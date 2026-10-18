/* convert_test.cpp -- rad-convert end to end on a tiny synthetic checkpoint (tests/tiny_model.h).
 *
 * The binary runs as a child on the host backend; the containers it writes are read back with
 * core's reader and decoder. The checkpoint is a two-layer LLaMA served by llama_dense_fp8, whose
 * recipe makes every linear block fp8 and keeps the norms, the embedding and the head as the
 * checkpoint holds them, so one conversion writes a quantised and a copied weight of every kind.
 */
#include "rad_test.h"
#include "tiny_model.h"

#include "format/encoding.h"
#include "format/radfile.h"
#include "gen_config.h"

#include <sys/stat.h>

#include <cstddef>
#include <map>

using namespace rad;

namespace {

std::vector<std::string> convert_env(const std::vector<std::string>& more = {}) {
    std::vector<std::string> e = { "ROCR_VISIBLE_DEVICES=99", "OMP_WAIT_POLICY=passive",
                                   "OMP_NUM_THREADS=2",
                                   std::string("RADIANCE_HOME=") + RAD_E2E_HOME };
    e.insert(e.end(), more.begin(), more.end());
    return e;
}

/* rad-convert `ckpt` with `args`: its exit status, and its output in `log`. */
int convert(const std::string& ckpt, const std::vector<std::string>& args, std::string* log,
            const std::vector<std::string>& env = {}) {
    std::vector<std::string> a = { RAD_E2E_CONVERT, ckpt };
    a.insert(a.end(), args.begin(), args.end());
    std::string out;
    const int st = tiny::run(a, convert_env(env), &out);
    if (log) *log = out;
    return st;
}

bool exists(const std::string& p) {
    struct stat st {};
    return stat(p.c_str(), &st) == 0;
}

bool copy_file(const std::string& from, const std::string& to) {
    std::error_code ec;
    std::filesystem::copy_file(from, to, std::filesystem::copy_options::overwrite_existing, ec);
    return !ec;
}

/* The checkpoint, its recipe and one conversion of it, shared by every case. */
struct World {
    tiny::TempDir dir{ "convert" };
    std::string ckpt, recipe, base;
    std::map<std::string, tiny::Tensor> src;   /* the checkpoint's tensors by name */
    std::string log;
    bool ok = false;

    World() {
        if (!dir.ok()) return;
        ckpt = dir / "llama";
        recipe = dir / "fp8.recipe";
        base = dir / "base.rad";
        if (mkdir(ckpt.c_str(), 0755) != 0 || !tiny::write_llama(ckpt, {}) ||
            !tiny::write_text(recipe, tiny::kLlamaFp8Recipe))
            return;
        for (tiny::Tensor& t : tiny::read_safetensors(ckpt + "/model.safetensors"))
            src[t.name] = std::move(t);
        ok = !src.empty() && convert(ckpt, { "-o", base, "--recipe", recipe }, &log) == 0;
        if (!ok) fprintf(stderr, "    the base conversion failed:\n%s\n", log.c_str());
    }
};

World& world() {
    static World w;
    return w;
}

/* A weight's planes, decoded to f32 [rows, cols] by the core's decoder. */
std::vector<float> decode(const RadFile& f, const RadFileEntry& e) {
    const int64_t cols = e.shape[e.rank - 1];
    int64_t rows = 1;
    for (uint32_t k = 0; k + 1 < e.rank; ++k) rows *= e.shape[k];
    const void* planes[RAD_ENC_MAX_PLANES] = {};
    for (uint32_t k = 0; k < e.n_planes; ++k) planes[k] = f.data(f.plane(e, (int)k));
    EncodedView v;
    enc_view(f.encoding(e), rows, cols, planes, &v);
    std::vector<float> out((size_t)(rows * cols));
    if (enc_decode_rows(v, 0, rows, out.data()) != RAD_OK) out.clear();
    return out;
}

/* The bytes of an entry's planes, concatenated. */
std::string entry_bytes(const RadFile& f, const RadFileEntry& e) {
    std::string s;
    for (uint32_t k = 0; k < e.n_planes; ++k) {
        const RadFilePlane& p = f.plane(e, (int)k);
        s.append((const char*)f.data(p), (size_t)p.bytes);
    }
    return s;
}

/* Every weight whose planes differ between two containers. */
std::vector<std::string> differing_weights(const RadFile& a, const RadFile& b) {
    std::vector<std::string> out;
    for (int64_t i = 0; i < a.entry_count(); ++i) {
        const RadFileEntry& e = a.entry(i);
        const RadFileEntry* o = b.find(a.str(e.name));
        if (!o || entry_bytes(a, e) != entry_bytes(b, *o)) out.push_back(a.str(e.name));
    }
    return out;
}

}  /* namespace */

/* ================================================================== what a conversion writes */

/* A recipe weight is block fp8 over the rows its name map joins, in the map's order: attn_qkv is
 * q, then k, then v, and ffn_gate_up is gate then up. Each element decodes to its source within
 * E4M3's half step (2^-4 relative) plus one subnormal step of its block's scale (amax / 448 /
 * 2^9). A part out of order, or a block's scale applied to another block's rows, is off by the
 * source's own magnitude. */
TEST(a_recipe_weight_decodes_to_its_source_rows_in_name_map_order) {
    World& w = world();
    REQUIRE(w.ok);
    RadFile f;
    REQUIRE_EQ(f.open(w.base.c_str()), RAD_OK);
    const struct { const char* weight; std::vector<const char*> parts; } kJoins[] = {
        { "blk.0.attn_qkv.weight", { "model.layers.0.self_attn.q_proj.weight",
                                     "model.layers.0.self_attn.k_proj.weight",
                                     "model.layers.0.self_attn.v_proj.weight" } },
        { "blk.1.ffn_gate_up.weight", { "model.layers.1.mlp.gate_proj.weight",
                                        "model.layers.1.mlp.up_proj.weight" } },
        { "blk.1.ffn_down.weight", { "model.layers.1.mlp.down_proj.weight" } },
        { "blk.0.attn_output.weight", { "model.layers.0.self_attn.o_proj.weight" } },
    };
    for (const auto& j : kJoins) {
        const RadFileEntry* e = f.find(j.weight);
        REQUIRE(e != nullptr);
        CHECK_EQ(enc_name(f.encoding(*e)), std::string("fp8_e4m3*bf16[128x128]"));
        CHECK_EQ(std::string(f.str(e->quantizer)), std::string("rtn"));
        const std::vector<float> got = decode(f, *e);
        REQUIRE(!got.empty());
        const int64_t rows = e->shape[0], cols = e->shape[1];
        std::vector<float> want;
        for (const char* part : j.parts) {
            const tiny::Tensor& t = w.src.at(part);
            REQUIRE_EQ(t.shape[1], cols);
            for (int64_t i = 0; i < t.shape[0] * cols; ++i) {
                uint16_t h;
                std::memcpy(&h, t.bytes.data() + 2 * i, 2);
                want.push_back(rad_bf16_to_f32(h));
            }
        }
        REQUIRE_EQ((int64_t)want.size(), rows * cols);
        int64_t bad = 0;
        for (int64_t r0 = 0; r0 < rows; r0 += 128)
            for (int64_t c0 = 0; c0 < cols; c0 += 128) {
                float amax = 0;
                for (int64_t r = r0; r < std::min(rows, r0 + 128); ++r)
                    for (int64_t c = c0; c < std::min(cols, c0 + 128); ++c)
                        amax = std::max(amax, std::fabs(want[(size_t)(r * cols + c)]));
                for (int64_t r = r0; r < std::min(rows, r0 + 128); ++r)
                    for (int64_t c = c0; c < std::min(cols, c0 + 128); ++c) {
                        const size_t i = (size_t)(r * cols + c);
                        const float tol = 0.0625f * std::fabs(want[i]) + amax / 448.0f / 512.0f;
                        if (!(std::fabs(got[i] - want[i]) <= tol)) ++bad;
                    }
            }
        if (bad)
            fprintf(stderr, "    %s: %lld elements outside the bound\n", j.weight, (long long)bad);
        CHECK_EQ(bad, (int64_t)0);
    }
}

/* A weight no rule matches keeps the checkpoint's encoding, byte for byte. */
TEST(a_weight_no_rule_matches_is_the_checkpoints_bytes) {
    World& w = world();
    REQUIRE(w.ok);
    RadFile f;
    REQUIRE_EQ(f.open(w.base.c_str()), RAD_OK);
    const std::pair<const char*, const char*> kCopies[] = {
        { "token_embd.weight", "model.embed_tokens.weight" },
        { "output.weight", "lm_head.weight" },
        { "output_norm.weight", "model.norm.weight" },
        { "blk.1.attn_norm.weight", "model.layers.1.input_layernorm.weight" },
        { "blk.0.ffn_norm.weight", "model.layers.0.post_attention_layernorm.weight" },
    };
    for (const auto& [name, from] : kCopies) {
        const RadFileEntry* e = f.find(name);
        REQUIRE(e != nullptr);
        CHECK_EQ(enc_name(f.encoding(*e)), std::string("bf16"));
        CHECK_EQ(std::string(f.str(e->quantizer)), std::string(""));
        const tiny::Tensor& t = w.src.at(from);
        CHECK(entry_bytes(f, *e) == std::string(t.bytes.begin(), t.bytes.end()));
    }
}

/* The header names the architecture and the recipe, the metadata carries config.json's
 * dimensions under RadModelMeta's names, and the provenance names the checkpoint without the
 * absolute path of the machine that converted it. */
TEST(the_header_and_metadata_describe_the_checkpoint) {
    World& w = world();
    REQUIRE(w.ok);
    RadFile f;
    REQUIRE_EQ(f.open(w.base.c_str()), RAD_OK);
    const RadFileHeader& h = f.header();
    CHECK_EQ(std::string(f.str(h.arch_id)), std::string("llama"));
    CHECK_EQ(std::string(f.str(h.quant)), std::string(""));
    const std::string recipe = f.str(h.recipe);
    for (const char* rule : { "blk.*.attn_qkv.weight", "blk.*.attn_output.weight",
                              "blk.*.ffn_gate_up.weight", "blk.*.ffn_down.weight" })
        CHECK(has(recipe, rule));
    const std::string by = f.str(h.created_by);
    CHECK(has(by, "rad-convert"));
    CHECK(!has(by, w.dir.path));
    const tiny::Llama s{};
    const std::pair<const char*, int64_t> kDims[] = {
        { "n_layers", s.layers }, { "n_embd", s.hidden }, { "n_head", s.heads },
        { "n_head_kv", s.kv_heads }, { "head_dim", s.head_dim }, { "n_ff", s.ffn },
        { "n_vocab", s.vocab }, { "n_ctx_train", s.ctx } };
    for (const auto& [key, want] : kDims) {
        int64_t v = -1;
        CHECK(f.get_i(key, &v));
        CHECK_EQ(v, want);
    }
    CHECK_EQ(f.entry_count(), (int64_t)15);
    CHECK(f.vocab());
}

/* Weights are quantised in parallel pieces; the bytes must not depend on how many threads. */
TEST(a_conversion_is_the_same_bytes_at_any_thread_count) {
    World& w = world();
    REQUIRE(w.ok);
    tiny::TempDir d("convert_t");
    for (const char* n : { "1", "7" }) {
        const std::string out = d / (std::string("t") + n + ".rad");
        REQUIRE_EQ(convert(w.ckpt, { "-o", out, "--recipe", w.recipe }, nullptr,
                           { std::string("OMP_NUM_THREADS=") + n }), 0);
        CHECK(tiny::read_text(out) == tiny::read_text(w.base));
    }
}

/* ================================================================== --reuse and --changed */

/* --reuse copies every weight converted the same way, which for an unchanged checkpoint is all
 * of them: the result is the same file. */
TEST(reuse_of_an_unchanged_checkpoint_writes_the_same_file) {
    World& w = world();
    REQUIRE(w.ok);
    tiny::TempDir d("convert_r");
    std::string log;
    REQUIRE_EQ(convert(w.ckpt, { "-o", d / "r.rad", "--recipe", w.recipe, "--reuse", w.base },
                       &log), 0);
    CHECK(has(log, "--reuse: 15 weight(s)"));
    CHECK(tiny::read_text(d / "r.rad") == tiny::read_text(w.base));
}

/* --reuse matches on how a weight was converted, not on its source bytes, so a changed tensor is
 * copied stale unless --changed names it; named, exactly the weights that read it are converted
 * again. */
TEST(changed_reconverts_exactly_the_weights_that_read_a_changed_tensor) {
    World& w = world();
    REQUIRE(w.ok);
    tiny::TempDir d("convert_c");
    const std::string ck = d / "llama";
    REQUIRE(mkdir(ck.c_str(), 0755) == 0);
    for (const char* f : { "config.json", "tokenizer.json" })
        REQUIRE(copy_file(w.ckpt + "/" + f, ck + "/" + f));
    std::vector<tiny::Tensor> ts = tiny::read_safetensors(w.ckpt + "/model.safetensors");
    for (tiny::Tensor& t : ts)
        if (t.name == "model.layers.0.mlp.down_proj.weight")
            for (size_t i = 0; i < t.bytes.size(); i += 2) t.bytes[i + 1] ^= 0x80;   /* negate */
    REQUIRE(tiny::write_safetensors(ck + "/model.safetensors", ts));
    REQUIRE(tiny::write_text(d / "changed",
                             "# the fine-tuned tensor\nmodel.layers.0.mlp.down_proj.weight\n"));

    RadFile base;
    REQUIRE_EQ(base.open(w.base.c_str()), RAD_OK);
    {
        REQUIRE_EQ(convert(ck, { "-o", d / "stale.rad", "--recipe", w.recipe, "--reuse", w.base },
                           nullptr), 0);
        RadFile f;
        REQUIRE_EQ(f.open((d / "stale.rad").c_str()), RAD_OK);
        CHECK(differing_weights(base, f).empty());
    }
    std::string log;
    REQUIRE_EQ(convert(ck, { "-o", d / "fresh.rad", "--recipe", w.recipe, "--reuse", w.base,
                             "--changed", d / "changed" }, &log), 0);
    RadFile f;
    REQUIRE_EQ(f.open((d / "fresh.rad").c_str()), RAD_OK);
    const std::vector<std::string> diff = differing_weights(base, f);
    CHECK_EQ(diff.size(), (size_t)1);
    CHECK(!diff.empty() && diff[0] == "blk.0.ffn_down.weight");
    /* And it is the negated weight, not merely different bytes. */
    const RadFileEntry* a = base.find("blk.0.ffn_down.weight");
    const RadFileEntry* b = f.find("blk.0.ffn_down.weight");
    REQUIRE(a && b);
    const std::vector<float> va = decode(base, *a), vb = decode(f, *b);
    REQUIRE(!va.empty() && va.size() == vb.size());
    int64_t neg = 0;
    for (size_t i = 0; i < va.size(); ++i) neg += va[i] == -vb[i];
    CHECK_EQ(neg, (int64_t)va.size());
}

/* --changed is a list for --reuse: without it, or naming a tensor no weight reads (a typo would
 * otherwise copy the stale weight it meant), the run stops before writing anything. */
TEST(changed_is_refused_without_reuse_or_for_a_tensor_nothing_reads) {
    World& w = world();
    REQUIRE(w.ok);
    tiny::TempDir d("convert_cr");
    REQUIRE(tiny::write_text(d / "unread", "model.layers.0.mlp.nope.weight\n"));
    REQUIRE(tiny::write_text(d / "real", "model.layers.0.mlp.down_proj.weight\n"));
    const std::vector<std::vector<std::string>> kRuns = {
        { "-o", d / "a.rad", "--recipe", w.recipe, "--changed", d / "real" },
        { "-o", d / "b.rad", "--recipe", w.recipe, "--reuse", w.base, "--changed", d / "unread" },
    };
    for (const auto& args : kRuns) {
        std::string log;
        CHECK(convert(w.ckpt, args, &log) != 0);
        CHECK(!exists(args[1]));
        CHECK(!exists(args[1] + ".blob.tmp"));
    }
}

/* ================================================================== --in-place */

/* Extending a container with nothing new keeps every weight at its offset, and the .pre-append
 * file (the old header and length) restores the original byte for byte. */
TEST(in_place_keeps_every_weight_where_it_was_and_can_be_undone) {
    World& w = world();
    REQUIRE(w.ok);
    tiny::TempDir d("convert_ip");
    const std::string ip = d / "ip.rad";
    REQUIRE(copy_file(w.base, ip));
    std::string log;
    REQUIRE_EQ(convert(w.ckpt, { "--recipe", w.recipe, "--reuse", ip, "--in-place" }, &log), 0);
    CHECK(has(log, "15 weight(s)"));
    RadFile a, b;
    REQUIRE_EQ(a.open(w.base.c_str()), RAD_OK);
    REQUIRE_EQ(b.open(ip.c_str()), RAD_OK);
    REQUIRE_EQ(a.entry_count(), b.entry_count());
    for (int64_t i = 0; i < a.entry_count(); ++i) {
        const RadFileEntry* e = b.find(a.str(a.entry(i).name));
        REQUIRE(e != nullptr);
        CHECK_EQ(e->offset, a.entry(i).offset);
    }
    b.close();

    const std::string bak = tiny::read_text(ip + ".pre-append");
    REQUIRE_EQ(bak.size(), sizeof(RadFileHeader) + sizeof(int64_t));
    int64_t len = 0;
    std::memcpy(&len, bak.data() + sizeof(RadFileHeader), sizeof len);
    std::string file = tiny::read_text(ip);
    REQUIRE(len > 0 && (size_t)len <= file.size());
    file.replace(0, sizeof(RadFileHeader), bak.data(), sizeof(RadFileHeader));
    file.resize((size_t)len);
    CHECK(file == tiny::read_text(w.base));
}

/* A weight the container holds in another movement group than the plugin now declares (a plugin
 * that regrouped it after the conversion) is kept in place, its entry taking the declared group:
 * the group places no bytes of a dense weight. */
TEST(in_place_keeps_a_weight_the_plugin_has_regrouped) {
    World& w = world();
    REQUIRE(w.ok);
    tiny::TempDir d("convert_ipg");
    const std::string ip = d / "ip.rad";
    REQUIRE(copy_file(w.base, ip));
    RadFile a;
    REQUIRE_EQ(a.open(w.base.c_str()), RAD_OK);
    REQUIRE(a.entry_count() > 0);
    const RadFileEntry& e0 = a.entry(0);
    REQUIRE(e0.expert < 0);
    const std::string name = a.str(e0.name);
    const int32_t layer = e0.layer;
    const uint64_t off = e0.offset;
    {
        std::string file = tiny::read_text(ip);
        const size_t at = (size_t)a.header().dir_off + offsetof(RadFileEntry, layer);
        const int32_t other = layer + 7;
        std::memcpy(&file[at], &other, sizeof other);
        REQUIRE(tiny::write_text(ip, file));
    }
    a.close();
    std::string log;
    REQUIRE_EQ(convert(w.ckpt, { "--recipe", w.recipe, "--reuse", ip, "--in-place" }, &log), 0);
    RadFile b;
    REQUIRE_EQ(b.open(ip.c_str()), RAD_OK);
    const RadFileEntry* e = b.find(name.c_str());
    REQUIRE(e != nullptr);
    CHECK_EQ(e->layer, layer);
    CHECK_EQ(e->offset, off);
}

/* In place, a weight cannot be replaced: a rule that would write one differently is refused and
 * the file is left as it was. Without --in-place, -o may not name the --reuse container, whose
 * weights the run reads while it truncates the output. */
TEST(in_place_refuses_to_change_a_weight_and_reuse_never_truncates_its_source) {
    World& w = world();
    REQUIRE(w.ok);
    tiny::TempDir d("convert_ipr");
    const std::string ip = d / "ip.rad";
    REQUIRE(copy_file(w.base, ip));
    REQUIRE(tiny::write_text(d / "search.recipe",
                             std::string("blk.0.attn_qkv.weight rtn codes=fp8_e4m3 block=128x128 "
                                         "scale=bf16 rule=search\n") + tiny::kLlamaFp8Recipe));
    std::string log;
    CHECK(convert(w.ckpt, { "--recipe", d / "search.recipe", "--reuse", ip, "--in-place" },
                  &log) != 0);
    CHECK(has(log, "cannot be replaced in place"));
    CHECK(tiny::read_text(ip) == tiny::read_text(w.base));
    CHECK_EQ(convert(w.ckpt, { "-o", ip, "--recipe", w.recipe, "--reuse", ip }, &log), 2);
    CHECK(tiny::read_text(ip) == tiny::read_text(w.base));
}

/* ================================================================== refusals and options */

/* --plan-only declares and plans every weight, and writes nothing. */
TEST(plan_only_writes_nothing) {
    World& w = world();
    REQUIRE(w.ok);
    tiny::TempDir d("convert_p");
    std::string log;
    CHECK_EQ(convert(w.ckpt, { "-o", d / "p.rad", "--recipe", w.recipe, "--plan-only" }, &log), 0);
    CHECK(has(log, "nothing was written"));
    CHECK(!exists(d / "p.rad"));
}

/* A recipe or flag that cannot be honoured stops the run with status 2 before any file exists:
 * an unknown quantiser, option, or non-integer value, a recipe that cannot be read, an unset
 * $VARIABLE, a --set without `=`, and a KV dtype that is not bf16 or fp8. */
TEST(a_recipe_or_flag_that_cannot_be_honoured_is_refused_before_writing) {
    World& w = world();
    REQUIRE(w.ok);
    tiny::TempDir d("convert_x");
    const char* kRule = "blk.*.attn_qkv.weight ";
    const std::pair<const char*, std::string> kRecipes[] = {
        { "quantiser", std::string(kRule) + "nope codes=fp8_e4m3" },
        { "option", std::string(kRule) + "rtn codez=fp8_e4m3 block=128x128 scale=bf16" },
        { "integer", std::string(kRule) + "rtn codes=i4 group=abc scale=f16" },
        { "variable", std::string(kRule) + "rtn codes=$RAD_CONVERT_TEST_UNSET block=128x128" },
    };
    int k = 0;
    for (const auto& [what, text] : kRecipes) {
        const std::string r = d / (std::string(what) + ".recipe");
        const std::string out = d / ("x" + std::to_string(k++));
        REQUIRE(tiny::write_text(r, text + "\n"));
        std::string log;
        const int st = convert(w.ckpt, { "-o", out, "--recipe", r }, &log,
                               { "RAD_CONVERT_TEST_UNSET" });
        if (st != 2) fprintf(stderr, "    %s: status %d\n%s\n", what, st, log.c_str());
        CHECK_EQ(st, 2);
        CHECK(!exists(out));
    }
    const std::vector<std::vector<std::string>> kFlags = {
        { "--recipe", d / "absent.recipe" },
        { "--recipe", w.recipe, "--set", "foo" },
        { "--recipe", w.recipe, "--kv-cache-dtype", "auto" },
    };
    for (const auto& args : kFlags) {
        std::vector<std::string> a = { "-o", d / "f.rad" };
        a.insert(a.end(), args.begin(), args.end());
        CHECK_EQ(convert(w.ckpt, a, nullptr), 2);
        CHECK(!exists(d / "f.rad"));
    }
}

/* ================================================================== an int4 release */

namespace {

/* compressed-tensors' value of one packed linear of a tiny int4 release (tiny_model.h,
 * write_llama_ct_int4): (q - zero_point) * scale, f32 [N, K]. */
std::vector<float> ct_values(const std::map<std::string, tiny::Tensor>& src, const std::string& b) {
    const tiny::Tensor& p = src.at(b + ".weight_packed");
    const tiny::Tensor& s = src.at(b + ".weight_scale");
    const auto z = src.find(b + ".weight_zero_point");
    const int64_t N = p.shape[0], K = p.shape[1] * 8, G = s.shape[1];
    std::vector<float> v((size_t)(N * K));
    for (int64_t n = 0; n < N; ++n)
        for (int64_t k = 0; k < K; ++k) {
            uint32_t w;
            std::memcpy(&w, p.bytes.data() + (n * (K / 8) + k / 8) * 4, 4);
            const int q = (int)((w >> (4 * (k % 8))) & 15) - 8;
            int zp = 0;
            if (z != src.end()) {
                std::memcpy(&w, z->second.bytes.data() + ((n / 8) * G + k / 128) * 4, 4);
                zp = (int)((w >> (4 * (n % 8))) & 15) - 8;
            }
            v[(size_t)(n * K + k)] =
                (float)(q - zp) * rad_load_f32(s.bytes.data(), RAD_BF16, n * G + k / 128);
        }
    return v;
}

}  /* namespace */

/* An int4 W4A16 release (compressed-tensors, asymmetric with zero points, and symmetric with
 * act-order static) converts with the import rule into the 4-bit GEMMs' form,
 * `u4*f16[1x128]-u4[1x128]`, each joined weight decoding to compressed-tensors' (q - zp) * scale
 * bit for bit, its rule recorded, and the run saying every shipped value is exact. Declaring it
 * resolves the cast, the f16 4-bit GEMM, the int8 quantiser and the int8 4-bit GEMM on the host
 * libraries. Without the rule the plugin refuses the shipped form by name. */
TEST(an_int4_release_imports_exactly_through_the_import_rule) {
    for (const bool asym : { true, false }) {
        tiny::TempDir d(asym ? "convert_i4a" : "convert_i4s");
        REQUIRE(d.ok());
        const std::string ck = d / "llama-int4", recipe = d / "int4.recipe", out = d / "i4.rad";
        REQUIRE_EQ(mkdir(ck.c_str(), 0755), 0);
        REQUIRE(tiny::write_llama_ct_int4(ck, {}, asym));
        REQUIRE(tiny::write_text(recipe, tiny::kInt4ImportRecipe));
        std::map<std::string, tiny::Tensor> src;
        for (tiny::Tensor& t : tiny::read_safetensors(ck + "/model.safetensors"))
            src[t.name] = std::move(t);
        std::string log;
        const int st = convert(ck, { "-o", out, "--recipe", recipe }, &log);
        if (st != 0) fprintf(stderr, "%s\n", log.c_str());
        REQUIRE_EQ(st, 0);
        CHECK(has(log, "every shipped value exact"));
        RadFile f;
        REQUIRE_EQ(f.open(out.c_str()), RAD_OK);
        const struct { const char* weight; std::vector<std::string> parts; } kJoins[] = {
            { "blk.0.attn_qkv.weight", { "model.layers.0.self_attn.q_proj",
                                         "model.layers.0.self_attn.k_proj",
                                         "model.layers.0.self_attn.v_proj" } },
            { "blk.1.ffn_gate_up.weight", { "model.layers.1.mlp.gate_proj",
                                            "model.layers.1.mlp.up_proj" } },
            { "blk.1.ffn_down.weight", { "model.layers.1.mlp.down_proj" } },
            { "blk.0.attn_output.weight", { "model.layers.0.self_attn.o_proj" } },
        };
        for (const auto& j : kJoins) {
            const RadFileEntry* e = f.find(j.weight);
            REQUIRE(e != nullptr);
            CHECK_EQ(enc_name(f.encoding(*e)), std::string("u4*f16[1x128]-u4[1x128]"));
            CHECK_EQ(std::string(f.str(e->quantizer)), std::string("import"));
            CHECK_EQ(std::string(f.str(e->options)), std::string("codes=u4,scale=f16,zero=u4"));
            std::vector<float> want;
            for (const std::string& part : j.parts) {
                const std::vector<float> v = ct_values(src, part);
                want.insert(want.end(), v.begin(), v.end());
            }
            const std::vector<float> got = decode(f, *e);
            REQUIRE_EQ(got.size(), want.size());
            int64_t bad = 0;
            for (size_t i = 0; i < got.size(); ++i) bad += got[i] != want[i];
            CHECK_EQ(bad, (int64_t)0);
        }
        /* The head the release leaves bf16 stays so. */
        const RadFileEntry* head = f.find("output.weight");
        REQUIRE(head != nullptr);
        CHECK_EQ(enc_name(f.encoding(*head)), std::string("bf16"));

        std::string log2;
        CHECK(convert(ck, { "-o", d / "none.rad" }, &log2) != 0);
        CHECK(has(log2, asym ? "u4*bf16[1x128]-u4[1x128]" : "i4*bf16[1x128]"));
        CHECK(has(log2, "import rule"));
    }
}

/* A rule that matches nothing is almost always a typo, so it is named; the run goes on. */
TEST(a_rule_that_matches_no_weight_is_warned_about) {
    World& w = world();
    REQUIRE(w.ok);
    tiny::TempDir d("convert_w");
    REQUIRE(tiny::write_text(d / "r", std::string("blk.*.attn_qvk.weight rtn codes=fp8_e4m3 "
                                                  "block=128x128 scale=bf16\n") +
                                          tiny::kLlamaFp8Recipe));
    std::string log;
    CHECK_EQ(convert(w.ckpt, { "-o", d / "w.rad", "--recipe", d / "r" }, &log), 0);
    CHECK(has(log, "'blk.*.attn_qvk.weight rtn' matches no weight"));
}

/* --set writes a key into the container, where declare and the engine both read it; setting the
 * quantization_config makes it the header's descriptor, which selects the plugin. */
TEST(set_writes_a_metadata_key_and_can_state_the_quantisation) {
    World& w = world();
    REQUIRE(w.ok);
    tiny::TempDir d("convert_s");
    REQUIRE_EQ(convert(w.ckpt, { "-o", d / "s.rad", "--recipe", w.recipe, "--set", "foo.bar=7",
                                 "--set", "quantization_config.quant_method=fp8",
                                 "--set", "quantization_config.fmt=e4m3" }, nullptr), 0);
    RadFile f;
    REQUIRE_EQ(f.open((d / "s.rad").c_str()), RAD_OK);
    const char* v = f.get_s("foo.bar");
    int64_t i = -1;
    CHECK((v && std::string(v) == "7") || (f.get_i("foo.bar", &i) && i == 7));
    CHECK_EQ(std::string(f.str(f.header().quant)), std::string("fp8_e4m3"));
}

/* generation_config.json beside the checkpoint travels as `generation.*` keys, which is where
 * the engine reads a container's sampling defaults. */
TEST(generation_config_travels_as_generation_keys) {
    World& w = world();
    REQUIRE(w.ok);
    tiny::TempDir d("convert_g");
    const std::string ck = d / "llama";
    REQUIRE(mkdir(ck.c_str(), 0755) == 0);
    for (const char* f : { "config.json", "tokenizer.json", "model.safetensors" })
        REQUIRE(copy_file(w.ckpt + "/" + f, ck + "/" + f));
    REQUIRE(tiny::write_text(
        ck + "/generation_config.json",
        R"({"temperature": 0.6, "top_k": 20, "top_p": 0.95, "do_sample": true})"));
    REQUIRE_EQ(convert(ck, { "-o", d / "g.rad", "--recipe", w.recipe }, nullptr), 0);
    RadFile f;
    REQUIRE_EQ(f.open((d / "g.rad").c_str()), RAD_OK);
    std::vector<std::string> keep;
    GenSampling g;
    CHECK(gen_sampling_from_kv(
        [&](const char* k) -> const char* {
            const std::string key = std::string("generation.") + k;
            if (const char* s = f.get_s(key.c_str())) return s;
            int64_t i;
            double x;
            if (f.get_i(key.c_str(), &i)) keep.push_back(std::to_string(i));
            else if (f.get_f(key.c_str(), &x)) keep.push_back(std::to_string(x));
            else return nullptr;
            return keep.back().c_str();
        },
        &g));
    CHECK_NEAR(g.temp, 0.6, 1e-6);
    CHECK_EQ(g.top_k, 20);
    CHECK_NEAR(g.top_p, 0.95, 1e-6);
}

/* Declare runs at the KV dtype the engine will use; fp8 resolves on the host kernels too. */
TEST(an_fp8_kv_cache_declares) {
    World& w = world();
    REQUIRE(w.ok);
    tiny::TempDir d("convert_k");
    CHECK_EQ(convert(w.ckpt, { "-o", d / "k.rad", "--recipe", w.recipe, "--kv-cache-dtype", "fp8" },
                     nullptr), 0);
    CHECK(exists(d / "k.rad"));
}

RAD_TEST_MAIN()
