/* checkpoint_test.cpp -- a checkpoint as a source of logical weights (core/format/checkpoint.h).
 *
 * rad-convert writes a container from a checkpoint and the loader serves a trivial one directly,
 * and both go through ckpt_resolve: which tensors make a declared weight, what encoding they
 * already are, and the rows of it. A wrong answer there is a model that loads and is wrong -- an
 * expert that is every expert, a scale applied to the next piece's rows -- so each case below is
 * one of the ways a checkpoint is laid out in practice, built small and read back.
 */
#include "rad_test.h"
#include "format/checkpoint.h"
#include "format/encoding.h"
#include "rad_plugin.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <sys/stat.h>
#include <unistd.h>
#include <unordered_set>
#include <vector>

using namespace rad;

namespace {

struct T {
    std::string name, dtype;          /* safetensors spelling: BF16, F32, F8_E4M3 */
    std::vector<int64_t> shape;
    std::vector<uint8_t> bytes;
};

std::string dir_path() {
    const char* d = std::getenv("TMPDIR");
    return std::string(d && *d ? d : "/tmp") + "/radiance_checkpoint_test_" +
           std::to_string((long long)getpid());
}

/* One safetensors file holding `ts`, in order. */
void write_shard(const std::string& path, const std::vector<T>& ts) {
    std::string hdr = "{";
    uint64_t off = 0;
    for (size_t i = 0; i < ts.size(); ++i) {
        std::string shape;
        for (size_t k = 0; k < ts[i].shape.size(); ++k)
            shape += (k ? "," : "") + std::to_string(ts[i].shape[k]);
        hdr += (i ? "," : "") + std::string("\"") + ts[i].name + "\":{\"dtype\":\"" + ts[i].dtype +
               "\",\"shape\":[" + shape + "],\"data_offsets\":[" + std::to_string(off) + "," +
               std::to_string(off + ts[i].bytes.size()) + "]}";
        off += ts[i].bytes.size();
    }
    hdr += "}";
    while (hdr.size() % 8) hdr += ' ';
    FILE* f = std::fopen(path.c_str(), "wb");
    const uint64_t n = hdr.size();
    std::fwrite(&n, 8, 1, f);
    std::fwrite(hdr.data(), 1, hdr.size(), f);
    for (const T& t : ts) std::fwrite(t.bytes.data(), 1, t.bytes.size(), f);
    std::fclose(f);
}

/* One model.safetensors and a config.json in a fresh directory. */
std::string write_checkpoint(const std::vector<T>& ts, const std::string& config) {
    const std::string dir = dir_path();
    ::mkdir(dir.c_str(), 0755);
    write_shard(dir + "/model.safetensors", ts);
    FILE* f = std::fopen((dir + "/config.json").c_str(), "wb");
    std::fwrite(config.data(), 1, config.size(), f);
    std::fclose(f);
    return dir;
}

void remove_checkpoint(const std::string& dir) {
    std::remove((dir + "/model.safetensors").c_str());
    std::remove((dir + "/config.json").c_str());
    ::rmdir(dir.c_str());
}

/* Element i of a tensor is a value that names it: tensor tag + i/1000, exact in bf16 for small i. */
T bf16(const std::string& name, std::vector<int64_t> shape, float tag) {
    T t{ name, "BF16", shape, {} };
    int64_t n = 1;
    for (int64_t s : shape) n *= s;
    t.bytes.resize((size_t)n * 2);
    for (int64_t i = 0; i < n; ++i) rad_store_f32(t.bytes.data(), RAD_BF16, i, tag + (float)(i % 64));
    return t;
}

T fp8(const std::string& name, std::vector<int64_t> shape) {
    T t{ name, "F8_E4M3", shape, {} };
    int64_t n = 1;
    for (int64_t s : shape) n *= s;
    t.bytes.resize((size_t)n);
    for (int64_t i = 0; i < n; ++i) rad_store_f32(t.bytes.data(), RAD_F8E4M3, i, (float)(i % 7) - 3.0f);
    return t;
}

RadNameMap map(const char* declared, int mode, std::vector<const char*> src,
               std::vector<int> index = {}, int concat_dim = 0) {
    RadNameMap m{};
    m.declared = declared;
    m.mode = mode;
    m.n_src = (int)src.size();
    m.concat_dim = concat_dim;
    for (int i = 0; i < 8; ++i) m.src_index[i] = -1;
    for (size_t i = 0; i < src.size(); ++i) m.src[i] = src[i];
    for (size_t i = 0; i < index.size(); ++i) m.src_index[i] = index[i];
    return m;
}

const char* kConfig =
    "{\"model_type\":\"qwen3_5\",\"hidden_size\":256,\"num_hidden_layers\":2,"
    "\"quantization_config\":{\"quant_method\":\"fp8\",\"fmt\":\"e4m3\","
    "\"weight_block_size\":[128,128]}}";

}  /* namespace */

TEST(the_metadata_names_the_architecture_and_the_quantisation) {
    const std::string dir = write_checkpoint({ bf16("x", { 2, 2 }, 0) }, kConfig);
    Checkpoint ck;
    REQUIRE_EQ(ck.open(dir), RAD_OK);
    CHECK_EQ(ck.meta().arch, std::string("qwen35"));
    CHECK_EQ(ck.meta().quant, std::string("fp8_e4m3"));
    CHECK_EQ(ck.meta().m.n_embd, (int64_t)256);
    CHECK_EQ(ck.fp8_block_rows(), (int64_t)128);
    CHECK_EQ(ck.dir(), dir);
    remove_checkpoint(dir);
}

/* rad-convert's --set of quantization_config re-derives the descriptor from the same keys, so a
 * bf16 checkpoint whose recipe writes block fp8 is converted for the fp8 plugin by stating the
 * format, and the container never states one format while selecting the plugin of another. */
TEST(the_descriptor_is_quant_method_and_fmt_whoever_set_them) {
    const std::string dir = write_checkpoint({ bf16("x", { 2, 2 }, 0) },
                                             "{\"model_type\":\"qwen3_5\",\"hidden_size\":256}");
    Checkpoint ck;
    REQUIRE_EQ(ck.open(dir), RAD_OK);
    CHECK_EQ(ck.meta().quant, std::string());
    MetaOwn m = ck.meta();
    m.set("quantization_config.quant_method", "fp8");
    CHECK_EQ(quant_descriptor(m), std::string("fp8"));
    m.set("quantization_config.fmt", "e4m3");
    CHECK_EQ(quant_descriptor(m), std::string("fp8_e4m3"));
    remove_checkpoint(dir);
}

TEST(a_plain_tensor_is_its_own_encoding_and_reads_back_as_itself) {
    const std::string dir = write_checkpoint({ bf16("a.weight", { 4, 8 }, 1.0f) }, "{}");
    Checkpoint ck;
    REQUIRE_EQ(ck.open(dir), RAD_OK);
    Program prog;
    prog.name_map.push_back(map("a", RAD_MAP_COPY, { "a.weight" }));
    CkptWeight w;
    std::string why;
    REQUIRE_EQ(ckpt_resolve(ck, prog, "a", false, &w, &why), RAD_OK);
    CHECK(w.trivial);
    CHECK(rad_enc_is(&w.enc, "plain"));
    CHECK_EQ(w.enc.plane[0].dtype, (uint32_t)RAD_BF16);
    CHECK_EQ(w.rows, (int64_t)4);
    CHECK_EQ(w.cols, (int64_t)8);
    std::vector<float> f(8);
    REQUIRE_EQ(ckpt_rows_f32(w, 2, 1, f.data(), &why), RAD_OK);
    for (int j = 0; j < 8; ++j) CHECK_EQ(f[(size_t)j], 1.0f + (float)((16 + j) % 64));
    std::vector<uint8_t> raw(8 * 2);
    REQUIRE_EQ(ckpt_plane_rows(w, 0, 2, 1, raw.data(), &why), RAD_OK);
    CHECK_EQ(std::memcmp(raw.data(), (const uint8_t*)ck.find("a.weight")->data + 32, 16), 0);
    remove_checkpoint(dir);
}

/* BLOCK-SCALED FP8 IS ONE WEIGHT OF TWO PLANES, and its value is the code times its block's
 * scale -- the scale of the block the element is in, along both axes. */
TEST(block_fp8_is_the_codes_and_their_scale_plane) {
    const int64_t N = 256, K = 384;
    T s = bf16("p.weight_scale_inv", { 2, 3 }, 0.0f);
    for (int i = 0; i < 6; ++i) rad_store_f32(s.bytes.data(), RAD_BF16, i, 0.5f * (float)(i + 1));
    const std::string dir = write_checkpoint({ fp8("p.weight", { N, K }), s }, kConfig);
    Checkpoint ck;
    REQUIRE_EQ(ck.open(dir), RAD_OK);
    Program prog;
    prog.name_map.push_back(map("p", RAD_MAP_COPY, { "p.weight" }));
    CkptWeight w;
    std::string why;
    REQUIRE_EQ(ckpt_resolve(ck, prog, "p", false, &w, &why), RAD_OK);
    REQUIRE(w.trivial);
    CHECK(rad_enc_is(&w.enc, "affine"));
    REQUIRE_EQ(w.enc.n_planes, 2);
    CHECK_EQ(w.enc.plane[1].block[0], (int64_t)128);
    CHECK_EQ(w.enc.plane[1].block[1], (int64_t)128);
    CHECK(rad_enc_valid(&w.enc));
    std::vector<float> f((size_t)(2 * K));
    REQUIRE_EQ(ckpt_rows_f32(w, 129, 2, f.data(), &why), RAD_OK);
    for (int64_t r = 0; r < 2; ++r)
        for (int64_t c : { (int64_t)0, (int64_t)130, (int64_t)383 }) {
            const int64_t i = (129 + r) * K + c;
            const float code = (float)(i % 7) - 3.0f;
            const float scale = 0.5f * (float)(3 + c / 128 + 1);   /* block row 1 */
            CHECK_EQ(f[(size_t)(r * K + c)], code * scale);
        }
    std::vector<uint8_t> sp(3 * 2);
    REQUIRE_EQ(ckpt_plane_rows(w, 1, 1, 1, sp.data(), &why), RAD_OK);
    CHECK_EQ(std::memcmp(sp.data(), s.bytes.data() + 6, 6), 0);
    remove_checkpoint(dir);
}

/* A FUSION JOINS THE PIECES -- and their scale planes -- along dim 0, and refuses a piece that ends
 * inside a scale block, whose scales could not follow it. */
TEST(a_fusion_joins_pieces_and_their_scales) {
    const std::string dir = write_checkpoint(
        { fp8("g.weight", { 256, 128 }), bf16("g.weight_scale_inv", { 2, 1 }, 10.0f),
          fp8("u.weight", { 128, 128 }), bf16("u.weight_scale_inv", { 1, 1 }, 20.0f),
          fp8("odd.weight", { 100, 128 }), bf16("odd.weight_scale_inv", { 1, 1 }, 30.0f) },
        kConfig);
    Checkpoint ck;
    REQUIRE_EQ(ck.open(dir), RAD_OK);
    Program prog;
    prog.name_map.push_back(map("gu", RAD_MAP_CONCAT, { "g.weight", "u.weight" }));
    prog.name_map.push_back(map("bad", RAD_MAP_CONCAT, { "odd.weight", "u.weight" }));
    CkptWeight w;
    std::string why;
    REQUIRE_EQ(ckpt_resolve(ck, prog, "gu", false, &w, &why), RAD_OK);
    CHECK(w.trivial);
    CHECK_EQ(w.shape[0], (int64_t)384);
    std::vector<uint8_t> sp(3 * 2);
    REQUIRE_EQ(ckpt_plane_rows(w, 1, 0, 3, sp.data(), &why), RAD_OK);
    CHECK_EQ(rad_load_f32(sp.data(), RAD_BF16, 0), 10.0f);
    CHECK_EQ(rad_load_f32(sp.data(), RAD_BF16, 1), 11.0f);
    CHECK_EQ(rad_load_f32(sp.data(), RAD_BF16, 2), 20.0f);
    std::vector<float> f(128);
    REQUIRE_EQ(ckpt_rows_f32(w, 300, 1, f.data(), &why), RAD_OK);   /* u's row 44, u's scale */
    CHECK_EQ(f[5], ((float)((44 * 128 + 5) % 7) - 3.0f) * 20.0f);

    REQUIRE_EQ(ckpt_resolve(ck, prog, "bad", false, &w, &why), RAD_OK);
    CHECK(!w.trivial);
    CHECK(w.why.find("inside a scale block") != std::string::npos);
    remove_checkpoint(dir);
}

/* A join along dim 1 interleaves the pieces row by row. */
TEST(a_join_along_columns_interleaves_rows) {
    const std::string dir = write_checkpoint({ bf16("l", { 3, 2 }, 0.0f),
                                               bf16("r", { 3, 4 }, 100.0f) }, "{}");
    Checkpoint ck;
    REQUIRE_EQ(ck.open(dir), RAD_OK);
    Program prog;
    prog.name_map.push_back(map("lr", RAD_MAP_CONCAT, { "l", "r" }, {}, 1));
    CkptWeight w;
    std::string why;
    REQUIRE_EQ(ckpt_resolve(ck, prog, "lr", false, &w, &why), RAD_OK);
    CHECK_EQ(w.shape[1], (int64_t)6);
    std::vector<float> f(6);
    REQUIRE_EQ(ckpt_rows_f32(w, 1, 1, f.data(), &why), RAD_OK);
    const float want[6] = { 2, 3, 104, 105, 106, 107 };
    for (int j = 0; j < 6; ++j) CHECK_EQ(f[(size_t)j], want[j]);
    std::vector<uint8_t> raw(6 * 2);
    REQUIRE_EQ(ckpt_plane_rows(w, 0, 2, 1, raw.data(), &why), RAD_OK);
    CHECK_EQ(rad_load_f32(raw.data(), RAD_BF16, 0), 4.0f);
    CHECK_EQ(rad_load_f32(raw.data(), RAD_BF16, 2), 108.0f);
    remove_checkpoint(dir);
}

/* AN EXPERT IS ITS OWN SLICE OF A STACKED TENSOR -- never the whole stack -- and a stacked FP8
 * tensor's scales are sliced at the same index. */
TEST(an_expert_is_its_slice_of_a_stack) {
    const std::string dir = write_checkpoint(
        { bf16("exps", { 4, 2, 3 }, 0.0f), fp8("q.weight", { 2, 128, 128 }),
          bf16("q.weight_scale_inv", { 2, 1, 1 }, 7.0f) }, kConfig);
    Checkpoint ck;
    REQUIRE_EQ(ck.open(dir), RAD_OK);
    Program prog;
    prog.name_map.push_back(map("e2", RAD_MAP_COPY, { "exps" }, { 2 }));
    prog.name_map.push_back(map("q1", RAD_MAP_COPY, { "q.weight" }, { 1 }));
    prog.name_map.push_back(map("e9", RAD_MAP_COPY, { "exps" }, { 9 }));
    CkptWeight w;
    std::string why;
    REQUIRE_EQ(ckpt_resolve(ck, prog, "e2", false, &w, &why), RAD_OK);
    CHECK_EQ(w.rank, 2u);
    std::vector<float> f(3);
    REQUIRE_EQ(ckpt_rows_f32(w, 1, 1, f.data(), &why), RAD_OK);
    CHECK_EQ(f[0], (float)(2 * 6 + 3));                       /* stack index 2, row 1, col 0 */
    REQUIRE_EQ(ckpt_resolve(ck, prog, "q1", false, &w, &why), RAD_OK);
    CHECK(w.trivial);
    std::vector<uint8_t> sp(2);
    REQUIRE_EQ(ckpt_plane_rows(w, 1, 0, 1, sp.data(), &why), RAD_OK);
    CHECK_EQ(rad_load_f32(sp.data(), RAD_BF16, 0), 8.0f);    /* the second expert's scale */
    CHECK(ckpt_resolve(ck, prog, "e9", false, &w, &why) < 0);
    remove_checkpoint(dir);
}

/* A CHANGED TENSOR IS NAMED AS THE CHECKPOINT NAMES IT. An expert is a slice "exps[2]" of its
 * stack, and it reads "exps"; a block-FP8 weight reads its scale plane as well as its codes, so a
 * changed scale changes the weight. */
TEST(a_weight_reads_the_checkpoint_tensors_its_pieces_and_scales_come_from) {
    const std::string dir = write_checkpoint(
        { bf16("exps", { 4, 2, 3 }, 0.0f), fp8("q.weight", { 2, 128, 128 }),
          bf16("q.weight_scale_inv", { 2, 1, 1 }, 7.0f), bf16("a.weight", { 2, 3 }, 0.0f),
          bf16("b.weight", { 2, 3 }, 0.0f) }, kConfig);
    Checkpoint ck;
    REQUIRE_EQ(ck.open(dir), RAD_OK);
    Program prog;
    prog.name_map.push_back(map("e2", RAD_MAP_COPY, { "exps" }, { 2 }));
    prog.name_map.push_back(map("q1", RAD_MAP_COPY, { "q.weight" }, { 1 }));
    prog.name_map.push_back(map("ab", RAD_MAP_CONCAT, { "a.weight", "b.weight" }));
    CkptWeight e2, q1, ab;
    std::string why;
    REQUIRE_EQ(ckpt_resolve(ck, prog, "e2", false, &e2, &why), RAD_OK);
    REQUIRE_EQ(ckpt_resolve(ck, prog, "q1", false, &q1, &why), RAD_OK);
    REQUIRE_EQ(ckpt_resolve(ck, prog, "ab", false, &ab, &why), RAD_OK);
    CHECK_EQ(ckpt_tensor_of(*e2.pieces[0]), std::string("exps"));

    std::unordered_set<std::string> hit;
    CHECK(ckpt_reads_any(e2, { "exps" }, &hit));
    CHECK(!ckpt_reads_any(e2, { "exps[2]" }));        /* a slice is not a checkpoint name */
    CHECK(ckpt_reads_any(q1, { "q.weight_scale_inv" }, &hit));
    CHECK(ckpt_reads_any(ab, { "b.weight" }, &hit));   /* one changed piece of a join */
    CHECK(!ckpt_reads_any(ab, { "q.weight", "exps" }));
    CHECK_EQ(hit.size(), (size_t)3);
    remove_checkpoint(dir);
}

/* A COPY's sources are alternatives, first present wins: a tied head reads the embedding where
 * the checkpoint ships no head of its own. */
TEST(alternatives_take_the_first_tensor_present) {
    const std::string dir = write_checkpoint({ bf16("embed", { 2, 2 }, 5.0f) }, "{}");
    Checkpoint ck;
    REQUIRE_EQ(ck.open(dir), RAD_OK);
    Program prog;
    prog.name_map.push_back(map("out", RAD_MAP_COPY, { "lm_head", "embed" }));
    prog.name_map.push_back(map("need", RAD_MAP_COPY, { "missing" }));
    CkptWeight w;
    std::string why;
    REQUIRE_EQ(ckpt_resolve(ck, prog, "out", false, &w, &why), RAD_OK);
    REQUIRE_EQ(w.pieces.size(), (size_t)1);
    CHECK_EQ(w.pieces[0]->name, std::string("embed"));
    CHECK_EQ(ckpt_resolve(ck, prog, "need", false, &w, &why), RAD_E_NOTFOUND);
    CHECK(why.find("missing") != std::string::npos);
    CHECK_EQ(ckpt_resolve(ck, prog, "need", true, &w, &why), RAD_E_NOTFOUND);
    CHECK_EQ(ckpt_resolve(ck, prog, "unmapped", false, &w, &why), RAD_E_NOTFOUND);
    remove_checkpoint(dir);
}

/* NOT TRIVIAL, AND SAID WHY: FP8 with no scale beside it, and pieces of different dtypes. */
TEST(what_is_not_served_as_it_is_says_why) {
    const std::string dir = write_checkpoint({ fp8("bare", { 128, 128 }),
                                               bf16("b16", { 128, 128 }, 0.0f) }, kConfig);
    Checkpoint ck;
    REQUIRE_EQ(ck.open(dir), RAD_OK);
    Program prog;
    prog.name_map.push_back(map("bare", RAD_MAP_COPY, { "bare" }));
    prog.name_map.push_back(map("mixed", RAD_MAP_CONCAT, { "bare", "b16" }));
    CkptWeight w;
    std::string why;
    REQUIRE_EQ(ckpt_resolve(ck, prog, "bare", false, &w, &why), RAD_OK);
    CHECK(!w.trivial);
    CHECK(w.why.find("no scale plane") != std::string::npos);
    REQUIRE_EQ(ckpt_resolve(ck, prog, "mixed", false, &w, &why), RAD_OK);
    CHECK(!w.trivial);
    remove_checkpoint(dir);
}

RAD_TEST_MAIN()

/* ================================================================== imported quantisations */
namespace {

/* A GPTQ-packed weight as auto_gptq's packer writes it: code (row n, column k) is the signed value
 * `q[n][k]`, stored unsigned with zero point 2^(bits-1) as field k % (32/bits) of int32 word
 * [k / (32/bits), n]; `qzeros` holds the v1 zero point (less one) in every field; `scales` is
 * f16 [K / group, N]. */
struct Gptq {
    std::vector<std::vector<int>> q;     /* [N][K] signed codes */
    std::vector<float> scale;            /* [G * N], row g of the scale tensor first */
    int64_t N = 0, K = 0, group = 0;
    int bits = 4;
};

Gptq make_gptq(int64_t N, int64_t K, int64_t group, int bits, int seed) {
    Gptq g;
    g.N = N; g.K = K; g.group = group; g.bits = bits;
    const int lo = -(1 << (bits - 1)), span = 1 << bits;
    g.q.assign((size_t)N, std::vector<int>((size_t)K));
    uint32_t x = 2463534242u ^ (uint32_t)seed;
    for (auto& row : g.q)
        for (int& v : row) {
            x ^= x << 13; x ^= x >> 17; x ^= x << 5;
            v = lo + (int)(x % (uint32_t)span);
        }
    /* f16 scales with low mantissa bits set and both signs, so the bf16 rounding and the sign
     * are both seen. */
    for (int64_t i = 0; i < (K / group) * N; ++i)
        g.scale.push_back((i % 3 == 1 ? -1.0f : 1.0f) * (0.0123f + 0.00071f * (float)i));
    return g;
}

void add_gptq(std::vector<T>* ts, const std::string& base, const Gptq& g, bool with_gidx = false) {
    const int64_t per = 32 / g.bits, G = g.K / g.group;
    T qw{ base + ".qweight", "I32", { g.K / per, g.N }, {} };
    qw.bytes.assign((size_t)(g.K / per * g.N * 4), 0);
    for (int64_t n = 0; n < g.N; ++n)
        for (int64_t k = 0; k < g.K; ++k) {
            const uint32_t u = (uint32_t)(g.q[(size_t)n][(size_t)k] + (1 << (g.bits - 1)));
            uint32_t* w = (uint32_t*)qw.bytes.data() + (k / per) * g.N + n;
            *w |= u << ((k % per) * g.bits);
        }
    T sc{ base + ".scales", "F16", { G, g.N }, {} };
    sc.bytes.resize((size_t)(G * g.N * 2));
    for (int64_t i = 0; i < G * g.N; ++i) rad_store_f32(sc.bytes.data(), RAD_F16, i, g.scale[(size_t)i]);
    T qz{ base + ".qzeros", "I32", { G, g.N * g.bits / 32 }, {} };
    qz.bytes.resize((size_t)(G * g.N * g.bits / 32 * 4));
    for (size_t i = 0; i < qz.bytes.size() / 4; ++i)
        ((uint32_t*)qz.bytes.data())[i] = g.bits == 4 ? 0x77777777u : 0x7f7f7f7fu;
    ts->push_back(qw);
    ts->push_back(sc);
    ts->push_back(qz);
    if (with_gidx) {
        T gi{ base + ".g_idx", "I32", { g.K }, {} };
        gi.bytes.resize((size_t)g.K * 4);
        for (int64_t k = 0; k < g.K; ++k) ((int32_t*)gi.bytes.data())[k] = (int32_t)(k / g.group);
        ts->push_back(gi);
    }
}

/* The scale the f16 tensor actually holds. */
float f16_of(float v) {
    uint16_t h;
    rad_store_f32(&h, RAD_F16, 0, v);
    return rad_load_f32(&h, RAD_F16, 0);
}

const char* kAutoRound =
    "{\"model_type\":\"qwen4_exp\",\"quantization_config\":{\"quant_method\":\"auto-round\","
    "\"packing_format\":\"auto_round:auto_gptq\",\"bits\":4,\"group_size\":128,\"sym\":true}}";

}  /* namespace */

/* AUTOROUND'S GPTQ EXPORT IS IMPORTED: gate and up -- each its own GPTQ-packed int4 weight --
 * become one `i4*bf16[1x128]` weight whose codes are the shipped ones, signed, row by row, and
 * whose scales are the shipped f16 ones rounded to bf16. Read as f32 it is the shipped value:
 * the code times the f16 scale. Codeberg #29. */
TEST(an_autoround_int4_weight_is_imported_as_its_codes_and_scales) {
    const Gptq gate = make_gptq(8, 256, 128, 4, 1), up = make_gptq(16, 256, 128, 4, 2);
    std::vector<T> ts;
    add_gptq(&ts, "e.0.gate_proj", gate);
    add_gptq(&ts, "e.0.up_proj", up, /*with_gidx=*/true);
    const std::string dir = write_checkpoint(ts, kAutoRound);
    Checkpoint ck;
    REQUIRE_EQ(ck.open(dir), RAD_OK);
    CHECK(ck.imports() == Checkpoint::Import::Gptq);
    CHECK_EQ(ck.meta().quant, std::string("auto-round"));
    Program prog;
    prog.name_map.push_back(map("gu", RAD_MAP_CONCAT, { "e.0.gate_proj.weight", "e.0.up_proj.weight" }));
    CkptWeight w;
    std::string why;
    REQUIRE_EQ(ckpt_resolve(ck, prog, "gu", false, &w, &why), RAD_OK);
    CHECK(!w.trivial);
    REQUIRE(w.imported);
    CHECK_EQ(w.rows, (int64_t)24);
    CHECK_EQ(w.cols, (int64_t)256);
    REQUIRE_EQ(w.enc.n_planes, 2);
    CHECK_EQ(w.enc.plane[0].dtype, (uint32_t)RAD_I4);
    CHECK_EQ(w.enc.plane[1].dtype, (uint32_t)RAD_BF16);
    CHECK_EQ(w.enc.plane[1].block[0], (int64_t)1);
    CHECK_EQ(w.enc.plane[1].block[1], (int64_t)128);
    CHECK(rad_enc_valid(&w.enc));

    /* every row, both pieces: codes, scales and the f32 value */
    std::vector<uint8_t> codes((size_t)(24 * 128)), scales((size_t)(24 * 2 * 2));
    REQUIRE_EQ(ckpt_plane_rows(w, 0, 0, 24, codes.data(), &why), RAD_OK);
    REQUIRE_EQ(ckpt_plane_rows(w, 1, 0, 24, scales.data(), &why), RAD_OK);
    std::vector<float> f((size_t)(24 * 256));
    REQUIRE_EQ(ckpt_rows_f32(w, 0, 24, f.data(), &why), RAD_OK);
    int bad_code = 0, bad_scale = 0, bad_f = 0;
    for (int64_t r = 0; r < 24; ++r) {
        const Gptq& g = r < 8 ? gate : up;
        const int64_t n = r < 8 ? r : r - 8;
        for (int64_t k = 0; k < 256; ++k) {
            const int q = g.q[(size_t)n][(size_t)k];
            if ((int)rad_load_f32(codes.data(), RAD_I4, r * 256 + k) != q) ++bad_code;
            const float s = f16_of(g.scale[(size_t)((k / 128) * g.N + n)]);
            if (f[(size_t)(r * 256 + k)] != (float)q * s) ++bad_f;
        }
        for (int64_t c = 0; c < 2; ++c) {
            const float s = f16_of(g.scale[(size_t)(c * g.N + n)]);
            if (rad_load_f32(scales.data(), RAD_BF16, r * 2 + c) !=
                rad_load_f32(std::vector<uint16_t>{ rad_f32_to_bf16(s) }.data(), RAD_BF16, 0))
                ++bad_scale;
        }
    }
    CHECK_EQ(bad_code, 0);
    CHECK_EQ(bad_scale, 0);
    CHECK_EQ(bad_f, 0);
    /* a part of the plane, from the middle of the second piece */
    std::vector<uint8_t> part(128);
    REQUIRE_EQ(ckpt_plane_rows(w, 0, 13, 1, part.data(), &why), RAD_OK);
    CHECK_EQ(std::memcmp(part.data(), codes.data() + 13 * 128, 128), 0);
    remove_checkpoint(dir);
}

/* AN INT8 PER-CHANNEL WEIGHT (one scale a row, g_idx all zero) is `i8*bf16[1x128]` with its row's
 * scale in every block. */
TEST(a_gptq_int8_per_channel_weight_repeats_its_scale) {
    const Gptq g = make_gptq(4, 384, 384, 8, 3);
    std::vector<T> ts;
    add_gptq(&ts, "o_proj", g, /*with_gidx=*/true);
    const std::string dir = write_checkpoint(ts, kAutoRound);
    Checkpoint ck;
    REQUIRE_EQ(ck.open(dir), RAD_OK);
    Program prog;
    prog.name_map.push_back(map("o", RAD_MAP_COPY, { "o_proj.weight" }));
    CkptWeight w;
    std::string why;
    REQUIRE_EQ(ckpt_resolve(ck, prog, "o", false, &w, &why), RAD_OK);
    REQUIRE(w.imported);
    CHECK_EQ(w.enc.plane[0].dtype, (uint32_t)RAD_I8);
    std::vector<uint8_t> codes(4 * 384), scales(4 * 3 * 2);
    REQUIRE_EQ(ckpt_plane_rows(w, 0, 0, 4, codes.data(), &why), RAD_OK);
    REQUIRE_EQ(ckpt_plane_rows(w, 1, 0, 4, scales.data(), &why), RAD_OK);
    int bad = 0;
    for (int64_t n = 0; n < 4; ++n) {
        for (int64_t k = 0; k < 384; ++k)
            if ((int8_t)codes[(size_t)(n * 384 + k)] != g.q[(size_t)n][(size_t)k]) ++bad;
        for (int c = 0; c < 3; ++c)
            if (rad_load_f32(scales.data(), RAD_BF16, n * 3 + c) !=
                rad_load_f32(scales.data(), RAD_BF16, n * 3))
                ++bad;
    }
    CHECK_EQ(bad, 0);
    remove_checkpoint(dir);
}

/* WHAT AN IMPORT CANNOT READ IT REFUSES BY NAME: an asymmetric zero point, an act-order group
 * index, a group narrower than the served 128, and a packing it does not know. */
TEST(an_import_refuses_what_it_cannot_read) {
    auto refused = [](void (*edit)(std::vector<T>*), const char* config, const char* expect) {
        Gptq g = make_gptq(8, 256, 128, 4, 4);
        std::vector<T> ts;
        add_gptq(&ts, "x", g, true);
        edit(&ts);
        const std::string dir = write_checkpoint(ts, config);
        Checkpoint ck;
        REQUIRE_EQ(ck.open(dir), RAD_OK);
        Program prog;
        prog.name_map.push_back(map("x", RAD_MAP_COPY, { "x.weight" }));
        CkptWeight w;
        std::string why;
        const int rc = ckpt_resolve(ck, prog, "x", false, &w, &why);
        CHECK(rc < 0);
        CHECK(why.find(expect) != std::string::npos);
        remove_checkpoint(dir);
    };
    refused([](std::vector<T>* ts) { (*ts)[2].bytes[0] = 0x76; }, kAutoRound, "symmetric");
    refused([](std::vector<T>* ts) { ((int32_t*)(*ts)[3].bytes.data())[0] = 1; }, kAutoRound,
            "act-order");

    /* group 64: four scale rows for 256 columns */
    {
        Gptq g = make_gptq(8, 256, 64, 4, 5);
        std::vector<T> ts;
        add_gptq(&ts, "x", g);
        const std::string dir = write_checkpoint(ts, kAutoRound);
        Checkpoint ck;
        REQUIRE_EQ(ck.open(dir), RAD_OK);
        Program prog;
        prog.name_map.push_back(map("x", RAD_MAP_COPY, { "x.weight" }));
        CkptWeight w;
        std::string why;
        CHECK(ckpt_resolve(ck, prog, "x", false, &w, &why) < 0);
        CHECK(why.find("multiple of 128") != std::string::npos);
        remove_checkpoint(dir);
    }
    /* AutoRound's own packing is another layout, and the checkpoint is not imported at all. */
    {
        const std::string dir = write_checkpoint(
            { bf16("y", { 2, 2 }, 0) },
            "{\"model_type\":\"qwen4_exp\",\"quantization_config\":{\"quant_method\":"
            "\"auto-round\",\"packing_format\":\"auto_round\"}}");
        Checkpoint ck;
        REQUIRE_EQ(ck.open(dir), RAD_OK);
        CHECK(ck.imports() == Checkpoint::Import::None);
        CHECK(ck.import_refusal().find("auto_round:auto_gptq") != std::string::npos);
        remove_checkpoint(dir);
    }
}

/* QUARK'S MXFP4 IS `fp4_e2m1*e8m0[1x32]` BYTE FOR BYTE, and so trivial: the U8 codes and U8
 * exponents are the planes, gate and up join along rows, and the f32 value is the e2m1 code times
 * 2^(e - 127). */
TEST(a_quark_mxfp4_weight_is_its_codes_and_exponents) {
    const int64_t K = 64;
    auto codes = [&](const std::string& n, int64_t N, int seed) {
        T t{ n, "U8", { N, K / 2 }, {} };
        for (int64_t i = 0; i < N * K / 2; ++i) t.bytes.push_back((uint8_t)((i * 37 + seed) & 0xff));
        return t;
    };
    auto exps = [&](const std::string& n, int64_t N, int seed) {
        T t{ n, "U8", { N, K / 32 }, {} };
        for (int64_t i = 0; i < N * K / 32; ++i) t.bytes.push_back((uint8_t)(120 + (i + seed) % 9));
        return t;
    };
    const T g = codes("g.weight", 2, 1), gs = exps("g.weight_scale", 2, 1);
    const T u = codes("u.weight", 3, 5), us = exps("u.weight_scale", 3, 4);
    const std::string dir = write_checkpoint(
        { g, gs, u, us, bf16("n.weight", { 4 }, 1.0f) },
        "{\"model_type\":\"qwen3_5\",\"quantization_config\":{\"quant_method\":\"quark\","
        "\"global_quant_config\":{\"weight\":{\"dtype\":\"fp4\",\"qscheme\":\"per_group\","
        "\"group_size\":32,\"scale_format\":\"e8m0\"}},\"layer_quant_config\":{}}}");
    Checkpoint ck;
    REQUIRE_EQ(ck.open(dir), RAD_OK);
    CHECK(ck.imports() == Checkpoint::Import::QuarkMx);
    Program prog;
    prog.name_map.push_back(map("gu", RAD_MAP_CONCAT, { "g.weight", "u.weight" }));
    prog.name_map.push_back(map("n", RAD_MAP_COPY, { "n.weight" }));
    CkptWeight w;
    std::string why;
    REQUIRE_EQ(ckpt_resolve(ck, prog, "gu", false, &w, &why), RAD_OK);
    REQUIRE(w.trivial);
    CHECK(!w.imported);
    CHECK_EQ(w.rows, (int64_t)5);
    CHECK_EQ(w.cols, K);
    REQUIRE_EQ(w.enc.n_planes, 2);
    CHECK_EQ(w.enc.plane[0].dtype, (uint32_t)RAD_FP4E2M1);
    CHECK_EQ(w.enc.plane[1].dtype, (uint32_t)RAD_E8M0);
    CHECK_EQ(w.enc.plane[1].block[1], (int64_t)32);
    CHECK(rad_enc_valid(&w.enc));
    std::vector<uint8_t> cp(5 * K / 2), sp(5 * K / 32);
    REQUIRE_EQ(ckpt_plane_rows(w, 0, 0, 5, cp.data(), &why), RAD_OK);
    REQUIRE_EQ(ckpt_plane_rows(w, 1, 0, 5, sp.data(), &why), RAD_OK);
    CHECK_EQ(std::memcmp(cp.data(), g.bytes.data(), g.bytes.size()), 0);
    CHECK_EQ(std::memcmp(cp.data() + g.bytes.size(), u.bytes.data(), u.bytes.size()), 0);
    CHECK_EQ(std::memcmp(sp.data() + gs.bytes.size(), us.bytes.data(), us.bytes.size()), 0);
    std::vector<float> f((size_t)K);
    REQUIRE_EQ(ckpt_rows_f32(w, 3, 1, f.data(), &why), RAD_OK);       /* u's row 1 */
    int bad = 0;
    for (int64_t k = 0; k < K; ++k) {
        const uint8_t b = u.bytes[(size_t)(K / 2 + k / 2)];
        const float code = rad_fp4e2m1_to_f32((uint8_t)(k % 2 ? b >> 4 : b & 15));
        const float sc = std::ldexp(1.0f, (int)us.bytes[(size_t)(K / 32 + k / 32)] - 127);
        if (f[(size_t)k] != code * sc) ++bad;
    }
    CHECK_EQ(bad, 0);
    /* an excluded layer is a plain tensor */
    REQUIRE_EQ(ckpt_resolve(ck, prog, "n", false, &w, &why), RAD_OK);
    CHECK(w.trivial && rad_enc_is(&w.enc, "plain"));
    remove_checkpoint(dir);
}

/* A DIRECTORY THAT MERGES TWO CHECKPOINTS holds some tensors in two shards: a finetune's changed
 * tensors in shards of its own beside links to every shard of the base model, which still carry
 * the old copies. The index is what says which copy is the model's, so it is what decides; a name
 * in two shards that no index entry settles is refused rather than read from whichever shard sorts
 * first. */
TEST(the_index_decides_which_shard_a_tensor_is_read_from) {
    const std::string dir = dir_path();
    ::mkdir(dir.c_str(), 0755);
    const T base_x = bf16("x.weight", { 2, 4 }, 1.0f), base_y = bf16("y.weight", { 4 }, 2.0f);
    const T ft_x = bf16("x.weight", { 2, 4 }, 5.0f);
    write_shard(dir + "/a.safetensors", { base_x, base_y });   /* sorts first */
    write_shard(dir + "/b.safetensors", { ft_x });
    auto put = [&](const char* name, const std::string& body) {
        FILE* f = std::fopen((dir + "/" + name).c_str(), "wb");
        std::fwrite(body.data(), 1, body.size(), f);
        std::fclose(f);
    };
    put("config.json", kConfig);
    put("model.safetensors.index.json",
        R"({"weight_map":{"x.weight":"b.safetensors","y.weight":"a.safetensors"}})");
    {
        Checkpoint ck;
        REQUIRE_EQ(ck.open(dir), RAD_OK);
        const SrcTensor* x = ck.find("x.weight");
        const SrcTensor* y = ck.find("y.weight");
        REQUIRE(x && y);
        CHECK_EQ(std::memcmp(x->data, ft_x.bytes.data(), ft_x.bytes.size()), 0);
        CHECK_EQ(std::memcmp(y->data, base_y.bytes.data(), base_y.bytes.size()), 0);
    }
    /* the index names both shards but not x: two copies and nothing to choose between them */
    put("model.safetensors.index.json",
        R"({"weight_map":{"y.weight":"a.safetensors","z.weight":"b.safetensors"}})");
    {
        Checkpoint ck;
        CHECK_EQ(ck.open(dir), RAD_E_FORMAT);
    }
    for (const char* f : { "a.safetensors", "b.safetensors", "config.json",
                           "model.safetensors.index.json" })
        std::remove((dir + "/" + f).c_str());
    ::rmdir(dir.c_str());
}

/* PAROQUANT MXFP4 is Quark's codes and exponents in a rotated basis plus the rotations: each
 * projection's pairs (group-local int16), theta and channel scales. Gate and up are two PARTS of
 * the fused weight, and the tables must say so: pairs made whole-row, angles widened, the channel
 * scale inverted, the part ends. Decoding the planes through the core must then give a weight
 * that, rotated forward again, is the codes the checkpoint shipped. */
TEST(a_paroquant_mxfp4_weight_carries_its_rotations_as_parts) {
    const int64_t K = 128, S = 2;
    uint32_t rng = 12345;
    auto next = [&]() { rng = rng * 1664525u + 1013904223u; return rng >> 8; };
    auto codes = [&](const std::string& n, int64_t N) {
        T t{ n, "U8", { N, K / 2 }, {} };
        for (int64_t i = 0; i < N * K / 2; ++i) t.bytes.push_back((uint8_t)(next() & 0xff));
        return t;
    };
    auto exps = [&](const std::string& n, int64_t N) {
        T t{ n, "U8", { N, K / 32 }, {} };
        for (int64_t i = 0; i < N * K / 32; ++i) t.bytes.push_back((uint8_t)(122 + next() % 5));
        return t;
    };
    auto pairs = [&](const std::string& n) {
        T t{ n, "I16", { S, K }, {} };
        for (int64_t s = 0; s < S; ++s) {
            std::vector<int16_t> perm((size_t)K);
            for (int64_t i = 0; i < K; ++i) perm[(size_t)i] = (int16_t)i;
            for (int64_t i = K - 1; i > 0; --i) std::swap(perm[(size_t)i], perm[next() % (i + 1)]);
            for (int64_t i = 0; i < K; ++i) {
                t.bytes.push_back((uint8_t)(perm[(size_t)i] & 0xff));
                t.bytes.push_back((uint8_t)(perm[(size_t)i] >> 8));
            }
        }
        return t;
    };
    auto f16s = [&](const std::string& n, std::vector<int64_t> shape, float lo, float hi) {
        T t{ n, "F16", shape, {} };
        int64_t cnt = 1;
        for (int64_t d : shape) cnt *= d;
        for (int64_t i = 0; i < cnt; ++i) {
            const float f = lo + (hi - lo) * (float)(next() % 1000) / 1000.0f;
            const uint16_t h = rad_f32_to_f16(f);
            t.bytes.push_back((uint8_t)(h & 0xff));
            t.bytes.push_back((uint8_t)(h >> 8));
        }
        return t;
    };
    const T gw = codes("g.weight", 16), gs = exps("g.weight_scale", 16), gp = pairs("g.pairs"),
            gt = f16s("g.theta", { S, K / 2 }, -3.0f, 3.0f), gc = f16s("g.channel_scales", { 1, K }, 0.5f, 2.0f);
    const T uw = codes("u.weight", 16), us = exps("u.weight_scale", 16), up = pairs("u.pairs"),
            ut = f16s("u.theta", { S, K / 2 }, -3.0f, 3.0f), uc = f16s("u.channel_scales", { 1, K }, 0.5f, 2.0f);
    // a per-row fp8 head beside them: weight_scale is one f32 a row
    T hs{ "h.weight_scale", "F32", { 4 }, {} };
    for (int i = 0; i < 4; ++i) { float f = 0.25f * (float)(i + 1); hs.bytes.resize(hs.bytes.size() + 4); std::memcpy(hs.bytes.data() + 4 * i, &f, 4); }
    const std::string dir = write_checkpoint(
        { gw, gs, gp, gt, gc, uw, us, up, ut, uc, fp8("h.weight", { 4, 8 }), hs },
        "{\"model_type\":\"qwen3_5\",\"quantization_config\":{\"quant_method\":\"paroquant_mxfp4\","
        "\"format\":\"mxfp4\",\"bits\":4,\"group_size\":128,\"mxfp4_block\":32,\"krot\":2}}");
    Checkpoint ck;
    REQUIRE_EQ(ck.open(dir), RAD_OK);
    CHECK(ck.imports() == Checkpoint::Import::ParoMx);
    Program prog;
    prog.name_map.push_back(map("gu", RAD_MAP_CONCAT, { "g.weight", "u.weight" }));
    prog.name_map.push_back(map("h", RAD_MAP_COPY, { "h.weight" }));
    CkptWeight w;
    std::string why;
    REQUIRE_EQ(ckpt_resolve(ck, prog, "gu", false, &w, &why), RAD_OK);
    REQUIRE(w.paro);
    CHECK(w.imported && !w.trivial);
    REQUIRE(rad_enc_valid(&w.enc));
    CHECK_EQ(std::string(w.enc.transform), std::string("givens"));
    CHECK_EQ(rad_enc_givens_parts(&w.enc), (int64_t)2);
    CHECK_EQ(w.rows, (int64_t)32);
    REQUIRE_EQ(w.enc.n_planes, 6);

    std::vector<std::vector<uint8_t>> pl((size_t)w.enc.n_planes);
    for (int k = 0; k < w.enc.n_planes; ++k) {
        int64_t pr = 0, pc = 0;
        rad_enc_plane_dims(&w.enc.plane[k], w.rows, w.cols, &pr, &pc);
        pl[(size_t)k].resize((size_t)(pr * rad_enc_row_bytes(w.enc.plane[k].dtype, pc)));
        REQUIRE_EQ(ckpt_plane_rows(w, k, 0, pr, pl[(size_t)k].data(), &why), RAD_OK);
    }
    // codes and exponents are the checkpoint's bytes, joined
    CHECK_EQ(std::memcmp(pl[0].data(), gw.bytes.data(), gw.bytes.size()), 0);
    CHECK_EQ(std::memcmp(pl[0].data() + gw.bytes.size(), uw.bytes.data(), uw.bytes.size()), 0);
    CHECK_EQ(std::memcmp(pl[1].data() + gs.bytes.size(), us.bytes.data(), us.bytes.size()), 0);
    const int32_t* tp = (const int32_t*)pl[2].data();
    const float* ta = (const float*)pl[3].data();
    const float* tsc = (const float*)pl[4].data();
    const int32_t* tpart = (const int32_t*)pl[5].data();
    CHECK_EQ(tpart[0], 16);
    CHECK_EQ(tpart[1], 32);
    for (int64_t c = 0; c < K; ++c) {
        CHECK_EQ(tp[c], (int32_t)(*(const int16_t*)(gp.bytes.data() + 2 * c)));             // part 0, stage 0
        CHECK_EQ(tp[(S + 1) * K + c], (int32_t)(*(const int16_t*)(up.bytes.data() + 2 * (K + c))));  // part 1, stage 1
        const uint16_t h = *(const uint16_t*)(gc.bytes.data() + 2 * c);
        CHECK(std::fabs(tsc[c] - 1.0f / rad_f16_to_f32(h)) < 1e-6f);
    }
    CHECK(std::fabs(ta[0] - rad_f16_to_f32(*(const uint16_t*)gt.bytes.data())) == 0.0f);

    // decode, rotate forward again with the checkpoint's own convention, compare to the codes
    const void* data[6];
    for (int k = 0; k < 6; ++k) data[k] = pl[(size_t)k].data();
    EncodedView v;
    enc_view(w.enc, w.rows, w.cols, data, &v);
    std::vector<float> dec((size_t)(w.rows * K));
    REQUIRE_EQ(enc_decode_rows(v, 0, w.rows, dec.data(), &why), RAD_OK);
    const T* cw[2] = { &gw, &uw };
    const T* es[2] = { &gs, &us };
    const T* pp[2] = { &gp, &up };
    const T* th[2] = { &gt, &ut };
    const T* cc[2] = { &gc, &uc };
    double worst = 0;
    for (int64_t r = 0; r < w.rows; ++r) {
        const int part = r >= 16, rr = r - 16 * part;
        std::vector<double> x((size_t)K);
        for (int64_t k = 0; k < K; ++k)
            x[(size_t)k] = (double)dec[(size_t)(r * K + k)] *
                           (1.0 / rad_f16_to_f32(*(const uint16_t*)(cc[part]->bytes.data() + 2 * k)));
        for (int64_t s = 0; s < S; ++s)
            for (int64_t p = 0; p < K / 2; ++p) {
                const int16_t* pv = (const int16_t*)pp[part]->bytes.data();
                const int64_t a = pv[s * K + 2 * p], b = pv[s * K + 2 * p + 1];
                const double t = rad_f16_to_f32(*(const uint16_t*)(th[part]->bytes.data() + 2 * (s * (K / 2) + p)));
                const double c = std::cos(t), sn = std::sin(t);
                const double xa = x[(size_t)a], xb = x[(size_t)b];
                x[(size_t)a] = xa * c + xb * sn;
                x[(size_t)b] = xb * c - xa * sn;
            }
        for (int64_t k = 0; k < K; ++k) {
            const uint8_t bt = cw[part]->bytes[(size_t)(rr * K / 2 + k / 2)];
            const float code = rad_fp4e2m1_to_f32((uint8_t)(k % 2 ? bt >> 4 : bt & 15));
            const float sc = std::ldexp(1.0f, (int)es[part]->bytes[(size_t)(rr * K / 32 + k / 32)] - 127);
            worst = std::max(worst, std::fabs(x[(size_t)k] - (double)code * sc));
        }
    }
    CHECK(worst < 1e-4);

    // the per-row fp8 head: one f32 scale a row, read as [N, 1]
    REQUIRE_EQ(ckpt_resolve(ck, prog, "h", false, &w, &why), RAD_OK);
    REQUIRE(w.trivial);
    CHECK_EQ(w.enc.plane[1].block[0], (int64_t)1);
    CHECK_EQ(w.enc.plane[1].block[1], (int64_t)0);
    std::vector<float> f(4 * 8);
    REQUIRE_EQ(ckpt_rows_f32(w, 0, 4, f.data(), &why), RAD_OK);
    for (int r = 0; r < 4; ++r) {
        const float q = rad_load_f32(fp8("h.weight", { 4, 8 }).bytes.data(), RAD_F8E4M3, r * 8);
        CHECK_EQ(f[(size_t)(r * 8)], q * 0.25f * (float)(r + 1));
    }
    remove_checkpoint(dir);
}

/* PAROQUANT INT5-BITPLANE: the low nibbles AWQ-packed along N (qweight [K, N/8], eight columns a
 * word in the order 0,2,4,6,1,3,5,7), a fifth-bit plane (qweight_hi [K, N/32], bit n % 32) and the
 * same two packings of the zero points. The logical [N, K] weight comes out as u5 codes (a bit
 * stream along K), an f16 scale and a u8 zero a 128 of K, plus the rotation as `givens` tables. */
TEST(a_paroquant_int5_weight_is_relaid_row_by_row) {
    const int64_t N = 64, K = 256, G = K / 128;
    uint32_t rng = 777;
    auto next = [&]() { rng = rng * 1664525u + 1013904223u; return rng >> 8; };
    std::vector<int> code((size_t)(N * K)), zero((size_t)(N * G));
    for (auto& c : code) c = (int)(next() % 32);
    for (auto& z : zero) z = (int)(next() % 32);
    static const int kOrder[8] = { 0, 2, 4, 6, 1, 3, 5, 7 };
    auto pack_awq = [&](const std::vector<int>& v, int64_t rows) {       // v[n * rows + r] -> [rows, N/8]
        T t{ "", "I32", { rows, N / 8 }, {} };
        t.bytes.assign((size_t)(rows * N / 8 * 4), 0);
        for (int64_t r = 0; r < rows; ++r)
            for (int64_t n = 0; n < N; ++n) {
                int j = 0;
                while (kOrder[j] != n % 8) ++j;
                uint32_t w;
                uint8_t* p = t.bytes.data() + (r * (N / 8) + n / 8) * 4;
                std::memcpy(&w, p, 4);
                w |= (uint32_t)(v[(size_t)(n * rows + r)] & 15) << (4 * j);
                std::memcpy(p, &w, 4);
            }
        return t;
    };
    auto pack_hi = [&](const std::vector<int>& v, int64_t rows) {
        T t{ "", "I32", { rows, N / 32 }, {} };
        t.bytes.assign((size_t)(rows * N / 32 * 4), 0);
        for (int64_t r = 0; r < rows; ++r)
            for (int64_t n = 0; n < N; ++n) {
                uint32_t w;
                uint8_t* p = t.bytes.data() + (r * (N / 32) + n / 32) * 4;
                std::memcpy(&w, p, 4);
                w |= (uint32_t)((v[(size_t)(n * rows + r)] >> 4) & 1) << (n % 32);
                std::memcpy(p, &w, 4);
            }
        return t;
    };
    T qw = pack_awq(code, K), qwh = pack_hi(code, K), qz = pack_awq(zero, G), qzh = pack_hi(zero, G);
    qw.name = "q.qweight"; qwh.name = "q.qweight_hi"; qz.name = "q.qzeros"; qzh.name = "q.qzeros_hi";
    T sc{ "q.scales", "F16", { G, N }, {} };
    for (int64_t i = 0; i < G * N; ++i) {
        const uint16_t h = rad_f32_to_f16(0.01f * (float)(1 + next() % 50));
        sc.bytes.push_back((uint8_t)(h & 0xff));
        sc.bytes.push_back((uint8_t)(h >> 8));
    }
    T pr{ "q.pairs", "I16", { 2, K }, {} };
    for (int64_t s = 0; s < 2; ++s)
        for (int64_t i = 0; i < K; ++i) {
            const int16_t v = (int16_t)(i % 128 ^ 1);                       // adjacent swap: a pairing
            pr.bytes.push_back((uint8_t)(v & 0xff));
            pr.bytes.push_back((uint8_t)(v >> 8));
        }
    T th{ "q.theta", "F16", { 2, K / 2 }, {} }, cs{ "q.channel_scales", "F16", { 1, K }, {} };
    for (int64_t i = 0; i < 2 * K / 2; ++i) { const uint16_t h = rad_f32_to_f16(0.3f); th.bytes.push_back(h & 0xff); th.bytes.push_back(h >> 8); }
    for (int64_t i = 0; i < K; ++i) { const uint16_t h = rad_f32_to_f16(1.5f); cs.bytes.push_back(h & 0xff); cs.bytes.push_back(h >> 8); }
    const std::string dir = write_checkpoint(
        { qw, qwh, qz, qzh, sc, pr, th, cs },
        "{\"model_type\":\"qwen3_5\",\"quantization_config\":{\"quant_method\":\"paroquant\","
        "\"bits\":5,\"group_size\":128,\"krot\":2,\"format\":\"int5-bitplane\"}}");
    Checkpoint ck;
    REQUIRE_EQ(ck.open(dir), RAD_OK);
    CHECK(ck.imports() == Checkpoint::Import::ParoI5);
    Program prog;
    prog.name_map.push_back(map("w", RAD_MAP_COPY, { "q.weight" }));
    CkptWeight w;
    std::string why;
    REQUIRE_EQ(ckpt_resolve(ck, prog, "w", false, &w, &why), RAD_OK);
    REQUIRE(w.paro && w.imported);
    CHECK_EQ(w.rows, N);
    CHECK_EQ(w.cols, K);
    REQUIRE(rad_enc_valid(&w.enc));
    CHECK_EQ(w.enc.plane[0].dtype, (uint32_t)RAD_U5);
    CHECK_EQ(w.enc.plane[1].dtype, (uint32_t)RAD_F16);
    CHECK_EQ(w.enc.plane[2].dtype, (uint32_t)RAD_U8);
    std::vector<std::vector<uint8_t>> pl((size_t)w.enc.n_planes);
    for (int k = 0; k < w.enc.n_planes; ++k) {
        int64_t pr_ = 0, pc = 0;
        rad_enc_plane_dims(&w.enc.plane[k], w.rows, w.cols, &pr_, &pc);
        pl[(size_t)k].assign((size_t)(pr_ * rad_enc_row_bytes(w.enc.plane[k].dtype, pc)) + 8, 0);
        REQUIRE_EQ(ckpt_plane_rows(w, k, 0, pr_, pl[(size_t)k].data(), &why), RAD_OK);
    }
    const int64_t rb = rad_enc_row_bytes(RAD_U5, K);
    int bad = 0;
    for (int64_t n = 0; n < N; ++n) {
        for (int64_t k = 0; k < K; ++k)
            if ((int)rad_load_f32(pl[0].data() + n * rb, RAD_U5, k) != code[(size_t)(n * K + k)]) ++bad;
        for (int64_t g = 0; g < G; ++g) {
            if (pl[2][(size_t)(n * G + g)] != zero[(size_t)(n * G + g)]) ++bad;
            if (std::memcmp(pl[1].data() + (n * G + g) * 2, sc.bytes.data() + (g * N + n) * 2, 2) != 0) ++bad;
        }
    }
    CHECK_EQ(bad, 0);
    remove_checkpoint(dir);
}
