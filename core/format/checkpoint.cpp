/* checkpoint.cpp -- see checkpoint.h. */
#include "checkpoint.h"
#include "rad_plugin.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <map>

namespace rad {

/* ================================================================== metadata */

void MetaOwn::set(const std::string& k, const std::string& v) {
    for (size_t i = 0; i < keys.size(); ++i)
        if (keys[i] == k) { vals[i] = v; return; }
    kv(k, v);
}

bool MetaOwn::has(const std::string& k) const {
    return std::find(keys.begin(), keys.end(), k) != keys.end();
}

const char* MetaOwn::get(const std::string& k) const {
    for (size_t i = 0; i < keys.size(); ++i)
        if (keys[i] == k) return vals[i].c_str();
    return nullptr;
}

void MetaOwn::finish() {
    kp.clear();
    vp.clear();
    for (auto& s : keys) kp.push_back(s.c_str());
    for (auto& s : vals) vp.push_back(s.c_str());
    m.arch_id = arch.c_str();
    m.name    = name.c_str();
    m.quant   = quant.c_str();
    m.n_kv    = (int)keys.size();
    m.kv_key  = kp.data();
    m.kv_val  = vp.data();
}

/* ---------------------------------------------------------------- config.json
 *
 * A safetensors checkpoint's architecture and dimensions live in config.json. The core parses it,
 * since the plugin only sees RadModelMeta, whose typed fields size everything at declare.
 *
 * A text-only model puts its dimensions at the top level; a multimodal one nests them under
 * `text_config` beside `vision_config`. Every key is flattened into the free-form list by dotted
 * path, and typed fields are read from `text_config.X`, falling back to `X`, so a plugin finds
 * `vision_config.depth` under that name either way.
 */
static void flatten_json(const nlohmann::json& j, const std::string& prefix, MetaOwn* out) {
    for (auto it = j.begin(); it != j.end(); ++it) {
        const std::string key = prefix.empty() ? it.key() : prefix + "." + it.key();
        const nlohmann::json& v = it.value();
        if (v.is_object())            flatten_json(v, key, out);
        else if (v.is_boolean())      out->kv(key, v.get<bool>() ? "1" : "0");
        else if (v.is_number_integer()) out->kv(key, fmt("%lld", (long long)v.get<int64_t>()));
        else if (v.is_number())       out->kv(key, fmt("%.17g", v.get<double>()));
        else if (v.is_string())       out->kv(key, v.get<std::string>());
        else if (v.is_array()) {
            /* An array of scalars becomes a space-separated string (mrope_section "11 11 10",
             * layer_types "linear_attention full_attention ..."). Arrays of objects are
             * skipped. */
            std::string s;
            bool ok = true;
            for (const auto& e : v) {
                if (!s.empty()) s += ' ';
                if (e.is_number_integer()) s += fmt("%lld", (long long)e.get<int64_t>());
                else if (e.is_number())    s += fmt("%.17g", e.get<double>());
                else if (e.is_string())    s += e.get<std::string>();
                else if (e.is_boolean())   s += e.get<bool>() ? "1" : "0";
                else { ok = false; break; }
            }
            if (ok) out->kv(key, s);
        }
    }
}

/* A second config.json (a drafter's), flattened under a prefix only. No typed extraction, which
 * would size the trunk from the draft head; the plugin reads `draft.` keys from the free-form
 * table. No `text_config` aliasing either: a drafter is a text model, and aliasing would collide
 * with the target's bare keys. */
int meta_merge_config_json(const std::string& dir, const std::string& prefix, MetaOwn* out) {
    const std::string path = dir + "/config.json";
    std::ifstream f(path);
    if (!f) {
        RAD_ERR("%s: no config.json. A draft checkpoint carries its own geometry there -- the "
                "block size it was trained for, which trunk layers it reads, the selector's rank "
                "-- and none of it is derivable from the target.", dir.c_str());
        return RAD_E_NOTFOUND;
    }
    nlohmann::json j;
    try { f >> j; }
    catch (const std::exception& e) { RAD_ERR("%s: %s", path.c_str(), e.what()); return RAD_E_FORMAT; }
    flatten_json(j, prefix, out);
    return RAD_OK;
}

/* ---------------------------------------------------------------- generation_config.json
 *
 * The checkpoint's sampling defaults, carried into the container so serving does not depend on a
 * separate file. Optional. Without them a model tuned with a truncated tail occasionally samples
 * far down its distribution, including end-of-turn, so replies stop early. Flattened under
 * `generation.` to avoid colliding with architecture keys such as `temperature` or `top_k`. */
int meta_merge_generation_config(const std::string& dir, MetaOwn* out) {
    const std::string path = dir + "/generation_config.json";
    std::ifstream f(path);
    if (!f) return RAD_OK;               /* optional */
    nlohmann::json j;
    try { f >> j; }
    catch (const std::exception& e) {
        RAD_WARN("%s: %s; no sampling defaults are carried", path.c_str(), e.what());
        return RAD_OK;
    }
    if (!j.is_object()) {
        RAD_WARN("%s: not a JSON object; no sampling defaults are carried", path.c_str());
        return RAD_OK;
    }
    flatten_json(j, "generation", out);
    return RAD_OK;
}

/* ---------------------------------------------------------------- the preprocessor configs
 *
 * How images and video become patches, carried into the container like the sampling defaults.
 * core/mm reads pixel bands, normalisation and frame rate from these; the encoder's geometry is
 * in config.json's vision_config. Each file has its own prefix (`size.shortest_edge` differs
 * between them). Optional. */
int meta_merge_preprocessor_configs(const std::string& dir, MetaOwn* out) {
    /* processor_config.json is the newer single file, one object per sub-processor
     * (`image_processor`, `video_processor`, `feature_extractor`) beside its own keys
     * (EmbeddingGemma 2's image budget and audio token rate are only there). */
    static const struct { const char* file; const char* prefix; } kFiles[] = {
        { "preprocessor_config.json", "preprocessor" },
        { "video_preprocessor_config.json", "video_preprocessor" },
        { "processor_config.json", "processor" },
        /* sentence-transformers: an embedding model's named prompts and default. */
        { "config_sentence_transformers.json", "sentence_transformers" },
    };
    for (const auto& f : kFiles) {
        const std::string path = dir + "/" + f.file;
        std::ifstream in(path);
        if (!in) continue;
        nlohmann::json j;
        try { in >> j; }
        catch (const std::exception& e) {
            RAD_WARN("%s: %s; it is not carried", path.c_str(), e.what());
            continue;
        }
        if (j.is_object()) flatten_json(j, f.prefix, out);
    }
    return RAD_OK;
}

int meta_from_config_json(const std::string& dir, MetaOwn* out) {
    const std::string path = dir + "/config.json";
    std::ifstream f(path);
    if (!f) {
        RAD_ERR("%s: no config.json. A safetensors checkpoint carries its architecture and its "
                "dimensions there; without it nothing can be sized.", dir.c_str());
        return RAD_E_NOTFOUND;
    }
    nlohmann::json j;
    try { f >> j; }
    catch (const std::exception& e) {
        RAD_ERR("%s: %s", path.c_str(), e.what());
        return RAD_E_FORMAT;
    }

    flatten_json(j, "", out);

    /* Alias text_config.X to a bare X where no bare X exists, so plugins read the same free-form
     * keys (e.g. `partial_rotary_factor`) with or without a vision tower. A bare key wins where
     * both exist. */
    {
        const size_t n = out->keys.size();
        for (size_t i = 0; i < n; ++i) {
            const std::string& k = out->keys[i];
            if (k.rfind("text_config.", 0) != 0) continue;
            const std::string bare = k.substr(12);
            bool have = false;
            for (size_t j2 = 0; j2 < n; ++j2) if (out->keys[j2] == bare) { have = true; break; }
            if (!have) out->kv(bare, out->vals[i]);
        }
    }

    auto gs = [&](const char* k) -> const char* { return out->get(k); };
    /* text_config first, then the top level. */
    auto gi = [&](const char* k, int64_t dflt) -> int64_t {
        const char* v = gs((std::string("text_config.") + k).c_str());
        if (!v) v = gs(k);
        return v ? strtoll(v, nullptr, 10) : dflt;
    };
    auto gf = [&](const char* k, double dflt) -> double {
        const char* v = gs((std::string("text_config.") + k).c_str());
        if (!v) v = gs(k);
        return v ? strtod(v, nullptr) : dflt;
    };

    /* HuggingFace's model_type minus underscores is the architecture id, matching GGUF's spelling
     * (`qwen3_5` is `qwen35`). */
    if (const char* mt = gs("model_type")) {
        out->arch.clear();
        for (const char* p = mt; *p; ++p) if (*p != '_') out->arch += *p;
        /* LFM2-VL names the wrapper, not the text model; one plugin serves LFM2 with or without
         * its tower. `model_type` stays in the free-form keys. */
        if (out->arch == "lfm2vl") out->arch = "lfm2";
    }
    if (const char* nm = gs("_name_or_path")) out->name = nm;

    RadModelMeta& m = out->m;
    m.n_layers    = gi("num_hidden_layers", 0);
    m.n_embd      = gi("hidden_size", 0);
    m.n_head      = gi("num_attention_heads", 0);
    m.n_head_kv   = gi("num_key_value_heads", m.n_head);
    /* Read head_dim; derive it only as a fallback. On qwen35 it is 256 while
     * hidden_size/num_attention_heads is 213. */
    m.head_dim    = gi("head_dim", m.n_head ? m.n_embd / m.n_head : 0);
    /* A routed config has no `intermediate_size`, only `moe_intermediate_size` (one expert's
     * width, which is what a GEMM sees and what n_ff means) and
     * `shared_expert_intermediate_size` (read by the plugin from the free-form table). */
    m.n_ff        = gi("intermediate_size", gi("moe_intermediate_size", 0));
    m.n_vocab     = gi("vocab_size", 0);
    m.n_ctx_train = gi("max_position_embeddings", 0);
    m.n_expert        = gi("num_experts", gi("num_local_experts", 0));
    m.n_expert_used   = gi("num_experts_per_tok", 0);
    m.n_expert_shared = gi("shared_expert_intermediate_size", 0) ? 1 : 0;
    m.rms_eps     = (float)gf("rms_norm_eps", gf("norm_eps", 1e-6));   /* LFM2: norm_eps */
    m.rope_theta  = (float)gf("rope_parameters.rope_theta", gf("rope_theta", 10000.0));
    /* Rotary scaling stays in the free-form keys with its type (arch/common/rad_arch.h,
     * yarn_from); a bare factor here would wrongly mean linear scaling. */
    m.rope_scale  = 1.0f;

    out->quant = quant_descriptor(*out);
    out->finish();
    return RAD_OK;
}

/* The quantisation descriptor, which with the architecture selects the plugin (spec §2.4): the
 * quant_method plus its fmt ("fp8_e4m3"), so the checkpoint, not a flag, distinguishes fp8 from
 * bf16 releases. */
std::string quant_descriptor(const MetaOwn& m) {
    std::string q;
    if (const char* qm = m.get("quantization_config.quant_method")) {
        q = qm;
        if (const char* fm = m.get("quantization_config.fmt")) { q += "_"; q += fm; }
    }
    return q;
}

int meta_from_gguf(GgufFile& g, MetaOwn* out) {
    if (!g.get_str("general.architecture", &out->arch)) {
        RAD_ERR("the GGUF has no general.architecture, so no architecture plugin can be chosen");
        return RAD_E_FORMAT;
    }
    g.get_str("general.name", &out->name);

    /* Dimensions come from GGUF's `<arch>.<field>` keys, so one plugin covers a family; other keys
     * pass through in the free-form list. */
    const std::string a = out->arch;
    auto gi = [&](const char* suffix, int64_t dflt) -> int64_t {
        int64_t v = 0;
        return g.get_i64(a + "." + suffix, &v) ? v : dflt;
    };
    auto gf = [&](const char* suffix, double dflt) -> double {
        double v = 0;
        return g.get_f64(a + "." + suffix, &v) ? v : dflt;
    };

    RadModelMeta& m = out->m;
    m.n_layers    = gi("block_count", 0);
    m.n_embd      = gi("embedding_length", 0);
    m.n_head      = gi("attention.head_count", 0);
    m.n_head_kv   = gi("attention.head_count_kv", m.n_head);
    m.head_dim    = gi("attention.key_length", m.n_head ? m.n_embd / m.n_head : 0);
    m.n_ff        = gi("feed_forward_length", 0);
    m.n_ctx_train = gi("context_length", 0);
    m.n_expert       = gi("expert_count", 0);
    m.n_expert_used  = gi("expert_used_count", 0);
    m.n_expert_shared= gi("expert_shared_count", 0);
    m.rms_eps     = (float)gf("attention.layer_norm_rms_epsilon", 1e-6);
    m.rope_theta  = (float)gf("rope.freq_base", 10000.0);
    m.rope_scale  = 1.0f;

    /* Rotary scaling, rewritten as config.json keys so one reader handles both formats
     * (arch/common/rad_arch.h, yarn_from). llama.cpp's YaRN attn_factor, ext_factor and log
     * multiplier have no config.json spelling, so a non-neutral value is refused rather than
     * served as plain YaRN. */
    {
        /* As llama.cpp reads it: type defaults to linear, the factor is rope.scaling.factor or
         * rope.scale_linear, and 0 or 1 means none. */
        std::string type = "linear";
        g.get_str(a + ".rope.scaling.type", &type);
        const double factor = gf("rope.scaling.factor", gf("rope.scale_linear", 0.0));
        const bool none = type == "none" || (type == "linear" && (factor == 0.0 || factor == 1.0));
        if (!none) {
            const struct { const char* key; double neutral; } kOff[] = {
                { "rope.scaling.attn_factor", 1.0 }, { "rope.scaling.yarn_attn_factor", 1.0 },
                { "rope.scaling.yarn_log_multiplier", 0.0 } };
            for (const auto& k : kOff)
                if (gf(k.key, k.neutral) != k.neutral) {
                    RAD_ERR("the GGUF sets %s.%s = %g; that modifier of the rotary's scaling is "
                            "not implemented", a.c_str(), k.key, gf(k.key, k.neutral));
                    return RAD_E_UNSUPPORTED;
                }
            const double ext = gf("rope.scaling.yarn_ext_factor", -1.0);
            if (ext >= 0.0 && ext != 1.0) {
                RAD_ERR("the GGUF sets %s.rope.scaling.yarn_ext_factor = %g; YaRN is implemented "
                        "at extrapolation factor 1", a.c_str(), ext);
                return RAD_E_UNSUPPORTED;
            }
            out->kv("rope_scaling.type", type);
            if (factor != 0.0) out->kv("rope_scaling.factor", fmt("%.17g", factor));
            const int64_t orig = gi("rope.scaling.original_context_length", 0);
            if (orig > 0) out->kv("rope_scaling.original_max_position_embeddings",
                                  fmt("%lld", (long long)orig));
            for (const char* beta : { "beta_fast", "beta_slow" }) {
                const double v = gf((std::string("rope.scaling.yarn_") + beta).c_str(), 0.0);
                if (v != 0.0) out->kv(std::string("rope_scaling.") + beta, fmt("%.17g", v));
            }
        }
    }

    std::vector<std::string> toks;
    if (g.get_strs("tokenizer.ggml.tokens", &toks)) m.n_vocab = (int64_t)toks.size();
    if (m.n_vocab == 0) m.n_vocab = gi("vocab_size", 0);

    /* Everything else, stringified, for the plugin. Arrays are summarised (small integer arrays
     * spelled out); tokenizer keys are skipped, since the vocab has its own section. */
    for (const GgufKV& kv : g.kv()) {
        if (kv.key.rfind("tokenizer.", 0) == 0) continue;
        std::string v;
        if (kv.is_array) {
            v = fmt("[%zu values]", kv.type == GGUF_STR ? kv.s.size()
                                  : !kv.i.empty()       ? kv.i.size() : kv.f.size());
            if (kv.type != GGUF_STR && kv.i.size() <= 8 && !kv.i.empty()) {
                v.clear();
                for (size_t j = 0; j < kv.i.size(); ++j)
                    v += fmt("%s%lld", j ? "," : "", (long long)kv.i[j]);
            }
        } else if (!kv.s.empty()) v = kv.s.front();
        else if (!kv.f.empty() && kv.type == GGUF_F32) v = fmt("%g", kv.f.front());
        else if (!kv.f.empty() && kv.type == GGUF_F64) v = fmt("%g", kv.f.front());
        else if (!kv.i.empty()) v = fmt("%lld", (long long)kv.i.front());
        if (v.size() > 240) v.resize(240);
        out->kv(kv.key, v);
    }

    /* The descriptor a safetensors release would carry. A float file type (0 all F32, 1 mostly
     * F16, 32 mostly BF16) is unquantised and selects the bf16 plugin; any other names its
     * quantisation, which no plugin serves directly. */
    int64_t ftype = 0;
    const bool typed = g.get_i64("general.file_type", &ftype);
    out->quant = !typed || ftype == 0 || ftype == 1 || ftype == 32
                     ? std::string() : fmt("gguf_ftype_%lld", (long long)ftype);
    out->finish();
    return RAD_OK;
}

/* ================================================================== the checkpoint */

int Checkpoint::open_gguf(const std::string& path) {
    RAD_TRY(gguf_.open(path.c_str()));
    kind_ = Gguf;
    for (const GgufTensor& t : gguf_.tensors()) {
        SrcTensor s;
        s.name  = t.name;
        s.ggml  = t.ggml_type;
        s.rad_dtype = gguf_to_rad_dtype(t.ggml_type);
        /* GGUF is fastest-first; everything from here on is row-major. */
        for (int d = (int)t.n_dims - 1; d >= 0; --d) s.shape.push_back(t.ne[d]);
        s.data  = gguf_.tensor_data(t);
        s.bytes = (int64_t)t.bytes;
        if (!s.data) {
            RAD_ERR("%s: '%s' has no readable extent", path.c_str(), t.name.c_str());
            return RAD_E_FORMAT;
        }
        index_.emplace(s.name, tensors_.size());
        tensors_.push_back(std::move(s));
    }
    return meta_from_gguf(gguf_, &meta_);
}

int Checkpoint::add_shard(const std::string& p, const std::string& prefix,
                          const std::unordered_map<std::string, std::string>* owner) {
    shards_.push_back(std::make_unique<SafeTensorsFile>());
    RAD_TRY(shards_.back()->open(p));
    const std::string file = p.substr(p.find_last_of('/') + 1);
    for (const auto& t : shards_.back()->tensors()) {
        /* The index decides which shard a tensor is read from: a directory merging two
         * checkpoints (a finetune's shards beside links to the base model's) holds some names
         * twice. */
        if (owner) {
            auto it = owner->find(t.name);
            if (it != owner->end() && it->second != file) continue;
        }
        if (index_.count(prefix + t.name)) {
            RAD_ERR("%s: '%s' is also in %s, and no index entry says which copy to read", p.c_str(),
                    t.name.c_str(), shards_[index_shard_[prefix + t.name]]->path().c_str());
            return RAD_E_FORMAT;
        }
        if (t.dtype == RAD_DT_INVALID) {
            RAD_ERR("%s: '%s' has dtype %s, which has no RAD equivalent", p.c_str(),
                    t.name.c_str(), t.dtype_str.c_str());
            return RAD_E_DTYPE;
        }
        SrcTensor s;
        s.name      = prefix + t.name;
        s.rad_dtype = t.dtype;
        s.shape     = t.shape;         /* safetensors is already row-major */
        s.data      = t.data;
        s.bytes     = (int64_t)t.bytes;
        index_shard_.emplace(s.name, shards_.size() - 1);
        index_.emplace(s.name, tensors_.size());
        tensors_.push_back(std::move(s));
    }
    return RAD_OK;
}

namespace {

/* Every shard an index names, once each and in name order, or the one model.safetensors; `map`
 * is the index's weight_map, empty without one. */
int shard_list(const std::string& dir, const std::string& index, bool index_given,
               std::vector<std::string>* out, std::unordered_map<std::string, std::string>* map) {
    const int s = st_load_index(index, map);
    if (s == RAD_OK) {
        for (auto& [t, f] : *map)
            if (std::find(out->begin(), out->end(), f) == out->end()) out->push_back(f);
        std::sort(out->begin(), out->end());
        for (auto& f : *out) f = dir + "/" + f;
        return RAD_OK;
    }
    if (s != RAD_E_NOTFOUND || index_given) return s;
    out->push_back(dir + "/model.safetensors");
    return RAD_OK;
}

}  /* namespace */

/* A JSON file flattened under `prefix`; absent is not an error when `optional`. */
static int meta_merge_json_file(const std::string& path, const std::string& prefix, bool optional,
                                MetaOwn* out) {
    std::ifstream f(path);
    if (!f) {
        if (optional) return RAD_OK;
        RAD_ERR("%s: cannot read it", path.c_str());
        return RAD_E_NOTFOUND;
    }
    nlohmann::json j;
    try { f >> j; }
    catch (const std::exception& e) { RAD_ERR("%s: %s", path.c_str(), e.what()); return RAD_E_FORMAT; }
    if (!j.is_object()) {
        RAD_ERR("%s: not a JSON object", path.c_str());
        return RAD_E_FORMAT;
    }
    flatten_json(j, prefix, out);
    return RAD_OK;
}

static bool path_exists(const std::string& p) { std::ifstream f(p); return (bool)f; }

/* "X.weight" -> "X"; false for a name with no such suffix. */
static bool weight_base(const std::string& name, std::string* base) {
    static const std::string suffix = ".weight";
    if (name.size() <= suffix.size() ||
        name.compare(name.size() - suffix.size(), suffix.size(), suffix) != 0)
        return false;
    *base = name.substr(0, name.size() - suffix.size());
    return true;
}

/* A diffusers pipeline: model_index.json at the top and one folder per component, each with its
 * own config and weights, read as one checkpoint:
 *
 *   text_encoder/   its config.json is the container's model config -- the typed dimensions and
 *                   the bare keys -- and its tensors keep their names;
 *   transformer/    tensors under `transformer.`, config.json under `transformer.` keys;
 *   vae/            tensors under `vae.`, config.json under `vae.` keys;
 *   scheduler/      scheduler_config.json under `scheduler.` keys;
 *   processor/ (or tokenizer/)  the tokenizer, the chat template and the preprocessor configs.
 *
 * model_index.json goes under `pipeline.` keys, and its `_class_name` names the architecture
 * (`QwenImage21Pipeline` is `qwenimage21`). A component folder may link to another pipeline's (a
 * distilled release sharing its base's text encoder). */
int Checkpoint::open_diffusers() {
    kind_ = SafeT;
    RAD_TRY(meta_merge_json_file(dir_ + "/model_index.json", "pipeline", false, &meta_));
    const std::string te = dir_ + "/text_encoder";
    if (path_exists(te + "/config.json")) {
        std::vector<std::string> shards;
        std::unordered_map<std::string, std::string> owner;
        RAD_TRY(shard_list(te, te + "/model.safetensors.index.json", false, &shards, &owner));
        for (const auto& sh : shards) RAD_TRY(add_shard(sh, "", owner.empty() ? nullptr : &owner));
        RAD_TRY(meta_from_config_json(te, &meta_));
        RAD_TRY(meta_merge_generation_config(te, &meta_));
    }
    for (const char* comp : { "transformer", "vae" }) {
        const std::string d = dir_ + "/" + comp;
        if (!path_exists(d + "/config.json")) continue;
        const std::string idx = d + "/diffusion_pytorch_model.safetensors.index.json";
        std::vector<std::string> shards;
        std::unordered_map<std::string, std::string> owner;
        if (path_exists(idx)) {
            RAD_TRY(shard_list(d, idx, true, &shards, &owner));
        } else {
            shards.push_back(d + "/diffusion_pytorch_model.safetensors");
        }
        for (const auto& sh : shards)
            RAD_TRY(add_shard(sh, std::string(comp) + ".", owner.empty() ? nullptr : &owner));
        RAD_TRY(meta_merge_json_file(d + "/config.json", comp, false, &meta_));
    }
    RAD_TRY(meta_merge_json_file(dir_ + "/scheduler/scheduler_config.json", "scheduler", true,
                                 &meta_));
    for (const char* sub : { "processor", "tokenizer" }) {
        const std::string d = dir_ + "/" + sub;
        if (!path_exists(d + "/tokenizer.json")) continue;
        tok_dir_ = d;
        RAD_TRY(meta_merge_preprocessor_configs(d, &meta_));
        break;
    }
    /* The architecture is the pipeline's, whatever the text encoder calls itself. */
    if (const char* cn = meta_.get("pipeline._class_name")) {
        std::string a = cn;
        const std::string suf = "Pipeline";
        if (a.size() > suf.size() && a.compare(a.size() - suf.size(), suf.size(), suf) == 0)
            a.resize(a.size() - suf.size());
        for (char& ch : a) ch = (char)std::tolower((unsigned char)ch);
        meta_.arch = a;
    }
    if (tensors_.empty()) {
        RAD_ERR("%s: a diffusers pipeline with no component weights", dir_.c_str());
        return RAD_E_NOTFOUND;
    }
    meta_.finish();
    return RAD_OK;
}

int Checkpoint::open_safetensors(const std::string& path) {
    kind_ = SafeT;
    /* One shard, an index, or the directory holding either, as one namespace. */
    std::vector<std::string> shards;
    std::unordered_map<std::string, std::string> owner;
    const bool is_index = path.size() > 10 && path.compare(path.size() - 10, 10, "index.json") == 0;
    const bool is_file = path.size() > 12 &&
                         path.compare(path.size() - 12, 12, ".safetensors") == 0;
    if (!is_file && !is_index && path_exists(dir_ + "/model_index.json") &&
        !path_exists(dir_ + "/config.json"))
        return open_diffusers();
    if (is_file) {
        shards.push_back(path);
    } else {
        RAD_TRY(shard_list(dir_, is_index ? path : dir_ + "/model.safetensors.index.json",
                           is_index, &shards, &owner));
    }
    for (const auto& sh : shards) RAD_TRY(add_shard(sh, "", owner.empty() ? nullptr : &owner));

    RAD_TRY(meta_from_config_json(dir_, &meta_));
    RAD_TRY(meta_merge_generation_config(dir_, &meta_));
    RAD_TRY(meta_merge_preprocessor_configs(dir_, &meta_));
    /* FP8's block, where the checkpoint states one. */
    if (const char* bs = meta_.get("quantization_config.weight_block_size")) {
        long long r = 0, c = 0;
        if (std::sscanf(bs, "%lld %lld", &r, &c) == 2 && r > 0 && c > 0) {
            fp8_br_ = r;
            fp8_bc_ = c;
        }
    }
    detect_import();
    meta_.finish();
    return RAD_OK;
}

/* Which published quantisation this is, from quantization_config alone; import_piece checks the
 * tensors weight by weight, refusing a file at the first weight that disagrees. */
void Checkpoint::detect_import() {
    auto qc = [&](const std::string& k) {
        const char* v = meta_.get("quantization_config." + k);
        return v ? std::string(v) : std::string();
    };
    auto any_under = [&](const std::string& prefix) {
        for (const std::string& k : meta_.keys)
            if (k.rfind("quantization_config." + prefix, 0) == 0) return true;
        return false;
    };
    const std::string qm = qc("quant_method");
    if (qm == "auto-round") {
        /* AutoRound's GPTQ-compatible export; its own `auto_round` packing is not read. */
        const std::string pf = qc("packing_format");
        if (pf == "auto_round:auto_gptq") import_ = Import::Gptq;
        else import_why_ = "AutoRound's packing_format '" + pf + "' is not one this reads; its "
                           "auto_round:auto_gptq export is";
    } else if (qm == "gptq") {
        /* v1 (auto_gptq's) stores the zero point less one; gptq_v2 does not, and reading it as v1
         * would shift every code by one. */
        const std::string cf = qc("checkpoint_format");
        if (cf.empty() || cf == "gptq") import_ = Import::Gptq;
        else import_why_ = "GPTQ's checkpoint_format '" + cf + "' is not one this reads; 'gptq' is";
    } else if (qm == "quark") {
        const std::string w = "global_quant_config.weight.";
        if (qc(w + "dtype") == "fp4" && qc(w + "qscheme") == "per_group" &&
            qc(w + "group_size") == "32" && qc(w + "scale_format") == "e8m0" &&
            !any_under("layer_quant_config.") && !any_under("layer_type_quant_config."))
            import_ = Import::QuarkMx;
        else
            import_why_ = "Quark's weight scheme here is not MXFP4 (fp4, per_group 32, e8m0 scales) "
                          "throughout, which is the one this reads";
    } else if (qm == "compressed-tensors") {
        /* compressed-tensors states each scheme as a config group with a format. Three are read:
         *   pack-quantized        symmetric grouped int4 (e.g. Google's Gemma 4 QAT w4a16-ct);
         *                         every group must be one, and a zero point, act-order, or a
         *                         dynamic or activation scheme would change what a code means
         *   nvfp4-pack-quantized  NVFP4: e2m1 codes, an e4m3 scale per 16 of K, an f32 global
         *                         scale per tensor
         *   float-quantized       8-bit float weights, read by the FP8 path
         * NVFP4 and float groups may be mixed (a mixed-precision release ships NVFP4 MLPs beside
         * FP8 linears); the tensors confirm each weight's form at import_piece and ckpt_resolve. */
        std::vector<std::string> groups;
        const std::string cg = "quantization_config.config_groups.";
        for (const std::string& k : meta_.keys)
            if (k.rfind(cg, 0) == 0) {
                const std::string g = k.substr(cg.size(), k.find('.', cg.size()) - cg.size());
                if (std::find(groups.begin(), groups.end(), g) == groups.end()) groups.push_back(g);
            }
        std::string why;
        int64_t group = 0;
        int n_int4 = 0, n_nvfp4 = 0, n_fp8 = 0;
        if (groups.empty()) why = "its quantization_config names no config group";
        for (const std::string& g : groups) {
            if (!why.empty()) break;
            const std::string w = "config_groups." + g + ".weights.";
            /* A config without per-group formats states the one format at the top level. */
            std::string f = qc("config_groups." + g + ".format");
            if (f.empty()) f = qc("format");
            if (f == "nvfp4-pack-quantized") {
                if (qc(w + "num_bits") != "4" || qc(w + "type") != "float" ||
                    qc(w + "group_size") != "16")
                    why = "config group '" + g + "' is nvfp4-pack-quantized but its weights are not "
                          "4-bit floats in groups of 16";
                ++n_nvfp4;
            } else if (f == "float-quantized") {
                if (qc(w + "num_bits") != "8" || qc(w + "type") != "float")
                    why = "config group '" + g + "' is float-quantized but its weights are not "
                          "8-bit floats";
                ++n_fp8;
            } else if (f == "pack-quantized") {
                const std::string gs = qc(w + "group_size");
                const long long gn = gs.empty() ? 0 : std::atoll(gs.c_str());
                if (qc("format") != "pack-quantized")
                    why = "its format is '" + qc("format") + "', and an int4 group is read only "
                          "in a pack-quantized config";
                else if (qc(w + "num_bits") != "4" || qc(w + "type") != "int" ||
                         qc(w + "symmetric") != "1" || qc(w + "strategy") != "group" ||
                         qc(w + "dynamic") != "0" || gn <= 0 || !qc(w + "actorder").empty() ||
                         any_under("config_groups." + g + ".input_activations."))
                    why = "config group '" + g + "' is not symmetric 4-bit int weights in groups, "
                          "packed, with no act-order and unquantised activations, which is the "
                          "one this reads";
                else if (group && gn != group)
                    why = "its config groups state different group sizes";
                else
                    group = gn;
                ++n_int4;
            } else {
                why = "config group '" + g + "' is in the format '" + f + "'";
            }
        }
        if (why.empty() && n_int4 && (n_nvfp4 || n_fp8))
            why = "it mixes pack-quantized int4 groups with float ones";
        if (!why.empty()) {
            import_why_ = "compressed-tensors: " + why + "; the formats read are pack-quantized "
                          "(symmetric int4), nvfp4-pack-quantized and float-quantized (8-bit float)";
        } else if (n_int4) {
            import_ = Import::CtInt4;
            ct_group_ = group;
        } else {
            import_ = n_nvfp4 ? Import::Nvfp4 : Import::CtFp8;
        }
    } else if (qm == "paroquant") {
        /* z-lab's ParoQuant at integer widths: "int5-bitplane" (AWQ nibbles for the low four bits
         * plus a fifth-bit plane) or AWQ-packed int4, group 128, asymmetric, with the learned
         * rotations beside. */
        if (qc("format") == "int5-bitplane" && qc("bits") == "5" && qc("group_size") == "128")
            import_ = Import::ParoI5;
        else if (qc("format").empty() && qc("bits") == "4" && qc("group_size") == "128")
            import_ = Import::ParoI4;       /* z-lab's own release: AWQ-packed int4, asymmetric */
        else
            import_why_ = "ParoQuant here is neither the int5-bitplane format (5-bit codes) nor z-lab's "
                          "AWQ-packed int4, in groups of 128, which are the ones this reads";
    } else if (qm == "paroquant_mxfp4") {
        /* Quark's MXFP4 planes in a rotated basis (ParoQuant retrained for the MXFP4 grid): e2m1
         * codes, an e8m0 exponent per 32 of K, and per projection the learned Givens rotations
         * (`pairs`, `theta`) and channel scales in groups of 128. */
        if (qc("format") == "mxfp4" && qc("mxfp4_block") == "32" && qc("group_size") == "128")
            import_ = Import::ParoMx;
        else
            import_why_ = "ParoQuant's MXFP4 here is not format mxfp4 with 32-element blocks and "
                          "rotation groups of 128, which is the one this reads";
    }
    /* Rename the tool's quantization_config: plugins read that key as the served format
     * (qwen35_fp8 checks quant_method is "fp8"), while an imported checkpoint states each weight's
     * encoding. It stays in the container as provenance. */
    if (import_ != Import::None)
        for (std::string& k : meta_.keys)
            if (k.rfind("quantization_config.", 0) == 0) k = "imported_" + k;
}

int Checkpoint::import_piece(const std::string& name, const SrcTensor** codes,
                             const SrcTensor** scale, std::string* why) {
    auto refuse = [&](int rc, std::string m) {
        if (why) *why = std::move(m);
        return rc;
    };
    /* CtFp8 has no imported form: its FP8 weights read stock (ckpt_resolve). */
    if (import_ == Import::None || import_ == Import::CtFp8) return RAD_E_NOTFOUND;
    if (auto hit = imported_.find(name); hit != imported_.end()) {
        *codes = hit->second.first;
        *scale = hit->second.second;
        return RAD_OK;
    }

    if (import_ == Import::ParoI5 || import_ == Import::ParoI4) {
        const bool i5 = import_ == Import::ParoI5;
        /* Paro int5: `X.weight` is X.qweight/.qweight_hi (codes), X.scales (f16 [G, N]) and
         * X.qzeros/.qzeros_hi (zero points). The view describes the logical [N, K] u5 matrix;
         * ckpt_plane_rows re-lays the planes row by row. */
        std::string base;
        if (!weight_base(name, &base)) return RAD_E_NOTFOUND;
        const SrcTensor* qw = find(base + ".qweight");
        if (!qw) return RAD_E_NOTFOUND;
        const SrcTensor* hi = i5 ? find(base + ".qweight_hi") : nullptr;
        const SrcTensor* sc = find(base + ".scales");
        const SrcTensor* qz = find(base + ".qzeros");
        const SrcTensor* qzh = i5 ? find(base + ".qzeros_hi") : nullptr;
        if ((i5 && (!hi || !qzh)) || !sc || !qz || qw->rad_dtype != RAD_I32 ||
            qz->rad_dtype != RAD_I32 || sc->rad_dtype != RAD_F16 || qw->shape.size() != 2 ||
            qz->shape.size() != 2 || sc->shape.size() != 2 ||
            (i5 && (hi->rad_dtype != RAD_I32 || qzh->rad_dtype != RAD_I32 ||
                    hi->shape.size() != 2 || qzh->shape.size() != 2)))
            return refuse(RAD_E_FORMAT, "'" + qw->name + "' lacks an int32 " +
                                        (i5 ? "qweight_hi, qzeros_hi, " : "") +
                                        "qzeros and an f16 scales beside it");
        const int64_t K = qw->shape[0], N = qw->shape[1] * 8, G = sc->shape[0];
        if (N % 32 || K % 128 || G * 128 != K || sc->shape[1] != N || qz->shape[0] != G ||
            qz->shape[1] != N / 8 ||
            (i5 && (hi->shape[0] != K || hi->shape[1] != N / 32 || qzh->shape[0] != G ||
                    qzh->shape[1] != N / 32)))
            return refuse(RAD_E_SHAPE, "'" + qw->name + "' planes do not fit a [" +
                                       std::to_string(N) + ", " + std::to_string(K) + "] weight");
        SrcTensor c = *qw;
        c.name = name;
        c.rad_dtype = i5 ? RAD_U5 : RAD_U4;
        c.shape = { N, K };
        c.pbits = i5 ? 5 : 4;
        c.aux[0] = hi; c.aux[1] = qz; c.aux[2] = qzh;
        sliced_.push_back(std::move(c));
        *codes = &sliced_.back();
        *scale = sc;
        imported_[name] = { *codes, *scale };
        return RAD_OK;
    }

    if (import_ == Import::QuarkMx || import_ == Import::ParoMx) {
        /* Quark's MXFP4 is `fp4_e2m1*e8m0[1x32]` as stored: two codes per byte (element 2j in the
         * low nibble), one biased exponent per 32 of K. The view only renames dtypes and the
         * logical width. Excluded layers are plain tensors. */
        const SrcTensor* w = find(name);
        const SrcTensor* e = w ? find(name + "_scale") : nullptr;
        if (!w || !e || w->rad_dtype != RAD_U8) return RAD_E_NOTFOUND;
        if (e->rad_dtype != RAD_U8 || w->shape.size() != 2 || e->shape.size() != 2 ||
            e->shape[0] != w->shape[0] || e->shape[1] * 32 != w->shape[1] * 2)
            return refuse(RAD_E_SHAPE, "'" + e->name + "' is not the [N, K/32] exponent plane of '" +
                                       w->name + "'");
        SrcTensor c = *w;
        c.rad_dtype = RAD_FP4E2M1;
        c.shape = { w->shape[0], w->shape[1] * 2 };
        SrcTensor x = *e;
        x.rad_dtype = RAD_E8M0;
        sliced_.push_back(std::move(c));
        *codes = &sliced_.back();
        sliced_.push_back(std::move(x));
        *scale = &sliced_.back();
        imported_[name] = { *codes, *scale };
        return RAD_OK;
    }

    if (import_ == Import::CtInt4) {
        /* CT-INT4: `X.weight` ships as X.weight_packed (i32 [N, K/8], eight codes per word from the
         * low bits, unsigned about 8) and X.weight_scale ([N, K/group]). The words are already the
         * [N, K] code matrix in row order; only the offset is left for the relayout. An ignored
         * layer is an ordinary X.weight. */
        std::string base;
        if (!weight_base(name, &base)) return RAD_E_NOTFOUND;
        const SrcTensor* pk = find(name + "_packed");
        if (!pk) return RAD_E_NOTFOUND;
        const SrcTensor* sc = find(name + "_scale");
        const std::string what = "'" + pk->name + "'";
        if (pk->rad_dtype != RAD_I32 || pk->shape.size() != 2)
            return refuse(RAD_E_FORMAT, what + " is not an int32 [N, K/8] matrix");
        if (find(name + "_zero_point"))
            return refuse(RAD_E_UNSUPPORTED, what + " has a zero point beside it, which a symmetric "
                                             "config does not; an asymmetric weight is not read");
        const int64_t N = pk->shape[0], K = pk->shape[1] * 8;
        if (K % ct_group_)
            return refuse(RAD_E_SHAPE, what + " has " + std::to_string(K) + " columns, not whole "
                                       "groups of " + std::to_string(ct_group_));
        if (!sc || sc->shape.size() != 2 || sc->shape[0] != N || sc->shape[1] != K / ct_group_ ||
            (sc->rad_dtype != RAD_F16 && sc->rad_dtype != RAD_BF16 && sc->rad_dtype != RAD_F32))
            return refuse(RAD_E_FORMAT, what + " has no float [" + std::to_string(N) + ", " +
                                        std::to_string(K / ct_group_) + "] '" + name +
                                        "_scale' beside it");
        if (const SrcTensor* ws = find(name + "_shape")) {
            int64_t dims[2] = { 0, 0 };
            for (int i = 0; i < 2 && i < (int)ws->numel(); ++i)
                dims[i] = (int64_t)rad_load_f32(ws->data, ws->rad_dtype, i);
            if (ws->numel() != 2 || dims[0] != N || dims[1] != K)
                return refuse(RAD_E_SHAPE, "'" + ws->name + "' does not state the [" +
                                           std::to_string(N) + ", " + std::to_string(K) +
                                           "] its codes hold");
        }
        SrcTensor c = *pk;
        c.rad_dtype = RAD_I4;
        c.shape = { N, K };
        c.ct_group = ct_group_;
        sliced_.push_back(std::move(c));
        *codes = &sliced_.back();
        *scale = sc;
        imported_[name] = { *codes, *scale };
        return RAD_OK;
    }

    if (import_ == Import::Nvfp4) {
        /* NVFP4 (compressed-tensors' nvfp4-pack-quantized): `X.weight_packed` U8 [N, K/2], two
         * e2m1 codes per byte with the even k in the low nibble; `X.weight_scale` e4m3 [N, K/16];
         * `X.weight_global_scale` f32, one per tensor. The global scale is a divisor: the value is
         * code * (e4m3 / global), as compressed-tensors' dequantise and vLLM's loader (which
         * stores 1 / global) compute it. The affine scheme multiplies, so the scale plane is that
         * quotient in f32, one per 16 of K, owned here. A weight without `X.weight_packed` is not
         * NVFP4 (a mixed release ships FP8 and bf16 linears beside it) and reads stock. */
        std::string base;
        if (!weight_base(name, &base)) return RAD_E_NOTFOUND;
        const SrcTensor* p = find(base + ".weight_packed");
        if (!p) return RAD_E_NOTFOUND;
        if (p->rad_dtype != RAD_U8 || p->shape.size() != 2)
            return refuse(RAD_E_FORMAT, "'" + p->name + "' is not the U8 [N, K/2] packed tensor the"
                                        " NVFP4 form ships beside its scales");
        const SrcTensor* s = find(base + ".weight_scale");
        if (!s || s->rad_dtype != RAD_F8E4M3 || s->shape.size() != 2)
            return refuse(RAD_E_FORMAT, "'" + name + "' has no e4m3 [N, K/16] weight_scale beside "
                                        "its packed codes, which is the form this reads");
        const SrcTensor* g = find(base + ".weight_global_scale");
        if (!g || g->rad_dtype != RAD_F32 || g->bytes != 4)
            return refuse(RAD_E_FORMAT, "'" + name + "' has no single f32 weight_global_scale "
                                        "beside its packed codes, which is the form this reads");
        const int64_t N = p->shape[0], K = p->shape[1] * 2;
        if (N <= 0 || p->shape[1] <= 0 || K % 16)
            return refuse(RAD_E_SHAPE, "'" + p->name + "' packs " + std::to_string(K) +
                                       " columns of K, which the 16-wide e4m3 scale does not "
                                       "divide");
        if (s->shape[0] != N || s->shape[1] != K / 16)
            return refuse(RAD_E_SHAPE, "'" + s->name + "' is not the [" + std::to_string(N) +
                                       ", " + std::to_string(K / 16) + "] e4m3 scale of '" +
                                       p->name + "'");
        if (const SrcTensor* ws = find(base + ".weight_shape")) {
            int64_t dims[2] = { 0, 0 };
            for (int i = 0; i < 2 && i < (int)ws->numel(); ++i)
                dims[i] = (int64_t)rad_load_f32(ws->data, ws->rad_dtype, i);
            if (ws->numel() != 2 || dims[0] != N || dims[1] != K)
                return refuse(RAD_E_SHAPE, "'" + ws->name + "' does not state the [" +
                                           std::to_string(N) + ", " + std::to_string(K) +
                                           "] its codes hold");
        }
        float d;
        std::memcpy(&d, g->data, 4);
        if (!(d > 0.0f) || !std::isfinite(d))
            return refuse(RAD_E_FORMAT, "'" + g->name + "' is not a usable f32 divisor: " +
                                        std::to_string(d));
        /* e4m3 is exact in f32, so each entry is one f32 division, as in the reference dequant. */
        const int64_t nsc = N * (K / 16);
        auto raw = std::make_unique<uint8_t[]>((size_t)nsc * 4);
        float* tab = reinterpret_cast<float*>(raw.get());
        const uint8_t* sb = (const uint8_t*)s->data;
        for (int64_t i = 0; i < nsc; ++i) {
            tab[i] = rad_fp8e4m3_to_f32(sb[(size_t)i]) / d;
            if (!std::isfinite(tab[i]))
                return refuse(RAD_E_FORMAT, "'" + s->name + "' holds a scale that is not finite "
                                            "at element " + std::to_string(i));
        }
        owned_.push_back(std::move(raw));
        SrcTensor sv = *s;
        sv.rad_dtype = RAD_F32;
        sv.data      = owned_.back().get();
        sv.bytes     = nsc * 4;
        SrcTensor pv = *p;
        pv.rad_dtype = RAD_FP4E2M1;
        pv.shape     = { N, K };
        sliced_.push_back(std::move(pv));
        *codes = &sliced_.back();
        sliced_.push_back(std::move(sv));
        *scale = &sliced_.back();
        imported_[name] = { *codes, *scale };
        return RAD_OK;
    }

    /* GPTQ-packed: `X.weight` ships as X.qweight, X.scales, X.qzeros and optionally X.g_idx. The
     * width is read from the tensors, not quantization_config, because a release may mix widths
     * (int4 experts beside int8 projections) and state them only per layer. */
    std::string base;
    if (!weight_base(name, &base)) return RAD_E_NOTFOUND;
    const SrcTensor* qw = find(base + ".qweight");
    if (!qw) return RAD_E_NOTFOUND;
    const SrcTensor* sc = find(base + ".scales");
    const SrcTensor* qz = find(base + ".qzeros");
    const SrcTensor* gi = find(base + ".g_idx");
    const std::string what = "'" + qw->name + "'";
    if (qw->rad_dtype != RAD_I32 || qw->shape.size() != 2)
        return refuse(RAD_E_FORMAT, what + " is not an int32 [K*bits/32, N] matrix");
    if (!sc || sc->shape.size() != 2 ||
        (sc->rad_dtype != RAD_F16 && sc->rad_dtype != RAD_BF16 && sc->rad_dtype != RAD_F32))
        return refuse(RAD_E_FORMAT, what + " has no float [groups, N] '" + base + ".scales' beside it");
    if (!qz || qz->rad_dtype != RAD_I32 || qz->shape.size() != 2)
        return refuse(RAD_E_FORMAT, what + " has no int32 '" + base + ".qzeros' beside it");
    const int64_t N = qw->shape[1];
    const int64_t bits = N > 0 && (qz->shape[1] * 32) % N == 0 ? qz->shape[1] * 32 / N : 0;
    if (bits != 4 && bits != 8)
        return refuse(RAD_E_UNSUPPORTED, what + " packs " + std::to_string(bits) + "-bit codes "
                                         "(from its qzeros' width); 4 and 8 are read");
    const int64_t K = qw->shape[0] * 32 / bits;
    const int64_t G = sc->shape[0];
    if (sc->shape[1] != N || G <= 0 || K % G)
        return refuse(RAD_E_SHAPE, "'" + sc->name + "' is not a [groups, " + std::to_string(N) +
                                   "] scale tensor for " + std::to_string(K) + " columns");
    const int64_t group = K / G;
    /* The served scale covers 128 of K: wider groups (per-channel included) repeat it; narrower
     * groups cannot be represented. */
    if (K % 128 || group % 128)
        return refuse(RAD_E_UNSUPPORTED, what + " has groups of " + std::to_string(group) +
                                         " columns; the imported form scales a 128 of K, so the "
                                         "group must be a multiple of 128");
    if (qz->shape[0] != G)
        return refuse(RAD_E_SHAPE, "'" + qz->name + "' has " + std::to_string(qz->shape[0]) +
                                   " rows for " + std::to_string(G) + " groups");
    /* Symmetric only: every zero point must be v1's 2^(bits-1) - 1, so a code offsets to its
     * signed value. An asymmetric weight would need a zero plane. */
    const uint32_t zero = bits == 4 ? 0x77777777u : 0x7f7f7f7fu;
    for (int64_t i = 0; i < qz->numel(); ++i) {
        uint32_t v;
        std::memcpy(&v, (const uint8_t*)qz->data + i * 4, 4);
        if (v != zero)
            return refuse(RAD_E_UNSUPPORTED, "'" + qz->name + "' holds a zero point other than "
                                             "the symmetric one; an asymmetric weight is not read");
    }
    /* In order only: act-order (desc_act) permutes column groups, which a [1x128] scale cannot
     * follow, so g_idx must be absent or the identity. */
    if (gi) {
        if (gi->rad_dtype != RAD_I32 || gi->numel() != K)
            return refuse(RAD_E_FORMAT, "'" + gi->name + "' is not an int32 [K] group index");
        for (int64_t k = 0; k < K; ++k) {
            int32_t v;
            std::memcpy(&v, (const uint8_t*)gi->data + k * 4, 4);
            if (v != k / group)
                return refuse(RAD_E_UNSUPPORTED, "'" + gi->name + "' reorders the columns' groups "
                                                 "(act-order); only an in-order index is read");
        }
    }
    SrcTensor c = *qw;
    c.rad_dtype = bits == 4 ? RAD_I4 : RAD_I8;
    c.shape = { N, K };
    c.gptq_bits = (int32_t)bits;
    c.gptq_group = group;
    sliced_.push_back(std::move(c));
    *codes = &sliced_.back();
    *scale = sc;
    imported_[name] = { *codes, *scale };
    return RAD_OK;
}

int Checkpoint::open(const std::string& path) {
    path_ = path;
    const bool gguf = path.size() > 5 && path.compare(path.size() - 5, 5, ".gguf") == 0;
    dir_ = path;
    const size_t slash = path.find_last_of('/');
    auto ends = [&](const char* s) {
        const size_t n = std::strlen(s);
        return path.size() > n && path.compare(path.size() - n, n, s) == 0;
    };
    const bool is_file = gguf || ends(".safetensors") || ends("index.json");
    if (is_file) dir_ = slash == std::string::npos ? "." : path.substr(0, slash);
    return gguf ? open_gguf(path) : open_safetensors(path);
}

int Checkpoint::add_draft(const std::string& dir, const std::string& prefix) {
    if (kind_ != SafeT) {
        RAD_ERR("a drafter is merged into a safetensors target's namespace; %s is not one",
                path_.c_str());
        return RAD_E_UNSUPPORTED;
    }
    RAD_TRY(meta_merge_config_json(dir, prefix.empty() || prefix.back() != '.'
                                            ? prefix
                                            : prefix.substr(0, prefix.size() - 1), &meta_));
    std::vector<std::string> shards;
    std::unordered_map<std::string, std::string> owner;
    RAD_TRY(shard_list(dir, dir + "/model.safetensors.index.json", false, &shards, &owner));
    for (const auto& sh : shards) RAD_TRY(add_shard(sh, prefix, owner.empty() ? nullptr : &owner));
    meta_.finish();
    return RAD_OK;
}

const SrcTensor* Checkpoint::find(const std::string& n) const {
    auto it = index_.find(n);
    return it == index_.end() ? nullptr : &tensors_[it->second];
}

/* One expert's slice of a stacked tensor, as a view: slicing dim 0 of a row-major tensor is an
 * offset and a shorter shape, with nothing copied. It is exact for block-quantised sources too:
 * ggml packs blocks along the row, so a dim-0 slice is whole blocks and `bytes / shape[0]` is the
 * exact byte count. Slicing other dimensions is not offered. */
const SrcTensor* Checkpoint::slice(const SrcTensor& st, int index, const char* declared) {
    if (st.shape.size() < 2) {
        RAD_ERR("'%s' asks for sub-tensor %d of '%s', which has rank %zu. Only a STACKED tensor "
                "-- rank 2 or more, with the stack on dim 0 -- has sub-tensors to take.",
                declared, index, st.name.c_str(), st.shape.size());
        return nullptr;
    }
    const int64_t n = st.shape[0];
    if (index < 0 || index >= n) {
        RAD_ERR("'%s' asks for sub-tensor %d of '%s', which holds %lld of them",
                declared, index, st.name.c_str(), (long long)n);
        return nullptr;
    }
    if (!st.data || st.bytes <= 0 || st.bytes % n != 0) {
        RAD_ERR("'%s': '%s' is %lld byte(s) over %lld sub-tensor(s), which does not divide -- the "
                "slice would straddle two of them",
                declared, st.name.c_str(), (long long)st.bytes, (long long)n);
        return nullptr;
    }
    SrcTensor v;
    /* The name carries the index so diagnostics distinguish the slices. */
    v.name      = st.name + "[" + std::to_string(index) + "]";
    v.ggml      = st.ggml;
    v.rad_dtype = st.rad_dtype;
    v.shape.assign(st.shape.begin() + 1, st.shape.end());
    v.bytes     = st.bytes / n;
    v.data      = (const uint8_t*)st.data + (size_t)index * (size_t)v.bytes;
    sliced_.push_back(std::move(v));
    return &sliced_.back();
}

/* ================================================================== logical weights */

namespace {

}  /* namespace */

std::string ckpt_tensor_of(const SrcTensor& t) {
    const size_t br = t.name.size() && t.name.back() == ']' ? t.name.rfind('[') : std::string::npos;
    return br == std::string::npos ? t.name : t.name.substr(0, br);
}

bool ckpt_reads_any(const CkptWeight& w, const std::unordered_set<std::string>& names,
                    std::unordered_set<std::string>* hit) {
    bool any = false;
    for (const auto* list : { &w.pieces, &w.scales, &w.pq_pairs, &w.pq_theta, &w.pq_scale })
        for (const SrcTensor* t : *list) {
            if (!t) continue;
            const std::string n = ckpt_tensor_of(*t);
            if (names.count(n)) {
                any = true;
                if (hit) hit->insert(n);
            }
        }
    return any;
}

namespace {

/* ParoQuant's rotation group: pairs address columns within a group of 128. */
constexpr int64_t kParoGroup = 128;

/* "model.layers.0.mlp.down_proj.weight" -> "model.layers.0.mlp.down_proj" */
std::string paro_base(const SrcTensor& piece) {
    const std::string n = ckpt_tensor_of(piece);
    std::string base;
    return weight_base(n, &base) ? base : n;
}

}  /* namespace */

/* Stages applied in table order (stage 0 first) as a' = a cos t + b sin t, b' = b cos t - a sin t
 * (libref's loop), in f64 throughout. The error (~1e-15) is far below an f16 step, so the output is
 * the exact composition rounded to f16 via f32 regardless of libm or summation order. */
void paro_compose_dense(int64_t S, const int32_t* ia, const int32_t* ib, const float* th,
                        uint16_t* out) {
    constexpr int64_t G = kParoGroup, H = kParoGroup / 2;
    std::vector<double> cs((size_t)(S * H)), sn((size_t)(S * H)), v((size_t)G);
    for (int64_t o = 0; o < S * H; ++o) {
        cs[(size_t)o] = std::cos((double)th[o]);
        sn[(size_t)o] = std::sin((double)th[o]);
    }
    for (int64_t i = 0; i < G; ++i) {
        std::fill(v.begin(), v.end(), 0.0);
        v[(size_t)i] = 1.0;
        for (int64_t o = 0; o < S * H; ++o) {
            const double va = v[(size_t)ia[o]], vb = v[(size_t)ib[o]];
            v[(size_t)ia[o]] = va * cs[(size_t)o] + vb * sn[(size_t)o];
            v[(size_t)ib[o]] = vb * cs[(size_t)o] - va * sn[(size_t)o];
        }
        for (int64_t j = 0; j < G; ++j) out[j * G + i] = rad_f32_to_f16((float)v[(size_t)j]);
    }
}

namespace {

/* Each piece of a rotated weight has its own rotation: a fused gate_up or q|k|v is several
 * projections with their own pairs, angles and channel scales, so the fused weight has one row
 * part per piece (abi/rad_encoding.h, "givens"). Adds the tables (t.pairs and t.angle stage-major
 * per part, t.scale one row per part, t.part each part's end row, t.dense) and refuses a rotation
 * the kernels cannot read: a pair leaving its group of 128, or a stage that does not use every
 * column of a group exactly once. */
int paro_attach(Checkpoint& ck, CkptWeight* out, std::string* why) {
    auto refuse = [&](std::string m) {
        if (why) *why = std::move(m);
        return (int)RAD_E_FORMAT;
    };
    const int64_t K = out->cols, P = (int64_t)out->pieces.size();
    if (K % kParoGroup) return refuse("'" + out->name + "' has " + std::to_string(K) +
                                      " columns, which is not whole rotation groups of 128");
    int64_t S = 0;
    out->pq_pairs.clear(); out->pq_theta.clear(); out->pq_scale.clear();
    for (const SrcTensor* piece : out->pieces) {
        const std::string base = paro_base(*piece);
        const SrcTensor* pr = ck.find(base + ".pairs");
        const SrcTensor* th = ck.find(base + ".theta");
        const SrcTensor* cs = ck.find(base + ".channel_scales");
        if (!pr || !th || !cs)
            return refuse("'" + piece->name + "' is a ParoQuant MXFP4 weight and '" + base +
                          "' lacks its pairs, theta or channel_scales");
        if (pr->rad_dtype != RAD_I16 || th->rad_dtype != RAD_F16 || cs->rad_dtype != RAD_F16 ||
            pr->shape.size() != 2 || th->shape.size() != 2 || cs->shape.size() != 2 ||
            pr->shape[1] != K || th->shape[0] != pr->shape[0] || th->shape[1] != K / 2 ||
            cs->shape[0] != 1 || cs->shape[1] != K || pr->shape[0] < 1)
            return refuse("'" + base + "' pairs/theta/channel_scales do not fit a " +
                          std::to_string(K) + "-column weight (int16 [S, K], f16 [S, K/2], f16 [1, K])");
        if (S == 0) S = pr->shape[0];
        if (pr->shape[0] != S)
            return refuse("the pieces of '" + out->name + "' have different numbers of rotation stages");
        /* Every stage must be a perfect matching of each group's 128 columns. */
        const int16_t* pv = (const int16_t*)pr->data;
        for (int64_t st = 0; st < S; ++st)
            for (int64_t g = 0; g < K / kParoGroup; ++g) {
                bool seen[kParoGroup] = {};
                for (int64_t i = 0; i < kParoGroup; ++i) {
                    const int v = pv[st * K + g * kParoGroup + i];
                    if (v < 0 || v >= kParoGroup || seen[v])
                        return refuse("'" + pr->name + "' stage " + std::to_string(st) + " group " +
                                      std::to_string(g) + " is not a pairing of its 128 columns");
                    seen[v] = true;
                }
            }
        out->pq_pairs.push_back(pr);
        out->pq_theta.push_back(th);
        out->pq_scale.push_back(cs);
    }
    RadEncoding& e = out->enc;
    rad_enc_copy_str(e.transform, "givens");
    rad_enc_add_table(&e, "t.pairs", RAD_I32, P * S, K);
    rad_enc_add_table(&e, "t.angle", RAD_F32, P * S, K / 2);
    rad_enc_add_table(&e, "t.scale", RAD_F32, P, K);
    if (P > 1) rad_enc_add_table(&e, "t.part", RAD_I32, 1, P);
    /* The dense matrices are composed and stored here, since the prefill quantiser
     * (givens_dense_quant) applies a (part, group) as one 128 x 128 matrix and a kernel library
     * must not compute weight values at load (abi/rad_abi.h, weight layout). The pairs and angles
     * stay for the 16-bit GEMMs' stage-wise rotation. */
    if (rad_enc_add_table(&e, "t.dense", RAD_F16, P * K / kParoGroup, kParoGroup * kParoGroup) < 0)
        return refuse("'" + out->name + "' has more planes than an encoding holds");
    if (!rad_enc_valid(&e)) return refuse("the rotation tables of '" + out->name + "' are malformed");
    out->imported = true;
    out->paro = true;
    out->trivial = false;
    out->why = "it is ParoQuant MXFP4, which rad-convert imports with its rotation tables -- "
               "serve the container it writes";
    return RAD_OK;
}

}  /* namespace */

int ckpt_resolve(Checkpoint& ck, const Program& prog, const std::string& name, bool optional,
                 CkptWeight* out, std::string* why) {
    auto refuse = [&](int rc, std::string m) {
        if (why) *why = std::move(m);
        return rc;
    };
    *out = CkptWeight{};
    out->name = name;

    /* Every map for the name in declaration order. Only a CONCAT map may be declared in pieces
     * (declare refused other repeats); the pieces join end to end. */
    std::vector<const RadNameMap*> maps;
    for (const auto& nm : prog.name_map)
        if (nm.declared && name == nm.declared) maps.push_back(&nm);
    if (maps.empty()) return refuse(RAD_E_NOTFOUND, "the name map has no entry for it");
    struct Src1 { const char* name; int index; };
    std::vector<Src1> srcs;
    for (const RadNameMap* pm : maps)
        for (int k = 0; k < pm->n_src; ++k) srcs.push_back({ pm->src[k], pm->src_index[k] });
    out->concat_dim = maps.front()->concat_dim;

    /* The mode decides what several sources mean:
     *
     *   RAD_MAP_CONCAT   all, joined (gate and up into gate_up).
     *   RAD_MAP_COPY     alternatives, first present wins (a tied lm_head is output.weight when
     *                    shipped, else the embedding). */
    const bool alternatives = maps.front()->mode == RAD_MAP_COPY && srcs.size() > 1;
    for (const Src1& s1 : srcs) {
        const SrcTensor* st = ck.find(s1.name);
        /* An FP8 weight's scale plane: `X.weight_scale_inv` (block-scaled releases) or
         * `X.weight_scale` (compressed-tensors' and Quark's naming, one per row or per block).
         * Only an FP8 tensor takes `X.weight_scale`, so a float weight beside an unrelated tensor
         * of that name reads plain. An imported weight's planes come from import_piece below. */
        const SrcTensor* sc = st ? ck.find(std::string(s1.name) + "_scale_inv") : nullptr;
        if (!sc && st && (st->rad_dtype == RAD_F8E4M3 || st->rad_dtype == RAD_F8E5M2))
            sc = ck.find(std::string(s1.name) + "_scale");
        /* A weight shipped in an imported form is read through that form's views. */
        {
            const SrcTensor *ic = nullptr, *is = nullptr;
            std::string iw;
            const int ir = ck.import_piece(s1.name, &ic, &is, &iw);
            if (ir == RAD_OK) {
                if (s1.index >= 0)
                    return refuse(RAD_E_UNSUPPORTED, "it is a slice of '" + std::string(s1.name) +
                                                     "', which the checkpoint ships quantised; an "
                                                     "imported tensor is read whole");
                st = ic;
                sc = is;
            } else if (ir != RAD_E_NOTFOUND) {
                return refuse(ir, iw);
            }
        }
        /* A sub-tensor index is always applied (ignoring it would give every expert the whole
         * stack). Its scale plane is sliced at the same index. */
        if (st && s1.index >= 0) {
            st = ck.slice(*st, s1.index, name.c_str());
            if (!st) return refuse(RAD_E_SHAPE, "a sub-tensor it names is not there");
            if (sc && !(sc = ck.slice(*sc, s1.index, name.c_str())))
                return refuse(RAD_E_SHAPE, "its scale plane cannot be sliced at the same index");
        }
        if (!st) {
            if (alternatives) continue;
            if (optional) { out->pieces.clear(); break; }
            return refuse(RAD_E_NOTFOUND, std::string("it needs checkpoint tensor '") + s1.name +
                                          "', which is not in " + ck.path());
        }
        out->pieces.push_back(st);
        out->scales.push_back(sc);
        if (alternatives) break;
    }
    if (out->pieces.empty()) {
        if (optional) return refuse(RAD_E_NOTFOUND, "none of its tensors is in the checkpoint");
        std::string tried;
        for (size_t k = 0; k < srcs.size(); ++k) tried += (k ? " or " : "") + std::string(srcs[k].name);
        return refuse(RAD_E_NOTFOUND, "it needs one of " + tried + ", and " + ck.path() +
                                      " holds none of them");
    }

    /* The logical shape: pieces joined along concat_dim, other extents equal. A rank-0 tensor
     * (safetensors' scalar, e.g. Gemma 4's clipped-linear bounds) reads as shape [1]. */
    const SrcTensor& p0 = *out->pieces[0];
    const int d = out->concat_dim;
    auto shape_of = [](const SrcTensor& t) {
        return t.shape.empty() ? std::vector<int64_t>{ 1 } : t.shape;
    };
    const std::vector<int64_t> s0 = shape_of(p0);
    if (s0.size() > RAD_MAX_RANK || d < 0 || d >= (int)s0.size())
        return refuse(RAD_E_SHAPE, "'" + p0.name + "' cannot be joined along dim " +
                                   std::to_string(d));
    out->rank = (uint32_t)s0.size();
    for (uint32_t i = 0; i < out->rank; ++i) out->shape[i] = s0[i];
    for (size_t k = 1; k < out->pieces.size(); ++k) {
        const SrcTensor& pk = *out->pieces[k];
        const std::vector<int64_t> sk = shape_of(pk);
        bool fits = sk.size() == s0.size();
        for (uint32_t i = 0; fits && i < out->rank; ++i)
            if ((int)i != d && sk[i] != s0[i]) fits = false;
        if (!fits)
            return refuse(RAD_E_SHAPE, "'" + pk.name + "' and '" + p0.name +
                                       "' do not join along dim " + std::to_string(d));
        out->shape[d] += sk[(size_t)d];
    }
    rad_enc_view(out->rank, out->shape, &out->rows, &out->cols);

    /* The pieces' encoding, when trivial (spec §4.3), or imported. */
    const uint32_t dt = p0.rad_dtype;
    bool same = true, scaled = true, unscaled = true;
    for (size_t k = 0; k < out->pieces.size(); ++k) {
        if (out->pieces[k]->rad_dtype != dt) same = false;
        if (out->scales[k]) unscaled = false; else scaled = false;
    }
    if (dt == RAD_DT_INVALID) {
        out->why = std::string("'") + p0.name + "' is a GGUF " +
                   (gguf_dtype(p0.ggml) ? gguf_dtype(p0.ggml)->name : "block") + " tensor";
    } else if (!same) {
        out->why = "its pieces are stored in different dtypes";
    } else if (p0.pbits) {
        /* Paro int5-bitplane: u5 codes, an f16 scale and u8 zero point per 128 of K, then the
         * rotation tables. Joined only along rows (gate|up), as every imported form. */
        bool fits = d == 0 || out->pieces.size() == 1;
        for (const SrcTensor* pk : out->pieces) fits = fits && pk->pbits == p0.pbits;
        if (!fits) {
            out->why = "its int5 pieces are joined along a column, across their groups";
        } else {
            rad_enc_clear(&out->enc);
            rad_enc_copy_str(out->enc.scheme, "affine");
            rad_enc_add_plane(&out->enc, "codes", p0.pbits == 5 ? RAD_U5 : RAD_U4, 1, 1);
            rad_enc_add_plane(&out->enc, "scale", RAD_F16, 1, 128);
            rad_enc_add_plane(&out->enc, "zero", RAD_U8, 1, 128);
            const int rc = paro_attach(ck, out, why);
            if (rc != RAD_OK) return rc;
            out->why = std::string("it is ParoQuant int") + (p0.pbits == 5 ? "5" : "4") +
                       ", which rad-convert imports with its zero points and rotation tables -- "
                       "serve the container it writes";
        }
    } else if (p0.gptq_bits) {
        /* GPTQ-packed: one width and group throughout, joined only along rows (a column join
         * would put two pieces' groups in one scale block). */
        out->imported = true;
        for (size_t k = 1; k < out->pieces.size() && out->imported; ++k)
            if (out->pieces[k]->gptq_bits != p0.gptq_bits ||
                out->pieces[k]->gptq_group != p0.gptq_group) {
                out->imported = false;
                out->why = "its GPTQ-packed pieces differ in width or group";
            }
        if (out->imported && d != 0 && out->pieces.size() > 1) {
            out->imported = false;
            out->why = "its GPTQ-packed pieces are joined along a column, across their groups";
        }
        if (out->imported) {
            rad_enc_clear(&out->enc);
            rad_enc_copy_str(out->enc.scheme, "affine");
            rad_enc_add_plane(&out->enc, "codes", dt, 1, 1);
            rad_enc_add_plane(&out->enc, "scale", RAD_BF16, 1, 128);
            out->why = "it is GPTQ-packed int" + std::to_string(p0.gptq_bits) + ", which "
                       "rad-convert imports with its scales rounded to bf16 -- serve the "
                       "container it writes";
        }
    } else if (dt == RAD_FP4E2M1 && scaled && ck.imports() == Checkpoint::Import::Nvfp4) {
        /* NVFP4 as import_piece reads it: e2m1 codes beside the f32 quotient e4m3 / global, one
         * per 16 of K, each piece with its own global. Not trivial (the scale is derived), so the
         * loader refuses it and rad-convert decodes it for a recipe. Joined only along rows: the
         * planes and the f32 read follow the pieces row by row. */
        bool f32 = true;
        for (const SrcTensor* s : out->scales) f32 = f32 && s->rad_dtype == RAD_F32;
        if (!f32) {
            out->why = "its FP4 pieces are not all NVFP4";
        } else if (d != 0 && out->pieces.size() > 1) {
            out->why = "its NVFP4 pieces are joined along a column, across their global scales";
        } else {
            out->imported = true;
            out->why = "it is NVFP4 (e2m1 codes, an e4m3 scale per 16 of K and an f32 global "
                       "scale per tensor), which no kernel serves; rad-convert decodes it for a "
                       "recipe rule";
            rad_enc_clear(&out->enc);
            rad_enc_copy_str(out->enc.scheme, "affine");
            rad_enc_add_plane(&out->enc, "codes", RAD_FP4E2M1, 1, 1);
            rad_enc_add_plane(&out->enc, "scale", RAD_F32, 1, 16);
        }
    } else if (dt == RAD_FP4E2M1 && scaled) {
        /* Quark's MXFP4 byte for byte: codes and exponents are the planes of
         * `fp4_e2m1*e8m0[1x32]`. Joined only along rows (gate and up). */
        out->trivial = d == 0 || out->pieces.size() == 1;
        if (!out->trivial) out->why = "its MXFP4 pieces are joined along a column";
        if (out->trivial) {
            rad_enc_clear(&out->enc);
            rad_enc_copy_str(out->enc.scheme, "affine");
            rad_enc_add_plane(&out->enc, "codes", RAD_FP4E2M1, 1, 1);
            rad_enc_add_plane(&out->enc, "scale", RAD_E8M0, 1, 32);
            if (ck.imports() == Checkpoint::Import::ParoMx) {
                const int rc = paro_attach(ck, out, why);
                if (rc != RAD_OK) return rc;
            }
        }
    } else if ((dt == RAD_F8E4M3 || dt == RAD_F8E5M2) && scaled) {
        /* FP8 codes with a scale plane of one dtype, in one of two shapes per piece:
         *   block    [ceil(N/br), ceil(K/bc)], the checkpoint's block (128x128 unless stated):
         *            `fp8_e4m3*<s>[brxbc]`
         *   per row  [N] or [N, 1] (compressed-tensors' channel strategy, ParoQuant's heads):
         *            `fp8_e4m3*<s>[1x*]`
         * Both are byte for byte when joined along rows, and a block join only where each piece
         * but the last fills its last row block. A row join that ends inside a block is still
         * read piece by piece, so it keeps its encoding without being trivial. Mixed shapes and
         * column joins state no encoding: ckpt_rows_f32 reads scales along rows only, so it
         * refuses them with `why`. */
        const int64_t br = ck.fp8_block_rows(), bc = ck.fp8_block_cols();
        const uint32_t sdt = out->scales[0]->rad_dtype;
        int n_row = 0, n_blk = 0;
        bool bad = false;
        for (size_t k = 0; k < out->pieces.size() && !bad; ++k) {
            const SrcTensor& pk = *out->pieces[k];
            const SrcTensor& sk = *out->scales[k];
            int64_t r = 0, c = 0;
            rad_enc_view((uint32_t)pk.shape.size(), pk.shape.data(), &r, &c);
            const int64_t want_r = (r + br - 1) / br, want_c = (c + bc - 1) / bc;
            int64_t sr = 0, scl = 0;
            rad_enc_view((uint32_t)sk.shape.size(), sk.shape.data(), &sr, &scl);
            /* A rank-1 plane is per row; a wider one is a block plane first, then [N, 1]. */
            const bool row1 = sk.shape.size() == 1 && sk.shape[0] == r;
            const bool blk = sk.shape.size() >= 2 && sr == want_r && scl == want_c;
            const bool row2 = sk.shape.size() >= 2 && sr == r && scl == 1;
            if (sk.rad_dtype != sdt) {
                bad = true;
                out->why = "its FP8 pieces' scale planes differ in dtype";
            } else if (row1) {
                ++n_row;
            } else if (blk) {
                ++n_blk;
                if (out->why.empty() && k + 1 < out->pieces.size() && r % br)
                    out->why = "'" + pk.name + "' ends inside a scale block, so the next piece's "
                               "scales cannot follow it";
            } else if (row2) {
                ++n_row;
            } else {
                bad = true;
                out->why = "'" + sk.name + "' is neither the [" + std::to_string(want_r) + ", " +
                           std::to_string(want_c) + "] " + std::to_string(br) + "x" +
                           std::to_string(bc) + " scale plane of '" + pk.name + "' nor one "
                           "scale per row";
            }
        }
        if (!bad && n_row && n_blk) {
            bad = true;
            out->why = "its FP8 pieces disagree: some scale per row, some per " +
                       std::to_string(br) + "x" + std::to_string(bc) + " block";
        }
        if (!bad && d != 0 && out->pieces.size() > 1) {
            bad = true;
            out->why = "its FP8 pieces are joined along a column, across their scale planes";
        }
        if (!bad) {
            out->trivial = out->why.empty();
            rad_enc_clear(&out->enc);
            rad_enc_copy_str(out->enc.scheme, "affine");
            rad_enc_add_plane(&out->enc, "codes", dt, 1, 1);
            if (n_row) rad_enc_add_plane(&out->enc, "scale", sdt, 1, 0);
            else rad_enc_add_plane(&out->enc, "scale", sdt, br, bc);
        }
    } else if (p0.ct_group) {
        /* CT-INT4: one group throughout, joined only along rows, as GPTQ. Codes become signed i4
         * (the offset removed at relayout) and the scale plane is as shipped, one per group of K. */
        out->imported = true;
        for (size_t k = 1; k < out->pieces.size() && out->imported; ++k)
            if (out->pieces[k]->ct_group != p0.ct_group ||
                out->scales[k]->rad_dtype != out->scales[0]->rad_dtype) {
                out->imported = false;
                out->why = "its CT-INT4 pieces differ in group or scale dtype";
            }
        if (out->imported && d != 0 && out->pieces.size() > 1) {
            out->imported = false;
            out->why = "its CT-INT4 pieces are joined along a column, across their groups";
        }
        if (out->imported) {
            rad_enc_clear(&out->enc);
            rad_enc_copy_str(out->enc.scheme, "affine");
            rad_enc_add_plane(&out->enc, "codes", RAD_I4, 1, 1);
            rad_enc_add_plane(&out->enc, "scale", out->scales[0]->rad_dtype, 1, p0.ct_group);
            out->why = "it is CT-INT4, which rad-convert imports -- serve the container it writes";
        }
    } else if (!unscaled) {
        out->why = "some of its pieces carry a scale plane and some do not";
    } else if (dt == RAD_F8E4M3 || dt == RAD_F8E5M2) {
        out->why = "it is FP8 with no scale plane beside it";
    } else {
        out->trivial = rad_dtype_bits(dt) >= 8;
        if (out->trivial) out->enc = rad_enc_plain(dt);
        else out->why = std::string("it is ") + rad_dtype_name(dt);
    }
    return RAD_OK;
}

namespace {

/* ---------------------------------------------------------------- GGUF block types
 *
 * ggml block types decodable to f32. Their geometry lives in core/text; decoding lives here, where
 * weights are read. K-quants are refused rather than half-implemented. */
enum { GGML_F32 = 0, GGML_F16 = 1, GGML_Q4_0 = 2, GGML_Q4_1 = 3, GGML_Q5_0 = 6, GGML_Q5_1 = 7,
       GGML_Q8_0 = 8, GGML_BF16 = 30 };

int ggml_rows_f32(uint32_t type, const uint8_t* p, int64_t n, float* dst) {
    auto half = [&](const uint8_t* q) { uint16_t h; std::memcpy(&h, q, 2); return rad_f16_to_f32(h); };
    switch (type) {
        case GGML_Q8_0:
            for (int64_t b = 0; b < n / 32; ++b, p += 34) {
                const float d = half(p);
                for (int j = 0; j < 32; ++j) dst[b * 32 + j] = d * (float)(int8_t)p[2 + j];
            }
            return RAD_OK;
        case GGML_Q4_0:
            for (int64_t b = 0; b < n / 32; ++b, p += 18) {
                const float d = half(p);
                for (int j = 0; j < 16; ++j) {
                    dst[b * 32 + j]      = d * (float)((p[2 + j] & 0x0F) - 8);
                    dst[b * 32 + j + 16] = d * (float)((p[2 + j] >> 4)   - 8);
                }
            }
            return RAD_OK;
        case GGML_Q4_1:
            for (int64_t b = 0; b < n / 32; ++b, p += 20) {
                const float d = half(p), m = half(p + 2);
                for (int j = 0; j < 16; ++j) {
                    dst[b * 32 + j]      = d * (float)(p[4 + j] & 0x0F) + m;
                    dst[b * 32 + j + 16] = d * (float)(p[4 + j] >> 4)   + m;
                }
            }
            return RAD_OK;
        case GGML_Q5_0:
            for (int64_t b = 0; b < n / 32; ++b, p += 22) {
                const float d = half(p);
                uint32_t qh; std::memcpy(&qh, p + 2, 4);
                for (int j = 0; j < 16; ++j) {
                    const int lo = (int)((p[6 + j] & 0x0F) | (((qh >> j) & 1u) << 4)) - 16;
                    const int hi = (int)((p[6 + j] >> 4)   | (((qh >> (j + 16)) & 1u) << 4)) - 16;
                    dst[b * 32 + j]      = d * (float)lo;
                    dst[b * 32 + j + 16] = d * (float)hi;
                }
            }
            return RAD_OK;
        case GGML_Q5_1:
            for (int64_t b = 0; b < n / 32; ++b, p += 24) {
                const float d = half(p), m = half(p + 2);
                uint32_t qh; std::memcpy(&qh, p + 4, 4);
                for (int j = 0; j < 16; ++j) {
                    const int lo = (int)((p[8 + j] & 0x0F) | (((qh >> j) & 1u) << 4));
                    const int hi = (int)((p[8 + j] >> 4)   | (((qh >> (j + 16)) & 1u) << 4));
                    dst[b * 32 + j]      = d * (float)lo + m;
                    dst[b * 32 + j + 16] = d * (float)hi + m;
                }
            }
            return RAD_OK;
        default:
            return RAD_E_UNSUPPORTED;
    }
}

/* Where flat element `i` of the joined tensor lives: which piece, and the element in it. A join
 * along dim d repeats, for each index of the dims before d, every piece's [d..] extent in turn. */
struct Joined {
    std::vector<int64_t> inner;   /* each piece's elements per outer index */
    int64_t total_inner = 0;
    int     n = 0;
    void init(const std::vector<const SrcTensor*>& pieces, int d) {
        n = (int)pieces.size();
        inner.assign((size_t)n, 1);
        total_inner = 0;
        for (int k = 0; k < n; ++k) {
            for (size_t i = (size_t)d; i < pieces[(size_t)k]->shape.size(); ++i)
                inner[(size_t)k] *= pieces[(size_t)k]->shape[i];
            total_inner += inner[(size_t)k];
        }
    }
    /* Visit [a, b) as runs that lie in one piece: f(piece, element in it, count, at). */
    template <class F> int walk(int64_t a, int64_t b, F f) const {
        while (a < b) {
            const int64_t o = a / total_inner;
            int64_t w = a % total_inner, k = 0;
            while (w >= inner[(size_t)k]) w -= inner[(size_t)k++];
            const int64_t run = std::min<int64_t>(b - a, inner[(size_t)k] - w);
            const int rc = f((int)k, o * inner[(size_t)k] + w, run);
            if (rc < 0) return rc;
            a += run;
        }
        return RAD_OK;
    }
};

/* `n` elements of a stored-dtype piece from element `e`, as f32. */
int piece_f32(const SrcTensor& t, int64_t e, int64_t n, float* out) {
    if (t.ct_group) {
        /* Element (row, k) of the logical [N, K] matrix is nibble k % 8 of word [row, k / 8],
         * less 8. */
        const int64_t K = t.shape[1];
        for (int64_t i = 0; i < n; ++i) {
            const int64_t row = (e + i) / K, k = (e + i) % K;
            uint32_t w;
            std::memcpy(&w, (const uint8_t*)t.data + (row * (K / 8) + k / 8) * 4, 4);
            out[i] = (float)((int)((w >> ((k % 8) * 4)) & 0xFu) - 8);
        }
        return RAD_OK;
    }
    if (t.gptq_bits) {
        /* Element (row, k) of the logical [N, K] matrix is field k % (32/bits) of word
         * [k / (32/bits), row], less the zero point. */
        const int64_t N = t.shape[0], K = t.shape[1];
        const int64_t per = 32 / t.gptq_bits;
        const uint32_t mask = (1u << t.gptq_bits) - 1u;
        const int zp = 1 << (t.gptq_bits - 1);
        for (int64_t i = 0; i < n; ++i) {
            const int64_t row = (e + i) / K, k = (e + i) % K;
            uint32_t w;
            std::memcpy(&w, (const uint8_t*)t.data + ((k / per) * N + row) * 4, 4);
            out[i] = (float)((int)((w >> ((k % per) * t.gptq_bits)) & mask) - zp);
        }
        return RAD_OK;
    }
    if (t.rad_dtype == RAD_DT_INVALID) {
        /* A GGUF block type: whole rows only (a row is whole blocks). */
        const int64_t cols = t.shape.back();
        if (cols <= 0 || e % cols || n % cols) return RAD_E_SHAPE;
        const int64_t rows = t.numel() / cols, rb = t.bytes / (rows ? rows : 1);
        return ggml_rows_f32(t.ggml, (const uint8_t*)t.data + (e / cols) * rb, n, out);
    }
    const uint8_t* p = (const uint8_t*)t.data;
    switch (t.rad_dtype) {
        case RAD_F32:
            std::memcpy(out, (const float*)t.data + e, (size_t)n * 4);
            return RAD_OK;
        case RAD_BF16: {
            /* bf16 is the top half of an f32: a shift in a loop the compiler vectorises (the
             * per-element switch below does not). Most checkpoints are mostly bf16. */
            const uint16_t* q = (const uint16_t*)t.data + e;
            for (int64_t i = 0; i < n; ++i) {
                const uint32_t u = (uint32_t)q[i] << 16;
                std::memcpy(out + i, &u, 4);
            }
            return RAD_OK;
        }
        case RAD_F16: case RAD_F8E4M3: case RAD_F8E5M2: case RAD_FP4E2M1:
        case RAD_I8: case RAD_U8: case RAD_I16: case RAD_I32: case RAD_U32: case RAD_U16:
            for (int64_t i = 0; i < n; ++i) out[i] = rad_load_f32(p, t.rad_dtype, e + i);
            return RAD_OK;
        default:
            return RAD_E_UNSUPPORTED;
    }
}

}  /* namespace */

int ckpt_rows_f32(const CkptWeight& w, int64_t r0, int64_t nr, float* out, std::string* why) {
    auto refuse = [&](int rc, std::string m) {
        if (why) *why = std::move(m);
        return rc;
    };
    if (r0 < 0 || nr < 0 || r0 + nr > w.rows) return refuse(RAD_E_SHAPE, "rows out of range");
    /* A rotated weight's codes are in the rotated basis; its f32 value needs the container's tables
     * to decode, and nothing re-quantises it. */
    if (w.paro)
        return refuse(RAD_E_UNSUPPORTED, "'" + w.name + "' is a ParoQuant weight, stored rotated; "
                                         "it is kept as it is and has no f32 view here");
    Joined j;
    j.init(w.pieces, w.concat_dim);
    /* Runs arrive in order, so the cursor is the count already written. */
    int64_t at = 0;
    const int rc = j.walk(r0 * w.cols, (r0 + nr) * w.cols, [&](int k, int64_t e, int64_t n) {
        const int s = piece_f32(*w.pieces[(size_t)k], e, n, out + at);
        at += n;
        return s;
    });
    if (rc < 0)
        return refuse(rc, "'" + w.pieces[0]->name + "' is stored in a form that cannot be read "
                          "as f32 (" + (w.pieces[0]->rad_dtype == RAD_DT_INVALID
                                            ? std::string(gguf_dtype(w.pieces[0]->ggml)
                                                              ? gguf_dtype(w.pieces[0]->ggml)->name
                                                              : "?")
                                            : std::string(rad_dtype_name(w.pieces[0]->rad_dtype))) +
                          ")");
    /* An imported code times its group's shipped scale: GPTQ's scales are [groups, N] (row r's are
     * a column), CT-INT4's are [N, groups] (a row). An NVFP4 import has neither group and takes
     * the scale-plane form below. */
    if (w.imported && (w.pieces[0]->gptq_group || w.pieces[0]->ct_group)) {
        int64_t base = 0;
        for (size_t k = 0; k < w.pieces.size(); ++k) {
            const SrcTensor& pk = *w.pieces[k];
            const SrcTensor& sk = *w.scales[k];
            const int64_t pr = pk.shape[0];
            for (int64_t r = std::max(r0, base); r < std::min(r0 + nr, base + pr); ++r)
                for (int64_t c = 0; c < w.cols; ++c)
                    out[(r - r0) * w.cols + c] *= rad_load_f32(
                        sk.data, sk.rad_dtype,
                        pk.ct_group ? (r - base) * (w.cols / pk.ct_group) + c / pk.ct_group
                                    : (c / pk.gptq_group) * pr + (r - base));
            base += pr;
        }
        return RAD_OK;
    }
    /* A code times its scale-plane element (FP8's block or row scale, MXFP4's exponent, NVFP4's
     * folded scale). Pieces join along rows, so logical row r takes its covering piece's plane.
     * A weight whose planes ckpt_resolve could not state an encoding for (mixed shapes, a
     * column join, a piece without a plane) is refused rather than read with a guessed block. */
    bool has_scale = false;
    for (const SrcTensor* s : w.scales)
        if (s) { has_scale = true; break; }
    if (has_scale) {
        if (w.enc.n_planes < 2)
            return refuse(RAD_E_UNSUPPORTED, "'" + w.name + "' has a scale plane with no stated "
                                             "encoding" + (w.why.empty() ? "" : ": " + w.why));
        /* A block extent of 0 is the whole extent (one scale per row or per column). */
        const int64_t br0 = w.enc.plane[1].block[0], bc0 = w.enc.plane[1].block[1];
        int64_t base = 0;
        for (size_t k = 0; k < w.pieces.size(); ++k) {
            const SrcTensor& pk = *w.pieces[k];
            const SrcTensor& sk = *w.scales[k];
            int64_t pr = 0, pc = 0;
            rad_enc_view((uint32_t)pk.shape.size(), pk.shape.data(), &pr, &pc);
            const int64_t br = br0 > 0 ? br0 : pr, bc = bc0 > 0 ? bc0 : pc;
            const int64_t sc = (pc + bc - 1) / bc;
            for (int64_t r = std::max(r0, base); r < std::min(r0 + nr, base + pr); ++r)
                for (int64_t c = 0; c < w.cols; ++c)
                    out[(r - r0) * w.cols + c] *=
                        rad_load_f32(sk.data, sk.rad_dtype, ((r - base) / br) * sc + c / bc);
            base += pr;
        }
    }
    return RAD_OK;
}

int ckpt_plane_rows(const CkptWeight& w, int k, int64_t r0, int64_t nr, void* dst,
                    std::string* why) {
    auto refuse = [&](int rc, std::string m) {
        if (why) *why = std::move(m);
        return rc;
    };
    if (!(w.trivial || w.imported) || k < 0 || k >= w.enc.n_planes)
        return refuse(RAD_E_STATE, "not a plane of a trivial or imported encoding");
    const std::vector<const SrcTensor*>& src = k == 0 ? w.pieces : w.scales;
    int64_t pr = 0, pc = 0;
    rad_enc_plane_dims(&w.enc.plane[k], w.rows, w.cols, &pr, &pc);
    if (r0 < 0 || nr < 0 || r0 + nr > pr) return refuse(RAD_E_SHAPE, "plane rows out of range");
    const uint32_t dt = w.enc.plane[k].dtype;
    const int64_t eb = rad_dtype_bits(dt) / 8;
    uint8_t* o = (uint8_t*)dst;
    if (w.paro && w.pieces[0]->pbits && k <= 2) {
        /* Paro int5-bitplane, re-laid one row of the logical [N, K] matrix at a time. The
         * checkpoint packs along N: qweight [K, N/8] holds eight low nibbles per word in AWQ order
         * (column c of a group of eight is in nibble nib(c)), and qweight_hi [K, N/32] the fifth
         * bit of 32 columns at bit n % 32. Zero points use the same two packings over [G, N]. */
        auto nib = [](int c) { return c % 2 == 0 ? c / 2 : 4 + c / 2; };      /* AWQ order */
        auto u32 = [](const SrcTensor& t, int64_t i) {
            uint32_t v;
            std::memcpy(&v, (const uint8_t*)t.data + i * 4, 4);
            return v;
        };
        int64_t base = 0;
        for (size_t pi = 0; pi < w.pieces.size(); ++pi) {
            const SrcTensor* pc = w.pieces[pi];
            const int64_t Np = pc->shape[0], K = pc->shape[1], G = K / 128;
            const SrcTensor *hip = pc->aux[0], *qzp = pc->aux[1], *qzhp = pc->aux[2];
            const SrcTensor& sc = *w.scales[pi];
            for (int64_t r = std::max(r0, base); r < std::min(r0 + nr, base + Np); ++r) {
                const int64_t n = r - base;
                const int cn = (int)(n % 8);
                if (k == 0) {
                    const int pb = pc->pbits;
                    const int64_t rb = (K * pb + 7) / 8;
                    std::memset(o, 0, (size_t)rb);
                    for (int64_t kk = 0; kk < K; ++kk) {
                        const uint32_t lo = (u32(*pc, kk * (Np / 8) + n / 8) >> (4 * nib(cn))) & 15u;
                        const uint32_t h = pb == 5 ? (u32(*hip, kk * (Np / 32) + n / 32) >> (n % 32)) & 1u : 0u;
                        const uint32_t code = lo | (h << 4);
                        const int64_t bit = kk * pb;
                        const uint32_t sh = (uint32_t)(bit & 7);
                        const uint32_t v = code << sh;
                        o[bit >> 3] |= (uint8_t)(v & 0xFFu);
                        if (sh + pb > 8) o[(bit >> 3) + 1] |= (uint8_t)(v >> 8);
                    }
                    o += rb;
                } else if (k == 1) {
                    for (int64_t g = 0; g < G; ++g) {
                        std::memcpy(o, (const uint8_t*)sc.data + (g * Np + n) * 2, 2);
                        o += 2;
                    }
                } else {
                    for (int64_t g = 0; g < G; ++g) {
                        const uint32_t lo = (u32(*qzp, g * (Np / 8) + n / 8) >> (4 * nib(cn))) & 15u;
                        const uint32_t h = pc->pbits == 5 ? (u32(*qzhp, g * (Np / 32) + n / 32) >> (n % 32)) & 1u : 0u;
                        *o++ = (uint8_t)(lo | (h << 4));
                    }
                }
            }
            base += Np;
        }
        return RAD_OK;
    }
    if (w.paro && std::strncmp(w.enc.plane[k].role, "t.", 2) == 0) {
        /* A rotation table, built a row at a time: pairs converted from group-local to whole-row
         * columns, angles widened, the channel scale inverted (the checkpoint multiplies the
         * activation by it; t.scale is the factor the stored row was multiplied by), and the part
         * ends. */
        const RadEncPlane& pl = w.enc.plane[k];
        const int64_t P = (int64_t)w.pieces.size(), S = w.pq_pairs[0]->shape[0], K = w.cols;
        const std::string role = pl.role;
        for (int64_t r = r0; r < r0 + nr; ++r) {
            if (role == "t.pairs") {
                const int16_t* pv = (const int16_t*)w.pq_pairs[(size_t)(r / S)]->data + (r % S) * K;
                int32_t* d = (int32_t*)o;
                for (int64_t c = 0; c < K; ++c)
                    d[c] = (int32_t)((c / kParoGroup) * kParoGroup + pv[c]);
                o += K * 4;
            } else if (role == "t.angle") {
                const SrcTensor& th = *w.pq_theta[(size_t)(r / S)];
                float* d = (float*)o;
                for (int64_t c = 0; c < K / 2; ++c)
                    d[c] = rad_load_f32(th.data, th.rad_dtype, (r % S) * (K / 2) + c);
                o += K / 2 * 4;
            } else if (role == "t.scale") {
                const SrcTensor& cs = *w.pq_scale[(size_t)r];
                float* d = (float*)o;
                for (int64_t c = 0; c < K; ++c) d[c] = 1.0f / rad_load_f32(cs.data, cs.rad_dtype, c);
                o += K * 4;
            } else if (role == "t.part") {
                int32_t* d = (int32_t*)o;
                int64_t end = 0;
                for (int64_t q = 0; q < P; ++q) {
                    end += w.pieces[(size_t)q]->shape[0];
                    d[q] = (int32_t)end;
                }
                o += P * 4;
            } else if (role == "t.dense") {
                /* Row r is part r / (K/128), group r % (K/128): its group-local stages composed. */
                const int64_t nkb = K / kParoGroup, H = kParoGroup / 2, q = r / nkb, g = r % nkb;
                const int16_t* pv = (const int16_t*)w.pq_pairs[(size_t)q]->data;
                const SrcTensor& th = *w.pq_theta[(size_t)q];
                std::vector<int32_t> ia((size_t)(S * H)), ib((size_t)(S * H));
                std::vector<float> an((size_t)(S * H));
                for (int64_t st = 0; st < S; ++st)
                    for (int64_t j = 0; j < H; ++j) {
                        const int64_t at = st * H + j;
                        ia[(size_t)at] = pv[st * K + g * kParoGroup + 2 * j];
                        ib[(size_t)at] = pv[st * K + g * kParoGroup + 2 * j + 1];
                        an[(size_t)at] = rad_load_f32(th.data, th.rad_dtype, st * (K / 2) + g * H + j);
                    }
                paro_compose_dense(S, ia.data(), ib.data(), an.data(), (uint16_t*)o);
                o += kParoGroup * kParoGroup * 2;
            } else {
                return refuse(RAD_E_STATE, "plane '" + role + "' of a ParoQuant weight is not a table");
            }
        }
        return RAD_OK;
    }
    if (w.imported && !w.paro && (w.pieces[0]->gptq_group || w.pieces[0]->ct_group)) {
        /* GPTQ-packed, re-laid: a row of codes is a column of the qweight, its words' bytes in order
         * (int4: two codes per byte, low nibble first), made signed by flipping each field's top
         * bit; a row of scales is a column of the scale tensor, one per 128 of K, rounded to
         * bf16. */
        int64_t base = 0;
        for (size_t p = 0; p < w.pieces.size(); ++p) {
            const SrcTensor& c = *w.pieces[p];
            const SrcTensor& s = *w.scales[p];
            const int64_t N = c.shape[0], K = c.shape[1];
            if (c.ct_group) {
                /* CT-INT4, re-laid: a row of codes is a row of the packed words (already two per
                 * byte, low nibble first), made signed by flipping each top bit; a row of scales is
                 * a row of the scale tensor, as shipped. */
                for (int64_t r = std::max(r0, base); r < std::min(r0 + nr, base + N); ++r) {
                    const int64_t n = r - base;
                    if (k == 0) {
                        const uint8_t* q = (const uint8_t*)c.data + n * (K / 2);
                        for (int64_t i = 0; i < K / 2; ++i) *o++ = q[i] ^ 0x88;
                    } else {
                        const int64_t g = K / c.ct_group, sb = rad_dtype_bits(s.rad_dtype) / 8;
                        std::memcpy(o, (const uint8_t*)s.data + n * g * sb, (size_t)(g * sb));
                        o += g * sb;
                    }
                }
                base += N;
                continue;
            }
            const uint8_t flip = c.gptq_bits == 4 ? 0x88 : 0x80;
            for (int64_t r = std::max(r0, base); r < std::min(r0 + nr, base + N); ++r) {
                const int64_t n = r - base;
                if (k == 0) {
                    const uint8_t* q = (const uint8_t*)c.data;
                    for (int64_t wi = 0; wi < K * c.gptq_bits / 32; ++wi) {
                        uint32_t v;
                        std::memcpy(&v, q + (wi * N + n) * 4, 4);
                        for (int b = 0; b < 4; ++b) *o++ = (uint8_t)(v >> (8 * b)) ^ flip;
                    }
                } else {
                    for (int64_t g = 0; g < K / 128; ++g) {
                        const uint16_t h = rad_f32_to_bf16(
                            rad_load_f32(s.data, s.rad_dtype, ((g * 128) / c.gptq_group) * N + n));
                        std::memcpy(o, &h, 2);
                        o += 2;
                    }
                }
            }
            base += N;
        }
        return RAD_OK;
    }
    if (k == 0) {
        /* Codes: the pieces joined, bytes as stored. Sub-byte codes (MXFP4) join along rows of even
         * width, so every run is byte-aligned. */
        const int64_t bits = rad_dtype_bits(dt);
        Joined j;
        j.init(src, w.concat_dim);
        return j.walk(r0 * pc, (r0 + nr) * pc, [&](int p, int64_t e, int64_t n) {
            if ((e * bits) % 8 || (n * bits) % 8) return (int)RAD_E_SHAPE;
            std::memcpy(o, (const uint8_t*)src[(size_t)p]->data + e * bits / 8, (size_t)(n * bits / 8));
            o += n * bits / 8;
            return (int)RAD_OK;
        });
    }
    /* A scale plane: the pieces' scale rows in sequence. */
    int64_t base = 0;
    for (const SrcTensor* s : src) {
        int64_t sr = 0, scl = 0;
        rad_enc_view((uint32_t)s->shape.size(), s->shape.data(), &sr, &scl);
        if (s->shape.size() == 1) { sr = s->shape[0]; scl = 1; }       /* one scale per row */
        const int64_t lo = std::max(r0, base), hi = std::min(r0 + nr, base + sr);
        if (hi > lo) {
            std::memcpy(o, (const uint8_t*)s->data + (lo - base) * scl * eb,
                        (size_t)((hi - lo) * scl * eb));
            o += (hi - lo) * scl * eb;
        }
        base += sr;
    }
    return RAD_OK;
}

}  /* namespace rad */
