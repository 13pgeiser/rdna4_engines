/* arch_test.cpp -- the declared graph of the architecture plugins, asserted statically.
 *
 * No GPU, no model file, no core. With two-phase construction (spec §3) the whole model is known
 * between declare and plan, so a unit test can check what a plugin declares. A fake builder
 * records every decl_weight, decl_buffer, decl_op, buffer use and name-map call, and a fake RadCtx
 * records every issue, so the run phase's order and operands are checked too.
 *
 * The fake is deliberately not core/build/'s RadBuilder: a defect there must not fail this test,
 * and a plugin that only works against one builder has an undeclared dependency.
 */
#define RAD_ARCH_NO_EXPORTS 1

#include "rad_test.h"

#include "rad_builder.h"
#include "rad_device.h"
#include "rad_runtime.h"

#include <algorithm>
#include <cstdarg>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <vector>
#include <array>
#include <thread>

/* ==================================================================== the recording builder */
namespace {

struct RecParam {
    std::string key;
    int         kind = 0;
    long long   ival = 0, ihi = 0;
    double      dval = 0.0;
    std::string sval;
};

struct RecWeight {
    std::string     name;
    RadWeightDecl   d{};
    /* The uneven slice this weight was handed (rad_weight_shard_span), in the container's
     * unsharded units, so tests can check which checkpoint rows each rank reads. Even shards have
     * `span_set` false. */
    bool            span_set = false;
    int64_t         span_lo = 0, span_hi = 0;
    /* rad_weight_shard_span_parts, recorded likewise. */
    int             n_part_spans = 0;
    int64_t         part_lo[RAD_MAX_ROW_PARTS] = {0}, part_hi[RAD_MAX_ROW_PARTS] = {0};
    int64_t         part_total[RAD_MAX_ROW_PARTS] = {0};
};
struct RecBuf    { std::string name; RadBufDecl    d{}; };
struct RecOp {
    std::string             op;
    std::vector<RecParam>   p;
    std::vector<rad_weight> w;
    std::vector<rad_buf>    reads, writes;
    bool                    resolved = true;
    bool                    uses_declared = false;   /* rad_op_reads/writes were called */
};
struct RecMap {
    std::string              declared;
    int                      mode = 0;
    std::vector<std::string> src;
    int                      concat_dim = 0;
};
struct RecKV    { std::string name; RadKVGroupDecl d{}; bool zero = false; };
struct RecIssue {
    rad_op                  op = 0;
    std::vector<RadOperand> opd;
    int64_t                 n = 0;
    int                     lane = 0;   /* the lane it was issued on (rad_lane) */
};
/* A rad_lane_join: `to` behind everything `from` issued before issue number `at`. */
struct RecJoin  { size_t at = 0; int from = 0, to = 0; };

}  /* namespace */

struct RadBuilder {
    std::vector<RecWeight> weights;
    std::vector<RecBuf>    bufs;
    std::vector<RecOp>     ops;
    std::vector<RecMap>    maps;
    std::vector<RecKV>     kv_groups;
    std::vector<std::pair<int, rad_kvgroup>> binds;
    std::vector<std::string> notes;

    /* Ops the fake kernel hierarchy refuses, to drive the fused-op fallback; and ops no loaded
     * library declares (rad_op_declared), as without an optional library. */
    std::set<std::string> refuse;
    std::set<std::string> undeclared;
    /* When set, the only ops any loaded library declares (a CPU-only engine's: libref's
     * vocabulary); every other op is undeclared. */
    const std::set<std::string>* vocab = nullptr;
    std::vector<std::string> asked_undeclared;   /* what a real engine would refuse */
    bool declared(const std::string& op) const {
        return vocab ? vocab->count(op) != 0 : undeclared.count(op) == 0;
    }
    int64_t kv_block = 16;

    /* The buffer the plugin named as this step's logits (rad_declare_logits), or as its pooled
     * output (rad_declare_pooled). */
    rad_buf logits = 0;
    rad_buf pooled = 0;

    /* The draft head's declaration (rad_declare_drafter), and whether it keeps no cache of its
     * own (rad_drafter_shared_kv) and at which rotary offset. */
    RadDrafterDecl drafter{};
    std::string    drafter_name;
    bool           drafter_shared = false;
    int            drafter_pos = -1;

    /* The buffers named to rad_buf_concurrent: the ones a second lane touches. */
    std::vector<rad_buf> concurrent;

    /* The vision and audio encoder declarations (rad_declare_encoder), if any. */
    RadEncoderDecl encoder{};
    std::string    encoder_name;
    RadEncoderDecl audio_encoder{};
    std::string    audio_encoder_name;

    /* The image generator's declaration (rad_declare_generator), if any. */
    RadGeneratorDecl generator{};
    std::string      generator_name;

    /* Adapter key names (rad_lora_root, rad_lora_alias): (native, spellings), and each alias as
     * (native, from, to, from_lo, from_hi, to_lo). */
    std::vector<std::pair<std::string, std::string>> lora_roots;
    struct LoraAliasRec { std::string native, from, to; int64_t from_lo, from_hi, to_lo; };
    std::vector<LoraAliasRec> lora_aliases;

    /* What rad_weight_encoding answers: the first entry whose key the source name contains. Empty
     * means no model, so every weight is plain in its declared dtype. */
    std::vector<std::pair<std::string, RadEncoding>> encs;

    const RecOp* op_of(rad_op h) const { return h ? &ops[h - 1] : nullptr; }
};

struct RadCtx {
    RadBuilder*     b     = nullptr;
    const RadBatch* batch = nullptr;
    int             rank  = 0;
    int             world = 1;
    std::vector<RecIssue> issues;
    std::vector<int>      routed_layers;
    std::string           step_fail;
    int                   lane = 0;     /* where the next issue goes */
    std::vector<RecJoin>  joins;
    /* Device copies into a buffer (rad_memcpy_async to a rad_buf_ptr address): the buffer and the
     * issue number the copy came before. The step lint counts each as a write there. */
    std::vector<std::pair<size_t, rad_buf>> copies;
};

/* -------------------------------------------------------------------- the declare-side ABI */
extern "C" rad_weight rad_decl_weight(RadBuilder* b, const char* name, const RadWeightDecl* d) {
    b->weights.push_back({name, *d});
    return (rad_weight)b->weights.size();
}

extern "C" int rad_weight_encoding(RadBuilder* b, const char* source, RadEncoding* out, int64_t*,
                                   uint32_t*) {
    for (const auto& [key, e] : b->encs)
        if (source && std::strstr(source, key.c_str())) { *out = e; return RAD_OK; }
    return RAD_E_NOTFOUND;
}
extern "C" rad_buf rad_decl_buffer(RadBuilder* b, const char* name, const RadBufDecl* d) {
    b->bufs.push_back({name, *d});
    return (rad_buf)b->bufs.size();
}

extern "C" rad_op rad_decl_op(RadBuilder* b, const char* op, const RadParam* params, int n_params,
                              const rad_weight* weights, int n_weights) {
    if (!b->declared(op)) {
        b->asked_undeclared.push_back(op);
        return RAD_NULL_HANDLE;
    }
    RecOp r;
    r.op = op;
    for (int i = 0; i < n_params; ++i) {
        RecParam p;
        p.key  = params[i].key ? params[i].key : "";
        p.kind = params[i].kind;
        p.ival = params[i].ival;
        p.ihi  = params[i].ihi;
        p.dval = params[i].dval;
        p.sval = params[i].sval ? params[i].sval : "";
        r.p.push_back(p);
    }
    for (int i = 0; i < n_weights; ++i) r.w.push_back(weights[i]);
    r.resolved = b->refuse.count(r.op) == 0;
    b->ops.push_back(r);
    /* Declare always completes. An unresolved op is RAD_NULL_HANDLE, an error only if the plugin
     * issues it (rad_builder.h). */
    return r.resolved ? (rad_op)b->ops.size() : RAD_NULL_HANDLE;
}

extern "C" int rad_op_reads(RadBuilder* b, rad_op h, const rad_buf* bufs, int n) {
    if (!h) return RAD_E_INVAL;
    RecOp& r = b->ops[h - 1];
    r.uses_declared = true;
    for (int i = 0; i < n; ++i) r.reads.push_back(bufs[i]);
    return RAD_OK;
}

extern "C" int rad_op_writes(RadBuilder* b, rad_op h, const rad_buf* bufs, int n) {
    if (!h) return RAD_E_INVAL;
    RecOp& r = b->ops[h - 1];
    r.uses_declared = true;
    for (int i = 0; i < n; ++i) r.writes.push_back(bufs[i]);
    return RAD_OK;
}

extern "C" int rad_op_resolved(RadBuilder* b, rad_op h) {
    return h != RAD_NULL_HANDLE && b->ops[h - 1].resolved;
}

extern "C" int rad_op_declared(RadBuilder* b, const char* op) {
    return op && b->declared(op);
}

extern "C" rad_kvgroup rad_decl_kv_group(RadBuilder* b, const char* name,
                                         const RadKVGroupDecl* d) {
    b->kv_groups.push_back({name, *d, false});
    return (rad_kvgroup)b->kv_groups.size();
}

extern "C" int64_t rad_kv_block_size(RadBuilder* b, rad_kvgroup g) { (void)g; return b->kv_block; }

extern "C" int rad_bind_layer_kv(RadBuilder* b, int layer, rad_kvgroup g) {
    b->binds.push_back({layer, g});
    return RAD_OK;
}

extern "C" int rad_decl_name_map(RadBuilder* b, const RadNameMap* m) {
    RecMap r;
    r.declared   = m->declared;
    r.mode       = m->mode;
    r.concat_dim = m->concat_dim;
    for (int i = 0; i < m->n_src; ++i) r.src.push_back(m->src[i]);
    b->maps.push_back(r);
    return RAD_OK;
}

extern "C" void rad_note(RadBuilder* b, const char* fmt, ...) {
    char buf[512];
    va_list ap; va_start(ap, fmt); vsnprintf(buf, sizeof buf, fmt, ap); va_end(ap);
    b->notes.push_back(buf);
}

/* The buffer the core samples from, recorded so tests can assert the plugin named the narrow
 * one: [max_tok, n_vocab] is legal but 8 GiB at this vocabulary. */
extern "C" int rad_declare_logits(RadBuilder* b, rad_buf logits) {
    if (!b || !logits) return RAD_E_INVAL;
    b->logits = logits;
    return RAD_OK;
}

/* An embedding program names its pooled output instead of logits; a program declares one. */
extern "C" int rad_declare_pooled(RadBuilder* b, rad_buf pooled) {
    if (!b || !pooled || b->logits) return RAD_E_INVAL;
    b->pooled = pooled;
    return RAD_OK;
}

/* An image generator's declaration, recorded whole; it too excludes logits. */
extern "C" int rad_declare_generator(RadBuilder* b, const RadGeneratorDecl* d) {
    if (!b || !d || !d->image_out || b->logits || b->pooled) return RAD_E_INVAL;
    b->generator = *d;
    b->generator_name = d->name ? d->name : "";
    return RAD_OK;
}

/* Recorded: a generator's adapter key names. */
extern "C" int rad_lora_root(RadBuilder* b, const char* native, const char* spellings) {
    if (!b || !native || !*native) return RAD_E_INVAL;
    b->lora_roots.push_back({ native, spellings ? spellings : "" });
    return RAD_OK;
}
extern "C" int rad_lora_alias(RadBuilder* b, const char* native, const char* from, const char* to,
                              int64_t from_lo, int64_t from_hi, int64_t to_lo) {
    if (!b || !native || !from || !to) return RAD_E_INVAL;
    b->lora_aliases.push_back({ native, from, to, from_lo, from_hi, to_lo });
    return RAD_OK;
}

/* Recorded: gemma4's prefill overlap names the buffers its second lane reduces in place. */
extern "C" int rad_buf_concurrent(RadBuilder* b, rad_buf buf) {
    if (!b || !buf) return RAD_E_INVAL;
    b->concurrent.push_back(buf);
    return RAD_OK;
}
extern "C" int rad_weight_shard_span(RadBuilder* b, rad_weight w, int64_t lo, int64_t hi) {
    if (!b || w == 0 || w > b->weights.size()) return RAD_E_INVAL;
    if (lo < 0 || hi <= lo) return RAD_E_INVAL;
    /* The real Builder also checks the declared extent against hi-lo (core/build/rad_builder.cpp);
     * the fake only records the span for tests. */
    RecWeight& rw = b->weights[w - 1];
    rw.span_set = true;
    rw.span_lo  = lo;
    rw.span_hi  = hi;
    return RAD_OK;
}
extern "C" int rad_weight_shard_span_parts(RadBuilder* b, rad_weight w, const int64_t* lo,
                                           const int64_t* hi, const int64_t* total, int n) {
    if (!b || w == 0 || w > b->weights.size() || !lo || !hi || !total) return RAD_E_INVAL;
    if (n < 2 || n > RAD_MAX_ROW_PARTS) return RAD_E_INVAL;
    RecWeight& rw = b->weights[w - 1];
    if (rw.d.n_row_parts != n) return RAD_E_INVAL;
    for (int g = 0; g < n; ++g) {
        if (lo[g] < 0 || hi[g] <= lo[g] || hi[g] > total[g]) return RAD_E_INVAL;
        if (rw.d.row_parts[g] != hi[g] - lo[g]) return RAD_E_SHAPE;
        rw.part_lo[g] = lo[g];
        rw.part_hi[g] = hi[g];
        rw.part_total[g] = total[g];
    }
    rw.n_part_spans = n;
    return RAD_OK;
}
extern "C" int rad_kv_group_zero(RadBuilder* b, rad_kvgroup g) {
    if (!b || g == 0 || g > b->kv_groups.size()) return RAD_E_INVAL;
    b->kv_groups[g - 1].zero = true;
    return RAD_OK;
}
extern "C" int rad_declare_drafter(RadBuilder* b, const RadDrafterDecl* d) {
    if (!b || !d) return RAD_E_INVAL;
    b->drafter = *d;
    b->drafter_name = d->name ? d->name : "";
    return RAD_OK;
}
extern "C" int rad_drafter_shared_kv(RadBuilder* b, int pos_offset) {
    if (!b || b->drafter.kind != RAD_DRAFT_SERIAL || (pos_offset != 0 && pos_offset != 1))
        return RAD_E_STATE;
    b->drafter_shared = true;
    b->drafter_pos = pos_offset;
    return RAD_OK;
}
extern "C" int rad_declare_encoder(RadBuilder* b, const RadEncoderDecl* d) {
    if (!b || !d || !d->name || !d->modalities) return RAD_E_INVAL;
    if (d->modalities == RAD_MM_AUDIO) {
        b->audio_encoder = *d;
        b->audio_encoder_name = d->name;
        return RAD_OK;
    }
    b->encoder = *d;
    b->encoder_name = d->name;
    return RAD_OK;
}

/* -------------------------------------------------------------------- model metadata */
extern "C" long long rad_meta_geti(const RadModelMeta* m, const char* key, long long dflt) {
    for (int i = 0; i < m->n_kv; ++i)
        if (std::strcmp(m->kv_key[i], key) == 0) return atoll(m->kv_val[i]);
    return dflt;
}
extern "C" double rad_meta_getf(const RadModelMeta* m, const char* key, double dflt) {
    for (int i = 0; i < m->n_kv; ++i)
        if (std::strcmp(m->kv_key[i], key) == 0) return atof(m->kv_val[i]);
    return dflt;
}
extern "C" const char* rad_meta_gets(const RadModelMeta* m, const char* key, const char* dflt) {
    for (int i = 0; i < m->n_kv; ++i)
        if (std::strcmp(m->kv_key[i], key) == 0) return m->kv_val[i];
    return dflt;
}

/* -------------------------------------------------------------------- the run-phase ABI */
extern "C" int rad_issue(RadCtx* c, rad_op op, const RadOperand* opd, int n_opd, int64_t n) {
    RecIssue r;
    r.op = op;
    r.n  = n;
    for (int i = 0; i < n_opd; ++i) r.opd.push_back(opd[i]);
    r.lane = c->lane;
    c->issues.push_back(r);
    return RAD_OK;
}
/* The second lane, recorded: each issue's lane, and each join's position among the issues. */
extern "C" int rad_lane(RadCtx* c, int lane) {
    if (lane != 0 && lane != 1) return RAD_E_INVAL;
    c->lane = lane;
    return RAD_OK;
}
extern "C" int rad_lane_join(RadCtx* c, int from, int to) {
    if (from < 0 || from > 1 || to < 0 || to > 1) return RAD_E_INVAL;
    c->joins.push_back({ c->issues.size(), from, to });
    return RAD_OK;
}
/* A block that refuses the batch records the reason for the tests. */
extern "C" int rad_step_fail(RadCtx* c, const char* what) {
    c->step_fail = what ? what : "";
    return RAD_E_SHAPE;
}
extern "C" const RadBatch* rad_batch(RadCtx* c)      { return c->batch; }
extern "C" int             rad_rank(RadCtx* c)       { return c->rank; }
extern "C" int             rad_world_size(RadCtx* c) { return c->world; }
/* The context itself, so a copy issued on the step's stream is recorded in it (rad_memcpy_async). */
extern "C" RadStream       rad_stream(RadCtx* c)     { return (RadStream)c; }
extern "C" int             rad_stream_sync(RadStream s) { (void)s; return RAD_OK; }
extern "C" int rad_route_report(RadCtx* c, int layer, const RadRouting* r) {
    (void)r;
    c->routed_layers.push_back(layer);
    return RAD_OK;
}
/* No card-side histogram here, so a router writes and reports its own buffer. */
extern "C" int32_t* rad_route_counts(RadCtx* c, int layer, int64_t n_expert) {
    (void)c; (void)layer; (void)n_expert;
    return nullptr;
}
/* No prefetch here: the three calls are no-ops, as in a core without one. */
extern "C" int rad_expert_prefetch(RadCtx* c, const RadPrefetchNext* next, rad_buf h, int64_t r0,
                                   int64_t rows, int64_t n_embd) {
    (void)c; (void)next; (void)h; (void)r0; (void)rows; (void)n_embd;
    return RAD_OK;
}
extern "C" int rad_expert_prefetch_mark(RadCtx* c) { (void)c; return RAD_OK; }
extern "C" int rad_expert_prefetch_step(RadCtx* c, int layer) { (void)c; (void)layer; return RAD_OK; }
/* Never dereferenced: a plugin passes these through to rad_route_report or offsets one into a raw
 * operand. Each handle gets its own 2^36-byte window above any user-space address, tagged by
 * kind in the top bits, so the step lint can map a raw operand inside a buffer back to that buffer
 * (fake_buf_of). */
constexpr int       kFakePtrShift = 36;
constexpr uintptr_t kFakeBufTag   = (uintptr_t)1 << 61;
constexpr uintptr_t kFakeWeightTag = (uintptr_t)1 << 62;
extern "C" void* rad_weight_ptr(RadCtx* c, rad_weight w) {
    (void)c;
    return (void*)(kFakeWeightTag | (uintptr_t)w << kFakePtrShift);
}
extern "C" void* rad_buf_ptr(RadCtx* c, rad_buf b) {
    (void)c;
    return (void*)(kFakeBufTag | (uintptr_t)b << kFakePtrShift);
}
/* The buffer a raw operand points into, if it came from rad_buf_ptr; 0 otherwise. */
static rad_buf fake_buf_of(const void* p, size_t n_bufs) {
    const uintptr_t u = (uintptr_t)p;
    if ((u & ~(kFakeBufTag - 1)) != kFakeBufTag) return 0;
    const uintptr_t h = (u & (kFakeBufTag - 1)) >> kFakePtrShift;
    return h >= 1 && h <= n_bufs ? (rad_buf)h : 0;
}
/* Pinned host memory a plugin fills itself (qwen4exp_fp8's piece bounds, the calibration tap's
 * staging), as host memory; nothing else is handed out. */
extern "C" void* rad_dev_alloc(int64_t bytes, int kind) {
    return kind == RAD_MEM_HOST_PINNED && bytes > 0 ? std::calloc(1, (size_t)bytes) : nullptr;
}
extern "C" void  rad_dev_free(void* p, int kind) { if (kind == RAD_MEM_HOST_PINNED) std::free(p); }
/* The device copies: nothing is copied. One into a buffer on a step's stream is recorded as a
 * write of that buffer before the next issue (RadCtx::copies). */
extern "C" int rad_memcpy_async(void* dst, const void* src, int64_t bytes, RadStream s) {
    (void)src; (void)bytes;
    RadCtx* c = (RadCtx*)s;
    if (c && c->b)
        if (const rad_buf h = fake_buf_of(dst, c->b->bufs.size()))
            c->copies.push_back({ c->issues.size(), h });
    return RAD_OK;
}
extern "C" int rad_memset_async(void* dst, int value, int64_t bytes, RadStream s) {
    (void)dst; (void)value; (void)bytes; (void)s;
    return RAD_OK;
}

/* ==================================================================== the plugin under test */
#include "../arch/qwen35_bf16/qwen35_bf16.cpp"
/* The production architecture, for its sizing declare (RadBuildCtx::shape_probe). */
#include "../arch/qwen4exp_fp8/qwen4exp_fp8.cpp"
/* The LLaMA family, for how it reads its rotary scaling. */
#include "../arch/llama_fp8/llama_fp8.cpp"
/* LFM2, for its own layer schedule and short convolution. */
#include "../arch/lfm2_bf16/lfm2_bf16.cpp"
/* Gemma 4 dense: two attention classes (sliding-window + proportional-RoPE global) on one stream. */
#include "../arch/gemma4_bf16/gemma4_bf16.cpp"
/* The bf16 / fp8-expert Gemma 4 plugin: the 26B-A4B's dense and routed arms side by side. */
#include "../arch/gemma3/gemma3.cpp"
#include "../arch/gemma4/gemma4.cpp"
/* Qwen3-ASR: a Whisper-style audio tower over a dense Qwen3 decoder. */
#include "../arch/qwen3asr_bf16/qwen3asr_bf16.cpp"
/* The PLE block alone: no architecture here declares it, and what its step() hands the two
 * windowed ops is its own contract (rad_block_ple.h). */
#include <arch/rad_block_ple.h>
/* Qwen3.5 / 3.6 / 3.8 with block-fp8 weights, dense and routed: one source built twice under
 * three macros, as arch/qwen35moe_fp8 builds it. */
#include "../arch/qwen35_fp8/qwen35_fp8.cpp"
#undef QWEN35_MOE
#undef QWEN35_NS
#undef QWEN35_ARCH_ID
#include "../arch/qwen35moe_fp8/qwen35moe_fp8.cpp"
/* The same source a third time, as qwen35_w8a8: the int8 trunk. */
#undef QWEN35_MOE
#undef QWEN35_W8A8
#undef QWEN35_NS
#undef QWEN35_ARCH_ID
#include "../arch/qwen35_w8a8/qwen35_w8a8.cpp"
/* The LLaMA family from a dense checkpoint: llama_fp8 under its other descriptor. */
#undef LLAMA_DENSE_SRC
#undef LLAMA_NS
#undef LLAMA_QUANT
#include "../arch/llama_dense_fp8/llama_dense_fp8.cpp"
/* EmbeddingGemma 2: a pooled encoder with no logits. */
#include "../arch/embgemma2/embgemma2.cpp"
/* Qwen-Image 2.1: a generator conditioned by Qwen3-VL. */
#include "../arch/qwenimage21/qwenimage21.cpp"
/* Anima: a cross-attention generator conditioned through two tokenizers. */
#include "../arch/anima/anima.cpp"
/* GLM-5.3-Flash: its connection, router, attention and cache blocks, and the plugin. */
#include <arch/rad_block_mhc.h>
#include <arch/rad_block_glm_router.h>
#include <arch/rad_block_mla_latent.h>
#include <arch/rad_block_kda_state.h>
#include <arch/rad_block_dsa_cache.h>
#include <arch/rad_block_glm_kda.h>
#include <arch/rad_block_glm_mla.h>
#include <arch/rad_block_glm_indexer.h>
#include "../arch/glm5next/glm5next.cpp"

/* ==================================================================== fixtures and helpers */
namespace {

/* Qwen3.8-27B's shape cut to eight layers, so failures stay readable. Eight is the smallest count
 * that exercises the alternation twice: with full_attention_interval 4, layers 3 and 7 are full
 * attention and the other six gated delta net, the 64-layer model's 1:3 ratio.
 *
 * head_dim is 256 while n_embd / n_head is 213, so using the latter would be caught. */
const char* k_keys[] = {
    "qwen35.full_attention_interval",
    "qwen35.rope.dimension_count",
    "qwen35.ssm.state_size",
    "qwen35.ssm.group_count",
    "qwen35.ssm.time_step_rank",
    "qwen35.ssm.inner_size",
    "qwen35.ssm.conv_kernel",
};
const char* k_vals[] = { "4", "64", "128", "16", "48", "6144", "4" };

RadModelMeta qwen35_meta() {
    RadModelMeta m{};
    m.arch_id     = "qwen35";
    m.name        = "test-qwen35-27b";
    m.quant       = "bf16";
    m.n_layers    = 8;
    m.n_embd      = 5120;
    m.n_head      = 24;
    m.n_head_kv   = 4;
    m.head_dim    = 256;
    m.n_ff        = 17408;
    m.n_vocab     = 248320;
    m.n_ctx_train = 262144;
    m.rms_eps     = 1e-6f;
    m.rope_theta  = 1e7f;
    m.rope_scale  = 1.0f;
    m.n_kv        = (int)(sizeof(k_keys) / sizeof(k_keys[0]));
    m.kv_key      = k_keys;
    m.kv_val      = k_vals;
    return m;
}

/* Gemma 4 31B's text geometry cut to twelve layers, the smallest count with two global layers at
 * (l + 1) % 6 == 0 (layers 5 and 11) over ten sliding ones, the model's 5:1 ratio. head_dim 256 is
 * the sliding width; the global class's 512 width and 4 heads are header keys the plugin must
 * read, not RadModelMeta fields. */
const char* k_gem4_keys[] = {
    "sliding_window",
    "global_head_dim",
    "num_global_key_value_heads",
    "rope_parameters.sliding_attention.rope_theta",
    "rope_parameters.full_attention.rope_theta",
    "rope_parameters.full_attention.partial_rotary_factor",
    "num_kv_shared_layers",
    "final_logit_softcapping",
    "full_attention_interval",
    "attention_k_eq_v",
};
const char* k_gem4_vals[] = { "1024", "512", "4", "10000", "1000000", "0.25", "0", "30", "6", "1" };

RadModelMeta gemma4_meta() {
    RadModelMeta m{};
    m.arch_id     = "gemma4";
    m.name        = "test-gemma4-31b";
    m.quant       = "bf16";
    m.n_layers    = 12;
    m.n_embd      = 5376;
    m.n_head      = 32;
    m.n_head_kv   = 16;
    m.head_dim    = 256;
    m.n_ff        = 21504;
    m.n_vocab     = 262144;
    m.n_ctx_train = 262144;
    m.rms_eps     = 1e-6f;
    m.rope_theta  = 1e6f;
    m.rope_scale  = 1.0f;
    m.n_kv        = (int)(sizeof(k_gem4_keys) / sizeof(k_gem4_keys[0]));
    m.kv_key      = k_gem4_keys;
    m.kv_val      = k_gem4_vals;
    return m;
}

/* The same twelve layers with Gemma's assistant head merged under `draft.` as
 * `rad-convert --draft-model` writes it: the head's config flattened, its text config under
 * `draft.text_config.`. The 31B's head: width 1024, four layers (three sliding, one full), the
 * trunk's head geometry and window, and its own 262144-row tied table. */
const char* k_gem4d_keys[] = {
    "sliding_window", "global_head_dim", "num_global_key_value_heads",
    "rope_parameters.sliding_attention.rope_theta", "rope_parameters.full_attention.rope_theta",
    "rope_parameters.full_attention.partial_rotary_factor", "num_kv_shared_layers",
    "final_logit_softcapping", "full_attention_interval", "attention_k_eq_v",
    "draft.architectures", "draft.backbone_hidden_size", "draft.use_ordered_embeddings",
    "draft.text_config.hidden_size", "draft.text_config.num_hidden_layers",
    "draft.text_config.layer_types", "draft.text_config.num_attention_heads",
    "draft.text_config.num_key_value_heads", "draft.text_config.num_global_key_value_heads",
    "draft.text_config.head_dim", "draft.text_config.global_head_dim",
    "draft.text_config.intermediate_size", "draft.text_config.sliding_window",
    "draft.text_config.vocab_size", "draft.text_config.rms_norm_eps",
    "draft.text_config.rope_parameters.sliding_attention.rope_theta",
    "draft.text_config.rope_parameters.full_attention.rope_theta",
    "draft.text_config.rope_parameters.full_attention.partial_rotary_factor",
};
const char* k_gem4d_vals[] = {
    "1024", "512", "4", "10000", "1000000", "0.25", "0", "30", "6", "1",
    "Gemma4AssistantForCausalLM", "5376", "0",
    "1024", "4",
    "sliding_attention sliding_attention sliding_attention full_attention", "32",
    "16", "4",
    "256", "512",
    "8192", "1024",
    "262144", "1e-06",
    "10000", "1000000", "0.25",
};

RadModelMeta gemma4_draft_meta() {
    RadModelMeta m = gemma4_meta();
    m.n_kv   = (int)(sizeof(k_gem4d_keys) / sizeof(k_gem4d_keys[0]));
    m.kv_key = k_gem4d_keys;
    m.kv_val = k_gem4d_vals;
    return m;
}

RadBuildCtx build_ctx(int rank = 0, int world = 1) {
    RadBuildCtx c{};
    c.rank       = rank;
    c.world_size = world;
    c.max_tok    = 256;
    c.max_seqs   = 8;
    c.max_ctx    = 32768;
    c.scope      = "";
    return c;
}

/* The sampled rows (RadBatch::out_ids). The plugin skips the lm_head when n_out is zero (a chunk
 * finishing no sequence), so a fake batch must set them or it silently tests the no-logits
 * path. */
static const int32_t kDecodeOut[]  = { 0, 1, 2, 3 };
static const int32_t kPrefillOut[] = { 31, 63 };   /* last token of each 32-token sequence */

RadBatch decode_batch() {
    RadBatch b{};
    b.phase       = RAD_PHASE_DECODE;
    b.n_tok       = 4;
    b.n_seq       = 4;
    b.max_q_len   = 1;
    b.max_ctx_len = 100;
    b.n_out       = 4;
    b.out_ids     = kDecodeOut;
    return b;
}

RadBatch prefill_batch() {
    RadBatch b{};
    b.phase       = RAD_PHASE_PREFILL;
    b.n_tok       = 64;
    b.n_seq       = 2;
    b.max_q_len   = 32;
    b.max_ctx_len = 32;
    b.n_out       = 2;
    b.out_ids     = kPrefillOut;
    return b;
}

/* --- small queries over the recorded graph --- */
std::vector<std::string> op_names(const RadBuilder& b) {
    std::vector<std::string> v;
    for (const auto& o : b.ops) v.push_back(o.op);
    return v;
}

const RecOp* find_op(const RadBuilder& b, const std::string& name, int nth = 0) {
    for (const auto& o : b.ops)
        if (o.op == name && nth-- == 0) return &o;
    return nullptr;
}

int count_op(const RadBuilder& b, const std::string& name) {
    int n = 0;
    for (const auto& o : b.ops) if (o.op == name) ++n;
    return n;
}

const RecParam* param(const RecOp* o, const char* key) {
    if (!o) return nullptr;
    for (const auto& p : o->p) if (p.key == key) return &p;
    return nullptr;
}

long long pint(const RecOp* o, const char* key) {
    const RecParam* p = param(o, key);
    return p ? p->ival : -12345;
}

double pflt(const RecOp* o, const char* key) {
    const RecParam* p = param(o, key);
    return p ? p->dval : -12345.0;
}

std::string pstr(const RecOp* o, const char* key) {
    const RecParam* p = param(o, key);
    return p ? p->sval : std::string("<absent>");
}

const RecWeight* find_w(const RadBuilder& b, const std::string& name) {
    for (const auto& w : b.weights) if (w.name == name) return &w;
    return nullptr;
}

const RecBuf* find_b(const RadBuilder& b, const std::string& name) {
    for (const auto& x : b.bufs) if (x.name == name) return &x;
    return nullptr;
}

const RecMap* find_map(const RadBuilder& b, const std::string& declared) {
    for (const auto& m : b.maps) if (m.declared == declared) return &m;
    return nullptr;
}
/* The gated attention layer's ops, listed once for every test.
 *
 * The fused qk_norm_rope_gate is not declared: its schema requires a precomputed cos/sin table
 * rather than theta, and this plugin has none (see rad_block_attn_gated.h). Declaring it with
 * parameters its schema lacks would fail declare outright rather than fall back to the four ops it
 * stands for. */
std::vector<std::string> attn_layer_ops(bool tp) {
    std::vector<std::string> v = { "rmsnorm", "gemm_nt", "gemm_nt", "gemm_nt" };
    v.push_back("rmsnorm"); v.push_back("rmsnorm");
    v.push_back("rope");    v.push_back("rope");
    v.push_back("attn_paged");
    v.push_back("kv_store");
    v.push_back("sigmoid");
    v.push_back("mul");
    v.push_back("gemm_nt");
    if (tp) v.push_back("all_reduce");
    v.push_back("add");
    return v;
}

std::vector<std::string> gdn_layer_ops(bool tp) {
    std::vector<std::string> v = { "rmsnorm", "gemm_nt", "gemm_nt",
                                   "gdn_conv_prep", "gdn_kkt_solve", "gdn_chunk_scan",
                                   "gdn_gated_rmsnorm",
                                   "gdn_conv_update", "gdn_recurrent_update",
                                   "gemm_nt" };
    if (tp) v.push_back("all_reduce");
    v.push_back("add");
    return v;
}

std::vector<std::string> mlp_ops(bool tp) {
    std::vector<std::string> v = { "rmsnorm", "gemm_nt", "silu_mul", "gemm_nt" };
    if (tp) v.push_back("all_reduce");
    v.push_back("add");
    return v;
}

void append(std::vector<std::string>& v, const std::vector<std::string>& t) {
    v.insert(v.end(), t.begin(), t.end());
}

/* The whole declared program for the 8-layer fixture: layers 3 and 7 are full attention. */
std::vector<std::string> program_ops(bool tp) {
    std::vector<std::string> v = { "embed_lookup" };
    /* The embedding table is vocab-sharded: a rank gathers only ids in its slice and zero-fills
     * the rest, and this collective sums the ranks into x. Hence 2 all_reduces per layer plus
     * one. */
    if (tp) v.push_back("all_reduce");
    for (int l = 0; l < 8; ++l) {
        if ((l + 1) % 4 == 0) append(v, attn_layer_ops(tp));
        else                  append(v, gdn_layer_ops(tp));
        append(v, mlp_ops(tp));
    }
    v.push_back("rmsnorm");
    /* gather_rows before the lm_head: the sampler needs one row per sequence, and the widest GEMM
     * over every prefill token would be n_tok/n_seq times the work and an 8 GiB logits buffer
     * (docs/OPS.md). */
    v.push_back("gather_rows");
    v.push_back("logits_gemm");
    return v;
}

std::string join(const std::vector<std::string>& v) {
    std::string s;
    for (size_t i = 0; i < v.size(); ++i) { if (i) s += " "; s += v[i]; }
    return s;
}

std::vector<std::string> issued_ops(const RadBuilder& b, const RadCtx& c) {
    std::vector<std::string> got;
    for (const auto& i : c.issues) {
        const RecOp* o = b.op_of(i.op);
        got.push_back(o ? o->op : "<null>");
    }
    return got;
}

}  /* namespace */

/* ==================================================================== the step lint
 *
 * The buffer plan takes liveness from declaration order: a transient lives over the [first, last]
 * op indices of its declared reads and writes, and two transients of one arena whose ranges are
 * disjoint may share bytes (core/build/rad_bufplan.cpp). A step that issues ops in another order,
 * or across passes of one engine step (trunk, drafter history, draft rounds), can write one buffer
 * while another that shares its bytes still holds rows a later issue reads. The runtime refuses
 * only an op outside a buffer's range, and only on the branch a real model runs, so the class is
 * checked here for every recorded step.
 *
 * lint_step walks the issues of one engine step's passes in order and reports:
 *   alias       a transient read whose last writer is followed, before the read, by a write of a
 *               transient of the same arena whose range is disjoint from its own;
 *   undeclared  an op issued outside the declared range of a buffer operand
 *               (core/runtime/issue.cpp refuses this);
 *   span        an operand that reaches past its buffer or weight, rows and columns applied as
 *               issue.cpp applies them;
 *   null op     an issue of RAD_NULL_HANDLE (a fused op that did not resolve, with no chain);
 *   unwritten   a transient read before any issue, or device copy into it, of the step wrote it;
 *   kv          a cache operand naming a group or layer no rad_bind_layer_kv bound, or a group
 *               this rank declared zero;
 *   strided     a GEMM whose input is a column slice narrower than its buffer (libr4d's GEMMs
 *               read whole rows and refuse it). */
namespace {

/* One buffer's planned lifetime. `planned` is false for a buffer live over the whole program
 * (PERSIST, DERIVED, rad_buf_concurrent, or no declared use), which shares bytes with nothing. */
struct Live {
    int32_t lo = 0, hi = 0;
    bool    planned = false;
    bool    transient = false;
    int     domain = RAD_DOMAIN_DEVICE;
};

/* Liveness by the rule of Builder::finish_buffer_liveness (core/build/rad_builder.cpp). Op handle
 * h is op index h: the real builder keeps a sentinel at index 0, and the fake hands out ops in
 * declaration order from 1. */
std::vector<Live> liveness(const RadBuilder& b) {
    const size_t  nb = b.bufs.size();
    const int32_t last_op = (int32_t)b.ops.size();
    std::vector<int32_t> first(nb + 1, -1), last(nb + 1, -1);
    for (size_t i = 0; i < b.ops.size(); ++i)
        for (const std::vector<rad_buf>* v : { &b.ops[i].reads, &b.ops[i].writes })
            for (rad_buf h : *v) {
                if (h == 0 || h > nb) continue;
                if (first[h] < 0) first[h] = (int32_t)(i + 1);
                last[h] = (int32_t)(i + 1);
            }
    std::set<rad_buf> conc(b.concurrent.begin(), b.concurrent.end());
    std::vector<Live> L(nb + 1);
    for (size_t h = 1; h <= nb; ++h) {
        const RadBufDecl& d = b.bufs[h - 1].d;
        Live& l = L[h];
        l.transient = d.kind == RAD_BUF_TRANSIENT;
        l.domain    = d.domain;
        if (!l.transient || conc.count((rad_buf)h) || first[h] < 0) {
            l.lo = 0;
            l.hi = last_op;
            continue;
        }
        l.planned = true;
        l.lo = first[h];
        l.hi = last[h];
        if ((rad_buf)h == b.logits || (rad_buf)h == b.pooled) l.hi = last_op;
    }
    return L;
}

/* Whether operand `o` stays inside a buffer or weight of `shape` (packed strides, rows and cols
 * applied as issue.cpp applies them), as issue.cpp's operand_fits decides. */
bool operand_fits(const int64_t* shape, uint32_t rank, const RadOperand& o) {
    if (o.offset < 0) return false;
    int64_t sh[RAD_MAX_RANK] = {}, st[RAD_MAX_RANK] = {};
    int64_t numel = 1;
    for (uint32_t d = 0; d < rank; ++d) { sh[d] = shape[d]; numel *= shape[d]; }
    int64_t s = 1;
    for (int d = (int)rank - 1; d >= 0; --d) { st[d] = s; s *= sh[d]; }
    if (o.rows > 0 && rank >= 1) sh[0] = o.rows;
    if (o.cols > 0 && rank >= 1) sh[rank - 1] = o.cols;
    int64_t span = o.offset;
    for (uint32_t d = 0; d < rank; ++d) {
        if (sh[d] <= 0) return true;
        span += (sh[d] - 1) * st[d];
    }
    return span < numel;
}

bool in(const std::vector<rad_buf>& v, rad_buf h) {
    return std::find(v.begin(), v.end(), h) != v.end();
}

std::vector<std::string> lint_step(const RadBuilder& b, const std::vector<const RadCtx*>& passes) {
    std::vector<std::string> bad;
    const std::vector<Live> L = liveness(b);
    const size_t nb = b.bufs.size();
    auto name = [&](rad_buf h) { return h && h <= nb ? b.bufs[h - 1].name : std::string("?"); };
    auto shares = [&](rad_buf x, rad_buf y) {
        const Live& p = L[x];
        const Live& q = L[y];
        return p.planned && q.planned && p.domain == q.domain && (p.hi < q.lo || q.hi < p.lo);
    };
    std::set<std::pair<int, rad_kvgroup>> bound(b.binds.begin(), b.binds.end());
    std::vector<int64_t> last_write(nb + 1, -1);
    int64_t pos = 0;
    char where[96];
    for (size_t pi = 0; pi < passes.size(); ++pi) {
        const RadCtx& c = *passes[pi];
        size_t next_copy = 0;
        for (size_t ii = 0; ii < c.issues.size(); ++ii, ++pos) {
            for (; next_copy < c.copies.size() && c.copies[next_copy].first <= ii; ++next_copy)
                last_write[c.copies[next_copy].second] = pos;
            const RecIssue& is = c.issues[ii];
            const RecOp* o = b.op_of(is.op);
            std::snprintf(where, sizeof where, "pass %zu issue %zu (%s, op %u)", pi, ii,
                          o ? o->op.c_str() : "<null>", (unsigned)is.op);
            if (!o) { bad.push_back(std::string(where) + ": null op issued"); continue; }
            std::vector<rad_buf> rd, wr;
            for (size_t k = 0; k < is.opd.size(); ++k) {
                const RadOperand& od = is.opd[k];
                const std::string at = std::string(where) + " operand " + std::to_string(k);
                if (od.kind == RAD_OPK_BUF) {
                    const rad_buf h = od.handle;
                    if (h == 0 || h > nb) {
                        bad.push_back(at + ": buffer handle out of range");
                        continue;
                    }
                    const RadBufDecl& d = b.bufs[h - 1].d;
                    if (!operand_fits(d.shape, d.rank, od))
                        bad.push_back(at + ": span past '" + name(h) + "' (offset " +
                                      std::to_string(od.offset) + ", rows " +
                                      std::to_string(od.rows) + ", cols " +
                                      std::to_string(od.cols) + ")");
                    const Live& l = L[h];
                    if (l.planned && ((int32_t)is.op < l.lo || (int32_t)is.op > l.hi))
                        bad.push_back(at + ": undeclared use of '" + name(h) + "' (planned ops " +
                                      std::to_string(l.lo) + ".." + std::to_string(l.hi) + ")");
                    const bool r = in(o->reads, h), w = in(o->writes, h);
                    if (r || !w) rd.push_back(h);
                    if (w || !r) wr.push_back(h);
                    /* A GEMM's input is read as contiguous rows: libr4d refuses a column slice
                     * narrower than its buffer. */
                    const int64_t width = d.rank ? d.shape[d.rank - 1] : 0;
                    if (k == 0 && o->op.rfind("gemm_nt", 0) == 0 && od.cols > 0 &&
                        (od.cols != width || od.offset % width))
                        bad.push_back(at + ": " + o->op + " reads " + std::to_string(od.cols) +
                                      " of " + std::to_string(width) + " columns of '" + name(h) +
                                      "' as its input, which is not contiguous");
                } else if (od.kind == RAD_OPK_WEIGHT) {
                    const uint32_t h = od.handle;
                    if (h == 0 || h > b.weights.size()) {
                        bad.push_back(at + ": weight handle out of range");
                        continue;
                    }
                    const RadWeightDecl& d = b.weights[h - 1].d;
                    if ((od.offset || od.rows > 0 || od.cols > 0) &&
                        !operand_fits(d.shape, d.rank, od))
                        bad.push_back(at + ": span past weight '" + b.weights[h - 1].name + "'");
                } else if (od.kind == RAD_OPK_WTAB) {
                    if (od.handle == 0 || od.rows <= 0 || od.offset || od.cols ||
                        (size_t)(od.handle + od.rows - 1) > b.weights.size())
                        bad.push_back(at + ": weight table out of range");
                } else if (od.kind == RAD_OPK_KV) {
                    const rad_kvgroup g = od.handle;
                    if (g == 0 || g > b.kv_groups.size())
                        bad.push_back(at + ": KV group handle out of range");
                    else if (b.kv_groups[g - 1].zero)
                        bad.push_back(at + ": KV group '" + b.kv_groups[g - 1].name +
                                      "' is declared zero on this rank");
                    else if (!bound.count({ (int)od.offset, g }))
                        bad.push_back(at + ": layer " + std::to_string(od.offset) +
                                      " is not bound to KV group '" + b.kv_groups[g - 1].name +
                                      "'");
                } else if (od.kind == RAD_OPK_RAW) {
                    if (od.offset && od.dtype == RAD_DT_INVALID)
                        bad.push_back(at + ": raw operand with an offset and no dtype");
                    /* A pointer into a buffer (rad_buf_ptr) is that buffer to the plan, so it is
                     * classified by the op's declared sets like a buffer operand. */
                    if (const rad_buf h = fake_buf_of(od.raw, nb)) {
                        const Live& l = L[h];
                        if (l.planned && ((int32_t)is.op < l.lo || (int32_t)is.op > l.hi))
                            bad.push_back(at + ": undeclared use of '" + name(h) +
                                          "' through a raw pointer");
                        const bool r = in(o->reads, h), w = in(o->writes, h);
                        if (r || !w) rd.push_back(h);
                        if (w || !r) wr.push_back(h);
                    }
                }
            }
            for (rad_buf h : rd) {
                if (!L[h].transient) continue;
                const int64_t w = last_write[h];
                if (w < 0) {
                    bad.push_back(std::string(where) + ": reads '" + name(h) +
                                  "' before any issue of this step wrote it");
                    continue;
                }
                for (size_t x = 1; x <= nb; ++x)
                    if (x != h && last_write[x] > w && shares(h, (rad_buf)x))
                        bad.push_back(std::string(where) + ": reads '" + name(h) + "' (ops " +
                                      std::to_string(L[h].lo) + ".." + std::to_string(L[h].hi) +
                                      ") after a write of '" + name((rad_buf)x) + "' (ops " +
                                      std::to_string(L[x].lo) + ".." + std::to_string(L[x].hi) +
                                      "), which may share its bytes");
            }
            for (rad_buf h : wr) last_write[h] = pos;
        }
    }
    return bad;
}

/* ---------------------------------------------------------------- whole step batches
 *
 * A batch as the core builds it: every per-token and per-sequence array filled and a group batch
 * for every KV group the builder declared, so a plugin's step can take any path. Decode sequences
 * come first (RadBatch::n_seq_decode); each `dq` entry is a decode sequence's query length (1, or
 * 1 + n_spec for a verify), each `pq` entry a prefill chunk. Every decode row is sampled (a verify
 * needs logits at each row) and a chunk's last row; a chunk continues `prefill_ctx` tokens of
 * history. */
struct Kit {
    std::vector<int32_t> tok, pos, cu, ctxl, qlen, out, nacc, dout;
    std::vector<int32_t> slot, table, used, sidx;
    std::vector<RadKVGroupBatch> kv;
    RadBatch b{};
};

std::unique_ptr<Kit> kit(const RadBuilder& bld, const std::vector<int>& dq,
                         const std::vector<int>& pq, int32_t prefill_ctx = 0) {
    auto k = std::make_unique<Kit>();
    const int64_t nd = (int64_t)dq.size(), ns = nd + (int64_t)pq.size();
    std::vector<int> ql(dq);
    ql.insert(ql.end(), pq.begin(), pq.end());
    int64_t T = 0, max_q = 0, max_ctx = 0, td = 0, mqd = 0, mqp = 0;
    k->cu.push_back(0);
    for (int64_t s = 0; s < ns; ++s) {
        const bool dec = s < nd;
        const int32_t ctx = dec ? (int32_t)(40 + 7 * s) : prefill_ctx;
        for (int i = 0; i < ql[(size_t)s]; ++i) {
            k->tok.push_back((int32_t)(3 + (T + i) % 97));
            k->pos.push_back(ctx + i);
            if (dec || i + 1 == ql[(size_t)s]) k->out.push_back((int32_t)(T + i));
        }
        T += ql[(size_t)s];
        k->cu.push_back((int32_t)T);
        k->ctxl.push_back(ctx);
        k->qlen.push_back(ql[(size_t)s]);
        k->nacc.push_back(1);
        k->dout.push_back((int32_t)T - 1);
        max_q = std::max<int64_t>(max_q, ql[(size_t)s]);
        max_ctx = std::max<int64_t>(max_ctx, ctx + ql[(size_t)s]);
        if (dec) { td += ql[(size_t)s]; mqd = std::max<int64_t>(mqd, ql[(size_t)s]); }
        else     mqp = std::max<int64_t>(mqp, ql[(size_t)s]);
    }
    k->tok.resize((size_t)T + 64, 5);   /* room for a following prompt (RadBatch::n_ahead) */
    const int64_t pitch = 16;
    for (int64_t i = 0; i < T; ++i) k->slot.push_back((int32_t)i);
    for (int64_t i = 0; i < ns * pitch; ++i) k->table.push_back((int32_t)i);
    for (int64_t s = 0; s < ns; ++s) {
        k->used.push_back(k->ctxl[(size_t)s] + k->qlen[(size_t)s]);
        k->sidx.push_back((int32_t)s);
    }
    for (size_t g = 0; g < bld.kv_groups.size(); ++g) {
        RadKVGroupBatch gb{};
        gb.group = (rad_kvgroup)(g + 1);
        gb.slot_mapping = k->slot.data();
        gb.block_table = k->table.data();
        gb.block_table_pitch = pitch;
        gb.seqused = k->used.data();
        gb.state_index = k->sidx.data();
        gb.state_index_pitch = 1;
        gb.block_size = bld.kv_block;
        gb.max_blocks = pitch;
        k->kv.push_back(gb);
    }
    RadBatch& b = k->b;
    b.phase = pq.empty() ? RAD_PHASE_DECODE : dq.empty() ? RAD_PHASE_PREFILL : RAD_PHASE_MIXED;
    b.n_tok = T;
    b.n_seq = ns;
    b.token_ids = k->tok.data();
    b.positions = k->pos.data();
    b.cu_seqlens = k->cu.data();
    b.n_out = (int64_t)k->out.size();
    b.out_ids = k->out.data();
    b.q_lens = k->qlen.data();
    b.ctx_lens = k->ctxl.data();
    b.max_q_len = (int32_t)max_q;
    b.max_ctx_len = (int32_t)max_ctx;
    b.n_kv_groups = (int)k->kv.size();
    b.kv = k->kv.data();
    b.n_spec = mqd > 1 ? (int)(mqd - 1) : 0;
    b.num_accepted = k->nacc.data();
    b.n_seq_decode = nd;
    b.n_tok_decode = td;
    b.max_q_len_decode = (int32_t)mqd;
    b.max_q_len_prefill = (int32_t)mqp;
    return k;
}

/* The draft passes the core stages for a step of `n_seq` sequences.
 *
 * A serial head's history pass (draft_pass -1) covers every row the trunk step had and carries
 * round 1 on each sequence's last row; round r (2..depth) is one row a sequence. A block
 * drafter's context pass (-1) covers the trunk's rows; its query pass (1) is `block` rows a
 * sequence, the anchor first, and samples the `steps` rows after it (RadDrafterDecl::block). */
std::unique_ptr<Kit> history_pass(const RadBuilder& bld, const Kit& trunk) {
    std::vector<int> rows;
    for (int64_t s = 0; s < trunk.b.n_seq; ++s) rows.push_back(trunk.qlen[(size_t)s]);
    auto k = kit(bld, rows, {});
    k->b.draft_pass = -1;
    k->out.clear();
    for (int64_t i = 0; i < k->b.n_tok; ++i) k->out.push_back((int32_t)i);
    k->b.out_ids = k->out.data();
    k->b.n_out = k->b.n_tok;
    k->b.n_draft_out = k->b.n_seq;
    k->b.draft_out_ids = k->dout.data();
    return k;
}
std::unique_ptr<Kit> round_pass(const RadBuilder& bld, int64_t n_seq, int r) {
    auto k = kit(bld, std::vector<int>((size_t)n_seq, 1), {});
    k->b.draft_pass = r;
    return k;
}
std::unique_ptr<Kit> context_pass(const RadBuilder& bld, const Kit& trunk) {
    auto k = history_pass(bld, trunk);
    k->b.n_draft_out = 0;
    k->b.draft_out_ids = nullptr;
    return k;
}
std::unique_ptr<Kit> query_pass(const RadBuilder& bld, int64_t n_seq, int64_t block,
                                int64_t steps) {
    auto k = kit(bld, std::vector<int>((size_t)n_seq, (int)block), {});
    k->b.draft_pass = 1;
    k->out.clear();
    for (int64_t s = 0; s < n_seq; ++s)
        for (int64_t i = 0; i < steps; ++i)
            k->out.push_back((int32_t)(s * block + block - steps + i));
    k->b.out_ids = k->out.data();
    k->b.n_out = (int64_t)k->out.size();
    return k;
}

using StepFn = void (*)(RadCtx*, const RadBatch*);

/* Runs one engine step's passes in order, each on a context of its own. */
std::vector<std::unique_ptr<RadCtx>> run_passes(RadBuilder& b, StepFn step,
                                                const std::vector<const Kit*>& passes,
                                                int rank = 0, int world = 1) {
    std::vector<std::unique_ptr<RadCtx>> out;
    for (const Kit* k : passes) {
        out.push_back(std::make_unique<RadCtx>());
        RadCtx& c = *out.back();
        c.b = &b;
        c.batch = &k->b;
        c.rank = rank;
        c.world = world;
        step(&c, &k->b);
    }
    return out;
}

std::vector<const RadCtx*> view(const std::vector<std::unique_ptr<RadCtx>>& v) {
    std::vector<const RadCtx*> p;
    for (const auto& c : v) p.push_back(c.get());
    return p;
}

/* The lint as a check: every finding is printed, and the step passes only with none. */
size_t report_lint(const std::vector<std::string>& f) {
    for (size_t i = 0; i < f.size() && i < 12; ++i) fprintf(stderr, "    lint: %s\n", f[i].c_str());
    if (f.size() > 12) fprintf(stderr, "    lint: ... %zu more\n", f.size() - 12);
    return f.size();
}

/* Every pass of one engine step ran without a step failure. */
bool steps_ok(const std::vector<std::unique_ptr<RadCtx>>& v) {
    bool ok = true;
    for (size_t i = 0; i < v.size(); ++i)
        if (!v[i]->step_fail.empty()) {
            fprintf(stderr, "    pass %zu failed: %s\n", i, v[i]->step_fail.c_str());
            ok = false;
        }
    return ok;
}

}  /* namespace */

#define CHECK_LINT(b, ...) CHECK_EQ(report_lint(lint_step((b), { __VA_ARGS__ })), (size_t)0)
/* The passes of one engine step (run_passes): none failed and the lint is clean over them. */
#define CHECK_PASSES(b, v)                                                                   \
    do {                                                                                     \
        CHECK(steps_ok(v));                                                                  \
        CHECK_EQ(report_lint(lint_step((b), view(v))), (size_t)0);                           \
    } while (0)

/* ==================================================================== tests */

TEST(declare_is_a_qwen35_trunk) {
    RadBuilder b;
    RadModelMeta meta = qwen35_meta();
    RadBuildCtx  ctx  = build_ctx();
    CHECK_OK(qwen35_bf16::declare(&b, &meta, &ctx));

    CHECK_EQ(join(op_names(b)), join(program_ops(/*tp=*/false)));

    /* Six gated delta net layers to two full attention ones, the interval's 1:3 ratio. */
    CHECK_EQ(count_op(b, "gdn_chunk_scan"), 6);
    CHECK_EQ(count_op(b, "attn_paged"), 2);
    /* The fused gate op is not declared (see attn_layer_ops), so each attention layer runs two
     * per-head rmsnorms and two ropes instead. */
    CHECK_EQ(count_op(b, "qk_norm_rope_gate"), 0);
    CHECK_EQ(count_op(b, "rope"), 4);
    /* One MLP per layer, the same block for both layer kinds. */
    CHECK_EQ(count_op(b, "silu_mul"), 8);
}

TEST(qwen35_bf16_yarn_reaches_every_text_rotary_op) {
    RadModelMeta meta = qwen35_meta();
    std::vector<const char*> keys(meta.kv_key, meta.kv_key + meta.n_kv);
    std::vector<const char*> vals(meta.kv_val, meta.kv_val + meta.n_kv);
    keys.insert(keys.end(), {"rope_parameters.rope_type", "rope_parameters.factor",
                            "rope_parameters.original_max_position_embeddings"});
    vals.insert(vals.end(), {"yarn", "2", "262144"});
    meta.n_kv = (int)keys.size(); meta.kv_key = keys.data(); meta.kv_val = vals.data();
    RadBuildCtx ctx = build_ctx();
    RadBuilder b;
    CHECK_OK(qwen35_bf16::declare(&b, &meta, &ctx));
    int rotary = 0;
    for (const auto& op : b.ops) if (op.op == "rope") {
        ++rotary;
        int found = 0;
        for (const auto& p : op.p) {
            if (p.key == "yarn_factor") { CHECK_EQ(p.dval, 2.0); ++found; }
            if (p.key == "yarn_magnitude") { CHECK_NEAR(p.dval, 1.069314718, 1e-7); ++found; }
            if (p.key == "scale") CHECK_EQ(p.dval, 1.0);
        }
        CHECK_EQ(found, 2);
    }
    CHECK_EQ(rotary, 4);
    vals[vals.size()-2] = "nan";
    RadBuilder invalid;
    CHECK_EQ(qwen35_bf16::declare(&invalid, &meta, &ctx), RAD_E_INVAL);
}

/* The LLaMA family reads rotary scaling through the same reader. Llama 3.1's `rope_type: llama3`
 * with factor 8, read as a bare factor, would be a linear position divide by 8, so it is refused
 * by name at declare. Linear stays linear and YaRN reaches the rope op. */
TEST(llama_reads_its_rotary_scaling_by_type) {
    auto declare = [](const char* type, const char* factor, RadBuilder* b) {
        const char* keys[] = {"rope_scaling.rope_type", "rope_scaling.factor",
                              "rope_scaling.original_max_position_embeddings"};
        const char* vals[] = {type, factor, "8192"};
        RadModelMeta m{};
        m.arch_id = "llama"; m.name = "test-llama"; m.quant = "fp8_e4m3";
        m.n_layers = 2; m.n_embd = 1024; m.n_head = 8; m.n_head_kv = 2; m.head_dim = 128;
        m.n_ff = 2048; m.n_vocab = 32000; m.n_ctx_train = 8192;
        m.rms_eps = 1e-5f; m.rope_theta = 5e5f; m.rope_scale = 1.0f;
        m.n_kv = type ? 3 : 0; m.kv_key = keys; m.kv_val = vals;
        RadBuildCtx c = build_ctx();
        return llama_fp8::declare(b, &m, &c);
    };
    for (const char* refused : {"llama3", "longrope", "dynamic"}) {
        RadBuilder b;
        CHECK_EQ(declare(refused, "8", &b), RAD_E_UNSUPPORTED);
    }
    {
        RadBuilder b;
        CHECK_OK(declare("linear", "2", &b));
        CHECK(count_op(b, "rope") > 0);
        for (const auto& op : b.ops) if (op.op == "rope") {
            CHECK_EQ(pflt(&op, "scale"), 2.0);
            CHECK(!param(&op, "yarn_factor"));
        }
    }
    {
        RadBuilder b;
        CHECK_OK(declare("yarn", "4", &b));
        CHECK(count_op(b, "rope") > 0);
        for (const auto& op : b.ops) if (op.op == "rope") {
            CHECK_EQ(pflt(&op, "scale"), 1.0);
            CHECK_EQ(pflt(&op, "yarn_factor"), 4.0);
        }
    }
    {
        RadBuilder b;
        CHECK_OK(declare(nullptr, "", &b));
        for (const auto& op : b.ops) if (op.op == "rope") {
            CHECK_EQ(pflt(&op, "scale"), 1.0);
            CHECK(!param(&op, "yarn_factor"));
        }
    }
}

TEST(the_alternation_puts_full_attention_at_3_and_7) {
    RadBuilder b;
    RadModelMeta meta = qwen35_meta();
    RadBuildCtx  ctx  = build_ctx();
    CHECK_OK(qwen35_bf16::declare(&b, &meta, &ctx));

    /* The weight manifest shows a layer's kind: full attention has a q|gate projection, linear
     * attention an in-projection, never both. */
    for (int l = 0; l < 8; ++l) {
        const bool full = ((l + 1) % 4 == 0);
        const std::string qg  = "blk." + std::to_string(l) + ".attn_qg.weight";
        const std::string in  = "blk." + std::to_string(l) + ".ssm_in.weight";
        CHECK_EQ(find_w(b, qg) != nullptr, full);
        CHECK_EQ(find_w(b, in) != nullptr, !full);
        /* Both kinds share the pre-norm name and feed-forward. */
        CHECK(find_w(b, "blk." + std::to_string(l) + ".attn_norm.weight") != nullptr);
        CHECK(find_w(b, "blk." + std::to_string(l) + ".ffn_gate_up.weight") != nullptr);
    }
}

TEST(attention_shapes_at_the_27b_dimensions) {
    RadBuilder b;
    RadModelMeta meta = qwen35_meta();
    RadBuildCtx  ctx  = build_ctx();
    CHECK_OK(qwen35_bf16::declare(&b, &meta, &ctx));

    /* q_dim = 24 * 256 = 6144; the projection is twice that, query and gate interleaved. */
    const RecWeight* qg = find_w(b, "blk.3.attn_qg.weight");
    CHECK(qg != nullptr);
    CHECK_EQ(qg->d.rank, 2u);
    CHECK_EQ(qg->d.shape[0], 12288);
    CHECK_EQ(qg->d.shape[1], 5120);
    CHECK_EQ(qg->d.dtype, (uint32_t)RAD_BF16);
    CHECK_EQ(qg->d.shard, RAD_SHARD_ROW);
    CHECK_EQ(qg->d.access, RAD_ACCESS_PER_TOKEN);

    /* k and v are separate projections, 4 kv heads of 256. */
    const RecWeight* kw = find_w(b, "blk.3.attn_k.weight");
    const RecWeight* vw = find_w(b, "blk.3.attn_v.weight");
    CHECK(kw != nullptr && vw != nullptr);
    CHECK_EQ(kw->d.shape[0], 1024);
    CHECK_EQ(vw->d.shape[0], 1024);
    CHECK_EQ(kw->d.shard, RAD_SHARD_ROW);

    const RecWeight* o = find_w(b, "blk.3.attn_output.weight");
    CHECK(o != nullptr);
    CHECK_EQ(o->d.shape[0], 5120);
    CHECK_EQ(o->d.shape[1], 6144);
    CHECK_EQ(o->d.shard, RAD_SHARD_COL);

    /* Per-head gains are head_dim wide and replicated, one vector shared by all heads. */
    const RecWeight* qn = find_w(b, "blk.3.attn_q_norm.weight");
    CHECK(qn != nullptr);
    CHECK_EQ(qn->d.shape[0], 256);
    CHECK_EQ(qn->d.shard, RAD_SHARD_NONE);
    CHECK_EQ(qn->d.dtype, (uint32_t)RAD_F32);

    /* The interleaved projection buffer is rank 3, [tokens, heads, 2*head_dim], so query and gate
     * are addressable as column slices. */
    const RecBuf* qgb = find_b(b, "attn_qg");
    CHECK(qgb != nullptr);
    CHECK_EQ(qgb->d.rank, 3u);
    CHECK_EQ(qgb->d.shape[0], 256);
    CHECK_EQ(qgb->d.shape[1], 24);
    CHECK_EQ(qgb->d.shape[2], 512);

    /* qk_norm_rope_gate must stay undeclared until something builds its cos/sin table (see
     * attn_layer_ops); the four ops it stands for carry the same numbers. */
    CHECK(find_op(b, "qk_norm_rope_gate") == nullptr);

    const RecOp* qn2 = find_op(b, "rope");
    CHECK(qn2 != nullptr);
    if (qn2) {
        CHECK_EQ(pint(qn2, "head_dim"), 256);
        CHECK_EQ(pint(qn2, "rotary_dim"), 64);   /* partial_rotary_factor 0.25 of 256 */
        CHECK_EQ(pstr(qn2, "mode"), std::string("mrope"));
        CHECK_NEAR(pflt(qn2, "theta"), 1e7, 1.0);
    }
    const RecOp* at = find_op(b, "attn_paged");
    CHECK_EQ(pint(at, "head_dim"), 256);
    CHECK_EQ(pint(at, "gqa"), 6);
    CHECK_EQ(pint(at, "causal"), 1);
    CHECK_EQ(pint(at, "window"), 0);
    /* block_size is declared deliberately. It is RAD_DERIVED (kernel-supplied) and every libr4d
     * attention row pins C_EQ("block_size", 16), so 16 changes nothing on the accelerated path.
     * But a geometry no accelerated row serves (e.g. head_dim 256 at gqa 4) resolves to libref,
     * which accepts any block size and names none, leaving the KV group unsizable. The block size
     * belongs to the cache the model asks for, so the model declares it. */
    CHECK_EQ(pint(at, "block_size"), 16);

    /* ...and it reaches kv_store, which needs it as a parameter. */
    const RecOp* ks = find_op(b, "kv_store");
    CHECK_EQ(pint(ks, "block_size"), 16);
    CHECK_EQ(pint(ks, "n_head_kv"), 4);

    /* The gate: a per-head sigmoid over head_dim, then an elementwise multiply over the attention
     * output, `out = attn * sigmoid(gate)`. */
    const RecOp* sg = find_op(b, "sigmoid");
    CHECK_EQ(pint(sg, "n"), 256);
    CHECK_EQ(param(sg, "M")->ihi, 256LL * 24);
    const RecOp* mu = find_op(b, "mul");
    CHECK_EQ(pint(mu, "n"), 6144);
}

TEST(gdn_shapes_and_both_paths) {
    RadBuilder b;
    RadModelMeta meta = qwen35_meta();
    RadBuildCtx  ctx  = build_ctx();
    CHECK_OK(qwen35_bf16::declare(&b, &meta, &ctx));

    /* conv_dim = 2*16*128 + 48*128 = 10240, followed by one `a` and one `b` per value head:
     * 10240 + 96 = 10336. The fusion lets gdn_conv_prep read the decay and beta inputs from `x`. */
    const RecWeight* in = find_w(b, "blk.0.ssm_in.weight");
    CHECK(in != nullptr);
    CHECK_EQ(in->d.shape[0], 10336);
    CHECK_EQ(in->d.shape[1], 5120);
    CHECK_EQ(in->d.shard, RAD_SHARD_ROW);

    const RecWeight* z = find_w(b, "blk.0.ssm_z.weight");
    CHECK(z != nullptr);
    CHECK_EQ(z->d.shape[0], 6144);            /* 48 value heads of 128 */

    const RecWeight* cv = find_w(b, "blk.0.ssm_conv1d.weight");
    CHECK(cv != nullptr);
    CHECK_EQ(cv->d.rank, 2u);
    CHECK_EQ(cv->d.shape[0], 10240);          /* depthwise: one filter per convolved channel */
    CHECK_EQ(cv->d.shape[1], 4);
    CHECK_EQ(cv->d.shard, RAD_SHARD_ROW);
    /* Must be BF16, and nothing else checks it: libr4d's conv reads this pointer as bf16, so an
     * f32 plane would be misread on the device path only (libref reads the operand's own dtype). */
    CHECK_EQ(cv->d.dtype, (uint32_t)RAD_BF16);
    /* Qwen3.5's conv1d has no bias, so none is declared; the operand is passed absent. */
    CHECK(find_w(b, "blk.0.ssm_conv1d.bias") == nullptr);

    /* The decay weight is named for its convention: it is a log and the kernels take exp(). */
    const RecWeight* al = find_w(b, "blk.0.ssm_a_log");
    CHECK(al != nullptr);
    CHECK_EQ(al->d.shape[0], 48);
    CHECK_EQ(al->d.dtype, (uint32_t)RAD_F32);
    CHECK(find_w(b, "blk.0.ssm_a") == nullptr);

    const RecWeight* on = find_w(b, "blk.0.ssm_norm.weight");
    CHECK(on != nullptr);
    CHECK_EQ(on->d.shape[0], 128);            /* over head_v, shared by every head */
    CHECK_EQ(on->d.shard, RAD_SHARD_NONE);

    const RecWeight* ow = find_w(b, "blk.0.ssm_out.weight");
    CHECK(ow != nullptr);
    CHECK_EQ(ow->d.shape[0], 5120);
    CHECK_EQ(ow->d.shape[1], 6144);
    CHECK_EQ(ow->d.shard, RAD_SHARD_COL);

    /* Both the chunked and recurrent paths are declared, so both appear in the graph dump and
     * resolve at declare. */
    CHECK_EQ(count_op(b, "gdn_conv_prep"), 6);
    CHECK_EQ(count_op(b, "gdn_kkt_solve"), 6);
    CHECK_EQ(count_op(b, "gdn_chunk_scan"), 6);
    CHECK_EQ(count_op(b, "gdn_gated_rmsnorm"), 6);
    CHECK_EQ(count_op(b, "gdn_conv_update"), 6);
    CHECK_EQ(count_op(b, "gdn_recurrent_update"), 6);

    const RecOp* cp = find_op(b, "gdn_conv_prep");
    CHECK_EQ(pint(cp, "head_k"), 128);
    CHECK_EQ(pint(cp, "head_v"), 128);
    CHECK_EQ(pint(cp, "conv_width"), 4);
    CHECK_EQ(pint(cp, "chunk"), 64);
    CHECK_EQ(param(cp, "M")->kind, RAD_P_RANGE);
    /* Weight operands: the conv weight and the two per-head vectors; no bias. */
    CHECK_EQ((int)cp->w.size(), 3);

    /* chunk is RAD_DERIVED (kernel-chosen) on the two ops that tile on it. */
    CHECK(param(find_op(b, "gdn_kkt_solve"), "chunk") == nullptr);
    CHECK(param(find_op(b, "gdn_chunk_scan"), "chunk") == nullptr);

    /* The recurrent path is ranged on q_len, since a speculative verify carries n_spec+1 rows per
     * sequence, and states n_head_v explicitly: with 48 value heads against 16 key heads, inferring
     * it from a flattened view gives a plausible wrong answer. */
    const RecOp* ru = find_op(b, "gdn_recurrent_update");
    CHECK_EQ(param(ru, "q_len")->kind, RAD_P_RANGE);
    CHECK_EQ(pint(ru, "n_head_v"), 48);
    CHECK_EQ(pstr(ru, "act"), std::string("silu"));
    CHECK_EQ(pint(find_op(b, "gdn_conv_update"), "conv_width"), 4);

    /* q and k have 16 heads and v 48: the 3:1 sharing is in the buffer shapes. */
    const RecBuf* gq = find_b(b, "gdn_q");
    const RecBuf* gv = find_b(b, "gdn_v");
    CHECK(gq != nullptr && gv != nullptr);
    CHECK_EQ(gq->d.shape[1], 16);
    CHECK_EQ(gq->d.shape[2], 128);
    CHECK_EQ(gv->d.shape[1], 48);
    CHECK_EQ(gv->d.shape[2], 128);
    /* Decay and beta are f32: the gate is a cumulative sum spanning hundreds of nats, and every
     * decay factor is an exponent of it.
     *
     * The chunk inverse is bf16 by contract: libr4d's gdn_kkt_solve writes it and gdn_chunk_scan
     * reads it as bf16, so f32 would double its size and misread every element. libref reads each
     * operand's own dtype, so only the device path would show it. */
    CHECK_EQ(find_b(b, "gdn_g")->d.dtype, (uint32_t)RAD_F32);
    CHECK_EQ(find_b(b, "gdn_beta")->d.dtype, (uint32_t)RAD_F32);
    CHECK_EQ(find_b(b, "gdn_kkt")->d.dtype, (uint32_t)RAD_BF16);
    CHECK_EQ(find_b(b, "gdn_kkt")->d.shape[2], 64);
}

TEST(three_kv_groups_with_every_layer_bound) {
    RadBuilder b;
    RadModelMeta meta = qwen35_meta();
    RadBuildCtx  ctx  = build_ctx();
    CHECK_OK(qwen35_bf16::declare(&b, &meta, &ctx));

    CHECK_EQ((int)b.kv_groups.size(), 3);
    CHECK_EQ(b.kv_groups[0].name, std::string("kv_attn"));
    CHECK_EQ(b.kv_groups[1].name, std::string("kv_gdn_state"));
    CHECK_EQ(b.kv_groups[2].name, std::string("kv_gdn_conv"));

    const RadKVGroupDecl& f = b.kv_groups[0].d;
    CHECK_EQ(f.kind, RAD_KV_FULL);
    CHECK_EQ(f.n_head_kv, 4);
    CHECK_EQ(f.head_dim, 256);
    CHECK_EQ(f.dtype, (uint32_t)RAD_BF16);

    /* 48 heads of 128x128 fp32: 3 MiB per layer, 144 MiB per sequence across the real model's 48
     * linear layers. fp32 because that is the state width libr4d has a kernel for. */
    const RadKVGroupDecl& s = b.kv_groups[1].d;
    CHECK_EQ(s.kind, RAD_KV_LINEAR);
    CHECK_EQ(s.n_head_kv, 48);
    CHECK_EQ(s.state_dim[0], 128);
    CHECK_EQ(s.state_dim[1], 128);
    CHECK_EQ(s.dtype, (uint32_t)RAD_F32);

    /* The conv window is counted in heads (2*16 + 48 of width 128 = 10240 channels), so a
     * tensor-parallel split never straddles a q/k/v boundary. */
    const RadKVGroupDecl& c = b.kv_groups[2].d;
    CHECK_EQ(c.kind, RAD_KV_CONV);
    CHECK_EQ(c.n_head_kv, 80);
    CHECK_EQ(c.head_dim, 128);
    CHECK_EQ(c.conv_width, 4);

    /* Every layer is bound to the groups it owns state in; a linear layer owns two. */
    std::map<rad_kvgroup, std::set<int>> bound;
    for (const auto& p : b.binds) bound[p.second].insert(p.first);
    CHECK_EQ((int)bound[1].size(), 2);          /* full attention: layers 3 and 7 */
    CHECK_EQ((int)bound[2].size(), 6);
    CHECK_EQ((int)bound[3].size(), 6);
    CHECK(bound[1].count(3) && bound[1].count(7));
    CHECK(bound[2].count(0) && bound[2].count(6));
    CHECK(bound[2] == bound[3]);                /* the two GDN groups cover the same layers */
    for (int l = 0; l < 8; ++l)
        CHECK_EQ(bound[1].count(l) + bound[2].count(l), (size_t)1);
}

TEST(the_name_map_owns_both_sides) {
    RadBuilder b;
    RadModelMeta meta = qwen35_meta();
    RadBuildCtx  ctx  = build_ctx();
    CHECK_OK(qwen35_bf16::declare(&b, &meta, &ctx));

    /* The interleaved query-and-gate projection is one checkpoint tensor and one declared weight. */
    const RecMap* qg = find_map(b, "blk.3.attn_qg.weight");
    CHECK(qg != nullptr);
    CHECK_EQ(qg->mode, RAD_MAP_COPY);
    CHECK_EQ((int)qg->src.size(), 1);
    CHECK_EQ(qg->src[0],
             std::string("model.language_model.layers.3.self_attn.q_proj.weight"));

    /* The GDN in-projection is a convert-time fusion of qkv, a and b, in gdn_conv_prep's row
     * order. */
    const RecMap* in = find_map(b, "blk.0.ssm_in.weight");
    CHECK(in != nullptr);
    CHECK_EQ(in->mode, RAD_MAP_CONCAT);
    CHECK_EQ((int)in->src.size(), 3);
    CHECK_EQ(in->src[0],
             std::string("model.language_model.layers.0.linear_attn.in_proj_qkv.weight"));
    CHECK_EQ(in->src[1],
             std::string("model.language_model.layers.0.linear_attn.in_proj_a.weight"));
    CHECK_EQ(in->src[2],
             std::string("model.language_model.layers.0.linear_attn.in_proj_b.weight"));
    CHECK_EQ(in->concat_dim, 0);

    /* A_log maps straight through; the checkpoint tensor is already a log. */
    const RecMap* al = find_map(b, "blk.0.ssm_a_log");
    CHECK(al != nullptr);
    CHECK_EQ(al->src[0], std::string("model.language_model.layers.0.linear_attn.A_log"));

    /* gate and up are one weight and one GEMM. */
    const RecMap* gu = find_map(b, "blk.1.ffn_gate_up.weight");
    CHECK(gu != nullptr);
    CHECK_EQ(gu->mode, RAD_MAP_CONCAT);
    CHECK_EQ((int)gu->src.size(), 2);
    CHECK_EQ(gu->src[0], std::string("model.language_model.layers.1.mlp.gate_proj.weight"));
    CHECK_EQ(gu->src[1], std::string("model.language_model.layers.1.mlp.up_proj.weight"));

    /* The feed-forward's norm is the post-attention norm, not the input norm. */
    const RecMap* fn = find_map(b, "blk.1.ffn_norm.weight");
    CHECK(fn != nullptr);
    CHECK_EQ(fn->src[0],
             std::string("model.language_model.layers.1.post_attention_layernorm.weight"));

    /* This model does not tie its lm_head; the second source serves family members that do. */
    const RecMap* lm = find_map(b, "output.weight");
    CHECK(lm != nullptr);
    CHECK_EQ(lm->mode, RAD_MAP_COPY);
    CHECK_EQ((int)lm->src.size(), 2);
    CHECK_EQ(lm->src[0], std::string("lm_head.weight"));
    CHECK_EQ(lm->src[1], std::string("model.language_model.embed_tokens.weight"));

    /* Every declared weight has a map entry, so rad-convert can fill all of them. */
    for (const auto& w : b.weights)
        CHECK(find_map(b, w.name) != nullptr);
}

TEST(every_op_declares_the_buffers_it_touches) {
    RadBuilder b;
    RadModelMeta meta = qwen35_meta();
    RadBuildCtx  ctx  = build_ctx();
    CHECK_OK(qwen35_bf16::declare(&b, &meta, &ctx));

    /* A transient without declared uses has no liveness, so the planner must keep it live for the
     * whole program. A GDN layer has eleven intermediates, so this decides the arena size. */
    for (const auto& o : b.ops)
        if (o.resolved) CHECK(o.uses_declared);

    const RecOp* cp = find_op(b, "gdn_conv_prep");
    CHECK_EQ((int)cp->reads.size(), 1);      /* the fused projection row */
    CHECK_EQ((int)cp->writes.size(), 5);     /* q, k, v, g, beta */

    /* kv_store writes the group's pool, not a declared buffer, so its write set is empty by
     * design; it reads the two projections it stores. */
    const RecOp* ks = find_op(b, "kv_store");
    CHECK_EQ((int)ks->reads.size(), 2);
    CHECK_EQ((int)ks->writes.size(), 0);

    /* The residual add reads the stream and the block output and writes the stream. */
    const RecOp* ad = find_op(b, "add");
    CHECK_EQ((int)ad->reads.size(), 2);
    CHECK_EQ((int)ad->writes.size(), 1);
    CHECK_EQ(ad->writes[0], ad->reads[0]);
}

/* The gated attention layer runs the four ops the undeclared fused op stands for; no refusal in
 * the fake hierarchy is needed to reach this sequence. */
TEST(the_gated_attention_layer_runs_the_unfused_sequence) {
    RadBuilder b;
    RadModelMeta meta = qwen35_meta();
    RadBuildCtx  ctx  = build_ctx();
    CHECK_OK(qwen35_bf16::declare(&b, &meta, &ctx));

    CHECK_EQ(join(op_names(b)), join(program_ops(/*tp=*/false)));

    /* The per-head norms are rmsnorm over head_dim with M ranged over tokens x heads. The q norm
     * reads the strided interleaved projection and writes the contiguous q buffer, de-interleaving
     * and normalising in one pass. */
    const RecOp* an = nullptr;      /* the first attention layer's q norm */
    for (size_t i = 0; i + 1 < b.ops.size(); ++i)
        if (b.ops[i].op == "rope") { an = &b.ops[i - 2]; break; }
    CHECK(an != nullptr);
    if (!an) return;
    CHECK_EQ(an->op, std::string("rmsnorm"));
    CHECK_EQ(pint(an, "n"), 256);
    CHECK_EQ(param(an, "M")->ihi, 256LL * 24);
    CHECK_EQ((an + 1)->op, std::string("rmsnorm"));
    CHECK_EQ(param(an + 1, "M")->ihi, 256LL * 4);

    const RecOp* rq = an + 2;
    CHECK_EQ(rq->op, std::string("rope"));
    CHECK_EQ(pint(rq, "head_dim"), 256);
    CHECK_EQ(pint(rq, "n_head"), 24);
    CHECK_EQ(pint(rq, "n_head_kv"), 0);      /* q only: no k or v heads */
    CHECK_EQ(pint(rq, "rotary_dim"), 64);
    CHECK_EQ(pstr(rq, "mode"), std::string("mrope"));
    CHECK_NEAR(pflt(rq, "theta"), 1e7, 1.0);

    const RecOp* rk = an + 3;
    CHECK_EQ(rk->op, std::string("rope"));
    CHECK_EQ(pint(rk, "n_head"), 4);         /* the k row, 4 heads */
    CHECK_EQ(pint(rk, "n_head_kv"), 0);

    /* The step issues them, once per full-attention layer. */
    RadBatch batch = decode_batch();
    RadCtx c{ .b = &b, .batch = &batch };
    qwen35_bf16::step(&c, &batch);
    CHECK_LINT(b, &c);
    int n_rope = 0;
    for (const auto& i : c.issues)
        if (b.op_of(i.op) && b.op_of(i.op)->op == "rope") ++n_rope;
    CHECK_EQ(n_rope, 4);                     /* two per attention layer, two layers */

    /* The sigmoid reads the interleaved projection: a column slice at offset head_dim, width
     * head_dim, of the buffer whose declared width is the pitch. */
    for (const auto& i : c.issues) {
        const RecOp* o = b.op_of(i.op);
        if (!o || o->op != "sigmoid") continue;
        CHECK_EQ(i.opd[0].kind, (uint8_t)RAD_OPK_BUF);
        CHECK_EQ(i.opd[0].offset, 256);      /* the gate half of the head's pair */
        CHECK_EQ(i.opd[0].cols, 256);
        CHECK_EQ(i.opd[0].rows, batch.n_tok);
    }
}

TEST(tensor_parallel_divides_dimensions_and_places_the_collectives) {
    RadBuilder b0, b1;
    RadModelMeta meta = qwen35_meta();
    RadBuildCtx  c0 = build_ctx(0, 2), c1 = build_ctx(1, 2);
    CHECK_OK(qwen35_bf16::declare(&b0, &meta, &c0));
    CHECK_OK(qwen35_bf16::declare(&b1, &meta, &c1));

    /* Declare runs once per rank, and the plugin declares already-divided dimensions. */
    const RecWeight* qg = find_w(b0, "blk.3.attn_qg.weight");
    CHECK_EQ(qg->d.shape[0], 6144);           /* 12 heads x 256 x 2 */
    CHECK_EQ(qg->d.shape[1], 5120);           /* the residual stream is not sharded */
    CHECK_EQ(find_w(b0, "blk.3.attn_k.weight")->d.shape[0], 512);      /* 2 kv heads */
    const RecWeight* o = find_w(b0, "blk.3.attn_output.weight");
    CHECK_EQ(o->d.shape[0], 5120);
    CHECK_EQ(o->d.shape[1], 3072);
    CHECK_EQ(o->d.shard, RAD_SHARD_COL);

    /* The GDN row halves by head: 8 key and 24 value heads give 2*8*128 + 24*128 = 5120 convolved
     * channels, plus 48 for a and b. */
    const RecWeight* in = find_w(b0, "blk.0.ssm_in.weight");
    CHECK_EQ(in->d.shape[0], 5168);
    CHECK_EQ(find_w(b0, "blk.0.ssm_z.weight")->d.shape[0], 3072);
    CHECK_EQ(find_w(b0, "blk.0.ssm_conv1d.weight")->d.shape[0], 5120);
    CHECK_EQ(find_w(b0, "blk.0.ssm_a_log")->d.shape[0], 24);
    CHECK_EQ(find_w(b0, "blk.0.ssm_out.weight")->d.shape[1], 3072);
    /* The gain vectors are head-width and stay whole. */
    CHECK_EQ(find_w(b0, "blk.0.ssm_norm.weight")->d.shape[0], 128);
    CHECK_EQ(find_w(b0, "blk.3.attn_q_norm.weight")->d.shape[0], 256);

    const RecWeight* gu = find_w(b0, "blk.0.ffn_gate_up.weight");
    CHECK_EQ(gu->d.shape[0], 17408);          /* 2 * (17408 / 2) */
    CHECK_EQ(find_w(b0, "blk.0.ffn_down.weight")->d.shape[1], 8704);

    /* gqa and head width are rank-invariant. */
    CHECK_EQ(pint(find_op(b0, "attn_paged"), "gqa"), 6);
    CHECK_EQ(pint(find_op(b0, "gdn_conv_prep"), "head_k"), 128);
    CHECK_EQ(pint(find_op(b0, "gdn_recurrent_update"), "n_head_v"), 24);

    /* The KV groups halve with the heads they hold. */
    CHECK_EQ(b0.kv_groups[0].d.n_head_kv, 2);
    CHECK_EQ(b0.kv_groups[1].d.n_head_kv, 24);
    CHECK_EQ(b0.kv_groups[2].d.n_head_kv, 40);   /* 2*8 + 24 */

    /* Two collectives per layer (the block's and the feed-forward's column-parallel projections)
     * plus one for the vocab-sharded embedding, and no others. */
    CHECK_EQ(count_op(b0, "all_reduce"), 17);
    CHECK_EQ(join(op_names(b0)), join(program_ops(/*tp=*/true)));

    const RecOp* ar = find_op(b0, "all_reduce");
    CHECK_EQ(pint(ar, "world_size"), 2);
    CHECK_EQ(pint(ar, "exact"), 1);           /* must be stated, or no kernel matches */
    CHECK_EQ(param(ar, "numel")->kind, RAD_P_RANGE);
    CHECK_EQ(param(ar, "numel")->ihi, 256LL * 5120);

    /* The vocabulary split is ragged: ceil(total/world) rows on every rank but the last, which
     * takes the rest. This must match the rule core/engine.cpp restates and checks; since 248320
     * divides by two, the sum and shard are asserted rather than a value "floor" would also give. */
    const RecWeight* lm0 = find_w(b0, "output.weight");
    const RecWeight* lm1 = find_w(b1, "output.weight");
    CHECK_EQ(lm0->d.shape[0] + lm1->d.shape[0], 248320);
    CHECK_EQ(lm0->d.shape[0], (248320 + 1) / 2);
    CHECK_EQ(lm0->d.shard, RAD_SHARD_ROW);
    CHECK_EQ(lm1->d.shard, RAD_SHARD_ROW);
    CHECK_EQ(pint(find_op(b0, "logits_gemm"), "n_vocab"), lm0->d.shape[0]);
    /* The draft head stays whole on every rank: its argmax is a row_topk over one rank's plane
     * that bypasses the sampler's merge, so a sharded plane would only propose tokens from that
     * rank's half. */
    if (const RecWeight* dh = find_w(b0, "mtp.draft_head.weight")) {
        CHECK_EQ(dh->d.shape[0], 248320);
        CHECK_EQ(dh->d.shard, RAD_SHARD_NONE);
    }
    /* The embedding uses the same ragged split as the lm_head (whole, it would be gigabytes of
     * replicated bf16 per card). The shards sum to the vocabulary and each rank's embed_lookup gets
     * its slice's vocab_offset. */
    const RecWeight* te0 = find_w(b0, "token_embd.weight");
    const RecWeight* te1 = find_w(b1, "token_embd.weight");
    CHECK_EQ(te0->d.shape[0] + te1->d.shape[0], 248320);
    CHECK_EQ(te0->d.shape[0], (248320 + 1) / 2);
    CHECK_EQ(te0->d.shard, RAD_SHARD_ROW);
    CHECK_EQ(te1->d.shard, RAD_SHARD_ROW);
    CHECK_EQ(te0->d.access, RAD_ACCESS_VOCAB);
    CHECK_EQ(pint(find_op(b0, "embed_lookup"), "n_vocab"), te0->d.shape[0]);
    CHECK_EQ(pint(find_op(b0, "embed_lookup"), "vocab_offset"), 0);
    CHECK_EQ(pint(find_op(b1, "embed_lookup"), "vocab_offset"), te0->d.shape[0]);

    /* A rank count that does not divide the heads is refused, not rounded. */
    RadBuilder b5;
    RadBuildCtx c5 = build_ctx(0, 5);
    CHECK_EQ(qwen35_bf16::declare(&b5, &meta, &c5), RAD_E_INVAL);

    /* With one rank there are no collectives. */
    RadBuilder bs;
    RadBuildCtx cs = build_ctx(0, 1);
    CHECK_OK(qwen35_bf16::declare(&bs, &meta, &cs));
    CHECK_EQ(count_op(bs, "all_reduce"), 0);
}

TEST(a_decode_step_issues_both_layer_types_in_order) {
    RadBuilder b;
    RadModelMeta meta = qwen35_meta();
    RadBuildCtx  ctx  = build_ctx();
    CHECK_OK(qwen35_bf16::declare(&b, &meta, &ctx));

    RadBatch batch = decode_batch();
    RadCtx c{ .b = &b, .batch = &batch };
    qwen35_bf16::step(&c, &batch);
    CHECK_LINT(b, &c);

    std::vector<std::string> want = { "embed_lookup" };
    for (int l = 0; l < 8; ++l) {
        if ((l + 1) % 4 == 0) {
            /* Issue order differs from declare order in one place: kv_store writes the cache
             * before attn_paged reads it. */
            const char* seq[] = { "rmsnorm", "gemm_nt", "gemm_nt", "gemm_nt",
                                  "rmsnorm", "rmsnorm", "rope", "rope",
                                  "kv_store", "attn_paged",
                                  "sigmoid", "mul", "gemm_nt", "add" };
            for (const char* s : seq) want.push_back(s);
        } else {
            /* Decode takes the recurrent path, which folds in the gated norm. */
            const char* seq[] = { "rmsnorm", "gemm_nt", "gemm_nt",
                                  "gdn_conv_update", "gdn_recurrent_update", "gemm_nt", "add" };
            for (const char* s : seq) want.push_back(s);
        }
        for (const char* s : { "rmsnorm", "gemm_nt", "silu_mul", "gemm_nt", "add" })
            want.push_back(s);
    }
    want.push_back("rmsnorm");
    want.push_back("gather_rows");
    want.push_back("logits_gemm");
    CHECK_EQ(join(issued_ops(b, c)), join(want));

    /* The ranged value: RAD_N_BATCH for token-ranged ops, the query length for q_len-banded ops,
     * an explicit row count where rows are tokens x heads, and n_out for the two vocabulary ops,
     * whose band is the number of sampled rows, not n_tok. */
    for (const auto& i : c.issues) {
        const RecOp* o = b.op_of(i.op);
        if (!o) continue;
        if (o->op == "attn_paged" || o->op == "gdn_conv_update" ||
            o->op == "gdn_recurrent_update")
            CHECK_EQ(i.n, (int64_t)batch.max_q_len);
        else if (o->op == "sigmoid")
            CHECK_EQ(i.n, batch.n_tok * 24);
        else if (o->op == "rmsnorm")
            /* Three rmsnorm row counts: layer pre-norms over tokens (RAD_N_BATCH), and the
             * per-head q and k norms at tokens x n_head and tokens x n_head_kv. */
            CHECK(i.n == RAD_N_BATCH || i.n == batch.n_tok * 24 || i.n == batch.n_tok * 4);
        else if (o->op == "gather_rows" || o->op == "logits_gemm")
            CHECK_EQ(i.n, batch.n_out);
        else
            CHECK_EQ(i.n, RAD_N_BATCH);
    }

    /* Every operand is a declared handle or a batch pointer, and every buffer operand carries the
     * step's row count, not the declared extent: kernels read extents from operands, so an
     * un-narrowed operand would compute a one-row decode over the whole arena. */
    for (const auto& i : c.issues)
        for (const auto& od : i.opd) {
            if (od.kind == RAD_OPK_BUF) {
                CHECK(od.handle >= 1 && od.handle <= b.bufs.size());
                CHECK_EQ(od.rows, batch.n_tok);
            }
            if (od.kind == RAD_OPK_WEIGHT) CHECK(od.handle >= 1 && od.handle <= b.weights.size());
        }
}

TEST(a_prefill_step_takes_the_chunked_gdn_path) {
    RadBuilder b;
    RadModelMeta meta = qwen35_meta();
    RadBuildCtx  ctx  = build_ctx();
    CHECK_OK(qwen35_bf16::declare(&b, &meta, &ctx));

    RadBatch batch = prefill_batch();
    RadCtx c{ .b = &b, .batch = &batch };
    qwen35_bf16::step(&c, &batch);
    CHECK_LINT(b, &c);

    std::vector<std::string> got = issued_ops(b, c);
    int n_prep = 0, n_scan = 0, n_recur = 0, n_gnorm = 0;
    for (const auto& s : got) {
        if (s == "gdn_conv_prep")        ++n_prep;
        if (s == "gdn_chunk_scan")       ++n_scan;
        if (s == "gdn_recurrent_update") ++n_recur;
        if (s == "gdn_gated_rmsnorm")    ++n_gnorm;
    }
    CHECK_EQ(n_prep, 6);
    CHECK_EQ(n_scan, 6);
    CHECK_EQ(n_gnorm, 6);
    CHECK_EQ(n_recur, 0);      /* declared but not issued in prefill */

    /* The decay and beta inputs are the last 2*n_head_v columns of the projection row: read
     * inside gdn_conv_prep on the chunked path, as strided views on the recurrent one. */
    const RecOp* cp = nullptr;
    for (const auto& i : c.issues) {
        const RecOp* o = b.op_of(i.op);
        if (o && o->op == "gdn_conv_prep") {
            cp = o;
            /* Sixteen operands, inputs before outputs (docs/OPS.md, "Linear attention").
             * `a` and `b_gate` (the recurrent path's strided views) are passed although optional:
             * libr4d dereferences them and `cache_idx` unconditionally, so omitting them would
             * silently resolve the layer to libref. */
            CHECK_EQ((int)i.opd.size(), 16);
            CHECK_EQ(i.opd[2].kind, (uint8_t)RAD_OPK_NONE);   /* no conv bias on this model */
            CHECK_EQ(i.opd[5].kind, (uint8_t)RAD_OPK_KV);     /* the conv state */
            CHECK_EQ(i.opd[6].rows, batch.n_seq + 1);         /* cu_seqlens */
            CHECK_EQ(i.opd[7].offset, 10240);                 /* a */
            CHECK_EQ(i.opd[7].cols, 48);
            CHECK_EQ(i.opd[8].offset, 10288);                 /* b_gate */
            CHECK_EQ(i.opd[10].rows, batch.n_seq);            /* has_init == ctx_lens */
            break;
        }
    }
    CHECK(cp != nullptr);

    /* The scan reads and writes the recurrent state group, as h0 and ht. */
    for (const auto& i : c.issues) {
        const RecOp* o = b.op_of(i.op);
        if (o && o->op == "gdn_chunk_scan") {
            /* Eleven: the state index after the two KV operands makes the scan write the
             * sequence's manager-assigned slot, not its batch row. */
            CHECK_EQ((int)i.opd.size(), 11);
            CHECK_EQ(i.opd[6].kind, (uint8_t)RAD_OPK_KV);
            CHECK_EQ(i.opd[9].kind, (uint8_t)RAD_OPK_KV);
            CHECK_EQ(i.opd[10].rows, batch.n_seq);
            break;
        }
    }
}

TEST(the_recurrent_path_slices_a_and_b_out_of_the_projection_row) {
    RadBuilder b;
    RadModelMeta meta = qwen35_meta();
    RadBuildCtx  ctx  = build_ctx();
    CHECK_OK(qwen35_bf16::declare(&b, &meta, &ctx));

    RadBatch batch = decode_batch();
    RadCtx c{ .b = &b, .batch = &batch };
    qwen35_bf16::step(&c, &batch);
    CHECK_LINT(b, &c);

    bool seen = false;
    for (const auto& i : c.issues) {
        const RecOp* o = b.op_of(i.op);
        if (!o || o->op != "gdn_recurrent_update") continue;
        seen = true;
        /* Sixteen, as both kernel registries declare for gdn_recurrent_update and
         * core/runtime/issue.cpp requires. The count comes from the schema, since this harness
         * only records operands. The last two (the folded out-projection quantiser) are unused by
         * the bf16 block; operands are positional, so an absent optional operand is a null
         * operand, not a shorter list. */
        CHECK_EQ((int)i.opd.size(), 16);
        CHECK_EQ(i.opd[14].kind, (uint8_t)RAD_OPK_NONE);
        CHECK_EQ(i.opd[15].kind, (uint8_t)RAD_OPK_NONE);
        CHECK_EQ(i.opd[3].offset, 10240);      /* a: first column after the conv row */
        CHECK_EQ(i.opd[3].cols, 48);
        CHECK_EQ(i.opd[4].offset, 10288);      /* b: one per value head after a */
        CHECK_EQ(i.opd[4].cols, 48);
        CHECK_EQ(i.opd[7].kind, (uint8_t)RAD_OPK_KV);
        break;
    }
    CHECK(seen);
}

TEST(what_the_plugin_refuses) {
    RadModelMeta meta = qwen35_meta();

    /* A drafter would need a second set of per-rank handles and name-map namespace; nothing
     * unimplemented returns success (spec §17). */
    RadBuilder bd;
    RadBuildCtx cd = build_ctx();
    cd.is_draft = 1;
    CHECK_EQ(qwen35_bf16::declare(&bd, &meta, &cd), RAD_E_UNSUPPORTED);

    /* A routed (MoE) checkpoint is a different graph. */
    RadBuilder bm;
    RadModelMeta moe = meta;
    moe.n_expert = 128;
    moe.n_expert_used = 8;
    RadBuildCtx cm = build_ctx();
    CHECK_EQ(qwen35_bf16::declare(&bm, &moe, &cm), RAD_E_UNSUPPORTED);

    /* A container that does not describe the linear layers is refused, not defaulted; a guessed
     * head count would load and produce wrong text. */
    RadBuilder bn;
    RadModelMeta bare = meta;
    bare.n_kv = 2;                 /* interval and rope width only; no ssm keys */
    RadBuildCtx cn = build_ctx();
    CHECK_EQ(qwen35_bf16::declare(&bn, &bare, &cn), RAD_E_INVAL);

    /* So is one that omits the rotary width. */
    RadBuilder br;
    RadModelMeta norot = meta;
    static const char* rk[] = { "qwen35.full_attention_interval", "qwen35.ssm.state_size",
                                "qwen35.ssm.group_count", "qwen35.ssm.time_step_rank",
                                "qwen35.ssm.inner_size", "qwen35.ssm.conv_kernel" };
    static const char* rv[] = { "4", "128", "16", "48", "6144", "4" };
    norot.n_kv = 6; norot.kv_key = rk; norot.kv_val = rv;
    RadBuildCtx cr = build_ctx();
    CHECK_EQ(qwen35_bf16::declare(&br, &norot, &cr), RAD_E_INVAL);
}

TEST(the_mtp_layer_sits_beside_the_trunk_and_not_inside_it) {
    /* `nextn_predict_layers` is not part of the layer count in this family: Qwen3.8-27B-FP8 sets
     * num_hidden_layers = 64, lists 64 layer_types, ships layers 0..63 under
     * `model.language_model.layers`, and keeps its MTP block in a separate file under `mtp.`.
     * Subtracting it would drop the last (full-attention) trunk layer and silently run one layer
     * short. So every declared layer is a trunk layer and `nextn` does not change the graph. */
    RadBuilder b;
    RadModelMeta meta = qwen35_meta();
    static const char* mk[] = { "qwen35.full_attention_interval", "qwen35.rope.dimension_count",
                                "qwen35.ssm.state_size", "qwen35.ssm.group_count",
                                "qwen35.ssm.time_step_rank", "qwen35.ssm.inner_size",
                                "qwen35.ssm.conv_kernel", "qwen35.nextn_predict_layers" };
    static const char* mv[] = { "4", "64", "128", "16", "48", "6144", "4", "1" };
    meta.n_kv = 8; meta.kv_key = mk; meta.kv_val = mv;
    RadBuildCtx ctx = build_ctx();
    CHECK_OK(qwen35_bf16::declare(&b, &meta, &ctx));

    /* The same graph as without nextn. */
    CHECK_EQ(join(op_names(b)), join(program_ops(/*tp=*/false)));
    /* The last trunk layer, blk.7, is declared; (7 + 1) % 4 == 0 makes it full attention, the
     * layer a subtraction would drop. */
    CHECK(find_w(b, "blk.7.attn_norm.weight") != nullptr);
    CHECK(find_w(b, "blk.8.attn_norm.weight") == nullptr);
}

/* The PLE windowed ops are issued once per half of the step, and the halves differ in whether the
 * ops see `num_accepted`, which picks their write-back layout. Decode rows may reject drafts, so
 * `ngram_ids` and `ple_conv` must see it; a prefill chunk commits everything, so they must not, and
 * the chunk's tail stays at offset zero for the scheduler's `num_accepted` of 1 next step. A mixed
 * step therefore needs two issues; a pure step issues one. */
TEST(the_ple_windows_are_issued_once_per_half_of_a_step) {
    using namespace rad::arch;
    RadBuilder b;
    Names nm("");
    Geom g{};
    g.n_layer = 1; g.n_embd = 64; g.max_tok = 64; g.max_seqs = 4; g.max_spec = 3;
    RadBufDecl hd{};
    hd.dtype = RAD_BF16; hd.rank = 2; hd.shape[0] = g.max_tok; hd.shape[1] = 4 * g.n_embd;
    PleLayer::Wire w{};
    w.h = rad_decl_buffer(&b, "h", &hd);
    PleLayer::Config pc{};
    pc.hc = 4; pc.embed_dim = 32; pc.heads = 4; pc.ngram = 3; pc.eos = 7; pc.rows = 1000;
    pc.shards = 2; pc.conv_width = 4; pc.dilation = 3;
    PleLayer::Src src{};
    src.shard_prefix = "ple.shard_"; src.shard_suffix = ".weight"; src.mult = "ple.mult";
    src.vocab_sizes = "ple.vsz"; src.offsets = "ple.off"; src.key_proj = "ple.k";
    src.value_proj = "ple.v"; src.norm_key = "ple.nk"; src.norm_query = "ple.nq";
    src.norm_conv = "ple.nc"; src.conv1d = "ple.conv";
    PleLayer ple;
    CHECK_OK(ple.declare(&b, nm, g, pc, 0, w, src));

    /* Two decode rows (a verify of 4 and a plain decode of 1) and one prefill chunk of 6. */
    static const int32_t tok[11] = { 1, 2, 3, 4, 5, 10, 11, 12, 13, 14, 15 };
    static const int32_t cu[4]   = { 0, 4, 5, 11 };
    static const int32_t ctxl[3] = { 40, 30, 12 };
    static const int32_t nacc[3] = { 2, 1, 1 };
    static int32_t tk_idx[3] = { 5, 6, 7 }, cv_idx[3] = { 1, 2, 3 };
    RadKVGroupBatch kv[2]{};
    kv[0].group = ple.kv_tok;  kv[0].state_index = tk_idx;
    kv[1].group = ple.kv_conv; kv[1].state_index = cv_idx;

    RadBatch mixed{};
    mixed.phase = RAD_PHASE_MIXED;
    mixed.n_tok = 11; mixed.n_seq = 3;
    mixed.n_seq_decode = 2; mixed.n_tok_decode = 5;
    mixed.token_ids = tok; mixed.cu_seqlens = cu; mixed.ctx_lens = ctxl;
    mixed.num_accepted = nacc;
    mixed.n_kv_groups = 2; mixed.kv = kv;

    RadCtx c{ .b = &b, .batch = &mixed };
    /* The hash is issued at the top of the step and the rest at the layer. */
    ple.ids(&c, &mixed);
    ple.step(&c, &mixed);

    const char* want[] = { "ngram_ids", "ngram_ids", "embed_lookup_q", "cast", "gemm_nt",
                           "gemm_nt", "ple_gate", "ple_conv", "ple_conv", "add" };
    std::vector<std::string> wv(want, want + 10);
    CHECK_EQ(join(issued_ops(b, c)), join(wv));
    REQUIRE(c.issues.size() == 10);

    /* The projections read the gather's rows from the card: the copy reads the gather's output
     * and both GEMMs read the copy, never the host buffer. */
    {
        const RecIssue& gth = c.issues[2];
        const RecIssue& cp  = c.issues[3];
        REQUIRE(gth.opd.size() == 5 && cp.opd.size() == 2);
        CHECK_EQ(cp.opd[0].handle, gth.opd[3].handle);
        CHECK_EQ(cp.opd[0].rows, (int64_t)11);
        CHECK(cp.opd[1].handle != gth.opd[3].handle);
        CHECK_EQ(c.issues[4].opd[0].handle, cp.opd[1].handle);
        CHECK_EQ(c.issues[5].opd[0].handle, cp.opd[1].handle);
    }

    /* ngram_ids: tok, state, cu, 3 weights, cache_idx, has_init, num_accepted, ids, ahead_ids. */
    {
        const RecIssue& d = c.issues[0];
        const RecIssue& p = c.issues[1];
        REQUIRE(d.opd.size() == 11 && p.opd.size() == 11);
        /* No following prompt (n_ahead 0); a decode half never has one. */
        CHECK_EQ(d.opd[10].kind, (uint8_t)RAD_OPK_NONE);
        CHECK_EQ(p.opd[10].kind, (uint8_t)RAD_OPK_NONE);
        CHECK_EQ(d.n, (int64_t)5);
        CHECK_EQ(d.opd[0].rows, (int64_t)5);
        CHECK(d.opd[2].raw == (void*)cu);             CHECK_EQ(d.opd[2].rows, (int64_t)3);
        CHECK(d.opd[6].raw == (void*)tk_idx);         CHECK_EQ(d.opd[6].rows, (int64_t)2);
        CHECK(d.opd[7].raw == (void*)ctxl);
        CHECK_EQ(d.opd[8].kind, (uint8_t)RAD_OPK_RAW);
        CHECK(d.opd[8].raw == (void*)nacc);           CHECK_EQ(d.opd[8].rows, (int64_t)2);
        CHECK_EQ(d.opd[9].rows, (int64_t)5);

        CHECK_EQ(p.n, (int64_t)11);
        CHECK_EQ(p.opd[0].rows, (int64_t)11);         /* token operands stay at their base */
        CHECK(p.opd[2].raw == (void*)(cu + 2));       CHECK_EQ(p.opd[2].rows, (int64_t)2);
        CHECK(p.opd[6].raw == (void*)(tk_idx + 2));   CHECK_EQ(p.opd[6].rows, (int64_t)1);
        CHECK(p.opd[7].raw == (void*)(ctxl + 2));
        CHECK_EQ(p.opd[8].kind, (uint8_t)RAD_OPK_NONE);
        CHECK_EQ(p.opd[9].rows, (int64_t)11);
    }
    /* ple_conv: x, w, state, cu, resid, cache_idx, has_init, num_accepted, y. */
    {
        const RecIssue& d = c.issues[7];
        const RecIssue& p = c.issues[8];
        REQUIRE(d.opd.size() == 9 && p.opd.size() == 9);
        CHECK_EQ(d.n, (int64_t)5);
        CHECK_EQ(d.opd[0].rows, (int64_t)5);
        CHECK(d.opd[3].raw == (void*)cu);
        CHECK(d.opd[5].raw == (void*)cv_idx);
        CHECK(d.opd[7].raw == (void*)nacc);
        CHECK_EQ(d.opd[8].rows, (int64_t)5);

        CHECK_EQ(p.n, (int64_t)11);
        CHECK_EQ(p.opd[0].rows, (int64_t)11);
        CHECK(p.opd[3].raw == (void*)(cu + 2));
        CHECK(p.opd[5].raw == (void*)(cv_idx + 2));
        CHECK(p.opd[6].raw == (void*)(ctxl + 2));
        CHECK_EQ(p.opd[7].kind, (uint8_t)RAD_OPK_NONE);
        CHECK_EQ(p.opd[8].rows, (int64_t)11);
    }

    /* A pure step issues one half: a decode step passes `num_accepted`, a prefill step does not. */
    for (int phase : { (int)RAD_PHASE_DECODE, (int)RAD_PHASE_PREFILL }) {
        RadBatch pure = mixed;
        pure.phase = phase;
        pure.n_seq_decode = 0; pure.n_tok_decode = 0;   /* the phase decides, not these counts */
        RadCtx cp{ .b = &b, .batch = &pure };
        ple.ids(&cp, &pure);
        ple.step(&cp, &pure);
        int n_ids = 0, n_conv = 0;
        for (const auto& i : cp.issues) {
            const RecOp* o = b.op_of(i.op);
            if (!o) continue;
            const bool ids = o->op == "ngram_ids", conv = o->op == "ple_conv";
            if (!ids && !conv) continue;
            n_ids += ids; n_conv += conv;
            CHECK_EQ(i.n, (int64_t)11);
            const RadOperand& na = i.opd[ids ? 8 : 7];
            if (phase == RAD_PHASE_DECODE) CHECK(na.kind == RAD_OPK_RAW && na.raw == (void*)nacc);
            else                           CHECK_EQ(na.kind, (uint8_t)RAD_OPK_NONE);
        }
        CHECK_EQ(n_ids, 1);
        CHECK_EQ(n_conv, 1);
    }

    /* A following prompt rides the prefill half only: its ngram_ids reads `tok` that many tokens
     * further and writes their ids into the `ple_ids` rows after the step's, which the gather also
     * gets. The decode half never sees them. */
    {
        RadBatch ahead = mixed;
        ahead.n_ahead = 7;
        RadCtx ca{ .b = &b, .batch = &ahead };
        ple.ids(&ca, &ahead);
        ple.step(&ca, &ahead);
        REQUIRE(ca.issues.size() == 10);
        const RecIssue& d = ca.issues[0];
        const RecIssue& p = ca.issues[1];
        const RecIssue& gth = ca.issues[2];
        CHECK_EQ(d.opd[0].rows, (int64_t)5);
        CHECK_EQ(d.opd[10].kind, (uint8_t)RAD_OPK_NONE);
        CHECK_EQ(p.n, (int64_t)11);
        CHECK_EQ(p.opd[0].rows, (int64_t)(11 + 7));
        CHECK_EQ(p.opd[9].rows, (int64_t)11);
        CHECK_EQ(p.opd[10].kind, (uint8_t)RAD_OPK_BUF);
        CHECK_EQ(p.opd[10].rows, (int64_t)7);
        CHECK_EQ(p.opd[10].offset, (int64_t)(11 * pc.heads));
        REQUIRE(gth.opd.size() == 5);
        CHECK_EQ(gth.opd[4].kind, (uint8_t)RAD_OPK_BUF);
        CHECK_EQ(gth.opd[4].handle, p.opd[10].handle);
        CHECK_EQ(gth.opd[4].rows, (int64_t)7);
        CHECK_EQ(gth.opd[4].offset, (int64_t)(11 * pc.heads));
    }
}

/* ==================================================================== the sizing declare
 *
 * The activation arena is lent out by level, and a level is the plugin declaring again at a
 * smaller max_tok. The plugin must keep this safe: the sizing declare must not replace the state
 * its step reads (written at the real size), and must describe the same buffers with only their
 * rows smaller, since every other extent is a stride the real kernels were chosen for.
 *
 * Qwen3.8-Flash-Next's metadata, cut to eight layers. */
namespace {
const char* q4_keys[] = {
    "full_attention_interval", "hc_count", "hc_lowrank", "heads_per_ngram",
    "hidden_act", "indexer_budget", "indexer_compress_ratio", "indexer_head_dim",
    "indexer_n_heads", "layer_types", "linear_conv_kernel_dim", "linear_key_head_dim",
    "linear_num_key_heads", "linear_num_value_heads", "linear_value_head_dim",
    "make_ngram_vocab_size_divisible_by",
    "moe_intermediate_size", "mtp_num_hidden_layers", "ngram_size", "ngram_vocab_size_base",
    "num_experts", "num_experts_per_tok", "output_gate_type", "partial_rotary_factor",
    "ple_conv_kernel_size", "ple_embed_dim", "ple_layer_ids", "shared_expert_intermediate_size",
    "split_ngram_parts", "rope_parameters.partial_rotary_factor", "eos_token_id",
};
const char* q4_vals[] = {
    "4", "4", "320", "8",
    "silu", "2048", "4", "128",
    "4",
    "linear_attention linear_attention linear_attention full_attention "
    "linear_attention linear_attention linear_attention full_attention",
    "4", "128",
    "16", "48", "128", "128",
    "640", "1", "3", "20000000",
    "512", "10", "sigmoid", "0.25",
    "4", "2560", "2", "640",
    "128", "0.25", "248044",
};

RadModelMeta qwen4exp_meta() {
    RadModelMeta m{};
    m.arch_id     = "qwen4exp";
    m.name        = "test-q38-flashnext";
    m.quant       = "";
    m.n_layers    = 8;
    m.n_embd      = 2560;
    m.n_head      = 24;
    m.n_head_kv   = 2;
    m.head_dim    = 256;
    m.n_ff        = 640;
    m.n_vocab     = 248320;
    m.n_ctx_train = 262144;
    m.rms_eps     = 1e-6f;
    m.rope_theta  = 1e7f;
    m.rope_scale  = 1.0f;
    m.n_kv        = (int)(sizeof(q4_keys) / sizeof(q4_keys[0]));
    m.kv_key      = q4_keys;
    m.kv_val      = q4_vals;
    return m;
}
}  /* namespace */

TEST(qwen4exp_yarn_reaches_attention_indexer_and_compressed_keys) {
    RadModelMeta meta = qwen4exp_meta();
    std::vector<const char*> keys(meta.kv_key, meta.kv_key + meta.n_kv);
    std::vector<const char*> vals(meta.kv_val, meta.kv_val + meta.n_kv);
    keys.insert(keys.end(), {"text_config.rope_parameters.rope_type", "text_config.rope_parameters.factor",
                            "text_config.rope_parameters.original_max_position_embeddings"});
    vals.insert(vals.end(), {"yarn", "2", "262144"});
    meta.n_kv = (int)keys.size(); meta.kv_key = keys.data(); meta.kv_val = vals.data();
    RadBuildCtx c = build_ctx(0, 1);
    c.max_tok = 2048; c.max_seqs = 8; c.max_spec = 3; c.max_ctx = 524288;
    RadBuilder b;
    REQUIRE_EQ(qwen4exp_fp8::declare(&b, &meta, &c), RAD_OK);
    int direct = 0, table = 0, compressed = 0;
    for (const auto& op : b.ops) {
        if (op.op == "rope") ++direct;
        else if (op.op == "rope_table") ++table;
        else if (op.op == "qsa_block_key") ++compressed;
        else continue;
        int found = 0;
        for (const auto& p : op.p) {
            if (p.key == "yarn_factor") { CHECK_EQ(p.dval, 2.0); ++found; }
            if (p.key == "yarn_magnitude") { CHECK_NEAR(p.dval, 1.069314718, 1e-7); ++found; }
            if (p.key == "scale") CHECK_EQ(p.dval, 1.0);
        }
        CHECK_EQ(found, 2);
    }
    CHECK(direct > 0); CHECK(table > 0); CHECK(compressed > 0);
}

/* The plugin declares by each weight's stored encoding. Flash-Next's experts, gated-residual
 * mixing matrices, MTP fc and 2-bit draft head each have a recipe-chosen format the plugin reads
 * from rad_weight_encoding, at convert and at load alike. With int4-rotated, E4M3-row and u2
 * encodings it declares the rotated expert kernels, the E4M3 mix and the 2-bit head; with no model
 * behind it, the fp8 experts and the bf16 mix. */
TEST(the_encodings_the_model_holds_choose_the_flash_next_kernels) {
    RadModelMeta meta = qwen4exp_meta();
    RadEncoding w4 = rad_enc_affine(RAD_I4, RAD_BF16, 1, 128);
    rad_enc_copy_str(w4.transform, "fwht128");
    const RadEncoding row8 = rad_enc_affine(RAD_F8E4M3, RAD_F32, 1, 128);
    RadEncoding u2 = rad_enc_affine(RAD_U2, RAD_F16, 1, 128);
    rad_enc_add_plane(&u2, "zero", RAD_U8, 1, 128);
    REQUIRE(rad_enc_valid(&w4) && rad_enc_valid(&row8) && rad_enc_valid(&u2));

    RadBuildCtx c = build_ctx(0, 1);
    c.max_tok = 2048; c.max_seqs = 8; c.max_spec = 3; c.max_ctx = 200000;
    RadBuilder b;
    b.encs = { { "ffn_gate_up_exps", w4 }, { "ffn_down_exps", w4 },
               { "_hc_down.weight", row8 }, { "_hc_up.weight", row8 },
               { "fc_hidden", row8 }, { "fc_embedding", row8 }, { "draft_head", u2 } };
    REQUIRE_EQ(qwen4exp_fp8::declare(&b, &meta, &c), RAD_OK);
    const qwen4exp_fp8::Model& m = qwen4exp_fp8::g_model[0];
    CHECK_EQ(m.moecfg.expert_w4, 1);
    CHECK_EQ(m.moecfg.expert_rot, 1);
    CHECK(m.layers[0].hc_mix.mix_e4m3);
    int rotated = 0, fp8_experts = 0, head_views = 0, scale_views = 0;
    for (const RecOp& o : b.ops)
        if (o.op.rfind("moe_gemm", 0) == 0)
            for (const auto& p : o.p) {
                if (p.key == "dtype" && p.sval == "w4a8h") ++rotated;
                if (p.key == "dtype" && p.sval == "fp8a8") ++fp8_experts;
            }
    for (const RecWeight& w : b.weights) {
        if (w.d.planes && std::string(w.d.planes) == "scale,zero") ++head_views;
        /* every expert's scale is a view of its logical weight, not a separate weight */
        if (w.name.find("_exps.") != std::string::npos &&
            w.name.size() > 6 && w.name.compare(w.name.size() - 6, 6, ".scale") == 0) {
            ++scale_views;
            CHECK(w.d.source != nullptr && w.d.planes && std::string(w.d.planes) == "scale");
        }
    }
    CHECK(rotated > 0);
    CHECK_EQ(fp8_experts, 0);
    CHECK_EQ(head_views, 1);
    CHECK(scale_views > 0);

    RadBuilder plain;
    REQUIRE_EQ(qwen4exp_fp8::declare(&plain, &meta, &c), RAD_OK);
    const qwen4exp_fp8::Model& p = qwen4exp_fp8::g_model[0];
    CHECK_EQ(p.moecfg.expert_w4, 0);
    CHECK(!p.layers[0].hc_mix.mix_e4m3);
}

TEST(manifold_connections_preserve_saved_coefficients_and_use_a_separate_residual_destination) {
    using namespace rad::arch;
    RadBuilder b;
    Names nm("");
    Geom g{};
    g.n_embd = 4096;
    g.max_tok = 32;
    g.world = 4;
    ManifoldConn::Config cfg;
    ManifoldConn::Src src{ "model.layers.3.hc_attn_fn", "model.layers.3.hc_attn_scale",
                           "model.layers.3.hc_attn_base" };
    ManifoldConn::Wire w;
    w.h = decl_b(&b, "h", RAD_BF16, { 32, 16384 });
    w.x = decl_b(&b, "x", RAD_BF16, { 32, 4096 });
    w.y = decl_b(&b, "y", RAD_BF16, { 32, 4096 });
    w.out = decl_b(&b, "out", RAD_BF16, { 32, 16384 });
    w.post = decl_b(&b, "post", RAD_F32, { 32, 4 });
    w.comb = decl_b(&b, "comb", RAD_F32, { 32, 16 });
    ManifoldConn conn;
    REQUIRE_EQ(conn.declare(&b, nm, g, 3, "blk.3.mhc_attn", cfg, w, src), RAD_OK);
    REQUIRE_EQ(b.weights.size(), (size_t)3);
    CHECK_EQ(b.weights[0].d.shape[0], 24);
    CHECK_EQ(b.weights[0].d.shape[1], 16384);
    CHECK_EQ(b.weights[1].d.shape[0], 3);
    CHECK_EQ(b.weights[2].d.shape[0], 24);
    for (const auto& weight : b.weights) {
        CHECK_EQ(weight.d.dtype, RAD_F32);
        CHECK_EQ(weight.d.shard, RAD_SHARD_NONE);
    }
    CHECK(b.ops[conn.op_pre - 1].reads == std::vector<rad_buf>({ w.h }));
    CHECK(b.ops[conn.op_pre - 1].writes == std::vector<rad_buf>({ w.x, w.post, w.comb }));
    CHECK(b.ops[conn.op_post - 1].reads == std::vector<rad_buf>({ w.y, w.h, w.post, w.comb }));
    CHECK(b.ops[conn.op_post - 1].writes == std::vector<rad_buf>({ w.out }));
    RadCtx ctx;
    conn.read(&ctx, 4);
    conn.write(&ctx, 4);
    REQUIRE_EQ(ctx.issues.size(), (size_t)2);
    CHECK_EQ(ctx.issues[0].op, conn.op_pre);
    CHECK_EQ(ctx.issues[1].op, conn.op_post);
    CHECK_EQ(ctx.issues[0].n, 4);
    CHECK_EQ(ctx.issues[1].n, 4);
    REQUIRE_EQ(ctx.issues[0].opd.size(), (size_t)7);
    CHECK_EQ(ctx.issues[0].opd[0].handle, w.h);
    const rad_weight coefficients[] = { conn.w_fn, conn.w_scale, conn.w_base };
    for (int i = 0; i < 3; ++i) {
        CHECK_EQ(ctx.issues[0].opd[i + 1].kind, RAD_OPK_WEIGHT);
        CHECK_EQ(ctx.issues[0].opd[i + 1].handle, coefficients[i]);
    }
    CHECK_EQ(ctx.issues[0].opd[4].handle, w.x);
    CHECK_EQ(ctx.issues[0].opd[5].handle, w.post);
    CHECK_EQ(ctx.issues[0].opd[6].handle, w.comb);
    REQUIRE_EQ(ctx.issues[1].opd.size(), (size_t)5);
    const rad_buf expected[] = { w.y, w.h, w.post, w.comb, w.out };
    for (int i = 0; i < 5; ++i) {
        CHECK_EQ(ctx.issues[1].opd[i].handle, expected[i]);
        CHECK_EQ(ctx.issues[1].opd[i].rows, 4);
    }
    auto invalid = w;
    invalid.out = w.h;
    ManifoldConn refused;
    CHECK_EQ(refused.declare(&b, nm, g, 3, "bad", cfg, invalid, src), RAD_E_INVAL);
}

TEST(manifold_read_splits_its_projection_by_chunks_over_ranks_from_the_row_threshold) {
    using namespace rad::arch;
    for (int world : { 1, 2, 4, 8 })
        for (int rank = 0; rank < world; ++rank) {
            RadBuilder b;
            Names nm("");
            Geom g{};
            g.n_embd = 4096;
            g.max_tok = 512;
            g.world = world;
            g.rank = rank;
            ManifoldConn::Config cfg;
            ManifoldConn::Src src{ "model.layers.3.hc_attn_fn", "model.layers.3.hc_attn_scale",
                                   "model.layers.3.hc_attn_base" };
            ManifoldConn::Wire w;
            w.h = decl_b(&b, "h", RAD_BF16, { 512, 16384 });
            w.x = decl_b(&b, "x", RAD_BF16, { 512, 4096 });
            w.y = decl_b(&b, "y", RAD_BF16, { 512, 4096 });
            w.out = decl_b(&b, "out", RAD_BF16, { 512, 16384 });
            w.post = decl_b(&b, "post", RAD_F32, { 512, 4 });
            w.comb = decl_b(&b, "comb", RAD_F32, { 512, 16 });
            ManifoldConn conn;
            REQUIRE_EQ(conn.declare(&b, nm, g, 3, "blk.3.mhc_attn", cfg, w, src), RAD_OK);
            const bool split = world > 1;
            const int64_t mine = 8 / world;
            CHECK_EQ(count_op(b, "mhc_pre"), 1);
            CHECK_EQ(count_op(b, "mhc_pre_partial"), split ? 1 : 0);
            CHECK_EQ(count_op(b, "all_gather"), split ? 1 : 0);
            CHECK_EQ(count_op(b, "mhc_pre_finish"), split ? 1 : 0);
            CHECK_EQ(b.weights[0].d.shard, RAD_SHARD_NONE);
            if (split) {
                const RecOp* part = find_op(b, "mhc_pre_partial");
                const RecOp* gather = find_op(b, "all_gather");
                const RecOp* finish = find_op(b, "mhc_pre_finish");
                REQUIRE(part && gather && finish);
                CHECK_EQ(pint(part, "chunk"), 2048);
                CHECK_EQ(pint(part, "first"), rank * mine);
                CHECK_EQ(pint(part, "chunks"), mine);
                CHECK(part->w == std::vector<rad_weight>({ conn.w_fn }));
                CHECK(part->reads == std::vector<rad_buf>({ w.h }));
                CHECK(part->writes == std::vector<rad_buf>({ conn.part }));
                CHECK_EQ(pint(gather, "world_size"), world);
                CHECK_EQ(pint(gather, "row"), mine * 28);
                CHECK_EQ(param(gather, "numel")->kind, RAD_P_RANGE);
                CHECK_EQ(param(gather, "numel")->ival, mine * 28);
                CHECK_EQ(param(gather, "numel")->ihi, 512 * mine * 28);
                CHECK_EQ(pstr(gather, "dtype"), "f32");
                CHECK(gather->reads == std::vector<rad_buf>({ conn.part }));
                CHECK(gather->writes == std::vector<rad_buf>({ conn.parts }));
                CHECK_EQ(pint(finish, "chunk"), 2048);
                CHECK_EQ(pint(finish, "iterations"), 20);
                CHECK(finish->w == std::vector<rad_weight>({ conn.w_scale, conn.w_base }));
                CHECK(finish->reads == std::vector<rad_buf>({ conn.parts, w.h }));
                CHECK(finish->writes == std::vector<rad_buf>({ w.x, w.post, w.comb }));
                CHECK_EQ(find_b(b, "blk.3.mhc_attn.parts")->d.shape[1], 8 * 28);
            }
            RadCtx ctx;
            conn.read(&ctx, 300);
            conn.read(&ctx, 255);
            if (split) {
                REQUIRE_EQ(ctx.issues.size(), (size_t)4);
                CHECK_EQ(ctx.issues[0].op, conn.op_partial);
                CHECK_EQ(ctx.issues[0].opd[0].handle, w.h);
                CHECK_EQ(ctx.issues[0].opd[0].rows, 300);
                CHECK_EQ(ctx.issues[0].opd[1].handle, conn.w_fn);
                CHECK_EQ(ctx.issues[0].opd[2].handle, conn.part);
                CHECK_EQ(ctx.issues[1].op, conn.op_gather);
                CHECK_EQ(ctx.issues[1].n, 300 * mine * 28);
                CHECK_EQ(ctx.issues[1].opd[1].handle, conn.parts);
                CHECK_EQ(ctx.issues[1].opd[1].rows, 300);
                CHECK_EQ(ctx.issues[2].op, conn.op_finish);
                REQUIRE_EQ(ctx.issues[2].opd.size(), (size_t)7);
                const uint32_t expected[] = { conn.parts, w.h,    conn.w_scale, conn.w_base,
                                              w.x,        w.post, w.comb };
                for (int i = 0; i < 7; ++i) CHECK_EQ(ctx.issues[2].opd[i].handle, expected[i]);
                CHECK_EQ(ctx.issues[3].op, conn.op_pre);
            } else {
                REQUIRE_EQ(ctx.issues.size(), (size_t)2);
                CHECK_EQ(ctx.issues[0].op, conn.op_pre);
                CHECK_EQ(ctx.issues[1].op, conn.op_pre);
            }
        }
}

TEST(glm_routing_keeps_f32_logits_and_correction_replicated_under_tp4) {
    using namespace rad::arch;
    for (int rank = 0; rank < 4; ++rank) {
        RadBuilder b;
        Names nm("");
        Geom g{};
        g.n_embd = 4096;
        g.max_tok = 128;
        g.world = 4;
        g.rank = rank;
        GlmRouter::Config cfg;
        GlmRouter::Wire w;
        w.input = decl_b(&b, "hidden", RAD_BF16, { 128, 4096 });
        w.logits = decl_b(&b, "logits", RAD_F32, { 32, 288 });
        w.ids = decl_b(&b, "ids", RAD_I32, { 32, 8 });
        w.weights = decl_b(&b, "weights", RAD_F32, { 32, 8 });
        GlmRouter::Src src{ "model.layers.3.mlp.gate.weight",
                            "model.layers.3.mlp.gate.e_score_correction_bias" };
        GlmRouter router;
        REQUIRE_EQ(router.declare(&b, nm, g, 3, "blk.3.router", cfg, w, src), RAD_OK);
        REQUIRE_EQ(b.weights.size(), (size_t)2);
        for (const auto& weight : b.weights) {
            CHECK_EQ(weight.d.dtype, RAD_F32);
            CHECK_EQ(weight.d.shard, RAD_SHARD_NONE);
        }
        CHECK_EQ(b.weights[0].d.shape[0], 288);
        CHECK_EQ(b.weights[0].d.shape[1], 4096);
        CHECK_EQ(b.ops[router.op_project - 1].op, std::string("gemm_nt_f32"));
        CHECK_EQ(b.ops[router.op_select - 1].op, std::string("router_sigmoid_topk"));
        RadCtx ctx;
        router.step(&ctx, 7, 4);
        REQUIRE_EQ(ctx.issues.size(), (size_t)2);
        CHECK_EQ(ctx.issues[0].opd[0].handle, w.input);
        CHECK_EQ(ctx.issues[0].opd[0].offset, 7 * 4096);
        CHECK_EQ(ctx.issues[0].opd[0].rows, 4);
        CHECK_EQ(ctx.issues[1].opd[1].handle, router.correction);
        CHECK_EQ(ctx.issues[1].opd[2].handle, w.ids);
        CHECK_EQ(ctx.issues[1].opd[3].handle, w.weights);
    }
}

TEST(glm_router_and_index_head_split_columns_over_ranks_from_the_row_threshold) {
    using namespace rad::arch;
    for (int world : { 1, 2, 4, 8 })
        for (int rank = 0; rank < world; ++rank) {
            RadBuilder b;
            Names nm("");
            Geom g{};
            g.n_embd = 4096;
            g.max_tok = 512;
            g.world = world;
            g.rank = rank;
            GlmRouter::Config cfg;
            cfg.rows = 512;
            GlmRouter::Wire w;
            w.input = decl_b(&b, "hidden", RAD_BF16, { 512, 4096 });
            w.logits = decl_b(&b, "logits", RAD_F32, { 512, 288 });
            w.ids = decl_b(&b, "ids", RAD_I32, { 512, 8 });
            w.weights = decl_b(&b, "weights", RAD_F32, { 512, 8 });
            GlmRouter::Src src{ "model.layers.3.mlp.gate.weight",
                                "model.layers.3.mlp.gate.e_score_correction_bias" };
            GlmRouter router;
            REQUIRE_EQ(router.declare(&b, nm, g, 3, "blk.3.router", cfg, w, src), RAD_OK);
            const bool split = world > 1;
            const int64_t cols = 288 / world;
            CHECK_EQ(count_op(b, "all_gather"), split ? 1 : 0);
            CHECK_EQ(count_op(b, "gemm_nt_f32"), split ? 2 : 1);
            // The weight stays whole and replicated: decode keeps the unsplit projection.
            CHECK_EQ(b.weights[0].d.shape[0], 288);
            CHECK_EQ(b.weights[0].d.shard, RAD_SHARD_NONE);
            if (split) {
                const RecOp* part = find_op(b, "gemm_nt_f32", 1);
                const RecOp* gather = find_op(b, "all_gather");
                REQUIRE(part && gather);
                CHECK_EQ(pint(part, "N"), cols);
                CHECK_EQ(pint(part, "K"), 4096);
                CHECK(part->w == std::vector<rad_weight>({ router.projection }));
                CHECK(part->reads == std::vector<rad_buf>({ w.input }));
                CHECK(part->writes == std::vector<rad_buf>({ router.split.part }));
                CHECK_EQ(pint(gather, "world_size"), world);
                CHECK_EQ(pint(gather, "row"), cols);
                CHECK_EQ(param(gather, "numel")->kind, RAD_P_RANGE);
                CHECK_EQ(param(gather, "numel")->ival, cols);
                CHECK_EQ(param(gather, "numel")->ihi, 512 * cols);
                CHECK_EQ(pstr(gather, "dtype"), "f32");
                CHECK(gather->reads == std::vector<rad_buf>({ router.split.part }));
                CHECK(gather->writes == std::vector<rad_buf>({ w.logits }));
            }
            RadCtx ctx;
            router.step(&ctx, 7, 300, 7);
            if (split) {
                REQUIRE_EQ(ctx.issues.size(), (size_t)3);
                CHECK_EQ(ctx.issues[0].op, router.split.op_part);
                CHECK_EQ(ctx.issues[0].n, 300);
                CHECK_EQ(ctx.issues[0].opd[0].handle, w.input);
                CHECK_EQ(ctx.issues[0].opd[0].offset, 7 * 4096);
                CHECK_EQ(ctx.issues[0].opd[0].rows, 300);
                CHECK_EQ(ctx.issues[0].opd[1].kind, RAD_OPK_WEIGHT);
                CHECK_EQ(ctx.issues[0].opd[1].handle, router.projection);
                CHECK_EQ(ctx.issues[0].opd[1].offset, rank * cols * 4096);
                CHECK_EQ(ctx.issues[0].opd[1].rows, cols);
                CHECK_EQ(ctx.issues[0].opd[2].handle, router.split.part);
                CHECK_EQ(ctx.issues[0].opd[2].rows, 300);
                CHECK_EQ(ctx.issues[1].op, router.split.op_gather);
                CHECK_EQ(ctx.issues[1].n, 300 * cols);
                CHECK_EQ(ctx.issues[1].opd[0].handle, router.split.part);
                CHECK_EQ(ctx.issues[1].opd[1].handle, w.logits);
                CHECK_EQ(ctx.issues[1].opd[1].offset, 7 * 288);
                CHECK_EQ(ctx.issues[1].opd[1].rows, 300);
                CHECK_EQ(ctx.issues[2].op, router.op_select);
            } else {
                REQUIRE_EQ(ctx.issues.size(), (size_t)2);
                CHECK_EQ(ctx.issues[0].op, router.op_project);
            }
            ctx.issues.clear();
            router.step(&ctx, 0, 255);
            REQUIRE_EQ(ctx.issues.size(), (size_t)2);
            CHECK_EQ(ctx.issues[0].op, router.op_project);
        }
}

TEST(absorbed_mla_shards_complete_heads_and_reads_one_latent_cache) {
    using namespace rad::arch;
    for (int rank = 0; rank < 4; ++rank) {
        RadBuilder b;
        b.kv_block = 4;
        Names nm("");
        Geom g{};
        g.n_embd = 4096;
        g.max_tok = 32;
        g.world = 4;
        g.rank = rank;
        RadKVGroupDecl kd{};
        kd.kind = RAD_KV_FULL;
        kd.dtype = RAD_BF16;
        kd.n_head_kv = 1;
        kd.head_dim = 256;
        const rad_kvgroup kv = rad_decl_kv_group(&b, "latent", &kd);
        MlaLatent::Config cfg;
        MlaLatent::Wire w;
        w.q = decl_b(&b, "q", RAD_BF16, { 32, 16, 256 });
        w.q_latent = decl_b(&b, "ql", RAD_BF16, { 32, 16, 512 });
        w.attn_latent = decl_b(&b, "al", RAD_BF16, { 32, 16, 512 });
        w.out = decl_b(&b, "out", RAD_BF16, { 32, 16, 256 });
        w.selected = decl_b(&b, "selected", RAD_I32, { 32, 513 });
        w.length = decl_b(&b, "length", RAD_I32, { 32 });
        MlaLatent block;
        REQUIRE_EQ(
            block.declare(&b, nm, g, 3, kv, cfg, w, "model.layers.3.self_attn.kv_b_proj.weight"),
            RAD_OK);
        REQUIRE_EQ(b.weights.size(), (size_t)1);
        CHECK_EQ(b.weights[0].d.dtype, RAD_BF16);
        CHECK_EQ(b.weights[0].d.shard, RAD_SHARD_ROW);
        CHECK_EQ(b.weights[0].d.shape[0], 8192);
        CHECK_EQ(b.weights[0].d.shape[1], 512);
        RadCtx ctx;
        block.step(&ctx, 4);
        REQUIRE_EQ(ctx.issues.size(), (size_t)3);
        CHECK_EQ(b.ops[ctx.issues[0].op - 1].op, std::string("mla_project"));
        CHECK_EQ(b.ops[ctx.issues[1].op - 1].op, std::string("mla_attn"));
        CHECK_EQ(b.ops[ctx.issues[2].op - 1].op, std::string("mla_project"));
        CHECK_EQ(ctx.issues[1].opd[1].kind, RAD_OPK_KV);
        CHECK_EQ(ctx.issues[1].opd[2].handle, w.selected);
        CHECK_EQ(ctx.issues[1].opd[3].handle, w.length);
        CHECK_EQ(ctx.issues[2].opd[1].handle, block.projection);
    }
}

TEST(
    glm53_native_graph_declares_all_layers_preserves_mixed_formats_and_runs_mtp_only_in_draft_passes) {
    using namespace rad::arch;
    const char* keys[] = { "num_nextn_predict_layers", "hc_mult", "n_routed_experts",
                           "index_topk" };
    const char* vals[] = { "1", "4", "288", "2048" };
    RadModelMeta meta{};
    meta.n_layers = 45;
    meta.n_embd = 4096;
    meta.n_vocab = 154880;
    meta.rms_eps = 1e-5f;
    meta.n_kv = 4;
    meta.kv_key = keys;
    meta.kv_val = vals;
    const auto mx = rad_enc_affine(RAD_FP4E2M1, RAD_E8M0, 1, 32);
    const auto fp8 = rad_enc_affine(RAD_F8E4M3, RAD_F32, 128, 128);
    for (int world : { 1, 4, 8 })
        for (int rank = 0; rank < world; ++rank) {
            RadBuilder b;
            b.kv_block = 4;
            b.encs = { { "mtp.blk.45.ffn_gate_up_exps", fp8 },
                       { "mtp.blk.45.ffn_down_exps", fp8 },
                       { "ffn_gate_up_exps", mx },
                       { "ffn_down_exps", mx },
                       { "ffn_gate_up_shexp", mx },
                       { "ffn_down_shexp", mx },
                       { "ffn_gate_up.weight", fp8 },
                       { "ffn_down.weight", fp8 } };
            RadBuildCtx build{};
            build.rank = rank;
            build.world_size = world;
            build.max_tok = rank == 0 ? 512 : 16;
            build.max_seqs = 4;
            build.max_ctx = 4096;
            build.max_spec = 3;
            REQUIRE_EQ(glm5next::declare(&b, &meta, &build), RAD_OK);
            const auto& m = *glm5next::models[rank];
            CHECK(m.have_mtp);
            CHECK_EQ(b.drafter.kind, RAD_DRAFT_SERIAL);
            CHECK_EQ(b.drafter.depth, 3);
            CHECK_EQ(m.layers.size(), (size_t)45);
            int full = 0;
            for (int l = 0; l < 45; ++l) {
                const auto& layer = m.layers[l];
                full += layer.full;
                CHECK(layer.attn_hc.op_pre < layer.norm.op);
                CHECK(layer.attn_hc.op_post < layer.ffn_hc.op_pre);
                if (layer.full) {
                    CHECK(layer.index.cache.op_select < layer.mla.core.op_attn);
                    CHECK(layer.mla.output.op < layer.attn_hc.op_post);
                } else
                    CHECK(layer.kda.output.op < layer.attn_hc.op_post);
                if (l >= 3) {
                    CHECK_EQ(layer.moe.c.expert_mx, 1);
                    CHECK(layer.moe.op_gather < layer.ffn_hc.op_post);
                }
            }
            CHECK_EQ(full, 11);
            CHECK_EQ(m.draft.moe.c.expert_f32_scale, 1);
            for (const auto& map : b.maps)
                if (map.declared.starts_with("mtp."))
                    for (const auto& source : map.src) CHECK(!source.starts_with("mtp.model."));
            int32_t cu[] = { 0, 2 }, positions[] = { 0, 1 }, ctxlens[] = { 0 }, ids[] = { 7, 9 },
                    slots[] = { 0, 1 }, out[] = { 1 };
            int32_t sidx[] = { 0, 0, 0, 1, 1, 1 }, ci[] = { 0 }, tables[32]{};
            std::vector<RadKVGroupBatch> kv(b.kv_groups.size());
            for (size_t i = 0; i < kv.size(); ++i) {
                kv[i].group = (rad_kvgroup)i + 1;
                kv[i].slot_mapping = slots;
                kv[i].block_table = tables;
                kv[i].block_table_pitch = 32;
                kv[i].state_index = b.kv_groups[i].d.kind == RAD_KV_CONV ? ci : sidx;
                kv[i].state_index_pitch = b.kv_groups[i].d.kind == RAD_KV_CONV ? 1 : 6;
            }
            RadBatch batch{};
            batch.phase = RAD_PHASE_PREFILL;
            batch.n_tok = 2;
            batch.n_seq = 1;
            batch.token_ids = ids;
            batch.positions = positions;
            batch.cu_seqlens = cu;
            batch.ctx_lens = ctxlens;
            batch.kv = kv.data();
            batch.n_kv_groups = (int)kv.size();
            batch.n_out = 1;
            batch.out_ids = out;
            RadCtx ctx;
            ctx.b = &b;
            ctx.rank = rank;
            ctx.world = world;
            ctx.batch = &batch;
            glm5next::step(&ctx, &batch);
            CHECK(ctx.step_fail.empty());
            const auto ops = issued_ops(b, ctx);
            CHECK_EQ(std::count(ops.begin(), ops.end(), "mhc_pre"), 90);
            CHECK_EQ(std::count(ops.begin(), ops.end(), "kda_update"), 34);
            CHECK_EQ(std::count(ops.begin(), ops.end(), "mla_attn"), 11);
            // Two rows stay under the split threshold: 42 routers and 11 index heads, replicated.
            CHECK_EQ(std::count(ops.begin(), ops.end(), "gemm_nt_f32"), 53);
            CHECK_EQ(std::count(ops.begin(), ops.end(), "all_gather"), 0);
            // The split is declared on every rank whatever its step size; only rank 0 issues it here.
            const bool split = world > 1;
            // 43 routers and 12 index heads (MTP included), and the 90 mHC reads.
            CHECK_EQ(count_op(b, "all_gather"), split ? 55 + 90 : 0);
            CHECK_EQ(count_op(b, "mhc_pre_partial"), split ? 90 : 0);
            ctx.issues.clear();
            batch.draft_pass = -1;
            batch.n_draft_out = 1;
            batch.draft_out_ids = out;
            glm5next::step(&ctx, &batch);
            CHECK(ctx.step_fail.empty());
            const auto draft_ops = issued_ops(b, ctx);
            CHECK_EQ(std::count(draft_ops.begin(), draft_ops.end(), "mhc_pre"), 0);
            CHECK_EQ(std::count(draft_ops.begin(), draft_ops.end(), "mla_attn"), 1);
            CHECK_EQ(ctx.issues.back().op, m.mhead);
            CHECK(m.mselect < m.draft.moe.op_router);
            for (const auto& issue : ctx.issues) {
                if (issue.op == m.draft.moe.op_router || issue.op == m.mnorm_op)
                    CHECK_EQ(issue.opd[0].rows, 1);   // Only the row producing a proposal.
                if (issue.op == m.draft.residual)
                    CHECK_EQ(issue.opd[0].rows, 2);   // Full history still fills attention caches.
            }
            ctx.issues.clear();
            batch.n_draft_out = 0;
            glm5next::step(&ctx, &batch);
            const auto history = issued_ops(b, ctx);
            CHECK(std::find(history.begin(), history.end(), "kv_store") != history.end());
            CHECK(std::find(history.begin(), history.end(), "moe_gemm_q") == history.end());
            if (rank == 0) {
                std::vector<int32_t> long_ids(512, 7), long_positions(512), long_slots(512),
                    long_tables(1024);
                for (int i = 0; i < 512; ++i) long_positions[i] = long_slots[i] = i;
                int32_t long_cu[] = { 0, 512 }, last = 511;
                for (auto& group : kv) {
                    group.slot_mapping = long_slots.data();
                    group.block_table = long_tables.data();
                    group.block_table_pitch = 1024;
                }
                batch.n_tok = 512;
                batch.token_ids = long_ids.data();
                batch.positions = long_positions.data();
                batch.cu_seqlens = long_cu;
                batch.out_ids = &last;
                batch.draft_pass = 0;
                batch.max_q_len = 512;
                ctx.issues.clear();
                glm5next::step(&ctx, &batch);
                CHECK(ctx.step_fail.empty());
                {
                    // 512 rows reach the split: each projection runs on its column slice and each
                    // mHC read on its chunks, both gathered; at TP1 nothing changes.
                    const auto long_ops = issued_ops(b, ctx);
                    CHECK_EQ(std::count(long_ops.begin(), long_ops.end(), "gemm_nt_f32"), 53);
                    CHECK_EQ(std::count(long_ops.begin(), long_ops.end(), "all_gather"),
                             world > 1 ? 53 + 90 : 0);
                    CHECK_EQ(std::count(long_ops.begin(), long_ops.end(), "mhc_pre"),
                             world > 1 ? 0 : 90);
                    CHECK_EQ(std::count(long_ops.begin(), long_ops.end(), "mhc_pre_finish"),
                             world > 1 ? 90 : 0);
                    for (const auto& issue : ctx.issues) {
                        if (b.ops[issue.op - 1].op != "gemm_nt_f32") continue;
                        const int64_t n = pint(&b.ops[issue.op - 1], "N");
                        if (world == 1)
                            CHECK(n == 288 || n == 32);
                        else {
                            CHECK(n == 288 / world || n == 32 / world);
                            CHECK_EQ(issue.opd[1].rows, n);
                        }
                    }
                }
                for (int l = 3; l < 45; ++l) {
                    // A prefill routes its full batch once, reusing each expert's
                    // weights across its rows instead of re-reading them in chunks.
                    CHECK_EQ(std::count_if(ctx.issues.begin(), ctx.issues.end(),
                                           [&](const auto& issue) {
                                               const auto& moe = m.layers[l].moe;
                                               return issue.op == moe.op_router ||
                                                      issue.op == moe.glm_router.split.op_part;
                                           }),
                             1);
                }
                // Arena-lending probes must not replace the model the serving
                // graph's handles refer to, even when they declare fewer rows.
                const auto* live = glm5next::models[rank].get();
                RadBuilder pb;
                pb.kv_block = 4;
                pb.encs = b.encs;
                auto small = build;
                small.shape_probe = 1;
                small.max_tok = 1;
                REQUIRE_EQ(glm5next::declare(&pb, &meta, &small), RAD_OK);
                CHECK_EQ(glm5next::models[rank].get(), live);
                CHECK_EQ(glm5next::models[rank]->g.max_tok, 512);
                RadArchProbe capabilities{};
                REQUIRE_EQ(glm5next::probe(&meta, &capabilities), RAD_OK);
                CHECK_EQ(capabilities.shape_probe_ok, 1);
                std::array<int, 8> results{};
                std::vector<std::thread> probes;
                for (int i = 0; i < 8; ++i)
                    probes.emplace_back([&, i] {
                        RadBuilder parallel;
                        parallel.kv_block = 4;
                        parallel.encs = b.encs;
                        auto level = build;
                        level.rank = i % 4;
                        level.max_tok = i < 4 ? 256 : 1024;
                        level.shape_probe = 1;
                        results[i] = glm5next::declare(&parallel, &meta, &level);
                    });
                for (auto& thread : probes) thread.join();
                for (int status : results) CHECK_EQ(status, RAD_OK);
                CHECK_EQ(glm5next::models[rank].get(), live);
            }
        }
}

/* The engine's arena levels declare again at a smaller max_tok (RadBuildCtx::shape_probe) and
 * must get the real declare's graph: the same ops with the same fixed parameters, only ranged row
 * counts and buffer rows smaller. The TP split declares its ops whatever the step size and its
 * gathers' message as a range. */
TEST(glm53_sizing_declares_match_the_real_graph_at_every_arena_level) {
    const char* keys[] = { "num_nextn_predict_layers", "hc_mult", "n_routed_experts",
                           "index_topk" };
    const char* vals[] = { "1", "4", "288", "2048" };
    RadModelMeta meta{};
    meta.n_layers = 45;
    meta.n_embd = 4096;
    meta.n_vocab = 154880;
    meta.rms_eps = 1e-5f;
    meta.n_kv = 4;
    meta.kv_key = keys;
    meta.kv_val = vals;
    const auto mx = rad_enc_affine(RAD_FP4E2M1, RAD_E8M0, 1, 32);
    const auto fp8 = rad_enc_affine(RAD_F8E4M3, RAD_F32, 128, 128);
    const int world = 8, rank = 3;
    for (int64_t R : { 256, 3072 }) {
        auto encs = [&](RadBuilder& b) {
            b.kv_block = 4;
            b.encs = { { "mtp.blk.45.ffn_gate_up_exps", fp8 },
                       { "mtp.blk.45.ffn_down_exps", fp8 },
                       { "ffn_gate_up_exps", mx },
                       { "ffn_down_exps", mx },
                       { "ffn_gate_up_shexp", mx },
                       { "ffn_down_shexp", mx },
                       { "ffn_gate_up.weight", fp8 },
                       { "ffn_down.weight", fp8 } };
        };
        RadBuildCtx c{};
        c.rank = rank;
        c.world_size = world;
        c.max_tok = 4096;
        c.max_seqs = 8;
        c.max_ctx = 32768;
        c.max_spec = 3;
        RadBuilder full;
        encs(full);
        REQUIRE_EQ(glm5next::declare(&full, &meta, &c), RAD_OK);
        RadBuildCtx p = c;
        p.max_tok = R;
        p.shape_probe = 1;
        RadBuilder small;
        encs(small);
        REQUIRE_EQ(glm5next::declare(&small, &meta, &p), RAD_OK);
        REQUIRE_EQ(small.ops.size(), full.ops.size());
        int differ = 0;
        for (size_t i = 0; i < full.ops.size(); ++i) {
            if (small.ops[i].op != full.ops[i].op) {
                ++differ;
                continue;
            }
            REQUIRE_EQ(small.ops[i].p.size(), full.ops[i].p.size());
            for (size_t k = 0; k < full.ops[i].p.size(); ++k) {
                const RecParam& a = full.ops[i].p[k];
                const RecParam& q = small.ops[i].p[k];
                if (a.kind == RAD_P_RANGE) continue;
                // A capacity may shrink: its kernel writes by its operands' extents.
                if (full.ops[i].op == "qsa_work" && a.key == "work" && q.ival <= a.ival)
                    continue;
                if (a.ival != q.ival || a.sval != q.sval || a.dval != q.dval) {
                    fprintf(stderr,
                            "    R %lld: op %zu '%s' param '%s': "
                            "%lld real, %lld small\n",
                            (long long)R, i, full.ops[i].op.c_str(), a.key.c_str(),
                            a.ival, q.ival);
                    ++differ;
                }
            }
        }
        CHECK_EQ(differ, 0);
        REQUIRE_EQ(small.bufs.size(), full.bufs.size());
        for (size_t i = 0; i < full.bufs.size(); ++i)
            for (uint32_t d = 1; d < full.bufs[i].d.rank; ++d)
                CHECK_EQ(small.bufs[i].d.shape[d], full.bufs[i].d.shape[d]);
    }
}

/* The MoE block's decode fusions: over a two-row trunk step, its draft pass and a 512-row prefill
 * (past the router fusion's one-workgroup sort), each fused launch stands where it fits and the
 * launches it replaces are gone, with the step lint clean throughout. The sort writes each slot's
 * row (`moe_scatter_inv`) and the gather reads it (`moe_gather_inv`) in every MoE pass. */
TEST(glm53_moe_decode_fusions_replace_their_launches_where_they_fit) {
    const char* keys[] = { "num_nextn_predict_layers", "hc_mult", "n_routed_experts",
                           "index_topk" };
    const char* vals[] = { "1", "4", "288", "2048" };
    RadModelMeta meta{};
    meta.n_layers = 45;
    meta.n_embd = 4096;
    meta.n_vocab = 154880;
    meta.rms_eps = 1e-5f;
    meta.n_kv = 4;
    meta.kv_key = keys;
    meta.kv_val = vals;
    const auto mx = rad_enc_affine(RAD_FP4E2M1, RAD_E8M0, 1, 32);
    const auto fp8 = rad_enc_affine(RAD_F8E4M3, RAD_F32, 128, 128);
    using Counts = std::map<std::string, int>;
    /* Launches per step: trunk, draft pass, prefill. The trunk has 42 MoE layers, each a routed
     * and a shared clamped SwiGLU, and three dense ones. */
    struct Want {
        const char* op;
        int trunk, draft, prefill;
    };
    const Want wants[] = {
        { "router_sigmoid_topk_scatter", 42, 1, 0 }, { "router_sigmoid_topk", 0, 0, 42 },
        { "swiglu_clamped_quant_fp8", 87, 2, 87 },   { "swiglu_clamped", 0, 0, 0 },
        { "moe_scatter", 0, 0, 0 },                  { "moe_gather", 0, 0, 0 },
    };
    for (int world : { 1, 8 }) {
        RadBuilder b;
        b.kv_block = 4;
        b.encs = { { "mtp.blk.45.ffn_gate_up_exps", fp8 },
                   { "mtp.blk.45.ffn_down_exps", fp8 },
                   { "ffn_gate_up_exps", mx },
                   { "ffn_down_exps", mx },
                   { "ffn_gate_up_shexp", mx },
                   { "ffn_down_shexp", mx },
                   { "ffn_gate_up.weight", fp8 },
                   { "ffn_down.weight", fp8 } };
        RadBuildCtx build{};
        build.rank = 0;
        build.world_size = world;
        build.max_tok = 512;
        build.max_seqs = 4;
        build.max_ctx = 4096;
        build.max_spec = 3;
        REQUIRE_EQ(glm5next::declare(&b, &meta, &build), RAD_OK);
        int32_t cu[] = { 0, 2 }, positions[] = { 0, 1 }, ctxlens[] = { 0 }, ids[] = { 7, 9 },
                slots[] = { 0, 1 }, out[] = { 1 };
        int32_t sidx[] = { 0, 0, 0, 1, 1, 1 }, ci[] = { 0 }, tables[32]{};
        std::vector<RadKVGroupBatch> kv(b.kv_groups.size());
        for (size_t i = 0; i < kv.size(); ++i) {
            kv[i].group = (rad_kvgroup)i + 1;
            kv[i].slot_mapping = slots;
            kv[i].block_table = tables;
            kv[i].block_table_pitch = 32;
            kv[i].state_index = b.kv_groups[i].d.kind == RAD_KV_CONV ? ci : sidx;
            kv[i].state_index_pitch = b.kv_groups[i].d.kind == RAD_KV_CONV ? 1 : 6;
        }
        RadBatch batch{};
        batch.phase = RAD_PHASE_PREFILL;
        batch.n_tok = 2;
        batch.n_seq = 1;
        batch.token_ids = ids;
        batch.positions = positions;
        batch.cu_seqlens = cu;
        batch.ctx_lens = ctxlens;
        batch.kv = kv.data();
        batch.n_kv_groups = (int)kv.size();
        batch.n_out = 1;
        batch.out_ids = out;
        RadCtx ctx;
        ctx.b = &b;
        ctx.rank = 0;
        ctx.world = world;
        ctx.batch = &batch;
        Counts got[3];
        /* With the inverse, the fused selection writes it as its ninth operand and the gather
         * reads it as its third. */
        const rad_buf inv = find_b(b, "expert_inverse")
                                ? (rad_buf)(find_b(b, "expert_inverse") - &b.bufs[0] + 1)
                                : 0;
        REQUIRE(inv != 0);
        auto run = [&](int k) {
            ctx.issues.clear();
            glm5next::step(&ctx, &batch);
            CHECK(ctx.step_fail.empty());
            CHECK_LINT(b, &ctx);
            for (const auto& op : issued_ops(b, ctx)) ++got[k][op];
            for (const auto& is : ctx.issues) {
                const RecOp* o = b.op_of(is.op);
                if (!o) continue;
                if (o->op == "router_sigmoid_topk_scatter")
                    CHECK(is.opd.size() > 8 && is.opd[8].kind == RAD_OPK_BUF &&
                          is.opd[8].handle == inv);
                if (o->op == "moe_gather_inv")
                    CHECK(is.opd.size() > 2 && is.opd[2].kind == RAD_OPK_BUF &&
                          is.opd[2].handle == inv);
            }
        };
        run(0);
        batch.draft_pass = -1;
        batch.n_draft_out = 1;
        batch.draft_out_ids = out;
        run(1);
        std::vector<int32_t> long_ids(512, 7), long_positions(512), long_slots(512),
            long_tables(1024);
        for (int i = 0; i < 512; ++i) long_positions[i] = long_slots[i] = i;
        int32_t long_cu[] = { 0, 512 }, last = 511;
        for (auto& group : kv) {
            group.slot_mapping = long_slots.data();
            group.block_table = long_tables.data();
            group.block_table_pitch = 1024;
        }
        batch.n_tok = 512;
        batch.token_ids = long_ids.data();
        batch.positions = long_positions.data();
        batch.cu_seqlens = long_cu;
        batch.out_ids = &last;
        batch.draft_pass = 0;
        batch.n_draft_out = 0;
        batch.max_q_len = 512;
        run(2);
        for (int k = 0; k < 3; ++k)
            for (const Want& w : wants) {
                const int want = k == 0 ? w.trunk : k == 1 ? w.draft : w.prefill;
                if (got[k][w.op] != want)
                    fprintf(stderr, "    world %d step %d: %s %d (want %d)\n", world, k, w.op,
                            got[k][w.op], want);
                CHECK_EQ(got[k][w.op], want);
            }
    }
}

TEST(dsa_cache_uses_distinct_index_and_attention_pages_and_a_speculative_tail_ring) {
    using namespace rad::arch;
    for (bool front : { false, true }) {
        RadBuilder b;
        b.kv_block = 4;
        Names nm("");
        Geom g{};
        g.n_embd = 4096;
        g.max_tok = 32;
        g.max_seqs = 4;
        g.max_spec = 3;
        RadKVGroupDecl ad{};
        ad.kind = RAD_KV_FULL;
        ad.dtype = RAD_BF16;
        ad.n_head_kv = 1;
        ad.head_dim = 256;
        const auto attn = rad_decl_kv_group(&b, "attn", &ad);
        // Claim the attention group's consumer before declaring the index group.
        RAD_OP(&b, "mla_attn", RAD_PARAMS(RAD_INT("block_size", 4)), RAD_NOWEIGHTS);
        RadKVGroupDecl td{};
        td.kind = RAD_KV_LINEAR;
        td.dtype = RAD_BF16;
        td.n_head_kv = 1;
        td.state_dim[0] = 7;
        td.state_dim[1] = 256;
        const auto tail = rad_decl_kv_group(&b, "tail", &td);
        DsaCache::Wire w;
        w.query = decl_b(&b, "query", RAD_BF16, { 1024, 128 });
        w.codes = decl_b(&b, "codes", RAD_F8E4M3, { 1024, 128 });
        w.scales = decl_b(&b, "scales", RAD_F32, { 1024, 1 });
        w.gate = decl_b(&b, "gates", RAD_F32, { 32, 32 });
        w.kg = decl_b(&b, "kg", RAD_BF16, { 32, 256 });
        w.stage = decl_b(&b, "stage", RAD_BF16, { 48, 256 });
        w.page = decl_b(&b, "page", RAD_I32, { 12 });
        w.bpos = decl_b(&b, "bpos", RAD_I32, { 12 });
        w.nc = decl_b(&b, "nc", RAD_I32, { 4 });
        w.pooled = decl_b(&b, "pooled", RAD_F8E4M3, { 12, 128 });
        w.pool_scale = decl_b(&b, "pool_scale", RAD_F32, { 12, 1 });
        w.score = decl_b(&b, "score", RAD_F32, { 32, 1024 });
        w.selected = decl_b(&b, "sel", RAD_I32, { 32, 513 });
        w.nsel = decl_b(&b, "nsel", RAD_I32, { 32 });
        w.length = decl_b(&b, "len", RAD_I32, { 32 });
        DsaCache block;
        DsaCache::Config cfg;
        cfg.blocks = 1024;
        cfg.work = 12;
        cfg.ring = 7;
        rad_kvgroup index = 0;
        GlmIndexer indexer;
        if (front) {
            GlmIndexer::Wire iw;
            iw.cache = w;
            iw.input = decl_b(&b, "input", RAD_BF16, { 32, 4096 });
            iw.q_low = decl_b(&b, "q_low", RAD_BF16, { 32, 1536 });
            iw.q_wide = decl_b(&b, "q_wide", RAD_BF16, { 32, 4096 });
            iw.kg_raw = decl_b(&b, "kg_raw", RAD_BF16, { 32, 256 });
            REQUIRE_EQ(indexer.declare(&b, nm, g, 3, iw, "indexer", cfg, attn, tail, &index),
                       RAD_OK);
            block = indexer.cache;
            CHECK_EQ(b.weights[indexer.head_weight - 1].d.dtype, RAD_F32);
            CHECK_EQ(b.weights[indexer.query.w - 1].d.shard, RAD_SHARD_NONE);
            CHECK_EQ(b.weights[indexer.key_gate.w - 1].d.shard, RAD_SHARD_NONE);
        } else
            REQUIRE_EQ(block.declare(&b, nm, g, 3, cfg, w, attn, tail, &index, "indexer.pool.ape"),
                       RAD_OK);
        CHECK_EQ(b.kv_groups[index - 1].d.dtype, RAD_U8);
        CHECK_EQ(b.kv_groups[index - 1].d.head_dim, 17);
        int32_t ibt[16]{}, abt[16]{}, sidx[12]{}, cu[] = { 0, 3, 8 }, pos[8]{};
        RadKVGroupBatch kv[3]{};
        kv[0].group = index;
        kv[0].block_table = ibt;
        kv[0].block_table_pitch = 8;
        kv[1].group = tail;
        kv[1].state_index = sidx;
        kv[1].state_index_pitch = 6;
        kv[2].group = attn;
        kv[2].block_table = abt;
        kv[2].block_table_pitch = 8;
        RadBatch batch{};
        batch.n_tok = 8;
        batch.n_seq = 2;
        batch.n_spec = 2;
        batch.kv = kv;
        batch.n_kv_groups = 3;
        batch.positions = pos;
        batch.cu_seqlens = cu;
        batch.max_ctx_len = 64;   // Mixed-row bound exceeds the eight physical table columns.
        RadCtx ctx;
        ctx.b = &b;
        ctx.batch = &batch;
        if (front) {
            indexer.step(&ctx, &batch);
            REQUIRE_EQ(ctx.issues.size(), (size_t)12);
            CHECK_EQ(ctx.issues[1].op, indexer.op_query_copy);
            CHECK_EQ(ctx.issues[1].opd[1].rows, 256);
            CHECK_EQ(ctx.issues[3].op, indexer.op_norm);
            CHECK_EQ(ctx.issues[4].opd[0].offset, 128);
            CHECK_EQ(ctx.issues[5].op, indexer.op_head);
            ctx.issues.erase(ctx.issues.begin(), ctx.issues.begin() + 6);
        } else
            block.step(&ctx, &batch);
        REQUIRE_EQ(ctx.issues.size(), (size_t)6);
        CHECK_EQ(ctx.issues[0].n, 256);
        CHECK(ctx.issues[1].opd[5].raw == ibt);
        CHECK_EQ(ctx.issues[1].opd[4].cols, 6);
        CHECK_EQ(ctx.issues[2].op, block.op_pool);
        CHECK_EQ(ctx.issues[2].opd[5].handle, index);
        CHECK_EQ(ctx.issues[3].op, block.op_tail);
        CHECK_EQ(ctx.issues[4].opd[0].rows, 256);
        CHECK_EQ(ctx.issues[4].opd[3].handle, index);
        CHECK_EQ(ctx.issues[4].opd[4].kind, RAD_OPK_NONE);
        CHECK(ctx.issues[4].opd[5].raw == ibt);
        CHECK_EQ(ctx.issues[4].opd[8].cols, 8);
        CHECK_EQ(ctx.issues[5].opd[0].cols, 8);
        CHECK(ctx.issues[5].opd[1].raw == abt);
    }
}

TEST(glm_dense_ffn_keeps_clipping_and_disables_the_unclipped_fp8_epilogue) {
    using namespace rad::arch;
    RadBuilder b;
    Names nm("");
    Geom g{};
    g.n_embd = 4096;
    g.n_ff = 3072;
    g.max_tok = 32;
    g.world = 4;
    const auto fp8 = rad_enc_affine(RAD_F8E4M3, RAD_F32, 128, 128);
    b.encs = { { "ffn_gate_up", fp8 }, { "ffn_down", fp8 } };
    MlpFP8::Wire w;
    w.x = decl_b(&b, "x", RAD_BF16, { 32, 4096 });
    w.h.x = decl_b(&b, "h", RAD_BF16, { 32, 4096 });
    REQUIRE_EQ(w.h.declare_qs(&b, nm, g, "h", 4096), RAD_OK);
    w.gate_up = decl_b(&b, "gate_up", RAD_BF16, { 32, 6144 });
    w.ffn.x = decl_b(&b, "ffn", RAD_BF16, { 32, 3072 });
    REQUIRE_EQ(w.ffn.declare_qs(&b, nm, g, "ffn", 3072), RAD_OK);
    MlpFP8 block;
    MlpFP8::Src src{ "post_attention_layernorm.weight", "mlp.gate_proj.weight",
                     "mlp.up_proj.weight", "mlp.down_proj.weight" };
    REQUIRE_EQ(block.declare(&b, nm, g, 0, w, src, false, false, kArNone, false, 10), RAD_OK);
    CHECK_EQ(block.op_gated, 0u);
    CHECK_EQ(b.ops[block.gq_ffn.op - 1].op, std::string("swiglu_clamped"));
    RadCtx ctx;
    ctx.b = &b;
    block.slice(&ctx, 8, 4, 4);
    const auto ops = issued_ops(b, ctx);
    CHECK_EQ(std::count(ops.begin(), ops.end(), "swiglu_clamped"), 1);
    CHECK(std::find(ops.begin(), ops.end(), "gemm_nt_q_gated") == ops.end());
    CHECK(std::find(ops.begin(), ops.end(), "quant_act_fp8") != ops.end());
}

TEST(glm_mla_stores_one_normalized_latent_and_exposes_q_lora_before_sparse_attention) {
    using namespace rad::arch;
    for (int rank = 0; rank < 4; ++rank) {
        RadBuilder b;
        b.kv_block = 4;
        Names nm("");
        Geom g{};
        g.n_embd = 4096;
        g.max_tok = 32;
        g.world = 4;
        g.rank = rank;
        RadKVGroupDecl kd{};
        kd.kind = RAD_KV_FULL;
        kd.dtype = RAD_BF16;
        kd.n_head_kv = 1;
        kd.head_dim = 256;
        const auto kv = rad_decl_kv_group(&b, "latent", &kd);
        auto buf = [&](const char* name, int64_t n) {
            return decl_b(&b, name, RAD_BF16, { 32, n });
        };
        GlmMla::Wire w;
        w.input.x = buf("input", 4096);
        w.q_raw = buf("q_raw", 1536);
        w.kv_raw = buf("kv_raw", 512);
        w.q_low.x = buf("q_low", 1536);
        w.latent = buf("latent_out", 512);
        w.out = buf("out", 4096);
        REQUIRE_EQ(w.input.declare_qs(&b, nm, g, "input", 4096), RAD_OK);
        REQUIRE_EQ(w.q_low.declare_qs(&b, nm, g, "q_low", 1536), RAD_OK);
        w.core.q = buf("q", 4096);
        w.core.q_latent = decl_b(&b, "q_latent", RAD_BF16, { 32, 16, 512 });
        w.core.attn_latent = decl_b(&b, "attn_latent", RAD_BF16, { 32, 16, 512 });
        w.core.out = buf("expanded", 4096);
        w.attn.x = w.core.out;
        REQUIRE_EQ(w.attn.declare_qs(&b, nm, g, "expanded", 4096), RAD_OK);
        w.core.selected = decl_b(&b, "selected", RAD_I32, { 32, 513 });
        w.core.length = decl_b(&b, "length", RAD_I32, { 32 });
        GlmMla block;
        REQUIRE_EQ(block.declare(&b, nm, g, 3, w, "attn", kv), RAD_OK);
        CHECK_EQ(b.weights[block.q_a.w - 1].d.shard, RAD_SHARD_NONE);
        CHECK_EQ(b.weights[block.q_b.w - 1].d.shard, RAD_SHARD_ROW);
        CHECK_EQ(b.weights[block.output.w - 1].d.shard, RAD_SHARD_COL);
        int32_t slots[4]{};
        RadKVGroupBatch kb{};
        kb.group = kv;
        kb.slot_mapping = slots;
        RadBatch batch{};
        batch.n_tok = 4;
        batch.kv = &kb;
        batch.n_kv_groups = 1;
        RadCtx ctx;
        ctx.b = &b;
        ctx.batch = &batch;
        block.prepare(&ctx, &batch);
        auto find_issue = [&](rad_op op) -> const RecIssue* {
            for (const auto& issue : ctx.issues)
                if (issue.op == op) return &issue;
            return nullptr;
        };
        const auto* store = find_issue(block.op_store);
        REQUIRE(store);
        CHECK_EQ(store->opd[0].handle, w.latent);
        CHECK_EQ(store->opd[0].cols, 256);
        CHECK_EQ(store->opd[1].offset, 256);
        CHECK_EQ(store->opd[1].cols, 256);
        CHECK(store->opd[2].raw == slots);
        const auto before = issued_ops(b, ctx);
        CHECK(std::find(before.begin(), before.end(), "mla_attn") == before.end());
        block.finish(&ctx, &batch);
        const auto* attn = find_issue(block.core.op_attn);
        REQUIRE(attn);
        CHECK_EQ(attn->opd[2].handle, w.core.selected);
        CHECK_EQ(ctx.issues.back().op, block.op_ar);
    }
}

TEST(glm_kda_projects_local_heads_and_reduces_only_after_the_output_projection) {
    using namespace rad::arch;
    for (int rank = 0; rank < 4; ++rank) {
        RadBuilder b;
        Names nm("");
        Geom g{};
        g.n_embd = 4096;
        g.max_tok = 32;
        g.world = 4;
        g.rank = rank;
        g.max_spec = 3;
        RadKVGroupDecl sd{};
        sd.kind = RAD_KV_LINEAR;
        sd.dtype = RAD_F32;
        sd.n_head_kv = 16;
        sd.state_dim[0] = sd.state_dim[1] = 128;
        RadKVGroupDecl cd{};
        cd.kind = RAD_KV_CONV;
        cd.dtype = RAD_BF16;
        cd.n_head_kv = 1;
        cd.head_dim = 6144;
        cd.conv_width = 4;
        const auto state = rad_decl_kv_group(&b, "state", &sd),
                   history = rad_decl_kv_group(&b, "conv", &cd);
        auto buf = [&](const char* name, int64_t width) {
            return decl_b(&b, name, RAD_BF16, { 32, width });
        };
        GlmKda::Wire w;
        w.input = buf("input", 4096);
        w.forget_low = buf("forget_low", 128);
        w.gate_low = buf("gate_low", 128);
        w.out = buf("out", 4096);
        w.core.qkv = buf("qkv", 6144);
        w.core.conv = buf("conv_out", 6144);
        w.core.a = buf("a", 2048);
        w.core.beta = buf("beta", 16);
        w.core.gate = buf("gate", 2048);
        w.core.out = buf("core_out", 2048);
        KdaState::Src src{ "attn.q_conv1d.weight", "attn.k_conv1d.weight", "attn.v_conv1d.weight",
                           "attn.A_log",           "attn.dt_bias",         "attn.o_norm.weight" };
        GlmKda block;
        REQUIRE_EQ(block.declare(&b, nm, g, 0, w, "attn", src, state, history), RAD_OK);
        CHECK_EQ(b.weights[block.qkv.w - 1].d.n_row_parts, 3);
        CHECK_EQ(b.weights[block.forget_a.w - 1].d.shard, RAD_SHARD_NONE);
        CHECK_EQ(b.weights[block.forget_b.w - 1].d.shard, RAD_SHARD_ROW);
        CHECK_EQ(b.weights[block.output.w - 1].d.shard, RAD_SHARD_COL);
        int32_t cu[] = { 0, 4 }, ctxlens[] = { 0 }, indices[] = { 0, 0, 0, 1, 1, 1 }, ci[] = { 0 };
        RadKVGroupBatch kv[2]{};
        kv[0].group = state;
        kv[0].state_index = indices;
        kv[0].state_index_pitch = 6;
        kv[1].group = history;
        kv[1].state_index = ci;
        RadBatch batch{};
        batch.phase = RAD_PHASE_PREFILL;
        batch.n_tok = 4;
        batch.n_seq = 1;
        batch.cu_seqlens = cu;
        batch.ctx_lens = ctxlens;
        batch.kv = kv;
        batch.n_kv_groups = 2;
        RadCtx ctx;
        ctx.b = &b;
        ctx.batch = &batch;
        block.step(&ctx, &batch);
        const auto ops = issued_ops(b, ctx);
        CHECK_EQ(std::count(ops.begin(), ops.end(), "gemm_nt"), 7);
        CHECK_EQ(std::count(ops.begin(), ops.end(), "silu"), 0);
        REQUIRE_EQ(ctx.issues.size(), (size_t)11);
        CHECK_EQ(ctx.issues[9].op, block.output.op);
        CHECK_EQ(ctx.issues[10].op, block.op_ar);
        CHECK_EQ(ctx.issues[10].opd[0].handle, w.out);
    }
}

TEST(kda_state_core_splits_mixed_batches_and_preserves_absolute_prefill_offsets) {
    using namespace rad::arch;
    RadBuilder b;
    Names nm("");
    Geom g{};
    g.n_embd = 4096;
    g.max_tok = 32;
    g.world = 4;
    g.max_spec = 3;
    RadKVGroupDecl sd{};
    sd.kind = RAD_KV_LINEAR;
    sd.dtype = RAD_F32;
    sd.n_head_kv = 16;
    sd.state_dim[0] = sd.state_dim[1] = 128;
    RadKVGroupDecl cd{};
    cd.kind = RAD_KV_CONV;
    cd.dtype = RAD_BF16;
    cd.n_head_kv = 1;
    cd.head_dim = 6144;
    cd.conv_width = 4;
    const rad_kvgroup st = rad_decl_kv_group(&b, "state", &sd),
                      cv = rad_decl_kv_group(&b, "conv", &cd);
    KdaState::Wire w;
    w.qkv = decl_b(&b, "qkv", RAD_BF16, { 32, 6144 });
    w.conv = decl_b(&b, "filtered", RAD_BF16, { 32, 6144 });
    w.a = decl_b(&b, "a", RAD_BF16, { 32, 2048 });
    w.beta = decl_b(&b, "beta", RAD_BF16, { 32, 16 });
    w.gate = decl_b(&b, "gate", RAD_BF16, { 32, 2048 });
    w.out = decl_b(&b, "out", RAD_BF16, { 32, 16, 128 });
    KdaState::Src src{ "attn.q_conv1d.weight", "attn.k_conv1d.weight", "attn.v_conv1d.weight",
                       "attn.A_log",           "attn.dt_bias",         "attn.o_norm.weight" };
    KdaState block;
    KdaState::Config cfg;
    REQUIRE_EQ(block.declare(&b, nm, g, 0, st, cv, cfg, w, src), RAD_OK);
    CHECK_EQ(b.weights[block.taps - 1].d.n_row_parts, 3);
    for (int p = 0; p < 3; ++p) CHECK_EQ(b.weights[block.taps - 1].d.row_parts[p], 2048);
    CHECK_EQ(b.weights[block.bias - 1].d.shape[0], 2048);
    for (rad_op op : { block.op_prefill, block.op_verify }) {
        bool found = false;
        for (const auto& p : b.ops[op - 1].p)
            if (p.key == "state_form") {
                CHECK_EQ(p.sval, std::string("anchor"));
                found = true;
            }
        CHECK(found);
    }
    int32_t cu[] = { 0, 3, 8 }, ctxlens[] = { 11, 0 }, accepted[] = { 2, 1 };
    int32_t indices[] = { 0, 0, 0, 2, 2, 2, 1, 1, 1, 3, 3, 3 }, conv_indices[] = { 0, 1 };
    RadKVGroupBatch kv[2]{};
    kv[0].group = st;
    kv[0].state_index = indices;
    kv[0].state_index_pitch = 6;
    kv[1].group = cv;
    kv[1].state_index = conv_indices;
    kv[1].state_index_pitch = 1;
    RadBatch batch{};
    batch.phase = RAD_PHASE_MIXED;
    batch.n_tok = 8;
    batch.n_seq = 2;
    batch.n_seq_decode = 1;
    batch.n_tok_decode = 3;
    batch.cu_seqlens = cu;
    batch.ctx_lens = ctxlens;
    batch.num_accepted = accepted;
    batch.kv = kv;
    batch.n_kv_groups = 2;
    RadCtx ctx;
    ctx.b = &b;
    ctx.batch = &batch;
    block.step(&ctx, &batch);
    REQUIRE_EQ(ctx.issues.size(), (size_t)5);
    CHECK_EQ(ctx.issues[0].op, block.op_conv);
    CHECK_EQ(ctx.issues[0].n, 3);
    CHECK_EQ(ctx.issues[1].op, block.op_verify);
    CHECK_EQ(ctx.issues[1].opd[0].cols, 2048);
    CHECK_EQ(ctx.issues[1].opd[1].offset, 2048);
    CHECK_EQ(ctx.issues[1].opd[2].offset, 4096);
    CHECK_EQ(ctx.issues[2].op, block.op_conv);
    CHECK_EQ(ctx.issues[2].n, 8);
    CHECK(ctx.issues[2].opd[3].raw == cu + 1);
    CHECK_EQ(ctx.issues[2].opd[7].kind, RAD_OPK_NONE);
    CHECK_EQ(ctx.issues[3].op, block.op_prefill);
    CHECK(ctx.issues[3].opd[8].raw == indices + 6);
    CHECK(ctx.issues[3].opd[9].raw == cu + 1);
    CHECK_EQ(ctx.issues[3].opd[0].offset, 0);
    CHECK_EQ(ctx.issues[3].opd[0].rows, 8);
    CHECK_EQ(ctx.issues[4].op, block.op_norm);
}

TEST(mxfp4_expert_declarations_preserve_tp4_and_tp8_scale_geometry_and_movement_groups) {
    using namespace rad::arch;
    const RadEncoding mx = rad_enc_affine(RAD_FP4E2M1, RAD_E8M0, 1, 32);
    REQUIRE(rad_enc_valid(&mx));
    CHECK_EQ(MoeFP8::format_of(mx), 8);
    for (int world : { 4, 8 })
        for (bool glm : { false, true })
            for (int rank = 0; rank < world; ++rank) {
                const int64_t F = 2048 / world;
                RadBuilder b;
                b.encs = { { "ffn_gate_up_exps", mx },
                           { "ffn_down_exps", mx },
                           { "ffn_gate_up_shexp", mx },
                           { "ffn_down_shexp", mx } };
                Names nm("");
                Geom g{};
                g.n_embd = 4096;
                g.max_tok = 32;
                g.world = world;
                g.rank = rank;
                MoeFP8::Src src{};
                src.router = "mlp.gate.weight";
                src.experts = "mlp.experts";
                src.cfg.n_expert = src.cfg.n_expert_all = 288;
                src.cfg.top_k = 8;
                src.cfg.n_ff_exp = F;
                src.cfg.rows = 32;
                if (glm) {
                    src.router_bias = "mlp.gate.e_score_correction_bias";
                    src.shared = "mlp.shared_experts";
                    src.cfg.sigmoid_router = 1;
                    src.cfg.route_scale = 2.5;
                    src.cfg.swiglu_limit = 10;
                    src.cfg.shared_ungated = 1;
                    src.cfg.n_ff_shared = F;
                }
                for (int q = 0; q < 2; ++q) {
                    src.cfg.ff_lo[q] = rank * F;
                    src.cfg.ff_hi[q] = (rank + 1) * F;
                }
                REQUIRE_EQ(MoeFP8::probe(&b, nm, 3, src, &src.cfg), RAD_OK);
                CHECK_EQ(src.cfg.expert_mx, 1);
                CHECK_EQ(src.cfg.expert_w4, 0);
                MoeFP8::Wire w{};
                auto buf = [&](const char* name, uint32_t dt, int64_t rows, int64_t cols) {
                    return decl_b(&b, name, dt, { rows, cols });
                };
                w.h.x = buf("h", RAD_BF16, 32, 4096);
                REQUIRE_EQ(w.h.declare_qs(&b, nm, g, "h", 4096), RAD_OK);
                w.logits = buf("logits", glm ? RAD_F32 : RAD_BF16, 32, 288);
                w.ids = buf("ids", RAD_I32, 32, 8);
                w.ew = buf("ew", glm ? RAD_F32 : RAD_BF16, 32, 8);
                w.eoff = decl_b(&b, "offset", RAD_I32, { 289 });
                w.ecnt = decl_b(&b, "count", RAD_I32, { 288 });
                w.sorted = decl_b(&b, "sorted", RAD_I32, { 256 });
                w.egu = buf("gate_up", RAD_BF16, 256, 2 * F);
                w.eff.x = buf("ff", RAD_BF16, 256, F);
                REQUIRE_EQ(w.eff.declare_qs(&b, nm, g, "ff", F, 256), RAD_OK);
                w.edn = buf("down", RAD_BF16, 256, 4096);
                if (glm) {
                    w.sgu = buf("shared_gu", RAD_BF16, 32, 2 * F);
                    w.sff.x = buf("shared_ff", RAD_BF16, 32, F);
                    REQUIRE_EQ(w.sff.declare_qs(&b, nm, g, "shared_ff", F), RAD_OK);
                    w.sout = buf("shared_out", RAD_BF16, 32, 4096);
                    w.sgate = buf("shared_one", RAD_BF16, 32, 1);
                }
                MoeFP8 moe;
                REQUIRE_EQ(moe.declare(&b, nm, g, 3, w, src, false, false), RAD_OK);
                CHECK_EQ(moe.ncls, 2);
                if (glm) {
                    CHECK_EQ(b.weights[moe.w_router - 1].d.dtype, RAD_F32);
                    CHECK_EQ(b.ops[moe.op_topk - 1].op, std::string("router_sigmoid_topk"));
                    CHECK_EQ(moe.op_topk_scatter, 0u);
                    CHECK_EQ(moe.op_router_fused, 0u);
                    CHECK_EQ(b.ops[moe.gq_exp.op - 1].op, std::string("swiglu_clamped"));
                    CHECK_EQ(b.ops[moe.gq_sh.op - 1].op, std::string("swiglu_clamped"));
                    CHECK_EQ(b.ops[moe.op_sgate - 1].op, std::string("fill"));
                    CHECK_EQ(moe.w_sgate, 0u);
                    bool plain_shared = false;
                    for (const auto& p : b.ops[moe.op_gather - 1].p)
                        if (p.key == "act") plain_shared = p.sval == "none";
                    CHECK(plain_shared);
                    RadCtx ctx;
                    ctx.b = &b;
                    ctx.rank = rank;
                    ctx.world = world;
                    moe.pass(&ctx, 8, 4, 4);
                    const auto ops = issued_ops(b, ctx);
                    CHECK(std::find(ops.begin(), ops.end(), "router_sigmoid_topk") != ops.end());
                    CHECK(std::find(ops.begin(), ops.end(), "router_topk") == ops.end());
                    CHECK_EQ(std::count(ops.begin(), ops.end(), "swiglu_clamped"), 2);
                    CHECK_EQ(std::count(ops.begin(), ops.end(), "fill"), 1);
                    for (const auto& issue : ctx.issues)
                        if (issue.op == moe.op_sgate) {
                            REQUIRE_EQ(issue.opd.size(), (size_t)1);
                            CHECK_EQ(issue.opd[0].handle, w.sgate);
                            CHECK_EQ(issue.opd[0].offset, 4);
                        }
                }
                for (int e = 0; e < 288; ++e) {
                    const auto& gu = b.weights[moe.w_gu[e] - 1];
                    const auto& gs = b.weights[moe.w_gus[e] - 1];
                    const auto& dn = b.weights[moe.w_dn[e] - 1];
                    const auto& ds = b.weights[moe.w_dns[e] - 1];
                    CHECK_EQ(gu.d.dtype, RAD_FP4E2M1);
                    CHECK_EQ(dn.d.dtype, RAD_FP4E2M1);
                    CHECK_EQ(gs.d.dtype, RAD_E8M0);
                    CHECK_EQ(ds.d.dtype, RAD_E8M0);
                    CHECK_EQ(gs.d.shape[0], 2 * F);
                    CHECK_EQ(gs.d.shape[1], 128);
                    CHECK_EQ(ds.d.shape[0], 4096);
                    CHECK_EQ(ds.d.shape[1], F / 32);
                    CHECK_EQ(gu.span_lo, rank * F);
                    CHECK_EQ(gs.span_lo, rank * F);
                    CHECK_EQ(dn.span_lo, rank * F);
                    CHECK_EQ(ds.span_lo, rank * (F / 32));
                    CHECK_EQ(gu.d.access, RAD_ACCESS_CONDITIONAL);
                    CHECK_EQ(dn.d.access, RAD_ACCESS_CONDITIONAL);
                }
                for (rad_op h : { moe.op_gu, moe.op_dn }) {
                    const RecOp* op = b.op_of(h);
                    REQUIRE(op != nullptr);
                    CHECK_EQ(op->op, std::string("moe_gemm_q"));
                    bool found = false;
                    for (const auto& p : op->p)
                        if (p.key == "dtype") {
                            CHECK_EQ(p.sval, std::string("mxfp4a8"));
                            found = true;
                        }
                    CHECK(found);
                }
            }
}

/* A quantised container may keep one linear as the checkpoint holds it. Flash-Next's recipe leaves
 * the QSA indexer's projection plain bf16 (its top-k is a hard choice between blocks), so in a
 * four-bit model with fp8 linears it is declared bf16 and read by `gemm_nt` over its input's bf16
 * plane, while the attention projections beside it stay block fp8. */
TEST(a_quantised_flash_next_reads_its_plain_indexer_as_bf16) {
    RadModelMeta meta = qwen4exp_meta();
    RadEncoding w4 = rad_enc_affine(RAD_I4, RAD_BF16, 1, 128);
    rad_enc_copy_str(w4.transform, "fwht128");
    const RadEncoding plain = rad_enc_plain(RAD_BF16);
    REQUIRE(rad_enc_valid(&w4) && rad_enc_valid(&plain));

    RadBuildCtx c = build_ctx(0, 1);
    c.max_tok = 2048; c.max_seqs = 8; c.max_spec = 3; c.max_ctx = 200000;
    RadBuilder b;
    b.encs = { { "ffn_gate_up_exps", w4 }, { "ffn_down_exps", w4 }, { "qsa_qk", plain } };
    REQUIRE_EQ(qwen4exp_fp8::declare(&b, &meta, &c), RAD_OK);
    int qk = 0, qk_bf16 = 0, attn_k_fp8 = 0;
    for (const RecWeight& w : b.weights) {
        if (w.name.find(".qsa_qk.weight") != std::string::npos) {
            ++qk;
            qk_bf16 += w.d.dtype == RAD_BF16 && !w.d.planes;
        }
        if (w.name.find(".attn_k.weight") != std::string::npos) attn_k_fp8 += w.d.dtype == RAD_F8E4M3;
    }
    CHECK(qk > 0);
    CHECK_EQ(qk_bf16, qk);
    CHECK(attn_k_fp8 > 0);
    /* every op that reads an indexer weight is the plain GEMM */
    int qk_ops = 0, qk_plain_ops = 0;
    for (const RecOp& o : b.ops)
        for (rad_weight wh : o.w)
            if (wh && b.weights[wh - 1].name.find(".qsa_qk.weight") != std::string::npos) {
                ++qk_ops;
                qk_plain_ops += o.op == "gemm_nt";
            }
    CHECK(qk_ops > 0);
    CHECK_EQ(qk_plain_ops, qk_ops);
}

/* A quantised layer may keep a few experts plain bf16. Layer 1 of a four-bit Flash-Next keeps
 * experts 3 and 10: that layer's sort places them last, its four-bit GEMMs serve the others in
 * slots 0..n-3 with no gap (a layer's experts are consecutive fixed-size units and routing counts
 * are in sort order), and the two are static bf16 weights behind a bf16 grouped GEMM pair, one of
 * which writes the same routed output as the four-bit down GEMM. Other layers are ordinary. */
TEST(a_four_bit_layer_serves_its_plain_experts_beside_the_rest) {
    RadModelMeta meta = qwen4exp_meta();
    RadEncoding w4 = rad_enc_affine(RAD_I4, RAD_BF16, 1, 128);
    rad_enc_copy_str(w4.transform, "fwht128");
    const RadEncoding plain = rad_enc_plain(RAD_BF16);
    REQUIRE(rad_enc_valid(&w4) && rad_enc_valid(&plain));

    RadBuildCtx c = build_ctx(0, 1);
    c.max_tok = 2048; c.max_seqs = 8; c.max_spec = 3; c.max_ctx = 200000;
    RadBuilder b;
    b.encs = { { "blk.1.ffn_gate_up_exps.3.weight", plain }, { "blk.1.ffn_down_exps.3.weight", plain },
               { "blk.1.ffn_gate_up_exps.10.weight", plain }, { "blk.1.ffn_down_exps.10.weight", plain },
               { "ffn_gate_up_exps", w4 }, { "ffn_down_exps", w4 } };
    REQUIRE_EQ(qwen4exp_fp8::declare(&b, &meta, &c), RAD_OK);
    const qwen4exp_fp8::Model& m = qwen4exp_fp8::g_model[0];
    CHECK_EQ(m.moecfg.expert_w4, 1);
    const int64_t ne = m.moecfg.n_expert;

    auto param = [](const RecOp& o, const char* k) -> const RecParam* {
        for (const RecParam& p : o.p) if (p.key == k) return &p;
        return nullptr;
    };
    auto layer_of = [&](const RecOp& o) -> int {
        for (rad_weight wh : o.w)
            if (wh) {
                const std::string& n = b.weights[wh - 1].name;
                if (n.rfind("blk.", 0) == 0) return std::atoi(n.c_str() + 4);
            }
        return -1;
    };
    /* the sorts: only layer 1's names the two experts */
    int prot_sorts = 0, plain_sorts = 0;
    for (const RecOp& o : b.ops)
        if (o.op == "moe_scatter" || o.op == "router_topk_scatter") {
            const RecParam* p = param(o, "protect");
            if (p) { ++prot_sorts; CHECK_EQ(p->sval, std::string("3,10")); }
            else ++plain_sorts;
        }
    CHECK(prot_sorts >= 1);
    CHECK(plain_sorts > 0);
    /* the grouped GEMMs of layer 1 */
    int q_ops = 0, bf_ops = 0;
    for (const RecOp& o : b.ops) {
        if (layer_of(o) != 1) continue;
        if (o.op == "moe_gemm_q") {
            ++q_ops;
            CHECK_EQ(param(o, "n_expert")->ival, ne - 2);
            for (rad_weight wh : o.w) {
                const std::string& n = b.weights[wh - 1].name;
                CHECK(n.find("exps.3.") == std::string::npos && n.find("exps.10.") == std::string::npos);
            }
        }
        if (o.op == "moe_gemm") {
            ++bf_ops;
            CHECK_EQ(param(o, "n_expert")->ival, 2);
            CHECK_EQ(param(o, "dtype")->sval, std::string("bf16"));
            REQUIRE_EQ(o.w.size(), (size_t)2);
            const std::string& n0 = b.weights[o.w[0] - 1].name;
            const std::string& n1 = b.weights[o.w[1] - 1].name;
            CHECK(n0.find("exps.3.weight") != std::string::npos);
            CHECK(n1.find("exps.10.weight") != std::string::npos);
        }
    }
    CHECK_EQ(q_ops, 2);
    CHECK_EQ(bf_ops, 2);
    /* layer 1's four-bit experts fill slots 0..ne-3 with none missing; the plain ones are static
     * bf16 weights */
    std::vector<int> slot_seen((size_t)ne, 0);
    int plain_w = 0;
    for (const RecWeight& w : b.weights) {
        if (w.name.rfind("blk.1.ffn_", 0) != 0 || w.name.find("_exps.") == std::string::npos) continue;
        const bool prot_name = w.name.find("exps.3.") != std::string::npos ||
                               w.name.find("exps.10.") != std::string::npos;
        if (prot_name) {
            ++plain_w;
            CHECK_EQ(w.d.dtype, (uint32_t)RAD_BF16);
            CHECK_EQ(w.d.group.expert, -1);
            CHECK_EQ(w.d.access, RAD_ACCESS_PER_TOKEN);
        } else if (w.d.group.expert >= 0 && w.d.group.expert < ne) {
            slot_seen[(size_t)w.d.group.expert] = 1;
        }
    }
    CHECK_EQ(plain_w, 4);
    int filled = 0;
    for (int64_t s = 0; s < ne; ++s) filled += slot_seen[(size_t)s];
    CHECK_EQ(filled, (int)(ne - 2));
    CHECK_EQ(slot_seen[(size_t)(ne - 1)] + slot_seen[(size_t)(ne - 2)], 0);

    /* An odd plain count leaves the rest unsplittable into two equal parity tables: refused. */
    RadBuilder odd;
    odd.encs = { { "blk.1.ffn_gate_up_exps.3.weight", plain }, { "blk.1.ffn_down_exps.3.weight", plain },
                 { "ffn_gate_up_exps", w4 }, { "ffn_down_exps", w4 } };
    CHECK(qwen4exp_fp8::declare(&odd, &meta, &c) != RAD_OK);

    /* With two ranks each holding a slice of every expert, a plain expert is sliced evenly too
     * (gate_up 2 x 320 rows, down 320 columns) and its GEMMs are that wide. */
    {
        RadBuildCtx c2 = build_ctx(0, 2);
        c2.max_tok = 2048; c2.max_seqs = 8; c2.max_spec = 3; c2.max_ctx = 200000;
        RadBuilder sb;
        sb.encs = b.encs;
        REQUIRE_EQ(qwen4exp_fp8::declare(&sb, &meta, &c2), RAD_OK);
        int sliced = 0;
        for (const RecWeight& w : sb.weights) {
            if (w.name != "blk.1.ffn_gate_up_exps.3.weight" && w.name != "blk.1.ffn_down_exps.3.weight")
                continue;
            ++sliced;
            if (w.name.find("gate_up") != std::string::npos) {
                CHECK_EQ(w.d.shape[0], 640);
                CHECK_EQ(w.d.shard, RAD_SHARD_ROW);
                CHECK_EQ(w.d.n_row_parts, 2);
            } else {
                CHECK_EQ(w.d.shape[1], 320);
                CHECK_EQ(w.d.shard, RAD_SHARD_COL);
            }
        }
        CHECK_EQ(sliced, 2);
        int widths_ok = 0;
        for (const RecOp& o : sb.ops)
            if (o.op == "moe_gemm" && param(o, "n_expert")->ival == 2)
                widths_ok += (param(o, "N")->ival == 640 && param(o, "K")->ival == 2560) ||
                             (param(o, "N")->ival == 2560 && param(o, "K")->ival == 320);
        CHECK_EQ(widths_ok, 2);
    }

    /* The expert the format is probed from may be a plain one; the probe reads on to the first
     * four-bit expert, so the model is four-bit, not bf16. */
    RadBuilder first;
    first.encs = { { "blk.0.ffn_gate_up_exps.0.weight", plain }, { "blk.0.ffn_down_exps.0.weight", plain },
                   { "blk.0.ffn_gate_up_exps.1.weight", plain }, { "blk.0.ffn_down_exps.1.weight", plain },
                   { "ffn_gate_up_exps", w4 }, { "ffn_down_exps", w4 } };
    REQUIRE_EQ(qwen4exp_fp8::declare(&first, &meta, &c), RAD_OK);
    CHECK_EQ(qwen4exp_fp8::g_model[0].moecfg.expert_w4, 1);
    CHECK_EQ(qwen4exp_fp8::g_model[0].moecfg.expert_bf16, 0);
}

TEST(a_sizing_declare_leaves_the_step_state_alone_and_shrinks_only_rows) {
    (void)&qwen4exp_fp8::step;     /* included for its declare */
    RadModelMeta meta = qwen4exp_meta();
    RadArchProbe pb{};
    CHECK_OK(qwen4exp_fp8::probe(&meta, &pb));
    CHECK(pb.shape_probe_ok);

    /* A 4096 budget under a span (RadBuildCtx::max_mixed_tok 2048: a step wider than 2048 is one
     * sequence) reads in pass-wide pieces and scores a pass-tall plane in windows, and at four
     * ranks pipelines its prefill (step_pipelined), declaring an all-reduce at `overlap` 1 a
     * block; the sizing declare, at a max_tok no wider than a pass, has to declare the same ones.
     * At two ranks the gated residual's fused write takes the all-reduce, so nothing pipelines.
     * Without the span a 4096 budget declares none of it. */
    struct Shape { int64_t tok, mixed; };
    for (const Shape sh : { Shape{ 2048, 0 }, Shape{ 4096, 0 }, Shape{ 4096, 2048 } })
    for (int world : { 1, 2, 4 }) {
        for (int rank = 0; rank < world; ++rank) {
            const int64_t real_tok = sh.tok;
            const bool mixed = sh.mixed > 0;
            RadBuildCtx c = build_ctx(rank, world);
            c.max_tok  = real_tok;
            c.max_mixed_tok = sh.mixed;
            c.max_seqs = 8;
            c.max_spec = 3;
            c.max_ctx  = 200000;
            RadBuilder full;
            REQUIRE_EQ(qwen4exp_fp8::declare(&full, &meta, &c), RAD_OK);
            const qwen4exp_fp8::Model& m = qwen4exp_fp8::g_model[rank];
            const int64_t max_tok = m.g.max_tok;
            const int64_t max_work = m.qsacfg.max_work;
            /* The score plane: a pass tall where a wider step is one sequence, max_tok rows
             * otherwise. */
            if (const RecBuf* sc = find_b(full, "qsa_score")) {
                const int64_t want = mixed ? 2048 : c.max_tok;
                CHECK_EQ(sc->d.shape[0], want);
                CHECK_EQ(m.qsacfg.plane_rows, want < c.max_tok ? want : 0);
            }
            /* And under a span every connection read is declared no wider than a pass: a wider
             * step reads in pieces, and the read's scratch follows its band. */
            {
                const int64_t want = (mixed && c.max_tok > 2048) ? 2048 : c.max_tok;
                int reads = 0, wide = 0;
                for (const auto& o : full.ops) {
                    if (o.op != "hc_read") continue;
                    ++reads;
                    const RecParam* pm = param(&o, "M");
                    if (!pm || pm->ihi != want) ++wide;
                }
                CHECK(reads > 0);
                CHECK_EQ(wide, 0);
            }
            /* The pipelined step's all-reduces at `overlap` 1, a block each, only where it runs,
             * with the buffer its second lane touches named concurrent. */
            {
                int lane = 0;
                for (const auto& o : full.ops)
                    if (o.op == "all_reduce" && pint(&o, "overlap") == 1) ++lane;
                const bool pipe = mixed && world > 2;
                CHECK_EQ(lane > 0, pipe);
                const RecBuf* x = find_b(full, "x");
                REQUIRE(x != nullptr);
                const rad_buf xh = (rad_buf)(x - full.bufs.data()) + 1;
                CHECK_EQ(std::count(full.concurrent.begin(), full.concurrent.end(), xh) > 0, pipe);
                CHECK_EQ(find_b(full, "pipe_cu") != nullptr, mixed);
            }
            const rad_op last_op = (rad_op)full.ops.size();
            CHECK_EQ(max_tok, real_tok);

            RadBuildCtx p = c;
            p.max_tok = 256;
            p.shape_probe = 1;
            RadBuilder small;
            REQUIRE_EQ(qwen4exp_fp8::declare(&small, &meta, &p), RAD_OK);

            /* The step's state is still the real declare's. */
            CHECK_EQ(m.g.max_tok, max_tok);
            CHECK_EQ(m.qsacfg.max_work, max_work);
            CHECK_EQ((rad_op)full.ops.size(), last_op);

            /* The same graph, and the same buffers with only their rows smaller. */
            REQUIRE_EQ(small.ops.size(), full.ops.size());
            int fixed_differ = 0;
            for (size_t i = 0; i < full.ops.size(); ++i) {
                CHECK_EQ(small.ops[i].op, full.ops[i].op);
                /* Only the ranged parameter may differ; a fixed one following max_tok would run
                 * the kernel at the real size against a level's buffers. */
                REQUIRE_EQ(small.ops[i].p.size(), full.ops[i].p.size());
                for (size_t k = 0; k < full.ops[i].p.size(); ++k) {
                    const RecParam& a = full.ops[i].p[k];
                    const RecParam& q = small.ops[i].p[k];
                    if (a.kind == RAD_P_RANGE) continue;
                    /* Capacities in libr4d's schema (RAD_PROLE_CAPACITY) may shrink: qsa_work
                     * launches as wide as its `page` operand regardless of `work`, and
                     * attn_paged_gate_quant by its block table's rows regardless of `max_seqs`. */
                    const bool cap = (full.ops[i].op == "qsa_work" && a.key == "work") ||
                                     (full.ops[i].op == "attn_paged_gate_quant" &&
                                      a.key == "max_seqs");
                    if (cap && q.ival <= a.ival) continue;
                    if (a.ival != q.ival || a.sval != q.sval || a.dval != q.dval) {
                        fprintf(stderr, "    op %zu '%s' param '%s': %lld at 2048, %lld at 256\n",
                                i, full.ops[i].op.c_str(), a.key.c_str(), a.ival, q.ival);
                        ++fixed_differ;
                    }
                }
            }
            CHECK_EQ(fixed_differ, 0);
            REQUIRE_EQ(small.bufs.size(), full.bufs.size());
            int shrank = 0;
            for (size_t i = 0; i < full.bufs.size(); ++i) {
                const RadBufDecl& f = full.bufs[i].d;
                const RadBufDecl& q = small.bufs[i].d;
                CHECK_EQ(small.bufs[i].name, full.bufs[i].name);
                CHECK_EQ(q.rank, f.rank);
                CHECK_EQ(q.kind, f.kind);
                for (uint32_t d = 1; d < f.rank; ++d) {
                    if (q.shape[d] != f.shape[d])
                        fprintf(stderr, "    '%s' extent %u: %lld at 2048, %lld at 256\n",
                                full.bufs[i].name.c_str(), d, (long long)f.shape[d],
                                (long long)q.shape[d]);
                    CHECK_EQ(q.shape[d], f.shape[d]);
                }
                CHECK(q.shape[0] <= f.shape[0]);
                shrank += q.shape[0] < f.shape[0];
            }
            CHECK(shrank > 10);
        }
    }
}

/* Four ranks on Flash-Next: KV heads replicate, expert widths split, most fused collectives drop.
 *
 *   - KV heads cannot be cut four ways, so they are replicated in pairs: one KV head per rank,
 *     ranks {0,1} reading rows [0,256) of the checkpoint's k/v projection and {2,3} [256,512).
 *     This is a span, not a shard index, and the only place a rank names the container rows it
 *     wants, so it is asserted as agreeing spans whose union is the whole projection.
 *   - An expert is 640 columns, five 128-column scale blocks, which four ranks cannot cut evenly.
 *     So the width divides, unevenly by class, not the expert set: every rank routes all 512
 *     experts and computes [lo, hi) of each, the odd block rotating with expert number mod 4, five
 *     blocks per four experts on every rank. Sharding the experts would not work: protection is
 *     per-rank, so a rank could own one plain bf16 expert and an odd count of quantised ones, while
 *     the grouped GEMM takes two equal parity tables (the recipe protects experts in pairs).
 *   - Fused all-reduces are world-2 kernels (`ar_wire_is_exact` and `ar_fused_ok` return false
 *     otherwise), except the gated residual's write: `ar_hc_write` and `ar_gather_hc_write` also
 *     exist at four, on the exact wire only (`wire` 0) and over a decode or verify step's rows,
 *     max_seqs * (1 + max_spec). Every block still declares its standalone `all_reduce`, and every
 *     MoE block its `moe_gather`, for the steps the fused write declines. The world-2 contrast
 *     below catches a fused declare gated on the wrong condition.
 *
 * Expected numbers derive from this file's meta (n_head 24, n_head_kv 2, head_dim 256,
 * num_experts 512); the production container's unsharded k_proj height (512 rows) is asserted as a
 * relation, not a literal. */
TEST(flashnext_tp4_replicates_kv_heads_and_splits_expert_widths) {
    RadModelMeta meta = qwen4exp_meta();
    const int      world    = 4;
    const int64_t  n_head   = meta.n_head;        /* 24 query heads */
    const int64_t  n_head_kv = meta.n_head_kv;    /* 2 KV heads, fewer than the ranks */
    const int64_t  head_dim = meta.head_dim;      /* 256 */
    const int64_t  ne_all   = atoll(rad_meta_gets(&meta, "num_experts", "0"));
    REQUIRE(n_head % world == 0);
    REQUIRE_EQ(world % n_head_kv, (int64_t)0);   /* replication needs the multiple */
    REQUIRE(ne_all % world == 0);

    /* Full-attention layers of the eight-layer fixture: 3 and 7 (interval 4). */
    const int lfull = 3;
    const std::string kq = "blk." + std::to_string(lfull) + ".attn_k.weight";
    const std::string kv = "blk." + std::to_string(lfull) + ".attn_v.weight";

    RadBuilder b[world];
    RadBuildCtx  c[world];
    for (int r = 0; r < world; ++r) {
        c[r] = build_ctx(r, world);
        c[r].max_tok  = 2048;
        c[r].max_seqs = 8;
        c[r].max_spec = 3;
        c[r].max_ctx  = 200000;
        /* geom_from must accept a world larger than the KV head count. */
        REQUIRE_EQ(qwen4exp_fp8::declare(&b[r], &meta, &c[r]), RAD_OK);
    }

    for (int r = 0; r < world; ++r) {
        /* (1) A quarter of the query heads and one KV head, so gqa is 6, and the page is the QSA
         * compress block of 4 (qwen4exp_fp8.cpp's lay.attn.kv_block must agree with the row's
         * C_EQ("block_size", 4) or nothing resolves). */
        const RecOp* at = find_op(b[r], "attn_paged");
        REQUIRE(at != nullptr);
        CHECK_EQ(pint(at, "n_head"), n_head / world);
        CHECK_EQ(pint(at, "gqa"), n_head / world);   /* one KV head, so gqa == the rank's heads */
        CHECK_EQ(pint(at, "block_size"), 4);
        /* The full-attention KV group (group 0, kv_attn, declared first) holds this rank's one
         * head. */
        REQUIRE(b[r].kv_groups.size() > 0);
        CHECK_EQ(b[r].kv_groups[0].name, std::string("kv_attn"));
        CHECK_EQ(b[r].kv_groups[0].d.n_head_kv, 1);
        CHECK_EQ(b[r].kv_groups[0].d.head_dim, head_dim);

        /* (2) One KV head's rows (256 per rank) and the span locating them in the entry. Each pair
         * of ranks declares the same span, and the two spans cover the checkpoint's whole k_proj
         * (asserted as a relation below). */
        for (const std::string& wn : {kq, kv}) {
            const RecWeight* w = find_w(b[r], wn);
            REQUIRE(w != nullptr);
            CHECK_EQ(w->d.shape[0], head_dim);          /* one head */
            CHECK_EQ(w->d.shape[1], meta.n_embd);       /* K is not sharded */
            CHECK_EQ(w->d.shard, RAD_SHARD_ROW);
            CHECK(w->span_set);
            CHECK_EQ(w->span_hi - w->span_lo, w->d.shape[0]);
            CHECK_EQ(w->span_lo, (int64_t)(r / (world / n_head_kv)) * head_dim);
            CHECK_EQ(w->span_hi, w->span_lo + head_dim);
        }
        /* (3) Every rank routes every expert and computes a slice of each. The router and expert
         * set are replicated (n_expert 512, base 0); ranks divide each expert's width in whole
         * 128-column scale blocks. */
        const RecOp* sc = find_op(b[r], "moe_scatter");
        REQUIRE(sc != nullptr);
        CHECK_EQ(pint(sc, "n_expert"), ne_all);
        CHECK_EQ(pint(sc, "expert_base"), (int64_t)0);
        const RecOp* tk = find_op(b[r], "router_topk");
        REQUIRE(tk != nullptr);
        CHECK_EQ(pint(tk, "n_expert"), ne_all);         /* replicated routing */
        if (const RecOp* ts = find_op(b[r], "router_topk_scatter")) {
            CHECK_EQ(pint(ts, "n_expert"), ne_all);
            CHECK_EQ(pint(ts, "expert_base"), (int64_t)0);
        }
        /* Each rank's width per class, as a span on every expert's down projection (no expert is
         * protected here, so class = expert number mod 4). These literals are the TP4 MoE contract:
         * five 128-blocks over four ranks, the odd block rotating with the class. They are written
         * out rather than recomputed from the plugin's formula on purpose. */
        const int64_t blk = 128;                    /* RAD_FP8_BLOCK, the scale block width */
        const int64_t nblk = meta.n_ff / blk;        /* five at the fixture's 640 */
        const int64_t cls_lo[4][4] = { { 0, 512, 384, 256 }, { 256, 0, 512, 384 },
                                       { 384, 256, 0, 512 }, { 512, 384, 256, 0 } };
        const int64_t cls_hi[4][4] = { { 256, 640, 512, 384 }, { 384, 256, 640, 512 },
                                       { 512, 384, 256, 640 }, { 640, 512, 384, 256 } };
        REQUIRE_EQ(nblk, (int64_t)5);
        int spanned[4] = { 0, 0, 0, 0 };
        for (const RecWeight& w : b[r].weights) {
            if (w.name.rfind("blk.0.ffn_down_exps.", 0) != 0) continue;
            if (w.name.compare(w.name.size() - 7, 7, ".weight") != 0) continue;
            const int q = atoi(w.name.c_str() + 20) % 4;      /* the expert's class */
            CHECK(w.span_set);
            CHECK_EQ(w.span_lo, cls_lo[r][q]);
            CHECK_EQ(w.span_hi, cls_hi[r][q]);
            ++spanned[q];
        }
        for (int q = 0; q < 4; ++q)
            CHECK_EQ(spanned[q], ne_all / 4);            /* every expert of each class declared */
        /* The four ranks' slices of one class tile its width exactly: no block unserved and none
         * claimed twice (the all-reduce would double it). */
        for (int q = 0; q < 4; ++q) {
            int64_t covered = 0;
            for (int r2 = 0; r2 < world; ++r2)
                for (int64_t i = 0; i < nblk; ++i) {
                    const int64_t lo = cls_lo[r2][q], hi = cls_hi[r2][q];
                    if (lo <= i * blk && i * blk < hi) ++covered;
                }
            CHECK_EQ(covered, nblk);   /* per class, exactly once each */
        }
        /* Every rank takes the same width, five blocks per four experts; that is why there are
         * four classes rather than two. */
        {
            int64_t sum = 0;
            for (int q = 0; q < 4; ++q) sum += cls_hi[r][q] - cls_lo[r][q];
            CHECK_EQ(sum, nblk * blk);
        }
        /* The gate/up GEMM's four classes follow the same split, doubled since the operand stacks
         * gate and up; buffers take the widest class. No parity op remains. */
        const RecOp* gu = find_op(b[r], "moe_gemm_q_mod4");
        REQUIRE(gu != nullptr);
        CHECK_EQ(pint(gu, "N"), 2 * (cls_hi[r][0] - cls_lo[r][0]));
        CHECK_EQ(pint(gu, "N_1"), 2 * (cls_hi[r][1] - cls_lo[r][1]));
        CHECK_EQ(pint(gu, "N_2"), 2 * (cls_hi[r][2] - cls_lo[r][2]));
        CHECK_EQ(pint(gu, "N_3"), 2 * (cls_hi[r][3] - cls_lo[r][3]));
        CHECK_EQ(pint(gu, "n_expert"), ne_all);
        CHECK_EQ(gu->w.size(), (size_t)(2 * ne_all));   /* codes and scales, each expert once */
        CHECK_EQ(count_op(b[r], "moe_gemm_q"), 0);

        /* (4) A standalone collective per reduction, plus the exact four-rank ar_hc_write and
         * ar_gather_hc_write (the gather form on each MoE connection), but no fused norm. Both
         * fused forms state the exact wire and range over a decode or verify step's rows,
         * 8 * (1 + 3), the architecture's own bound. */
        CHECK(count_op(b[r], "all_reduce") > 0);
        CHECK_EQ(pint(find_op(b[r], "all_reduce"), "world_size"), world);
        CHECK(count_op(b[r], "ar_hc_write") > 0);
        CHECK(count_op(b[r], "moe_gather") > 0);
        CHECK(count_op(b[r], "ar_gather_hc_write") > 0);
        CHECK(count_op(b[r], "ar_gather_hc_write") <= count_op(b[r], "moe_gather"));
        for (const RecOp& o : b[r].ops) {
            if (o.op != "ar_hc_write" && o.op != "ar_gather_hc_write") continue;
            CHECK_EQ(pint(&o, "world_size"), world);
            CHECK_EQ(pint(&o, "wire"), 0);                 /* exact only, stated */
            const RecParam* m = param(&o, "M");
            REQUIRE(m != nullptr);
            CHECK_EQ(m->kind, (int)RAD_P_RANGE);
            CHECK_EQ(m->ihi, c[r].max_seqs * (1 + c[r].max_spec));
        }
        CHECK_EQ(count_op(b[r], "ar_rmsnorm_quant_fp8"), 0);
    }

    /* The pairs agree and their union is the whole projection, n_head_kv * head_dim rows (a
     * container fact, so asserted as a relation). */
    for (const std::string& wn : {kq, kv}) {
        CHECK_EQ(find_w(b[0], wn)->span_hi, find_w(b[1], wn)->span_hi);
        CHECK_EQ(find_w(b[2], wn)->span_lo, find_w(b[3], wn)->span_lo);
        CHECK_EQ(find_w(b[0], wn)->span_lo, find_w(b[2], wn)->span_hi - n_head_kv * head_dim);
        const int64_t lo = find_w(b[0], wn)->span_lo;
        const int64_t hi = find_w(b[2], wn)->span_hi;
        CHECK_EQ(hi - lo, n_head_kv * head_dim);
        CHECK_EQ(hi - lo, world * find_w(b[0], wn)->d.shape[0] / (world / n_head_kv));
    }

    /* (5) All four ranks declare the same ops; shards differ only in parameters and spans, since
     * the plan must not vary by rank. */
    for (int r = 1; r < world; ++r) {
        if (op_names(b[r]) != op_names(b[0])) {
            fprintf(stderr, "    rank %d differs from rank 0 in its op list (%zu vs %zu)\n", r,
                    b[r].ops.size(), b[0].ops.size());
            for (size_t i = 0; i < b[0].ops.size() && i < b[r].ops.size(); ++i)
                if (b[0].ops[i].op != b[r].ops[i].op) {
                    fprintf(stderr, "    first at %zu: '%s' vs '%s'\n", i,
                            b[0].ops[i].op.c_str(), b[r].ops[i].op.c_str());
                    break;
                }
        }
        CHECK_EQ(join(op_names(b[r])), join(op_names(b[0])));
    }

    /* The shared expert is split by whole blocks, unevenly at four: 640 is five 128-blocks, and
     * rank 2 takes the extra one, after the plain experts' extra tiles on ranks 0 and 1
     * (qwen4exp_fp8.cpp's shared_place). Ranks 2, 3, 0, 1 cut [0,256) [256,384) [384,512)
     * [512,640). At two ranks, rank 0 takes three of five. */
    const int64_t sh_lo[] = { 384, 512, 0,   256 };
    const int64_t sh_hi[] = { 512, 640, 256, 384 };
    for (int r = 0; r < world; ++r) {
        const RecWeight* sg = find_w(b[r], "blk.0.ffn_gate_up_shexp.weight");
        REQUIRE(sg != nullptr);
        CHECK(sg->span_set);
        CHECK_EQ(sg->span_lo, sh_lo[r]);
        CHECK_EQ(sg->span_hi, sh_hi[r]);
        CHECK_EQ(sg->d.shape[0], 2 * (sh_hi[r] - sh_lo[r]));   /* gate's and up's rows */
    }
    /* Every block of the shared expert has exactly one owner. */
    for (int64_t i = 0; i < 640; i += 128) {
        int owners = 0;
        for (int r = 0; r < world; ++r) {
            const RecWeight* sg = find_w(b[r], "blk.0.ffn_gate_up_shexp.weight");
            owners += sg->span_lo <= i && i < sg->span_hi;
        }
        CHECK_EQ(owners, 1);
    }

    /* At two ranks the KV heads split evenly, so the even shard selects the rows and no span is
     * declared. The fused forms resolve at world 2 over every step. */
    {
        RadBuilder h[2];
        for (int r = 0; r < 2; ++r) {
            RadBuildCtx c2 = build_ctx(r, 2);
            c2.max_tok  = 2048;
            c2.max_seqs = 8;
            c2.max_spec = 3;
            c2.max_ctx  = 200000;
            REQUIRE_EQ(qwen4exp_fp8::declare(&h[r], &meta, &c2), RAD_OK);
            const RecWeight* w = find_w(h[r], kq);
            REQUIRE(w != nullptr);
            CHECK_EQ(w->d.shape[0], n_head_kv / 2 * head_dim);   /* one of two heads */
            CHECK(!w->span_set);
            CHECK_EQ(h[r].kv_groups[0].d.n_head_kv, n_head_kv / 2);
            CHECK_EQ(pint(find_op(h[r], "attn_paged"), "gqa"), (n_head / 2) / (n_head_kv / 2));
            CHECK_EQ(pint(find_op(h[r], "moe_scatter"), "n_expert"), ne_all);
            CHECK_EQ(pint(find_op(h[r], "moe_scatter"), "expert_base"), 0);
            CHECK(count_op(h[r], "ar_hc_write") > 0);
            CHECK(count_op(h[r], "ar_gather_hc_write") > 0);
            /* Two ranks keep the exact write over every step, prefill chunks included. */
            for (const RecOp& o : h[r].ops) {
                if (o.op != "ar_hc_write") continue;
                CHECK_EQ(pint(&o, "wire"), 0);
                const RecParam* m = param(&o, "M");
                REQUIRE(m != nullptr);
                CHECK_EQ(m->ihi, c2.max_tok);
            }
            /* No ar_rmsnorm_quant_fp8 at either world: qwen4exp hands every block's norm
             * `kArNone`, since the reduction rides in the gated residual's write instead
             * (qwen4exp_fp8.cpp's block declares). */
            CHECK_EQ(count_op(h[r], "ar_rmsnorm_quant_fp8"), 0);
        }
    }

    /* Which steps the four-rank fused write takes: decode and verify steps, up to
     * max_seqs * (1 + max_spec) rows and never more than max_tok, whose message goes out exact (on
     * the lossy wire: under its floor or off its 128-element tile). The bound is the architecture's,
     * not a message size: 64 sequences at depth 3 are 256 rows (1.25 MiB) and still take it, while
     * a prefill chunk never does. The preceding block asks the same predicate. At two ranks it
     * equals ar_wire_is_exact, which itself stays false at four. */
    {
        rad::arch::Geom g4{};
        g4.world = 4; g4.n_embd = 2560; g4.wire_exact = 1; g4.wire_min_bytes = 128 * 1024;
        g4.max_tok = 2048; g4.max_seqs = 8; g4.max_spec = 3;
        CHECK_EQ(rad::arch::ar_wide_write_rows(g4), (int64_t)32);
        CHECK_EQ(rad::arch::ar_exact_write_rows(g4), (int64_t)32);
        CHECK(!rad::arch::ar_exact_write_takes(g4, 0));
        CHECK(rad::arch::ar_exact_write_takes(g4, 1));
        CHECK(rad::arch::ar_exact_write_takes(g4, 8));      /* a decode step */
        CHECK(rad::arch::ar_exact_write_takes(g4, 32));     /* every sequence verifying 1 + 3 */
        CHECK(!rad::arch::ar_exact_write_takes(g4, 33));    /* more: a prefill chunk */
        CHECK(!rad::arch::ar_exact_write_takes(g4, 2048));
        CHECK(!rad::arch::ar_wire_is_exact(g4, 8));
        g4.wire_exact = 0;
        CHECK(rad::arch::ar_exact_write_takes(g4, 25));     /* 125 KiB, under the floor */
        CHECK(!rad::arch::ar_exact_write_takes(g4, 26));    /* 130 KiB, goes on the ti8 wire */
        CHECK(!rad::arch::ar_exact_write_takes(g4, 32));
        g4.wire_exact = 1;
        g4.max_spec = 0;                                    /* one row per sequence */
        CHECK_EQ(rad::arch::ar_wide_write_rows(g4), (int64_t)8);
        CHECK(rad::arch::ar_exact_write_takes(g4, 8));
        CHECK(!rad::arch::ar_exact_write_takes(g4, 9));
        g4.max_seqs = 64; g4.max_spec = 3;
        CHECK(rad::arch::ar_exact_write_takes(g4, 256));    /* 1.25 MiB */
        CHECK(!rad::arch::ar_exact_write_takes(g4, 257));
        g4.max_tok = 128;                                   /* caps a step's rows */
        CHECK_EQ(rad::arch::ar_wide_write_rows(g4), (int64_t)128);
        CHECK(rad::arch::ar_exact_write_takes(g4, 128));
        CHECK(!rad::arch::ar_exact_write_takes(g4, 129));
        rad::arch::Geom g8 = g4;
        g8.world = 8;
        CHECK_EQ(rad::arch::ar_exact_write_rows(g8), (int64_t)0);
        CHECK(!rad::arch::ar_exact_write_takes(g8, 8));
        rad::arch::Geom g2 = g4;
        g2.world = 2; g2.max_tok = 4096; g2.max_seqs = 8;
        CHECK_EQ(rad::arch::ar_exact_write_rows(g2), g2.max_tok);
        for (int exact : { 0, 1 }) {
            g2.wire_exact = exact;
            for (int64_t T : { 1, 8, 25, 26, 33, 205, 4096 })
                CHECK(rad::arch::ar_exact_write_takes(g2, T) == rad::arch::ar_wire_is_exact(g2, T));
        }
    }
    /* The four-rank fused gather takes the fused write's steps, within one MoE pass. Only the
     * exact bound exists at four (the rotated form is never declared, so `rows6` is 0), and a step
     * the write declines keeps its plain gather. At two ranks ar_wire_is_exact picks the bound. */
    {
        rad::arch::Geom g4{};
        g4.world = 4; g4.n_embd = 2560; g4.wire_exact = 1; g4.wire_min_bytes = 128 * 1024;
        g4.max_tok = 2048; g4.max_seqs = 8; g4.max_spec = 3;
        CHECK(rad::arch::gather_taken(g4, 8, 32, 0));        /* a decode step */
        CHECK(rad::arch::gather_taken(g4, 32, 32, 0));       /* a verify step, one pass */
        CHECK(!rad::arch::gather_taken(g4, 33, 2048, 0));    /* the write declines it */
        CHECK(!rad::arch::gather_taken(g4, 8, 0, 0));        /* not declared */
        CHECK(!rad::arch::gather_taken(g4, 20, 16, 0));      /* two passes */
        g4.wire_exact = 0;
        CHECK(rad::arch::gather_taken(g4, 25, 32, 0));       /* under the floor */
        CHECK(!rad::arch::gather_taken(g4, 26, 32, 0));      /* goes on the ti8 wire */
        rad::arch::Geom g2 = g4;
        g2.world = 2;
        for (int64_t T : { 1, 8, 25, 26, 205, 4096 })
            for (int64_t r6 : { 0, 2048 }) {
                const int64_t r = rad::arch::ar_wire_is_exact(g2, T) ? 2048 : r6;
                CHECK(rad::arch::gather_taken(g2, T, 2048, r6) == (r > 0 && T <= r));
            }
    }
    /* At four, the lossy wire declares no rotated write or gather: only the exact forms, stating
     * `wire` 0, for decode and verify steps under the floor. */
    {
        RadBuilder b4;
        RadBuildCtx c4 = build_ctx(0, world);
        c4.max_tok = 2048; c4.max_seqs = 8; c4.max_spec = 3; c4.max_ctx = 200000;
        c4.tp_wire_lossy = 1;
        c4.tp_wire_min_bytes = 128 * 1024;
        REQUIRE_EQ(qwen4exp_fp8::declare(&b4, &meta, &c4), RAD_OK);
        CHECK(count_op(b4, "ar_hc_write") > 0);
        CHECK(count_op(b4, "ar_gather_hc_write") > 0);
        for (const RecOp& o : b4.ops) {
            if (o.op != "ar_hc_write" && o.op != "ar_gather_hc_write") continue;
            CHECK_EQ(pint(&o, "wire"), 0);
            const RecParam* m = param(&o, "M");
            REQUIRE(m != nullptr);
            CHECK_EQ(m->ihi, c4.max_seqs * (1 + c4.max_spec));
        }
    }

    /* A world that is not a multiple of the KV head count still declares: qwen4exp opts in via
     * geom_from's `attn_world_ok`, so attention runs on the largest multiple below the world and the
     * remaining ranks compute none (see the tp3 test). Nothing is rounded into a half head. A
     * plugin that does not opt in is refused at this world by geom_from. */
    {
        RadBuilder b3;
        RadBuildCtx c3 = build_ctx(0, 3);
        c3.max_tok = 2048; c3.max_seqs = 8; c3.max_spec = 3; c3.max_ctx = 200000;
        CHECK_EQ(qwen4exp_fp8::declare(&b3, &meta, &c3), RAD_OK);
    }
    {
        rad::arch::Geom g3{};
        RadBuildCtx c3 = build_ctx(0, 3);
        c3.max_tok = 2048; c3.max_seqs = 8; c3.max_spec = 3; c3.max_ctx = 200000;
        CHECK_EQ(rad::arch::geom_from(g3, &meta, &c3, "bf16", RAD_BF16, /*arch_splits_ff=*/true), RAD_E_INVAL);
        CHECK_EQ(rad::arch::geom_from(g3, &meta, &c3, "bf16", RAD_BF16, /*arch_splits_ff=*/true,
                           /*attn_world_ok=*/true), RAD_OK);
        CHECK(g3.attn_zero == false);
        CHECK_EQ(g3.attn_world, 2);
    }

    /* ---- A layer that keeps experts plain stays at four classes. Two plain experts leave 510
     * quantised ones, four leave 508; either way `moe_gemm_q_mod4` serves the layer, the class being
     * the expert's place in the sort mod 4 (plain ones sorted last). At 510, classes 0 and 1 hold
     * 128 experts and 2 and 3 hold 127 (the first tables take the remainder). The plain experts'
     * extra tiles go to ranks 0 and 1, so every rank holds the same routed width; the shared
     * expert's extra block is rank 2's. */
    {
        const RadEncoding w4p = [] { RadEncoding e = rad_enc_affine(RAD_I4, RAD_BF16, 1, 128);
                                     rad_enc_copy_str(e.transform, "fwht128"); return e; }();
        const RadEncoding plain = rad_enc_plain(RAD_BF16);
        /* The four-class slices, [lo, hi) by rank and place mod 4. */
        const int64_t c4_lo[4][4] = { { 0, 512, 384, 256 }, { 256, 0, 512, 384 },
                                      { 384, 256, 0, 512 }, { 512, 384, 256, 0 } };
        const int64_t c4_hi[4][4] = { { 256, 640, 512, 384 }, { 384, 256, 640, 512 },
                                      { 512, 384, 256, 640 }, { 640, 512, 384, 256 } };
        for (int nprot : { 2, 4 }) {
            const std::vector<int> pe = nprot == 2 ? std::vector<int>{ 3, 10 }
                                                   : std::vector<int>{ 3, 10, 21, 40 };
            for (int r = 0; r < world; ++r) {
                RadBuilder bp;
                for (int e : pe) {
                    bp.encs.push_back({ "blk.1.ffn_gate_up_exps." + std::to_string(e) + ".weight",
                                        plain });
                    bp.encs.push_back({ "blk.1.ffn_down_exps." + std::to_string(e) + ".weight",
                                        plain });
                }
                bp.encs.push_back({ "ffn_gate_up_exps", w4p });
                bp.encs.push_back({ "ffn_down_exps", w4p });
                RadBuildCtx cp = build_ctx(r, world);
                cp.max_tok = 2048; cp.max_seqs = 8; cp.max_spec = 3; cp.max_ctx = 200000;
                REQUIRE_EQ(qwen4exp_fp8::declare(&bp, &meta, &cp), RAD_OK);
                const int64_t n_reg = ne_all - nprot;
                /* layer 1's quantised GEMMs */
                int l1_ops = 0, other_ops = 0;
                for (const RecOp& o : bp.ops) {
                    if (o.op != "moe_gemm_q" && o.op != "moe_gemm_q_mod4") continue;
                    bool l1 = false;
                    for (rad_weight wh : o.w)
                        l1 = l1 || bp.weights[wh - 1].name.rfind("blk.1.", 0) == 0;
                    if (!l1) { CHECK_EQ(o.op, std::string("moe_gemm_q_mod4")); ++other_ops; continue; }
                    ++l1_ops;
                    CHECK_EQ(o.op, std::string("moe_gemm_q_mod4"));
                    CHECK_EQ(pint(&o, "n_expert"), n_reg);
                    CHECK_EQ(o.w.size(), (size_t)(2 * n_reg));
                    /* four runs, codes then scales, class by class; class q holds places q,
                     * q + 4, ... (128, 128, 127, 127 at 510) */
                    size_t at = 0;
                    for (int q = 0; q < 4; ++q) {
                        const int64_t nq = (n_reg + 3 - q) / 4;
                        for (int half = 0; half < 2; ++half)
                            for (int64_t j = 0; j < nq && at < o.w.size(); ++j, ++at) {
                                const std::string& wn = bp.weights[o.w[at] - 1].name;
                                const int e = atoi(wn.c_str() + wn.find("_exps.") + 6);
                                int place = e;
                                for (int x : pe) place -= x < e ? 1 : 0;
                                CHECK_EQ(place, (int)(q + 4 * j));
                                CHECK((wn.find(".scale") != std::string::npos) == (half == 1));
                            }
                    }
                    CHECK_EQ(at, o.w.size());
                }
                CHECK_EQ(l1_ops, 2);
                CHECK(other_ops > 0);
                /* each quantised expert of layer 1 gets its class's slice */
                int seen = 0;
                for (const RecWeight& w : bp.weights) {
                    if (w.name.rfind("blk.1.ffn_down_exps.", 0) != 0) continue;
                    if (w.name.compare(w.name.size() - 7, 7, ".weight") != 0) continue;
                    if (w.d.dtype == RAD_BF16) continue;          /* plain ones checked below */
                    const int e = atoi(w.name.c_str() + 20);
                    int place = e;
                    for (int x : pe) place -= x < e ? 1 : 0;
                    CHECK(w.span_set);
                    CHECK_EQ(w.span_lo, c4_lo[r][place % 4]);
                    CHECK_EQ(w.span_hi, c4_hi[r][place % 4]);
                    ++seen;
                }
                CHECK_EQ(seen, (int)n_reg);
                /* the plain experts' tiles by rank: 3, 3, 2, 2 */
                const int64_t p4_lo[4] = { 0, 192, 384, 512 },   p4_hi[4] = { 192, 384, 512, 640 };
                int pseen = 0;
                for (const RecWeight& w : bp.weights) {
                    if (w.name.rfind("blk.1.ffn_", 0) != 0 || w.d.dtype != RAD_BF16) continue;
                    if (w.name.find("_exps.") == std::string::npos) continue;
                    if (w.name.compare(w.name.size() - 7, 7, ".weight") != 0) continue;
                    CHECK(w.span_set);
                    CHECK_EQ(w.span_lo, p4_lo[r]);
                    CHECK_EQ(w.span_hi, p4_hi[r]);
                    ++pseen;
                }
                CHECK_EQ(pseen, 2 * nprot);
            }
        }
    }

    /* ---- The plain bf16 experts at four ranks.
     *
     * A second grouped GEMM serves them over their own slice of the same width: whole 64-wide
     * output tiles, one range per rank, uniform across a layer's protected experts (their GEMM
     * takes one table, so every entry has one geometry; they do not alternate by parity). The
     * width must be the expert's own, not derived from this rank's parity slices, which only match
     * at two ranks.
     *
     * Ten tiles over four ranks is 3, 3, 2, 2: uneven, whole tiles, each owned once. Ranks 0 and 1
     * compute 192 columns of every protected expert, ranks 2 and 3 compute 128; every rank holds
     * the same routed width, and rank 2 the shared expert's extra block (shared_place). Two
     * protected experts, as the production recipe pairs them. */
    {
        const RadEncoding w4p = [] { RadEncoding e = rad_enc_affine(RAD_I4, RAD_BF16, 1, 128);
                                     rad_enc_copy_str(e.transform, "fwht128"); return e; }();
        const RadEncoding plain = rad_enc_plain(RAD_BF16);
        /* [lo, hi) of the plain experts' columns per rank, written out: 3, 3, 2, 2 tiles. */
        const int64_t p_lo[4] = { 0,   192, 384, 512 };
        const int64_t p_hi[4] = { 192, 384, 512, 640 };
        for (int r = 0; r < world; ++r) {
            RadBuilder bp;
            bp.encs = { { "blk.1.ffn_gate_up_exps.3.weight", plain },
                        { "blk.1.ffn_down_exps.3.weight", plain },
                        { "blk.1.ffn_gate_up_exps.10.weight", plain },
                        { "blk.1.ffn_down_exps.10.weight", plain },
                        { "ffn_gate_up_exps", w4p }, { "ffn_down_exps", w4p } };
            RadBuildCtx cp = build_ctx(r, world);
            cp.max_tok = 2048; cp.max_seqs = 8; cp.max_spec = 3; cp.max_ctx = 200000;
            REQUIRE_EQ(qwen4exp_fp8::declare(&bp, &meta, &cp), RAD_OK);
            const int64_t wp = p_hi[r] - p_lo[r];
            int seen = 0;
            for (const RecWeight& w : bp.weights) {
                if (w.name.rfind("blk.1.ffn_gate_up_exps.", 0) != 0 &&
                    w.name.rfind("blk.1.ffn_down_exps.", 0) != 0) continue;
                if (w.d.dtype != RAD_BF16) continue;      /* plain ones only */
                if (w.name.compare(w.name.size() - 7, 7, ".weight") != 0) continue;
                CHECK(w.span_set);
                CHECK_EQ(w.span_lo, p_lo[r]);
                CHECK_EQ(w.span_hi, p_hi[r]);
                ++seen;
            }
            /* Both projections of both protected experts. */
            CHECK_EQ(seen, 4);
            /* The gate/up GEMM's N is twice this rank's share (stacked pair) and the down GEMM's
             * K is the share. */
            int gu_ok = 0, dn_ok = 0;
            for (const RecOp& o : bp.ops) {
                if (o.op != "moe_gemm") continue;
                if (pint(&o, "N") == 2 * wp && pint(&o, "n_expert") == 2) ++gu_ok;
                if (pint(&o, "K") == wp && pint(&o, "n_expert") == 2) ++dn_ok;
            }
            CHECK_EQ(gu_ok, 1);
            CHECK_EQ(dn_ok, 1);
        }
    }
}

/* ==================================================================== media */
namespace {

/* The fixture above plus what a Qwen3.8 container adds for pictures: the text config's
 * interleaved M-RoPE and a vision tower cut to two blocks at the real widths, enough to see a
 * block repeat. */
const char* k_vl_keys[] = {
    "qwen35.full_attention_interval",
    "qwen35.rope.dimension_count",
    "qwen35.ssm.state_size",
    "qwen35.ssm.group_count",
    "qwen35.ssm.time_step_rank",
    "qwen35.ssm.inner_size",
    "qwen35.ssm.conv_kernel",
    "rope_parameters.mrope_section",
    "rope_parameters.mrope_interleaved",
    "vision_config.depth",
    "vision_config.hidden_size",
    "vision_config.num_heads",
    "vision_config.intermediate_size",
    "vision_config.patch_size",
    "vision_config.temporal_patch_size",
    "vision_config.in_channels",
    "vision_config.spatial_merge_size",
    "vision_config.out_hidden_size",
    "vision_config.num_position_embeddings",
    "vision_config.deepstack_visual_indexes",
    "vision_config.hidden_act",
};
const char* k_vl_vals[] = { "4", "64", "128", "16", "48", "6144", "4",
                            "11 11 10", "1",
                            "2", "1152", "16", "4304", "16", "2", "3", "2", "5120", "2304", "",
                            "gelu_pytorch_tanh" };

RadModelMeta qwen35_vl_meta() {
    RadModelMeta m = qwen35_meta();
    m.n_kv   = (int)(sizeof(k_vl_keys) / sizeof(k_vl_keys[0]));
    m.kv_key = k_vl_keys;
    m.kv_val = k_vl_vals;
    return m;
}

RadBuildCtx vl_ctx(int64_t patches) {
    RadBuildCtx c = build_ctx();
    c.max_enc_patches = patches;
    return c;
}

const RecOp* issued_op(const RadBuilder& b, const RadCtx& c, const std::string& name, int nth = 0) {
    for (const auto& i : c.issues) {
        const RecOp* o = b.op_of(i.op);
        if (o && o->op == name && nth-- == 0) return o;
    }
    return nullptr;
}
const RecIssue* issue_of(const RadBuilder& b, const RadCtx& c, const std::string& name,
                         int nth = 0) {
    for (const auto& i : c.issues) {
        const RecOp* o = b.op_of(i.op);
        if (o && o->op == name && nth-- == 0) return &i;
    }
    return nullptr;
}

}  /* namespace */

/* Three ranks over two KV heads: the attention world is 2 (rad_arch.h), so rank 2 computes no
 * attention (a `fill` of its output wire and a zero KV group). The delta net's 16 K heads go 5, 5,
 * 6 with the V heads at the same multiple, the extra head on the rank without attention, declared
 * as container spans with no padding. */
TEST(flashnext_tp3_zeroes_the_third_ranks_attention_and_splits_gdn_heads_unevenly) {
    RadModelMeta meta = qwen4exp_meta();
    const int      world     = 3;
    const int64_t  n_head    = meta.n_head;       /* 24 */
    const int64_t  n_head_kv = meta.n_head_kv;    /* 2: does not divide 3 */
    const int64_t  head_dim  = meta.head_dim;     /* 256 */
    const int64_t  nk = atoll(rad_meta_gets(&meta, "linear_num_key_heads", "0"));     /* 16 */
    const int64_t  nv = atoll(rad_meta_gets(&meta, "linear_num_value_heads", "0"));   /* 48 */
    const int64_t  hk = atoll(rad_meta_gets(&meta, "linear_key_head_dim", "0"));      /* 128 */
    const int64_t  hv = atoll(rad_meta_gets(&meta, "linear_value_head_dim", "0"));    /* 128 */
    REQUIRE(n_head_kv < world && world % n_head_kv != 0);
    REQUIRE(nk % world != 0);
    const int attn_world = (int)(n_head_kv * (world / n_head_kv));   /* 2 */

    const int lfull = 3;
    const std::string kq = "blk." + std::to_string(lfull) + ".attn_k.weight";
    const std::string kv = "blk." + std::to_string(lfull) + ".attn_v.weight";
    const std::string qg = "blk." + std::to_string(lfull) + ".attn_qg.weight";
    const std::string oo = "blk." + std::to_string(lfull) + ".attn_output.weight";

    RadBuilder b[world];
    RadBuildCtx  c[world];
    for (int r = 0; r < world; ++r) {
        c[r] = build_ctx(r, world);
        c[r].max_tok  = 2048;
        c[r].max_seqs = 8;
        c[r].max_spec = 3;
        c[r].max_ctx  = 200000;
        REQUIRE_EQ(qwen4exp_fp8::declare(&b[r], &meta, &c[r]), RAD_OK);
    }

    /* Every rank declares the same KV groups in the same order, since the scheduler indexes groups
     * by position and one block table serves all ranks. Ops may differ per rank (issue walks each
     * rank's own program), but the attention ranks declare the same ops. */
    for (int r = 1; r < world; ++r) {
        REQUIRE_EQ(b[r].kv_groups.size(), b[0].kv_groups.size());
        for (size_t g = 0; g < b[0].kv_groups.size(); ++g) {
            CHECK_EQ(b[r].kv_groups[g].name, b[0].kv_groups[g].name);
            CHECK_EQ(b[r].kv_groups[g].d.kind, b[0].kv_groups[g].d.kind);
        }
        if (r < attn_world) {
            REQUIRE_EQ(b[r].ops.size(), b[0].ops.size());
            for (size_t i = 0; i < b[0].ops.size(); ++i) CHECK_EQ(b[r].ops[i].op, b[0].ops[i].op);
        }
    }

    std::vector<std::pair<int64_t, int64_t>> k_span((size_t)world);   /* each rank's K heads [lo, hi) */
    for (int r = 0; r < world; ++r) {
        const bool zero = r >= attn_world;
        CHECK_EQ(b[r].kv_groups[0].d.n_head_kv, 1);
        /* The attention-zero rank has no attention op or projection weight, fills its output
         * wire, and carves no cache for the attention group. */
        if (zero) {
            CHECK(find_op(b[r], "attn_paged") == nullptr);
            CHECK(find_op(b[r], "kv_store") == nullptr);
            CHECK(find_op(b[r], "fill") != nullptr);
            CHECK(find_w(b[r], kq) == nullptr);
            CHECK(find_w(b[r], qg) == nullptr);
            CHECK(find_w(b[r], oo) == nullptr);
            CHECK(b[r].kv_groups[0].zero);
            /* It still joins the all-reduce and runs the indexer. */
            CHECK(find_op(b[r], "qsa_select") != nullptr);
        } else {
            CHECK(!b[r].kv_groups[0].zero);
            CHECK(find_op(b[r], "fill") == nullptr);
        }
        /* (1) Each attention rank has the two-rank geometry: 12 heads, one KV head, gqa 12. */
        const RecOp* at = find_op(b[r], "attn_paged");
        if (!zero) {
            REQUIRE(at != nullptr);
            CHECK_EQ(pint(at, "n_head"), n_head / attn_world);
            CHECK_EQ(pint(at, "gqa"), n_head / attn_world);
        }
        /* (2) The projections: the span of this rank's heads in the container. */
        for (const std::string& wn : {kq, kv}) {
            if (zero) break;
            const RecWeight* w = find_w(b[r], wn);
            REQUIRE(w != nullptr);
            CHECK_EQ(w->d.shape[0], head_dim);
            CHECK(w->span_set);
            const int ar = zero ? attn_world - 1 : r;
            CHECK_EQ(w->span_lo, (int64_t)ar * head_dim);
            CHECK_EQ(w->span_hi, w->span_lo + head_dim);
        }
        if (!zero) {
            const RecWeight* w = find_w(b[r], qg);
            REQUIRE(w != nullptr);
            const int ar = zero ? attn_world - 1 : r;
            const int64_t qg_rows = 2 * (n_head / attn_world) * head_dim;
            CHECK_EQ(w->d.shape[0], qg_rows);
            CHECK(w->span_set);
            CHECK_EQ(w->span_lo, (int64_t)ar * qg_rows);
            const RecWeight* o = find_w(b[r], oo);
            REQUIRE(o != nullptr);
            CHECK_EQ(o->d.shard, RAD_SHARD_COL);
            CHECK(o->span_set);
            CHECK_EQ(o->span_lo, (int64_t)ar * (n_head / attn_world) * head_dim);
        }
        /* (3) The delta net: 5, 5, 6 K heads and 15, 15, 18 V heads on every rank, as per-part
         * spans of [q|k|v|z] and of the conv channels, and a single span of A_log. */
        const int64_t base = nk / world, rem = nk % world;
        /* The extra head goes to the rank without attention first (rad_arch.h remainder_place):
         * rank 2 takes 6, ranks 0 and 1 take 5. */
        const int64_t pl   = r >= attn_world ? r - attn_world : (world - attn_world) + r % attn_world;
        const int64_t my_k = base + (pl < rem ? 1 : 0);
        const int64_t k0   = pl * base + (pl < rem ? pl : rem);
        const int64_t my_v = (nv / nk) * my_k, v0 = (nv / nk) * k0;
        const RecWeight* in = find_w(b[r], "blk.0.ssm_inz.weight");
        REQUIRE(in != nullptr);
        REQUIRE_EQ(in->d.n_row_parts, 4);
        CHECK_EQ(in->d.row_parts[0], my_k * hk);
        CHECK_EQ(in->d.row_parts[2], my_v * hv);
        REQUIRE_EQ(in->n_part_spans, 4);
        CHECK_EQ(in->part_lo[0], k0 * hk);
        CHECK_EQ(in->part_lo[1], k0 * hk);
        CHECK_EQ(in->part_lo[2], v0 * hv);
        CHECK_EQ(in->part_lo[3], v0 * hv);
        CHECK_EQ(in->part_total[0], nk * hk);
        CHECK_EQ(in->part_total[2], nv * hv);
        k_span[(size_t)r] = { in->part_lo[0] / hk, in->part_hi[0] / hk };
        const RecWeight* cv = find_w(b[r], "blk.0.ssm_conv1d.weight");
        REQUIRE(cv != nullptr);
        REQUIRE_EQ(cv->n_part_spans, 3);
        CHECK_EQ(cv->part_lo[2], v0 * hv);
        CHECK_EQ(cv->part_hi[2], (v0 + my_v) * hv);
        const RecWeight* al = find_w(b[r], "blk.0.ssm_a_log");
        REQUIRE(al != nullptr);
        CHECK_EQ(al->d.shape[0], my_v);
        CHECK(al->span_set);
        CHECK_EQ(al->span_lo, v0);
        CHECK_EQ(al->span_hi, v0 + my_v);
        const RecWeight* ow = find_w(b[r], "blk.0.ssm_out.weight");
        REQUIRE(ow != nullptr);
        CHECK_EQ(ow->d.shard, RAD_SHARD_COL);
        CHECK(ow->span_set);
        CHECK_EQ(ow->span_lo, v0 * hv);
        CHECK_EQ(ow->span_hi, (v0 + my_v) * hv);
        /* The GDN state group holds this rank's V heads. */
        bool seen = false;
        for (const auto& g : b[r].kv_groups)
            if (g.name == "kv_gdn_state") { seen = true; CHECK_EQ(g.d.n_head_kv, (int)my_v); }
        CHECK(seen);
    }
    /* The three ranks' K head spans cover the container's 16 heads exactly once. */
    {
        std::vector<int> owner((size_t)nk, -1);
        for (int r = 0; r < world; ++r)
            for (int64_t h = k_span[(size_t)r].first; h < k_span[(size_t)r].second; ++h) {
                REQUIRE(h >= 0 && h < nk);
                CHECK_EQ(owner[(size_t)h], -1);
                owner[(size_t)h] = r;
            }
        for (int64_t h = 0; h < nk; ++h) CHECK(owner[(size_t)h] >= 0);
    }
}

TEST(a_vision_container_declares_the_tower_when_the_engine_asks_for_it) {
    /* With no encoder budget the tower is not declared, whatever the container carries. */
    {
        RadBuilder b;
        RadModelMeta meta = qwen35_vl_meta();
        RadBuildCtx  ctx  = vl_ctx(0);
        CHECK_OK(qwen35_bf16::declare(&b, &meta, &ctx));
        CHECK(b.encoder_name.empty());
        CHECK(find_w(b, "v.patch.weight") == nullptr);
        CHECK_EQ(count_op(b, "scatter_rows"), 0);
    }
    RadBuilder b;
    RadModelMeta meta = qwen35_vl_meta();
    RadBuildCtx  ctx  = vl_ctx(1024);
    CHECK_OK(qwen35_bf16::declare(&b, &meta, &ctx));

    CHECK_EQ(b.encoder_name, std::string("vision"));
    CHECK_EQ(b.encoder.modalities, (uint32_t)(RAD_MM_IMAGE | RAD_MM_VIDEO));
    CHECK_EQ(b.encoder.patch_dim, 3 * 2 * 16 * 16);
    CHECK_EQ(b.encoder.merge, 4);
    CHECK_EQ(b.encoder.max_patches, 1024);
    CHECK_EQ(b.encoder.n_embd, 5120);
    const RecBuf* out = find_b(b, "v.out");
    REQUIRE(out != nullptr);
    CHECK_EQ(out->d.shape[0], 256);
    CHECK_EQ(out->d.shape[1], 5120);

    /* 3 stem + 12 per block + 6 merger, each mapped from the checkpoint's model.visual.*. */
    int nv = 0;
    for (const auto& w : b.weights)
        if (w.name.rfind("v.", 0) == 0) {
            ++nv;
            CHECK(find_map(b, w.name) != nullptr);
            CHECK_EQ(w.d.shard, (int)RAD_SHARD_NONE);
        }
    CHECK_EQ(nv, 3 + 12 * 2 + 6);
    const RecMap* qkv = find_map(b, "v.blk.1.qkv.weight");
    REQUIRE(qkv != nullptr);
    REQUIRE_EQ(qkv->src.size(), (size_t)1);
    CHECK_EQ(qkv->src[0], std::string("model.visual.blocks.1.attn.qkv.weight"));

    /* The tower's rotary is 2-D, half of the 72-wide head per axis. */
    int axial = 0;
    for (const auto& o : b.ops)
        if (o.op == "rope" && pstr(&o, "mode") == "axial") {
            ++axial;
            CHECK_EQ(pstr(&o, "sections"), std::string("18 18"));
            CHECK_EQ(pint(&o, "head_dim"), 72);
        }
    CHECK_EQ(axial, 2);
    /* The language model's is the interleaved M-RoPE with the checkpoint's sections. */
    int im = 0;
    for (const auto& o : b.ops)
        if (o.op == "rope" && pstr(&o, "mode") == "imrope") {
            ++im;
            CHECK_EQ(pstr(&o, "sections"), std::string("11 11 10"));
            CHECK_EQ(pint(&o, "rotary_dim"), 64);
        }
    CHECK_EQ(im, 4);
    CHECK_EQ(count_op(b, "scatter_rows"), 1);

    /* A sizing declare sizes the tower without declaring it to the core. */
    RadBuilder bp;
    RadBuildCtx cp = ctx;
    cp.shape_probe = 1;
    CHECK_OK(qwen35_bf16::declare(&bp, &meta, &cp));
    CHECK(bp.encoder_name.empty());
    CHECK(find_w(bp, "v.patch.weight") != nullptr);
}

TEST(an_encoder_pass_issues_the_tower_and_nothing_else) {
    RadBuilder b;
    RadModelMeta meta = qwen35_vl_meta();
    RadBuildCtx  ctx  = vl_ctx(1024);
    CHECK_OK(qwen35_bf16::declare(&b, &meta, &ctx));

    /* Two segments of 8 x 8 patches: two pictures or two temporal groups of one video. */
    static uint16_t pix[128 * 1536];
    static int32_t  coord[4 * 128];
    static const int32_t cu[] = { 0, 64, 128 };
    RadBatch batch{};
    batch.enc         = 1;
    batch.enc_n_patch = 128;
    batch.enc_pixels  = pix;
    batch.enc_coord   = coord;
    batch.enc_n_seg   = 2;
    batch.enc_cu      = cu;
    batch.enc_max_seg = 64;
    RadCtx c{ .b = &b, .batch = &batch };
    qwen35_bf16::step(&c, &batch);
    CHECK_LINT(b, &c);
    CHECK(c.step_fail.empty());

    std::vector<std::string> want = { "gemm_nt_bias", "grid_embed" };
    for (int l = 0; l < 2; ++l)
        for (const char* s : { "layernorm", "gemm_nt_bias", "rope", "attn_dense", "gemm_nt_bias",
                               "layernorm", "gemm_nt_bias", "gemm_nt_bias" })
            want.push_back(s);
    for (const char* s : { "layernorm", "gemm_nt_bias", "gemm_nt_bias" }) want.push_back(s);
    CHECK_EQ(join(issued_ops(b, c)), join(want));

    /* The merger runs over merged rows, a quarter of the patches. */
    const RecIssue* last = &c.issues.back();
    CHECK_EQ(last->n, 32);
    /* The attention reads the segment boundaries, and q, k, v as column slices. */
    const RecIssue* at = issue_of(b, c, "attn_dense");
    REQUIRE(at != nullptr);
    REQUIRE_EQ(at->opd.size(), (size_t)5);
    CHECK_EQ(at->opd[0].cols, 1152);
    CHECK_EQ(at->opd[1].offset, 1152);
    CHECK_EQ(at->opd[2].offset, 2304);
    CHECK(at->opd[3].raw == cu);
    CHECK_EQ(at->opd[3].rows, 3);
    /* The rotary reads the row and column planes; the grid embedding all four. */
    const RecIssue* ro = issue_of(b, c, "rope");
    REQUIRE(ro != nullptr);
    CHECK(ro->opd[1].raw == coord);
    CHECK_EQ(ro->opd[1].rows, 2);
    CHECK_EQ(ro->opd[1].cols, 128);
    const RecIssue* ge = issue_of(b, c, "grid_embed");
    REQUIRE(ge != nullptr);
    CHECK_EQ(ge->opd[2].rows, 4);

    /* A pass larger than the declared tower fails the step rather than overrunning. */
    RadBatch big = batch;
    big.enc_n_patch = 2048;
    RadCtx c2{ .b = &b, .batch = &big };
    qwen35_bf16::step(&c2, &big);
    CHECK(!c2.step_fail.empty());
    CHECK(c2.issues.empty());
}

TEST(a_step_with_pictures_scatters_their_rows_and_rotates_by_three_components) {
    RadBuilder b;
    RadModelMeta meta = qwen35_vl_meta();
    RadBuildCtx  ctx  = vl_ctx(1024);
    CHECK_OK(qwen35_bf16::declare(&b, &meta, &ctx));

    /* A prefill chunk whose tokens 4..11 are one picture's rows. */
    static int32_t rope[3 * 64];
    static const int32_t rows[] = { 4, 5, 6, 7, 8, 9, 10, 11 };
    static uint16_t embd[8 * 5120];
    RadBatch batch = prefill_batch();
    batch.rope_pos   = rope;
    batch.rope_mixed = 1;
    batch.n_mm_rows  = 8;
    batch.mm_rows    = rows;
    batch.mm_embd    = embd;
    RadCtx c{ .b = &b, .batch = &batch };
    qwen35_bf16::step(&c, &batch);
    CHECK_LINT(b, &c);
    CHECK(c.step_fail.empty());

    const std::vector<std::string> ops = issued_ops(b, c);
    REQUIRE(ops.size() > 2);
    CHECK_EQ(ops[0], std::string("embed_lookup"));
    CHECK_EQ(ops[1], std::string("scatter_rows"));
    const RecIssue* sc = issue_of(b, c, "scatter_rows");
    REQUIRE(sc != nullptr);
    CHECK_EQ(sc->n, 8);
    CHECK(sc->opd[0].raw == embd);
    CHECK_EQ(sc->opd[0].rows, 8);
    CHECK_EQ(sc->opd[0].cols, 5120);
    CHECK(sc->opd[1].raw == rows);
    CHECK_EQ(sc->opd[2].rows, 64);

    /* Every language-model rotary takes the [3, T] planes. */
    int n = 0;
    for (const auto& i : c.issues) {
        const RecOp* o = b.op_of(i.op);
        if (!o || o->op != "rope" || pstr(o, "mode") != "imrope") continue;
        ++n;
        CHECK(i.opd[1].raw == rope);
        CHECK_EQ(i.opd[1].rows, 3);
        CHECK_EQ(i.opd[1].cols, 64);
    }
    CHECK_EQ(n, 4);

    /* A text-only step on the same program: no scatter, one position per token. */
    static const int32_t pos[] = { 90, 91, 92, 93 };
    RadBatch text = decode_batch();
    text.positions = pos;
    RadCtx ct{ .b = &b, .batch = &text };
    qwen35_bf16::step(&ct, &text);
    CHECK_LINT(b, &ct);
    CHECK(ct.step_fail.empty());
    CHECK(issued_op(b, ct, "scatter_rows") == nullptr);
    for (const auto& i : ct.issues) {
        const RecOp* o = b.op_of(i.op);
        if (!o || o->op != "rope") continue;
        CHECK(i.opd[1].raw == pos);
        CHECK_EQ(i.opd[1].cols, 0);
    }
}

namespace {
/* The Flash-Next fixture with its vision tower and its interleaved M-RoPE. */
std::vector<const char*> q4vl_keys, q4vl_vals;
RadModelMeta qwen4exp_vl_meta() {
    RadModelMeta m = qwen4exp_meta();
    if (q4vl_keys.empty()) {
        q4vl_keys.assign(m.kv_key, m.kv_key + m.n_kv);
        q4vl_vals.assign(m.kv_val, m.kv_val + m.n_kv);
        for (int i = 7; i < (int)(sizeof(k_vl_keys) / sizeof(k_vl_keys[0])); ++i) {
            q4vl_keys.push_back(k_vl_keys[i]);
            /* Flash-Next's tower emits rows at its width 2560, not the 27B's 5120. */
            q4vl_vals.push_back(std::strcmp(k_vl_keys[i], "vision_config.out_hidden_size") ? k_vl_vals[i]
                                                                                           : "2560");
        }
    }
    m.n_kv   = (int)q4vl_keys.size();
    m.kv_key = q4vl_keys.data();
    m.kv_val = q4vl_vals.data();
    return m;
}
}  /* namespace */

TEST(a_sizing_declare_with_a_vision_tower_is_the_same_graph) {
    RadModelMeta meta = qwen4exp_vl_meta();
    /* A serving configuration (fp8 cache, lossy wire, three MTP rounds) at levels below a
     * 2048-token chunk. */
    for (int world : { 1, 2 }) for (int lossy : { 0, 1 }) for (int64_t R : { 256, 1024 }) {
        for (int rank = 0; rank < world; ++rank) {
            RadBuildCtx c = build_ctx(rank, world);
            c.max_tok  = 2048;
            c.max_seqs = 8;
            c.max_spec = 3;
            c.max_ctx  = 200000;
            c.max_enc_patches = 16384;
            c.kv_dtype = RAD_F8E4M3;
            c.tp_wire_lossy = lossy;
            c.tp_wire_min_bytes = 64 * 1024;
            RadBuilder full;
            REQUIRE_EQ(qwen4exp_fp8::declare(&full, &meta, &c), RAD_OK);
            CHECK_EQ(full.encoder_name, std::string("vision"));
            RadBuildCtx p = c;
            p.max_tok = R;
            p.shape_probe = 1;
            RadBuilder small;
            REQUIRE_EQ(qwen4exp_fp8::declare(&small, &meta, &p), RAD_OK);
            CHECK(small.encoder_name.empty());
            size_t first = 0;
            while (first < full.ops.size() && first < small.ops.size() &&
                   full.ops[first].op == small.ops[first].op)
                ++first;
            if (first < full.ops.size() || first < small.ops.size())
                fprintf(stderr, "    world %d rank %d lossy %d R %lld: %zu ops vs %zu, first "
                                "difference at %zu ('%s' vs '%s')\n", world, rank, lossy,
                        (long long)R, full.ops.size(), small.ops.size(),
                        first, first < full.ops.size() ? full.ops[first].op.c_str() : "-",
                        first < small.ops.size() ? small.ops[first].op.c_str() : "-");
            REQUIRE_EQ(small.ops.size(), full.ops.size());
            CHECK_EQ(first, full.ops.size());
            /* Every fixed parameter matches the real declare's, the tower's included: a level runs
             * the real declare's kernels, so only ranged row counts may differ. */
            int fixed_differ = 0;
            for (size_t i = 0; i < full.ops.size(); ++i) {
                REQUIRE_EQ(small.ops[i].p.size(), full.ops[i].p.size());
                for (size_t k = 0; k < full.ops[i].p.size(); ++k) {
                    const RecParam& a = full.ops[i].p[k];
                    const RecParam& q = small.ops[i].p[k];
                    if (a.kind == RAD_P_RANGE) continue;
                    const bool cap = (full.ops[i].op == "qsa_work" && a.key == "work") ||
                                     (full.ops[i].op == "attn_paged_gate_quant" &&
                                      a.key == "max_seqs");
                    if (cap && q.ival <= a.ival) continue;
                    if (a.ival != q.ival || a.sval != q.sval || a.dval != q.dval) {
                        fprintf(stderr, "    op %zu '%s' param '%s': %lld real, %lld at %lld\n", i,
                                full.ops[i].op.c_str(), a.key.c_str(), a.ival, q.ival,
                                (long long)R);
                        ++fixed_differ;
                    }
                }
            }
            CHECK_EQ(fixed_differ, 0);
            /* A level shrinks only a buffer's rows; other extents stay the real declare's. */
            REQUIRE_EQ(small.bufs.size(), full.bufs.size());
            for (size_t i = 0; i < full.bufs.size(); ++i) {
                const RadBufDecl& f = full.bufs[i].d;
                const RadBufDecl& q = small.bufs[i].d;
                CHECK_EQ(q.rank, f.rank);
                for (uint32_t d = 1; d < f.rank && d < q.rank; ++d) {
                    if (q.shape[d] != f.shape[d])
                        fprintf(stderr, "    '%s' extent %u: %lld real, %lld at %lld\n",
                                full.bufs[i].name.c_str(), d, (long long)f.shape[d],
                                (long long)q.shape[d], (long long)R);
                    CHECK_EQ(q.shape[d], f.shape[d]);
                }
                CHECK(q.shape[0] <= f.shape[0]);
            }
        }
    }
}

/* CalibExperts stores each expert's Gram as a packed upper triangle; a packing slip could move
 * entries between experts or transpose them while every result stays k x k and symmetric. Two
 * passes of distinct symmetric Grams go in, and each expert must come back as the sum of its own
 * two, entry for entry, at an awkward width. */
TEST(the_per_expert_tap_keeps_each_experts_gram_whole) {
    const int64_t ne = 5, k = 7;
    rad::arch::CalibExperts ce;
    ce.init(ne, k);
    std::vector<float> want((size_t)(ne * k * k), 0.0f);
    for (int pass = 0; pass < 2; ++pass) {
        std::vector<float> full((size_t)(ne * k * k));
        for (int64_t e = 0; e < ne; ++e)
            for (int64_t i = 0; i < k; ++i)
                for (int64_t j = 0; j < k; ++j) {
                    const int64_t lo = std::min(i, j), hi = std::max(i, j);
                    const float v = (float)(1000 * e + 31 * lo + hi + 7 * pass);
                    full[(size_t)((e * k + i) * k + j)] = v;
                    want[(size_t)((e * k + i) * k + j)] += v;
                }
        ce.add(full.data());
    }
    std::vector<float> got((size_t)(k * k));
    int bad = 0;
    for (int64_t e = 0; e < ne; ++e) {
        ce.unpack(e, got.data());
        for (int64_t x = 0; x < k * k; ++x) bad += got[(size_t)x] != want[(size_t)(e * k * k + x)];
    }
    CHECK_EQ(bad, 0);
}

/* ==================================================================== LFM2 */
namespace {

/* d1-3B's graph cut to four layers: conv, conv, attention, conv, an irregular schedule the plugin
 * must read, not derive. A VL checkpoint (it has a text_config) with a tied head. */
const char* k_lfm2_keys[] = { "text_config.num_hidden_layers", "layer_types", "conv_L_cache",
                              "tie_word_embeddings" };
const char* k_lfm2_vals[] = { "4", "conv conv full_attention conv", "3", "true" };

RadModelMeta lfm2_meta(int n_kv = 4) {
    RadModelMeta m{};
    m.arch_id = "lfm2"; m.name = "test-lfm2"; m.quant = "";
    m.n_layers = 4; m.n_embd = 64; m.n_head = 4; m.n_head_kv = 1; m.head_dim = 16;
    m.n_ff = 128; m.n_vocab = 512; m.n_ctx_train = 32768;
    m.rms_eps = 1e-5f; m.rope_theta = 1e6f; m.rope_scale = 1.0f;
    m.n_kv = n_kv; m.kv_key = k_lfm2_keys; m.kv_val = k_lfm2_vals;
    return m;
}

std::vector<std::string> lfm2_layer_ops(bool attention) {
    std::vector<std::string> v = { "rmsnorm" };
    if (attention) {
        for (const char* s : { "gemm_nt", "gemm_nt", "gemm_nt", "rmsnorm", "rmsnorm", "rope",
                               "rope", "attn_paged", "kv_store", "gemm_nt" })
            v.push_back(s);
    } else {
        for (const char* s : { "gemm_nt", "mul", "causal_conv1d", "mul", "gemm_nt" })
            v.push_back(s);
    }
    v.push_back("add");
    append(v, mlp_ops(false));
    return v;
}

}  /* namespace */

TEST(lfm2_declares_the_checkpoints_own_schedule) {
    RadBuilder b;
    RadModelMeta meta = lfm2_meta();
    RadBuildCtx ctx = build_ctx();
    CHECK_OK(lfm2_bf16::declare(&b, &meta, &ctx));

    std::vector<std::string> want = { "embed_lookup" };
    for (bool att : { false, false, true, false }) append(want, lfm2_layer_ops(att));
    for (const char* s : { "gather_rows", "rmsnorm", "logits_gemm" }) want.push_back(s);
    CHECK_EQ(join(op_names(b)), join(want));

    /* Two state groups: paged KV bound to the attention layer, and a conv window (n_embd channels,
     * the checkpoint's width) bound to the other three. */
    REQUIRE(b.kv_groups.size() == 2);
    CHECK_EQ(b.kv_groups[0].d.kind, (int)RAD_KV_FULL);
    CHECK_EQ(b.kv_groups[1].d.kind, (int)RAD_KV_CONV);
    CHECK_EQ(b.kv_groups[1].d.conv_width, (int64_t)3);
    CHECK_EQ(b.kv_groups[1].d.n_head_kv * b.kv_groups[1].d.head_dim, (int64_t)64);
    REQUIRE(b.binds.size() == 4);
    for (const auto& bd : b.binds)
        CHECK_EQ(bd.second, bd.first == 2 ? (rad_kvgroup)1 : (rad_kvgroup)2);

    /* A plain RMSNorm at the checkpoint's epsilon, NeoX over the whole head. */
    const RecOp* norm = find_op(b, "rmsnorm");
    CHECK_NEAR(pflt(norm, "eps"), 1e-5, 1e-9);
    CHECK_EQ(param(norm, "wadd") ? pflt(norm, "wadd") : 0.0, 0.0);
    CHECK_EQ(pstr(find_op(b, "rope"), "mode"), "neox");
    CHECK_EQ(pint(find_op(b, "rope"), "rotary_dim"), 16);
    CHECK_EQ(pint(find_op(b, "attn_paged"), "gqa"), 4);
    CHECK_EQ(pint(find_op(b, "causal_conv1d"), "width"), 3);

    /* The VL checkpoint's names, and a tied head read from the embedding when no lm_head
     * exists. */
    const RecMap* in = find_map(b, "blk.0.conv_in.weight");
    REQUIRE(in != nullptr);
    CHECK_EQ(in->src[0], "model.language_model.layers.0.conv.in_proj.weight");
    const RecMap* head = find_map(b, "output.weight");
    REQUIRE(head != nullptr && head->src.size() == 2);
    CHECK_EQ(head->src[0], "lm_head.weight");
    CHECK_EQ(head->src[1], "model.language_model.embed_tokens.weight");
}

TEST(lfm2_text_checkpoint_keeps_its_names_under_model) {
    /* No text_config: a text-only LFM2, with layers under `model.` */
    static const char* keys[] = { "layer_types", "conv_L_cache" };
    static const char* vals[] = { "conv conv full_attention conv", "3" };
    RadBuilder b;
    RadModelMeta meta = lfm2_meta();
    meta.n_kv = 2; meta.kv_key = keys; meta.kv_val = vals;
    RadBuildCtx ctx = build_ctx();
    CHECK_OK(lfm2_bf16::declare(&b, &meta, &ctx));
    const RecMap* q = find_map(b, "blk.2.attn_q.weight");
    REQUIRE(q != nullptr);
    CHECK_EQ(q->src[0], "model.layers.2.self_attn.q_proj.weight");
}

TEST(lfm2_refuses_what_it_does_not_serve) {
    {
        RadBuilder b;
        RadModelMeta meta = lfm2_meta(1);   /* no layer_types */
        RadBuildCtx ctx = build_ctx();
        CHECK(lfm2_bf16::declare(&b, &meta, &ctx) < 0);
    }
    {
        RadBuilder b;
        RadModelMeta meta = lfm2_meta();
        RadBuildCtx ctx = build_ctx(0, 2);
        CHECK_EQ(lfm2_bf16::declare(&b, &meta, &ctx), RAD_E_UNSUPPORTED);
    }
    {
        static const char* keys[] = { "layer_types", "conv_L_cache" };
        static const char* vals[] = { "conv sliding_attention conv conv", "3" };
        RadBuilder b;
        RadModelMeta meta = lfm2_meta();
        meta.n_kv = 2; meta.kv_key = keys; meta.kv_val = vals;
        RadBuildCtx ctx = build_ctx();
        CHECK_EQ(lfm2_bf16::declare(&b, &meta, &ctx), RAD_E_UNSUPPORTED);
    }
}

TEST(lfm2_issues_the_convolution_once_per_half_of_a_mixed_step) {
    RadBuilder b;
    RadModelMeta meta = lfm2_meta();
    RadBuildCtx ctx = build_ctx();
    CHECK_OK(lfm2_bf16::declare(&b, &meta, &ctx));

    /* Two decode rows (a verify of 3 and a plain decode) and one prefill chunk of 5. */
    static const int32_t tok[9]  = { 1, 2, 3, 4, 5, 6, 7, 8, 9 };
    static const int32_t pos[9]  = { 40, 41, 42, 30, 0, 1, 2, 3, 4 };
    static const int32_t cu[4]   = { 0, 3, 4, 9 };
    static const int32_t ctxl[3] = { 40, 30, 0 };
    static const int32_t nacc[3] = { 2, 1, 1 };
    static const int32_t out[3]  = { 2, 3, 8 };
    static int32_t cv_idx[3] = { 4, 5, 6 };
    RadKVGroupBatch kv[2]{};
    kv[0].group = lfm2_bf16::g_model[0].kv_attn;
    kv[1].group = lfm2_bf16::g_model[0].kv_conv; kv[1].state_index = cv_idx;
    RadBatch mixed{};
    mixed.phase = RAD_PHASE_MIXED;
    mixed.n_tok = 9; mixed.n_seq = 3; mixed.max_q_len = 5;
    mixed.n_seq_decode = 2; mixed.n_tok_decode = 4;
    mixed.token_ids = tok; mixed.positions = pos; mixed.cu_seqlens = cu; mixed.ctx_lens = ctxl;
    mixed.num_accepted = nacc; mixed.n_out = 3; mixed.out_ids = out;
    mixed.n_kv_groups = 2; mixed.kv = kv;

    RadCtx c{ .b = &b, .batch = &mixed };
    lfm2_bf16::step(&c, &mixed);
    CHECK_LINT(b, &c);
    CHECK(c.step_fail.empty());

    int convs = 0;
    for (const auto& i : c.issues) {
        const RecOp* o = b.op_of(i.op);
        if (!o) continue;
        if (o->op == "mul" && i.opd.size() == 3 && i.opd[0].cols > 0) {
            /* conv(B * x): the B and x thirds of the [B | C | x] row as column slices. */
            if (i.opd[1].cols > 0) {
                CHECK_EQ(i.opd[0].offset, (int64_t)0);
                CHECK_EQ(i.opd[1].offset, (int64_t)128);
            } else {
                CHECK_EQ(i.opd[0].offset, (int64_t)64);    /* C * conv(...) */
            }
            CHECK_EQ(i.opd[0].cols, (int64_t)64);
        }
        if (o->op != "causal_conv1d") continue;
        REQUIRE(i.opd.size() == 8);
        if (convs++ % 2 == 0) {      /* the decode half */
            CHECK_EQ(i.n, (int64_t)4);
            CHECK_EQ(i.opd[0].rows, (int64_t)4);
            CHECK(i.opd[3].raw == (void*)cu);         CHECK_EQ(i.opd[3].rows, (int64_t)3);
            CHECK(i.opd[4].raw == (void*)cv_idx);
            CHECK(i.opd[6].raw == (void*)nacc);
        } else {                     /* prefill half: per-seq arrays past the decode seqs */
            CHECK_EQ(i.n, (int64_t)9);
            CHECK_EQ(i.opd[0].rows, (int64_t)9);
            CHECK(i.opd[3].raw == (void*)(cu + 2));   CHECK_EQ(i.opd[3].rows, (int64_t)2);
            CHECK(i.opd[4].raw == (void*)(cv_idx + 2));
            CHECK(i.opd[5].raw == (void*)(ctxl + 2));
            CHECK_EQ(i.opd[6].kind, (uint8_t)RAD_OPK_NONE);
        }
    }
    CHECK_EQ(convs, 6);   /* three conv layers, two halves each */
}

/* ==================================================================== LFM2-VL's tower */
namespace {

/* lfm2_meta's text model with a SigLIP2 tower cut to two layers of 64 (four heads of 16), d1's
 * patch, position table and unshuffle, and a 96-wide projector. */
const char* k_lfm2vl_keys[] = { "text_config.num_hidden_layers", "layer_types", "conv_L_cache",
                                "tie_word_embeddings", "vision_config.model_type",
                                "vision_config.num_hidden_layers", "vision_config.hidden_size",
                                "vision_config.num_attention_heads", "vision_config.intermediate_size",
                                "vision_config.patch_size", "vision_config.num_patches",
                                "vision_config.hidden_act", "vision_config.vision_use_head",
                                "downsample_factor", "projector_hidden_size",
                                "projector_hidden_act", "projector_bias",
                                "projector_use_layernorm", "max_num_patches" };
const char* k_lfm2vl_vals[] = { "4", "conv conv full_attention conv", "3", "true",
                                "siglip2_vision_model", "2", "64", "4", "128", "16", "256",
                                "gelu_pytorch_tanh", "false", "2", "96", "gelu", "true", "false",
                                "1024" };
const char* k_lfm2vl_norm_vals[] = { "4", "conv conv full_attention conv", "3", "true",
                                     "siglip2_vision_model", "2", "64", "4", "128", "16", "256",
                                     "gelu_pytorch_tanh", "false", "2", "96", "gelu", "true", "true",
                                     "1024" };

RadModelMeta lfm2vl_meta(bool proj_norm = false) {
    RadModelMeta m = lfm2_meta();
    m.n_kv = (int)(sizeof(k_lfm2vl_keys) / sizeof(k_lfm2vl_keys[0]));
    m.kv_key = k_lfm2vl_keys;
    m.kv_val = proj_norm ? k_lfm2vl_norm_vals : k_lfm2vl_vals;
    return m;
}

}  /* namespace */

TEST(lfm2_vl_declares_the_siglip2_tower_and_its_projector) {
    {   /* No encoder budget: text only, whatever the container carries. */
        RadBuilder b;
        RadModelMeta meta = lfm2vl_meta();
        RadBuildCtx ctx = vl_ctx(0);
        CHECK_OK(lfm2_bf16::declare(&b, &meta, &ctx));
        CHECK(b.encoder_name.empty());
        CHECK_EQ(count_op(b, "scatter_rows"), 0);
    }
    RadBuilder b;
    RadModelMeta meta = lfm2vl_meta();
    RadBuildCtx ctx = vl_ctx(1024);
    CHECK_OK(lfm2_bf16::declare(&b, &meta, &ctx));
    CHECK_EQ(b.encoder_name, std::string("vision"));
    CHECK_EQ(b.encoder.modalities, (uint32_t)RAD_MM_IMAGE);
    CHECK_EQ(b.encoder.patch_dim, 3 * 16 * 16);
    CHECK_EQ(b.encoder.merge, 4);
    CHECK_EQ(b.encoder.max_patches, 1024);
    CHECK_EQ(b.encoder.n_embd, 64);
    CHECK_EQ(count_op(b, "scatter_rows"), 1);

    /* 3 stem + 12 per layer + 2 post-norm + 4 projector, each mapped from the checkpoint. */
    int nv = 0;
    for (const auto& w : b.weights)
        if (w.name.rfind("v.", 0) == 0) {
            ++nv;
            CHECK(find_map(b, w.name) != nullptr);
        }
    CHECK_EQ(nv, 3 + 2 * 12 + 2 + 4);
    /* q, k and v are concatenated at convert into one QKV weight and bias. */
    const RecMap* qkv = find_map(b, "v.blk.1.qkv.weight");
    REQUIRE(qkv != nullptr);
    REQUIRE_EQ(qkv->src.size(), (size_t)3);
    CHECK_EQ(qkv->src[0],
             std::string("model.vision_tower.vision_model.encoder.layers.1.self_attn.q_proj.weight"));
    CHECK_EQ(qkv->src[2],
             std::string("model.vision_tower.vision_model.encoder.layers.1.self_attn.v_proj.weight"));
    const RecMap* l1 = find_map(b, "v.proj.fc1.weight");
    REQUIRE(l1 != nullptr);
    CHECK_EQ(l1->src[0], std::string("model.multi_modal_projector.linear_1.weight"));
    /* The projector reads four patches as one row; it uses erf GELU, the tower tanh GELU. */
    const RecWeight* w1 = find_w(b, "v.proj.fc1.weight");
    REQUIRE(w1 != nullptr);
    CHECK_EQ(w1->d.shape[0], 96);
    CHECK_EQ(w1->d.shape[1], 4 * 64);
    int tanh = 0, erf = 0;
    for (const auto& o : b.ops)
        if (o.op == "gemm_nt_bias") {
            tanh += pstr(&o, "act") == "gelu_tanh";
            erf += pstr(&o, "act") == "gelu";
        }
    CHECK_EQ(tanh, 2);
    CHECK_EQ(erf, 1);
    /* No rotary in the tower; the position table is resampled as torch's antialiased bilinear,
     * and attention's block height is bounded by the longest segment the processor makes. */
    CHECK_EQ(count_op(b, "grid_embed_aa"), 1);
    CHECK_EQ(count_op(b, "grid_embed"), 0);
    CHECK_EQ(pint(find_op(b, "attn_dense"), "max_seqlen"), 1024);
    CHECK_EQ(pint(find_op(b, "attn_dense"), "head_dim"), 16);

    /* A projector with a layer norm declares it over the merged rows. */
    RadBuilder bn;
    RadModelMeta mn = lfm2vl_meta(true);
    CHECK_OK(lfm2_bf16::declare(&bn, &mn, &ctx));
    CHECK(find_w(bn, "v.proj.norm.weight") != nullptr);
    CHECK_EQ(count_op(bn, "layernorm"), count_op(b, "layernorm") + 1);
}

TEST(lfm2_vl_encoder_pass_and_the_rows_it_scatters) {
    RadBuilder b;
    RadModelMeta meta = lfm2vl_meta();
    RadBuildCtx ctx = vl_ctx(1024);
    CHECK_OK(lfm2_bf16::declare(&b, &meta, &ctx));

    /* A tile of 8 x 8 patches and a thumbnail of 4 x 8. */
    static uint16_t pix[96 * 768];
    static int32_t  coord[4 * 96];
    static const int32_t cu[] = { 0, 64, 96 };
    RadBatch batch{};
    batch.enc         = RAD_MM_IMAGE;
    batch.enc_n_patch = 96;
    batch.enc_pixels  = pix;
    batch.enc_coord   = coord;
    batch.enc_n_seg   = 2;
    batch.enc_cu      = cu;
    batch.enc_max_seg = 64;
    RadCtx c{ .b = &b, .batch = &batch };
    lfm2_bf16::step(&c, &batch);
    CHECK_LINT(b, &c);
    CHECK(c.step_fail.empty());
    std::vector<std::string> want = { "gemm_nt_bias", "grid_embed_aa" };
    for (int l = 0; l < 2; ++l)
        for (const char* s : { "layernorm", "gemm_nt_bias", "attn_dense", "gemm_nt_bias",
                               "layernorm", "gemm_nt_bias", "gemm_nt_bias" })
            want.push_back(s);
    for (const char* s : { "layernorm", "gemm_nt_bias", "gemm_nt_bias" }) want.push_back(s);
    CHECK_EQ(join(issued_ops(b, c)), join(want));
    /* The position resample reads all four coordinate planes; the projector runs on merged rows. */
    const RecIssue* ge = issue_of(b, c, "grid_embed_aa");
    REQUIRE(ge != nullptr);
    CHECK(ge->opd[2].raw == coord);
    CHECK_EQ(ge->opd[2].rows, 4);
    CHECK_EQ(c.issues.back().n, 24);
    const RecIssue* at = issue_of(b, c, "attn_dense");
    REQUIRE(at != nullptr);
    CHECK_EQ(at->opd[1].offset, 64);
    CHECK_EQ(at->opd[2].offset, 128);
    CHECK(at->opd[3].raw == cu);

    /* A text step with media rows scatters them over the lookup before the first layer. */
    static const int32_t tok[4] = { 1, 2, 3, 4 }, pos[4] = { 0, 1, 2, 3 }, cus[2] = { 0, 4 };
    static const int32_t ctxl[1] = { 0 }, out[1] = { 3 }, rows[2] = { 1, 2 };
    static uint16_t embd[2 * 64];
    static int32_t cv_idx[1] = { 0 };
    RadKVGroupBatch kv[2]{};
    kv[0].group = lfm2_bf16::g_model[0].kv_attn;
    kv[1].group = lfm2_bf16::g_model[0].kv_conv; kv[1].state_index = cv_idx;
    RadBatch tb{};
    tb.phase = RAD_PHASE_PREFILL;
    tb.n_tok = 4; tb.n_seq = 1; tb.max_q_len = 4;
    tb.token_ids = tok; tb.positions = pos; tb.cu_seqlens = cus; tb.ctx_lens = ctxl;
    tb.n_out = 1; tb.out_ids = out; tb.n_kv_groups = 2; tb.kv = kv;
    tb.n_mm_rows = 2; tb.mm_rows = rows; tb.mm_embd = embd;
    RadCtx ct{ .b = &b, .batch = &tb };
    lfm2_bf16::step(&ct, &tb);
    CHECK_LINT(b, &ct);
    CHECK(ct.step_fail.empty());
    const std::vector<std::string> ops = issued_ops(b, ct);
    REQUIRE(ops.size() > 2);
    CHECK_EQ(ops[0], std::string("embed_lookup"));
    CHECK_EQ(ops[1], std::string("scatter_rows"));
}

/* =============================== gemma4 dense: two attention classes =========================== */

TEST(declare_is_a_gemma4_trunk) {
    RadBuilder b;
    RadModelMeta meta = gemma4_meta();
    RadBuildCtx  ctx  = build_ctx();
    CHECK_OK(gemma4_bf16::declare(&b, &meta, &ctx));

    /* 12 layers: 10 sliding + 2 global (layers 5 and 11). */
    CHECK_EQ(count_op(b, "attn_paged"),  12);
    CHECK_EQ(count_op(b, "kv_store"),    12);
    CHECK_EQ(count_op(b, "rmsnorm_p"),   12);   /* weightless value norm, one per layer */
    CHECK_EQ(count_op(b, "rope"),        24);   /* q + k */
    CHECK_EQ(count_op(b, "gelu_mul"),    12);   /* Gemma's FFN activation, not SwiGLU */
    CHECK_EQ(count_op(b, "silu_mul"),     0);
    CHECK_EQ(count_op(b, "gemm_nt"),     70);   /* 10x6 sliding (q,k,v,o,gu,down) + 2x5 global (no v) */
    CHECK_EQ(count_op(b, "rmsnorm"),     73);   /* 4 attn + 2 ffn per layer, plus the final norm */
    CHECK_EQ(count_op(b, "add"),         24);   /* the two GemmaRMSNorm residuals per layer */
    CHECK_EQ(count_op(b, "embed_lookup"),  1);
    CHECK_EQ(count_op(b, "gather_rows"),   1);
    CHECK_EQ(count_op(b, "logits_gemm"),   1);
    CHECK_EQ(count_op(b, "logit_softcap"), 1);
    CHECK_EQ(count_op(b, "all_reduce"),    0);  /* world 1 */
}

/* The plugin states the reply format, since template analysis cannot read Gemma 4's template.
 * Calls are the object family (a name and one object in Gemma's dialect) and the thought channel
 * may reopen after the answer. chatparse_gemma4_test covers parsing; this checks the export. */
TEST(gemma4_declares_its_reply_format) {
    RadModelMeta meta = gemma4_meta();
    const RadChatFormat* f = gemma4_bf16::chat_format(&meta);
    REQUIRE(f != nullptr);
    CHECK_EQ(f->struct_size, (uint32_t)sizeof(RadChatFormat));
    CHECK_EQ(f->args_form, (uint32_t)RAD_CHAT_ARGS_OBJECT);
    CHECK_EQ(std::string(f->call_open), std::string("<|tool_call>"));
    CHECK_EQ(std::string(f->call_close), std::string("<tool_call|>"));
    CHECK_EQ(std::string(f->name_prefix), std::string("call:"));
    CHECK_EQ(std::string(f->name_suffix), std::string(""));
    CHECK_EQ(std::string(f->object_string_open), std::string("<|\"|>"));
    CHECK_EQ(std::string(f->object_string_close), std::string("<|\"|>"));
    CHECK(f->object_bare_keys != 0);
    CHECK_EQ(std::string(f->reasoning_open), std::string("<|channel>thought\n"));
    CHECK_EQ(std::string(f->reasoning_close), std::string("<channel|>"));
    CHECK(f->reasoning_anywhere != 0);
    REQUIRE(f->n_triggers == 1);
    CHECK_EQ(std::string(f->triggers[0]), std::string("<|tool_call>"));
    CHECK(f->arg_prefix == nullptr && f->arg_value_suffix == nullptr);
}

TEST(gemma4_declares_two_kv_groups_and_binds_the_schedule) {
    RadBuilder b;
    RadModelMeta meta = gemma4_meta();
    RadBuildCtx  ctx  = build_ctx();
    CHECK_OK(gemma4_bf16::declare(&b, &meta, &ctx));

    CHECK_EQ((int)b.kv_groups.size(), 2);
    CHECK_EQ((int)b.kv_groups[0].d.kind, RAD_KV_WINDOW);   /* sliding, declared first */
    CHECK_EQ(b.kv_groups[0].d.window,    1024);
    CHECK_EQ(b.kv_groups[0].d.n_head_kv, 16);
    CHECK_EQ(b.kv_groups[0].d.head_dim,  256);
    CHECK_EQ((int)b.kv_groups[1].d.kind, RAD_KV_FULL);     /* global */
    CHECK_EQ(b.kv_groups[1].d.window,    0);
    CHECK_EQ(b.kv_groups[1].d.n_head_kv, 4);
    CHECK_EQ(b.kv_groups[1].d.head_dim,  512);

    /* (l + 1) % 6 == 0 gives global layers 5 and 11; group 1 is sliding (declared first), 2
     * global. */
    CHECK_EQ((int)b.binds.size(), 12);
    for (int l = 0; l < 12; ++l) {
        const bool want_global = ((l + 1) % 6 == 0);
        CHECK_EQ((int)b.binds[(size_t)l].first, l);
        CHECK_EQ((int)b.binds[(size_t)l].second, want_global ? 2 : 1);
    }
}

TEST(gemma4_attention_shapes_for_both_classes) {
    RadBuilder b;
    RadModelMeta meta = gemma4_meta();
    RadBuildCtx  ctx  = build_ctx();
    CHECK_OK(gemma4_bf16::declare(&b, &meta, &ctx));

    /* layer 0 (sliding): q 32x256, k/v 16x256, windowed. */
    {
        const RecOp* at = find_op(b, "attn_paged", 0);
        CHECK(at != nullptr);
        CHECK_EQ(pint(at, "head_dim"), 256);
        CHECK_EQ(pint(at, "gqa"),      2);
        CHECK_EQ(pint(at, "window"),   1024);
        CHECK_EQ(pint(at, "n_head"),   32);
        CHECK_EQ(pint(at, "block_size"), 16);
        CHECK_EQ(pstr(at, "q_dtype"), std::string("bf16"));
    }
    /* layer 5 (the first global, (5+1)%6==0): q 32x512, k 4x512, windowless. */
    {
        const RecOp* at = find_op(b, "attn_paged", 5);
        CHECK(at != nullptr);
        CHECK_EQ(pint(at, "head_dim"), 512);
        CHECK_EQ(pint(at, "gqa"),      8);
        CHECK_EQ(pint(at, "window"),   0);
        CHECK_EQ(pint(at, "n_head"),   32);
    }

    /* rope: sliding is standard over the full 256 (theta 1e4); global is proportional over 128 of
     * 512 (the "head" ladder, theta 1e6). Ropes are declared q, k per layer, so index =
     * 2*layer + (q ? 0 : 1). */
    {
        const RecOp* r  = find_op(b, "rope", 0);   /* layer 0, q */
        CHECK(r != nullptr);
        CHECK_EQ(pint(r, "head_dim"),   256);
        CHECK_EQ(pint(r, "rotary_dim"), 256);
        CHECK_NEAR(pflt(r, "theta"),    1e4, 1.0);
        CHECK_EQ(pstr(r, "ladder"), std::string("rotary"));
        CHECK_EQ(pint(r, "n_head"),    32);

        const RecOp* rg = find_op(b, "rope", 10);  /* layer 5, q */
        CHECK(rg != nullptr);
        CHECK_EQ(pint(rg, "head_dim"),   512);
        CHECK_EQ(pint(rg, "rotary_dim"), 128);
        CHECK_NEAR(pflt(rg, "theta"),    1e6, 1.0);
        CHECK_EQ(pstr(rg, "ladder"), std::string("head"));
        CHECK_EQ(pint(rg, "n_head"),    32);

        /* the k rope is a separate buffer, run as a q-only form over the k heads: n_head is the
         * count to rotate (n_head_kv = 4) and n_head_kv is 0. */
        const RecOp* rkg = find_op(b, "rope", 11); /* layer 5, k */
        CHECK(rkg != nullptr);
        CHECK_EQ(pint(rkg, "n_head"),    4);
        CHECK_EQ(pint(rkg, "n_head_kv"), 0);
    }

    /* the weightless value norm reads 256 dims on a sliding layer and 512 on a global one, and
     * the global layer has no v_proj (v is k). */
    {
        const RecOp* vn  = find_op(b, "rmsnorm_p", 0);   /* layer 0 */
        CHECK(vn != nullptr);
        CHECK_EQ(pint(vn, "n"), 256);
        const RecOp* vng = find_op(b, "rmsnorm_p", 5);   /* layer 5 */
        CHECK(vng != nullptr);
        CHECK_EQ(pint(vng, "n"), 512);
    }
    CHECK(find_w(b, "blk.0.attn_v.weight") != nullptr);  /* sliding has a v projection */
    CHECK(find_w(b, "blk.5.attn_v.weight") == nullptr);  /* global: k == v */
    CHECK(find_w(b, "blk.5.attn_q.weight") != nullptr);
    CHECK(find_w(b, "blk.5.attn_o.weight") != nullptr);
}

TEST(gemma4_final_norm_softcap_and_vocab) {
    RadBuilder b;
    RadModelMeta meta = gemma4_meta();
    RadBuildCtx  ctx  = build_ctx();
    CHECK_OK(gemma4_bf16::declare(&b, &meta, &ctx));

    const RecOp* cap = find_op(b, "logit_softcap", 0);
    CHECK(cap != nullptr);
    CHECK_EQ(pint(cap, "n"), 262144);
    CHECK_NEAR(pflt(cap, "cap"), 30.0, 1e-9);
    CHECK_EQ(pstr(cap, "dtype"), std::string("f32"));

    const RecOp* lg = find_op(b, "logits_gemm", 0);
    CHECK(lg != nullptr);
    CHECK_EQ(pint(lg, "n_vocab"), 262144);
    CHECK_EQ(pint(lg, "n_embd"),  5376);

    const RecWeight* emb = find_w(b, "token_embd.weight");
    CHECK(emb != nullptr);
    CHECK_EQ(emb->d.shape[0], 262144);
    CHECK_EQ(emb->d.shape[1], 5376);
    CHECK_EQ(emb->d.shard, RAD_SHARD_ROW);

    /* lm_head is tied to the embedding. */
    CHECK(find_map(b, "output.weight") != nullptr);

    /* The embedding is scaled by sqrt(n_embd) at lookup (as the reference's build_inp_embd and
     * HF's ScaledWordEmbedding do), not baked into the table, so the table stays tied to the head.
     * HF multiplies by `embed_scale.to(bf16)`, so sqrt(5376) = 73.32 is 73.5. */
    const RecOp* el = find_op(b, "embed_lookup", 0);
    CHECK(el != nullptr);
    CHECK_NEAR(pflt(el, "mult"), 73.5, 1e-9);
}

TEST(gemma4_every_op_declares_the_buffers_it_touches) {
    RadBuilder b;
    RadModelMeta meta = gemma4_meta();
    RadBuildCtx  ctx  = build_ctx();
    CHECK_OK(gemma4_bf16::declare(&b, &meta, &ctx));
    for (const auto& o : b.ops)
        if (o.resolved) CHECK(o.uses_declared);
}

/* attention_k_eq_v decides whether global layers have a v_proj: 1 (31B) reads V from K, 0 declares
 * one, and absence is refused. */
TEST(gemma4_global_v_follows_attention_k_eq_v) {
    for (const char* v : { "1", "0", "" }) {
        std::vector<const char*> keys(std::begin(k_gem4_keys), std::end(k_gem4_keys));
        std::vector<const char*> vals(std::begin(k_gem4_vals), std::end(k_gem4_vals));
        if (*v) vals.back() = v;
        else { keys.pop_back(); vals.pop_back(); }
        RadModelMeta meta = gemma4_meta();
        meta.n_kv = (int)keys.size(); meta.kv_key = keys.data(); meta.kv_val = vals.data();
        RadBuilder b;
        RadBuildCtx ctx = build_ctx();
        const int rc = gemma4_bf16::declare(&b, &meta, &ctx);
        if (!*v) { CHECK(rc != RAD_OK); continue; }
        CHECK_OK(rc);
        CHECK(find_w(b, "blk.0.attn_v.weight") != nullptr);              /* sliding: always */
        CHECK_EQ(find_w(b, "blk.5.attn_v.weight") != nullptr, *v == '0');  /* global */
    }
}

TEST(gemma4_every_weight_has_a_name_map) {
    RadBuilder b;
    RadModelMeta meta = gemma4_meta();
    RadBuildCtx  ctx  = build_ctx();
    CHECK_OK(gemma4_bf16::declare(&b, &meta, &ctx));
    for (const auto& w : b.weights)
        CHECK(find_map(b, w.name) != nullptr);
}

TEST(a_gemma4_decode_step_runs_both_classes_in_order) {
    RadBuilder b;
    b.refuse.insert("qkv_norm_rope");   /* force the unfused prologue; fused is checked below */
    RadModelMeta meta = gemma4_meta();
    RadBuildCtx  ctx  = build_ctx();
    CHECK_OK(gemma4_bf16::declare(&b, &meta, &ctx));

    RadBatch batch = decode_batch();
    RadCtx c{ .b = &b, .batch = &batch };
    gemma4_bf16::step(&c, &batch);
    CHECK_LINT(b, &c);

    std::vector<std::string> want = { "embed_lookup" };
    for (int l = 0; l < 12; ++l) {
        if ((l + 1) % 6 == 0) {
            /* global: k == v, so no v gemm; the weightless v norm derives from k. */
            for (const char* s : { "rmsnorm", "gemm_nt", "gemm_nt",
                                   "rmsnorm_p", "rmsnorm", "rmsnorm", "rope", "rope",
                                   "kv_store", "attn_paged", "gemm_nt", "rmsnorm", "add" })
                want.push_back(s);
        } else {
            /* sliding: a v gemm feeds the weightless v norm. */
            for (const char* s : { "rmsnorm", "gemm_nt", "gemm_nt", "gemm_nt",
                                   "rmsnorm_p", "rmsnorm", "rmsnorm", "rope", "rope",
                                   "kv_store", "attn_paged", "gemm_nt", "rmsnorm", "add" })
                want.push_back(s);
        }
        for (const char* s : { "rmsnorm", "gemm_nt", "gelu_mul", "gemm_nt", "rmsnorm", "add" })
            want.push_back(s);
        want.push_back("scale");   /* per-layer scalar over the residual stream */
    }
    for (const char* s : { "rmsnorm", "gather_rows", "logits_gemm", "logit_softcap" })
        want.push_back(s);
    CHECK_EQ(join(issued_ops(b, c)), join(want));

    /* The sampled-row ops band on n_out; the attention bands on q_len. */
    for (const auto& i : c.issues) {
        const RecOp* o = b.op_of(i.op);
        if (!o) continue;
        if (o->op == "attn_paged")
            CHECK_EQ(i.n, (int64_t)batch.max_q_len);
        else if (o->op == "gather_rows" || o->op == "logits_gemm" || o->op == "logit_softcap")
            CHECK_EQ(i.n, (int64_t)batch.n_out);
    }
}

/* The w4a8 trunk fuses its residual junctions: the first block's entry, each attention ->
 * feed-forward and each feed-forward -> next attention is one `rmsnorm_quant_i8` (post-norm, add,
 * layer scalar, norm and quantiser), and q, k and v read one quantised pair. The last layer goes to
 * the final norm unfused. Without the fused op the whole chain stays unfused, still with the shared
 * q/k/v pair. */
static std::vector<std::string> gemma4_w4_step(const std::set<std::string>& refuse, int world = 1) {
    RadBuilder b;
    const RadEncoding w4 = rad_enc_affine(RAD_I4, RAD_F16, 1, 128);
    b.encs = { { "attn_q.", w4 }, { "attn_k.", w4 }, { "attn_v.", w4 }, { "attn_o.", w4 },
               { "ffn_gate_up.", w4 }, { "ffn_down.", w4 } };
    for (const auto& r : refuse) b.refuse.insert(r);
    RadModelMeta meta = gemma4_meta();
    RadBuildCtx  ctx  = build_ctx(0, world);
    CHECK_OK(gemma4_bf16::declare(&b, &meta, &ctx));
    RadBatch batch = decode_batch();
    RadCtx c{ .b = &b, .batch = &batch, .world = world };
    gemma4_bf16::step(&c, &batch);
    CHECK_LINT(b, &c);
    return issued_ops(b, c);
}

/* At two ranks the all-reduce before each junction is folded into it; only the embedding's and
 * the last feed-forward's (which feeds the final norm, not a quantiser) remain standalone. Without
 * the fused collective, junctions keep their preceding all-reduces. */
TEST(a_gemma4_w4a8_step_at_two_ranks_folds_the_all_reduce_into_the_junction) {
    std::vector<std::string> want = { "embed_lookup", "all_reduce", "rmsnorm_quant_i8" };
    for (int l = 0; l < 12; ++l) {
        const bool global = (l + 1) % 6 == 0;
        for (int i = 0; i < (global ? 2 : 3); ++i) want.push_back("gemm_nt_q");
        for (const char* s : { "qkv_norm_rope", "kv_store", "attn_paged", "quant_act_i8", "gemm_nt_q",
                               "ar_rmsnorm_quant_i8", "gemm_nt_q", "gelu_mul_quant_i8", "gemm_nt_q" })
            want.push_back(s);
        if (l < 11) want.push_back("ar_rmsnorm_quant_i8");
        else for (const char* s : { "all_reduce", "rmsnorm", "add", "scale" }) want.push_back(s);
    }
    for (const char* s : { "rmsnorm", "gather_rows", "logits_gemm", "logit_softcap" })
        want.push_back(s);
    CHECK_EQ(join(gemma4_w4_step({}, 2)), join(want));

    std::vector<std::string> plain = { "embed_lookup", "all_reduce", "rmsnorm_quant_i8" };
    for (int l = 0; l < 12; ++l) {
        const bool global = (l + 1) % 6 == 0;
        for (int i = 0; i < (global ? 2 : 3); ++i) plain.push_back("gemm_nt_q");
        for (const char* s : { "qkv_norm_rope", "kv_store", "attn_paged", "quant_act_i8", "gemm_nt_q",
                               "all_reduce", "rmsnorm_quant_i8", "gemm_nt_q", "gelu_mul_quant_i8",
                               "gemm_nt_q", "all_reduce" })
            plain.push_back(s);
        if (l < 11) plain.push_back("rmsnorm_quant_i8");
        else for (const char* s : { "rmsnorm", "add", "scale" }) plain.push_back(s);
    }
    for (const char* s : { "rmsnorm", "gather_rows", "logits_gemm", "logit_softcap" })
        plain.push_back(s);
    CHECK_EQ(join(gemma4_w4_step({ "ar_rmsnorm_quant_i8" }, 2)), join(plain));
}

TEST(a_gemma4_w4a8_step_fuses_its_residual_junctions) {
    std::vector<std::string> want = { "embed_lookup", "rmsnorm_quant_i8" };
    for (int l = 0; l < 12; ++l) {
        const bool global = (l + 1) % 6 == 0;
        for (int i = 0; i < (global ? 2 : 3); ++i) want.push_back("gemm_nt_q");
        for (const char* s : { "qkv_norm_rope", "kv_store",
                               "attn_paged", "quant_act_i8", "gemm_nt_q", "rmsnorm_quant_i8",
                               "gemm_nt_q", "gelu_mul_quant_i8", "gemm_nt_q" })
            want.push_back(s);
        if (l < 11) want.push_back("rmsnorm_quant_i8");
        else for (const char* s : { "rmsnorm", "add", "scale" }) want.push_back(s);
    }
    for (const char* s : { "rmsnorm", "gather_rows", "logits_gemm", "logit_softcap" })
        want.push_back(s);
    CHECK_EQ(join(gemma4_w4_step({})), join(want));
}

TEST(a_gemma4_w4a8_step_without_the_fused_ops_keeps_the_chain_and_one_qkv_quantiser) {
    std::vector<std::string> want = { "embed_lookup" };
    for (int l = 0; l < 12; ++l) {
        const bool global = (l + 1) % 6 == 0;
        want.push_back("rmsnorm");
        want.push_back("quant_act_i8");
        for (int i = 0; i < (global ? 2 : 3); ++i) want.push_back("gemm_nt_q");
        for (const char* s : { "rmsnorm_p", "rmsnorm", "rmsnorm", "rope", "rope", "kv_store",
                               "attn_paged", "quant_act_i8", "gemm_nt_q", "rmsnorm", "add",
                               "rmsnorm", "quant_act_i8", "gemm_nt_q", "gelu_mul", "quant_act_i8",
                               "gemm_nt_q", "rmsnorm", "add", "scale" })
            want.push_back(s);
    }
    for (const char* s : { "rmsnorm", "gather_rows", "logits_gemm", "logit_softcap" })
        want.push_back(s);
    CHECK_EQ(join(gemma4_w4_step({ "rmsnorm_quant_i8", "qkv_norm_rope", "gelu_mul_quant_i8" })), join(want));
}

/* =============================== gemma4: the assistant (MTP) head =============================== */

namespace {
/* The draft passes the core stages for a head with no cache of its own: one row per drafting
 * sequence. Four sequences; the history pass marks every row as a draft row. */
static const int32_t kHeadRows[] = { 0, 1, 2, 3 };
RadBatch head_batch(int pass) {
    RadBatch b{};
    b.phase         = RAD_PHASE_DECODE;
    b.n_tok         = 4;
    b.n_seq         = 4;
    b.max_q_len     = 1;
    b.max_ctx_len   = 100;
    b.draft_pass    = pass;
    if (pass < 0) {
        b.n_out         = 4;
        b.out_ids       = kHeadRows;
        b.n_draft_out   = 4;
        b.draft_out_ids = kHeadRows;
    }
    return b;
}
}  /* namespace */

TEST(gemma4_declares_no_head_without_one_or_without_speculation) {
    {
        RadBuilder b;
        RadModelMeta meta = gemma4_meta();
        RadBuildCtx  ctx  = build_ctx();
        ctx.max_spec = 3;
        CHECK_OK(gemma4_bf16::declare(&b, &meta, &ctx));
        CHECK_EQ((int)b.drafter.kind, (int)RAD_DRAFT_NONE);
        CHECK(find_w(b, "mtp.pre_proj.weight") == nullptr);
    }
    {
        RadBuilder b;
        RadModelMeta meta = gemma4_draft_meta();
        RadBuildCtx  ctx  = build_ctx();      /* max_spec 0 */
        CHECK_OK(gemma4_bf16::declare(&b, &meta, &ctx));
        CHECK_EQ((int)b.drafter.kind, (int)RAD_DRAFT_NONE);
        CHECK(find_w(b, "mtp.pre_proj.weight") == nullptr);
        CHECK_EQ(count_op(b, "attn_paged"), 12);
    }
}

TEST(gemma4_assistant_is_a_serial_head_with_no_cache_of_its_own) {
    RadBuilder b;
    RadModelMeta meta = gemma4_draft_meta();
    RadBuildCtx  ctx  = build_ctx();
    ctx.max_spec = 3;
    CHECK_OK(gemma4_bf16::declare(&b, &meta, &ctx));

    CHECK_EQ((int)b.drafter.kind, (int)RAD_DRAFT_SERIAL);
    CHECK_EQ(b.drafter.depth, 3);
    CHECK_EQ(b.drafter_name, std::string("mtp"));
    CHECK_EQ((int)b.drafter.proposal, 0);         /* proposes through the sampler chain */
    CHECK(b.drafter_shared);
    CHECK_EQ(b.drafter_pos, gemma4_bf16::kHeadPosOffset);

    /* It binds no layer and writes no K or V; only the trunk's twelve bindings exist. */
    CHECK_EQ((int)b.binds.size(), 12);
    CHECK_EQ((int)b.kv_groups.size(), 2);

    /* Four head attention layers beside the trunk's twelve, none storing. */
    CHECK_EQ(count_op(b, "attn_paged"), 16);
    CHECK_EQ(count_op(b, "kv_store"), 12);
    CHECK_EQ(count_op(b, "logit_softcap"), 1);    /* trunk only; head logits are not capped */
    CHECK_EQ(count_op(b, "logits_gemm"), 2);

    for (int l = 0; l < 4; ++l) {
        const RecOp* at = find_op(b, "attn_paged", 12 + l);
        REQUIRE(at != nullptr);
        const bool full = (l == 3);
        CHECK_EQ(pint(at, "head_dim"), full ? 512 : 256);
        CHECK_EQ(pint(at, "gqa"), full ? 8 : 2);
        CHECK_EQ(pint(at, "window"), full ? 0 : 1024);
        CHECK_EQ(pint(at, "causal"), 1);
        CHECK_NEAR(pflt(at, "scale"), 1.0, 1e-12);
        const RecParam* ql = param(at, "q_len");
        REQUIRE(ql != nullptr);
        CHECK_EQ(ql->ival, 1);                        /* one row per sequence: decode band only */
        CHECK_EQ(ql->ihi, 1);
    }

    /* Weights: q-only attention, the two projections and the head's own tied table. */
    const RecWeight* wq0 = find_w(b, "mtp.blk.0.attn_q.weight");
    const RecWeight* wq3 = find_w(b, "mtp.blk.3.attn_q.weight");
    REQUIRE(wq0 != nullptr); REQUIRE(wq3 != nullptr);
    CHECK_EQ(wq0->d.shape[0], 32 * 256); CHECK_EQ(wq0->d.shape[1], 1024);
    CHECK_EQ(wq3->d.shape[0], 32 * 512); CHECK_EQ(wq3->d.shape[1], 1024);
    const RecWeight* wo3 = find_w(b, "mtp.blk.3.attn_o.weight");
    REQUIRE(wo3 != nullptr);
    CHECK_EQ(wo3->d.shape[0], 1024); CHECK_EQ(wo3->d.shape[1], 32 * 512);
    CHECK(find_w(b, "mtp.blk.0.attn_k.weight") == nullptr);
    CHECK(find_w(b, "mtp.blk.0.attn_v.weight") == nullptr);
    const RecWeight* gu = find_w(b, "mtp.blk.1.ffn_gate_up.weight");
    REQUIRE(gu != nullptr);
    CHECK_EQ(gu->d.shape[0], 2 * 8192); CHECK_EQ(gu->d.shape[1], 1024);
    const RecWeight* pre  = find_w(b, "mtp.pre_proj.weight");
    const RecWeight* post = find_w(b, "mtp.post_proj.weight");
    REQUIRE(pre != nullptr); REQUIRE(post != nullptr);
    CHECK_EQ(pre->d.shape[0], 1024);  CHECK_EQ(pre->d.shape[1], 2 * 5376);
    CHECK_EQ(post->d.shape[0], 5376); CHECK_EQ(post->d.shape[1], 1024);
    CHECK_EQ(pre->d.shard, RAD_SHARD_NONE);
    const RecWeight* lm = find_w(b, "mtp.output.weight");
    REQUIRE(lm != nullptr);
    CHECK_EQ(lm->d.shape[0], 262144); CHECK_EQ(lm->d.shape[1], 1024);
    CHECK_EQ(lm->d.shard, RAD_SHARD_ROW);
    CHECK_EQ(lm->d.access, RAD_ACCESS_VOCAB);

    /* Checkpoint names merged under `draft.`: the head's table is its own embedding (no
     * fallback), and gate and up are fused like the trunk's. */
    const RecMap* mlm = find_map(b, "mtp.output.weight");
    REQUIRE(mlm != nullptr);
    CHECK_EQ(mlm->src[0], std::string("draft.model.embed_tokens.weight"));
    const RecMap* mgu = find_map(b, "mtp.blk.1.ffn_gate_up.weight");
    REQUIRE(mgu != nullptr);
    CHECK_EQ(mgu->mode, RAD_MAP_CONCAT);
    CHECK_EQ(mgu->src[0], std::string("draft.model.layers.1.mlp.gate_proj.weight"));
    CHECK_EQ(mgu->src[1], std::string("draft.model.layers.1.mlp.up_proj.weight"));
    const RecMap* mpre = find_map(b, "mtp.pre_proj.weight");
    REQUIRE(mpre != nullptr);
    CHECK_EQ(mpre->src[0], std::string("draft.pre_projection.weight"));

    /* The entry embeds through the trunk's table with the trunk's scale, sqrt(5376) rounded to
     * bf16 as the reference's embedding module holds it. */
    const RecOp* el = find_op(b, "embed_lookup", 1);
    REQUIRE(el != nullptr);
    CHECK_EQ(pint(el, "n_embd"), 5376);
    CHECK_NEAR(pflt(el, "mult"), 73.5, 1e-9);
    CHECK(el->w.size() == 1 && find_op(b, "embed_lookup", 0)->w == el->w);

    for (const auto& o : b.ops)
        if (o.resolved) CHECK(o.uses_declared);
    for (const auto& w : b.weights)
        CHECK(find_map(b, w.name) != nullptr);
}

TEST(gemma4_head_passes_read_the_last_trunk_layer_of_their_class) {
    RadBuilder b;
    RadModelMeta meta = gemma4_draft_meta();
    RadBuildCtx  ctx  = build_ctx();
    ctx.max_spec = 3;
    CHECK_OK(gemma4_bf16::declare(&b, &meta, &ctx));

    for (int pass : { -1, 2 }) {
        RadBatch batch = head_batch(pass);
        RadCtx c{ .b = &b, .batch = &batch };
        gemma4_bf16::step(&c, &batch);
        CHECK(c.step_fail.empty());

        std::vector<std::string> want;
        want.push_back(pass < 0 ? "gather_rows" : "cast");   /* the trunk's h or the head's h' */
        for (const char* s : { "embed_lookup", "cast", "gemm_nt" }) want.push_back(s);
        for (int l = 0; l < 4; ++l)
            for (const char* s : { "rmsnorm", "gemm_nt", "rmsnorm", "rope", "attn_paged", "gemm_nt",
                                   "rmsnorm", "add", "rmsnorm", "gemm_nt", "gelu_mul", "gemm_nt",
                                   "rmsnorm", "add", "scale" })
                want.push_back(s);
        for (const char* s : { "rmsnorm", "logits_gemm", "gemm_nt" }) want.push_back(s);
        CHECK_EQ(join(issued_ops(b, c)), join(want));

        /* Attention reads the trunk's cache: head layers 0..2 the last sliding trunk layer (10),
         * layer 3 the last full one (11). Groups: 1 sliding, 2 full. */
        int seen = 0;
        for (const auto& i : c.issues) {
            const RecOp* o = b.op_of(i.op);
            if (!o || o->op != "attn_paged") continue;
            REQUIRE(i.opd.size() > 1);
            CHECK_EQ((int)i.opd[1].kind, (int)RAD_OPK_KV);
            CHECK_EQ((int)i.opd[1].handle, seen < 3 ? 1 : 2);
            CHECK_EQ((long long)i.opd[1].offset, seen < 3 ? 10LL : 11LL);
            CHECK_EQ(i.n, 1);
            ++seen;
        }
        CHECK_EQ(seen, 4);
    }
}

TEST(gemma4_head_at_two_ranks_reduces_before_each_post_norm) {
    RadBuilder b;
    RadModelMeta meta = gemma4_draft_meta();
    RadBuildCtx  ctx  = build_ctx(0, 2);
    ctx.max_spec = 3;
    CHECK_OK(gemma4_bf16::declare(&b, &meta, &ctx));

    const RecWeight* wq0 = find_w(b, "mtp.blk.0.attn_q.weight");
    REQUIRE(wq0 != nullptr);
    CHECK_EQ(wq0->d.shape[0], 16 * 256);
    const RecWeight* lm = find_w(b, "mtp.output.weight");
    REQUIRE(lm != nullptr);
    CHECK_EQ(lm->d.shape[0], 131072);

    RadBatch batch = head_batch(-1);
    RadCtx c{ .b = &b, .batch = &batch, .world = 2 };
    gemma4_bf16::step(&c, &batch);
    CHECK(c.step_fail.empty());
    const std::vector<std::string> got = issued_ops(b, c);
    std::vector<std::string> want = { "gather_rows", "embed_lookup", "all_reduce", "cast", "gemm_nt" };
    for (int l = 0; l < 4; ++l)
        for (const char* s : { "rmsnorm", "gemm_nt", "rmsnorm", "rope", "attn_paged", "gemm_nt",
                               "all_reduce", "rmsnorm", "add", "rmsnorm", "gemm_nt", "gelu_mul",
                               "gemm_nt", "all_reduce", "rmsnorm", "add", "scale" })
            want.push_back(s);
    for (const char* s : { "rmsnorm", "logits_gemm", "gemm_nt" }) want.push_back(s);
    CHECK_EQ(join(got), join(want));
}

TEST(gemma4_head_refuses_a_history_pass_shaped_for_a_cache_of_its_own) {
    RadBuilder b;
    RadModelMeta meta = gemma4_draft_meta();
    RadBuildCtx  ctx  = build_ctx();
    ctx.max_spec = 3;
    CHECK_OK(gemma4_bf16::declare(&b, &meta, &ctx));
    RadBatch batch = head_batch(-1);
    batch.n_draft_out = 1;                 /* a history pass with non-drafting rows */
    RadCtx c{ .b = &b, .batch = &batch };
    gemma4_bf16::step(&c, &batch);
    CHECK(!c.step_fail.empty());
    CHECK(c.issues.empty());
}

TEST(gemma4_probe_names_the_heads_operating_depth) {
    RadArchProbe p{};
    RadModelMeta with = gemma4_draft_meta();
    CHECK_OK(gemma4_bf16::probe(&with, &p));
    CHECK_EQ(p.draft_depth, gemma4_bf16::kHeadDepth);
    CHECK_EQ(p.draft_depth_fixed, 0);
    RadArchProbe q{};
    RadModelMeta without = gemma4_meta();
    CHECK_OK(gemma4_bf16::probe(&without, &q));
    CHECK_EQ(q.draft_depth, 0);
}

/* ===================== gemma4: the 26B-A4B, a dense and a routed arm side by side ===================== */

namespace {

/* gemma-4-26B-A4B-it's text geometry cut to six layers (five sliding then one full, the model's
 * 5:1) with the routed arm on. Keys are the checkpoint's flattened text_config, as arch/gemma4
 * reads them; quant is empty, the container this plugin claims. */
const char* k_g4moe_keys[] = {
    "hidden_size", "num_attention_heads", "num_key_value_heads", "head_dim", "global_head_dim",
    "num_global_key_value_heads", "attention_k_eq_v", "vocab_size", "intermediate_size",
    "rms_norm_eps", "sliding_window", "final_logit_softcapping", "hidden_activation", "layer_types",
    "rope_parameters.sliding_attention.rope_theta", "rope_parameters.sliding_attention.rope_type",
    "rope_parameters.full_attention.rope_theta", "rope_parameters.full_attention.rope_type",
    "rope_parameters.full_attention.partial_rotary_factor", "enable_moe_block", "num_experts",
    "top_k_experts", "moe_intermediate_size",
};
const char* k_g4moe_vals[] = {
    "2816", "16", "8", "256", "512",
    "2", "true", "262144", "2112",
    "1e-06", "1024", "30.0", "gelu_pytorch_tanh",
    "[\"sliding_attention\", \"sliding_attention\", \"sliding_attention\", \"sliding_attention\", "
    "\"sliding_attention\", \"full_attention\"]",
    "10000.0", "default",
    "1000000.0", "proportional",
    "0.25", "true", "128",
    "8", "704",
};
static_assert(sizeof(k_g4moe_keys) == sizeof(k_g4moe_vals), "one value a key");

RadModelMeta gemma4_moe_meta() {
    RadModelMeta m{};
    m.arch_id     = "gemma4";
    m.name        = "test-gemma4-26b-a4b";
    m.quant       = "";
    m.n_layers    = 6;
    m.n_embd      = 2816;
    m.n_head      = 16;
    m.n_head_kv   = 8;
    m.head_dim    = 256;
    m.n_ff        = 2112;
    m.n_vocab     = 262144;
    m.n_ctx_train = 262144;
    m.rms_eps     = 1e-6f;
    m.n_kv        = (int)(sizeof(k_g4moe_keys) / sizeof(k_g4moe_keys[0]));
    m.kv_key      = k_g4moe_keys;
    m.kv_val      = k_g4moe_vals;
    return m;
}

int count_resolved(const RadBuilder& b, const std::string& name) {
    int n = 0;
    for (const auto& o : b.ops) if (o.op == name && o.resolved) ++n;
    return n;
}

int count_issued(const std::vector<std::string>& v, const std::string& name) {
    return (int)std::count(v.begin(), v.end(), name);
}

}  /* namespace */

TEST(a_gemma4_moe_trunk_declares_both_arms_and_the_joins) {
    RadBuilder b;
    RadModelMeta meta = gemma4_moe_meta();
    RadBuildCtx  ctx  = build_ctx();
    CHECK_OK(gemma4::declare(&b, &meta, &ctx));

    /* five sliding layers read q from the fused row (attn_paged_qrow); the full one attn_paged */
    CHECK_EQ(count_op(b, "attn_paged_qrow"), 5);
    CHECK_EQ(count_op(b, "attn_paged"),      1);
    CHECK_EQ(count_op(b, "kv_store"),   6);
    /* Per layer, one join after attention and one after the two feed-forward arms. */
    CHECK_EQ(count_resolved(b, "norm_add_norm"),  6);
    CHECK_EQ(count_resolved(b, "join_add_scale"), 6);
    /* The routed arm: experts are bf16 where the container has no encoding for them. */
    CHECK(count_op(b, "moe_gemm") > 0);
    CHECK_EQ(count_op(b, "moe_gemm_q"), 0);
    CHECK_EQ(count_op(b, "logit_softcap"), 1);
    CHECK_EQ(count_op(b, "all_reduce"),    0);   /* world 1 */

    for (const auto& o : b.ops)
        if (o.resolved) CHECK(o.uses_declared);
    for (const auto& w : b.weights)
        CHECK(find_map(b, w.name) != nullptr);
}

/* The sliding group is RAD_KV_FULL with the window on the attention op, since a window group is
 * not prefix-cached (arch/gemma4/gemma4.cpp, the KV groups). */
TEST(gemma4_moe_kv_groups_are_both_full_and_the_window_rides_the_op) {
    RadBuilder b;
    RadModelMeta meta = gemma4_moe_meta();
    RadBuildCtx  ctx  = build_ctx();
    CHECK_OK(gemma4::declare(&b, &meta, &ctx));

    REQUIRE((int)b.kv_groups.size() == 2);
    for (const auto& g : b.kv_groups) {
        CHECK_EQ((int)g.d.kind, RAD_KV_FULL);
        CHECK_EQ(g.d.window, 0);
    }
    const RecOp* sl = find_op(b, "attn_paged_qrow", 0);
    REQUIRE(sl != nullptr);
    CHECK_EQ(pint(sl, "head_dim"), 256);
    CHECK_EQ(pint(sl, "window"),   1024);
    const RecOp* fu = find_op(b, "attn_paged", 0);
    REQUIRE(fu != nullptr);
    CHECK_EQ(pint(fu, "head_dim"), 512);
    CHECK_EQ(pint(fu, "gqa"),      8);
    CHECK_EQ(pint(fu, "window"),   0);
}

/* Proportional RoPE is not a partial rotary: full layers rotate the whole 512-wide head as NeoX
 * pairs (j, j + 256), with frequencies past pair 63 removed by the YaRN ramp. rotary_dim 128 would
 * pair (j, j + 64), which is wrong. */
TEST(gemma4_full_rope_is_the_whole_head_with_the_frequencies_past_64_pairs_cut) {
    RadBuilder b;
    RadModelMeta meta = gemma4_moe_meta();
    RadBuildCtx  ctx  = build_ctx();
    CHECK_OK(gemma4::declare(&b, &meta, &ctx));

    int full = 0;
    for (const auto& o : b.ops) {
        if (o.op != "rope" || pint(&o, "head_dim") != 512) continue;
        ++full;
        CHECK_EQ(pint(&o, "rotary_dim"), 512);
        CHECK_EQ(pstr(&o, "mode"), std::string("neox"));
        CHECK_NEAR(pflt(&o, "theta"),     1e6, 1.0);
        CHECK_NEAR(pflt(&o, "yarn_low"),  63.5, 1e-9);
        CHECK_NEAR(pflt(&o, "yarn_high"), 64.0, 1e-9);
        CHECK(pflt(&o, "yarn_factor") >= 1e30);
    }
    CHECK(full > 0);
}

/* Where no kernel library serves a join, the plugin declares and issues the rmsnorm / add chain;
 * libref deliberately has no row for the joins so this path stays reachable (docs/OPS.md). */
TEST(gemma4_falls_back_to_the_chain_where_no_join_resolves) {
    std::vector<std::string> fused, chain;
    for (int refuse = 0; refuse < 2; ++refuse) {
        RadBuilder b;
        if (refuse) { b.refuse.insert("norm_add_norm"); b.refuse.insert("join_add_scale"); }
        RadModelMeta meta = gemma4_moe_meta();
        RadBuildCtx  ctx  = build_ctx();
        CHECK_OK(gemma4::declare(&b, &meta, &ctx));

        RadBatch batch = decode_batch();
        RadCtx c{ .b = &b, .batch = &batch };
        gemma4::step(&c, &batch);
        CHECK_LINT(b, &c);
        CHECK(c.step_fail.empty());
        (refuse ? chain : fused) = issued_ops(b, c);
    }
    CHECK_EQ(count_issued(fused, "norm_add_norm"),  6);
    CHECK_EQ(count_issued(fused, "join_add_scale"), 6);
    CHECK_EQ(count_issued(chain, "norm_add_norm"),  0);
    CHECK_EQ(count_issued(chain, "join_add_scale"), 0);
    for (const auto* v : { &fused, &chain }) {
        CHECK_EQ(count_issued(*v, "attn_paged_qrow"), 5);
        CHECK_EQ(count_issued(*v, "attn_paged"),      1);
    }
    /* The chain is longer by the norms and adds the joins fold in. */
    CHECK(count_issued(chain, "rmsnorm") > count_issued(fused, "rmsnorm"));
    CHECK(count_issued(chain, "add") > count_issued(fused, "add"));
}

/* Both Gemma 4 plugins export the same reply-format declaration. */
TEST(both_gemma4_plugins_export_one_reply_format) {
    RadModelMeta moe = gemma4_moe_meta(), dense = gemma4_meta();
    CHECK(gemma4::chat_format(&moe) == rad::arch::gemma4_chat_format());
    CHECK(gemma4_bf16::chat_format(&dense) == rad::arch::gemma4_chat_format());
}

/* ================== gemma4 + DFlash: the drafter's taps, live where they are written ================== */

namespace {

/* The MoE trunk above with z-lab's DFlash drafter merged: five layers (four sliding, one full)
 * reading the trunk at layers 1, 3 and 5. */
std::vector<const char*> g4df_keys() {
    std::vector<const char*> k(std::begin(k_g4moe_keys), std::end(k_g4moe_keys));
    for (const char* s : { "draft.architectures", "draft.num_hidden_layers", "draft.head_dim",
                           "draft.num_attention_heads", "draft.num_key_value_heads",
                           "draft.intermediate_size", "draft.rms_norm_eps", "draft.rope_theta",
                           "draft.hidden_size", "draft.vocab_size", "draft.block_size",
                           "draft.dflash_config.target_layer_ids",
                           "draft.dflash_config.mask_token_id", "draft.layer_types",
                           "draft.sliding_window" })
        k.push_back(s);
    return k;
}
std::vector<const char*> g4df_vals() {
    std::vector<const char*> v(std::begin(k_g4moe_vals), std::end(k_g4moe_vals));
    for (const char* s : { "[\"DFlashDraftModel\"]", "5", "128", "32", "8", "5632", "1e-06",
                           "1000000.0", "2816", "262144", "16", "[1, 3, 5]", "4",
                           "[\"sliding_attention\", \"sliding_attention\", \"sliding_attention\", "
                           "\"sliding_attention\", \"full_attention\"]",
                           "2048" })
        v.push_back(s);
    return v;
}

int buf_named(const RadBuilder& b, const char* name) {
    for (size_t i = 0; i < b.bufs.size(); ++i)
        if (b.bufs[i].name == name) return (int)i + 1;
    return 0;
}

}  /* namespace */

/* DFlash proposes a block in one pass, so the probe's depth is the trained block size minus the
 * anchor, and fixed. Without a drafter, or with DFlash2 (not served by this plugin), it is 0. */
TEST(gemma4_probe_names_the_dflash_blocks_depth) {
    std::vector<const char*> keys = g4df_keys(), vals = g4df_vals();
    RadModelMeta meta = gemma4_moe_meta();
    meta.n_kv   = (int)keys.size();
    meta.kv_key = keys.data();
    meta.kv_val = vals.data();
    RadArchProbe p{};
    CHECK_OK(gemma4::probe(&meta, &p));
    CHECK_EQ(p.draft_depth, 15);
    CHECK_EQ(p.draft_depth_fixed, 1);

    for (size_t i = 0; i < keys.size(); ++i)
        if (!std::strcmp(keys[i], "draft.architectures")) vals[i] = "[\"DFlash2DraftModel\"]";
    meta.kv_val = vals.data();
    RadArchProbe q{};
    CHECK_OK(gemma4::probe(&meta, &q));
    CHECK_EQ(q.draft_depth, 0);

    RadModelMeta plain = gemma4_moe_meta();
    RadArchProbe r{};
    CHECK_OK(gemma4::probe(&plain, &r));
    CHECK_EQ(r.draft_depth, 0);
}

/* Each tap is declared where it is issued. The buffer planner takes liveness from declaration
 * order, so `dflash_src` is live from its first declared write; a tap declared after the trunk but
 * issued inside it would let the plan alias it with `moe_edn`, which every MoE layer writes,
 * silently corrupting what the drafter reads. The two live ranges must overlap. */
TEST(gemma4_dflash_taps_are_live_from_the_layer_that_writes_them) {
    const std::vector<const char*> keys = g4df_keys(), vals = g4df_vals();
    RadModelMeta meta = gemma4_moe_meta();
    meta.n_kv   = (int)keys.size();
    meta.kv_key = keys.data();
    meta.kv_val = vals.data();
    RadBuilder b;
    RadBuildCtx ctx = build_ctx();
    ctx.max_spec = 15;
    CHECK_OK(gemma4::declare(&b, &meta, &ctx));

    const int src = buf_named(b, "dflash_src"), edn = buf_named(b, "moe_edn");
    REQUIRE(src > 0 && edn > 0);
    int first_src = -1, last_edn = -1;
    for (size_t i = 0; i < b.ops.size(); ++i)
        for (rad_buf w : b.ops[i].writes) {
            if ((int)w == src && first_src < 0) first_src = (int)i;
            if ((int)w == edn) last_edn = (int)i;
        }
    REQUIRE(first_src >= 0 && last_edn >= 0);
    CHECK(first_src < last_edn);

    /* A trunk step writes each tap once: three copies into `dflash_src`. */
    RadBatch batch = decode_batch();
    RadCtx c{ .b = &b, .batch = &batch };
    gemma4::step(&c, &batch);
    CHECK_LINT(b, &c);
    CHECK(c.step_fail.empty());
    int taps = 0;
    for (const auto& i : c.issues) {
        const RecOp* o = b.op_of(i.op);
        if (o && std::find(o->writes.begin(), o->writes.end(), (rad_buf)src) != o->writes.end())
            ++taps;
    }
    CHECK_EQ(taps, 3);
}

/* Sliding attention reads q in place in the fused projection row where attn_paged_qrow is served,
 * and lifts q into its own buffer for attn_paged where it is not. */
TEST(gemma4_sliding_attention_reads_q_in_the_fused_row) {
    /* Served: attn_paged_qrow reads the slice and nothing is lifted. Refused: q is lifted into q_s
     * for attn_paged. attn_paged never gets the slice, since its kernel may read q only dense. */
    for (int refuse = 0; refuse < 2; ++refuse) {
        RadBuilder b;
        if (refuse) b.refuse.insert("attn_paged_qrow");
        RadModelMeta meta = gemma4_moe_meta();
        RadBuildCtx  ctx  = build_ctx();
        CHECK_OK(gemma4::declare(&b, &meta, &ctx));
        const int qkv = buf_named(b, "qkv_s"), qs = buf_named(b, "q_s");
        REQUIRE(qkv > 0);
        CHECK_EQ(qs > 0, refuse == 1);
        auto reads = [](const RecOp& o, int buf) {
            return buf > 0 && std::find(o.reads.begin(), o.reads.end(), (rad_buf)buf) != o.reads.end();
        };
        int qrow = 0, paged_slice = 0, paged_lifted = 0;
        for (const auto& o : b.ops) {
            if (!o.resolved) continue;
            if (o.op == "attn_paged_qrow" && reads(o, qkv)) ++qrow;
            if (o.op == "attn_paged" && reads(o, qkv)) ++paged_slice;
            if (o.op == "attn_paged" && reads(o, qs)) ++paged_lifted;
        }
        const int sliding = refuse ? paged_lifted : qrow;
        CHECK(sliding > 0);
        CHECK_EQ(refuse ? qrow : paged_lifted, 0);
        CHECK_EQ(paged_slice, 0);
        CHECK_EQ(count_op(b, "cast"), refuse ? sliding : 0);   /* the lifts (no drafter taps) */

        RadBatch batch = decode_batch();
        RadCtx c{ .b = &b, .batch = &batch };
        gemma4::step(&c, &batch);
        CHECK_LINT(b, &c);
        CHECK(c.step_fail.empty());
        const std::vector<std::string> issued = issued_ops(b, c);
        CHECK_EQ(count_issued(issued, "attn_paged_qrow"), refuse ? 0 : sliding);
        CHECK_EQ(count_issued(issued, "cast"), refuse ? sliding : 0);
        for (const auto& o : b.ops)
            if (o.resolved) CHECK(o.uses_declared);
    }
}

/* The feed-forward join writes the next layer's input norm. With joins served, the only rmsnorm
 * from residual `x` into `h` is layer 0's, the last join produces the final norm, and every join
 * writes `h`. Refused, the chain declares them all. */
TEST(gemma4_feed_forward_join_writes_the_next_input_norm) {
    for (int refuse = 0; refuse < 2; ++refuse) {
        RadBuilder b;
        if (refuse) { b.refuse.insert("norm_add_norm"); b.refuse.insert("join_add_scale"); }
        RadModelMeta meta = gemma4_moe_meta();
        RadBuildCtx  ctx  = build_ctx();
        CHECK_OK(gemma4::declare(&b, &meta, &ctx));
        const int x = buf_named(b, "x"), h = buf_named(b, "h");
        REQUIRE(x > 0 && h > 0);
        int x_to_h = 0, joins_to_h = 0;
        for (const auto& o : b.ops) {
            if (!o.resolved) continue;
            const bool rx = std::find(o.reads.begin(), o.reads.end(), (rad_buf)x) != o.reads.end();
            const bool wh = std::find(o.writes.begin(), o.writes.end(), (rad_buf)h) != o.writes.end();
            if (o.op == "rmsnorm" && rx && wh) ++x_to_h;
            if (o.op == "join_add_scale" && wh) ++joins_to_h;
        }
        /* fused: layer 0's input norm. Chained: six input norms, six pre-feedforward norms and the
         * final norm. */
        CHECK_EQ(x_to_h, refuse ? 13 : 1);
        CHECK_EQ(joins_to_h, refuse ? 0 : 6);

        RadBatch batch = decode_batch();
        RadCtx c{ .b = &b, .batch = &batch };
        gemma4::step(&c, &batch);
        CHECK_LINT(b, &c);
        CHECK(c.step_fail.empty());
        const std::vector<std::string> issued = issued_ops(b, c);
        CHECK_EQ(count_issued(issued, "join_add_scale"), refuse ? 0 : 6);
        CHECK_EQ(count_issued(issued, "logits_gemm"), 1);
    }
}

/* At decode heights the router GEMM, top-k and sort are one launch (libr4d's
 * router_gemm_topk_scatter); if refused, router_topk_scatter follows the router GEMM; if both are
 * refused, three separate ops. The routing buffers are the same in all cases. */
TEST(gemma4_routes_a_decode_step_in_one_launch) {
    for (int refuse = 0; refuse < 3; ++refuse) {
        RadBuilder b;
        if (refuse >= 1) b.refuse.insert("router_gemm_topk_scatter");
        if (refuse >= 2) b.refuse.insert("router_topk_scatter");
        RadModelMeta meta = gemma4_moe_meta();
        RadBuildCtx  ctx  = build_ctx();
        CHECK_OK(gemma4::declare(&b, &meta, &ctx));
        RadBatch batch = decode_batch();
        RadCtx c{ .b = &b, .batch = &batch };
        gemma4::step(&c, &batch);
        CHECK_LINT(b, &c);
        CHECK(c.step_fail.empty());
        const std::vector<std::string> issued = issued_ops(b, c);
        CHECK_EQ(count_issued(issued, "router_gemm_topk_scatter"), refuse == 0 ? 6 : 0);
        CHECK_EQ(count_issued(issued, "router_topk_scatter"), refuse == 1 ? 6 : 0);
        CHECK_EQ(count_issued(issued, "router_topk"), refuse == 2 ? 6 : 0);
        CHECK_EQ(count_issued(issued, "moe_scatter"), refuse == 2 ? 6 : 0);
        /* the per-expert scale and gather run whichever routing form was used */
        CHECK_EQ(count_issued(issued, "expert_scale"), 6);
        CHECK_EQ(count_issued(issued, "moe_gather"), 6);
    }
}

/* The per-expert scale is one launch where `expert_scale` is served, else a table lookup and a
 * multiply over the same buffers. */
TEST(gemma4_scales_the_routing_weights_in_one_launch) {
    for (int refuse = 0; refuse < 2; ++refuse) {
        RadBuilder b;
        if (refuse) b.refuse.insert("expert_scale");
        RadModelMeta meta = gemma4_moe_meta();
        RadBuildCtx  ctx  = build_ctx();
        CHECK_OK(gemma4::declare(&b, &meta, &ctx));
        RadBatch batch = decode_batch();
        RadCtx c{ .b = &b, .batch = &batch };
        gemma4::step(&c, &batch);
        CHECK_LINT(b, &c);
        CHECK(c.step_fail.empty());
        const std::vector<std::string> issued = issued_ops(b, c);
        CHECK_EQ(count_issued(issued, "expert_scale"), refuse ? 0 : 6);
        CHECK_EQ(count_issued(issued, "embed_lookup"), refuse ? 1 + 6 : 1);   /* 1: the token embedding */
        CHECK_EQ(count_issued(issued, "mul"), refuse ? 6 : 0);
        for (const auto& o : b.ops)
            if (o.resolved) CHECK(o.uses_declared);
    }
}

/* The drafter's head reads the container's block-fp8 copy of the target's table
 * (`dflash.draft_head.weight`) through quant_act_fp8 and gemm_nt_q where present, else the bf16
 * table through gemm_nt. The copy is only declared when present, so older containers are served
 * unchanged. */
TEST(gemma4_dflash_head_reads_the_fp8_copy_where_the_container_holds_one) {
    const std::vector<const char*> keys = g4df_keys(), vals = g4df_vals();
    for (int fp8 = 0; fp8 < 2; ++fp8) {
        RadModelMeta meta = gemma4_moe_meta();
        meta.n_kv   = (int)keys.size();
        meta.kv_key = keys.data();
        meta.kv_val = vals.data();
        RadBuilder b;
        if (fp8)
            b.encs = { { "dflash.draft_head", rad_enc_affine(RAD_F8E4M3, RAD_BF16, 128, 128) } };
        RadBuildCtx ctx = build_ctx();
        ctx.max_spec = 15;
        CHECK_OK(gemma4::declare(&b, &meta, &ctx));
        CHECK_EQ(find_w(b, "dflash.draft_head.weight") != nullptr, fp8 == 1);

        const int dlog = buf_named(b, "df_dlog"), selh = buf_named(b, "df_selh");
        REQUIRE(dlog > 0 && selh > 0);
        const RecOp* head = nullptr;
        for (const auto& o : b.ops)
            if (o.resolved &&
                std::find(o.writes.begin(), o.writes.end(), (rad_buf)dlog) != o.writes.end())
                head = &o;
        REQUIRE(head != nullptr);
        CHECK_EQ(head->op, std::string(fp8 ? "gemm_nt_q" : "gemm_nt"));
        const bool reads_selh =
            std::find(head->reads.begin(), head->reads.end(), (rad_buf)selh) != head->reads.end();
        CHECK_EQ(reads_selh, fp8 == 0);

        /* a draft pass issues it */
        RadBatch batch = decode_batch();
        batch.draft_pass = 1;
        RadCtx c{ .b = &b, .batch = &batch };
        gemma4::step(&c, &batch);
        CHECK(c.step_fail.empty());
        int head_issues = 0;
        for (const auto& i : c.issues)
            if (b.op_of(i.op) == head) ++head_issues;
        CHECK_EQ(head_issues, 1);
    }
}

/* The drafter fuses residual add, norm and quantiser into `rmsnorm_quant_fp8`: ten per draft pass
 * over five layers (each layer's two inputs; layer 0's attention input takes x as the embedding
 * left it), leaving one `add`, the last layer's, before the final norm. Refused: two adds and two
 * quantisers of x per layer. */
TEST(gemma4_dflash_folds_the_residual_add_norm_and_quantiser_into_one_launch) {
    const std::vector<const char*> keys = g4df_keys(), vals = g4df_vals();
    for (int refuse = 0; refuse < 2; ++refuse) {
        RadModelMeta meta = gemma4_moe_meta();
        meta.n_kv   = (int)keys.size();
        meta.kv_key = keys.data();
        meta.kv_val = vals.data();
        RadBuilder b;
        if (refuse) b.refuse.insert("rmsnorm_quant_fp8");
        RadBuildCtx ctx = build_ctx();
        ctx.max_spec = 15;
        CHECK_OK(gemma4::declare(&b, &meta, &ctx));

        RadBatch batch = decode_batch();
        batch.draft_pass = 1;
        RadCtx c{ .b = &b, .batch = &batch };
        gemma4::step(&c, &batch);
        CHECK(c.step_fail.empty());
        const std::vector<std::string> ops = issued_ops(b, c);
        CHECK_EQ(count_issued(ops, "rmsnorm_quant_fp8"), refuse ? 0 : 10);
        CHECK_EQ(count_issued(ops, "add"), refuse ? 10 : 1);
        /* chain: x quantised twice per layer; attention output quantised once per layer always */
        CHECK_EQ(count_issued(ops, "quant_act_fp8"), refuse ? 15 : 5);
        if (refuse) continue;

        /* the delta then the residual x past layer 0; layer 0 reads x and adds nothing */
        const uint32_t x = (uint32_t)buf_named(b, "df_x"), h = (uint32_t)buf_named(b, "df_h");
        int k = 0;
        for (const auto& i : c.issues) {
            const RecOp* o = b.op_of(i.op);
            if (!o || o->op != "rmsnorm_quant_fp8") continue;
            REQUIRE(i.opd.size() == 6);
            if (k++ == 0) {
                CHECK_EQ(i.opd[0].handle, x);
                CHECK_EQ((int)i.opd[1].kind, (int)RAD_OPK_NONE);
            } else {
                CHECK_EQ(i.opd[0].handle, h);
                CHECK_EQ(i.opd[1].handle, x);
            }
        }
    }
}

/* ============ gemma4: the prefill overlap -- two slices, the reductions on lane 1 ============ */

namespace {

/* The elements [lo, hi) of its buffer an operand can reach, in whole rows at the declared width.
 * False for a non-buffer operand. */
bool reach(const RadBuilder& b, const RadOperand& o, rad_buf* buf, int64_t* lo, int64_t* hi) {
    if (o.kind != RAD_OPK_BUF || o.handle == 0 || o.handle > b.bufs.size()) return false;
    const RadBufDecl& d = b.bufs[o.handle - 1].d;
    int64_t pitch = 1;
    for (uint32_t i = 1; i < d.rank; ++i) pitch *= d.shape[i];
    *buf = o.handle;
    *lo  = o.offset - o.offset % pitch;
    *hi  = *lo + (o.rows > 0 ? o.rows : d.shape[0]) * pitch;
    return true;
}

/* Whether issue a runs before issue b (a < b): on one lane by issue order; across lanes only via a
 * join from a's lane to b's issued between them. */
bool ordered(const RadCtx& c, size_t a, size_t b) {
    const int la = c.issues[a].lane, lb = c.issues[b].lane;
    if (la == lb) return true;
    for (const RecJoin& j : c.joins)
        if (j.from == la && j.to == lb && j.at > a && j.at <= b) return true;
    return false;
}

/* Pairs of unordered issues on different lanes that reach a common element. Every operand counts
 * as read and written, so this can only over-count. */
int races(const RadBuilder& b, const RadCtx& c) {
    int n = 0;
    for (size_t i = 0; i < c.issues.size(); ++i)
        for (size_t j = i + 1; j < c.issues.size(); ++j) {
            if (ordered(c, i, j)) continue;
            bool meet = false;
            for (const RadOperand& p : c.issues[i].opd)
                for (const RadOperand& q : c.issues[j].opd) {
                    rad_buf bp = 0, bq = 0;
                    int64_t pl = 0, ph = 0, ql = 0, qh = 0;
                    if (reach(b, p, &bp, &pl, &ph) && reach(b, q, &bq, &ql, &qh) && bp == bq &&
                        pl < qh && ql < ph)
                        meet = true;
                }
            n += meet ? 1 : 0;
        }
    return n;
}

/* One prompt of T tokens, sampling its last. */
RadBatch one_prompt(int64_t T, const int32_t* out) {
    RadBatch b{};
    b.phase       = RAD_PHASE_PREFILL;
    b.n_tok       = T;
    b.n_seq       = 1;
    b.max_q_len   = T;
    b.max_ctx_len = T;
    b.n_out       = 1;
    b.out_ids     = out;
    return b;
}

/* Rows issued, summed by op: a slice's n is its rows; RAD_N_BATCH counts as T. */
std::map<std::string, int64_t> rows_by_op(const RadBuilder& b, const RadCtx& c, int64_t T) {
    std::map<std::string, int64_t> w;
    for (const auto& i : c.issues) {
        const RecOp* o = b.op_of(i.op);
        w[o ? o->op : "<null>"] += i.n < 0 ? T : i.n;
    }
    return w;
}

/* The n of each issue of one op, in issue order. */
std::vector<int64_t> ns_of(const RadBuilder& b, const RadCtx& c, const char* op) {
    std::vector<int64_t> v;
    for (const auto& i : c.issues) {
        const RecOp* o = b.op_of(i.op);
        if (o && o->op == op) v.push_back(i.n);
    }
    return v;
}

bool noted(const RadBuilder& b, const char* text) {
    for (const auto& n : b.notes)
        if (n.find(text) != std::string::npos) return true;
    return false;
}

}  /* namespace */

/* A prefill chunk in two slices (arch/gemma4, trunk_overlap). Checked: every reduction runs on
 * lane 1 after its producer and before its consumer, and nothing else does; the cut is a whole
 * number of routed passes; and the slices issue exactly the whole step's rows (same routed
 * passes, same drafter taps), so the output matches. A chunk under two passes, and a decode step,
 * run whole: reductions on lane 0 and the dense arm on lane 1 beside the routed arm, all ordered. */
TEST(a_gemma4_prefill_chunk_runs_as_two_slices_with_its_reductions_on_the_second_lane) {
    const std::vector<const char*> keys = g4df_keys(), vals = g4df_vals();
    const int64_t E = 2816, L = 6;
    auto model = [&](RadBuilder& b, bool whole) {
        RadModelMeta meta = gemma4_moe_meta();
        meta.n_kv   = (int)keys.size();
        meta.kv_key = keys.data();
        meta.kv_val = vals.data();
        RadBuildCtx ctx = build_ctx(0, 4);
        ctx.max_tok  = 8192;
        ctx.max_spec = 15;
        /* The lossy wire runs every chunk whole, since a slice is not shown to get the same bits
         * as the whole step. */
        if (whole) ctx.tp_wire_lossy = 1;
        return gemma4::declare(&b, &meta, &ctx);
    };
    auto run = [&](RadBuilder& b, RadCtx& c, RadBatch& batch) {
        c = RadCtx{ .b = &b, .batch = &batch, .world = 4 };
        gemma4::step(&c, &batch);
        CHECK_LINT(b, &c);
    };

    struct Case { int64_t T, cut; };
    for (const Case k : { Case{8192, 4096}, Case{6144, 4096}, Case{6143, 2048}, Case{4096, 2048},
                          Case{4095, 0} }) {
        const int32_t out[1] = { (int32_t)(k.T - 1) };
        RadBuilder b;
        CHECK_OK(model(b, false));
        CHECK(noted(b, "prefill overlap: a chunk of two routed passes"));
        CHECK_EQ((int)b.concurrent.size(), 5);
        for (const char* name : { "o", "ff2", "h", "gate_up", "ffn" })
            CHECK(std::count(b.concurrent.begin(), b.concurrent.end(),
                             (rad_buf)buf_named(b, name)) == 1);
        RadBatch batch = one_prompt(k.T, out);
        RadCtx c;
        run(b, c, batch);
        CHECK(c.step_fail.empty());
        CHECK_EQ(c.lane, 0);
        CHECK_EQ(races(b, c), 0);

        std::vector<int64_t> lane1;   /* reductions on lane 1 */
        int dense1 = 0;               /* dense-arm launches on lane 1 */
        for (const auto& i : c.issues) {
            if (i.lane != 1) continue;
            const RecOp* o = b.op_of(i.op);
            REQUIRE(o != nullptr);
            if (o->op == "all_reduce") lane1.push_back(i.n);
            else ++dense1;
        }
        if (k.cut == 0) {
            CHECK(lane1.empty());
            CHECK_EQ(dense1, 3 * (int)L);    /* gate_up, act_mul, down beside the routed arm */
        } else {
            CHECK_EQ(dense1, 0);
            REQUIRE(lane1.size() == (size_t)(4 * L));
            for (int64_t l = 0; l < L; ++l) {
                CHECK_EQ(lane1[4 * l + 0], k.cut * E);               /* attention, slice A */
                CHECK_EQ(lane1[4 * l + 1], (k.T - k.cut) * E);       /* attention, slice B */
                CHECK_EQ(lane1[4 * l + 2], 2 * k.cut * E);           /* both arms, slice A */
                CHECK_EQ(lane1[4 * l + 3], 2 * (k.T - k.cut) * E);   /* both arms, slice B */
            }
        }

        /* The whole step on the lossy wire: the same rows by op and routed passes, reductions on
         * lane 0, only the dense arm on lane 1. */
        RadBuilder w;
        CHECK_OK(model(w, true));
        CHECK(noted(w, "prefill overlap: off (the lossy all-reduce wire)"));
        CHECK(std::count(w.concurrent.begin(), w.concurrent.end(), (rad_buf)buf_named(w, "o")) == 0);
        RadBatch wbatch = one_prompt(k.T, out);
        RadCtx wc;
        run(w, wc, wbatch);
        CHECK(wc.step_fail.empty());
        CHECK_EQ(races(w, wc), 0);
        for (const auto& i : wc.issues) {
            const RecOp* o = w.op_of(i.op);
            if (i.lane == 1) CHECK(o == nullptr || o->op != "all_reduce");
        }
        CHECK(rows_by_op(b, c, k.T) == rows_by_op(w, wc, k.T));
        CHECK(ns_of(b, c, "moe_gather") == ns_of(w, wc, "moe_gather"));
        CHECK_EQ((int)ns_of(b, c, "cast").size(), k.cut ? 6 : 3);   /* three taps, per slice */
    }

    /* A decode step: reductions on lane 0, the dense arm on lane 1 joined before the feed-forward
     * reduction, all ordered. */
    RadBuilder b;
    CHECK_OK(model(b, false));
    CHECK(noted(b, "the two arms: in a step that runs whole"));
    RadBatch d = decode_batch();
    RadCtx c;
    run(b, c, d);
    CHECK(c.step_fail.empty());
    CHECK_EQ(c.lane, 0);
    CHECK_EQ(races(b, c), 0);
    int dense1 = 0;
    for (const auto& i : c.issues) {
        const RecOp* o = b.op_of(i.op);
        REQUIRE(o != nullptr);
        if (o->op == "all_reduce") CHECK_EQ(i.lane, 0);
        if (i.lane == 1) {
            ++dense1;
            CHECK(o->op == "gemm_nt" || o->op == "act_mul");
        }
    }
    CHECK_EQ(dense1, 3 * (int)L);
    CHECK_EQ((int)c.joins.size(), 2 * (int)L);
}

/* Where no loaded library declares the fused ops (CPU-only, or no libr4d), asking for one would
 * stop the engine, so the plugin checks rad_op_declared first and declares the chains: rmsnorm /
 * add joins, embed_lookup and mul for the expert scale, and q lifted for attn_paged. */
TEST(gemma4_without_the_fused_ops_declares_the_chains_and_asks_for_no_undeclared_op) {
    const RadEncoding fp8 = rad_enc_affine(RAD_F8E4M3, RAD_BF16, 128, 128);
    RadBuilder b;
    b.encs = { { "ffn_gate_up_exps", fp8 } };
    for (const char* op : { "norm_add_norm", "norm_add_norm_route", "join_add_scale",
                            "expert_scale", "attn_paged_qrow" })
        b.undeclared.insert(op);
    RadModelMeta meta = gemma4_moe_meta();
    RadBuildCtx  ctx  = build_ctx();
    CHECK_OK(gemma4::declare(&b, &meta, &ctx));
    CHECK(b.asked_undeclared.empty());
    CHECK(noted(b, "sandwich joins: 0 of 12 fused"));
    CHECK(noted(b, "routed inputs: 0 of 6"));
    CHECK(noted(b, "sliding attention: q lifted"));

    RadBatch batch = decode_batch();
    RadCtx c{ .b = &b, .batch = &batch };
    gemma4::step(&c, &batch);
    CHECK_LINT(b, &c);
    CHECK(c.step_fail.empty());
    const std::vector<std::string> ops = issued_ops(b, c);
    CHECK_EQ(count_issued(ops, "rmsnorm_quant_fp8"), 6);
    CHECK_EQ(count_issued(ops, "cast"), 5);
    CHECK(count_issued(ops, "add") > 0);
}

/* On a layer whose gate_up is block fp8 the attention join (norm_add_norm_route) also writes the
 * routed arm's inputs: the router's norm into moe_rn and the experts' codes into moe_hq / moe_hqs,
 * so neither the router's rmsnorm nor rmsnorm_quant_fp8 is issued. Refused: norm_add_norm plus
 * those two launches. A tp-4 prefill through the overlap keeps every reduction ordered too. */
TEST(gemma4_attention_join_writes_the_routed_inputs_on_fp8_experts) {
    const RadEncoding fp8 = rad_enc_affine(RAD_F8E4M3, RAD_BF16, 128, 128);
    std::map<std::string, int> issued[2];
    for (int refuse = 0; refuse < 2; ++refuse) {
        RadBuilder b;
        b.encs = { { "ffn_gate_up_exps", fp8 } };
        if (refuse) b.refuse.insert("norm_add_norm_route");
        RadModelMeta meta = gemma4_moe_meta();
        RadBuildCtx  ctx  = build_ctx();
        CHECK_OK(gemma4::declare(&b, &meta, &ctx));
        CHECK_EQ(count_resolved(b, "norm_add_norm_route"), refuse ? 0 : 6);
        CHECK_EQ(count_resolved(b, "norm_add_norm"), refuse ? 6 : 0);
        CHECK(noted(b, refuse ? "routed inputs: 0 of 6" : "routed inputs: 6 of 6"));
        if (!refuse) {
            const RecOp* r = find_op(b, "norm_add_norm_route");
            REQUIRE(r != nullptr);
            for (const char* name : { "moe_rn", "moe_hq", "moe_hqs", "x", "h" })
                CHECK(std::count(r->writes.begin(), r->writes.end(),
                                 (rad_buf)buf_named(b, name)) == 1);
        }

        RadBatch batch = decode_batch();
        RadCtx c{ .b = &b, .batch = &batch };
        gemma4::step(&c, &batch);
        CHECK_LINT(b, &c);
        CHECK(c.step_fail.empty());
        for (const auto& n : issued_ops(b, c)) ++issued[refuse][n];
    }
    CHECK_EQ(issued[0]["norm_add_norm_route"], 6);
    CHECK_EQ(issued[0]["norm_add_norm"], 0);
    CHECK_EQ(issued[0]["rmsnorm_quant_fp8"], 0);
    CHECK_EQ(issued[1]["norm_add_norm_route"], 0);
    CHECK_EQ(issued[1]["norm_add_norm"], 6);
    CHECK_EQ(issued[1]["rmsnorm_quant_fp8"], 6);
    CHECK_EQ(issued[1]["rmsnorm"] - issued[0]["rmsnorm"], 6);   /* the router's input norm */
    CHECK_EQ(issued[0]["moe_gemm_q"], 6);

    /* tp 4, an 8192-token chunk in two slices */
    RadBuilder b;
    b.encs = { { "ffn_gate_up_exps", fp8 } };
    RadModelMeta meta = gemma4_moe_meta();
    RadBuildCtx  ctx  = build_ctx(0, 4);
    ctx.max_tok = 8192;
    CHECK_OK(gemma4::declare(&b, &meta, &ctx));
    CHECK(noted(b, "prefill overlap: a chunk of two routed passes"));
    const int32_t out[1] = { 8191 };
    RadBatch batch = one_prompt(8192, out);
    RadCtx c{ .b = &b, .batch = &batch, .world = 4 };
    gemma4::step(&c, &batch);
    CHECK_LINT(b, &c);
    CHECK(c.step_fail.empty());
    CHECK_EQ(races(b, c), 0);
    CHECK_EQ(count_issued(issued_ops(b, c), "norm_add_norm_route"), 12);
    CHECK_EQ(count_issued(issued_ops(b, c), "rmsnorm_quant_fp8"), 0);
}

/* At four ranks the junctions carry their reductions at decode heights, in the same issue order as
 * at two ranks (only the embedding's and the last layer's feed-forward reduction stand alone),
 * through ar_rmsnorm_quant_i8 declared at world_size 4 up to 4 rows (kAr4FoldRows: where the fold
 * still beats the pair, inside libr4d's one-shot band of 524288 / 5376 = 97). A taller step, a DFlash
 * verify's 16 rows included, issues each reduction and its junction apart, and where no library
 * declares the fold every step issues the pair. */
TEST(a_gemma4_w4a8_step_at_four_ranks_folds_the_all_reduce_into_the_junction_at_decode_heights) {
    const int L = 12;
    std::vector<std::string> folded = { "embed_lookup", "all_reduce", "rmsnorm_quant_i8" };
    std::vector<std::string> apart  = folded;
    for (int l = 0; l < L; ++l) {
        const bool global = (l + 1) % 6 == 0;
        for (int i = 0; i < (global ? 2 : 3); ++i) { folded.push_back("gemm_nt_q"); apart.push_back("gemm_nt_q"); }
        for (const char* s : { "qkv_norm_rope", "kv_store", "attn_paged", "quant_act_i8", "gemm_nt_q",
                               "ar_rmsnorm_quant_i8", "gemm_nt_q", "gelu_mul_quant_i8", "gemm_nt_q" })
            folded.push_back(s);
        for (const char* s : { "qkv_norm_rope", "kv_store", "attn_paged", "quant_act_i8", "gemm_nt_q",
                               "all_reduce", "rmsnorm_quant_i8", "gemm_nt_q", "gelu_mul_quant_i8",
                               "gemm_nt_q", "all_reduce" })
            apart.push_back(s);
        if (l < L - 1) {
            folded.push_back("ar_rmsnorm_quant_i8");
            apart.push_back("rmsnorm_quant_i8");
        } else {
            for (const char* s : { "all_reduce", "rmsnorm", "add", "scale" }) folded.push_back(s);
            for (const char* s : { "rmsnorm", "add", "scale" }) apart.push_back(s);
        }
    }
    for (const char* s : { "rmsnorm", "gather_rows", "logits_gemm", "logit_softcap" }) {
        folded.push_back(s);
        apart.push_back(s);
    }
    /* a decode step folds */
    CHECK_EQ(join(gemma4_w4_step({}, 4)), join(folded));

    auto declare4 = [](RadBuilder& b) {
        const RadEncoding w4 = rad_enc_affine(RAD_I4, RAD_F16, 1, 128);
        b.encs = { { "attn_q.", w4 }, { "attn_k.", w4 }, { "attn_v.", w4 }, { "attn_o.", w4 },
                   { "ffn_gate_up.", w4 }, { "ffn_down.", w4 } };
        RadModelMeta meta = gemma4_meta();
        RadBuildCtx  ctx  = build_ctx(0, 4);
        return gemma4_bf16::declare(&b, &meta, &ctx);
    };
    RadBuilder b;
    CHECK_OK(declare4(b));
    CHECK(noted(b, "folded in at steps of 4 rows or fewer"));
    auto fold_hi = [](const RadBuilder& bb, int* n) {
        int64_t hi = -1;
        *n = 0;
        for (const auto& o : bb.ops) {
            if (o.op != "ar_rmsnorm_quant_i8") continue;
            ++*n;
            CHECK_EQ(pint(&o, "world_size"), 4);
            for (const auto& p : o.p)
                if (p.key == "M") hi = p.ihi;
        }
        return hi;
    };
    int n_ar = 0;
    CHECK_EQ(fold_hi(b, &n_ar), 4);
    CHECK_EQ(n_ar, 2 * L - 1);

    /* a verify's height and a step past the one-shot's band both issue the pair */
    const int32_t out[1] = { 127 };
    for (int rows : { 16, 128 }) {
        RadBatch tall = one_prompt(rows, out);
        RadCtx c; c.b = &b; c.rank = 0; c.world = 4; c.batch = &tall;
        gemma4_bf16::step(&c, &tall);
        CHECK(c.step_fail.empty());
        CHECK_EQ(join(issued_ops(b, c)), join(apart));
    }


    /* no library declares the fold: none declared, every step issues the pair */
    RadBuilder off;
    off.undeclared.insert("ar_rmsnorm_quant_i8");
    CHECK_OK(declare4(off));
    CHECK(noted(off, "issued apart"));
    for (const auto& o : off.ops) CHECK(o.op != "ar_rmsnorm_quant_i8");
}

/* ==================================================================== the step lint, checked */

namespace {

/* A hand-built program for the lint's own checks: two transients of one arena, A and X, and four
 * ops declared in the order write A, read A, write X, read X, so A lives over ops 1..2 and X over
 * 3..4 and the plan may give them the same bytes. A KV group binds layer 0 only. */
struct LintProgram {
    RadBuilder b;
    rad_buf    A = 0, X = 0;
    rad_op     wA = 0, rA = 0, wX = 0, rX = 0;
    rad_kvgroup kv = 0;
    LintProgram() {
        RadBufDecl d{};
        d.dtype = RAD_BF16; d.rank = 2; d.shape[0] = 8; d.shape[1] = 16;
        A = rad_decl_buffer(&b, "A", &d);
        X = rad_decl_buffer(&b, "X", &d);
        wA = rw(b, "fill", {}, { A });
        rA = rw(b, "copy", { A }, {});
        wX = rw(b, "fill", {}, { X });
        rX = rw(b, "copy", { X }, {});
        RadKVGroupDecl g{};
        g.kind = RAD_KV_FULL; g.dtype = RAD_BF16; g.n_head_kv = 1; g.head_dim = 16;
        kv = rad_decl_kv_group(&b, "kv", &g);
        rad_bind_layer_kv(&b, 0, kv);
    }
    static rad_op rw(RadBuilder& b, const char* name, std::initializer_list<rad_buf> r,
                     std::initializer_list<rad_buf> w) {
        const rad_op h = rad_decl_op(&b, name, nullptr, 0, nullptr, 0);
        rad_op_reads(&b, h, r.begin(), (int)r.size());
        rad_op_writes(&b, h, w.begin(), (int)w.size());
        return h;
    }
    std::vector<std::string> lint(std::initializer_list<std::pair<rad_op, RadOperand>> issues) {
        RadCtx c{ .b = &b };
        for (const auto& [op, o] : issues) rad_issue(&c, op, &o, 1, RAD_N_BATCH);
        return lint_step(b, { &c });
    }
};

bool any_has(const std::vector<std::string>& v, const char* s) {
    for (const auto& x : v) if (has(x, s)) return true;
    return false;
}

}  /* namespace */

/* The lint's rules, each on a program built to break exactly one of them. A lint that cannot
 * fire would let every plugin test below pass vacuously. */
TEST(the_step_lint_reports_each_rule_it_checks) {
    {   /* Issued in declaration order: nothing to report. */
        LintProgram p;
        const auto f = p.lint({ { p.wA, RAD_B(p.A) }, { p.rA, RAD_B(p.A) },
                                { p.wX, RAD_B(p.X) }, { p.rX, RAD_B(p.X) } });
        CHECK_EQ(report_lint(f), (size_t)0);
    }
    {   /* X written between A's write and its read, with disjoint planned ranges. */
        LintProgram p;
        const auto f = p.lint({ { p.wA, RAD_B(p.A) }, { p.wX, RAD_B(p.X) },
                                { p.rA, RAD_B(p.A) }, { p.rX, RAD_B(p.X) } });
        REQUIRE_EQ(f.size(), (size_t)1);
        CHECK(has(f[0], "reads 'A' (ops 1..2) after a write of 'X' (ops 3..4)"));
    }
    {   /* The same order is safe when A stays live through X's ops: the plan keeps them apart. */
        LintProgram p;
        rad_op_reads(&p.b, p.rX, &p.A, 1);
        const auto f = p.lint({ { p.wA, RAD_B(p.A) }, { p.wX, RAD_B(p.X) },
                                { p.rA, RAD_B(p.A) }, { p.rX, RAD_B(p.X) } });
        CHECK_EQ(report_lint(f), (size_t)0);
    }
    {   /* An op issued with a buffer outside the buffer's planned ops. */
        LintProgram p;
        const auto f = p.lint({ { p.wA, RAD_B(p.A) }, { p.wX, RAD_B(p.A) } });
        CHECK(any_has(f, "undeclared use of 'A' (planned ops 1..2)"));
    }
    {   /* Rows past the declared extent, and an offset that runs a slice off the end. */
        LintProgram p;
        CHECK(any_has(p.lint({ { p.wA, RAD_B_N(p.A, 0, 9) } }), "span past 'A'"));
        CHECK(any_has(p.lint({ { p.wA, RAD_B_N(p.A, 16 * 7 + 1, 1) } }), "span past 'A'"));
        CHECK_EQ(report_lint(p.lint({ { p.wA, RAD_B_N(p.A, 16 * 7, 1) } })), (size_t)0);
        /* A column slice keeps the declared pitch: 8 rows of the last 4 columns fit. */
        RadOperand col = RAD_B_COL(p.A, 12, 4);
        col.rows = 8;
        CHECK_EQ(report_lint(p.lint({ { p.wA, col } })), (size_t)0);
        col.offset = 13;
        CHECK(any_has(p.lint({ { p.wA, col } }), "span past 'A'"));
    }
    {   /* A fused op that did not resolve, issued anyway. */
        LintProgram p;
        CHECK(any_has(p.lint({ { RAD_NULL_HANDLE, RAD_B(p.A) } }), "null op issued"));
    }
    {   /* A transient read before anything in the step wrote it. */
        LintProgram p;
        CHECK(any_has(p.lint({ { p.rA, RAD_B(p.A) } }),
                      "reads 'A' before any issue of this step wrote it"));
    }
    {   /* A cache operand naming a layer no binding gave the group. */
        LintProgram p;
        CHECK_EQ(report_lint(p.lint({ { p.wA, RAD_KV(p.kv, 0) } })), (size_t)0);
        CHECK(any_has(p.lint({ { p.wA, RAD_KV(p.kv, 1) } }), "layer 1 is not bound"));
    }
    {   /* A GEMM fed a column slice narrower than its buffer; the whole width is fine. */
        LintProgram p;
        const rad_op mm = LintProgram::rw(p.b, "gemm_nt", { p.A }, { p.X });
        const RadOperand half = RAD_B_COL(p.A, 0, 8), whole = RAD_B_COL(p.A, 0, 16);
        CHECK(any_has(p.lint({ { p.wA, RAD_B(p.A) }, { mm, half } }), "not contiguous"));
        CHECK_EQ(report_lint(p.lint({ { p.wA, RAD_B(p.A) }, { mm, whole } })), (size_t)0);
    }
    {   /* A raw pointer into a buffer counts as that buffer. */
        LintProgram p;
        RadCtx c{ .b = &p.b };
        const RadOperand o = RAD_P_T((char*)rad_buf_ptr(&c, p.A) + 32, RAD_BF16);
        CHECK(any_has(p.lint({ { p.wX, o } }), "undeclared use of 'A' through a raw pointer"));
    }
}

/* ====================================================== qwen35_fp8 and qwen35moe_fp8 */

namespace {

/* `fn`'s status, with what it prints kept out of the test output. */
int stderr_of_rc(const std::function<int()>& fn) {
    int rc = 0;
    stderr_of([&] { rc = fn(); });
    return rc;
}

/* Model metadata built key by key; `meta()` is valid while this object lives. */
struct MetaKV {
    RadModelMeta              m{};
    std::vector<std::string>  keys, vals;
    std::vector<const char*>  kp, vp;
    explicit MetaKV(const RadModelMeta& base) : m(base) {
        for (int i = 0; i < base.n_kv; ++i) set(base.kv_key[i], base.kv_val[i]);
    }
    MetaKV& set(const char* k, const char* v) {
        for (size_t i = 0; i < keys.size(); ++i)
            if (keys[i] == k) { vals[i] = v; return *this; }
        keys.push_back(k);
        vals.push_back(v);
        return *this;
    }
    MetaKV& drop(const char* k) {
        for (size_t i = 0; i < keys.size(); ++i)
            if (keys[i] == k) {
                keys.erase(keys.begin() + (long)i);
                vals.erase(vals.begin() + (long)i);
                break;
            }
        return *this;
    }
    const RadModelMeta* meta() {
        kp.clear(); vp.clear();
        for (size_t i = 0; i < keys.size(); ++i) {
            kp.push_back(keys[i].c_str());
            vp.push_back(vals[i].c_str());
        }
        m.n_kv = (int)kp.size();
        m.kv_key = kp.data();
        m.kv_val = vp.data();
        return &m;
    }
};

/* Qwen3.8-27B-FP8's shape cut to the eight-layer fixture, its MTP head declared. */
MetaKV q35fp8_meta() {
    MetaKV k(qwen35_meta());
    k.m.quant = "fp8_e4m3";
    k.m.name  = "test-qwen35-27b-fp8";
    k.set("qwen35.nextn_predict_layers", "1");
    return k;
}

/* The 35B-A3B's routed feed-forward on the same trunk: 16 experts of 512, 4 a token, and a shared
 * expert of 512 (cut from 256 / 8 at 512). `n_ff` is the expert width, as rad-convert writes it. */
MetaKV q35moe_meta() {
    MetaKV k = q35fp8_meta();
    k.m.name = "test-qwen35-35b-a3b";
    k.m.n_ff = 512;
    k.m.n_expert = 16;
    k.m.n_expert_used = 4;
    k.set("shared_expert_intermediate_size", "512");
    return k;
}

/* A DFlash2 drafter merged under `draft.`: two layers reading trunk layers 1, 3 and 5, a block
 * of 8 (seven drafts), the selector of rank 64 over 8 candidates. */
MetaKV with_dflash2(MetaKV k) {
    k.set("draft.architectures", "DFlash2DraftModel")
     .set("draft.num_hidden_layers", "2")
     .set("draft.head_dim", "128")
     .set("draft.num_attention_heads", "32")
     .set("draft.num_key_value_heads", "8")
     .set("draft.intermediate_size", "8192")
     .set("draft.rms_norm_eps", "1e-06")
     .set("draft.rope_theta", "10000000")
     .set("draft.vocab_size", "248320")
     .set("draft.dflash_config.target_layer_ids", "1 3 5")
     .set("draft.dflash_config.block_size", "8")
     .set("draft.dflash_config.mask_token_id", "248070")
     .set("draft.dflash_config.selector_rank", "64")
     .set("draft.dflash_config.selector_top_k", "8")
     .set("draft.dflash_config.conv_kernel_size", "2")
     .set("draft.dflash_config.conv_group_size", "16");
    return k;
}

RadBuildCtx spec_ctx(int rank, int world, int max_spec) {
    RadBuildCtx c = build_ctx(rank, world);
    c.max_spec = max_spec;
    return c;
}

/* Every gemm_nt handle a pass issues is issued once: the bf16 GEMMs take a whole pass in one
 * launch (libr4d's bf16 row serves M to 65536), so a slice loop would be a stale band. */
bool each_gemm_nt_once(const RadBuilder& b, const RadCtx& c) {
    std::map<rad_op, int> n;
    for (const auto& i : c.issues) {
        const RecOp* o = b.op_of(i.op);
        if (o && o->op == "gemm_nt") ++n[i.op];
    }
    for (const auto& [op, k] : n)
        if (k != 1) {
            fprintf(stderr, "    gemm_nt op %u issued %d times in one pass\n", (unsigned)op, k);
            return false;
        }
    return true;
}

/* The engine steps a serial-drafter or block-drafter model runs, each linted whole: decode,
 * prefill, a mixed step, and a verify followed by the drafter's passes. */
template <class Fn>
void lint_qwen35_fp8_family(Fn declare, StepFn step, const RadModelMeta* meta, int max_spec,
                            bool block_drafter, int64_t block) {
    for (int world : { 1, 2 })
        for (int rank = 0; rank < world; ++rank) {
            RadBuilder b;
            RadBuildCtx ctx = spec_ctx(rank, world, max_spec);
            REQUIRE_EQ(declare(&b, meta, &ctx), RAD_OK);
            const auto dec = kit(b, { 1, 1, 1, 1 }, {});
            const auto pre = kit(b, {}, { 32, 32 });
            const auto mix = kit(b, { 1, 1 + max_spec }, { 6 });
            const auto ver = kit(b, std::vector<int>(4, 1 + max_spec), {});
            for (const Kit* k : { dec.get(), pre.get(), mix.get() }) {
                const auto v = run_passes(b, step, { k }, rank, world);
                CHECK_PASSES(b, v);
            }
            std::vector<std::unique_ptr<Kit>> passes;
            if (block_drafter) {
                passes.push_back(context_pass(b, *ver));
                passes.push_back(query_pass(b, 4, block, block - 1));
            } else {
                passes.push_back(history_pass(b, *ver));
                for (int r = 2; r <= max_spec; ++r) passes.push_back(round_pass(b, 4, r));
            }
            std::vector<const Kit*> all = { ver.get() };
            for (const auto& p : passes) all.push_back(p.get());
            const auto v = run_passes(b, step, all, rank, world);
            CHECK_PASSES(b, v);
            for (size_t i = 1; i < v.size(); ++i) CHECK(each_gemm_nt_once(b, *v[i]));
        }
}

}  /* namespace */

/* The fp8 sibling of qwen35_bf16: the same layers with every large linear a block-fp8 gemm_nt_q
 * read at fp8a8, one quantiser per activation, and the attention's output gate, multiply and
 * quantiser fused into gate_quant_fp8 (the sigmoid/mul pair is the bf16-linear form only). The
 * delta net's a|b projection stays bf16: its 48 rows are not a whole 128-row scale block. */
TEST(qwen35_fp8_declares_fp8_linears_and_the_fused_output_gate) {
    MetaKV mk = q35fp8_meta();
    RadBuilder b;
    RadBuildCtx ctx = build_ctx();
    REQUIRE_EQ(qwen35_fp8::declare(&b, mk.meta(), &ctx), RAD_OK);
    int q = 0;
    for (const auto& o : b.ops) {
        if (o.op != "gemm_nt_q") continue;
        ++q;
        CHECK_EQ(pstr(&o, "dtype"), std::string("fp8a8"));
        CHECK_EQ(pint(&o, "group"), 128);
    }
    /* q|gate, k, v, o for 2 attention layers; qkv|z and out for 6 delta-net layers; gate_up and
     * down for 8 feed-forwards. */
    CHECK_EQ(q, 2 * 4 + 6 * 2 + 8 * 2);
    CHECK_EQ(count_op(b, "gate_quant_fp8"), 2);
    CHECK_EQ(count_op(b, "sigmoid"), 0);
    CHECK_EQ(count_op(b, "mul"), 0);
    CHECK_EQ(count_op(b, "gemm_nt"), 6);
    for (const auto& o : b.ops)
        if (o.op == "gemm_nt") CHECK(in(o.writes, (rad_buf)buf_named(b, "gdn_ab")));
    for (const auto& o : b.ops)
        if (o.resolved) CHECK(o.uses_declared);
    for (const auto& w : b.weights)
        if (!w.d.source) CHECK(find_map(b, w.name) != nullptr);
    /* No speculation: no head, no drafter. */
    CHECK_EQ((int)b.drafter.kind, (int)RAD_DRAFT_NONE);
    CHECK(find_b(b, "mtp_x") == nullptr);
}

/* Every engine step the dense fp8 model runs lints clean at one and two ranks, the MTP head's
 * history pass and rounds included. */
TEST(qwen35_fp8_steps_lint_clean_with_its_mtp_head) {
    MetaKV mk = q35fp8_meta();
    lint_qwen35_fp8_family(qwen35_fp8::declare, qwen35_fp8::step, mk.meta(), 3, false, 0);
}

namespace {

/* Every trunk and MTP linear of a qwen35 container in the 4-bit form an int4 W4A16 release
 * imports to: u4 codes, an f16 scale and a u4 zero point per 128 of K. */
std::vector<std::pair<std::string, RadEncoding>> int4_linears() {
    RadEncoding w4 = rad_enc_affine(RAD_U4, RAD_F16, 1, 128);
    rad_enc_add_plane(&w4, "zero", RAD_U4, 1, 128);
    std::vector<std::pair<std::string, RadEncoding>> e;
    for (const char* k : { "attn_qg.weight", "attn_k.weight", "attn_v.weight",
                           "attn_output.weight", "ssm_inz.weight", "ssm_out.weight",
                           "ffn_gate_up.weight", "ffn_down.weight" })
        e.emplace_back(k, w4);
    return e;
}

int qwen35_fp8_int4_declare(RadBuilder* b, const RadModelMeta* m, const RadBuildCtx* c) {
    b->encs = int4_linears();
    return qwen35_fp8::declare(b, m, c);
}

}  /* namespace */

/* An int4 container's linears are read by the 4-bit GEMMs over their input's bf16 plane: at or
 * below 64 rows a `cast` to f16 and w4a16_asym, above that quant_act_i8g's int8 codes and
 * w4a8i8_asym, both over the codes and the joined (scale, zero) view. No linear keeps an fp8a8
 * GEMM, and the gated fold (an fp8a8 tile) is not declared over a 4-bit gate_up. */
TEST(qwen35_fp8_reads_an_int4_container_with_the_4bit_gemms) {
    MetaKV mk = q35fp8_meta();
    RadBuilder b;
    RadBuildCtx ctx = build_ctx();
    REQUIRE_EQ(qwen35_fp8_int4_declare(&b, mk.meta(), &ctx), RAD_OK);
    int f16 = 0, i8 = 0, other = 0;
    for (const auto& o : b.ops) {
        if (o.op != "gemm_nt_q") continue;
        const std::string dt = pstr(&o, "dtype");
        const RecParam* m = param(&o, "M");
        REQUIRE(m != nullptr);
        if (dt == "w4a16_asym") {
            ++f16;
            CHECK_EQ(m->ival, 1LL);
            CHECK_EQ(m->ihi, 64LL);
        } else if (dt == "w4a8i8_asym") {
            ++i8;
            CHECK_EQ(m->ival, 65LL);
            CHECK_EQ(m->ihi, 256LL);
        } else {
            ++other;
        }
        CHECK_EQ(pint(&o, "group"), 128);
    }
    /* q|gate, k, v, o for 2 attention layers; qkv|z and out for 6 delta-net layers; gate_up and
     * down for 8 feed-forwards: 36 linears, each with both GEMMs, a cast and a quantiser. */
    CHECK_EQ(f16, 36);
    CHECK_EQ(i8, 36);
    CHECK_EQ(other, 0);
    int casts = 0;
    for (const auto& o : b.ops)
        if (o.op == "cast" && pstr(&o, "to") == "f16") {
            ++casts;
            CHECK_EQ(pstr(&o, "from"), std::string("bf16"));
            CHECK_EQ(param(&o, "M")->ihi, 64LL);
        }
    CHECK_EQ(casts, 36);
    CHECK_EQ(count_op(b, "quant_act_i8g"), 36);
    CHECK_EQ(count_op(b, "gemm_nt_q_gated"), 0);
    int joined = 0;
    for (const RecWeight& w : b.weights)
        if (w.d.planes && std::string(w.d.planes) == "scale,zero") {
            ++joined;
            CHECK_EQ(w.d.dtype, (uint32_t)RAD_F16);
        }
    CHECK_EQ(joined, 36);
}

/* Every engine step an int4 container runs lints clean at one and two ranks, the MTP head's
 * history pass and rounds included: the 4-bit linears' row parts and shards are cut like the
 * block-fp8 ones. */
TEST(qwen35_fp8_int4_steps_lint_clean_with_its_mtp_head) {
    MetaKV mk = q35fp8_meta();
    lint_qwen35_fp8_family(qwen35_fp8_int4_declare, qwen35_fp8::step, mk.meta(), 3, false, 0);
}

/* The head is a serial drafter at the deployment's depth, declared after the trunk's lm_head:
 * the planner reads declaration order as liveness, and the head's passes run after the trunk. */
TEST(qwen35_fp8_declares_its_mtp_head_after_the_trunk) {
    MetaKV mk = q35fp8_meta();
    RadBuilder b;
    RadBuildCtx ctx = spec_ctx(0, 1, 3);
    REQUIRE_EQ(qwen35_fp8::declare(&b, mk.meta(), &ctx), RAD_OK);
    CHECK_EQ((int)b.drafter.kind, (int)RAD_DRAFT_SERIAL);
    CHECK_EQ(b.drafter.depth, 3);
    CHECK_EQ(b.drafter_name, std::string("mtp"));
    size_t logits = 0, first_head = 0;
    for (size_t i = 0; i < b.ops.size() && !logits; ++i)
        if (b.ops[i].op == "logits_gemm") logits = i + 1;
    const rad_buf hx = (rad_buf)buf_named(b, "mtp_x");
    for (size_t i = 0; i < b.ops.size() && !first_head; ++i)
        if (in(b.ops[i].reads, hx) || in(b.ops[i].writes, hx)) first_head = i + 1;
    REQUIRE(logits > 0 && first_head > 0);
    CHECK(first_head > logits);
    /* The head binds one more layer of the trunk's attention group, at index n_layers. */
    bool bound = false;
    for (const auto& [layer, g] : b.binds) bound |= layer == 8 && g == (rad_kvgroup)1;
    CHECK(bound);
    /* A draft round-1 pass is refused: the history pass carries it. */
    const auto one = round_pass(b, 4, 1);
    const auto v = run_passes(b, qwen35_fp8::step, { one.get() });
    CHECK(!v[0]->step_fail.empty());
}

/* DFlash2 replaces the MTP head when the container carries one: a block drafter whose block is
 * one longer than the depth, a power of two (its convolution masks positions with block - 1),
 * reading trunk taps copied out after layers 1, 3 and 5. */
TEST(qwen35_fp8_dflash2_drafter_takes_a_power_of_two_block) {
    MetaKV mk = with_dflash2(q35fp8_meta());
    {
        RadBuilder b;
        RadBuildCtx ctx = spec_ctx(0, 1, 7);
        REQUIRE_EQ(qwen35_fp8::declare(&b, mk.meta(), &ctx), RAD_OK);
        CHECK_EQ((int)b.drafter.kind, (int)RAD_DRAFT_BLOCK);
        CHECK_EQ(b.drafter_name, std::string("dflash2"));
        CHECK_EQ(b.drafter.depth, 7);
        CHECK_EQ(b.drafter.block, 8);
        CHECK_EQ(b.drafter.mask_token, 248070);
        CHECK(find_b(b, "mtp_x") == nullptr);           /* no MTP head beside it */
        CHECK(noted(b, "the DFlash2 drafter replaces the 1-layer MTP head"));
        const RecBuf* src = find_b(b, "dflash_src");
        REQUIRE(src != nullptr);
        CHECK_EQ(src->d.shape[1], 3 * 5120);
        /* The drafter's layers follow the trunk's in its own group. */
        std::set<int> drafted;
        for (const auto& [layer, g] : b.binds)
            if (g == (rad_kvgroup)b.kv_groups.size()) drafted.insert(layer);
        CHECK(drafted == (std::set<int>{ 8, 9 }));
    }
    {   /* Six drafts make a block of seven, which the convolution cannot mask. */
        RadBuilder b;
        RadBuildCtx ctx = spec_ctx(0, 1, 6);
        CHECK_EQ(stderr_of_rc([&] { return qwen35_fp8::declare(&b, mk.meta(), &ctx); }),
                 RAD_E_UNSUPPORTED);
    }
    {   /* Another power of two runs, with a note that it was trained at 8. */
        RadBuilder b;
        RadBuildCtx ctx = spec_ctx(0, 1, 15);
        REQUIRE_EQ(qwen35_fp8::declare(&b, mk.meta(), &ctx), RAD_OK);
        CHECK_EQ(b.drafter.block, 16);
        CHECK(noted(b, "DFlash2 was trained at block_size 8"));
    }
    {   /* The probe fixes the depth at the trained block, so the engine asks for 7. */
        RadArchProbe p{};
        CHECK_OK(qwen35_fp8::probe(mk.meta(), &p));
        CHECK_EQ(p.draft_depth, 7);
        CHECK_EQ(p.draft_depth_fixed, 1);
    }
}

/* The block drafter's engine step (verify, context pass, query pass) lints clean at one and two
 * ranks, and its bf16 GEMMs run once a pass. */
TEST(qwen35_fp8_steps_lint_clean_with_a_dflash2_drafter) {
    MetaKV mk = with_dflash2(q35fp8_meta());
    lint_qwen35_fp8_family(qwen35_fp8::declare, qwen35_fp8::step, mk.meta(), 7, true, 8);
}

/* =============================== gemma3: TranslateGemma 27B's trunk =========================== */

namespace {

/* TranslateGemma 27B's text geometry cut to twelve layers, two global (5 and 11) over ten sliding
 * ones by the legacy (l + 1) % 6 rule. One head geometry for both classes (head_dim 128, 32 / 16),
 * scores scaled by query_pre_attn_scalar^-0.5, linear rotary scaling on the global class only. */
MetaKV gemma3_meta() {
    RadModelMeta m{};
    m.arch_id     = "gemma3";
    m.name        = "test-translategemma-27b";
    m.quant       = "";
    m.n_layers    = 12;
    m.n_embd      = 5376;
    m.n_head      = 32;
    m.n_head_kv   = 16;
    m.head_dim    = 128;
    m.n_ff        = 21504;
    m.n_vocab     = 262208;
    m.n_ctx_train = 131072;
    m.rms_eps     = 1e-6f;
    m.rope_theta  = 1e6f;
    m.rope_scale  = 1.0f;
    MetaKV k(m);
    k.set("sliding_window", "1024")
     .set("_sliding_window_pattern", "6")
     .set("rope_local_base_freq", "10000")
     .set("rope_theta", "1000000")
     .set("query_pre_attn_scalar", "168")
     .set("rope_scaling.rope_type", "linear")
     .set("rope_scaling.factor", "8");
    return k;
}

MetaKV gemma3_dflash2_meta() {
    MetaKV k = gemma3_meta();
    k.set("draft.architectures", "DFlash2DraftModel")
     .set("draft.num_hidden_layers", "2")
     .set("draft.head_dim", "128")
     .set("draft.num_attention_heads", "32")
     .set("draft.num_key_value_heads", "8")
     .set("draft.hidden_size", "5376")
     .set("draft.intermediate_size", "8192")
     .set("draft.rms_norm_eps", "1e-06")
     .set("draft.rope_theta", "1000000")
     .set("draft.vocab_size", "262208")
     .set("draft.use_sliding_window", "1")
     .set("draft.sliding_window", "2048")
     .set("draft.dflash_config.input_embedding_scale", "73.5")
     .set("draft.dflash_config.target_layer_ids", "1 5 10")
     .set("draft.dflash_config.block_size", "8")
     .set("draft.dflash_config.mask_token_id", "6")
     .set("draft.dflash_config.selector_rank", "64")
     .set("draft.dflash_config.selector_top_k", "8")
     .set("draft.dflash_config.conv_kernel_size", "2")
     .set("draft.dflash_config.conv_group_size", "16");
    return k;
}

}  /* namespace */

/* Sliding layers in a window group declared first, the global ones in a full group, both at the
 * one head geometry; layers bound by the (l + 1) % 6 rule. */
TEST(gemma3_declares_two_kv_groups_and_binds_the_schedule) {
    MetaKV mk = gemma3_meta();
    RadBuilder b;
    RadBuildCtx ctx = build_ctx();
    REQUIRE_EQ(gemma3::declare(&b, mk.meta(), &ctx), RAD_OK);
    REQUIRE_EQ((int)b.kv_groups.size(), 2);
    CHECK_EQ(b.kv_groups[0].name, std::string("kv_sliding"));
    CHECK_EQ((int)b.kv_groups[0].d.kind, RAD_KV_WINDOW);
    CHECK_EQ(b.kv_groups[0].d.window, 1024);
    CHECK_EQ(b.kv_groups[0].d.n_head_kv, 16);
    CHECK_EQ(b.kv_groups[0].d.head_dim, 128);
    CHECK_EQ(b.kv_groups[1].name, std::string("kv_global"));
    CHECK_EQ((int)b.kv_groups[1].d.kind, RAD_KV_FULL);
    CHECK_EQ(b.kv_groups[1].d.window, 0);
    CHECK_EQ(b.kv_groups[1].d.head_dim, 128);
    CHECK_EQ((int)b.binds.size(), 12);
    for (int l = 0; l < 12; ++l) {
        CHECK_EQ((int)b.binds[(size_t)l].first, l);
        CHECK_EQ((int)b.binds[(size_t)l].second, (l + 1) % 6 == 0 ? 2 : 1);
    }
}

/* layer_types, where the config has it, names the schedule instead of the legacy pattern, and a
 * count that is not the model's is refused. */
TEST(gemma3_layer_types_name_the_global_layers) {
    MetaKV mk = gemma3_meta();
    mk.set("layer_types", "[\"full_attention\", \"sliding_attention\", \"sliding_attention\", "
                          "\"sliding_attention\", \"sliding_attention\", \"sliding_attention\", "
                          "\"sliding_attention\", \"sliding_attention\", \"sliding_attention\", "
                          "\"sliding_attention\", \"sliding_attention\", \"full_attention\"]");
    RadBuilder b;
    RadBuildCtx ctx = build_ctx();
    REQUIRE_EQ(gemma3::declare(&b, mk.meta(), &ctx), RAD_OK);
    for (int l = 0; l < 12; ++l)
        CHECK_EQ((int)b.binds[(size_t)l].second, (l == 0 || l == 11) ? 2 : 1);

    mk.set("layer_types", "[\"full_attention\", \"sliding_attention\"]");
    RadBuilder c;
    CHECK(stderr_of_rc([&] { return gemma3::declare(&c, mk.meta(), &ctx); }) != RAD_OK);
}

/* Both classes attend at 1/sqrt(query_pre_attn_scalar), not 1/sqrt(head_dim); the sliding class
 * is windowed and rotates at rope_local_base_freq unscaled, the global class at rope_theta with
 * the linear factor; every norm is GemmaRMSNorm (wadd 1). */
TEST(gemma3_attention_scale_rope_and_norms_per_class) {
    MetaKV mk = gemma3_meta();
    RadBuilder b;
    RadBuildCtx ctx = build_ctx();
    REQUIRE_EQ(gemma3::declare(&b, mk.meta(), &ctx), RAD_OK);
    std::vector<const RecOp*> at;
    for (const auto& o : b.ops) if (o.op == "attn_paged") at.push_back(&o);
    REQUIRE(!at.empty());
    bool windowed = false, full = false;
    for (const RecOp* o : at) {
        CHECK_EQ(pint(o, "head_dim"), 128);
        CHECK_EQ(pint(o, "gqa"), 2);
        CHECK(std::fabs(pflt(o, "scale") - 1.0 / std::sqrt(168.0)) < 1e-9);
        if (pint(o, "window") == 1024) windowed = true;
        if (pint(o, "window") == 0) full = true;
    }
    CHECK(windowed && full);
    std::set<std::pair<double, double>> rot;   /* (theta, scale) of every rotary op */
    for (const auto& o : b.ops)
        if (param(&o, "theta")) rot.insert({ pflt(&o, "theta"), pflt(&o, "scale") });
    CHECK(rot.count({ 10000.0, 1.0 }) == 1);
    CHECK(rot.count({ 1000000.0, 0.125 }) == 1 || rot.count({ 1000000.0, 8.0 }) == 1);
    for (const auto& o : b.ops)
        if (param(&o, "wadd")) CHECK_EQ(pflt(&o, "wadd"), 1.0);
}

/* Gemma 2's soft-capping and a YaRN scaling are refused rather than served without them. */
TEST(gemma3_refuses_softcapping_and_yarn) {
    RadBuildCtx ctx = build_ctx();
    {
        MetaKV mk = gemma3_meta();
        mk.set("final_logit_softcapping", "30");
        RadBuilder b;
        CHECK_EQ(stderr_of_rc([&] { return gemma3::declare(&b, mk.meta(), &ctx); }), RAD_E_UNSUPPORTED);
    }
    {
        MetaKV mk = gemma3_meta();
        mk.set("rope_scaling.rope_type", "yarn").set("rope_scaling.original_max_position_embeddings", "8192");
        RadBuilder b;
        CHECK(stderr_of_rc([&] { return gemma3::declare(&b, mk.meta(), &ctx); }) != RAD_OK);
    }
}

TEST(gemma3_every_op_declares_its_buffers_and_every_weight_has_a_name_map) {
    MetaKV mk = gemma3_meta();
    RadBuilder b;
    RadBuildCtx ctx = build_ctx();
    REQUIRE_EQ(gemma3::declare(&b, mk.meta(), &ctx), RAD_OK);
    for (const auto& o : b.ops)
        if (o.resolved) CHECK(o.uses_declared);
    for (const auto& w : b.weights)
        if (!w.d.source) CHECK(find_map(b, w.name) != nullptr);
}

/* A decode step and a prefill step lint clean at one and two ranks. */
TEST(gemma3_steps_lint_clean_at_one_and_two_ranks) {
    MetaKV mk = gemma3_meta();
    for (int world : { 1, 2 })
        for (int rank = 0; rank < world; ++rank) {
            RadBuilder b;
            RadBuildCtx ctx = build_ctx(rank, world);
            REQUIRE_EQ(gemma3::declare(&b, mk.meta(), &ctx), RAD_OK);
            RadBatch batch = decode_batch();
            RadCtx c{ .b = &b, .batch = &batch };
            gemma3::step(&c, &batch);
            CHECK_LINT(b, &c);
            CHECK(c.step_fail.empty());
            CHECK_EQ(count_op(b, "all_reduce") + count_op(b, "ar_rmsnorm_quant_fp8") +
                     count_op(b, "ar_rmsnorm_post_quant_fp8") > 0, world == 2);
        }
}

/* The DFlash2 drafter: a block of 8 (depth 7, fixed), taps copied out after the named layers into
 * one column-sliced buffer, the mask token from the config, and the drafter's own KV group after
 * the trunk's two. Its input_embedding_scale must be the trunk's sqrt(hidden) held in bf16
 * (73.5 on the 27B), which the embedding lookup applies. */
TEST(gemma3_dflash2_drafter_taps_and_block) {
    MetaKV mk = gemma3_dflash2_meta();
    RadBuilder b;
    RadBuildCtx ctx = spec_ctx(0, 1, 7);
    REQUIRE_EQ(gemma3::declare(&b, mk.meta(), &ctx), RAD_OK);
    CHECK_EQ((int)b.drafter.kind, (int)RAD_DRAFT_BLOCK);
    CHECK_EQ(b.drafter.depth, 7);
    CHECK_EQ(b.drafter.block, 8);
    CHECK_EQ(b.drafter.mask_token, 6);
    const RecBuf* src = find_b(b, "dflash_src");
    REQUIRE(src != nullptr);
    CHECK_EQ(src->d.shape[1], 3 * 5376);
    CHECK_EQ((int)b.kv_groups.size(), 3);
    CHECK(!noted(b, "input_embedding_scale"));
    {   /* A drafter trained at another embedding scale still serves, with a note. */
        MetaKV other = gemma3_dflash2_meta();
        other.set("draft.dflash_config.input_embedding_scale", "64");
        RadBuilder o;
        REQUIRE_EQ(gemma3::declare(&o, other.meta(), &ctx), RAD_OK);
        CHECK(noted(o, "input_embedding_scale"));
    }

    RadArchProbe p{};
    CHECK_OK(gemma3::probe(mk.meta(), &p));
    CHECK_EQ(p.draft_depth, 7);
    CHECK_EQ(p.draft_depth_fixed, 1);
    MetaKV plain = gemma3_meta();
    RadArchProbe q{};
    CHECK_OK(gemma3::probe(plain.meta(), &q));
    CHECK_EQ(q.draft_depth, 0);
}

/* --kv-cache-dtype fp8_v4: the trunk's attention cache is E4M3 K (256 bytes), int4 V (128) and
 * eight f16 scales, 400 bytes a token and head, declared as 25 * 256 / 32 = 200 E4M3 columns so the
 * core sizes its pages; every trunk op on it bands on kv_dtype fp8_v4. The DFlash2 drafter's own
 * group stays E4M3 at its head_dim. Without kv_quant the same build is plain fp8. */
TEST(qwen35_fp8_fp8_v4_cache_is_400_bytes_a_token_and_head) {
    MetaKV mk = with_dflash2(q35fp8_meta());
    auto count = [](const RadBuilder& b, const char* kd) {
        int n = 0;
        for (const auto& o : b.ops) if (pstr(&o, "kv_dtype") == kd) ++n;
        return n;
    };
    RadBuilder f8;
    RadBuildCtx c8 = spec_ctx(0, 1, 7);
    c8.kv_dtype = RAD_F8E4M3;
    REQUIRE_EQ(qwen35_fp8::declare(&f8, mk.meta(), &c8), RAD_OK);
    CHECK_EQ(f8.kv_groups[0].d.head_dim, 256);
    CHECK_EQ(count(f8, "fp8_v4"), 0);
    const int n8 = count(f8, "fp8_e4m3");

    RadBuilder b;
    RadBuildCtx ctx = c8;
    ctx.kv_quant = RAD_KVQ_V4;
    REQUIRE_EQ(qwen35_fp8::declare(&b, mk.meta(), &ctx), RAD_OK);
    CHECK_EQ(b.kv_groups[0].name, std::string("kv_attn"));
    const RadKVGroupDecl& f = b.kv_groups[0].d;
    CHECK_EQ(f.kind, RAD_KV_FULL);
    CHECK_EQ(f.dtype, (uint32_t)RAD_F8E4M3);
    CHECK_EQ(f.n_head_kv, 4);
    CHECK_EQ(f.head_dim, 200);
    const RadKVGroupDecl& d = b.kv_groups.back().d;
    CHECK_EQ(d.dtype, (uint32_t)RAD_F8E4M3);
    CHECK_EQ(d.head_dim, 128);
    const int v4 = count(b, "fp8_v4"), e4 = count(b, "fp8_e4m3");
    CHECK(v4 > 0);
    CHECK(e4 > 0);                       /* the drafter's attention */
    CHECK_EQ(v4 + e4, n8);
    CHECK(noted(b, "kv cache fp8_v4"));

    RadArchProbe p{};
    CHECK_OK(qwen35_fp8::probe(mk.meta(), &p));
    CHECK_EQ(p.kv_quant_ok, (1 << RAD_KVQ_V4) | (1 << RAD_KVQ_K6V4));
}

/* --kv-cache-dtype k6_v4: int6 K and int4 V, each in groups of 32 with an f16 scale, 352 bytes a
 * token and head, declared as 176 E4M3 columns; the trunk's ops band on kv_dtype k6_v4 and the
 * drafter's group stays E4M3. */
TEST(qwen35_fp8_k6_v4_cache_is_352_bytes_a_token_and_head) {
    MetaKV mk = with_dflash2(q35fp8_meta());
    RadBuilder b;
    RadBuildCtx ctx = spec_ctx(0, 1, 7);
    ctx.kv_dtype = RAD_F8E4M3;
    ctx.kv_quant = RAD_KVQ_K6V4;
    REQUIRE_EQ(qwen35_fp8::declare(&b, mk.meta(), &ctx), RAD_OK);
    const RadKVGroupDecl& f = b.kv_groups[0].d;
    CHECK_EQ(f.dtype, (uint32_t)RAD_F8E4M3);
    CHECK_EQ(f.head_dim, 176);
    CHECK_EQ(b.kv_groups.back().d.head_dim, 128);
    int k6 = 0, v4 = 0, e4 = 0;
    for (const auto& o : b.ops) {
        const std::string kd = pstr(&o, "kv_dtype");
        k6 += kd == "k6_v4"; v4 += kd == "fp8_v4"; e4 += kd == "fp8_e4m3";
    }
    CHECK(k6 > 0);
    CHECK(e4 > 0);
    CHECK_EQ(v4, 0);
    CHECK(noted(b, "kv cache k6_v4"));
}

/* libr4d's fp8_v4 and k6_v4 rows are h256/gqa6 only: the probe answers no for another geometry, so the
 * engine refuses the flag at startup, and declare refuses it too if asked anyway. */
TEST(qwen35_fp8_quantised_kv_is_refused_outside_h256_gqa6) {
    MetaKV mk = q35fp8_meta();
    mk.m.n_head_kv = 2;                  /* gqa 12 */
    RadArchProbe p{};
    CHECK_OK(qwen35_fp8::probe(mk.meta(), &p));
    CHECK_EQ(p.kv_quant_ok, 0);
    for (int q : { (int)RAD_KVQ_V4, (int)RAD_KVQ_K6V4 }) {
        RadBuilder b;
        RadBuildCtx ctx = build_ctx();
        ctx.kv_dtype = RAD_F8E4M3;
        ctx.kv_quant = q;
        CHECK_EQ(stderr_of_rc([&] { return qwen35_fp8::declare(&b, mk.meta(), &ctx); }),
                 RAD_E_UNSUPPORTED);
    }
}

/* The 35B-A3B: a replicated router over every expert, a normalised top-k (norm_topk_prob
 * defaults on in this family; without it the weights sum to about 0.4), fp8a8 grouped GEMMs over
 * the expert planes, and the shared expert folded into the gather behind a sigmoid gate. */
TEST(qwen35moe_fp8_routes_every_expert_and_folds_the_shared_arm) {
    MetaKV mk = q35moe_meta();
    for (int world : { 1, 2 }) {
        RadBuilder b;
        RadBuildCtx ctx = build_ctx(0, world);
        REQUIRE_EQ(qwen35moe_fp8::declare(&b, mk.meta(), &ctx), RAD_OK);
        CHECK_EQ(count_op(b, "router_topk"), 8);
        for (const auto& o : b.ops) {
            if (o.op == "router_topk") {
                CHECK_EQ(pint(&o, "n_expert"), 16);
                CHECK_EQ(pint(&o, "top_k"), 4);
                CHECK_EQ(pint(&o, "norm"), 1);
            }
            if (o.op == "moe_gemm_q") {
                CHECK_EQ(pstr(&o, "dtype"), std::string("fp8a8"));
                CHECK_EQ(pint(&o, "n_expert"), 16);
                const long long N = pint(&o, "N"), K = pint(&o, "K");
                CHECK((N == 2 * 512 / world && K == 5120) || (N == 5120 && K == 512 / world));
            }
            if (o.op == "moe_gather") CHECK_EQ(pstr(&o, "act"), std::string("sigmoid"));
        }
        CHECK_EQ(count_op(b, "moe_gather"), 8);
        CHECK(count_op(b, "moe_gemm_q") >= 16);
        /* The shared arm's buffers sit on the concurrent list, out of the linear packing. */
        CHECK(std::count(b.concurrent.begin(), b.concurrent.end(),
                         (rad_buf)buf_named(b, "shexp_gate_up")) > 0);
    }
    /* Without experts this is the dense plugin's checkpoint, and the reverse. */
    {
        MetaKV dense = q35fp8_meta();
        RadBuilder b;
        RadBuildCtx ctx = build_ctx();
        CHECK_EQ(qwen35moe_fp8::declare(&b, dense.meta(), &ctx), RAD_E_UNSUPPORTED);
        RadBuilder b2;
        CHECK_EQ(qwen35_fp8::declare(&b2, mk.meta(), &ctx), RAD_E_UNSUPPORTED);
    }
}

/* The routed model's engine steps, its routed MTP head included, lint clean at one and two
 * ranks. */
TEST(qwen35moe_fp8_steps_lint_clean_with_its_routed_mtp_head) {
    MetaKV mk = q35moe_meta();
    lint_qwen35_fp8_family(qwen35moe_fp8::declare, qwen35moe_fp8::step, mk.meta(), 3, false, 0);
}

/* `layer_types` is checked against the interval rule, not ignored: a model built with the wrong
 * layers linear still generates fluent text. A disagreeing, short or unknown schedule is refused
 * by every plugin of the family; an agreeing one declares. */
TEST(the_qwen35_family_refuses_a_layer_schedule_that_contradicts_its_interval) {
    const char* agree = "linear_attention linear_attention linear_attention full_attention "
                        "linear_attention linear_attention linear_attention full_attention";
    const char* wrong = "linear_attention linear_attention full_attention linear_attention "
                        "linear_attention linear_attention linear_attention full_attention";
    const char* shrt  = "linear_attention linear_attention linear_attention full_attention";
    const char* odd   = "linear_attention linear_attention linear_attention sliding_attention "
                        "linear_attention linear_attention linear_attention full_attention";
    for (const char* s : { agree, wrong, shrt, odd }) {
        const int want = s == agree ? RAD_OK : RAD_E_UNSUPPORTED;
        MetaKV b16(qwen35_meta());
        b16.set("layer_types", s);
        MetaKV f8 = q35fp8_meta();
        f8.set("layer_types", s);
        RadBuildCtx ctx = build_ctx();
        RadBuilder x, y;
        CHECK_EQ(stderr_of_rc([&] { return qwen35_bf16::declare(&x, b16.meta(), &ctx); }), want);
        CHECK_EQ(stderr_of_rc([&] { return qwen35_fp8::declare(&y, f8.meta(), &ctx); }), want);
    }
    /* Flash-Next reads the same key under its own spelling. */
    MetaKV q4(qwen4exp_meta());
    q4.set("layer_types", wrong);
    RadBuildCtx c4 = build_ctx(0, 1);
    c4.max_tok = 2048; c4.max_seqs = 8; c4.max_spec = 3; c4.max_ctx = 200000;
    RadBuilder z;
    CHECK_EQ(stderr_of_rc([&] { return qwen4exp_fp8::declare(&z, q4.meta(), &c4); }),
             RAD_E_UNSUPPORTED);
}

/* ==================================================================== qwen35_w8a8 */

namespace {

/* The dense fixture as data/recipes/q38-27b-w8a8-df2.recipe states it: w8a8 / fwht128. */
MetaKV q35w8a8_meta() {
    MetaKV k = q35fp8_meta();
    k.m.quant = "w8a8_fwht128";
    k.m.name  = "test-qwen35-27b-w8a8";
    k.set("quantization_config.quant_method", "w8a8").set("quantization_config.fmt", "fwht128");
    return k;
}

/* The recipe's encodings by declared name: the eight trunk linears int8 (`/fwht128` unless
 * `rotated` is false, which the plugin refuses), an int8 lm_head, and the DFlash2 drafter's MXFP4
 * linears with its 2-bit head. The drafter's keys come first because its names contain the
 * trunk's (dflash.blk.0.attn_k.weight), and the head's last because `output.weight` is in
 * `attn_output.weight`. */
void w8a8_encs(RadBuilder* b, bool rotated = true) {
    RadEncoding i8 = rad_enc_affine(RAD_I8, RAD_BF16, 1, 128);
    if (rotated) rad_enc_copy_str(i8.transform, "fwht128");
    RadEncoding u2 = rad_enc_affine(RAD_U2, RAD_F16, 1, 128);
    rad_enc_add_plane(&u2, "zero", RAD_U8, 1, 128);
    b->encs = { { "dflash.draft_head", u2 },
                { "dflash.", rad_enc_affine(RAD_FP4E2M1, RAD_E8M0, 1, 32) } };
    for (const char* t : { "attn_qg.weight", "attn_k.weight", "attn_v.weight",
                           "attn_output.weight", "ssm_inz.weight", "ssm_out.weight",
                           "ffn_gate_up.weight", "ffn_down.weight" })
        b->encs.push_back({ t, i8 });
    b->encs.push_back({ "output.weight", rad_enc_affine(RAD_I8, RAD_BF16, 1, 128) });
}

/* qwen35_w8a8::declare with the recipe's encodings, in lint_qwen35_fp8_family's shape. */
int w8a8_declare(RadBuilder* b, const RadModelMeta* meta, const RadBuildCtx* ctx) {
    w8a8_encs(b);
    return qwen35_w8a8::declare(b, meta, ctx);
}

}  /* namespace */

/* The `_had_quant_i8g` producers, which libr4d serves and libref does not. */
const char* const kW8a8Fused[] = { "rmsnorm_had_quant_i8g", "ar_rmsnorm_had_quant_i8g",
                                   "gated_had_quant_i8g" };

/* Where the `_had_quant_i8g` producers are not served, every trunk linear of a rotated int8
 * container is an i8a8h GEMM right behind its own had_quant_act_i8g, at one rank and two: 2
 * attention layers of four linears, 6 delta-net layers of two and 8 feed-forwards of two. None is
 * left at fp8, which is what a linear that missed the encoding falls back to, and the MLP declares
 * no gated fold (an fp8a8 tile over E4M3 codes). The four trunk activations carry an int8 pair;
 * the delta net's `gdn_in` feeds no int8 linear. */
TEST(qwen35_w8a8_serves_every_trunk_linear_rotated_int8) {
    MetaKV mk = q35w8a8_meta();
    for (int world : { 1, 2 }) {
        RadBuilder b;
        for (const char* op : kW8a8Fused) b.undeclared.insert(op);
        w8a8_encs(&b);
        RadBuildCtx ctx = build_ctx(0, world);
        REQUIRE_EQ(qwen35_w8a8::declare(&b, mk.meta(), &ctx), RAD_OK);
        int rot = 0, paired = 0;
        for (size_t i = 0; i < b.ops.size(); ++i) {
            const RecOp& o = b.ops[i];
            if (o.op != "gemm_nt_q") continue;
            CHECK_EQ(pstr(&o, "dtype"), std::string("i8a8h"));
            if (pstr(&o, "dtype") != "i8a8h") continue;
            ++rot;
            paired += i > 0 && b.ops[i - 1].op == "had_quant_act_i8g";
        }
        CHECK_EQ(rot, 2 * 4 + 6 * 2 + 8 * 2);
        CHECK_EQ(paired, rot);
        CHECK_EQ(count_op(b, "quant_act_i8g"), 0);
        CHECK_EQ(count_op(b, "gemm_nt_q_gated"), 0);
        for (const char* a : { "h.q8", "attn_out.q8", "gdn_o.q8", "ffn.q8" })
            CHECK(find_b(b, a) != nullptr);
        CHECK(find_b(b, "gdn_in.q8") == nullptr);
    }
}

/* Where they are served, the norms and the gated product write had_quant_act_i8g's codes into the
 * int8 pair, and the projections reading them declare no quantiser: had_quant_act_i8g is left only
 * before the attention and delta-net output projections (2 + 6), every norm is the rotated twin (at
 * two ranks with its fused all-reduce form beside it), and every feed-forward's product is
 * gated_had_quant_i8g. No E4M3 norm or gated quantiser is left, and the 36 GEMMs are unchanged. */
TEST(qwen35_w8a8_feeds_its_rotated_codes_from_the_producers_where_libr4d_serves_them) {
    MetaKV mk = q35w8a8_meta();
    for (int world : { 1, 2 }) {
        RadBuilder b;
        w8a8_encs(&b);
        RadBuildCtx ctx = build_ctx(0, world);
        REQUIRE_EQ(qwen35_w8a8::declare(&b, mk.meta(), &ctx), RAD_OK);
        int rot = 0;
        for (const RecOp& o : b.ops) rot += o.op == "gemm_nt_q" && pstr(&o, "dtype") == "i8a8h";
        CHECK_EQ(rot, 2 * 4 + 6 * 2 + 8 * 2);
        CHECK_EQ(count_op(b, "had_quant_act_i8g"), 2 + 6);
        CHECK_EQ(count_op(b, "rmsnorm_had_quant_i8g"), 8 + 8);
        CHECK_EQ(count_op(b, "gated_had_quant_i8g"), 8);
        for (const char* op : { "rmsnorm_quant_fp8", "ar_rmsnorm_quant_fp8", "gated_quant_fp8",
                                "quant_act_i8g" })
            CHECK_EQ(count_op(b, op), 0);
        CHECK_EQ(count_op(b, "ar_rmsnorm_had_quant_i8g") > 0, world == 2);
    }
}

/* What the int8 build refuses: a trunk that is not int8 (no stated encoding, which a block-fp8
 * plugin reads as fp8), a plain int8 trunk (the descriptor names the rotation, and a plain weight
 * served behind the rotated quantiser would be wrong), and a container stated for another format
 * or without the rotation's `fmt`. */
TEST(qwen35_w8a8_refuses_what_is_not_a_rotated_int8_trunk) {
    RadBuildCtx ctx = build_ctx();
    {
        MetaKV mk = q35w8a8_meta();
        RadBuilder b;
        CHECK_EQ(stderr_of_rc([&] { return qwen35_w8a8::declare(&b, mk.meta(), &ctx); }),
                 RAD_E_DTYPE);
    }
    {
        MetaKV mk = q35w8a8_meta();
        RadBuilder b;
        w8a8_encs(&b, false);
        CHECK_EQ(stderr_of_rc([&] { return qwen35_w8a8::declare(&b, mk.meta(), &ctx); }),
                 RAD_E_DTYPE);
    }
    {
        MetaKV mk = q35w8a8_meta();
        mk.set("quantization_config.quant_method", "fp8").set("quantization_config.fmt", "e4m3");
        RadBuilder b;
        w8a8_encs(&b);
        CHECK_EQ(stderr_of_rc([&] { return qwen35_w8a8::declare(&b, mk.meta(), &ctx); }),
                 RAD_E_UNSUPPORTED);
    }
    {
        MetaKV mk = q35w8a8_meta();
        mk.drop("quantization_config.fmt");
        RadBuilder b;
        w8a8_encs(&b);
        CHECK_EQ(stderr_of_rc([&] { return qwen35_w8a8::declare(&b, mk.meta(), &ctx); }),
                 RAD_E_UNSUPPORTED);
    }
}

/* Every engine step lints clean at one and two ranks, with the MTP head's passes and with the
 * DFlash2 drafter's, whose depth the probe fixes at its trained block as qwen35_fp8's does. */
TEST(qwen35_w8a8_steps_lint_clean_with_its_mtp_head_and_a_dflash2_drafter) {
    MetaKV mtp = q35w8a8_meta();
    lint_qwen35_fp8_family(w8a8_declare, qwen35_w8a8::step, mtp.meta(), 3, false, 0);
    MetaKV df = with_dflash2(q35w8a8_meta());
    lint_qwen35_fp8_family(w8a8_declare, qwen35_w8a8::step, df.meta(), 7, true, 8);
    RadArchProbe p{};
    CHECK_OK(qwen35_w8a8::probe(df.meta(), &p));
    CHECK_EQ(p.draft_depth, 7);
}

/* ==================================================================== llama_fp8 and DSpark */

namespace {

/* MiniCPM5-2B's graph cut to four layers at a quarter of its width, with the DSpark drafter
 * MiniCPM5-2B-DSpark ships merged under `draft.`: two layers reading trunk layers 0, 2 and 3, a
 * block of seven, the rank-64 bigram head. */
MetaKV llama_meta(const char* quant = "fp8_e4m3") {
    RadModelMeta m{};
    m.arch_id = "llama"; m.name = "test-minicpm5"; m.quant = quant;
    m.n_layers = 4; m.n_embd = 1024; m.n_head = 8; m.n_head_kv = 2; m.head_dim = 128;
    m.n_ff = 2048; m.n_vocab = 32000; m.n_ctx_train = 32768;
    m.rms_eps = 1e-5f; m.rope_theta = 5e6f; m.rope_scale = 1.0f;
    return MetaKV(m);
}

MetaKV with_dspark(MetaKV k) {
    k.set("draft.architectures", "DSparkDraftModel")
     .set("draft.num_hidden_layers", "2")
     .set("draft.head_dim", "128")
     .set("draft.num_attention_heads", "8")
     .set("draft.num_key_value_heads", "2")
     .set("draft.intermediate_size", "2048")
     .set("draft.rms_norm_eps", "1e-06")
     .set("draft.rope_theta", "1000000")
     .set("draft.vocab_size", "32000")
     .set("draft.target_layer_ids", "0 2 3")
     .set("draft.block_size", "7")
     .set("draft.mask_token_id", "31999")
     .set("draft.markov_rank", "64")
     .set("draft.markov_head_type", "vanilla");
    return k;
}

RadBuildCtx dspark_ctx(int rank, int world) {
    RadBuildCtx c = spec_ctx(rank, world, 7);
    c.max_seqs = 16;     /* room for a query pass past 64 rows */
    return c;
}

/* Moves the declarations of `moved`, in their order, to just before `before`, as a plugin that
 * declared them there would have, and renumbers the recorded issues to match. */
void redeclare_before(RadBuilder& b, const std::vector<std::unique_ptr<RadCtx>>& ctxs,
                      const std::vector<rad_op>& moved, rad_op before) {
    std::set<rad_op> mv(moved.begin(), moved.end());
    std::vector<rad_op> order;
    for (rad_op h = 1; h <= (rad_op)b.ops.size(); ++h) {
        if (mv.count(h)) continue;
        if (h == before) order.insert(order.end(), moved.begin(), moved.end());
        order.push_back(h);
    }
    std::vector<rad_op> to(b.ops.size() + 1, 0);
    std::vector<RecOp> ops;
    for (size_t i = 0; i < order.size(); ++i) {
        to[order[i]] = (rad_op)(i + 1);
        ops.push_back(b.ops[order[i] - 1]);
    }
    b.ops = ops;
    for (const auto& c : ctxs)
        for (RecIssue& is : c->issues) is.op = to[is.op];
}

}  /* namespace */

/* The plain LLaMA graph's engine steps lint clean at one and two ranks, with the DSpark drafter's
 * context and query passes behind a verify. Ten sequences at a block of seven is 70 query rows,
 * past the 64 the fused gate_up GEMM serves, so the unfused feed-forward path runs too. */
TEST(llama_fp8_steps_lint_clean_with_a_dspark_drafter) {
    MetaKV mk = with_dspark(llama_meta());
    for (int world : { 1, 2 })
        for (int rank = 0; rank < world; ++rank) {
            RadBuilder b;
            RadBuildCtx ctx = dspark_ctx(rank, world);
            REQUIRE_EQ(llama_fp8::declare(&b, mk.meta(), &ctx), RAD_OK);
            CHECK_EQ((int)b.drafter.kind, (int)RAD_DRAFT_BLOCK);
            CHECK_EQ(b.drafter.depth, 7);
            CHECK_EQ(b.drafter.block, 7);       /* every row predicts, the anchor included */
            CHECK_EQ(b.drafter.mask_token, 31999);
            const int64_t gated = llama_fp8::g_model[rank].ds.gated_rows;
            for (const auto& k : { kit(b, { 1, 1, 1 }, {}), kit(b, {}, { 40 }),
                                   kit(b, { 8, 1 }, { 5 }) }) {
                const auto v = run_passes(b, llama_fp8::step, { k.get() }, rank, world);
                CHECK_PASSES(b, v);
            }
            for (int64_t n_seq : { 4, 10 }) {
                const auto ver = kit(b, std::vector<int>((size_t)n_seq, 8), {});
                const auto ctxp = context_pass(b, *ver);
                const auto qry = query_pass(b, n_seq, 7, 7);
                const auto v = run_passes(b, llama_fp8::step, { ver.get(), ctxp.get(), qry.get() },
                                          rank, world);
                CHECK_PASSES(b, v);
                /* The feed-forward of the query pass takes the fold up to its band, the pair past
                 * it. */
                const bool fold = gated > 0 && n_seq * 7 <= gated;
                const auto ops = issued_ops(b, *v[2]);
                CHECK_EQ(count_issued(ops, "silu_mul") > 0, !fold);
            }
        }
}

/* Declaration order is liveness, and DSpark's norms are declared where its step issues them. Moved
 * back to where they once were, all ahead of the layers, the residual stream's planned life ends
 * before the first layer's temporaries begin; past the fused GEMM's 64 rows the unfused
 * feed-forward writes those temporaries, and the lint reports the stream being read after them. */
TEST(the_step_lint_catches_dspark_norms_declared_ahead_of_their_layers) {
    MetaKV mk = with_dspark(llama_meta());
    for (int64_t n_seq : { 4, 10 }) {
        RadBuilder b;
        RadBuildCtx ctx = dspark_ctx(0, 1);
        REQUIRE_EQ(llama_fp8::declare(&b, mk.meta(), &ctx), RAD_OK);
        const rad::arch::Dspark& ds = llama_fp8::g_model[0].ds;
        const auto ver = kit(b, std::vector<int>((size_t)n_seq, 8), {});
        const auto ctxp = context_pass(b, *ver);
        const auto qry = query_pass(b, n_seq, 7, 7);
        const auto v = run_passes(b, llama_fp8::step, { ver.get(), ctxp.get(), qry.get() });
        CHECK_PASSES(b, v);

        std::vector<rad_op> norms;
        for (int64_t l = 0; l < ds.g.n_layer; ++l) {
            norms.push_back(ds.op_anorm[l]);
            norms.push_back(ds.op_fnorm[l]);
        }
        norms.push_back(ds.op_outnorm);
        redeclare_before(b, v, norms, ds.op_qat);
        const auto f = lint_step(b, view(v));
        bool stream = false;
        for (const auto& s : f)
            stream |= has(s, "reads 'ds_x'") && (has(s, "'ds_gate_up'") || has(s, "'ds_ffn'"));
        if (n_seq * 7 > ds.gated_rows) CHECK(stream);
        else                           CHECK(!stream);
    }
}

/* A drafter's taps name trunk layers. Out-of-range ids are dropped, not clamped (the drafter
 * belongs to another target), parsing stops at the first non-number, and a drafter naming no
 * layer of this model is refused. So are a trained block other than the serve's depth, a Markov
 * rank the walk cannot apply, and a head type other than the bigram. */
TEST(llama_reads_its_drafters_taps_and_refuses_what_dspark_does_not_implement) {
    {
        MetaKV mk = with_dspark(llama_meta());
        mk.set("draft.target_layer_ids", "[1, 99, 3, -2]");
        RadBuilder b;
        RadBuildCtx ctx = dspark_ctx(0, 1);
        REQUIRE_EQ(llama_fp8::declare(&b, mk.meta(), &ctx), RAD_OK);
        CHECK_EQ(llama_fp8::g_model[0].n_taps, 2);
        CHECK_EQ(llama_fp8::g_model[0].tap_layer[0], 1);
        CHECK_EQ(llama_fp8::g_model[0].tap_layer[1], 3);
        CHECK_EQ(find_b(b, "draft_src")->d.shape[1], 2 * 1024);
    }
    auto refused = [](MetaKV mk, int max_spec = 7) {
        RadBuilder b;
        RadBuildCtx ctx = dspark_ctx(0, 1);
        ctx.max_spec = max_spec;
        return stderr_of_rc([&] { return llama_fp8::declare(&b, mk.meta(), &ctx); });
    };
    CHECK_EQ(refused(with_dspark(llama_meta()).set("draft.target_layer_ids", "7 9")),
             RAD_E_UNSUPPORTED);
    CHECK_EQ(refused(with_dspark(llama_meta()), 6), RAD_E_UNSUPPORTED);
    CHECK_EQ(refused(with_dspark(llama_meta()).set("draft.markov_rank", "48")), RAD_E_UNSUPPORTED);
    CHECK_EQ(refused(with_dspark(llama_meta()).set("draft.markov_head_type", "mlp")),
             RAD_E_UNSUPPORTED);
    CHECK_EQ(refused(with_dspark(llama_meta()).drop("draft.block_size")), RAD_E_FORMAT);
    CHECK_EQ(refused(with_dspark(llama_meta()).drop("draft.mask_token_id")), RAD_E_FORMAT);
    /* Without speculation the drafter is ignored and the trunk declares alone. */
    RadBuilder b;
    MetaKV mk = with_dspark(llama_meta());
    RadBuildCtx ctx = build_ctx();
    REQUIRE_EQ(llama_fp8::declare(&b, mk.meta(), &ctx), RAD_OK);
    CHECK_EQ((int)b.drafter.kind, (int)RAD_DRAFT_NONE);
    CHECK(find_b(b, "draft_src") == nullptr);
    /* The probe reports the trained block, fixed. */
    RadArchProbe p{};
    CHECK_OK(llama_fp8::probe(mk.meta(), &p));
    CHECK_EQ(p.draft_depth, 7);
    CHECK_EQ(p.draft_depth_fixed, 1);
}

/* One graph, two descriptors: llama_dense_fp8 serves the dense checkpoint a recipe quantises
 * (descriptor ""), llama_fp8 the fp8 one, each refusing the other's, and both declare the same
 * ops and weights. */
TEST(llama_dense_fp8_declares_the_fp8_graph_for_the_dense_descriptor) {
    MetaKV fp8 = with_dspark(llama_meta());
    MetaKV dense = with_dspark(llama_meta(""));
    RadBuildCtx ctx = dspark_ctx(0, 1);
    RadBuilder a, d;
    REQUIRE_EQ(llama_fp8::declare(&a, fp8.meta(), &ctx), RAD_OK);
    REQUIRE_EQ(llama_dense_fp8::declare(&d, dense.meta(), &ctx), RAD_OK);
    CHECK_EQ(join(op_names(a)), join(op_names(d)));
    REQUIRE_EQ(a.weights.size(), d.weights.size());
    for (size_t i = 0; i < a.weights.size(); ++i) {
        CHECK_EQ(a.weights[i].name, d.weights[i].name);
        CHECK_EQ(a.weights[i].d.dtype, d.weights[i].d.dtype);
    }
    RadBuilder x, y;
    CHECK_EQ(stderr_of_rc([&] { return llama_fp8::declare(&x, dense.meta(), &ctx); }),
             RAD_E_UNSUPPORTED);
    CHECK_EQ(stderr_of_rc([&] { return llama_dense_fp8::declare(&y, fp8.meta(), &ctx); }),
             RAD_E_UNSUPPORTED);
    /* Its steps are llama_fp8's. A plugin keeps one model per rank, so declare again after the
     * refusal above replaced it. */
    d = RadBuilder{};
    REQUIRE_EQ(llama_dense_fp8::declare(&d, dense.meta(), &ctx), RAD_OK);
    const auto ver = kit(d, std::vector<int>(4, 8), {});
    const auto ctxp = context_pass(d, *ver);
    const auto qry = query_pass(d, 4, 7, 7);
    const auto v = run_passes(d, llama_dense_fp8::step, { ver.get(), ctxp.get(), qry.get() });
    CHECK_PASSES(d, v);
}

/* ==================================================================== qwen4exp_fp8 stepped */

namespace {

RadBuildCtx q4_ctx(int rank, int world, int64_t max_seqs = 8) {
    RadBuildCtx c = build_ctx(rank, world);
    c.max_tok = 2048; c.max_seqs = max_seqs; c.max_spec = 3; c.max_ctx = 200000;
    return c;
}

/* The 2-bit draft head's encoding (u2 codes, an f16 scale and a u8 zero per 128), as the recipe
 * writes `mtp.draft_head.weight`. */
RadEncoding u2_head() {
    RadEncoding u2 = rad_enc_affine(RAD_U2, RAD_F16, 1, 128);
    rad_enc_add_plane(&u2, "zero", RAD_U8, 1, 128);
    return u2;
}

/* The Flash-Next fixture with 32 experts instead of 512: the step and its lint do not depend on
 * the expert count, and declaring 512 experts on every rank is most of a test's time. */
MetaKV q4_small() {
    MetaKV k(qwen4exp_meta());
    k.set("num_experts", "32");
    return k;
}

/* The issues of op `name` in a pass. */
std::vector<const RecIssue*> issues_of(const RadBuilder& b, const RadCtx& c, const char* name) {
    std::vector<const RecIssue*> v;
    for (const auto& i : c.issues)
        if (const RecOp* o = b.op_of(i.op); o && o->op == name) v.push_back(&i);
    return v;
}

}  /* namespace */

/* Flash-Next's engine steps lint clean at one, two and four ranks: decode, a prefill chunk after
 * a long history (the sparse attention), a mixed step, and a verify followed by the MTP head's
 * history pass and rounds, the 2-bit draft head and its vocab-parallel gather included. */
TEST(flashnext_steps_lint_clean_at_one_two_and_four_ranks) {
    MetaKV mk = q4_small();
    const RadModelMeta& meta = *mk.meta();
    for (int world : { 1, 2, 4 })
        for (int rank = 0; rank < world; ++rank) {
            RadBuilder b;
            b.encs = { { "draft_head", u2_head() } };
            RadBuildCtx ctx = q4_ctx(rank, world);
            REQUIRE_EQ(qwen4exp_fp8::declare(&b, &meta, &ctx), RAD_OK);
            REQUIRE_EQ((int)b.drafter.kind, (int)RAD_DRAFT_SERIAL);
            for (const auto& k : { kit(b, { 1, 1, 1, 1 }, {}), kit(b, {}, { 64, 64 }, 9000),
                                   kit(b, { 1, 4 }, { 6 }, 300) }) {
                const auto v = run_passes(b, qwen4exp_fp8::step, { k.get() }, rank, world);
                CHECK_PASSES(b, v);
            }
            const auto ver = kit(b, std::vector<int>(4, 4), {});
            const auto his = history_pass(b, *ver);
            const auto r2 = round_pass(b, 4, 2), r3 = round_pass(b, 4, 3);
            const auto v = run_passes(b, qwen4exp_fp8::step,
                                      { ver.get(), his.get(), r2.get(), r3.get() }, rank, world);
            CHECK_PASSES(b, v);
        }
}

/* Sparse attention issues one row per query, chunked by the split-KV scratch it declared, never
 * by `max_seqs`: with four sequences allowed, a 1024-token chunk after a long history is one
 * launch of the gated form, or four of 256 rows where only the plain form resolves. Rows of four
 * would be 256 launches a layer. */
TEST(flashnext_sparse_attention_chunks_by_its_scratch_not_by_max_seqs) {
    MetaKV mk = q4_small();
    const RadModelMeta& meta = *mk.meta();
    for (int gated = 0; gated < 2; ++gated) {
        RadBuilder b;
        if (!gated) b.refuse.insert("attn_paged_gate_quant");
        RadBuildCtx ctx = q4_ctx(0, 1, 4);
        REQUIRE_EQ(qwen4exp_fp8::declare(&b, &meta, &ctx), RAD_OK);
        const auto k = kit(b, {}, { 1024 }, 9000);
        const auto v = run_passes(b, qwen4exp_fp8::step, { k.get() });
        CHECK_PASSES(b, v);
        const auto at = issues_of(b, *v[0], gated ? "attn_paged_gate_quant" : "attn_paged");
        /* Two full-attention layers. */
        REQUIRE_EQ(at.size(), (size_t)(gated ? 2 : 2 * 4));
        for (const RecIssue* i : at) {
            CHECK_EQ(i->n, 1);                               /* q_len 1: a row is a query */
            CHECK_EQ(i->opd[0].rows, gated ? 1024 : 256);    /* queries in this launch */
            CHECK_EQ(i->opd[2].rows, i->opd[0].rows);        /* one selection row a query */
        }
    }
}

/* The indexer's unfinished block is state, not scratch: each QSA layer stores it in the linear
 * `kv_qsa_tail` group at its own layer, so two layers never share one tail and a slot survives
 * across steps. */
TEST(flashnext_qsa_tail_is_per_layer_state) {
    MetaKV mk = q4_small();
    const RadModelMeta& meta = *mk.meta();
    RadBuilder b;
    RadBuildCtx ctx = q4_ctx(0, 1);
    REQUIRE_EQ(qwen4exp_fp8::declare(&b, &meta, &ctx), RAD_OK);
    rad_kvgroup tail = 0;
    for (size_t g = 0; g < b.kv_groups.size(); ++g)
        if (b.kv_groups[g].name == "kv_qsa_tail") tail = (rad_kvgroup)(g + 1);
    REQUIRE(tail != 0);
    CHECK_EQ((int)b.kv_groups[tail - 1].d.kind, (int)RAD_KV_LINEAR);
    const auto k = kit(b, { 1, 1 }, {});
    const auto v = run_passes(b, qwen4exp_fp8::step, { k.get() });
    CHECK_PASSES(b, v);
    std::set<int64_t> layers;
    for (const RecIssue* i : issues_of(b, *v[0], "qsa_tail_store"))
        for (const RadOperand& o : i->opd)
            if (o.kind == RAD_OPK_KV) {
                CHECK_EQ(o.handle, tail);
                layers.insert(o.offset);
            }
    CHECK(layers == (std::set<int64_t>{ 3, 7 }));
}

/* The rope table has a row for every position a step can name: a verify and the draft rounds run
 * up to max_spec past max_ctx, so a table ending at max_ctx would rotate them by unwritten rows.
 * Every draft pass fills its own rows before anything reads them, since a round's positions are
 * ones the trunk step never wrote. */
TEST(flashnext_rope_table_reaches_past_the_context_and_draft_passes_fill_it_first) {
    MetaKV mk = q4_small();
    const RadModelMeta& meta = *mk.meta();
    RadBuilder b;
    RadBuildCtx ctx = q4_ctx(0, 1);
    REQUIRE_EQ(qwen4exp_fp8::declare(&b, &meta, &ctx), RAD_OK);
    const RecBuf* cs = find_b(b, "rope_cs");
    REQUIRE(cs != nullptr);
    CHECK(cs->d.shape[0] >= ctx.max_ctx + ctx.max_spec);
    const rad_buf h = (rad_buf)buf_named(b, "rope_cs");
    const auto ver = kit(b, std::vector<int>(4, 4), {});
    const auto his = history_pass(b, *ver);
    const auto r2 = round_pass(b, 4, 2), r3 = round_pass(b, 4, 3);
    const auto v = run_passes(b, qwen4exp_fp8::step, { ver.get(), his.get(), r2.get(), r3.get() });
    CHECK_PASSES(b, v);
    for (size_t p = 1; p < v.size(); ++p) {
        int first_write = -1, first_read = -1;
        for (size_t i = 0; i < v[p]->issues.size(); ++i) {
            const RecIssue& is = v[p]->issues[i];
            const RecOp* o = b.op_of(is.op);
            bool names = false;
            for (const RadOperand& od : is.opd) names |= od.kind == RAD_OPK_BUF && od.handle == h;
            if (!names) continue;
            if (in(o->writes, h) && first_write < 0) first_write = (int)i;
            if (in(o->reads, h) && first_read < 0) first_read = (int)i;
        }
        CHECK(first_write >= 0);
        CHECK(first_read < 0 || first_write < first_read);
        CHECK_EQ(b.op_of(v[p]->issues[0].op)->op, std::string("rope_table"));
    }
}

/* A draft pass refills the rope table before the head's first op, while the carry `mtp_hin` holds
 * the rows the previous pass left, so the table's planned life must reach the carry's or the
 * planner may give the two one set of bytes. Without qk_norm_rope_gate (the fused attention
 * prologue and the indexer's) no head op reads the table, so only the head's own declaration of
 * the refill carries its life there. At one rank and two, with the fused prologue and without, the
 * table is live at the carry's first op and a verify, the history pass and two rounds lint clean. */
TEST(flashnext_draft_passes_refill_the_rope_table_where_the_carry_is_live) {
    MetaKV mk = q4_small();
    const RadModelMeta& meta = *mk.meta();
    for (int fused = 0; fused < 2; ++fused)
        for (int world : { 1, 2 }) {
            RadBuilder b;
            if (!fused) b.undeclared.insert("qk_norm_rope_gate");
            RadBuildCtx ctx = q4_ctx(0, world);
            REQUIRE_EQ(qwen4exp_fp8::declare(&b, &meta, &ctx), RAD_OK);
            CHECK_EQ(count_op(b, "qk_norm_rope_gate") > 0, fused == 1);
            const std::vector<Live> L = liveness(b);
            const rad_buf cs = (rad_buf)buf_named(b, "rope_cs");
            const rad_buf hin = (rad_buf)buf_named(b, "mtp_hin");
            REQUIRE(cs && hin);
            CHECK(L[cs].planned && L[hin].planned);
            CHECK(L[cs].lo <= L[hin].lo && L[hin].lo <= L[cs].hi);
            const auto ver = kit(b, std::vector<int>(4, 4), {});
            const auto his = history_pass(b, *ver);
            const auto r2 = round_pass(b, 4, 2), r3 = round_pass(b, 4, 3);
            const auto v = run_passes(b, qwen4exp_fp8::step,
                                      { ver.get(), his.get(), r2.get(), r3.get() }, 0, world);
            CHECK_PASSES(b, v);
        }
}

/* The vocab-parallel draft head gathers each rank's top-R as (id, value) pairs: a row is 2R
 * floats, whole 16-byte words, and the gather moves every draft row's. A row of one pair (8 bytes)
 * is what the two-rank gather cannot take. */
TEST(flashnext_draft_head_gathers_whole_candidate_rows_at_two_ranks) {
    MetaKV mk = q4_small();
    const RadModelMeta& meta = *mk.meta();
    RadBuilder b;
    b.encs = { { "draft_head", u2_head() } };
    RadBuildCtx ctx = q4_ctx(0, 2);
    REQUIRE_EQ(qwen4exp_fp8::declare(&b, &meta, &ctx), RAD_OK);
    const RecOp* topk = nullptr;
    for (const auto& o : b.ops)
        if (o.op == "row_topk" && param(&o, "vocab_off")) topk = &o;
    const RecOp* ag = find_op(b, "all_gather");
    REQUIRE(topk != nullptr && ag != nullptr);
    const long long R = pint(topk, "R");
    CHECK(R >= 2);
    CHECK_EQ(pint(ag, "row"), 2 * R);
    CHECK_EQ(pint(ag, "row") * 4 % 16, 0);
    CHECK_EQ(pint(ag, "world_size"), 2);
    const long long rows = ctx.max_seqs * (1 + ctx.max_spec);
    CHECK_EQ(pint(ag, "numel"), rows * 2 * R);
    const RecBuf* pair = find_b(b, "mtp_dpair");
    const RecBuf* gath = find_b(b, "mtp_dgath");
    REQUIRE(pair != nullptr && gath != nullptr);
    CHECK_EQ(pair->d.shape[1] * pair->d.shape[2], 2 * R);
    CHECK_EQ(gath->d.shape[1] * gath->d.shape[2], 2 * 2 * R);
}

/* ==================================================================== embgemma2 */

namespace {

/* EmbeddingGemma 2 cut to six layers (five sliding, one whole-input, its 5:1), with its vision
 * tower and audio encoder cut to two layers each. */
MetaKV eg2_meta() {
    RadModelMeta m{};
    m.arch_id = "embeddinggemma2"; m.name = "test-embgemma2"; m.quant = "";
    m.n_layers = 6; m.n_embd = 512; m.n_head = 4; m.n_head_kv = 2; m.head_dim = 256;
    m.n_ff = 2048; m.n_vocab = 262144; m.n_ctx_train = 2048;
    m.rms_eps = 1e-6f; m.rope_theta = 1e4f; m.rope_scale = 1.0f;
    MetaKV k(m);
    k.set("layer_types", "sliding_attention sliding_attention sliding_attention sliding_attention "
                         "sliding_attention full_attention")
     .set("rope_parameters.sliding_attention.rope_theta", "10000")
     .set("rope_parameters.full_attention.rope_theta", "1000000")
     .set("sliding_window", "512")
     .set("per_layer_config.05.head_dim", "512")
     .set("per_layer_config.05.num_key_value_heads", "1")
     .set("hidden_size_per_layer_input", "512")
     .set("embedding_dim", "768")
     .set("vision_config.model_type", "gemma4_vision")
     .set("vision_config.hidden_size", "768")
     .set("vision_config.num_attention_heads", "12")
     .set("vision_config.head_dim", "64")
     .set("vision_config.intermediate_size", "3072")
     .set("vision_config.pooling_kernel_size", "3")
     .set("vision_config.position_embedding_size", "10240")
     .set("vision_config.num_hidden_layers", "2")
     .set("vision_config.patch_size", "16")
     .set("vision_config.rope_parameters.rope_theta", "100")
     .set("vision_config.rope_parameters.rope_type", "default")
     .set("vision_config.hidden_activation", "gelu_pytorch_tanh")
     .set("audio_config.model_type", "gemma4_audio")
     .set("audio_config.hidden_size", "1024")
     .set("audio_config.num_attention_heads", "8")
     .set("audio_config.output_proj_dims", "1536")
     .set("audio_config.attention_chunk_size", "12")
     .set("audio_config.attention_context_left", "13")
     .set("audio_config.attention_context_right", "0")
     .set("audio_config.attention_logit_cap", "50")
     .set("audio_config.num_hidden_layers", "2")
     .set("audio_config.conv_kernel_size", "5")
     .set("audio_config.residual_weight", "0.5")
     .set("audio_config.hidden_act", "silu")
     .set("audio_config.subsampling_conv_channels", "128 32")
     .set("preprocessor.feature_size", "128");
    return k;
}

RadBuildCtx eg2_ctx() {
    RadBuildCtx c = build_ctx();
    c.max_enc_patches = 9 * 64;
    c.max_audio_frames = 400;
    return c;
}

/* An encoder pass of `n` patches in one segment, every coordinate plane filled. */
struct EncKit {
    std::vector<uint16_t> pix;
    std::vector<int32_t>  coord, cu;
    RadBatch b{};
};
std::unique_ptr<EncKit> enc_pass(int32_t mod, int64_t n, int64_t patch_dim) {
    auto k = std::make_unique<EncKit>();
    k->pix.assign((size_t)(n * patch_dim), 0);
    k->coord.assign((size_t)(4 * n), 1);
    k->cu = { 0, (int32_t)n };
    k->b.enc = mod;
    k->b.enc_n_patch = n;
    k->b.enc_pixels = k->pix.data();
    k->b.enc_coord = k->coord.data();
    k->b.enc_n_seg = 1;
    k->b.enc_cu = k->cu.data();
    k->b.enc_max_seg = (int32_t)n;
    return k;
}

}  /* namespace */

/* An embedding model: a pooled f32 output and no logits, attention bidirectional on every layer
 * (a window on the sliding ones, one more than the reference's radius since attn_dense's bound is
 * strict), each layer's residual gated by its slice of the per-layer inputs and scaled by its
 * learned scalar, and both media encoders declared. */
TEST(embgemma2_declares_a_pooled_encoder_with_both_media_towers) {
    MetaKV mk = eg2_meta();
    RadBuilder b;
    RadBuildCtx ctx = eg2_ctx();
    REQUIRE_EQ(embgemma2::declare(&b, mk.meta(), &ctx), RAD_OK);
    CHECK_EQ(b.logits, (rad_buf)0);
    REQUIRE(b.pooled != 0);
    const RadBufDecl& p = b.bufs[b.pooled - 1].d;
    CHECK_EQ(p.dtype, (uint32_t)RAD_F32);
    CHECK_EQ(p.shape[0], ctx.max_seqs);
    CHECK_EQ(p.shape[1], 768);
    CHECK_EQ(count_op(b, "logits_gemm"), 0);
    std::vector<long long> win;
    for (const auto& o : b.ops)
        if (o.op == "attn_dense" && pint(&o, "causal") == 0 && pint(&o, "head_dim") >= 256) {
            win.push_back(pint(&o, "window"));
            CHECK_NEAR(pflt(&o, "scale"), 1.0, 0.0);
        }
    CHECK(win == (std::vector<long long>{ 513, 513, 513, 513, 513, 0 }));
    CHECK_EQ(count_op(b, "mul_weight"), 6);       /* the layer scalars */
    CHECK_EQ(count_op(b, "seg_mean"), 1);
    CHECK_EQ(b.encoder_name, std::string("vision"));
    CHECK_EQ(b.audio_encoder_name, std::string("audio"));
    CHECK_EQ(b.audio_encoder.max_patches, 400);
    CHECK_EQ(b.audio_encoder.merge, 4);
    CHECK_EQ(count_op(b, "scatter_rows"), 1);
    for (const auto& o : b.ops)
        if (o.resolved) CHECK(o.uses_declared);
    for (const auto& w : b.weights) CHECK(find_map(b, w.name) != nullptr);
}

/* What it refuses: a second rank, a container without layer_types or with a kind it does not
 * know or a count that is not the model's, and an audio tower that is not the conformer it
 * implements. */
TEST(embgemma2_refuses_what_it_does_not_serve) {
    RadBuildCtx ctx = eg2_ctx();
    auto rc = [&](MetaKV mk, const RadBuildCtx& c) {
        RadBuilder b;
        return stderr_of_rc([&] { return embgemma2::declare(&b, mk.meta(), &c); });
    };
    CHECK_EQ(rc(eg2_meta(), build_ctx(0, 2)), RAD_E_UNSUPPORTED);
    CHECK_EQ(rc(eg2_meta().drop("layer_types"), ctx), RAD_E_INVAL);
    CHECK_EQ(rc(eg2_meta().set("layer_types", "sliding_attention sliding_attention "
                                              "sliding_attention sliding_attention "
                                              "sliding_attention chunked_attention"), ctx),
             RAD_E_UNSUPPORTED);
    CHECK_EQ(rc(eg2_meta().set("layer_types", "sliding_attention full_attention"), ctx),
             RAD_E_INVAL);
    CHECK_EQ(rc(eg2_meta().set("audio_config.residual_weight", "1.0"), ctx), RAD_E_UNSUPPORTED);
    CHECK_EQ(rc(eg2_meta().drop("embedding_dim"), ctx), RAD_E_INVAL);
}

/* The text pass over three whole inputs (one with an image's rows scattered in), the vision pass
 * and the audio pass each lint clean, and every GEMM reads whole rows: the audio encoder's clamp
 * copy feeding a GEMM is a buffer of its own width, not a slice of a wider one. */
TEST(embgemma2_text_and_media_passes_lint_clean_and_feed_gemms_whole_rows) {
    MetaKV mk = eg2_meta();
    RadBuilder b;
    RadBuildCtx ctx = eg2_ctx();
    REQUIRE_EQ(embgemma2::declare(&b, mk.meta(), &ctx), RAD_OK);
    const auto text = kit(b, {}, { 12, 7, 30 });
    static const int32_t rows[] = { 2, 3, 4 };
    static uint16_t embd[3 * 512];
    text->b.n_mm_rows = 3;
    text->b.mm_rows = rows;
    text->b.mm_embd = embd;
    const auto vis = enc_pass(RAD_MM_IMAGE, 9 * 16, 3 * 16 * 16);
    const auto aud = enc_pass(RAD_MM_AUDIO, 400, 128);
    std::vector<const RadBatch*> passes = { &text->b, &vis->b, &aud->b };
    for (const RadBatch* pb : passes) {
        RadCtx c{ .b = &b, .batch = pb };
        embgemma2::step(&c, pb);
        CHECK(c.step_fail.empty());
        CHECK(!c.issues.empty());
        CHECK_LINT(b, &c);
        if (pb == &text->b) {
            /* the pooled rows, one per input */
            const RecIssue& last = c.issues.back();
            CHECK_EQ(last.n, 3);
            CHECK_EQ(last.opd.back().handle, b.pooled);
        }
    }
    /* A pass past the declared audio frames, or not whole output rows, is refused. */
    for (int64_t n : { 404, 398 }) {
        const auto bad = enc_pass(RAD_MM_AUDIO, n, 128);
        RadCtx c{ .b = &b, .batch = &bad->b };
        embgemma2::step(&c, &bad->b);
        CHECK(!c.step_fail.empty());
        CHECK(c.issues.empty());
    }
}

/* ==================================================================== qwenimage21 */

namespace {

/* Qwen-Image 2.1 with every network cut: a two-layer conditioning language model of 512, a
 * two-block DiT of four 128-wide heads (rotary axes 16, 56, 56), and the residual VAE's own
 * shape. */
MetaKV qimg_meta() {
    RadModelMeta m{};
    m.arch_id = "qwenimage21"; m.name = "test-qwen-image"; m.quant = "";
    m.n_layers = 2; m.n_embd = 512; m.n_head = 4; m.n_head_kv = 2; m.head_dim = 128;
    m.n_ff = 1024; m.n_vocab = 151936; m.n_ctx_train = 32768;
    m.rms_eps = 1e-6f; m.rope_theta = 5e6f; m.rope_scale = 1.0f;
    MetaKV k(m);
    std::string mean, sd;
    for (int i = 0; i < 64; ++i) {
        mean += (i ? " " : "") + std::to_string(0.01 * i);
        sd += (i ? " " : "") + std::to_string(1.0 + 0.01 * i);
    }
    k.set("pipeline._class_name", "QwenImage21Pipeline")
     .set("transformer.patch_size", "1")
     .set("transformer.causal_condition", "1")
     .set("transformer.num_layers", "2")
     .set("transformer.attention_head_dim", "128")
     .set("transformer.num_attention_heads", "4")
     .set("transformer.mlp_ratio", "4")
     .set("transformer.in_channels", "64")
     .set("transformer.context_in_dim", "512")
     .set("transformer.axes_dims_rope", "16 56 56")
     .set("vae.is_residual", "1")
     .set("vae.z_dim", "64")
     .set("vae.latents_mean", mean.c_str())
     .set("vae.latents_std", sd.c_str())
     .set("vae.dim_mult", "1 2 4 4 4")
     .set("vae.temperal_downsample", "0 1 1 0");
    return k;
}

/* A generator pass: COND over a text prompt of `n` tokens, DENOISE over `entries` latents of `n`
 * rows each, or DECODE of a 64 x 64 picture. */
struct GenKit {
    std::vector<int32_t> tok, rope, cu, src, seg, seg_h, seg_bidi, seg_cu, seg_start, slot;
    std::vector<int32_t> pair, lane, flags, table, used;
    std::vector<uint32_t> seed;
    std::vector<float> sigma, sigma_next, cfg;
    RadKVGroupBatch kv{};
    RadGenBatch g{};
    RadBatch b{};
};

std::unique_ptr<GenKit> gen_pass(const RadBuilder& bld, int pass, int64_t n, int64_t entries = 1) {
    auto k = std::make_unique<GenKit>();
    const int64_t S = pass == RAD_GEN_PASS_DENOISE ? entries : 1, N = n * S;
    for (int64_t i = 0; i < N; ++i) {
        k->tok.push_back((int32_t)(3 + i % 50));
        k->src.push_back((int32_t)i);
        k->slot.push_back((int32_t)i);
    }
    k->rope.assign((size_t)(3 * N), 0);
    for (int64_t s = 0; s <= S; ++s) k->cu.push_back((int32_t)(s * n));
    k->seg = { 0, (int32_t)n };
    k->seg_h = k->seg;
    k->seg_bidi = { 0 };
    k->seg_cu = { 0, (int32_t)n };
    k->seg_start = { 0 };
    k->sigma.assign((size_t)S, pass == RAD_GEN_PASS_COND ? 0.0f : 0.5f);
    k->sigma_next.assign((size_t)S, 0.25f);
    k->cfg.assign((size_t)S, 1.0f);
    k->pair.assign((size_t)S, -1);
    for (int64_t s = 0; s < S; ++s) k->lane.push_back((int32_t)s);
    k->flags.assign((size_t)S, RAD_GEN_F_FIRST);
    k->seed.assign((size_t)(2 * S), 7u);
    k->table.assign((size_t)(S * 16), 0);
    k->used.assign((size_t)S, (int32_t)n);
    k->kv.group = 1;
    k->kv.slot_mapping = k->slot.data();
    k->kv.block_table = k->table.data();
    k->kv.block_table_pitch = 16;
    k->kv.seqused = k->used.data();
    k->kv.block_size = bld.kv_block;
    k->kv.max_blocks = 16;
    RadGenBatch& g = k->g;
    g.rows = n;
    g.src = k->src.data();
    g.rope = k->rope.data();
    g.n_seg = 1;
    g.seg = k->seg.data();
    g.seg_h = k->seg_h.data();
    g.seg_bidi_h = k->seg_bidi.data();
    g.seg_cu = k->seg_cu.data();
    g.seg_start = k->seg_start.data();
    g.slot_mapping = k->slot.data();
    g.sigma = k->sigma.data();
    g.sigma_next = k->sigma_next.data();
    g.cfg = k->cfg.data();
    g.pair = k->pair.data();
    g.lane = k->lane.data();
    g.lane_dh = k->lane.data();
    g.n_lane = S;
    g.flags = k->flags.data();
    g.seed = k->seed.data();
    g.dec_lane = 0;
    g.dec_height = g.dec_width = 64;
    RadBatch& b = k->b;
    b.gen = pass;
    b.gen_b = &k->g;
    b.n_tok = pass == RAD_GEN_PASS_DECODE ? 16 : N;
    b.n_seq = S;
    b.token_ids = k->tok.data();
    b.rope_pos = k->rope.data();
    b.cu_seqlens = k->cu.data();
    b.n_kv_groups = 1;
    b.kv = &k->kv;
    return k;
}

RadBuildCtx qimg_ctx(int rank, int world, int lossy = 0) {
    RadBuildCtx c = build_ctx(rank, world);
    c.max_tok = 512;
    c.max_seqs = 2;
    c.tp_wire_lossy = lossy;
    c.tp_wire_min_bytes = 64 * 1024;
    return c;
}

}  /* namespace */

/* An image generator, not a language model: no logits, the generator declared with the model's
 * prompt and sizes. One modulation projection is shared by every block; the DiT rotates three
 * sections of adjacent pairs from axes 16, 56, 56; the prefix is cached in a full KV group, read
 * causally by text and both ways by a picture. A Turbo container's fixed schedule names it. */
TEST(qwenimage21_declares_a_generator_with_one_shared_modulation) {
    MetaKV mk = qimg_meta();
    RadBuilder b;
    RadBuildCtx ctx = qimg_ctx(0, 1);
    REQUIRE_EQ(qwenimage21::declare(&b, mk.meta(), &ctx), RAD_OK);
    CHECK_EQ(b.logits, (rad_buf)0);
    CHECK_EQ(b.generator_name, std::string("qwen-image-2.1"));
    CHECK_EQ(b.generator.layout, (int32_t)RAD_GEN_LAYOUT_JOINT_GRID);
    CHECK_EQ(b.generator.latent_channels, 64);
    CHECK_EQ(b.generator.n_sigmas, 0);
    CHECK(b.generator.image_out != 0);
    int mods = 0;
    for (const auto& w : b.weights) mods += has(w.name, "dit.mod.weight");
    CHECK_EQ(mods, 1);
    const rad_buf mod = (rad_buf)buf_named(b, "dit.mod");
    int readers = 0;
    for (const auto& o : b.ops)
        if ((o.op == "ln_mod" || o.op == "gate_add" || o.op == "ln_mod_quant_fp8") &&
            in(o.reads, mod))
            ++readers;
    CHECK(readers >= 2 * 2);          /* both blocks' norms read the one modulation */
    bool dit_rope = false;
    for (const auto& o : b.ops)
        if (pstr(&o, "sections") == "8 28 28") dit_rope = true;
    CHECK(dit_rope);
    /* Per block: the causal and the bidirectional form, and (fp8 linears) the denoising pass's
     * bidirectional form with int8 scores. */
    int causal = 0, bidi = 0, bidi8 = 0;
    for (const auto& o : b.ops)
        if (o.op == "attn_prefix_self")
            (pint(&o, "causal") ? causal : pstr(&o, "qk") == "i8" ? bidi8 : bidi)++;
    CHECK_EQ(causal, 2);
    CHECK_EQ(bidi, 2);
    CHECK_EQ(bidi8, 2);
    REQUIRE_EQ(b.kv_groups.size(), (size_t)1);
    CHECK_EQ((int)b.kv_groups[0].d.kind, (int)RAD_KV_FULL);

    MetaKV turbo = qimg_meta();
    turbo.set("pipeline.sample_sigmas", "1.0 0.9 0.75 0.6 0.45 0.3 0.15 0.05");
    RadBuilder bt;
    REQUIRE_EQ(qwenimage21::declare(&bt, turbo.meta(), &ctx), RAD_OK);
    CHECK_EQ(bt.generator_name, std::string("qwen-image-2.1-turbo"));
    CHECK_EQ(bt.generator.n_sigmas, 8);
    CHECK_EQ(bt.generator.default_steps, 8);
}

/* Every collective follows --tp-wire: exact unless the lossy wire was accepted, and the floor
 * under which a message stays exact is the deployment's. */
/* The DiT's adapter names: one root before any GEMM, and every alias lands on a checkpoint tensor
 * the plugin maps, with the gate half of ComfyUI's `gate_up` on `gate_layer`. */
TEST(qwenimage21_names_its_adapter_keys_onto_mapped_tensors) {
    MetaKV mk = qimg_meta();
    RadBuilder b;
    const RadBuildCtx ctx = qimg_ctx(0, 1);
    REQUIRE_EQ(qwenimage21::declare(&b, mk.meta(), &ctx), RAD_OK);
    REQUIRE_EQ(b.lora_roots.size(), (size_t)1);
    CHECK_EQ(b.lora_roots[0].first, std::string("transformer."));
    CHECK(b.lora_roots[0].second.find("lora_unet_") != std::string::npos);
    std::set<std::string> srcs;
    for (const RecMap& m : b.maps) for (const std::string& x : m.src) srcs.insert(x);
    REQUIRE_EQ(b.lora_aliases.size(), (size_t)2);
    for (const auto& a : b.lora_aliases) {
        std::string to = a.native + a.to;
        const size_t h = to.find('#');
        if (h != std::string::npos) to.replace(h, 1, "0");
        CHECK(srcs.count(to + ".weight"));
    }
    CHECK(b.lora_aliases[0].to.find("gate_layer") != std::string::npos);
    CHECK_EQ(b.lora_aliases[0].from_lo, (int64_t)0);
    CHECK_EQ(b.lora_aliases[1].from_lo, b.lora_aliases[0].from_hi);
}

TEST(qwenimage21_collectives_follow_the_tp_wire) {
    MetaKV mk = qimg_meta();
    for (int lossy : { 0, 1 }) {
        RadBuilder b;
        RadBuildCtx ctx = qimg_ctx(0, 2, lossy);
        REQUIRE_EQ(qwenimage21::declare(&b, mk.meta(), &ctx), RAD_OK);
        int n = 0;
        for (const auto& o : b.ops) {
            if (o.op != "all_reduce") continue;
            ++n;
            CHECK_EQ(pint(&o, "exact"), lossy ? 0 : 1);
            CHECK_EQ(pint(&o, "min_bytes"), 64 * 1024);
        }
        CHECK(n >= 2 * 2 + 2 * 2);      /* two a layer in each network */
    }
}

/* The three passes the core drives (a text prompt's conditioning, denoising one image or two at
 * once, decoding one) lint clean at one and two ranks, the split all-reduces included. */
TEST(qwenimage21_generator_passes_lint_clean) {
    MetaKV mk = qimg_meta();
    for (int world : { 1, 2 })
        for (int rank = 0; rank < world; ++rank) {
            RadBuilder b;
            RadBuildCtx ctx = qimg_ctx(rank, world);
            REQUIRE_EQ(qwenimage21::declare(&b, mk.meta(), &ctx), RAD_OK);
            for (const auto& k : { gen_pass(b, RAD_GEN_PASS_COND, 12),
                                   gen_pass(b, RAD_GEN_PASS_DENOISE, 16),
                                   gen_pass(b, RAD_GEN_PASS_DENOISE, 16, 2),
                                   gen_pass(b, RAD_GEN_PASS_DECODE, 0) }) {
                RadCtx c{ .b = &b, .batch = &k->b, .rank = rank, .world = world };
                qwenimage21::step(&c, &k->b);
                CHECK(c.step_fail.empty());
                CHECK(!c.issues.empty());
                CHECK_EQ(c.lane, 0);
                CHECK_LINT(b, &c);
            }
            /* A token step is not a generator pass. */
            const auto tok = kit(b, { 1 }, {});
            RadCtx c{ .b = &b, .batch = &tok->b, .rank = rank, .world = world };
            qwenimage21::step(&c, &tok->b);
            CHECK(!c.step_fail.empty());
        }
}

/* ==================================================================== anima */

namespace {

/* Anima's text encoder (Qwen3-0.6B-Base) as the container's config states it; the DiT, adapter
 * and VAE are the plugin's own constants. */
MetaKV anima_meta(const char* variant) {
    RadModelMeta m{};
    m.arch_id = "anima"; m.name = "test-anima"; m.quant = "";
    m.n_layers = 28; m.n_embd = 1024; m.n_head = 16; m.n_head_kv = 8; m.head_dim = 128;
    m.n_ff = 3072; m.n_vocab = 151936; m.n_ctx_train = 32768;
    m.rms_eps = 1e-6f; m.rope_theta = 1e6f; m.rope_scale = 1.0f;
    MetaKV k(m);
    k.set("num_hidden_layers", "28").set("hidden_size", "1024").set("anima.variant", variant);
    return k;
}

RadBuildCtx anima_ctx() {
    RadBuildCtx c = build_ctx(0, 1);
    c.max_tok = 2 * (4096 + 512);
    c.max_seqs = 4;
    return c;
}

/* A JOINT_STEP pass as the driver lays one out for Anima: COND over a prompt of `nq` Qwen3 tokens
 * (padded to 512) and `nt` T5 ids, DENOISE over `entries` entries of 512 context rows and `ni`
 * image rows each, or DECODE of a 64 x 64 picture. */
struct AnimaKit {
    std::vector<int32_t> tok, pos, cu, src, lane, flags, pair, aux, aux_cu, valid_cu;
    std::vector<int32_t> img_cu, img_rows, mode_cu, mode_rows, mode_src, mode_ccu;
    std::vector<int64_t> n_mode;
    std::vector<uint32_t> seed;
    std::vector<float> sigma, sigma_next, cfg, rope_f, mode_rope_f;
    RadGenBatch g{};
    RadBatch b{};
};

std::unique_ptr<AnimaKit> anima_pass(int pass, int64_t nq = 12, int64_t nt = 9, int64_t ni = 16,
                                     int64_t entries = 1) {
    auto k = std::make_unique<AnimaKit>();
    RadGenBatch& g = k->g;
    RadBatch& b = k->b;
    b.gen = pass;
    b.gen_b = &k->g;
    if (pass == RAD_GEN_PASS_COND) {
        for (int64_t i = 0; i < 512; ++i) {
            k->tok.push_back(i < nq ? (int32_t)(100 + i) : 151643);
            k->pos.push_back(i < nq ? (int32_t)i : 1);
        }
        for (int64_t i = 0; i < nt; ++i) k->aux.push_back((int32_t)(3 + i));
        k->aux_cu = { 0, (int32_t)nt };
        k->valid_cu = { 0, (int32_t)nq };
        k->cu = { 0, 512 };
        k->lane = { 0, 1 };
        k->sigma = { 0.0f };
        b.n_tok = 512;
        b.n_seq = 1;
        b.token_ids = k->tok.data();
        b.positions = k->pos.data();
        b.cu_seqlens = k->cu.data();
        g.rows = 512;
        g.n_pad = (int32_t)(512 - nq);
        g.valid_cu = k->valid_cu.data();
        g.n_lane = 2;
        g.lane = k->lane.data();
        g.lane_h = k->lane.data();
        g.branch = 1;
        g.sigma = k->sigma.data();
        g.n_aux = nt;
        g.aux_ids = k->aux.data();
        g.aux_cu = k->aux_cu.data();
        return k;
    }
    if (pass == RAD_GEN_PASS_DECODE) {
        k->cu = { 0, 16 };
        b.n_tok = 16;
        b.n_seq = 1;
        b.cu_seqlens = k->cu.data();
        g.dec_lane = 1;
        g.dec_height = g.dec_width = 64;
        return k;
    }
    const int64_t S = entries, N = S * (512 + ni);
    k->cu.push_back(0);
    k->img_cu.push_back(0);
    for (int64_t s = 0; s < S; ++s) {
        k->mode_cu.push_back((int32_t)(s * (512 + ni)));
        k->mode_cu.push_back((int32_t)(s * (512 + ni) + 512));
        k->mode_cu.push_back((int32_t)(s * (512 + ni) + 512));
        for (int64_t r = 0; r < 512; ++r) k->src.push_back((int32_t)(s * 512 + r));
        for (int64_t r = 0; r < ni; ++r) {
            k->src.push_back((int32_t)(-1 - (s * ni + r)));
            k->img_rows.push_back((int32_t)(s * (512 + ni) + 512 + r));
        }
        k->cu.push_back((int32_t)((s + 1) * (512 + ni)));
        k->img_cu.push_back((int32_t)((s + 1) * ni));
        k->lane.push_back((int32_t)(s / 2));
        k->pair.push_back(S > 1 ? (s % 2 ? -2 : (int32_t)(s + 1)) : -1);
    }
    k->mode_cu.push_back((int32_t)N);
    for (int64_t s = 0; s < S; ++s)
        for (int64_t r = 0; r < 512; ++r) {
            k->mode_rows.push_back((int32_t)(s * (512 + ni) + r));
            k->mode_src.push_back((int32_t)(s * 512 + r));
        }
    for (int64_t s = 0; s < S; ++s)
        for (int64_t r = 0; r < ni; ++r) {
            k->mode_rows.push_back((int32_t)(s * (512 + ni) + 512 + r));
            k->mode_src.push_back((int32_t)(-1 - (s * ni + r)));
        }
    k->n_mode = { 512 * S, 0, ni * S };
    for (int m = 0; m < 3; ++m)
        for (int64_t s = 0; s <= S; ++s)
            k->mode_ccu.push_back((int32_t)(s * (m == 0 ? 512 : m == 1 ? 0 : ni)));
    k->rope_f.assign((size_t)(4 * N), 0.0f);
    k->mode_rope_f.assign((size_t)(4 * N), 0.0f);
    k->tok.assign((size_t)N, 0);
    k->pos.assign((size_t)N, 0);
    k->sigma.assign((size_t)S, 0.75f);
    k->sigma_next.assign((size_t)S, 0.5f);
    k->cfg.assign((size_t)S, 4.0f);
    k->flags.assign((size_t)S, RAD_GEN_F_FIRST);
    k->seed.assign((size_t)(2 * S), 7u);
    b.n_tok = N;
    b.n_seq = S;
    b.token_ids = k->tok.data();
    b.positions = k->pos.data();
    b.cu_seqlens = k->cu.data();
    g.src = k->src.data();
    g.n_img = ni * S;
    g.img_cu = k->img_cu.data();
    g.img_rows = k->img_rows.data();
    g.mode_cu = k->mode_cu.data();
    g.rope_f = k->rope_f.data();
    g.mode_rows = k->mode_rows.data();
    g.mode_src = k->mode_src.data();
    g.mode_ccu = k->mode_ccu.data();
    g.n_mode_h = k->n_mode.data();
    g.mode_rope_f = k->mode_rope_f.data();
    g.sigma = k->sigma.data();
    g.sigma_next = k->sigma_next.data();
    g.cfg = k->cfg.data();
    g.pair = k->pair.data();
    g.lane = k->lane.data();
    g.lane_dh = k->lane.data();
    g.flags = k->flags.data();
    g.seed = k->seed.data();
    return k;
}

/* The linears a LoRA for Anima names: every DiT block's attention, MLP and modulation linears
 * and the text encoder's. */
bool anima_lora_target(const std::string& w) {
    for (const char* t : { "attn_qkv", "attn_output", "cross_q", "cross_k", "cross_v",
                           "cross_output", "mlp_layer1", "mlp_layer2", ".mod_", "ffn_gate_up",
                           "ffn_down" })
        if (has(w, t) && !has(w, ".scale")) return true;
    return false;
}

}  /* namespace */

/* A JOINT_STEP generator that reads a second tokenizer, pads an empty prompt to one token and
 * samples on ComfyUI's simple schedule; the variant sets its defaults and the DiT keys' prefix is
 * the checkpoint's, `model.diffusion_model.` here and `net.` for Base. */
TEST(anima_declares_a_two_tokenizer_generator_under_the_checkpoints_prefix) {
    const RadEncoding plain = rad_enc_plain(RAD_BF16);
    RadBuildCtx ctx = anima_ctx();
    MetaKV mk = anima_meta("aesthetic");
    RadBuilder b;
    b.encs = { { "dit.t_norm.mdm", plain }, { "dit.blk.0.attn_qkv", plain } };
    REQUIRE_EQ(anima::declare(&b, mk.meta(), &ctx), RAD_OK);
    CHECK_EQ(b.logits, (rad_buf)0);
    CHECK_EQ(b.generator_name, std::string("anima-aesthetic"));
    const RadGeneratorDecl& d = b.generator;
    CHECK_EQ(d.layout, (int32_t)RAD_GEN_LAYOUT_JOINT_STEP);
    CHECK_EQ(d.aux_tokens, 1);
    CHECK_EQ(d.aux_max_tokens, (int64_t)512);
    CHECK_EQ(d.min_tokens, 1);
    CHECK_EQ(d.pad_multiple, 512);
    CHECK_EQ(d.cond_rows, (int64_t)512);
    CHECK_EQ(d.sigma_grid, 1000);
    CHECK_EQ(d.shift, 3.0);
    CHECK_EQ(d.dyn_shift, 0);
    CHECK_EQ(d.cfg_original, 0);
    CHECK_EQ(d.default_steps, (int64_t)30);
    CHECK_EQ(d.default_cfg, 4.0);
    CHECK_EQ(d.latent_channels, (int64_t)64);
    CHECK(b.kv_groups.empty());
    const RecMap* q = find_map(b, "dit.blk.27.attn_qkv.weight");
    REQUIRE(q != nullptr);
    REQUIRE_EQ(q->src.size(), (size_t)3);
    CHECK_EQ(q->src[0], std::string("model.diffusion_model.blocks.27.self_attn.q_proj.weight"));
    const RecMap* te = find_map(b, "te.blk.3.ffn_gate_up.weight");
    REQUIRE(te != nullptr);
    CHECK_EQ(te->src[1], std::string("model.layers.3.mlp.up_proj.weight"));
    /* The patch embedding over its first 64 columns, the final linear's rows in eight pieces. */
    const RecMap* xe = find_map(b, "dit.x_embed.weight");
    REQUIRE(xe != nullptr);
    CHECK_EQ(xe->mode, (int)RAD_MAP_COLS);
    int final_maps = 0;
    for (const auto& m : b.maps)
        if (m.declared == "dit.final.weight") { ++final_maps; CHECK_EQ(m.mode, (int)RAD_MAP_ROWS); }
    CHECK_EQ(final_maps, 8);
    /* The DiT's rotary: 22, 21 and 21 pairs, NeoX, over (frame, row, column). */
    int axial = 0;
    for (const auto& o : b.ops)
        if (o.op == "qk_norm_rope" && pstr(&o, "mode") == "axial") {
            CHECK_EQ(pstr(&o, "sections"), std::string("22 21 21"));
            ++axial;
        }
    CHECK_EQ(axial, 28);

    MetaKV turbo = anima_meta("turbo");
    RadBuilder bt;
    bt.encs = { { "dit.t_norm.mdm", plain } };
    REQUIRE_EQ(anima::declare(&bt, turbo.meta(), &ctx), RAD_OK);
    CHECK_EQ(bt.generator_name, std::string("anima-turbo"));
    CHECK_EQ(bt.generator.default_steps, (int64_t)10);
    CHECK_EQ(bt.generator.default_cfg, 1.0);

    MetaKV base = anima_meta("base");
    RadBuilder bb;
    bb.encs = { { "dit.t_norm.net", plain } };
    REQUIRE_EQ(anima::declare(&bb, base.meta(), &ctx), RAD_OK);
    const RecMap* nq = find_map(bb, "dit.blk.0.mlp_layer1.weight");
    REQUIRE(nq != nullptr);
    CHECK_EQ(nq->src[0], std::string("net.blocks.0.mlp.layer1.weight"));

    /* Neither spelling, or an unknown variant, is refused. */
    RadBuilder none;
    CHECK(anima::declare(&none, mk.meta(), &ctx) < 0);
    MetaKV odd = anima_meta("deluxe");
    RadBuilder bo;
    bo.encs = { { "dit.t_norm.mdm", plain } };
    CHECK(anima::declare(&bo, odd.meta(), &ctx) < 0);
}

/* Every linear a LoRA for Anima names is read by a plain GEMM a runtime LoRA can follow: gemm_nt
 * in a bf16 container, gemm_nt_q without rotated parts in a block-fp8 one, its activation a
 * separate op. */
TEST(anima_lora_targets_are_plain_gemms) {
    const RadEncoding plain = rad_enc_plain(RAD_BF16);
    const auto fp8 = rad_enc_affine(RAD_F8E4M3, RAD_BF16, 128, 128);
    RadBuildCtx ctx = anima_ctx();
    MetaKV mk = anima_meta("base");
    for (bool q : { false, true }) {
        RadBuilder b;
        b.encs = { { "dit.t_norm.net", plain } };
        if (!q) b.encs.push_back({ "dit.blk.0.attn_qkv", plain });
        if (q)
            for (const char* t : { "attn_qkv", "attn_output", "cross_q", "cross_k", "cross_v",
                                   "cross_output", "mlp_layer1", "mlp_layer2", "ffn_gate_up",
                                   "ffn_down" })
                b.encs.push_back({ t, fp8 });
        REQUIRE_EQ(anima::declare(&b, mk.meta(), &ctx), RAD_OK);
        int targets = 0;
        for (const auto& o : b.ops)
            for (rad_weight w : o.w) {
                const std::string& n = b.weights[w - 1].name;
                if (!anima_lora_target(n)) continue;
                ++targets;
                const bool ok = o.op == "gemm_nt" || (q && o.op == "gemm_nt_q" &&
                                                      pint(&o, "pb1") <= 0 && pint(&o, "pb2") <= 0);
                if (!ok) fprintf(stderr, "    '%s' read by %s\n", n.c_str(), o.op.c_str());
                CHECK(ok);
                CHECK(o.op != "gemm_nt_bias");
            }
        /* 28 blocks of 8 linears (two views a block-fp8 one) and 6 modulation linears; 28 text
         * encoder layers of 4. */
        CHECK(targets >= 28 * (8 + 6) + 28 * 4);
    }
}

/* The three passes the core drives (a conditioning with its T5 ids, denoising one image or a
 * guided pair, decoding one) lint clean in a bf16 and a block-fp8 container. */
TEST(anima_generator_passes_lint_clean) {
    const RadEncoding plain = rad_enc_plain(RAD_BF16);
    const auto fp8 = rad_enc_affine(RAD_F8E4M3, RAD_BF16, 128, 128);
    RadBuildCtx ctx = anima_ctx();
    MetaKV mk = anima_meta("aesthetic");
    for (bool q : { false, true }) {
        RadBuilder b;
        b.encs = { { "dit.t_norm.mdm", plain } };
        if (!q) b.encs.push_back({ "dit.blk.0.attn_qkv", plain });
        if (q) b.encs.push_back({ "attn_qkv", fp8 }), b.encs.push_back({ "attn_output", fp8 }),
               b.encs.push_back({ "cross_", fp8 }), b.encs.push_back({ "mlp_layer", fp8 }),
               b.encs.push_back({ "ffn_", fp8 });
        REQUIRE_EQ(anima::declare(&b, mk.meta(), &ctx), RAD_OK);
        for (const auto& k : { anima_pass(RAD_GEN_PASS_COND, 12, 9),
                               anima_pass(RAD_GEN_PASS_COND, 3, 40),
                               anima_pass(RAD_GEN_PASS_DENOISE, 12, 9, 16, 1),
                               anima_pass(RAD_GEN_PASS_DENOISE, 12, 9, 16, 2),
                               anima_pass(RAD_GEN_PASS_DECODE) }) {
            RadCtx c{ .b = &b, .batch = &k->b, .rank = 0, .world = 1 };
            anima::step(&c, &k->b);
            CHECK(c.step_fail.empty());
            if (!c.step_fail.empty()) fprintf(stderr, "    %s\n", c.step_fail.c_str());
            CHECK(!c.issues.empty());
            CHECK_LINT(b, &c);
        }
        const auto tok = kit(b, { 1 }, {});
        RadCtx c{ .b = &b, .batch = &tok->b, .rank = 0, .world = 1 };
        anima::step(&c, &tok->b);
        CHECK(!c.step_fail.empty());
    }
}

/* ======================================================== the conventional vocabulary */

namespace {

/* libref's vocabulary: every op its schemas name. A CPU-only engine, or a new hardware library
 * that implements only what libref does, declares exactly these (libavx's are a subset). Empty
 * when libref.so is not under $RADIANCE_HOME/kernels (ctest sets it to the build's plugin home). */
const std::set<std::string>& libref_vocab() {
    static std::set<std::string> v;
    static bool tried = false;
    if (tried) return v;
    tried = true;
    const char* home = std::getenv("RADIANCE_HOME");
    if (!home) return v;
    const std::string path = std::string(home) + "/kernels/libref.so";
    void* h = dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
    if (!h) {
        fprintf(stderr, "    cannot load %s: %s\n", path.c_str(), dlerror());
        return v;
    }
    auto count = (int (*)(void))dlsym(h, "rad_kernel_schema_count");
    auto at = (const RadOpSchema* (*)(int))dlsym(h, "rad_kernel_schema_at");
    if (count && at)
        for (int i = 0; i < count(); ++i)
            if (const RadOpSchema* s = at(i); s && s->op) v.insert(s->op);
    return v;
}

/* One architecture and configuration. `pinned` lists the device-only ops it may still ask for on
 * libref's vocabulary: an upper bound that may only shrink. An op libref gains drops out of the
 * expectation by itself; one the plugin stops asking for must be removed here. */
struct ArchCase {
    std::string name;
    std::function<int(RadBuilder*)> declare;
    StepFn step = nullptr;
    int max_spec = 0;
    std::vector<std::string> pinned;
    /* The declared graph's fingerprint with every op declared (graph_hash). */
    uint64_t graph = 0;
};

/* Every architecture in the tree, at the fixtures above. */
struct ArchCases {
    MetaKV q35 = q35fp8_meta(), q35df = with_dflash2(q35fp8_meta()), q35moe = q35moe_meta();
    MetaKV q35w8df = with_dflash2(q35w8a8_meta());
    MetaKV lla = with_dspark(llama_meta()), eg2 = eg2_meta(), qimg = qimg_meta();
    MetaKV q4s = q4_small();
    RadModelMeta q35b = qwen35_meta(), q4 = *q4s.meta(), lfm = lfm2_meta();
    RadModelMeta g4 = gemma4_meta(), g4m = gemma4_moe_meta(), g4df = gemma4_moe_meta();
    std::vector<const char*> g4dk = g4df_keys(), g4dv = g4df_vals();
    RadEncoding w4{}, w4g{}, fp8{};
    std::vector<ArchCase> list;

    ArchCases() {
        g4df.n_kv = (int)g4dk.size(); g4df.kv_key = g4dk.data(); g4df.kv_val = g4dv.data();
        w4 = rad_enc_affine(RAD_I4, RAD_BF16, 1, 128);
        rad_enc_copy_str(w4.transform, "fwht128");
        w4g = rad_enc_affine(RAD_I4, RAD_F16, 1, 128);
        fp8 = rad_enc_affine(RAD_F8E4M3, RAD_BF16, 128, 128);
        auto sc = [](int world, int max_spec) { return spec_ctx(0, world, max_spec); };
        auto g4w4 = [this](RadBuilder* b) {
            b->encs = { { "attn_q.", w4g }, { "attn_k.", w4g }, { "attn_v.", w4g },
                        { "attn_o.", w4g }, { "ffn_gate_up.", w4g }, { "ffn_down.", w4g } };
        };
        /* The ops libref lacks that each still asks for, both two-rank fusions with no chain:
         * qwen35_fp8's all-reduce norm (layer_fold assumes it at two ranks) and Flash-Next's
         * all-reduce into the wide stream. */
        auto add = [&](const char* name, std::function<int(RadBuilder*)> d, StepFn step,
                       int max_spec, std::vector<std::string> pinned, uint64_t graph) {
            list.push_back({ name, std::move(d), step, max_spec, std::move(pinned), graph });
        };
        add("qwen35_bf16", [this, sc](RadBuilder* b) {
                auto c = sc(1, 0); return qwen35_bf16::declare(b, &q35b, &c); },
            qwen35_bf16::step, 0, {}, 0xf82117051023e608ull);
        add("qwen35_fp8 + MTP", [this, sc](RadBuilder* b) {
                auto c = sc(1, 3); return qwen35_fp8::declare(b, q35.meta(), &c); },
            qwen35_fp8::step, 3, {}, 0x74fd4495028c1525ull);
        add("qwen35_fp8 + MTP, two ranks", [this, sc](RadBuilder* b) {
                auto c = sc(2, 3); return qwen35_fp8::declare(b, q35.meta(), &c); },
            nullptr, 3, { "ar_rmsnorm_quant_fp8" }, 0x1c28909125c12f52ull);
        add("qwen35_fp8 + DFlash2", [this, sc](RadBuilder* b) {
                auto c = sc(1, 7); return qwen35_fp8::declare(b, q35df.meta(), &c); },
            qwen35_fp8::step, 7, {}, 0xb5691456a0ce7e9dull);
        add("qwen35_w8a8 + DFlash2", [this, sc](RadBuilder* b) {
                auto c = sc(1, 7); return w8a8_declare(b, q35w8df.meta(), &c); },
            qwen35_w8a8::step, 7, {}, 0xed8219da9fcc0ceaull);
        add("qwen35_w8a8 + DFlash2, two ranks", [this, sc](RadBuilder* b) {
                auto c = sc(2, 7); return w8a8_declare(b, q35w8df.meta(), &c); },
            nullptr, 7, { "ar_rmsnorm_quant_fp8" }, 0x2c00f131709dedcfull);
        add("qwen35moe_fp8 + MTP", [this, sc](RadBuilder* b) {
                auto c = sc(1, 3); return qwen35moe_fp8::declare(b, q35moe.meta(), &c); },
            qwen35moe_fp8::step, 3, {}, 0x1bd3b53543cd06eeull);
        add("qwen4exp_fp8", [this](RadBuilder* b) {
                auto c = q4_ctx(0, 1); return qwen4exp_fp8::declare(b, &q4, &c); },
            qwen4exp_fp8::step, 3, {}, 0xbe74d476c7fd04b8ull);
        add("qwen4exp_fp8, two ranks", [this](RadBuilder* b) {
                auto c = q4_ctx(0, 2); return qwen4exp_fp8::declare(b, &q4, &c); },
            nullptr, 3, { "ar_hc_write" }, 0x9d7df7914590d6d8ull);
        add("qwen4exp_fp8, four-bit experts", [this](RadBuilder* b) {
                b->encs = { { "ffn_gate_up_exps", w4 }, { "ffn_down_exps", w4 } };
                auto c = q4_ctx(0, 1); return qwen4exp_fp8::declare(b, &q4, &c); },
            qwen4exp_fp8::step, 3, {}, 0xb308d06b168a61f1ull);
        add("llama_fp8 + DSpark", [this](RadBuilder* b) {
                auto c = dspark_ctx(0, 1); return llama_fp8::declare(b, lla.meta(), &c); },
            llama_fp8::step, 7, {}, 0xdcf36a1d31aba73cull);
        add("lfm2_bf16", [this, sc](RadBuilder* b) {
                auto c = sc(1, 0); return lfm2_bf16::declare(b, &lfm, &c); },
            lfm2_bf16::step, 0, {}, 0x1775b188dc4a5120ull);
        add("gemma4_bf16", [this, sc](RadBuilder* b) {
                auto c = sc(1, 0); return gemma4_bf16::declare(b, &g4, &c); },
            gemma4_bf16::step, 0, {}, 0xafc23087c634a537ull);
        add("gemma4_bf16, w4a8", [this, sc, g4w4](RadBuilder* b) {
                g4w4(b); auto c = sc(1, 0); return gemma4_bf16::declare(b, &g4, &c); },
            gemma4_bf16::step, 0, {}, 0x719fa07788f8b98aull);
        add("gemma4_bf16, w4a8, two ranks", [this, sc, g4w4](RadBuilder* b) {
                g4w4(b); auto c = sc(2, 0); return gemma4_bf16::declare(b, &g4, &c); },
            nullptr, 0, {}, 0x93b174ab52cc0d31ull);
        add("gemma4", [this, sc](RadBuilder* b) {
                auto c = sc(1, 0); return gemma4::declare(b, &g4m, &c); },
            gemma4::step, 0, {}, 0x0f227f5853b6488cull);
        add("gemma4, fp8 experts", [this, sc](RadBuilder* b) {
                b->encs = { { "ffn_gate_up_exps", fp8 } };
                auto c = sc(1, 0); return gemma4::declare(b, &g4m, &c); },
            gemma4::step, 0, {}, 0x532d06c46f2e7488ull);
        add("gemma4 + DFlash", [this, sc](RadBuilder* b) {
                auto c = sc(1, 15); return gemma4::declare(b, &g4df, &c); },
            gemma4::step, 15, {}, 0x37bffcc9bd749b88ull);
        add("embgemma2", [this](RadBuilder* b) {
                auto c = eg2_ctx(); return embgemma2::declare(b, eg2.meta(), &c); },
            embgemma2::step, 0, {}, 0xac72c705bc97d2b3ull);
        add("qwenimage21", [this](RadBuilder* b) {
                auto c = qimg_ctx(0, 1); return qwenimage21::declare(b, qimg.meta(), &c); },
            nullptr, 0, {}, 0xd7990ce44671b171ull);
    }
};

ArchCases& arch_cases() {
    static ArchCases c;
    return c;
}

/* The engine steps a model runs: decode, a prefill chunk, a mixed step, and with a drafter a
 * verify followed by its passes, each run on its own contexts. */
std::vector<std::vector<std::unique_ptr<RadCtx>>> model_steps(RadBuilder& b, StepFn step,
                                                               int max_spec) {
    std::vector<std::vector<std::unique_ptr<RadCtx>>> out;
    for (const auto& k : { kit(b, { 1, 1, 1 }, {}), kit(b, {}, { 24 }), kit(b, { 1 }, { 6 }) })
        out.push_back(run_passes(b, step, { k.get() }));
    if (b.drafter.kind == RAD_DRAFT_NONE) return out;
    const auto ver = kit(b, std::vector<int>(3, 1 + max_spec), {});
    std::vector<std::unique_ptr<Kit>> ps;
    if (b.drafter.kind == RAD_DRAFT_SERIAL) {
        ps.push_back(history_pass(b, *ver));
        for (int r = 2; r <= b.drafter.depth; ++r) ps.push_back(round_pass(b, 3, r));
    } else {
        ps.push_back(context_pass(b, *ver));
        ps.push_back(query_pass(b, 3, b.drafter.block, b.drafter.proposal_pitch));
    }
    std::vector<const Kit*> all = { ver.get() };
    for (const auto& p : ps) all.push_back(p.get());
    out.push_back(run_passes(b, step, all));
    return out;
}

std::string joined(const std::set<std::string>& s) {
    std::string r;
    for (const auto& x : s) r += (r.empty() ? "" : " ") + x;
    return r.empty() ? std::string("(none)") : r;
}

/* A 64-bit FNV-1a fingerprint of everything a declare recorded: ops with their parameters,
 * weights, read and write sets and resolution; buffers, weights, KV groups, bindings, name maps,
 * the logits / pooled / drafter / generator declarations and the concurrent list. Two declares
 * give one value only if the core would build the same program from them. */
struct Fnv {
    uint64_t h = 1469598103934665603ull;
    Fnv& raw(const void* p, size_t n) {
        const auto* c = (const unsigned char*)p;
        for (size_t i = 0; i < n; ++i) { h ^= c[i]; h *= 1099511628211ull; }
        return *this;
    }
    Fnv& s(const std::string& x) { return raw(x.data(), x.size()).i((int64_t)x.size()); }
    Fnv& s(const char* x) { return s(std::string(x ? x : "")); }
    Fnv& i(int64_t v) { return raw(&v, sizeof v); }
    Fnv& f(double v) { return raw(&v, sizeof v); }
};
uint64_t graph_hash(const RadBuilder& b) {
    Fnv f;
    for (const RecOp& o : b.ops) {
        f.s(o.op).i(o.resolved).i((int64_t)o.p.size());
        for (const RecParam& p : o.p) f.s(p.key).i(p.kind).i(p.ival).i(p.ihi).f(p.dval).s(p.sval);
        for (const auto* v : { &o.w, &o.reads, &o.writes }) {
            f.i((int64_t)v->size());
            for (uint32_t x : *v) f.i(x);
        }
    }
    for (const RecBuf& x : b.bufs) {
        f.s(x.name).i(x.d.dtype).i(x.d.rank).i(x.d.kind).i(x.d.domain).i(x.d.kv).s(x.d.derive);
        for (uint32_t d = 0; d < x.d.rank; ++d) f.i(x.d.shape[d]);
    }
    for (const RecWeight& w : b.weights) {
        f.s(w.name).i(w.d.dtype).i(w.d.rank).i(w.d.shard).i(w.d.access).i(w.d.group.layer)
         .i(w.d.group.expert).s(w.d.source).s(w.d.planes).i(w.d.n_row_parts).i(w.span_lo)
         .i(w.span_hi).i(w.n_part_spans);
        for (uint32_t d = 0; d < w.d.rank; ++d) f.i(w.d.shape[d]);
    }
    for (const RecKV& g : b.kv_groups)
        f.s(g.name).i(g.d.kind).i(g.d.dtype).i(g.d.window).i(g.d.n_head_kv).i(g.d.head_dim)
         .i(g.d.state_dim[0]).i(g.d.state_dim[1]).i(g.d.conv_width).i(g.zero);
    for (const auto& [l, g] : b.binds) f.i(l).i(g);
    for (const RecMap& m : b.maps) {
        f.s(m.declared).i(m.mode).i(m.concat_dim);
        for (const auto& x : m.src) f.s(x);
    }
    f.i(b.logits).i(b.pooled).s(b.drafter_name).i(b.drafter.kind).i(b.drafter.depth)
     .i(b.drafter.block).i(b.drafter.proposal).i(b.drafter.mask_token).i(b.drafter_shared)
     .s(b.encoder_name).s(b.audio_encoder_name).s(b.generator_name);
    for (rad_buf x : b.concurrent) f.i(x);
    return f.h;
}

}  /* namespace */

/* For every architecture, declaring with only libref's vocabulary either asks for nothing else and
 * then issues only libref's ops (a CPU-only engine serves it), or asks for exactly the pinned
 * device-only ops libref lacks. A new fused op declared without a chain, or a block that starts
 * calling a device-only op, fails here rather than at a CPU-only engine's startup, where a
 * request for an op no library declares stops the declare. */
TEST(each_architecture_declares_on_librefs_vocabulary_or_names_its_device_only_ops) {
    const std::set<std::string>& vocab = libref_vocab();
    if (vocab.empty()) {
        fprintf(stderr, "    SKIP: libref.so not found under $RADIANCE_HOME/kernels\n");
        return;
    }
    for (const ArchCase& ac : arch_cases().list) {
        RadBuilder b;
        b.vocab = &vocab;
        const int rc = stderr_of_rc([&] { return ac.declare(&b); });
        const std::set<std::string> asked(b.asked_undeclared.begin(), b.asked_undeclared.end());
        std::set<std::string> want;
        for (const auto& op : ac.pinned) {
            if (vocab.count(op))
                fprintf(stderr, "    note: %s: libref now declares '%s'; drop it from the pin\n",
                        ac.name.c_str(), op.c_str());
            else
                want.insert(op);
        }
        if (asked != want)
            fprintf(stderr, "    %s asks for: %s\n    pinned: %s\n", ac.name.c_str(),
                    joined(asked).c_str(), joined(want).c_str());
        CHECK(asked == want);
        if (!asked.empty()) continue;
        CHECK_EQ(rc, RAD_OK);
        if (rc != RAD_OK || !ac.step) continue;
        for (const auto& v : model_steps(b, ac.step, ac.max_spec)) {
            CHECK_PASSES(b, v);
            for (const auto& c : v)
                for (const auto& op : issued_ops(b, *c))
                    if (!vocab.count(op)) {
                        fprintf(stderr, "    %s issues '%s'\n", ac.name.c_str(), op.c_str());
                        CHECK(vocab.count(op));
                    }
        }
    }
}

/* With every op declared, as an engine with libr4d sees it, each architecture declares its pinned
 * graph: the conventional-vocabulary fallbacks are taken only where a fused op is not declared or
 * does not resolve, so they leave these values alone. A snapshot (the fingerprint of the recorded
 * declare), since nothing independent describes a whole graph. */
TEST(with_every_op_declared_each_architecture_declares_its_unchanged_graph) {
    for (const ArchCase& ac : arch_cases().list) {
        RadBuilder b;
        const int rc = stderr_of_rc([&] { return ac.declare(&b); });
        CHECK_EQ(rc, RAD_OK);
        const uint64_t h = graph_hash(b);
        if (h != ac.graph)
            fprintf(stderr, "    %s: graph 0x%016llx, pinned 0x%016llx\n", ac.name.c_str(),
                    (unsigned long long)h, (unsigned long long)ac.graph);
        CHECK_EQ(h, ac.graph);
    }
}

/* Where gate_quant_fp8 is not declared, or declared and not resolved, the fp8 gated attention runs
 * the chain it fuses: the per-head sigmoid into a gate buffer of the layer's own, the multiply
 * into the attention output, and quant_act_fp8 into the codes the output projection reads, the
 * three buffers the fused op writes. */
TEST(the_fp8_output_gate_runs_its_chain_where_gate_quant_fp8_is_not_served) {
    MetaKV mk = q35fp8_meta();
    for (int how = 0; how < 2; ++how) {
        RadBuilder b;
        if (how == 0) b.undeclared.insert("gate_quant_fp8");
        else          b.refuse.insert("gate_quant_fp8");
        RadBuildCtx ctx = build_ctx();
        REQUIRE_EQ(qwen35_fp8::declare(&b, mk.meta(), &ctx), RAD_OK);
        CHECK_EQ(count_resolved(b, "gate_quant_fp8"), 0);
        CHECK_EQ(count_op(b, "sigmoid"), 2);
        CHECK_EQ(count_op(b, "mul"), 2);
        const rad_buf ax = (rad_buf)buf_named(b, "attn_out");
        const rad_buf aq = (rad_buf)buf_named(b, "attn_out.q");
        const rad_buf as = (rad_buf)buf_named(b, "attn_out.s");
        REQUIRE(ax && aq && as);
        CHECK(find_b(b, "blk.3.attn_gate") != nullptr);
        CHECK(find_b(b, "blk.7.attn_gate") != nullptr);
        const auto k = kit(b, { 1, 1 }, {});
        const auto v = run_passes(b, qwen35_fp8::step, { k.get() });
        CHECK_PASSES(b, v);
        /* attention, then gate, multiply and quantiser, then the out projection, per layer */
        const std::vector<std::string> ops = issued_ops(b, *v[0]);
        int chains = 0;
        for (size_t i = 0; i + 4 < ops.size(); ++i)
            if (ops[i] == "attn_paged") {
                CHECK_EQ(ops[i + 1], std::string("sigmoid"));
                CHECK_EQ(ops[i + 2], std::string("mul"));
                CHECK_EQ(ops[i + 3], std::string("quant_act_fp8"));
                CHECK_EQ(ops[i + 4], std::string("gemm_nt_q"));
                const RecIssue& q = v[0]->issues[i + 3];
                CHECK_EQ(q.opd[0].handle, ax);
                CHECK_EQ(q.opd[1].handle, aq);
                CHECK_EQ(q.opd[2].handle, as);
                ++chains;
            }
        CHECK_EQ(chains, 2);
    }
}

/* Where gated_had_quant_fp8 is not served, a four-bit routed down projection's input is made by
 * silu_mul into the bf16 plane and had_quant_act_fp8 from it, into the rotated codes the fused op
 * writes. */
TEST(the_rotated_expert_gate_runs_its_chain_where_gated_had_quant_fp8_is_not_served) {
    MetaKV mk = q4_small();
    const RadModelMeta& meta = *mk.meta();
    const RadEncoding w4 = [] { RadEncoding e = rad_enc_affine(RAD_I4, RAD_BF16, 1, 128);
                                rad_enc_copy_str(e.transform, "fwht128"); return e; }();
    RadBuilder b;
    b.encs = { { "ffn_gate_up_exps", w4 }, { "ffn_down_exps", w4 } };
    b.undeclared.insert("gated_had_quant_fp8");
    RadBuildCtx ctx = q4_ctx(0, 1);
    REQUIRE_EQ(qwen4exp_fp8::declare(&b, &meta, &ctx), RAD_OK);
    CHECK(b.asked_undeclared.empty());
    CHECK_EQ(count_op(b, "gated_had_quant_fp8"), 0);
    /* One a routed layer (eight trunk layers and the MTP head), from the product into the down
     * projection's codes. */
    const rad_buf eff = (rad_buf)buf_named(b, "moe_eff"), effq = (rad_buf)buf_named(b, "moe_eff.q");
    int chained = 0;
    for (const auto& o : b.ops)
        if (o.op == "had_quant_act_fp8" && in(o.reads, eff)) {
            ++chained;
            CHECK_EQ(pint(&o, "group"), 128);
            CHECK(in(o.writes, effq));
        }
    CHECK_EQ(chained, 9);
    const auto k = kit(b, { 1, 1 }, { 6 });
    const auto v = run_passes(b, qwen4exp_fp8::step, { k.get() });
    CHECK_PASSES(b, v);
    const std::vector<std::string> ops = issued_ops(b, *v[0]);
    int pairs = 0;
    for (size_t i = 0; i + 1 < ops.size(); ++i)
        pairs += ops[i] == "silu_mul" && ops[i + 1] == "had_quant_act_fp8";
    CHECK(pairs >= 8);
}

/* What each plugin's probe tells the engine before declare: a drafter's depth (an MTP head's
 * operating point, a block drafter's trained block, fixed), an encoder's pooling window and audio
 * budget, a generator's widest pass. */
TEST(the_probes_size_what_declare_needs) {
    {   /* the routed and dense fp8 plugins share the MTP head's depth of 3, not fixed */
        MetaKV moe = q35moe_meta();
        RadArchProbe p{};
        CHECK_OK(qwen35moe_fp8::probe(moe.meta(), &p));
        CHECK_EQ(p.draft_depth, 3);
        CHECK_EQ(p.draft_depth_fixed, 0);
        MetaKV none = q35moe_meta();
        none.drop("qwen35.nextn_predict_layers");
        RadArchProbe q{};
        CHECK_OK(qwen35moe_fp8::probe(none.meta(), &q));
        CHECK_EQ(q.draft_depth, 0);
    }
    {   /* the dense llama build reads DSpark's trained block like llama_fp8 */
        MetaKV d = with_dspark(llama_meta(""));
        RadArchProbe p{};
        CHECK_OK(llama_dense_fp8::probe(d.meta(), &p));
        CHECK_EQ(p.draft_depth, 7);
        CHECK_EQ(p.draft_depth_fixed, 1);
    }
    {   /* the vision tower pools 3 x 3; 30 s of audio at a 10 ms hop is 3000 frames, whole rows of
         * four */
        MetaKV e = eg2_meta();
        RadArchProbe p{};
        CHECK_OK(embgemma2::probe(e.meta(), &p));
        CHECK_EQ(p.enc_merge, 9);
        CHECK_EQ(p.audio_frames, 3000);
    }
    {   /* two entries (an image and its guidance) of the largest image's 16800 latent rows; ten
         * references of 1024 x 1024 pixels at 16 x 16 a patch, with a tenth for rounding */
        MetaKV q = qimg_meta();
        RadArchProbe p{};
        CHECK_OK(qwenimage21::probe(q.meta(), &p));
        CHECK_EQ(p.gen_rows, 2 * 16800);
        CHECK_EQ(p.gen_patches, 10 * 4096 * 11 / 10);
    }
}

/* ==================================================================== rad_arch.h's helpers */

/* The vocabulary split, swept: every rank's slice starts where the previous ends, the slices sum
 * to the vocabulary, a split that divides is even, and one that does not cuts only at whole
 * 128-row tiles (a block-fp8 lm_head's tile), the remainder going to the last rank with rows. A
 * rank can be left empty (301 rows over four ranks: 128, 128, 45, 0), and geom_from refuses that
 * world rather than declare a zero-row head. */
TEST(the_vocabulary_split_tiles_the_vocabulary) {
    int bad = 0;
    for (int64_t total : { 1, 7, 127, 128, 129, 300, 301, 1000, 32000, 151669, 248320, 262144 })
        for (int world = 1; world <= 8; ++world) {
            int64_t at = 0;
            for (int r = 0; r < world; ++r) {
                const int64_t lo = rad::arch::shard_begin(total, world, r);
                const int64_t n = rad::arch::shard_extent(total, world, r);
                const bool ok = lo == at && n >= 0 && lo + n <= total &&
                                (total % world == 0 ? n == total / world
                                                    : (n % 128 == 0 || lo + n == total));
                if (!ok) {
                    ++bad;
                    fprintf(stderr, "    total %lld world %d rank %d: [%lld, +%lld)\n",
                            (long long)total, world, r, (long long)lo, (long long)n);
                }
                at = lo + n;
            }
            if (at != total) ++bad;
        }
    CHECK_EQ(bad, 0);
    CHECK_EQ(rad::arch::shard_extent(301, 4, 2), 45);
    CHECK_EQ(rad::arch::shard_extent(301, 4, 3), 0);
    CHECK_EQ(rad::arch::shard_extent(300, 4, 3), 75);
    CHECK_EQ(rad::arch::shard_extent(248320, 3, 2), 82688);
    rad::arch::Geom g{};
    RadModelMeta m = lfm2_meta();
    m.n_vocab = 301;
    m.n_head = 4; m.n_head_kv = 4;
    RadBuildCtx c = build_ctx(3, 4);
    CHECK_EQ(stderr_of_rc([&] { return rad::arch::geom_from(g, &m, &c, "bf16", RAD_BF16); }),
             RAD_E_INVAL);
}

/* A layer schedule: names map to full (1) or linear (0), an unknown name is refused by its
 * index, a list shorter than the trunk is refused, and entries past the trunk (an MTP block's)
 * are not compared. */
TEST(the_layer_schedule_readers) {
    std::vector<int> k;
    CHECK_OK(rad::arch::parse_layer_kinds("t", "[\"a\", \"b\", \"a\"]", { "b" }, "a", &k));
    CHECK(k == (std::vector<int>{ 0, 1, 0 }));
    CHECK_EQ(stderr_of_rc([&] {
                 return rad::arch::parse_layer_kinds("t", "a c", { "b" }, "a", &k); }),
             RAD_E_UNSUPPORTED);
    CHECK(rad::arch::is_full_attention(3, 4) && rad::arch::is_full_attention(7, 4));
    CHECK(!rad::arch::is_full_attention(4, 4) && !rad::arch::is_full_attention(3, 0));
    const char* lin = "linear_attention", * ful = "full_attention";
    const std::string four = std::string(lin) + " " + lin + " " + lin + " " + ful;
    CHECK_OK(rad::arch::check_layer_schedule("t", four.c_str(), 4, 4, { ful }));
    CHECK_OK(rad::arch::check_layer_schedule("t", (four + " " + ful).c_str(), 4, 4, { ful }));
    CHECK_OK(rad::arch::check_layer_schedule("t", nullptr, 4, 4, { ful }));
    CHECK_OK(rad::arch::check_layer_schedule("t", "", 4, 4, { ful }));
    CHECK_EQ(stderr_of_rc([&] {
                 return rad::arch::check_layer_schedule("t", four.c_str(), 8, 4, { ful }); }),
             RAD_E_UNSUPPORTED);
    CHECK_EQ(stderr_of_rc([&] {
                 return rad::arch::check_layer_schedule("t", four.c_str(), 4, 2, { ful }); }),
             RAD_E_UNSUPPORTED);
}

/* A drafter's taps: out-of-range ids are dropped, not clamped; parsing stops at the first token
 * that is not a number; at most `max` are kept. */
TEST(parse_layer_ids_keeps_the_trunks_layers_in_order) {
    int out[4] = {};
    CHECK_EQ(rad::arch::parse_layer_ids("1 10 20 30 39", 40, out, 4), 4);
    CHECK(out[0] == 1 && out[1] == 10 && out[2] == 20 && out[3] == 30);
    CHECK_EQ(rad::arch::parse_layer_ids("[3, 64, -1, 5]", 40, out, 4), 2);
    CHECK(out[0] == 3 && out[1] == 5);
    CHECK_EQ(rad::arch::parse_layer_ids("2 x 5", 40, out, 4), 1);
    CHECK_EQ(rad::arch::parse_layer_ids(nullptr, 40, out, 4), 0);
    CHECK_EQ(rad::arch::parse_layer_ids("", 40, out, 4), 0);
}

/* The remainder order is a permutation of the ranks for every rotation: attention-zero ranks
 * take places 0.. first, then the attention ranks rotated by `parity`. */
TEST(remainder_place_orders_every_rank_once) {
    for (int world : { 1, 2, 3, 4, 6 })
        for (int attn_world : { world, world - 1 }) {
            if (attn_world < 1) continue;
            rad::arch::Geom g{};
            g.world = world;
            g.attn_world = attn_world;
            for (int parity = 0; parity < world; ++parity) {
                std::set<int64_t> seen;
                for (int r = 0; r < world; ++r) {
                    const int64_t p = rad::arch::remainder_place(g, r, parity);
                    seen.insert(p);
                    if (r >= attn_world) CHECK_EQ(p, (int64_t)(r - attn_world));
                }
                CHECK_EQ((int)seen.size(), world);
                CHECK_EQ(*seen.begin(), 0);
                CHECK_EQ(*seen.rbegin(), (int64_t)(world - 1));
            }
        }
}

/* divide_or_fail refuses rather than rounds, a world of zero included; batch_split takes a pure
 * step's split from its phase and a mixed step's from its fields. */
TEST(divide_or_fail_and_batch_split) {
    int64_t q = -1;
    CHECK_OK(rad::arch::divide_or_fail(48, 4, "x", &q));
    CHECK_EQ(q, 12);
    CHECK_EQ(stderr_of_rc([&] { return rad::arch::divide_or_fail(48, 5, "x", &q); }), RAD_E_INVAL);
    CHECK_EQ(stderr_of_rc([&] { return rad::arch::divide_or_fail(48, 0, "x", &q); }), RAD_E_INVAL);
    RadBatch b{};
    b.n_seq = 5; b.n_tok = 40; b.n_seq_decode = 2; b.n_tok_decode = 7;
    int64_t s = -1, t = -1;
    b.phase = RAD_PHASE_PREFILL; rad::arch::batch_split(&b, &s, &t);
    CHECK(s == 0 && t == 0);
    b.phase = RAD_PHASE_DECODE;  rad::arch::batch_split(&b, &s, &t);
    CHECK(s == 5 && t == 40);
    b.phase = RAD_PHASE_MIXED;   rad::arch::batch_split(&b, &s, &t);
    CHECK(s == 2 && t == 7);
}

/* Gemma 4's assistant head keeps no cache of its own: its history pass is one row per drafting
 * sequence, gathered from the trunk's verify rows, and each later round one row per sequence. The
 * trunk's verify and the head's passes, linted as the one engine step they are, at one and two
 * ranks. */
TEST(gemma4_assistant_head_engine_step_lints_clean) {
    RadModelMeta meta = gemma4_draft_meta();
    for (int world : { 1, 2 })
        for (int rank = 0; rank < world; ++rank) {
            RadBuilder b;
            RadBuildCtx ctx = spec_ctx(rank, world, 3);
            REQUIRE_EQ(gemma4_bf16::declare(&b, &meta, &ctx), RAD_OK);
            const auto ver = kit(b, std::vector<int>(4, 4), {});
            auto his = kit(b, { 1, 1, 1, 1 }, {});
            his->b.draft_pass = -1;
            his->out = { 3, 7, 11, 15 };               /* each sequence's last verify row */
            his->b.out_ids = his->out.data();
            his->b.n_out = 4;
            his->dout = { 0, 1, 2, 3 };
            his->b.draft_out_ids = his->dout.data();
            his->b.n_draft_out = 4;
            const auto r2 = round_pass(b, 4, 2), r3 = round_pass(b, 4, 3);
            const auto v = run_passes(b, gemma4_bf16::step,
                                      { ver.get(), his.get(), r2.get(), r3.get() }, rank, world);
            CHECK_PASSES(b, v);
        }
}

/* gemma4_bf16 asks for its fused prologue and int8 junctions only where a library declares them:
 * on libref's and libavx's vocabulary a w4a8 (compressed-tensors) model declares the unfused
 * chains and asks for no op outside it, at one rank and at two, issuing what it issues when the
 * fused ops are declared and do not resolve. With them declared nothing changes (the graph
 * fingerprints above). */
TEST(gemma4_bf16_without_the_fused_ops_declares_the_chains_and_asks_for_no_undeclared_op) {
    const RadEncoding w4 = rad_enc_affine(RAD_I4, RAD_F16, 1, 128);
    const std::set<std::string> fused = { "qkv_norm_rope", "rmsnorm_quant_i8", "gelu_mul_quant_i8",
                                          "ar_rmsnorm_quant_i8" };
    for (int world : { 1, 2 }) {
        std::vector<std::string> seq[2];
        for (int undeclared = 0; undeclared < 2; ++undeclared) {
            RadBuilder b;
            b.encs = { { "attn_q.", w4 }, { "attn_k.", w4 }, { "attn_v.", w4 }, { "attn_o.", w4 },
                       { "ffn_gate_up.", w4 }, { "ffn_down.", w4 } };
            (undeclared ? b.undeclared : b.refuse) = fused;
            RadModelMeta meta = gemma4_meta();
            RadBuildCtx ctx = build_ctx(0, world);
            REQUIRE_EQ(gemma4_bf16::declare(&b, &meta, &ctx), RAD_OK);
            CHECK(b.asked_undeclared.empty());
            for (const auto& op : fused) CHECK_EQ(count_resolved(b, op), 0);
            const auto k = kit(b, { 1, 1, 1, 1 }, {});
            const auto v = run_passes(b, gemma4_bf16::step, { k.get() }, 0, world);
            CHECK_PASSES(b, v);
            seq[undeclared] = issued_ops(b, *v[0]);
        }
        CHECK_EQ(join(seq[1]), join(seq[0]));
        CHECK(count_issued(seq[1], "quant_act_i8") > 0);
    }
}

/* ==================================================================== qwen3asr */

/* Qwen3-ASR 1.7B's shape, its decoder and its audio tower each cut to TWO layers. */
const char* k_asr_keys[] = {
    "audio_config.model_type", "audio_config.d_model", "audio_config.encoder_attention_heads",
    "audio_config.encoder_ffn_dim", "audio_config.downsample_hidden_size",
    "audio_config.num_mel_bins", "audio_config.encoder_layers", "audio_config.n_window",
    "audio_config.n_window_infer", "audio_config.output_dim", "audio_config.num_key_value_heads",
    "audio_config.activation_function", "audio_token_id",
};
const char* k_asr_vals[] = { "qwen3_asr_encoder", "1024", "16", "4096", "480", "128", "2", "50",
                             "800", "2048", "16", "gelu", "151676" };

RadModelMeta qwen3asr_meta() {
    RadModelMeta m{};
    m.arch_id     = "qwen3asr";
    m.name        = "test-qwen3-asr-1.7b";
    m.quant       = "bf16";
    m.n_layers    = 2;
    m.n_embd      = 2048;
    m.n_head      = 16;
    m.n_head_kv   = 8;
    m.head_dim    = 128;
    m.n_ff        = 6144;
    m.n_vocab     = 151936;
    m.n_ctx_train = 65536;
    m.rms_eps     = 1e-6f;
    m.rope_theta  = 1e6f;
    m.rope_scale  = 1.0f;
    m.n_kv        = (int)(sizeof(k_asr_keys) / sizeof(k_asr_keys[0]));
    m.kv_key      = k_asr_keys;
    m.kv_val      = k_asr_vals;
    return m;
}

/* The decoder is plain Qwen3 and the tower is declared as the audio encoder: GQA 2 paged attention
 * with NeoX rotary over the whole head and a tied head; the two 480-channel convolutions as unfold
 * plus GEMM against the checkpoint's kernels read as matrices; positions a second at a time and
 * attention in blocks of 104 rows read at cu_div 8. */
TEST(qwen3asr_declares_a_qwen3_decoder_and_its_audio_tower) {
    RadModelMeta meta = qwen3asr_meta();
    RadArchProbe pb{};
    CHECK_OK(qwen3asr_bf16::probe(&meta, &pb));
    CHECK_EQ(pb.audio_frames, (int64_t)3120);

    RadBuilder b;
    RadBuildCtx ctx = build_ctx();
    ctx.max_audio_frames = pb.audio_frames;
    CHECK_OK(qwen3asr_bf16::declare(&b, &meta, &ctx));

    CHECK_EQ(b.audio_encoder_name, std::string("audio"));
    CHECK_EQ(b.audio_encoder.modalities, (uint32_t)RAD_MM_AUDIO);
    CHECK_EQ(b.audio_encoder.patch_dim, 128);
    CHECK_EQ(b.audio_encoder.merge, 8);
    CHECK_EQ(b.audio_encoder.max_patches, 3120);
    CHECK_EQ(b.audio_encoder.n_embd, 2048);

    for (const auto& o : b.ops)
        if (o.op == "attn_paged") {
            CHECK_EQ(pint(&o, "gqa"), 2);
            CHECK_EQ(pint(&o, "head_dim"), 128);
            CHECK_EQ(pint(&o, "causal"), 1);
        }
    int rope = 0;
    for (const auto& o : b.ops)
        if (o.op == "rope") {
            ++rope;
            CHECK_EQ(pstr(&o, "mode"), std::string("neox"));
            CHECK_EQ(pint(&o, "rotary_dim"), 128);
        }
    CHECK_EQ(rope, 4);
    const RecMap* head = find_map(b, "output.weight");
    REQUIRE(head != nullptr);
    REQUIRE_EQ(head->src.size(), (size_t)2);
    CHECK_EQ(head->src[1], std::string("model.language_model.embed_tokens.weight"));

    CHECK_EQ(count_op(b, "conv2d"), 1);
    CHECK_EQ(count_op(b, "conv2d_unfold"), 3);
    CHECK_EQ(count_op(b, "seg_sinusoid"), 1);
    const RecWeight* c2 = find_w(b, "a.conv2.w");
    REQUIRE(c2 != nullptr);
    CHECK_EQ(c2->d.rank, 2u);
    CHECK_EQ(c2->d.shape[0], 480);
    CHECK_EQ(c2->d.shape[1], 4320);
    const RecMap* qkv = find_map(b, "a.blk.1.qkv.w");
    REQUIRE(qkv != nullptr);
    REQUIRE_EQ(qkv->src.size(), (size_t)3);
    CHECK_EQ(qkv->src[2], std::string("model.audio_tower.layers.1.self_attn.v_proj.weight"));
    int dense = 0;
    for (const auto& o : b.ops)
        if (o.op == "attn_dense") {
            ++dense;
            CHECK_EQ(pint(&o, "block"), 104);
            CHECK_EQ(pint(&o, "cu_div"), 8);
            CHECK_EQ(pint(&o, "head_dim"), 64);
        }
    CHECK_EQ(dense, 2);
    for (const auto& o : b.ops)
        if (o.op == "conv2d_unfold" || o.op == "conv2d")
            CHECK_EQ(pint(&o, "t_last"), pint(&o, "kh") == 3 ? 1 : 0);
    CHECK_EQ(count_op(b, "scatter_rows"), 1);

    /* How it hears: the clip's markers and a transcription turn with every slot. */
    const RadAudioFormat* f = qwen3asr_bf16::audio_format(&meta);
    REQUIRE(f != nullptr);
    CHECK_EQ(f->struct_size, (uint32_t)sizeof(RadAudioFormat));
    CHECK_EQ(std::string(f->audio_open), std::string("<|audio_start|>"));
    CHECK(std::string(f->transcribe_prompt).find("{audio}") != std::string::npos);
    CHECK(std::string(f->transcribe_prompt).find("{context}") != std::string::npos);
    CHECK_EQ(f->n_languages, 30u);
}

/* An audio pass runs the tower, in order, and reads the core's planes: the first convolution and the
 * unfolds take plane 1 as their mask, the positions and the attention the pass's clip bounds. */
TEST(a_qwen3asr_audio_pass_issues_the_tower_and_nothing_else) {
    RadModelMeta meta = qwen3asr_meta();
    RadBuilder b;
    RadBuildCtx ctx = build_ctx();
    ctx.max_audio_frames = 3120;
    CHECK_OK(qwen3asr_bf16::declare(&b, &meta, &ctx));

    /* One clip of two whole seconds: 208 staged frames, 26 rows. */
    static uint16_t frames[208 * 128];
    static int32_t  coord[4 * 208];
    static const int32_t cu[] = { 0, 208 };
    RadBatch batch{};
    batch.enc         = RAD_MM_AUDIO;
    batch.enc_n_patch = 208;
    batch.enc_pixels  = frames;
    batch.enc_coord   = coord;
    batch.enc_n_seg   = 1;
    batch.enc_cu      = cu;
    batch.enc_max_seg = 208;
    RadCtx c; c.b = &b; c.rank = 0; c.world = 1; c.batch = &batch;
    qwen3asr_bf16::step(&c, &batch);
    CHECK(c.step_fail.empty());

    std::vector<std::string> want = { "conv2d", "conv2d_unfold", "gemm_nt_bias", "conv2d_unfold",
                                      "gemm_nt_bias", "conv2d_unfold", "gemm_nt", "seg_sinusoid",
                                      "add" };
    for (int l = 0; l < 2; ++l)
        for (const char* s : { "layernorm", "gemm_nt_bias", "attn_dense", "gemm_nt_bias",
                               "layernorm", "gemm_nt_bias", "gemm_nt_bias" })
            want.push_back(s);
    for (const char* s : { "layernorm", "gemm_nt_bias", "gemm_nt_bias" }) want.push_back(s);
    CHECK_EQ(join(issued_ops(b, c)), join(want));

    const RecIssue* c1 = issue_of(b, c, "conv2d");
    REQUIRE(c1 != nullptr);
    CHECK_EQ(c1->n, 208);
    CHECK(c1->opd[3].raw == coord + 208);
    /* The second convolution's GEMM runs over 52 time rows of 32 positions. */
    const RecIssue* g2 = issue_of(b, c, "gemm_nt_bias");
    REQUIRE(g2 != nullptr);
    CHECK_EQ(g2->n, 52 * 32);
    const RecIssue* at = issue_of(b, c, "attn_dense");
    REQUIRE(at != nullptr);
    CHECK_EQ(at->n, 26);
    CHECK(at->opd[3].raw == cu);
    CHECK_EQ(c.issues.back().n, 26);
}

/* What this tower does not carry is refused by name. */
TEST(qwen3asr_refuses_a_tower_it_does_not_serve) {
    static const char* vals[] = { "qwen3_asr_encoder", "1024", "16", "4096", "480", "128", "2",
                                  "50", "800", "2048", "16", "silu", "151676" };
    RadModelMeta meta = qwen3asr_meta();
    meta.kv_val = vals;
    RadBuilder b;
    RadBuildCtx ctx = build_ctx();
    ctx.max_audio_frames = 3120;
    CHECK_EQ(qwen3asr_bf16::declare(&b, &meta, &ctx), RAD_E_UNSUPPORTED);
    /* No audio budget: the decoder alone, and no tower. */
    RadModelMeta ok = qwen3asr_meta();
    RadBuilder bt;
    RadBuildCtx text = build_ctx();
    CHECK_OK(qwen3asr_bf16::declare(&bt, &ok, &text));
    CHECK(bt.audio_encoder_name.empty());
    CHECK_EQ(count_op(bt, "conv2d"), 0);
}

RAD_TEST_MAIN()

/* ==================================================================== a 4096 budget without a span */
namespace {

/* What a step issued, as a fingerprint: each launch's op (by name and declared parameters), row
 * count and lane, every operand by kind, buffer or weight name, offset and extent, a raw operand
 * by the batch array and element it points at, and every lane join. */
uint64_t issue_hash(const RadBuilder& b, const std::vector<std::unique_ptr<RadCtx>>& v,
                    const std::vector<const Kit*>& kits) {
    Fnv f;
    auto raw_at = [&](const void* p) {
        if (!p) { f.i(-1); return; }
        if (const rad_buf h = fake_buf_of(p, b.bufs.size())) {
            f.i(1).s(b.bufs[h - 1].name)
             .i((int64_t)((uintptr_t)p - (uintptr_t)rad_buf_ptr(nullptr, h)));
            return;
        }
        for (size_t k = 0; k < kits.size(); ++k) {
            const Kit& t = *kits[k];
            const std::vector<int32_t>* arrays[] = { &t.tok, &t.pos, &t.cu, &t.ctxl, &t.qlen,
                                                     &t.out, &t.nacc, &t.dout, &t.slot, &t.table,
                                                     &t.used, &t.sidx };
            for (size_t a = 0; a < sizeof arrays / sizeof arrays[0]; ++a) {
                const std::vector<int32_t>& x = *arrays[a];
                const int32_t* q = (const int32_t*)p;
                if (!x.empty() && q >= x.data() && q < x.data() + x.size()) {
                    f.i(2).i((int64_t)a).i((int64_t)(q - x.data()));
                    return;
                }
            }
        }
        f.i(3);
    };
    for (const auto& c : v) {
        f.i((int64_t)c->issues.size());
        for (const RecIssue& r : c->issues) {
            const RecOp* o = b.op_of(r.op);
            f.s(o ? o->op : "<null>");
            if (o)
                for (const RecParam& p : o->p) f.s(p.key).i(p.ival).i(p.ihi).f(p.dval).s(p.sval);
            f.i(r.n).i(r.lane).i((int64_t)r.opd.size());
            for (const RadOperand& d : r.opd) {
                f.i(d.kind).i(d.dtype).i(d.offset).i(d.rows).i(d.cols);
                if (d.kind == RAD_OPK_BUF)
                    f.s(d.handle && d.handle <= b.bufs.size() ? b.bufs[d.handle - 1].name : "?");
                else if (d.kind == RAD_OPK_WEIGHT || d.kind == RAD_OPK_WTAB)
                    f.s(d.handle && d.handle <= b.weights.size() ? b.weights[d.handle - 1].name
                                                                 : "?");
                else if (d.kind == RAD_OPK_RAW)
                    raw_at(d.raw);
                else
                    f.i(d.handle);
            }
        }
        for (const RecJoin& j : c->joins) f.i((int64_t)j.at).i(j.from).i(j.to);
    }
    return f.h;
}

/* The steps of the fingerprint: a 4096-row prefill after 4096 tokens of history (wider than a
 * MoE pass), a mixed step, a decode step, and with a drafter a verify and its first passes. */
uint64_t pinned_steps(RadBuilder& b, StepFn step, int rank, int world) {
    const auto pf = kit(b, {}, { 4096 }, 4096);
    const auto mx = kit(b, { 1, 1 }, { 2000 }, 3000);
    const auto dc = kit(b, { 1, 1, 1 }, {});
    std::vector<std::unique_ptr<RadCtx>> all;
    std::vector<const Kit*> kits;
    for (const Kit* k : { pf.get(), mx.get(), dc.get() }) {
        auto v = run_passes(b, step, { k }, rank, world);
        CHECK_PASSES(b, v);
        for (auto& c : v) all.push_back(std::move(c));
        kits.push_back(k);
    }
    std::unique_ptr<Kit> ver, his, r2;
    if (b.drafter.kind == RAD_DRAFT_SERIAL) {
        ver = kit(b, std::vector<int>(3, 4), {});
        his = history_pass(b, *ver);
        r2 = round_pass(b, 3, 2);
        auto v = run_passes(b, step, { ver.get(), his.get(), r2.get() }, rank, world);
        CHECK_PASSES(b, v);
        for (auto& c : v) all.push_back(std::move(c));
        kits.push_back(ver.get()); kits.push_back(his.get()); kits.push_back(r2.get());
    }
    return issue_hash(b, all, kits);
}

}  /* namespace */

/* Without --prefill-span a 4096 budget declares and steps the pinned graph and launches: for
 * Flash-Next and the other plugins built on the same fp8 blocks, at one, two and four ranks, taken
 * from the plugins as they were before the span. */
TEST(a_4096_budget_without_a_span_is_the_pinned_graph_and_steps) {
    MetaKV q4s = q4_small(), q35 = q35fp8_meta(), q35moe = q35moe_meta();
    struct Case {
        const char* name;
        int world, rank;
        uint64_t graph, steps;
    };
    const Case cases[] = {
        { "qwen4exp_fp8", 1, 0, 0x14b9cab0bea2591full, 0x26a9e2c23cf85696ull },
        { "qwen4exp_fp8", 2, 0, 0xc9bcbf1b538b597full, 0x889c6b36c014609full },
        { "qwen4exp_fp8", 2, 1, 0x8b2f1786a4ac12b8ull, 0x959ef6b7c6f57d8bull },
        { "qwen4exp_fp8", 4, 0, 0x1e97a26b759d29dfull, 0xfbe74f2b9d7020bfull },
        { "qwen4exp_fp8", 4, 3, 0x6669aa0573931943ull, 0x36df57cac613c6cfull },
        { "qwen35_fp8", 1, 0, 0xcfc82ca915ec549dull, 0x4d6fca823d1e97efull },
        { "qwen35_fp8", 2, 1, 0x36da8d762194e24eull, 0xe4022daff02ec8bdull },
        { "qwen35moe_fp8", 1, 0, 0x7fa7b015e1bc8873ull, 0x3fcf71676e558d79ull },
        { "qwen35moe_fp8", 2, 0, 0xf60c9508dd197370ull, 0x05e0778848bf416cull },
    };
    for (const Case& cs : cases) {
        RadBuilder b;
        RadBuildCtx ctx = spec_ctx(cs.rank, cs.world, 3);
        ctx.max_tok = 4096;
        ctx.max_seqs = 8;
        ctx.max_ctx = 200000;
        StepFn step = nullptr;
        int rc = RAD_E_INVAL;
        if (!std::strcmp(cs.name, "qwen4exp_fp8")) {
            b.encs = { { "draft_head", u2_head() } };
            rc = stderr_of_rc([&] { return qwen4exp_fp8::declare(&b, q4s.meta(), &ctx); });
            step = qwen4exp_fp8::step;
        } else if (!std::strcmp(cs.name, "qwen35_fp8")) {
            rc = stderr_of_rc([&] { return qwen35_fp8::declare(&b, q35.meta(), &ctx); });
            step = qwen35_fp8::step;
        } else {
            rc = stderr_of_rc([&] { return qwen35moe_fp8::declare(&b, q35moe.meta(), &ctx); });
            step = qwen35moe_fp8::step;
        }
        REQUIRE_EQ(rc, RAD_OK);
        const uint64_t g = graph_hash(b);
        const uint64_t s = pinned_steps(b, step, cs.rank, cs.world);
        fprintf(stderr, "    pin: %s world %d rank %d: graph 0x%016llxull, steps 0x%016llxull\n",
                cs.name, cs.world, cs.rank, (unsigned long long)g, (unsigned long long)s);
        CHECK_EQ(g, cs.graph);
        CHECK_EQ(s, cs.steps);
    }
}

/* GLM declares and steps the pinned graph and launches at one and eight ranks: a 4096-row prefill
 * after history, a mixed step, a decode step, a verify and its draft passes. */
TEST(glm53_is_the_pinned_graph_and_steps) {
    const char* keys[] = { "num_nextn_predict_layers", "hc_mult", "n_routed_experts",
                           "index_topk" };
    const char* vals[] = { "1", "4", "288", "2048" };
    RadModelMeta meta{};
    meta.n_layers = 45;
    meta.n_embd = 4096;
    meta.n_vocab = 154880;
    meta.rms_eps = 1e-5f;
    meta.n_kv = 4;
    meta.kv_key = keys;
    meta.kv_val = vals;
    const auto mx = rad_enc_affine(RAD_FP4E2M1, RAD_E8M0, 1, 32);
    const auto fp8 = rad_enc_affine(RAD_F8E4M3, RAD_F32, 128, 128);
    struct Case {
        int world;
        uint64_t graph, steps;
    };
    const Case cases[] = { { 1, 0xed3fc26c486ea1b5ull, 0x98ba2952bb37b075ull },
                           { 8, 0x8204e073efdeee0cull, 0x1279aecfe4e32d9aull } };
    for (const Case& cs : cases) {
        RadBuilder b;
        b.kv_block = 4;
        b.encs = { { "mtp.blk.45.ffn_gate_up_exps", fp8 },
                   { "mtp.blk.45.ffn_down_exps", fp8 },
                   { "ffn_gate_up_exps", mx },
                   { "ffn_down_exps", mx },
                   { "ffn_gate_up_shexp", mx },
                   { "ffn_down_shexp", mx },
                   { "ffn_gate_up.weight", fp8 },
                   { "ffn_down.weight", fp8 } };
        RadBuildCtx ctx = spec_ctx(0, cs.world, 3);
        ctx.max_tok = 4096;
        ctx.max_seqs = 8;
        ctx.max_ctx = 200000;
        REQUIRE_EQ(glm5next::declare(&b, &meta, &ctx), RAD_OK);
        const uint64_t g = graph_hash(b);
        const uint64_t s = pinned_steps(b, glm5next::step, 0, cs.world);
        fprintf(stderr, "    pin: glm5next world %d: graph 0x%016llxull, steps 0x%016llxull\n",
                cs.world, (unsigned long long)g, (unsigned long long)s);
        CHECK_EQ(g, cs.graph);
        CHECK_EQ(s, cs.steps);
    }
}

/* Under a span a Flash-Next prefill of two intervals runs a pass at a time. At every world its
 * connection reads take a pass each and the indexer scores its pass-tall plane in windows, the MTP
 * head's pass included. At four ranks the step is pipelined (step_pipelined): every layer's
 * all-reduce carries one pass's rows on the second lane at `overlap` 1 and the delta net runs each
 * piece over its own bounds, while the embedding's collective stays whole, on the second lane at
 * no overlap. At two ranks the gated residual's fused write takes the layers' all-reduces and the
 * step runs on one lane. */
TEST(flashnext_a_spanned_prefill_runs_a_pass_at_a_time_with_its_all_reduces_beside_it) {
    MetaKV mk = q4_small();
    const RadModelMeta& meta = *mk.meta();
    for (int world : { 1, 2, 4 })
        for (int rank = 0; rank < world; ++rank) {
            RadBuilder b;
            b.encs = { { "draft_head", u2_head() } };
            RadBuildCtx ctx = q4_ctx(rank, world);
            ctx.max_tok = 4096;
            ctx.max_mixed_tok = 2048;
            REQUIRE_EQ(stderr_of_rc([&] { return qwen4exp_fp8::declare(&b, &meta, &ctx); }),
                       RAD_OK);
            const qwen4exp_fp8::Model& m = qwen4exp_fp8::g_model[rank];
            CHECK(m.span_on);
            CHECK_EQ(m.pipe_on, world > 2);
            CHECK_EQ(m.qsacfg.plane_rows, 2048);
            const auto pf = kit(b, {}, { 4096 }, 4096);
            const auto his = history_pass(b, *pf);
            const auto v = run_passes(b, qwen4exp_fp8::step, { pf.get(), his.get() }, rank, world);
            CHECK_PASSES(b, v);
            const RadCtx& c = *v[0];
            CHECK(c.step_fail.empty());
            /* The MTP head's pass over the same rows reads and scores a pass at a time too. */
            CHECK(v[1]->step_fail.empty());
            for (const RecIssue& i : v[1]->issues)
                if (const RecOp* o = b.op_of(i.op); o && (o->op == "hc_read" || o->op == "qsa_score"))
                    CHECK(i.n <= 2048);
            const int64_t piece = 2048 * m.g.n_embd;
            int reads = 0, wide_reads = 0, ar = 0, ar_lane0 = 0, ar_piece = 0, ar_overlap = 0;
            int score = 0, wide_score = 0, conv = 0, wide_conv = 0;
            for (const RecIssue& i : c.issues) {
                const RecOp* o = b.op_of(i.op);
                if (!o) continue;
                if (o->op == "hc_read") { ++reads; wide_reads += i.n > 2048; }
                if (o->op == "qsa_score") { ++score; wide_score += i.n > 2048; }
                if (o->op == "gdn_conv_prep") { ++conv; wide_conv += i.n > 2048; }
                if (o->op == "all_reduce") {
                    ++ar;
                    ar_lane0 += i.lane == 0;
                    ar_piece += i.n == piece;
                    ar_overlap += pint(o, "overlap") == 1;
                }
            }
            CHECK(reads > 0);
            CHECK_EQ(wide_reads, 0);
            CHECK(score > 0);
            CHECK_EQ(wide_score, 0);
            CHECK(conv > 0);
            if (world < 4) {
                CHECK_EQ(ar, world > 1 ? 1 : 0);
                CHECK_EQ(ar_lane0, ar);
                CHECK(c.joins.empty());
                continue;
            }
            CHECK_EQ(wide_conv, 0);
            CHECK(ar > 2);
            CHECK_EQ(ar_lane0, 0);
            CHECK_EQ(ar_piece, ar - 1);
            CHECK_EQ(ar_overlap, ar - 1);
            CHECK(!c.joins.empty());
        }
}
