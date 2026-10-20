/* rad_fp8.h -- the pieces every block-scaled FP8 block is built from: a linear whose weight is E4M3
 * with a block scale plane, the activation quantisers that feed it, and the fused norm/residual/
 * all-reduce front op.
 *
 * ============================== BLOCK-SCALED FP8 ==============================
 *
 * The checkpoint's format. Qwen3.8-27B-FP8 and other
 * `quantization_config: {quant_method: fp8, weight_block_size: [128,128]}` releases store each
 * linear as
 *
 *   <linear>.weight             F8_E4M3, row-major [N][K]
 *   <linear>.weight_scale_inv   BF16,    row-major [ceil(N/128)][ceil(K/128)]
 *
 * and libr4d's fp8 GEMMs read those bytes (grid verified in libr4d/r4d_fp8_layout.cpp), so
 * rad-convert copies them unchanged. Widening a 27B FP8 checkpoint to bf16 would double the
 * weight bytes.
 *
 * ============================== THE SCALE PLANE IS A WEIGHT ==============================
 *
 * `gemm_nt_q` takes `a`, `a_scale`, `b`(w), `b_scale`(w) -> `y`, so the scale plane is its own
 * rad_weight rather than an `aux` stream: RadArgs cannot deliver an aux stream to a launch
 * (libr4d/r4d_fp8_layout.cpp), and a weight can be placed, moved and load-time sharded on its own
 * (a row-sharded [136,40] plane is sliced like its [17408,5120] weight). The cost is one extra
 * weight per linear (2.1 KiB of scales against 89 MiB at gate_proj's shape) and a name-map entry,
 * `<src>_scale_inv`, derived here.
 *
 * ============================== FP8 ACTIVATIONS ==============================
 *
 * Every block-fp8 linear is `fp8a8`, whose two kernels cover every M (`nt_m16` to 16, `tiled`
 * above), at the cost of quantising each activation that feeds a GEMM. libr4d's `fp8a16` matvec,
 * which reads bf16 directly, is not used: it assumes 16 consecutive k of a row per lane, which the
 * stored WMMA fragment order (libr4d/r4d_fp8_frag.h) does not provide.
 *
 * Quantisation is per activation, not per linear: the three attention projections, or the GDN's
 * in and z projections, share one quantised input. Hence QuantFP8 is separate from LinearFP8.
 */
#ifndef RAD_FP8_H
#define RAD_FP8_H

#include "rad_arch.h"

#include <cstring>
#include <vector>

namespace rad {
namespace arch {

/* The scale block in both dimensions, fixed by the checkpoint (`weight_block_size`). libr4d
 * compiles it in and its rows carry C_EQ("group", 128), so other widths resolve to no kernel. */
enum { RAD_FP8_BLOCK = 128 };

/* Rows of the scale plane for an N-row weight, and columns for a K-wide one. */
inline int64_t fp8_blocks(int64_t n) { return (n + RAD_FP8_BLOCK - 1) / RAD_FP8_BLOCK; }

/* An activation and its quantised codes. The bf16 buffer is what the rest of the graph uses; the
 * codes exist only between the quantiser and the GEMMs that consume them. */
struct ActFP8 {
    rad_buf x = 0;   /* [max_tok, n]                bf16 */
    rad_buf q = 0;   /* [max_tok, n]                e4m3 */
    rad_buf s = 0;   /* [max_tok, ceil(n/block)]    f32 */
    /* Int8 codes of the same activation, for an int8 linear (LinearFP8::i8): [max_tok, n] i8 and
     * an f32 scale per (row, block). Declared only for an int8 trunk (declare_qs's `i8`); written
     * by the linear's own quant_act_i8g from `x`, or by the producer when `q8_fed`. */
    rad_buf q8 = 0;
    rad_buf s8 = 0;
    /* The producer writes q8/s8 instead of the E4M3 pair (a connection read with
     * HyperConn::Config::codes_i8, a gated quantiser, the attention gate or delta-net output
     * epilogue), so an int8 linear reading it declares no quantiser. Set by the model when it
     * declares the pair, before the wires are copied. */
    bool    q8_fed = false;

    /* The code pair the producer writes: int8 when the model fed it, E4M3 otherwise. */
    rad_buf cq() const { return q8_fed ? q8 : q; }
    rad_buf cs() const { return q8_fed ? s8 : s; }

    /* Declares the codes (and the int8 pair with `i8`); `x` is set by the caller.
     *
     * `rows` defaults to g.max_tok and must be given for any buffer that is not one row per token
     * (a MoE expert plane is `rows * top_k`, an encoder tower's its patch count); too small a value
     * overruns the arena (core/runtime/issue.cpp refuses such an issue). */
    /* `i8` declares the int8 pair; `fp8_pair` the E4M3 one. A trunk read entirely by int8 linears
     * (a W8A8 model's) passes i8 with fp8_pair false: no consumer reads the E4M3 codes, and an
     * unused plane still takes arena the KV cache needs. */
    int declare_qs(RadBuilder* b, Names& nm, const Geom& g, const char* name, int64_t n,
                   int64_t rows = 0, bool i8 = false, bool fp8_pair = true) {
        /* bf16 linears read only `x` (Geom::w_bf16), so declare no codes; an unused plane would
         * still take arena space. */
        if (g.w_bf16) { q = s = q8 = s8 = 0; return RAD_OK; }
        const int64_t m = rows > 0 ? rows : g.max_tok;
        if (fp8_pair) {
            q = decl_b(b, nm.f("%s.q", name), RAD_F8E4M3, { m, n });
            s = decl_b(b, nm.f("%s.s", name), RAD_F32,    { m, fp8_blocks(n) });
            if (!q || !s) return RAD_E_INVAL;
        }
        if (i8) {
            q8 = decl_b(b, nm.f("%s.q8", name), RAD_I8,  { m, n });
            s8 = decl_b(b, nm.f("%s.s8", name), RAD_F32, { m, fp8_blocks(n) });
            if (!q8 || !s8) return RAD_E_INVAL;
        }
        return (fp8_pair || i8) ? RAD_OK : RAD_E_INVAL;
    }
};

/* The fused attention prologue's row band, shared by the block issuing it, the rope table fill,
 * the QSA indexer reading that table, and the step deciding whether to fill it.
 *
 * The fusion is bit-exact against the unfused path (r4d_selftest asserts byte equality over 1.8M
 * elements at both gain widths; it reproduces `r4d_rmsnorm_bf16`'s sum-of-squares order). It stops
 * at the 64-row decode band, like the narrow GEMM and the gated fold: at a 2048-token chunk the
 * four separate kernels already fill the device and the fused form, which recomputes each rotary
 * partner's normed value, is slightly slower. */
inline int64_t qk_fuse_rows(const Geom& g) {
    return g.max_tok < 64 ? g.max_tok : 64;
}

/* The rope table's height. `rope_table` writes each position at its own row, so the table needs a
 * row for every position a step can name, and speculative steps (verify and draft rounds) name up
 * to `max_spec` positions past `max_ctx`; one more row is margin. */
inline int64_t rope_table_rows(const Geom& g) {
    return g.max_ctx + (g.max_spec > 0 ? (int64_t)g.max_spec : 0) + 1;
}

/* ================================================================== the activation quantiser */
struct QuantFP8 {
    ActFP8  a{};
    int64_t n  = 0;
    rad_op  op = 0;

    int  declare(RadBuilder* b, const Geom& g, const ActFP8& act, int64_t width);
    /* `r0`/`rows` narrow it to a row range: in a mixed step only the prefill rows need the
     * standalone quantiser, since the recurrent path quantises in its epilogue. */
    void step(RadCtx* c, int64_t T, int64_t r0 = 0, int64_t rows = 0) const;
};

inline int QuantFP8::declare(RadBuilder* b, const Geom& g, const ActFP8& act, int64_t width) {
    a = act;
    n = width;
    /* Nothing to quantise for a bf16 model: no op, and step() issues nothing. */
    if (g.w_bf16) { op = 0; return a.x ? RAD_OK : RAD_E_INVAL; }
    if (!a.x || !a.cq() || !a.cs() || n <= 0) return RAD_E_INVAL;
    /* `dtype` is the input's (bf16); the op name gives the output format. An input fed at int8
     * codes is quantised by quant_act_i8g into its int8 pair. */
    op = rw(b, RAD_OP(b, a.q8_fed ? "quant_act_i8g" : "quant_act_fp8",
               RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok), RAD_INT("n", n),
                          RAD_INT("group", RAD_FP8_BLOCK), RAD_STR("dtype", g.dtype)),
               RAD_NOWEIGHTS),
           { a.x }, { a.cq(), a.cs() });
    return RAD_OK;
}

inline void QuantFP8::step(RadCtx* c, int64_t T, int64_t r0, int64_t rows) const {
    if (!op) return;
    if (rows <= 0) rows = T - r0;
    RAD_ISSUE(c, op, brow_slice(a.x, r0, rows, n), brow_slice(a.cq(), r0, rows, n),
              brow_slice(a.cs(), r0, rows, n / RAD_FP8_BLOCK));
}

/* ================================== the quantiser folded into the op that produces the row
 *
 * A standalone quantiser is mostly launch cost (an 82 KiB row), and most have a producer in the
 * same block:
 *
 *     rmsnorm -> quant     the input of every attention, GDN and MLP linear
 *     silu_mul -> quant    down_proj's input
 *
 * libr4d's `rmsnorm_quant_fp8` and `gated_quant_fp8` quantise in the producer's pass. They still
 * write the bf16 output and take the codes from the bf16-rounded value, so the bytes match the
 * unfused pair exactly.
 *
 * The residual add folds in too. Every block ends with out-projection into the shared `h`,
 * all-reduce, `x += h`, and begins with a norm of `x`, so the add moves to the next block's norm,
 * which already reads both buffers; `fold` says a block has a predecessor to add. The kernel rounds
 * the sum to bf16 and stores it before normalising (unlike the int8 sibling, which keeps the more
 * accurate f32 sum), reproducing the unfused pair byte for byte. */
/* Whether the two-rank all-reduce at `T` rows uses the exact wire. Must match the plugin's
 * `all_reduce` row, which decides from the same three facts: lossy wire not accepted, a message
 * the rotation's group of 64 does not tile, or a message under the byte floor. Every rank
 * evaluates it on the same T.
 *
 * The 6-bit wire carries 2.6x fewer bytes and the fused kernel carries both payloads, so this
 * chooses the fused op's wire; `ar_fused_ok` says whether a fused form exists. The floor exists
 * because small messages (one sequence, T around 8) lose more to the rotation's arithmetic than
 * they save in bytes, while from two sequences up the compressed wire wins increasingly; 128 KiB
 * sits between the regimes.
 */
inline bool ar_wire_is_exact(const Geom& g, int64_t T) {
    if (g.world != 2) return false;
    const int64_t numel = T * g.n_embd;
    if (numel <= 0) return false;
    if (g.wire_exact) return true;
    if (numel % 64) return true;
    return numel * 2 < g.wire_min_bytes;   /* bf16 on the wire */
}

/* The compressed fused form's packing chunk, in elements. The kernel gives each block whole rows
 * (the sum of squares closes over one row), so a chunk must not straddle a row. Fixed here so
 * every rank decides identically at declare whether the op exists. */
static const int64_t kArWire6Chunk = 512;

/* Whether the all-reduce at `T` rows can be folded into the following norm: always on the exact
 * wire, on the compressed one when a chunk tiles the row. The preceding block asks the same
 * question before skipping its own collective. */
inline bool ar_fused_ok(const Geom& g, int64_t T) {
    if (g.world != 2 || T * g.n_embd <= 0) return false;
    return ar_wire_is_exact(g, T) || (g.n_embd % kArWire6Chunk) == 0;
}

/* The four-rank lossy wire's tile: tiered int8 in groups of 32 that may not straddle a shard, so a
 * message whose element count is not a multiple of 128 goes out exact. */
static const int64_t kArWideWireTile = 128;

/* The most rows a decode or verify step carries: 1 + max_spec per sequence, at most max_tok. The
 * four-rank fused write and gather are declared up to this, and ar_wide_write_exact takes no
 * larger step. */
inline int64_t ar_wide_write_rows(const Geom& g) {
    const int64_t r = g.max_seqs * (1 + (g.max_spec > 0 ? g.max_spec : 0));
    return r < g.max_tok ? r : g.max_tok;
}

/* Rows the exact fused write is declared for: every step at two ranks, decode and verify steps at
 * four (ar_wide_write_rows), none otherwise. */
inline int64_t ar_exact_write_rows(const Geom& g) {
    if (g.world == 2) return g.max_tok;
    if (g.world == 4) return ar_wide_write_rows(g);
    return 0;
}

/* Whether a four-rank step takes the fused write (ar_hc_write, exact one-shot only). Both must hold:
 *
 *   rows      T <= ar_wide_write_rows (decode and verify steps). Larger steps keep the separate
 *             `all_reduce`, whose size bands choose one-shot or two-shot. This is a cost bound,
 *             not a correctness one.
 *   exact     --tp-wire exact, or a message the four-rank lossy row sends exact anyway
 *             (not a multiple of kArWideWireTile, or under `min_bytes`).
 *
 * ar_wire_is_exact stays false off two ranks: the embedding's collective uses it to ask whether
 * layer 0's fused norm absorbs the reduction, and at four ranks none does. */
inline bool ar_wide_write_exact(const Geom& g, int64_t T) {
    if (g.world != 4 || T < 1 || T > ar_wide_write_rows(g)) return false;
    const int64_t numel = T * g.n_embd;
    if (numel <= 0) return false;
    if (g.wire_exact) return true;
    if (numel % kArWideWireTile) return true;
    return numel * 2 < g.wire_min_bytes;   /* bf16 on the wire */
}

/* Whether a gated residual's exact fused write takes the all-reduce at T rows: at two ranks when
 * the wire is exact, at four per ar_wide_write_exact. Asked by both the write (HyperConn::ar_write)
 * and the preceding block (ar_taken with kArOutExact). */
inline bool ar_exact_write_takes(const Geom& g, int64_t T) {
    return g.world == 2 ? ar_wire_is_exact(g, T) : ar_wide_write_exact(g, T);
}

/* Which wires the consumer of a block's all-reduce takes; a block is declared without knowing what
 * follows, so the architecture tells it:
 *
 *   kArOutNorm   the next block's norm, which takes either wire it has a fused form for
 *                (ar_fused_ok);
 *   kArOutExact  a gated residual's write with the exact fused form only, so a message that would
 *                go out compressed stays with the block (ar_exact_write_takes -- at four ranks
 *                also one of a step larger than a decode or verify step);
 *   kArOutBoth   a gated residual's write with a fused form on each wire, which takes every
 *                message (HyperConn::ar_take). */
enum ArOut : int {
    kArOutNorm  = 0,
    kArOutExact = 1,
    kArOutBoth  = 2,
};

/* Whether a block skips its own all-reduce because its consumer takes it. The consumer asks the
 * same question of the same T, so exactly one of them issues it. */
inline bool ar_taken(const Geom& g, int64_t T, bool out, int take) {
    if (!out) return false;
    if (take == kArOutBoth)  return true;
    if (take == kArOutExact) return ar_exact_write_takes(g, T);
    return ar_fused_ok(g, T);
}

/* A caller that takes a block's all-reduce a piece at a time (qwen4exp_fp8's pipelined prefill).
 * A block whose last stages are per row finishes its output `rows` rows at a time and calls `done`
 * behind every piece but the last, so the caller can start that piece's reduction while the block
 * works on the next; `handed` counts the calls. A block that cannot cut its output calls nothing
 * and the caller issues every piece itself. The last piece is always the caller's: it waits for the
 * first pieces, and a wait on the second lane is a wait on everything issued there. */
struct ArPieces {
    int64_t rows = 0;
    void  (*done)(void* u, int64_t r0, int64_t n) = nullptr;
    void*   u = nullptr;
    int64_t handed = 0;
    /* Optional, on the card: {0, rows, T} for a step of one prefill sequence in two pieces, so a
     * block whose recurrence runs over cu_seqlens can run it a piece at a time (each piece the
     * chunk a span of 1 makes, the state carried through the cache) and hand off the first
     * piece's reduction before the second piece's projection. Null for any other step. */
    const int32_t* cu = nullptr;
    bool cuts(int64_t T) const { return done && rows > 0 && T > rows; }
    void hand(int64_t r0, int64_t n) { done(u, r0, n); ++handed; }
};

/* Whether a MoE block's gather is folded into the following reduction (ar_gather_hc_write).
 * `rows` and `rows6` are the fused op's declared bounds on the exact and rotated wires (0 if not
 * declared), at most one MoE pass, so a served step is a single pass. The step's wire picks the
 * bound: `rows` wherever ar_exact_write_takes (at four ranks `rows6` is always 0). The block and
 * the write ask this with the same T and bounds, so they agree. */
inline bool gather_taken(const Geom& g, int64_t T, int64_t rows, int64_t rows6) {
    const int64_t r = ar_exact_write_takes(g, T) ? rows : rows6;
    return r > 0 && T <= r;
}

/* Which preceding collective a norm may absorb.
 *
 * `kArAny`: a layer boundary, where the predecessor reduced partial sums and the compressed wire is
 * acceptable.
 *
 * `kArExact`: the embedding. Its collective sums disjoint rows (one rank has the token, the others
 * zeros), so a lossy wire would only degrade an exact value at the model's input, and would differ
 * from the drafter's exact embedding reduction, hurting acceptance. A prompt of a few dozen tokens
 * crosses --tp-wire-min-kb, so allowing it changes greedy text. Exact-only keeps the decode fold
 * and leaves prefill on the standalone collective. */
enum ArFold : int {
    kArNone  = 0,
    kArAny   = 1,
    kArExact = 2,
};

/* Where trunk layer `l` of `n` does its residual add, for a trunk of fp8 blocks whose feed-forward
 * always folds its mixer's delta into its own norm (NormQuantFP8), so a mixer never adds.
 *
 * The mixer folds its predecessor's delta unless it is layer 0 or follows a layer that kept its
 * add. A feed-forward keeps its add where something reads `x` between blocks: a drafter tap (in
 * `taps`), which must see the layer's real output, and the final norm, a plain `rmsnorm` with no
 * residual operand.
 *
 * The all-reduce moves with the add, at two ranks only (tp1 has no collective and there is no
 * fused kernel at four): a block's norm absorbs the preceding collective where it folds. Layer 0's
 * mixer does not fold but absorbs the vocab-sharded embedding's reduction (`embed_ar`), exact
 * only (kArExact). */
struct LayerFold {
    bool mix_fold   = false;
    int  mix_ar_in  = kArNone;
    bool mix_ar_out = false;   /* the mixer leaves its collective to the feed-forward's norm */
    bool mlp_add    = false;
    int  mlp_ar_in  = kArNone;
    bool mlp_ar_out = false;
};

inline LayerFold layer_fold(const Geom& g, int64_t l, int64_t n, const int* taps, int n_taps,
                            bool embed_ar) {
    auto tapped = [&](int64_t i) { return std::find(taps, taps + n_taps, (int)i) != taps + n_taps; };
    const bool tp2 = g.world == 2;
    LayerFold f;
    f.mix_fold   = l > 0 && !tapped(l - 1);
    f.mix_ar_in  = !tp2 ? kArNone : f.mix_fold ? kArAny : (l == 0 && embed_ar) ? kArExact : kArNone;
    f.mix_ar_out = tp2;
    f.mlp_add    = tapped(l) || l + 1 >= n;
    f.mlp_ar_in  = tp2 ? kArAny : kArNone;
    f.mlp_ar_out = tp2 && !f.mlp_add;
    return f;
}

struct NormQuantFP8 {
    ActFP8     a{};
    rad_weight w = 0;
    /* The norm's input: the residual stream, or with a fold the previous block's delta, which the
     * op first adds into the stream (`res`). */
    rad_buf    src = 0;
    rad_buf    res = 0;
    rad_op     op = 0;
    /* The same op with the preceding all-reduce folded in, declared when the caller passes
     * `fuse_ar` (this norm is the collective's sole consumer). Both forms are declared and step()
     * chooses, since the wire depends on T; the predecessor asks the same question and skips its
     * own all_reduce on the same answer. */
    rad_op     op_arn = 0;
    /* The fused form on the rotated 6-bit wire, taken from the second concurrent sequence up
     * (the T * n_embd * 2-byte message then crosses --tp-wire-min-kb). The wire is a declare-time
     * parameter, hence a separate op. */
    rad_op     op_arn6 = 0;
    Geom       gm{};
    /* The row width, for turning a row slice into element offsets. */
    int64_t    n = 0;

    int  declare(RadBuilder* b, const Geom& g, rad_weight gain, rad_buf x, const ActFP8& act,
                 int64_t n, rad_buf residual = 0, int64_t max_rows = 0, int fuse_ar = kArNone);
    void step(RadCtx* c, int64_t T, int64_t r0 = 0, int64_t rows = 0) const;
    /* The op this step runs: the fused form for the step's wire if declared, else the plain op. */
    rad_op pick(int64_t T) const {
        if (op_arn && ar_wire_is_exact(gm, T)) return op_arn;
        if (op_arn6 && ar_fused_ok(gm, T)) return op_arn6;
        return op;
    }
    /* Asked by the preceding block before issuing its own collective. */
    bool takes_ar(int64_t T) const { return pick(T) != op; }
};

inline int NormQuantFP8::declare(RadBuilder* b, const Geom& g, rad_weight gain, rad_buf x,
                                 const ActFP8& act, int64_t n, rad_buf residual,
                                 int64_t max_rows, int fuse_ar) {
    a = act;
    w = gain;
    src = x;
    res = residual;
    gm = g;
    this->n = n;                    /* the parameter shadows the member; the member is the pitch */
    /* bf16 models of this family take block inputs from hc_read and declare no block norm; a
     * pre-norm bf16 model uses the bf16 blocks. */
    if (g.w_bf16) {
        fprintf(stderr, "radiance: a fused norm and quantiser was declared on a model whose "
                        "linears are bf16; its blocks take a prepared input instead\n");
        return RAD_E_UNSUPPORTED;
    }
    if (!src || !a.x || !a.q || !a.s || !w || n <= 0) return RAD_E_INVAL;
    op = RAD_OP(b, "rmsnorm_quant_fp8",
                RAD_PARAMS(RAD_RANGE("M", 1, max_rows > 0 ? max_rows : g.max_tok),
                           RAD_INT("n", n), RAD_F64("eps", g.eps),
                           RAD_INT("group", RAD_FP8_BLOCK), RAD_STR("dtype", g.dtype),
                           RAD_F64("wadd", g.wadd)),
                RAD_WEIGHTS(w));
    if (!op) return RAD_E_INVAL;
    /* Arrays rather than rw(): the sets are conditional, and a braced list in a ternary is not
     * lifetime-extended. `residual` is INOUT, so it is in both sets; otherwise the planner could
     * reorder a reader around it. */
    rad_buf rd[2] = { src, res };
    rad_buf wr[4] = { a.q, a.s, a.x, res };
    rad_op_reads (b, op, rd, res ? 2 : 1);
    rad_op_writes(b, op, wr, res ? 4 : 3);

    /* The fused form also writes the reduced message into `src`. With a fold, `src` is `a.x`,
     * already in the write set; without one it is the residual stream, so it is added (as `res`'s
     * slot) to keep liveness correct.
     *
     * `res` is optional: the embedding's partial (vocab-sharded) lookup also needs this reduction,
     * with `x` the stream and no residual. The kernel supports it (r4d_fused_quant_fp8.hip) and the
     * schema marks `residual` optional. */
    if (fuse_ar && g.world == 2) {
        op_arn = RAD_OP(b, "ar_rmsnorm_quant_fp8",
                        RAD_PARAMS(RAD_RANGE("M", 1, max_rows > 0 ? max_rows : g.max_tok),
                                   RAD_INT("n", n), RAD_F64("eps", g.eps),
                                   RAD_INT("group", RAD_FP8_BLOCK),
                                   RAD_INT("world_size", g.world),
                                   RAD_STR("dtype", g.dtype), RAD_F64("wadd", g.wadd)),
                        RAD_WEIGHTS(w));
        if (!op_arn) return RAD_E_INVAL;
        rad_buf wr_ar[4] = { a.q, a.s, a.x, res ? res : src };
        rad_op_reads (b, op_arn, rd, res ? 2 : 1);
        rad_op_writes(b, op_arn, wr_ar, 4);

        /* Only where the compressed wire is reachable and servable: a lossy payload accepted, and
         * the width tiled by the chunk (otherwise a fatal step-time shape refusal). */
        if (fuse_ar != kArExact && !g.wire_exact && (n % kArWire6Chunk) == 0) {
            op_arn6 = RAD_OP(b, "ar_rmsnorm_quant_fp8",
                             RAD_PARAMS(RAD_RANGE("M", 1, max_rows > 0 ? max_rows : g.max_tok),
                                        RAD_INT("n", n), RAD_F64("eps", g.eps),
                                        RAD_INT("group", RAD_FP8_BLOCK),
                                        RAD_INT("world_size", g.world),
                                        RAD_STR("dtype", g.dtype), RAD_F64("wadd", g.wadd),
                                        RAD_INT("wire", 6)),
                             RAD_WEIGHTS(w));
            if (!op_arn6) return RAD_E_INVAL;
            rad_op_reads (b, op_arn6, rd, res ? 2 : 1);
            rad_op_writes(b, op_arn6, wr_ar, 4);
        }
    }
    return RAD_OK;
}

/* `r0`/`rows` select a row slice, defaulting to the whole step (`brow_slice` at offset 0 equals
 * `brows`). Slicing lets a block run in passes (the routed feed-forward). Each slice must land in
 * the same bucket-table band as the whole (core/runtime/ctx.h), or a different kernel with a
 * different summation order would run. */
inline void NormQuantFP8::step(RadCtx* c, int64_t T, int64_t r0, int64_t rows) const {
    if (rows <= 0) rows = T - r0;
    RAD_ISSUE_N(c, pick(rows), rows, brow_slice(src, r0, rows, n),
                res ? brow_slice(res, r0, rows, n) : RAD_NONE, RAD_W(w),
                brow_slice(a.q, r0, rows, n),
                brow_slice(a.s, r0, rows, n / RAD_FP8_BLOCK),
                brow_slice(a.x, r0, rows, n));
}

/* Gate activation then quantise, over a [gate|up] input. `act` names the gate: "silu" for
 * down_proj's input, "sigmoid" for a gated attention o_proj's. */
struct GateQuantFP8 {
    ActFP8 a{};
    rad_op op = 0;
    /* Whether the bf16 product is written. A consumer of the bf16 value must request it
     * (docs/OPS.md), but a block-fp8 down_proj reads only codes and scales; only one that reads
     * the bf16 plane (LinearFP8::reads_x) needs it. Skipping it saves [T, n_ff] per layer (18.2 GB
     * over a 16k prefill). Left out of the write list, not just passed absent, so the plan is
     * accurate. */
    bool   want_x = true;
    /* The output width; `gate_up` is 2n wide. */
    int64_t n = 0;
    /* `silu_mul` into the bf16 plane alone, for a model whose linears are bf16 (Geom::w_bf16). */
    bool   bf16 = false;
    /* The rotated form's chain, where no loaded library declares `gated_had_quant_fp8` or it does
     * not resolve: `silu_mul` into the bf16 plane (`op`), then `had_quant_act_fp8` over it into the
     * codes (`op_q`). */
    rad_op op_q = 0;

    /* `silu_only` takes the `silu_mul`-into-bf16 path regardless of `g.w_bf16`: an int8 trunk whose
     * down projection quantises its own input (LinearFP8::i8's op_q) needs only the bf16 product,
     * and `gated_quant_fp8` cannot supply it alone (its code and scale outputs are mandatory). */
    int  declare(RadBuilder* b, const Geom& g, rad_buf gate_up, const ActFP8& act, int64_t n,
                 const char* act_name = "silu", int64_t max_rows = 0, bool bf16_out = true,
                 bool rot = false, bool silu_only = false);
    void step(RadCtx* c, rad_buf gate_up, int64_t T, int64_t r0 = 0, int64_t rows = 0) const;
};

/* `rot` selects a different op (`gated_had_quant_fp8`), whose rotated codes only a weight stored
 * with the matching rotation can read; used for the routed MoE down projection at four bits. The
 * bf16 output is unrotated either way. */
inline int GateQuantFP8::declare(RadBuilder* b, const Geom& g, rad_buf gate_up, const ActFP8& act,
                                 int64_t n, const char* act_name, int64_t max_rows,
                                 bool bf16_out, bool rot, bool silu_only) {
    a = act;
    want_x = bf16_out;
    this->n = n;
    /* bf16 linears read the product itself, and an int8 trunk's down projection quantises its own
     * input (`silu_only`): `silu_mul` into the bf16 plane only, and no rotation. */
    bf16 = g.w_bf16 || silu_only;
    if (bf16) {
        if (!gate_up || !a.x || n <= 0 || rot || std::strcmp(act_name, "silu") != 0) {
            fprintf(stderr, "radiance: a bf16 gate is silu over [gate|up] into a bf16 plane; "
                            "asked for '%s'%s\n", act_name, rot ? " rotated" : "");
            return RAD_E_INVAL;
        }
        want_x = true;
        op = rw(b, RAD_OP(b, "silu_mul",
                   RAD_PARAMS(RAD_RANGE("M", 1, max_rows > 0 ? max_rows : g.max_tok),
                              RAD_INT("n", n), RAD_STR("dtype", g.dtype)),
                   RAD_NOWEIGHTS),
               { gate_up }, { a.x });
        return op ? RAD_OK : RAD_E_INVAL;
    }
    if (!gate_up || !a.x || !a.cq() || !a.cs() || n <= 0) return RAD_E_INVAL;
    /* Rotated codes feed a four-bit expert, which reads E4M3 only. */
    if (rot && a.q8_fed) return RAD_E_INVAL;
    const int64_t rows = max_rows > 0 ? max_rows : g.max_tok;
    if (!rot || rad_op_declared(b, "gated_had_quant_fp8"))
        op = RAD_OP(b, rot ? "gated_had_quant_fp8" : "gated_quant_fp8",
                    RAD_PARAMS(RAD_RANGE("M", 1, rows), RAD_INT("n", n),
                               RAD_INT("group", RAD_FP8_BLOCK), RAD_STR("dtype", g.dtype),
                               RAD_STR("act", act_name)),
                    RAD_NOWEIGHTS);
    /* The rotated op's chain, which writes the bf16 plane whether or not a consumer asked. */
    if (!op && rot && std::strcmp(act_name, "silu") == 0) {
        want_x = true;
        op = rw(b, RAD_OP(b, "silu_mul",
                   RAD_PARAMS(RAD_RANGE("M", 1, rows), RAD_INT("n", n), RAD_STR("dtype", g.dtype)),
                   RAD_NOWEIGHTS),
               { gate_up }, { a.x });
        op_q = rw(b, RAD_OP(b, "had_quant_act_fp8",
                     RAD_PARAMS(RAD_RANGE("M", 1, rows), RAD_INT("n", n),
                                RAD_INT("group", RAD_FP8_BLOCK), RAD_STR("dtype", g.dtype)),
                     RAD_NOWEIGHTS),
                 { a.x }, { a.cq(), a.cs() });
        return op && op_q ? RAD_OK : RAD_E_INVAL;
    }
    if (!op) return RAD_E_INVAL;
    const rad_buf wr[3] = { a.cq(), a.cs(), a.x };
    rad_op_reads (b, op, &gate_up, 1);
    rad_op_writes(b, op, wr, want_x ? 3 : 2);
    return RAD_OK;
}

inline void GateQuantFP8::step(RadCtx* c, rad_buf gate_up, int64_t T, int64_t r0,
                               int64_t rows) const {
    if (rows <= 0) rows = T - r0;
    if (bf16 || op_q) {
        RAD_ISSUE_N(c, op, rows, brow_slice(gate_up, r0, rows, 2 * n), brow_slice(a.x, r0, rows, n));
        if (op_q)
            RAD_ISSUE_N(c, op_q, rows, brow_slice(a.x, r0, rows, n),
                        brow_slice(a.cq(), r0, rows, n),
                        brow_slice(a.cs(), r0, rows, n / RAD_FP8_BLOCK));
        return;
    }
    RAD_ISSUE_N(c, op, rows, brow_slice(gate_up, r0, rows, 2 * n),
                brow_slice(a.cq(), r0, rows, n),
                brow_slice(a.cs(), r0, rows, n / RAD_FP8_BLOCK),
                want_x ? brow_slice(a.x, r0, rows, n) : RAD_NONE);
}

/* A linear's stacked row parts (RadWeightDecl::row_parts) in weight rows (`w`) and in the rows of
 * its scale plane (`s`, `div` times fewer). */
struct RowParts {
    int64_t w[RAD_MAX_ROW_PARTS] = {0}, s[RAD_MAX_ROW_PARTS] = {0};
    int     n = 0;

    /* Refuses more than RAD_MAX_ROW_PARTS parts, or a part that is not whole `unit`-row tiles
     * (`tile` names them), since a row shard cuts the weight at part boundaries. */
    int set(std::initializer_list<int64_t> parts, int64_t unit, int64_t div, const char* base,
            const char* tile) {
        n = 0;
        for (int64_t p : parts) {
            if (n >= RAD_MAX_ROW_PARTS) return RAD_E_INVAL;
            if (p % unit) {
                fprintf(stderr, "radiance: '%s' stacks a %lld-row part, which is not a whole "
                                "number of %lld-row %s; it cannot be row-sharded\n",
                        base, (long long)p, (long long)unit, tile);
                return RAD_E_INVAL;
            }
            w[n] = p;
            s[n] = p / div;
            ++n;
        }
        return RAD_OK;
    }
};

/* ================================================================== one block-scaled linear */
struct LinearFP8 {
    rad_weight w = 0, ws = 0;
    rad_op     op = 0;
    int64_t    N = 0, K = 0;
    /* Weight rows per scale-plane row: RAD_FP8_BLOCK for block fp8, 1 otherwise. Row spans on `ws`
     * are converted with it (declare, span_parts). */
    int64_t    ws_row_div = 1;
    /* Plain bf16 weight read by `gemm_nt` over the bf16 activation (Geom::w_bf16): no scale plane,
     * no quantiser. */
    bool       bf16 = false;
    /* Int8 weight read by `gemm_nt_q` at dtype i8a8: i8 codes with a bf16 scale per row per 128 of
     * K (libquant `codes=i8 group=128 scale=bf16`), against an i8 activation with an f32 scale per
     * (row, 128). A quarter of E4M3's error at the same bytes, and iu8 WMMA runs at the fp8 rate.
     * `op_q` (quant_act_i8g) writes the activation codes from the bf16 plane when the producer
     * writes E4M3. Read from the weight's encoding. */
    bool       i8 = false;
    rad_op     op_q = 0;
    /* OCP-MXFP4 weight read by `gemm_nt_q` at dtype mxfp4a8 against the usual E4M3 activation pair:
     * e2m1 codes with one E8M0 exponent per 32 of K (libquant `codes=fp4_e2m1 group=32
     * scale=e8m0`). libr4d folds each block's exponent into the e4m3 operand against a per-row
     * reference, so the scale plane is passed twice, as `b_scale` and as `b_ref` (`wref`, a second
     * view whose layout hook reduces it). Read from the weight's encoding. */
    bool       mx = false;
    rad_weight wref = 0;
    /* Rotated (ParoQuant) variant of `mx`: the same GEMM over an activation that
     * `givens_dense_quant` makes from the bf16 plane (channel scale, dense per-group rotation and
     * E4M3 quantiser in one pass), once per independently rotated part of a fused projection
     * (gate|up, qkv|z), into this linear's own code and scale planes side by side. The tables are
     * views of the weight's planes. The GEMM takes the parts' row bounds as `pb1`/`pb2` and meets
     * copy p with part p's rows. */
    bool       paro = false;
    /* The 5-bit sibling: the same rotation to f16 (`rot_q` is the f16 plane, no `rot_s`), then
     * gemm_nt_q at w5a16 over u5 codes with an f16 scale and a zero point. */
    bool       paro5 = false;
    /* A 4-bit weight with a zero point and no rotation (linear_is_w4): the int4 W4A16 releases.
     * At or below `w4_split` rows `op_a16` casts the input's bf16 plane to f16 (`a16`) for the
     * w4a16_asym GEMM `op`; above it `op_q` quantises it to int8 with a scale per (row, 128)
     * (`a8q`, `a8s`) for the w4a8i8_asym GEMM `op2`. */
    bool       w4 = false;
    int64_t    w4_split = 0;
    rad_buf    a16 = 0, a8q = 0, a8s = 0;
    rad_op     op_a16 = 0;
    /* Prefill band for z-lab's int4 and the int5: above `rot_split` rows the f16 skinny kernel is
     * bound by the slow f16 matrix instruction, so a second chain runs: the same rotation to int8
     * with a scale per (row, 128) into `rot_q2`/`rot_s2`, then the int8 tiled GEMM (`w4a8i8_asym` /
     * `w5a8i8`) over the same weight views. At or below the split the f16 chain runs. */
    int64_t    rot_split = 0;
    rad_buf    rot_q2 = 0, rot_s2 = 0;
    rad_op     op_rot2 = 0, op2 = 0;
    rad_weight rot_pairs = 0, rot_angle = 0, rot_scale = 0, rot_dense = 0;
    rad_buf    rot_q = 0, rot_s = 0;
    rad_op     op_rot = 0;
    int64_t    rot_parts = 1, rot_stages = 0;
    /* probe() found an int8 weight; lets a model learn its trunk's format from one linear before
     * declaring activation buffers (ActFP8::declare_qs's `i8`). */
    bool       probed_i8 = false;
    /* Whether this linear reads the input's bf16 plane, known from probe() before the producer is
     * declared. A producer writes the bf16 plane only when asked, so the block must ask for it when
     * this is set; otherwise the linear reads unwritten arena bytes. */
    bool       reads_x = false;
    bool       mapped  = false;   /* probe() made the name map; declare() does not make it again */
    /* The weight's access class (rad_builder.h): read every pass by default; a linear run once a
     * request (an encoder's) says so, which lets placement keep it off the card. Set before
     * declare(). */
    int        access  = RAD_ACCESS_PER_TOKEN;

    /* Declares the name map and reads the weight's format (reads_x, probed_i8). `base` and `src`
     * are the same as for declare(). */
    int  probe(RadBuilder* b, Names& nm, const Geom& g, const char* base,
               std::initializer_list<const char*> src);

    /* `base` is the declared name without suffix ("blk.7.ffn_gate_up"); the logical weight is
     * `<base>.weight` with views `.weight` (codes) and `.scale`. `src` is one checkpoint name, or
     * several concatenated along dim 0; block-fp8 scale planes concatenate with them, so each part
     * must be a whole number of 128-row blocks ([17408,5120] x 2 -> [34816,5120] with scales
     * [136,40] x 2 -> [272,40]).
     *
     * `span_lo`/`span_hi` (when hi is set): this rank's uneven slice of the split dimension in
     * model units (rows of each row part for RAD_SHARD_ROW, columns for RAD_SHARD_COL), whole scale
     * blocks; n_out / n_in is already its width (rad_weight_shard_span).
     *
     * `rot_part_rows`: rows of each independently rotated part of a rotated weight. Equal to
     * `row_parts` when the cuts coincide (gate|up), not otherwise (the delta net's qkv|z has four
     * shard parts and two rotated ones). */
    int  declare(RadBuilder* b, Names& nm, const Geom& g, const char* base,
                 int64_t n_out, int64_t n_in, int shard, RadWeightGroup grp,
                 std::initializer_list<const char*> src,
                 const ActFP8& a, rad_buf y, int64_t max_rows = 0,
                 std::initializer_list<int64_t> row_parts = {},
                 int64_t span_lo = 0, int64_t span_hi = 0,
                 std::initializer_list<int64_t> rot_part_rows = {});
    /* Declares a rotated weight's rotation: parts and stages (`rot_parts`, `rot_stages`) and the
     * channel scale as a view of the weight's plane. `cut` is each rotated part's rows; the GEMM's
     * part bounds are returned in pb1 and pb2. */
    int  declare_rotation(RadBuilder* b, Names& nm, const char* base, const char* wn,
                          const RadEncoding& enc, int64_t N, int64_t K, RadWeightGroup grp,
                          const std::vector<int64_t>& cut, int64_t* pb1, int64_t* pb2);
    /* The sequential rotation network (cheap at decode's few rows): pairs and angles as views, and
     * givens_rotate over them to f16 for M in [lo, hi]. */
    int  decl_seq_tables(RadBuilder* b, Names& nm, const char* base, const char* wn, RadWeightGroup grp);
    rad_op decl_rot_op(RadBuilder* b, const Geom& g, const ActFP8& a, int64_t lo, int64_t hi,
                       std::initializer_list<rad_buf> outs) const;
    /* The dense rotation (rot_dense, decl_dense): givens_dense_quant to `out` ("fp8" or "i8") codes
     * and a scale, for M in [lo, hi]. */
    rad_op decl_dense_op(RadBuilder* b, const Geom& g, const ActFP8& a, const char* out, int64_t lo,
                         int64_t hi, std::initializer_list<rad_buf> outs) const;
    /* gemm_nt_q over this weight, M in [lo, hi]. `pb1`/`pb2` are passed only for rotated parts,
     * so a one-part op keeps the plain geometry (the tune cache key). */
    rad_op decl_gemm_q(RadBuilder* b, int64_t lo, int64_t hi, int64_t group, const char* dtype,
                       int64_t pb1, int64_t pb2, std::initializer_list<rad_weight> wts,
                       std::initializer_list<rad_buf> reads, rad_buf y) const;
    rad_weight decl_dense(RadBuilder* b, Names& nm, const char* base, const char* wn, const RadEncoding& enc,
                          int64_t K,
                          RadWeightGroup grp);
    void step(RadCtx* c, const ActFP8& a, rad_buf y, int64_t T, int64_t r0 = 0,
              int64_t rows = 0) const;
    /* A span per stacked row part (rad_weight_shard_span_parts), in weight rows; the scale plane
     * gets the same spans in its rows. Call after declare(). With a 1/128 plane every span and
     * total must be whole scale blocks. */
    int  span_parts(RadBuilder* b, const int64_t* lo, const int64_t* hi, const int64_t* total,
                    int n) const {
        RAD_ARCH_TRY(rad_weight_shard_span_parts(b, w, lo, hi, total, n));
        if (!ws) return RAD_OK;
        if (ws_row_div == 1) return rad_weight_shard_span_parts(b, ws, lo, hi, total, n);
        int64_t slo[RAD_MAX_ROW_PARTS], shi[RAD_MAX_ROW_PARTS], stot[RAD_MAX_ROW_PARTS];
        for (int g = 0; g < n && g < RAD_MAX_ROW_PARTS; ++g) {
            if (lo[g] % ws_row_div || hi[g] % ws_row_div || total[g] % ws_row_div) {
                fprintf(stderr, "radiance: a part span [%lld, %lld) of %lld rows is not whole "
                                "%lld-row scale blocks\n", (long long)lo[g], (long long)hi[g],
                        (long long)total[g], (long long)ws_row_div);
                return RAD_E_INVAL;
            }
            slo[g] = lo[g] / ws_row_div;
            shi[g] = hi[g] / ws_row_div;
            stot[g] = total[g] / ws_row_div;
        }
        return rad_weight_shard_span_parts(b, ws, slo, shi, stot, n);
    }
};

/* A linear's name map, declared before its weight because the weight's encoding is looked up
 * through it (rad_weight_encoding). One source is copied; several are concatenated along dim 0
 * (with block-fp8 scale planes, so each part must be whole 128-row blocks). */
inline int map_linear(RadBuilder* b, const char* wn, std::initializer_list<const char*> src) {
    if (src.size() == 1) return map_copy(b, wn, *src.begin());
    RadNameMap mw = map_base(wn, RAD_MAP_CONCAT);
    int i = 0;
    for (const char* q : src) {
        if (i >= 8) return RAD_E_INVAL;
        mw.src[i++] = q;
    }
    mw.n_src = i;
    mw.concat_dim = 0;
    return rad_decl_name_map(b, &mw);
}

/* The int8 weight LinearFP8 serves at i8a8: i8 codes, a bf16 scale per [1 x 128], no transform or
 * other plane (libr4d's i8 block layout hook). */
inline bool linear_is_w8(const RadEncoding& e) {
    if (!rad_enc_is(&e, "affine") || e.n_planes != 2 || e.transform[0]) return false;
    const RadEncPlane* sc = rad_enc_plane(&e, "scale");
    return e.plane[0].dtype == RAD_I8 && e.plane[0].block[1] == 1 && sc && sc->dtype == RAD_BF16 &&
           sc->block[0] == 1 && sc->block[1] == RAD_FP8_BLOCK;
}

/* The OCP-MXFP4 weight LinearFP8 serves at mxfp4a8: e2m1 codes and an E8M0 exponent per [1 x 32],
 * no transform (libr4d's MXFP4 layout hook). */
enum { RAD_MX_GROUP = 32 };
inline bool linear_is_mxfp4(const RadEncoding& e) {
    if (!rad_enc_is(&e, "affine") || e.n_planes != 2 || e.transform[0]) return false;
    const RadEncPlane* sc = rad_enc_plane(&e, "scale");
    return e.plane[0].dtype == RAD_FP4E2M1 && e.plane[0].block[1] == 1 && sc &&
           sc->dtype == RAD_E8M0 && sc->block[0] == 1 && sc->block[1] == RAD_MX_GROUP;
}

/* A rotated weight (ParoQuant): an affine encoding in the basis of a "givens" transform, with the
 * rotation tables as planes (int32 column pairs, f32 angles, f32 channel scales; one set per part
 * of a fused projection). The GEMM reads the codes as stored; the activation is rotated first. */
inline bool linear_is_givens(const RadEncoding& e) {
    if (!rad_enc_is(&e, "affine") || std::strncmp(e.transform, "givens", RAD_ENC_STR) != 0) return false;
    if (!rad_enc_valid(&e)) return false;
    const RadEncPlane *pr = rad_enc_plane(&e, "t.pairs"), *an = rad_enc_plane(&e, "t.angle"),
                      *rs = rad_enc_plane(&e, "t.scale");
    return pr && an && rs && pr->dtype == RAD_I32 && an->dtype == RAD_F32 && rs->dtype == RAD_F32;
}

/* A rotated weight with OCP-MXFP4 codes and exponents, read like the plain MXFP4 weight. */
inline bool linear_is_paro_mxfp4(const RadEncoding& e) {
    const RadEncPlane* sc = rad_enc_plane(&e, "scale");
    return linear_is_givens(e) && e.plane[0].dtype == RAD_FP4E2M1 && e.plane[0].block[1] == 1 && sc &&
           sc->dtype == RAD_E8M0 && sc->block[0] == 1 && sc->block[1] == RAD_MX_GROUP;
}

/* A 4-bit weight with a zero point and no transform: u4 codes, an f16 scale and a u4 or u8 zero
 * point per 128 of K (libquant `codes=u4 zero=u4 group=128 scale=f16`, or rad-convert's import rule
 * over an int4 release). Read by gemm_nt_q at w4a16_asym over an f16 activation, or w4a8i8_asym
 * over an int8 one, the rotated int4's GEMMs with no rotation. */
inline bool linear_is_w4(const RadEncoding& e) {
    if (!rad_enc_is(&e, "affine") || e.n_planes != 3 || e.transform[0]) return false;
    const RadEncPlane *sc = rad_enc_plane(&e, "scale"), *zp = rad_enc_plane(&e, "zero");
    return e.plane[0].dtype == RAD_U4 && e.plane[0].block[1] == 1 && sc && zp &&
           sc->dtype == RAD_F16 && sc->block[0] == 1 && sc->block[1] == RAD_FP8_BLOCK &&
           (zp->dtype == RAD_U4 || zp->dtype == RAD_U8) && zp->block[0] == 1 &&
           zp->block[1] == RAD_FP8_BLOCK;
}

/* A rotated weight with ParoQuant int5 (u5) or z-lab int4 (u4) asymmetric codes: an f16 scale and
 * u8 zero point per 128 of K. Read by gemm_nt_q at w5a16 / w4a16_asym over an f16 activation, or
 * w5a8i8 / w4a8i8_asym over an int8 one. */
inline bool linear_is_paro_w5(const RadEncoding& e) {
    const RadEncPlane *sc = rad_enc_plane(&e, "scale"), *zp = rad_enc_plane(&e, "zero");
    return linear_is_givens(e) && (e.plane[0].dtype == RAD_U5 || e.plane[0].dtype == RAD_U4) && sc &&
           zp && sc->dtype == RAD_F16 && sc->block[0] == 1 && sc->block[1] == RAD_FP8_BLOCK &&
           zp->dtype == RAD_U8 && zp->block[0] == 1 && zp->block[1] == RAD_FP8_BLOCK;
}

inline int LinearFP8::probe(RadBuilder* b, Names& nm, const Geom& g, const char* base,
                            std::initializer_list<const char*> src) {
    if (src.size() == 0) return RAD_E_INVAL;
    const char* wn = nm.f("%s.weight", base);
    RAD_ARCH_TRY(map_linear(b, wn, src));
    mapped = true;
    RadEncoding enc{};
    const bool known = weight_enc(b, wn, &enc);
    const bool plain = known && rad_enc_is(&enc, "plain") && enc.n_planes == 1 &&
                       enc.plane[0].dtype == RAD_BF16;
    probed_i8 = known && linear_is_w8(enc);
    /* A rotated weight rotates and quantises the bf16 plane itself; a 4-bit one casts or
     * quantises it. */
    const bool rotated = known && (linear_is_paro_mxfp4(enc) || linear_is_paro_w5(enc));
    reads_x = g.w_bf16 || plain || probed_i8 || rotated || (known && linear_is_w4(enc));
    return RAD_OK;
}

inline rad_op LinearFP8::decl_gemm_q(RadBuilder* b, int64_t lo, int64_t hi, int64_t group,
                                     const char* dtype, int64_t pb1, int64_t pb2,
                                     std::initializer_list<rad_weight> wts,
                                     std::initializer_list<rad_buf> reads, rad_buf y) const {
    const RadParam ps[] = { RAD_RANGE("M", lo, hi), RAD_INT("N", N), RAD_INT("K", K),
                            RAD_INT("group", group), RAD_STR("dtype", dtype),
                            RAD_INT("pb1", pb1), RAD_INT("pb2", pb2) };
    const int n_ps = (int)(sizeof ps / sizeof ps[0]) - (rot_parts > 1 ? 0 : 2);
    return rw(b, rad_decl_op(b, "gemm_nt_q", ps, n_ps, wts.begin(), (int)wts.size()), reads, { y });
}

inline int LinearFP8::declare_rotation(RadBuilder* b, Names& nm, const char* base, const char* wn,
                                       const RadEncoding& enc, int64_t N, int64_t K,
                                       RadWeightGroup grp, const std::vector<int64_t>& cut_in,
                                       int64_t* pb1, int64_t* pb2) {
    /* The container records the number of rotated parts but not their row bounds, so the block
     * supplies them (`rot_part_rows`, or its row parts when those are the cut). */
    const int64_t P = rad_enc_givens_parts(&enc);
    const RadEncPlane* tp = rad_enc_plane(&enc, "t.pairs");
    if (P < 1 || !tp || tp->extent[0] % P || tp->extent[1] != K) {
        fprintf(stderr, "radiance: '%s' has rotation tables that do not fit %lld columns\n",
                wn, (long long)K);
        return RAD_E_SHAPE;
    }
    const int64_t S = tp->extent[0] / P;
    std::vector<int64_t> cut = cut_in;
    if (cut.empty() && P == 1) cut.push_back(N);
    int64_t sum = 0;
    for (int64_t p : cut) sum += p;
    if ((int64_t)cut.size() != P || sum != N || P > 3) {
        fprintf(stderr, "radiance: '%s' was rotated as %lld part(s) and is declared as %zu "
                        "rotated part(s) of %lld rows; the GEMM meets up to three copies "
                        "of the activation\n", wn, (long long)P, cut.size(), (long long)sum);
        return RAD_E_SHAPE;
    }
    for (int64_t i = 0; i + 1 < P; ++i) if (cut[(size_t)i] % 128) {
        fprintf(stderr, "radiance: '%s' rotation part of %lld rows is not whole 128s\n",
                wn, (long long)cut[(size_t)i]);
        return RAD_E_SHAPE;
    }
    rot_parts = P;
    rot_stages = S;
    *pb1 = P > 1 ? cut[0] : 0;
    *pb2 = P > 2 ? cut[0] + cut[1] : 0;
    rot_scale = decl_view(b, nm.f("%s.rot_scale", base), wn, "t.scale", RAD_F32,
                          { P, K }, RAD_ACCESS_PER_TOKEN, RAD_SHARD_NONE, grp);
    return rot_scale ? RAD_OK : RAD_E_INVAL;
}

inline int LinearFP8::decl_seq_tables(RadBuilder* b, Names& nm, const char* base, const char* wn,
                                      RadWeightGroup grp) {
    const int64_t PS = rot_parts * rot_stages;
    rot_pairs = decl_view(b, nm.f("%s.rot_pairs", base), wn, "t.pairs", RAD_I32,
                          { PS, K }, RAD_ACCESS_PER_TOKEN, RAD_SHARD_NONE, grp);
    rot_angle = decl_view(b, nm.f("%s.rot_angle", base), wn, "t.angle", RAD_F32,
                          { PS, K / 2 }, RAD_ACCESS_PER_TOKEN, RAD_SHARD_NONE, grp);
    return rot_pairs && rot_angle ? RAD_OK : RAD_E_INVAL;
}

inline rad_op LinearFP8::decl_rot_op(RadBuilder* b, const Geom& g, const ActFP8& a, int64_t lo,
                                     int64_t hi, std::initializer_list<rad_buf> outs) const {
    return rw(b, RAD_OP(b, "givens_rotate",
                  RAD_PARAMS(RAD_RANGE("M", lo, hi), RAD_INT("n", K), RAD_INT("group", RAD_FP8_BLOCK),
                             RAD_STR("dtype", g.dtype), RAD_INT("stages", rot_stages),
                             RAD_INT("parts", rot_parts)),
                  RAD_WEIGHTS(rot_pairs, rot_angle, rot_scale)),
              { a.x }, outs);
}

inline rad_op LinearFP8::decl_dense_op(RadBuilder* b, const Geom& g, const ActFP8& a, const char* out,
                                       int64_t lo, int64_t hi, std::initializer_list<rad_buf> outs) const {
    return rw(b, RAD_OP(b, "givens_dense_quant",
                  RAD_PARAMS(RAD_RANGE("M", lo, hi), RAD_INT("n", K), RAD_INT("group", RAD_FP8_BLOCK),
                             RAD_STR("dtype", g.dtype), RAD_INT("parts", rot_parts),
                             RAD_STR("out", out)),
                  RAD_WEIGHTS(rot_dense, rot_scale)),
              { a.x }, outs);
}

/* The dense rotation matrices of a rotated weight: a view of its "t.dense" table, f16
 * [P * K/128, 128*128], which rad-convert composes from the givens tables (abi/rad_encoding.h). A
 * container with only pairs and angles is refused with a message to reconvert. */
inline rad_weight LinearFP8::decl_dense(RadBuilder* b, Names& nm, const char* base, const char* wn,
                                        const RadEncoding& enc, int64_t K, RadWeightGroup grp) {
    if (!rad_enc_plane(&enc, "t.dense")) {
        fprintf(stderr, "radiance: '%s' carries its rotation as pairs and angles only and the dense "
                        "quantiser reads a composed \"t.dense\" table; convert it again (--reuse "
                        "<this container> copies every other weight)\n", wn);
        return 0;
    }
    return decl_view(b, nm.f("%s.rot_dense", base), wn, "t.dense", RAD_F16,
                     { rot_parts * (K / RAD_FP8_BLOCK), RAD_FP8_BLOCK * RAD_FP8_BLOCK },
                     RAD_ACCESS_PER_TOKEN, RAD_SHARD_NONE, grp);
}

inline int LinearFP8::declare(RadBuilder* b, Names& nm, const Geom& g, const char* base,
                              int64_t n_out, int64_t n_in, int shard, RadWeightGroup grp,
                              std::initializer_list<const char*> src,
                              const ActFP8& a, rad_buf y, int64_t max_rows,
                              std::initializer_list<int64_t> row_parts,
                              int64_t span_lo, int64_t span_hi,
                              std::initializer_list<int64_t> rot_part_rows) {
    N = n_out;
    K = n_in;
    if (N <= 0 || K <= 0 || src.size() == 0) return RAD_E_INVAL;

    /* Name map first: the weight's encoding, looked up through it, picks the branch below. */
    const char* wn = nm.f("%s.weight", base);
    if (!mapped) RAD_ARCH_TRY(map_linear(b, wn, src));
    RadEncoding enc{};
    const bool known = weight_enc(b, wn, &enc);
    const bool plain = known && rad_enc_is(&enc, "plain") && enc.n_planes == 1 &&
                       enc.plane[0].dtype == RAD_BF16;

    /* ---- bf16 weight, `gemm_nt` over `a.x`: every linear of a bf16 model, or one a quantised
     * container keeps at checkpoint precision (e.g. the QSA indexer's projection, whose top-k is a
     * hard choice; 20 MB to keep exact). Row parts and uneven slices apply without block
     * constraints. */
    bf16 = g.w_bf16 || plain;
    if (bf16) {
        if (known && !plain) {
            char en[256];
            rad_enc_format(&enc, en, sizeof en);
            fprintf(stderr, "radiance: '%s' is %s and this model's linears are plain bf16; a "
                            "bf16 container is served bf16 throughout\n", wn, en);
            return RAD_E_DTYPE;
        }
        /* A bf16 weight needs the input's bf16 plane, which a quantised model's producer may not
         * write (a gated quantiser writes only codes). */
        if (!a.x) {
            fprintf(stderr, "radiance: '%s' is plain bf16 and its input has no bf16 plane in this "
                            "model; quantise it with its neighbours\n", wn);
            return RAD_E_DTYPE;
        }
        RowParts rp;
        RAD_ARCH_TRY(rp.set(row_parts, 1, 1, base, "rows"));
        w = decl_w_n(b, wn, RAD_BF16, { N, K }, access, shard, grp, 0, rp.w, rp.n);
        if (!w) return RAD_E_INVAL;
        if (span_hi > 0) RAD_ARCH_TRY(rad_weight_shard_span(b, w, span_lo, span_hi));
        op = rw(b, RAD_OP(b, "gemm_nt",
                   RAD_PARAMS(RAD_RANGE("M", 1, max_rows > 0 ? max_rows : g.max_tok),
                              RAD_INT("N", N), RAD_INT("K", K), RAD_STR("dtype", g.dtype)),
                   RAD_WEIGHTS(w)),
               { a.x }, { y });
        return op ? RAD_OK : RAD_E_INVAL;
    }

    /* ---- int8 weight, `gemm_nt_q` at i8a8 over the input's `a.q8` / `a.s8`. The scale plane is
     * one bf16 per row per 128 of K: row parts and row slices apply as is, column slices in groups
     * of 128. The codes come from the producer (ActFP8::q8_fed) or from op_q over `a.x` (probe()
     * set reads_x). The model must have declared the pair (ActFP8::declare_qs's `i8`). */
    i8 = known && linear_is_w8(enc);
    if (i8) {
        /* The codes come from the producer (`q8_fed`, which declares no `op_q` and so needs no bf16
         * plane) or from this linear's own `op_q` over `a.x`. Either way the int8 pair must exist;
         * the bf16 plane is required only when this linear quantises, exactly as the fp8 branch
         * needs `a.q`/`a.s` without demanding `a.x`. */
        if (!a.q8 || !a.s8 || (!a.q8_fed && !a.x)) {
            fprintf(stderr, "radiance: '%s' is an int8 weight and its input lacks the int8 code "
                            "pair%s (the model declared its input before learning the trunk is "
                            "int8)\n", wn, a.q8_fed ? "" : " or the bf16 plane it quantises");
            return RAD_E_DTYPE;
        }
        if (N % 16 || K % RAD_FP8_BLOCK) {
            fprintf(stderr, "radiance: '%s' is %lldx%lld per rank; an int8 weight takes rows in "
                            "16s and K in whole %d-column scale groups\n",
                    base, (long long)N, (long long)K, RAD_FP8_BLOCK);
            return RAD_E_INVAL;
        }
        RowParts rp;
        RAD_ARCH_TRY(rp.set(row_parts, 1, 1, base, "rows"));
        const char* sn = nm.f("%s.scale", base);
        w  = decl_view(b, wn, nullptr, "codes", RAD_I8, { N, K }, access, shard, grp,
                       rp.w, rp.n);
        ws = decl_view(b, sn, wn, "scale", RAD_BF16, { N, K / RAD_FP8_BLOCK }, access,
                       shard, grp, rp.w, rp.n);
        if (!w || !ws) return RAD_E_INVAL;
        if (span_hi > 0) {
            const bool col = shard == RAD_SHARD_COL;
            if (col && (span_lo % RAD_FP8_BLOCK || span_hi % RAD_FP8_BLOCK)) {
                fprintf(stderr, "radiance: '%s' is sliced [%lld, %lld) on this rank, which cuts a "
                                "%d-column scale group\n", base, (long long)span_lo,
                        (long long)span_hi, RAD_FP8_BLOCK);
                return RAD_E_INVAL;
            }
            RAD_ARCH_TRY(rad_weight_shard_span(b, w, span_lo, span_hi));
            RAD_ARCH_TRY(rad_weight_shard_span(b, ws, col ? span_lo / RAD_FP8_BLOCK : span_lo,
                                               col ? span_hi / RAD_FP8_BLOCK : span_hi));
        }
        const int64_t hi = max_rows > 0 ? max_rows : g.max_tok;
        if (!a.q8_fed)
            op_q = rw(b, RAD_OP(b, "quant_act_i8g",
                     RAD_PARAMS(RAD_RANGE("M", 1, hi), RAD_INT("n", K),
                                RAD_INT("group", RAD_FP8_BLOCK), RAD_STR("dtype", "bf16")),
                     RAD_NOWEIGHTS),
                  { a.x }, { a.q8, a.s8 });
        op = rw(b, RAD_OP(b, "gemm_nt_q",
                   RAD_PARAMS(RAD_RANGE("M", 1, hi), RAD_INT("N", N), RAD_INT("K", K),
                              RAD_INT("group", RAD_FP8_BLOCK), RAD_STR("dtype", "i8a8")),
                   RAD_WEIGHTS(w, ws)),
               { a.q8, a.s8 }, { y });
        return (op_q || a.q8_fed) && op ? RAD_OK : RAD_E_INVAL;
    }

    /* ---- Rotated 5-bit or asymmetric 4-bit weight (ParoQuant int5, z-lab int4): the input's bf16
     * plane rotated into this linear's own f16 plane, then the w5a16 / w4a16_asym GEMM. Two views:
     * codes, and (scale, zero) as `b_scale`. */
    paro5 = known && linear_is_paro_w5(enc);
    if (paro5) {
        const int bits = enc.plane[0].dtype == RAD_U4 ? 4 : 5;
        if (!a.x) {
            fprintf(stderr, "radiance: '%s' is a rotated %d-bit weight, rotated from its input's "
                            "bf16 plane, and its input has none\n", wn, bits);
            return RAD_E_DTYPE;
        }
        if (g.world > 1 || span_hi > 0) {
            fprintf(stderr, "radiance: '%s' is a rotated %d-bit weight; its rotation is not cut by "
                            "a tensor-parallel shard (world %d)\n", wn, bits, g.world);
            return RAD_E_UNSUPPORTED;
        }
        if (N % 16 || K % RAD_FP8_BLOCK) {
            fprintf(stderr, "radiance: '%s' is %lldx%lld; a %d-bit weight takes rows in 16s and K "
                            "in whole 128s\n", base, (long long)N, (long long)K, bits);
            return RAD_E_INVAL;
        }
        std::vector<int64_t> cut;
        for (int64_t p : rot_part_rows) cut.push_back(p);
        if (cut.empty()) for (int64_t p : row_parts) cut.push_back(p);
        int64_t pb1 = 0, pb2 = 0;
        const int64_t hi = max_rows > 0 ? max_rows : g.max_tok;
        const bool bits5 = enc.plane[0].dtype == RAD_U5;
        const int64_t kSplit = 64;                      /* the skinny kernel's M ceiling */
        const bool band = hi > kSplit;
        const int64_t f16_hi = band ? kSplit : hi;      /* the f16 chain's rows */
        RAD_ARCH_TRY(declare_rotation(b, nm, base, wn, enc, N, K, grp, cut, &pb1, &pb2));
        RAD_ARCH_TRY(decl_seq_tables(b, nm, base, wn, grp));
        const int64_t P = rot_parts;
        rot_q = decl_b(b, nm.f("%s.rot_y", base), RAD_F16, { f16_hi, P * K });
        if (!rot_q) return RAD_E_INVAL;
        op_rot = decl_rot_op(b, g, a, 1, f16_hi, { rot_q });
        if (!op_rot) return RAD_E_INVAL;
        w  = decl_view(b, wn, nullptr, "codes", enc.plane[0].dtype, { N, K }, access,
                       shard, grp);
        ws = decl_view(b, nm.f("%s.scale", base), wn, "scale,zero", RAD_F16, { N, K },
                       access, shard, grp);
        if (!w || !ws) return RAD_E_INVAL;
        op = decl_gemm_q(b, 1, f16_hi, RAD_FP8_BLOCK, bits5 ? "w5a16" : "w4a16_asym",
                         pb1, pb2, { w, ws }, { rot_q }, y);
        if (!op) return RAD_E_INVAL;
        if (band) {
            /* Prefill band: rows above the skinny kernel's ceiling rotate to int8 and use the int8
             * tiled GEMM, over the same tables and weight views. */
            rot_split = kSplit;
            rot_q2 = decl_b(b, nm.f("%s.rot_q2", base), RAD_I8, { hi, P * K });
            rot_s2 = decl_b(b, nm.f("%s.rot_s2", base), RAD_F32, { hi, P * (K / RAD_FP8_BLOCK) });
            if (!rot_q2 || !rot_s2) return RAD_E_INVAL;
            rot_dense = decl_dense(b, nm, base, wn, enc, K, grp);
            if (!rot_dense) return RAD_E_INVAL;
            op_rot2 = decl_dense_op(b, g, a, "i8", kSplit + 1, hi, { rot_q2, rot_s2 });
            op2 = decl_gemm_q(b, kSplit + 1, hi, RAD_FP8_BLOCK, bits5 ? "w5a8i8" : "w4a8i8_asym",
                              pb1, pb2, { w, ws }, { rot_q2, rot_s2 }, y);
            if (!op_rot2 || !op2) return RAD_E_INVAL;
        }
        return RAD_OK;
    }

    /* ---- 4-bit weight with a zero point and no rotation: the rotated int4's two GEMMs over this
     * linear's own copies of the input's bf16 plane. At or below kSplit rows (the skinny f16
     * kernel's ceiling) the plane is cast to f16 for w4a16_asym; above it, where that kernel is
     * bound by the f16 matrix instruction, it is quantised to int8 with an f32 scale per
     * (row, 128) for w4a8i8_asym. Two views: codes, and (scale, zero) as `b_scale`, both at the
     * logical extents, so row parts and spans apply to both as they are; a column slice must be
     * whole 128s, the scale group. */
    w4 = known && linear_is_w4(enc);
    if (w4) {
        if (!a.x) {
            fprintf(stderr, "radiance: '%s' is a 4-bit weight, read from its input's bf16 plane, "
                            "and its input has none\n", wn);
            return RAD_E_DTYPE;
        }
        if (N % 16 || K % RAD_FP8_BLOCK) {
            fprintf(stderr, "radiance: '%s' is %lldx%lld per rank; a 4-bit weight takes rows in "
                            "16s and K in whole %d-column groups\n",
                    base, (long long)N, (long long)K, RAD_FP8_BLOCK);
            return RAD_E_INVAL;
        }
        RowParts rp;
        RAD_ARCH_TRY(rp.set(row_parts, 1, 1, base, "rows"));
        w  = decl_view(b, wn, nullptr, "codes", RAD_U4, { N, K }, access, shard, grp, rp.w, rp.n);
        ws = decl_view(b, nm.f("%s.scale", base), wn, "scale,zero", RAD_F16, { N, K }, access,
                       shard, grp, rp.w, rp.n);
        if (!w || !ws) return RAD_E_INVAL;
        if (span_hi > 0) {
            if (shard == RAD_SHARD_COL && (span_lo % RAD_FP8_BLOCK || span_hi % RAD_FP8_BLOCK)) {
                fprintf(stderr, "radiance: '%s' is sliced [%lld, %lld) on this rank, which cuts a "
                                "%d-column scale group\n", base, (long long)span_lo,
                        (long long)span_hi, RAD_FP8_BLOCK);
                return RAD_E_INVAL;
            }
            RAD_ARCH_TRY(rad_weight_shard_span(b, w, span_lo, span_hi));
            RAD_ARCH_TRY(rad_weight_shard_span(b, ws, span_lo, span_hi));
        }
        const int64_t hi = max_rows > 0 ? max_rows : g.max_tok;
        const int64_t kSplit = 64;
        const int64_t f16_hi = hi > kSplit ? kSplit : hi;
        a16 = decl_b(b, nm.f("%s.a16", base), RAD_F16, { f16_hi, K });
        if (!a16) return RAD_E_INVAL;
        op_a16 = rw(b, RAD_OP(b, "cast",
                       RAD_PARAMS(RAD_RANGE("M", 1, f16_hi), RAD_INT("n", K),
                                  RAD_STR("from", g.dtype), RAD_STR("to", "f16")),
                       RAD_NOWEIGHTS),
                    { a.x }, { a16 });
        op = decl_gemm_q(b, 1, f16_hi, RAD_FP8_BLOCK, "w4a16_asym", 0, 0, { w, ws }, { a16 }, y);
        if (!op_a16 || !op) return RAD_E_INVAL;
        if (hi > kSplit) {
            w4_split = kSplit;
            a8q = decl_b(b, nm.f("%s.a8q", base), RAD_I8, { hi, K });
            a8s = decl_b(b, nm.f("%s.a8s", base), RAD_F32, { hi, K / RAD_FP8_BLOCK });
            if (!a8q || !a8s) return RAD_E_INVAL;
            op_q = rw(b, RAD_OP(b, "quant_act_i8g",
                         RAD_PARAMS(RAD_RANGE("M", kSplit + 1, hi), RAD_INT("n", K),
                                    RAD_INT("group", RAD_FP8_BLOCK), RAD_STR("dtype", g.dtype)),
                         RAD_NOWEIGHTS),
                      { a.x }, { a8q, a8s });
            op2 = decl_gemm_q(b, kSplit + 1, hi, RAD_FP8_BLOCK, "w4a8i8_asym", 0, 0, { w, ws },
                              { a8q, a8s }, y);
            if (!op_q || !op2) return RAD_E_INVAL;
        }
        return RAD_OK;
    }

    /* The remaining formats read the E4M3 pair, which an int8-fed input lacks (a trunk mixing
     * formats that the model's probe did not detect). */
    if (a.q8_fed) {
        fprintf(stderr, "radiance: '%s' is not an int8 weight, and its input is fed int8 codes "
                        "because this model's trunk is int8; quantise it with its neighbours\n", wn);
        return RAD_E_DTYPE;
    }

    /* ---- OCP-MXFP4 weight over the E4M3 pair. Three views: codes, exponents as `b_scale`, and
     * the same plane as `b_ref`. Row parts apply as is (the exponent grid is one row tall); column
     * slices must be whole 128s, the activation's scale block. */
    paro = known && linear_is_paro_mxfp4(enc);
    mx = known && (linear_is_mxfp4(enc) || paro);
    if (mx) {
        if (paro) {
            /* The rotation is applied to the activation, so this linear needs the input's bf16
             * plane (probe() set reads_x) and quantises it itself. */
            if (!a.x) {
                fprintf(stderr, "radiance: '%s' is a rotated MXFP4 weight, rotated and quantised "
                                "from its input's bf16 plane, and its input has none\n", wn);
                return RAD_E_DTYPE;
            }
            if (g.world > 1 || span_hi > 0) {
                fprintf(stderr, "radiance: '%s' is a rotated MXFP4 weight; its rotation is per "
                                "projection and is not cut by a tensor-parallel shard (world %d)\n",
                        wn, g.world);
                return RAD_E_UNSUPPORTED;
            }
        } else if (!a.q || !a.s) {
            fprintf(stderr, "radiance: '%s' is an MXFP4 weight read against its input's E4M3 "
                            "pair, and its input has none\n", wn);
            return RAD_E_DTYPE;
        }
        if (N % 16 || K % RAD_FP8_BLOCK) {
            fprintf(stderr, "radiance: '%s' is %lldx%lld per rank; an MXFP4 weight takes rows in "
                            "16s and K in whole %d-column activation scale blocks\n",
                    base, (long long)N, (long long)K, RAD_FP8_BLOCK);
            return RAD_E_INVAL;
        }
        RowParts rp;
        RAD_ARCH_TRY(rp.set(row_parts, 16, 1, base, "MXFP4 tiles"));
        int64_t pb1 = 0, pb2 = 0;
        if (paro) {
            std::vector<int64_t> cut;
            for (int64_t p : rot_part_rows) cut.push_back(p);
            if (cut.empty()) for (int64_t p : row_parts) cut.push_back(p);
            RAD_ARCH_TRY(declare_rotation(b, nm, base, wn, enc, N, K, grp, cut, &pb1, &pb2));
            const int64_t rows = max_rows > 0 ? max_rows : g.max_tok;
            rot_q = decl_b(b, nm.f("%s.rot_q", base), RAD_F8E4M3, { rows, rot_parts * K });
            rot_s = decl_b(b, nm.f("%s.rot_s", base), RAD_F32, { rows, rot_parts * (K / RAD_FP8_BLOCK) });
            if (!rot_q || !rot_s) return RAD_E_INVAL;
            rot_dense = decl_dense(b, nm, base, wn, enc, K, grp);
            if (!rot_dense) return RAD_E_INVAL;
            op_rot = decl_dense_op(b, g, a, "fp8", 1, rows, { rot_q, rot_s });
            if (!op_rot) return RAD_E_INVAL;
        }
        const char* sn = nm.f("%s.scale", base);
        const char* rn = nm.f("%s.ref", base);
        w    = decl_view(b, wn, nullptr, "codes", RAD_FP4E2M1, { N, K }, access,
                         shard, grp, rp.w, rp.n);
        ws   = decl_view(b, sn, wn, "scale", RAD_E8M0, { N, K / RAD_MX_GROUP },
                         access, shard, grp, rp.w, rp.n);
        wref = decl_view(b, rn, wn, "scale", RAD_E8M0, { N, K / RAD_MX_GROUP },
                         access, shard, grp, rp.w, rp.n);
        if (!w || !ws || !wref) return RAD_E_INVAL;
        if (span_hi > 0) {
            const bool col = shard == RAD_SHARD_COL;
            if (col && (span_lo % RAD_FP8_BLOCK || span_hi % RAD_FP8_BLOCK)) {
                fprintf(stderr, "radiance: '%s' is sliced [%lld, %lld) on this rank, which cuts a "
                                "%d-column activation scale block\n", base, (long long)span_lo,
                        (long long)span_hi, RAD_FP8_BLOCK);
                return RAD_E_INVAL;
            }
            const int64_t lo = col ? span_lo / RAD_MX_GROUP : span_lo;
            const int64_t hi = col ? span_hi / RAD_MX_GROUP : span_hi;
            RAD_ARCH_TRY(rad_weight_shard_span(b, w, span_lo, span_hi));
            RAD_ARCH_TRY(rad_weight_shard_span(b, ws, lo, hi));
            RAD_ARCH_TRY(rad_weight_shard_span(b, wref, lo, hi));
        }
        /* One GEMM: a rotated weight reads its own rotated pair, a plain one the producer's. */
        op = decl_gemm_q(b, 1, max_rows > 0 ? max_rows : g.max_tok, RAD_MX_GROUP, "mxfp4a8", pb1, pb2,
                         { w, ws, wref }, { paro ? rot_q : a.q, paro ? rot_s : a.s }, y);
        return op ? RAD_OK : RAD_E_INVAL;
    }

    /* Block fp8 is the last form; any other encoding is refused by name with the ones read. */
    if (known) {
        const RadEncPlane* sc = rad_enc_plane(&enc, "scale");
        if (!rad_enc_is(&enc, "affine") || enc.n_planes != 2 || enc.transform[0] ||
            enc.plane[0].dtype != RAD_F8E4M3 || !sc || sc->dtype != RAD_BF16 ||
            sc->block[0] != RAD_FP8_BLOCK || sc->block[1] != RAD_FP8_BLOCK) {
            char en[256];
            rad_enc_format(&enc, en, sizeof en);
            fprintf(stderr, "radiance: '%s' is %s, which this model's linears do not read; they "
                            "read bf16, fp8_e4m3*bf16[128x128], i8*bf16[1x128], "
                            "fp4_e2m1*e8m0[1x32], u4*f16[1x128]-u4[1x128] (or a u8 zero) and the "
                            "ParoQuant forms. A recipe rule stores the weight in one of them "
                            "(docs/TOOLS.md, the import rule for a shipped int4 grid)\n", wn, en);
            return RAD_E_DTYPE;
        }
    }

    /* Block fp8: per-rank extents must be whole scale blocks, checked here to name the weight (this
     * also catches a tensor-parallel width that cuts a block). */
    if (N % RAD_FP8_BLOCK || K % RAD_FP8_BLOCK) {
        fprintf(stderr, "radiance: '%s' is %lldx%lld per rank, which is not a whole number of "
                        "%dx%d scale blocks. A block-scaled fp8 weight cannot be split inside a "
                        "block; use a bf16 linear here or a world_size that divides cleanly\n",
                base, (long long)N, (long long)K, RAD_FP8_BLOCK, RAD_FP8_BLOCK);
        return RAD_E_INVAL;
    }

    /* The scale plane is split where the weight is, in RAD_FP8_BLOCK rows: fused [q|k|v] parts of
     * 1024/1024/3072 rows are 8/8/24 scale rows. Without this the scale would be sliced
     * contiguously while the weight was not, passing shape checks with wrong values. Parts must be
     * whole blocks. */
    RowParts rp;
    RAD_ARCH_TRY(rp.set(row_parts, RAD_FP8_BLOCK, RAD_FP8_BLOCK, base, "scale blocks"));

    const char* sn = nm.f("%s.scale",  base);
    /* One stored weight, two operands: the codes and scale plane of `<base>.weight`, block fp8 in
     * the checkpoint or made so by rad-convert's recipe. */
    w  = decl_view(b, wn, nullptr, "codes", RAD_F8E4M3, { N, K },
                   access, shard, grp, rp.w, rp.n);
    ws_row_div = RAD_FP8_BLOCK;
    ws = decl_view(b, sn, wn, "scale", RAD_BF16, { fp8_blocks(N), fp8_blocks(K) },
                   access, shard, grp, rp.s, rp.n);
    if (!w || !ws) return RAD_E_INVAL;
    /* An uneven slice; the scale plane takes the same slice in blocks. */
    if (span_hi > 0) {
        if (span_lo % RAD_FP8_BLOCK || span_hi % RAD_FP8_BLOCK) {
            fprintf(stderr, "radiance: '%s' is sliced [%lld, %lld) on this rank, which cuts a %d-"
                            "wide scale block\n", base, (long long)span_lo, (long long)span_hi,
                    RAD_FP8_BLOCK);
            return RAD_E_INVAL;
        }
        RAD_ARCH_TRY(rad_weight_shard_span(b, w, span_lo, span_hi));
        RAD_ARCH_TRY(rad_weight_shard_span(b, ws, span_lo / RAD_FP8_BLOCK,
                                           span_hi / RAD_FP8_BLOCK));
    }

    /* `M` ranges over the token count and selects libr4d's narrow (M <= 16) or tiled kernel. A
     * caller with a tighter bound (the lm_head, over sampled rows only) passes `max_rows`, so the
     * narrow kernel's split-K scratch is not sized for max_tok. */
    const int64_t hi = max_rows > 0 ? max_rows : g.max_tok;
    op = rw(b, RAD_OP(b, "gemm_nt_q",
               RAD_PARAMS(RAD_RANGE("M", 1, hi), RAD_INT("N", N), RAD_INT("K", K),
                          RAD_INT("group", RAD_FP8_BLOCK), RAD_STR("dtype", "fp8a8")),
               RAD_WEIGHTS(w, ws)),
           { a.q, a.s }, { y });
    return RAD_OK;
}

/* `gemm_nt_q` has seven positional operands; the optional `a_sum` and `b_ref` after `y` (for
 * asymmetric integer and mxfp4 grids) are passed as RAD_NONE when unused, since the runtime checks
 * the operand count against the schema (core/runtime/issue.cpp). */
inline void LinearFP8::step(RadCtx* c, const ActFP8& a, rad_buf y, int64_t T, int64_t r0,
                            int64_t rows) const {
    if (rows <= 0) rows = T - r0;
    if (bf16) {
        RAD_ISSUE_N(c, op, rows, brow_slice(a.x, r0, rows, K), RAD_W(w), brow_slice(y, r0, rows, N));
        return;
    }
    if (i8) {
        if (op_q)
            RAD_ISSUE_N(c, op_q, rows, brow_slice(a.x, r0, rows, K), brow_slice(a.q8, r0, rows, K),
                        brow_slice(a.s8, r0, rows, K / RAD_FP8_BLOCK));
        RAD_ISSUE_N(c, op, rows, brow_slice(a.q8, r0, rows, K),
                    brow_slice(a.s8, r0, rows, K / RAD_FP8_BLOCK),
                    RAD_W(w), RAD_W(ws), brow_slice(y, r0, rows, N), RAD_NONE, RAD_NONE);
        return;
    }
    /* The 4-bit weight's copies of the input start at their buffers' first row: they live only
     * between the two ops issued here. */
    if (w4 && w4_split && rows > w4_split) {
        RAD_ISSUE_N(c, op_q, rows, brow_slice(a.x, r0, rows, K), brow_slice(a8q, 0, rows, K),
                    brow_slice(a8s, 0, rows, K / RAD_FP8_BLOCK));
        RAD_ISSUE_N(c, op2, rows, brow_slice(a8q, 0, rows, K),
                    brow_slice(a8s, 0, rows, K / RAD_FP8_BLOCK), RAD_W(w), RAD_W(ws),
                    brow_slice(y, r0, rows, N), RAD_NONE, RAD_NONE);
        return;
    }
    if (w4) {
        RAD_ISSUE_N(c, op_a16, rows, brow_slice(a.x, r0, rows, K), brow_slice(a16, 0, rows, K));
        RAD_ISSUE_N(c, op, rows, brow_slice(a16, 0, rows, K), RAD_NONE, RAD_W(w), RAD_W(ws),
                    brow_slice(y, r0, rows, N), RAD_NONE, RAD_NONE);
        return;
    }
    if (paro5 && rot_split && rows > rot_split) {
        RAD_ISSUE_N(c, op_rot2, rows, brow_slice(a.x, r0, rows, K), RAD_W(rot_dense),
                    RAD_W(rot_scale), brow_slice(rot_q2, r0, rows, rot_parts * K),
                    brow_slice(rot_s2, r0, rows, rot_parts * (K / RAD_FP8_BLOCK)));
        RAD_ISSUE_N(c, op2, rows, brow_slice(rot_q2, r0, rows, rot_parts * K),
                    brow_slice(rot_s2, r0, rows, rot_parts * (K / RAD_FP8_BLOCK)),
                    RAD_W(w), RAD_W(ws), brow_slice(y, r0, rows, N), RAD_NONE, RAD_NONE);
        return;
    }
    if (paro5) {
        RAD_ISSUE_N(c, op_rot, rows, brow_slice(a.x, r0, rows, K), RAD_W(rot_pairs),
                    RAD_W(rot_angle), RAD_W(rot_scale), brow_slice(rot_q, r0, rows, rot_parts * K));
        RAD_ISSUE_N(c, op, rows, brow_slice(rot_q, r0, rows, rot_parts * K), RAD_NONE,
                    RAD_W(w), RAD_W(ws), brow_slice(y, r0, rows, N), RAD_NONE, RAD_NONE);
        return;
    }
    if (mx) {
        /* A rotated weight first rotates its input into its own codes, `rot_parts` copies side by
         * side; a plain weight reads the producer's. */
        const rad_buf aq = paro ? rot_q : a.q, as = paro ? rot_s : a.s;
        const int64_t ak = rot_parts * K;
        if (paro)
            RAD_ISSUE_N(c, op_rot, rows, brow_slice(a.x, r0, rows, K), RAD_W(rot_dense),
                        RAD_W(rot_scale), brow_slice(aq, r0, rows, ak),
                        brow_slice(as, r0, rows, ak / RAD_FP8_BLOCK));
        RAD_ISSUE_N(c, op, rows, brow_slice(aq, r0, rows, ak),
                    brow_slice(as, r0, rows, ak / RAD_FP8_BLOCK),
                    RAD_W(w), RAD_W(ws), brow_slice(y, r0, rows, N), RAD_NONE, RAD_W(wref));
        return;
    }
    RAD_ISSUE_N(c, op, rows, brow_slice(a.q, r0, rows, K),
                brow_slice(a.s, r0, rows, K / RAD_FP8_BLOCK),
                RAD_W(w), RAD_W(ws), brow_slice(y, r0, rows, N), RAD_NONE, RAD_NONE);
}

/* ================================================== an fp8 weight with no op of its own
 *
 * LinearFP8's views without the GEMM, for blocks that declare their own op over the weight: the
 * DFlash2 and DSpark drafters' projections, shipped bf16 and made block fp8 by the recipe (read
 * whole every decode step, and drafts are verified).
 *
 * Views are declared at what they select ([N][K] E4M3 codes, [N/128][K/128] BF16 scales), the
 * shapes libr4d checks before running. Row parts are rows for the weight and 128-row blocks for
 * the plane, as in LinearFP8; parts must be whole blocks.
 */
struct DenseFP8 {
    rad_weight w = 0, s = 0;
    /* Or OCP-MXFP4 when the container holds that (`codes=fp4_e2m1 group=32 scale=e8m0`): e2m1
     * codes, the E8M0 plane, and that plane again as the per-row reference (LinearFP8's mx branch).
     * Blocks declare the GEMM with `gemm()` and pass `bref()` as its last operand, so either format
     * works against the same E4M3 activation pair. */
    rad_weight ref = 0;
    bool       mx = false;
    /* The caller made the name map (a row range of a checkpoint tensor); declare() does not. */
    bool       mapped = false;

    int declare(RadBuilder* b, Names& nm, const char* base, int64_t N, int64_t K,
                int access, int shard, RadWeightGroup grp,
                std::initializer_list<const char*> src,
                std::initializer_list<int64_t> row_parts = {});
    /* `gemm_nt_q` over this weight, `fp8a8` or `mxfp4a8` as it was declared, ranged M 1..hi. */
    rad_op gemm(RadBuilder* b, int64_t hi, int64_t N, int64_t K,
                std::initializer_list<rad_buf> reads, std::initializer_list<rad_buf> writes) const {
        if (mx)
            return rw(b, RAD_OP(b, "gemm_nt_q",
                          RAD_PARAMS(RAD_RANGE("M", 1, hi), RAD_INT("N", N), RAD_INT("K", K),
                                     RAD_INT("group", RAD_MX_GROUP), RAD_STR("dtype", "mxfp4a8")),
                          RAD_WEIGHTS(w, s, ref)),
                      reads, writes);
        return rw(b, RAD_OP(b, "gemm_nt_q",
                      RAD_PARAMS(RAD_RANGE("M", 1, hi), RAD_INT("N", N), RAD_INT("K", K),
                                 RAD_INT("group", RAD_FP8_BLOCK), RAD_STR("dtype", "fp8a8")),
                      RAD_WEIGHTS(w, s)),
                  reads, writes);
    }
    /* The op's trailing `b_ref` operand: the reference view for MXFP4, absent for fp8. */
    RadOperand bref() const { return mx ? RAD_W(ref) : RAD_NONE; }
};

inline int DenseFP8::declare(RadBuilder* b, Names& nm, const char* base, int64_t N, int64_t K,
                             int access, int shard, RadWeightGroup grp,
                             std::initializer_list<const char*> src,
                             std::initializer_list<int64_t> row_parts) {
    if (N <= 0 || K <= 0 || src.size() == 0) return RAD_E_INVAL;

    /* ---- OCP-MXFP4, from the weight's encoding. The exponent grid is one row tall, so parts need
     * only whole 16-row tiles; K takes whole 128s for the activation's per-(row, 128) scale. */
    {
        const char* wn = nm.f("%s.weight", base);
        if (!mapped) RAD_ARCH_TRY(map_linear(b, wn, src));
        RadEncoding enc{};
        mx = weight_enc(b, wn, &enc) && linear_is_mxfp4(enc);
        if (mx) {
            if (N % 16 || K % RAD_FP8_BLOCK) {
                fprintf(stderr, "radiance: '%s' is %lldx%lld per rank; an MXFP4 weight takes rows "
                                "in 16s and K in whole %d-column activation scale blocks\n",
                        base, (long long)N, (long long)K, RAD_FP8_BLOCK);
                return RAD_E_INVAL;
            }
            RowParts rp;
            RAD_ARCH_TRY(rp.set(row_parts, 16, 1, base, "MXFP4 tiles"));
            w   = decl_view(b, wn, nullptr, "codes", RAD_FP4E2M1, { N, K }, access, shard, grp,
                            rp.w, rp.n);
            s   = decl_view(b, nm.f("%s.scale", base), wn, "scale", RAD_E8M0, { N, K / RAD_MX_GROUP },
                            access, shard, grp, rp.w, rp.n);
            ref = decl_view(b, nm.f("%s.ref", base), wn, "scale", RAD_E8M0, { N, K / RAD_MX_GROUP },
                            access, shard, grp, rp.w, rp.n);
            return w && s && ref ? RAD_OK : RAD_E_INVAL;
        }
    }
    if (N % RAD_FP8_BLOCK || K % RAD_FP8_BLOCK) {
        fprintf(stderr, "radiance: '%s' is %lldx%lld per rank, which is not a whole number of "
                        "%dx%d scale blocks. A block-scaled fp8 weight cannot be split inside a "
                        "block; use a bf16 linear here or a world_size that divides cleanly\n",
                base, (long long)N, (long long)K, RAD_FP8_BLOCK, RAD_FP8_BLOCK);
        return RAD_E_INVAL;
    }

    RowParts rp;
    RAD_ARCH_TRY(rp.set(row_parts, RAD_FP8_BLOCK, RAD_FP8_BLOCK, base, "scale blocks"));

    const char* wn = nm.f("%s.weight", base);
    const char* sn = nm.f("%s.scale",  base);   /* the name map was made above */
    w = decl_view(b, wn, nullptr, "codes", RAD_F8E4M3, { N, K }, access, shard, grp, rp.w, rp.n);
    s = decl_view(b, sn, wn, "scale", RAD_BF16, { fp8_blocks(N), fp8_blocks(K) }, access, shard,
                  grp, rp.s, rp.n);
    return w && s ? RAD_OK : RAD_E_INVAL;
}

/* gate_up, silu_mul and the E4M3 quantiser in one GEMM epilogue (`gemm_nt_q_gated`, an fp8a8
 * tile over a block-fp8 [2n, K] weight), declared for M up to min(rows, 64): the kernel refuses M
 * past that band. Optional: when it does not resolve the caller's unfused ops run, and the issue
 * site checks `*gated_rows`, set to the band only when the op exists. */
inline rad_op decl_gated_fold(RadBuilder* b, int64_t rows, int64_t N, int64_t K, rad_weight w,
                              rad_weight ws, std::initializer_list<rad_buf> reads,
                              std::initializer_list<rad_buf> writes, int64_t* gated_rows) {
    const int64_t hi = rows < 64 ? rows : 64;
    if (hi <= 0) return 0;
    const rad_op op = rw(b, RAD_OP(b, "gemm_nt_q_gated",
                             RAD_PARAMS(RAD_RANGE("M", 1, hi), RAD_INT("N", N), RAD_INT("K", K),
                                        RAD_INT("group", RAD_FP8_BLOCK), RAD_STR("dtype", "fp8a8"),
                                        RAD_STR("act", "silu")),
                             RAD_WEIGHTS(w, ws)),
                         reads, writes);
    if (op) *gated_rows = hi;
    return op;
}

}  /* namespace arch */
}  /* namespace rad */

#endif /* RAD_FP8_H */
