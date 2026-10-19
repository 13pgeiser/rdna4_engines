// r4d_sam_attn_bf16 -- Unlimited-OCR's DeepEncoder (SAM ViT-B/16) attention with decomposed relative
// position (`sam_attn`; arch/common/rad_block_sam.h, libref/ref_sam_attn.cpp is the oracle).
//
// The logits of a query row i against a key j of the same window tile are
//
//     scale * q.k  +  q.Rh[i_h, j_h]  +  q.Rw[i_w, j_w]
//
// over the head's whole head_dim, the tables indexed by the offset inside the tile. With `window`
// > 0 the grid is padded with ZERO rows to a multiple of the window and cut into window x window
// tiles (deepencoder's window_partition, after norm1): the pad rows are keys and values of their
// tile and take part in the softmax, their v being zero. `window` 0 is one tile over the segment.
//
// Shape. One block per (segment, tile, head), so the tile's keys are shared by the block's query
// rows and the k/v rows a wave reads at each step are one address (an L1/L2 hit, broadcast across
// the wave). Each thread owns one query row: q and the output accumulator are HD floats in
// registers, the softmax is the online form (running max and sum), and the rel_pos dot of a tile
// row is computed once per (query row, tile row) rather than once per key.
//
// Head widths. HD is a template parameter and only 64 is instantiated (SAM ViT-B's 12 x 64); the
// shim refuses any other width rather than reading past the registers. The tables must be exactly
// 2*tile-1 rows, which every geometry the tower declares is; the shim declines a table that would
// need get_rel_pos's resampling (libref and libavx serve that).
//
// Exactness. f32 throughout, one bf16 rounding of the output, exp2 (the hardware transcendental)
// with the log2(e) factor; the selftest holds it to the house tolerance against a host port of the
// reference, not to bits.
#include "r4d.h"
#include "r4d_args.h"
#include "r4d_common.h"

#include <cmath>

namespace {

constexpr uint32_t kThreads = 128;
constexpr float    kLog2e   = 1.4426950408889634f;

template <int HD>
__global__ __launch_bounds__(kThreads) void sam_attn_kernel(
    const uint16_t* __restrict__ gq, const uint16_t* __restrict__ gk,
    const uint16_t* __restrict__ gv, uint16_t* __restrict__ go,
    const int* __restrict__ cu, const uint16_t* __restrict__ rh,
    const uint16_t* __restrict__ rw,
    int q_ld, int k_ld, int v_ld, int o_ld, int nhead, int gh, int gw, int th, int tw,
    int nth, int ntw, float scale) {
    // blockIdx.x = ((seq * nth + ty) * ntw + tx) * nhead + head: heads adjacent, so the tile's k/v
    // rows stay hot across them.
    int b = blockIdx.x;
    const int head = b % nhead; b /= nhead;
    const int tx   = b % ntw;   b /= ntw;
    const int ty   = b % nth;   b /= nth;
    const int seq  = b;

    const int s0 = cu[seq];
    if (cu[seq + 1] - s0 != gh * gw) return;   // a segment is exactly one grid
    const int keys = th * tw;

    for (int r = (int)threadIdx.x; r < keys; r += (int)kThreads) {
        const int lh = r / tw, lw = r % tw;
        const int ih = ty * th + lh, iw = tx * tw + lw;
        if (ih >= gh || iw >= gw) continue;    // a pad row: no output

        const size_t qrow = (size_t)(s0 + ih * gw + iw);
        const uint16_t* qp = gq + qrow * (size_t)q_ld + (size_t)head * HD;
        float qf[HD];
#pragma unroll
        for (int c = 0; c < HD; ++c) qf[c] = bf16_to_f32(qp[c]);

        float acc[HD];
#pragma unroll
        for (int c = 0; c < HD; ++c) acc[c] = 0.0f;
        float m = -INFINITY, l = 0.0f;

        for (int jh = 0; jh < th; ++jh) {
            const uint16_t* hrp = rh + (size_t)(lh - jh + th - 1) * HD;
            float hr = 0.0f;
#pragma unroll
            for (int c = 0; c < HD; ++c) hr += qf[c] * bf16_to_f32(hrp[c]);

            const int kh = ty * th + jh;
            const bool hreal = kh < gh;
            for (int jw = 0; jw < tw; ++jw) {
                const uint16_t* wrp = rw + (size_t)(lw - jw + tw - 1) * HD;
                float s = hr;
#pragma unroll
                for (int c = 0; c < HD; ++c) s += qf[c] * bf16_to_f32(wrp[c]);

                const int kw = tx * tw + jw;
                const bool real = hreal && kw < gw;
                const uint16_t* vp = nullptr;
                if (real) {
                    const size_t krow = (size_t)(s0 + kh * gw + kw);
                    const uint16_t* kp = gk + krow * (size_t)k_ld + (size_t)head * HD;
                    float d = 0.0f;
#pragma unroll
                    for (int c = 0; c < HD; ++c) d += qf[c] * bf16_to_f32(kp[c]);
                    s += scale * d;
                    vp = gv + krow * (size_t)v_ld + (size_t)head * HD;
                }

                // Online softmax: a new maximum rescales what has been accumulated.
                if (s > m) {
                    const float corr = ex2((m - s) * kLog2e);
#pragma unroll
                    for (int c = 0; c < HD; ++c) acc[c] *= corr;
                    l *= corr;
                    m = s;
                }
                const float p = ex2((s - m) * kLog2e);
                l += p;
                if (real) {
#pragma unroll
                    for (int c = 0; c < HD; ++c) acc[c] += p * bf16_to_f32(vp[c]);
                }
            }
        }

        const float inv = l > 0.0f ? 1.0f / l : 0.0f;
        uint16_t* op = go + qrow * (size_t)o_ld + (size_t)head * HD;
#pragma unroll
        for (int c = 0; c < HD; ++c) op[c] = f32_to_bf16(acc[c] * inv);
    }
}

int launch_sam_attn(const uint16_t* q, const uint16_t* k, const uint16_t* v, uint16_t* o,
                    const int* cu, const uint16_t* rh, const uint16_t* rw,
                    int q_ld, int k_ld, int v_ld, int o_ld, int nseq, int nhead, int gh, int gw,
                    int win, float scale, long stream) {
    const int th = win > 0 ? win : gh, tw = win > 0 ? win : gw;
    const int nth = (gh + th - 1) / th, ntw = (gw + tw - 1) / tw;
    const size_t blocks = (size_t)nseq * nth * ntw * nhead;
    if (blocks == 0 || blocks > 0x7fffffff) return R4D_NO(RAD_E_SHAPE);
    hipLaunchKernelGGL((sam_attn_kernel<64>), dim3((uint32_t)blocks), dim3(kThreads), 0,
                       (hipStream_t)stream, q, k, v, o, cu, rh, rw, q_ld, k_ld, v_ld, o_ld, nhead,
                       gh, gw, th, tw, nth, ntw, scale);
    return hipGetLastError() == hipSuccess ? RAD_OK : R4D_NO(RAD_E_DEVICE);
}

}  // namespace

// ================================================================================ entry point
//
// The shim checks what the kernel assumes: bf16 operands, rank 2 with dense columns and any row
// pitch (the tower's q/k/v are the three column slices of one fused projection), an i32 cu with one
// entry a segment plus the terminator, tables of exactly 2*tile-1 rows, head_dim 64. The segment
// lengths are the kernel's to check against cu (it cannot be read on the host at issue).

#define R4D_NEED(x) do { const int _s = (x); if (_s != RAD_OK) return R4D_NO(_s); } while (0)

static inline int bf16_rows(const RadTensor* t, int64_t cols, int64_t* rows, int64_t* ld) {
    if (!t || !t->data) return RAD_E_INVAL;
    if (t->dtype != RAD_BF16) return RAD_E_DTYPE;
    if (t->rank != 2) return RAD_E_SHAPE;
    if (t->shape[1] != cols || t->stride[1] != 1) return RAD_E_SHAPE;
    *rows = t->shape[0];
    *ld   = t->stride[0];
    return RAD_OK;
}

extern "C" int r4d_rad_sam_attn(const RadArgs* a, RadStream s) {
    enum { V_Q = 0, V_K, V_V, V_CU, V_OUT, V_RH, V_RW };
    const RadTensor* q  = r4d_opt(a, V_Q);
    const RadTensor* k  = r4d_opt(a, V_K);
    const RadTensor* v  = r4d_opt(a, V_V);
    const RadTensor* cu = r4d_opt(a, V_CU);
    const RadTensor* o  = r4d_opt(a, V_OUT);
    const RadTensor* rh = r4d_opt(a, V_RH);
    const RadTensor* rw = r4d_opt(a, V_RW);
    if (!q || !k || !v || !cu || !o || !rh || !rw) return R4D_NO(RAD_E_INVAL);

    const long long hd  = r4d_geti_or(a, "head_dim", 0);
    const long long nh  = r4d_geti_or(a, "n_head", 0);
    const long long gh  = r4d_geti_or(a, "grid_h", 0);
    const long long gw  = r4d_geti_or(a, "grid_w", 0);
    const long long win = r4d_geti_or(a, "window", 0);
    if (hd != 64 || nh <= 0 || gh <= 0 || gw <= 0 || win < 0) return R4D_NO(RAD_E_UNSUPPORTED);
    const long long W = nh * hd;

    int64_t qrows = 0, qld = 0, krows = 0, kld = 0, vrows = 0, vld = 0, orows = 0, old = 0;
    R4D_NEED(bf16_rows(q, W, &qrows, &qld));
    R4D_NEED(bf16_rows(k, W, &krows, &kld));
    R4D_NEED(bf16_rows(v, W, &vrows, &vld));
    R4D_NEED(bf16_rows(o, W, &orows, &old));
    if (qrows <= 0 || krows < qrows || vrows < qrows || orows < qrows) return R4D_NO(RAD_E_SHAPE);

    if (!cu->data) return R4D_NO(RAD_E_INVAL);
    if (cu->dtype != RAD_I32 || !r4d_contig(cu)) return R4D_NO(RAD_E_DTYPE);
    const long long nseq = r4d_numel(cu) - 1;
    if (nseq < 1) return R4D_NO(RAD_E_SHAPE);

    // The tables: [2*tile-1, head_dim] each, the tile being the window or the whole grid.
    const long long th = win > 0 ? win : gh, tw = win > 0 ? win : gw;
    int64_t rhrows = 0, rhld = 0, rwrows = 0, rwld = 0;
    R4D_NEED(bf16_rows(rh, hd, &rhrows, &rhld));
    R4D_NEED(bf16_rows(rw, hd, &rwrows, &rwld));
    if (rhrows != 2 * th - 1 || rwrows != 2 * tw - 1) return R4D_NO(RAD_E_UNSUPPORTED);

    const float scale = (float)r4d_getf(a, "scale", 1.0 / std::sqrt((double)hd));

    return launch_sam_attn((const uint16_t*)r4d_addr(q), (const uint16_t*)r4d_addr(k),
                           (const uint16_t*)r4d_addr(v), (uint16_t*)r4d_addr(o),
                           (const int*)r4d_addr(cu), (const uint16_t*)r4d_addr(rh),
                           (const uint16_t*)r4d_addr(rw), (int)qld, (int)kld, (int)vld, (int)old,
                           (int)nseq, (int)nh, (int)gh, (int)gw, (int)win, scale,
                           (long)(uintptr_t)s);
}
