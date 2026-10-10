// The ten small bf16 ops the engine issues between the GEMMs, and the reason they are here at all.
//
// The rest of this library is kernels worth writing: a WMMA GEMM, a paged attention, a chunked
// scan.  These are not -- they are an embedding gather, a norm, an add, a sigmoid.  They
// are here because a step is not a set of interesting kernels, it is a CHAIN, and a chain with one
// host link does not run: the engine's buffers live in VRAM, libref is domain HOST, and
// `rad_issue` refuses an op with no launch for the domain its operands are on.  Twenty-one ops
// short of a device implementation is not "slower", it is a server that answers the first request
// with `no device kernel for this band` and exits.  So: one kernel library, and this is where a
// kernel with no other home goes -- including the ops nobody would choose to write.
//
// THE ARITHMETIC IS libref's, DELIBERATELY.  Ref is the oracle (spec §17) and these are the
// ops whose disagreements would be invisible -- an RMS norm that accumulates in bf16, a gather that
// reads a negative id as unsigned, a rotary that pairs the wrong halves.  Each kernel below states
// which ref function it must equal.  Where ref rounds, this rounds, in the same place:
//
//   * every reduction accumulates in f32 and only the store narrows to bf16;
//   * `f32_to_bf16` is round-to-nearest-even (r4d_common.h), which is ref's rounder;
//   * a NEGATIVE index -- token id, gather row, KV slot -- is the PADDING convention and writes a
//     zero row or skips the store.  It is not an error and it is not clamped to zero, which would
//     silently duplicate token 0 across every padding position of a ragged batch.
//
// ONE DELIBERATE DIVERGENCE FROM REF, and it is about what a kernel can do rather than about the
// arithmetic.  Ref reads its index arrays on the host and REFUSES the launch when one is out of
// range -- an id at or above n_vocab, a gather row past the source, a KV slot past the last block.
// A device kernel cannot: the indices are in VRAM, checking them means a copy and a synchronisation
// on the step path, and a per-row `return` cannot become a status.  So the bound is an ARGUMENT and
// an out-of-range index takes the same path as a negative one -- a zero row, or no store.  That is
// a defined answer to a caller bug; the alternative is a page fault that takes the server down.
//
// AND ONE ON PRECISION: `expf`, `powf` and `sincosf` here are the ACCURATE forms, not the `__`
// hardware approximations.  A rope angle is position * inv_freq and position reaches 262144, where
// __sincosf's argument reduction has nothing left; and every one of these ops is an oracle
// comparison away from libref, which uses the libm forms.  None of them is on a hot path
// that would notice.
//
// WHAT IS NOT HERE.  No fusion.  `rmsnorm` and `add` next to each other are the unfused sequence
// the architecture plugin asked for, and fusing them is a SELECTION the plugin makes by declaring
// `rmsnorm_add` (spec §2.3), not something a kernel library does behind its back.  When the fused
// forms are wanted they are new entry points and new rows, not a rewrite of these.
//
// Every entry point returns a negative code rather than throwing, matching r4d_gemm_fp8a8.hip.

#include "r4d.h"
#include "r4d_args.h"
#include "r4d_emit.h"
#include "r4d_common.h"
#include "rad_yarn.h"
#include "r4d_fp8_wmma.h"

#include <cstdlib>
#include <vector>
#include <cmath>
#include <cstring>

// ================================================================== attn_paged verify
// Copy n bytes from a (possibly host-page-table-unmapped) device VA into a regular allocation, via
// the DEVICE page table (mapped) -- a host DtoH copy on such a VA faults. Global scope so the
// verify (which reads the KV cache) can use it. Utility for the env-gated verify path only.
__global__ void r4d_copy8_kernel(const uint8_t* __restrict__ src, uint8_t* __restrict__ dst, size_t n) {
    for (size_t i = blockIdx.x * (size_t)256 + threadIdx.x; i < n;
         i += (size_t)gridDim.x * 256)
        dst[i] = src[i];
}
// DEBUG (env-gated, R4D_VERIFY_OFF): run the paged-attention kernel, then recompute the SAME
// attention on the host (the libref two-pass-softmax formula, f32) from the bf16/fp8 q and the
// paged k/v the kernel read, and report the max deviation. A deviation well above the bf16
// rounding (~1e-3) means the device kernel is wrong for the real (paged, ragged) inputs. GLOBAL
// scope (not the anonymous namespace below) so the attention wrappers -- a different translation
// unit -- can call it. Only the first few of each band (decode / prefill) are checked.
static float attn_verify_bf16(uint16_t b) {
    uint32_t u = (uint32_t)b << 16;
    float f;
    std::memcpy(&f, &u, 4);
    return f;
}
static float attn_verify_e4m3(uint8_t b) {
    const float sign = (b & 0x80u) ? -1.0f : 1.0f;
    const uint32_t ex = (uint32_t)(b >> 3) & 0x0Fu, mn = (uint32_t)b & 0x07u;
    if (ex == 0) return sign * (float)mn * (1.0f / 512.0f);
    return sign * (1.0f + (float)mn / 8.0f) * std::ldexp(1.0f, (int)ex - 7);
}
int g_attn_decode_verify_n = 0;
int g_attn_prefill_verify_n = 0;

/* Diagnostics master switch (degenerate-output investigation, 2026-10-08): OFF by default. Set
 * R4D_DIAG=1 to arm the env-gated probes; the probe families each keep their own env:
 * R4D_VERIFY_ATTN=1 -> verify_attn_paged (KV store + attention recompute, maxd);
 * the [KVCHK]/[HIDD]/[LOGITS] store/GEMM probes and [SAMPIN] (sampler input) run their capped
 * dumps under R4D_DIAG alone. Cleared 2026-10: KV store byte-exact, attention maxd ~0.04,
 * hidden state rich, logits sharp, softcap applied, sampler argmax correct. */
int r4d_diag_on() {
    static const int v = std::getenv("R4D_DIAG") ? 1 : 0;
    return v;
}

void verify_attn_paged(const R4DArgs* r, long stream, int kv_is_fp8, int is_decode) {
    static int on = -1;
    if (on < 0) { const char* e = std::getenv("R4D_VERIFY_ATTN"); on = (e && *e) ? 1 : 0; }
    if (!on) return;
    int* cnt = is_decode ? &g_attn_decode_verify_n : &g_attn_prefill_verify_n;
    const int cap = is_decode ? 10 : 4;   /* decode 10 covers the first global (h512) layer too */
    if (*cnt >= cap) return;
    (*cnt)++;
    /* SYNC BEFORE ANY DtoH COPY (see verify_rmsnorm_p): the q / kv / out buffers are the products
     * of earlier ops on `stream`, and a subsequent op reuses the workspace. The DtoH copies below
     * run on the DEFAULT stream (concurrent with `stream`), so without this sync they can read a
     * buffer while a later op overwrites it. Syncing `stream` first guarantees the producers + this
     * kernel have completed and the consumer has not yet been launched. */
    hipStreamSynchronize((hipStream_t)stream);

    const int ew         = kv_is_fp8 ? 1 : 2;
    const int num_seqs   = r->num_seqs, q_len = r->q_len, q_heads = r->q_heads,
              kv_heads   = r->kv_heads, hd = r->head_dim, bs = r->block_size,
              max_blocks = r->max_blocks;
    const bool ragged    = (r->cu_q != nullptr);
    const int total_q    = ragged ? r->total_q : num_seqs * q_len;
    const int gqa        = kv_heads > 0 ? q_heads / kv_heads : 1;
    fprintf(stderr, "[VERIFY attn_%s] START seqs=%d q_len=%d tot_q=%d qh=%d kvh=%d hd=%d bs=%d mb=%d kvs=%ld kvh_s=%ld\n",
            is_decode ? "decode" : "prefill", num_seqs, q_len, total_q,
            q_heads, kv_heads, hd, bs, max_blocks,
            (long)r->kv_block_stride, (long)r->kv_head_stride);

    std::vector<int> bt_h((size_t)num_seqs * (size_t)max_blocks);
    hipMemcpy(bt_h.data(), r->block_table, bt_h.size() * 4, hipMemcpyDeviceToHost);
    std::vector<int> su_h(num_seqs);
    hipMemcpy(su_h.data(), r->seqused_k, su_h.size() * 4, hipMemcpyDeviceToHost);
    std::vector<int> cu_h(num_seqs + 1);
    if (ragged) hipMemcpy(cu_h.data(), r->cu_q, cu_h.size() * 4, hipMemcpyDeviceToHost);
    else for (int s = 0; s <= num_seqs; ++s) cu_h[s] = s * q_len;
    std::vector<float> kds_h(num_seqs * kv_heads), vds_h(num_seqs * kv_heads);
    if (r->k_descale) hipMemcpy(kds_h.data(), r->k_descale, kds_h.size() * 4, hipMemcpyDeviceToHost);
    if (r->v_descale) hipMemcpy(vds_h.data(), r->v_descale, vds_h.size() * 4, hipMemcpyDeviceToHost);
    fprintf(stderr, "[VERIFY attn_%s] P1 small-copies done su0=%d\n", is_decode ? "decode" : "prefill", su_h[0]);

    /* Only the blocks the block tables actually reference are read, so copy just those (a full
     * cache can be gigabytes; a decode step touches one). */
    std::vector<int> used;
    for (int s = 0; s < num_seqs; ++s) {
        int nb = (su_h[s] + bs - 1) / bs;
        if (nb > max_blocks) nb = max_blocks;
        for (int b = 0; b < nb; ++b) {
            int blk = bt_h[(size_t)s * max_blocks + b];
            if (blk < 0) continue;
            bool seen = false;
            for (int u : used) if (u == blk) { seen = true; break; }
            if (!seen) used.push_back(blk);
        }
    }
    fprintf(stderr, "[VERIFY attn_%s] P2 used_n=%zu used0=%d\n", is_decode ? "decode" : "prefill", used.size(), used.empty() ? -1 : used[0]);
    /* The cache DtoH copy faults (host page table unmapped). Read the used blocks via a DEVICE copy
     * kernel (the device page table IS mapped -- the store kernel wrote the cache fine) into a
     * regular allocation, then DtoH that. TEMP (degenerate-output investigation). */
    const size_t per_bytes = (size_t)r->kv_block_stride * (size_t)ew;
    std::vector<uint8_t> kv_h(used.size() * per_bytes);
    {
        uint8_t* reg = nullptr;
        if (hipMalloc(&reg, kv_h.size()) == hipSuccess && reg) {
            for (size_t i = 0; i < used.size(); ++i)
                r4d_copy8_kernel<<<(unsigned)((per_bytes + 255) / 256), 256>>>(
                    (const uint8_t*)r->kv + (size_t)used[i] * per_bytes, reg + i * per_bytes, per_bytes);
            hipDeviceSynchronize();
            hipMemcpy(kv_h.data(), reg, kv_h.size(), hipMemcpyDeviceToHost);
            hipFree(reg);
        }
    }
    fprintf(stderr, "[VERIFY attn_%s] P3 kv_h done bytes=%zu\n", is_decode ? "decode" : "prefill", kv_h.size());
    auto blkidx = [&](int blk) -> int {
        for (size_t i = 0; i < used.size(); ++i) if (used[i] == blk) return (int)i;
        return -1;
    };

    std::vector<uint16_t> qh((size_t)total_q * (size_t)q_heads * (size_t)hd);
    hipMemcpy(qh.data(), r->q, qh.size() * 2, hipMemcpyDeviceToHost);
    fprintf(stderr, "[VERIFY attn_%s] P4 qh done bytes=%zu\n", is_decode ? "decode" : "prefill", qh.size() * 2);
    std::vector<uint16_t> oh((size_t)total_q * (size_t)q_heads * (size_t)hd);
    hipMemcpy(oh.data(), r->out, oh.size() * 2, hipMemcpyDeviceToHost);
    fprintf(stderr, "[VERIFY attn_%s] P5 oh done bytes=%zu\n", is_decode ? "decode" : "prefill", oh.size() * 2);

    float maxq = 0.0f;
    for (size_t i = 0; i < qh.size(); ++i) { float a = fabsf(attn_verify_bf16(qh[i])); if (a > maxq) maxq = a; }

    /* Cap the host recompute so a large prefill (total_q * ctx big) cannot stall the server; the
     * decode band (total_q 1) is always well under it. */
    int ctx_max = 0;
    for (int s = 0; s < num_seqs; ++s) if (su_h[s] > ctx_max) ctx_max = su_h[s];
    const long long work = (long long)total_q * (long long)q_heads * (long long)ctx_max * (long long)hd;
    if (work > 200000000LL) {
        fprintf(stderr, "[VERIFY attn_%s] SKIPPED (work=%lld>cap) seqs=%d q_len=%d tot_q=%d qh=%d kvh=%d hd=%d win=%d scale=%.3g maxq=%.6g\n",
                is_decode ? "decode" : "prefill", work, num_seqs, q_len, total_q, q_heads, kv_heads,
                hd, r->window, r->scale, maxq);
        return;
    }

    float maxd = 0.0f; int bad = 0, ntot = 0;
    for (int t = 0; t < total_q; ++t) {
        int seq = 0, i = 0, ctx = 0, qlen_seq = 0;
        if (ragged) {
            for (int s = 0; s < num_seqs; ++s)
                if (t >= cu_h[s] && t < cu_h[s + 1]) { seq = s; break; }
            i = t - cu_h[seq];
            ctx = su_h[seq];
            qlen_seq = cu_h[seq + 1] - cu_h[seq];
        } else {
            seq = t / q_len;
            i = t % q_len;
            ctx = su_h[seq];
            qlen_seq = q_len;
        }
        const int pos = ctx - qlen_seq + i;
        for (int h = 0; h < q_heads; ++h) {
            const int kh = h / gqa;
            const float kds = r->k_descale ? kds_h[seq * kv_heads + kh] : 1.0f;
            const float vds = r->v_descale ? vds_h[seq * kv_heads + kh] : 1.0f;
            auto visible = [&](int j) -> bool {
                if (r->causal) {
                    if (j > pos) return false;
                    if (r->window > 0 && (pos - j) >= r->window) return false;
                } else {
                    if (j >= ctx) return false;
                    if (r->window > 0) { int d = pos - j; if (d >= r->window || d <= -r->window) return false; }
                }
                return true;
            };
            auto kdot = [&](int j, int idx) -> float {
                float dot = 0.0f;
                for (int d = 0; d < hd; ++d) {
                    const long off = (long)idx * r->kv_block_stride + (long)kh * r->kv_head_stride
                                   + (long)(j % bs) * 2 * hd + d;
                    const uint8_t* p = kv_h.data() + off * ew;
                    float k = kv_is_fp8 ? attn_verify_e4m3(p[0]) : attn_verify_bf16(*(const uint16_t*)p);
                    dot += attn_verify_bf16(qh[(size_t)t * q_heads * hd + (size_t)h * hd + d]) * (k * kds);
                }
                return dot * r->scale;
            };
            float mx = -INFINITY;
            for (int j = 0; j < ctx; ++j) {
                if (!visible(j)) continue;
                if (j / bs >= max_blocks) break;
                int blk = bt_h[(size_t)seq * max_blocks + (j / bs)];
                if (blk < 0) continue;
                int idx = blkidx(blk);
                if (idx < 0) continue;
                float dot = kdot(j, idx);
                if (dot > mx) mx = dot;
            }
            std::vector<float> ref(hd, 0.0f);
            if (mx != -INFINITY) {
                float sum = 0.0f;
                for (int j = 0; j < ctx; ++j) {
                    if (!visible(j)) continue;
                    if (j / bs >= max_blocks) break;
                    int blk = bt_h[(size_t)seq * max_blocks + (j / bs)];
                    if (blk < 0) continue;
                    int idx = blkidx(blk);
                    if (idx < 0) continue;
                    float dot = kdot(j, idx);
                    const float w = std::exp(dot - mx);
                    sum += w;
                    for (int d = 0; d < hd; ++d) {
                        const long off = (long)idx * r->kv_block_stride + (long)kh * r->kv_head_stride
                                       + (long)(j % bs) * 2 * hd + hd + d;
                        const uint8_t* p = kv_h.data() + off * ew;
                        float v = kv_is_fp8 ? attn_verify_e4m3(p[0]) : attn_verify_bf16(*(const uint16_t*)p);
                        ref[d] += w * (v * vds);
                    }
                }
                const float inv = sum > 0.0f ? 1.0f / sum : 0.0f;
                for (int d = 0; d < hd; ++d) ref[d] *= inv;
            }
            for (int d = 0; d < hd; ++d) {
                float dv = fabsf(attn_verify_bf16(oh[(size_t)t * q_heads * hd + (size_t)h * hd + d]) - ref[d]);
                if (dv > maxd) maxd = dv;
                if (dv > 1e-2f) bad++;
                ntot++;
            }
        }
    }
    fprintf(stderr, "[VERIFY attn_%s] seqs=%d q_len=%d tot_q=%d qh=%d kvh=%d hd=%d bs=%d mb=%d win=%d caus=%d scale=%.3g maxd=%.6g bad=%d/%d maxq=%.6g\n",
            is_decode ? "decode" : "prefill", num_seqs, q_len, total_q, q_heads, kv_heads, hd, bs,
            max_blocks, r->window, r->causal, r->scale, maxd, bad, ntot, maxq);
    if (*cnt == 1) {
        fprintf(stderr, "  q  :"); for (int d = 0; d < 8 && d < hd; ++d) fprintf(stderr, " %g", attn_verify_bf16(qh[d]));
        fprintf(stderr, "  out:"); for (int d = 0; d < 8 && d < hd; ++d) fprintf(stderr, " %g", attn_verify_bf16(oh[d]));
        fprintf(stderr, "\n");
    }
}

namespace {

// DEBUG (env-gated, R4D_VERIFY_OFF=1): run the device kernel, then recompute the same op on the
// host (the libref formula, f32) from the bf16 input the kernel read, and report the max deviation.
// A deviation well above the bf16 rounding (~1e-3) means the device kernel is wrong for the real
// shape; a deviation at the bf16 rounding means it is correct. Only the first few calls are
// checked. OFF by default: the DtoH syncs inside perturb stream timing, which the KV-isolation
// investigation cannot afford.
int r4d_verify_on() {
    static int on = -1;
    if (on < 0) { const char* e = std::getenv("R4D_VERIFY_OFF"); on = (e && *e) ? 1 : 0; }
    return on;
}
int g_rmsnorm_p_verify_n = 0;
int g_gelu_mul_verify_n = 0;

// The CORRECT bf16 -> f32 decode: the 16-bit pattern is the high half of the f32 bits. (Casting a
// uint16_t to float gives the integer value of the pattern, 0..65535, which is NOT the value.)
static float bf16_to_f32(uint16_t b) {
    uint32_t u = (uint32_t)b << 16;
    float f;
    std::memcpy(&f, &u, 4);
    return f;
}

void verify_rmsnorm_p(const uint16_t* x, const uint16_t* y, int M, int n, int x_ld, int y_ld,
                      float eps, long stream) {
    if (!r4d_verify_on() || g_rmsnorm_p_verify_n >= 4) return;
    g_rmsnorm_p_verify_n++;
    /* SYNC BEFORE ANY DtoH COPY: the input `x` is the output of an earlier GEMM on `stream`, and a
     * subsequent GEMM (launched after this verify returns) reuses the workspace and overwrites it.
     * The DtoH copy below runs on the DEFAULT stream (concurrent with `stream`), so without this
     * sync it can read `x` while the next GEMM is overwriting it -- a race that produced false
     * O(1e37) "garbage" readings. Syncing `stream` first guarantees the producer GEMM + this kernel
     * have completed and the consumer GEMM has not yet been launched. */
    hipStreamSynchronize((hipStream_t)stream);
    std::vector<uint16_t> xh((size_t)M * (size_t)x_ld);
    hipMemcpy(xh.data(), x, (size_t)M * (size_t)x_ld * 2, hipMemcpyDeviceToHost);
    std::vector<float> xf((size_t)M * (size_t)x_ld);
    for (size_t i = 0; i < xf.size(); ++i) xf[i] = bf16_to_f32(xh[i]);
    std::vector<float> ref((size_t)M * (size_t)y_ld);
    for (int m = 0; m < M; ++m) {
        float ss = 0.0f;
        for (int i = 0; i < n; ++i) { float v = xf[(size_t)m * (size_t)x_ld + i]; ss += v * v; }
        float s = 1.0f / sqrtf(ss / (float)n + eps);
        for (int i = 0; i < n; ++i)
            ref[(size_t)m * (size_t)y_ld + i] = xf[(size_t)m * (size_t)x_ld + i] * s;
    }
    std::vector<uint16_t> yh((size_t)M * (size_t)y_ld);
    hipMemcpy(yh.data(), y, (size_t)M * (size_t)y_ld * 2, hipMemcpyDeviceToHost);
    float maxd = 0.0f; int bad = 0;
    float maxin = 0.0f;
    for (size_t i = 0; i < xf.size(); ++i) { float a = fabsf(xf[i]); if (a > maxin) maxin = a; }
    for (int m = 0; m < M; ++m) for (int i = 0; i < n; ++i) {
        float d = fabsf(bf16_to_f32(yh[(size_t)m * (size_t)y_ld + i]) - ref[(size_t)m * (size_t)y_ld + i]);
        if (d > maxd) maxd = d;
        if (d > 1e-2f) bad++;
    }
    // sample: first 8 input, first 8 device-out, first 8 ref (all as f32); maxin = max-abs of the input (realistic ~O(1) vs GARBAGE ~O(1e37))
    fprintf(stderr, "[VERIFY rmsnorm_p] M=%d n=%d x_ld=%d y_ld=%d eps=%.3g maxd=%.6g bad=%d/%d maxin=%.6g\n",
            M, n, x_ld, y_ld, eps, maxd, bad, M * n, maxin);
    if (g_rmsnorm_p_verify_n == 1) {
        fprintf(stderr, "  in :"); for (int i = 0; i < 8 && i < n; ++i) fprintf(stderr, " %g", xf[i]);
        fprintf(stderr, "  out:"); for (int i = 0; i < 8 && i < n; ++i) fprintf(stderr, " %g", bf16_to_f32(yh[i]));
        fprintf(stderr, "  ref:"); for (int i = 0; i < 8 && i < n; ++i) fprintf(stderr, " %g", ref[i]);
        fprintf(stderr, "\n");
    }
}

void verify_gelu_mul(const uint16_t* gu, const uint16_t* y, int M, int n, int gu_ld, int y_ld,
                     long stream) {
    if (!r4d_verify_on() || g_gelu_mul_verify_n >= 4) return;
    g_gelu_mul_verify_n++;
    /* SYNC BEFORE ANY DtoH COPY (see verify_rmsnorm_p): the input `gu` is the output of the
     * gate_up GEMM on `stream`, and a subsequent GEMM reuses the workspace and overwrites it.
     * The DtoH copy below runs on the DEFAULT stream (concurrent with `stream`), so without this
     * sync it can read `gu` while the next GEMM is overwriting it -- a race that produced false
     * O(1e37) "garbage" readings. Syncing `stream` first guarantees the gate_up GEMM + this kernel
     * have completed and the consumer GEMM has not yet been launched. */
    hipStreamSynchronize((hipStream_t)stream);
    std::vector<uint16_t> guh((size_t)M * (size_t)gu_ld);
    hipMemcpy(guh.data(), gu, (size_t)M * (size_t)gu_ld * 2, hipMemcpyDeviceToHost);
    std::vector<float> ref((size_t)M * (size_t)y_ld);
    const double k0 = 0.7978845608028654, k1 = 0.044715;
    for (int m = 0; m < M; ++m)
        for (int i = 0; i < n; ++i) {
            double g = (double)bf16_to_f32(guh[(size_t)m * (size_t)gu_ld + i]);
            double u = (double)bf16_to_f32(guh[(size_t)m * (size_t)gu_ld + n + i]);
            ref[(size_t)m * (size_t)y_ld + i] = (float)(0.5 * g * (1.0 + std::tanh(k0 * (g + k1 * g * g * g))) * u);
        }
    std::vector<uint16_t> yh((size_t)M * (size_t)y_ld);
    hipMemcpy(yh.data(), y, (size_t)M * (size_t)y_ld * 2, hipMemcpyDeviceToHost);
    float maxd = 0.0f; int bad = 0;
    float maxg = 0.0f;
    for (int m = 0; m < M; ++m) for (int i = 0; i < n; ++i) {
        float a = fabsf((float)bf16_to_f32(guh[(size_t)m * (size_t)gu_ld + i]));
        if (a > maxg) maxg = a;
    }
    for (int m = 0; m < M; ++m) for (int i = 0; i < n; ++i) {
        float d = fabsf(bf16_to_f32(yh[(size_t)m * (size_t)y_ld + i]) - ref[(size_t)m * (size_t)y_ld + i]);
        if (d > maxd) maxd = d;
        if (d > 1e-2f) bad++;
    }
    /* maxg = max-abs of the gate (first half of gu): realistic ~O(1) vs GARBAGE ~O(1e37). If maxg is
     * now realistic after the sync-before-copy fix, the O(1e37) readings were a verify artifact
     * (a race with a subsequent GEMM overwriting the workspace). If maxg is still ~O(1e37), the
     * gate_up GEMM is genuinely producing garbage. */
    fprintf(stderr, "[VERIFY gelu_mul] M=%d n=%d gu_ld=%d y_ld=%d maxd=%.6g bad=%d/%d maxg=%.6g\n",
            M, n, gu_ld, y_ld, maxd, bad, M * n, maxg);
    if (g_gelu_mul_verify_n == 1) {
        fprintf(stderr, "  g  :"); for (int i = 0; i < 8 && i < n; ++i) fprintf(stderr, " %g", bf16_to_f32(guh[i]));
        fprintf(stderr, "  out:"); for (int i = 0; i < 8 && i < n; ++i) fprintf(stderr, " %g", bf16_to_f32(yh[i]));
        fprintf(stderr, "  ref:"); for (int i = 0; i < 8 && i < n; ++i) fprintf(stderr, " %g", ref[i]);
        fprintf(stderr, "\n");
    }
}

constexpr uint32_t kWave    = 32;
constexpr uint32_t kThreads = 256;
constexpr uint32_t kWaves   = kThreads / kWave;

__device__ __forceinline__ float bf2f(uint16_t h) {
    return __builtin_bit_cast(float, (uint32_t)h << 16);
}

/* An orthonormal Walsh-Hadamard over N values a THREAD owns, in place and its own inverse. Only
 * `kv_store_kernel`'s quality probe uses it, and only there does a sequential butterfly make sense:
 * r4d_fwht.h's is the one to reuse anywhere speed matters, and it wants a WAVE per N-wide block
 * with N/32 values a lane. N must be a power of two, which every scale group the probe accepts is. */
template <uint32_t N>
__device__ __forceinline__ void hadamard(float* a) {
    static_assert(N && (N & (N - 1u)) == 0u, "the Walsh-Hadamard width must be a power of two");
    for (uint32_t len = 1; len < N; len <<= 1)
        for (uint32_t i = 0; i < N; i += len << 1)
            for (uint32_t j = 0; j < len; ++j) {
                const float u = a[i + j], w = a[i + j + len];
                a[i + j] = u + w;
                a[i + j + len] = u - w;
            }
    const float s = rsqrtf((float)N);
    for (uint32_t e = 0; e < N; ++e) a[e] *= s;
}

// Sum across the block, broadcast to every thread. One LDS round trip after the wave reduction,
// which is what the norms need and nothing else here does.
template <uint32_t NW>
__device__ __forceinline__ float block_sum(float v, float* lds) {
    v = r4d_wave_sum(v);
    const uint32_t lane = threadIdx.x & (kWave - 1), wave = threadIdx.x / kWave;
    if (lane == 0) lds[wave] = v;
    __syncthreads();
    float s = 0.0f;
#pragma unroll
    for (uint32_t w = 0; w < NW; ++w) s += lds[w];
    return s;
}

// ---------------------------------------------------------------------------- embed_lookup
// ref_embed_lookup.  One block a token; a negative id -- or one at or above n_vocab, see the
// divergence note at the top -- writes a zero row.
//
// `vocab_off` is the first GLOBAL row this rank's table stands for, which makes the embedding
// vocab-parallel: ids stay global, a rank subtracts the offset, and a token belonging to another
// rank takes the same zero-row path the padding id takes.  Summing the ranks then reconstructs x.
// At vocab_off = 0 this is the plain gather, to the bit.
// `mult` multiplies every gathered entry, the product rounded once (Gemma's sqrt(hidden)); 1 is the
// plain copy, to the bit.
__global__ __launch_bounds__(kThreads) void embed_lookup_kernel(
    const int32_t* __restrict__ tok, const uint16_t* __restrict__ wte, uint16_t* __restrict__ x,
    uint32_t n_embd, int32_t n_vocab, int32_t vocab_off, uint32_t x_ld, float mult) {
    const uint32_t m = blockIdx.x;
    const int32_t id = tok[m] - vocab_off;
    uint16_t* __restrict__ dst = x + (size_t)m * x_ld;
    if (id < 0 || id >= n_vocab) {
        for (uint32_t i = threadIdx.x; i < n_embd; i += kThreads) dst[i] = 0;
        return;
    }
    const uint16_t* __restrict__ src = wte + (size_t)id * n_embd;
    if (mult == 1.0f) {
        for (uint32_t i = threadIdx.x; i < n_embd; i += kThreads) dst[i] = src[i];
        return;
    }
    for (uint32_t i = threadIdx.x; i < n_embd; i += kThreads) dst[i] = f32_to_bf16(bf2f(src[i]) * mult);
}

// ---------------------------------------------------------------------------- embed_lookup_q
// ref_embed_lookup_q.  x[m] = bf16(wte[id] * scale), from a table stored as E4M3 codes with ONE
// scale (Q) or as plain bf16, where an absent scale is one.  This is the n-gram table's gather
// (arch/common/rad_block_ple.h) wherever --ngram-placement lets a card read the table: in VRAM, or
// in pinned host memory the card reads in place over the link.
//
// A THREAD A QUAD OF ENTRIES, NOT A BLOCK A ROW.  A row of the n-gram table is 160 entries, so a
// block a row leaves most of its threads idle and the grid is as deep as the chunk's row count.
// Flattened, consecutive threads take consecutive quads, and a row's 40 threads read its 160
// contiguous bytes.  V = 1 is the same walk an entry at a time, for a width or a pitch the quad's
// 4- and 8-byte accesses would misalign.
//
// THE VALUE libavx's LOOKUP TABLE HOLDS, to the bit: a code decodes to f32 exactly
// (v_cvt_pk_f32_fp8, or the software form beside it in r4d_fp8_wmma.h), the product with the
// scale rounds once in f32, and f32_to_bf16 rounds to nearest even.  An id outside [0, n_vocab)
// after the offset writes a zero row (the divergence note at the top of this file).
template <uint32_t V, bool Q>
__global__ __launch_bounds__(kThreads) void embed_lookup_q_kernel(
    const int32_t* __restrict__ tok, const void* __restrict__ wte, const void* __restrict__ sc,
    uint32_t sc_f32, uint16_t* __restrict__ x, uint32_t M, uint32_t n_embd, int32_t n_vocab,
    int32_t vocab_off, uint32_t x_ld) {
    const uint32_t per_row = n_embd / V;
    const uint64_t g = (uint64_t)blockIdx.x * kThreads + threadIdx.x;
    if (g >= (uint64_t)M * per_row) return;
    const uint32_t m = (uint32_t)(g / per_row), j = (uint32_t)(g % per_row) * V;
    const int32_t id = tok[m] - vocab_off;
    uint16_t* __restrict__ dst = x + (size_t)m * x_ld + j;
    if (id < 0 || id >= n_vocab) {
        if constexpr (V == 4) *(uint2*)dst = make_uint2(0u, 0u);
        else                  dst[0] = 0;
        return;
    }
    const float s = !sc ? 1.0f : sc_f32 ? *(const float*)sc : bf2f(*(const uint16_t*)sc);
    const size_t at = (size_t)id * n_embd + j;
    float v[V];
    if constexpr (Q && V == 4) {
        const uint32_t q = *(const uint32_t*)((const uint8_t*)wte + at);
        const v2f lo = r4d_fp8::e4m3_pk<false>(q), hi = r4d_fp8::e4m3_pk<true>(q);
        v[0] = lo.x; v[1] = lo.y; v[2] = hi.x; v[3] = hi.y;
    } else if constexpr (Q) {
        v[0] = r4d_fp8::e4m3(((const uint8_t*)wte)[at]);
    } else if constexpr (V == 4) {
        const uint2 b = *(const uint2*)((const uint16_t*)wte + at);
        v[0] = bf2f((uint16_t)b.x); v[1] = bf2f((uint16_t)(b.x >> 16));
        v[2] = bf2f((uint16_t)b.y); v[3] = bf2f((uint16_t)(b.y >> 16));
    } else {
        v[0] = bf2f(((const uint16_t*)wte)[at]);
    }
    if constexpr (V == 4) {
        *(uint2*)dst = make_uint2(
            (uint32_t)f32_to_bf16(v[0] * s) | ((uint32_t)f32_to_bf16(v[1] * s) << 16),
            (uint32_t)f32_to_bf16(v[2] * s) | ((uint32_t)f32_to_bf16(v[3] * s) << 16));
    } else {
        dst[0] = f32_to_bf16(v[0] * s);
    }
}

// ---------------------------------------------------------------------------- gather_rows
// ref_gather_rows.  Same shape as the embedding gather and the same negative-index rule, but the
// source row pitch is the caller's rather than n: a gather off a padded activation buffer is what
// the op exists for (one logits row per sequence out of a whole prefill chunk).
__global__ __launch_bounds__(kThreads) void gather_rows_kernel(
    const uint16_t* __restrict__ x, const int32_t* __restrict__ idx, uint16_t* __restrict__ y,
    uint32_t n, int32_t src_rows, uint32_t x_ld, uint32_t y_ld) {
    const uint32_t m = blockIdx.x;
    const int32_t r = idx[m];
    uint16_t* __restrict__ dst = y + (size_t)m * y_ld;
    if (r < 0 || r >= src_rows) {
        for (uint32_t i = threadIdx.x; i < n; i += kThreads) dst[i] = 0;
        return;
    }
    const uint16_t* __restrict__ src = x + (size_t)r * x_ld;
    for (uint32_t i = threadIdx.x; i < n; i += kThreads) dst[i] = src[i];
}


// ---------------------------------------------------------------- byte row gather / scatter
//
// ref_gather_rows and ref_scatter_rows at dtype u8.  A SEPARATE PAIR FROM THE bf16 ONES, and not
// a widened version of them, because these move OPAQUE BYTES: the KV pool's tier traffic is the
// caller, a row is one layer's fragment of one block, and what is in it is whatever the attention
// kernel wrote.  Reading that as bf16 would be a lie with teeth -- libref is the oracle for every
// op here, its bf16 path loads through float, and a bf16 NaN payload is not guaranteed to survive
// that round trip.  At u8 every value is 0..255 and float carries it exactly, so oracle and kernel
// agree bit for bit on any payload at all.
//
// 16 BYTES A THREAD WHERE THE ROW ALLOWS IT.  A KV fragment is a tensor extent and is aligned; the
// registry constrains `n` to a multiple of 16 so this is the only path, and a row that did not
// satisfy it would not have resolved here.
__global__ __launch_bounds__(kThreads) void gather_rows_u8_kernel(
    const uchar4* __restrict__ x, const int32_t* __restrict__ idx, uchar4* __restrict__ y,
    uint32_t n4, int32_t src_rows, uint32_t x_ld4, uint32_t y_ld4) {
    const uint32_t m = blockIdx.x;
    const int32_t r = idx[m];
    uchar4* __restrict__ dst = y + (size_t)m * y_ld4;
    if (r < 0 || r >= src_rows) {
        const uchar4 z = make_uchar4(0, 0, 0, 0);
        for (uint32_t i = threadIdx.x; i < n4; i += kThreads) dst[i] = z;
        return;
    }
    const uchar4* __restrict__ src = x + (size_t)r * x_ld4;
    for (uint32_t i = threadIdx.x; i < n4; i += kThreads) dst[i] = src[i];
}

// The mirror.  A negative index writes NOTHING -- the destination is a live pool and the rows this
// call was not asked about belong to somebody else, so the zero-fill the gather does for an absent
// row would be data loss here rather than padding.
__global__ __launch_bounds__(kThreads) void scatter_rows_u8_kernel(
    const uchar4* __restrict__ v, const int32_t* __restrict__ idx, uchar4* __restrict__ x,
    uint32_t n4, int32_t dst_rows, uint32_t v_ld4, uint32_t x_ld4) {
    const uint32_t m = blockIdx.x;
    const int32_t r = idx[m];
    if (r < 0 || r >= dst_rows) return;
    const uchar4* __restrict__ src = v + (size_t)m * v_ld4;
    uchar4* __restrict__ dst = x + (size_t)r * x_ld4;
    for (uint32_t i = threadIdx.x; i < n4; i += kThreads) dst[i] = src[i];
}

// ---------------------------------------------------------------------------- rmsnorm
// ref_rmsnorm: y = x * rsqrt(mean(x^2) + eps) * (w + wadd).
//
// THE GAIN IS ADDED IN f32 AFTER THE bf16 LOAD.  `wadd` is GemmaRMSNorm's `1 +` (Qwen3.5,
// Qwen3-Next, Gemma store a zero-centred gamma); forming `1 + w` in bf16 first loses 0.3% per
// channel, which is why r4d_rmsnorm_had_quant_i8 takes the same argument and why it is not the
// caller's job to bake it into the weight.
//
// THE GAIN IS f32 OR bf16 AND THE CALLER SAYS WHICH.  Every norm gain in this project's
// architecture plugins is declared RAD_F32 -- they are a few thousand elements against gigabytes of
// weights, and the norm is where a rounded gain shows -- while the activation either side is bf16.
// A kernel that assumed one width would read the other as two halves of the wrong number, so the
// width is a template parameter and the shim reads it off the operand.
//
// Two passes over the row.  n is 256 (a QK-norm head) or 5120 (the hidden state), so the row is in
// L1 for the second pass and a one-pass Welford would buy nothing but a different rounding.
template <bool WF32, uint32_t THR>
__global__ __launch_bounds__(THR) void rmsnorm_kernel(
    const uint16_t* __restrict__ x, const void* __restrict__ w, uint16_t* __restrict__ y,
    uint32_t n, float eps, float wadd, float mult, uint32_t x_ld, uint32_t y_ld) {
    __shared__ float lds[THR / kWave];
    const uint32_t m = blockIdx.x;
    const uint16_t* __restrict__ xr = x + (size_t)m * x_ld;
    uint16_t* __restrict__ yr = y + (size_t)m * y_ld;

    float ss = 0.0f;
    for (uint32_t i = threadIdx.x; i < n; i += THR) { const float v = bf2f(xr[i]); ss += v * v; }
    ss = block_sum<THR / kWave>(ss, lds);
    const float s = rsqrtf(ss / (float)n + eps);
    for (uint32_t i = threadIdx.x; i < n; i += THR) {
        const float g = !w ? 1.0f : WF32 ? ((const float*)w)[i] : bf2f(((const uint16_t*)w)[i]);
        yr[i] = f32_to_bf16(bf2f(xr[i]) * s * (g + wadd) * mult);
    }
}

// ---------------------------------------------------------------------------- rmsnorm_p
// ref_rmsnorm_p: y = x * rsqrt(mean(x^2) + eps) -- the SAME normalisation as rmsnorm with NO gain.
// It is the weightless value normalisation Gemma's attention uses on V (llama.cpp's
// ggml_rms_norm(V, eps), no tensor); there is no checkpoint plane to feed a gain through, so an
// rmsnorm call would need a phantom all-zero weight. Two passes over the row, the sum of squares
// in f32; the second pass drops the gain multiply rmsnorm has. The mean accumulates in f32 in
// ascending index order, exactly as rmsnorm does, so the two agree bit-for-bit where rmsnorm's
// gain happens to be one.
__global__ __launch_bounds__(kThreads) void rmsnorm_p_kernel(
    const uint16_t* __restrict__ x, uint16_t* __restrict__ y,
    uint32_t n, float eps, uint32_t x_ld, uint32_t y_ld) {
    __shared__ float lds[kThreads / kWave];
    const uint32_t m = blockIdx.x;
    const uint16_t* __restrict__ xr = x + (size_t)m * x_ld;
    uint16_t* __restrict__ yr = y + (size_t)m * y_ld;
    float ss = 0.0f;
    for (uint32_t i = threadIdx.x; i < n; i += kThreads) { const float v = bf2f(xr[i]); ss += v * v; }
    ss = block_sum<kThreads / kWave>(ss, lds);
    const float s = rsqrtf(ss / (float)n + eps);
    for (uint32_t i = threadIdx.x; i < n; i += kThreads)
        yr[i] = f32_to_bf16(bf2f(xr[i]) * s);
}

// ---------------------------------------------------------------------------- add / mul
// ref_add / ref_mul, including THE ONE BROADCAST RULE: an operand carrying exactly n elements
// against an output of rows*n broadcasts along rows, and nothing else broadcasts at all.  That is
// what makes a bias add this op instead of a second one, and the narrowness is the point -- silent
// broadcasting of an arbitrary shape is how a shape bug becomes a plausible number.
// `b_step` 0 reads one element for the whole row -- with b_ld 0 too, a scalar for the whole op.
__global__ __launch_bounds__(kThreads) void binary_kernel(
    const uint16_t* __restrict__ a, const uint16_t* __restrict__ b, uint16_t* __restrict__ y,
    uint32_t n, uint32_t a_ld, uint32_t b_ld, uint32_t y_ld, int mul, uint32_t b_step) {
    const uint32_t m = blockIdx.x;
    const uint16_t* __restrict__ ar = a + (size_t)m * a_ld;
    const uint16_t* __restrict__ br = b + (size_t)m * b_ld;
    uint16_t* __restrict__ yr = y + (size_t)m * y_ld;
    for (uint32_t i = threadIdx.x; i < n; i += kThreads) {
        const float u = bf2f(ar[i]), v = bf2f(br[i * b_step]);
        yr[i] = f32_to_bf16(mul ? u * v : u + v);
    }
}

// ---------------------------------------------------------------------------- sigmoid
// ref_sigmoid.  Its own op and not a mode on a unary kernel for the same reason it is its own row:
// a gated attention output multiplies by sigmoid(gate) and the selector has to be able to name that.
//
// ROW-WISE WITH A PITCH, not flat over the extent, because the operand it is issued against is a
// COLUMN SLICE: the gate is the second half of each head of an interleaved [tokens, heads,
// 2*head_dim] projection, so its rows are `n` wide inside a stride of 2n and there is no contiguous
// extent to walk.  A flat kernel would either refuse it or read the query.
__global__ __launch_bounds__(kThreads) void sigmoid_kernel(
    const uint16_t* __restrict__ x, uint16_t* __restrict__ y,
    uint32_t n, uint32_t x_ld, uint32_t y_ld) {
    const uint32_t m = blockIdx.x;
    const uint16_t* __restrict__ xr = x + (size_t)m * x_ld;
    uint16_t* __restrict__ yr = y + (size_t)m * y_ld;
    for (uint32_t i = threadIdx.x; i < n; i += kThreads)
        yr[i] = f32_to_bf16(1.0f / (1.0f + expf(-bf2f(xr[i]))));
}

// ---------------------------------------------------------------------------- logit_softcap
// ref_logit_softcap: y = cap * tanh(x / cap), the final-logit softcap. THE ROW-WISE-WITH-A-PITCH
// SHAPE IS COPIED FROM sigmoid on purpose: Gemma's arch issues it against the logits_gemm's output,
// and the pitch form costs nothing on a contiguous buffer -- it is x_ld == n.
__global__ __launch_bounds__(kThreads) void logit_softcap_kernel(
    const uint16_t* __restrict__ x, uint16_t* __restrict__ y,
    uint32_t n, float cap, uint32_t x_ld, uint32_t y_ld) {
    const uint32_t m = blockIdx.x;
    const uint16_t* __restrict__ xr = x + (size_t)m * x_ld;
    uint16_t* __restrict__ yr = y + (size_t)m * y_ld;
    for (uint32_t i = threadIdx.x; i < n; i += kThreads)
        yr[i] = f32_to_bf16(cap * tanhf(bf2f(xr[i]) / cap));
}

/* The f32 leg: the 1.3.0 sampling chain carries the logits in f32 from the gemm, so the softcap
 * is issued against f32 rows. The same row-with-pitch shape as the bf16 leg, and the math is the
 * op's definition verbatim -- no rounding pass, the buffer already holds the arithmetic dtype. */
__global__ __launch_bounds__(kThreads) void logit_softcap_kernel_f32(
    const float* __restrict__ x, float* __restrict__ y,
    uint32_t n, float cap, uint32_t x_ld, uint32_t y_ld) {
    const uint32_t m = blockIdx.x;
    const float* __restrict__ xr = x + (size_t)m * x_ld;
    float* __restrict__ yr = y + (size_t)m * y_ld;
    for (uint32_t i = threadIdx.x; i < n; i += kThreads)
        yr[i] = cap * tanhf(xr[i] / cap);
}

// ---------------------------------------------------------------------------- scale
// ref_scale: y = x * s, s a [1] weight broadcast over every element. THE ROW-WISE-WITH-A-PITCH
// SHAPE IS COPIED FROM sigmoid on purpose: Gemma's arch issues it against the residual stream at
// the end of every layer (the per-layer scalar), and the pitch form costs nothing on a contiguous
// buffer -- it is x_ld == n. The scalar is read once per block (a single load, cached), not per
// element: a per-element reload of a [1] tensor is the same value every time and costs a load per
// element. `s` is f32 or bf16 (the layer_scalar weight is bf16 in the checkpoint); the branch is
// on its dtype, so the op is dtype-generic in the scalar and the bf16 kernel is the residual
// stream's, which is what a served Gemma 4 issues it against.
__global__ __launch_bounds__(kThreads) void scale_kernel(
    const uint16_t* __restrict__ x, const void* __restrict__ s, uint32_t s_dt,
    uint16_t* __restrict__ y, uint32_t n, uint32_t x_ld, uint32_t y_ld) {
    const uint32_t m = blockIdx.x;
    const float sv = (s_dt == RAD_F32) ? ((const float*)s)[0] : bf2f(((const uint16_t*)s)[0]);
    const uint16_t* __restrict__ xr = x + (size_t)m * x_ld;
    uint16_t* __restrict__ yr = y + (size_t)m * y_ld;
    for (uint32_t i = threadIdx.x; i < n; i += kThreads)
        yr[i] = f32_to_bf16(bf2f(xr[i]) * sv);
}

// ---------------------------------------------------------------------------- silu_mul
// ref_silu_mul: y[.., i] = silu(gu[.., i]) * gu[.., n + i].  THE GATE IS THE FIRST HALF, which is
// the order rad-convert fuses gate_proj and up_proj into one weight in; the two conventions cannot
// drift because both are stated and neither is inferred.
__global__ __launch_bounds__(kThreads) void silu_mul_kernel(
    const uint16_t* __restrict__ gu, uint16_t* __restrict__ y,
    uint32_t n, uint32_t gu_ld, uint32_t y_ld) {
    const uint32_t m = blockIdx.x;
    const uint16_t* __restrict__ gr = gu + (size_t)m * gu_ld;
    uint16_t* __restrict__ yr = y + (size_t)m * y_ld;
    for (uint32_t i = threadIdx.x; i < n; i += kThreads) {
        const float g = bf2f(gr[i]);
        yr[i] = f32_to_bf16((g / (1.0f + expf(-g))) * bf2f(gr[n + i]));
    }
}

// ---------------------------------------------------------------------------- gelu_mul
// ref_gelu_mul: y[.., i] = gelu_tanh(gu[.., i]) * gu[.., n + i]. THE GATE IS THE FIRST HALF, the
// same order-lock as silu_mul. The tanh-approx GELU (gemma's gelu_pytorch_tanh), NOT the erf one:
// gated feed-forwards are trained with the tanh form, and the constants are the standard fit the
// gemm_nt_bias "gelu_tanh" epilogue uses (the identical function, so a GELU-gated FFN and a
// tanh-gelu epilogue agree to the last bit).
__global__ __launch_bounds__(kThreads) void gelu_mul_kernel(
    const uint16_t* __restrict__ gu, uint16_t* __restrict__ y,
    uint32_t n, uint32_t gu_ld, uint32_t y_ld) {
    const uint32_t m = blockIdx.x;
    const uint16_t* __restrict__ gr = gu + (size_t)m * gu_ld;
    uint16_t* __restrict__ yr = y + (size_t)m * y_ld;
    for (uint32_t i = threadIdx.x; i < n; i += kThreads) {
        const float g = bf2f(gr[i]);
        const float k0 = 0.7978845608028654f, k1 = 0.044715f;
        const float gelu = 0.5f * g * (1.0f + tanhf(k0 * (g + k1 * g * g * g)));
        yr[i] = f32_to_bf16(gelu * bf2f(gr[n + i]));
    }
}

// ---------------------------------------------------------------------------- rope
// ref_rope, in place over a fused [n_head + 2*n_head_kv] head row -- the V heads at the end are
// left alone, which is the whole reason the op takes the fused projection rather than two tensors.
//
// NeoX pairs element i with i + rot/2; GPT-J pairs 2i with 2i+1.  Which one a checkpoint wants is a
// property of the checkpoint, and getting it wrong produces fluent text with a broken sense of
// position -- the hardest kind of wrong to notice.  `mrope` is NOT a third case: RadBatch carries
// one position component per token, and mrope over equal components is bit-identical to neox
// (rad_block_attn_gated.h says so and says what would have to change first), so the shim maps it.
//
// PARTIAL ROTARY.  `rot` may be less than head_dim -- 64 of 256 on this target -- and the tail of
// the head is untouched.  The base ladder's exponent divides by `den`, which `ladder` resolves:
// `rot` for `rotary` (the rotated prefix, a complete rotary of its own width) and `head_dim` for
// `head` (the proportional variant, a prefix of the full-width table). `den` = `rot` on the
// default path, so existing models are bit-for-bit unchanged.
//
// One thread a PAIR.  The pair's two loads and two stores are the whole body, so there is no
// reduction and no LDS; the grid is (M, heads).
//
// THE MULTI-COMPONENT MODES (ref_rope's rope_freqs): `mode` 2 mrope, 3 imrope, 4 axial, with up to
// three sections, read a position per component out of `plane`-strided rows -- component c of
// token m at pos[c * plane + m]. `plane` 0 is the one-component form, where every mode is neox.
__device__ __forceinline__ int rope_comp(int mode, uint32_t i, const int* sec, int n_sec,
                                         uint32_t* j) {
    *j = i;
    if (mode == 3) {
        if (i % 3 == 1 && (int)i < 3 * sec[1]) return 1;
        if (i % 3 == 2 && (int)i < 3 * sec[2]) return 2;
        return 0;
    }
    int c = 0, start = 0;
    while (c + 1 < n_sec && (int)i >= start + sec[c]) { start += sec[c]; ++c; }
    *j = i - (uint32_t)start;
    return c;
}

__global__ __launch_bounds__(kThreads) void rope_kernel(
    uint16_t* __restrict__ qkv, const int32_t* __restrict__ pos,
    uint32_t head_dim, uint32_t n_rope_heads, uint32_t rot, uint32_t nfreq, int den, int gptj,
    float theta, float rscale, uint32_t row_pitch,
    int mode, int s0, int s1, int s2, int n_sec, uint32_t plane, RadYarn yarn) {
    const uint32_t m = blockIdx.x;
    const uint32_t half = rot / 2;
    const float div = rscale != 0.0f ? rscale : 1.0f;
    const float p0 = (float)pos[m] / div;
    const int sec[3] = { s0, s1, s2 };

    for (uint32_t t = threadIdx.x + blockIdx.y * kThreads; t < n_rope_heads * half;
         t += kThreads * gridDim.y) {
        const uint32_t h = t / half, i = t % half;
        uint16_t* __restrict__ hd = qkv + (size_t)m * row_pitch + (size_t)h * head_dim;
        float inv = 1.0f / powf(theta, (float)(2 * i) / (float)den);
        float p = p0;
        if (plane) {
            uint32_t j = i;
            const int c = rope_comp(mode, i, sec, n_sec, &j);
            p = (float)pos[(size_t)c * plane + m] / div;
            if (mode == 4)
                inv = 1.0f / powf(theta, (float)(2 * j) / (float)(2 * sec[c]));
        }
        inv = yarn.frequency(inv, (int)i);
        /* THE PROPORTIONAL LADDER rotates the whole head with every frequency past nfreq zero
         * (HF's _compute_proportional_rope_parameters), so pair i is (i, i + head_dim/2) and the
         * pairs past nfreq turn by 0: cos 1, sin 0, the value back exactly. */
        if (i >= nfreq) inv = 0.0f;
        const float ang = p * inv;
        float c, s;
        sincosf(ang, &s, &c);
        if (yarn.factor != 0.0f) { c *= yarn.magnitude; s *= yarn.magnitude; }
        const uint32_t i0 = gptj ? 2 * i : i;
        const uint32_t i1 = gptj ? 2 * i + 1 : i + half;
        const float x0 = bf2f(hd[i0]), x1 = bf2f(hd[i1]);
        hd[i0] = f32_to_bf16(x0 * c - x1 * s);
        hd[i1] = f32_to_bf16(x0 * s + x1 * c);
    }
}

// ---------------------------------------------------------------------------- kv_store
// ref_kv_store, and THE PAGED LAYOUT IS THE ONE EVERY KERNEL IN THIS PROJECT SHARES:
//
//     kv_cache[num_blocks, kv_heads, block_size, 2 * head_dim]
//
// K first and V second inside a slot: element d of the key at [..., d] and of the value at
// [..., head_dim + d].  A flat slot index decomposes as block = slot / block_size and
// offset = slot % block_size, which is what `slot_mapping` carries.  A NEGATIVE slot is a padding
// position and stores nothing.
template <uint32_t QG, uint32_t QF, uint32_t QW, uint32_t QB>
__global__ __launch_bounds__(kThreads) void kv_store_kernel(
    const uint16_t* __restrict__ k, const uint16_t* __restrict__ v,
    const int32_t* __restrict__ slot, uint16_t* __restrict__ kv,
    uint32_t head_dim, uint32_t kv_heads, uint32_t block_size, uint32_t n_blocks,
    uint32_t k_ld, uint32_t v_ld) {
    const uint32_t t = blockIdx.x;
    const int32_t s = slot[t];
    if (s < 0 || (uint32_t)s / block_size >= n_blocks) return;
    const uint32_t blk = (uint32_t)s / block_size, off = (uint32_t)s % block_size;
    const size_t slot_base = ((size_t)blk * kv_heads * block_size + (size_t)off) * 2 * head_dim;
    const size_t head_step = (size_t)block_size * 2 * head_dim;

    if constexpr (QG) {
        /* ---- A QUALITY PROBE, NOT A FORMAT. `R4D_KV_QSIM=<g>` round-trips every `g`-element
         * group of k and of v through a quantiser with its own scale and stores the result as
         * ordinary bf16. Nothing about the cache's layout, size or bandwidth changes -- the point
         * is to answer "would this format still write correct code" BEFORE any of that is built,
         * rather than building a real cache to find out.
         *
         * THREE KNOBS, and each one is a question:
         *   R4D_KV_QSIM=<g>   the scale group. int8 spends its 8 bits on the range the group
         *                     occupies, so the components that carry the dot product land near
         *                     bf16 and only the small ones are coarse.
         *   R4D_KV_QSIMF=1    snap to E4M3 instead of int8. E4M3 keeps 3 mantissa bits at EVERY
         *                     magnitude -- ~6% relative everywhere -- so a scale cannot change its
         *                     step, only keep a small head out of the subnormals below 2^-6.
         *   R4D_KV_QSIMW=1    ROTATE the group by an orthonormal Walsh-Hadamard first. A scale is
         *                     set by the group's largest element, so one outlier channel costs
         *                     every other element its resolution; the rotation spreads that
         *                     outlier over all QG channels. It is why this engine's own WIRE codec
         *                     is `wht6` and not `u6`.
         *
         * THE ROTATION IS FREE TO DEPLOY ON BOTH SIDES, which is what makes it worth measuring.
         * The transform is orthonormal and its own inverse, so
         *
         *     score[t] = q . k[t]      = (R q) . (R k[t])
         *     out      = sum p[t] v[t] = R^T ( sum p[t] (R v[t]) )
         *
         * -- a cache holding R k and R v needs the QUERY rotated once per (token, head) and the
         * OUTPUT rotated back once per (token, head). Neither is per CACHED token, so the cost
         * does not grow with context, which is the whole reason a KV cache is expensive. This
         * probe rotates back itself instead, because it stores bf16: same values, one extra round.
         *
         * WHAT THE PROBE SAYS ABOUT EIGHT BITS. Top-1 agreement against a bf16 control is a
         * Bernoulli trial a prompt, so at p ~ 0.93 its standard error is a couple of points at any
         * prompt count worth running -- and every 8-bit arm sits inside that band at ~93-94%
         * agreement. The mean absolute delta, which has thousands of samples rather than hundreds,
         * separates them by only ~5% end to end: int8 with a per-head scale and a rotated K at
         * 0.196, int8 with a scale per 32 at 0.199, unscaled fp8 e4m3 at 0.205.
         *
         * THE CASE FOR AN INTEGER KV CACHE IS THEREFORE WEAK: fp8 is already built, is DENSER
         * (50.0% of bf16 against 50.8% for scaled int8), and measures the same. The rotation does
         * not change that. The rotation MECHANISM is real all the same -- at int4 it is worth
         * +5.0 points of agreement and takes |delta| from 0.308 to 0.255 -- there is simply little
         * left for it to fix at eight bits.
         *
         * ONE THREAD OWNS A GROUP, and k and v are done in SEPARATE PASSES so the thread holds QG
         * floats and not 2*QG -- at QG = 256 that is the difference between 1 KiB of scratch a
         * lane and 2. This runs once per stored token and its speed is irrelevant; a plain
         * sequential butterfly is used rather than r4d_fwht.h's, which wants a WAVE per block. */
        const uint32_t ng = kv_heads * head_dim / QG;
        for (uint32_t g = threadIdx.x; g < ng; g += kThreads) {
            const uint32_t j0 = g * QG, h = j0 / head_dim, d0 = j0 % head_dim;
            uint16_t* __restrict__ c = kv + slot_base + (size_t)h * head_step;
            const uint16_t* const src[2] = { k + (size_t)t * k_ld + j0,
                                             v + (size_t)t * v_ld + j0 };
            uint16_t* const dst[2] = { c + d0, c + head_dim + d0 };
#pragma unroll 1
            for (uint32_t which = 0; which < 2u; ++which) {
                /* QW: 0 none, 1 both, 2 K only, 3 V only. The asymmetry is the point -- the
                 * outlier structure a rotation exists to break up is a KEY phenomenon (channels
                 * that are consistently large after RoPE), and a value row has no such structure,
                 * so rotating V may only be mixing its own quantisation noise across channels. */
                constexpr bool rot_k = (QW == 1u || QW == 2u);
                constexpr bool rot_v = (QW == 1u || QW == 3u);
                const bool rot = which == 0u ? rot_k : rot_v;
                float b[QG];
                for (uint32_t e = 0; e < QG; ++e) b[e] = bf2f(src[which][e]);
                if constexpr (rot_k || rot_v) { if (rot) hadamard<QG>(b); }
                float amax = 0.0f;
                for (uint32_t e = 0; e < QG; ++e) amax = fmaxf(amax, fabsf(b[e]));
                /* QB is the integer width: 8 bits is +-127 and 4 is +-7. Four is where a rotation
                 * has the most to gain, because that is where one outlier really does cost every
                 * other element in the group its resolution. E4M3 ignores it. */
                const float lvl = (float)((1u << (QB - 1u)) - 1u);
                const float sc =
                    amax > 0.0f ? amax * (QF ? (1.0f / 448.0f) : (1.0f / lvl)) : 1.0f;
                for (uint32_t e = 0; e < QG; ++e) {
                    const float q = b[e] / sc;
                    if constexpr (QF) {
                        /* The clamp is the fp8 quantiser's own: the builtin answers an
                         * out-of-range input with a NaN rather than saturating, and amax/scale is
                         * 448 in exact arithmetic but can land an ulp above it. */
                        int pk = 0;
                        pk = __builtin_amdgcn_cvt_pk_fp8_f32(fminf(fmaxf(q, -448.0f), 448.0f),
                                                             0.0f, pk, false);
                        b[e] = r4d_fp8::e4m3((uint32_t)pk & 0xFFu) * sc;
                    } else {
                        b[e] = fminf(fmaxf(rintf(q), -lvl), lvl) * sc;
                    }
                }
                if constexpr (rot_k || rot_v) { if (rot) hadamard<QG>(b); }
                for (uint32_t e = 0; e < QG; ++e) dst[which][e] = f32_to_bf16(b[e]);
            }
        }
        return;
    }
    for (uint32_t j = threadIdx.x; j < kv_heads * head_dim; j += kThreads) {
        const uint32_t h = j / head_dim, d = j % head_dim;
        uint16_t* __restrict__ c = kv + slot_base + (size_t)h * head_step;
        c[d]            = k[(size_t)t * k_ld + j];
        c[head_dim + d] = v[(size_t)t * v_ld + j];
    }
}

// The same scatter into an E4M3 cache.  It is the write side of the fp8kv attention family and its
// `k_descale` / `v_descale` operands: without it a cache declared fp8 has no way to be filled.
//
// THE SCALE IS A CALLER CONSTANT AND IT HAS TO BE.  The reader's descale is indexed
// `[seq * kv_heads + kv_head]` (r4d.h), which is ONE number for a whole sequence -- so it cannot
// depend on the token being written, because tokens written at different steps share it.  vLLM has
// the same contract and defaults both scales to 1.0.  This takes them as arguments so a calibrated
// pair costs nothing later; at 1.0 the store is a plain E4M3 round of the bf16 the attention would
// otherwise have cached.
//
// FOUR COLUMNS A THREAD, for the reason quant_act_fp8 takes four: it is exactly the two
// v_cvt_pk_fp8_f32 that fill one dword, and the store is then a single dword rather than four
// byte writes.  head_dim is a multiple of 4 on every shape this serves and the entry point refuses
// anything else, so a head's 4-column group never straddles two heads and there is no ragged tail.
__global__ __launch_bounds__(kThreads) void kv_store_fp8_kernel(
    const uint16_t* __restrict__ k, const uint16_t* __restrict__ v,
    const int32_t* __restrict__ slot, uint8_t* __restrict__ kv,
    uint32_t head_dim, uint32_t kv_heads, uint32_t block_size, uint32_t n_blocks,
    uint32_t k_ld, uint32_t v_ld, float k_inv, float v_inv) {
    const uint32_t t = blockIdx.x;
    const int32_t s = slot[t];
    if (s < 0 || (uint32_t)s / block_size >= n_blocks) return;
    const uint32_t blk = (uint32_t)s / block_size, off = (uint32_t)s % block_size;
    const size_t slot_base = ((size_t)blk * kv_heads * block_size + (size_t)off) * 2 * head_dim;
    const size_t head_step = (size_t)block_size * 2 * head_dim;

    for (uint32_t j = threadIdx.x * 4u; j < kv_heads * head_dim; j += kThreads * 4u) {
        const uint32_t h = j / head_dim, d = j % head_dim;
        uint8_t* __restrict__ c = kv + slot_base + (size_t)h * head_step;
        float kf[4], vf[4];
#pragma unroll
        for (int e = 0; e < 4; ++e) {
            kf[e] = bf2f(k[(size_t)t * k_ld + j + (uint32_t)e]) * k_inv;
            vf[e] = bf2f(v[(size_t)t * v_ld + j + (uint32_t)e]) * v_inv;
        }
        // Clamped because v_cvt_pk_fp8_f32 answers an out-of-range input with NaN, not a
        // saturate -- the same reason quant_act_fp8 clamps.
        int pk = 0, pv = 0;
        pk = __builtin_amdgcn_cvt_pk_fp8_f32(fminf(fmaxf(kf[0], -448.0f), 448.0f),
                                             fminf(fmaxf(kf[1], -448.0f), 448.0f), pk, false);
        pk = __builtin_amdgcn_cvt_pk_fp8_f32(fminf(fmaxf(kf[2], -448.0f), 448.0f),
                                             fminf(fmaxf(kf[3], -448.0f), 448.0f), pk, true);
        pv = __builtin_amdgcn_cvt_pk_fp8_f32(fminf(fmaxf(vf[0], -448.0f), 448.0f),
                                             fminf(fmaxf(vf[1], -448.0f), 448.0f), pv, false);
        pv = __builtin_amdgcn_cvt_pk_fp8_f32(fminf(fmaxf(vf[2], -448.0f), 448.0f),
                                             fminf(fmaxf(vf[3], -448.0f), 448.0f), pv, true);
        *(uint32_t*)(c + d)            = (uint32_t)pk;
        *(uint32_t*)(c + head_dim + d) = (uint32_t)pv;
    }
}

}  // namespace

// ================================================================================ entry points
//
// Pitches are in ELEMENTS and 0 means "tight".  Every pointer is bf16 except the index arrays,
// which are i32 -- the core's derived buffers are all i32 (core/sched/batch.cpp refuses anything
// else), so there is no 64-bit index form to serve.

/* THE BODY TAKES A SINK; THE PUBLIC ENTRY IS A WRAPPER. `r4d.h`'s long-stream signature is
 * libr4d's driver API and callers outside this tree use it, so it stays exactly as it was -- but
 * the launch has to be describable, and writing the geometry twice is the one thing ABI v10's
 * contract says not to do. One body, two ways in. */
static int embed_lookup_bf16_emit(long tok, long wte, long x, int M, int n_embd, int n_vocab,
                                  int vocab_off, int x_ld, float mult, const R4dSink& sk) {
    if (!tok || !wte || !x) return -1;
    if (M <= 0 || n_embd <= 0 || n_vocab <= 0 || vocab_off < 0) return -2;
    if (!x_ld) x_ld = n_embd;
    return R4D_EMIT(sk, embed_lookup_kernel, dim3((uint32_t)M), dim3(kThreads), 0,
                    (const int32_t*)tok, (const uint16_t*)wte,
                    (uint16_t*)x, (uint32_t)n_embd, (int32_t)n_vocab, (int32_t)vocab_off,
                    (uint32_t)x_ld, mult) == RAD_OK ? 0 : -4;
}

extern "C" int r4d_embed_lookup_bf16(long tok, long wte, long x, int M, int n_embd, int n_vocab,
                                     int vocab_off, int x_ld, long stream) {
    R4dSink sk; sk.stream = (RadStream)(uintptr_t)stream;
    return embed_lookup_bf16_emit(tok, wte, x, M, n_embd, n_vocab, vocab_off, x_ld, 1.0f, sk);
}

/* The quad walk when the width, the pitch and both bases allow its 4- and 8-byte accesses (a bf16
 * table's quad is 8 bytes), an entry a thread otherwise. */
static int embed_lookup_q_emit(const int32_t* tok, const void* wte, bool q, const void* sc,
                               bool sc_f32, uint16_t* x, long long M, int n_embd, int n_vocab,
                               int vocab_off, long long x_ld, const R4dSink& sk) {
    const bool quad = n_embd % 4 == 0 && x_ld % 4 == 0 && r4d_aligned(x, 8) &&
                      r4d_aligned(wte, q ? 4 : 8);
    const uint64_t threads = (uint64_t)M * (uint64_t)(quad ? n_embd / 4 : n_embd);
    const dim3 grid((uint32_t)((threads + kThreads - 1) / kThreads));
#define R4D_ELQ(V, Q)                                                                             \
    R4D_EMIT(sk, (embed_lookup_q_kernel<V, Q>), grid, dim3(kThreads), 0, tok, wte, sc,          \
             (uint32_t)(sc_f32 ? 1u : 0u), x, (uint32_t)M, (uint32_t)n_embd, (int32_t)n_vocab,   \
             (int32_t)vocab_off, (uint32_t)x_ld)
    const int rc = quad ? (q ? R4D_ELQ(4, true) : R4D_ELQ(4, false))
                        : (q ? R4D_ELQ(1, true) : R4D_ELQ(1, false));
#undef R4D_ELQ
    return rc == RAD_OK ? 0 : -4;
}

extern "C" int r4d_embed_lookup_bf16_scaled(long tok, long wte, long x, int M, int n_embd,
                                           int n_vocab, int vocab_off, int x_ld, float scale,
                                           long stream) {
    R4dSink sk; sk.stream = (RadStream)(uintptr_t)stream;
    return embed_lookup_bf16_emit(tok, wte, x, M, n_embd, n_vocab, vocab_off, x_ld, scale, sk);
}

static int gather_rows_bf16_emit(long x, long idx, long y, int M, int n, int src_rows,
                                 int x_ld, int y_ld, const R4dSink& sk) {
    if (!x || !idx || !y) return -1;
    if (M <= 0 || n <= 0 || src_rows <= 0) return -2;
    if (!x_ld) x_ld = n;
    if (!y_ld) y_ld = n;
    return R4D_EMIT(sk, gather_rows_kernel, dim3((uint32_t)M), dim3(kThreads), 0,
                    (const uint16_t*)x, (const int32_t*)idx,
                    (uint16_t*)y, (uint32_t)n, (int32_t)src_rows,
                    (uint32_t)x_ld, (uint32_t)y_ld) == RAD_OK ? 0 : -4;
}

extern "C" int r4d_gather_rows_bf16(long x, long idx, long y, int M, int n, int src_rows,
                                    int x_ld, int y_ld, long stream) {
    R4dSink sk; sk.stream = (RadStream)(uintptr_t)stream;
    return gather_rows_bf16_emit(x, idx, y, M, n, src_rows, x_ld, y_ld, sk);
}


/* Byte rows. `n`, `x_ld` and `y_ld` are in BYTES at this boundary and are divided down to uchar4
 * inside, so a caller that thinks in a pool's byte strides never has to convert -- and the one
 * place the /4 happens is the one place it can be wrong. */
static int gather_rows_u8_emit(long x, long idx, long y, int M, int n, int src_rows,
                               int x_ld, int y_ld, const R4dSink& sk) {
    if (!x || !idx || !y) return -1;
    if (M <= 0 || n <= 0 || src_rows <= 0) return -2;
    if (!x_ld) x_ld = n;
    if (!y_ld) y_ld = n;
    if ((n | x_ld | y_ld) & 3) return -2;        /* the registry constrains n; pitches too */
    return R4D_EMIT(sk, gather_rows_u8_kernel, dim3((uint32_t)M), dim3(kThreads), 0,
                    (const uchar4*)x, (const int32_t*)idx, (uchar4*)y,
                    (uint32_t)(n / 4), (int32_t)src_rows,
                    (uint32_t)(x_ld / 4), (uint32_t)(y_ld / 4)) == RAD_OK ? 0 : -4;
}

extern "C" int r4d_gather_rows_u8(long x, long idx, long y, int M, int n, int src_rows,
                                  int x_ld, int y_ld, long stream) {
    R4dSink sk; sk.stream = (RadStream)(uintptr_t)stream;
    return gather_rows_u8_emit(x, idx, y, M, n, src_rows, x_ld, y_ld, sk);
}

static int scatter_rows_u8_emit(long v, long idx, long x, int M, int n, int dst_rows,
                                int v_ld, int x_ld, const R4dSink& sk) {
    if (!v || !idx || !x) return -1;
    if (M <= 0 || n <= 0 || dst_rows <= 0) return -2;
    if (!v_ld) v_ld = n;
    if (!x_ld) x_ld = n;
    if ((n | v_ld | x_ld) & 3) return -2;
    return R4D_EMIT(sk, scatter_rows_u8_kernel, dim3((uint32_t)M), dim3(kThreads), 0,
                    (const uchar4*)v, (const int32_t*)idx, (uchar4*)x,
                    (uint32_t)(n / 4), (int32_t)dst_rows,
                    (uint32_t)(v_ld / 4), (uint32_t)(x_ld / 4)) == RAD_OK ? 0 : -4;
}

extern "C" int r4d_scatter_rows_u8(long v, long idx, long x, int M, int n, int dst_rows,
                                   int v_ld, int x_ld, long stream) {
    R4dSink sk; sk.stream = (RadStream)(uintptr_t)stream;
    return scatter_rows_u8_emit(v, idx, x, M, n, dst_rows, v_ld, x_ld, sk);
}

// `w_f32` says whether the gain is f32 (every norm gain the architecture plugins declare) or bf16;
// a null `w` is a gain of one. `mult` scales the row before its one rounding (rmsnorm's `mult`).
extern "C" int r4d_rmsnorm_mult_bf16(long x, long w, long y, int M, int n, float eps, float wadd,
                                     float mult, int w_f32, int x_ld, int y_ld, long stream) {
    if (!x || !y) return -1;
    if (M <= 0 || n <= 0) return -2;
    if (!x_ld) x_ld = n;
    if (!y_ld) y_ld = n;
    // ONE WORKGROUP A ROW, so the grid at a decode batch is a handful of workgroups on a 64-CU
    // part and the row is latency bound. r4d_act_threads is the measured crossover table the int8
    // activation kernels already use; r4d_rmsnorm_quant_fp8 calls the SAME function, and it has to,
    // because the width decides which columns a thread folds into the sum of squares. Two kernels
    // that disagree about it disagree in the last bits of the norm, and the fused one is required
    // to be byte-identical to this kernel followed by the quantiser.
    const int thr = r4d_act_threads(M, n);
#define R4D_RN_T(wf, T)                                                                         \
    hipLaunchKernelGGL((rmsnorm_kernel<wf, T>), dim3((uint32_t)M), dim3(T), 0,                  \
                       (hipStream_t)stream, (const uint16_t*)x, (const void*)w, (uint16_t*)y,   \
                       (uint32_t)n, eps, wadd, mult, (uint32_t)x_ld, (uint32_t)y_ld)
#define R4D_RN(wf)                                                                              \
    do { if (thr == 1024)     R4D_RN_T(wf, 1024);                                               \
         else if (thr == 512) R4D_RN_T(wf, 512);                                                \
         else                 R4D_RN_T(wf, 256); } while (0)
    if (w_f32) R4D_RN(true);
    else       R4D_RN(false);
#undef R4D_RN
#undef R4D_RN_T
    return hipGetLastError() == hipSuccess ? 0 : -4;
}

/* The unscaled form, the entry point every caller before `mult` used: times one is exact. */
extern "C" int r4d_rmsnorm_bf16(long x, long w, long y, int M, int n, float eps, float wadd,
                                int w_f32, int x_ld, int y_ld, long stream) {
    return r4d_rmsnorm_mult_bf16(x, w, y, M, n, eps, wadd, 1.0f, w_f32, x_ld, y_ld, stream);
}

// The weightless norm: rmsnorm without the gain multiply.  It is not templated on a thread count
// the way rmsnorm is, because rmsnorm's width is pinned to the fused rmsnorm_quant_fp8 by their
// shared byte-identical requirement, and rmsnorm_p has no fused quantiser counterpart to match --
// kThreads is the same choice the other non-templated row kernels (binary, sigmoid, silu_mul) make.
extern "C" int r4d_rmsnorm_p_bf16(long x, long y, int M, int n, float eps,
                                  int x_ld, int y_ld, long stream) {
    if (!x || !y) return -1;
    if (M <= 0 || n <= 0) return -2;
    if (!x_ld) x_ld = n;
    if (!y_ld) y_ld = n;
    hipLaunchKernelGGL(rmsnorm_p_kernel, dim3((uint32_t)M), dim3(kThreads), 0, (hipStream_t)stream,
                       (const uint16_t*)x, (uint16_t*)y, (uint32_t)n, eps,
                       (uint32_t)x_ld, (uint32_t)y_ld);
    return hipGetLastError() == hipSuccess ? 0 : -4;
}

// `op` is 0 for a sum and 1 for a product.  `b_ld` of 0 with the same n is the broadcast form: the
// caller passes a b_ld of zero and every row reads the same n elements.
// b_ld -1 is a ONE-ELEMENT b, the scalar broadcast; 0 is one row broadcast down the rows.
extern "C" int r4d_binary_bf16(long a, long b, long y, int M, int n, int op,
                               int a_ld, int b_ld, int y_ld, long stream) {
    if (!a || !b || !y) return -1;
    if (M <= 0 || n <= 0) return -2;
    if (!a_ld) a_ld = n;
    if (!y_ld) y_ld = n;
    const uint32_t b_step = b_ld < 0 ? 0u : 1u;
    if (b_ld < 0) b_ld = 0;
    hipLaunchKernelGGL(binary_kernel, dim3((uint32_t)M), dim3(kThreads), 0, (hipStream_t)stream,
                       (const uint16_t*)a, (const uint16_t*)b, (uint16_t*)y, (uint32_t)n,
                       (uint32_t)a_ld, (uint32_t)b_ld, (uint32_t)y_ld, op, b_step);
    return hipGetLastError() == hipSuccess ? 0 : -4;
}

extern "C" int r4d_sigmoid_bf16(long x, long y, int M, int n, int x_ld, int y_ld, long stream) {
    if (!x || !y) return -1;
    if (M <= 0 || n <= 0) return -2;
    if (!x_ld) x_ld = n;
    if (!y_ld) y_ld = n;
    hipLaunchKernelGGL(sigmoid_kernel, dim3((uint32_t)M), dim3(kThreads), 0, (hipStream_t)stream,
                       (const uint16_t*)x, (uint16_t*)y, (uint32_t)n,
                       (uint32_t)x_ld, (uint32_t)y_ld);
    return hipGetLastError() == hipSuccess ? 0 : -4;
}

// ---------------------------------------------------------------------------- fill
// y[m, i] = v for every element of an [M, n] extent with a row pitch -- the op a rank issues where
// it has nothing to compute and a downstream reader (an all-reduce, a residual) still expects a
// defined buffer. A memset cannot do this: the wire may have a pitch and bf16 is two bytes.
__global__ __launch_bounds__(kThreads) void fill_kernel(uint16_t* __restrict__ y, uint32_t n,
                                                        uint32_t y_ld, uint16_t v) {
    uint16_t* __restrict__ yr = y + (size_t)blockIdx.x * y_ld;
    for (uint32_t i = threadIdx.x; i < n; i += kThreads) yr[i] = v;
}

// The host form of r4d_common.h's f32_to_bf16 (which is __device__): the same round-to-nearest-even.
static inline uint16_t f32_to_bf16_host(float f) {
    uint32_t u;
    std::memcpy(&u, &f, sizeof u);
    return (uint16_t)((u + 0x7fffu + ((u >> 16) & 1u)) >> 16);
}

extern "C" int r4d_fill_bf16(long y, int M, int n, int y_ld, float value, long stream) {
    if (!y) return -1;
    if (M <= 0 || n <= 0) return -2;
    if (!y_ld) y_ld = n;
    hipLaunchKernelGGL(fill_kernel, dim3((uint32_t)M), dim3(kThreads), 0, (hipStream_t)stream,
                       (uint16_t*)y, (uint32_t)n, (uint32_t)y_ld, f32_to_bf16_host(value));
    return hipGetLastError() == hipSuccess ? 0 : -4;
}

extern "C" int r4d_logit_softcap_bf16(long x, long y, int M, int n, float cap,
                                      int x_ld, int y_ld, long stream) {
    if (!x || !y) return -1;
    if (M <= 0 || n <= 0) return -2;
    if (!(cap > 0.0f)) return -3;
    if (!x_ld) x_ld = n;
    if (!y_ld) y_ld = n;
    hipLaunchKernelGGL(logit_softcap_kernel, dim3((uint32_t)M), dim3(kThreads), 0,
                       (hipStream_t)stream,
                       (const uint16_t*)x, (uint16_t*)y, (uint32_t)n, cap,
                       (uint32_t)x_ld, (uint32_t)y_ld);
    return hipGetLastError() == hipSuccess ? 0 : -4;
}

extern "C" int r4d_logit_softcap_f32(long x, long y, int M, int n, float cap,
                                     int x_ld, int y_ld, long stream) {
    if (!x || !y) return -1;
    if (M <= 0 || n <= 0) return -2;
    if (!(cap > 0.0f)) return -3;
    if (!x_ld) x_ld = n;
    if (!y_ld) y_ld = n;
    hipLaunchKernelGGL(logit_softcap_kernel_f32, dim3((uint32_t)M), dim3(kThreads), 0,
                       (hipStream_t)stream,
                       (const float*)x, (float*)y, (uint32_t)n, cap,
                       (uint32_t)x_ld, (uint32_t)y_ld);
    return hipGetLastError() == hipSuccess ? 0 : -4;
}

extern "C" int r4d_scale_bf16(long x, long s, uint32_t s_dt, long y, int M, int n,
                              int x_ld, int y_ld, long stream) {
    if (!x || !s || !y) return -1;
    if (M <= 0 || n <= 0) return -2;
    if (s_dt != RAD_F32 && s_dt != RAD_BF16) return -3;
    if (!x_ld) x_ld = n;
    if (!y_ld) y_ld = n;
    hipLaunchKernelGGL(scale_kernel, dim3((uint32_t)M), dim3(kThreads), 0,
                       (hipStream_t)stream,
                       (const uint16_t*)x, (const void*)s, s_dt, (uint16_t*)y, (uint32_t)n,
                       (uint32_t)x_ld, (uint32_t)y_ld);
    return hipGetLastError() == hipSuccess ? 0 : -4;
}

extern "C" int r4d_silu_mul_bf16(long gu, long y, int M, int n, int gu_ld, int y_ld, long stream) {
    if (!gu || !y) return -1;
    if (M <= 0 || n <= 0) return -2;
    if (!gu_ld) gu_ld = 2 * n;
    if (!y_ld) y_ld = n;
    hipLaunchKernelGGL(silu_mul_kernel, dim3((uint32_t)M), dim3(kThreads), 0, (hipStream_t)stream,
                       (const uint16_t*)gu, (uint16_t*)y, (uint32_t)n,
                       (uint32_t)gu_ld, (uint32_t)y_ld);
    return hipGetLastError() == hipSuccess ? 0 : -4;
}

// GELU-gated product, the gemma FFN activation.  Same shape and order-lock as silu_mul (the gate
// is the first half of the fused [M, 2n] row); the only difference is the nonlinearity, which is
// the tanh approximation, not the erf one, to match ref_gelu_mul's act_gelu_tanh.
extern "C" int r4d_gelu_mul_bf16(long gu, long y, int M, int n, int gu_ld, int y_ld, long stream) {
    if (!gu || !y) return -1;
    if (M <= 0 || n <= 0) return -2;
    if (!gu_ld) gu_ld = 2 * n;
    if (!y_ld) y_ld = n;
    hipLaunchKernelGGL(gelu_mul_kernel, dim3((uint32_t)M), dim3(kThreads), 0, (hipStream_t)stream,
                       (const uint16_t*)gu, (uint16_t*)y, (uint32_t)n,
                       (uint32_t)gu_ld, (uint32_t)y_ld);
    return hipGetLastError() == hipSuccess ? 0 : -4;
}

// `n_rope_heads` is n_head + n_head_kv: the Q heads and the K heads of a fused projection, or just
// the one buffer's heads when q and k are separate tensors (which is what the gated block does).
extern "C" int r4d_rope_bf16(long qkv, long pos, int M, int head_dim, int n_rope_heads,
                             int rot, int gptj, float theta, float rscale, int row_pitch,
                             long stream) {
    return r4d_rope_mc_bf16(qkv, pos, M, head_dim, n_rope_heads, rot, rot, gptj, theta, rscale,
                            row_pitch, 0, nullptr, 0, 0, stream);
}

// THE ONE LAUNCH OF rope_kernel. `yarn` with factor 0 is the plain rotary: the kernel's YaRN
// blend returns the frequency it was given and the magnitude is not applied, so the C entry points
// below and the op without YaRN run the same arithmetic with the same arguments. `den` is the
// ladder's width (`rot` for `rotary`, `head_dim` for `head`) -- the multi-component form passes
// the caller's, and the plain entry passes `rot`, as it always has.
static int rope_launch(long qkv, long pos, int M, int head_dim, int n_rope_heads, int rot, int den,
                       int gptj, float theta, float rscale, int row_pitch, int mode,
                       const int* sec, int n_sec, int plane, RadYarn yarn, long stream,
                       int nfreq = -1) {
    if (!qkv || !pos) return -1;
    if (M <= 0 || head_dim <= 0 || n_rope_heads <= 0) return -2;
    if (rot <= 0 || rot > head_dim || (rot & 1)) return -3;
    if (n_sec < 0 || n_sec > 3 || (n_sec && !sec)) return -3;
    if (!row_pitch) row_pitch = n_rope_heads * head_dim;
    const uint32_t pairs = (uint32_t)n_rope_heads * (uint32_t)(rot / 2);
    const uint32_t gy = (pairs + kThreads - 1) / kThreads;
    const int s0 = n_sec > 0 ? sec[0] : 0, s1 = n_sec > 1 ? sec[1] : 0, s2 = n_sec > 2 ? sec[2] : 0;
    if (nfreq < 0) nfreq = rot / 2;
    hipLaunchKernelGGL(rope_kernel, dim3((uint32_t)M, gy), dim3(kThreads), 0, (hipStream_t)stream,
                       (uint16_t*)qkv, (const int32_t*)pos, (uint32_t)head_dim,
                       (uint32_t)n_rope_heads, (uint32_t)rot, (uint32_t)nfreq, (int)den, gptj, theta,
                       rscale,
                       (uint32_t)row_pitch, mode, s0, s1, s2, n_sec, (uint32_t)plane, yarn);
    return hipGetLastError() == hipSuccess ? 0 : -4;
}

// The multi-component form: `mode` 2 mrope, 3 imrope, 4 axial, `sec[0..n_sec)` the sections and
// `plane` the component pitch of `pos`; plane 0 is the one-component rotary above.
extern "C" int r4d_rope_mc_bf16(long qkv, long pos, int M, int head_dim, int n_rope_heads,
                                int rot, int den, int gptj, float theta, float rscale,
                                int row_pitch, int mode, const int* sec, int n_sec, int plane,
                                long stream) {
    return rope_launch(qkv, pos, M, head_dim, n_rope_heads, rot, den, gptj, theta, rscale,
                       row_pitch, mode, sec, n_sec, plane, RadYarn{}, stream);
}

// Copy n bf16 from a (possibly host-page-table-unmapped) device VA into a regular allocation.
// The device kernel reads via the DEVICE page table (mapped); a host DtoH copy on such a VA faults.
__global__ void r4d_kv_copy_kernel(const uint16_t* __restrict__ src, uint16_t* __restrict__ dst, size_t n) {
    for (size_t i = blockIdx.x * (size_t)kThreads + threadIdx.x; i < n;
         i += (size_t)gridDim.x * kThreads)
        dst[i] = src[i];
}
__global__ void r4d_kv_copy8_kernel(const uint8_t* __restrict__ src, uint8_t* __restrict__ dst, size_t n) {
    for (size_t i = blockIdx.x * (size_t)kThreads + threadIdx.x; i < n;
         i += (size_t)gridDim.x * kThreads)
        dst[i] = src[i];
}

extern "C" int r4d_kv_store_bf16(long k, long v, long slot, long kv, int M, int head_dim,
                                 int kv_heads, int block_size, int n_blocks,
                                 int k_ld, int v_ld, long stream) {
    if (!k || !v || !slot || !kv) return -1;
    if (M <= 0 || head_dim <= 0 || kv_heads <= 0 || block_size <= 0 || n_blocks <= 0) return -2;
    if (!k_ld) k_ld = kv_heads * head_dim;
    if (!v_ld) v_ld = kv_heads * head_dim;
    /* NO QUANTISATION-SIMULATION BRANCH ON THE SERVING PATH. The probe above is compiled only when
     * its template parameters are set; this launch pins them off. A simulation that stays in the
     * store path is four branches and a macro tree between every token and its cache, for a
     * question that is answered once.
     *
     * The real fp8 cache is a separate entry point (r4d_kv_store_fp8, below) selected by
     * --kv-cache-dtype, not a mode of this one. */
    hipLaunchKernelGGL((kv_store_kernel<0u, 0u, 0u, 8u>), dim3((uint32_t)M), dim3(kThreads), 0,
                       (hipStream_t)stream,
                       (const uint16_t*)k, (const uint16_t*)v, (const int32_t*)slot,
                       (uint16_t*)kv, (uint32_t)head_dim, (uint32_t)kv_heads,
                       (uint32_t)block_size, (uint32_t)n_blocks, (uint32_t)k_ld, (uint32_t)v_ld);
    /* ==== TEMP DIAGNOSTIC (degenerate-output investigation): the store-populate check. ====
     * Established: kv/k/v/slot are DEVICE pointers; the store kernel launches+completes clean;
     * a DtoH copy from `slot` succeeds but a DtoH copy from `kv` FAULTS; kv is not a VMM allocation
     * (hipMemRetainAllocationHandle=invalid). Coherent reading: the cache is mapped in the DEVICE
     * page table (device kernels read/write it fine -> the model runs) and NOT in the HOST page
     * table (host copies fault). So the model's cache is fine; the degenerate output is a forward
     * bug. To confirm the store kernel POPULATED the cache, read it via a DEVICE copy kernel (the
     * device page table is mapped -> succeeds) into a regular allocation, then DtoH that regular
     * buffer. Compare the cached K against the store kernel's input k. Gated: R4D_DIAG=1. */
    if (r4d_diag_on()) {
        static int s_kvchk = 0;
        if (s_kvchk < 2) {
            ++s_kvchk;
            hipDeviceSynchronize();
            const size_t blk_elems = (size_t)kv_heads * block_size * 2 * head_dim;  // stride[0]
            uint16_t* reg = nullptr;
            hipError_t am = hipMalloc(&reg, blk_elems * 2);
            fprintf(stderr, "[KVCHK] call=%d M=%d hd=%d kvh=%d bs=%d nb=%d kld=%d vld=%d malloc=%d reg=%p\n",
                    s_kvchk, M, head_dim, kv_heads, block_size, n_blocks, k_ld, v_ld, (int)am, (void*)reg);
            fflush(stderr);
            if (am == hipSuccess && reg) {
                r4d_kv_copy_kernel<<<(unsigned)((blk_elems + kThreads - 1) / kThreads), kThreads>>>(
                    (const uint16_t*)kv, reg, blk_elems);
                hipError_t cs = hipDeviceSynchronize();
                std::vector<uint16_t> kv_h(blk_elems);
                hipError_t kc = hipMemcpy(kv_h.data(), reg, blk_elems * 2, hipMemcpyDeviceToHost);
                fprintf(stderr, "[KVCHK] call=%d devcopy_sync=%d kvcopy=%d\n", s_kvchk, (int)cs, (int)kc);
                fflush(stderr);
                if (cs == hipSuccess && kc == hipSuccess) {
                    std::vector<int32_t> slot_h(M);
                    hipMemcpy(slot_h.data(), (const int32_t*)(uintptr_t)slot, M * 4, hipMemcpyDeviceToHost);
                    std::vector<uint16_t> k_h((size_t)M * k_ld);
                    hipMemcpy(k_h.data(), (const uint16_t*)(uintptr_t)k, k_h.size() * 2, hipMemcpyDeviceToHost);
                    for (int t = 0; t < M && t < 6; ++t) {
                        int s = slot_h[t];
                        if (s < 0) continue;
                        const int blk = s / block_size, off = s % block_size;
                        const size_t base = ((size_t)blk * kv_heads * block_size + (size_t)off) * 2 * head_dim;
                        if (base + 2 * head_dim > kv_h.size()) break;
                        fprintf(stderr, "[KVCHK] call=%d t=%d slot=%d blk=%d off=%d cacheK[0..3]=", s_kvchk, t, s, blk, off);
                        for (int i = 0; i < 4; ++i) fprintf(stderr, " %.4f", attn_verify_bf16(kv_h[base + i]));
                        fprintf(stderr, "  inputK[0..3]=");
                        for (int i = 0; i < 4; ++i) fprintf(stderr, " %.4f", attn_verify_bf16(k_h[(size_t)t * k_ld + i]));
                        fprintf(stderr, "\n");
                    }
                    fflush(stderr);
                }
            }
            if (reg) hipFree(reg);
        }
    }
    return hipGetLastError() == hipSuccess ? 0 : -4;
}

// bf16 k and v in, E4M3 cache out.  `k_scale` and `v_scale` are what the READER will multiply back
// by -- they are the descale, so this divides -- and 1.0 is the vLLM default and the value the
// attention kernels assume when their descale pointers are null.
static int kv_store_fp8_emit(long k, long v, long slot, long kv, int M, int head_dim,
                             int kv_heads, int block_size, int n_blocks,
                             int k_ld, int v_ld, float k_scale, float v_scale,
                             const R4dSink& sk) {
    if (!k || !v || !slot || !kv) return -1;
    if (M <= 0 || head_dim <= 0 || kv_heads <= 0 || block_size <= 0 || n_blocks <= 0) return -2;
    // Four columns a thread is the packing, so a head must be a whole number of groups; otherwise
    // one group would straddle two heads and write the next head's first bytes.
    if (head_dim % 4) return -3;
    if (!(k_scale > 0.0f) || !(v_scale > 0.0f)) return -5;
    if (!k_ld) k_ld = kv_heads * head_dim;
    if (!v_ld) v_ld = kv_heads * head_dim;
    return R4D_EMIT(sk, kv_store_fp8_kernel, dim3((uint32_t)M), dim3(kThreads), 0,
                    (const uint16_t*)k, (const uint16_t*)v, (const int32_t*)slot,
                    (uint8_t*)kv, (uint32_t)head_dim, (uint32_t)kv_heads,
                    (uint32_t)block_size, (uint32_t)n_blocks, (uint32_t)k_ld, (uint32_t)v_ld,
                    1.0f / k_scale, 1.0f / v_scale) == RAD_OK ? 0 : -4;
}

extern "C" int r4d_kv_store_fp8(long k, long v, long slot, long kv, int M, int head_dim,
                                int kv_heads, int block_size, int n_blocks,
                                int k_ld, int v_ld, float k_scale, float v_scale, long stream) {
    R4dSink sk; sk.stream = (RadStream)(uintptr_t)stream;
    return kv_store_fp8_emit(k, v, slot, kv, M, head_dim, kv_heads, block_size, n_blocks,
                             k_ld, v_ld, k_scale, v_scale, sk);
}

// ================================================================================ logits_gemm
//
// THE OUTPUT IS f32 AND THAT IS THE WHOLE REASON THIS KERNEL EXISTS.  `logits_gemm` is the same
// arithmetic as r4d_gemm_bf16_nt_m16 under another parameter spelling, so dispatching it there
// looks correct -- but that kernel writes `__hip_bfloat16* C`, and the engine declares its logits
// buffer f32, because a bf16 logit has eight bits of mantissa and a nucleus cut is a comparison of
// neighbouring probabilities.  The projection would write bf16 pairs into an f32 plane: every
// logit wrong, the magnitudes roughly right, and fluent nonsense out of the sampler.  rad-kbench
// catches it as rel_l2 1.2 against the oracle, but only where it has a device domain to run in.
//
// One workgroup a vocabulary column, 256 threads striding K.  At n_vocab 248320 and K 5120 that is
// a 2.5 GiB weight read a step and the read is the cost; the arithmetic is a v_dot2 per pair.  It
// is not a tuned kernel and does not pretend to be -- there is one variant, it is correct, and the
// tuning axis is open.
//
// Rows come in chunks of kLgRows so the accumulators stay in registers whatever M is.  M here is
// one row per SAMPLED position -- `gather_rows` has already reduced a prefill chunk to one row a
// sequence -- so a chunk is almost always the whole of it.

namespace {

constexpr uint32_t kLgRows = 8;    // rows held in registers at once
constexpr uint32_t kLgVec  = 8;    // bf16 per 128-bit load

__global__ __launch_bounds__(kThreads) void logits_gemm_kernel(
    const uint16_t* __restrict__ a, const uint16_t* __restrict__ w, float* __restrict__ c,
    uint32_t M, uint32_t K, uint32_t N, uint32_t a_ld, uint32_t c_ld) {
    __shared__ float lds[kWaves][kLgRows];
    const uint32_t j = blockIdx.x;
    if (j >= N) return;
    const uint16_t* __restrict__ wj = w + (size_t)j * K;

    for (uint32_t r0 = 0; r0 < M; r0 += kLgRows) {
        const uint32_t nr = (M - r0) < kLgRows ? (M - r0) : kLgRows;
        float acc[kLgRows];
#pragma unroll
        for (uint32_t i = 0; i < kLgRows; ++i) acc[i] = 0.0f;

        for (uint32_t k = threadIdx.x * kLgVec; k < K; k += kThreads * kLgVec) {
            uint16_t wv[kLgVec];
#pragma unroll
            for (uint32_t p = 0; p < kLgVec; ++p) wv[p] = wj[k + p];
            for (uint32_t i = 0; i < nr; ++i) {
                const uint16_t* __restrict__ ar = a + (size_t)(r0 + i) * a_ld + k;
                float s = 0.0f;
#pragma unroll
                for (uint32_t p = 0; p < kLgVec; ++p) s += bf2f(ar[p]) * bf2f(wv[p]);
                acc[i] += s;
            }
        }

        for (uint32_t i = 0; i < nr; ++i) {
            float v = acc[i];
            v = r4d_wave_sum(v);
            const uint32_t lane = threadIdx.x & (kWave - 1), wave = threadIdx.x / kWave;
            if (lane == 0) lds[wave][i] = v;
        }
        __syncthreads();
        if (threadIdx.x < nr) {
            float v = 0.0f;
#pragma unroll
            for (uint32_t q = 0; q < kWaves; ++q) v += lds[q][threadIdx.x];
            c[(size_t)(r0 + threadIdx.x) * c_ld + j] = v;
        }
        __syncthreads();
    }
}

}  // namespace

//   a  [M][K]       bf16, `a_ld` elements a row (0 = K)
//   w  [N][K]       bf16, tight
//   c  [M][N]       f32,  `c_ld` elements a row (0 = N)
extern "C" int r4d_logits_gemm_bf16(long a, long w, long c, int M, int K, int N,
                                    int a_ld, int c_ld, long stream) {
    if (!a || !w || !c) return -1;
    if (M <= 0 || K <= 0 || N <= 0) return -2;
    if (K % (int)kLgVec) return -3;      // a thread owns eight columns; see the kernel
    if (!a_ld) a_ld = K;
    if (!c_ld) c_ld = N;
    hipLaunchKernelGGL(logits_gemm_kernel, dim3((uint32_t)N), dim3(kThreads), 0,
                       (hipStream_t)stream, (const uint16_t*)a, (const uint16_t*)w, (float*)c,
                       (uint32_t)M, (uint32_t)K, (uint32_t)N, (uint32_t)a_ld, (uint32_t)c_ld);
    /* ==== TEMP DIAGNOSTIC (degenerate-output investigation): is the final hidden state featureless? ====
     * KV store + attention verified fine; the logits are near-flat (top-5 within a few points) ->
     * a near-constant hidden state would make every logit the same. a is a regular device buffer
     * (DtoH succeeds). Dump per-row mean/std over K + a few elements: std~0 = featureless (forward
     * collapses); std large = hidden state varies and the flat logits come from elsewhere.
     * Gated: R4D_DIAG=1. */
    if (r4d_diag_on()) {
        static int s_hidd = 0;
        if (s_hidd < 2) {
            ++s_hidd;
            hipDeviceSynchronize();
            const int M1 = (M < 2) ? M : 2;
            std::vector<uint16_t> a_h((size_t)M1 * a_ld);
            hipMemcpy(a_h.data(), (const uint16_t*)a, a_h.size() * 2, hipMemcpyDeviceToHost);
            for (int r = 0; r < M1; ++r) {
                double s = 0, s2 = 0;
                for (int d = 0; d < K; ++d) { double v = attn_verify_bf16(a_h[(size_t)r * a_ld + d]); s += v; s2 += v * v; }
                const double mean = s / K, var = s2 / K - mean * mean;
                fprintf(stderr, "[HIDD] call=%d M=%d K=%d r=%d mean=%.4f std=%.4f a[0..7]=", s_hidd, M, K, r, mean, sqrt(var));
                for (int d = 0; d < 8 && d < K; ++d) fprintf(stderr, " %.4f", attn_verify_bf16(a_h[(size_t)r * a_ld + d]));
                fprintf(stderr, "\n");
            }
            fflush(stderr);
        }
    }
    /* ==== TEMP DIAGNOSTIC (degenerate-output investigation): the logits-GEMM output. ====
     * Hidden state verified RICH (std~4.6). A rich hidden state + a correct logits GEMM => sharp
     * logits; observed logits are near-flat (top-5 within a few points). Dump c (f32, regular
     * buffer, DtoH succeeds): mean/std over the full vocab + max/argmax. std~0 => GEMM output flat
     * (kernel bug); std large => GEMM fine and the sampler/softcap downstream is the suspect.
     * Gated: R4D_DIAG=1. */
    if (r4d_diag_on()) {
        static int s_log = 0;
        if (s_log < 2) {
            ++s_log;
            hipDeviceSynchronize();
            std::vector<float> c_h((size_t)N);
            hipMemcpy(c_h.data(), (const float*)c, (size_t)N * 4, hipMemcpyDeviceToHost);
            long double s = 0, s2 = 0; float mx = -1e30f; int mi = 0;
            for (int j = 0; j < N; ++j) { const double v = c_h[j]; s += v; s2 += v * v; if (c_h[j] > mx) { mx = c_h[j]; mi = j; } }
            const double mean = (double)(s / N), var = (double)(s2 / N) - mean * mean;
            fprintf(stderr, "[LOGITS] call=%d c=%p N=%d ld=%d mean=%.4f std=%.4f max=%.4f argmax=%d c[0..7]=", s_log, (void*)(uintptr_t)c, N, c_ld, mean, sqrt(var), mx, mi);
            for (int j = 0; j < 8 && j < N; ++j) fprintf(stderr, " %.4f", c_h[j]);
            fprintf(stderr, "\n");
            fflush(stderr);
        }
    }
    return hipGetLastError() == hipSuccess ? 0 : -4;
}

// ============================================================== the plugin ABI entry points
//
// Unpacking RadArgs and re-deriving the row pitches IS this kernel's own business, so it lives
// here rather than in a translation layer an out-of-tree author cannot write.  See
// libr4d/r4d_args.h.
//
// WHAT THESE CHECK, and why here and not deeper in.  The kernels take raw pointers and one
// pitch, so anything they do not pitch has to be tight -- the "necessary condition only"
// boundary of spec 2.1.  Every op below therefore establishes, before the launch:
//
//   * the dtype.  These kernels are bf16 and read through a uint16_t; an f32 operand would be
//     read as two bf16 halves and the output would be plausible garbage.  The rows constrain
//     `dtype`, but a constraint is a NECESSARY condition on the geometry and not a promise
//     about the tensors, so the operand's own dtype is what is tested.
//   * the row pitch.  r4d_rows_pitch() gives it in elements, or -1 when an inner axis is
//     strided, and -1 is RAD_E_STRIDE rather than a silently wrong walk.
//   * the row count.  It comes from the OPERAND and never from the `M` parameter: M is RANGED,
//     so the value in the args is the band's upper bound and a loop bounded by it walks off
//     the arena (libref/ref_ops.h makes the same point at more length).
/* bf16 is the only activation width these kernels read. Stated once. */
static inline int dt_bf16(const RadTensor* t) {
    return (t && t->dtype == RAD_BF16) ? RAD_OK : RAD_E_DTYPE;
}
static inline int dt_i32(const RadTensor* t) {
    return (t && (t->dtype == RAD_I32 || t->dtype == RAD_U32)) ? RAD_OK : RAD_E_DTYPE;
}

/* Opaque bytes. The tier traffic moves KV fragments whose contents are whatever the attention
 * kernel wrote, so u8 is the dtype that describes them without claiming to interpret them. */
static inline int dt_u8(const RadTensor* t) {
    return (t && (t->dtype == RAD_U8 || t->dtype == RAD_I8)) ? RAD_OK : RAD_E_DTYPE;
}

#define R4D_NEED(x) do { const int _s = (x); if (_s != RAD_OK) return R4D_NO(_s); } while (0)

/* ------------------------------------------------------------------ embed_lookup */

enum { EL_TOK = 0, EL_WTE, EL_X };

static int rad_embed_lookup_body(const RadArgs* a, const R4dSink& sk) {
    const RadTensor* tok = r4d_opt(a, EL_TOK);
    const RadTensor* wte = r4d_opt(a, EL_WTE);
    const RadTensor* x   = r4d_opt(a, EL_X);
    if (!tok || !wte || !x) return R4D_NO(RAD_E_INVAL);
    R4D_NEED(dt_i32(tok));
    R4D_NEED(dt_bf16(wte));
    R4D_NEED(dt_bf16(x));
    if (!r4d_contig(tok) || !r4d_contig(wte)) return R4D_NO(RAD_E_STRIDE);

    const long long n_embd = r4d_geti_or(a, "n_embd", r4d_dim(wte, (int)wte->rank - 1));
    const long long n_vocab = r4d_geti_or(a, "n_vocab", r4d_dim(wte, (int)wte->rank - 2));
    const long long voff = r4d_geti_or(a, "vocab_offset", 0);
    if (n_embd <= 0 || n_vocab <= 0 || voff < 0) return R4D_NO(RAD_E_SHAPE);
    if (r4d_numel(wte) < n_vocab * n_embd) return R4D_NO(RAD_E_SHAPE);

    long long M = 0;
    const long long x_ld = r4d_rows_pitch(x, n_embd, &M);
    if (x_ld < 0) return R4D_NO(RAD_E_STRIDE);
    if (M <= 0 || r4d_numel(tok) < M) return R4D_NO(RAD_E_SHAPE);

    const float mult = (float)r4d_getf(a, "mult", 1.0);
    return embed_lookup_bf16_emit(r4d_addr(tok), r4d_addr(wte), r4d_addr(x),
                                  (int)M, (int)n_embd, (int)n_vocab, (int)voff, (int)x_ld,
                                  mult, sk) == 0 ? RAD_OK : RAD_E_DEVICE;
}
R4D_ENTRY(r4d_rad_embed_lookup, rad_embed_lookup_body)

/* ------------------------------------------------------------------ embed_lookup_q */

/* Operand 4, `ahead`, is not read. It carries the ids of the prompt after this step for a host
 * reader that fetches table rows from a file before they are wanted; a card reads its table where
 * it already is, so there is nothing to fetch. */
enum { ELQ_TOK = 0, ELQ_WTE, ELQ_SCALE, ELQ_X };

static int rad_embed_lookup_q_body(const RadArgs* a, const R4dSink& sk) {
    const RadTensor* tok = r4d_opt(a, ELQ_TOK);
    const RadTensor* wte = r4d_opt(a, ELQ_WTE);
    const RadTensor* sc  = r4d_opt(a, ELQ_SCALE);
    const RadTensor* x   = r4d_opt(a, ELQ_X);
    if (!tok || !wte || !x) return R4D_NO(RAD_E_INVAL);
    R4D_NEED(dt_i32(tok));
    R4D_NEED(dt_bf16(x));
    const bool q = wte->dtype == RAD_F8E4M3;
    if (!q && wte->dtype != RAD_BF16) return R4D_NO(RAD_E_DTYPE);
    if (sc && sc->dtype != RAD_BF16 && sc->dtype != RAD_F32) return R4D_NO(RAD_E_DTYPE);
    if (sc && r4d_numel(sc) < 1) return R4D_NO(RAD_E_SHAPE);
    if (!r4d_contig(tok) || !r4d_contig(wte)) return R4D_NO(RAD_E_STRIDE);

    const long long n_embd = r4d_geti_or(a, "n_embd", r4d_dim(wte, (int)wte->rank - 1));
    const long long n_vocab = r4d_geti_or(a, "n_vocab", r4d_dim(wte, (int)wte->rank - 2));
    const long long voff = r4d_geti_or(a, "vocab_offset", 0);
    if (n_embd <= 0 || n_vocab <= 0 || voff < 0) return R4D_NO(RAD_E_SHAPE);
    if (r4d_numel(wte) < n_vocab * n_embd) return R4D_NO(RAD_E_SHAPE);

    long long M = 0;
    const long long x_ld = r4d_rows_pitch(x, n_embd, &M);
    if (x_ld < 0) return R4D_NO(RAD_E_STRIDE);
    if (M <= 0 || r4d_numel(tok) < M) return R4D_NO(RAD_E_SHAPE);

    return embed_lookup_q_emit((const int32_t*)r4d_addr(tok), (const void*)r4d_addr(wte), q,
                               sc ? (const void*)r4d_addr(sc) : nullptr,
                               sc && sc->dtype == RAD_F32, (uint16_t*)r4d_addr(x), M,
                               (int)n_embd, (int)n_vocab, (int)voff, x_ld, sk) == 0
               ? RAD_OK : RAD_E_DEVICE;
}
R4D_ENTRY(r4d_rad_embed_lookup_q, rad_embed_lookup_q_body)

/* ------------------------------------------------------------------ gather_rows */

enum { GR_X = 0, GR_IDX, GR_Y };

static int rad_gather_rows_body(const RadArgs* a, const R4dSink& sk) {
    const RadTensor* x   = r4d_opt(a, GR_X);
    const RadTensor* idx = r4d_opt(a, GR_IDX);
    const RadTensor* y   = r4d_opt(a, GR_Y);
    if (!x || !idx || !y) return R4D_NO(RAD_E_INVAL);
    R4D_NEED(dt_bf16(x));
    R4D_NEED(dt_i32(idx));
    R4D_NEED(dt_bf16(y));
    if (!r4d_contig(idx)) return R4D_NO(RAD_E_STRIDE);

    const long long n = r4d_geti_or(a, "n", r4d_dim(y, (int)y->rank - 1));
    if (n <= 0) return R4D_NO(RAD_E_SHAPE);
    long long M = 0, src = 0;
    const long long y_ld = r4d_rows_pitch(y, n, &M);
    const long long x_ld = r4d_rows_pitch(x, n, &src);
    if (y_ld < 0 || x_ld < 0) return R4D_NO(RAD_E_STRIDE);
    if (M <= 0 || src <= 0 || r4d_numel(idx) < M) return R4D_NO(RAD_E_SHAPE);

    return gather_rows_bf16_emit(r4d_addr(x), r4d_addr(idx), r4d_addr(y),
                                 (int)M, (int)n, (int)src, (int)x_ld, (int)y_ld,
                                 sk) == 0 ? RAD_OK : RAD_E_DEVICE;
}
R4D_ENTRY(r4d_rad_gather_rows, rad_gather_rows_body)


/* ------------------------------------------------------------------ byte rows */
/* The u8 forms. `n` here is a COUNT OF BYTES, because at dtype u8 an element is a byte -- so the
 * pitches r4d_rows_pitch returns are byte pitches and go straight through. */
static int rad_gather_rows_u8_body(const RadArgs* a, const R4dSink& sk) {
    const RadTensor* x   = r4d_opt(a, GR_X);
    const RadTensor* idx = r4d_opt(a, GR_IDX);
    const RadTensor* y   = r4d_opt(a, GR_Y);
    if (!x || !idx || !y) return R4D_NO(RAD_E_INVAL);
    R4D_NEED(dt_u8(x));
    R4D_NEED(dt_i32(idx));
    R4D_NEED(dt_u8(y));
    if (!r4d_contig(idx)) return R4D_NO(RAD_E_STRIDE);

    const long long n = r4d_geti_or(a, "n", r4d_dim(y, (int)y->rank - 1));
    if (n <= 0) return R4D_NO(RAD_E_SHAPE);
    long long M = 0, src = 0;
    const long long y_ld = r4d_rows_pitch(y, n, &M);
    const long long x_ld = r4d_rows_pitch(x, n, &src);
    if (y_ld < 0 || x_ld < 0) return R4D_NO(RAD_E_STRIDE);
    if (M <= 0 || src <= 0 || r4d_numel(idx) < M) return R4D_NO(RAD_E_SHAPE);

    return gather_rows_u8_emit(r4d_addr(x), r4d_addr(idx), r4d_addr(y),
                               (int)M, (int)n, (int)src, (int)x_ld, (int)y_ld,
                               sk) == 0 ? RAD_OK : RAD_E_DEVICE;
}
R4D_ENTRY(r4d_rad_gather_rows_u8, rad_gather_rows_u8_body)

/* The operand order is (v, idx, x) and x is INOUT: the destination is the pool and the rows this
 * call does not name stay as they were. */
enum { SR_V = 0, SR_IDX, SR_X };

static int rad_scatter_rows_u8_body(const RadArgs* a, const R4dSink& sk) {
    const RadTensor* v   = r4d_opt(a, SR_V);
    const RadTensor* idx = r4d_opt(a, SR_IDX);
    const RadTensor* x   = r4d_opt(a, SR_X);
    if (!v || !idx || !x) return R4D_NO(RAD_E_INVAL);
    R4D_NEED(dt_u8(v));
    R4D_NEED(dt_i32(idx));
    R4D_NEED(dt_u8(x));
    if (!r4d_contig(idx)) return R4D_NO(RAD_E_STRIDE);

    const long long n = r4d_geti_or(a, "n", r4d_dim(v, (int)v->rank - 1));
    if (n <= 0) return R4D_NO(RAD_E_SHAPE);
    long long M = 0, dst = 0;
    const long long v_ld = r4d_rows_pitch(v, n, &M);
    const long long x_ld = r4d_rows_pitch(x, n, &dst);
    if (v_ld < 0 || x_ld < 0) return R4D_NO(RAD_E_STRIDE);
    if (M <= 0 || dst <= 0 || r4d_numel(idx) < M) return R4D_NO(RAD_E_SHAPE);

    return scatter_rows_u8_emit(r4d_addr(v), r4d_addr(idx), r4d_addr(x),
                                (int)M, (int)n, (int)dst, (int)v_ld, (int)x_ld,
                                sk) == 0 ? RAD_OK : RAD_E_DEVICE;
}
R4D_ENTRY(r4d_rad_scatter_rows_u8, rad_scatter_rows_u8_body)

/* THE SAME COPY, for rows a caller declared bf16: the kernel moves bytes, so a bf16 row of n is
 * 2n bytes and nothing else about it changes. An encoder's rows over a language model's
 * embeddings are the case -- both sides are activations, declared bf16. */
static int rad_scatter_rows_bf16_body(const RadArgs* a, const R4dSink& sk) {
    const RadTensor* v   = r4d_opt(a, SR_V);
    const RadTensor* idx = r4d_opt(a, SR_IDX);
    const RadTensor* x   = r4d_opt(a, SR_X);
    if (!v || !idx || !x) return R4D_NO(RAD_E_INVAL);
    R4D_NEED(dt_bf16(v));
    R4D_NEED(dt_i32(idx));
    R4D_NEED(dt_bf16(x));
    if (!r4d_contig(idx)) return R4D_NO(RAD_E_STRIDE);

    const long long n = r4d_geti_or(a, "n", r4d_dim(v, (int)v->rank - 1));
    if (n <= 0) return R4D_NO(RAD_E_SHAPE);
    long long M = 0, dst = 0;
    const long long v_ld = r4d_rows_pitch(v, n, &M);
    const long long x_ld = r4d_rows_pitch(x, n, &dst);
    if (v_ld < 0 || x_ld < 0) return R4D_NO(RAD_E_STRIDE);
    if (M <= 0 || dst <= 0 || r4d_numel(idx) < M) return R4D_NO(RAD_E_SHAPE);

    return scatter_rows_u8_emit(r4d_addr(v), r4d_addr(idx), r4d_addr(x),
                                (int)M, (int)(2 * n), (int)dst, (int)(2 * v_ld), (int)(2 * x_ld),
                                sk) == 0 ? RAD_OK : RAD_E_DEVICE;
}
R4D_ENTRY(r4d_rad_scatter_rows_bf16, rad_scatter_rows_bf16_body)

/* ------------------------------------------------------------------ cast (bf16 -> bf16) */
/* A STRIDED ROW COPY, which is what `cast` degenerates to when the two ends are the same dtype,
 * and the reason there is a device row for it at all: an architecture that conditions a drafter on
 * the trunk's INTERMEDIATE hidden states has to move rows out of the residual stream before the
 * next layer overwrites it, and the only other implementation is ref's -- on the host, which the
 * reference gate refuses in a serve.
 *
 * libr4d's 2D copy takes byte pitches, so a column slice of a wider buffer is expressible and no
 * kernel has to learn about the layout: this is exactly how five taps land side by side in one
 * [T, 5 * n_embd] row. */
enum { CS_X = 0, CS_Y };

extern "C" int r4d_rad_cast(const RadArgs* a, RadStream s) {
    const RadTensor* x = r4d_opt(a, CS_X);
    const RadTensor* y = r4d_opt(a, CS_Y);
    if (!x || !y) return R4D_NO(RAD_E_INVAL);
    R4D_NEED(dt_bf16(x));
    R4D_NEED(dt_bf16(y));

    const long long n = r4d_geti_or(a, "n", r4d_dim(y, (int)y->rank - 1));
    if (n <= 0) return R4D_NO(RAD_E_SHAPE);
    long long M = 0, Mx = 0;
    const long long y_ld = r4d_rows_pitch(y, n, &M);
    const long long x_ld = r4d_rows_pitch(x, n, &Mx);
    if (y_ld < 0 || x_ld < 0) return R4D_NO(RAD_E_STRIDE);
    /* The ROW COUNT is the destination's. A tap writes a column slice of a buffer sized for the
     * step's ceiling while the source is the live rows, so the two extents differ by design; what
     * must hold is that the source has the rows the destination asks for. */
    if (M <= 0 || Mx < M) return R4D_NO(RAD_E_SHAPE);

    return r4d_p2p_copy2d(r4d_addr(y), r4d_addr(x), y_ld * 2, x_ld * 2, n * 2, M,
                          (long)(uintptr_t)s) == 0 ? RAD_OK : RAD_E_DEVICE;
}

/* ------------------------------------------------------------------ rmsnorm */

enum { RN_X = 0, RN_W, RN_Y };

extern "C" int r4d_rad_rmsnorm(const RadArgs* a, RadStream s) {
    const RadTensor* x = r4d_opt(a, RN_X);
    const RadTensor* w = r4d_opt(a, RN_W);
    const RadTensor* y = r4d_opt(a, RN_Y);
    if (!x || !y) return R4D_NO(RAD_E_INVAL);
    R4D_NEED(dt_bf16(x));
    R4D_NEED(dt_bf16(y));
    /* An ABSENT gain is one: a norm with no scale (Gemma 4's value norm, its multimodal
     * embedder's pre-projection norm). `wadd` belongs to a gain and is refused without one. */
    if (!w) {
        const long long n = r4d_geti_or(a, "n", r4d_dim(y, (int)y->rank - 1));
        if (n <= 0) return R4D_NO(RAD_E_SHAPE);
        if (r4d_getf(a, "wadd", 0.0) != 0.0) return R4D_NO(RAD_E_INVAL);
        long long M = 0, My = 0;
        const long long x_ld = r4d_rows_pitch(x, n, &M);
        const long long y_ld = r4d_rows_pitch(y, n, &My);
        if (x_ld < 0 || y_ld < 0) return R4D_NO(RAD_E_STRIDE);
        if (M <= 0 || My < M) return R4D_NO(RAD_E_SHAPE);
        return r4d_rmsnorm_mult_bf16(r4d_addr(x), 0, r4d_addr(y), (int)M, (int)n,
                                (float)r4d_getf(a, "eps", 1e-6), 0.0f,
                                (float)r4d_getf(a, "mult", 1.0), 0, (int)x_ld, (int)y_ld,
                                (long)(uintptr_t)s) == 0 ? RAD_OK : RAD_E_DEVICE;
    }
    if (!r4d_contig(w)) return R4D_NO(RAD_E_STRIDE);
    /* THE GAIN IS THE ONE OPERAND THAT IS NOT bf16. Every architecture plugin here declares its
     * norm gains RAD_F32 -- a few thousand elements against gigabytes of weights, and the norm is
     * where a rounded gain shows -- so the width is read off the operand and handed to the kernel
     * rather than assumed. Assuming it would read an f32 gain as two bf16 halves of the wrong
     * number, which is the silent kind of wrong. */
    if (w->dtype != RAD_F32 && w->dtype != RAD_BF16) return R4D_NO(RAD_E_DTYPE);
    const int w_f32 = w->dtype == RAD_F32;

    const long long n = r4d_geti_or(a, "n", r4d_dim(w, (int)w->rank - 1));
    if (n <= 0 || r4d_numel(w) < n) return R4D_NO(RAD_E_SHAPE);
    long long M = 0, My = 0;
    const long long x_ld = r4d_rows_pitch(x, n, &M);
    const long long y_ld = r4d_rows_pitch(y, n, &My);
    if (x_ld < 0 || y_ld < 0) return R4D_NO(RAD_E_STRIDE);
    if (M <= 0 || My < M) return R4D_NO(RAD_E_SHAPE);

    const float eps  = (float)r4d_getf(a, "eps", 1e-6);
    const float wadd = (float)r4d_getf(a, "wadd", 0.0);
    return r4d_rmsnorm_mult_bf16(r4d_addr(x), r4d_addr(w), r4d_addr(y), (int)M, (int)n, eps, wadd,
                            (float)r4d_getf(a, "mult", 1.0),
                            w_f32, (int)x_ld, (int)y_ld, (long)(uintptr_t)s) == 0 ? RAD_OK
                                                                                 : RAD_E_DEVICE;
}

/* ------------------------------------------------------------------ rmsnorm_p */

extern "C" int r4d_rad_rmsnorm_p(const RadArgs* a, RadStream s) {
    const RadTensor* x = r4d_opt(a, 0);
    const RadTensor* y = r4d_opt(a, 1);
    if (!x || !y) return R4D_NO(RAD_E_INVAL);
    R4D_NEED(dt_bf16(x));
    R4D_NEED(dt_bf16(y));

    const long long n = r4d_geti_or(a, "n", r4d_dim(x, (int)x->rank - 1));
    if (n <= 0) return R4D_NO(RAD_E_SHAPE);
    long long M = 0, My = 0;
    const long long x_ld = r4d_rows_pitch(x, n, &M);
    const long long y_ld = r4d_rows_pitch(y, n, &My);
    if (x_ld < 0 || y_ld < 0) return R4D_NO(RAD_E_STRIDE);
    if (M <= 0 || My < M) return R4D_NO(RAD_E_SHAPE);

    const float eps = (float)r4d_getf(a, "eps", 1e-6);
    const int rc = r4d_rmsnorm_p_bf16(r4d_addr(x), r4d_addr(y), (int)M, (int)n, eps,
                                      (int)x_ld, (int)y_ld, (long)(uintptr_t)s);
    if (rc != 0) return RAD_E_DEVICE;
    verify_rmsnorm_p((const uint16_t*)(uintptr_t)r4d_addr(x), (const uint16_t*)(uintptr_t)r4d_addr(y),
                     (int)M, (int)n, (int)x_ld, (int)y_ld, eps, (long)(uintptr_t)s);
    return RAD_OK;
}

/* ------------------------------------------------------------------ add / mul */

enum { BI_A = 0, BI_B, BI_Y };

static int binary(const RadArgs* a, int mul, RadStream s) {
    const RadTensor* A = r4d_opt(a, BI_A);
    const RadTensor* B = r4d_opt(a, BI_B);
    const RadTensor* Y = r4d_opt(a, BI_Y);
    if (!A || !B || !Y) return R4D_NO(RAD_E_INVAL);
    R4D_NEED(dt_bf16(A));
    R4D_NEED(dt_bf16(B));
    R4D_NEED(dt_bf16(Y));

    const long long n = r4d_geti_or(a, "n", r4d_dim(Y, (int)Y->rank - 1));
    if (n <= 0) return R4D_NO(RAD_E_SHAPE);
    long long M = 0, Ma = 0;
    const long long y_ld = r4d_rows_pitch(Y, n, &M);
    const long long a_ld = r4d_rows_pitch(A, n, &Ma);
    if (y_ld < 0 || a_ld < 0) return R4D_NO(RAD_E_STRIDE);
    if (M <= 0 || Ma < M) return R4D_NO(RAD_E_SHAPE);

    /* THE ONE BROADCAST RULE: an operand carrying exactly n elements against an output of M rows
     * broadcasts along rows, and nothing else broadcasts at all. A pitch of 0 is how the kernel is
     * told. Anything in between -- more than n, fewer than M*n -- is a shape bug and is refused
     * rather than made plausible. */
    long long b_ld = 0;
    if (r4d_numel(B) == n && M > 1) {
        if (!r4d_contig(B)) return R4D_NO(RAD_E_STRIDE);
    } else {
        long long Mb = 0;
        b_ld = r4d_rows_pitch(B, n, &Mb);
        if (b_ld < 0) return R4D_NO(RAD_E_STRIDE);
        if (Mb < M) return R4D_NO(RAD_E_SHAPE);
        if (b_ld == 0) return R4D_NO(RAD_E_STRIDE);   /* 0 means broadcast; a real pitch cannot be 0 */
    }

    return r4d_binary_bf16(r4d_addr(A), r4d_addr(B), r4d_addr(Y), (int)M, (int)n, mul,
                           (int)a_ld, (int)b_ld, (int)y_ld,
                           (long)(uintptr_t)s) == 0 ? RAD_OK : RAD_E_DEVICE;
}

extern "C" int r4d_rad_add(const RadArgs* a, RadStream s) { return binary(a, 0, s); }
extern "C" int r4d_rad_mul(const RadArgs* a, RadStream s) { return binary(a, 1, s); }

/* mul_weight: y = x * w, `w` a bf16 weight of one element (pitch -1, a scalar) or one row (pitch
 * 0, broadcast down the rows) -- the binary kernel's two broadcast forms. */
extern "C" int r4d_rad_mul_weight(const RadArgs* a, RadStream s) {
    const RadTensor* X = r4d_opt(a, 0);
    const RadTensor* W = r4d_opt(a, 1);
    const RadTensor* Y = r4d_opt(a, 2);
    if (!X || !W || !Y) return R4D_NO(RAD_E_INVAL);
    R4D_NEED(dt_bf16(X));
    R4D_NEED(dt_bf16(W));
    R4D_NEED(dt_bf16(Y));
    const long long n = r4d_geti_or(a, "n", r4d_dim(Y, (int)Y->rank - 1));
    if (n <= 0) return R4D_NO(RAD_E_SHAPE);
    long long M = 0, Mx = 0;
    const long long y_ld = r4d_rows_pitch(Y, n, &M);
    const long long x_ld = r4d_rows_pitch(X, n, &Mx);
    if (y_ld < 0 || x_ld < 0) return R4D_NO(RAD_E_STRIDE);
    if (M <= 0 || Mx < M) return R4D_NO(RAD_E_SHAPE);
    long long w_ld;
    if (r4d_numel(W) == 1) w_ld = -1;
    else if (r4d_numel(W) == n && r4d_contig(W)) w_ld = 0;
    else return R4D_NO(RAD_E_SHAPE);
    return r4d_binary_bf16(r4d_addr(X), r4d_addr(W), r4d_addr(Y), (int)M, (int)n, 1,
                           (int)x_ld, (int)w_ld, (int)y_ld,
                           (long)(uintptr_t)s) == 0 ? RAD_OK : RAD_E_DEVICE;
}

/* ------------------------------------------------------------------ sigmoid */

extern "C" int r4d_rad_sigmoid(const RadArgs* a, RadStream s) {
    const RadTensor* x = r4d_opt(a, 0);
    const RadTensor* y = r4d_opt(a, 1);
    if (!x || !y) return R4D_NO(RAD_E_INVAL);
    R4D_NEED(dt_bf16(x));
    R4D_NEED(dt_bf16(y));

    /* ROW-WISE, WITH A PITCH, because the input is usually a COLUMN SLICE. A gated attention block
     * issues this against the second half of each head of an interleaved [tokens, heads,
     * 2*head_dim] projection: `n` wide inside a stride of 2n, with no contiguous extent anywhere.
     * Treating it as a flat elementwise op refused it outright. */
    const long long n = r4d_geti_or(a, "n", r4d_dim(y, (int)y->rank - 1));
    if (n <= 0) return R4D_NO(RAD_E_SHAPE);
    long long M = 0, My = 0;
    const long long x_ld = r4d_rows_pitch(x, n, &M);
    const long long y_ld = r4d_rows_pitch(y, n, &My);
    if (x_ld < 0 || y_ld < 0) return R4D_NO(RAD_E_STRIDE);
    if (M <= 0 || My < M) return R4D_NO(RAD_E_SHAPE);

    return r4d_sigmoid_bf16(r4d_addr(x), r4d_addr(y), (int)M, (int)n, (int)x_ld, (int)y_ld,
                            (long)(uintptr_t)s) == 0 ? RAD_OK : RAD_E_DEVICE;
}

/* ------------------------------------------------------------------ fill */

extern "C" int r4d_rad_fill(const RadArgs* a, RadStream s) {
    const RadTensor* y = r4d_opt(a, 0);
    if (!y) return R4D_NO(RAD_E_INVAL);
    R4D_NEED(dt_bf16(y));
    const long long n = r4d_geti_or(a, "n", r4d_dim(y, (int)y->rank - 1));
    if (n <= 0) return R4D_NO(RAD_E_SHAPE);
    long long M = 0;
    const long long y_ld = r4d_rows_pitch(y, n, &M);
    if (y_ld < 0) return R4D_NO(RAD_E_STRIDE);
    if (M <= 0) return R4D_NO(RAD_E_SHAPE);
    const double v = r4d_getf(a, "value", 0.0);
    return r4d_fill_bf16(r4d_addr(y), (int)M, (int)n, (int)y_ld, (float)v,
                         (long)(uintptr_t)s) == 0 ? RAD_OK : RAD_E_DEVICE;
}

/* ------------------------------------------------------------------ logit_softcap */

extern "C" int r4d_rad_logit_softcap(const RadArgs* a, RadStream s) {
    const RadTensor* x = r4d_opt(a, 0);
    const RadTensor* y = r4d_opt(a, 1);
    if (!x || !y) return R4D_NO(RAD_E_INVAL);
    /* Two legs, one op: bf16 (the logits buffer as the gemm writes it on the bf16 chain) and f32
     * (the 1.3.0 sampling chain, where the gemm hands the sampler f32 rows). The dtype check
     * moved from the entry to the call below because the legs differ, not the geometry. */
    const bool leg_bf16 = (x->dtype == RAD_BF16) && (y->dtype == RAD_BF16);
    const bool leg_f32  = (x->dtype == RAD_F32)  && (y->dtype == RAD_F32);
    if (!leg_bf16 && !leg_f32) return R4D_NO(RAD_E_DTYPE);

    /* `cap` is where the model lives: Gemma 4 issues it at 30.0. The default exists for the same
     * reason every other one here does (a direct call with no declare in front), and the `> 0`
     * test is load-bearing: a zero cap is a division by zero in every lane, not a bad answer. */
    const float cap = (float)r4d_getf(a, "cap", 30.0);
    if (!(cap > 0.0f)) return R4D_NO(RAD_E_INVAL);

    /* ROW-WISE WITH A PITCH, same as sigmoid: issued against a column slice there is no contiguous
     * extent to walk, and the contiguous case costs nothing. */
    const long long n = r4d_geti_or(a, "n", r4d_dim(y, (int)y->rank - 1));
    if (n <= 0) return R4D_NO(RAD_E_SHAPE);
    long long M = 0, My = 0;
    const long long x_ld = r4d_rows_pitch(x, n, &M);
    const long long y_ld = r4d_rows_pitch(y, n, &My);
    if (x_ld < 0 || y_ld < 0) return R4D_NO(RAD_E_STRIDE);
    if (M <= 0 || My < M) return R4D_NO(RAD_E_SHAPE);

    const int rc = leg_bf16
        ? r4d_logit_softcap_bf16(r4d_addr(x), r4d_addr(y), (int)M, (int)n, cap,
                                 (int)x_ld, (int)y_ld, (long)(uintptr_t)s)
        : r4d_logit_softcap_f32(r4d_addr(x), r4d_addr(y), (int)M, (int)n, cap,
                                (int)x_ld, (int)y_ld, (long)(uintptr_t)s);
    return rc == 0 ? RAD_OK : RAD_E_DEVICE;
}

/* ------------------------------------------------------------------ scale */

extern "C" int r4d_rad_scale(const RadArgs* a, RadStream s) {
    const RadTensor* x = r4d_opt(a, 0);
    const RadTensor* sc = r4d_opt(a, 1);
    const RadTensor* y = r4d_opt(a, 2);
    if (!x || !sc || !y) return R4D_NO(RAD_E_INVAL);
    R4D_NEED(dt_bf16(x));
    R4D_NEED(dt_bf16(y));

    /* `s` is a [1] weight, f32 or bf16 (the layer_scalar weight is bf16 in the checkpoint). A
     * scalar with no element is a shape error, not a broadcast of zero -- a zero scalar would be
     * a plausible number, which is the thing the oracle rules out. Read on the device (the kernel
     * reads s[0] once per block) rather than a host copy: a device-to-host copy of one element is
     * a synchronisation on the step path, and the per-block read is cached. */
    if (sc->dtype != RAD_F32 && sc->dtype != RAD_BF16) return R4D_NO(RAD_E_DTYPE);
    if (r4d_numel(sc) < 1) return R4D_NO(RAD_E_SHAPE);

    /* ROW-WISE WITH A PITCH, same as sigmoid: issued against the residual stream at the end of a
     * layer there is no contiguous extent to walk, and the contiguous case costs nothing. */
    const long long n = r4d_geti_or(a, "n", r4d_dim(y, (int)y->rank - 1));
    if (n <= 0) return R4D_NO(RAD_E_SHAPE);
    long long M = 0, My = 0;
    const long long x_ld = r4d_rows_pitch(x, n, &M);
    const long long y_ld = r4d_rows_pitch(y, n, &My);
    if (x_ld < 0 || y_ld < 0) return R4D_NO(RAD_E_STRIDE);
    if (M <= 0 || My < M) return R4D_NO(RAD_E_SHAPE);

    return r4d_scale_bf16(r4d_addr(x), r4d_addr(sc), sc->dtype, r4d_addr(y), (int)M, (int)n,
                          (int)x_ld, (int)y_ld,
                          (long)(uintptr_t)s) == 0 ? RAD_OK : RAD_E_DEVICE;
}

/* ------------------------------------------------------------------ silu_mul */

extern "C" int r4d_rad_silu_mul(const RadArgs* a, RadStream s) {
    const RadTensor* gu = r4d_opt(a, 0);
    const RadTensor* y  = r4d_opt(a, 1);
    if (!gu || !y) return R4D_NO(RAD_E_INVAL);
    R4D_NEED(dt_bf16(gu));
    R4D_NEED(dt_bf16(y));

    const long long n = r4d_geti_or(a, "n", r4d_dim(y, (int)y->rank - 1));
    if (n <= 0) return R4D_NO(RAD_E_SHAPE);
    long long M = 0, Mg = 0;
    const long long y_ld  = r4d_rows_pitch(y, n, &M);
    const long long gu_ld = r4d_rows_pitch(gu, 2 * n, &Mg);
    if (y_ld < 0 || gu_ld < 0) return R4D_NO(RAD_E_STRIDE);
    if (M <= 0 || Mg < M) return R4D_NO(RAD_E_SHAPE);

    return r4d_silu_mul_bf16(r4d_addr(gu), r4d_addr(y), (int)M, (int)n, (int)gu_ld, (int)y_ld,
                             (long)(uintptr_t)s) == 0 ? RAD_OK : RAD_E_DEVICE;
}

/* ------------------------------------------------------------------ gelu_mul */

extern "C" int r4d_rad_gelu_mul(const RadArgs* a, RadStream s) {
    const RadTensor* gu = r4d_opt(a, 0);
    const RadTensor* y  = r4d_opt(a, 1);
    if (!gu || !y) return R4D_NO(RAD_E_INVAL);
    R4D_NEED(dt_bf16(gu));
    R4D_NEED(dt_bf16(y));

    const long long n = r4d_geti_or(a, "n", r4d_dim(y, (int)y->rank - 1));
    if (n <= 0) return R4D_NO(RAD_E_SHAPE);
    long long M = 0, Mg = 0;
    const long long y_ld  = r4d_rows_pitch(y, n, &M);
    const long long gu_ld = r4d_rows_pitch(gu, 2 * n, &Mg);
    if (y_ld < 0 || gu_ld < 0) return R4D_NO(RAD_E_STRIDE);
    if (M <= 0 || Mg < M) return R4D_NO(RAD_E_SHAPE);

    const int rc = r4d_gelu_mul_bf16(r4d_addr(gu), r4d_addr(y), (int)M, (int)n, (int)gu_ld, (int)y_ld,
                                     (long)(uintptr_t)s);
    if (rc != 0) return RAD_E_DEVICE;
    verify_gelu_mul((const uint16_t*)(uintptr_t)r4d_addr(gu), (const uint16_t*)(uintptr_t)r4d_addr(y),
                    (int)M, (int)n, (int)gu_ld, (int)y_ld, (long)(uintptr_t)s);
    return RAD_OK;
}

/* ------------------------------------------------------------------ rope */

enum { RP_QKV = 0, RP_POS };

extern "C" int r4d_rad_rope(const RadArgs* a, RadStream s) {
    const RadTensor* qkv = r4d_opt(a, RP_QKV);
    const RadTensor* pos = r4d_opt(a, RP_POS);
    if (!qkv || !pos) return R4D_NO(RAD_E_INVAL);
    R4D_NEED(dt_bf16(qkv));
    R4D_NEED(dt_i32(pos));
    if (!r4d_contig(pos)) return R4D_NO(RAD_E_STRIDE);

    const long long head_dim  = r4d_geti_or(a, "head_dim", 0);
    const long long n_head    = r4d_geti_or(a, "n_head", 0);
    const long long n_head_kv = r4d_geti_or(a, "n_head_kv", 0);
    const long long rot       = r4d_geti_or(a, "rotary_dim", head_dim);
    if (head_dim <= 0 || n_head <= 0) return R4D_NO(RAD_E_SHAPE);
    if (rot <= 0 || rot > head_dim || (rot & 1)) return R4D_NO(RAD_E_SHAPE);

    /* `ladder`: `rotary` (default) measures the base ladder in `rot`; `head` in `head_dim`, the
     * proportional variant. head_dim is already validated above, so `den = head_dim` is safe. */
    const char* lad = r4d_gets(a, "ladder");
    long long den = rot;
    if (lad) {
        if (!std::strcmp(lad, "head")) den = head_dim;
        else if (std::strcmp(lad, "rotary") && std::strcmp(lad, "head"))
            return R4D_NO(RAD_E_UNSUPPORTED);
    }
    /* `head` is HF's proportional RoPE: the rotation spans the WHOLE head (rotate_half over
     * head_dim, so pair i is (i, i + head_dim/2)) and only the first rotary_dim/2 frequencies are
     * non-zero. Rotating the first rotary_dim dims as a rotary of their own pairs (i, i + rot/2)
     * instead -- Gemma 4's global layers, 128 of 512 -- turns the wrong channels, and long-range
     * attention degrades with distance. */

    /* THE MODE, and with it how many components a position has. With [M] positions -- one
     * component a token -- mrope and imrope ARE neox: every frequency reads the same position and
     * the ladder is the same. With [C, M] positions and `sections`, each frequency reads the
     * component its section names (docs/OPS.md, `rope`). */
    const char* mode = r4d_gets(a, "mode");
    int gptj = 0, mc = 0;
    if (mode) {
        if (!std::strcmp(mode, "gptj")) gptj = 1;
        else if (!std::strcmp(mode, "mrope")) mc = 2;
        else if (!std::strcmp(mode, "imrope")) mc = 3;
        else if (!std::strcmp(mode, "axial")) mc = 4;
        else if (std::strcmp(mode, "neox")) return R4D_NO(RAD_E_SCHEMA);
    }
    int sec[3] = { 0, 0, 0 }, n_sec = 0;
    for (const char* v = r4d_gets(a, "sections"); v && *v;) {
        char* e = nullptr;
        const long x = std::strtol(v, &e, 10);
        if (e == v) break;
        if (x <= 0 || n_sec >= 3) return R4D_NO(RAD_E_SHAPE);
        sec[n_sec++] = (int)x;
        v = e;
    }

    const long long row = (n_head + 2 * n_head_kv) * head_dim;
    long long M = 0;
    const long long pitch = r4d_rows_pitch(qkv, row, &M);
    if (pitch < 0) return R4D_NO(RAD_E_STRIDE);
    const bool multi = pos->rank >= 2;
    const long long Mp = multi ? pos->shape[1] : r4d_numel(pos);
    if (Mp <= 0 || M < Mp) return R4D_NO(RAD_E_SHAPE);
    int plane = 0;
    if (multi) {
        if (!mc || n_sec == 0) return R4D_NO(RAD_E_SHAPE);
        if (pos->shape[0] < (mc == 3 ? 3 : n_sec)) return R4D_NO(RAD_E_SHAPE);
        if (mc == 3 && n_sec != 3) return R4D_NO(RAD_E_SHAPE);
        if (mc != 3 && [&] { int t = 0; for (int c = 0; c < n_sec; ++c) t += sec[c];
                             return t != (int)(rot / 2); }())
            return R4D_NO(RAD_E_SHAPE);
        plane = (int)pos->shape[1];
    }

    const float theta  = (float)r4d_getf(a, "theta", 10000.0);
    const float rscale = (float)r4d_getf(a, "scale", 1.0);
    const RadYarn yarn = rad_yarn_args(a);
    if (!rad_yarn_valid(yarn)) return R4D_NO(RAD_E_INVAL);
    /* YaRN over the axial vision rotary, or over a linear position divide, is no model's. */
    if (yarn.factor != 0.0f && (mc == 4 || rscale != 1.0f)) return R4D_NO(RAD_E_UNSUPPORTED);
    /* The Q heads and the K heads rotate; the V heads that follow them do not, which is the whole
     * reason this op takes the fused projection rather than two tensors. */
    long long rot_l = rot, nfreq = rot / 2;
    if (den == head_dim && rot < head_dim && !multi && !gptj && mc == 0) rot_l = head_dim;
    return rope_launch(r4d_addr(qkv), r4d_addr(pos), (int)Mp, (int)head_dim,
                       (int)(n_head + n_head_kv), (int)rot_l, (int)den, gptj, theta, rscale,
                       (int)pitch, mc, sec, n_sec, plane, yarn, (long)(uintptr_t)s,
                       (int)nfreq) == 0
        ? RAD_OK
        : RAD_E_DEVICE;
}

/* ------------------------------------------------------------------ kv_store */

enum { KS_K = 0, KS_V, KS_SLOT, KS_CACHE };

/* The geometry both stores share. The ONLY difference between them is the cache's dtype and what
 * the kernel does with a value on the way in, so everything up to the launch is one function --
 * two copies of a paged-layout check is how the two drift into disagreeing about the layout. */
struct KvStoreGeom {
    const RadTensor *k, *v, *sm, *kv;
    long long head_dim, block_size, kv_heads, n_blocks, k_ld, v_ld, T;
};

static int kv_store_geom(const RadArgs* a, KvStoreGeom* g) {
    g->k  = r4d_opt(a, KS_K);
    g->v  = r4d_opt(a, KS_V);
    g->sm = r4d_opt(a, KS_SLOT);
    g->kv = r4d_opt(a, KS_CACHE);
    if (!g->k || !g->v || !g->sm || !g->kv) return R4D_NO(RAD_E_INVAL);
    R4D_NEED(dt_bf16(g->k));
    R4D_NEED(dt_bf16(g->v));
    R4D_NEED(dt_i32(g->sm));
    if (!r4d_contig(g->sm) || !r4d_contig(g->kv)) return R4D_NO(RAD_E_STRIDE);

    g->head_dim   = r4d_geti_or(a, "head_dim", r4d_dim(g->k, (int)g->k->rank - 1));
    g->block_size = r4d_geti_or(a, "block_size", 0);
    g->kv_heads   = r4d_geti_or(a, "n_head_kv", g->kv->rank == 4 ? g->kv->shape[1] : 0);
    if (g->head_dim <= 0 || g->block_size <= 0 || g->kv_heads <= 0) return R4D_NO(RAD_E_SHAPE);
    /* The paged layout is [n_blocks, kv_heads, block_size, 2*head_dim] and every kernel in this
     * project shares it. A cache that disagrees is refused rather than walked with the parameters'
     * idea of the shape. */
    if (g->kv->rank != 4) return R4D_NO(RAD_E_SHAPE);
    if (g->kv->shape[1] != g->kv_heads || g->kv->shape[2] != g->block_size ||
        g->kv->shape[3] != 2 * g->head_dim) return R4D_NO(RAD_E_SHAPE);
    g->n_blocks = g->kv->shape[0];

    const long long lane = g->kv_heads * g->head_dim;
    long long M = 0, Mv = 0;
    g->k_ld = r4d_rows_pitch(g->k, lane, &M);
    g->v_ld = r4d_rows_pitch(g->v, lane, &Mv);
    if (g->k_ld < 0 || g->v_ld < 0) return R4D_NO(RAD_E_STRIDE);
    g->T = r4d_numel(g->sm);
    if (g->T <= 0 || M < g->T || Mv < g->T) return R4D_NO(RAD_E_SHAPE);
    return RAD_OK;
}

extern "C" int r4d_rad_kv_store(const RadArgs* a, RadStream s) {
    KvStoreGeom g{};
    const int rc = kv_store_geom(a, &g);
    if (rc != RAD_OK) return rc;
    R4D_NEED(dt_bf16(g.kv));

    return r4d_kv_store_bf16(r4d_addr(g.k), r4d_addr(g.v), r4d_addr(g.sm), r4d_addr(g.kv),
                             (int)g.T, (int)g.head_dim, (int)g.kv_heads, (int)g.block_size,
                             (int)g.n_blocks, (int)g.k_ld, (int)g.v_ld,
                             (long)(uintptr_t)s) == 0 ? RAD_OK : RAD_E_DEVICE;
}

/* The E4M3 twin. `k_scale` and `v_scale` are the DESCALES the attention side multiplies back by --
 * one number per (sequence, head) there, so a constant here -- and 1.0 is both vLLM's default and
 * what the fp8kv kernels assume when their descale operands are absent. */
static int rad_kv_store_fp8_body(const RadArgs* a, const R4dSink& sk) {
    KvStoreGeom g{};
    const int rc = kv_store_geom(a, &g);
    if (rc != RAD_OK) return rc;
    if (g.kv->dtype != RAD_F8E4M3) return R4D_NO(RAD_E_DTYPE);
    /* Four columns a thread is the packing the kernel uses, so a head must be a whole number of
     * groups; otherwise one group straddles two heads and writes the next head's first bytes. */
    if (g.head_dim % 4) return R4D_NO(RAD_E_SHAPE);

    const double ks = r4d_getf(a, "k_scale", 1.0);
    const double vs = r4d_getf(a, "v_scale", 1.0);
    if (!(ks > 0.0) || !(vs > 0.0)) return R4D_NO(RAD_E_INVAL);

    return kv_store_fp8_emit(r4d_addr(g.k), r4d_addr(g.v), r4d_addr(g.sm), r4d_addr(g.kv),
                             (int)g.T, (int)g.head_dim, (int)g.kv_heads, (int)g.block_size,
                             (int)g.n_blocks, (int)g.k_ld, (int)g.v_ld,
                             (float)ks, (float)vs, sk) == 0 ? RAD_OK : RAD_E_DEVICE;
}
R4D_ENTRY(r4d_rad_kv_store_fp8, rad_kv_store_fp8_body)

/* ------------------------------------------------------------------ logits_gemm
 *
 * ITS OWN KERNEL, AND THE OUTPUT DTYPE IS WHY. `logits_gemm` is `gemm_nt`'s arithmetic under the
 * vocabulary edge's parameter spelling -- `n_vocab` and `n_embd` where gemm_nt says N and K -- so
 * the skinny bf16 GEMM looks like the right target for it. That GEMM writes bf16. The engine
 * declares its logits buffer f32, because a bf16 logit has eight bits of mantissa and a nucleus
 * cut is a comparison of neighbouring probabilities, so that dispatch writes bf16 pairs into an
 * f32 plane: every logit wrong, the magnitudes roughly right, fluent nonsense out of the sampler.
 *
 * M is one row per SAMPLED position, not per token: `gather_rows` runs first and reduces a prefill
 * chunk to one row a sequence. At max_tok 8192 against a 248K vocabulary the ungathered logits
 * buffer alone would be 8 GiB, which is what that op is for.
 *
 */

enum { LG_X = 0, LG_W, LG_Y };

extern "C" int r4d_rad_logits_gemm(const RadArgs* a, RadStream s) {
    const RadTensor* x = r4d_opt(a, LG_X);
    const RadTensor* w = r4d_opt(a, LG_W);
    const RadTensor* y = r4d_opt(a, LG_Y);
    if (!x || !w || !y) return R4D_NO(RAD_E_INVAL);
    R4D_NEED(dt_bf16(x));
    R4D_NEED(dt_bf16(w));
    /* f32, and checked rather than assumed: the bf16 GEMM this op resembles writes bf16 into
     * exactly this buffer. See the note above. */
    if (y->dtype != RAD_F32) return R4D_NO(RAD_E_DTYPE);
    if (!r4d_contig(w)) return R4D_NO(RAD_E_STRIDE);

    const long long n_vocab = r4d_geti_or(a, "n_vocab", r4d_dim(w, 0));
    const long long n_embd  = r4d_geti_or(a, "n_embd", r4d_dim(w, 1));
    if (n_vocab <= 0 || n_embd <= 0) return R4D_NO(RAD_E_SHAPE);
    if (w->rank != 2 || w->shape[0] != n_vocab || w->shape[1] != n_embd)
        return R4D_NO(RAD_E_SHAPE);

    long long M = 0, My = 0;
    const long long x_ld = r4d_rows_pitch(x, n_embd, &M);
    const long long y_ld = r4d_rows_pitch(y, n_vocab, &My);
    if (x_ld < 0 || y_ld < 0) return R4D_NO(RAD_E_STRIDE);
    if (M <= 0 || My < M) return R4D_NO(RAD_E_SHAPE);
    if (!r4d_aligned(x->data, 16) || !r4d_aligned(w->data, 16)) return R4D_NO(RAD_E_ALIGN);

    return r4d_logits_gemm_bf16(r4d_addr(x), r4d_addr(w), r4d_addr(y), (int)M, (int)n_embd,
                                (int)n_vocab, (int)x_ld, (int)y_ld,
                                (long)(uintptr_t)s) == 0 ? RAD_OK : RAD_E_DEVICE;
}
