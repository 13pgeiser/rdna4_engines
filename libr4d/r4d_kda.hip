/* Exact recurrent baseline for GLM KDA, including speculative candidate snapshots.
 * One wave owns one, two or four value rows and keeps each row's 128-wide state in four
 * registers per lane.
 * Prefill uses the same recurrence and commits only its final state. */
#include <hip/hip_runtime.h>
#include "r4d_args.h"
#include "r4d_common.h"
#include "r4d_emit.h"
#include <climits>
#include <cstdlib>
#include <cstring>

namespace {
struct KdaMeta {
    int row[5], head[5];
    long os0, os1, ss0, ss1, ss2;
    long is0, is1, cs0, as0, hs0;
    int tokens, slots, width, verify, anchor;
    float lower, eps, scale;
};
struct KdaJob : KdaMeta {
    const uint16_t* input[5];
    const float *alog, *bias;
    float* state;
    uint16_t* out;
    const int *idx, *cu, *accepted, *init;
};

__device__ float sum(float v) {
    for (int d = 16; d; d >>= 1) v += __shfl_down(v, d, 32);
    return __shfl(v, 0, 32);
}
/* r4d_wave_sum adds the same partners in the same order as sum() does for lane 0, and every
 * other lane forms the same additions with the operands swapped, so all lanes hold lane 0's bits
 * without the broadcast. */
template <bool Dpp>
__device__ __forceinline__ float wave_total(float v) {
    if constexpr (Dpp)
        return r4d_wave_sum(v);
    else
        return sum(v);
}
__device__ long factor_offset(long base, int logical, long pitch) {
    return base + (logical / 128) * pitch + (logical % 128);
}
/* Separate dispatch: all old factor reads finish before the next window overwrites them. */
template <bool Dpp>
__global__ void kda_replay_kernel(float* state, const int* idx, const int* cu, const int* accepted,
                                  const int* init, KdaMeta m) {
    const int lane = threadIdx.x & 31, e = blockIdx.x * 4 + (threadIdx.x >> 5), head = blockIdx.y,
              seq = blockIdx.z;
    const int anchor = idx[(long)seq * m.is0],
              scratch = idx[(long)seq * m.is0 + (m.width / 2) * m.is1];
    if (anchor < 0 || anchor >= m.slots || scratch < 0 || scratch >= m.slots || anchor == scratch)
        return;
    const long fb = (long)scratch * m.ss0 + (long)head * m.ss1;
    const long sb = (long)anchor * m.ss0 + (long)head * m.ss1 + (long)e * m.ss2;
    if (init && !init[(long)seq * m.hs0]) {
#pragma unroll
        for (int d = 0; d < 4; ++d) state[sb + lane + d * 32] = 0;
        return;
    }
    const float count = state[fb];
    const int na = accepted ? accepted[(long)seq * m.as0] : 1;
    if (!(count >= 0 && count <= 32) || count != floorf(count) || na < 1 || na > 32 ||
        (count > 0 && na > count))
        return;
    float s[4];
#pragma unroll
    for (int d = 0; d < 4; ++d) s[d] = state[sb + lane + d * 32];
    for (int t = 0; t < (count > 0 ? na : 0); ++t) {
        const int f = 1 + t * 385;
        float k[4], prediction = 0;
#pragma unroll
        for (int d = 0; d < 4; ++d) {
            const int c = lane + d * 32;
            k[d] = state[factor_offset(fb, f + c, m.ss2)];
            s[d] *= state[factor_offset(fb, f + 256 + c, m.ss2)];
            prediction += s[d] * k[d];
        }
        const float delta =
            state[factor_offset(fb, f + 384, m.ss2)] *
            (state[factor_offset(fb, f + 128 + e, m.ss2)] - wave_total<Dpp>(prediction));
#pragma unroll
        for (int d = 0; d < 4; ++d) s[d] += delta * k[d];
    }
#pragma unroll
    for (int d = 0; d < 4; ++d) state[sb + lane + d * 32] = s[d];
}
__global__ void kda_update_kernel(const uint16_t* q, const uint16_t* k, const uint16_t* v,
                                  const uint16_t* g, const uint16_t* beta, const float* alog,
                                  const float* bias, float* state_ptr, uint16_t* out,
                                  const int* idx, const int* cu, const int* accepted_ptr,
                                  const int* init, KdaMeta meta) {
    KdaJob j{};
    static_cast<KdaMeta&>(j) = meta;
    j.input[0] = q;
    j.input[1] = k;
    j.input[2] = v;
    j.input[3] = g;
    j.input[4] = beta;
    j.alog = alog;
    j.bias = bias;
    j.state = state_ptr;
    j.out = out;
    j.idx = idx;
    j.cu = cu;
    j.accepted = accepted_ptr;
    j.init = init;
    const int lane = threadIdx.x & 31, e = blockIdx.x * 4 + (threadIdx.x >> 5);
    const int head = blockIdx.y, seq = blockIdx.z;
    const int begin = j.cu[(long)seq * j.cs0], end = j.cu[(long)(seq + 1) * j.cs0];
    const int accepted = j.accepted ? j.accepted[(long)seq * j.as0] : 1;
    if (begin < 0 || end < begin || end > j.tokens || accepted < 1 ||
        accepted > (j.anchor ? 32 : j.width) ||
        (j.verify && end - begin > (j.anchor ? 32 : j.width)))
        return;
    const int read = j.idx[(long)seq * j.is0 + (j.verify && !j.anchor ? accepted - 1 : 0) * j.is1];
    if (read >= j.slots) return;
    if (read < 0) {
        if (!lane)
            for (int t = begin; t < end; ++t) j.out[(long)t * j.os0 + (long)head * j.os1 + e] = 0;
        return;
    }
    const bool initialized = !j.init || j.init[(long)seq * j.hs0] != 0;
    const int scratch = j.anchor ? j.idx[(long)seq * j.is0 + (j.width / 2) * j.is1] : 0;
    if (j.anchor && (scratch < 0 || scratch >= j.slots || scratch == read)) return;
    const long fb = (long)scratch * j.ss0 + (long)head * j.ss1;
    const long initial = (long)read * j.ss0 + (long)head * j.ss1 + (long)e * j.ss2;
    float state[4];
#pragma unroll
    for (int d = 0; d < 4; ++d) state[d] = initialized ? j.state[initial + lane + 32 * d] : 0;
    const float gain = expf(j.alog[head]);
    for (int t = begin; t < end; ++t) {
        float q[4], k[4], decay[4], sq = 0, sk = 0;
#pragma unroll
        for (int d = 0; d < 4; ++d) {
            const int channel = lane + d * 32;
            q[d] = bf16_to_f32(j.input[0][(long)t * j.row[0] + (long)head * j.head[0] + channel]);
            k[d] = bf16_to_f32(j.input[1][(long)t * j.row[1] + (long)head * j.head[1] + channel]);
            sq += q[d] * q[d];
            sk += k[d] * k[d];
            const float raw =
                bf16_to_f32(j.input[3][(long)t * j.row[3] + (long)head * j.head[3] + channel]);
            decay[d] = expf(j.lower / (1.0f + expf(-gain * (raw + j.bias[head * 128 + channel]))));
        }
        const float qi = j.scale * rsqrtf(sum(sq) + j.eps), ki = rsqrtf(sum(sk) + j.eps);
        const float beta =
            1.0f /
            (1.0f + expf(-bf16_to_f32(j.input[4][(long)t * j.row[4] + (long)head * j.head[4]])));
        float predicted = 0;
#pragma unroll
        for (int d = 0; d < 4; ++d) {
            q[d] *= qi;
            k[d] *= ki;
            state[d] *= decay[d];
            predicted += state[d] * k[d];
        }
        predicted = sum(predicted);
        const float value =
            bf16_to_f32(j.input[2][(long)t * j.row[2] + (long)head * j.head[2] + e]);
        if (j.anchor && j.verify) {
            const int f = 1 + (t - begin) * 385;
            if (e == 0) {
#pragma unroll
                for (int d = 0; d < 4; ++d) {
                    const int c = lane + 32 * d;
                    j.state[factor_offset(fb, f + c, j.ss2)] = k[d];
                    j.state[factor_offset(fb, f + 256 + c, j.ss2)] = decay[d];
                }
                if (!lane) j.state[factor_offset(fb, f + 384, j.ss2)] = beta;
            }
            if (!lane) j.state[factor_offset(fb, f + 128 + e, j.ss2)] = value;
        }
        const float delta = beta * (value - predicted);
        float result = 0;
#pragma unroll
        for (int d = 0; d < 4; ++d) {
            state[d] += delta * k[d];
            result += state[d] * q[d];
        }
        result = sum(result);
        if (!lane) j.out[(long)t * j.os0 + (long)head * j.os1 + e] = f32_to_bf16(result);
        if ((!j.anchor && j.verify) || (!j.verify && t + 1 == end)) {
            const int write = j.idx[(long)seq * j.is0 + (j.verify ? t - begin : 0) * j.is1];
            if (write >= 0 && write < j.slots) {
                const long target = (long)write * j.ss0 + (long)head * j.ss1 + (long)e * j.ss2;
#pragma unroll
                for (int d = 0; d < 4; ++d) j.state[target + lane + 32 * d] = state[d];
            }
        }
    }
    if (j.anchor && e == 0 && !lane) j.state[fb] = j.verify ? (float)(end - begin) : 0.0f;
}
/* kda_update_kernel with R value rows a wave: the loads, norms and decay exponentials, which do
 * not depend on the state, are formed once for R rows, and the rows' reductions interleave. Each
 * row's arithmetic is kda_update_kernel's in its order, so the bytes match (selftest --perfkda).
 * With Prefetch the next token's operands are loaded while this one is reduced. */
template <int R, bool Prefetch>
__global__ void kda_rows_kernel(const uint16_t* q, const uint16_t* k, const uint16_t* v,
                                const uint16_t* g, const uint16_t* beta, const float* alog,
                                const float* bias, float* state_ptr, uint16_t* out, const int* idx,
                                const int* cu, const int* accepted_ptr, const int* init,
                                KdaMeta meta) {
    static_assert(32 % R == 0, "the 128 value rows split evenly over the waves");
    KdaJob j{};
    static_cast<KdaMeta&>(j) = meta;
    j.input[0] = q;
    j.input[1] = k;
    j.input[2] = v;
    j.input[3] = g;
    j.input[4] = beta;
    j.alog = alog;
    j.bias = bias;
    j.state = state_ptr;
    j.out = out;
    j.idx = idx;
    j.cu = cu;
    j.accepted = accepted_ptr;
    j.init = init;
    const int lane = threadIdx.x & 31, e0 = (blockIdx.x * 4 + (threadIdx.x >> 5)) * R;
    const int head = blockIdx.y, seq = blockIdx.z;
    const int begin = j.cu[(long)seq * j.cs0], end = j.cu[(long)(seq + 1) * j.cs0];
    const int accepted = j.accepted ? j.accepted[(long)seq * j.as0] : 1;
    if (begin < 0 || end < begin || end > j.tokens || accepted < 1 ||
        accepted > (j.anchor ? 32 : j.width) ||
        (j.verify && end - begin > (j.anchor ? 32 : j.width)))
        return;
    const int read = j.idx[(long)seq * j.is0 + (j.verify && !j.anchor ? accepted - 1 : 0) * j.is1];
    if (read >= j.slots) return;
    uint16_t* const out_row = j.out + (long)head * j.os1 + e0;
    if (read < 0) {
        if (lane < R)
            for (int t = begin; t < end; ++t) out_row[(long)t * j.os0 + lane] = 0;
        return;
    }
    const bool initialized = !j.init || j.init[(long)seq * j.hs0] != 0;
    const int scratch = j.anchor ? j.idx[(long)seq * j.is0 + (j.width / 2) * j.is1] : 0;
    if (j.anchor && (scratch < 0 || scratch >= j.slots || scratch == read)) return;
    const long fb = (long)scratch * j.ss0 + (long)head * j.ss1;
    const long initial = (long)read * j.ss0 + (long)head * j.ss1 + (long)e0 * j.ss2;
    float state[R][4];
#pragma unroll
    for (int r = 0; r < R; ++r)
#pragma unroll
        for (int d = 0; d < 4; ++d)
            state[r][d] = initialized ? j.state[initial + r * j.ss2 + lane + 32 * d] : 0;
    const float gain = expf(j.alog[head]);
    float shift[4];
#pragma unroll
    for (int d = 0; d < 4; ++d) shift[d] = j.bias[head * 128 + lane + 32 * d];
    const uint16_t* const qb = j.input[0] + (long)head * j.head[0] + lane;
    const uint16_t* const kb = j.input[1] + (long)head * j.head[1] + lane;
    const uint16_t* const vb = j.input[2] + (long)head * j.head[2] + e0;
    const uint16_t* const gb = j.input[3] + (long)head * j.head[3] + lane;
    const uint16_t* const bb = j.input[4] + (long)head * j.head[4];
    struct Token {
        uint32_t q[4], k[4], g[4], beta, value[R];
    };
    auto fetch = [&](int t, Token& o) {
#pragma unroll
        for (int d = 0; d < 4; ++d) {
            o.q[d] = ldg_u16(qb + (long)t * j.row[0] + 32 * d);
            o.k[d] = ldg_u16(kb + (long)t * j.row[1] + 32 * d);
            o.g[d] = ldg_u16(gb + (long)t * j.row[3] + 32 * d);
        }
        o.beta = ldg_u16(bb + (long)t * j.row[4]);
#pragma unroll
        for (int r = 0; r < R; ++r) o.value[r] = ldg_u16(vb + (long)t * j.row[2] + r);
    };
    Token cur;
    if (Prefetch && begin < end) fetch(begin, cur);
    for (int t = begin; t < end; ++t) {
        [[maybe_unused]] Token next;
        /* The last token loads itself again rather than a row past `end`. */
        if constexpr (Prefetch)
            fetch(t + 1 < end ? t + 1 : t, next);
        else
            fetch(t, cur);
        float qv[4], kv[4], decay[4], sq = 0, sk = 0;
#pragma unroll
        for (int d = 0; d < 4; ++d) {
            qv[d] = bf16w_lo(cur.q[d]);
            kv[d] = bf16w_lo(cur.k[d]);
            sq += qv[d] * qv[d];
            sk += kv[d] * kv[d];
            const float raw = bf16w_lo(cur.g[d]);
            decay[d] = expf(j.lower / (1.0f + expf(-gain * (raw + shift[d]))));
        }
        const float qi = j.scale * rsqrtf(r4d_wave_sum(sq) + j.eps),
                    ki = rsqrtf(r4d_wave_sum(sk) + j.eps);
        const float b = 1.0f / (1.0f + expf(-bf16w_lo(cur.beta)));
        float predicted[R];
#pragma unroll
        for (int r = 0; r < R; ++r) predicted[r] = 0;
#pragma unroll
        for (int d = 0; d < 4; ++d) {
            qv[d] *= qi;
            kv[d] *= ki;
#pragma unroll
            for (int r = 0; r < R; ++r) {
                state[r][d] *= decay[d];
                predicted[r] += state[r][d] * kv[d];
            }
        }
        float value[R];
#pragma unroll
        for (int r = 0; r < R; ++r) {
            predicted[r] = r4d_wave_sum(predicted[r]);
            value[r] = bf16w_lo(cur.value[r]);
        }
        if (j.anchor && j.verify) {
            const int f = 1 + (t - begin) * 385;
            if (e0 == 0) {
#pragma unroll
                for (int d = 0; d < 4; ++d) {
                    const int c = lane + 32 * d;
                    j.state[factor_offset(fb, f + c, j.ss2)] = kv[d];
                    j.state[factor_offset(fb, f + 256 + c, j.ss2)] = decay[d];
                }
                if (!lane) j.state[factor_offset(fb, f + 384, j.ss2)] = b;
            }
            if (!lane) {
#pragma unroll
                for (int r = 0; r < R; ++r)
                    j.state[factor_offset(fb, f + 128 + e0 + r, j.ss2)] = value[r];
            }
        }
        float result[R];
#pragma unroll
        for (int r = 0; r < R; ++r) {
            const float delta = b * (value[r] - predicted[r]);
            result[r] = 0;
#pragma unroll
            for (int d = 0; d < 4; ++d) {
                state[r][d] += delta * kv[d];
                result[r] += state[r][d] * qv[d];
            }
        }
        float mine = 0;
#pragma unroll
        for (int r = 0; r < R; ++r) {
            result[r] = r4d_wave_sum(result[r]);
            if (lane == r) mine = result[r];
        }
        if (lane < R) out_row[(long)t * j.os0 + lane] = f32_to_bf16(mine);
        if ((!j.anchor && j.verify) || (!j.verify && t + 1 == end)) {
            const int write = j.idx[(long)seq * j.is0 + (j.verify ? t - begin : 0) * j.is1];
            if (write >= 0 && write < j.slots) {
                const long target = (long)write * j.ss0 + (long)head * j.ss1 + (long)e0 * j.ss2;
#pragma unroll
                for (int r = 0; r < R; ++r)
#pragma unroll
                    for (int d = 0; d < 4; ++d)
                        j.state[target + r * j.ss2 + lane + 32 * d] = state[r][d];
            }
        }
        if constexpr (Prefetch) cur = next;
    }
    if (j.anchor && e0 == 0 && !lane) j.state[fb] = j.verify ? (float)(end - begin) : 0.0f;
}

template <class Pitch>
bool headed(const RadTensor& t, long tokens, long heads, long dim, Pitch* row, Pitch* head) {
    if (!t.data || t.dtype != RAD_BF16 || t.shape[0] != tokens || t.stride[0] > INT_MAX ||
        (t.rank == 3 && t.stride[1] > INT_MAX))
        return false;
    if (t.rank == 3 && t.shape[1] == heads && t.shape[2] == dim && t.stride[2] == 1 &&
        t.stride[1] >= dim && t.stride[0] >= heads * t.stride[1]) {
        *row = t.stride[0];
        *head = t.stride[1];
        return true;
    }
    if (t.rank == 2 && t.shape[1] == heads * dim && t.stride[1] == 1 &&
        t.stride[0] >= heads * dim) {
        *row = t.stride[0];
        *head = dim;
        return true;
    }
    return false;
}
bool vector(const RadTensor& t, uint32_t dt, long n) {
    return t.data && t.dtype == dt && t.rank == 1 && t.shape[0] >= n && t.stride[0] == 1;
}
int body(const RadArgs* a, RadStream stream) {
    if (!a || a->n_t < 13) return RAD_E_INVAL;
    const long H = r4d_geti_or(a, "heads", 0);
    if (H < 1 || H > 65535 || r4d_geti_or(a, "head_k", 0) != 128 ||
        r4d_geti_or(a, "head_v", 0) != 128)
        return RAD_E_SHAPE;
    KdaJob j{};
    const auto &state = a->t[7], &idx = a->t[8], &cu = a->t[9];
    const long tokens = a->t[0].shape[0];
    if (tokens < 1 || tokens > INT_MAX || state.rank != 4 || state.dtype != RAD_F32 ||
        !state.data || state.shape[0] < 1 || state.shape[0] > INT_MAX || state.shape[1] != H ||
        state.shape[2] != 128 || state.shape[3] != 128 || state.stride[3] != 1 ||
        state.stride[2] < 128 || state.stride[1] < 128 * state.stride[2] ||
        state.stride[0] < H * state.stride[1] || cu.rank != 1 || cu.dtype != RAD_I32 || !cu.data ||
        cu.shape[0] < 2 || cu.shape[0] > 65536 || cu.stride[0] < 1 || idx.rank != 2 ||
        idx.dtype != RAD_I32 || !idx.data || idx.shape[0] < cu.shape[0] - 1 || idx.shape[1] < 1 ||
        idx.shape[1] > INT_MAX || idx.stride[1] < 1 || idx.stride[0] < idx.shape[1] * idx.stride[1])
        return RAD_E_SHAPE;
    for (int i = 0; i < 5; ++i) {
        if (!headed(a->t[i], tokens, H, i == 4 ? 1 : 128, &j.row[i], &j.head[i]))
            return RAD_E_SHAPE;
        j.input[i] = (const uint16_t*)a->t[i].data;
    }
    if (!headed(a->t[12], tokens, H, 128, &j.os0, &j.os1) || !vector(a->t[5], RAD_F32, H) ||
        !vector(a->t[6], RAD_F32, H * 128))
        return RAD_E_SHAPE;
    const RadTensor *accepted = rad_arg_in(a, 10), *init = rad_arg_in(a, 11);
    if ((accepted && !vector(*accepted, RAD_I32, cu.shape[0] - 1)) ||
        (init && !vector(*init, RAD_I32, cu.shape[0] - 1)))
        return RAD_E_SHAPE;
    const char* mode = r4d_gets(a, "mode");
    if (!mode || (std::strcmp(mode, "verify") && std::strcmp(mode, "prefill"))) return RAD_E_INVAL;
    j.verify = !std::strcmp(mode, "verify");
    const char* form = r4d_gets(a, "state_form");
    j.anchor = form && !std::strcmp(form, "anchor");
    if (form && !j.anchor && std::strcmp(form, "per_candidate")) return RAD_E_INVAL;
    if (j.anchor && idx.shape[1] < 2) return RAD_E_SHAPE;
    j.lower = (float)r4d_getf(a, "lower_bound", 0);
    j.eps = (float)r4d_getf(a, "l2_eps", 0);
    j.scale = (float)r4d_getf(a, "scale", 1);
    if (!std::isfinite(j.lower) || j.lower >= 0 || !std::isfinite(j.eps) || j.eps <= 0 ||
        !std::isfinite(j.scale))
        return RAD_E_SHAPE;
    j.tokens = (int)tokens;
    j.slots = (int)state.shape[0];
    j.width = (int)idx.shape[1];
    j.alog = (const float*)a->t[5].data;
    j.bias = (const float*)a->t[6].data;
    j.state = (float*)state.data;
    j.out = (uint16_t*)a->t[12].data;
    j.ss0 = state.stride[0];
    j.ss1 = state.stride[1];
    j.ss2 = state.stride[2];
    j.idx = (const int*)idx.data;
    j.is0 = idx.stride[0];
    j.is1 = idx.stride[1];
    j.cu = (const int*)cu.data;
    j.cs0 = cu.stride[0];
    if (accepted) {
        j.accepted = (const int*)accepted->data;
        j.as0 = accepted->stride[0];
    }
    if (init) {
        j.init = (const int*)init->data;
        j.hs0 = init->stride[0];
    }
    /* RADIANCE_KDA_FORM=old is the one-row kernel on shuffle sums, the reference the selftest
     * holds every form to byte for byte; RADIANCE_KDA_ROWS (1, 2 or 4 value rows a wave) and
     * RADIANCE_KDA_PREFETCH=0 pin the other form's variants for the microbenchmark. */
    const char* kform = std::getenv("RADIANCE_KDA_FORM");
    const bool old = kform && !std::strcmp(kform, "old");
    const unsigned seqs = (unsigned)(cu.shape[0] - 1);
    if (j.anchor && j.verify) {
        const int rc =
            old ? R4D_EMIT(stream, kda_replay_kernel<false>, dim3(32, (unsigned)H, seqs), dim3(128),
                           0, j.state, j.idx, j.cu, j.accepted, j.init, static_cast<KdaMeta>(j))
                : R4D_EMIT(stream, kda_replay_kernel<true>, dim3(32, (unsigned)H, seqs), dim3(128),
                           0, j.state, j.idx, j.cu, j.accepted, j.init, static_cast<KdaMeta>(j));
        if (rc != RAD_OK) return rc;
    }
    if (old)
        return R4D_EMIT(stream, kda_update_kernel, dim3(32, (unsigned)H, seqs), dim3(128), 0,
                        j.input[0], j.input[1], j.input[2], j.input[3], j.input[4], j.alog, j.bias,
                        j.state, j.out, j.idx, j.cu, j.accepted, j.init, static_cast<KdaMeta>(j));
    const char* rows_env = std::getenv("RADIANCE_KDA_ROWS");
    const long rows = rows_env && *rows_env ? std::strtol(rows_env, nullptr, 10) : 4;
    const char* prefetch_env = std::getenv("RADIANCE_KDA_PREFETCH");
    const bool prefetch = !prefetch_env || prefetch_env[0] != '0';
#define KDA_ROWS(RR, PF)                                                                           \
    return R4D_EMIT(stream, (kda_rows_kernel<RR, PF>), dim3(32 / RR, (unsigned)H, seqs),           \
                    dim3(128), 0, j.input[0], j.input[1], j.input[2], j.input[3], j.input[4],      \
                    j.alog, j.bias, j.state, j.out, j.idx, j.cu, j.accepted, j.init,               \
                    static_cast<KdaMeta>(j))
    if (rows == 1) {
        if (prefetch) KDA_ROWS(1, true);
        KDA_ROWS(1, false);
    }
    if (rows == 2) {
        if (prefetch) KDA_ROWS(2, true);
        KDA_ROWS(2, false);
    }
    if (prefetch) KDA_ROWS(4, true);
    KDA_ROWS(4, false);
#undef KDA_ROWS
}
}
extern "C" int r4d_rad_kda_update(const RadArgs* a, RadStream stream) { return body(a, stream); }
