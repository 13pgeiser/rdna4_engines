/* Absorbed MLA: one latent vector is both the attention key and value. The
 * per-query table already contains the selected complete pages plus its tail.
 * `scale` comes from the original query/key head width, NOT the latent width. */
#include "ref_common.h"
#include "ref_ops.h"
#include <algorithm>
#include <cmath>
using namespace ref;

/* Both projections share the checkpoint's [head,(K|V),latent] matrix.
 * Absorb: q @ Wk. Expand: attention_latent @ Wv^T. No per-head KV cache. */
int ref_mla_project(const RadArgs* a, RadStream) {
    if (!have(a, 3)) return RAD_E_INVAL;
    RadTensor x = a->t[0], y = a->t[2];
    const auto& w = a->t[1];
    const int64_t h = p_int(a, "heads", 0), d = p_int(a, "head_dim", 0),
                  l = p_int(a, "latent_dim", 0);
    const char* mode = p_str(a, "mode", "");
    const bool absorb = !std::strcmp(mode, "absorb"), expand = !std::strcmp(mode, "expand");
    auto heads_view = [&](RadTensor& t, int64_t width) {
        if (h > 0 && width > 0 && t.rank == 2 && t.shape[1] == h * width) {
            t.rank = 3;
            t.shape[1] = h;
            t.shape[2] = width;
            t.stride[2] = t.stride[1];
            t.stride[1] *= width;
        }
    };
    heads_view(x, absorb ? d : l);
    heads_view(y, absorb ? l : d);
    if ((!absorb && !expand) || h < 1 || d < 1 || l < 1 || x.rank != 3 || w.rank != 2 ||
        y.rank != 3 || x.shape[0] < 1 || x.shape[1] != h || x.shape[2] != (absorb ? d : l) ||
        w.shape[0] != h * 2 * d || w.shape[1] != l || y.shape[0] != x.shape[0] || y.shape[1] != h ||
        y.shape[2] != (absorb ? l : d))
        return RAD_E_SHAPE;
    if (x.dtype != RAD_BF16 || w.dtype != RAD_BF16 || y.dtype != RAD_BF16) return RAD_E_DTYPE;
    if (!x.data || !w.data || !y.data) return RAD_E_INVAL;
#pragma omp parallel for schedule(static)
    for (int64_t r = 0; r < x.shape[0]; ++r)
        for (int64_t head = 0; head < h; ++head)
            for (int64_t col = 0; col < (absorb ? l : d); ++col) {
                float sum = 0;
                for (int64_t j = 0; j < (absorb ? d : l); ++j) {
                    const int64_t wr = head * 2 * d + (absorb ? j : d + col), wc = absorb ? col : j;
                    sum += ldt(&x, r * x.stride[0] + head * x.stride[1] + j * x.stride[2]) *
                           ldt(&w, wr * w.stride[0] + wc * w.stride[1]);
                }
                stt(&y, r * y.stride[0] + head * y.stride[1] + col * y.stride[2], sum);
            }
    return RAD_OK;
}

int ref_mla_attn(const RadArgs* a, RadStream) {
    if (!have(a, 5)) return RAD_E_INVAL;
    const auto &q = a->t[0], &kv = a->t[1], &bt = a->t[2], &len = a->t[3], &out = a->t[4];
    const int64_t d = p_int(a, "latent_dim", 0), h = p_int(a, "heads", 0),
                  bs = p_int(a, "block_size", 0);
    const float scale = p_f32(a, "scale", 0);
    if (d < 1 || d > 2048 || h < 1 || bs < 1 || !std::isfinite(scale) || scale <= 0 ||
        q.rank != 3 || q.shape[0] < 1 || q.shape[1] != h || q.shape[2] != d || kv.rank != 4 ||
        kv.shape[1] != 1 || kv.shape[2] != bs || kv.shape[3] != d || bt.rank != 2 ||
        bt.shape[0] != q.shape[0] || len.rank != 1 || len.shape[0] != q.shape[0] || out.rank != 3 ||
        out.shape[0] != q.shape[0] || out.shape[1] != h || out.shape[2] != d ||
        bt.dtype != RAD_I32 || len.dtype != RAD_I32)
        return RAD_E_SHAPE;
    for (int i = 0; i < 5; ++i)
        if (!a->t[i].data) return RAD_E_INVAL;
#pragma omp parallel for schedule(static)
    for (int64_t job = 0; job < q.shape[0] * h; ++job) {
        const int64_t r = job / h, head = job % h;
        const int64_t live = std::clamp(ld_int(len.dtype, len.data, r * len.stride[0]), (int64_t)0,
                                        bt.shape[1] * bs);
        auto key_offset = [&](int64_t token) -> int64_t {
            const int64_t page =
                ld_int(bt.dtype, bt.data, r * bt.stride[0] + (token / bs) * bt.stride[1]);
            return page < 0 || page >= kv.shape[0]
                       ? -1
                       : page * kv.stride[0] + (token % bs) * kv.stride[2];
        };
        auto dot = [&](int64_t base) -> float {
            float v = 0;
            for (int64_t c = 0; c < d; ++c)
                v += ldt(&q, r * q.stride[0] + head * q.stride[1] + c * q.stride[2]) *
                     ldt(&kv, base + c * kv.stride[3]);
            return v * scale;
        };
        float max = -INFINITY;
        for (int64_t j = 0; j < live; ++j) {
            const auto base = key_offset(j);
            if (base >= 0) max = std::max(max, dot(base));
        }
        float val[2048] = {}, den = 0;
        for (int64_t j = 0; j < live; ++j) {
            const auto base = key_offset(j);
            if (base < 0) continue;
            const float w = std::exp(dot(base) - max);
            den += w;
            for (int64_t c = 0; c < d; ++c) val[c] += w * ldt(&kv, base + c * kv.stride[3]);
        }
        for (int64_t c = 0; c < d; ++c)
            stt(&out, r * out.stride[0] + head * out.stride[1] + c * out.stride[2],
                den > 0 ? val[c] / den : 0);
    }
    return RAD_OK;
}
