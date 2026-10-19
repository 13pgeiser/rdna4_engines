/* GLM's replicated F32 work split over the tensor-parallel ranks: each rank computes a disjoint
 * part with the replicated kernel's own arithmetic and an all_gather moves the parts to every rank,
 * so every value is bit-identical to the replicated computation's. */
#ifndef RAD_TP_SPLIT_H
#define RAD_TP_SPLIT_H
#include "rad_arch.h"
#include <cstdlib>

namespace rad {
namespace arch {
/* Rows from which a replicated F32 projection splits its columns over the ranks
 * (RADIANCE_GLM_TP_SPLIT_ROWS, 0 never). Below it the collective's latency outweighs the
 * saved arithmetic, so decode keeps the replicated form. */
inline int64_t glm_tp_split_rows() {
    const char* e = std::getenv("RADIANCE_GLM_TP_SPLIT_ROWS");
    if (!e || !*e) return 256;
    const long long v = std::strtoll(e, nullptr, 10);
    return v > 0 ? (int64_t)v : 0;
}

/* The split's ops are declared whatever the step size, and each gather's message as a range of
 * rows: the engine's arena levels declare again at a smaller max_tok and must get the same graph
 * with the same fixed parameters.
 *
 * A replicated [n, k] F32 projection computed column-split: rank r runs the same gemm_nt_f32 on
 * rows [r*n/world, (r+1)*n/world) of the weight and a row-interleaved all_gather lays the slices
 * out as the full [M, n]. The kernel computes every output column independently of n and the
 * gather moves bytes, so each value is bit-identical to the replicated projection's. */
struct SplitF32Proj {
    rad_buf part = 0;
    rad_op op_part = 0, op_gather = 0;
    int64_t cols = 0, k = 0, min_rows = 0, first_col = 0;

    int declare(RadBuilder* b, Names& nm, const Geom& g, const char* name, int64_t rows, int64_t n,
                int64_t width, rad_weight weight, rad_buf input, rad_buf output) {
        min_rows = glm_tp_split_rows();
        if (g.world < 2 || !min_rows || n % g.world || (n / g.world * 4) % 16) {
            min_rows = 0;
            return RAD_OK;
        }
        cols = n / g.world;
        k = width;
        first_col = g.rank * cols;
        part = decl_b(b, nm.f("%s.part", name), RAD_F32, { rows, cols });
        op_part = rw(b,
                     RAD_OP(b, "gemm_nt_f32",
                            RAD_PARAMS(RAD_RANGE("M", 1, rows), RAD_INT("N", cols), RAD_INT("K", k),
                                       RAD_STR("dtype", "bf16")),
                            RAD_WEIGHTS(weight)),
                     { input }, { part });
        op_gather = rw(
            b,
            RAD_OP(b, "all_gather",
                   RAD_PARAMS(RAD_INT("world_size", g.world), RAD_RANGE("numel", cols, rows * cols),
                              RAD_STR("dtype", "f32"), RAD_INT("row", cols)),
                   RAD_NOWEIGHTS),
            { part }, { output });
        return part && op_part && op_gather ? RAD_OK : RAD_E_INVAL;
    }
    bool takes(int64_t rows) const { return min_rows && rows >= min_rows; }
    void step(RadCtx* ctx, int64_t rows, RadOperand input, rad_weight weight,
              RadOperand output) const {
        RadOperand slice = RAD_W(weight);
        slice.offset = first_col * k;
        slice.rows = cols;
        RAD_ISSUE_N(ctx, op_part, rows, input, slice, brows(part, rows));
        RAD_ISSUE_N(ctx, op_gather, rows * cols, brows(part, rows), output);
    }
};
}
} // namespace rad::arch
#endif
