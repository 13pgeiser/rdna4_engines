/* lq_gptq.h -- error-feedback quantisation on any grid, against a calibration Hessian.
 *
 * GPTQ picks the codes one column at a time and pushes each column's rounding error into the
 * columns not yet reached, weighted by the inverse Hessian of the layer's real input. Plain
 * rounding treats every column as independent, and a 4-bit grid lacks the levels to absorb that.
 *
 * This file holds the Hessian and the loop, not the grid: codes are chosen on whatever grid the
 * rule names (lq_grid.h), against scales fixed before the loop from the (transformed) weight. This
 * "static groups" form keeps a scale plane a function of the weight alone.
 *
 * The Hessian is the expert's own where the tap recorded one, else the layer's (lq_hess.h,
 * calib_resolve). A layer's pooled Hessian is found from the weight's logical name with the expert
 * index stripped, so its experts share one factor; an expert's own is factored for its one call.
 * The engine's calibration tap writes both (arch/common/rad_arch.h, RADIANCE_CALIB_DIR);
 * docs/MOE-W4.md describes the bootstrap.
 */
#pragma once
#include "lq_grid.h"
#include "lq_hess.h"

namespace lq {

/* Upper Cholesky factor U of the damped inverse Hessian, U^T U = (H + lambda I)^-1, as a K x K
 * row-major f32 plane, for weight `name`. Null when `dir` has no calibration for it (not an error:
 * the weight is then rounded to nearest). Also null, with a message, when a file exists but is
 * unusable, so a mistyped directory is not a silent quality loss. With `perm` (gptq_act_order's)
 * it factors the permuted Hessian P H P^T. `domain` is the grid's transform width, 0 without one
 * (calib_sum). A layer's factor is cached for the process by (dir, stem, K, domain, damp, act
 * order); an expert's own is not. */
Gram gptq_factor(const char* dir, const char* name, int64_t K, int64_t domain, double damp,
                 const int32_t* perm = nullptr);

/* Act order: columns by descending Hessian diagonal (the input energy each carries), ties in
 * column order. The identity when `dir` holds nothing for the weight. */
std::vector<int32_t> gptq_act_order(const char* dir, const char* name, int64_t K, int64_t domain);

/* Quantise `w` [N, K] in place on grid `g` against `scales` (grid_scales'), writing codes to
 * `codes` [N, K]. On return `w` holds the decoded values. With `u` null each value is rounded to
 * nearest on its own, which is byte-identical to grid_rtn. */
void gptq_quantise(const Grid& g, float* w, int32_t* codes, const Scales& scales, int64_t N,
                   int64_t K, const float* u);

/* Coordinate descent after the GPTQ loop: `passes` sweeps over each row's columns, re-choosing each
 * code at its conditional optimum (the value minimising the row's e (H + lambda I) e^T,
 * e = decoded - x, other columns held) against the same damped Hessian. GPTQ fixes each column
 * before later ones are known; a sweep revisits it with all in place and never raises the
 * objective. `x` [N, K] is the (transformed) weight GPTQ was given, `w` its decoded result,
 * updated in place with `codes`. `h` is the undamped Gram (calib_gram's); null means no-op. */
void gptq_refine(const Grid& g, const float* x, float* w, int32_t* codes, const Scales& scales,
                 int64_t N, int64_t K, const float* h, double damp, int passes);


}  /* namespace lq */
