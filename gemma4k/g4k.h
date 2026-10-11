/* g4k.h -- what gemma4k's row table (g4k_rows.cpp, host C++) needs from its device units
 * (g4k_attn.hip, g4k_pick.hip, g4k_norm.hip). Plain C declarations: the row table compiles
 * without HIP. */
#ifndef G4K_H
#define G4K_H

#include "rad_abi.h"

#ifdef __cplusplus
extern "C" {
#endif

/* attn_paged at (head_dim, gqa, kv dtype). One entry point per instantiation, named for its
 * geometry, refusing anything else at launch. */
#define G4K_ATTN_DECL(HD, G)                                                                       \
    int g4k_attn_h##HD##_gqa##G##_bf16kv(const RadArgs* a, RadStream s);                           \
    int g4k_attn_h##HD##_gqa##G##_fp8kv(const RadArgs* a, RadStream s);
G4K_ATTN_DECL(256, 1) G4K_ATTN_DECL(256, 2) G4K_ATTN_DECL(256, 4) G4K_ATTN_DECL(256, 8)
G4K_ATTN_DECL(512, 1) G4K_ATTN_DECL(512, 2) G4K_ATTN_DECL(512, 4) G4K_ATTN_DECL(512, 8)
#undef G4K_ATTN_DECL

/* The split-KV partials one launch can need, at declare (rad_abi.h RadScratchFn). */
int64_t g4k_attn_h256_scratch(const RadArgs* a);
int64_t g4k_attn_h512_scratch(const RadArgs* a);

/* pair_argmax on the device: the top-1 of a row of gathered (id, value) pairs, as one i32 a row. */
int g4k_pair_argmax_f32(const RadArgs* a, RadStream s);

/* Gemma 4's sandwich-norm joins, one kernel each (g4k_norm.hip):
 *   norm_add_norm   operands a, w_a, x (inout), w_y, y
 *   join_add_scale  operands d, w_d, r, w_r, w_j, x (inout), s */
int g4k_norm_add_norm_bf16(const RadArgs* a, RadStream s);

/* expert_scale: each routing weight times its expert's scale, in place (g4k_escale.hip):
 *   operands ids, w_e, ew (inout) */
int g4k_expert_scale_bf16(const RadArgs* a, RadStream s);
int g4k_join_add_scale_bf16(const RadArgs* a, RadStream s);

#ifdef __cplusplus
}
#endif
#endif /* G4K_H */
