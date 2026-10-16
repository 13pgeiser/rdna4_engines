/* ref_host.h -- host C++ versions of the libr4d units r4d_emu runs.
 *
 * Each restates a device kernel's arithmetic, RadArgs in and RAD_OK or an error out. Tests hold
 * the emulated kernels to these, beside the unfused chain each test builds (the joins,
 * expert_scale) or a brute force (pair_argmax). No HIP, no libr4d header. */
#pragma once

#include "rad_abi.h"
#include "rad_plugin.h"

/* bf16 as libr4d converts it: exact widening, round-to-nearest-even narrowing. rb() is a value
 * stored to bf16 and read back. */
inline float b2f(uint16_t h) { return rad_bf16_to_f32(h); }
inline uint16_t f2b(float f) { return rad_f32_to_bf16(f); }
inline float rb(float f) { return b2f(f2b(f)); }

int host_pair_argmax(const RadArgs* a);
int host_norm_add_norm(const RadArgs* a);
int host_norm_add_norm_route(const RadArgs* a);
int host_join_add_scale(const RadArgs* a);
int host_expert_scale(const RadArgs* a);
