/* ref_host.h -- what the four libr4d units r4d_emu runs compute, in plain C++ on the host.
 *
 * The device kernels' arithmetic restated, each a RadArgs in and RAD_OK or an error out: the
 * reference the emulated kernels are held to beside the unfused chain each test builds itself
 * (the joins, expert_scale) or a brute force (pair_argmax). No HIP, no libr4d header. */
#pragma once

#include "rad_abi.h"

int host_pair_argmax(const RadArgs* a);
int host_norm_add_norm(const RadArgs* a);
int host_norm_add_norm_route(const RadArgs* a);
int host_join_add_scale(const RadArgs* a);
int host_expert_scale(const RadArgs* a);
