#pragma once
// RADIANCE_PREFILL_PIPELINE: whether an architecture with a pipelined prefill (a prefill step of one
// sequence in two pieces, the collectives between blocks on the second lane, beside the other piece's
// compute) uses it.
//
//   auto (or unset)  at two ranks, where it was measured (Qwen3.8-27B 41k prefill -6.6 %,
//                    TranslateGemma-27B 6k -12 %, bit for bit the serial step);
//   on               also at three, four and eight ranks: libr4d has the narrow `overlap` all-reduce
//                    there (the wide rows' entry points at a launch that leaves the CUs to compute),
//                    but the step has not been measured on such hardware;
//   off              never.
//
// Any other rank count has no narrow row, and a wide all-reduce beside the GEMMs ate the whole gain at
// two (it slows the GEMM 1.45-2.1x), so those stay serial.
#include <arch/rad_arch.h>

#include <cstdlib>
#include <cstring>

namespace rad::arch {

inline bool prefill_pipeline(RadBuilder* b, int world, const char* who) {
    const char* e = std::getenv("RADIANCE_PREFILL_PIPELINE");
    const bool on = e && !std::strcmp(e, "on"), off = e && !std::strcmp(e, "off");
    if (e && *e && !on && !off && std::strcmp(e, "auto") != 0)
        rad_note(b, "%s: RADIANCE_PREFILL_PIPELINE=%s is not auto, on or off; taken as auto", who, e);
    if (off || world < 2) return false;
    if (world == 2) return true;
    if (on && (world == 3 || world == 4 || world == 8)) {
        rad_note(b, "%s: the pipelined prefill at %d ranks (RADIANCE_PREFILL_PIPELINE=on) is not yet "
                    "measured on hardware", who, world);
        return true;
    }
    return false;
}

}  // namespace rad::arch
