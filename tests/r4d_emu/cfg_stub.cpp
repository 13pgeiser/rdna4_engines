/* cfg_stub.cpp -- the two tuned-configuration readers the M2XFP4 GEMM units call
 * (r4d_cfg_mxfp4_decode, r4d_cfg_mxfp4_tiled), which libr4d defines on its declaration side
 * (r4d_rows.cpp) beside the row tables. That side is not in the emu build, and every r4d_emu_*
 * binary links every unit, so the readers are defined once here. Every axis default of
 * tGemmMxfp4Decode and tGemmMxfp4Tiled is 0, the kernels' own rule, so reading the key or 0 is
 * what r4d_rows.cpp's readers return for a caller with no tuned point. */
#include "r4d_args.h"

extern "C" void r4d_cfg_mxfp4_decode(const RadArgs* a, R4dMxDecCfg* out) {
    out->ks = (int)rad_args_geti_or(a, "ks", 0);
    out->bk = (int)rad_args_geti_or(a, "bk", 0);
    out->nt = (int)rad_args_geti_or(a, "nt", 0);
}

extern "C" void r4d_cfg_mxfp4_tiled(const RadArgs* a, R4dMxTileCfg* out) {
    out->tn = (int)rad_args_geti_or(a, "tn", 0);
}
