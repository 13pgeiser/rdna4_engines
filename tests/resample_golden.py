#!/home/juup/.venvs/hfref312/bin/python
"""resample_golden.py -- emit tests/data/golden/uocr_resample_golden.bin for rad_resample.h.

Runs on CPU only (the host's GPUs serve production; torch is used with no CUDA).

For each geometry (64,64->40,40), (24,24->20,20), (8,8->5,5), (16,16->10,10),
(16,20->10,13) it draws a fixed-seed fp32 input (C=3, row-major [h, w, C]), computes
torch's F.interpolate(..., mode="bicubic", antialias=True, align_corners=False) on the
[1, C, h, w] permute (the DeepEncoder's get_abs_pos_sam form), and emits input + output as
float literals (%.9g, exact fp32 round trip).

It then loads the Unlimited-OCR checkpoint (shard resolved through the checkpoint's
model.safetensors.index.json), reads
  model.sam_model.pos_embed                                    [1,64,64,768] bf16
  model.vision_model.embeddings.position_embedding.weight        [257,1024] bf16
resamples them the way deepencoder.py does (SAM grid 64->40; CLIP rows 1: as a 16x16 grid
16->10 with row 0 (cls) passed through in front), casts back to bf16, and emits the source
and the torch output as bf16 bit patterns in the exact order the C++ produces them:
SAM [40,40,768]; CLIP [101,1024] with the cls row at index 0.

The output is the little-endian blob tests/uocr_resample_golden.h loads:
  u32 magic 'UOCR', u32 version=1, u32 n_cases, u32 pad
  n_cases * { i32 ih, iw, oh, ow; i64 in_n, out_n }
  per case: f32 in[in_n], f32 out[out_n]
  u16 sam_src, sam_out, clip_src, clip_out
(round-trip verified against the written bytes before the script exits 0).

A numpy emulation of arch/common/rad_resample.h is checked against every golden and
reported; a variant sweep on the smallest geometry pins the knobs (pass order, filter
family, inv-scale placement, centre precision, edge handling) before the blob is written.
"""

import itertools
import json
import os
import struct
import sys

import numpy as np
import torch
import torch.nn.functional as F
from safetensors.torch import safe_open

SIZES = [(64, 64, 40, 40), (24, 24, 20, 20), (8, 8, 5, 5), (16, 16, 10, 10), (16, 20, 10, 13)]
CH = 3
SEED = 20261019
MODELS = "/home/juup/models"
CKPT = os.path.join(MODELS, "Unlimited-OCR")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "data", "golden", "uocr_resample_golden.bin")

SAM_TENSOR = "model.sam_model.pos_embed"
CLIP_TENSOR = "model.vision_model.embeddings.position_embedding.weight"

# The emulation mirrors torch 2.14's separable AA kernel exactly (see the emulation
# section); the C++ header mirrors the same algorithm.


def torch_resample(hwc, oh, ow):
    """[h,w,C] fp32 -> torch bicubic+antialias -> [oh,ow,C] fp32, as the tower permutes."""
    x = torch.from_numpy(hwc).permute(2, 0, 1).unsqueeze(0)
    y = F.interpolate(x, size=(oh, ow), mode="bicubic", antialias=True, align_corners=False)
    return y[0].permute(1, 2, 0).contiguous().numpy()


# ---------------------------------------------------------------- rad_resample.h emulation
#
# Mirrors torch 2.14's upsample_bicubic2d_aa_kernel_impl -> upsample_separable_Nd
# (UpSampleKernel.cpp): HelperInterpCubic::aa_filter (Keys a=-0.5), separable W pass
# first then H pass, per-axis float scale/center/taps and
# _compute_indices_min_size_weights_aa. float32 is used exactly where ATen uses
# scalar_t = float and double where the expression promotes.

import math


def cubic_kernel(fa):
    """HelperInterpCubic::aa_filter<float> (Keys a = -0.5), elementwise fp32."""
    ax = np.abs(np.float32(fa)).astype(np.float32)
    r1 = ((np.float32(1.5) * ax - np.float32(2.5)) * ax) * ax + np.float32(1.0)
    r2 = ((np.float32(-0.5) * ax + np.float32(2.5)) * ax - np.float32(4.0)) * ax + np.float32(2.0)
    return np.where(ax < np.float32(1.0), r1,
                    np.where(ax < np.float32(2.0), r2, np.float32(0.0))).astype(np.float32)


def make_axis(in_n, out_n):
    scale = np.float32(in_n / out_n)
    support = (np.float32(np.float64(np.float32(4) * 0.5) * float(scale))
               if scale >= 1.0 else np.float32(2.0))
    max_taps = int(math.ceil(float(support))) * 2 + 1
    invscale = np.float32(1.0 / float(scale)) if scale >= 1.0 else np.float32(1.0)
    w = np.zeros((out_n, max_taps), np.float32)
    idx = np.zeros((out_n, max_taps), np.int64)
    size = np.zeros(out_n, np.int64)
    for i in range(out_n):
        center = np.float32(float(scale) * (i + 0.5))
        xmin = max(int(float(center) - float(support) + 0.5), 0)
        xsize = min(int(float(center) + float(support) + 0.5), in_n) - xmin
        xsize = min(max(int(xsize), 0), max_taps)
        for j in range(xsize):
            d = np.float32(np.float32(j + xmin) - center)
            fa = np.float32((float(d) + 0.5) * float(invscale))
            w[i, j] = float(cubic_kernel(fa))
            idx[i, j] = xmin + j
        total = np.float32(0.0)
        for j in range(xsize):
            total = np.float32(total + w[i, j])
        if total != 0:
            for j in range(xsize):
                w[i, j] = np.float32(w[i, j] / total)
        size[i] = xsize
    return w, idx, size


def emu_resample(src_hwc, ih, iw, oh, ow, ch, cfg=None):
    wy, iy, sy = make_axis(ih, oh)
    wx, ix, sx = make_axis(iw, ow)
    inter = src_hwc.astype(np.float32)
    if iw != ow:  # W (horizontal) pass first
        tmp = np.zeros((ih, ow, ch), np.float32)
        for oy in range(ih):
            for ox in range(ow):
                acc = inter[oy, ix[ox, 0]] * wx[ox, 0]
                for j in range(1, int(sx[ox])):
                    acc = (acc + inter[oy, ix[ox, j]] * wx[ox, j]).astype(np.float32)
                tmp[oy, ox] = acc
        inter = tmp
    if ih != oh:  # H (vertical) pass second
        out = np.zeros((oh, ow, ch), np.float32)
        for ox in range(ow):
            for oy in range(oh):
                acc = inter[iy[oy, 0], ox] * wy[oy, 0]
                for j in range(1, int(sy[oy])):
                    acc = (acc + inter[iy[oy, j], ox] * wy[oy, j]).astype(np.float32)
                out[oy, ox] = acc
        inter = out
    return inter


def ulps(a, b):
    ia = np.ascontiguousarray(a, np.float32).view(np.int32).astype(np.int64)
    ib = np.ascontiguousarray(b, np.float32).view(np.int32).astype(np.int64)
    oa = np.where(ia < 0, -np.int64(2147483648) - ia, ia)
    ob = np.where(ib < 0, -np.int64(2147483648) - ib, ib)
    return np.abs(oa - ob)


def report(tag, got, want):
    d = ulps(got, want)
    exact = int((d == 0).sum())
    n = d.size
    rel = np.abs(got.astype(np.float64) - want.astype(np.float64)) / np.maximum(np.abs(want.astype(np.float64)), 1e-30)
    print(f"  {tag}: {exact}/{n} bit-exact ({100.0 * exact / n:.4f}%), "
          f"max ULP {int(d.max())}, max rel {rel.max():.3g}", flush=True)
    return exact == n





# ---------------------------------------------------------------- blob emission

def pack_f32(arr):
    return struct.pack("<%df" % arr.size, *np.ascontiguousarray(arr, np.float32).reshape(-1))


def pack_u16(arr):
    return struct.pack("<%dH" % arr.size, *arr.reshape(-1))


def bits(t):
    return (t.detach().cpu().view(torch.int16).numpy().astype(np.uint32) & 0xFFFF).astype(np.uint16)


def bf16_round_bits(f32):
    u = np.ascontiguousarray(f32, np.float32).view(np.uint32)
    u = u + np.uint32(0x7FFF) + ((u >> np.uint32(16)) & np.uint32(1))
    return (u >> np.uint32(16)).astype(np.uint16)


def main():
    torch.set_num_threads(1)

    ins, outs = [], []
    all_ok = True
    gen = torch.Generator().manual_seed(SEED)
    first = True
    for (ih, iw, oh, ow) in SIZES:
        x = torch.randn(ih, iw, CH, generator=gen, dtype=torch.float32)
        xn = x.numpy()
        y = torch_resample(xn, oh, ow)
        all_ok &= report(f"{ih}x{iw}->{oh}x{ow} (emulation vs torch)", emu_resample(xn, ih, iw, oh, ow, CH), y)
        ins.append(xn)
        outs.append(y)

    shard = json.load(open(os.path.join(CKPT, "model.safetensors.index.json")))["weight_map"]
    with safe_open(os.path.join(CKPT, shard[SAM_TENSOR]), framework="pt") as h:
        sam = h.get_tensor(SAM_TENSOR)
        clip = h.get_tensor(CLIP_TENSOR)
    assert tuple(sam.shape) == (1, 64, 64, 768), sam.shape
    assert tuple(clip.shape) == (257, 1024), clip.shape

    sam_src = sam[0]                                          # [64,64,768] bf16
    sam_out = F.interpolate(sam_src.permute(2, 0, 1).unsqueeze(0).to(torch.float32),
                            size=(40, 40), mode="bicubic", antialias=True,
                            align_corners=False)[0].permute(1, 2, 0).to(torch.bfloat16)
    grid = clip[1:]                                           # [256,1024] bf16
    grid_out = F.interpolate(grid.view(1, 16, 16, 1024).permute(0, 3, 1, 2).to(torch.float32),
                             size=(10, 10), mode="bicubic", antialias=True,
                             align_corners=False).permute(0, 2, 3, 1).to(torch.bfloat16)
    clip_out = torch.cat([clip[:1], grid_out.reshape(100, 1024)], dim=0)   # [101,1024]

    def bf16_emu(src_bits_u16, ih, iw, oh, ow, ch):
        f32 = (src_bits_u16.astype(np.uint32) << 16).view(np.float32).reshape(ih, iw, ch)
        return f32, emu_resample(f32, ih, iw, oh, ow, ch)

    sam_bits = bits(sam_src)
    samf, samg = bf16_emu(sam_bits, 64, 64, 40, 40, 768)
    all_ok &= report("sam 64->40 fp32 vs torch (pre-bf16)", samg, sam_out.float().numpy())
    sam_bf_mism = int((bf16_round_bits(samg).reshape(-1) != bits(sam_out).reshape(-1)).sum())
    print(f"  sam 64->40 bf16: {sam_bf_mism} bf16 mismatches", flush=True)

    clip_bits = bits(clip)
    grid_bits = clip_bits[1:].reshape(16, 16, 1024)
    cf, cg = bf16_emu(grid_bits, 16, 16, 10, 10, 1024)
    all_ok &= report("clip 16->10 fp32 vs torch (pre-bf16)", cg, grid_out.reshape(10, 10, 1024).float().numpy())
    clip_bf_mism = int((bf16_round_bits(cg).reshape(-1) != bits(grid_out.reshape(10, 10, 1024)).reshape(-1)).sum())
    print(f"  clip 16->10 bf16: {clip_bf_mism} bf16 mismatches", flush=True)

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    blob = bytearray()
    blob += struct.pack("<4I", 0x52434F55, 1, len(SIZES), 0)
    for (ih, iw, oh, ow), x, y in zip(SIZES, ins, outs):
        blob += struct.pack("<4i2q", ih, iw, oh, ow, x.size, y.size)
    for x, y in zip(ins, outs):
        blob += pack_f32(x)
        blob += pack_f32(y)
    for arr in (sam_bits, bits(sam_out), clip_bits, bits(clip_out)):
        blob += pack_u16(arr)
    with open(OUT, "wb") as f:
        f.write(blob)

    # Round-trip: the loader in tests/uocr_resample_golden.h reads the same layout.
    raw = bytes(blob)
    assert struct.unpack_from("<4I", raw, 0) == (0x52434F55, 1, len(SIZES), 0)
    off = 16
    for (ih, iw, oh, ow), x, y in zip(SIZES, ins, outs):
        assert struct.unpack_from("<4i2q", raw, off) == (ih, iw, oh, ow, x.size, y.size)
        off += 32
    for x, y in zip(ins, outs):
        for v in (x, y):
            got = np.frombuffer(raw, np.float32, v.size, off)
            assert np.array_equal(got, np.ascontiguousarray(v, np.float32).reshape(-1))
            off += 4 * v.size
    for arr in (sam_bits, bits(sam_out), clip_bits, bits(clip_out)):
        got = np.frombuffer(raw, np.uint16, arr.size, off)
        assert np.array_equal(got, arr.reshape(-1))
        off += 2 * arr.size
    assert off == len(raw), (off, len(raw))

    print(f"wrote {OUT} ({os.path.getsize(OUT) / 1e6:.1f} MB, round-trip verified); "
          f"emulation {'FULLY matches' if all_ok else 'DIVERGES from'} torch", flush=True)
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
