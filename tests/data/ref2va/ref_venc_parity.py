#!/usr/bin/env python3
"""Device-input-anchored parity for the MiniMax-H3 **video / image** VAE encoder (#14 value-parity).

The audio half (ref_aenc_parity.py) measured the device encoder against the reference and found a real
DIVERGENCE (relFro 0.79). This is the OTHER half: the `conv3d_ndhwc` video/image encoder, a SEPARATE
kernel family, dumped the same way. It answers one question: does the device video encoder match the
reference, or does it diverge like the audio path did?

WHY IT DOES NOT IMPORT THE COMFY PACKAGE.  Importing `comfy.ldm.minimax.vae` drags
`comfy.ldm.modules.attention` -> an aiter JIT kernel build that fails on this box (same reason
vae_geom/gen_golden.py AST-extracts instead of importing). The ENCODER path uses no attention, so we
extract just the encoder classes (CausalConv3d / TemporalIsolatedGroupNorm / Downsample3D /
ResnetBlock3D / EncoderFCN3D) and the two wrapper methods `_normalize_pixels` / `_encode_moments`, exec
them on torch alone with the tiny CPU shims gen_golden.py proved sufficient, then load the REAL fp16
reference weights into the encoder + quant_conv (upcast to fp32). The decoder (ViT3DDecoder, with its
attention) is irrelevant to `encode` and never built.

WHY A SORTED COMPARISON, NOT ELEMENTWISE.  The device stores `b_evc` as the PATCH-FOLDED latent: `op_ve_fold`
turns the raw latent `[flat, kC=24, H/16, W/16]` into `flat*ppc` rows of `kPatchVals=96` (= 24 channels x a
2x2 spatial patch), applying the per-channel latent norm `(z - mean)/std` on the way (`ve_fold_mul = 1/std`,
`ve_fold_add = -mean/std`). The fold is a PURE BIJECTION and the norm is PER-CHANNEL, so the MULTISET of
values in the device slot region equals the multiset of the reference's normalised latent regardless of the
fold's row/column order. Comparing the two SORTED value vectors is therefore exact for a faithful encoder and
immune to having to re-derive the fold's (t, h, w) and channel packing — the mistake that once cost a bogus
audio reading. A gross difference between sorted vectors can only mean a gross difference in the encoder.

A single reference IMAGE is the cleanest case: `encode()` takes its T==1 branch (no temporal chunking), the
device encodes one clip, there is exactly ONE latent frame, so keep-first/keep-last is moot and there is no
temporal ambiguity at all. That is what this harness targets (pass a picture's dumps; it refuses T_pad!=1).

Layout it relies on (from arch/minimaxh3/minimaxh3.cpp + rad_vae_geom.h, keep in sync):
    b_evc : [kEncSlotsV * kEncSlotRowsV, kPatchVals] f32 = the NORMALISED patch-folded latent.
    kC=24, kPatch=2, kPatchVals=96, kVaeRatio=16, kEncMaxSide=256, kEncSlotsV=13,
    kEncSlotRowsV = 82*(256/32)*(256/32) = 5248.
    picture slot k row-base -> (k-1)*5248  (pictures number 1..9).  ppc = (H/32)*(W/32).
    enc_pixels : [stage_frames, H, W, 3] f32, 0..255 UN-normalised (the driver dumps exactly this).
    enc_geom (25 i32) : g0=(clipT,H,W) -> gd[1]=H, gd[2]=W ; gd[18]=short_last ; gd[22]=n_clips ; gd[23]=T_pad.

Run under the ComfyUI torch venv (only for numpy/torch/safetensors — no comfy import):
    COMFYUI_HOME=/home/dustin/AI/ComfyUI /home/dustin/AI/ComfyUI/.venv/bin/python \
        ref_venc_parity.py --evc evc.f32 --encpx encpx_1_1.f32 --encg encg_1_1.f32 --slot 1
    # or validate the harness itself with no device at all:
    ref_venc_parity.py --selftest
"""
import argparse
import ast
import math
import os
import sys
import types

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

COMFYUI = os.environ.get("COMFYUI_HOME", "/home/dustin/AI/ComfyUI")
VAE_PY = os.path.join(COMFYUI, "comfy/ldm/minimax/vae.py")
VAE_SAFETENSORS = os.path.join(COMFYUI, "models/vae/minimax_h3_video_vae_fp16.safetensors")

# --- keep these in sync with arch/minimaxh3/minimaxh3.cpp + rad_vae_geom.h ---
KC = 24                     # video latent channels (vae.latent_channels)
KPATCH = 2                  # patch_fold spatial patch
KPATCHVALS = KC * KPATCH * KPATCH   # 96 = b_evc row width
KVAE_RATIO = 16             # pixels per latent side
KENC_MAX_SIDE = 256
KENC_SLOT_ROWS_V = 82 * (KENC_MAX_SIDE // (KVAE_RATIO * KPATCH)) ** 2   # 82*8*8 = 5248


# --------------------------------------------------------------------- shims (as gen_golden.py)
class _Ops:
    class Conv3d(nn.Conv3d):
        def forward(self, x, autopad=None, **kw):
            if autopad == "causal_zero":
                k = self.kernel_size[0]
                x = F.pad(x, (0, 0, 0, 0, k - 1, 0))
            return super().forward(x)
    Conv1d = nn.Conv1d
    GroupNorm = nn.GroupNorm
    LayerNorm = nn.LayerNorm
    Linear = nn.Linear


def _cast_to_input(w, x):
    return w.to(device=x.device, dtype=x.dtype)


class _ComfyOpsNs:
    cast_bias_weight = None
    uncast_bias_weight = None
    cast_to_input = staticmethod(_cast_to_input)


comfy = types.ModuleType("comfy")
comfy.quant_ops = types.ModuleType("comfy.quant_ops")   # absent ck -> kitchen fast paths self-disable on CPU
comfy.ops = _ComfyOpsNs()


def _top_level(path, names):
    src = open(path).read()
    tree = ast.parse(src)
    out = {}
    for node in tree.body:
        key = getattr(node, "name", None)
        if key is None and isinstance(node, ast.Assign):
            keys = [t.id for t in node.targets if isinstance(t, ast.Name)]
        else:
            keys = [key]
        for k in keys:
            if k in names:
                out[k] = ast.get_source_segment(src, node)
    missing = set(names) - set(out)
    if missing:
        raise SystemExit(f"not found in {path}: {sorted(missing)}")
    return out


def _class_method(path, cls_name, method_name):
    src = open(path).read()
    tree = ast.parse(src)
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == cls_name:
            for m in node.body:
                if isinstance(m, ast.FunctionDef) and m.name == method_name:
                    return ast.get_source_segment(src, m)
    raise SystemExit(f"{cls_name}.{method_name} not found in {path}")


class _Enc(nn.Module):
    """the reference encoder chain + quant_conv, the only parts `encode()` touches; decoder excluded."""
    def __init__(self, ns):
        super().__init__()
        self.encoder = ns["EncoderFCN3D"](
            ch=128, ch_mult=[1, 2, 2, 4, 4, 8], space_down=[2, 2, 2, 2, 1, 1],
            time_down=[1, 2, 2, 1, 1, 1], num_res_blocks=2, in_channels=3, z_channels=24,
            double_z=True)
        self.quant_conv = _Ops.Conv3d(48, 48, 1)

    def forward(self, xn):
        return self.quant_conv(self.encoder(xn))


def load_encoder():
    from safetensors.torch import load_file
    ns = dict(torch=torch, nn=nn, F=F, math=math, comfy=comfy, ops=_Ops)
    need = ["IMAGENET_MEAN", "IMAGENET_STD", "LATENTS_MEAN", "LATENTS_STD",
            "_kitchen_ndhwc", "_fused_norm_pad", "_fp16_accum_conv",
            "CausalConv3d", "TemporalIsolatedGroupNorm", "group_norm_3d",
            "Downsample3D", "ResnetBlock3D", "EncoderFCN3D"]
    for name in need:
        exec(_top_level(VAE_PY, [name])[name], ns)

    enc = _Enc(ns).to(torch.float32).eval()
    raw = load_file(VAE_SAFETENSORS)
    # load only the encoder + quant_conv tensors (fp16 -> fp32); skip the decoder (ViT3DDecoder).
    sd = {k: v.float() for k, v in raw.items() if k.startswith(("encoder.", "quant_conv."))}
    enc.load_state_dict(sd, strict=True)

    # bind the two reference methods we replicate, on a shim self with the reference constants.
    self = types.SimpleNamespace()
    self.encoder, self.quant_conv = enc.encoder, enc.quant_conv
    self.pixel_mean = torch.tensor(ns["IMAGENET_MEAN"]).view(1, 3, 1, 1, 1)
    self.pixel_std = torch.tensor(ns["IMAGENET_STD"]).view(1, 3, 1, 1, 1)
    self.latents_mean = torch.tensor(ns["LATENTS_MEAN"]).view(1, KC, 1, 1, 1)
    self.latents_std = torch.tensor(ns["LATENTS_STD"]).view(1, KC, 1, 1, 1)
    mns = dict(ns)
    for meth in ("_normalize_pixels", "_encode_moments"):
        exec(_class_method(VAE_PY, "MiniMaxH3VideoVAE", meth), mns)
        setattr(self, meth, types.MethodType(mns[meth], self))
    return self


def ref_image_latent(model, encpx, H, W):
    """encpx: f32 numpy [H, W, 3] (0..255, the device's staged single frame) -> the reference
    normalised latent [24, 1, H/16, W/16] flattened (a value multiset; order-independent)."""
    px = torch.from_numpy(encpx).permute(2, 0, 1).unsqueeze(0).unsqueeze(2)   # [1, 3, 1, H, W]
    x = px.float() * (2.0 / 255.0) - 1.0                                       # 0..255 -> [-1, 1]
    with torch.no_grad():
        x = model._normalize_pixels(x)
        moments = model._encode_moments(x)
        moments = moments[:, :, -1:, :, :]                                     # T==1 keep-last
        mean = torch.chunk(moments.float(), 2, dim=1)[0]
        z = (mean - model.latents_mean) / model.latents_std
    assert z.shape[1] == KC and z.shape[3] == H // KVAE_RATIO and z.shape[4] == W // KVAE_RATIO, tuple(z.shape)
    return z.reshape(-1).double().numpy()


def rel_frobenius_sorted(a, b):
    """Sort both value vectors (patch_fold is a bijection + the norm is per-channel, so a faithful
    encoder has an identical multiset) and compare."""
    sa, sb = np.sort(a), np.sort(b)
    if sa.shape != sb.shape:
        raise ValueError(f"multiset sizes differ: dev {sa.size} vs ref {sb.size}; check --slot / geometry")
    num = float(np.linalg.norm(sa - sb))
    den = float(np.linalg.norm(sb)) + 1e-12
    return num / den, float(np.max(np.abs(sa - sb)))


def do_parity(m, evc_f, encpx_f, encg_f, slot):
    gd = np.fromfile(encg_f, dtype=np.int32)
    if gd.size < 25:
        sys.exit(f"encg file has {gd.size} ints, expected 25")
    H, W = int(gd[1]), int(gd[2])
    short_last, n_clips, t_pad = int(gd[18]), int(gd[22]), int(gd[23])
    if t_pad != 1:
        sys.exit(f"this harness targets a single reference IMAGE (T_pad=1); got T_pad={t_pad}. "
                 "It does not model the multi-clip temporal keep-last — feed a picture's dumps.")
    if H % (KVAE_RATIO * KPATCH) or W % (KVAE_RATIO * KPATCH):
        sys.exit(f"picture sides {H}x{W} not multiples of {KVAE_RATIO * KPATCH}; wrong dump?")

    encpx = np.fromfile(encpx_f, dtype=np.float32)
    if encpx.size != t_pad * H * W * 3:
        sys.exit(f"encpx {encpx.size} floats != T_pad*H*W*3 = {t_pad * H * W * 3}")
    encpx = encpx.reshape(t_pad, H, W, 3)

    ref = ref_image_latent(m, encpx[0], H, W)                                  # [24*1*H/16*W/16]

    evc = np.fromfile(evc_f, dtype=np.float32).reshape(-1, KPATCHVALS)
    ppc = (H // (KVAE_RATIO * KPATCH)) * (W // (KVAE_RATIO * KPATCH))
    n_rows = short_last * ppc                                                  # rows the device wrote
    base = (slot - 1) * KENC_SLOT_ROWS_V
    dev = evc[base:base + n_rows]
    if dev.shape != (n_rows, KPATCHVALS):
        sys.exit(f"device slice {dev.shape} != ({n_rows}, {KPATCHVALS}); slot {slot} past the buffer?")
    if dev.size != ref.size:
        sys.exit(f"device region {dev.size} values != reference {ref.size}; "
                 f"n_rows={n_rows}*{KPATCHVALS} vs 24*1*{H // 16}*{W // 16}")

    rf, mx = rel_frobenius_sorted(dev.reshape(-1), ref)
    print(f"picture slot={slot} H={H} W={W} short_last={short_last} ppc={ppc} rows={n_rows} values={dev.size}")
    print(f"  SORTED relative Frobenius (dev vs ref) = {rf:.3e}")
    print(f"  max abs deviation (sorted)             = {mx:.3e}")
    print(f"  reference std={ref.std():.4f} mean={ref.mean():.4f}; "
          f"device std={dev.std():.4f} mean={dev.mean():.4f}")
    # fp8 weights + bf16 activations are far coarser than the fp16 reference, so the floor is well above
    # the audio path's 5e-3. Interpret against the audio DIVERGENCE (0.79), not an absolute pass/fail.
    floor = 5e-2
    print(f"  verdict: {'consistent within fp8/bf16 quant' if rf < floor else 'DIVERGED like the audio path'} "
          f"(fp8/bf16 headroom {floor:.0e}; audio DIVERGED at 7.9e-01)")
    return 0 if rf < floor else 1


def do_selftest(m):
    """No device: build a 'device' region from the reference itself (perfect) and two corruptions
    (a dropped normalisation, and a spread-collapse), and confirm the sorted metric reads ~0 and gross.
    Validates the reshape / normalisation / sorted-metric logic with zero device involvement."""
    H = W = 64
    grid = np.indices((H, W, 3)).sum(axis=0).astype(np.float32) % 256          # deterministic tile
    ref = ref_image_latent(m, grid, H, W)

    rf0, _ = rel_frobenius_sorted(ref.copy(), ref)
    lmean = m.latents_mean.reshape(-1).double().numpy()
    lstd = m.latents_std.reshape(-1).double().numpy()
    raw = (ref.reshape(KC, -1) * lstd[:, None] + lmean[:, None]).reshape(-1)   # corruption: drop norm
    rf1, _ = rel_frobenius_sorted(raw, ref)
    sm = ref.reshape(KC, -1)
    sm = sm * 0.1 + sm.mean(axis=1, keepdims=True) * 0.9                        # corruption: spread collapse
    rf2, _ = rel_frobenius_sorted(sm.reshape(-1), ref)
    print(f"selftest  perfect       sorted relFro={rf0:.3e}  (want ~0)")
    print(f"selftest  un-normalised sorted relFro={rf1:.3e}  (want gross)")
    print(f"selftest  collapse      sorted relFro={rf2:.3e}  (want gross)")
    ok = rf0 < 1e-9 and rf1 > 1e-2 and rf2 > 1e-2
    print("selftest:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--evc", help="raw b_evc readback (f32, [slots*5248, 96])")
    ap.add_argument("--encpx", help="staged enc_pixels readback (f32, [T_pad, H, W, 3])")
    ap.add_argument("--encg", help="staged enc_geom readback (i32, 25)")
    ap.add_argument("--slot", type=int, default=1)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    m = load_encoder()
    if a.selftest:
        sys.exit(do_selftest(m))
    if not (a.evc and a.encpx and a.encg):
        sys.exit("give --evc, --encpx and --encg (or --selftest)")
    sys.exit(do_parity(m, a.evc, a.encpx, a.encg, a.slot))


if __name__ == "__main__":
    main()
