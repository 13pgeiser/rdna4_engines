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
    # Decompose the gap. A single global scale+bias fitting the two sorted distributions well means the
    # divergence is mostly ONE network-level gain (a fixable mis-scaled norm/weight), not an intractable
    # structural transform. This scale survives the fold's PER-CHANNEL norm — and those norm constants are
    # byte-identical to the reference — so a COMMON-across-channels gain must live in the RAW encoder latent
    # (a network gain), not the fold. The residual relFro after removing the affine is the STRUCTURAL part.
    sa, sb = np.sort(dev.reshape(-1)), np.sort(ref)
    A = np.vstack([sb, np.ones_like(sb)]).T
    (aa, bb), *_ = np.linalg.lstsq(A, sa, rcond=None)
    resid = sa - (aa * sb + bb)
    rf_aff = float(np.linalg.norm(resid)) / (float(np.linalg.norm(sb)) + 1e-12)
    r2 = 1.0 - float(np.sum(resid ** 2)) / (float(np.sum((sa - sa.mean()) ** 2)) + 1e-30)
    print(f"  global-affine fit  dev ~ {aa:.4f}*ref {bb:+.4f}:  R^2={r2:.3f}  residual(relFro after affine)={rf_aff:.3e}")
    print(f"    (a large rf->rf_aff drop => mostly ONE network gain; rf_aff near rf => structural)")
    # fp8 weights + bf16 activations are far coarser than the fp16 reference, so the floor is well above
    # the audio path's 5e-3. Interpret against the audio DIVERGENCE (0.79), not an absolute pass/fail.
    floor = 5e-2
    print(f"  verdict: {'consistent within fp8/bf16 quant' if rf < floor else 'DIVERGED like the audio path'} "
          f"(fp8/bf16 headroom {floor:.0e}; audio DIVERGED at 7.9e-01)")
    return 0 if rf < floor else 1


def ref_encoder_out(m, encpx, H, W, which):
    """The reference RAW (pre-fold) latent values at a tail stage, as a value multiset (order-free).
      'nout' -> the norm_out output (silu(GroupNorm(h))) at the last level, ALL dims[5]=1024 channels
                (h, the conv_out input, is captured by a forward hook; conv_out applies pre_norm inside).
      'q48'  -> the conv_out output (encoder(x)), 48ch; first 24 = mu.
      'q2'   -> the quant_conv output (moments), 48ch; first 24 = raw mu.
    Each is exactly what the corresponding device f32 shadow (b_vnout/b_evq48/b_evq2) holds, so a gain
    that is already present at 'nout' points at norm_out itself, one that first appears at 'q48' points
    at conv_out, and one that first appears at 'q2' points at quant_conv."""
    px = torch.from_numpy(encpx).permute(2, 0, 1).unsqueeze(0).unsqueeze(2)   # [1,3,1,H,W]
    x = px.float() * (2.0 / 255.0) - 1.0
    cap = {}
    def hook(mod, inp, out):
        cap["h"] = inp[0]
    hk = m.encoder.conv_out.register_forward_hook(hook)
    try:
        with torch.no_grad():
            x = m._normalize_pixels(x)
            out = m.encoder(x)                       # conv_out output, [1,48,1,H/16,W/16]
            if which == "q2":
                out = m.quant_conv(out)              # moments
            if which == "trunk":
                return cap["h"].float().reshape(-1).double().numpy()   # norm_out INPUT (level-5 stream)
            if which == "nout":
                h = cap["h"]                         # the level-5 stream feeding conv_out
                no = m.encoder.norm_out
                g = F.group_norm(h.float(), no.num_groups, no.weight.float(), no.bias.float(),
                                 eps=no.eps)         # T==1: TemporalIsolated == plain GroupNorm
                out = F.silu(g)                       # [1,1024,1,H/16,W/16]
                return out.reshape(-1).double().numpy()
    finally:
        hk.remove()
    out = out[:, :, -1:, :, :]                       # T==1 keep-last
    mu = out.float()[:, :KC, ...]                    # first 24 channels = the latent mean
    return mu.reshape(-1).double().numpy()


def do_stage_raw(m, dev_f, encpx_f, encg_f, which):
    """Compare a device f32 shadow against the reference RAW latent at the same stage, to localise
    where the ~0.37 network gain enters."""
    gd = np.fromfile(encg_f, dtype=np.int32)
    H, W = int(gd[1]), int(gd[2])
    short_last = int(gd[18])
    if int(gd[23]) != 1:
        sys.exit("these raw stages target a single reference IMAGE (T_pad=1)")
    encpx = np.fromfile(encpx_f, dtype=np.float32).reshape(int(gd[23]), H, W, 3)
    ref = ref_encoder_out(m, encpx[0], H, W, which)
    rows = short_last * (H // KVAE_RATIO) * (W // KVAE_RATIO)   # real rows the device wrote
    ncol = 1024 if which == "nout" else 48
    sh = np.fromfile(dev_f, dtype=np.float32).reshape(-1, ncol)
    dev = sh[:rows] if which == "nout" else sh[:rows, :KC]      # nout: all 1024; mu: first 24
    if dev.size != ref.size:
        sys.exit(f"device shadow region {dev.size} != reference {ref.size} (rows={rows}, ncol={ncol})")
    rf, mx = rel_frobenius_sorted(dev.reshape(-1), ref)
    sa, sb = np.sort(dev.reshape(-1)), np.sort(ref)
    A = np.vstack([sb, np.ones_like(sb)]).T
    (aa, bb), *_ = np.linalg.lstsq(A, sa, rcond=None)
    resid = sa - (aa * sb + bb)
    rf_aff = float(np.linalg.norm(resid)) / (float(np.linalg.norm(sb)) + 1e-12)
    print(f"[{which}] device shadow vs reference: values={dev.size}")
    print(f"  sorted relFro={rf:.3e}  maxabs={mx:.3e}  dev std={dev.std():.4f} ref std={ref.std():.4f}")
    print(f"  affine dev ~ {aa:.4f}*ref {bb:+.4f}  residual-after-affine={rf_aff:.3e}  "
          f"(gain {'PRESENT here' if abs(aa-1.0) > 0.1 else 'already ~1.0 here'})")


def _gn_on(trunk, w, b, G, eps, per_frame_rows):
    """GroupNorm over a [rows, C] trunk. per_frame_rows>0 -> recompute statistics every that many
    rows (per temporal frame, the reference's TemporalIsolatedGroupNorm); 0 -> one whole-volume
    statistic over all rows (what the device group_norm kernel does, M=ry over the full buffer)."""
    t = torch.from_numpy(trunk).double()             # [rows, C]
    rows, C = t.shape
    w = torch.from_numpy(w).double(); b = torch.from_numpy(b).double()
    cpg = C // G
    tg = t.reshape(rows, G, cpg)                     # per row: G groups of cpg channels
    if per_frame_rows:
        nf = rows // per_frame_rows
        # statistics over each frame's (per_frame_rows x cpg) block, per group
        blocks = tg.reshape(nf, per_frame_rows, G, cpg)
        mu = blocks.mean(dim=(1, 3), keepdim=True)
        var = blocks.var(dim=(1, 3), unbiased=False, keepdim=True)
        rstd = 1.0 / torch.sqrt(var + eps)
        normed = (blocks - mu) * rstd
        tg = normed.reshape(rows, G, cpg)
    else:
        mu = tg.mean(dim=(0, 2), keepdim=True)
        var = tg.var(dim=(0, 2), unbiased=False, keepdim=True)
        rstd = 1.0 / torch.sqrt(var + eps)
        tg = (tg - mu) * rstd
    y = tg.reshape(rows, C) * w + b
    return (y * torch.sigmoid(y)).numpy()            # SiLU


def do_gn_extent(m, trunk_f, nout_f, encpx_f, encg_f):
    """Is the device norm_out whole-volume (over every temporal slot it materialises) while the
    reference is per-frame? Dump the trunk (v_lv[5].a) AND its shadowed norm_out output, recompute
    both statistics modes on the device's OWN trunk, and see which reproduces the device output --
    and which reproduces the reference. Confines a T==1 latent-scale gain to the norm extent (#20)."""
    gd = np.fromfile(encg_f, np.int32)
    H, W = int(gd[1]), int(gd[2])
    if int(gd[23]) != 1:
        sys.exit("this test targets a single reference IMAGE (T_pad=1)")
    C = 1024
    PF = (H // KVAE_RATIO) * (W // KVAE_RATIO)       # rows per latent frame
    trunk = np.fromfile(trunk_f, np.float32).reshape(-1, C)
    nout = np.fromfile(nout_f, np.float32).reshape(-1, C)
    nrow = trunk.shape[0]
    ntf = nrow // PF                                # temporal frames the device materialised
    no = m.encoder.norm_out
    w = no.weight.float().double().detach().numpy(); b = no.bias.float().double().detach().numpy()
    G = int(no.num_groups); eps = float(no.eps)
    # reference trunk (norm_out input) -- reference processes 1 frame -> PF rows
    encpx = np.fromfile(encpx_f, np.float32).reshape(1, H, W, 3)
    ref_trunk = ref_encoder_out(m, encpx[0], H, W, "trunk")
    ref_nout = ref_encoder_out(m, encpx[0], H, W, "nout")
    print(f"device trunk rows={nrow} = {ntf} frames x PF={PF}; norm groups={G} eps={eps} C={C}")
    rf_t, _ = rel_frobenius_sorted(trunk[:PF].reshape(-1), ref_trunk)
    print(f"[A] device trunk frame0 vs reference trunk : relFro={rf_t:.3e}  "
          f"(dev std={trunk[:PF].std():.4f} ref std={ref_trunk.std():.4f})")
    pred_whole = _gn_on(trunk, w, b, G, eps, 0)
    pred_frame = _gn_on(trunk[:PF], w, b, G, eps, 0)   # per-frame == whole on the single-frame slice
    for name, pred, target in (("device nout", pred_whole[:PF], nout[:PF]),
                               ("reference nout", pred_frame[:PF], ref_nout)):
        rf, _ = rel_frobenius_sorted(pred.reshape(-1), target.reshape(-1))
        print(f"[B] whole-volume-GN(trunk frame0) vs {name:14s}: relFro={rf:.3e}")
    # and the per-frame reading of the device output: does the DEVICE output match a per-frame norm?
    rf_dev_frame, _ = rel_frobenius_sorted(pred_frame.reshape(-1), nout[:PF].reshape(-1))
    print(f"[C] per-frame-GN(trunk frame0) vs device nout frame0: relFro={rf_dev_frame:.3e}")
    print("  reading: [B device nout ~0] => device norm is WHOLE-VOLUME over all "
          f"{ntf} frames; [B reference nout ~0] => reference is PER-FRAME; if both, the fix is "
          "per-frame norm statistics (matches TemporalIsolatedGroupNorm).")


def do_expand_test(m, trunk_f, encpx_f, encg_f):
    """Is the trunk divergence just the device's T==1->clip temporal EXPANSION (replicate the image
    across the clip, temporal-downsample) rather than a trunk kernel/weight bug? Feed the reference the
    image replicated to N frames; if its (per-frame-normed) trunk std drops to the device's, the whole
    gap is the staging geometry, not the convs."""
    gd = np.fromfile(encg_f, np.int32)
    H, W = int(gd[1]), int(gd[2])
    C = 1024
    PF = (H // KVAE_RATIO) * (W // KVAE_RATIO)
    trunk = np.fromfile(trunk_f, np.float32).reshape(-1, C)
    ntf = trunk.shape[0] // PF
    encpx = np.fromfile(encpx_f, np.float32).reshape(1, H, W, 3)[0]

    def ref_trunk_std(nframes, keep):
        px = torch.from_numpy(encpx).permute(2, 0, 1)                 # [3,H,W]
        px = px.unsqueeze(0).unsqueeze(2).repeat(1, 1, nframes, 1, 1) # [1,3,nframes,H,W]
        x = px.float() * (2.0 / 255.0) - 1.0
        cap = {}
        hk = m.encoder.conv_out.register_forward_hook(lambda md, i, o: cap.__setitem__("h", i[0]))
        try:
            with torch.no_grad():
                x = m._normalize_pixels(x); m.encoder(x)
        finally:
            hk.remove()
        h = cap["h"].float()                                          # [1,1024,T,H/16,W/16]
        t = h.shape[2]
        fr = h[:, :, (t - 1) if keep == "last" else 0, ...]           # one frame
        return t, fr.reshape(-1).double().numpy()

    dev_f0 = trunk[:PF]; dev_fl = trunk[-PF:]
    print(f"device trunk: {ntf} frames x PF={PF}; frame0 std={dev_f0.std():.4f} "
          f"last-frame std={dev_fl.std():.4f}")
    for n in (1, 4, 8, 17):
        for keep in ("first", "last"):
            t, r = ref_trunk_std(n, keep)
            rf0, _ = rel_frobenius_sorted(r, dev_f0.reshape(-1))
            rfl, _ = rel_frobenius_sorted(r, dev_fl.reshape(-1))
            print(f"  ref N={n:2d} -> T_out={t} keep={keep:5s}: std={r.std():.4f}  "
                  f"relFro vs dev-frame0={rf0:.3e}  vs dev-last={rfl:.3e}")
    print("  reading: if some (N,keep) gives relFro ~0 to the device frame, the trunk gap is purely "
          "the T==1 clip EXPANSION (staging), not the convs.")


def do_zeropad_test(m, trunk_f, encpx_f, encg_f):
    """Confirm the single-frame trunk dilution: the device stages a picture as frame0 in a zero-padded
    clip, so its temporal downsamplers average over ZERO future slots. Feed the reference a
    [real,0,0,...] clip and see whether its frame0 trunk drops to the device's std / matches it.
    Host-only; the decisive golden comparison for the temporal-padding root cause (no device change)."""
    gd = np.fromfile(encg_f, np.int32)
    H, W = int(gd[1]), int(gd[2])
    C = 1024
    PF = (H // KVAE_RATIO) * (W // KVAE_RATIO)
    trunk = np.fromfile(trunk_f, np.float32).reshape(-1, C)
    ntf = trunk.shape[0] // PF
    encpx = np.fromfile(encpx_f, np.float32).reshape(1, H, W, 3)[0]
    dev_f0 = trunk[:PF]
    print(f"device trunk {ntf} frames x PF={PF}: frame0 std={dev_f0.std():.4f} last std={trunk[-PF:].std():.4f}")
    px = torch.from_numpy(encpx).permute(2, 0, 1)                       # [3,H,W]
    for n in (1, 2, 5, 9, 17):
        clip = torch.zeros(1, 3, n, H, W)
        clip[0, :, 0] = px.float() * (2.0 / 255.0) - 1.0                # real frame0, zeros after
        cap = {}
        hk = m.encoder.conv_out.register_forward_hook(lambda md, i, o: cap.__setitem__("h", i[0]))
        try:
            with torch.no_grad():
                x = m._normalize_pixels(clip); m.encoder(x)
        finally:
            hk.remove()
        h = cap["h"].float()                                            # [1,1024,T,H/16,W/16]
        t = h.shape[2]
        f0 = h[:, :, 0, ...].reshape(-1).double().numpy()
        rf0, _ = rel_frobenius_sorted(f0, dev_f0.reshape(-1))
        print(f"  ref clip N={n:2d} -> T_out={t}: frame0 std={f0.std():.4f}  relFro vs dev-frame0={rf0:.3e}")
    print("  reading: a ~0 relFro / matching std means the reference on a zero-padded clip reproduces "
          "the device => the gap is the single-frame temporal DILUTION (device averages zeros).")


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
    ap.add_argument("--evcq2", help="device f32 shadow of the quant_conv output b_eq2 (R4D_ENC_SEL=q2)")
    ap.add_argument("--evcq48", help="device f32 shadow of the conv_out output b_eq48 (R4D_ENC_SEL=q48)")
    ap.add_argument("--evcnout", help="device f32 shadow of the norm_out output b_vnout (R4D_ENC_SEL=nout)")
    ap.add_argument("--evctrunk", help="device f32 shadow of the norm_out input v_lv[5].a (R4D_ENC_SEL=all)")
    ap.add_argument("--expand", action="store_true",
                    help="with --evctrunk: test whether the trunk gap is just the T==1 clip expansion")
    ap.add_argument("--zeropad", action="store_true",
                    help="with --evctrunk: feed the reference a [real,0,...] clip to confirm temporal dilution")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    m = load_encoder()
    if a.selftest:
        sys.exit(do_selftest(m))
    if a.zeropad:
        if not (a.evctrunk and a.encpx and a.encg):
            sys.exit("--zeropad needs --evctrunk, --encpx and --encg")
        sys.exit(do_zeropad_test(m, a.evctrunk, a.encpx, a.encg))
    if a.expand:
        if not (a.evctrunk and a.encpx and a.encg):
            sys.exit("--expand needs --evctrunk, --encpx and --encg")
        sys.exit(do_expand_test(m, a.evctrunk, a.encpx, a.encg))
    if a.evctrunk:
        if not (a.evcnout and a.encpx and a.encg):
            sys.exit("--evctrunk needs --evcnout, --encpx and --encg")
        sys.exit(do_gn_extent(m, a.evctrunk, a.evcnout, a.encpx, a.encg))
    if a.evcq2 or a.evcq48 or a.evcnout:
        if not (a.encpx and a.encg):
            sys.exit("raw-latent stages need --encpx and --encg too")
        # print the tail chain in order so a gain's entry point is visible top-to-bottom
        if a.evcnout:
            do_stage_raw(m, a.evcnout, a.encpx, a.encg, "nout")
        if a.evcq48:
            do_stage_raw(m, a.evcq48, a.encpx, a.encg, "q48")
        if a.evcq2:
            do_stage_raw(m, a.evcq2, a.encpx, a.encg, "q2")
        sys.exit(0)
    if not (a.evc and a.encpx and a.encg):
        sys.exit("give --evc, --encpx and --encg (or --evcq2/--evcq48, or --selftest)")
    sys.exit(do_parity(m, a.evc, a.encpx, a.encg, a.slot))


if __name__ == "__main__":
    main()
