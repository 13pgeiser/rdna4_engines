#!/usr/bin/env python3
"""Generate tests/data/vae_geom/golden.json from the REFERENCE VAEs themselves.

Same discipline as tests/data/ref2va/gen_golden.py: the module source is AST-parsed and only the
needed classes/helpers are extracted and exec'd on torch alone -- importing the comfy package
dies on this machine (aiter JIT build failure). Unlike the pack golden this one runs real CPU
networks, so expectations are the encoders' own output shapes:

  * the video encoder (EncoderFCN3D with the reference's exact constructor arguments) + the
    reference's own encode()/encode_temporal()/split_tiles()/tiled_encode() methods bound to a
    shim self -> latent counts, per-axis tile plans, and blended tile-grid shapes;
  * the audio DAC encoder -> latent frames for a sweep of sample counts (its posterior head,
    AttnProjection + 1x1 mean_proj, applies no strided op, so it cannot change the frame count;
    NOTE-P2-ENCODE.md makes that argument explicitly).

Only shims: comfy.quant_ops must exist without `ck` (the fused-pad fast path then self-disables
and the CPU path is the plain torch one), and Conv3d must accept the reference's autopad kwarg --
on CPU the only use is the single-frame causal case ("causal_zero" = implicit front zero pad of
kernel_t - 1, which is what physically skipping the pad means for a causal conv).

Usage:  python gen_golden.py <ComfyUI>/comfy/ldm/minimax/vae.py <...>/audio_vae.py golden.json
"""
import ast
import json
import math
import sys
import types

import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)


# ----------------------------------------------------------------------------- shims

class _Ops:
    class Conv3d(nn.Conv3d):
        def forward(self, x, autopad=None, **kw):
            if autopad == "causal_zero":
                k = self.kernel_size[0]
                x = F.pad(x, (0, 0, 0, 0, k - 1, 0))
            return super().forward(x)

    Conv1d = nn.Conv1d
    ConvTranspose1d = nn.ConvTranspose1d
    GroupNorm = nn.GroupNorm
    LayerNorm = nn.LayerNorm
    Linear = nn.Linear


class _QuantOps:
    pass


def _cast_to_input(w, x):
    return w.to(device=x.device, dtype=x.dtype)


class _ComfyOpsNs:
    """what the extracted code reaches for as bare `comfy.ops.*`: cast_to_input is the real
    semantics (a plain device/dtype cast); the kitchen fast paths are never reached on CPU, so
    placeholders are enough there"""
    cast_bias_weight = None
    uncast_bias_weight = None
    cast_to_input = staticmethod(_cast_to_input)


comfy = types.ModuleType("comfy")
comfy.quant_ops = _QuantOps()
comfy.ops = _ComfyOpsNs()
ops = _Ops


# --------------------------------------------------------------------------- extract

def top_level(path, names):
    src = open(path).read()
    tree = ast.parse(src)
    out = {}
    for node in tree.body:
        key = getattr(node, "name", None)
        if key is None and isinstance(node, ast.Assign):
            key = [t.id for t in node.targets if isinstance(t, ast.Name)]
        else:
            key = [key]
        for k in key:
            if k in names:
                out[k] = ast.get_source_segment(src, node)
    missing = set(names) - set(out)
    if missing:
        raise SystemExit(f"not found in {path}: {sorted(missing)}")
    return out


def class_methods(path, cls_name, method_names):
    src = open(path).read()
    tree = ast.parse(src)
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == cls_name:
            found = {m.name: ast.get_source_segment(src, m) for m in node.body
                     if isinstance(m, ast.FunctionDef) and m.name in method_names}
            missing = set(method_names) - set(found)
            if missing:
                raise SystemExit(f"{cls_name} methods not found: {sorted(missing)}")
            return found
    raise SystemExit(f"class {cls_name} not found in {path}")


ns = dict(torch=torch, nn=nn, F=F, math=math, comfy=comfy, ops=ops)

# --- video: the encoder chain classes + module constants
VIDEO_NEED = [
    "IMAGENET_MEAN", "IMAGENET_STD", "LATENTS_MEAN", "LATENTS_STD",
    "_kitchen_ndhwc", "_fused_norm_pad", "_fp16_accum_conv",
    "CausalConv3d", "TemporalIsolatedGroupNorm", "group_norm_3d",
    "Downsample3D", "ResnetBlock3D", "EncoderFCN3D",
]
_vsrc = top_level(sys.argv[1], VIDEO_NEED)
for name in VIDEO_NEED:
    exec(_vsrc[name], ns)

# --- video: the wrapper methods, bound to a shim self (no decoder, no checkpoint)
VIDEO_METHODS = ["_encode_moments", "_normalize_pixels", "split_tiles", "blend",
                 "tiled_encode", "_adaptive_encode", "encode_temporal", "encode"]
mns = dict(ns)
for name, code in class_methods(sys.argv[1], "MiniMaxH3VideoVAE", VIDEO_METHODS).items():
    exec(code, mns)


def make_video_vae():
    self = types.SimpleNamespace()
    self.vae_ratio = 16
    self.vae_ratio_t = 8
    self.clip_length = 17
    self.token_drop = 3
    self.tile_size = 256
    self.tile_overlap_min = 64
    self.tiling = True
    self.encoder = ns["EncoderFCN3D"](
        ch=128, ch_mult=[1, 2, 2, 4, 4, 8], space_down=[2, 2, 2, 2, 1, 1],
        time_down=[1, 2, 2, 1, 1, 1], num_res_blocks=2, in_channels=3, z_channels=24,
        double_z=True)
    self.quant_conv = nn.Conv3d(48, 48, 1)
    self.latents_mean = torch.tensor(ns["LATENTS_MEAN"])
    self.latents_std = torch.tensor(ns["LATENTS_STD"])
    self.pixel_mean = torch.tensor(ns["IMAGENET_MEAN"]).view(1, 3, 1, 1, 1)
    self.pixel_std = torch.tensor(ns["IMAGENET_STD"]).view(1, 3, 1, 1, 1)
    for name in VIDEO_METHODS:
        setattr(self, name, types.MethodType(mns[name], self))
    return self


golden = {"video": {}, "audio": {}}
golden["video"]["constants"] = {
    "latents_mean": list(ns["LATENTS_MEAN"]), "latents_std": list(ns["LATENTS_STD"]),
    "pixel_mean": list(ns["IMAGENET_MEAN"]), "pixel_std": list(ns["IMAGENET_STD"]),
}

vae = make_video_vae()

# --- 1. latent count per frame count, from the reference encode() end to end
FRAME_COUNTS = [1, 2, 3, 5, 8, 16, 17, 18, 22, 34, 39, 40, 56, 72, 73, 90]
cases = []
with torch.no_grad():
    for t in FRAME_COUNTS:
        x = torch.zeros(1, 3, t, 32, 32)
        z = vae.encode(x)
        cases.append({"T": t, "z_shape": list(z.shape)})
        print(f"video encode T={t:3d} -> z {list(z.shape)}", file=sys.stderr)
golden["video"]["encode"] = cases

# --- 2. split_tiles raw output (the pure arithmetic, exact reference return values)
TILE_LENS = [1, 16, 64, 128, 255, 256, 257, 272, 320, 448, 512, 640, 720, 768, 1024, 1088, 1344, 2048]
golden["video"]["split_tiles"] = [
    {"input_len": L, "starts": starts, "lens": lens, "overlaps": ovl}
    for L in TILE_LENS
    for starts, lens, ovl in [vae.split_tiles(L)]
]

# --- 3. the tile grid end to end: the blend/cut keeps the moments at exactly ratio-reduced size
with torch.no_grad():
    tiled = []
    for x_shape in [(1, 3, 5, 320, 320), (1, 3, 2, 512, 256), (1, 3, 17, 288, 192)]:
        moments = vae.tiled_encode(torch.zeros(*x_shape))
        tiled.append({"x_shape": list(x_shape), "moments_shape": list(moments.shape)})
        print(f"video tiled {list(x_shape)} -> moments {list(moments.shape)}", file=sys.stderr)
golden["video"]["tiled_encode"] = tiled

# --- 4. audio DAC encoder: latent frames for a sweep of (padded) sample counts
AUDIO_NEED = ["snake", "Snake1d", "ResidualUnit", "EncoderBlock", "Encoder"]
ans = dict(torch=torch, nn=nn, F=F, math=math, comfy=comfy, ops=ops)
for name in AUDIO_NEED:
    exec(top_level(sys.argv[2], [name])[name], ans)
with torch.no_grad():
    enc = ans["Encoder"](64, (2, 4, 4, 5, 5), 2048)
    samples_cases = []
    for s in [1, 2, 400, 799, 800, 801, 1599, 1600, 1601, 4000, 32000, 32001, 96000]:
        hop = 800
        padded = math.ceil(s / hop) * hop
        out = enc(torch.zeros(2, 1, padded))     # b*s = 2 stereo streams, mono encoder
        samples_cases.append({"samples": s, "padded": padded, "T": out.shape[2]})
        print(f"audio encode {s:6d} (pad {padded}) -> T {out.shape[2]}", file=sys.stderr)
golden["audio"]["latents"] = samples_cases

json.dump(golden, open(sys.argv[3], "w"), indent=1)
print(f"wrote {sys.argv[3]}", file=sys.stderr)
