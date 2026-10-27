#!/usr/bin/env python3
"""Generate tests/data/vae_geom/golden.json from the reference VAEs themselves.

As tests/data/ref2va/gen_golden.py does, the module source is AST-parsed and only the classes and
helpers needed are extracted and run on torch alone, without importing the comfy package and the
runtime it brings. Unlike the pack golden this one runs the networks on CPU, so the expectations
are the encoders' own output shapes:

  * the video encoder (EncoderFCN3D with the reference's constructor arguments) and the
    reference's own encode()/encode_temporal()/tiled_encode() methods bound to a stand-in self:
    the latent count of each frame count, and the moments' shape a tiled encode keeps;
  * the audio DAC encoder: the latent frames of a sweep of sample counts. Its posterior head
    (AttnProjection and a 1x1 mean_proj) has no strided op, so the encoder alone fixes the count.

Two stand-ins are needed: comfy.quant_ops must exist without `ck` (the fused-pad fast path then
disables itself and the CPU path is plain torch), and Conv3d must take the reference's `autopad`
keyword. On CPU its one use is the single-frame causal case, "causal_zero": kernel_t - 1 zero
frames in front, which is what dropping the pad's taps comes to for a causal convolution.

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


def dump(v, f, ind=0):
    """JSON with an object a key a line and every list on one line, so a golden diffs by key."""
    if isinstance(v, dict) and v:
        pad = " " * (ind + 1)
        items = list(v.items())
        f.write("{\n")
        for i, (k, x) in enumerate(items):
            f.write(pad + json.dumps(k) + ": ")
            dump(x, f, ind + 1)
            f.write(",\n" if i + 1 < len(items) else "\n")
        f.write(" " * ind + "}")
    else:
        f.write(json.dumps(v, separators=(", ", ": ")))


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

# --- 2. the tile grid end to end: the blend/cut keeps the moments at exactly ratio-reduced size
with torch.no_grad():
    tiled = []
    for x_shape in [(1, 3, 5, 320, 320), (1, 3, 2, 512, 256), (1, 3, 17, 288, 192)]:
        moments = vae.tiled_encode(torch.zeros(*x_shape))
        tiled.append({"x_shape": list(x_shape), "moments_shape": list(moments.shape)})
        print(f"video tiled {list(x_shape)} -> moments {list(moments.shape)}", file=sys.stderr)
golden["video"]["tiled_encode"] = tiled

# --- 3. audio DAC encoder: latent frames for a sweep of (padded) sample counts
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

with open(sys.argv[3], "w") as f:
    dump(golden, f)
    f.write("\n")
print(f"wrote {sys.argv[3]}", file=sys.stderr)
