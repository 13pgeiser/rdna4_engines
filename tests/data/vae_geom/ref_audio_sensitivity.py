#!/usr/bin/env python3
"""#14 golden — reference-half integrity + content-sensitivity (audio encoder).

The device Aencode reportedly produces a content-INVARIANT latent (all sufficiently-nonzero
audio collapses to one vector, only silence separates) while the host scatter/staging provably
carry distinct content. #14 asks whether the DEVICE encoder MATH is wrong. This script builds the
REFERENCE half of that comparison on the CPU, in the ComfyUI venv's torch, using the reference
network classes themselves and the REAL HF weights -- no device, no shared kernels, read-only, so
it cannot regress the build.

What it establishes:

  1. INTEGRITY -- the real `encoder.*` / `pre_block.*` / `mean_proj.*` checkpoint loads under
     `load_state_dict(strict=True)` into the extracted classes. A clean load means the port's
     `aenc.*` container names are name-consistent with the genuine model (a mapping typo would
     surface here as a missing/unexpected key, exactly the class of bug that could masquerade as
     "kernel math").

  2. REFERENCE IS CONTENT-SENSITIVE, AT BOTH STAGES -- the reference, on the SAME synthetic
     contents the on-device audio test used (a 440 Hz tone, an 880 Hz tone, white noise, silence),
     gives pairwise-DISTINCT outputs at BOTH:
       (a) the raw conv encoder stream `Encoder(x)`  [1, 2048, T]   -- the f32 conv chain, and
       (b) the FULL latent `mean_proj(pre_block(Encoder(x)))`  [1, 32, T] -- exactly the quantity
           the device writes into `b_enca` (the per-channel latents_mean/std is a bijection and is
           omitted; it cannot turn distinct latents into identical ones).
     If both are distinct, the reference does not collapse the way the device does, so the
     divergence is on the device side. The stage-(b) latent is the target a device dump compares
     against once `b_enca` is read back.

Run:  <comfyui-venv>/bin/python ref_audio_sensitivity.py [audio_vae.py] [audio_vae.safetensors]
"""
import ast
import hashlib
import math
import sys
import types

import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)

VAE_PY = sys.argv[1] if len(sys.argv) > 1 else \
    "/home/dustin/AI/ComfyUI/comfy/ldm/minimax/audio_vae.py"
WEIGHTS = sys.argv[2] if len(sys.argv) > 2 else \
    "/home/dustin/AI/models/MiniMaxAI/MiniMax-H3/audio_vae/diffusion_pytorch_model.safetensors"


# ---- minimal comfy.ops shim (same discipline as gen_golden.py: CPU path only) ----
# The encode path reaches for these comfy.ops symbols:
#   cast_to_input      -- a plain dtype/device cast of a weight to the activation's dtype/device
#                         (faithful to comfy/ops.py on CPU: no offload, no requant).
#   cast_bias_weight   -- returns (weight, None, None) on CPU (no offload stream, no bias requant;
#                         CausalAttention re-supplies its own packed q/k/v bias below).
#   uncast_bias_weight -- no-op on CPU (nothing to un-offload).
#   scaled_dot_product_attention -- torch's SDPA (is_causal handled by torch).
# disable_weight_init is used by the module as `ops = comfy.ops.disable_weight_init` then
# `ops.Conv1d(...)`: it must be an OBJECT exposing the plain nn classes (init is irrelevant here,
# every weight is overwritten by load_state_dict).
class _DisableWeightInit:
    Conv1d = nn.Conv1d
    ConvTranspose1d = nn.ConvTranspose1d
    LayerNorm = nn.LayerNorm
    Linear = nn.Linear


class _Ops:
    Conv1d = nn.Conv1d
    ConvTranspose1d = nn.ConvTranspose1d
    GroupNorm = nn.GroupNorm
    LayerNorm = nn.LayerNorm
    Linear = nn.Linear
    disable_weight_init = _DisableWeightInit()

    @staticmethod
    def cast_to_input(weight, input, non_blocking=False, copy=True):
        return weight.to(dtype=input.dtype, device=input.device)

    @staticmethod
    def cast_bias_weight(module, input, bias=None, offloadable=False, **kw):
        return module.weight.to(dtype=input.dtype, device=input.device), None, None

    @staticmethod
    def uncast_bias_weight(module, weight, bias, offload_stream):
        return None

    @staticmethod
    def scaled_dot_product_attention(query, key, value, is_causal=False, **kw):
        return F.scaled_dot_product_attention(query, key, value, is_causal=is_causal)


comfy = types.ModuleType("comfy")
comfy.quant_ops = types.ModuleType("comfy.quant_ops")
comfy.ops = _Ops()


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


ans = dict(torch=torch, nn=nn, F=F, math=math, comfy=comfy, ops=_Ops.disable_weight_init)
for name in ["snake", "Snake1d", "ResidualUnit", "EncoderBlock", "Encoder",
             "GeGluMlp", "CausalAttention", "AttnProjection"]:
    exec(top_level(VAE_PY, [name])[name], ans)


class RefEncoder(nn.Module):
    """The encode-side half of MiniMaxH3AudioVAE -- encoder, pre_block, mean_proj -- without
    constructing the BigVGAN decoder (whose alias-free / Kaiser classes this harness does not
    need to run). Mirrors `MiniMaxH3AudioVAE.__init__`'s encoder fields exactly."""

    def __init__(self):
        super().__init__()
        self.encoder = ans["Encoder"](64, (2, 4, 4, 5, 5), 2048)
        self.pre_block = ans["AttnProjection"](2048, 32, num_heads=8)
        self.mean_proj = nn.Conv1d(32, 32, 1)

    def forward(self, x, full=True):
        x = self.encoder(x)                                   # [b, 2048, T]
        if not full:
            return x
        x = self.pre_block(x.transpose(1, 2)).transpose(1, 2)  # [b, 32, T]
        return self.mean_proj(x)                              # [b, 32, T]


def load_encoder():
    """Reference encode chain with the REAL HF weights bound. strict=True is the integrity gate."""
    enc = RefEncoder()
    sd = _read_state(WEIGHTS)
    enc.load_state_dict(sd, strict=True)                       # raises on any mismatch
    return enc


def _read_state(path):
    """Pull the encode-side tensors out of the checkpoint as a bare state_dict, folding the
    weight-norm parametrization (`weight_g`/`weight_v`) into the single plain `.weight` the
    extracted Conv1d holds. The convert splits this the same way -- it stores `v` and a separate
    per-output-row scale g[o]/||v[o]|| (core/format/checkpoint.cpp resolve_wnorm); folding into
    .weight here reproduces the identical effective linear map, and a per-row scale cannot flatten
    content (distinct inputs still give distinct dot products through any nonzero v), so this is
    the faithful reference. Only the `encoder.*` convs are weight-norm; `pre_block.*` /
    `mean_proj.*` Linears/Conv/LayerNorms load verbatim. latents_mean/std live in the container
    meta, not here, and are a per-channel bijection omitted for the distinctness test."""
    from safetensors import safe_open
    raw = {}
    with safe_open(path, framework="pt") as f:
        for k in f.keys():
            if k.startswith(("encoder.", "pre_block.", "mean_proj.")):
                raw[k] = f.get_tensor(k)
    out = {}
    for k, g in raw.items():
        if not k.endswith(".weight_g"):
            continue
        stem = k[: -len("weight_g")]                     # e.g. "encoder.block.1.block.0.block.1."
        v = raw[stem + "weight_v"]
        # classic weight_norm (dim=0): w[o] = g[o] * v[o] / ||v[o]|| over every non-output dim
        norm = v.flatten(1).norm(dim=1).view([-1] + [1] * (v.dim() - 1))
        out[stem + "weight"] = g / norm * v
    for k, v in raw.items():
        if k.endswith((".weight_g", ".weight_v")):
            continue
        out[k] = v
    return out


def contents():
    """The same four contents the on-device audio differential used, same length, padded to the
    encoder hop (800) exactly as the driver pads an audio tail to a whole hop. The reference
    encode takes a stereo [B, 2, L] waveform; the port's encoder is mono and the server duplicates
    the mono stream across the two channels, so a mono reference is stacked twice here."""
    hop, n = 800, 800 * 40                       # 32000 samples = 40 latent frames
    def mono(x):
        return torch.zeros(1, hop * math.ceil(n / hop)).scatter_(
            1, torch.arange(n).view(1, n), x.view(1, n))
    t = torch.arange(n, dtype=torch.float32) / 16000.0
    def stereo(x):
        m = mono(x)
        return torch.stack([m, m], dim=1)         # [1, 2, L] -- mono spread to both channels
    return {
        "silence": stereo(torch.zeros(n)),
        "tone440": stereo(0.3 * torch.sin(2 * math.tau * 440 * t)),
        "tone880": stereo(0.3 * torch.sin(2 * math.tau * 880 * t)),
        "noise":   stereo(0.3 * torch.randn(n)),
    }


def digest(z):
    b = z.detach().float().numpy().tobytes()
    return hashlib.sha256(b).hexdigest()[:12], float(z.abs().max()), float(z.sum())


def run(enc, full, label):
    sigs = {}
    with torch.no_grad():
        for name, x in contents().items():
            # encode(): pad each channel to the hop and fold the two stereo channels into the batch
            b, s, length = x.shape
            pad = (hop := 800) * math.ceil(length / hop) - length
            x = F.pad(x, (0, pad)).reshape(b * s, 1, -1)
            z = enc(x, full=full)
            sigs[name] = digest(z)
            print(f"    {name:8s} shape={list(z.shape)} sha={sigs[name][0]} "
                  f"maxabs={sigs[name][1]:.4f} sum={sigs[name][2]:.2f}", file=sys.stderr)
    distinct = len({v[0] for v in sigs.values()}) == len(sigs)
    n = len(set(v[0] for v in sigs.values()))
    print(f"  {label}: content-sensitive={distinct}  ({n}/{len(sigs)} distinct)", file=sys.stderr)
    return distinct


hop = 800

if __name__ == "__main__":
    enc = load_encoder()
    enc.eval()
    print("integrity: encoder.* + pre_block.* + mean_proj.* loaded strict=True OK", file=sys.stderr)
    a = run(enc, full=False, label="raw Encoder(x) [f32 conv stream, 2048xT]")
    b = run(enc, full=True, label="FULL mean_proj(pre_block(Encoder(x))) [b_enca, 32xT]")
    sys.exit(0 if (a and b) else 3)
