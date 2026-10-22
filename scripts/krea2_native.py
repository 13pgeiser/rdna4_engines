#!/usr/bin/env python3
# krea2_native.py -- a Krea 2 DiT single file in the reference code's key names, as diffusers'
# Krea2Transformer2DModel file, so rad-convert reads it as the transformer/ of a pipeline directory.
#
#   krea2_native.py <single file> <pipeline>/transformer/diffusion_pytorch_model.safetensors
#
# Krea's own turbo.safetensors / raw.safetensors, and the finetunes civitai and ComfyUI ship, name
# the DiT by krea-ai/krea-2 mmdit.py (`blocks.N.attn.wq`, `txtfusion.refiner_blocks.N.mlp.up`,
# `tproj.1`) and carry no text encoder or VAE. Each tensor maps to one diffusers tensor, the same
# bytes under a new name; the blocks' modulation tables are flat ([6 * D]) where diffusers stores
# them [6, D], so only their shape is restated. Checked against krea/Krea-2-Turbo, whose turbo.safetensors
# and diffusers shards hold byte-identical tensors under this map (norms, tables, embedders, linears).
#
# The rest of the pipeline directory (model_index.json, scheduler/, text_encoder/, tokenizer/, vae/,
# transformer/config.json) is the release's: Raw and Turbo ship identical parts but for
# model_index.json, whose `is_distilled` selects Turbo's sampling (8 steps, no guidance). A finetune
# stored as plain fp8 converts with data/recipes/krea-2-fp8-unscaled.recipe.
#
# Needs only python3; the data is copied, never decoded.
import json
import os
import re
import shutil
import struct
import sys

RULES = [
    (r"^blocks\.(\d+)\.", r"transformer_blocks.\1."),
    (r"^txtfusion\.", r"text_fusion."),
    (r"\.attn\.wq\.", ".attn.to_q."), (r"\.attn\.wk\.", ".attn.to_k."), (r"\.attn\.wv\.", ".attn.to_v."),
    (r"\.attn\.wo\.", ".attn.to_out.0."), (r"\.attn\.gate\.", ".attn.to_gate."),
    (r"\.attn\.qknorm\.qnorm\.scale$", ".attn.norm_q.weight"),
    (r"\.attn\.qknorm\.knorm\.scale$", ".attn.norm_k.weight"),
    (r"\.mlp\.(gate|up|down)\.", r".ff.\1."),
    (r"\.prenorm\.scale$", ".norm1.weight"), (r"\.postnorm\.scale$", ".norm2.weight"),
    (r"\.mod\.lin$", ".scale_shift_table"),
    (r"^first\.", "img_in."),
    (r"^last\.linear\.", "final_layer.linear."), (r"^last\.norm\.scale$", "final_layer.norm.weight"),
    (r"^last\.modulation\.lin$", "final_layer.scale_shift_table"),
    (r"^tmlp\.0\.", "time_embed.linear_1."), (r"^tmlp\.2\.", "time_embed.linear_2."),
    (r"^tproj\.1\.", "time_mod_proj."),
    (r"^txtmlp\.0\.scale$", "txt_in.norm.weight"),
    (r"^txtmlp\.1\.", "txt_in.linear_1."), (r"^txtmlp\.3\.", "txt_in.linear_2."),
]


def to_diffusers(k):
    for a, b in RULES:
        k = re.sub(a, b, k)
    return k


def main():
    if len(sys.argv) != 3:
        sys.exit("usage: krea2_native.py <single file> <out .safetensors>")
    src, dst = sys.argv[1], sys.argv[2]
    with open(src, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        h = json.loads(f.read(n))
    meta = h.pop("__metadata__", None)
    out = {}
    for k, v in h.items():
        dk = to_diffusers(k)
        if dk == k or dk in out:
            sys.exit(f"krea2_native.py: '{k}' is not a Krea 2 DiT tensor this map knows")
        v = dict(v)
        if dk.endswith(".scale_shift_table") and len(v["shape"]) == 1:
            if v["shape"][0] % 6:
                sys.exit(f"krea2_native.py: '{k}' is {v['shape']}, not six rows of the width")
            v["shape"] = [6, v["shape"][0] // 6]
        out[dk] = v
    if meta:
        out["__metadata__"] = meta
    hb = json.dumps(out, separators=(",", ":")).encode()
    hb += b" " * (-len(hb) % 8)
    with open(src, "rb") as f, open(dst + ".tmp", "wb") as g:
        g.write(struct.pack("<Q", len(hb)))
        g.write(hb)
        f.seek(8 + n)
        shutil.copyfileobj(f, g, 64 << 20)
    os.replace(dst + ".tmp", dst)
    print(f"{len(h)} tensors -> {dst}")


if __name__ == "__main__":
    main()
