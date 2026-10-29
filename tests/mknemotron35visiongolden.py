"""Offline official Transformers/Pillow oracle for Nemotron 3.5's vision path.

Never called by CMake/CTest. Requires CPU torch/torchvision, transformers==5.20.0, NumPy and Pillow.
Random tiny weights retain the production patch size, head width, activation and merger layout.
"""

import json
import re
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import PIL
import torch
import torchvision
import transformers
from PIL import Image
from transformers import (
    Ministral3Config,
    Mistral3Config,
    PixtralVisionConfig,
    PixtralVisionModel,
)
from transformers.models.mistral3.modeling_mistral3 import Mistral3MultiModalProjector
from transformers.models.pixtral.image_processing_pil_pixtral import (
    PixtralImageProcessorPil,
)
from transformers.models.pixtral.processing_pixtral import PixtralProcessor

torch.set_num_threads(1)
torch.manual_seed(8275)
vc = PixtralVisionConfig(
    hidden_size=64,
    intermediate_size=96,
    num_hidden_layers=2,
    num_attention_heads=1,
    patch_size=14,
    image_size=1540,
    hidden_act="silu",
    rope_parameters={"rope_type": "axial", "rope_theta": 10000.0},
)
vc._attn_implementation = "eager"
tc = Ministral3Config(
    hidden_size=16,
    intermediate_size=32,
    num_hidden_layers=2,
    num_attention_heads=4,
    num_key_value_heads=1,
    head_dim=128,
    vocab_size=32,
    rms_norm_eps=1e-5,
    is_causal=False,
)
config = Mistral3Config(
    vision_config=vc,
    text_config=tc,
    spatial_merge_size=2,
    multimodal_projector_bias=False,
    projector_hidden_act="gelu",
    vision_feature_layer=-1,
)
vision = PixtralVisionModel(vc).eval()
frequencies = vision.patch_positional_embedding.inv_freq.clone()
vision.to(torch.bfloat16)
# Match from_pretrained(dtype=bfloat16): rotary frequency buffers stay float32.
vision.patch_positional_embedding.inv_freq = frequencies
vision.patch_positional_embedding.original_inv_freq = frequencies.clone()
projector = Mistral3MultiModalProjector(config).to(torch.bfloat16).eval()
with torch.no_grad():
    # The projector class is not a PreTrainedModel; initialise its linears explicitly.
    for name, weight in projector.named_parameters():
        weight.copy_(
            torch.randn(weight.shape) * 0.02
            if weight.ndim == 2
            else 1 + torch.randn(weight.shape) * 0.1
        )
    for name, weight in vision.named_parameters():
        if "norm" in name or "ln_pre" in name:
            weight.copy_(1 + 0.1 * torch.randn(weight.shape))

processor = PixtralImageProcessorPil(patch_size=28, size={"longest_edge": 1120})


def frame(f, h, w):
    y, x, c = np.indices((h, w, 3), dtype=np.uint64)
    with np.errstate(over="ignore"):
        z = np.uint64(f) * np.uint64(0x9E3779B97F4A7C15) + y * np.uint64(0xBF58476D1CE4E5B9)
        z += (
            x * np.uint64(0x94D049BB133111EB) + c * np.uint64(0x2545F4914F6CDD1D) + np.uint64(12345)
        )
        z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    return ((z ^ (z >> np.uint64(31))) & 255).astype(np.uint8)


def patchify(x):
    # Only fixture serialisation: the oracle's convolution still consumes real [C,H,W] pixels.
    c, h, w = x.shape
    return x.reshape(c, h // 14, 14, w // 14, 14).permute(1, 3, 0, 2, 4).reshape(-1, 3 * 14 * 14)


def bf16(x):
    return x.detach().contiguous().flatten().view(torch.uint16).tolist()


def fnv(values):
    h = 0xCBF29CE484222325
    for v in values:
        for b in (v & 255, v >> 8):
            h = ((h ^ b) * 0x100000001B3) & ((1 << 64) - 1)
    return f"{h:016x}"


images, pixels = [], []
for f, h, w in [
    (0, 1, 1),
    (1, 28, 56),
    (2, 29, 57),
    (3, 56, 28),
    (4, 56, 56),
    (5, 1500, 1000),
    (6, 1, 10000),
]:
    try:
        data = processor(images=[Image.fromarray(frame(f, h, w))], return_tensors="pt")
    except ValueError:
        if f != 6:
            raise
        # Flooring this extreme aspect ratio produces a zero side in the official processor.
        images.append({"f": f, "h": h, "w": w, "error": "zero resize side"})
        pixels.append(None)
        continue
    x = data["pixel_values"][0].to(torch.bfloat16)
    values = bf16(patchify(x))
    formatter = SimpleNamespace(
        patch_size=14,
        spatial_merge_size=2,
        image_token="[IMG]",
        image_break_token="[IMG_BREAK]",
        image_end_token="[IMG_END]",
    )
    expansion = PixtralProcessor.replace_image_token(formatter, data, 0)
    token_ids = {"[IMG]": 10, "[IMG_BREAK]": 12, "[IMG_END]": 13}
    images.append(
        {
            "f": f,
            "h": h,
            "w": w,
            "size": list(x.shape[1:]),
            "fnv": fnv(values),
            "expand": [token_ids[t] for t in re.findall(r"\[IMG(?:_BREAK|_END)?\]", expansion)],
            "patches": values if x.numel() < 20000 else [],
        }
    )
    pixels.append(x)

weights = {}
for prefix, model in [("vision_tower", vision), ("multi_modal_projector", projector)]:
    for name, w in model.state_dict().items():
        weights[prefix + "." + name] = {"shape": list(w.shape), "bf16": bf16(w)}

captured = {}
projector.norm.register_forward_hook(
    lambda m, a, y: captured.update(post_norm=y.detach().float().tolist())
)
projector.patch_merger.merging_layer.register_forward_pre_hook(
    lambda m, a: captured.update(packed=a[0].detach().float().tolist())
)
projector.patch_merger.register_forward_hook(
    lambda m, a, y: captured.update(merged=y.detach().float().tolist())
)
cases = []
for name, indices in [("wide", [1]), ("tall", [3]), ("square", [4]), ("mixed", [2, 0])]:
    xs = [pixels[i] for i in indices]
    sizes = torch.tensor([list(x.shape[1:]) for x in xs])
    h, w = sizes.max(0).values.tolist()
    padded = torch.stack(
        [torch.nn.functional.pad(x, (0, w - x.shape[2], 0, h - x.shape[1])) for x in xs]
    )
    with torch.no_grad():
        hidden = vision(padded, image_sizes=sizes).last_hidden_state.squeeze(0)
        out = projector(hidden, sizes)
    cases.append(
        {
            "name": name,
            "images": indices,
            "hidden": hidden.float().tolist(),
            **captured,
            "out": out.float().tolist(),
        }
    )

# Isolate every frequency and dtype-rounding seam, including coordinates near the max grid.
qkv = torch.linspace(-2, 2, 6 * 3 * 64).reshape(6, 3, 64).to(torch.bfloat16)
coords = torch.tensor([[0, 0], [1, 2], [3, 1], [63, 79], [79, 63], [109, 109]])
from transformers.models.pixtral.modeling_pixtral import apply_rotary_pos_emb

with torch.no_grad():
    cos, sin = vision.patch_positional_embedding(qkv, coords)
    q, k = apply_rotary_pos_emb(qkv[:, 0][None], qkv[:, 1][None], cos, sin, unsqueeze_dim=0)
    rotary = {
        "input": bf16(qkv),
        "coord": coords.T.flatten().tolist(),
        "output": bf16(torch.stack((q.squeeze(0), k.squeeze(0), qkv[:, 2]), dim=1)),
    }

result = {
    "provenance": {
        "transformers": transformers.__version__,
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "pillow": PIL.__version__,
        "seed": 8275,
        "processor": "PixtralImageProcessorPil",
        "attention": "eager",
        "dtype": "bfloat16",
        "rotary_dtype": "float32",
    },
    "images": images,
    "weights": weights,
    "cases": cases,
    "rotary": rotary,
}
path = Path(__file__).parent / "data" / "golden" / "nemotron35_vision.json"
path.write_text(json.dumps(result, separators=(",", ":")) + "\n")
print(path, path.stat().st_size)
