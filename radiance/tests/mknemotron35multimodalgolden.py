"""Offline full Mistral3Model oracle, reusing the checked-in tiny text and vision weights.
Never run by CMake/CTest. Requires CPU torch and transformers==5.20.0.
"""

import json
from pathlib import Path

import torch
import transformers
from transformers import (
    Ministral3Config,
    Mistral3Config,
    Mistral3Model,
    PixtralVisionConfig,
)

root = Path(__file__).parent / "data/golden"
text = json.loads((root / "nemotron35.json").read_text())
vision = json.loads((root / "nemotron35_vision.json").read_text())
torch.set_num_threads(1)
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
    use_cache=False,
    rope_parameters={
        "rope_type": "yarn",
        "rope_theta": 1e6,
        "factor": 16.0,
        "original_max_position_embeddings": 16384,
        "beta_fast": 32.0,
        "beta_slow": 1.0,
        "mscale": 1.0,
        "mscale_all_dim": 1.0,
        "llama_4_scaling_beta": 0.1,
    },
)
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
config = Mistral3Config(
    text_config=tc,
    vision_config=vc,
    spatial_merge_size=2,
    image_token_index=10,
    multimodal_projector_bias=False,
    projector_hidden_act="gelu",
    vision_feature_layer=-1,
    is_causal=False,
)
model = Mistral3Model(config).eval()
frequencies = {name: b.clone() for name, b in model.named_buffers()}
model.to(torch.bfloat16)
for name, b in frequencies.items():
    parent, attr = name.rsplit(".", 1)
    setattr(model.get_submodule(parent), attr, b)
model.set_attn_implementation("eager")
assert model.language_model.config._attn_implementation == "eager"
assert model.vision_tower.config._attn_implementation == "eager"
state = text["weights"] | vision["weights"]
model.load_state_dict(
    {
        name: torch.tensor(w["bf16"], dtype=torch.uint16).view(torch.bfloat16).reshape(w["shape"])
        for name, w in state.items()
    },
    strict=True,
)


def sequence(parts):
    ids, placements = [], []
    for p in parts:
        if isinstance(p, list):
            ids.extend(p)
        else:
            placements.append({"image": p, "tok": len(ids)})
            ids.extend(vision["images"][p]["expand"])
    return {"tokens": ids, "placements": placements}


def pixels(index):
    image = vision["images"][index]
    h, w = image["size"]
    patches = torch.tensor(image["patches"], dtype=torch.uint16).view(torch.bfloat16)
    return patches.reshape(h // 14, w // 14, 3, 14, 14).permute(2, 0, 3, 1, 4).reshape(3, h, w)


image_only = sequence([1])
combined = sequence([[3, 4], 2, [5, 7, 8, 6]])
two_images = sequence([[3], 1, [5], 3, [5, 7, 9]])
plain = sequence([[3, 4, 7, 8]])
http_combined = sequence([2, [5, 7, 8, 6]])
http_two = sequence([[3, 4, 5], 1, [5], 3, [5, 7, 9]])
cases = []
for name, inputs in [
    ("image_only", [image_only]),
    ("image_text", [combined]),
    ("two_images", [two_images]),
    ("ragged", [image_only, combined, two_images]),
    ("text_and_image_batch", [plain, image_only]),
    ("http_image_text", [http_combined]),
    ("http_two_images", [http_two]),
]:
    length = max(len(x["tokens"]) for x in inputs)
    ids = torch.full((len(inputs), length), 11, dtype=torch.long)
    mask = torch.zeros_like(ids)
    image_indices = []
    for s, x in enumerate(inputs):
        ids[s, : len(x["tokens"])] = torch.tensor(x["tokens"])
        mask[s, : len(x["tokens"])] = 1
        image_indices.extend(p["image"] for p in x["placements"])
    imgs = [pixels(i) for i in image_indices]
    sizes = torch.tensor([list(p.shape[1:]) for p in imgs])
    h, w = sizes.max(0).values.tolist()
    padded = torch.stack(
        [torch.nn.functional.pad(x, (0, w - x.shape[2], 0, h - x.shape[1])) for x in imgs]
    )
    with torch.no_grad():
        output = model(
            input_ids=ids,
            attention_mask=mask,
            pixel_values=padded,
            image_sizes=sizes,
            position_ids=torch.arange(length)[None].expand(len(inputs), -1),
            use_cache=False,
        )
        # NVIDIA's documented masked BF16 sum/divide; padding is not part of the valid-token mean.
        pooled = (
            (output.last_hidden_state * mask[..., None]).sum(1) / mask.sum(1)[:, None]
        ).float()
    cases.append(
        {
            "name": name,
            "inputs": inputs,
            "pooled": pooled.tolist(),
            "hidden": [
                output.last_hidden_state[s, : len(x["tokens"])].float().tolist()
                for s, x in enumerate(inputs)
            ],
        }
    )
result = {
    "provenance": {
        "transformers": transformers.__version__,
        "torch": torch.__version__,
        "attention": "eager",
        "dtype": "bfloat16",
        "rotary_dtype": "float32",
        "weights": ["nemotron35.json", "nemotron35_vision.json"],
    },
    "cases": cases,
}
path = root / "nemotron35_multimodal.json"
path.write_text(json.dumps(result, separators=(",", ":")) + "\n")
print(path, path.stat().st_size)
