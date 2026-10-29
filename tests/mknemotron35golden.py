"""Offline reference generator; never invoked by CMake or CTest.

Run with CPU PyTorch and transformers==5.20.0. No checkpoint download is needed. The fixture
uses the official Ministral3Model with BF16 weights/activations and eager bidirectional attention.
The recorded weights let C++ execute the architecture's actual issued graph against the oracle.
"""

import json
from pathlib import Path

import torch
import transformers
from transformers import Ministral3Config, Ministral3Model

torch.set_num_threads(1)
torch.manual_seed(413)
config = Ministral3Config(
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
config._attn_implementation = "eager"
model = Ministral3Model(config).eval()
frequencies = model.rotary_emb.inv_freq.clone()
model.to(torch.bfloat16)
# from_pretrained(dtype=bfloat16) keeps these nonpersistent buffers in float32. A blanket .to()
# would narrow them and generate a different reference, most visibly at long positions.
model.rotary_emb.inv_freq = frequencies
model.rotary_emb.original_inv_freq = frequencies.clone()
# Non-unit gains catch misplaced/omitted norms. Keep them near one to resemble real weights.
with torch.no_grad():
    for name, weight in model.named_parameters():
        if "norm" in name:
            weight.copy_(1 + 0.1 * torch.randn(weight.shape))

weights = {
    "language_model." + name: {
        "shape": list(w.shape),
        "bf16": w.detach().flatten().view(torch.uint16).tolist(),
    }
    for name, w in model.state_dict().items()
}
cases = []
for name, inputs, offsets in [
    ("short", [[3, 4, 7, 12, 6]], [0]),
    ("ragged", [[3, 4, 7], [12, 6, 2, 8, 9]], [0, 0]),
    ("q_scale_boundary", [[3, 4, 7, 12]], [16382]),
]:
    expected = []
    for ids, offset in zip(inputs, offsets):
        with torch.no_grad():
            hidden = model(
                torch.tensor([ids]),
                attention_mask=torch.ones((1, len(ids)), dtype=torch.long),
                position_ids=torch.arange(offset, offset + len(ids))[None],
                use_cache=False,
            ).last_hidden_state
            # Reproduce NVIDIA's documented BF16 masked sum/division before float normalization.
            pooled = (hidden.sum(1) / len(ids)).float()
        expected.append(pooled.flatten().tolist())
    cases.append({"name": name, "inputs": inputs, "offsets": offsets, "pooled": expected})

output = {
    "provenance": {
        "transformers": transformers.__version__,
        "torch": torch.__version__,
        "seed": 413,
        "attention": "eager",
        "dtype": "bfloat16",
        "rotary_dtype": "float32",
    },
    "weights": weights,
    "cases": cases,
}
path = Path(__file__).parent / "data" / "golden" / "nemotron35.json"
path.write_text(json.dumps(output, separators=(",", ":")) + "\n")
print(path, path.stat().st_size)
