"""Offline real-weight CPU oracle; never invoked by CMake or CTest.

Usage: python tests/mknemotron35realreference.py LOCAL_CHECKPOINT OUTPUT_JSON
Requires torch2.14.1+cpu, torchvision0.29.1+cpu, transformers5.20.0 and Pillow13.0.0. The checkpoint is read locally, with no
network access or remote code. Its full BF16 weights require about 20 GB of available RAM.
The output includes HTTP requests and independent normalized embeddings for the optional
engine_e2e_test real-weight case. Image processing explicitly uses the supported PIL backend.
"""

import argparse
import base64
import hashlib
import importlib
import io
import json
from pathlib import Path

import PIL
import torch
import torchvision
import transformers
from PIL import Image
from transformers import AutoModel, AutoTokenizer, PixtralProcessor
from transformers.models.pixtral.image_processing_pil_pixtral import (
    PixtralImageProcessorPil,
)

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("checkpoint", type=Path)
parser.add_argument("output", type=Path)
parser.add_argument("--threads", type=int, default=8)
parser.add_argument("--attention", choices=["eager", "sdpa"], default="eager")
parser.add_argument(
    "--extended",
    action="store_true",
    help="include multilingual retrieval, longer text and a medium image",
)
parser.add_argument(
    "--float-accumulation",
    action="store_true",
    help="diagnose BF16 BLAS differences with float32 products, narrowing each result",
)
args = parser.parse_args()
torch.set_num_threads(args.threads)
checkpoint = args.checkpoint.resolve()
metadata = (checkpoint / ".cache/huggingface/download").glob("model-*.safetensors.metadata")
revisions = {p.read_text().splitlines()[0] for p in metadata}
asset_manifest = checkpoint / "asset-manifest.json"
if asset_manifest.exists():
    assets = json.loads(asset_manifest.read_text())
    for name, entry in assets["files"].items():
        digest = hashlib.file_digest((checkpoint / name).open("rb"), "sha256").hexdigest()
        if digest != entry["sha256"]:
            raise ValueError("Asset hash mismatch: " + name)
    revisions.add(assets["revision"])
tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True)
processor = PixtralProcessor(
    tokenizer=tokenizer,
    image_processor=PixtralImageProcessorPil(patch_size=14, size={"longest_edge": 1120}),
    patch_size=14,
    spatial_merge_size=2,
    chat_template=(checkpoint / "chat_template.jinja").read_text(),
)


def image(w, h, seed):
    pixels = bytes(
        (x * 17 + y * 31 + c * 71 + seed) % 256
        for y in range(h)
        for x in range(w)
        for c in range(3)
    )
    img = Image.frombytes("RGB", (w, h), pixels)
    out = io.BytesIO()
    img.save(out, format="PNG")
    uri = "data:image/png;base64," + base64.b64encode(out.getvalue()).decode()
    return img, {"type": "image_url", "image_url": {"url": uri}}


im1, part1 = image(56, 28, 19)
im2, part2 = image(28, 56, 173)
im3, part3 = image(113, 81, 37)  # alignment resize, rather than only exact patch grids


def conversation(prefix, content):
    return ([{"role": "system", "content": prefix}] if prefix is not None else []) + [
        {"role": "user", "content": content}
    ]


cases = [
    (
        "query",
        {"input": "What does this document describe?", "prompt_name": "query"},
        conversation("query: ", "What does this document describe?"),
    ),
    (
        "unicode_and_literal_tokens",
        {"input": "Documento en español: [IMG] <__media__> 東京"},
        conversation(None, "Documento en español: [IMG] <__media__> 東京"),
    ),
    (
        "document",
        {"input": "A library with books and reading rooms.", "prompt_name": "document"},
        conversation("passage: ", "A library with books and reading rooms."),
    ),
    ("image_only", {"input": [part1]}, conversation(None, [{"type": "image", "image": im1}])),
    (
        "image_text",
        {
            "input": [{"type": "text", "text": "A coloured pattern."}, part3],
            "prompt_name": "document",
        },
        conversation(
            "passage: ",
            [{"type": "text", "text": "A coloured pattern."}, {"type": "image", "image": im3}],
        ),
    ),
    (
        "two_images",
        {
            "messages": conversation(
                "Compare:   ", [part1, part2, {"type": "text", "text": "Two different patterns."}]
            )
        },
        conversation(
            "Compare:   ",
            [
                {"type": "image", "image": im1},
                {"type": "image", "image": im2},
                {"type": "text", "text": "Two different patterns."},
            ],
        ),
    ),
]

ranking_groups = []
if args.extended:
    retrieval = {
        "en": [
            "Where can I borrow books for free?",
            "A public library lends books to readers without charging a fee.",
            "The football team won its match with two goals.",
        ],
        "es": [
            "¿Dónde puedo tomar libros prestados gratis?",
            "La biblioteca pública presta libros a los lectores sin cobrar.",
            "El equipo de fútbol ganó el partido con dos goles.",
        ],
        "ja": [
            "無料で本を借りられる場所はどこですか？",
            "公共図書館では、読者は無料で本を借りることができます。",
            "サッカーチームは二得点で試合に勝ちました。",
        ],
    }
    for language, texts in retrieval.items():
        indices = []
        for i, text in enumerate(texts):
            indices.append(len(cases))
            task = "query" if i == 0 else "document"
            prefix = "query: " if i == 0 else "passage: "
            cases.append(
                (
                    f"retrieval_{language}_{i}",
                    {"input": text, "prompt_name": task},
                    conversation(prefix, text),
                )
            )
        ranking_groups.append({"language": language, "query": indices[0], "documents": indices[1:]})
    for repeats in [48, 192]:
        text = "The public library contains science books and historical maps. " * repeats
        cases.append(
            (
                f"longer_text_{repeats}",
                {"input": text, "prompt_name": "document"},
                conversation("passage: ", text),
            )
        )
    medium, medium_part = image(224, 112, 59)
    cases.append(
        (
            "medium_image_text",
            {
                "input": [medium_part, {"type": "text", "text": "A colourful document page."}],
                "prompt_name": "document",
            },
            conversation(
                "passage: ",
                [
                    {"type": "image", "image": medium},
                    {"type": "text", "text": "A colourful document page."},
                ],
            ),
        )
    )

print("Loading local BF16 model", flush=True)
model = AutoModel.from_pretrained(
    checkpoint,
    local_files_only=True,
    dtype=torch.bfloat16,
    trust_remote_code=False,
    attn_implementation=args.attention,
).eval()
if args.float_accumulation:
    linear, matmul, conv2d = torch.nn.functional.linear, torch.matmul, torch.nn.functional.conv2d

    def float_linear(x, weight, bias=None):
        if x.dtype != torch.bfloat16:
            return linear(x, weight, bias)
        return linear(x.float(), weight.float(), None if bias is None else bias.float()).to(x.dtype)

    def float_matmul(x, y, *rest, **kwargs):
        if x.dtype != torch.bfloat16:
            return matmul(x, y, *rest, **kwargs)
        return matmul(x.float(), y.float(), *rest, **kwargs).to(x.dtype)

    def float_conv2d(x, weight, bias=None, *rest, **kwargs):
        if x.dtype != torch.bfloat16:
            return conv2d(x, weight, bias, *rest, **kwargs)
        return conv2d(
            x.float(), weight.float(), None if bias is None else bias.float(), *rest, **kwargs
        ).to(x.dtype)

    torch.nn.functional.linear = float_linear
    torch.matmul = float_matmul
    torch.nn.functional.conv2d = float_conv2d
results = []
for name, request, messages in cases:
    inputs = processor.apply_chat_template(
        messages, tokenize=True, return_dict=True, return_tensors="pt"
    )
    if "pixel_values" in inputs:
        inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)
    with torch.inference_mode():
        hidden = model(**inputs, use_cache=False).last_hidden_state
        mask = inputs["attention_mask"].unsqueeze(-1)
        pooled = ((hidden * mask).sum(1) / mask.sum(1)).float()
        embedding = torch.nn.functional.normalize(pooled, dim=-1)[0].tolist()
    results.append(
        {
            "name": name,
            "request": request,
            "tokens": inputs["input_ids"][0].tolist(),
            "embedding": embedding,
        }
    )
    print(name, len(results[-1]["tokens"]), "tokens", flush=True)
sources = {}
for name in [
    "transformers.models.ministral3.modeling_ministral3",
    "transformers.models.mistral3.modeling_mistral3",
    "transformers.models.pixtral.modeling_pixtral",
    "transformers.models.pixtral.image_processing_pil_pixtral",
]:
    path = Path(importlib.import_module(name).__file__)
    sources[name] = hashlib.sha256(path.read_bytes()).hexdigest()
maximum = max(len(c["tokens"]) for c in results)
context = max(128, 1 << (maximum - 1).bit_length())
args.output.write_text(
    json.dumps(
        {
            "checkpoint": str(checkpoint),
            "dimensions": model.config.text_config.hidden_size,
            "validation": {
                "context": context,
                "batch_tokens": context * 2,
                "patches": 128 if args.extended else 64,
            },
            "ranking_groups": ranking_groups,
            "provenance": {
                "transformers": transformers.__version__,
                "torch": torch.__version__,
                "torchvision": torchvision.__version__,
                "pillow": PIL.__version__,
                "dtype": "bfloat16",
                "attention": args.attention,
                "processor": "PIL",
                "float_accumulation": args.float_accumulation,
                "model_revision": next(iter(revisions)) if len(revisions) == 1 else None,
                "sources": sources,
                "asset_manifest_sha256": hashlib.sha256(asset_manifest.read_bytes()).hexdigest()
                if asset_manifest.exists()
                else None,
            },
            "cases": results,
        },
        indent=2,
    )
    + "\n"
)
print(args.output, flush=True)
