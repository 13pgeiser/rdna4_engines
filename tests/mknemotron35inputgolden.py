"""Offline input oracle; never invoked by CMake/CTest.

Use --tokenizer with tokenizer.json from nvidia/Nemotron-3.5-Embed-8B-BF16 revision
781e779ee4ca9c2d2a4f38321993556baeab49a1. Requires Jinja2 and Hugging Face tokenizers.
The checked-in template is copied from that same checkpoint revision.
"""

import argparse
import base64
import copy
import hashlib
import json
from pathlib import Path

import jinja2
import tokenizers
from jinja2 import StrictUndefined
from jinja2.sandbox import ImmutableSandboxedEnvironment

args = argparse.ArgumentParser()
args.add_argument("--tokenizer", type=Path, required=True)
opts = args.parse_args()
root = Path(__file__).parent
source = (root / "data/chat/nemotron35/template.jinja").read_text()
env = ImmutableSandboxedEnvironment(undefined=StrictUndefined)


def refuse(message):
    raise ValueError(message)


env.globals["raise_exception"] = refuse
template = env.from_string(source)
tok = tokenizers.Tokenizer.from_file(str(opts.tokenizer))
vision = json.loads((root / "data/golden/nemotron35_vision.json").read_text())
cases = []
expanded_checks = []
user_inputs = [
    ("plain", "What is NVIDIA NeMo Framework?"),
    ("literal_controls", "Treat [IMG], [IMG_BREAK], <s>, </s> and <__media__> literally."),
    ("unicode", "¿Qué ves? Καλημέρα 世界 👁️\n"),
    (
        "parts",
        [
            {"type": "text", "text": "first"},
            {"type": "text", "text": ""},
            {"type": "text", "text": "last"},
        ],
    ),
    ("image_only", [{"type": "image_url", "image_url": {"url": "data:image/png;base64,AA=="}}]),
    (
        "text_before_images",
        [
            {"type": "text", "text": "A [page]"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,AQ=="}},
            {"type": "text", "text": "<details>"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,Ag=="}},
        ],
    ),
    (
        "empty_blocks",
        [
            {"type": "text", "text": ""},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,Aw=="}},
            {"type": "text", "text": ""},
        ],
    ),
]
for prefix_name, prefix in [
    ("none", None),
    ("query", "query:   "),
    ("document", "passage: "),
    ("system_blocks", [{"type": "text", "text": "x "}, {"type": "text", "text": "y\t  "}]),
]:
    for name, content in user_inputs:
        messages = [] if prefix is None else [{"role": "system", "content": prefix}]
        messages.append({"role": "user", "content": content})
        rendered = template.render(messages=messages)
        # Record text segments between genuine image placeholders. Escaped user literals contain
        # a ZWNJ, so they cannot be confused with these boundaries.
        segments = rendered.split("[IMG]")
        image_blocks = (
            [] if isinstance(content, str) else [b for b in content if b["type"] == "image_url"]
        )
        expanded = segments[0]
        order = []
        for i, block in enumerate(image_blocks):
            index = base64.b64decode(block["image_url"]["url"].split(",")[1])[0]
            order.append(index)
            expanded += (
                "".join(tok.id_to_token(t) for t in vision["images"][index]["expand"])
                + segments[i + 1]
            )
        cases.append(
            {
                "name": prefix_name + "/" + name,
                "messages": messages,
                "rendered": rendered,
                "ids": tok.encode(rendered, add_special_tokens=False).ids,
                "expanded_ids": tok.encode(expanded, add_special_tokens=False).ids,
                "image_order": order,
                "segments": [
                    {"text": s, "ids": tok.encode(s, add_special_tokens=False).ids}
                    for s in segments
                ],
            }
        )
        if prefix_name == "none" and isinstance(content, str):
            cases[-1]["legacy_text"] = content
            cases[-1]["legacy_ids"] = tok.encode(content, add_special_tokens=False).ids
        expanded_checks.append((expanded, cases[-1]["expanded_ids"]))

# A sparse fixture vocabulary, not a deployable tokenizer: retain every byte and added token,
# and every BPE candidate that can occur in a fixture segment. Preserve original ids and merge
# ordering. Both HF and the C++ tests run this subset; HF must reproduce the full tokenizer first.
original = json.loads(opts.tokenizer.read_text())
subset = copy.deepcopy(original)
wanted = set(tokenizers.pre_tokenizers.ByteLevel.alphabet())
wanted.update(t["content"] for t in original["added_tokens"])
bytelevel = tokenizers.pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False)
for case in cases:
    samples = [seg["text"] for seg in case["segments"]]
    if "legacy_text" in case:
        samples.append(case["legacy_text"])
    for sample in samples:
        for text in (sample, tok.normalizer.normalize_str(sample)):
            encoded = "".join(p for p, offsets in bytelevel.pre_tokenize_str(text))
            wanted.update(
                encoded[i:j] for i in range(len(encoded)) for j in range(i + 1, len(encoded) + 1)
            )
vocab = {t: i for t, i in original["model"]["vocab"].items() if t in wanted}
merges = []
for merge in original["model"]["merges"]:
    a, b = merge if isinstance(merge, list) else merge.split(" ")
    if a in vocab and b in vocab and a + b in vocab:
        merges.append(merge)
subset["model"]["vocab"] = vocab
subset["model"]["merges"] = merges
subset_text = json.dumps(subset, ensure_ascii=False, separators=(",", ":"))
small = tokenizers.Tokenizer.from_str(subset_text)
for case in cases:
    assert small.encode(case["rendered"], add_special_tokens=False).ids == case["ids"]
    for seg in case["segments"]:
        assert small.encode(seg["text"], add_special_tokens=False).ids == seg["ids"]
    if "legacy_text" in case:
        assert small.encode(case["legacy_text"], add_special_tokens=False).ids == case["legacy_ids"]
for expanded, ids in expanded_checks:
    assert small.encode(expanded, add_special_tokens=False).ids == ids
(root / "data/chat/nemotron35/tokenizer-subset.json").write_text(subset_text + "\n")

result = {
    "provenance": {
        "model_revision": "781e779ee4ca9c2d2a4f38321993556baeab49a1",
        "jinja2": jinja2.__version__,
        "tokenizers": tokenizers.__version__,
        "template_sha256": hashlib.sha256(source.encode()).hexdigest(),
        "tokenizer_sha256": hashlib.sha256(opts.tokenizer.read_bytes()).hexdigest(),
        "subset_sha256": hashlib.sha256((subset_text + "\n").encode()).hexdigest(),
    },
    "cases": cases,
}
path = root / "data/golden/nemotron35_input.json"
path.write_text(json.dumps(result, ensure_ascii=False, separators=(",", ":")) + "\n")
print(path, path.stat().st_size)
