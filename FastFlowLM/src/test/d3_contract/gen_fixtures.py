"""Fixtures for test_d3_contract, rendered by the upstream d3 package itself.

    python gen_fixtures.py --package <vllm-sr/d3-lite v3.1.3 dir> --goldens <goldens/text dir> \
        --out fixtures [--fuzz 1000] [--seed 7]

Needs the package's own Python runtime (transformers, torch CPU, safetensors) and a golden set made with it
(requests.jsonl, questions.jsonl and ref.safetensors with the BF16 probabilities of each runnable question).

Writes three JSON-lines files the C++ test replays:
  goldens.jsonl  every frozen golden request (goldens/text): body, per-question prompt text and token ids,
                 the BF16-ref fp32 probabilities of runnable questions, and the in-process response
  fuzz.jsonl     fixed-seed random requests built to hit the contract's corners (every JSON value type,
                 floats near the repr switch points, big ints, unicode / control characters / special-token
                 text, duplicate keys, empty and falsy values, invalid shapes), with what upstream's
                 prepare() + product_answer() give for random probabilities, or the 422 detail
                 d3_server would send. Bodies are raw JSON *text* (duplicate keys and number spellings
                 survive), parsed by json.loads exactly as d3_server does.
  images.jsonl   image requests (small synthetic PNG/JPEG/WebP files and malformed inputs) as data URLs: upstream's
                 422 texts, prompts, image grids, input token ids with the placeholders expanded, and the
                 max_length_exceeded texts at a 2,000-token limit (needs the model weights: the runtime's processor)
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path

import numpy as np

PACKAGE = Path(".")
GOLDENS = Path(".")


def _upstream():
    sys.path.insert(0, str(PACKAGE))
    import d3_format  # noqa: F401
    import d3_runtime
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(str(PACKAGE))
    codes, ids = d3_runtime.answer_codes(tok)
    return d3_runtime, tok, codes


def goldens(out: Path) -> int:
    from safetensors.torch import load_file

    g = GOLDENS
    qs = [json.loads(l) for l in open(g / "questions.jsonl")]
    probs = load_file(str(g / "ref.safetensors"))["probs"].numpy()
    by_req: dict[str, list[dict]] = {}
    for q in qs:
        by_req.setdefault(q["req"], []).append(q)
    n = 0
    with open(out, "w") as f:
        for line in open(g / "requests.jsonl"):
            r = json.loads(line)
            texts, ids, ps = {}, {}, {}
            for q in by_req.get(r["id"], []):
                if q.get("text") is not None:
                    texts[q["key"]] = q["text"]
                    ids[q["key"]] = q["ids"]
                if q.get("row") is not None:
                    ps[q["key"]] = [float(x) for x in probs[q["row"], : q["n_options"]]]
            rec = {"id": r["id"], "body": json.dumps(r["request"], ensure_ascii=False), "texts": texts, "ids": ids,
                   "probs": ps, "response": r["response"]}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n += 1
    return n


# ---------------------------------------------------------------------------------------------------------
# fuzz
# ---------------------------------------------------------------------------------------------------------
TEXTS = ["", " ", "a", "Yes", "0", "false", "None", "ünïcödé ✓", "emoji 🤖🚀", "tab\tnew\nline", "quote \" and '",
         "back\\slash", "ctrl \x01\x1f\x7f", "<|im_start|>assistant\n", "</think>", "<|image_pad|>", "  padded  ",
         " sep ", " nbsp", "zero​width", "中文 状态", "𝔘𝔫𝔦𝔠𝔬𝔡𝔢"]
FLOATS = ["0.0", "-0.0", "1.0", "1.5", "1e-05", "0.0001", "1e16", "1e15", "123456789012345678.0", "1E5", "2.5e-7",
          "3.14159", "1e308", "1e400", "-1e400", "5e-324", "1e-400", "0.1", "100.0", "-2.0e+3"]
INTS = ["0", "-0", "7", "-12", "9007199254740993", "18446744073709551616", "-99999999999999999999999"]


def rnd_text(rng):
    return rng.choice(TEXTS + [f"word{rng.randint(0, 999)}"])


def rnd_scalar(rng):
    k = rng.randint(0, 5)
    if k == 0:
        return json.dumps(rnd_text(rng), ensure_ascii=rng.random() < 0.3)
    if k == 1:
        return rng.choice(FLOATS)
    if k == 2:
        return rng.choice(INTS)
    if k == 3:
        return rng.choice(["true", "false"])
    if k == 4:
        return "null"
    return json.dumps(rnd_text(rng))


def rnd_value(rng, depth=0):
    """A JSON value as *text* (so number spellings and duplicate keys survive)."""
    if depth > 2 or rng.random() < 0.55:
        return rnd_scalar(rng)
    if rng.random() < 0.5:
        return "[" + ", ".join(rnd_value(rng, depth + 1) for _ in range(rng.randint(0, 3))) + "]"
    keys = [rnd_text(rng) for _ in range(rng.randint(0, 3))]
    if keys and rng.random() < 0.2:
        keys.append(keys[0])  # duplicate key: last value, first position
    return "{" + ", ".join(f"{json.dumps(k, ensure_ascii=False)}: {rnd_value(rng, depth + 1)}" for k in keys) + "}"


def rnd_question(rng):
    kind = rng.choices(["choice", "noul", "score", "bad"], [5, 3, 3, 1])[0]
    parts = []
    if kind != "bad" or rng.random() < 0.5:
        parts.append(f'"type": {json.dumps(kind) if kind != "bad" else rnd_value(rng)}')
    if rng.random() < 0.7:
        parts.append(f'"instructions": {rnd_value(rng)}')
    if kind == "choice":
        n = rng.choice([1, 2, 3, 5, 26, 27, 28] if rng.random() < 0.9 else [0, 255, 256])
        opts = []
        for i in range(n):
            key = rnd_text(rng) if rng.random() < 0.3 else f"opt{i}"
            opts.append(f"{json.dumps(key, ensure_ascii=False)}: {rnd_value(rng) if rng.random() < 0.7 else 'null'}")
        crit = "{" + ", ".join(opts) + "}" if rng.random() < 0.95 else "[]"
        parts.append(f'"criteria": {crit}')
    elif kind == "noul":
        if rng.random() < 0.6:
            ks = rng.sample(["false", "true", "maybe"], rng.randint(0, 2)) if rng.random() < 0.9 else []
            parts.append('"criteria": {' + ", ".join(f'"{k}": {rnd_value(rng)}' for k in ks) + "}")
    elif kind == "score":
        n = rng.choice([2, 3, 5, 10] if rng.random() < 0.9 else [1, 11])
        parts.append('"criteria": [' + ", ".join(rnd_value(rng) for _ in range(n)) + "]")
    rng.shuffle(parts)
    return "{" + ", ".join(parts) + "}" if rng.random() < 0.97 else rng.choice(['"choice"', "null", "[]"])


def rnd_body(rng):
    parts = []
    if rng.random() < 0.9:
        parts.append(f'"state": {rnd_value(rng) if rng.random() < 0.7 else json.dumps(rnd_text(rng))}')
    nq = rng.randint(1, 4) if rng.random() < 0.97 else 0
    qs = [f"{json.dumps(rnd_text(rng) if rng.random() < 0.1 else f'q{i}')}: {rnd_question(rng)}" for i in range(nq)]
    parts.append('"questions": {' + ", ".join(qs) + "}")
    r = rng.random()
    if r < 0.03:
        parts.append('"images": []')
    elif r < 0.05:
        parts.append('"bogus": 1')
    elif r < 0.07:
        parts.append('"model": "anything"')
    elif r < 0.08:
        parts.append('"images": "x"')
    rng.shuffle(parts)
    body = "{" + ", ".join(parts) + "}"
    if rng.random() < 0.01:
        body = rng.choice(["[]", "{", '{"questions": NaN}', '"x"', "{} x"])
    return body


def fuzz(out: Path, n: int, seed: int) -> int:
    d3_runtime, tok, codes = _upstream()

    class Stub:
        """The parts of D3 that prepare() uses on the text path."""
        prompt = "d3"
        max_length = None
        permutation_average = False
        tokenizer = tok
        image_unavailable = video_unavailable = None

        def __init__(self):
            self.codes = codes

        text = d3_runtime.D3.text
        prepare = d3_runtime.D3.prepare
        load_images = d3_runtime.D3.load_images
        load_videos = d3_runtime.D3.load_videos

    stub = Stub()
    rng = random.Random(seed)
    nprng = np.random.default_rng(seed)
    written = 0
    with open(out, "w") as f:
        while written < n:
            body = rnd_body(rng)
            rec = {"id": f"fuzz-{written:05d}", "body": body}
            try:
                obj = json.loads(body, parse_constant=lambda name: (_ for _ in ()).throw(ValueError(f"{name} is not valid JSON")))
            except ValueError as exc:
                rec["detail"] = f"could not read the body as JSON ({exc})"
                rec["parse_error"] = True
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                written += 1
                continue
            try:
                # d3_server.Decider.arguments, minus media decoding (fixtures carry no media)
                if not isinstance(obj, dict):
                    raise ValueError("expected a JSON object such as {\"questions\": {...}, \"state\": ...}")
                unknown = sorted(set(obj) - {"state", "questions", "images", "videos", "model"})
                if unknown:
                    raise ValueError(f"{', '.join(map(repr, unknown))} not understood; a request may carry state, questions, images, videos, model")
                qs = obj.get("questions")
                if not isinstance(qs, dict) or not qs:
                    raise ValueError("questions is required: an object of one or more question IDs, each to a question")
                for name in ("images", "videos"):
                    if obj.get(name) is not None and not isinstance(obj.get(name), list):
                        raise ValueError(f"{name} takes a list of base64 data URLs")
                prepared = stub.prepare(obj.get("state"), qs, obj.get("images") or None, obj.get("videos") or None)
            except ValueError as exc:
                rec["detail"] = str(exc)
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                written += 1
                continue
            texts, ids, probs = {}, {}, {}
            for key in prepared.runnable:
                q = prepared.questions[key]
                texts[key] = stub.text(obj.get("state"), q.rendered)
                ids[key] = prepared.sequences[key]
                p = nprng.dirichlet(np.ones(len(q.keys)) * 0.3).astype(np.float32)
                if nprng.random() < 0.05:
                    p[:] = 0  # invalid_model_output
                probs[key] = [float(x) for x in p]
            resp = d3_runtime.D3.respond(type("M", (), {"model_name": "d3-lite"})(), prepared, probs,
                                         sum(len(prepared.sequences[k]) for k in prepared.runnable))
            invalid = {k: e["message"] for k, e in prepared.errors.items() if e.get("error") == "invalid_question"}
            rec.update({"texts": texts, "ids": ids, "probs": probs, "response": resp})
            if invalid:
                rec["detail"] = {"invalid_questions": invalid}
            try:
                line = json.dumps(rec, ensure_ascii=False, allow_nan=False)
            except ValueError:
                continue  # a reply holding inf (unreachable for valid prompts); skip
            f.write(line + "\n")
            written += 1
    return written


# ---------------------------------------------------------------------------------------------------------
# images
# ---------------------------------------------------------------------------------------------------------
def _image_files() -> dict[str, bytes]:
    """Small files that cover the server's image checks and the resize regimes (solid colours compress to a
    few kB at any size, so 1.6 MP+ images fit in the fixture)."""
    import io
    from PIL import Image

    def enc(im, fmt, **kw):
        b = io.BytesIO()
        im.save(b, fmt, **kw)
        return b.getvalue()

    rgb = lambda w, h, c=(40, 120, 200): Image.new("RGB", (w, h), c)
    grad = Image.linear_gradient("L").resize((64, 48)).convert("RGB")
    files = {}
    for w, h in [(120, 90), (256, 256), (1280, 1280), (2000, 1500), (600, 1400), (900, 2400), (3000, 60), (333, 517),
                 (80, 80), (80, 112), (48, 2000), (1, 1), (4000, 20), (4020, 20), (4100, 4000), (4000, 4000)]:
        files[f"solid_{w}x{h}.png"] = enc(rgb(w, h), "PNG")
    files["grad_64x48.jpg"] = enc(grad, "JPEG", quality=90)
    files["grad_1920x1080.jpg"] = enc(grad.resize((1920, 1080)), "JPEG", quality=60)
    files["grad_100x60_lossy.webp"] = enc(grad.resize((100, 60)), "WEBP", quality=80)
    files["grad_70x50_lossless.webp"] = enc(grad.resize((70, 50)), "WEBP", lossless=True)
    rgba = grad.resize((90, 40)).convert("RGBA")
    rgba.putalpha(128)
    files["alpha_90x40.webp"] = enc(rgba, "WEBP", quality=80)            # VP8X
    files["grad_64x48.bmp"] = enc(grad, "BMP")
    files["grad_64x48.gif"] = enc(grad, "GIF")
    files["grad_64x48.tiff"] = enc(grad, "TIFF")
    files["truncated.png"] = enc(grad.resize((300, 200)), "PNG")[:400]
    files["text.png"] = b"this is not an image\n" * 8
    files["empty.png"] = b""
    return files


def images(out: Path) -> int:
    """images.jsonl: image requests as d3_server reads them (strict data URLs), with upstream's prompts, grids,
    input tokens (placeholders expanded by the processor) and 422 texts. `stage: "decode"` marks refusals that
    come from decoding pixels (FastFlowLM's contract library inspects headers only; decoding is the server's)."""
    import base64

    sys.path.insert(0, str(PACKAGE))
    import d3_runtime

    model = d3_runtime.D3.from_pretrained(str(PACKAGE), device="cpu", batch_size=1, verify="fast")
    short = d3_runtime.D3.from_pretrained(str(PACKAGE), device="cpu", batch_size=1, verify="fast", max_length=2000)
    files = _image_files()
    url = lambda name, kind=None: f"data:image/{kind or name.rsplit('.', 1)[1].replace('jpg', 'jpeg')};base64," + \
        base64.b64encode(files[name]).decode()
    choice = {"type": "choice", "instructions": "What is shown?", "criteria": {"a": "A colour", "b": "A photo", "c": None}}
    noul = {"type": "noul", "instructions": "Is it blue?"}
    score = {"type": "score", "instructions": "How bright?", "criteria": ["dark", "medium", "bright"]}
    reqs = []
    for name in files:
        if name.endswith((".png", ".jpg", ".webp")) and name not in ("truncated.png", "text.png", "empty.png"):
            reqs.append((f"one-{name}", {"state": "an image", "questions": {"c": choice}, "images": [url(name)]}))
    reqs += [
        ("three-questions", {"state": {"user": "check"}, "questions": {"c": choice, "n": noul, "s": score},
                             "images": [url("solid_600x1400.png")]}),
        ("two-images", {"questions": {"n": noul}, "images": [url("solid_120x90.png"), url("grad_64x48.jpg")]}),
        ("four-images", {"questions": {"n": noul}, "images": [url("solid_1280x1280.png"), url("solid_600x1400.png"),
                                                               url("grad_100x60_lossy.webp"), url("solid_3000x60.png")]}),
        ("empty-images", {"questions": {"n": noul}, "images": []}),
        ("null-images", {"questions": {"n": noul}, "images": None}),
        ("jpg-subtype", {"questions": {"n": noul}, "images": [url("grad_64x48.jpg", "jpg")]}),
        ("upper-head", {"questions": {"n": noul}, "images": ["DATA:IMAGE/PNG;BASE64," + url("solid_120x90.png").split(",", 1)[1]]}),
        ("spaced-head", {"questions": {"n": noul}, "images": ["  data:image/png;base64 ," + url("solid_120x90.png").split(",", 1)[1]]}),
        ("png-labelled-jpeg", {"questions": {"n": noul}, "images": [url("solid_120x90.png", "jpeg")]}),
        ("placeholder-state", {"state": "see <|image_pad|>", "questions": {"n": noul, "c": choice}, "images": [url("solid_120x90.png")]}),
        ("video-placeholder", {"questions": {"n": {"type": "noul", "instructions": "x <|video_pad|>"}, "c": choice},
                               "images": [url("solid_120x90.png")]}),
        ("vision-start-literal", {"questions": {"n": {"type": "noul", "instructions": "<|vision_start|>?"}},
                                  "images": [url("solid_120x90.png")]}),
        ("placeholder-text-path", {"state": "see <|image_pad|>", "questions": {"n": noul}}),
        ("invalid-and-valid", {"questions": {"bad": {"type": "rank"}, "n": noul}, "images": [url("solid_120x90.png")]}),
        ("bad-bmp", {"questions": {"n": noul}, "images": [url("grad_64x48.bmp", "png")]}),
        ("bad-gif", {"questions": {"n": noul}, "images": [url("grad_64x48.gif", "png")]}),
        ("bad-tiff", {"questions": {"n": noul}, "images": [url("grad_64x48.tiff", "png")]}),
        ("bad-subtype", {"questions": {"n": noul}, "images": [url("grad_64x48.gif", "gif")]}),
        ("bad-svg", {"questions": {"n": noul}, "images": ["data:image/svg+xml;base64,PHN2Zy8+"]}),
        ("bad-not-data", {"questions": {"n": noul}, "images": ["https://example.com/a.png"]}),
        ("bad-number", {"questions": {"n": noul}, "images": [7]}),
        ("bad-no-base64-flag", {"questions": {"n": noul}, "images": ["data:image/png," + url("solid_120x90.png").split(",", 1)[1]]}),
        ("bad-no-comma", {"questions": {"n": noul}, "images": ["data:image/png;base64"]}),
        ("bad-text-head", {"questions": {"n": noul}, "images": ["data:text/plain;base64,QUJD"]}),
        ("bad-base64-chars", {"questions": {"n": noul}, "images": ["data:image/png;base64,QU JD"]}),
        ("bad-base64-pad", {"questions": {"n": noul}, "images": ["data:image/png;base64,QQ="]}),
        ("bad-base64-five", {"questions": {"n": noul}, "images": ["data:image/png;base64,QUJDQ"]}),
        ("bad-text-file", {"questions": {"n": noul}, "images": [url("text.png")]}),
        ("bad-empty-file", {"questions": {"n": noul}, "images": [url("empty.png")]}),
        ("bad-16mp", {"questions": {"n": noul}, "images": [url("solid_4100x4000.png")]}),
        ("ok-16mp-exact", {"questions": {"n": noul}, "images": [url("solid_4000x4000.png")]}),
        ("bad-aspect", {"questions": {"n": noul}, "images": [url("solid_4020x20.png")]}),
        ("bad-truncated", {"questions": {"n": noul}, "images": [url("truncated.png")]}),
        ("bad-second-of-two", {"questions": {"n": noul}, "images": [url("solid_120x90.png"), url("grad_64x48.bmp", "png")]}),
        ("bad-truncated-then-type", {"questions": {"n": noul}, "images": [url("truncated.png"), url("grad_64x48.gif", "gif")]}),
        ("bad-images-string", {"questions": {"n": noul}, "images": "x"}),
    ]
    n = 0
    with open(out, "w") as f:
        for rid, body in reqs:
            rec = {"id": rid, "body": json.dumps(body, ensure_ascii=False)}
            items = body.get("images")
            try:
                if items is not None and not isinstance(items, list):
                    raise ValueError("images takes a list of base64 data URLs")
                loaded = model.load_images(items, strict=True) if items else None
            except ValueError as exc:
                rec["detail"] = str(exc)
                msg = str(exc).split(": ", 1)[-1]
                if msg.startswith(("the image failed the integrity check", "the image could not be decoded")):
                    rec["stage"] = "decode"
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                n += 1
                continue
            try:
                prepared = model.prepare(body.get("state"), body["questions"], loaded or None, None)
            except ValueError as exc:
                rec["detail"] = str(exc)
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                n += 1
                continue
            invalid = {k: e["message"] for k, e in prepared.errors.items() if e.get("error") == "invalid_question"}
            if invalid:
                rec["detail"] = {"invalid_questions": invalid}
            texts, ids, lengths = {}, {}, {}
            for key in prepared.runnable:
                if loaded:
                    texts[key] = prepared.texts[key]
                    enc = model.processor(text=[texts[key]], images=list(loaded), return_tensors="pt")
                    ids[key] = enc["input_ids"][0].tolist()
                    rec["grids"] = [[int(t), int(h), int(w)] for t, h, w in enc["image_grid_thw"].tolist()]
                else:
                    ids[key] = prepared.sequences[key]
                lengths[key] = prepared.lengths[key] if loaded else len(prepared.sequences[key])
            rec.update({"texts": texts, "ids": ids, "lengths": lengths})
            if loaded:
                p2 = short.prepare(body.get("state"), body["questions"], loaded, None)
                rec["max_length_2000"] = {k: e["message"] for k, e in p2.errors.items() if e.get("error") == "max_length_exceeded"}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n += 1
    return n


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--package", required=True)
    ap.add_argument("--goldens", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--fuzz", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--images-only", action="store_true", help="regenerate images.jsonl only")
    a = ap.parse_args(argv)
    global PACKAGE, GOLDENS
    PACKAGE, GOLDENS = Path(a.package), Path(a.goldens)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if not a.images_only:
        print("goldens", goldens(out / "goldens.jsonl"))
        print("fuzz", fuzz(out / "fuzz.jsonl", a.fuzz, a.seed))
    print("images", images(out / "images.jsonl"))


if __name__ == "__main__":
    main()
