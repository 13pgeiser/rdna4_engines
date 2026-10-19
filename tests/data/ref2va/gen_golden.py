#!/usr/bin/env python3
"""Regenerate golden.json -- the reference numbers arch/minimaxh3/rad_ref2va_pack.cpp is tested
against (tests/ref2va_pack_test.cpp).

    gen_golden.py /path/to/ComfyUI/comfy/ldm/minimax/model.py [out.json]

The reference implementation lives in the ComfyUI tree; importing its package pulls the whole
device stack, so this extracts exactly the layout's source -- the four constants, the grid helpers
and the PackedLayout class -- with `ast`, execs them on torch alone, and runs the cases below.
Nothing here is a re-derivation: every number in golden.json is computed by the reference code
itself, at float64, from the file you point it at. If a radiance bump changes golden.json, the port
must change with it.

The `mod` entries are NOT produced by the reference class (its _forward folds them into per-row
mask tensors); they are the reference's segment-level rule -- the seg_tag table and the
{stream, pin} timestep labels transcribed from _forward -- applied to self.segments here, which is
the arithmetic the port replicates (sorted distinct labels, collapse on equality, t_row * 3 + tag).
"""

import ast
import json
import sys

NEED = ["_axis_from_sqrt_area", "_frame_grid", "_video_t_spans", "_video_t_grid", "_ref_t_span",
        "_audio_grid", "_video_grid", "PackedLayout"]
NEED_ASSIGN = ["FRAME_PER_TOKEN", "FRAME_RESCALE", "VISUAL_COND_TIMESTEP", "AUDIO_COND_TIMESTEP"]


def extract(path):
    """The reference's layout source, exec'd with torch/math only."""
    import math
    import torch
    with open(path) as f:
        src = f.read()
    tree = ast.parse(src)
    picked, got = [], set()
    for node in tree.body:
        names = ([t.id for t in node.targets if isinstance(t, ast.Name)]
                 if isinstance(node, ast.Assign) else
                 [node.name] if isinstance(node, (ast.FunctionDef, ast.ClassDef)) else [])
        if any(n in NEED or n in NEED_ASSIGN for n in names):
            picked.append(ast.get_source_segment(src, node))
            got.update(names)
    missing = (set(NEED) | set(NEED_ASSIGN)) - got
    if missing:
        sys.exit(f"{path} does not define: {sorted(missing)} -- reference moved?")
    ns = {"math": math, "torch": torch}
    exec("\n\n".join(picked), ns)
    return ns


# The reference _forward's per-kind (timestep label, modality tag), transcribed.
SEG_TAG = {"text": 1, "video": 0, "audio": 2, "cond": 0, "ref_img": 0, "cond_audio": 2,
           "ref_audio": 2}


def t_label(kind, t_v, t_a):
    if kind in ("cond", "ref_img"):
        return max(t_v, 0.999)
    if kind in ("cond_audio", "ref_audio"):
        return max(t_a, 1.0)
    if kind == "audio":
        return t_a
    return t_v


def mod_rows(layout, t_v, t_a):
    # non-empty segments only: the reference derives labels from mask TENSORS, so a zero-row
    # segment contributes no rows to the distinct-label set. An empty segment's own row index has
    # no defined value -- 0 by convention, matching the port.
    uniq = sorted({t_label(k, t_v, t_a) for (s, e, k) in layout.segments if e > s})
    return [uniq.index(t_label(k, t_v, t_a)) * 3 + SEG_TAG[k] if e > s else 0
            for (s, e, k) in layout.segments]


CASES = {
    # plain t2va: the pack must reproduce the old three-segment shape exactly
    "t2va": dict(text_len=5, target=(3, 8, 10, 4), keyframes=[], refs=[]),
    # fl2va keyframes at frame 0 (shares the target origin) and mid-timeline, video-only and both
    "fl2va": dict(text_len=4, target=(3, 8, 8, 3),
                  keyframes=[dict(frame=0, video_t=1, audio_t=2), dict(frame=7, video_t=2, audio_t=0)],
                  refs=[]),
    # every reference kind in one request: a photo, a standalone soundtrack, a silent clip, and a
    # clip with sound (its audio packs before its video, at the clip's own width extremes); a
    # second photo after them; audio_t=0 targets the silent-video edge
    "ref2va_all": dict(text_len=6, target=(2, 8, 8, 0),
                       keyframes=[],
                       refs=[dict(kind="image", latent_h=6, latent_w=10, latent_t=0, ref_audio_t=0),
                             dict(kind="audio", latent_h=0, latent_w=0, latent_t=0, ref_audio_t=3),
                             dict(kind="video", latent_h=4, latent_w=6, latent_t=2, ref_audio_t=0),
                             dict(kind="video_audio", latent_h=5, latent_w=7, latent_t=3,
                                  ref_audio_t=4),
                             dict(kind="image", latent_h=4, latent_w=4, latent_t=0, ref_audio_t=0)]),
    # a realistic signature: keyframes + a big photo + a clip with sound, real-ish sizes
    "chain": dict(text_len=32, target=(7, 16, 24, 6),
                  keyframes=[dict(frame=0, video_t=1, audio_t=2)],
                  refs=[dict(kind="image", latent_h=8, latent_w=12, latent_t=0, ref_audio_t=0),
                        dict(kind="video_audio", latent_h=4, latent_w=6, latent_t=5,
                             ref_audio_t=20)]),
    # degenerate prompt: text_len 0, nothing but targets
    "no_text": dict(text_len=0, target=(2, 4, 6, 1), keyframes=[], refs=[]),
}

# (t_v, t_a) probes: apart (four distinct labels), and past both pins (every label collapses to
# two, and cond/ref share the video row, cond_audio/ref_audio the audio row)
SIGMAS = [(0.5, 0.45), (0.9995, 0.999), (0.2, 0.15)]


def main():
    import torch
    model_py = sys.argv[1]
    out = sys.argv[2] if len(sys.argv) > 2 else "golden.json"
    ns = extract(model_py)
    PackedLayout = ns["PackedLayout"]
    golden = {}
    for name, c in CASES.items():
        t, ah, aw, at = c["target"]
        refs = [dict(kind=r["kind"], latent_h=r["latent_h"], latent_w=r["latent_w"],
                     latent_t=r["latent_t"], ref_audio_t=r["ref_audio_t"]) for r in c["refs"]]
        # the class reads only the latents' shapes (video .shape[2], audio .shape[-1]); zeros do
        kfs = [dict(resolved_frame_index=k["frame"],
                    latent=torch.zeros(1, 24, k["video_t"], ah, aw) if k["video_t"] else None,
                    audio_latent=torch.zeros(1, 32, 2, k["audio_t"]) if k["audio_t"] else None)
               for k in c["keyframes"]]
        lay = PackedLayout(c["text_len"], t, ah, aw, at, keyframes=kfs or None, refs=refs or None)
        golden[name] = {
            "text_len": c["text_len"], "latent_t": t, "latent_h": ah, "latent_w": aw,
            "audio_t": at,
            "keyframes": c["keyframes"], "refs": c["refs"],
            "seq_len": lay.seq_len,
            "segments": [[s, e, k] for (s, e, k) in lay.segments],
            "positions": [[float(x) for x in row] for row in lay.position_ids.tolist()],
            "img_update": [bool(b) for b in lay.img_update.tolist()],
            "audio_update": [bool(b) for b in lay.audio_update.tolist()],
            "img_pos": [int(i) for i in lay.img_pos.tolist()],
            "audio_pos": [int(i) for i in lay.audio_pos.tolist()],
            "mod": [dict(t_v=tv, t_a=ta, rows=mod_rows(lay, tv, ta)) for tv, ta in SIGMAS],
        }
    with open(out, "w") as f:
        json.dump(golden, f, indent=1)
    print(f"{out}: " + ", ".join(f"{n} {g['seq_len']} rows" for n, g in golden.items()))


main()
