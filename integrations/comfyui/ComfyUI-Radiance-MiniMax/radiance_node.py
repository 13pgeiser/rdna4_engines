"""Radiance MiniMax: drive a Radiance MiniMax-H3 server (/v1/videos) from ComfyUI.

The node is a pure HTTP client: the ~50-60 GiB model (container-dependent) runs in
the radiance server process (this box or any TP2 box on the LAN), so it costs
ComfyUI no VRAM.

Prompt: `prompt` is a widget AND an input socket -- wire your LLM prompt
enhancer's STRING output straight into it and the wire wins over what is typed.

Media is purely wire-driven: image_1..9, video_1..3, video_audio_1..3 (a
video's soundtrack, its own socket) and audio_1..3 (standalone reference
audio). Which ones are live depends on the task:

    t2va    -- none
    i2va    -- image_1 -> first frame
    l2va    -- image_1 -> last frame
    fl2va   -- image_1 -> first frame, image_2 -> last frame
    ref2va  -- all: image_1..9 + video_1..3 + video_audio_1..3 + audio_1..3

Connections a task does not use are ignored (with a console warning), and for
ref2va the slots must be filled in order -- ref image 3 without 2 would shift
the server's conditioning groups. Everything is documented again in the node's
? help pane (the description below), which ComfyUI renders as markdown.
"""

import io as _io
import json
import logging
import os
import re
import tempfile
import time
import wave

import requests

import folder_paths
from comfy_api.latest import io, ui, Types
from comfy_api.input_impl import VideoFromFile

POLL_SECONDS = 3.0
LOG = "RadianceMiniMax"

# The size law the minimaxh3 plugin declares (arch/minimaxh3/minimaxh3.cpp):
# sides are multiples of 32 px from 256, at most 4:1, at most 1344x768 of area
# -- and that law now holds for EVERY task. Keyframes are resized to the exact
# output canvas and VAE-encoded as one frame; the plugin's encoder buffers are
# sized max(full clip, one canvas-sized frame), so keyframe tasks carry no
# extra cap. NOTE: true 1280x720 is impossible with this law (720 is not a
# multiple of 32); the HD-shaped options are 1024x576 (exact 16:9), 1280x704
# or 1280x736 (through custom), and the model's native 1344x768 = 7:4.
SIZE_MULT, MIN_SIDE = 32, 256
MAX_AREA, MAX_ASPECT = 1344 * 768, 4.0

# DaSiWa-style picking: a p-tier sets the SHORT side (raised to the 256 floor),
# the aspect ratio computes the long side -- both snapped to 32-multiples and
# scaled back under the canvas if needed. Tiers stop at 720p because a single
# pass cannot exceed the canvas: DaSiWa's 900/1024/1080p presets are megapixel
# budgets run as base pass + latent-upscaler second stage, which Radiance has
# no equivalent of yet.
TIERS = {"144p": 144, "240p": 240, "360p": 360, "480p": 480, "540p": 540,
         "576p": 576, "720p": 720}
ASPECTS = ["5:4", "4:3", "3:2", "16:9", "2:1", "1:1", "1:2", "9:16", "2:3", "3:4", "4:5"]

# The plain-text twin of the schema description below: shown inside the node when
# the `help` widget is set to on (for frontends whose header ? glyph is missing).
HELP_TEXT = """Radiance MiniMax -- input guide (no VRAM here; the model
lives in the radiance server process).

TASK decides which sockets are live; all others are IGNORED:
  t2va    no media at all
  i2va    image_1 = the FIRST frame
  l2va    image_1 = the LAST frame (video grows back to it)
  fl2va   image_1 = first frame, image_2 = last frame
  ref2va  every media socket below

MEDIA SOCKETS (wire Load Image / Video / Audio or generated media):
  image_1..9         reference images: identity/style (max 9)
  video_1..3         reference clips: motion/camera (max 3)
  video_audio_1..3   the SOUNDTRACK of video_N -- same number =
                     same video; wire that video or this errors
  audio_1..3         STANDALONE reference audio: a voice/timbre to
                     speak with. NOT a video soundtrack!
  ref2va: fill slots in order -- image_3 without image_2 is an
  error, the socket number IS the conditioning position.

PROMPT: widget AND wire; a wired STRING (your LLM enhancer) wins over
the typed text. The model was trained on the MiniMax format
(subject_definitions / summary / retention_analysis /
detailed_description, <Picture N> <Subject N> <Setting N> <Style N>
tags, timed shots, <d>[English] ...</d> dialogue). Short one-line
prompts garble the audio.
There is NO negative prompt: H3 is guidance-distilled (CFG is
baked into the weights, one forward a step) and the server
REFUSES a negative_prompt ("this model takes no guidance"),
so a "negative" here could only ever fail the job. Negative
concepts belong in the prompt's own wording.
loras / lora 2: TWO server-side adapters, picked from
dropdowns ('none' = off) -- the pair mirrors the server's
--max-loras 2 slots; either may stand alone. The list is
the radiance server's own (GET /v1/loras; the FILES live
there under --lora-dir) -- 'refresh lora list' re-fetches
it, and it ONLY refreshes the list, never fills anything
in. Each slot has its own strength slider (sent as
name:strength). Beyond two: <lora:name:0.8> prompt tags
plus a raised --max-loras on the server. ComfyUI's Load
LoRA node can NOT drive this node -- it patches a local
torch model; radiance bakes adapters into the served
weights. Works with refs/keyframes too.
resolution + aspect (DaSiWa-style): the p-tier is the SHORT
side, the aspect computes the LONG side -- 480p + 16:9 ->
864x480, 480p + 9:16 -> 480x864, 480p + 1:1 -> 480x480. The
node title shows the computed WxH. Both sides snap to
multiples of 32 (256 floor) under the model's canvas (area
at most 1344*768), so 720p 16:9 lands on 1312x736: true
1280x720 is impossible (720 is not a multiple of 32). Tiers
stop at 720p because a single pass cannot exceed the canvas
-- DaSiWa's 900/1024/1080p run a second latent-upscaler
stage Radiance has no equivalent of. EVERY size works for
EVERY task, keyframes included. custom (aspect ignored) is
ADJUSTED, never refused -- it snaps to the nearest size the
law allows (480x720 -> 480x736, 4096x1024 -> 1920x480) and
says so on the console; only non-size text errors.
seconds / seed / num_inference_steps / sampler / shift / audio_shift:
server dials (sampler: euler or res_multistep). seed caps
at 9007199254740991 (2^53-1, 16 digits) -- the largest
seed radiance holds EXACTLY; KSampler/DaSiWa allow 20
digits but past 2^53 the engine rounds the seed and past
2^63 many different seeds collapse to one constant, so
bigger values are clamped here rather than silently
changing the run.
latent_upscale (advanced): the container's built-in latent
upscaler -- early steps at size/2, refine at full size.
Works with t2va and ref2va (references carry their own
grids); REFUSED with i2va/l2va/fl2va keyframes -- a pin
sits at the exact output frame on the exact target grid
and a draft canvas leaves it nowhere to sit. The output
still stays within the canvas, so it buys speed and
detail, not 1080p. off by default.
server_url: http://<box>:8133. LAN needs the server on
--host 0.0.0.0; NO auth -- trusted LAN only.
model: served id, GET /v1/models lists them.

OUTPUTS: clip saved to the output folder under radiance/ AND returned
as VIDEO (preview/chain) + file path + server job id. Progress bar
tracks the job; `timeout` bounds the wait. The node prints a
PROGRESS LINE to the ComfyUI console every poll (~3s):
"job_x: 45%, step 12/26, 5.2 s/step, eta ~1m 07s (+decode),
elapsed 55s" -- the s/step is MEASURED from the step counter
(smoothed, so it tracks a latent-upscale's draft->refine pace
change), and the eta is remaining-steps x that pace. Real step
counts come from a server built after Oct 2026; older servers
show ~step derived from the job progress (the steps' share of
0..90 by design) and extrapolate the eta from it. On the SERVER
itself, a new-enough build logs one RAD_INFO line per step, so
`docker logs -f` shows the same pace live."""


def _image_pngs(images):
    """Core IMAGE tensor [B,H,W,C] float 0..1 -> [(name, png bytes)] for each row."""
    import torch
    from PIL import Image
    arr = images.cpu().clamp(0.0, 1.0).mul(255.0).to(dtype=torch.uint8).numpy()
    out = []
    for i in range(arr.shape[0]):
        buf = _io.BytesIO()
        Image.fromarray(arr[i], "RGB").save(buf, format="PNG")
        out.append((f"connected_{i + 1}.png", buf.getvalue()))
    return out


def _image_png(image):
    """First row of a wired IMAGE as (name, png bytes)."""
    return _image_pngs(image)[0]


def _audio_wav(audio):
    """Core AUDIO {'waveform':[B,C,N] float, 'sample_rate': int} -> (name, wav bytes)."""
    import torch
    wf = audio["waveform"][0]
    sr = int(audio["sample_rate"])
    pcm = (wf.transpose(0, 1).clamp(-1.0, 1.0) * 32767.0).cpu().to(
        dtype=torch.int16).contiguous().numpy()
    buf = _io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(pcm.shape[1])
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm.reshape(-1).tobytes())
    return "connected_audio.wav", buf.getvalue()


def _video_mp4(video):
    """Core VIDEO -> (name, mp4 bytes) via its own save_to (stream copy when the
    source is already a compatible mp4, transcode otherwise)."""
    fd, path = tempfile.mkstemp(suffix=".mp4")
    os.close(fd)
    try:
        video.save_to(path, format=Types.VideoContainer.MP4)
        with open(path, "rb") as f:
            return "connected_video.mp4", f.read()
    finally:
        os.unlink(path)


def _contiguous(items, prefix):
    """Number of filled leading slots; a gap is a mistake -- the server reads the
    ref_* parts as one ordered list."""
    n = 0
    while n < len(items) and items[n] is not None:
        n += 1
    for i in range(n, len(items)):
        if items[i] is not None:
            raise ValueError(f"{prefix}_{i + 1} is connected but {prefix}_{n if n else 1} is "
                             "not -- fill the slots in order, the socket number IS the "
                             "conditioning position")
    return n


def _fit(w, h, cap):
    """The largest canvas at the w:h ratio, sides snapped DOWN to 32-multiples,
    whose area fits under `cap` pixels."""
    s = (cap / float(w * h)) ** 0.5
    fw = max(MIN_SIDE, int(w * s) // SIZE_MULT * SIZE_MULT)
    fh = max(MIN_SIDE, int(h * s) // SIZE_MULT * SIZE_MULT)
    while fw * fh > cap:  # the 256 floor can push a very narrow ratio over
        if fw >= fh and fw > MIN_SIDE:
            fw -= SIZE_MULT
        elif fh > MIN_SIDE:
            fh -= SIZE_MULT
        else:
            break
    if max(fw, fh) > MAX_ASPECT * min(fw, fh):  # rounding can skew a boundary ratio
        if fw >= fh:
            fw = int(MAX_ASPECT * fh) // SIZE_MULT * SIZE_MULT
        else:
            fh = int(MAX_ASPECT * fw) // SIZE_MULT * SIZE_MULT
    return fw, fh


def _snap_text(text):
    """`custom` size text: like ComfyUI's own resize, an illegal size is
    ADJUSTED, not refused -- each side is rounded to a multiple of 32 px (ties
    up, 256 px floor), and an aspect over 4:1 or an area over the model's
    1344x768 canvas is scaled back at the same ratio. The adjustment is said
    on the console -- the server still has the last word, it enforces the
    same law."""
    import re
    m = re.match(r"(\d+)\s*[xX×*]\s*(\d+)", text.strip())
    if not m:
        raise ValueError(f'"{text}" is not a size -- give WIDTHxHEIGHT, e.g. 512x512')
    w, h = int(m.group(1)), int(m.group(2))
    # nearest 32-multiple, ties up, never under the minimum side
    lw = max(MIN_SIDE, (w + SIZE_MULT // 2) // SIZE_MULT * SIZE_MULT)
    lh = max(MIN_SIDE, (h + SIZE_MULT // 2) // SIZE_MULT * SIZE_MULT)
    if lw * lh > MAX_AREA or max(lw, lh) > MAX_ASPECT * min(lw, lh):
        lw, lh = _fit(lw, lh, MAX_AREA)
    if (lw, lh) != (w, h):
        logging.warning(f"[{LOG}] size {w}x{h} is not valid for this model -- using "
                        f"{lw}x{lh} instead (sides multiples of {SIZE_MULT} px and at least "
                        f"{MIN_SIDE}, ratio <= {MAX_ASPECT:.0f}:1, area <= {MAX_AREA} px)")
    return f"{lw}x{lh}"


def _tier_ratio(aspect):
    a, b = aspect.split(":")
    return int(a), int(b)


def _tier_size(tier, aspect):
    """The DaSiWa-style pick: the p-tier is the SHORT side (raised to the 256
    floor), the ratio computes the LONG side; both snapped to 32-multiples,
    ties up, and the canvas scaled back proportionally if it overshoots
    1344x768. Returns "WxH" with the orientation the ratio implies."""
    a, b = _tier_ratio(aspect)
    hi, lo = max(a, b), min(a, b)
    short = max(MIN_SIDE, (TIERS[tier] + SIZE_MULT // 2) // SIZE_MULT * SIZE_MULT)
    long_ = max(short, (int(short * hi / lo) + SIZE_MULT // 2) // SIZE_MULT * SIZE_MULT)
    if long_ * short > MAX_AREA:
        long_, short = _fit(long_, short, MAX_AREA)
    return f"{long_}x{short}" if a >= b else f"{short}x{long_}"


def _resolve_size(resolution, aspect, custom_size=""):
    """`resolution` is the p-tier combo ("480p", ... or "custom" -> the typed
    text); the frontend delivers a v3 DynamicCombo as a dict with the selected
    key and any option sub-inputs folded in. The p-tier path computes the
    other dimension from `aspect`; the server still has the last word -- it
    enforces the same law."""
    if isinstance(resolution, dict):  # what the frontend really delivers
        custom_size = resolution.get("custom_size", custom_size)
        resolution = resolution.get("resolution", "")
    if resolution == "custom":
        return _snap_text(custom_size)
    if resolution not in TIERS:
        raise ValueError(f"unknown resolution {resolution!r} -- pick one of "
                         f"{', '.join(TIERS)} or custom")
    return _tier_size(resolution, aspect)


def _parse_loras(text):
    """'film:0.8, ink' -> ['film:0.8', 'ink'] -- the server's entry grammar
    (core/server/loras.cpp: the LAST ':' splits a strength, anything else is
    the whole name), minus empty entries. Entries go to the server verbatim;
    only a colon tail that looks like a botched decimal ('0.8x', '1.2.3') is
    caught here, because the server would silently read such a name as a
    plain adapter name and answer with a confusing 'unknown lora'."""
    out = []
    for entry in (text or "").split(","):
        entry = entry.strip()
        if not entry:
            continue
        _head, sep, tail = entry.rpartition(":")
        # a tail that STARTS like a decimal ('0.8x', '1.2.3', '0.5e') was meant as the
        # strength; the server's strtod would instead read the WHOLE entry as a name
        if sep and tail and re.match(r"^[+-]?(\d+(\.\d*)?|\.\d+)", tail):
            try:
                float(tail)
            except ValueError:
                raise ValueError(f"lora entry {entry!r}: {tail!r} after the ':' is not a "
                                 "number -- give a plain decimal strength ('name:0.8') or no "
                                 "colon for strength 1")
        out.append(entry)
    return out


class RadianceMiniMax(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        images = [io.Image.Input(f"image_{i}", optional=True) for i in range(1, 10)]
        videos = [io.Video.Input(f"video_{i}", optional=True) for i in range(1, 4)]
        vauds = [io.Audio.Input(f"video_audio_{i}", optional=True) for i in range(1, 4)]
        adus = [io.Audio.Input(f"audio_{i}", optional=True) for i in range(1, 4)]
        images[0].tooltip = ("ref2va: reference image 1. i2va/l2va: THE frame (first / last). "
                             "fl2va: the FIRST frame")
        images[1].tooltip = "ref2va: reference image 2. fl2va: the LAST frame. Else ignored"
        for im in images[2:]:
            im.tooltip = "ref2va only: reference image -- ignored by every other task"
        for i, v in enumerate(videos):
            v.tooltip = (f"ref2va only: reference video {i + 1}; its soundtrack (if any) goes "
                         f"in video_audio_{i + 1}. Ignored by every other task")
        for i, a in enumerate(vauds):
            a.tooltip = (f"ref2va only: soundtrack of video_{i + 1} -- wire video_{i + 1} too. "
                         "Ignored by every other task")
        for i, a in enumerate(adus):
            a.tooltip = (f"ref2va only: standalone reference audio {i + 1} (voice/timbre) -- "
                         "NOT a video soundtrack (that is video_audio_N). Ignored otherwise")
        return io.Schema(
            node_id="RadianceMiniMax",
            display_name="Radiance MiniMax",
            category="video/radiance",
            description=(
                "Generates video + audio on a **Radiance MiniMax-H3 server** over HTTP "
                "(`POST /v1/videos`). The node is a pure HTTP client — **no VRAM**; the model "
                "lives in the server process (this box or any TP2 box on the LAN).\n\n"
                "## Which sockets are live depends on `task` — everything else is ignored\n\n"
                "| task | media used |\n"
                "|---|---|\n"
                "| `t2va` | none — text only |\n"
                "| `i2va` | `image_1` → the **first frame** of the clip |\n"
                "| `l2va` | `image_1` → the **last frame** (video grows *backwards* to it) |\n"
                "| `fl2va` | `image_1` → **first** frame, `image_2` → **last** frame "
                "(completion between two pinned endpoints) |\n"
                "| `ref2va` | every socket below |\n\n"
                "## Media sockets (all optional; wire Load Image / Video / Audio or generated "
                "media)\n\n"
                "- **image_1..image_9** — `ref2va` reference images (identity/style). "
                "For the keyframe tasks only image_1/image_2 matter, as the table says.\n"
                "- **video_1..video_3** — `ref2va` reference clips: motion/camera reference. "
                "Uploads as mp4 (ComfyUI re-encodes only when the source can't stream-copy).\n"
                "- **video_audio_1..video_audio_3** — `ref2va` **soundtrack of `video_N`** "
                "(same number = same video). Wire `video_N` too or this is an error.\n"
                "- **audio_1..audio_3** — `ref2va` **standalone reference audio**: a voice/"
                "timbre to speak with. NOT a video soundtrack — that is `video_audio_N`.\n\n"
                "For `ref2va` fill slots **in order** (no `image_3` without `image_2`): the "
                "socket number is the conditioning position the server groups references by.\n\n"
                "## Prompt & knobs\n\n"
                "- **prompt** — widget *and* STRING socket: wire your prompt enhancer in and "
                "the wire wins. The model was trained on the MiniMax format "
                "(`subject_definitions` / `summary` / `retention_analysis` /\n"
                "  `detailed_description`, `<Picture N>`/`<Subject N>`/`<Setting N>`/`<Style N>` "
                "tags, timed shots, `<d>[English] …</d>` dialogue); short one-liners garble "
                "the audio.\n"
                "- **no negative_prompt** — MiniMax-H3 is guidance-distilled (CFG is baked "
                "into the weights, one forward per step); the server *refuses* a "
                "`negative_prompt` (\"this model takes no guidance\") rather than ignoring "
                "it, so negative concepts belong in the prompt's own wording.\n"
                "- **loras / lora 2** — TWO server-side adapters in dropdown slots "
                "(`none` = off; either may stand alone) — the pair mirrors the server's "
                "`--max-loras 2`. The dropdowns list what the box in `server_url` serves "
                "(`GET /v1/loras`) — fetched when a workflow loads, when you edit the "
                "URL, and by the **refresh lora list** button, which refreshes the list "
                "only and never fills anything in. Each slot gets its own strength "
                "slider (sent as `NAME:STRENGTH`). Beyond two adapters: `<lora:name:0.8>` "
                "prompt tags plus a raised `--max-loras` on the server. The files live "
                "on the **radiance server** (started with `--lora-dir`). **ComfyUI's "
                "Load LoRA node cannot drive this node** — it patches a local torch "
                "model, while radiance applies adapters inside the served weights. "
                "Adapters work with every task (refs and keyframes included).\n"
                "- **resolution / aspect** — DaSiWa-style picking: the p-tier (144p…720p) is "
                "the **short side**, the aspect ratio computes the long side — 480p + 16:9 → "
                "864×480, 480p + 9:16 → 480×864, 1:1 → square. The node **title shows the "
                "computed WxH**. Both sides snap to multiples of 32 px (256 floor) under the "
                "model's canvas (area ≤ 1344×768 = 1,032,192 px), so 720p 16:9 lands on "
                "1312×736 — true 1280×720 is impossible (720 is not a multiple of 32). Tiers "
                "stop at 720p: a single pass cannot exceed the canvas (DaSiWa's 900/1024/1080p "
                "presets run a **latent-upscaler second stage** Radiance has no equivalent of "
                "yet). **Every size works for every task, keyframes included**: a keyframe is "
                "encoded whole at the exact output canvas and the encoder buffers cover the "
                "full canvas (needs a server built with the grown encoder buffers, Oct 2026). "
                "`custom` (aspect ignored) is **auto-adjusted, not refused** — like ComfyUI's "
                "own resize, any WxH snaps to the nearest size satisfying the law (480×720 → "
                "480×736) and the change is logged to the console.\n"
                "- **seconds / seed / num_inference_steps / sampler / shift / audio_shift** — "
                "server dials; `sampler` is `euler` or `res_multistep`; `seed` caps at "
                "**9007199254740991** (2^53−1, 16 digits) — the largest value radiance holds "
                "exactly. KSampler/DaSiWa allow 20 digits, but beyond 2^53 the engine rounds "
                "the seed and beyond 2^63 it collapses to one constant, so bigger values "
                "would silently change the run; paste-clamping is the honest failure.\n"
                "- **latent_upscale** — advanced, **off** by default: a container converted with "
                "the latent-upscaler weights (DaSiWa's companion — the server log names it at "
                "boot) runs the early steps at size/2 or /4, enlarges the clean latent, and "
                "refines at full size in the last steps. Works with **t2va and ref2va** — "
                "references carry their own grids, so they condition both stages identically — "
                "and is **refused with i2va/l2va/fl2va keyframes**: a pin sits AT the exact "
                "output frame on the exact target grid, and a draft canvas leaves it nowhere to "
                "sit. The output stays within the canvas, so it buys speed/detail, not 1080p.\n"
                "- **server_url / model** — which box (`http://<host>:8133`; LAN needs the "
                "server on `--host 0.0.0.0`, no auth — trusted LAN only) and which served "
                "model id (`GET /v1/models`).\n\n"
                "## Outputs\n\n"
                "The clip is **saved to the output folder** under `radiance/` AND returned as "
                "a `VIDEO` (preview/chain), plus the saved `file` path and server `job id`. "
                "The node polls the job, drives its progress bar, and prints a live line to "
                "the ComfyUI console every ~3 s (`45%, step 12/26, 5.2 s/step, eta ~1m 07s "
                "(+decode)` — the pace is measured from the step counter and the eta is "
                "remaining steps × that pace; real steps on a server built after Oct 2026, "
                "`~step` derived from progress otherwise); `timeout` (advanced) bounds the "
                "wait."),
            inputs=[
                io.String.Input("server_url", default="http://127.0.0.1:8133",
                                tooltip="the radiance server: http://<host>:8133 -- point it "
                                        "at any of your TP2 boxes"),
                io.String.Input("model", default="minimax-h3",
                                tooltip="model id as served; GET /v1/models lists them"),
                io.Combo.Input("loras", options=["none"], default="none", display_name="lora 1",
                                tooltip="ONE server-side LoRA adapter; 'none' = off. The "
                                        "dropdown lists what GET /v1/loras on the box "
                                        "serves -- the 'refresh lora list' button (and "
                                        "loading a workflow) re-fetches it; the button only "
                                        "refreshes the list, it never fills anything in. "
                                        "The FILES live on the radiance server (started "
                                        "with --lora-dir, e.g. a mount of ComfyUI's "
                                        "models/loras). ComfyUI's Load LoRA node CANNOT "
                                        "drive this node: it patches a local torch model, "
                                        "radiance bakes adapters into the served weights. "
                                        "Strength: lora strength below; a second adapter "
                                        "goes in lora 2. Beyond two: <lora:name:0.8> "
                                        "prompt tags. Works with every task (refs and "
                                        "keyframes included)"),
                io.Float.Input("lora_strength", default=1.0, min=0.0, max=2.0, step=0.05,
                               display_name="lora strength",
                               tooltip="strength of the loras pick -- sent as NAME:STRENGTH; "
                                       "inert while loras is 'none'"),
                io.Combo.Input("lora2", options=["none"], default="none", display_name="lora 2",
                               tooltip="second adapter -- the radiance server holds "
                                       "--max-loras 2 by default, and this pair mirrors "
                                       "that exactly. Same dropdown rules as loras "
                                       "('none' = off); either slot may stand alone. "
                                       "Beyond two: <lora:name:0.8> prompt tags and a "
                                       "raised --max-loras on the server"),
                io.Float.Input("lora2_strength", default=1.0, min=0.0, max=2.0, step=0.05,
                               display_name="lora 2 strength",
                               tooltip="strength of the lora2 pick -- sent as NAME:STRENGTH; "
                                       "inert while lora2 is 'none'"),
                io.Combo.Input("task", options=["t2va", "i2va", "l2va", "fl2va", "ref2va"],
                               tooltip="which sockets are live changes with this -- see the "
                                       "node description"),
                io.String.Input("prompt", multiline=True, default="",
                                tooltip="wire your LLM prompt enhancer's STRING output here "
                                        "and it replaces the text typed below. The MiniMax "
                                        "format (subject_definitions / summary / "
                                        "retention_analysis / detailed_description with "
                                        "<d>[lang] ...</d> dialogue) is what the model was "
                                        "trained on -- short prompts garble the audio"),
                io.DynamicCombo.Input("resolution", tooltip=  # no default kwarg here --
                    # DynamicCombo.Input has none; the frontend shows the first option and
                    # web/radiance_size.js nudges fresh nodes to 480p (a loaded workflow
                    # overwrites that with its own stored value in onConfigure)
                    "DaSiWa-style pick: the tier is the SHORT side, the aspect combo "
                    "computes the long side (the node title shows the resulting WxH). "
                    "Both snap to 32-multiples (256 floor) under the model's canvas "
                    "(area <= 1344x768) -- so 720p 16:9 becomes 1312x736, true 1280x720 "
                    "is impossible (720 is not a multiple of 32). Tiers stop at 720p: a "
                    "single pass cannot exceed the canvas (DaSiWa's 900/1024/1080p run "
                    "a latent-upscaler second stage Radiance has no equivalent of). "
                    "Every size works for every task, keyframes included; custom is "
                    "auto-adjusted, not refused (480x720 -> 480x736), aspect ignored",
                    options=[
                        io.DynamicCombo.Option("144p", []),
                        io.DynamicCombo.Option("240p", []),
                        io.DynamicCombo.Option("360p", []),
                        io.DynamicCombo.Option("480p", []),
                        io.DynamicCombo.Option("540p", []),
                        io.DynamicCombo.Option("576p", []),
                        io.DynamicCombo.Option("720p", []),
                        io.DynamicCombo.Option("custom", [
                            io.String.Input("custom_size", display_name="WIDTHxHEIGHT",
                                            default="512x512"),
                        ]),
                    ]),
                io.Combo.Input("aspect", options=ASPECTS, default="16:9",
                               tooltip="ratio of the tier pick: first number is the "
                                       "WIDTH (16:9 landscape, 9:16 portrait, 1:1 "
                                       "square); ignored for a custom size"),
                io.Int.Input("seconds", default=12, min=1, max=120, step=1),
                io.Int.Input("seed", default=1234, min=0, max=2**53 - 1, step=1,
                             tooltip="up to 9007199254740991 (2^53-1, 16 digits) -- every "
                                     "seed radiance can hold EXACTLY (it carries seeds as "
                                     "doubles). KSampler/DaSiWa allow 20 digits, but larger "
                                     "values only lose precision here: past 2^53 the engine "
                                     "rounds, and past 2^63 thousands of different seeds "
                                     "collapse to the SAME one -- so a DaSiWa random seed "
                                     "above this range clamps to the max when pasted in"),
                io.Int.Input("num_inference_steps", default=25, min=1, max=200, step=1),
                io.Combo.Input("sampler", options=["euler", "res_multistep"]),
                io.Float.Input("shift", default=12.0, min=0.0, max=100.0, step=0.5),
                io.Float.Input("audio_shift", default=3.0, min=0.0, max=100.0, step=0.5),
                io.Combo.Input("latent_upscale", options=["off", "2", "4"], default="off",
                               advanced=True, tooltip=
                    "the container's built-in latent upscaler (the DaSiWa companion weights, "
                    "baked into the .rad at convert time -- the server log names it at boot): "
                    "the early steps run at size/2 (or /4), the clean latent is enlarged, "
                    "re-noised, and the last steps refine at the FULL size. Works with t2va and "
                    "ref2va (references carry their own grids); refused with i2va/l2va/fl2va "
                    "keyframes (a pin needs the exact target grid). The OUTPUT stays within "
                    "the canvas (the refine steps run full size): this buys speed and detail, "
                    "not 1080p -- sizes stay multiples of 32 and under 1344x768 either way"),
                *images, *videos, *vauds, *adus,
                io.Int.Input("timeout", default=1800, min=30, max=7200, step=30, advanced=True,
                             tooltip="seconds to wait for the job before giving up"),
                io.DynamicCombo.Input("show_help", display_name="help",
                                      options=[
                                          io.DynamicCombo.Option("off", []),
                                          io.DynamicCombo.Option("on", [
                                              io.String.Input("docs", multiline=True,
                                                              display_name="input guide",
                                                              default=HELP_TEXT),
                                          ]),
                                      ],
                                      tooltip="the full input guide inside the node (the header "
                                              "? pane holds the same text)"),
            ],
            hidden=[io.Hidden.unique_id],
            outputs=[
                io.Video.Output("video"),
                io.String.Output("file", display_name="file",
                                 tooltip="path of the saved clip, relative to the output folder"),
                io.String.Output("job_id", display_name="job id"),
            ],
        )

    @classmethod
    def validate_inputs(cls, loras, lora2):
        """The loras/lora2 dropdowns are populated CLIENT-side from the radiance
        server (GET /v1/loras -- names differ per box), so a chosen adapter is
        not in the schema's static options. ComfyUI's membership check
        (execution.py value_not_in_list) covers only inputs a validator does
        NOT name -- so naming them here is the sanctioned skip. We still refuse
        a botched entry locally instead of after the job round-trips."""
        for pick in (loras, lora2):
            _parse_loras("" if (pick or "none") in ("", "none") else pick)
        return True

    @classmethod
    def execute(cls, server_url, model, task, prompt, resolution, aspect, seconds, seed,
                num_inference_steps, sampler, shift, audio_shift, custom_size="",
                timeout=1800, show_help="off", docs="",
                latent_upscale="off", loras="none", lora_strength=1.0,
                lora2="none", lora2_strength=1.0, **media):
        size = _resolve_size(resolution, aspect, custom_size)
        use_prompt = (prompt or "").strip()
        if not use_prompt:
            raise ValueError("give a prompt (type it or wire one in)")

        images = [media.get(f"image_{i}") for i in range(1, 10)]
        videos = [media.get(f"video_{i}") for i in range(1, 4)]
        vauds = [media.get(f"video_audio_{i}") for i in range(1, 4)]
        adus = [media.get(f"audio_{i}") for i in range(1, 4)]

        parts = []  # (field, (filename, bytes, mime)) -- every part is wired content

        def warn(what):
            logging.warning(f"[{LOG}] {task}: {what} -- this task does not take it, ignored")

        if task == "ref2va":
            n_img = _contiguous(images, "image")
            n_vid = _contiguous(videos, "video")
            n_aud = _contiguous(adus, "audio")
            for i, a in enumerate(vauds):
                if a is not None and videos[i] is None:
                    raise ValueError(f"video_audio_{i + 1} is the soundtrack of video_{i + 1} "
                                     f"-- wire video_{i + 1} too")
            for i in range(n_img):
                name, png = _image_png(images[i])
                parts.append((f"ref_image_{i + 1}", (name, png, "image/png")))
            for i in range(n_vid):
                name, body = _video_mp4(videos[i])
                parts.append((f"ref_video_{i + 1}", (name, body, "video/mp4")))
            for i, a in enumerate(vauds):
                if a is not None:
                    name, body = _audio_wav(a)
                    parts.append((f"ref_video_audio_{i + 1}", (name, body, "audio/wav")))
            for i in range(n_aud):
                name, body = _audio_wav(adus[i])
                parts.append((f"ref_audio_{i + 1}", (name, body, "audio/wav")))
            if not parts:
                raise ValueError("ref2va needs at least one wired image, video or audio")
        else:
            if any(videos) or any(vauds) or any(adus):
                warn("video/audio connections")
            n_img = sum(x is not None for x in images)
            if task == "i2va":
                if n_img > 1:
                    warn("image_2 and up")
                if images[0] is not None:
                    name, png = _image_png(images[0])
                    parts.append(("first_frame", (name, png, "image/png")))
            elif task == "l2va":
                if n_img > 1:
                    warn("image_2 and up -- l2va pins image_1 as the LAST frame")
                if images[0] is not None:
                    name, png = _image_png(images[0])
                    parts.append(("last_frame", (name, png, "image/png")))
            elif task == "fl2va":
                if n_img > 2:
                    warn("image_3 and up -- fl2va takes only the two endpoint frames")
                for slot, field in ((0, "first_frame"), (1, "last_frame")):
                    if images[slot] is not None:
                        name, png = _image_png(images[slot])
                        parts.append((field, (name, png, "image/png")))
            elif n_img:
                warn("image connections -- t2va is text only")

        base = server_url.rstrip("/")
        s = requests.Session()
        data = {
            "model": model, "task": task, "prompt": use_prompt, "size": size,
            "seconds": str(seconds), "seed": str(seed),
            "num_inference_steps": str(num_inference_steps), "sampler": sampler,
            "shift": str(shift), "audio_shift": str(audio_shift),
        }
        if str(latent_upscale) not in ("off", "", "1"):
            # the draft/refine split re-lays the reference pack per stage -- refs carry their own
            # grids, so ref2va rides an upscale fine. A KEYFRAME pin sits AT the output frame's row
            # on the exact target grid, and two canvases leave it nowhere to sit -- the driver
            # refuses those by name (video_driver.cpp), so say it here instead of eating a round trip
            if task in ("i2va", "l2va", "fl2va"):
                raise ValueError(f"latent_upscale {latent_upscale} with task {task}: a keyframe "
                                 "pins an exact output frame on the exact target grid, and latent "
                                 "upscale drafts at a smaller grid first -- run the keyframe "
                                 "request without latent_upscale, or send the frame as a "
                                 "reference image (ref2va supports the upscale)")
            data["latent_upscale"] = str(latent_upscale)
        # The two dropdown slots become the server's adapter list IN ORDER;
        # 'none' is off and either slot may stand alone. A slot holding
        # NAME:STRENGTH or legacy multi-entry text (saved by pre-dropdown
        # graphs) passes through verbatim; the strength slider composes a
        # single plain name -- refusing beats silently overriding the text.
        def _slot(pick, strength, label):
            pick = (pick or "none").strip()
            entries = [] if pick in ("", "none") else _parse_loras(pick)
            if entries and float(strength) != 1.0:
                if len(entries) > 1 or ":" in entries[0]:
                    raise ValueError(f"{label}_strength applies to one plain adapter name -- "
                                     "set it back to 1 and give per-adapter strengths as "
                                     "NAME:STRENGTH text or <lora:name:0.8> prompt tags")
                entries = [f"{entries[0]}:{float(strength):g}"]
            return entries

        def _entry_name(entry):
            head, sep, tail = entry.rpartition(":")   # the server's own split rule
            return head if sep and re.match(r"^[+-]?(\d+(\.\d*)?|\.\d+)$", tail) else entry

        lora_list = _slot(loras, lora_strength, "lora") + _slot(lora2, lora2_strength, "lora2")
        names = [_entry_name(e) for e in lora_list]
        if len(names) != len(set(names)):
            raise ValueError("the same adapter in both slots -- one is enough; the server "
                             "would apply it twice")
        if len(lora_list) > 2:
            raise ValueError(f"{len(lora_list)} adapters selected -- the radiance server "
                             "loads --max-loras 2 by default (raise the flag on the server "
                             "for more; free-text from an old graph can also add up)")
        if lora_list:
            data["loras"] = lora_list  # server grammar: array of "name[:strength]"

        # The server takes multipart (a form with files) or JSON. requests sends a
        # files=[] post as urlencoded, which is neither -- with no uploads go JSON.
        try:
            if parts:
                # a list field must arrive as JSON TEXT (the server json-parses a
                # loras field that starts with '[') -- not requests' str(list) repr
                form = {k: json.dumps(v) if isinstance(v, list) else v
                        for k, v in data.items()}
                r = s.post(f"{base}/v1/videos", data=form, files=parts, timeout=300)
            else:
                r = s.post(f"{base}/v1/videos", json=data, timeout=300)
        except requests.ConnectionError as e:
            raise RuntimeError(f"cannot reach the radiance server at {base}: {e} "
                               "(is it running with --host 0.0.0.0 for LAN use?)") from e
        if r.status_code >= 400:
            try:
                msg = r.json().get("error", {}).get("message") or r.text[:400]
            except ValueError:
                msg = r.text[:400]
            raise RuntimeError(f"radiance refused the job ({r.status_code}): {msg}")
        job_id = r.json()["id"]

        bar = _progress_bar(cls)
        deadline = time.time() + timeout
        t0 = time.time()
        pace = _PaceTracker()
        logging.info(f"[{LOG}] job {job_id} submitted -- progress lines every "
                     f"{POLL_SECONDS:.0f}s below; `docker logs -f` on the server shows them "
                     "per STEP too (a server built after Oct 2026)")
        while True:
            jr = s.get(f"{base}/v1/videos/{job_id}", timeout=30)
            jr.raise_for_status()
            j = jr.json()
            status = j.get("status")
            if status == "completed":
                break
            if status in ("failed", "cancelled"):
                err = (j.get("error") or {}).get("message", "") if isinstance(j.get("error"), dict) else str(j.get("error"))
                raise RuntimeError(f"job {job_id} {status}: {err or 'no reason given'}")
            p = j.get("progress")
            if isinstance(p, (int, float)) and bar is not None:
                bar.update_absolute(min(max(float(p), 0.0), 100.0))
            # the heartbeat: ComfyUI's console gets a live line per poll, so a
            # waiting user sees pace and eta instead of a silent spinner
            now = time.time()
            logging.info(f"[{LOG}] {job_id}: {_progress_line(j, num_inference_steps, now - t0, pace.update(j, now))}")
            if now > deadline:
                raise RuntimeError(f"job {job_id} still {status} after {timeout}s")
            time.sleep(POLL_SECONDS)

        cr = s.get(f"{base}/v1/videos/{job_id}/content", timeout=600)
        cr.raise_for_status()

        out_dir = folder_paths.get_output_directory()
        full_folder, filename, counter, subfolder, _prefix = folder_paths.get_save_image_path(
            "radiance/Radiance", out_dir, 0, 0)
        file = f"{filename}_{counter:05}_.mp4"
        path = os.path.join(full_folder, file)
        with open(path, "wb") as f:
            f.write(cr.content)

        if bar is not None:
            bar.update_absolute(100.0)
        video = VideoFromFile(path)
        return io.NodeOutput(
            video,
            os.path.join(subfolder, file) if subfolder else file,
            job_id,
            ui=ui.PreviewVideo([ui.SavedResult(file, subfolder, io.FolderType.output)]),
        )


def _fmt_secs(s):
    # Round to WHOLE seconds first, then split with integer ops: `.0f`/`.0f`
    # each round INDEPENDENTLY, so 175 s once printed "3m 55s" (2.92m rounded
    # to 3, its seconds left over) and 119.5 s printed "1m 60s".
    s = int(max(0.0, float(s)) + 0.5)
    return f"{s}s" if s < 90 else f"{s // 60}m {s % 60:02d}s"


class _PaceTracker:
    """Measured sec/step for the console heartbeat -- the client-side mirror of
    the server's per-step RAD_INFO log. Polls are ~3 s apart, so ONE step
    cannot be timed directly: a 12.7 s step sampled every 3 s appears as a
    single increment between two adjacent polls, and per-increment deltas read
    as the POLL interval -- a constant "3.0 s/step" no matter the real pace
    (the bug this replaces). The rate is therefore measured over a WINDOW: the
    anchor observation is held until at least two more steps are visible, then
    rate = elapsed / steps-in-between, and the anchor moves on. A light EMA
    (~40% new) still follows pace changes -- a latent-upscaled job's
    draft->refine grid switch re-prices the steps within a couple of windows.
    Needs a post-Oct-2026 server (the `steps_done` counter); reports None
    otherwise."""

    def __init__(self):
        self.anchor_t = None
        self.anchor_done = None
        self.sps = None

    def update(self, job, now):
        done = job.get("steps_done")
        if not isinstance(done, int):
            return self.sps
        if self.anchor_t is None:
            if done > 0:  # anchor on LIVE steps only -- the encode warm-up
                self.anchor_t, self.anchor_done = now, done  # must not ride in
        elif done - self.anchor_done >= 2:                   # the first rate
            inst = (now - self.anchor_t) / (done - self.anchor_done)
            if inst > 0:
                self.sps = inst if self.sps is None else 0.4 * inst + 0.6 * self.sps
            self.anchor_t, self.anchor_done = now, done
        return self.sps


def _progress_line(job, steps_requested, elapsed, sps=None):
    """One console line per poll -- the waiting client's heartbeat. A server
    built after Oct 2026 reports real `steps_done`/`steps_total` in the job
    JSON; every server's `progress` is the steps' share of 0..90 by design
    (videos.cpp), so an older one shows the derived estimate marked `~`. The
    eta rides the MEASURED s/step while steps remain (it tracks the live pace
    through phase changes and ignores the encode warm-up that skews a
    whole-job extrapolation), falling back to extrapolating `progress` on
    older servers -- that also covers the decode tail, which the step-based
    eta flags as extra."""
    status = job.get("status", "?")
    if status == "queued":
        return f"queued (waiting for the server's slot), elapsed {_fmt_secs(elapsed)}"
    p = job.get("progress")
    p = float(p) if isinstance(p, (int, float)) else 0.0
    parts = [f"{p:.0f}%"]
    done, total = job.get("steps_done"), job.get("steps_total")
    have_steps = isinstance(done, int) and isinstance(total, int) and total > 0
    if have_steps:
        parts.append(f"step {done}/{total}")
    elif p > 0 and steps_requested:
        est = min(int(round(p / 90.0 * steps_requested)), int(steps_requested))
        parts.append(f"~step {est}/{steps_requested}")
    if sps:
        parts.append(f"{sps:.1f} s/step")
    if p >= 90:
        parts.append("decoding/muxing")
    if have_steps and sps and 0 < done < total:
        parts.append(f"eta ~{_fmt_secs((total - done) * sps)} (+decode)")
    elif not have_steps and 0 < p < 100 and elapsed > 0:
        parts.append(f"eta ~{_fmt_secs(elapsed * (100.0 - p) / p)}")
    parts.append(f"elapsed {_fmt_secs(elapsed)}")
    return ", ".join(parts)


def _progress_bar(node_cls):
    """comfy.utils.ProgressBar scaled to the server's 0..100 job progress. (It
    lives in comfy.utils, NOT nodes — and its update() ADDS a delta, so the job
    poll must call update_absolute.)"""
    uid = getattr(getattr(node_cls, "hidden", None), "unique_id", None)
    try:
        from comfy.utils import ProgressBar
        return ProgressBar(100.0, node_id=uid)
    except Exception:
        return None
