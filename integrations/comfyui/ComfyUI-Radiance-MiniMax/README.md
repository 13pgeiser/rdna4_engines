# ComfyUI-Radiance-MiniMax

One node: **Radiance MiniMax**. It calls a Radiance MiniMax-H3 server's
`POST /v1/videos` over HTTP and hands back the finished clip.

The node is a pure HTTP client — it loads **no model and no VRAM**. The ~50–60 GiB
checkpoint (size depends on the served container) lives in the radiance server process,
which can be this machine
(`http://127.0.0.1:8133`) or any TP2 box on the LAN (`http://192.168.1.x:8133`).
Run three servers, point three copies of the node at them, one ComfyUI
frontend drives all of them. No contention with radiance's VRAM, even when
ComfyUI runs on the same box: the node only sends bytes.

## Server side

    radiance --model /path/to/minimax-h3-ref2va-*.rad --radiance-home ... \
        --host 0.0.0.0 --port 8133 --tp 2 --vision-tower-placement ram --max-model-len 4096

`--host 0.0.0.0` (or your LAN address) is what makes the box reachable from
another machine; the server has **no authentication**, so keep it on a trusted
LAN. Open the port in the firewall (`ufw allow 8133/tcp`). CORS is on, so a
LAN client needs nothing else.

To serve LoRA files, add `--lora-dir DIR`: any `NAME.safetensors` under it
(subdirs one level deep as `sub/NAME`) can then be named by a request. Same-box
tidbit: mount ComfyUI's `models/loras` into the container and point `--lora-dir`
at the mount, and the LoRAs you browse for `Load LoRA` and the ones radiance
lists at `GET /v1/loras` are the **same files** (radiance reads the kohya /
ComfyUI / PEFT key spellings as they come). New LoRA without a restart:
`POST /v1/load_lora_adapter {"lora_name": "...", "lora_path": "..."}` with a
path on the **server's** filesystem (loopback-only unless `--api-key` is set).

**LoRA slots and VRAM — tight on 32 GB cards.** Every adapter slot is *reserved
at server start* and lives in VRAM whether used or not (the startup log prints
"N slots of rank R, X a slot on this rank"). A video generator's activation
arena is sized for its largest legal step and is not shrinkable, so the slots
compete head-on with the KV cache for whatever slack remains. Measured on this
node's flagship serve — MiniMax-H3, TP2, 32 GB cards, rank 16 — one slot costs
**265 MiB per card and exactly one slot of slack remains** after the arena:
`--max-loras 1` boots by a hair; `--max-loras 2` boots **only if nothing else
holds VRAM** (stop ComfyUI on the same box — it holds ~1.7 GiB itself — and any
other GPU process). The failure signature is a boot that stops with
`the static weights alone want … of the … this card can give` then
`VRAM pool budgets are empty: … vram_kv resolved to 0 MiB` (exit 1).
Lowering `--max-num-batched-tokens` to claw room back does NOT work for a
video generator: its auto value is a floor — the largest denoise step must fit
the arena — and a lower one is refused at declare. **This contention exists
only between GPU processes on the machine that HOSTS radiance.** The node is a
pure HTTP client, so the standard split topology — ComfyUI on its own box/GPU,
radiance headless on another — leaves the server's cards otherwise idle and
`--max-loras 2` fits fine there; that is the way to run two slots. **On one
shared machine, serve with `--max-loras 1` as the default** (this node's
`lora 1` keeps working, keep `lora 2` at `none`); go to 2 only when nothing
else on that box touches the GPU.

## Inputs

- **server_url / model** — which box, which served model (`GET /v1/models` lists ids).
- **prompt** — widget *and* STRING socket: wire your LLM prompt-enhancer's
  output straight into `prompt` and the wire wins over the typed text. The
  MiniMax format (`subject_definitions` / `summary` / `retention_analysis` /
  `detailed_description`, `<Picture N>` / `<Subject N>` / `<Setting N>` /
  `<Style N>` tags, timed shots, `<d>[English] …</d>` dialogue) is the format
  the model was trained on; short one-line prompts garble the audio.
- **there is no negative_prompt** — deliberately. MiniMax-H3 is
  guidance-distilled: CFG is baked into the weights and every step is a single
  forward pass, so the model has no negative branch to steer. The server says so
  too — a request that names `negative_prompt` on this model is *refused*
  ("this model takes no guidance"), never silently ignored. Negative concepts
  belong in the prompt's own wording.
- **loras / lora 2** — **two dropdown slots** of the server-side adapters,
  `none` = off, either slot able to stand alone — the pair mirrors the
  server's `--max-loras 2`. The list is fetched from the box in `server_url`
  (`GET /v1/loras`): when a workflow loads, when you edit the URL, and via
  the **refresh lora list** button — the button refreshes the list **only**,
  it never writes anything into the widgets. Each slot has its own strength
  slider (sent as `NAME:STRENGTH`). More than two adapters: `<lora:name:0.8>`
  tags in the prompt plus a raised `--max-loras` on the server (and a graph
  saved against older builds whose box still holds free text like
  `film:0.8, ink` keeps working verbatim). The `.safetensors` files must live
  on the **radiance server** — it was started with `--lora-dir DIR` (see
  Server side below). **ComfyUI's `Load LoRA` node cannot drive this node**:
  its output patches a local torch model, while radiance applies adapters to
  the served weights inside its own process. Adapters work with every task
  (`ref2va` references and `fl2va` keyframes included — unlike
  `latent_upscale`). Both container quants carry them (fp8 and the int8 trunk
  alike).
- **seconds, seed, num_inference_steps, sampler, shift, audio_shift** —
  the usual dials; `sampler` is `euler` or `res_multistep`. `seed` caps at
  `9007199254740991` (2^53−1, 16 digits) — the largest seed radiance holds
  **exactly** (it carries seeds as doubles). KSampler/DaSiWa allow 20 digits,
  but beyond 2^53 radiance *rounds* the seed and beyond 2^63 everything
  collapses to one constant (verified live) — so 16 digits is the real engine
  max, and a DaSiWa random seed pasted from above gets an honest clamp instead
  of a silent rewrite.
- **resolution / aspect** — DaSiWa-style size picking. The p-tier dropdown
  (144p … 720p) sets the **short side**; the aspect dropdown (landscape→portrait:
  `5:4, 4:3, 3:2, 16:9, 2:1, 1:1, 1:2, 9:16, 2:3, 3:4, 4:5`) computes the long
  side — `480p + 16:9` → 864×480, `480p + 9:16` → 480×864, `1:1` → square —
  and the **node title shows the computed WxH** live. Both sides snap to
  multiples of 32 px (256 floor) and the canvas scales back proportionally if
  needed, so every pick satisfies the model's size law (ratio ≤ 4:1, area ≤
  1344×768 = 1,032,192 px): `720p + 16:9` lands on **1312×736** — true
  `1280x720` is impossible under the law (720 is not a multiple of 32).
  **Every size works for every task, keyframes included** — `i2va`/`l2va`/
  `fl2va` resize their keyframe(s) to the exact output canvas and encode them
  whole (needs a server built with the grown one-frame encoder buffers, Oct
  2026). Tiers stop at 720p because a *single pass* cannot exceed the canvas:
  DaSiWa's 900/1024/1080p presets are megapixel budgets run through a
  **second stage** (latent upscaler + refiner) that Radiance's single DiT pass
  has no equivalent of yet — its `latent_upscale` refine steps still run at
  full size. `custom` (aspect ignored) is **auto-adjusted, not refused** —
  like ComfyUI's own resize, any `WxH` snaps to the nearest size satisfying
  the law (`480x720` → `480x736`, `4096x1024` → `1920x480`) and the change is
  logged to the console; only text that isn't a size errors.
- **latent_upscale** — advanced, off by default. A container converted with
  the latent-upscaler weights (the DaSiWa companion folder; `minimax-h3-dasiwa-*.rad`
  has them — the server log names the upscaler at boot) diffuses the early
  steps at size/2 (or /4), enlarges the clean latent, then refines at full
  size in the last steps. Works with **`t2va` and `ref2va`** — reference rows
  carry their own grids, so they condition the draft and the refine stage
  identically — and is **refused with `i2va`/`l2va`/`fl2va` keyframes**: a pin
  sits AT the exact output frame on the exact target grid, and a draft canvas
  leaves it nowhere to sit (send the frame as a reference image instead). The
  output still stays within the canvas, so it buys speed and detail, not 1080p.

### Media sockets — purely wire-driven, task-sensitive

18 sockets: `image_1..image_9` (IMAGE), `video_1..video_3` (VIDEO),
`video_audio_1..video_audio_3` (AUDIO), `audio_1..audio_3` (AUDIO). Wire
anything — Load Image / Load Video / Load Audio or generated media. **Which
sockets are live depends on the task; everything else is ignored** (a console
warning says so). *The node header's **? button opens a full help pane
documenting every input; this section is the same text.*

| task | live sockets |
|---|---|
| `t2va` | none — text only |
| `i2va` | `image_1` → first frame |
| `l2va` | `image_1` → last frame |
| `fl2va` | `image_1` → first frame, `image_2` → last frame |
| `ref2va` | all of them |

For `ref2va`, roles are by socket kind, never by position-magic:
`video_audio_N` is the **soundtrack of `video_N`** (wire that video or it's an
error); `audio_N` is a **standalone reference audio** (voice/timbre), no video
involved. Fill slots **in order** (`image_3` without `image_2` is a hard error
— the socket number IS the conditioning position, and the server groups
references by it). Images upload as PNG, audio as WAV, video via ComfyUI's own
encoder (stream copy when the source is already mp4/h264) — nothing touches
the `input/` folder.

## Outputs

- **video** — a `VIDEO` the frontend previews natively; chain it to SaveVideo,
  VHS nodes, etc.
- **file** — the clip is **already saved** by the node into the output folder
  under `radiance/Radiance_00001_.mp4` (the clip lands in ComfyUI's output
  folder even with no SaveVideo downstream; adding one would write a second
  copy).
- **job id** — the server-side job id, for `GET /v1/videos/<id>` or curl replay.

While a job runs the node polls every 3 s and drives ComfyUI's node progress
bar, and **prints a live line to the ComfyUI console once per completed step**
(a keep-alive every 30 s covers encoding and any stall) —
`[RadianceMiniMax] job_x: 45%, step 12/26, 5.2 s/step, eta ~1m 07s (+decode),
elapsed 55s` — mirroring the server's own per-step log, so a waiting user sees
pace and progress, not a silent spinner and not a poll-spammed wall of nearly
identical lines. The `s/step` is **measured over ≥2-step windows** from the
step counter and lightly smoothed, so it tracks the live pace — including the
step-price jump when a latent-upscaled job switches from draft to refine grid
— and the eta is remaining steps × that pace (the `+decode` tail is extra, as
on the server's own log). Polling is unavoidable (the HTTP API pushes nothing);
*printing* is event-driven off the step counter. Real step counts come from a radiance server built after Oct 2026
(it added `steps_done`/`steps_total` to the job object); an older server shows
`~step` derived from the job's `progress` field, which is the steps' share of
0..90 by design, and extrapolates its eta from that progress. That same new
server also prints one `RAD_INFO` line per step, so `docker logs -f` shows the
identical pace on the server box. `timeout` (advanced, default 1800 s) bounds
the wait. A refusal, a failed job or an unreachable server all
raise with the server's own error text.
