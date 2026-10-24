# Integrations

Client-side companions to the radiance server — code that runs *beside* other
tools, not inside radiance.

## comfyui/ComfyUI-Radiance-MiniMax

Radiance's first ComfyUI node: one node, **Radiance MiniMax**, that drives a
radiance server's `POST /v1/videos` over HTTP and returns the finished clip.
It is a pure HTTP client — no model, no VRAM on the ComfyUI side — so it
coexists with the server even on the same box, and one frontend can point at
several servers on the LAN.

Drop the `ComfyUI-Radiance-MiniMax/` folder into
`ComfyUI/custom_nodes/` and restart ComfyUI; `README.md` inside it documents
every input (tasks, media sockets, LoRA slots, the size law) and the server
side it expects (`--lora-dir`, CORS, no auth — trusted LAN only).

The folder layout is exactly what ComfyUI loads, so the repo copy and the
installed copy are the same files — `cp -r` one way. (The location inside
this repo is a first-draft guess; move it if `contrib/` or `clients/` suits
better.)
