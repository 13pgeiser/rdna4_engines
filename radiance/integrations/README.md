# Integrations

Clients of the radiance server that run inside other tools.

## comfyui/ComfyUI-Radiance-MiniMax

A ComfyUI node that renders a clip on a radiance server serving MiniMax-H3, over
`POST /v1/videos`, and returns it. The folder is a ComfyUI custom node as it stands: copy it into
`ComfyUI/custom_nodes/`. Its README documents the inputs and the server it expects.
