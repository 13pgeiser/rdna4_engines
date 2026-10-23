"""glm5_vision_reference.py MODEL_DIR OUT_DIR IMAGE [IMAGE ...] [--dtype bfloat16|float32] [--taps]
                           [--max-image-tokens N] [--compare DUMP_DIR]

Transformers' own GLM-5.3 vision tower, run on the CPU, as the reference arch/glm5next's tower is
checked against. Only `model.visual.*` is loaded (the checkpoint keeps the whole tower, bf16, in its
last shard), so this needs about 3 GB of memory, not the model's 180.

All the pictures go through the tower in one call, as one encoder pass carries them: one attention
segment each. For each picture it writes, under OUT_DIR/<picture's file name>/:

  pix.bf16     the patches Glm5NextProcessor makes [P, 1176], merge-block order, bf16 -- the
               layout core/mm stages and RADIANCE_DEBUG_MM_DUMP writes as .pix.bf16
  out.f32      the merger's rows [P / 4, 4096], the rows that replace the picture's placeholders
  meta.json    the grid, the patch and token counts, the dtype, and the prompt ids the processor
               makes of <|begin_of_image|><|image|><|end_of_image|>

and with --taps the tensors on the way, each as f32 in the tower's own order:

  patch.f32    the patch embedding [P, 1024]
  q0.f32 k0.f32  block 0's q and k after the per-head norm and the rotary [P, 16 * 64]
  blk0.f32 blk11.f32 blk23.f32  the residual stream after those blocks [P, 1024]
  post.f32     post_layernorm [P, 1024]
  down.f32     the 2 x 2 downsample [P / 4, 4096]

--max-image-tokens is the processor's budget, the checkpoint's 8000 by default: a server whose
encoder pass holds fewer (--mm-max-patches auto is 4096 tokens) sizes a large picture smaller, and
the reference must be asked for the same budget.

The tower's input is the processor's patches rounded to bf16 in both dtypes, which is what the
engine stages: --dtype float32 then measures the tower's arithmetic alone. Every number here comes
from transformers, not from this script.

--compare DUMP_DIR reads what an engine serving the same pictures wrote with
RADIANCE_DEBUG_MM_DUMP=DUMP_DIR (mm_<key>_1xHxW.pix.bf16 and .bf16, matched to the pictures by
grid) and checks it against the reference: the staged patches bit for bit, and the encoder rows to
the tolerance of two bf16 computations of the same tower -- a cosine of at least 0.99 over the
picture and a median per-row relative error of at most 0.08 against the bfloat16 reference.
Measured against it on the CPU (libref, libavx) and one R9700 (libr4d): cosine 0.996 to 0.998 and
a median per-row error of 0.03 to 0.045 at up to 4144 patches, 0.993 and 0.049 at 16384, the same
order as the bfloat16 reference's own against float32; a handful of rows with a small norm differ
more in both (docs/ARCHITECTURES.md). Exits 1 past either bound."""
import argparse, json, os

import numpy as np
import torch
from PIL import Image
from safetensors import safe_open
from transformers import AutoConfig, AutoProcessor
from transformers.models.glm5_next import modeling_glm5_next as M


def load_tower(model_dir, dtype):
    cfg = AutoConfig.from_pretrained(model_dir).vision_config
    cfg._attn_implementation = "sdpa"
    tower = M.Glm5NextVisionModel(cfg)
    index = json.load(open(os.path.join(model_dir, "model.safetensors.index.json")))
    want = {k: v for k, v in index["weight_map"].items() if k.startswith("model.visual.")}
    state = {}
    for shard in sorted(set(want.values())):
        with safe_open(os.path.join(model_dir, shard), "pt") as f:
            for k in f.keys():
                if k in want:
                    state[k[len("model.visual."):]] = f.get_tensor(k)
    missing, unexpected = tower.load_state_dict(state, strict=False)
    if unexpected or [k for k in missing if "inv_freq" not in k]:
        raise SystemExit(f"tower weights: missing {missing}, unexpected {unexpected}")
    return tower.to(dtype).eval()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("out")
    ap.add_argument("images", nargs="+")
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float32"])
    ap.add_argument("--taps", action="store_true")
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--max-image-tokens", type=int, default=0)
    ap.add_argument("--compare", default=None, metavar="DUMP_DIR")
    a = ap.parse_args()
    if a.threads:
        torch.set_num_threads(a.threads)
    dtype = getattr(torch, a.dtype)
    proc = AutoProcessor.from_pretrained(a.model)
    tower = load_tower(a.model, dtype)

    pics = [Image.open(p).convert("RGB") for p in a.images]
    budget = {"max_image_tokens": a.max_image_tokens} if a.max_image_tokens else {}
    enc = proc.image_processor(images=pics, return_tensors="pt", **budget)
    pv, thw = enc["pixel_values"], enc["image_grid_thw"]
    counts = [int(t.prod()) for t in thw]
    pv16 = pv.to(torch.bfloat16)

    taps = {}
    hooks = []
    if a.taps:
        def keep(name, fn=lambda o: o):
            return lambda mod, inp, out: taps.__setitem__(name, fn(out).detach().float())
        hooks.append(tower.patch_embed.register_forward_hook(keep("patch")))
        for i in (0, 11, len(tower.blocks) - 1):
            hooks.append(tower.blocks[i].register_forward_hook(keep(f"blk{i}")))
        hooks.append(tower.post_layernorm.register_forward_hook(keep("post")))
        hooks.append(tower.downsample.register_forward_hook(
            keep("down", lambda o: o.reshape(-1, o.shape[1]))))
        rope = M.apply_rotary_pos_emb_vision

        def rope_tap(q, k, cos, sin):
            qe, ke = rope(q, k, cos, sin)
            if "q0" not in taps:
                taps["q0"] = qe.detach().float().reshape(qe.shape[0], -1)
                taps["k0"] = ke.detach().float().reshape(ke.shape[0], -1)
            return qe, ke
        M.apply_rotary_pos_emb_vision = rope_tap
    with torch.no_grad():
        out = tower(pv16.to(dtype), grid_thw=thw).pooler_output.float()
    if a.taps:
        M.apply_rotary_pos_emb_vision = rope
        for h in hooks:
            h.remove()

    p0 = r0 = 0
    for path, im, t, n in zip(a.images, pics, thw.tolist(), counts):
        d = os.path.join(a.out, os.path.basename(path))
        os.makedirs(d, exist_ok=True)
        r = n // 4
        pv16[p0:p0 + n].view(torch.int16).numpy().astype("<i2").tofile(os.path.join(d, "pix.bf16"))
        out[r0:r0 + r].numpy().astype("<f4").tofile(os.path.join(d, "out.f32"))
        for name, v in taps.items():
            lo, hi = (r0, r0 + r) if name == "down" else (p0, p0 + n)
            v[lo:hi].numpy().astype("<f4").tofile(os.path.join(d, name + ".f32"))
        ids = proc(text=["<|begin_of_image|><|image|><|end_of_image|>"], images=[im],
                   return_tensors="pt", **budget)["input_ids"][0].tolist()
        json.dump({"grid_thw": t, "patches": n, "tokens": r, "dtype": a.dtype,
                   "prompt_ids_len": len(ids), "prompt_ids_head": ids[:3], "prompt_ids_tail": ids[-2:],
                   "taps": sorted(taps)}, open(os.path.join(d, "meta.json"), "w"), indent=1)
        p0 += n
        r0 += r
    print(f"wrote {len(a.images)} picture(s) to {a.out}: {counts} patches, dtype {a.dtype}")
    if a.compare:
        raise SystemExit(compare(a.compare, a.images, thw.tolist(), pv16, out, counts))


def compare(dump, images, grids, pv16, out, counts):
    """The engine's patches and rows against the reference's; 0 when every picture is within."""
    def bf16(path):
        return (np.fromfile(path, dtype="<u2").astype(np.uint32) << 16).view(np.float32)
    bad = 0
    p0 = r0 = 0
    for path, (t, h, w), n in zip(images, grids, counts):
        r = n // 4
        name = os.path.basename(path)
        rows = [f for f in os.listdir(dump) if f.endswith(f"_{t}x{h}x{w}.bf16")]
        pix = [f for f in os.listdir(dump) if f.endswith(f"_{t}x{h}x{w}.pix.bf16")]
        if len(rows) != 1 or len(pix) != 1:
            print(f"  {name}: {len(rows)} row dump(s) and {len(pix)} patch dump(s) for a "
                  f"{t}x{h}x{w} grid in {dump}; expected one each")
            bad += 1
            p0 += n
            r0 += r
            continue
        same = np.array_equal(np.fromfile(os.path.join(dump, pix[0]), dtype="<u2"),
                              pv16[p0:p0 + n].view(torch.int16).numpy().astype("<u2").ravel())
        e = bf16(os.path.join(dump, rows[0])).reshape(-1, 4096).astype(np.float64)
        ref = out[r0:r0 + r].numpy().astype(np.float64)
        if e.shape != ref.shape:
            print(f"  {name}: {e.shape[0]} rows, the reference {ref.shape[0]}")
            bad += 1
        else:
            cos = float((e * ref).sum() / np.sqrt((e * e).sum() * (ref * ref).sum()))
            row = np.linalg.norm(e - ref, axis=1) / np.linalg.norm(ref, axis=1)
            med, worst = float(np.median(row)), float(row.max())
            ok = same and cos >= 0.99 and med <= 0.08
            bad += not ok
            print(f"  {name}: patches {'identical' if same else 'DIFFER'}; rows: cosine {cos:.5f}, "
                  f"per-row relative error median {med:.4f}, largest {worst:.4f} -- "
                  f"{'within' if ok else 'OUTSIDE'}")
        p0 += n
        r0 += r
    return 1 if bad else 0


if __name__ == "__main__":
    main()
