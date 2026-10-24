"""mkuocrtilegoldens.py tile|e2e|multi -- goldens for the Unlimited-OCR 640-tile geometry and
for multi-page requests, every number from the checkpoint's own code (CODEDIR's deepencoder.py
and modeling_unlimitedocr.py as shipped, weights from MODELDIR's safetensors).

  tile  (out: OUTDIR)   the 640x640 tile reference for a 1024x1024 document:
    sam_pos40.npz/.json   model.sam_model.pos_embed [1,64,64,768] through get_abs_pos_sam to 40
                          (F.interpolate bicubic antialias in fp32, cast back bf16): 40x40x768 of
                          bf16 bit patterns as raw uint16.
    clip_pos10.npz/.json  model.vision_model.embeddings.position_embedding.weight [257,1024]
                          through get_abs_pos(..., 101): cls row passthrough + 16x16 resampled to
                          10x10, 101x1024 of bf16 raw uint16.
    tile0.pix.bf16        the FIRST dynamic_preprocess crop (the picture resized to 1280x1280,
                          crop [0:640,0:640]), normalized (x/255-0.5)/0.5 to bf16, staged the way
                          Processor::append (core/mm/processor.cpp:1102) stages a 40x44 grid: 40
                          patch rows of 44 patch columns, the last 4 columns zeros, each patch
                          (channel, y, x) row-major, 40*44*768 bf16 raw uint16.
    tile0.npz             the DeepEncoder stage taps over that one crop, the uocr_ref.py key set
                          (st_x0, st_pre/attn/b 0..11, st_full0..2, st_n0..3, st_map, st_map2,
                          sam, clip, fuse, proj, rows, image, patches, crop_rgb); the stage taps
                          carry the script's first-256-rows range.
    rows.bin              the 110 layout rows (10 feature rows of 10 proj rows + image_newline)
                          as raw bf16, view_seperator excluded.
    tile0.json            shapes of every artifact, the crop derivation, the verification cosines.
  e2e   (out: OUTDIR/e2e_document.{json,txt})
        the checkpoint's infer() on document.png at its defaults (base 1024, tile 640, crop_mode,
        greedy, ngram 35/128) for 200 max tokens, with the prompt ids and the per-picture layout
        it built.
  multi (out: OUTDIR/multi_document_screenshot.{json,txt})
        the checkpoint's infer_multi() on document.png + screenshot.png at image_size 640, greedy,
        ngram 35/128, 200 max tokens, with the per-page token/row layout.

CPU only: CUDA is disabled at import; the checkpoint's .cuda() calls are patched to identity.
"""
import json
import math
import os
import sys
import types

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import numpy as np  # noqa: E402
import torch  # noqa: E402

torch.Tensor.cuda = lambda self, *a, **k: self
torch.nn.Module.cuda = lambda self, *a, **k: self
try:
    torch.cuda.empty_cache = lambda: None
except Exception:
    pass

MODELDIR = "/home/juup/models/Unlimited-OCR"
CODEDIR = "/tmp/kilo/uocr"
IMGDIR = "/tmp/kilo/uocr-images"
TILE_DIR = "/tmp/kilo/uocr-ref-tile"
MULTI_DIR = "/tmp/kilo/uocr-ref-multi"

sys.path.insert(0, CODEDIR)
import deepencoder as de  # noqa: E402

PATCH, DS = 16, 4
BASE, TILE = 1024, 640
IMAGE_TOKEN_ID = 128815


def uocr_pkg():
    """the checkpoint's remote code as a package (its relative imports need one)."""
    pkg = types.ModuleType("uocr")
    pkg.__path__ = [MODELDIR]
    sys.modules["uocr"] = pkg
    import importlib
    return importlib.import_module("uocr.modeling_unlimitedocr")


def build_vision():
    from safetensors.torch import load_file
    sd = load_file(os.path.join(MODELDIR, "model-00001-of-000001.safetensors"))
    w = {}
    for k, v in sd.items():
        for p, m in (("model.sam_model.", "sam"), ("model.vision_model.", "clip"),
                     ("model.projector.", "proj")):
            if k.startswith(p):
                w[(m, k[len(p):])] = v
        if k == "model.image_newline":
            w[("newline", "")] = v
        if k == "model.view_seperator":
            w[("sep", "")] = v
    del sd
    sam = de.build_sam_vit_b()
    vision = de.build_clip_l()
    proj = de.MlpProjector(de.adict(projector_type="linear", input_dim=2048, n_embed=1280))
    sam.load_state_dict({k: v for (m, k), v in w.items() if m == "sam"}, strict=True)
    vision.load_state_dict({k: v for (m, k), v in w.items() if m == "clip"}, strict=False)
    proj.load_state_dict({k: v for (m, k), v in w.items() if m == "proj"}, strict=True)
    for m in (sam, vision, proj):
        m.eval().to(torch.bfloat16)
    return sam, vision, proj, w[("newline", "")], w[("sep", "")]


def u16(t):
    return t.view(torch.int16).detach().cpu().numpy().astype(np.uint16)


def cosine(a, b):
    a = a.double().flatten()
    b = b.double().flatten()
    return float((a @ b) / (a.norm() * b.norm()))


def load_gate_image(path):
    im = uocr_pkg().load_image(path)
    return im.convert("RGB")


# --------------------------------------------------------------------- tile


def tile_crop(M):
    """the first 640 crop infer(crop_mode=True) feeds as images_crop_list[0], verbatim:
    dynamic_preprocess(image) at its defaults, then the crop. Returns (crop, derivation)."""
    im = load_gate_image(os.path.join(IMGDIR, "document.png"))
    crops, ratio = M.dynamic_preprocess(im)
    w, h = im.size
    box = (0, 0, TILE, TILE)
    derivation = {
        "source": os.path.join(IMGDIR, "document.png"),
        "source_size_wh": [w, h],
        "aspect": w / h,
        "crop_ratio_cols_rows": [int(ratio[0]), int(ratio[1])],
        "resized_to_wh": [TILE * ratio[0], TILE * ratio[1]],
        "resize": "PIL Image.resize(BICUBIC) (the call's default resample)",
        "crop_box_xyxy": list(box),
        "crop_count": len(crops),
    }
    return crops[0], ratio, crops, derivation


def transform(im):
    """BasicImageTransform(mean=.5,std=.5) -> bf16, the model's own."""
    M = uocr_pkg()
    tr = M.BasicImageTransform(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), normalize=True)
    return tr(im).to(torch.bfloat16)


def stage_engine(x, q, cols, extra):
    """the Processor::append staging: rows of (cols+extra) patch columns, the last `extra` all
    zeros, each patch (channel, y, x); bf16 bit patterns as raw uint16 [q, cols+extra, 3,P,P]."""
    c, h, w = x.shape
    grid = x.reshape(c, q, PATCH, cols, PATCH).permute(1, 3, 0, 2, 4)
    out = torch.zeros(q, cols + extra, c, PATCH, PATCH, dtype=x.dtype)
    out[:, :cols] = grid
    return out


def tile_forward(sam, vision, proj, newline, sep, X):
    """the DeepEncoder over one 640 crop, the checkpoint's own ops, taps in uocr_ref.py layouts."""
    out = {}
    t = sam.patch_embed(X)
    pos40 = de.get_abs_pos_sam(sam.pos_embed, X.size(2) // PATCH)
    t = t + pos40
    out["x0"] = t.reshape(-1, 768)
    H = W = X.size(2) // PATCH
    for i, blk in enumerate(sam.blocks):
        sc = t
        h = blk.norm1(t)
        if blk.window_size > 0:
            hp, pad_hw = de.window_partition(h, blk.window_size)
            a = blk.attn(hp)
            a = de.window_unpartition(a, blk.window_size, pad_hw, (H, W))
        else:
            a = blk.attn(h)
        out["attn%d" % i] = a.reshape(-1, 768)
        out["pre%d" % i] = (sc + a).reshape(-1, 768)
        t = sc + a
        t = t + blk.mlp(blk.norm2(t))
        out["b%d" % i] = t.reshape(-1, 768)
        if i in (0, 1, 2):
            out["full%d" % i] = t.reshape(-1, 768)
    mm = t.permute(0, 3, 1, 2).contiguous()
    n0 = sam.neck[0](mm)
    n1 = sam.neck[1](n0)
    n2 = sam.neck[2](n1)
    n3 = sam.neck[3](n2)
    out["map"] = n3.flatten(2).permute(0, 2, 1)[0]
    out["n0"] = n0.flatten(2).permute(0, 2, 1)[0]
    out["n1"] = n1.flatten(2).permute(0, 2, 1)[0]
    out["n2"] = n2.flatten(2).permute(0, 2, 1)[0]
    out["n3"] = n3.flatten(2).permute(0, 2, 1)[0]
    out["map2"] = sam.net_2(n3).flatten(2).permute(0, 2, 1)[0]
    m1 = sam.net_3(sam.net_2(n3))
    m2 = vision(X, m1)
    sam_rows = m1.flatten(2).permute(0, 2, 1)
    clip_rows = m2[:, 1:]
    fuse = torch.cat((clip_rows, sam_rows), dim=-1)
    pf = proj(fuse)
    h = w = int(pf.shape[1] ** 0.5)
    g = pf.view(h, w, -1)
    g = torch.cat([g, newline.view(1, 1, -1).expand(h, 1, -1)], dim=1).view(-1, pf.shape[-1])
    rows = torch.cat([g, sep.view(1, -1)], dim=0)
    return out, dict(pos40=pos40, sam=sam_rows[0], clip=clip_rows[0], fuse=fuse[0], proj=pf[0],
                     layout110=g, rows=rows)


def clip_pos_rows(vision, tgt_grid):
    pos = vision.embeddings.position_embedding(vision.embeddings.position_ids)
    return de.get_abs_pos(pos, tgt_grid * tgt_grid + 1)


def cmd_tile():
    out = sys.argv[2] if len(sys.argv) > 2 else TILE_DIR
    os.makedirs(out, exist_ok=True)
    M = uocr_pkg()
    crop, ratio, crops, derivation = tile_crop(M)
    x = transform(crop)
    q = TILE // PATCH

    with torch.inference_mode():
        # A1: the SAM absolute table at the 40 grid.
        sam, vision, proj, newline, sep = build_vision()
        pos40 = de.get_abs_pos_sam(sam.pos_embed, q)
        np.savez(os.path.join(out, "sam_pos40.npz"), pos=u16(pos40[0]))
        json.dump({"pos": {"shape": [q, q, 768], "dtype": "bf16 raw uint16",
                           "source": "model.sam_model.pos_embed [1,64,64,768]",
                           "op": "get_abs_pos_sam: F.interpolate bicubic antialias fp32 "
                                 "align_corners=False, cast back bf16"}},
                  open(os.path.join(out, "sam_pos40.json"), "w"), indent=1)

        # A2: the CLIP absolute table at the 10 grid (cls row 0 passthrough).
        pos10 = clip_pos_rows(vision, 10)
        np.savez(os.path.join(out, "clip_pos10.npz"), pos=u16(pos10[0]))
        json.dump({"pos": {"shape": [101, 1024], "dtype": "bf16 raw uint16",
                           "source": "model.vision_model.embeddings.position_embedding.weight "
                                     "[257,1024]",
                           "op": "get_abs_pos(..., 101): row 0 passthrough, rows 1: as 16x16 "
                                 "resampled to 10x10 (bicubic antialias fp32, cast bf16), cls "
                                 "concatenated first"}},
                  open(os.path.join(out, "clip_pos10.json"), "w"), indent=1)

        # B: the engine-staged tile pixels (40 rows x 44 cols, 4 zero columns).
        staged = stage_engine(x, q, q, 4)
        staged.view(torch.int16).detach().cpu().numpy().astype(np.uint16).tofile(
            os.path.join(out, "tile0.pix.bf16"))

        stages, fin = tile_forward(sam, vision, proj, newline, sep, x.unsqueeze(0))

        p = np.array(crop, dtype=np.uint8)
        patches = torch.nn.functional.unfold(x.unsqueeze(0), kernel_size=16, stride=16)[0].t()
        np.savez(os.path.join(out, "tile0.npz"),
                 image=x.to(torch.float32).numpy().astype(np.float16),
                 patches=patches.float().numpy().astype(np.float32),
                 crop_rgb=p,
                 rows=fin["rows"].float().numpy().astype(np.float32),
                 sam=fin["sam"].float().numpy().astype(np.float32),
                 clip=fin["clip"].float().numpy().astype(np.float32),
                 fuse=fin["fuse"].float().numpy().astype(np.float32),
                 proj=fin["proj"].float().numpy().astype(np.float32),
                 newline=newline.float().numpy().astype(np.float32),
                 sep=sep.float().numpy().astype(np.float32),
                 **{"st_" + k: v[:256].float().numpy().astype(np.float32)
                    for k, v in stages.items()})
        fin["layout110"].view(torch.int16).detach().cpu().numpy().astype(np.uint16).tofile(
            os.path.join(out, "rows.bin"))

        # VERIFY: the staged pixels reproduce the crop, and st_x0 is the staged crop through
        # patch_embed + the 40 table.
        raw = np.fromfile(os.path.join(out, "tile0.pix.bf16"), dtype=np.uint16)
        rec = torch.from_numpy(raw.astype(np.int16)).view(torch.bfloat16).view(q, q + 4, 3,
                                                                               PATCH, PATCH)
        bits_equal = bool(torch.equal(rec[:, :q].permute(2, 0, 3, 1, 4).reshape(3, TILE, TILE), x))
        X2 = rec[:, :q].permute(2, 0, 3, 1, 4).reshape(1, 3, TILE, TILE).contiguous()
        x0b = sam.patch_embed(X2) + de.get_abs_pos_sam(sam.pos_embed, q)
        x0b = x0b.reshape(-1, 768)
        verif = {"staged_bits_equal_to_transform": bits_equal,
                 "cos_x0_from_pix_vs_stage0": cosine(x0b, stages["x0"]),
                 "bit_equal_x0_from_pix_vs_stage0": bool(torch.equal(x0b, stages["x0"])),
                 "stage_cosines": {}, "stage_bit_equal": {}}
        stages2, fin2 = tile_forward(sam, vision, proj, newline, sep, X2)
        for k in stages:
            verif["stage_cosines"]["st_" + k] = cosine(stages2[k], stages[k])
            verif["stage_bit_equal"]["st_" + k] = bool(torch.equal(stages2[k], stages[k]))
        for k in ("sam", "clip", "fuse", "proj", "rows"):
            verif["stage_cosines"][k] = cosine(fin2[k], fin[k])
            verif["stage_bit_equal"][k] = bool(torch.equal(fin2[k], fin[k]))

    manifest = {
        "crop_derivation": derivation,
        "artifacts": {
            "sam_pos40.npz:pos": [q, q, 768], "clip_pos10.npz:pos": [101, 1024],
            "tile0.pix.bf16": [q, q + 4, 3, PATCH, PATCH],
            "rows.bin": [110, 1280],
            "tile0.npz": {"image": [3, 640, 640], "patches": [1600, 768],
                          "crop_rgb": [640, 640, 3], "rows": [111, 1280],
                          "sam": [100, 1024], "clip": [100, 1024], "fuse": [100, 2048],
                          "proj": [100, 1280], "newline": [1280], "sep": [1280],
                          "st_x0/st_pre*/st_attn*/st_b*": [256, 768],
                          "st_full0..2": [1600, 768], "st_n0..3": [256, 256],
                          "st_map": [256, 1024], "st_map2": [256, 512]},
        },
        "notes": {
            "window_pad": "40 % 14 = 12 -> window_partition pads to 42x42 (bottom/right zeros), "
                          "9 windows of 14x14; unpartition crops back to 40x40",
            "global_rel_pos": "blocks 2/5/8/11 are global over 40x40: get_rel_pos resizes the "
                              "127-row table (input 64x64) to 2*40-1 = 79 rows, linear",
            "rows_bin": "110 rows of 10x10 proj rows + image_newline every 10, raw bf16; the "
                        "npz rows key adds the view_seperator row (111 rows, fp32)",
            "pix_layout": "Processor::append (processor.cpp:1102): py rows of gw=44 patch cols, "
                          "cols 40..43 zeros, patch = (channel, y, x) of (x/255-0.5)/0.5 bf16",
        },
        "verify": verif,
    }
    json.dump(manifest, open(os.path.join(out, "tile0.json"), "w"), indent=1)
    print("crop:", json.dumps(derivation))
    print("staged bits equal:", bits_equal, " x0 cos:", verif["cos_x0_from_pix_vs_stage0"],
          " x0 bit-equal:", verif["bit_equal_x0_from_pix_vs_stage0"])
    for k, v in verif["stage_cosines"].items():
        print("cos %-12s %.9f" % (k, v))
    print("wrote", out)


# --------------------------------------------------------------------- e2e / multi


def capture_generate(model, max_new_tokens=200):
    """wrap model.generate to record the prefill tensors infer()/infer_multi() built. The cap is
    forced through max_new_tokens: the checkpoint's generation_config.json ships max_new_tokens
    32768, which transformers makes take precedence over the max_length infer() passes."""
    box = {"max_new_tokens": max_new_tokens}
    orig = model.generate

    def gen(*a, **kw):
        kw = dict(kw)
        kw.pop("max_length", None)
        kw["max_new_tokens"] = max_new_tokens
        ids = kw.get("input_ids", a[0] if a else None)
        box["input_ids"] = ids[0].tolist()
        imgs = kw.get("images")
        box["images_crop_shape"] = list(imgs[0][0].shape)
        box["images_ori_shape"] = list(imgs[0][1].shape)
        box["images_spatial_crop"] = kw["images_spatial_crop"].tolist()
        box["seq_mask_true"] = int(kw["images_seq_mask"].sum())
        out = orig(*a, **kw)
        box["output_ids"] = out[0].tolist()
        return out
    model.generate = gen
    return box


def load_full():
    from transformers import AutoTokenizer, AutoModel
    tok = AutoTokenizer.from_pretrained(MODELDIR, trust_remote_code=True)
    m = AutoModel.from_pretrained(MODELDIR, trust_remote_code=True, use_safetensors=True,
                                  torch_dtype=torch.bfloat16).eval()
    return tok, m


def gen_report(tok, box, prompt_len):
    full = box["output_ids"]
    gen = full[prompt_len:]
    rep = {
        "prompt_token_ids_len": prompt_len,
        "generation_cap_max_new_tokens": box.get("max_new_tokens"),
        "input_ids": box["input_ids"],
        "images_spatial_crop": box["images_spatial_crop"],
        "images_crop_shape": box["images_crop_shape"],
        "images_ori_shape": box["images_ori_shape"],
        "image_seq_mask_true_count": box["seq_mask_true"],
        "generated_len": len(gen),
        "first_24_generated_ids": gen[:24],
        "first_24_generated_text": tok.decode(gen[:24], skip_special_tokens=False),
        "first_12_generated_ids": gen[:12],
        "full_generated_ids": gen,
        "eos_token_id": tok.eos_token_id,
        "ended_with_eos": bool(gen and gen[-1] == tok.eos_token_id),
        "decoded_raw": tok.decode(gen, skip_special_tokens=False),
    }
    stop = "<｜end▁of▁sentence｜>"
    txt = tok.decode(gen, skip_special_tokens=True)
    if txt.endswith(stop):
        txt = txt[:-len(stop)]
    rep["decoded_strip_stop"] = txt
    return rep, txt


def cmd_e2e():
    out = sys.argv[2] if len(sys.argv) > 2 else TILE_DIR
    tok, model = load_full()
    box = capture_generate(model)
    M = uocr_pkg()
    img = os.path.join(IMGDIR, "document.png")
    im = load_gate_image(img)
    crops, ratio = M.dynamic_preprocess(im)
    nq_b, nq = BASE // PATCH // DS, TILE // PATCH // DS
    global_tokens = (nq_b + 1) * nq_b + 1
    local_tokens = (nq * ratio[0] + 1) * nq * ratio[1]
    model.infer(tok, prompt="<image>document parsing.", image_file=img,
                output_path=os.path.join(out, "hfout"), base_size=BASE, image_size=TILE,
                crop_mode=True, max_length=200, no_repeat_ngram_size=35, ngram_window=128,
                temperature=0.0, save_results=False)
    rep, txt = gen_report(tok, box, len(box["input_ids"]))
    rep["picture"] = img
    rep["infer_args"] = {"prompt": "<image>document parsing.", "base_size": BASE,
                         "image_size": TILE, "crop_mode": True, "max_length": 200,
                         "no_repeat_ngram_size": 35, "ngram_window": 128, "temperature": 0.0}
    rep["layout"] = {
        "num_tiles": len(crops), "grid_cols_x_rows": [int(ratio[0]), int(ratio[1])],
        "token_segments": {"global": global_tokens, "local": local_tokens,
                           "total_image_tokens": global_tokens + local_tokens},
        "row_segments": {"local_rows": (ratio[1] * nq) * (ratio[0] * nq + 1),
                         "global_rows": nq_b * (nq_b + 1), "separator_rows": 1},
    }
    assert rep["layout"]["token_segments"]["total_image_tokens"] == box["seq_mask_true"]
    json.dump(rep, open(os.path.join(out, "e2e_document.json"), "w"), indent=1, ensure_ascii=False)
    open(os.path.join(out, "e2e_document.txt"), "w").write(rep["decoded_raw"])
    print("first12", rep["first_12_generated_ids"])
    print("generated_len", rep["generated_len"])
    print(txt[:400])
    print("wrote", os.path.join(out, "e2e_document.json"))


def cmd_multi():
    out = sys.argv[2] if len(sys.argv) > 2 else MULTI_DIR
    tok, model = load_full()
    box = capture_generate(model)
    pages = [os.path.join(IMGDIR, "document.png"), os.path.join(IMGDIR, "screenshot.png")]
    model.infer_multi(tok, prompt="<image>Multi page parsing.", image_files=pages,
                      output_path=os.path.join(out, "hfout"), image_size=TILE, max_length=200,
                      no_repeat_ngram_size=35, ngram_window=128, temperature=0.0,
                      save_results=False)
    rep, txt = gen_report(tok, box, len(box["input_ids"]))
    nq = TILE // PATCH // DS
    per_tokens = (nq + 1) * nq + 1
    rep["pages"] = pages
    rep["infer_args"] = {"prompt": "<image>Multi page parsing.", "image_size": TILE,
                         "max_length": 200, "no_repeat_ngram_size": 35, "ngram_window": 128,
                         "temperature": 0.0}
    rep["layout"] = {
        "num_pages": len(pages),
        "per_page": {"resized_to": [TILE, TILE], "tokens": per_tokens,
                     "grid_rows_x_cols": [nq, nq], "feature_rows": (nq + 1) * nq,
                     "view_seperator_rows": 1, "rows_per_page": (nq + 1) * nq + 1},
        "inter_page": "every page carries its own trailing image_token_id token; the rows of page "
                      "i end with the view_seperator row, page i+1 follows in the same image run",
        "total_image_tokens": per_tokens * len(pages),
        "total_rows": ((nq + 1) * nq + 1) * len(pages),
    }
    assert rep["layout"]["total_image_tokens"] == box["seq_mask_true"]
    json.dump(rep, open(os.path.join(out, "multi_document_screenshot.json"), "w"), indent=1,
              ensure_ascii=False)
    open(os.path.join(out, "multi_document_screenshot.txt"), "w").write(rep["decoded_raw"])
    print("input_len", rep["prompt_token_ids_len"], "mask_true", box["seq_mask_true"])
    print("first12", rep["first_12_generated_ids"])
    print(txt[:400])
    print("wrote", os.path.join(out, "multi_document_screenshot.json"))


if __name__ == "__main__":
    cmd = sys.argv[1]
    {"tile": cmd_tile, "e2e": cmd_e2e, "multi": cmd_multi}[cmd]()
