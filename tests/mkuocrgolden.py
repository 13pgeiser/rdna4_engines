"""mkuocrgolden.py MODELDIR OUT.h [REPORT] -- goldens for the Unlimited-OCR processor in
tests/mm_test.cpp (P3 of aijuus/plans/UNLIMITED-OCR-PLUGIN.md), every number from the checkpoint's
own remote code as MODELDIR ships it:

  - the tile grid, from modeling_unlimitedocr.dynamic_preprocess (find_closest_aspect_ratio over
    its sorted set of ratios), for a sweep of sizes including the exact ties its set order resolves;
  - the placeholders that replace a picture's <image>, from infer() (gundam mode) and infer_multi()
    (base mode: the whole page, and the 640-pixel page the reference's default asks for);
  - the pixels: the global view (ImageOps.pad) and the local one (the picture resized to its tile
    grid and cut), transformed by the model's BasicImageTransform and rounded to bf16, in the patch
    order the radiance processor stages them (Item::patch_raster: row by row, each patch
    (channel, y, x)). The segment plan is the radiance Item's (see the header of
    Processor::image_unlimitedocr): a line of tiles per tile row, the global view's grid, and the
    view separator's single row -- each with `downsample` filler patch columns for the newline
    column the reference's layout puts after every line of features.

REPORT, when given, gets one line per picture and mode: the grid, the placeholder count and
whether the segment plan's counts match, for the gate's log.
"""
import sys, math, json
import numpy as np, torch
from PIL import Image, ImageOps
from torchvision import transforms

model_dir, out = sys.argv[1], sys.argv[2]
report = open(sys.argv[3], "w") if len(sys.argv) > 3 else None

# ---- the checkpoint's two configs, and its remote module as a package (its relative imports need
# one)
import types, importlib
pkg = types.ModuleType("uocr")
pkg.__path__ = [model_dir]
sys.modules["uocr"] = pkg
M = importlib.import_module("uocr.modeling_unlimitedocr")
pre = json.load(open(model_dir + "/preprocessor_config.json"))
proc = json.load(open(model_dir + "/processor_config.json"))

PATCH = int(proc["patch_size"])
DS = int(proc["downsample_ratio"])
BASE = int(pre["size"]["height"])
TILE = int(pre["tile_size"])
MIN_PATCHES = int(pre["min_patches"])
MAX_PATCHES = int(pre["max_patches"])
CROP = bool(pre["crop_to_patches"])
MEAN = tuple(float(x) for x in pre["image_mean"])
STD = tuple(float(x) for x in pre["image_std"])
assert proc["image_token"] == "<image>" and proc["normalize"]
assert [int(x) for x in pre["background_color"]] == [127, 127, 127]
assert abs(MEAN[0] - 0.5) < 1e-9 and abs(STD[0] - 0.5) < 1e-9

from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=False)
IMAGE_TOKEN = tok.convert_tokens_to_ids("<image>")
assert IMAGE_TOKEN == 128815, IMAGE_TOKEN

# the model's own transform: ToTensor then Normalize(0.5, 0.5), in f32, rounded to bf16
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize(MEAN, STD)])


def bf16_patches(im, grids):
    """The patches of `im`, in Item::patch_raster order: (channel, y, x) per patch, row by row,
    bf16 bit patterns. `grids` is (gh, gw): the patch grid the tensor is cut into."""
    t = transform(im).to(torch.bfloat16)
    ch, h, w = t.shape
    gh, gw = grids
    assert h == gh * PATCH and w == gw * PATCH, (h, w, gh, gw)
    a = t.reshape(ch, gh, PATCH, gw, PATCH).permute(1, 3, 0, 2, 4).contiguous()
    return a.view(torch.int16).numpy().astype(np.uint16)


def fnv(arr):
    b = arr.tobytes()
    h = 0xcbf29ce484222325
    mul = 0x100000001b3
    mask = (1 << 64) - 1
    for v in b:
        h = ((h ^ v) * mul) & mask
    return h


def expand_gundam(im):
    """modeling_unlimitedocr.infer, verbatim for one picture: the global view at BASE, then the
    local views if the picture is larger than a tile. Returns (placeholders, cols, rows, tiles,
    global view)."""
    w, h = im.size
    global_view = ImageOps.pad(im, (BASE, BASE), color=tuple(int(x * 255) for x in MEAN))
    if w > TILE or h > TILE:
        tiles, ratio = M.dynamic_preprocess(im, MIN_PATCHES, MAX_PATCHES, TILE, False)
        cols, rows = ratio
    else:
        tiles, cols, rows = [], 1, 1
    nq_base = math.ceil((BASE // PATCH) / DS)
    n = (nq_base + 1) * nq_base + 1                       # ([tok] * n + [tok]) * n + [tok]
    if cols * rows > 1:
        nq = math.ceil((TILE // PATCH) / DS)
        n += (nq * cols + 1) * nq * rows                   # ([tok] * nq * cols + [tok]) * nq * rows
    return n, cols, rows, tiles, global_view


def expand_base(im, image_size):
    """modeling_unlimitedocr.infer_multi, verbatim for one page: the page squared when the side is
    a tile or less, then padded to image_size."""
    w, h = im.size
    if image_size <= TILE:
        im = im.resize((image_size, image_size))
    global_view = ImageOps.pad(im, (image_size, image_size),
                              color=tuple(int(x * 255) for x in MEAN))
    nq = math.ceil((image_size // PATCH) / DS)
    n = (nq + 1) * nq + 1
    return n, 1, 1, [], global_view


def plan_segments(cols, rows, base_grid):
    """The Item's segment plan: one line of tiles per tile row (split so a pass holds it; the last
    chunk carries the newline column), the global view's grid, the separator. (gh, gw, real)."""
    q = TILE // PATCH
    segs = []
    if cols * rows > 1:
        c_max = (16384 // q - 4) // q                    # the server's pass, --mm-max-patches
        c_max = max(1, c_max)
        for r in range(rows):
            c0 = 0
            while c0 < cols:
                n = min(c_max, cols - c0)
                last = (c0 + n) == cols
                segs.append((q, q * n + (4 if last else 0), q * n))
                c0 += n
    segs.append((base_grid, base_grid + 4, base_grid))
    segs.append((DS, DS, 0))
    return segs


def picture(f, h, w):
    y, x, c = np.meshgrid(np.arange(h, dtype=np.uint64), np.arange(w, dtype=np.uint64),
                          np.arange(3, dtype=np.uint64), indexing="ij")
    with np.errstate(over="ignore"):
        z = (np.uint64(f) * np.uint64(0x9E3779B97F4A7C15) + y * np.uint64(0xBF58476D1CE4E5B9) +
             x * np.uint64(0x94D049BB133111EB) + c * np.uint64(0x2545F4914F6CDD1D) + np.uint64(12345))
        z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        z = z ^ (z >> np.uint64(31))
    return Image.fromarray((z & np.uint64(255)).astype(np.uint8))


def names():
    return ["kUocrPictures", "kUocrSegs", "kUocrGrid"]


L = ["/* uocr_golden.h -- GENERATED by mkuocrgolden.py from the checkpoint's remote code; do not "
     "edit. */",
     "#pragma once", "#include <cstdint>", "",
     f"static const int32_t kUocrPatch = {PATCH}, kUocrDownsample = {DS};",
     f"static const int32_t kUocrBaseSize = {BASE}, kUocrTile = {TILE};",
     f"static const int32_t kUocrMinTiles = {MIN_PATCHES}, kUocrMaxTiles = {MAX_PATCHES};",
     f"static const bool kUocrCropMode = {'true' if CROP else 'false'};",
     f"static const int32_t kUocrImageToken = {IMAGE_TOKEN};", ""]

# ---- the tile grid, for sizes that cover every shape and the exact ties of dynamic_preprocess
sizes = [(1, 1), (40, 30), (100, 100), (300, 500), (500, 300), (640, 640), (641, 641), (500, 700),
         (700, 500), (800, 1000), (1000, 800), (1024, 1024), (1280, 1024), (1024, 1280), (900, 1400),
         (1400, 900), (2000, 600), (600, 2000), (2400, 400), (400, 2400), (1080, 1920), (1920, 1080),
         (515, 1031), (1031, 515), (100, 5000), (5000, 100), (3000, 3000), (720, 1280), (800, 640),
         (640, 800), (960, 720), (720, 960), (1600, 1200), (1200, 1600), (33, 4096), (4096, 33)]
L.append("struct GoldUocrGrid { int64_t h, w; int32_t cols, rows; };")
L.append("static const GoldUocrGrid kUocrGrid[] = {")
for h, w in sizes:
    im = Image.new("RGB", (w, h))
    if w > TILE or h > TILE:
        _, ratio = M.dynamic_preprocess(im, MIN_PATCHES, MAX_PATCHES, TILE, False)
        cols, rows = ratio
    else:
        cols, rows = 1, 1
    L.append(f"    {{ {h}, {w}, {cols}, {rows} }},")
L.append("};")
L.append("")

# ---- whole pictures: gundam (1), base at 1024 (2) and base at 640 (3), and a two-page pair
pics = [(0, 300, 500, 1), (1, 900, 1400, 1), (2, 2400, 400, 1), (3, 500, 700, 1), (4, 64, 48, 1),
        (5, 800, 1000, 1), (0, 300, 500, 2), (1, 900, 1400, 2), (2, 2400, 400, 2), (3, 500, 700, 2),
        (4, 64, 48, 2), (5, 800, 1000, 2), (1, 900, 1400, 3), (3, 500, 700, 3)]
segs, rows, seg0 = [], [], 0
L.append("struct GoldUocrPic { int32_t f, h, w, mode; int32_t cols, rows, n_ids; int32_t n_seg, "
         "seg0; uint64_t global_fnv, local_fnv; };")
for f, h, w, mode in pics:
    im = picture(f, h, w)
    if mode == 1:
        n, cols, rows_, tiles, gv = expand_gundam(im)
        base_grid = BASE // PATCH
        g_fnv = fnv(bf16_patches(gv, (base_grid, base_grid)))
        if cols * rows_ > 1:
            big = im.resize((TILE * cols, TILE * rows_), Image.Resampling.BICUBIC)
            l_fnv = fnv(bf16_patches(big, (TILE // PATCH * rows_, TILE // PATCH * cols)))
        else:
            l_fnv = 0
    else:
        size = 1024 if mode == 2 else 640
        n, cols, rows_, tiles, gv = expand_base(im, size)
        base_grid = size // PATCH
        g_fnv = fnv(bf16_patches(gv, (base_grid, base_grid)))
        l_fnv = 0
    pl = plan_segments(cols, rows_, base_grid)
    # every segment's row count is its patches / downsample^2, and the sum is the reference's
    total = sum((gh // DS) * (gw // DS) for gh, gw, _ in pl)
    assert total == n, (h, w, mode, total, n)
    rows.append(f"    {{ {f}, {h}, {w}, {mode}, {cols}, {rows_}, {n}, {len(pl)}, {seg0}, "
                f"0x{g_fnv:016x}ull, 0x{l_fnv:016x}ull }},")
    for gh, gw, real in pl:
        segs.append((gh, gw, real))
    seg0 += len(pl)
    if report:
        report.write(f"picture {f} {w}x{h} mode {'gundam' if mode == 1 else 'base@%d' % size}: "
                     f"reference grid {cols}x{rows_} ({cols * rows_} tiles), placeholders {n}; "
                     f"item plan {len(pl)} segments, {total} rows: MATCH\n")
        if cols * rows_ > 1:
            report.write(f"    tiles row-major {cols} wide x {rows_} tall, one line of "
                         f"{10 * cols} features + 1 newline per feature row: "
                         f"{(10 * cols + 1) * 10 * rows_} placeholders\n")
L.append("static const GoldUocrPic kUocrPictures[] = {")
L += rows
L.append("};")
L.append("struct GoldUocrSeg { int32_t h, w, real; };")
L.append("static const GoldUocrSeg kUocrSegs[] = {")
for gh, gw, real in segs:
    L.append(f"    {{ {gh}, {gw}, {real} }},")
L.append("};")
L.append("")

# ---- the ratios' set order: the reference's tie resolution, area by area
target = set((i, j) for n in range(MIN_PATCHES, MAX_PATCHES + 1) for i in range(1, n + 1)
             for j in range(1, n + 1) if i * j <= MAX_PATCHES and i * j >= MIN_PATCHES)
ordered = sorted(target, key=lambda t: t[0] * t[1])
tall_last = []
for (i, j) in sorted(target):
    if i < j and (j, i) in target and ordered.index((i, j)) > ordered.index((j, i)):
        tall_last.append((i, j))
# ---- Pillow's bicubic resize: both pass orders (its Resample.c runs the vertical pass first when
# the height shrinks by more than twice what the width does), the one-axis forms and an exact tie.
L.append("/* Pillow's Image.resize(BICUBIC) of a splitmix picture: its two-pass order is part of the")
L.append(" * function (the intermediate rounds to eight bits). */")
L.append("struct GoldUocrResize { int32_t sh, sw, dh, dw; uint64_t fnv; };")
L.append("static const GoldUocrResize kUocrResize[] = {")
resize_cases = [(0, 2400, 400, 1024, 171), (0, 300, 300, 90, 100), (0, 400, 300, 171, 1024),
                (0, 300, 400, 1024, 171), (0, 400, 1000, 200, 500), (0, 1000, 400, 200, 300),
                (0, 700, 500, 650, 470), (0, 500, 700, 640, 650), (0, 320, 480, 320, 480),
                (0, 240, 320, 160, 320), (0, 240, 320, 240, 200), (0, 64, 48, 1024, 1024)]
for f, sh, sw, dh, dw in resize_cases:
    im = picture(f, sh, sw)
    r = np.asarray(im.resize((dw, dh), Image.Resampling.BICUBIC))
    L.append(f"    {{ {sh}, {sw}, {dh}, {dw}, 0x{fnv(r):016x}ull }},")
L.append("};")
L.append("")

L.append("/* The transposed pairs whose tall side the reference's set visits last. */")
L.append("struct GoldUocrTie { int32_t c, r; };")
L.append("static const GoldUocrTie kUocrTallLast[] = {")
L.append("    " + " ".join(f"{{{c}, {r}}}," for c, r in tall_last))
L.append("};")
L.append(f"static const int32_t kUocrNTallLast = {len(tall_last)};")
if report:
    # The multi-page form: the reference concatenates the pages' runs at one <image>.
    pair = [picture(0, 500, 300), picture(1, 1400, 900)]
    per = [expand_base(im, 1024)[0] for im in pair]
    report.write(f"multi-page pair (infer_multi, 2 pages at one marker, image_size 1024): "
                 f"{per[0]} + {per[1]} = {sum(per)} placeholders, "
                 f"{'MATCH' if per == [273, 273] else 'MISMATCH'}\n")
open(out, "w").write("\n".join(L) + "\n")
print("wrote", out, len(L), "lines;", len(pics), "pictures,", len(segs), "segments,",
      len(tall_last), "tie pairs")
