#!/usr/bin/env python3
# krea2_rebalance_calib.py -- the constants of arch/krea2's rebalance presets (`rebalance`:
# "balanced", "detail", "subtle"): each preset's per-layer weights times s, the factor
# comfyui-krea2-conditioning's renormalize applies at multiplier 1.
#
#   krea2_rebalance_calib.py <Krea 2 pipeline directory> [out.json]
#
# The node scales tap t by w_t and renormalises so the taps keep their total energy:
# s = sqrt(sum_t m_t / sum_t w_t^2 m_t), m_t the mean square of tap t over the prompt's rows. The
# plugin cannot measure m_t per request without a reduction across the encoder's rows, so it takes
# s as a constant: m_t averaged over the twenty prompts below, each run through the release's text
# encoder in arch/krea2's template with its system prefix dropped, as the plugin keeps the rows.
# The script prints, per preset, that s, the per-prompt range of s, and the twelve gains, and for
# the default gains (`rebalance: true`) their factor on the taps' RMS.
#
# Measured on krea/Krea-2-Turbo's text_encoder (bf16, CPU, about a minute): balanced 0.44109
# (per prompt 0.43449 to 0.44879), detail 0.35702, subtle 0.84759; the default gains are 9.07x the
# taps' RMS. Needs torch and transformers.
import json
import sys

import torch
from transformers import AutoModel, AutoTokenizer

# Tap t is the hidden state after layer TAPS[t] - 1 (hidden_states[TAPS[t]]), arch/krea2's kTaps.
TAPS = [2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 32, 35]
SYS = ("Describe the image by detailing the color, shape, size, texture, quantity, text, spatial "
       "relationships of the objects and background:")
DROP = f"<|im_start|>system\n{SYS}<|im_end|>\n<|im_start|>user\n"
PRESETS = {
    "balanced": [1, 1, 1, 1, 1, 1, 1, 2.5, 5, 1.1, 4, 1],
    "detail": [0.8, 0.8, 0.9, 0.9, 1, 1, 1.2, 3, 6, 1.5, 5, 1.2],
    "subtle": [1, 1, 1, 1, 1, 1, 1, 1.5, 2, 1, 1.5, 1],
}
# ConditioningKrea2Rebalance's defaults: multiplier 4 times these, no renormalisation.
DEFAULT = [4 * w for w in PRESETS["balanced"]]
PROMPTS = [
    "a cat",
    "portrait of an old fisherman with a weathered face, grey beard, wool cap, soft window light",
    "a young woman with freckles and red hair laughing in a sunflower field, golden hour, 35mm photo",
    "close-up of a man's face, sharp focus on the eyes, studio lighting, black background",
    "a mountain lake at dawn with mist over the water and pine trees on the shore",
    "a busy street market in Marrakech, colorful spices, people in motion, afternoon sun",
    "a neon sign that reads \"OPEN 24 HOURS\" above the door of a small diner at night, rain on the street",
    "a hand-written note on yellow paper saying 'Buy milk, call mom, feed the dog'",
    "an astronaut riding a horse on the moon, earth in the sky, cinematic",
    "three red apples and two green pears on a wooden table, top-down view",
    "a cozy living room with a fireplace, a grey sofa, a knitted blanket and a sleeping dog",
    "architectural photo of a brutalist concrete building against a blue sky, wide angle",
    "a watercolor painting of a lighthouse on a cliff during a storm",
    "anime style illustration of a girl with a sword standing on a rooftop at sunset",
    "macro photo of a dewdrop on a spider web, bokeh background",
    "a group of five friends sitting around a campfire at night, smiling, warm light on their faces",
    "a vintage red car parked in front of a gas station in the desert, 1960s, film grain",
    "a bowl of ramen with a soft-boiled egg, chopsticks resting on the side, steam rising",
    "an elderly woman knitting in a rocking chair next to a window, her cat on her lap, overcast daylight, "
    "muted colors, documentary photography, shallow depth of field, the room is full of old books and plants",
    "a futuristic city skyline with flying cars, holographic billboards in Japanese and English, dusk",
]


def main():
    if len(sys.argv) not in (2, 3):
        sys.exit(__doc__ or "usage: krea2_rebalance_calib.py <pipeline directory> [out.json]")
    pipe = sys.argv[1]
    tok = AutoTokenizer.from_pretrained(f"{pipe}/tokenizer")
    model = AutoModel.from_pretrained(f"{pipe}/text_encoder", dtype=torch.bfloat16).eval()
    lm = getattr(model, "language_model", model)
    n_drop = len(tok(DROP, add_special_tokens=False).input_ids)
    rows = []
    with torch.no_grad():
        for p in PROMPTS:
            text = DROP + p + "<|im_end|>\n<|im_start|>assistant\n"
            ids = tok(text, add_special_tokens=False, return_tensors="pt").input_ids
            hs = lm(input_ids=ids, output_hidden_states=True).hidden_states
            rows.append([hs[i][0, n_drop:].float().pow(2).mean().item() for i in TAPS])
    M = torch.tensor(rows, dtype=torch.float64)   # (prompts, taps)
    m = M.mean(0)
    out = {"taps": TAPS, "m_mean": m.tolist(), "presets": {}}
    for name, w in PRESETS.items():
        w2 = torch.tensor(w, dtype=torch.float64) ** 2
        per = (M.sum(1) / (M * w2).sum(1)).sqrt()
        s = float((m.sum() / (m * w2).sum()).sqrt())
        out["presets"][name] = {"s": s, "s_min": float(per.min()), "s_max": float(per.max()),
                                "gains": [x * s for x in w]}
    d2 = torch.tensor(DEFAULT, dtype=torch.float64) ** 2
    out["default_rms_gain"] = float(((m * d2).sum() / m.sum()).sqrt())
    for name, p in out["presets"].items():
        print(f"{name:9s} s {p['s']:.5f} ({p['s_min']:.5f} to {p['s_max']:.5f})  gains "
              + ", ".join(f"{g:.5g}" for g in p["gains"]))
    print(f"default gains: {out['default_rms_gain']:.2f}x the taps' RMS")
    if len(sys.argv) == 3:
        with open(sys.argv[2], "w") as f:
            json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
