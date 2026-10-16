"""qwen3_asr_reference.py MODEL_DIR WAV OUT_DIR [--language English] [--dtype bfloat16]

Transformers' own Qwen3-ASR, run on the CPU, as the reference arch/qwen3asr_bf16 is checked against.
It writes, for one 16 kHz mono WAV:

  OUT_DIR/features.f32     the log-mel features [frames, 128] (Qwen3ASRFeatureExtractor), real
                           frames only, frame-major -- the layout core/mm stages
  OUT_DIR/audio.bf16       the projected audio rows [tokens, hidden] the decoder's placeholders take
  OUT_DIR/result.json      shapes, the prompt token ids, the greedy reply's token ids and its text

RADIANCE_DEBUG_MM_DUMP writes the engine's own encoder rows as raw bf16, so the two files compare
byte for byte in layout. Every number here comes from transformers, not from this script."""
import argparse, json, os, wave

import numpy as np
import torch
from transformers import AutoProcessor, Qwen3ASRForConditionalGeneration


def read_wav(path):
    with wave.open(path, "rb") as w:
        if w.getsampwidth() != 2:
            raise SystemExit(f"{path}: expected 16-bit PCM")
        rate, ch, n = w.getframerate(), w.getnchannels(), w.getnframes()
        x = np.frombuffer(w.readframes(n), dtype="<i2").astype(np.float32) / 32768.0
    if ch > 1:
        x = x.reshape(-1, ch).mean(axis=1)
    if rate != 16000:
        raise SystemExit(f"{path}: expected 16 kHz, got {rate}")
    return x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("wav")
    ap.add_argument("out")
    ap.add_argument("--language", default=None, help="a name the model knows (English), or none")
    ap.add_argument("--context", default="", help="system-prompt context text")
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float32"])
    ap.add_argument("--max-new-tokens", type=int, default=256)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    torch.manual_seed(0)

    proc = AutoProcessor.from_pretrained(a.model)
    model = Qwen3ASRForConditionalGeneration.from_pretrained(a.model, dtype=getattr(torch, a.dtype))
    model.eval()

    audio = read_wav(a.wav)
    conv = [{"role": "system", "content": [{"type": "text", "text": a.context}]},
            {"role": "user", "content": [{"type": "audio", "audio": audio}]}]
    inputs = proc.apply_chat_template(conv, tokenize=True, add_generation_prompt=True,
                                      return_dict=True, return_tensors="pt", sampling_rate=16000)
    if a.language:
        prefill = proc.tokenizer(f"language {a.language}<asr_text>", add_special_tokens=False,
                                 return_tensors="pt").input_ids
        inputs["input_ids"] = torch.cat([inputs["input_ids"], prefill], dim=1)
        inputs["attention_mask"] = torch.ones_like(inputs["input_ids"])

    inputs["input_features"] = inputs["input_features"].to(model.dtype)
    feats = inputs["input_features"][0]                       # [128, padded frames]
    fmask = inputs["input_features_mask"][0].bool()
    real = feats[:, fmask].T.contiguous().float().numpy()     # [frames, 128]
    real.tofile(os.path.join(a.out, "features.f32"))

    with torch.no_grad():
        rows = model.get_audio_features(inputs["input_features"],
                                        inputs["input_features_mask"]).pooler_output
        rows.to(torch.bfloat16).view(torch.int16).numpy().tofile(os.path.join(a.out, "audio.bf16"))
        out = model.generate(**{k: v for k, v in inputs.items()}, max_new_tokens=a.max_new_tokens,
                             do_sample=False)
    prompt = inputs["input_ids"][0].tolist()
    reply = out[0, len(prompt):].tolist()
    text = proc.tokenizer.decode(reply, skip_special_tokens=True)
    res = {"frames": int(real.shape[0]), "audio_rows": int(rows.shape[0]), "hidden": int(rows.shape[1]),
           "prompt_ids": prompt, "reply_ids": reply, "text": text, "dtype": a.dtype,
           "language": a.language}
    with open(os.path.join(a.out, "result.json"), "w") as f:
        json.dump(res, f, indent=1)
    print(json.dumps({k: v for k, v in res.items() if k not in ("prompt_ids",)}, indent=1))


if __name__ == "__main__":
    main()
