#!/usr/bin/env python3
"""CPU-only differential test against a Spark checkpoint's modeling_spark.py.

Requires torch, transformers and safetensors. It creates a tiny two-layer random checkpoint,
converts and serves it on the host, then compares prefill and cached decode with the HF reference.
No model weights are copied from the reference checkpoint, and GPUs are hidden from both processes.
ROCm may still open device nodes when probing; use a GPU-denying device cgroup on shared machines.
"""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import time
import types
import urllib.error
import urllib.request

# Set before importing torch or launching a Radiance child, even on a HIP-enabled build.
os.environ.update(ROCR_VISIBLE_DEVICES="GPU-no-such-device", HIP_VISIBLE_DEVICES="",
                  CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="2", OMP_WAIT_POLICY="passive")


def load_reference(path):
    pkg = types.ModuleType("spark25_reference")
    pkg.__path__ = [str(path)]
    sys.modules[pkg.__name__] = pkg
    for name in ("configuration_spark", "modeling_spark"):
        spec = importlib.util.spec_from_file_location(pkg.__name__ + "." + name, path / (name + ".py"))
        if spec is None or spec.loader is None:
            raise RuntimeError("cannot load " + name)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = mod
        spec.loader.exec_module(mod)
    return sys.modules[pkg.__name__ + ".configuration_spark"], sys.modules[pkg.__name__ + ".modeling_spark"]


def write_tokenizer(path):
    def symbol(b):
        def printable(c):
            return 33 <= c <= 126 or 161 <= c <= 172 or 174 <= c <= 255
        return chr(b if printable(b) else 256 + sum(not printable(c) for c in range(b)))
    vocab = {symbol(b): b for b in range(256)}
    added = []
    for i, s in enumerate(("<|endoftext|>", "<|im_start|>", "<|im_end|>")):
        vocab[s] = 256 + i
        added.append(dict(id=256 + i, content=s, single_word=False, lstrip=False,
                          rstrip=False, normalized=False, special=True))
    bytelevel = dict(type="ByteLevel", add_prefix_space=False, trim_offsets=False, use_regex=True)
    tok = dict(version="1.0", added_tokens=added, normalizer=None, pre_tokenizer=bytelevel,
               post_processor=None, decoder=bytelevel,
               model=dict(type="BPE", dropout=None, unk_token=None, byte_fallback=False,
                          vocab=vocab, merges=[]))
    (path / "tokenizer.json").write_text(json.dumps(tok, ensure_ascii=False))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--reference", required=True, type=Path, help="directory with modeling_spark.py")
    ap.add_argument("--build", required=True, type=Path, help="Radiance build directory")
    args = ap.parse_args()
    import torch
    import transformers
    from safetensors.torch import save_file

    torch.set_num_threads(2)
    cfgmod, modelmod = load_reference(args.reference.resolve())
    # The checkpoint targets Transformers 4.57; v5 changed this declaration from a list to a map.
    if int(transformers.__version__.split(".")[0]) >= 5:
        modelmod.Spark2_5ForCausalLM._tied_weights_keys = {"lm_head.weight": "model.embedding.weight"}
    config = cfgmod.Spark2_5Config(
        vocab_size=259, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, head_dim=8, hidden_act="gelu",
        max_position_embeddings=128, rms_norm_eps=1e-6, tie_word_embeddings=True,
        headwise_attn_output_gate=True, gate_attn_act_mode="sigmoid", sliding_window=4,
        layer_types=["sliding_attention", "full_attention"],
        rope_parameters={"sliding_attention": {"rope_theta": 10000.0, "partial_rotary_factor": 1.0},
                         "full_attention": {"rope_theta": 5000000.0, "partial_rotary_factor": 0.5}},
        bos_token_id=256, eos_token_id=256, pad_token_id=256)
    torch.manual_seed(19)
    model = modelmod.Spark2_5ForCausalLM(config).eval().to(torch.bfloat16)
    with torch.no_grad():
        for name, w in model.named_parameters():
            if name.endswith("layernorm.weight") or name == "model.norm.weight":
                w.copy_((1 + 0.08 * torch.randn(w.shape)).to(w.dtype))
            elif name == "model.embedding.weight":
                w.copy_((0.4 * torch.randn(w.shape)).to(w.dtype))
            elif name.endswith("g_proj.weight"):
                w.copy_((0.2 * torch.randn(w.shape)).to(w.dtype))
            else:
                w.copy_((0.16 * torch.randn(w.shape)).to(w.dtype))

    def reference(ids):
        n = len(ids)
        q, k = torch.arange(n)[:, None], torch.arange(n)[None, :]
        full = torch.where(k <= q, 0.0, float("-inf"))[None, None, :, :]
        local = torch.where((k <= q) & (k > q - config.sliding_window), 0.0,
                            float("-inf"))[None, None, :, :]
        # Explicit masks also bridge the Transformers 4.57/5.x mask-helper signature change.
        with torch.inference_mode():
            logits = model(torch.tensor([ids]), attention_mask={"full_attention": full,
                           "sliding_attention": local}, use_cache=False).logits[0, -1].float()
        return torch.log_softmax(logits, dim=-1)

    build = args.build.resolve()
    env = os.environ.copy()
    env.update(RADIANCE_HOME=str(build / "radiance_home"), RADIANCE_HOST_VRAM_MIB="512")
    with tempfile.TemporaryDirectory(prefix="spark25-reference-") as tmp:
        root = Path(tmp)
        ckpt = root / "checkpoint"
        ckpt.mkdir()
        config_json = config.to_dict()
        config_json.update(model_type="spark2_5", architectures=["Spark2_5ForCausalLM"])
        (ckpt / "config.json").write_text(json.dumps(config_json, indent=2))
        state = {k: v.contiguous().clone() for k, v in model.state_dict().items() if k != "lm_head.weight"}
        save_file(state, ckpt / "model.safetensors")
        write_tokenizer(ckpt)
        rad = root / "model.rad"
        subprocess.run([build / "bin/rad-convert", ckpt, "-o", rad, "--max-ctx", "64",
                        "--max-tok", "32"], env=env, check=True)
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        with (root / "server.log").open("w+") as log:
            proc = subprocess.Popen([build / "bin/radiance", "--model", rad, "--host", "127.0.0.1",
                "--port", str(port), "--kernels", "libref", "--debug-accept-reference-kernels",
                "--max-model-len", "64", "--max-num-batched-tokens", "32", "--max-num-seqs", "1"],
                env=env, stdout=log, stderr=subprocess.STDOUT)
            try:
                url = f"http://127.0.0.1:{port}"
                for _ in range(200):
                    if proc.poll() is not None:
                        raise RuntimeError("Radiance exited before listening")
                    try:
                        urllib.request.urlopen(url + "/health", timeout=1).close()
                        break
                    except (OSError, urllib.error.URLError):
                        time.sleep(0.1)
                else:
                    raise RuntimeError("Radiance did not start")
                # The longer prompts exceed the local window, and the last also crosses the
                # 32-token prefill chunk boundary. Compare cached decode to the same HF continuation.
                for prompt in ("a", "hello", "longer prompt", "the quick brown fox jumps over the lazy dog twice"):
                    request = dict(prompt=prompt, max_tokens=5, temperature=0, logprobs=10,
                                   ignore_eos=True)
                    data = json.dumps(request).encode()
                    req = urllib.request.Request(url + "/v1/completions", data,
                                                 {"Content-Type": "application/json"})
                    result = json.load(urllib.request.urlopen(req, timeout=30))
                    rows = result["choices"][0]["logprobs"]["content"]
                    if len(rows) != 5:
                        raise AssertionError(f"{prompt!r}: expected 5 decode rows, got {len(rows)}")
                    ids = list(prompt.encode())
                    max_error = 0.0
                    for row in rows:
                        expected = reference(ids)
                        if row["id"] != int(expected.argmax()):
                            raise AssertionError(f"{prompt!r}: token {row['id']} != reference {int(expected.argmax())}")
                        for candidate in row["top_logprobs"]:
                            max_error = max(max_error,
                                abs(candidate["logprob"] - float(expected[candidate["id"]])))
                        ids.append(row["id"])
                    # BF16 logits / attention reduction order differ slightly from CPU torch.
                    if max_error >= 0.15:
                        raise AssertionError(f"{prompt!r}: max top-10 log-prob error {max_error:.5f}")
                    print(f"{prompt!r}: 5/5 top-1, max top-10 log-prob error {max_error:.5f}")
            except Exception:
                log.flush()
                log.seek(0)
                print(log.read(), file=sys.stderr)
                raise
            finally:
                proc.terminate()
                try:
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait()


if __name__ == "__main__":
    main()
