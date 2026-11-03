#!/usr/bin/env python3
"""Repair the ssm alpha/beta tables the FLM_Q4NX_Converter emits for qwen35moe
models (Qwen3.5/3.6 35B-A3B geometry) so the qwen3.6-moe NPU2 engine can load
them.

The converter's quantized branch (q4nx/models/qwen35_moe.py) packs
model.layer.N.linear_attn.ssm_{alpha,beta}_proj.weight as I8 and stashes the
dequantized values in a ".bf16.weight" side-entry, transposed ([32, 2048]).
The engine instead expects the MAIN entry to be a plain BF16 [2048, 32] table:
the converter's own bf16 branch saves the main entry column-major ("save
matrix in colum major order instead"), and the reference
Qwen3.6-35B-A3B-NPU2 pack has exactly that. As emitted, the engine reads
131 072 B from a 37 888 B slot and segfaults in qwen3_6_reorder_cpy during
load_weights.

This tool promotes the transposed side-copies to the main entries (BF16,
shape reversed) and drops the side-entries. In-place (tmp + rename),
idempotent (a correct file is left untouched), streaming (16 MiB chunks),
stdlib only.

usage: fix_q4nx_ssm.py FILE [FILE ...]
"""
import json
import os
import struct
import sys
from array import array


def read_header(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(n))


def transpose_bf16(data, r, c):
    """Row-major BF16 [r, c] -> [c, r]; moves 16-bit words only."""
    w = array("H")
    w.frombytes(data)
    assert len(w) == r * c, f"expected {r * c} bf16 words, got {len(w)}"
    out = array("H", bytes(2 * r * c))
    for j in range(r):
        row = j * c
        for i in range(c):
            out[i * r + j] = w[row + i]
    return out.tobytes()


def is_ssm(name):
    return name.endswith(".ssm_alpha_proj.weight") or name.endswith(".ssm_beta_proj.weight")


def fix(path):
    header = read_header(path)
    meta = header.get("__metadata__")
    entries = {k: v for k, v in header.items() if k != "__metadata__"}

    sides = {k for k in entries if ".bf16." in k}
    needs_fix = bool(sides)
    for k in entries:
        if is_ssm(k) and not (entries[k]["dtype"] == "BF16"
                              and list(entries[k]["shape"]) == [2048, 32]):
            needs_fix = True
    if not needs_fix:
        print(f"{path}: already correct, nothing to do")
        return

    plan = []  # (name, dtype, shape, size, source-entry-or-None)
    trans = {}  # main name -> repaired bytes
    with open(path, "rb") as f:
        for k, v in entries.items():
            if k in sides:
                continue
            if is_ssm(k):
                side = k.replace(".weight", ".bf16.weight")
                if side in entries:
                    sv = entries[side]
                    if sv["dtype"] != "BF16":
                        sys.exit(f"{path}: {side} is not BF16")
                    r, c = sv["shape"]
                    f.seek(sv["data_offsets"][0])
                    trans[k] = transpose_bf16(
                        f.read(sv["data_offsets"][1] - sv["data_offsets"][0]), r, c)
                    plan.append((k, "BF16", [c, r], len(trans[k]), None))
                elif v["dtype"] == "BF16" and list(v["shape"]) == [2048, 32]:
                    plan.append((k, "BF16", [2048, 32],
                                 v["data_offsets"][1] - v["data_offsets"][0], k))
                else:
                    sys.exit(f"{path}: {k} is packed and has no .bf16. side-copy; "
                             "cannot repair without dequantizing (re-run the converter)")
            else:
                plan.append((k, v["dtype"], list(v["shape"]),
                             v["data_offsets"][1] - v["data_offsets"][0], k))

        # header length depends on the offsets and vice versa: iterate to a fixpoint
        L = 85000
        for _ in range(10):
            off = 8 + L
            d = {"__metadata__": meta} if meta is not None else {}
            for name, dt, sh, sz, _ in plan:
                d[name] = {"dtype": dt, "shape": sh, "data_offsets": [off, off + sz]}
                off += sz
            L = len(json.dumps(d).encode())

        tmp = path + ".ssmfix.tmp"
        with open(tmp, "wb") as out:
            out.write(struct.pack("<Q", L))
            out.write(json.dumps(d).encode())
            for name, dt, sh, sz, src in plan:
                if src is None:
                    out.write(trans[name])
                else:
                    a, b = entries[src]["data_offsets"]
                    f.seek(a)
                    rem = b - a
                    while rem > 0:
                        chunk = f.read(min(rem, 1 << 24))
                        out.write(chunk)
                        rem -= len(chunk)
        os.replace(tmp, path)
    print(f"{path}: rewrote {len(plan)} tensors ({len(sides)} .bf16. side-entries absorbed)")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    for p in sys.argv[1:]:
        fix(p)
