#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Kaden Schutt
# hipfire — see LICENSE and NOTICE in the project root.
"""CPU-only CLI tests for scripts/perf_config_receipt.py.

Every test drives the script through ``subprocess`` with a temporary startup log
and receipt, so only the CLI contract is exercised:

    perf_config_receipt.py --log PATH --arch gfx1151|gfx1201 --receipt PATH [--model PATH]

It writes ``startup_config`` and ``product_default_check`` (``status`` in
match|mismatch|unknown plus ``reasons``) into the receipt and preserves every
other field. Log lines are copied from real Flash-Next daemon logs referenced by
product-config-audit.md (2026-10-16).
"""
import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "perf_config_receipt.py"

BASE_MODEL_SHA = "8b15b6fede7d7c5bfed0db4720a8295bedda51bc93e545fa242bd50d0f200972"
HOTSET_SHA = "7f6760e11a86e4ffe0b05c050ce64e6eb772ab748b6b11c8b2e6dbc566e57086"

# --- realistic startup lines -------------------------------------------------

PLE_Q4_HALO = (
    "  qwen4 ple rows: q4 from /home/kaden/pm-wave/fnds/models-ov/"
    "qwen3.8-flash-next-gptq3.mq4.ple-q4 (Q4LM, rank order, 28.8 GB)"
)
PLE_Q4_R9700 = (
    "  qwen4 ple rows: q4 from /home/kaden/qcal/models/"
    "qwen3.8-flash-next-gptq3.mq4.ple-q4 (Q4LM, rank order, 28.8 GB)"
)
PLE_Q8_EMBEDDED = "  qwen4 ple rows: q8 from embedded model file (Q8_0, row order, 54.4 GB)"
PLE_Q4_MISSING_NOTE = (
    "  qwen4 ple rows: q4 PLE sidecar not found beside "
    "/home/kaden/.hipfire/models/qwen3.8-flash-next-gptq3.mq4; "
    "using the embedded q8 rows"
)

OVERLAY_APPLIED = (
    "  qwen4 trunk overlay: applied 108 QT44 tensors from "
    "/home/kaden/pm-wave/fnds/models-ov/qwen3.8-flash-next-gptq3.mq4.trunk-overlay "
    "(default gfx1151-zab-v1; bound to base 125288544792 B, index sha256 8703bb1626ff, "
    "file sha256 8b15b6fede7d; 449280000 -> 305510400 bytes)"
)
OVERLAY_NOT_APPLIED_1201 = (
    "  qwen4 trunk overlay: /home/kaden/qcal/models/"
    "qwen3.8-flash-next-gptq3.mq4.trunk-overlay present, not applied on gfx1201 "
    "(overlay default gfx1151-zab-v1 is gfx1151 only; HIPFIRE_QWEN4_TRUNK_OVERLAY=1 applies it)"
)

# Synthetic log line: the script only parses it. `cal_sha256` is an opaque sample digest, not the hash of
# the current built-in record, so it is deliberately not regenerated when the calibration key changes.
NPU_AUTO_HALO = (
    'npu-trunk: mode=auto contract=bf16 families=qkv,zab,q,kvidx '
    'cal_key={"device":{"arch":"gfx1151","pci_bdf":"0000:c0:00.0","npu_fw_version":"1.1.2.65"},'
    '"build":"hipfire-0.4.2:npu-trunk-bf16-rowmajor-v1","kernel":{"uname_release":"7.3.0-rc7-perfopt+"},'
    '"model_trunk_index_sha256":"098bd75fd3a6a5174d06033f2b3ec7febeb7777e7eba4696b64cecca5e3418b6",'
    '"epilogue":"bf16","trunk_route":"overlay:gfx1151-zab-v1"} '
    "cal_sha256=b3cbbcba42068ed2f9b6bd43f64876c95ccf08cbdf107903c31de4dfe7015bfd"
)
NPU_ROUTE_GPU = [
    "npu-trunk: rows=1131 plan=[gpu 0..1131] predicted=+0.0%",
    "npu-trunk: rows=763 plan=[gpu 0..763] predicted=+0.0%",
    "npu-trunk: rows=243 plan=[gpu 0..243] predicted=+0.0%",
    "npu-trunk: rows=37 plan=[gpu 0..37] predicted=+0.0%",
]

HOTSET_APPLIED = (
    "  qwen4 expert hot set: /home/kaden/qcal/models/qwen3.8-flash-next-gptq3.mq4.expert-hotset "
    f"sha256 {HOTSET_SHA}, budget 16136 MiB, 6479 experts in VRAM (16134 MiB), "
    "18097 host-mapped (45065 MiB), generation 1, digest 40554bac16dbca87, "
    "bound to base 125288544792 B, index sha256 8703bb1626ff"
)
HOTSET_BUDGET = (
    "  qwen4 expert hot set budget: 32550 MiB free - 5115 MiB non-expert weights - "
    "8816 MiB reserve (8816 MiB = max(8816 MiB historical floor, 7151 MiB census [...])) "
    "- 417792 bytes tier tables - 512 MiB safety margin - 1969 MiB long-prefill stage buffers = 16136 MiB"
)
HOTSET_IDENTITY = (
    "  qwen4 expert hot set identity: aggregate-MQ4 structure validated (no full artifact digest "
    f"held by this loader; artifact not rehashed); model_sha256 {BASE_MODEL_SHA}, "
    "counts_sha256 97b6406be3cdd3281ed6799d2621ad062d2d5481af0e36191953ba842245de4b"
)
HOTSET_DISABLED = (
    "  qwen4 expert hot set: /home/kaden/qcal/release-0.4.3/lever1-r9700/model/"
    "qwen3.8-flash-next-gptq3.mq4.expert-hotset present, not applied "
    "(HIPFIRE_QWEN4_EXPERT_HOTSET=0); whole-layer placement"
)
WHOLE_LAYER_PLACEMENT = [
    "  qwen4 expert placement: HIPFIRE_QWEN4_EXPERT_VRAM_LAYERS unset, auto (62475 MiB routed experts "
    "+ 5115 MiB non-expert weights + 9328 MiB reserve exceed 32550 MiB free VRAM)",
    "  qwen4 routed experts: layers 0..14 in VRAM, 70 expert tensors (43.6 GiB) in pinned host RAM "
    "(MemAvailable 48.8 GiB + 50.2 GiB estimated in TTM's page pool)",
]

PREAMBLE = [
    "[gpu-lock] directory=/home/kaden/qcal/release-0.4.3/mtpdepth-r9700/runs/hd-grd-p0/new@c/gq3/lockdir",
    "  pre-compiled kernels: /home/kaden/qcal/release-0.4.3/mtpdepth-r9700/kcache/gfx1201",
    "  qwen4 native MTP speculator enabled (K=3)",
]


def halo_log(ple, *, overlay=True, npu=True):
    lines = list(PREAMBLE)
    lines.append(ple)
    if overlay:
        lines.append(OVERLAY_APPLIED)
    if npu:
        lines.append(NPU_AUTO_HALO)
        lines.extend(NPU_ROUTE_GPU)
    return "\n".join(lines) + "\n"


def r9700_log(ple, *, hotset=True, overlay_note=False):
    lines = list(PREAMBLE)
    lines.append(ple)
    if overlay_note:
        lines.append(OVERLAY_NOT_APPLIED_1201)
    if hotset:
        lines += [HOTSET_BUDGET, HOTSET_APPLIED, HOTSET_IDENTITY]
    else:
        lines += WHOLE_LAYER_PLACEMENT
    return "\n".join(lines) + "\n"


class PerfConfigReceiptCliTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.tmp = Path(self._tmp.name)

    # -- helpers ----------------------------------------------------------

    def run_cli(self, log_text, arch, *, receipt=None, model=None, expect_rc=0):
        log = self.tmp / "daemon.log"
        log.write_text(log_text, encoding="utf-8")
        receipt_path = self.tmp / "receipt.json"
        if receipt is not None:
            receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
        cmd = [
            sys.executable, str(SCRIPT),
            "--log", str(log), "--arch", arch, "--receipt", str(receipt_path),
        ]
        if model is not None:
            cmd += ["--model", str(model)]
        proc = subprocess.run(
            cmd, cwd=str(REPO_ROOT), capture_output=True, text=True, timeout=120,
        )
        self.assertEqual(
            proc.returncode, expect_rc,
            f"rc={proc.returncode}\nstdout={proc.stdout}\nstderr={proc.stderr}",
        )
        data = json.loads(receipt_path.read_text(encoding="utf-8"))
        return data, proc

    def check(self, data):
        self.assertIn("startup_config", data)
        self.assertIsInstance(data["startup_config"], dict)
        chk = data["product_default_check"]
        self.assertIn(chk["status"], ("match", "mismatch", "unknown"))
        self.assertIsInstance(chk["reasons"], list)
        return chk

    @staticmethod
    def reasons_text(chk):
        return "\n".join(str(r) for r in chk["reasons"]).lower()

    # -- Halo (gfx1151) ---------------------------------------------------

    def test_halo_explicit_q4_is_mismatch_against_fresh_pull_q8(self):
        data, _ = self.run_cli(halo_log(PLE_Q4_HALO), "gfx1151")
        chk = self.check(data)
        self.assertEqual(chk["status"], "mismatch")
        text = self.reasons_text(chk)
        self.assertIn("q4", text)
        self.assertIn("q8", text)
        raw = json.dumps(data["startup_config"])
        self.assertIn("qwen3.8-flash-next-gptq3.mq4.ple-q4", raw)
        self.assertIn("trunk overlay: applied 108 QT44 tensors", raw)

    def test_halo_embedded_q8_overlay_npu_auto_route_gpu_matches(self):
        data, _ = self.run_cli(halo_log(PLE_Q8_EMBEDDED), "gfx1151")
        chk = self.check(data)
        self.assertEqual(chk["status"], "match", chk["reasons"])
        # All request plans are GPU-only: recorded as not NPU-executed evidence.
        self.assertIn("gpu-only", json.dumps(chk.get("notes", "")).lower())
        raw = json.dumps(data["startup_config"])
        # Raw evidence lines are kept verbatim.
        self.assertIn("q8 from embedded model file", raw)
        self.assertIn("mode=auto", raw)
        self.assertIn("plan=[gpu 0..1131]", raw)
        self.assertIn("trunk overlay: applied 108 QT44 tensors", raw)

    def test_halo_fresh_pull_fallback_note_matches_q8(self):
        log = halo_log(PLE_Q4_MISSING_NOTE)
        data, _ = self.run_cli(log, "gfx1151")
        chk = self.check(data)
        self.assertEqual(chk["status"], "match", chk["reasons"])
        self.assertIn("using the embedded q8 rows", json.dumps(data["startup_config"]))

    def test_halo_missing_overlay_is_mismatch(self):
        data, _ = self.run_cli(halo_log(PLE_Q8_EMBEDDED, overlay=False), "gfx1151")
        chk = self.check(data)
        self.assertEqual(chk["status"], "mismatch")
        self.assertIn("overlay", self.reasons_text(chk))

    # -- R9700 (gfx1201) --------------------------------------------------

    def test_r9700_q8_hotset_matches_without_overlay_or_npu(self):
        data, _ = self.run_cli(r9700_log(PLE_Q8_EMBEDDED), "gfx1201")
        chk = self.check(data)
        self.assertEqual(chk["status"], "match", chk["reasons"])
        raw = json.dumps(data["startup_config"])
        self.assertIn("6479 experts in VRAM", raw)
        self.assertIn(HOTSET_SHA, raw)

    def test_r9700_overlay_present_not_applied_is_expected(self):
        data, _ = self.run_cli(r9700_log(PLE_Q8_EMBEDDED, overlay_note=True), "gfx1201")
        chk = self.check(data)
        self.assertEqual(chk["status"], "match", chk["reasons"])
        self.assertIn("present, not applied on gfx1201", json.dumps(data["startup_config"]))

    def test_r9700_explicit_hotset_disabled_whole_layer_is_mismatch(self):
        lines = list(PREAMBLE) + [PLE_Q8_EMBEDDED, HOTSET_DISABLED] + WHOLE_LAYER_PLACEMENT
        data, _ = self.run_cli("\n".join(lines) + "\n", "gfx1201")
        chk = self.check(data)
        self.assertEqual(chk["status"], "mismatch")
        text = self.reasons_text(chk)
        self.assertIn("hot", text)
        self.assertIn("whole-layer", text)

    def test_r9700_no_hotset_line_whole_layer_placement_is_mismatch(self):
        data, _ = self.run_cli(r9700_log(PLE_Q8_EMBEDDED, hotset=False), "gfx1201")
        chk = self.check(data)
        self.assertEqual(chk["status"], "mismatch")
        self.assertIn("hot", self.reasons_text(chk))

    def test_r9700_historical_q4_plus_hotset_is_not_product_default(self):
        # fn-floor-hotset-r9700 / mtp-reconcile floor4 g2: explicit q4 + pinned hot set.
        data, _ = self.run_cli(r9700_log(PLE_Q4_R9700), "gfx1201")
        chk = self.check(data)
        self.assertEqual(chk["status"], "mismatch")
        text = self.reasons_text(chk)
        self.assertIn("q4", text)
        self.assertIn("q8", text)

    # -- insufficient evidence -------------------------------------------

    def test_insufficient_log_is_unknown(self):
        log = "\n".join(PREAMBLE[:2]) + "\n"
        for arch in ("gfx1151", "gfx1201"):
            with self.subTest(arch=arch):
                data, _ = self.run_cli(log, arch)
                chk = self.check(data)
                self.assertEqual(chk["status"], "unknown")
                self.assertTrue(chk["reasons"])

    def test_empty_log_is_unknown(self):
        data, _ = self.run_cli("", "gfx1201")
        self.assertEqual(self.check(data)["status"], "unknown")

    # -- receipt preservation --------------------------------------------

    def test_existing_receipt_fields_are_preserved(self):
        existing = {
            "schema": "perf-receipt/1",
            "tok_s": [50.47, 50.77],
            "nested": {"prompts": 11, "k": 3, "labels": ["a", "b"]},
            "note": "historical q4 band",
            "null_field": None,
        }
        data, _ = self.run_cli(
            r9700_log(PLE_Q8_EMBEDDED), "gfx1201", receipt=dict(existing),
        )
        for key, value in existing.items():
            self.assertEqual(data[key], value, key)
        self.assertEqual(self.check(data)["status"], "match")

    def test_rerun_overwrites_check_but_keeps_other_fields(self):
        existing = {"tok_s": 61.0}
        first, _ = self.run_cli(
            halo_log(PLE_Q4_HALO), "gfx1151", receipt=dict(existing),
        )
        self.assertEqual(self.check(first)["status"], "mismatch")
        second, _ = self.run_cli(
            halo_log(PLE_Q8_EMBEDDED), "gfx1151", receipt=first,
        )
        self.assertEqual(second["tok_s"], 61.0)
        self.assertEqual(self.check(second)["status"], "match")

    # -- model identity ---------------------------------------------------

    def test_model_sha_and_path_recorded_when_model_given(self):
        model = self.tmp / "qwen3.8-flash-next-gptq3.mq4"
        payload = b"hipfire-receipt-test-model\x00" * 17
        model.write_bytes(payload)
        expect = hashlib.sha256(payload).hexdigest()
        data, _ = self.run_cli(r9700_log(PLE_Q8_EMBEDDED), "gfx1201", model=model)
        self.check(data)
        raw = json.dumps(data["startup_config"])
        self.assertIn(expect, raw)
        self.assertIn(str(model), raw)

    def test_model_identity_not_fabricated_without_model(self):
        # A hash that is never in the log must not appear; the only hashes present
        # are the ones the daemon log printed itself.
        decoy = self.tmp / "decoy.mq4"
        decoy.write_bytes(b"decoy-not-passed-to-cli")
        decoy_sha = hashlib.sha256(decoy.read_bytes()).hexdigest()
        data, _ = self.run_cli(r9700_log(PLE_Q8_EMBEDDED), "gfx1201")
        raw = json.dumps(data)
        self.assertNotIn(decoy_sha, raw)
        self.assertNotIn(str(decoy), raw)
        # Evidence from the log itself (hot-set identity) may be carried through.
        self.assertIn(BASE_MODEL_SHA, raw)

    def test_missing_log_fails(self):
        receipt = self.tmp / "receipt.json"
        proc = subprocess.run(
            [sys.executable, str(SCRIPT), "--log", str(self.tmp / "nope.log"),
             "--arch", "gfx1201", "--receipt", str(receipt)],
            cwd=str(REPO_ROOT), capture_output=True, text=True, timeout=120,
        )
        self.assertNotEqual(proc.returncode, 0)


if __name__ == "__main__":
    unittest.main()
