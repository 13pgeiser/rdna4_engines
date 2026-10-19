#!/usr/bin/env python3
"""Device-input-anchored parity for the MiniMax-H3 **audio** encoder (#14 value-parity).

This is the strict per-tensor check the NOTE-REF2VA gap note leaves open: does the device's
strided-causal-conv1d Aencode kernel chain compute the *same* normalised latent the reference HF
audio VAE does, to within the fp8/bf16 quantisation the container ships? The reference is run in
**fp32** (minimax_h3_audio_vae_fp32.safetensors), so a correct port shows a relative-Frobenius at or
below the container's own quant error (the convert log showed ~1.6e-3 on the audio tensors); a kernel
bug — e.g. a strided conv reducing the wrong span — shows up as a gross, order-of-magnitude error.

Device-INPUT-anchored means we do NOT re-derive the port's media preprocessing. We feed the reference
encoder the port's *own* staged `enc_samples` buffer (the exact hop-padded waveform it handed its
kernels) and compare the reference latent to the port's `b_enca` readback for the same slot. That
removes every decode/normalisation/padding confound: the two sides are handed identical input, so any
disagreement is encoder *math*, not preprocessing.

Layout it relies on (from arch/minimaxh3/minimaxh3.cpp, keep in sync):
    b_enca : [kEncSlotsA* kEncSlotRowsA, kAudioC] f32, holding the NORMALISED latent (op_ae_norm),
             kAudioHop = 800, kAudioC = 32, kAudioCh = 2, kEncSlotRowsA = 2*(500000//800) = 1250.
    slot row-base : standalone-audio slot k -> (3 + k - 1)*1250 ; video-soundtrack slot k -> (k-1)*1250.
    a slot's used rows = kAudioCh * T, T = ceil(n / 800); rows are CHANNEL-MAJOR (ch0 frames, ch1
    frames), matching the reference's encode() reshape (b*s, 1, L).
    staged enc_samples : [kAudioCh, stage_samples] channel-major, right-padded with zeros to a whole
    hop (stage_samples = ceil(n/800)*800).

Run under a torch environment; the reference import drags ComfyUI, so we force its CPU device branch
(the TP2 server holds both GPUs — this script opens no GPU context) by setting args.cpu.

    # capture the two device files (debug-gated dump, R4D_ENC_DEBUG) then:
    python3 ref_aenc_parity.py --enca enca.f32 --encsm encsm.f32 --kind audio --slot 1 --n 64000
    # or validate the harness itself with no device at all:
    python3 ref_aenc_parity.py --selftest
"""
import argparse
import math
import os
import sys

sys.path.insert(0, os.environ.get("COMFYUI_HOME", "/home/dustin/AI/ComfyUI"))

# Force ComfyUI's import-time device probe onto the CPU branch WITHOUT opening a ROCm context
# (the running TP2 server owns both GPUs). comfy.cli_args exposes a module-level `args`;
# model_management reads args.cpu before it ever touches a device.
from comfy.cli_args import args as _cargs
_cargs.cpu = True

import numpy as np
import torch
from safetensors.torch import load_file
from comfy.ldm.minimax.audio_vae import MiniMaxH3AudioVAE

# --- keep these in sync with arch/minimaxh3/minimaxh3.cpp ---
AUDIO_HOP = 800
AUDIO_C = 32
AUDIO_CH = 2
ENC_MAX_SAMPLES = 500000
SLOT_ROWS_A = AUDIO_CH * (ENC_MAX_SAMPLES // AUDIO_HOP)   # 1250
MAXA = SLOT_ROWS_A // AUDIO_CH                            # per-channel row stride within a slot (625)
VAE_SAFETENSORS = os.path.join(
    os.environ.get("COMFYUI_HOME", "/home/dustin/AI/ComfyUI"),
    "models/vae/minimax_h3_audio_vae_fp32.safetensors")


def load_encoder():
    m = MiniMaxH3AudioVAE().to(torch.float32).eval()
    m.load_state_dict(load_file(VAE_SAFETENSORS), strict=True)
    return m


def ref_latent(model, staged, ch, T):
    """staged: f32 numpy [ch, stage_samples] channel-major -> reference normalised latent [ch*T, 32]
    laid out exactly as b_enca stores it (channel-major rows)."""
    wave = torch.from_numpy(staged).unsqueeze(0)                 # [1, ch, L]
    with torch.no_grad():
        z = model.encode(wave)                                    # [1, 32, ch, T']
    assert z.shape[1] == AUDIO_C and z.shape[2] == ch, tuple(z.shape)
    zp = z[0].permute(1, 2, 0)                                    # [ch, T', 32]
    out = zp.reshape(ch * z.shape[3], AUDIO_C)                     # channel-major rows (contiguous)
    return out[:ch * T].numpy()                                    # used rows only


def slot_row(kind, slot):
    idx = (slot - 1) if kind == "vaudio" else (3 + slot - 1)
    return idx * SLOT_ROWS_A


def rel_frobenius(a, b):
    num = float(np.linalg.norm(a - b))
    den = float(np.linalg.norm(b)) + 1e-12
    return num / den, float(np.max(np.abs(a - b)))


def do_parity(m, enca_f, encsm_f, kind, slot, n):
    ch = AUDIO_CH
    allsm = np.fromfile(encsm_f, dtype=np.float32)
    # Device-input-anchored: the driver dumped the FULL staged buffer (hop-padded, channel-major),
    # so stage_samples is simply the file size over the channel count — no need to re-derive n.
    if allsm.size % ch != 0:
        sys.exit(f"encsm file has {allsm.size} floats, not a multiple of ch={ch}")
    stage_samples = allsm.size // ch
    if stage_samples % AUDIO_HOP != 0:
        sys.exit(f"staged length {stage_samples} is not a whole hop (x{AUDIO_HOP}); wrong file?")
    T = stage_samples // AUDIO_HOP
    staged = allsm.reshape(ch, stage_samples)
    if n and not (stage_samples - AUDIO_HOP < n <= stage_samples):
        print(f"  note: --n {n} is not within one hop below stage_samples {stage_samples} "
              f"(informational; the staged buffer drives the comparison)")
    ref = ref_latent(m, staged, ch, T)                             # [ch*T, 32]
    enca = np.fromfile(enca_f, dtype=np.float32).reshape(-1, AUDIO_C)
    base = slot_row(kind, slot)
    # Device layout: within a slot, channel c occupies rows [c*MAXA, c*MAXA+T). Stack them back to
    # the reference's contiguous channel-major [ch*T, 32] so the two sides align row for row.
    dev = np.vstack([enca[base:base + T], enca[base + MAXA:base + MAXA + T]])
    if dev.shape != ref.shape:
        sys.exit(f"device slice {dev.shape} vs reference {ref.shape}; check --kind/--slot/--n")
    rf, mx = rel_frobenius(dev, ref)
    print(f"slot kind={kind} slot={slot} n={n} T={T} rows={ch * T}")
    print(f"  relative Frobenius (dev vs ref) = {rf:.3e}")
    print(f"  max abs deviation               = {mx:.3e}")
    print(f"  reference latent |mean|={ref.mean():.4f} std={ref.std():.4f}; "
          f"device |mean|={dev.mean():.4f} std={dev.std():.4f}")
    # Decompose the gap (see ref_venc_parity.py for the rationale): a single global scale+bias that fits the
    # two sorted distributions well means the divergence is mostly ONE network-level gain, not structural.
    sa, sb = np.sort(dev.reshape(-1)), np.sort(ref.reshape(-1))
    A = np.vstack([sb, np.ones_like(sb)]).T
    (aa, bb), *_ = np.linalg.lstsq(A, sa, rcond=None)
    resid = sa - (aa * sb + bb)
    rf_aff = float(np.linalg.norm(resid)) / (float(np.linalg.norm(sb)) + 1e-12)
    r2 = 1.0 - float(np.sum(resid ** 2)) / (float(np.sum((sa - sa.mean()) ** 2)) + 1e-30)
    print(f"  global-affine fit  dev ~ {aa:.4f}*ref {bb:+.4f}:  R^2={r2:.3f}  residual(relFro after affine)={rf_aff:.3e}")
    print(f"    (a large rf->rf_aff drop => mostly ONE network gain; rf_aff near rf => structural)")
    floor = 5e-3   # fp8 audio quant floor is ~1.6e-3; allow headroom for the bf16 activations
    print(f"  verdict: {'MATCH within quant floor' if rf < floor else 'DIVERGED beyond quant floor'} "
          f"(floor {floor:.0e})")
    return 0 if rf < floor else 1


def do_selftest(m):
    """No device: build a device-shaped b_enca from the reference itself (perfect) and a
    temporally-mis-pooled copy (a stand-in for the strided-conv bug #14 suspected), and confirm the
    harness reads ~0 and gross respectively. This validates the reshape / slot / metric logic."""
    torch.manual_seed(0)
    n = 800 * 12                                   # T=12 latent frames
    stage = (n + AUDIO_HOP - 1) // AUDIO_HOP * AUDIO_HOP
    t = np.arange(stage, dtype=np.float32) / 32000.0
    ch0 = (0.3 * np.sin(2 * math.pi * 440.0 * t)).astype(np.float32)
    ch1 = (0.2 * np.sin(2 * math.pi * 880.0 * t)).astype(np.float32)
    staged = np.stack([ch0, ch1], axis=0)           # [2, stage] not mono-spread (so stereo ordering is tested)
    T = stage // AUDIO_HOP
    ref = ref_latent(m, staged, AUDIO_CH, T)        # [2T, 32]

    # perfect device = reference
    rf0, mx0 = rel_frobenius(ref.copy(), ref)
    # buggy device: collapse each strided-conv frame onto its neighbour (temporal reduction bug)
    bad = ref.copy()
    for s in range(AUDIO_CH):
        blk = bad[s * T:(s + 1) * T]
        blk[:-1] = blk[1:]                          # shift frames: a wrong temporal alignment
    rf1, mx1 = rel_frobenius(bad, ref)
    print(f"selftest  perfect  relFro={rf0:.3e} max={mx0:.3e}  (want ~0)")
    print(f"selftest  misalign relFro={rf1:.3e} max={mx1:.3e}  (want gross)")
    ok = rf0 < 1e-5 and rf1 > 1e-2
    print("selftest:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--enca", help="raw b_enca readback (f32, [slots*1250, 32])")
    ap.add_argument("--encsm", help="staged enc_samples readback (f32, [2, stage_samples])")
    ap.add_argument("--kind", choices=["audio", "vaudio"], default="audio")
    ap.add_argument("--slot", type=int, default=1)
    ap.add_argument("--n", type=int, help="raw enc_samples count (before hop padding)")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    m = load_encoder()
    if a.selftest:
        sys.exit(do_selftest(m))
    if not (a.enca and a.encsm):
        sys.exit("give --enca and --encsm (or --selftest); --n is optional/informational")
    sys.exit(do_parity(m, a.enca, a.encsm, a.kind, a.slot, a.n))


if __name__ == "__main__":
    main()
