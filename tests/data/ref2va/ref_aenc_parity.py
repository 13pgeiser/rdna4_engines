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


def ref_pre_norm_z(model, wave_ch, T):
    """The reference mean_proj OUTPUT for one mono channel — the latent BEFORE the fixed
    (z - latents_mean)/latents_std, exactly what the device's b_amb2 (aenc.mp) holds. encode() folds
    the norm into its last line (audio_vae.py:439-442), so we reproduce the stages up to mean_proj
    directly: encoder -> pre_block -> mean_proj. wave_ch: f32 [L] (already hop-padded). Returns
    [T, 32] (row-major, to match the device's [rows, kAudioC])."""
    x = torch.from_numpy(wave_ch.reshape(1, 1, -1))              # [1, 1, L]
    with torch.no_grad():
        h = model.encoder(x)                                     # [1, latent_dim, T]
        h = model.pre_block(h.transpose(1, 2)).transpose(1, 2)   # [1, 32, T]
        z = model.mean_proj(h)                                   # [1, 32, T], pre-norm
    return z[0].permute(1, 0).numpy()                            # [T, 32]


def ref_pre_block_out(model, wave_ch, T):
    """The reference pre_block (AttnProjection) OUTPUT for one mono channel — one stage before
    mean_proj, what the device's b_ax2 (aenc.xfin) holds: encoder -> pre_block, WITHOUT mean_proj."""
    x = torch.from_numpy(wave_ch.reshape(1, 1, -1))              # [1, 1, L]
    with torch.no_grad():
        h = model.encoder(x)                                     # [1, latent_dim, T]
        h = model.pre_block(h.transpose(1, 2)).transpose(1, 2)   # [1, 32, T]
    return h[0].permute(1, 0).numpy()                            # [T, 32]


def _read_bf16(f):
    u = np.fromfile(f, dtype=np.uint16)
    return torch.from_numpy(u).view(torch.bfloat16).float().numpy()


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


def _affine_report(dev, ref):
    """Sorted-multiset relFro + best global affine fit dev ~ a*ref + b. SORTED so it is immune to the
    row/channel packing (b_amb2 stores one channel's [rows,32]; the reference mean_proj returns
    [32,T] transposed here) — only the value DISTRIBUTION is compared, which is what a scale gap is."""
    sa, sb = np.sort(dev.reshape(-1)), np.sort(ref.reshape(-1))
    if sa.size != sb.size:
        sys.exit(f"multiset sizes differ: device {sa.size} vs reference {sb.size}; check T / channel")
    rf = float(np.linalg.norm(sa - sb)) / (float(np.linalg.norm(sb)) + 1e-12)
    A = np.vstack([sb, np.ones_like(sb)]).T
    (aa, bb), *_ = np.linalg.lstsq(A, sa, rcond=None)
    resid = sa - (aa * sb + bb)
    rf_aff = float(np.linalg.norm(resid)) / (float(np.linalg.norm(sb)) + 1e-12)
    r2 = 1.0 - float(np.sum(resid ** 2)) / (float(np.sum((sa - sa.mean()) ** 2) + 1e-30))
    return rf, rf_aff, aa, bb, r2


def do_stage(m, dev_f, encsm_f, ch, T, stage):
    """Walk the audio final chain backward to localise a latent-scale gap. Each stage compares the
    DEVICE buffer (the last channel) against the reference at the SAME stage, pre-norm, so the fixed
    (z-mean)/std is excluded and only the network is compared:
        amb : device b_amb   (f32) vs reference pre_block (AttnProjection) output  [reliable]
        mp  : device b_amb2  (f32) vs reference mean_proj output (the latent before the fixed norm)
        pre : device b_ax2   (bf16) vs reference pre_block output — the raw bf16 buffer, whose
              readback lags the arena (non-finite stale rows); kept only to show the lag. 'amb' is
              its f32 mean_proj-input twin and is what to trust for the pre_block value.
    Because pre_block's LayerNorms reset the deep-encoder scale, a ratio ~ ref at 'amb' but a collapse
    only at 'mp' implicates mean_proj; a ratio already collapsed at 'amb' implicates pre_block (its LN
    affines / branches), since the encoder body's absolute scale is normalised away before it."""
    allsm = np.fromfile(encsm_f, dtype=np.float32)
    stage_samples = allsm.size // AUDIO_CH
    staged = allsm.reshape(AUDIO_CH, stage_samples)
    ref_fn = ref_pre_norm_z if stage == "mp" else ref_pre_block_out
    ref = ref_fn(m, staged[ch], T)                                # [T, 32]
    # 'pre' reads the raw bf16 b_ax2 (arena-rotated, readback can lag -> untrustworthy); 'amb' reads
    # b_amb, its f32 mean_proj-input twin, which is the reliable pre_block value.
    raw = _read_bf16(dev_f) if stage == "pre" else np.fromfile(dev_f, dtype=np.float32)
    dev = raw.reshape(-1, AUDIO_C)[:T]                            # last channel's used rows
    if dev.shape != ref.shape:
        sys.exit(f"device slice {dev.shape} vs reference {stage} {ref.shape}; check T / buffer rows")
    nbad = int((~np.isfinite(dev)).sum())
    if nbad:
        # A readback with non-finite values cannot go through the sorted-multiset fit (it would need
        # equal, all-finite sizes). The bf16 b_ax2 ('pre') lags the arena this way; 'amb' is its
        # reliable f32 twin, so just report the corruption and point at the right stage.
        print(f"STAGE '{stage}' READBACK UNUSABLE: {nbad} non-finite device values "
              f"(min {np.nanmin(dev):.4g} max {np.nanmax(dev):.4g}). "
              + ("The bf16 b_ax2 readback lags the arena — re-run with R4D_ENC_SEL=amb "
                 "(--stage amb) for the reliable pre_block value."
                 if stage == "pre" else "unexpected: check the buffer size/dtype."))
        return
    rf, rf_aff, aa, bb, r2 = _affine_report(dev, ref)
    ratio = float(np.std(dev)) / (float(np.std(ref)) + 1e-12)
    print(f"STAGE '{stage}' (device vs reference {stage} output, pre-norm), ch={ch} T={T}")
    print(f"  reference |mean|={ref.mean():.4f} std={ref.std():.4f}; "
          f"device |mean|={dev.mean():.4f} std={dev.std():.4f}")
    print(f"  device/reference std ratio            = {ratio:.4f}")
    print(f"  sorted relFro (dev vs ref)            = {rf:.3e}")
    print(f"  global-affine dev ~ {aa:.4f}*ref {bb:+.4f}: R^2={r2:.3f}  relFro after affine={rf_aff:.3e}")
    verdict = ("MATCHES reference — this stage is fine; the collapse is at/before it only if an earlier stage differs"
               if ratio > 0.85 else
               "COLLAPSED here — the gain is introduced AT this stage or upstream of it")
    print(f"  verdict: {verdict}")


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
    ap.add_argument("--dev", help="device stage readback (see --stage): 'mp'=b_amb2 f32, "
                                  "'amb'=b_amb f32 (reliable pre_block out), 'pre'=b_ax2 bf16 (arena-laggy)")
    ap.add_argument("--stage", choices=["mp", "amb", "pre"], default="mp",
                    help="which final-chain stage --dev is")
    ap.add_argument("--ch", type=int, default=AUDIO_CH - 1, help="channel in --dev (device keeps the last)")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    m = load_encoder()
    if a.selftest:
        sys.exit(do_selftest(m))
    if a.dev:
        if not a.encsm:
            sys.exit("--dev also needs --encsm (it gives T and the reference waveform)")
        allsm = np.fromfile(a.encsm, dtype=np.float32)
        T = (allsm.size // AUDIO_CH) // AUDIO_HOP
        sys.exit(do_stage(m, a.dev, a.encsm, a.ch, T, a.stage))
    if not (a.enca and a.encsm):
        sys.exit("give --enca and --encsm, or --dev/--stage and --encsm (or --selftest); --n is informational")
    sys.exit(do_parity(m, a.enca, a.encsm, a.kind, a.slot, a.n))


if __name__ == "__main__":
    main()
