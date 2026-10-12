# Measured floating edges

`gfx1151-gfx1201-core-edges-v2.tar.zst` is a lossless combined **empirical raw hardware capture**, not a numerical-model certificate. Architecture directories prevent basename collisions. It includes both completed core manifests, every referenced scalar/packed/VOPD table and exception payload, both packed-extra payloads per architecture, reconstruction readers, raw producer sources/generators/headers, predicate smoke records, and saved build provenance. Each `build/` contains the matching executable, linked device code object, host/device relocatable objects, preprocessed source snapshots, disassembly, build JSON and inspection tables. No pending WMMA, full-domain SFU or finite-FMAS corpus is included.

Extract with `zstd -dc gfx1151-gfx1201-core-edges-v2.tar.zst | tar -xf -`. Raw source artifacts remain at `/home/kaden/qcal/release-0.4.1/pm-r2/probe/`; gfx1151 originated at `hipx:/home/kaden/pm-wave/pm-r2-edge/`. Corrected build products are saved in `v2-gfx1151/` and `edge_v2_gfx1201/` under the local artifact root. The manifests retain exact original field pointers (`file`, pair-exception and packed-extra paths); bundled readers relocate absent paths by basename beside the manifest. When original absolute paths still exist, the reader prefers them; independent archive verification explicitly used extracted member paths.

## Scalar, packed and VOPD

| Architecture | Manifest | Ordinary table specs | Ordinary tuples per state | Table/pair payloads verified | Extra payloads verified |
|---|---|---:|---:|---:|---:|
| gfx1151 | `gfx1151/capture-v2-gfx1151.manifest.json` | 99 | 26,303,051 | 313 | 2 |
| gfx1201 | `gfx1201/edge-capture-v2-gfx1201.manifest.json` | 101 | 26,311,243 | 319 | 2 |

Both captures measured all ordinary tuples in all four VCC/SCC initial states: `s=0..3`, `vcc_in=(s>>1)&1`, `scc_in=s&1`. Literal and SGPR operands are wave-uniform; unused lanes are not counted. Manifests retain typed operand sets, exact native instructions/modifiers, axis order, descriptor FP modes, seeds and object fingerprints.

A table records one or two raw destination words and per-lane VCC/SCC bits. Wave32 VCC masks are reconstructed from lane/axis tiling; these fields are not independent whole-wave masks. `fold` retains a state-zero baseline plus every exception; `full` retains every state. Hashes and lengths describe decompressed bytes. Run `python3 gfx1151/edge_recon.py gfx1151/capture-v2-gfx1151.manifest.json verify` (analogously for gfx1201); `spec NAME --idx i,j,k [--state S]` reconstructs raw operands and outputs. The reader's `verify` covers table/pair files; packed extras require their separate manifest SHA256/length checks. Each architecture includes 18 extra tuples × four states in a 1,152-byte record payload and a 2,270-byte index payload.

The v2 SGPR predicate correction applies to gid 29, `v_div_scale_f32.sdst_vvv`: the generator emits `@PREDD1`, waits for the VALU-to-SGPR result (`s_waitcnt_depctr 0` on gfx1151, `s_wait_alu 0` on gfx1201), then reads this lane's actual predicate bit into D1 with `v_cndmask_b32_e64`. D1 is an early-clobber pure output, not the `EDGE_D_INIT` seed. A generator contract checks that SGPR-predicate writers declare and perform the matching readback. Both new captures include this corrected D1 payload.

VOPD uses native paired instructions over full Cartesian grids, not sampled pair contexts. On **each** architecture, fmac/fmac checks 274,877,906,944 tuple-states (`64^6 × 4`); mul/add-literal and its forced-literal twin each check 67,108,864 (`64^4 × 4`). All three report zero pair-context and half-reference exception records. Half-reference checks cover 2,097,152 tuple-states for fmac/fmac and 32,768 for each mul/add variant. Independent native-half tables plus retained exceptions reconstruct full pair outputs. This is an empirical context-independence measurement, not CPU arithmetic-model equality.

## Archive integrity

Independent temporary extraction verified all **636** lossless core payloads (315 gfx1151, 321 gfx1201), including extras, against raw SHA256 and exact length with zero failures. Their decompressed lengths total 139,489,930 and 139,555,466 bytes respectively. Every one of the **682 file members**, including manifests, sources and objects, was also compared byte-for-byte against its original. Both executable/device-object/build-JSON fingerprints matched the capture manifests. A second streamed archive build was byte-identical.

The deterministic archive uses sorted USTAR members, zero timestamps/uid/gid, mode `0644`, and streamed `zstd -19 -T4 --long=27` compression. No measured core payload was discarded for size.

- Compressed size: **5,452,076 bytes**
- SHA256: `6dbf478885069842f3df0d63028cac3287dca487ffcffb8c4a8516279794c98f`
- MD5: `da87324f363e9c4072cefa427b82e639`

## Historical WMMA evidence

The superseded `gfx1151-edges-v1.tar.zst` is removed from this checkout rather than duplicated. Its original gfx1151 f16/bf16 WMMA raw artifacts remain unmodified under the artifact root (`wmma-capture-gfx1151.{f16,bf16}.v0.*`), with readers and own-run reports; the v1 archive remains in commit `42a51b3f9`. The new core-only archive does not replace or include that WMMA corpus.

Each historical operation captured 66 physical packed-K fragments × 66 fragments × 34 FP32 accumulator values × four flag states = 592,416 tuples. Historical own-run `mma.rs` comparisons reported 151,658,496 destination words per operation and zero mismatches for f16 and bf16. Those are captured-corpus results only, not randomized or full-domain qualification; this archive-integrity work did not rerun numeric comparisons.

## Local gfx1201 WMMA census

The later `wmma-capture-gfx1201-v2.{f16,bf16,fp8}.v0.*` raw artifacts remain under the artifact root, outside this core archive. Own production-linked comparisons checked **151,658,496** words each: f16 had **zero** differences; BF16 had **32,384** differences across 40 operand triples and is **UNQUALIFIED**. The BF16 report is `edge-wmma-model-gfx1201-v2-bf16.json`; for example, factors `A=[0001;16]`, `B=[4040,4100,0,…]`, and `C=80000001` produced hardware `000afffc`, versus model `000b0000`.

The FP8 capture contains **382,024** tuples (**97,798,144** destination words). Every one of 54 experimental traversal/arithmetic candidates was compared against all words; none had zero differences (`edge-wmma-fp8-candidates-gfx1201-54.json`). FP8 is **UNQUALIFIED** and has no production value API. BF16/FP8 WMMA are outside the QSA/sym-NT4 fold; the emulator rejects them rather than using these unqualified models.

## Opcode-fill qualification (pending, not certified)

The pm-opcode-fill work added emulator semantics for the missing gfx1100/gfx1151/gfx1201 rows. This section records what is and is not qualified; nothing here is a hardware certificate until its receipt is listed as completed.

Probe harness: `isa/emu/numerics-probe/{numerics_probe.hip,build.sh,run.sh}`. Raw receipts stay outside the checkout under `/home/kaden/qcal/release-0.4.2/pm-opcode-fill-numerics/rcpt/`. Hardware identities: gfx1201 card B job #1069 (UUID `GPU-e475645fe0200397`, BDF `0000:13:00.0`) and card C (UUID `GPU-085289909a86cc63`, BDF `0000:e3:00.0`, receipt `gfx1201-C-r1`, 213 records); gfx1151 Halo job #112 (BDF `0000:c0:00.0`; no ROCr UUID, `run.sh` captures the unique_id).

Measured (gfx1201 receipt `gfx1201-C-r1`; card B ran the same probe and its ~900 MB of outputs are bit-identical to card C):
- `v_dot4_i32_iu8`, `v_dot8_i32_iu4`: exact integer sum; NEG_LO[0]/[1] select src0/src1 signed (0 unsigned); CLAMP is **one final signed i32 saturation** (per-term saturation refuted), otherwise the dword wraps; NEG_LO[2], NEG_HI and MODE have no effect. 0 mismatches over 402,304 full-grid evaluations and 72,304 per modifier set.
- `v_dot2_f32_f16`, `v_dot2_f32_bf16`: one two-term `dot_step` pair of the WMMA FP32 accumulation, including the threshold-23 accumulator rule (the exact single-rounded sum is wrong: 55k-58k mismatches). 0 mismatches over 1,321,984 full-set words x 8 MODE settings per opcode (NaN payloads, infinities, denormals in/out, sNaN, -0) and 99,742 words x 15 modifier variants. Round/denorm MODE and CLAMP have no effect. `neg_lo[0]/[1]` and `neg_hi[0]/[1]` flip the sign of A/B's lo/hi element, `op_sel[0]/[1]` pick the half that is the lo element, `op_sel_hi[0]/[1]` the hi element, `neg_lo[2]`=-C, `neg_hi[2]`=|C|, both -|C|. op_sel[2]/op_sel_hi[2] were never varied and an inline constant as a half source was not measured: refused.
- `v_wmma_f32_16x16x16_bf16` (and f16): the `mma.rs` chain (four 4-term steps, K-order swap of physical bits 2/3, FP32 accumulator floor rule: a negative C discards its tail with a -1 unit decrement iff the tail is non-zero and `grid - exponent(C) <= 23`) equals hardware: 0 mismatches over 7,437,056 words (controls, single-product Cartesian, subnormal rows, C-tail window sweep, two-term Cartesian, 10,000 random waves). MODE has no effect. This closes the 32,384-word BF16 mismatch above; the frozen `magnitude != 0` guard was the bug. The raw discriminator `A=[0001;16] B=[4040,4100,0..] C=80000001 -> 000afffc` is pinned in `mma.rs`/`wmma.rs`. This supersedes "UNQUALIFIED" above for **gfx1201 only**; gfx1151 BF16 keeps its historical captured-corpus evidence (the changed rule is gfx12-only code), with no fresh receipt yet.
- `v_swmmac_i32_16x16x32_iu8` index and fragment map (exact one-hot probe, 3,840 + 16,384 row reads, 0 deviations; `emu/wmma.rs::swmmac` documents it): lane `l` = row `l & 15`, half `l >> 4` owns compressed slots `8h + i` (byte `i & 3` of VGPR `i >> 2`); the index word is the same lane's, selector of slot `i` at `idx[2i+1:2i]` (`index_key:0`); slot `j` lands on dense K = `4*(j>>1) + selector` and equal selectors accumulate; dense B lane `l` = column `l & 15`, element `e` (VGPR `e>>2`, byte `e&3`) is K = `16*(l>>4) + e`; D as WMMA. A recorded hardware wave is a regression fixture.
- `v_swmmac_f32_16x16x32_f16`, `v_swmmac_f32_16x16x32_bf16` (receipts `gfx1201-{B,C}-r1` `swmmac.*.numerics`): exactly the dense WMMA FP32 chain applied to the 16 compressed slots of a row in slot order (term `c` taken from slot `(c&3)|((c&4)<<1)|((c&8)>>1)`, B paired by `K = 4*(j>>1)+selector`, duplicates are separate terms, zeros never enter), threshold-23 rule. 0 mismatches over 76,800 random-selector words and 76,800 dense-equivalent-selector words per kind (an independent port of the Rust gather also had 0 mismatches over 38,400 sampled words per kind). 16-bit packed layout: A element `i` is VGPR `i>>1`, half `i&1`; B element `e` is VGPR `e>>1`, half `e&1`. Hardware waves are regression fixtures.
- `v_wmma_i32_16x16x16_iu8` (gfx1201, `rcpt/gfx1201-C-r1`): exact integer sum, NEG_LO[0]/[1] = A/B signed, wrap without CLAMP (0 mismatches / 295,680 words per sign combination). **CLAMP saturates to i32 after `C + products(K 0..8)` and again after the K 8..16 products** (0 mismatches / 295,680 words for each clamp variant; one final saturation has 4,864, per-4-K 372, per-product 1,291). MODE has no effect. gfx1151 IU8 CLAMP has no receipt and is refused.

Still not qualified, no `semantics.tbl` row, hard error (no provisional execution):
- `v_wmma_bf16_16x16x16_bf16` (bf16-out rounding and packing unmeasured).
- SWMMAC `index_key:1` (OPSEL[0] placement), NEG_LO signedness, NEG_HI and CLAMP of the iu8 and f32 forms (refused inside the models), and `v_swmmac_i32_16x16x32_iu4`, `..x64_iu4`, the 16-bit-out SWMMAC (`f16_16x16x32_f16`, `bf16_16x16x32_bf16`, same unresolved 16-bit accumulator as the 16-bit-out WMMA) and every FP8/BF8 WMMA and SWMMAC (accumulation order, sparse zero-product handling, E4M3/E5M2 NaN rules; the 54 dense FP8 candidates above all failed).
- gfx1151/gfx1100 float DOT2, integer-dot CLAMP/NEG[2]/NEG_HI and gfx1151 IU8 WMMA CLAMP (no gfx11 receipt; Halo receipt `gfx1151-halo-r1` has not completed), gfx1100 WMMA of any type (no probe corpus).
- The IU8 WMMA gfx11 fragment map is by analogy to the F16/IU4 gfx11 maps and is unobservable for dense D (A and B share one map); it awaits the Halo receipt.

## Exhaustive gfx1201 exp qualification

Outside the core archive, `gfx1201-sfu-exp-from00.range.json` records one completed comparison of **all 4,294,967,296** input bit patterns against the production Rust `exp_f32`: **zero mismatches**. Own full-cover/mask-CRC/popcount certification passed in `gfx1201-exp-full-own-cert.json`; the 536,870,912-byte decompressed mask has CRC32 `1840808736` and zero set bits.

The exact compiled-model chain is frozen in `sfu-build-final/gfx1201/sfu_build.json`: native SHA256 `50bf70f7a6a5144370d45b1dfc36c0b36467989c4e583cece04a55cf8f204069`, MD5 `1a023f97747c028cb0ccc409143ec902`, plus source/bridge/tables, compiler flags/hashes, static library, and the linked device ELF embedded in that native executable. Historical partial ranges are **excluded** because their compiled-library/executable lineage was not reproduced byte-identically.

This is an online GPU-versus-model comparison with retained mismatch mask and sparse samples, **not** an offline recomparison of a retained full GPU-output stream. The measured context is the pinned Wave32 probe descriptor (round32/16_64=0, denorm32/16_64=3); arbitrary MODE states are not qualified.

At the hard drain, gfx1201 rcp/rcp_iflag and all three gfx1151 SFU full-domain gates remain pending, as do randomized F16/IU4 qualification on both architectures and Halo DIV_FMAS discriminators. Their queued holds were cancelled; no completion is inferred from prepared binaries or the captured WMMA census.

## Qualification limits

The scalar F32 set does **not** contain DIV_FMAS scaled-rounding discriminator words `0x14800000` and `0x88800000`, or the `2^80` overflow discriminator `0x67800000`. Do not infer generic VCC=1 DIV_FMAS rounding semantics from these captures. Ordinary ABS/CLAMP/OMOD variants absent from the audited census are not claimed measured.

Raw table integrity is **not CPU-model numerical proof**. This archive makes no full SFU, WMMA, or finite-FMAS qualification claim. The fold covers only QSA convert/attend and sym-NT4 gate-up/down. Every exercised SFU and F16/IU4 WMMA model requires its numerical gate; unused SFUs and BF16/FP8 WMMA remain **UNQUALIFIED** and have no permitted emulator execution path. Preliminary source-derived models must not be described as hardware-certified.
