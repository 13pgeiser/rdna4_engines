# Explicit native MQ6 railgun programs

The native gfx1151 lane-row MQ6 single/x4 kernels can be authored and executed
as a fixed-binding railgun program through
`rdna_compute::mq6_railgun::PreparedMq6Program`. This is the railgun follow-up
to the native MQ6 capture work in #941. It supplies an explicit reusable
operator API; automatic model capture integration remains a separate step.

`railgun::mq6::Mq6Facts::from_hit` requires the canonical bundle's actual
schema-7 A1/G8 receipt from a verified `JitCorpus` lookup. The author derives
memory effects from those facts through `kernel::author`, then the runtime
executes the resulting `plan::Pm4Plan` with ROCr's existing PM4 transport.
No name-based MQ6 `pointer_effects` entry is added to the Redline recorder.

The kernels are `gemv_mq6g256v2_xbatch_pm` and
`gemv_mq6g256v2_x4_xbatch_pm`. Their loader kernarg segments are 36 and 96 bytes,
respectively. The single-kernel launch record is zero-padded to 48 bytes; this
padding is not part of its loader segment. The code checks the receipt,
embedded bundle, extracted device ELF and loaded ABI before preparing an IB.

## Execution and ownership

- Exact gfx1151, F32 activation/output, MQ6G256V2 weights, K positive and
  divisible by 256, batch 1 through 8. An inactive x4 slot is `None`, encoded
  as null weight/output pointers with M=0. Shapes and bindings are constants.
- Prepare probes ordinary, same-device HIP allocations and subviews. It
  refuses out-of-bounds or unaligned views, conflicting allocation identities,
  and same-node output/input aliases. VMM and host-mapped allocations need
  their own range/lifetime proof and are refused here.
- Multiple nodes may form a dependency chain. The program uses conservative
  allocation-wide M1 effects and a private WaitIdle + System cache table for
  all hazards. This does not qualify global gfx1151 cache rows or claim A2
  span proofs. IF/WHILE from #963 is unnecessary for this straight-line API.
- Replay joins the current HIP producer stream, submits the retained planned
  PM4 IB and synchronously waits for terminal completion before returning to
  a HIP consumer. The existing PM4 transport is reused; no Redline recording
  or graph-capture admission is enabled by this API.
- Prepare/replay are unsafe: the caller retains every original allocation,
  including the owners behind subviews, at the same addresses until the
  program is quiesced and destroyed. Shallow tensor views do not own memory.
  Other-stream producers must be joined onto the current HIP stream.
- An execution failure poisons the program. Unknown quiescence quarantines
  its owned transport, code and kernargs; external allocations must remain
  live until GPU reset or process exit. A new address or shape requires a new
  prepare. Existing model routes and Redline defaults are unchanged.

## Reproduce the numerical and replay oracle

Use a private Cargo target directory for this checkout. No model is needed.
Create new output directories rather than overwriting previous evidence:

```bash
cargo build --release --locked -p railgun-cert
cargo build --release --locked -p rdna-compute --features lab \
  --example mq6_railgun_oracle
cargo test --release --locked -p railgun --lib mq6::tests

"$CARGO_TARGET_DIR/release/railgun-cert" certify --out "$CERT_OUT" \
  kernels/qwen4_lane_rows_pm_gfx1151.hxaco
python3 scripts/railgun/mq6_native_corpus.py \
  --receipt "$CERT_OUT/receipts/gfx1151/bc34ee389bd990386bdfc93d94b66752affcd3fbd33bec212a6b4edcaa51a963.receipt.json" \
  --out "$CORPUS_OUT"

# Acquire the machine's GPU locks before running this command.
"$CARGO_TARGET_DIR/release/examples/mq6_railgun_oracle" \
  --corpus "$CORPUS_OUT" --out "$ORACLE_OUT" \
  --source-head "$(git rev-parse HEAD)"
```

The packing helper only constructs a native fixture corpus from the real
certificate. It does not certify an object, invoke hipcc or admit a default
program. The CPU fixture contains the complete actual bundle receipt, with
negative mutations tested separately.

The GPU oracle compares against independently launched incumbent HIP singleton
rows, including full output guards. It covers single/x4 B1..8, ragged M,
inactive slots, native GDN/QSA projection shapes, and two-node RAW chains.
Each prepared program receives changing inputs A/B/A without changing its IB
or kernargs. No-write, poison, wrong-binding and real device-corruption
controls must be detected. Raw per-case program/plan/ABI identities and a
summary are written to the requested directory. This is component correctness
and retained-replay evidence; it does not measure model quality or speed.
