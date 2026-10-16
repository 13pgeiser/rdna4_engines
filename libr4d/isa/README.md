# libr4d/isa: reading the machine code the kernels became

Nine scripts and four microbenchmarks. A wall-clock timer cannot tell "this kernel is slow" from
"this kernel is at the memory wall"; the disassembly plus one calibration number can, without a GPU
run. The trace tools are the exception: they need the GPU, and they are the only instruments here
that report the geometry a kernel was launched with rather than every geometry it could have had.

    BUILD=/path/to/build ./res.sh | head -40        # every kernel: vgpr/lds/scratch
    BUILD=... ./dis.sh r4d_gemm_fp8a8 ILb1ELb0ELj8ELj1ELj8ELj64ELb0E
    BUILD=... ./waterfall.sh                        # divergent VGPR indices, per kernel
    ./loop.sh                                       # its K loop, classified
    ./trace.sh traces/dec -- ./bin/radiance ...     # then: ./tsum.sh traces/dec/k_kernel_trace.csv

`res.sh` prints mangled names; the substring `dis.sh` takes picks one template instantiation, and
the parameter order is in the kernel's declaration (`gemm_fp8a8_kernel` is
`<Full, AW, NT, MT, MW, FK, GATED, I8>`, so `ILb1ELb0ELj8ELj1ELj8ELj64ELb0E` is the prefill tile).
`loop.sh` anchors on `v_wmma` by default and takes the smallest backward branch containing every
anchor, because a largest-loop search finds the epilogue, a long unrolled store chain.

## Tools

| tool | what it answers |
|---|---|
| `co.sh` | extracts the gfx1201 code object from one built `.hip.o`; the other scripts read that |
| `dis.sh` | disassembles one template instantiation from one translation unit |
| `loop.sh` | isolates the working loop of that disassembly and classifies its body by instruction class |
| `res.sh` | one row per device kernel in a build tree: VGPRs, SGPRs, LDS, scratch, spills, waves a SIMD, workgroup size |
| `waterfall.sh` | every kernel that indexes a VGPR array with an index the compiler could not prove uniform |
| `trace.sh` | runs a command under `rocprofv3 --kernel-trace` and leaves a CSV |
| `tsum.sh` | summarises that CSV over a steady-state window: where device time goes, per (kernel, grid) |
| `gaps.sh` | where the device is idle, and which op it was waiting for |
| `rskew.sh` | per-kernel device time on each agent, and the skew between ranks |
| `isarate.hip` | what one instruction slot is worth on gfx1201 (below) |
| `graphrate.hip` | what a dispatch costs the host to submit, and what capturing a fixed sequence saves |
| `launchreg.hip` + `launchreg_lib.hip` | whether a launch costs more when the module registers more kernels, and what crossing into a `.so` costs |
| `blockcopy.hip` | contiguous versus scattered device-to-host copy, at KV-block granularity |

Each script's header states its traps and how to read its output. The microbenchmarks are built by
hand: `clang++ -x hip -O3 -std=c++17 --offload-arch=gfx1201 <name>.hip -o <name>` for `isarate`,
`graphrate` and `blockcopy`, and `build_launchreg.sh` for the two-object launchreg pair.

## Instruction slot cost on gfx1201

`isarate.hip` runs each instruction back to back in eight independent chains, so nothing stalls
but issue. It reports a ratio, which is clock-independent, so it reads no clock.

One `v_wmma_f32_16x16x16_fp8_fp8` costs about 12.9 plain VALU issue slots, and this holds with
36 KiB of LDS limiting occupancy to what a real prefill tile gets (there it measures nearer 11.6,
for a peak of about 434 TFLOP/s). A kernel's ceiling is therefore

    TFLOP/s = 434 * wmma_slots / (wmma_slots + other_instructions_issued)

with `wmma_slots = 12.9 * (wmma count)`, both counts per loop iteration from `loop.sh`.

This is a ceiling, not a prediction. A kernel cannot beat it, and a kernel far below it is held
back by something else. Issue is often not the binding constraint: cutting a kernel's non-WMMA
instruction count by more than half can make it slower, because the removed work was covering
memory and LDS latency and its removal raises register pressure, and a memory-bound kernel moves by
about a percent under the same cut. Use the model to rule a lever out, not to predict a gain.

gfx1201 has no dense 16x16x32 fp8 WMMA (`llvm-mc -mcpu=gfx1201` rejects it). The only 32-deep fp8
form is `v_swmmac_f32_16x16x32_fp8_fp8`, the 2:4 sparse one, so the GEMMs here already use the
widest dense op there is.

## The profiler

`rocprofv3 --pmc` does not work on this platform. With any counter set and any kernel filter, on
this tree or on `isarate` alone, it collects, then hangs in `queue.cpp:939 Timeout while waiting
for queue sync: 1 kernels still active` and aborts on SIGABRT. There are no hardware counters, so
the dynamic instruction count is not directly observable; `loop.sh` plus knowing which blocks are
predicated is how to get it.

`--kernel-trace` works without masking any device. The abort that names an unsupported agent

    rocprofiler_iterate_agent_supported_counters failed for agent 3 (gfx1036)
      :: Agent HW architecture is not supported, no counter metrics found.

comes from counter enumeration and appears only under `--pmc`. It names an integrated display GPU,
which has no counter support; masking it does not make `--pmc` work. `trace.sh` sets
`ROCR_VISIBLE_DEVICES=0,1` when it is unset, only to pin the agent set and keep that line out of
the log. It runs `$ROCM/bin/rocprofv3`, with `ROCM` defaulting to `/opt/rocm`.

**A killed `rocprofv3` can outlive its own `timeout`.** The hang is in finalisation, after the
alarm has fired, so the child ignores the timeout and waits in queue sync indefinitely. It holds no
CUs, so it does not disturb a timing, but it holds VRAM, so the next tool reports allocation
failures that look like miscompiled kernels. Follow a `--pmc` attempt with `pkill -9 -f rocprofv3`;
`trace.sh` does this itself.

### Reading a trace

* Tracing costs about 2 us per dispatch per rank. The ranks are separate host threads and slow down
  in parallel, so per-rank dispatch counts are what matter, not the total.
* Kernel durations are hardware timestamps and do not change under tracing. The gaps absorb the
  whole interception cost, so busy time is real and idle time is an upper bound. `tsum.sh` does
  that subtraction and prints both.
* Generated text does not change under tracing, so a traced run can use a real serving
  configuration rather than a proxy.
* Busy is not utilisation. A dispatch counts as busy if it is resident at all, so a 5 us kernel on
  1 of 64 CUs reads the same as a GEMM saturating the card. "96% busy" says the step is almost never
  waiting for a launch, nothing about occupancy; for that, read the blocks column, where anything
  under ~64 cannot fill the device however long it runs.
* A trace can go quiet and still look complete. A run that idles before its workload records the
  startup and nothing else, and the CSV is well formed either way. Send the workload as soon as
  health answers, keep the run short, and check `dispatches/step` in `tsum.sh`'s header against what
  the model should issue before trusting any table built from it.

## General results

Two results recur and decide which levers are worth pulling.

**Every dispatch has a fixed cost of a few microseconds.** Across the shapes a decode step asks of
the narrow fp8 GEMM, the time follows one line,

    t = fixed + bytes / achievable_stream_rate

with a fixed term of roughly 4.5 us and a stream rate near the card's achievable bandwidth. The same
fixed cost shows on every small op (rmsnorm, quant_act, add, gated_quant) whatever it moves. A
"slow" small shape is usually a fixed cost that does not amortise, not a kernel with bandwidth left,
and fewer kernels is the only lever that reaches it.

**Most weight-streaming kernels are at the memory wall.** A narrow GEMM whose K loop issues a few
hundred instructions per few kilobytes of weight per wave has about four times more issue capacity
than the card has bandwidth, and runs at the achievable stream rate. Assembly-level work on such a
kernel does not pay; confirm with `loop.sh` before starting.

**Read a kernel's own header before proposing changes to it.** Each hot kernel's source records
its ablations, sweeps and rejected variants with the reason for each, and most assembly-level ideas
are already answered there.
