/* r4d_launch.h -- the one spelling of a kernel launch in the libr4d units tests/r4d_emu runs on the
 * host: the windowed paged attention, pair_argmax, the sandwich-norm joins and expert_scale.
 *
 * A template kernel's name carries commas, so it goes through R4D_KNAME the way HIP_KERNEL_NAME
 * exists for hipLaunchKernelGGL. R4D_KLAUNCH is overridable so the same sources can be driven by a
 * host-side SIMT emulator in a test -- the kernels are ordinary C++ once the HIP names are
 * supplied. A build never defines it, and gets the triple-chevron launch, which the engine answers
 * through hipLaunchKernel on its own queues (docs/PLUGIN.md §5). */
#ifndef R4D_LAUNCH_H
#define R4D_LAUNCH_H
#define R4D_KNAME(...) __VA_ARGS__
#ifndef R4D_KLAUNCH
#define R4D_KLAUNCH(kernel, grid, block, stream, ...) kernel<<<grid, block, 0, stream>>>(__VA_ARGS__)
#endif
#endif
