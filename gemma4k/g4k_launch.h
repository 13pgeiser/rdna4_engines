/* g4k_launch.h -- the one spelling of a kernel launch in gemma4k's device units.
 *
 * A template kernel's name carries commas, so it goes through G4K_KNAME the way HIP_KERNEL_NAME
 * exists for hipLaunchKernelGGL. G4K_LAUNCH is overridable so the same sources can be driven by a
 * host-side SIMT emulator in a test -- the kernels are ordinary C++ once the HIP names are
 * supplied. A build never defines it, and gets the triple-chevron launch, which the engine answers
 * through hipLaunchKernel on its own queues (docs/PLUGIN.md §5). */
#ifndef G4K_LAUNCH_H
#define G4K_LAUNCH_H
#define G4K_KNAME(...) __VA_ARGS__
#ifndef G4K_LAUNCH
#define G4K_LAUNCH(kernel, grid, block, stream, ...) kernel<<<grid, block, 0, stream>>>(__VA_ARGS__)
#endif
#endif
