/* vmemoff.so -- LD_PRELOAD shim for this host.
 *
 * The HSA runtime's KMT path (hsaKmtMemoryVaMap, used by hsa_amd_vmem_set_access)
 * hangs in a DRM syncobj wait whose KIQ TLB-flush fence is never signalled on
 * this kernel (6.12.113+deb13) + gfx1201. Radiance checks
 * HSA_AMD_MEMORY_POOL_INFO_RUNTIME_ALLOC_REC_GRANULE: a zero granule means "no
 * virtual memory management" and its KV pool falls back to one-shot allocation
 * through the plain path, which works.
 *
 * run_radiance.sh derives local/radiance-vmemoff from local/radiance with this
 * shim at /opt/vmemoff.so and sets LD_PRELOAD. vmemoff.so is gitignored but kept
 * on disk; it is rebuilt from vmemoff.c with plain host gcc when missing (the
 * source is self-contained, no ROCm toolchain needed), falling back to a build
 * inside the radiance-build image. VMEMOFF=0 ./run_radiance.sh runs plain
 * local/radiance without it.
 *
 * Build:  gcc -shared -fPIC -O2 -o vmemoff.so vmemoff.c -ldl
 */
#define _GNU_SOURCE
#include <dlfcn.h>
#include <stddef.h>
#include <stdint.h>

/* From ROCm's hsa_ext_amd.h (values stable across ROCm 7.x and 10.x):
 *   HSA_AMD_MEMORY_POOL_INFO_RUNTIME_ALLOC_REC_GRANULE = 18,  type size_t
 * hsa_handle_t is struct { uint64_t handle; }, passed by value on x86-64.
 * Declaring them here keeps this file buildable without the ROCm headers. */
typedef int hsa_status_t;
typedef struct { uint64_t handle; } hsa_amd_memory_pool_t;
#define HSA_AMD_MEMORY_POOL_INFO_RUNTIME_ALLOC_REC_GRANULE 18
#define HSA_STATUS_SUCCESS 0

static hsa_status_t (*real_get_info)(hsa_amd_memory_pool_t, int, void *) = NULL;

hsa_status_t hsa_amd_memory_pool_get_info(hsa_amd_memory_pool_t pool, int info, void *p)
{
    if (!real_get_info)
        real_get_info = (hsa_status_t (*)(hsa_amd_memory_pool_t, int, void *))dlsym(RTLD_NEXT,
                                                                                    "hsa_amd_memory_pool_get_info");
    hsa_status_t st = real_get_info(pool, info, p);
    if (st == HSA_STATUS_SUCCESS && info == HSA_AMD_MEMORY_POOL_INFO_RUNTIME_ALLOC_REC_GRANULE)
        *(size_t *)p = 0;
    return st;
}
