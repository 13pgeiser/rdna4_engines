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
 * on disk; it is rebuilt from vmemoff.c inside the radiance-build image when
 * missing. VMEMOFF=0 ./run_radiance.sh runs plain local/radiance without it.
 *
 * Build:  gcc -shared -fPIC -o vmemoff.so vmemoff.c -I/opt/rocm/include
 */
#define _GNU_SOURCE
#include <dlfcn.h>
#include <hsa/hsa_ext_amd.h>

static hsa_status_t (*real_get_info)(hsa_amd_memory_pool_t,
                                     hsa_amd_memory_pool_info_t, void *) = NULL;

hsa_status_t hsa_amd_memory_pool_get_info(hsa_amd_memory_pool_t pool,
                                          hsa_amd_memory_pool_info_t info, void *p)
{
    if (!real_get_info)
        real_get_info = (hsa_status_t (*)(hsa_amd_memory_pool_t, hsa_amd_memory_pool_info_t,
                                          void *))dlsym(RTLD_NEXT, "hsa_amd_memory_pool_get_info");
    hsa_status_t st = real_get_info(pool, info, p);
    if (st == HSA_STATUS_SUCCESS && info == HSA_AMD_MEMORY_POOL_INFO_RUNTIME_ALLOC_REC_GRANULE)
        *(size_t *)p = 0;
    return st;
}
