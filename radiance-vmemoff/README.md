# radiance-vmemoff (host-specific shim)

On this host (kernel 6.12.113+deb13 + gfx1201) the HSA runtime's KMT path hangs
in a DRM syncobj wait whose KIQ TLB-flush fence is never signalled, so plain
`radiance` dies at startup. `radiance-vmemoff/vmemoff.c` is a self-contained
`LD_PRELOAD` shim that zeroes one HSA pool field so radiance falls back to its
supported "no virtual memory management" allocation path. `run_radiance.sh`
enables it with `VMEMOFF=1` (the `.so` is gitignored and rebuilt from source
when missing). See the long comment block in `run_radiance.sh` for the full
diagnosis.

The trixie-backports kernel solves the issue: on a system running the
trixie-backports kernel the KMT path does not hang, so the shim is not needed.
