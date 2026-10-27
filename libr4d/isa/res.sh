#!/bin/sh
# One row per device kernel in a build tree: VGPRs, SGPRs, LDS bytes, scratch bytes, spilled
# registers, the waves a SIMD the VGPR count alone allows, and the workgroup size.
#
#   BUILD=/path/to/build res.sh
#
# SPILL IS THE LOUD COLUMN, and SCRATCH after it: spilled registers are a memory round trip the
# source does not show, and scratch without spills is a private array the compiler could not keep
# in registers. WAVES is the other one: on gfx1201 a SIMD has 1536 VGPRs in wave32 allocated in
# groups of 24, so 96 VGPRs is 16 waves, 97 is 12 and 256 is 5 -- and LDS can cut it further, which
# this column does NOT model. The parse is rows.awk, which selftest.sh checks at those boundaries.
: "${ROCM:=/opt/rocm}"
: "${BUILD:=build}"
: "${ISA_OUT:=/tmp/r4disa}"
D="$ISA_OUT/res"; rm -rf "$D"; mkdir -p "$D"
here=$(cd "$(dirname "$0")" && pwd)
for o in $(find "$BUILD" -name '*.hip.o'); do
  n=$(basename "$o" .hip.o)
  ISA_OUT="$D" "$here/co.sh" "$o" >/dev/null 2>&1 || continue
  "$ROCM/llvm/bin/llvm-readelf" --notes "$D/$n.co" 2>/dev/null > "$D/$n.md"
done
printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' vgpr sgpr lds scratch spill waves wg kernel
  awk -f "$here/rows.awk" "$D"/*.md | sort -k5,5nr -k4,4nr -k1,1nr
