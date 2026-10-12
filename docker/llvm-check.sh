#!/bin/bash
# docker/llvm-check.sh BUILD -- the image's HIP compiler held to the two things it is for, on one of
# the build's own compile commands: r4d_hc_bf16.hip, the file whose code varied from build to build
# under ROCm's own clang. The command is run again four ways, outputs in a scratch directory:
#   1. pairing off, by ROCm's clang and by the image's: the device code must match byte for byte,
#      so the llvm stage's libLLVM is ROCm's but for the patch;
#   2. pairing on, exactly as the build ran it, twice more by the image's clang: both must match
#      the build's own object, so the pairing is repeatable.
# A mismatch fails the image build.
set -euo pipefail
B=$1
STOCK=/opt/rocm/lib/llvm/bin/clang++
IMAGE=/opt/rocm-repeatable/lib/llvm/bin/clang++
OBJCOPY=/opt/rocm/lib/llvm/bin/llvm-objcopy
O=libr4d/CMakeFiles/libr4d.dir/r4d_hc_bf16.hip.o
fail() { echo "llvm-check: $*" >&2; exit 1; }

T=$(mktemp -d)
trap 'rm -rf "$T"' EXIT
cd "$B"
cmds=$(ninja -t commands libr4d)
line=$(grep -m1 -F -- "-o $O " <<< "$cmds") || fail "no compile command for $O"
eval "set -- $line"
args=("$@")
[ "${args[0]}" = "$IMAGE" ] || fail "the build compiled with ${args[0]}, not $IMAGE"
for a in "${args[@]}"; do
    case "$a" in *amdgpu-enable-vopd*) fail "the build turned VOPD pairing off ($a)" ;; esac
done

# variant NAME COMPILER [FLAG]: the build's command with this compiler, its object and depfile in
# the scratch directory, and FLAG appended; then the object's device code as NAME.fb.
variant() {
    local name=$1 cc=$2 extra=${3:-} i=1
    local v=("$cc")
    while [ $i -lt ${#args[@]} ]; do
        case "${args[$i]}" in
            -o)  v+=(-o "$T/$name.o");  i=$((i + 2)); continue ;;
            -MF) v+=(-MF "$T/$name.d"); i=$((i + 2)); continue ;;
            *)   v+=("${args[$i]}") ;;
        esac
        i=$((i + 1))
    done
    [ -z "$extra" ] || v+=("$extra")
    "${v[@]}" > "$T/$name.log" 2>&1 || { cat "$T/$name.log" >&2; return 1; }
    "$OBJCOPY" --dump-section=.hip_fatbin="$T/$name.fb" "$T/$name.o" /dev/null
}

OFF=-mllvm=-amdgpu-enable-vopd=false
pids=()
variant stock_off "$STOCK" "$OFF" & pids+=($!)
variant image_off "$IMAGE" "$OFF" & pids+=($!)
variant image_on1 "$IMAGE" & pids+=($!)
variant image_on2 "$IMAGE" & pids+=($!)
for p in "${pids[@]}"; do wait "$p" || fail "a compile of $O failed"; done
"$OBJCOPY" --dump-section=.hip_fatbin="$T/build.fb" "$O" /dev/null

cmp -s "$T/stock_off.fb" "$T/image_off.fb" \
    || fail "with pairing off the image's clang and ROCm's emit different device code: the llvm stage did not rebuild ROCm's libLLVM"
cmp -s "$T/build.fb" "$T/image_on1.fb" && cmp -s "$T/build.fb" "$T/image_on2.fb" \
    || fail "with pairing on, three compiles of $O emitted different device code"
echo "llvm-check: r4d_hc_bf16.hip -- pairing off, ROCm's clang and the image's agree; pairing on, three compiles agree"
