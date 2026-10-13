#!/bin/sh
# tests/r4d_emu/run.sh -- build and run r4d_emu with a clang that is not the build's C++ compiler.
#
# A build whose C++ compiler is clang builds these as ordinary targets (tests/CMakeLists.txt). The
# docker image compiles host C++ with g++, which has no ext_vector_type; its HIP compiler is ROCm's
# own clang++, and with `-x c++` that compiles the kernel sources as the host code this test needs.
#
#   run.sh <clang++> <abi dir> <libr4d dir> <work dir> [sweep cases]
set -e
CXX=${1:?usage: run.sh <clang++> <abi dir> <libr4d dir> <work dir> [sweep cases]}
ABI=$2; G=$3; B=$4; N=${5:-40}
HERE=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$B"
F="-std=c++20 -O2 -ffp-contract=off -Wno-psabi -I$HERE/emu -I$HERE -I$ABI -I$G"
$CXX $F -x c++ -c "$G/r4d_attn_paged_window.hip" -o "$B/k_attn.o"
$CXX $F -x c++ -c "$G/r4d_pair_argmax_f32.hip" -o "$B/k_pick.o"
$CXX $F -x c++ -c "$G/r4d_norm_join_bf16.hip" -o "$B/k_norm.o"
$CXX $F -x c++ -c "$G/r4d_expert_scale_bf16.hip" -o "$B/k_escale.o"
$CXX $F -c "$HERE/ref_host.cpp" -o "$B/ref.o"
$CXX $F -c "$HERE/emu/emu.cpp" -o "$B/emu.o"
for t in attn pick norm escale; do
    $CXX $F -c "$HERE/test_$t.cpp" -o "$B/test_$t.o"
    $CXX "$B/test_$t.o" "$B/k_attn.o" "$B/k_pick.o" "$B/k_norm.o" "$B/k_escale.o" "$B/ref.o" \
        "$B/emu.o" -o "$B/test_$t"
done
"$B/test_attn"
"$B/test_attn" sweep "$N"
"$B/test_pick"
"$B/test_norm"
"$B/test_escale"
