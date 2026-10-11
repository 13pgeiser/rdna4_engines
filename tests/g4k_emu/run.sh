#!/bin/sh
# tests/g4k_emu/run.sh -- build and run g4k_emu with a clang that is not the build's C++ compiler.
#
# A build whose C++ compiler is clang builds these as ordinary targets (tests/CMakeLists.txt). The
# docker image compiles host C++ with g++, which has no ext_vector_type; its HIP compiler is ROCm's
# own clang++, and with `-x c++` that compiles the kernel sources as the host code this test needs.
#
#   run.sh <clang++> <abi dir> <gemma4k dir> <work dir> [sweep cases]
set -e
CXX=${1:?usage: run.sh <clang++> <abi dir> <gemma4k dir> <work dir> [sweep cases]}
ABI=$2; G=$3; B=$4; N=${5:-40}
HERE=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$B"
F="-std=c++20 -O2 -ffp-contract=off -Wno-psabi -I$HERE/emu -I$ABI -I$G"
for u in attn pick norm escale; do $CXX $F -x c++ -c "$G/g4k_$u.hip" -o "$B/k_$u.o"; done
$CXX $F -c "$G/g4k_rows.cpp" -o "$B/rows.o"
$CXX $F -c "$HERE/emu/emu.cpp" -o "$B/emu.o"
for t in attn pick norm escale; do
    $CXX $F -c "$HERE/test_$t.cpp" -o "$B/test_$t.o"
    $CXX "$B/test_$t.o" "$B/k_attn.o" "$B/k_pick.o" "$B/k_norm.o" "$B/k_escale.o" "$B/rows.o" \
        "$B/emu.o" -o "$B/test_$t"
done
"$B/test_attn"
"$B/test_attn" sweep "$N"
"$B/test_pick"
"$B/test_norm"
"$B/test_escale"
