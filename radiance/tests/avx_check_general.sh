#!/bin/sh
# avx_check_general.sh CHECK LIBREF SUBSET
#
# rad-avx-check as a general host-library checker, run on tests/avx_subset_plugin.cpp (part of
# libref, named "subset"):
#   1. it passes although the library lacks three of libref's ops, lists them, counts the f16
#      rmsnorm cases its constraint excludes and the hand-built cases for ops it does not serve, and
#      runs one pass at the library's own level with no file-backed gather;
#   2. a row that declines a geometry libref serves still fails.
set -u
CHECK=$1; REF=$2; SUBSET=$3
export RAD_SUBSET_OF="$REF" OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
fail() { printf 'FAIL: %s\n' "$*"; printf '%s\n' "$out" | tail -30; exit 1; }
has() { printf '%s\n' "$out" | grep -q -- "$1" || fail "missing: $1"; }
lacks() { printf '%s\n' "$out" | grep -q -- "$1" && fail "unexpected: $1"; return 0; }

out=$("$CHECK" --ref "$REF" --avx "$SUBSET" 2>&1); rc=$?
[ $rc -eq 0 ] || fail "exit $rc for a library that lacks ops"
has "checked as a host library"
has "3 of libref's ops have no host row here"
gaps=$(printf '%s\n' "$out" | grep -A1 "not checked:" | tail -1)
for op in softmax sample_merge_topk dflash_select; do
  printf '%s\n' "$gaps" | grep -qw "$op" || fail "$op is not listed as a gap"
done
has "== subset =="
has "admitted by no row's constraints"
has "for ops subset does not serve"
has "^PASS\$"
lacks "== level"
lacks "embed_lookup_q from a file"
lacks "COVER "

out=$(RAD_SUBSET_DECLINE=gemm_nt "$CHECK" --ref "$REF" --avx "$SUBSET" --op gemm_nt 2>&1); rc=$?
[ $rc -ne 0 ] || fail "exit 0 although gemm_nt declines every launch"
has "DECLINED"
has "^FAIL\$"
echo "avx_check_general: ok"
