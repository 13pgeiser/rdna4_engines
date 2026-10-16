#!/bin/sh
# kbench-analysis.sh -- what a static analyser says about the kernel sources and a sanitised build
# says about a replay, as one markdown fragment that rad-kbench splices into its report with
# --analysis. Separate from rad-kbench because it runs in CI on machines without a GPU, while
# rad-kbench runs on a card.
#
# Usage: [JOBS=N] tools/kbench-analysis.sh <build-dir> <out.md>   (JOBS defaults to every core)
#
# Advisory: it never fails the build. clang-tidy reports language-level noise on this code, and a
# sanitised replay is a different binary from the one that ships.
set -eu

BUILD=${1:-build}
OUT=${2:-analysis.md}
ROOT=$(cd "$(dirname "$0")/.." && pwd)
CC_JSON="$BUILD/compile_commands.json"

TIDY=$(command -v clang-tidy || echo /opt/rocm/llvm/bin/clang-tidy)

{
  echo "## Static and dynamic analysis"
  echo
  echo "Advisory, and generated separately from the measurements above:"
  echo "\`tools/kbench-analysis.sh\` runs on a machine that need not have a GPU."
  echo
} > "$OUT"

# ---------------------------------------------------------------- clang-tidy
# The defect-finding packs only; the readability and naming packs would bury them under style
# findings this tree has decided against. performance-enum-size is excluded too: the ABI's enums
# are int-sized on purpose.
CHECKS='-*,bugprone-*,clang-analyzer-*,performance-*,-bugprone-easily-swappable-parameters,-bugprone-narrowing-conversions,-performance-enum-size'

if [ ! -f "$CC_JSON" ]; then
  echo "clang-tidy: skipped, no $CC_JSON (configure with -DCMAKE_EXPORT_COMPILE_COMMANDS=ON)" >> "$OUT"
  echo >> "$OUT"
elif [ ! -x "$TIDY" ] && ! command -v clang-tidy > /dev/null 2>&1; then
  echo "clang-tidy: not installed on this machine" >> "$OUT"
  echo >> "$OUT"
else
  TMP=$(mktemp)
  # Host C++ only. A HIP translation unit needs the offload flags clang-tidy will not replay, and
  # it reports the __device__ attributes as errors rather than analysing the code. One process per
  # unit, each into its own file, and a warning in a header counts once however many units
  # include it.
  PARTS=$(mktemp -d)
  find "$ROOT/libref" "$ROOT/core" "$ROOT/tools" -name '*.cpp' | sort |
    xargs -P "${JOBS:-$(nproc)}" -I{} sh -c \
      '"$1" -p "$2" --checks="$3" --quiet "$4" > "$5/$(echo "$4" | tr / _).log" 2>/dev/null || true' \
      sh "$TIDY" "$BUILD" "$CHECKS" {} "$PARTS"
  cat "$PARTS"/*.log | grep "warning:" | sort -u > "$TMP"
  rm -rf "$PARTS"

  N=$(grep -c "warning:" "$TMP" || true)
  echo "### clang-tidy" >> "$OUT"
  echo >> "$OUT"
  echo "\`bugprone-*\`, \`clang-analyzer-*\` and \`performance-*\` over every host translation unit" >> "$OUT"
  echo "(\`libref/\`, \`core/\`, \`tools/\`). HIP units are excluded: clang-tidy cannot replay the" >> "$OUT"
  echo "offload flags and reports \`__device__\` as an error rather than analysing the body." >> "$OUT"
  echo >> "$OUT"
  echo "- $N distinct warning(s)" >> "$OUT"
  if [ "$N" -gt 0 ]; then
    echo >> "$OUT"
    echo "| check | count |" >> "$OUT"
    echo "|---|---|" >> "$OUT"
    grep -o '\[[a-z][a-z-]*-[a-z][a-z.-]*\]$' "$TMP" | sort | uniq -c | sort -rn |
      while read -r n c; do echo "| \`${c}\` | $n |" >> "$OUT"; done
  fi
  echo >> "$OUT"
  rm -f "$TMP"
fi

# ---------------------------------------------------------------- sanitised replay
# A second build tree: -fsanitize changes every object, and mixing sanitised and plain objects can
# link into a binary that reports nothing.
echo "### AddressSanitizer + UndefinedBehaviorSanitizer" >> "$OUT"
echo >> "$OUT"
SAN="$BUILD-asan"
if [ -x "$SAN/bin/rad-kbench" ]; then
  LOG=$(mktemp)
  # Both fixtures, one run per distinct case, and the reference always run: a cached answer
  # would leave libref's code unexecuted under the sanitiser.
  ASAN_OPTIONS=detect_leaks=0:abort_on_error=0 \
  UBSAN_OPTIONS=print_stacktrace=1 \
    "$SAN/bin/rad-kbench" --home "$SAN/radiance_home" --max-cases 1 --no-cache \
      --fixture "$SAN/radiance_home/kernels.rkb" \
      --fixture "$SAN/radiance_home/kernels_moe.rkb" > "$LOG" 2>&1 || true
  E=$(grep -c "ERROR: AddressSanitizer\|runtime error:" "$LOG" || true)
  echo "A sanitised replay of both fixtures. Host code only -- see the note in the" >> "$OUT"
  echo "**Memory safety** section for why the device side cannot be sanitised on this card." >> "$OUT"
  echo >> "$OUT"
  echo "- $E report(s)" >> "$OUT"
  if [ "$E" -gt 0 ]; then
    echo >> "$OUT"
    echo '```' >> "$OUT"
    grep -h "ERROR: AddressSanitizer\|runtime error:" "$LOG" | sort | uniq -c | head -40 >> "$OUT"
    echo '```' >> "$OUT"
  fi
  rm -f "$LOG"
else
  echo "Not run: no sanitised build at \`$SAN\`. Build one with" >> "$OUT"
  echo >> "$OUT"
  echo '```sh' >> "$OUT"
  echo "cmake -S . -B $SAN -DRAD_ASAN=ON && cmake --build $SAN -j8" >> "$OUT"
  echo '```' >> "$OUT"
fi
echo >> "$OUT"
echo "analysis written to $OUT" >&2
