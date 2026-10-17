#!/bin/sh
# Rebuild the Qwen3.8-Flash-Next container `fnserve.sh` serves; the canonical conversion.
#
# The recipe names what each weight is quantised to and the container records it (rad-info
# --recipe), so the engine serves the file in the format it was packed in. The default,
# data/recipes/qwen4exp-w4nl64-i8-hc8m.recipe, builds the production container w4nl64-i8lm.rad;
# its header describes each weight's format.
#
# That recipe requires CALIB=<dir>: its routed-expert rule runs GPTQ against `calib=$CALIB`, and
# the converter refuses a recipe that names an unset variable. The directory holds each layer's
# pooled Grams and each expert's down Gram, recorded from the bf16 model of this checkpoint
# (docs/MOE-W4.md, "Calibration"). Grams from another checkpoint do not transfer, and a layer with
# no file is packed without error feedback.
#
# A recipe that does not name $CALIB (e.g. RECIPE=data/recipes/qwen4exp-w4.recipe) rounds experts
# to nearest with absmax scales; setting CALIB=<dir> then applies int4 GPTQ through the --quant
# rule that recipe's header names, which takes precedence over the recipe's own rule.
#
# OUT is removed before conversion starts, so it defaults to a .new.rad beside the served
# container.
#
#   CALIB=<dir> [SRC=<checkpoint dir>] [OUT=<container>] [RECIPE=<file>] scripts/convfn.sh
set -e
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BUILD="${BUILD:-$ROOT/build}"
cd "$BUILD" || exit 1
SRC=${SRC:-$HOME/models/Qwen/Qwen3.8-Flash-Next}
OUT=${OUT:-$HOME/models/rad/w4nl64-i8lm.new.rad}
RECIPE=${RECIPE:-$ROOT/data/recipes/qwen4exp-w4nl64-i8-hc8m.recipe}
[ -d "$SRC" ] || { echo "no checkpoint at $SRC"; exit 1; }
[ -f "$SRC/tokenizer.json" ] || { echo "no tokenizer.json in $SRC"; exit 1; }
[ -f "$RECIPE" ] || { echo "no recipe at $RECIPE"; exit 1; }
[ -z "$CALIB" ] || [ -d "$CALIB" ] || { echo "no calibration directory at $CALIB"; exit 1; }
if grep -q '^[^#]*\$CALIB' "$RECIPE"; then
    [ -n "$CALIB" ] || { echo "$RECIPE quantises against \$CALIB: set CALIB=<calibration dir>"; exit 1; }
    export CALIB
    set --
elif [ -n "$CALIB" ]; then
    export CALIB
    set -- --quant 'blk.*.ffn_*_exps.*.weight=gptq:codes=i4,group=128,scale=bf16,transform=fwht128,calib=$CALIB'
else
    set --
fi
rm -f "$OUT"
./bin/rad-convert "$SRC" -o "$OUT" \
    --home radiance_home \
    --tokenizer "$SRC/tokenizer.json" \
    --recipe "$RECIPE" "$@" \
    -v 2>&1 | tail -30
[ -s "$OUT" ] || { echo CONVERT_FAILED; exit 1; }
echo CONVERT_OK "$OUT"
