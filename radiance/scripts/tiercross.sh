#!/bin/sh
# tiercross.sh -- two conversations through the KV tiers at once; checks that neither is served
# the other's state.
#
# tierident.sh checks that one session survives the round trip, but with one conversation there is
# nothing else to be served. Cross-contamination (an entry holding reassigned block ids, a
# checkpoint slot freed by one session and still named by another) needs two sessions in flight.
#
# Start the server with a host tier the filler overflows, so the arena's drops run as well as its
# restores:
#
#   --prefix-cache-mib 512
#
# It also needs snapshot slots for both sessions. On a hybrid model a hit is clamped to the deepest
# snapshot it reaches; a store too small for both makes each session retire the other's, and every
# arm re-prefills from zero (checked below via the device arm's reuse). The default
# --checkpoint-slots holds both.
#
# Each session is compared only to itself: A's restored answer must equal A's reference. Corrupt
# state typically shows as B's train of thought under A's question.
set -e

P=${P:-8100}
H=${H:-127.0.0.1}
TO_HOST=${TO_HOST:-10}
FILL=${FILL:-1200}
N=${N:-48}
work=$(mktemp -d "${TMPDIR:-/tmp}/tiercross.XXXXXX")
trap 'rm -rf "$work"' EXIT INT TERM

command -v jq >/dev/null || { echo "tiercross: jq is required" >&2; exit 2; }

# Two prompts that diverge at the first token: a shared opening would legitimately share blocks
# and blur the comparison.
#
# Each must produce a real answer. An instruction-tuned model may answer a wall of repeated
# sentences with a single EOS, and every arm would then compare empty strings; both lengths are
# checked below.
mk() {                        # mk <file> <subject> <opening> <question>
    : > "$1"
    i=0
    while [ $i -lt 300 ]; do
        printf '%s %d. %s ' "$2" "$i" "$3" >> "$1"
        i=$((i + 1))
    done
    printf '\n%s\n\nAnswer: ' "$4" >> "$1"
}
mk "$work/a.txt" "Chapter" "The paged pool is layer-major, so one logical block is many small fragments rather than one extent." \
   "Using only the passage above, write three sentences about it."
mk "$work/b.txt" "Entry"   "The linear layer keeps one state per sequence and updates it token by token." \
   "Using only the passage above, write three sentences about it."
# The files are sent verbatim through jq's --rawfile. Shell variables would lose the line breaks,
# and a question joined to the passage's last sentence gets an immediate EOS (as in tierident.sh).

ask() {                       # ask <who> <label> -> sha of the answer
    f="$work/$(printf '%s' "$1" | tr 'AB' 'ab').txt"
    jq -n --rawfile p "$f" --argjson n "$N" \
        '{model:"m", prompt:$p, max_tokens:$n, temperature:0, seed:1234, stream:false}' \
        > "$work/req.json"
    curl -sS --fail -X POST "http://$H:$P/v1/completions" \
         -H 'Content-Type: application/json' --data @"$work/req.json" > "$work/$2.json"
    jq -r '.choices[0].text' < "$work/$2.json" > "$work/$2.txt"
    sha1sum < "$work/$2.txt" | cut -d' ' -f1
}

# The hit length decides how the prefill is chunked and so every GEMM's shape. Restores that
# recovered different prefix lengths are different computations, so the hit is printed beside every
# arm and checked before a mismatch is called corruption.
hit() { jq -r '.usage.prompt_tokens_details.cached_tokens' < "$work/$1.json"; }

fail=0
# A hybrid session's reuse is clamped by the scheduler to the last linear checkpoint it can reach,
# not the attention hit (core/sched/scheduler.cpp, admit_one: RadBatch carries one query length for
# every KV group). Reuse is therefore a multiple of the checkpoint interval, depending on which
# snapshots survived: 0 on a cold run, 4096 or 6144 on a restored one.
#
# On Qwen3.8-Flash-Next with no tiers, the same prompt at reuse 0, 4096 and 6144 gives the same
# bytes, so a differing hash is a real difference. On another model, re-check that first: a
# near-tie that flips at the chunk boundary looks like corruption.
#
# Restored arms are compared against the device-hit arm, not the cold one: the device hit is the
# same query served from VRAM, while a cold run always has a different hit length.
cmp_arm() {                   # cmp_arm <label> <sha> <ref sha> <reuse> <ref reuse>
    if [ "$2" = "$3" ]; then
        echo "  ok    $1 (reuse $4)"
        return
    fi
    fail=$((fail + 1))
    if [ "$4" = "$5" ]; then
        echo "  FAIL  $1: SAME reuse ($4), different answer. This is corrupt state."
    else
        echo "  FAIL  $1: different answer, and it reused $4 where the reference reused $5."
        echo "        Almost certainly corruption -- reuse does not change the answer on the model"
        echo "        this was written against -- but confirm that on THIS model with the tiers"
        echo "        off before reporting it as one."
    fi
}
# ---- pushing a session out of host memory --------------------------------------------------------
# A full host tier drops its least recently used entries, so the sessions are pushed out by a
# prompt that shares no prefix with them and exceeds the arena's free room. The host tier must be
# small enough for FILL sections to overflow it: --prefix-cache-mib 512 holds about 27K tokens on
# Qwen3.8-Flash-Next, and the default FILL is about 30K.
fill_arena() {
    : > "$work/fill.txt"
    i=0
    while [ $i -lt "$FILL" ]; do
        printf 'Ledger line %d: warehouse north, shelf eleven, a part number and a count that nobody will ask about. ' "$i" >> "$work/fill.txt"
        i=$((i + 1))
    done
    printf '\nName the warehouse.\n\nAnswer: ' >> "$work/fill.txt"
    jq -n --rawfile p "$work/fill.txt" '{model:"m", prompt:$p, max_tokens:4, temperature:0, stream:false}' \
        > "$work/fill.json"
    curl -sS --fail -X POST "http://$H:$P/v1/completions" -H 'Content-Type: application/json' \
         --data @"$work/fill.json" > /dev/null
}
echo "tiercross: $H:$P, two sessions, ${TO_HOST}s for VRAM to go, $FILL filler sections"
CKS=$(curl -sS --fail "http://$H:$P/sessions" | jq -r .snapshot_store.bytes_each)
[ "$CKS" = "0" ] && echo "  (no recurrent state in this model; the attention half is still tested)"

A0=$(ask A a0); B0=$(ask B b0)
echo "  first     A $A0  reuse $(hit a0)"
echo "            B $B0  reuse $(hit b0)"
# The two answers must differ, or every comparison below would pass regardless.
if [ "$A0" = "$B0" ]; then
    echo "tiercross: both prompts produced the same answer, so no comparison here can tell the" >&2
    echo "           two sessions apart. Change the prompts." >&2
    exit 1
fi
for f in a0 b0; do
    CN=$(wc -c < "$work/$f.txt")
    [ "$CN" -lt 32 ] && { echo "tiercross: $f answered $CN bytes; every sha for it would match for that" >&2
                            echo "           reason alone. Give that prompt something to answer." >&2; exit 1; }
done

# ---- the control: both sessions again, straight out of VRAM -------------------------------------
# The same query the restored arms run. Without this control a differing hash below could come from
# non-tiering irreproducibility, or from contamination while both sessions are still resident.
AH=$(ask A ah); BH=$(ask B bh)
echo "  device    A $AH  reuse $(hit ah)"
echo "            B $BH  reuse $(hit bh)"
# A device hit that reused nothing makes every arm below a fresh prefill, testing nothing.
for f in ah bh; do
    case "$(hit $f)" in 0|null|"") echo "  FAIL  the $f arm reused nothing -- no hit, so no restore below can be tested
        (a snapshot store too small for both sessions? --checkpoint-slots)"; fail=$((fail + 1)) ;; esac
done
if [ "$AH" != "$A0" ] || [ "$BH" != "$B0" ]; then
    echo "  note: a device hit does not reproduce the cold answer. That is not necessarily wrong --"
    echo "        the hit reuses $(hit ah)/$(hit bh) tokens where the cold run reused nothing -- but every"
    echo "        comparison below is against the DEVICE HIT from here on, which is the like-for-like one."
fi

echo "  waiting $((TO_HOST + 8))s..."
sleep $((TO_HOST + 8))
A1=$(ask A a1); B1=$(ask B b1)
echo "  from RAM  A $A1  reuse $(hit a1)"
echo "            B $B1  reuse $(hit b1)"
cmp_arm "A restored from RAM is still A" "$A1" "$AH" "$(hit a1)" "$(hit ah)"
cmp_arm "B restored from RAM is still B" "$B1" "$BH" "$(hit b1)" "$(hit bh)"

echo "  filling host memory with $FILL sections of something else..."
fill_arena
sleep $((TO_HOST + 8))
A2=$(ask A a2); B2=$(ask B b2)
echo "  after a full host tier"
echo "            A $A2  reuse $(hit a2)"
echo "            B $B2  reuse $(hit b2)"
cmp_arm "A after the arena's drops is still A" "$A2" "$AH" "$(hit a2)" "$(hit ah)"
cmp_arm "B after the arena's drops is still B" "$B2" "$BH" "$(hit b2)" "$(hit bh)"

curl -sS --fail "http://$H:$P/sessions" > "$work/s.json"
echo "  moved: blocks to_host=$(jq -r .moved.to_host < "$work/s.json") promoted=$(jq -r .moved.promoted < "$work/s.json") host_drops=$(jq -r .moved.host_drops < "$work/s.json")"
# The drop path runs only if the arena ran out of room.
[ "$(jq -r .moved.host_drops < "$work/s.json")" = "0" ] && \
    echo "  note: the host tier never overflowed -- lower --prefix-cache-mib or raise FILL"

if [ "$fail" -ne 0 ]; then
    echo
    echo "A session answering with the OTHER session's train of thought is what corruption looks"
    echo "like. A reuse mismatch above is a different finding and the diffs below do not speak to it."
    for pair in "ah a1" "ah a2" "bh b1" "bh b2"; do
        set -- $pair
        cmp -s "$work/$1.txt" "$work/$2.txt" || { echo "--- $1 vs $2"; diff "$work/$1.txt" "$work/$2.txt" | head -4; }
    done
fi

echo
[ "$fail" -eq 0 ] && echo "tiercross: PASS -- neither session was served the other's context or state"
[ "$fail" -ne 0 ] && { echo "tiercross: $fail FAILURE(S)"; exit 1; }
exit 0
