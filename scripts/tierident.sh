#!/bin/sh
# tierident.sh -- does a session that left VRAM come back BYTE FOR BYTE, both halves of it?
#
# This is the acceptance test for the idle session tiers (core/mem/kvtier.h). It is a byte
# comparison because CORRUPT KV DOES NOT READ AS BABBLE -- it reads as fluent, deterministic,
# coherent, plausible text that simply is not the text the model would have produced, and nothing
# that looks at the engine from outside notices it. Only a byte comparison against a known-good
# arm sees it.
#
# THE CONTROL IS THE WHOLE TEST. Arm 2 asks the same prompt again while the session is still in
# VRAM. If that does not reproduce arm 1 exactly, then this model at this temperature is not
# reproducible across requests at all and every later comparison is meaningless -- a differing sha
# would prove nothing, because a cache hit COULD have been legitimately tie-divergent. Run the
# control or do not run the test.
#
# ============================== AND ON A HYBRID MODEL, HALF THE TEST IS INVISIBLE ===============
#
# A session on a hybrid model is attention blocks AND a recurrent-state snapshot. The two are
# stored separately and can be lost separately -- and losing the snapshot IS NOT VISIBLE IN THE
# TEXT. The engine simply replays the linear layers from an older checkpoint, or from token zero,
# and produces exactly the same answer, slower. So byte identity proves the blocks came back and
# says nothing whatever about the state.
#
# That is why every snapshot assertion below is a counter and not a sha. A run where the blocks
# moved and the snapshots did not is a PASS by text and a FAILURE here, and it is the failure that
# matters: it is the difference between a restored 200K session costing a second and costing a
# minute of linear replay.
#
#   scripts/tierident.sh                  # against a server already up on $P (default 8100)
#
# The server must be started with the host tier on, small enough for the filler below to overflow
# it (see fill_arena):
#   --prefix-cache-mib 512
# With TTL=<the server's --prefix-cache-ttl> set here as well, a last arm waits it out and checks
# that the session is gone from every tier and that asking again recomputes the same answer; start
# the server with a TTL long enough for the arms before it (a few minutes).
set -e

P=${P:-8100}
H=${H:-127.0.0.1}
TO_HOST=${TO_HOST:-10}
FILL=${FILL:-1200}
TTL=${TTL:-0}
N=${N:-48}
work=$(mktemp -d "${TMPDIR:-/tmp}/tierident.XXXXXX")
trap 'rm -rf "$work"' EXIT INT TERM

command -v jq >/dev/null || { echo "tierident: jq is required" >&2; exit 2; }

# A PROMPT LONG ENOUGH TO BE WORTH TIERING, AND LONGER THAN A CHECKPOINT INTERVAL. The unit of the
# attention half is the block, so a prompt of a few tokens occupies one and exercises nothing. The
# unit of the linear half is the checkpoint interval -- 2048 tokens by default -- and a prompt
# shorter than that produces NO snapshot at all, so every snapshot assertion below would be
# vacuously satisfied by a server where snapshot tiering is entirely broken.
#
# AND IT HAS TO ASK FOR SOMETHING. A wall of repeated sentences with no question at the end is a
# passage an instruction-tuned model can answer with a single EOS, and every arm would then compare
# an empty string to an empty string and report the control green. That is the same failure this
# whole script is built around -- a comparison that passes because neither side has any content --
# so the answer length is checked below as well.
: > "$work/p.txt"
i=0
while [ $i -lt 300 ]; do
    printf 'Section %d. The cache stores conversation prefixes as content-hashed blocks, and the identity of a block includes every token before it. ' "$i" >> "$work/p.txt"
    i=$((i + 1))
done
printf '\nUsing only the passage above, write three sentences about it.\n\nAnswer: ' >> "$work/p.txt"
# THE FILE IS THE PROMPT, READ VERBATIM, through jq's --rawfile. Escaping it by hand into a shell
# variable doubles the backslash jq adds before every quote, and flattening its newlines welds the
# question onto the passage ("...why it matters.Answer:" on one line), which this model answers
# with an immediate EOS -- the empty-completion failure described above.

ask() {                       # ask <label> -> writes $work/<label>.txt, echoes its sha
    jq -n --rawfile p "$work/p.txt" --argjson n "$N" \
        '{model:"m", prompt:$p, max_tokens:$n, temperature:0, seed:1234, stream:false}' \
        > "$work/req.json"
    curl -sS --fail -X POST "http://$H:$P/v1/completions" \
         -H 'Content-Type: application/json' --data @"$work/req.json" > "$work/$1.json"
    jq -r '.choices[0].text' < "$work/$1.json" > "$work/$1.txt"
    sha1sum < "$work/$1.txt" | cut -d' ' -f1
}
# ---- pushing a session out of host memory --------------------------------------------------------
# A FULL HOST TIER DROPS ITS LEAST RECENTLY USED ENTRIES, so the way to push a session out is to give
# the arena something else to hold: a prompt that shares no prefix with anything here and is bigger
# than the room the arena has left. The server must be started with a host tier FILL sections of it
# overflow; --prefix-cache-mib 512 holds about 27K tokens on Qwen3.8-Flash-Next, and the default
# FILL is about 30K.
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
snap() { curl -sS --fail "http://$H:$P/sessions" > "$work/s.json"; }
q()    { jq -r "$1" < "$work/s.json"; }
# PROMPT TOKENS A LOOKUP COVERED WITH RECURRENT STATE. Moving a snapshot and moving it back proves
# the bytes travelled and nothing more; this is the number that says a hit then SERVED one. They
# are not the same claim -- a restore that puts the state back somewhere the chain cannot reach
# scores perfectly on the first and zero on this.
lin()  { jq -r ".linear.cached_tokens" < "$work/s.json"; }

where() {                     # where -> "vram=N ram=N [+ snapshots]" for the whole server
    snap
    w="vram=$(q .blocks_sum_overlapping.device) ram=$(q .blocks_sum_overlapping.host)"
    if [ "$(q .snapshot_store.bytes_each)" != "0" ]; then
        w="$w  snapshots=$(q .snapshots_sum_overlapping)"
    fi
    echo "$w"
}

fail=0
check() {                     # check <label> <sha> <expected>
    if [ "$2" = "$3" ]; then echo "  ok    $1"
    else echo "  FAIL  $1: $2 != $3"; fail=$((fail + 1)); fi
}
note() { echo "  FAIL  $1"; fail=$((fail + 1)); }

echo "tierident: $H:$P, ${TO_HOST}s for VRAM to go, $FILL filler sections to push host memory out"

# IS THERE A LINEAR HALF AT ALL. A pure-attention model has no snapshots and must not be failed
# for having none; a hybrid one with a snapshot store of zero slots is a real misconfiguration and
# is named as one rather than skipped.
snap
HYBRID=0
[ "$(q .snapshot_store.bytes_each)" != "0" ] && HYBRID=1
if [ "$HYBRID" = 1 ]; then
    echo "  hybrid: snapshots are $(q .snapshot_store.bytes_each) bytes, $(q .snapshot_store.slots) slot(s)"
    [ "$(q .snapshot_store.slots)" = "0" ] && \
        note "the snapshot store has no room; a session's linear state has nowhere to be kept"
else
    echo "  this model has no recurrent state; the linear-state arms are not applicable"
fi

COLD=$(ask cold)
echo "  cold                       $COLD   [$(where)]"

# AN EMPTY ANSWER MAKES EVERY COMPARISON BELOW VACUOUS, and it is the easiest way for this script
# to report a confident green having tested nothing: three identical shas of the empty string.
CN=$(wc -c < "$work/cold.txt")
if [ "$CN" -lt 32 ]; then
    echo "tierident: the model answered $CN bytes. Every sha below would match for that reason" >&2
    echo "           alone. Give the prompt something to answer and run again." >&2
    exit 1
fi

# ---- the control -------------------------------------------------------------------------------
HOT=$(ask hot)
echo "  device hit                 $HOT   [$(where)]"
check "the same prompt twice from VRAM is byte-identical  (THE CONTROL)" "$HOT" "$COLD"
if [ "$HOT" != "$COLD" ]; then
    echo "tierident: the control failed, so nothing below would mean anything. Stopping." >&2
    echo "           This server is not reproducible across requests; fix that first." >&2
    exit 1
fi

# ---- host --------------------------------------------------------------------------------------
echo "  waiting $((TO_HOST + 8))s for the session to leave VRAM..."
sleep $((TO_HOST + 8))
W=$(where)
echo "  after the host threshold   [$W]"
# A RUN THAT MOVED NOTHING IS A FAILURE, NOT A PASS. This is the trap the whole script exists to
# avoid, turned on itself: if the session never left VRAM then every arm below is an ordinary
# device hit, the shas match for a reason that has nothing to do with tiering, and reporting PASS
# would be exactly the false confidence a byte comparison is supposed to replace.
snap
# A MISSING FIELD IS THE SAME FAILURE AS A ZERO: if the /sessions layout changes, this guard reads
# `null`, and treating that as a pass would pass vacuously.
case "$(q .blocks_sum_overlapping.host)" in 0|null|"") false ;; *) true ;; esac || note "the session never left VRAM -- nothing below would test anything
        (--prefix-cache-mib set? is the maintenance pass running?)"
if [ "$HYBRID" = 1 ]; then
    snap
    # A SNAPSHOT IS IN HOST MEMORY FROM THE MOMENT IT IS SAVED (the checkpoint store), so the
    # session's state must be there -- nowhere else is a place it can be.
    case "$(q .snapshots_sum_overlapping)" in 0|null|"") false ;; *) true ;; esac || \
        note "the session holds no linear-state snapshot, so the restore below tests the blocks only"
fi
snap
LIN0=$([ "$HYBRID" = 1 ] && lin || echo 0)
FROMHOST=$(ask fromhost)
echo "  restored from RAM          $FROMHOST   [$(where)]"
check "a session restored from RAM reproduces the cold answer" "$FROMHOST" "$COLD"
if [ "$HYBRID" = 1 ]; then
    snap
    # A SNAPSHOT NEEDS NO READ: it is restored straight out of its slot in the store. Whether it was
    # SERVED is the check.
    if [ "$(lin)" -gt "$LIN0" ]; then
        echo "  ok    and the restored state was SERVED  ($(( $(lin) - LIN0 )) prompt tokens covered by it)"
    else
        note "the snapshot came back and no lookup used it -- restored somewhere the chain
        cannot reach, which costs the replay it was moved to avoid"
    fi
fi

# ---- a full host tier ---------------------------------------------------------------------------
# THE FILLER OVERFLOWS THE ARENA, and the arena drops its least recently used entries -- this
# session's -- to make room. What comes back then is a recompute, or a restore of what survived; the
# answer is the cold one either way, and an entry the arena dropped that the cache went on serving
# would show here as another conversation's context.
echo "  filling host memory with $FILL sections of something else..."
snap
DROP0=$(q .moved.host_drops)
fill_arena
echo "  and waiting $((TO_HOST + 8))s for VRAM to go..."
sleep $((TO_HOST + 8))
W=$(where)
echo "  after the filler           [$W]"
snap
[ "$(q .moved.host_drops)" -gt "$DROP0" ] || note "the host tier dropped nothing -- the arm below tests nothing
        (a host tier FILL=$FILL sections overflow?)"
AFTER=$(ask afterfill)
echo "  after a full host tier     $AFTER   [$(where)]"
check "a session the full host tier dropped from reproduces the cold answer" "$AFTER" "$COLD"

# ---- expiry ------------------------------------------------------------------------------------
if [ "$TTL" -gt 0 ]; then
    echo "  waiting $((TTL + 5))s for --prefix-cache-ttl to take everything..."
    sleep $((TTL + 5))
    snap
    echo "  after the ttl              [$(where)]  expired: $(q .expired.entries) entries, $(q .expired.snapshots) snapshots"
    case "$(q '.sessions | length')/$(q .blocks_sum_overlapping.host)/$(q .occupancy.host_bytes)" in
        0/0/0) echo "  ok    nothing is left in any tier" ;;
        *) note "the ttl left something behind: $(q '.sessions | length') sessions, host bytes $(q .occupancy.host_bytes)" ;;
    esac
    [ "$(q .expired.entries)" -gt 0 ] || note "no entry was counted as expired"
    EXPIRED=$(ask expired)
    echo "  after expiry               $EXPIRED   [$(where)]"
    check "a session asked again after it expired reproduces the cold answer" "$EXPIRED" "$COLD"
fi

# ---- and the text itself, so a failure says WHERE ----------------------------------------------
if [ "$fail" -ne 0 ]; then
    echo
    echo "The first differing line, cold vs the failing arm:"
    for arm in fromhost afterfill expired; do
        [ -f "$work/$arm.txt" ] || continue
        if ! cmp -s "$work/cold.txt" "$work/$arm.txt"; then
            echo "--- $arm"
            diff "$work/cold.txt" "$work/$arm.txt" | head -6
        fi
    done
    echo
    echo "A DIFFERENCE IN THE TEXT IS CORRUPT KV, not a sampling artefact: the control above proved"
    echo "this server reproduces itself from VRAM, so the only thing that changed is where the"
    echo "bytes came back from. A snapshot failure with matching text is not corruption -- it is"
    echo "the restore silently costing a linear replay it was built to avoid."
fi

echo
[ "$fail" -eq 0 ] && echo "tierident: PASS -- a session survives VRAM -> RAM byte for byte, state included"
[ "$fail" -ne 0 ] && { echo "tierident: $fail FAILURE(S)"; exit 1; }
exit 0
