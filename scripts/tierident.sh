#!/bin/sh
# tierident.sh -- checks that a session that left VRAM comes back byte for byte, attention blocks
# and recurrent state both.
#
# Acceptance test for the idle session tiers (core/mem/kvtier.h). Corrupt KV produces fluent,
# deterministic, plausible text that is simply not what the model would have produced, so only a
# byte comparison against a known-good arm detects it.
#
# The control is essential: arm 2 asks the same prompt while the session is still in VRAM. If it
# does not reproduce arm 1 exactly, the server is not reproducible across requests and every later
# comparison is meaningless, so the script stops.
#
# On a hybrid model a session is attention blocks plus a recurrent-state snapshot, stored and lost
# separately. A lost snapshot does not change the text: the engine replays the linear layers from
# an older checkpoint or token zero and gives the same answer, slower. Byte identity therefore
# proves only the blocks, and every snapshot assertion is a counter. Blocks moving without the
# snapshot is a failure here: for a restored 200K session it is the difference between a second
# and a minute of linear replay.
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

# The prompt must span many blocks and exceed a checkpoint interval (2048 tokens by default):
# shorter, it produces no snapshot and every snapshot assertion passes vacuously.
#
# It must also ask a question. An instruction-tuned model may answer a wall of repeated sentences
# with a single EOS, and empty answers would pass the control, so the answer length is checked
# below.
: > "$work/p.txt"
i=0
while [ $i -lt 300 ]; do
    printf 'Section %d. The cache stores conversation prefixes as content-hashed blocks, and the identity of a block includes every token before it. ' "$i" >> "$work/p.txt"
    i=$((i + 1))
done
printf '\nUsing only the passage above, write three sentences about it.\n\nAnswer: ' >> "$work/p.txt"
# The file is sent verbatim through jq's --rawfile. Escaping it into a shell variable doubles the
# backslash jq adds before every quote, and flattening newlines joins the question to the passage
# ("...why it matters.Answer:"), which gets an immediate EOS.

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
# A full host tier drops its least recently used entries, so the session is pushed out by a
# prompt that shares no prefix with it and exceeds the arena's free room. The host tier must be
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
snap() { curl -sS --fail "http://$H:$P/sessions" > "$work/s.json"; }
q()    { jq -r "$1" < "$work/s.json"; }
# Prompt tokens a lookup covered with recurrent state. A snapshot moving out and back only proves
# the bytes travelled; this counter shows a hit actually served it. A restore to a place the chain
# cannot reach passes the first and scores zero here.
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

# A pure-attention model has no snapshots and is not failed for it; a hybrid model with zero
# snapshot slots is a misconfiguration and is reported.
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

# An empty answer would make every comparison below pass vacuously.
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
# A run that moved nothing is a failure: every arm below would be an ordinary device hit.
snap
# A missing field counts as zero, so a changed /sessions layout (`null`) fails rather than passes.
case "$(q .blocks_sum_overlapping.host)" in 0|null|"") false ;; *) true ;; esac || note "the session never left VRAM -- nothing below would test anything
        (--prefix-cache-mib set? is the maintenance pass running?)"
if [ "$HYBRID" = 1 ]; then
    snap
    # Snapshots live in host memory (the checkpoint store) from the moment they are saved, so the
    # session's state must be there.
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
    # A snapshot is restored straight from its slot in the store; the check is whether it was
    # served.
    if [ "$(lin)" -gt "$LIN0" ]; then
        echo "  ok    and the restored state was SERVED  ($(( $(lin) - LIN0 )) prompt tokens covered by it)"
    else
        note "the snapshot came back and no lookup used it -- restored somewhere the chain
        cannot reach, which costs the replay it was moved to avoid"
    fi
fi

# ---- a full host tier ---------------------------------------------------------------------------
# The filler overflows the arena, which drops this session's entries as least recently used. The
# next answer is a recompute or a partial restore, and must equal the cold one either way; a dropped
# entry the cache kept serving would show as another conversation's context.
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

# ---- the differing text, to locate a failure ---------------------------------------------------
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
