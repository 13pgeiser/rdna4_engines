# common.sh -- shared helpers for the run_*.sh entry points (source it, don't execute it)
[ "${BASH_SOURCE[0]}" != "$0" ] || { echo "common.sh is sourced, not executed" >&2; exit 1; }

# -------------------------------
# GPU arch detection (run_hipfire.sh, run_radiance.sh, run_fastflowlm.sh)
# Card identity by PCI ID from /sys; the amdgpu dmesg GFX IP block version
# ("amdgpu <bdf>: ... <gfx_vX_Y_Z> ...") is a family fallback for unknown IDs
# (the IP version alone is not the target, e.g. both the R9700 and the Krackan
# report a plain x.y.0).
# -------------------------------
# gpu_arch_gate <caller> <accepted-arch>...
# Scans the AMD DRM cards and maps each to an arch (0x7551 -> gfx1201,
# 0x1114 -> gfx1152, unknown ID -> the dmesg IP family). Exits early unless the
# box has exactly one card of an accepted arch; on success sets GPU_ARCH and
# prints it.
gpu_arch_gate() {
    local caller="$1"; shift
    local want="" a dev DMSGS supported AMD_IDS id bdf ip arch
    for a in "$@"; do want="${want:+$want or }$a"; done
    DMSGS="$( { dmesg 2>/dev/null || sudo dmesg 2>/dev/null; } 2>/dev/null)"
    for dev in /sys/class/drm/card*/device; do
        [ "$(cat "$dev/vendor" 2>/dev/null)" = "0x1002" ] || continue
        id="$(cat "$dev/device" 2>/dev/null)"
        case " $AMD_IDS " in *" $id "*) ;; *) AMD_IDS="$AMD_IDS $id" ;; esac
        bdf="$(basename "$(readlink -f "$dev")")"
        ip="$(printf '%s\n' "$DMSGS" | awk -v b="amdgpu $bdf:" 'index($0, b) && match($0, /gfx_v[0-9]+_[0-9]+_[0-9]+/) { print substr($0, RSTART, RLENGTH); exit }')"
        case "$id" in
            0x7551) arch=gfx1201 ;;        # Radeon AI PRO R9700
            0x1114) arch=gfx1152 ;;        # Ryzen AI 7 350 (Krackan Point)
            *)
                case "$ip" in
                    gfx_v12_0_*) arch=gfx1201 ;;   # RDNA4 family
                    gfx_v11_5_*) arch=gfx1152 ;;   # RDNA3.5 family (e.g. RX 9060)
                    *) continue ;;
                esac ;;
        esac
        case " $* " in
            *" $arch "*) case " $supported " in *" $arch "*) ;; *) supported="$supported $arch" ;; esac ;;
        esac
    done
    supported="${supported# }"
    AMD_IDS="${AMD_IDS# }"
    if [ -z "$supported" ]; then
        echo "no supported GPU: $caller needs $want (AMD cards found: ${AMD_IDS:-none})" >&2
        exit 1
    fi
    if [ "$(echo $supported | wc -w)" -gt 1 ]; then
        echo "ambiguous: multiple supported GPUs found ($supported); narrow to one first" >&2
        exit 1
    fi
    GPU_ARCH="$supported"
    echo "GPU arch: $GPU_ARCH"
}

# -------------------------------
# model selection (run_hipfire.sh, run_radiance.sh, run_fastflowlm.sh)
# -------------------------------
# select_model <header> <arg> <label>...
# Picks one of the labels and sets MODEL_IDX to its 0-based index: <arg> when
# non-empty (a menu index, or a substring matching exactly one label), else an
# interactive menu, else label 1 when non-interactive. Prints the options and
# exits on failure.
select_model() {
    local header="$1" arg="$2"
    shift 2
    local labels=("$@")
    local n="${#labels[@]}" how pick hits=() i
    if [ -n "$arg" ]; then
        pick="$arg"; how=arg
    elif [ ! -t 0 ]; then
        pick=1
    else
        how=menu
        echo "$header"
        for i in "${!labels[@]}"; do echo "     $((i+1))) ${labels[$i]}"; done
        read -rp "Select model [1-$n] (default 1): " pick
        pick=${pick:-1}
    fi
    if [[ "$pick" =~ ^[0-9]+$ ]] && (( pick >= 1 && pick <= n )); then
        MODEL_IDX=$((pick-1))
        return 0
    fi
    for i in "${!labels[@]}"; do
        [[ "${labels[$i]}" == *"$pick"* ]] && hits+=("$i")
    done
    if (( ${#hits[@]} == 1 )); then
        MODEL_IDX="${hits[0]}"
        return 0
    fi
    if [ "$how" = arg ]; then
        echo "no model matching '$pick'; options:" >&2
        for i in "${!labels[@]}"; do echo "     $((i+1))) ${labels[$i]}" >&2; done
    else
        echo "invalid selection: $pick" >&2
    fi
    exit 1
}
