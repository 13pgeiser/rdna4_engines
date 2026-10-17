#!/bin/sh
# Keep the caller's process as PID 1 after granting device access.
set -eu

if [ ! -e /dev/kfd ]; then
    printf '%s\n' 'hipfire: /dev/kfd missing: GPU passthrough not configured' >&2
fi

if [ "$#" -eq 0 ]; then
    set -- hipfire serve
else
    case "$1" in
        -*) set -- hipfire "$@" ;;
        *)
            executable=$(command -v "$1" 2>/dev/null || :)
            if [ -z "$executable" ] || [ ! -x "$executable" ]; then
                set -- hipfire "$@"
            fi
            ;;
    esac
fi

# HIPFIRE_SERVE_ARGS: extra `hipfire serve` arguments, split on whitespace
# (no quoting). Passed as an env var because compose implementations differ on
# splitting an empty interpolated `command` string (one turns it into a "" arg).
if [ "$1" = hipfire ] && [ "${2:-}" = serve ] && [ -n "${HIPFIRE_SERVE_ARGS:-}" ]; then
    set -f
    # shellcheck disable=SC2086 # word splitting is the point
    set -- "$@" $HIPFIRE_SERVE_ARGS
    set +f
fi

if [ "$(id -u)" -eq 0 ] && [ "${HIPFIRE_RUN_AS_ROOT:-0}" != 1 ]; then
    export HOME=/home/hipfire
    keep=0
    # Rootless runtimes (crun keep-groups / run.oci.keep_original_groups)
    # preserve the user's host groups but prohibit setgroups.
    [ "$(cat /proc/self/setgroups 2>/dev/null || :)" != deny ] || keep=1

    groups=$(id -g hipfire)
    for node in /dev/kfd /dev/dri/renderD* /dev/dri/card* /dev/accel/accel*; do
        [ -e "$node" ] || continue
        gid=$(stat -c %g "$node" 2>/dev/null) || continue
        # 65534 is the overflow GID: the owning host group is not mapped into
        # this user namespace, so only the inherited groups can grant access.
        if [ "$gid" = 65534 ]; then
            keep=1
            continue
        fi
        case ",$groups," in
            *,"$gid",*) ;;
            *) groups="$groups,$gid" ;;
        esac
    done
    if [ "$keep" = 1 ]; then
        exec setpriv --reuid=hipfire --regid=hipfire --keep-groups -- "$@"
    fi
    exec setpriv --reuid=hipfire --regid=hipfire --groups="$groups" -- "$@"
fi

exec "$@"
