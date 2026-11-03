#!/bin/bash
# FastFlowLM (FLM): chat models running on the Ryzen AI XDNA2 NPU, built from the FastFlowLM/ subtree.
# The ornith:35b-a3b tag is not in any FLM registry: it converts the public GGUF to Q4NX
# (FLM_Q4NX_Converter/ subtree) and registers it locally — see the ornith block below.
CURDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$CURDIR/common.sh" || exit 1

# -------------------------------
# GPU arch gate: gfx1152 only (early exit; detection in common.sh)
# -------------------------------
gpu_arch_gate run_fastflowlm.sh gfx1152

# -------------------------------
# NPU gate: FLM runs on the XDNA2 NPU, not the GPU
# -------------------------------
if ! lsmod | grep -q '^amdxdna'; then
    echo "amdxdna kernel module not loaded: FLM needs the NPU driver (kernel 7.0+ in-tree, or amdxdna-dkms)" >&2
    exit 1
fi
if ! ls /dev/accel/accel* >/dev/null 2>&1; then
    echo "no NPU device node under /dev/accel: nothing to serve on" >&2
    exit 1
fi

# -------------------------------
# model selection (multiple choice)
# -------------------------------
# TAGS - registry tags (`flm serve`/`pull` take this)
TAGS=(
      "gemma4-it:12b"
      "qwen3.5:9b"
      "nanbeige4.1:3b"
      "phi4-mini-it:4b"
      "gpt-oss:20b"
      "qwen3.6-moe:35b-a3b"
      "qwen3.8-dflash2:27b"   # source catalog only (unreleased in the 1.0.11 tarballs)
      "qwen3.8-mtp:27b"       # the 27B with MTP speculative decoding
      "ornith:35b-a3b"        # not in the FLM registry: GGUF->Q4NX convert + local registration (one-time ~22 GB download + CPU convert)
)

select_model "Available FastFlowLM models:" "$1" "${TAGS[@]}"
TAG="${TAGS[$MODEL_IDX]}"
echo "serving $TAG"

# -------------------------------
# model storage (host /models/flm, bind-mounted at /models in the container)
# -------------------------------
MODELS_DIR=/models/flm
# /models can end up root-owned (a sudo cleanup, or bind dirs docker created
# as root): claim the model dir and the three ornith dirs up front so the
# user-land fetch/cp steps below can write them; a cheap no-op when the tree
# is already ours
sudo mkdir -p "$MODELS_DIR" "$MODELS_DIR-ornith" "$MODELS_DIR-ornith-catalog" "$MODELS_DIR-xclbins"
sudo chown -R "$USER:users" /models/flm-* 2>/dev/null || true

# -------------------------------
# docker image (built from the FastFlowLM/ subtree; the tag is the subtree's git
# tree hash, so a ./git.sh sync that moves FastFlowLM/ builds a fresh image, old
# ones survive for A/B testing, and unrelated commits reuse the existing one)
# -------------------------------
FLM_SRC_REV="$(git -C "$CURDIR" rev-parse --short=8 HEAD:FastFlowLM)"
IMAGE="local/fastflowlm-src:$FLM_SRC_REV"
HAVE_REV="$(docker image inspect --format '{{index .Config.Labels "flm.src-rev"}}' "$IMAGE" 2>/dev/null)"
if [ "$HAVE_REV" = "$FLM_SRC_REV" ]; then
    echo "image $IMAGE already present (subtree rev $FLM_SRC_REV), skipping build"
else
    # the third_party submodules are gitlinks and arrive empty: clone them at the pinned revisions
    docker build --progress=plain -t "$IMAGE" \
        --build-arg TOKENIZERS_REVISION="$(git -C "$CURDIR" ls-tree HEAD FastFlowLM/third_party/tokenizers-cpp | awk '{print $3}')" \
        --build-arg XGRAMMAR_REVISION="$(git -C "$CURDIR" ls-tree HEAD FastFlowLM/third_party/xgrammar | awk '{print $3}')" \
        --label flm.src-rev="$FLM_SRC_REV" \
        -f "$CURDIR/Dockerfile.fastflowlm-src" "$CURDIR/FastFlowLM" \
        || { echo "image build failed" >&2; exit 1; }
fi

# -------------------------------
# ornith:35b-a3b — not in any FLM registry. One-time setup (re-runs are no-ops).
# By default it downloads the public GGUF (Q4_K_M, the only quant whose dequant
# fits 32 GB), converts it with the FLM_Q4NX_Converter/ subtree, and repairs
# the ssm alpha/beta tables the converter emits in a form the engine cannot
# load (fastflowlm-scripts/fix_q4nx_ssm.py — see its docstring). ORNITH_PKG=
# <hf-repo> fetches a prebuilt NPU2 package instead of converting; it is run
# through the same repair (a no-op for a correctly-converted package), but
# note Atomic-Germ/Ornith-1.5-35B-A3B-NPU2 is NOT loadable by this engine:
# its tensors are Q8-repacked (the engine expects the Q4NX layout) and it
# ships an extra MTP layer 40 the engine has no slot for. Either way the model
# is registered locally — model files, a catalog override (passed via
# FLM_CONFIG_PATH) and an xclbins copy under the new model name — all under
# /models so they survive image rebuilds, running on the same qwen3.6-moe NPU2
# engine. The model's config.json/tokenizer files must be in FLM registry
# format (flat config with xclbin names + NPU addr fields + QWEN3_6_MOE_*
# constants): from a pulled Qwen3.6-35B-A3B-NPU2 if present, else the three
# small files are fetched from the registry — the HF-native ones segfault the
# load. chat_template.jinja is the froggeric fixed template shared with the
# radiance-vllm flow. VISION=0 skips the vision tower; ORNITH_FORCE overrides
# the converter's arch detection.
# -------------------------------
ORNITH=0
if [ "$TAG" = "ornith:35b-a3b" ]; then
    ORNITH=1
    VISION="${VISION:-1}"
    O_NAME="Ornith-1.5-35B-A3B"
    O_WORK=/models/flm-ornith
    O_CAT=/models/flm-ornith-catalog
    O_XCL=/models/flm-xclbins
    O_GGUF="$O_WORK/Ornith-1.5-35B-Q4_K_M.gguf"
    O_MM="$O_WORK/mmproj-Ornith-1.5-35B-BF16.gguf"
    O_CONV="$CURDIR/FLM_Q4NX_Converter"
    O_GGUF_URL="https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B-GGUF/resolve/main"
    # local GGUF->Q4NX conversion is the default; ORNITH_PKG=<hf-repo> fetches
    # a prebuilt NPU2 package instead
    ORNITH_PKG="${ORNITH_PKG:-}"
    O_PKG_URL="https://huggingface.co/$ORNITH_PKG/resolve/main"

    fetch() {  # $1=url $2=dest — skips when already complete, else resumes.
        # The size probe is a GET range request (1 byte): HF answers HEAD on
        # resolve/main with a 404, so a content-length probe never matches.
        local want
        want=$(curl -fsS -r 0-0 -D - -o /dev/null -L "$1" 2>/dev/null \
            | awk '{ sub(/\r$/, "") } tolower($1) == "content-range:" { split($3, a, "/"); if (a[2] ~ /^[0-9]+$/) print a[2] }')
        if [ -f "$2" ] && [ "$(stat -c%s "$2")" = "$want" ]; then
            echo "  present: $(basename "$2")"
        else
            curl -fL --retry 3 -C - -o "$2" "$1" || { echo "download failed: $1" >&2; return 1; }
        fi
    }

    mkdir -p "$O_WORK" "$O_CAT" || exit 1

    if [ -n "$ORNITH_PKG" ]; then
        # 1. prebuilt NPU2 package (~24 GB, mixed Q8_0/Q4_1/BF16): fetch the
        #    files the catalog entry names
        mkdir -p "$O_WORK/pkg"
        fetch "$O_PKG_URL/model.q4nx" "$O_WORK/pkg/model.q4nx" || exit 1
        if [ "$VISION" = 1 ]; then
            fetch "$O_PKG_URL/vision_weight.q4nx" "$O_WORK/pkg/vision_weight.q4nx" || exit 1
        fi
        for f in config.json tokenizer.json tokenizer_config.json chat_template.jinja; do
            fetch "$O_PKG_URL/$f" "$O_WORK/pkg/$f" || exit 1
        done
        # 2. repair ssm tables if needed (no-op for a correctly-converted package)
        python3 "$CURDIR/fastflowlm-scripts/fix_q4nx_ssm.py" "$O_WORK/pkg/model.q4nx" || exit 1
        O_LANG_SRC="$O_WORK/pkg/model.q4nx"
        O_VIS_SRC="$O_WORK/pkg/vision_weight.q4nx"
    else
        # 1. inputs: Q4_K_M is the only quant that fits 32 GB (the converter
        #    dequantizes it to FP32 and re-quantizes to Q4NX); plus mmproj
        #    (vision). The GGUFs are only fetched while a conversion is
        #    pending — safe to delete them afterwards to reclaim ~23 GB
        #    without triggering re-downloads.
        if [ ! -f "$O_WORK/lang/model.q4nx" ] || { [ "$VISION" = 1 ] && [ ! -f "$O_WORK/vision/model.q4nx" ]; }; then
            fetch "$O_GGUF_URL/Ornith-1.5-35B-Q4_K_M.gguf" "$O_GGUF" || exit 1
            if [ "$VISION" = 1 ]; then
                fetch "$O_GGUF_URL/mmproj-Ornith-1.5-35B-BF16.gguf" "$O_MM" || exit 1
            fi
        fi
        # 1b. aux files in FLM registry format — the engine needs the flat
        #     config (xclbin names, NPU addr fields, QWEN3_6_MOE_* constants)
        #     and a tokenizer_config without an embedded chat_template; the
        #     HF-native files the model repo ships lack those fields and the
        #     engine segfaults in load. From a pulled reference model if
        #     present, else just the three small files from the registry (no
        #     35 GB pull needed). Same geometry, so the reference's files are
        #     the right ones.
        if [ ! -f "$MODELS_DIR/models/Qwen3.6-35B-A3B-NPU2/config.json" ]; then
            mkdir -p "$O_WORK/ref"
            # file_url is the HF tree-API base; the download base is the
            # resolve URL derived from it
            refurl=$(docker run --rm "$IMAGE" python3 -c 'import json; print(json.load(open("/opt/fastflowlm/model_list.json"))["models"]["qwen3.6-moe"]["35b-a3b"]["file_url"])' \
                | sed -E 's#^https://huggingface\.co/api/models/#https://huggingface.co/#; s#/tree/#/resolve/#')
            for f in config.json tokenizer.json tokenizer_config.json; do
                fetch "$refurl/$f" "$O_WORK/ref/$f" || exit 1
            done
        fi
        # chat template: the froggeric fixed one, shared with the radiance-vllm flow
        if [ -f "$CURDIR/radiance-vllm-mxfp4-scripts/chat_template.jinja" ]; then
            cp -f "$CURDIR/radiance-vllm-mxfp4-scripts/chat_template.jinja" "$O_WORK/chat_template.jinja"
        else
            fetch "https://huggingface.co/froggeric/Qwen-Fixed-Chat-Templates/resolve/main/chat_template.jinja" "$O_WORK/chat_template.jinja" || exit 1
        fi

        # 2. converter venv (one-time: torch, amd-quark, transformers, gguf from llama.cpp)
        if [ ! -x "$O_CONV/venv/bin/python" ]; then
            ( cd "$O_CONV" && ./setup_venv.sh ) || exit 1
            rmdir "$CURDIR/gguf_files" "$CURDIR/q4nx_files" 2>/dev/null
        fi

        # 3. convert (CPU, slow one-off; the language GGUF auto-detects (arch
        #    qwen35moe), the mmproj needs the arch forced, per the converter's own
        #    usage examples)
        if [ ! -f "$O_WORK/lang/model.q4nx" ]; then
            echo "[ornith] converting language weights (Q4_K_M -> Q4NX, CPU: slow)"
            ( cd "$O_CONV" && venv/bin/python convert.py -i "$O_GGUF" -o "$O_WORK/lang" ${ORNITH_FORCE:+-f "$ORNITH_FORCE"} ) || exit 1
        fi
        if [ "$VISION" = 1 ] && [ ! -f "$O_WORK/vision/model.q4nx" ]; then
            echo "[ornith] converting vision weights"
            ( cd "$O_CONV" && venv/bin/python convert.py -i "$O_MM" -o "$O_WORK/vision" -t vision -f "${ORNITH_FORCE:-qwen35moe}" ) || exit 1
        fi

        # 4. repair the ssm alpha/beta tables: the converter packs them as I8
        #    and stashes the dequantized values in transposed .bf16. side-entries,
        #    while the qwen3.6-moe engine wants a plain BF16 [2048, 32] main entry
        #    (unpatched, load segfaults in qwen3_6_reorder_cpy). Idempotent; the
        #    vision pack has no ssm layers, so this is a no-op for it.
        python3 "$CURDIR/fastflowlm-scripts/fix_q4nx_ssm.py" "$O_WORK/lang/model.q4nx" || exit 1
        if [ "$VISION" = 1 ]; then
            python3 "$CURDIR/fastflowlm-scripts/fix_q4nx_ssm.py" "$O_WORK/vision/model.q4nx" || exit 1
        fi
        O_LANG_SRC="$O_WORK/lang/model.q4nx"
        O_VIS_SRC="$O_WORK/vision/model.q4nx"
    fi

    # 5. model dir: FLM's model_path is CWD-relative "models" and the container
    #    runs from /, so pulled models live in $MODELS_DIR/models/<name> on the host
    O_ROOT="$MODELS_DIR/models"
    if [ ! -d "$O_ROOT" ]; then
        # no model root yet: prefer an existing one (from a prior flm pull);
        # else create the canonical one — a pre-populated /models is not needed:
        # the Qwen3.6 reference is only read for the flm_version stamp (which
        # falls back), and its xclbins come from the image, not /models
        ref=$(find "$MODELS_DIR" -maxdepth 3 -type d -name "Qwen3.6-35B-A3B-NPU2" 2>/dev/null | head -1)
        [ -n "$ref" ] && O_ROOT=$(dirname "$ref")
    fi
    mkdir -p "$O_ROOT/$O_NAME" || exit 1
    cp -f "$O_LANG_SRC" "$O_ROOT/$O_NAME/model.q4nx"
    if [ "$VISION" = 1 ]; then
        cp -f "$O_VIS_SRC" "$O_ROOT/$O_NAME/vision_weight.q4nx"
    fi
    # aux files: a prebuilt package ships its own; the converted model takes
    # them from the FLM registry (reference model, else the fetched copies)
    if [ -n "$ORNITH_PKG" ]; then
        O_AUX2="$O_WORK/pkg"
        O_CHAT2="$O_WORK/pkg/chat_template.jinja"
    else
        if [ -f "$O_ROOT/Qwen3.6-35B-A3B-NPU2/config.json" ]; then
            O_AUX2="$O_ROOT/Qwen3.6-35B-A3B-NPU2"
        elif [ -f "$O_WORK/ref/config.json" ]; then
            O_AUX2="$O_WORK/ref"
        else
            echo "no FLM-format aux files: pull qwen3.6-moe:35b-a3b once" >&2
            exit 1
        fi
        O_CHAT2="$O_WORK/chat_template.jinja"
    fi
    cp -f "$O_AUX2/config.json" "$O_AUX2/tokenizer.json" "$O_AUX2/tokenizer_config.json" "$O_ROOT/$O_NAME/"
    cp -f "$O_CHAT2" "$O_ROOT/$O_NAME/chat_template.jinja"
    # stamp flm_version into the copied config: a transformers-native config has
    # none, so the compatibility check reads "0.0.0" < flm_min_version, marks the
    # model Outdated and runs a file-verification pass that has no records for a
    # local model. Take the value from the reference model's config (it works
    # with this FLM build); fall back to the entry's flm_min_version floor.
    python3 - "$O_ROOT/$O_NAME/config.json" "$O_ROOT/Qwen3.6-35B-A3B-NPU2/config.json" <<'PYEOF' || exit 1
import json, sys
cfg, ref = sys.argv[1], sys.argv[2]
d = json.load(open(cfg))
try:
    v = json.load(open(ref)).get("flm_version")
except (OSError, ValueError):
    v = None
d["flm_version"] = v or "1.0.3"
json.dump(d, open(cfg, "w"), indent=2)
print("[ornith] config: flm_version =", d["flm_version"])
PYEOF

    # 6. catalog override: re-derived from the image each run (tracks rebuilds),
    #    then an "ornith" family is upserted, cloned from qwen3.6-moe:35b-a3b —
    #    same engine (details family qwen3.6-moe), same geometry, dummy url
    #    because the files are local and never pulled
    docker run --rm "$IMAGE" cat /opt/fastflowlm/model_list.json > "$O_CAT/model_list.json" || exit 1
    docker run --rm "$IMAGE" cat /opt/fastflowlm/model_info.json > "$O_CAT/model_info.json" || exit 1
    python3 - "$O_CAT/model_list.json" "$O_NAME" "$VISION" <<'PYEOF' || exit 1
import json, sys
path, name, vision = sys.argv[1], sys.argv[2], sys.argv[3] == "1"
d = json.load(open(path))
e = json.loads(json.dumps(d["models"]["qwen3.6-moe"]["35b-a3b"]))
e["name"] = name
e["url"] = "https://local/or"
e["file_url"] = "https://local/or"
# keep the reference entry's "size" (and everything else): runtime code reads it
# for feature toggles (e.g. enable_tool = size > 800MB)
e.pop("ms_url", None)
if not vision:
    e["files"] = [f for f in e["files"] if f != "vision_weight.q4nx"]
    e["vlm"] = False
d["models"]["ornith"] = {"35b-a3b": e}
json.dump(d, open(path, "w"), indent=2)
print("[ornith] catalog: registered ornith:35b-a3b ->", name)
PYEOF

    # 7. xclbins: kernels are looked up by model name; give Ornith its own copy
    #    of Qwen3.6-35B-A3B's (identical geometry). The host dir is extracted
    #    once from the image and bind-mounted over /opt/fastflowlm/xclbins
    # check the reference folder, not the dir: the sudo mkdir up front leaves
    # an empty one, and an empty O_XCL must still trigger the extraction
    if [ ! -d "$O_XCL/Qwen3.6-35B-A3B-NPU2" ]; then
        mkdir -p "$O_XCL"
        docker run --rm -v "$O_XCL":/out "$IMAGE" sh -c "cp -r /opt/fastflowlm/xclbins/. /out/" || exit 1
    fi
    if [ ! -d "$O_XCL/$O_NAME" ]; then
        cp -r "$O_XCL/Qwen3.6-35B-A3B-NPU2" "$O_XCL/$O_NAME" \
            || { echo "xclbins copy failed" >&2; exit 1; }
    fi
    [ -f "$O_XCL/$O_NAME/layer.xclbin" ] || { echo "xclbins for $O_NAME incomplete" >&2; exit 1; }
fi

# -------------------------------
# model (pulled into host /models/flm; `flm pull` is a no-op when present)
# -------------------------------
if [ "$ORNITH" = 0 ]; then
    docker run --rm -v "$MODELS_DIR":/models "$IMAGE" flm pull "$TAG" \
        || { echo "flm pull $TAG failed (tag not in this FLM build's catalog - check 'flm list')" >&2; exit 1; }
fi

# -------------------------------
# run
# -------------------------------
docker rm -f fastflowlm_server
EXTRA=()
if [ "$ORNITH" = 1 ]; then
    # local catalog (FLM_CONFIG_PATH) + xclbins overlay for the custom model name;
    # the catalog mounts outside /models so it cannot nest under the models bind
    EXTRA=(-e FLM_CONFIG_PATH=/opt/flm-ornith-catalog/model_list.json
           -v "$O_CAT":/opt/flm-ornith-catalog
           -v "$O_XCL":/opt/fastflowlm/xclbins)
fi
# memlock must be unlimited for NPU execution; seccomp unconfined like the GPU servers
docker run --name fastflowlm_server --network host --init --ulimit memlock=-1 \
    --device /dev/accel --security-opt seccomp=unconfined \
    "${EXTRA[@]}" \
    -v "$MODELS_DIR":/models \
    "$IMAGE" flm serve "$TAG" --host 0.0.0.0 -p 9931
