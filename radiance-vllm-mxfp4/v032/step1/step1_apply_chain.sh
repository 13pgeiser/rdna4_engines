#!/bin/bash
# Step 1: apply GGZ14/vllm-mxfp4 patches (Dockerfile order, then serve-mxfp4.sh order) to a vLLM
# source checkout used as fake site-packages, one git commit per patch.
# usage: step1_apply_chain.sh <worktree> <ggz14-repo> <logfile>
set -u
WT=$1; G=$2; LOG=$3
R=$(cd "$(dirname "$0")/.." && pwd)
PY=$R/tools/py312/python/bin/python3
: > "$LOG"
cd "$WT" || exit 1

commit() { git add -A >/dev/null; git commit -q --allow-empty -m "$1" -m "$2" ; }

# 0. Dockerfile COPY of runtime modules (before the patch loop)
cp "$G"/radiance_amdsmi.py "$G"/radiance_amdsmi.pth "$G"/radiance_kernels.py "$G"/radiance_vit_attn.py \
   "$G"/radiance_allreduce.py "$G"/radiance_draft.py "$G"/radiance_draft_gpu.py "$G"/radiance_drafthead.py \
   "$G"/radiance_gemm.py "$G"/radiance_r4d_attn.py "$G"/radiance_gdn.py "$G"/radiance_w4.py "$G"/radiance_mxfp4.py .
commit "radiance: Dockerfile COPY runtime modules" "source: GGZ14/vllm-mxfp4 Dockerfile r.228-231"

DOCKER="patch_gfx1201 patch_radiance_dispatch patch_skinny_gemm patch_unified_attention_lds \
 patch_gdn_wmma patch_preshuffle patch_radiance_fusion install_radiance_hooks \
 patch_unpad patch_mtp_mm_mask patch_mtp_loopbreak patch_qwen3_toolparse patch_from_json_filter \
 patch_dynamo_metrics patch_conv1d_blockn patch_r4d patch_dflash_base patch_dflash2 \
 patch_dflash_fused_kv_fp8 patch_dflash_w4 patch_gdn_metadata \
 patch_quark_mxfp4 patch_ar_maxbytes patch_topk_triton_rows patch_qwen3_thinkoff"
SERVE="patch_quark_mxfp4 patch_nvfp4_mxfp4 patch_tp3_pad patch_ar_maxbytes patch_topk_triton_rows \
 patch_dflash_calib patch_dflash_mxfp4_kv patch_rmsquant_fusion patch_verify_head patch_kv_group_size \
 patch_topk_composite patch_gdn_shared_build patch_dflash_selector_topk patch_gdn_merge_inproj \
 patch_dynwidth patch_async_dynwidth patch_step_trace patch_ar_geometry patch_ar_qbits patch_ar_3rank \
 patch_gdn_glue patch_qwen3_thinkoff"

run() {  # $1 = phase, $2 = patch
  out=$("$PY" "$R/tools/run_patch.py" "$WT" "$G/$2.py" ${SKIP_NONVLLM:+--skip-nonvllm} 2>&1); rc=$?
  ok=$(grep -c '^  OK\|OK ' <<<"$out"); fail=$(grep -c 'FAIL' <<<"$out"); noop=$(grep -c 'NOOP' <<<"$out")
  printf '%s\t%s\trc=%s\tOK=%s\tNOOP=%s\tFAIL=%s\n' "$1" "$2" "$rc" "$ok" "$noop" "$fail" | tee -a "$LOG"
  sed 's/^/    | /' <<<"$out" >> "$LOG"
  commit "radiance[$1]: $2 (rc=$rc OK=$ok NOOP=$noop FAIL=$fail)" "$out"
}
for p in $DOCKER; do run docker "$p"; done
for p in $SERVE; do run serve "$p"; done

# serve-mxfp4.sh copies these modules into site-packages after the patch run
cp "$G"/radiance_nvfp4.py "$G"/radiance_mxfp4.py "$G"/radiance_gdn.py "$G"/radiance_gdn_lazy.py \
   "$G"/radiance_rmsquant.py "$G"/radiance_drafthead.py "$G"/radiance_verifyhead.py "$G"/radiance_gdnmerge.py \
   "$G"/radiance_aroverlap.py "$G"/radiance_topk.py "$G"/radiance_arnq.py "$G"/radiance_tp3pad.py .
commit "radiance: serve-mxfp4.sh cp runtime modules" "source: serve-mxfp4.sh r.1049-1051"
