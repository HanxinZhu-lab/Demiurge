#!/usr/bin/env bash
# Modified for Demiurge; see THIRD_PARTY_NOTICES.md.
set -Eeuo pipefail

# Settings can be overridden through environment variables.
IDX="${IDX:-1}"
OUT_DIR="${OUT_DIR:?Please launch through Demiurge/run.sh}"
TC_EXP_NAME="${TC_EXP_NAME:-test}"
TC_EXP_DIR="${OUT_DIR}/${TC_EXP_NAME}"
SKIP_MASK_COPY="${SKIP_MASK_COPY:-0}"      # 1 skips mask_raw_B.mp4 -> mask-IDX.mp4 during phase 2 refinement.

mkdir -p "${OUT_DIR}"

copy_must() {
  local src="$1" dst="$2"
  if [[ ! -f "${src}" ]]; then
    echo "[ERR] missing: ${src}" >&2
    exit 1
  fi
  cp -f "${src}" "${dst}"
  echo "[cp] ${src} -> ${dst}"
}

# Preserve iteration outputs in OUT_DIR with the -${IDX} suffix.
copy_must "${OUT_DIR}/video_Wan2.1-Fun-1.3B-InP.mp4"   "${OUT_DIR}/video_Wan2.1-Fun-1.3B-InP-${IDX}.mp4"
copy_must "${OUT_DIR}/poisson_aligned_metric.npz"      "${OUT_DIR}/poisson_aligned_metric-${IDX}.npz"
copy_must "${OUT_DIR}/poisson_aligned_metric.mp4"      "${OUT_DIR}/poisson_aligned_metric-${IDX}.mp4"
if [[ "${SKIP_MASK_COPY}" -eq 0 ]]; then
  copy_must "${TC_EXP_DIR}/mask_raw_B.mp4"             "${OUT_DIR}/mask-${IDX}.mp4"
else
  echo "[skip] mask_raw_B.mp4 -> mask-${IDX}.mp4 (SKIP_MASK_COPY=1)"
fi
copy_must "${OUT_DIR}/depth_gray_video_1.mp4"          "${OUT_DIR}/depth_gray_video-${IDX}.mp4"
copy_must "${OUT_DIR}/poisson_blend_naive.mp4"         "${OUT_DIR}/poisson_blend_naive-${IDX}.mp4"
copy_must "${OUT_DIR}/poisson_blend_naive.npz"         "${OUT_DIR}/poisson_blend_naive-${IDX}.npz"
