#!/usr/bin/env bash
DEMIURGE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export DEMIURGE_ROOT

# N is the final round index;
export N="${N:-1}"
export START_IDX="${START_IDX:-1}"
export REFINE_ITERS="${REFINE_ITERS:-0}"
export SKIP_TC="${SKIP_TC:-0}"
export GPU_ID="${GPU_ID:-0}"
export TC_DEVICE="${TC_DEVICE:-cuda:0}"
export TC_EXP_NAME="${TC_EXP_NAME:-test}"
export TC_VIDEO_PATH="${TC_VIDEO_PATH:-${DEMIURGE_ROOT}/assets/p7.mp4}"
export OUT_DIR="${OUT_DIR:-${DEMIURGE_ROOT}/results}"
export LOG_DIR="${LOG_DIR:-${DEMIURGE_ROOT}/logs}"

# Reuse existing environments without installing or upgrading packages.
export TC_ENV="${TC_ENV:-trajcrafter_new}"
export DS_ENV="${DS_ENV:-diffsynth_cuda}"
export CONDA_BASE="${CONDA_BASE:-}"
if [[ -z "${CONDA_BASE}" ]]; then
  for base in "${DEMIURGE_ROOT}/../miniconda3" "${HOME}/miniconda3" "${HOME}/anaconda3"; do
    if [[ -f "${base}/etc/profile.d/conda.sh" ]]; then
      export CONDA_BASE="$(cd "${base}" && pwd)"
      break
    fi
  done
fi

export DIFFSYNTH_MODEL_BASE_PATH="${DIFFSYNTH_MODEL_BASE_PATH:-${DEMIURGE_ROOT}/models/diffsynth}"
export TC_CKPT="${TC_CKPT:-${DEMIURGE_ROOT}/models/trajectorycrafter}"

# Prefer release filenames; retain legacy checkpoints for existing installations.
_demiurge_checkpoint() {
  local preferred="${DIFFSYNTH_MODEL_BASE_PATH}/Demiurge/$1"
  local legacy="${DIFFSYNTH_MODEL_BASE_PATH}/$2"
  if [[ ! -e "${preferred}" && -f "${legacy}" ]]; then
    printf '%s\n' "${legacy}"
  else
    printf '%s\n' "${preferred}"
  fi
}

export DS_RGB_GEN_CKPT="${DS_RGB_GEN_CKPT:-$(_demiurge_checkpoint rgb_generation.safetensors train/Wan2.1-Fun-1.3B-InP_full_rgb_refine/step-27000.safetensors)}"
export DS_RGB_REFINE_CKPT="${DS_RGB_REFINE_CKPT:-$(_demiurge_checkpoint rgb_refinement.safetensors train/Wan2.1-Fun-1.3B-InP_full_rgb_refine-2/step-24000.safetensors)}"
export DS_DEPTH_CKPT="${DS_DEPTH_CKPT:-$(_demiurge_checkpoint depth_completion.safetensors train/Wan2.1-Fun-1.3B-InP_full_depth/step-10000.safetensors)}"
unset -f _demiurge_checkpoint
