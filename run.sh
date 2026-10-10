#!/usr/bin/env bash
set -Eeuo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${ROOT}/config.sh"
CHECK_ONLY=0
SOURCE_ONLY=0
for arg in "$@"; do
  case "${arg}" in
    --check) CHECK_ONLY=1 ;;
    --source-only) SOURCE_ONLY=1 ;;
    --skip-tc) export SKIP_TC=1 ;;
    -h|--help)
      printf '%s\n' 'Demiurge: TrajectoryCrafter -> RGB/Depth -> optional refine -> alignment -> collect' \
        'Usage: bash run.sh [--check | --source-only] [--skip-tc]' \
        'Options are configured in config.sh or supplied as environment variables.' \
        'Examples: N=8 GPU_ID=1 bash run.sh; REFINE_ITERS=2 bash run.sh' \
        '--check: static file/configuration check only; no torch import or GPU access.' \
        '--source-only: source syntax and release hygiene; no models or Conda required.'
      exit 0 ;;
    *) echo "Unknown argument: ${arg}" >&2; exit 2 ;;
  esac
done
cd "${ROOT}"
DS_ROOT="${ROOT}/code/DiffSynth-Studio"
TC_ROOT="${ROOT}/code/TrajectoryCrafter"
VALIDATE_DIR="${DS_ROOT}/examples/wanvideo/model_training/validate_full"
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
unset PYTHONHOME PYTHONPATH
PYTHON_CHECK="${PYTHON_CHECK:-python3}"
if [[ "${SOURCE_ONLY}" -eq 1 ]]; then
  exec "${PYTHON_CHECK}" -B "${ROOT}/tools/check.py" --source-only
fi
"${PYTHON_CHECK}" -B "${ROOT}/tools/check.py"
if [[ "${CHECK_ONLY}" -eq 1 ]]; then exit 0; fi

export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export DIFFSYNTH_SKIP_DOWNLOAD=true MODELSCOPE_OFFLINE=1 HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1
export XDG_CACHE_HOME="${ROOT}/cache"
export HF_HOME="${ROOT}/cache/huggingface"
export MODELSCOPE_CACHE="${ROOT}/cache/modelscope"
export TORCH_HOME="${ROOT}/cache/torch"
export TORCH_EXTENSIONS_DIR="${ROOT}/cache/torch_extensions"
export TRITON_CACHE_DIR="${ROOT}/cache/triton"
export MPLCONFIGDIR="${ROOT}/cache/matplotlib"
mkdir -p "${OUT_DIR}" "${LOG_DIR}" "${XDG_CACHE_HOME}"
exec 9>"${OUT_DIR}/.demiurge.lock"
flock -n 9 || { echo "Another run is using OUT_DIR=${OUT_DIR}" >&2; exit 1; }
TS="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${LOG_DIR}/run_${TS}_idx${START_IDX}-${N}_$$.log"
exec > >(tee -a "${LOG_FILE}") 2>&1
trap 'code=$?; echo "[FATAL] line ${LINENO}, exit=${code}; log=${LOG_FILE}"; exit "${code}"' ERR
printf '%s\n' "Demiurge ${TS}" "IDX=${START_IDX}..${N} REFINE_ITERS=${REFINE_ITERS} GPU_ID=${GPU_ID}" \
  "INPUT=${TC_VIDEO_PATH}" "OUT_DIR=${OUT_DIR}" "TC_ENV=${TC_ENV} DS_ENV=${DS_ENV}" "LOG=${LOG_FILE}"

in_env() (
  env_name="$1"; work_dir="$2"; shift 2
  source "${CONDA_BASE}/etc/profile.d/conda.sh"
  conda activate "${env_name}"
  export PYTHONPATH="${work_dir}"
  cd "${work_dir}"
  "$@"
)
run_step() {
  local name="$1" t0; shift
  t0="$(date +%s)"
  echo "----- ${name} @ $(date '+%F %T') -----"
  "$@"
  echo "----- ${name}: $(( $(date +%s) - t0 ))s -----"
}
ds_python() { in_env "${DS_ENV}" "${DS_ROOT}" python "$@"; }

for ((IDX=START_IDX; IDX<=N; IDX++)); do
  echo "========== ROUND ${IDX}/${N} =========="
  tc_inputs="$(IDX="${IDX}" "${PYTHON_CHECK}" -B "${DS_ROOT}/gen_source_configs.py")"
  export SOURCE_CONFIGS="$(printf '%s\n' "${tc_inputs}" | sed -n '1p')"
  export WARP_TARGET_POSE_STR="$(printf '%s\n' "${tc_inputs}" | sed -n '2p')"
  export SOURCE_BASE_PATH="${OUT_DIR}" TC_OUT_DIR="${OUT_DIR}"
  echo "SOURCE_CONFIGS=${SOURCE_CONFIGS}"
  echo "WARP_TARGET_POSE=${WARP_TARGET_POSE_STR}"
  if [[ "${SKIP_TC}" -eq 0 ]]; then
    run_step "TrajectoryCrafter" in_env "${TC_ENV}" "${TC_ROOT}" bash "${TC_ROOT}/run_cuda.sh"
  fi
  rm -f -- "${OUT_DIR}/video_Wan2.1-Fun-1.3B-InP.mp4" \
    "${OUT_DIR}/video_Wan2.1-Fun-1.3B-InP-rgb.npz" "${OUT_DIR}/depth_gray_video_1.mp4" \
    "${OUT_DIR}/video_single_channel_expanded.npz" "${OUT_DIR}/poisson_aligned_metric.mp4" \
    "${OUT_DIR}/poisson_aligned_metric.npz" "${OUT_DIR}/poisson_blend_naive.mp4" \
    "${OUT_DIR}/poisson_blend_naive.npz"
  run_step "RGB generate" ds_python "${VALIDATE_DIR}/Wan2.1-Fun-1.3B-InP-rgb_generate_cuda.py"
  run_step "Depth generate" ds_python "${VALIDATE_DIR}/Wan2.1-Fun-1.3B-InP-depth_generate_cuda.py"
  for ((IT=1; IT<=REFINE_ITERS; IT++)); do
    run_step "RGB refine ${IT}/${REFINE_ITERS}" ds_python "${VALIDATE_DIR}/Wan2.1-Fun-1.3B-InP-rgb_refine_cuda.py"
    run_step "Depth regenerate ${IT}/${REFINE_ITERS}" ds_python "${VALIDATE_DIR}/Wan2.1-Fun-1.3B-InP-depth_generate_cuda.py"
  done
  run_step "Depth align" ds_python "${DS_ROOT}/video_depth_align.py"
  run_step "RGB align" ds_python "${DS_ROOT}/video_rgb_align.py"
  export IDX
  run_step "Collect" bash "${DS_ROOT}/copy_file.sh"
done
echo "ALL ROUNDS DONE: IDX=${START_IDX}..${N}, OUT_DIR=${OUT_DIR}"
