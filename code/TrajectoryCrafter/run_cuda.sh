#!/bin/bash
# Modified for Demiurge; see THIRD_PARTY_NOTICES.md.
# Launch TrajectoryCrafter inference on CUDA.
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

TC_DEVICE="${TC_DEVICE:-cuda:0}"

if [ -z "${SOURCE_CONFIGS:-}" ]; then
    SOURCE_CONFIGS='[]'
fi
SOURCE_BASE_PATH="${SOURCE_BASE_PATH:?Please launch through Demiurge/run.sh}"
if [ -n "${WARP_TARGET_POSE_STR:-}" ]; then
    read -r -a WARP_TARGET_POSE <<< "${WARP_TARGET_POSE_STR}"
else
    WARP_TARGET_POSE=(0.0 -5 0 0 0.0)
fi
OUT_DIR_ARG="${TC_OUT_DIR:?Please launch through Demiurge/run.sh}"
EXP_NAME_ARG="${TC_EXP_NAME:-test}"
VIDEO_PATH_ARG="${TC_VIDEO_PATH:?Please launch through Demiurge/run.sh}"
CKPT="${TC_CKPT:?Please launch through Demiurge/run.sh}"
mkdir -p "${OUT_DIR_ARG}"

python inference_depth_warping_2.py \
    --video_path "${VIDEO_PATH_ARG}" \
    --stride 1 \
    --out_dir "${OUT_DIR_ARG}" \
    --radius_scale 1 \
    --camera 'target' \
    --mode 'gradual' \
    --mask \
    --target_pose 0 0 0 0 0 \
    --traj_txt 'test/trajs/loop2.txt' \
    --video_length 81 \
    --exp_name "${EXP_NAME_ARG}" \
    --source_configs "$SOURCE_CONFIGS" \
    --source_base_path "$SOURCE_BASE_PATH" \
    --warp_target_pose "${WARP_TARGET_POSE[@]}" \
    --device "${TC_DEVICE}" \
    --model_name       "$CKPT/CogVideoX-Fun-V1.1-5b-InP" \
    --transformer_path "$CKPT/TrajectoryCrafter" \
    --blip_path        "$CKPT/blip2-opt-2.7b" \
    --unet_path        "$CKPT/DepthCrafter" \
    --pre_train_path   "$CKPT/stable-video-diffusion-img2vid"
