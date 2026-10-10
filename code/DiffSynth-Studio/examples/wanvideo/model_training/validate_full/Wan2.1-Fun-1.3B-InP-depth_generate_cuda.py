# Modified for Demiurge; see THIRD_PARTY_NOTICES.md.
import os
import torch
from diffsynth.utils.data import VideoData
from diffsynth.core import load_state_dict
from diffsynth.pipelines.wan_video import WanVideoPipeline, ModelConfig
import numpy as np


# H20 + cuBLASLt 12.1 workaround for integer division by zero in bf16 GEMM at N=9216.
import torch.nn.functional as _F

_orig_linear = _F.linear
_BAD_N = {9216}

def _safe_linear(input, weight, bias=None):
    if weight.dtype is torch.bfloat16 and weight.shape[0] in _BAD_N:
        out = _orig_linear(input.half(), weight.half(),
                           None if bias is None else bias.half())
        return out.to(input.dtype)
    return _orig_linear(input, weight, bias)

_F.linear = _safe_linear
torch.nn.functional.linear = _safe_linear
print("[h20_fix] patched F.linear for N in", _BAD_N)

# Shared output root supplied by the workflow environment.
OUT_DIR     = os.environ["OUT_DIR"]
TC_EXP_NAME = os.environ.get("TC_EXP_NAME", "test")
TC_EXP_DIR  = os.path.join(OUT_DIR, TC_EXP_NAME)
os.makedirs(OUT_DIR, exist_ok=True)
print(f"[depth_generate] OUT_DIR    = {OUT_DIR}")
print(f"[depth_generate] TC_EXP_DIR = {TC_EXP_DIR}")


mode = 'depth_gen'

MB = os.environ["DIFFSYNTH_MODEL_BASE_PATH"]
CONV = f"{MB}/DiffSynth-Studio/Wan-Series-Converted-Safetensors"

pipe = WanVideoPipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    mode=mode,
    model_configs=[
        ModelConfig(path=f"{MB}/PAI/Wan2.1-Fun-1.3B-InP/diffusion_pytorch_model.safetensors"),
        ModelConfig(path=f"{CONV}/models_t5_umt5-xxl-enc-bf16.safetensors"),
        ModelConfig(path=f"{CONV}/Wan2.1_VAE.safetensors"),
        ModelConfig(path=f"{CONV}/models_clip_open-clip-xlm-roberta-large-vit-huge-14.safetensors"),
    ],
    tokenizer_config=ModelConfig(path=f"{MB}/Wan-AI/Wan2.1-T2V-1.3B/google/umt5-xxl"),
)

state_dict = load_state_dict(os.environ["DS_DEPTH_CKPT"])
pipe.dit.load_state_dict(state_dict)



# RGB comes from generation/refinement; depth and both masks come from TC.
input_video_1 = VideoData(os.path.join(OUT_DIR, "video_Wan2.1-Fun-1.3B-InP.mp4"), height=480, width=832)
input_video_1 = [input_video_1[i] for i in range(81)]
input_video_2 = os.path.join(TC_EXP_DIR, "render_depth.npz")
input_video_3 = VideoData(os.path.join(TC_EXP_DIR, "mask.mp4"), height=480, width=832)
input_video_3 = [input_video_3[i] for i in range(81)]
input_video_4 = VideoData(os.path.join(TC_EXP_DIR, "mask.mp4"), height=480, width=832)
input_video_4 = [input_video_4[i] for i in range(81)]


video = pipe(
    prompt="A reasonable, clean scene.",
    negative_prompt=os.environ.get(
        "DEMIURGE_NEGATIVE_PROMPT",
        "Vivid colors, overexposure, static, unclear details, subtitles, style, works, paintings, images, "
        "stillness, overall gray cast, worst quality, low quality, JPEG compression artifacts, ugly, "
        "incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, "
        "malformed limbs, fused fingers, motionless frames, cluttered background, three legs, "
        "many people in the background, walking backwards",
    ),
    seed=59, tiled=True,
    input_video_1=input_video_1,
    input_video_2=input_video_2,
    input_video_3=input_video_3,
    input_video_4=input_video_4,
    num_frames=81,
    output_type='floatpoint',
)


# Save grayscale video and the expanded three-channel depth array in OUT_DIR.
video = (video.float().cpu() + 1) / 2
video = video.squeeze(0)
channel_mean = torch.mean(video, dim=0)

channel_mean = (channel_mean - channel_mean.min()) / (channel_mean.max() - channel_mean.min())

video_gray_np = (channel_mean.cpu().numpy() * 255).astype(np.uint8)

import imageio

fps = 15
out_path = os.path.join(OUT_DIR, "depth_gray_video_1.mp4")
with imageio.get_writer(out_path, fps=fps, codec='libx264', pixelformat='gray') as writer:
    for frame in video_gray_np:
        writer.append_data(frame)
print(f"Grayscale video saved: {out_path}")

video_single_channel_expanded = channel_mean.repeat(3, 1, 1, 1)
video_np_3ch = video_single_channel_expanded.cpu().numpy()

save_path = os.path.join(OUT_DIR, "video_single_channel_expanded.npz")
np.savez_compressed(
    save_path,
    video=video_np_3ch,
    fps=15,
    dtype=str(video_np_3ch.dtype),
)
print(f"Expanded single-channel NPZ saved: {save_path}")
