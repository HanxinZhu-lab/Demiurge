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
print(f"[rgb_refine] OUT_DIR    = {OUT_DIR}")
print(f"[rgb_refine] TC_EXP_DIR = {TC_EXP_DIR}")


import random
seed = random.randint(0, 2**16 - 1)
print(f"[seed] this run = {seed}")


mode = 'rgb_refine'

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
state_dict = load_state_dict(os.environ["DS_RGB_REFINE_CKPT"])
pipe.dit.load_state_dict(state_dict)



# TC supplies RGB and masks; DiffSynth supplies generated depth.
input_video_1 = os.path.join(TC_EXP_DIR, "render_rgb_D.npz")
input_video_2 = os.path.join(OUT_DIR,   "video_single_channel_expanded.npz")
input_video_3 = VideoData(os.path.join(TC_EXP_DIR, "mask_D.mp4"), height=480, width=832)
input_video_3 = [input_video_3[i] for i in range(81)]
input_video_4 = VideoData(os.path.join(TC_EXP_DIR, "mask_D.mp4"), height=480, width=832)
input_video_4 = [input_video_4[i] for i in range(81)]


video = pipe(
    prompt=os.environ.get(
        "DEMIURGE_RGB_PROMPT",
        "A blue bird stands beside a large, smooth apple resting on a moss-covered rock. "
        "The small bird has bright blue plumage, from a pale powder-blue crown to deep cobalt wings, "
        "with alert black eyes and slender gray feet. The oversized apple has rich dark-red skin, "
        "blending warm red-orange and yellow-green highlights near its short woody stem, dotted with pale specks. "
        "Both sit on thick green moss covered with gray-green lichen. The background is a soft blur of "
        "green and gray tones, suggesting a forest setting. The striking contrast between the bird's cool blue "
        "and the apple's warm red creates a tranquil, rich natural scene.",
    ),
    negative_prompt=os.environ.get(
        "DEMIURGE_NEGATIVE_PROMPT",
        "Vivid colors, overexposure, static, unclear details, subtitles, style, works, paintings, images, "
        "stillness, overall gray cast, worst quality, low quality, JPEG compression artifacts, ugly, "
        "incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, "
        "malformed limbs, fused fingers, motionless frames, cluttered background, three legs, "
        "many people in the background, walking backwards",
    ),
    seed=seed, tiled=True,
    input_video_1=input_video_1,
    input_video_2=input_video_2,
    input_video_3=input_video_3,
    input_video_4=input_video_4,
    num_frames=81,
    output_type='floatpoint',
)


# Save RGB video and the RGB array in OUT_DIR.
video_rgb = video.float().cpu()
if video_rgb.dim() == 5:
    video_rgb = video_rgb.squeeze(0)
assert video_rgb.dim() == 4 and video_rgb.shape[0] == 3, \
    f"unexpected video shape: {tuple(video_rgb.shape)}"

video_rgb = ((video_rgb + 1.0) / 2.0).clamp_(0.0, 1.0)

video_rgb_np_uint8 = (
    video_rgb.permute(1, 2, 3, 0).cpu().numpy() * 255.0
).round().astype(np.uint8)
print(f"RGB video array shape: {video_rgb_np_uint8.shape}")

import imageio
fps = 15
out_mp4 = os.path.join(OUT_DIR, "video_Wan2.1-Fun-1.3B-InP.mp4")
with imageio.get_writer(
    out_mp4,
    fps=fps,
    codec="libx264",
    pixelformat="yuv420p",
    quality=5,
) as writer:
    for frame in video_rgb_np_uint8:
        writer.append_data(frame)
print(f"RGB video saved (lossy): {out_mp4}")

video_rgb_np = (
    video_rgb.permute(1, 2, 3, 0).cpu().numpy().astype(np.float16)
)
print(f"NPZ array shape: {video_rgb_np.shape}, dtype={video_rgb_np.dtype}")

save_path = os.path.join(OUT_DIR, "video_Wan2.1-Fun-1.3B-InP-rgb.npz")
np.savez_compressed(
    save_path,
    rgb=video_rgb_np,
    fps=int(fps),
    dtype=str(video_rgb_np.dtype),
)
print(f"RGB NPZ saved (lossless compression of float16 data): {save_path}")
