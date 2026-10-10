# -*- coding: utf-8 -*-
# Modified for Demiurge; see THIRD_PARTY_NOTICES.md.
# Inference with optional NPU support.
import torch
from npu_patch import DEFAULT_DEVICE, _HAS_NPU, _HAS_CUDA  # Enable NPU support when available.

from demo import TrajCrafter
import os
import json
from datetime import datetime
import argparse



def get_parser():
    parser = argparse.ArgumentParser()
    ckpt_root = os.environ.get("TC_CKPT", os.path.abspath(os.path.join(os.path.dirname(__file__), "../../models/trajectorycrafter")))

    ## general
    parser.add_argument('--video_path', type=str, help='Input path')
    parser.add_argument(
        '--out_dir', type=str, default='./experiments/', help='Output dir'
    )
    parser.add_argument(
        '--device', type=str, default=DEFAULT_DEVICE, help='The device to use (npu:0 / cuda:0 / cpu)'
    )
    parser.add_argument(
        '--exp_name',
        type=str,
        default=None,
        help='Experiment name, use video file name by default',
    )
    parser.add_argument(
        '--seed', type=int, default=43, help='Random seed for reproducibility'
    )
    parser.add_argument(
        '--video_length', type=int, default=49, help='Length of the video frames'
    )
    parser.add_argument('--fps', type=int, default=10, help='Fps for saved video')
    parser.add_argument(
        '--stride', type=int, default=1, help='Sampling stride for input video'
    )
    parser.add_argument('--server_name', type=str, help='Server IP address')

    ## render
    parser.add_argument(
        '--radius_scale',
        type=float,
        default=1.0,
        help='Scale factor for the spherical radius',
    )
    parser.add_argument('--camera', type=str, default='traj', help='traj or target')
    parser.add_argument(
        '--mode', type=str, default='gradual', help='gradual, bullet or direct'
    )
    parser.add_argument(
        '--mask', action='store_true', default=False, help='Clean the pcd if true'
    )
    parser.add_argument(
        '--traj_txt',
        type=str,
        help="Required for 'traj' camera, a txt file that specify camera trajectory",
    )
    parser.add_argument(
        '--target_pose',
        nargs=5,
        type=float,
        help="Required for 'target' mode, specify target camera pose, <theta phi r x y>",
    )
    parser.add_argument(
        '--near', type=float, default=0.0001, help='Near clipping plane distance'
    )
    parser.add_argument(
        '--far', type=float, default=10000.0, help='Far clipping plane distance'
    )
    parser.add_argument('--anchor_idx', type=int, default=0, help='One GT frame')

    ## diffusion
    parser.add_argument(
        '--low_gpu_memory_mode',
        type=bool,
        default=False,
        help='Enable low GPU/NPU memory mode',
    )
    parser.add_argument(
        '--model_name',
        type=str,
        default=os.path.join(ckpt_root, 'CogVideoX-Fun-V1.1-5b-InP'),
        help='Path to the model',
    )
    parser.add_argument(
        '--sampler_name',
        type=str,
        choices=["Euler", "Euler A", "DPM++", "PNDM", "DDIM_Cog", "DDIM_Origin"],
        default='DDIM_Origin',
        help='Choose the sampler',
    )
    parser.add_argument(
        '--transformer_path',
        type=str,
        default=os.path.join(ckpt_root, 'TrajectoryCrafter'),
        help='Path to the pretrained transformer model',
    )
    parser.add_argument(
        '--sample_size',
        type=int,
        nargs=2,
        default=[384, 672],
        help='Sample size as [height, width]',
    )
    parser.add_argument(
        '--diffusion_guidance_scale',
        type=float,
        default=6.0,
        help='Guidance scale for inference',
    )
    parser.add_argument(
        '--diffusion_inference_steps',
        type=int,
        default=50,
        help='Number of inference steps',
    )
    parser.add_argument(
        '--prompt', type=str, default=None, help='Prompt for video generation'
    )
    parser.add_argument(
        '--negative_prompt',
        type=str,
        default="The video is not of a high quality, it has a low resolution. Watermark present in each frame. The background is solid. Strange body and strange trajectory. Distortion.",
        help='Negative prompt for video generation',
    )
    parser.add_argument(
        '--refine_prompt',
        type=str,
        default=". The video is of high quality, and the view is very clear. High quality, masterpiece, best quality, highres, ultra-detailed, fantastic.",
        help='Prompt for video generation',
    )
    parser.add_argument('--blip_path', type=str, default=os.path.join(ckpt_root, 'blip2-opt-2.7b'))

    ## depth
    parser.add_argument(
        '--unet_path',
        type=str,
        default=os.path.join(ckpt_root, 'DepthCrafter'),
        help='Path to the UNet model',
    )
    parser.add_argument(
        '--pre_train_path',
        type=str,
        default=os.path.join(ckpt_root, 'stable-video-diffusion-img2vid'),
        help='Path to the pre-trained model',
    )
    parser.add_argument(
        '--cpu_offload', type=str, default='model', help='CPU offload strategy'
    )
    parser.add_argument(
        '--depth_inference_steps', type=int, default=5, help='Number of inference steps'
    )
    parser.add_argument(
        '--depth_guidance_scale',
        type=float,
        default=1.0,
        help='Guidance scale for inference',
    )
    parser.add_argument(
        '--window_size', type=int, default=110, help='Window size for processing'
    )
    parser.add_argument(
        '--overlap', type=int, default=25, help='Overlap size for processing'
    )
    parser.add_argument(
        '--max_res', type=int, default=1024, help='Maximum resolution for processing'
    )

    ## multi-source warping (infer_depth_warping_2)
    parser.add_argument(
        '--source_configs',
        type=str,
        default='[]',
        help='JSON list of source configs, each item: [npz_name, video_name, mask_path, [dtheta,dphi,dr,dx,dy]]',
    )
    parser.add_argument(
        '--source_base_path',
        type=str,
        default='',
        help='Base path for source_configs files (required when --source_configs is non-empty)',
    )
    parser.add_argument(
        '--warp_target_pose',
        nargs=5,
        type=float,
        default=[0.0, 20, -0.2, 0, 0.0],
        help='Final target pose used for pose_t in infer_depth_warping_2: <dtheta dphi dr dx dy>',
    )

    return parser


if __name__ == "__main__":
    parser = get_parser()
    opts = parser.parse_args()

    # Prefer float16 on NPU because some operators have limited bfloat16 support.
    if _HAS_NPU:
        opts.weight_dtype = torch.float16
    else:
        opts.weight_dtype = torch.bfloat16

    if opts.exp_name is None:
        prefix = datetime.now().strftime("%Y%m%d_%H%M")
        opts.exp_name = (
            f'{prefix}_{os.path.splitext(os.path.basename(opts.video_path))[0]}'
        )
    opts.save_dir = os.path.join(opts.out_dir, opts.exp_name)
    os.makedirs(opts.save_dir, exist_ok=True)

    print(f"[inference] Using device: {opts.device}, dtype: {opts.weight_dtype}")

    pvd = TrajCrafter(opts)

    input_depth = True
    pvd.infer_depth_warping_2(opts, input_depth)

    
