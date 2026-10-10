# Modified for Demiurge; see THIRD_PARTY_NOTICES.md.
import gc
import os
import torch
from npu_patch import empty_cache as _empty_cache, _HAS_NPU  # Enable NPU redirection and shared cache handling.
from models.infer import DepthCrafterDemo
import numpy as np
import torch

# Work around unsupported NPU float64 interpolation and CANN 8.x 5D nearest crashes.
if _HAS_NPU:
    import torch.nn.functional as _F_mod
    _orig_interpolate = _F_mod.interpolate

    def _npu_safe_interpolate(input, *args, **kwargs):
        mode = kwargs.get('mode', 'nearest')
        # Run 5D nearest interpolation on CPU to avoid NPU kernel crashes.
        if input.dim() == 5 and mode == 'nearest':
            _orig_device = input.device
            _orig_dtype = input.dtype
            out = _orig_interpolate(
                input.detach().to('cpu').float(), *args, **kwargs
            )
            return out.to(_orig_device).to(_orig_dtype)

        # NPU linear interpolation modes do not support float64.
        if mode in ('bilinear', 'bicubic', 'trilinear') and input.dtype == torch.float64:
            orig_dtype = input.dtype
            out = _orig_interpolate(input.float(), *args, **kwargs)
            return out.to(orig_dtype)

        return _orig_interpolate(input, *args, **kwargs)

    _F_mod.interpolate = _npu_safe_interpolate
    print("[demo.py] NPU: F.interpolate patched (5D nearest -> CPU, bilinear fp64 -> fp32)")
from transformers import T5EncoderModel
from omegaconf import OmegaConf
from PIL import Image
from models.crosstransformer3d import CrossTransformer3DModel
from models.autoencoder_magvit import AutoencoderKLCogVideoX
from models.pipeline_trajectorycrafter import TrajCrafter_Pipeline
from models.utils import *
from diffusers import (
    AutoencoderKL,
    CogVideoXDDIMScheduler,
    DDIMScheduler,
    DPMSolverMultistepScheduler,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    PNDMScheduler,
)
from transformers import AutoProcessor, Blip2ForConditionalGeneration


class TrajCrafter:
    def __init__(self, opts, gradio=False):
        self.funwarp = Warper(device=opts.device)
        self.depth_estimater = DepthCrafterDemo(
            unet_path=opts.unet_path,
            pre_train_path=opts.pre_train_path,
            cpu_offload=opts.cpu_offload,
            device=opts.device,
        )
        self.caption_processor = AutoProcessor.from_pretrained(opts.blip_path)
        self.captioner = Blip2ForConditionalGeneration.from_pretrained(
            opts.blip_path, torch_dtype=torch.float16
        ).to(opts.device)
        self.setup_diffusion(opts)
        if gradio:
            self.opts = opts

    def infer_gradual(self, opts):
        frames = read_video_frames(
            opts.video_path, opts.video_length, opts.stride, opts.max_res
        )
        prompt = self.get_caption(opts, frames[opts.video_length // 2])
        depths = self.depth_estimater.infer(
            frames,
            opts.near,
            opts.far,
            opts.depth_inference_steps,
            opts.depth_guidance_scale,
            window_size=opts.window_size,
            overlap=opts.overlap,
        ).to(opts.device)
        frames = (
            torch.from_numpy(frames).permute(0, 3, 1, 2).to(opts.device) * 2.0 - 1.0
        )  # 49 576 1024 3 -> 49 3 576 1024, [-1,1]
        assert frames.shape[0] == opts.video_length
        pose_s, pose_t, K = self.get_poses_fixed(opts, depths, num_frames=opts.video_length)
        warped_images = []
        masks = []
        warped_depth = []
        for i in tqdm(range(opts.video_length)):
            warped_frame2, mask2, warped_depth2, _, flow12 = self.funwarp.forward_warp(
                frames[i : i + 1],
                None,
                depths[i : i + 1],
                pose_s[i : i + 1],
                pose_t[i : i + 1],
                K[i : i + 1],
                None,
                opts.mask,
                twice=False,
            )
            warped_depth2 = warped_depth2.repeat(1,3,1,1)
            warped_depth2[warped_depth2 == 0] = 1e4
            warped_depth2 = 1 / (warped_depth2 + 1)

            warped_images.append(warped_frame2)
            masks.append(mask2)
            warped_depth.append(warped_depth2)

        cond_video = (torch.cat(warped_images) + 1.0) / 2.0
        cond_masks = torch.cat(masks)
        cond_video_depth = torch.cat(warped_depth)

        print('*' * 100, cond_video.shape)

        frames = F.interpolate(
            frames, size=opts.sample_size, mode='bilinear', align_corners=False
        )
        cond_video = F.interpolate(
            cond_video, size=opts.sample_size, mode='bilinear', align_corners=False
        )
        cond_masks = F.interpolate(cond_masks, size=opts.sample_size, mode='nearest')

        cond_video_depth = F.interpolate(
            cond_video_depth, size=[480, 832], mode='bilinear', align_corners=False
        )

        print('*' * 100, cond_video.shape)

        depths = depths.permute(0,2,3,1).repeat(1,1,1,3).detach()
        depths = 1 / (depths)
        save_video(
            depths,
            os.path.join(opts.save_dir, 'depth.mp4'),
            fps=opts.fps,
        )

        save_video(
            (frames.permute(0, 2, 3, 1) + 1.0) / 2.0,
            os.path.join(opts.save_dir, 'input.mp4'),
            fps=opts.fps,
        )
        save_video(
            cond_video.permute(0, 2, 3, 1),
            os.path.join(opts.save_dir, 'render.mp4'),
            fps=opts.fps,
        )
        save_video(
            cond_masks.repeat(1, 3, 1, 1).permute(0, 2, 3, 1),
            os.path.join(opts.save_dir, 'mask.mp4'),
            fps=opts.fps,
        )
        save_video(
            cond_video_depth.permute(0, 2, 3, 1),
            os.path.join(opts.save_dir, 'render_depth.mp4'),
            fps=opts.fps,
        )

        render_depth = cond_video_depth.detach().cpu()  # (T,1,H,W)

        render_depth_np = render_depth[:, 0].numpy().astype(np.float16)
        np.savez_compressed(
            os.path.join(opts.save_dir, "render_depth.npz"),
            depth=render_depth_np,
        )


        frames = (frames.permute(1, 0, 2, 3).unsqueeze(0) + 1.0) / 2.0
        frames_ref = frames[:, :, :10, :, :]
        cond_video = cond_video.permute(1, 0, 2, 3).unsqueeze(0)
        cond_masks = (1.0 - cond_masks.permute(1, 0, 2, 3).unsqueeze(0)) * 255.0
        generator = torch.Generator(device=opts.device).manual_seed(opts.seed)


        print('*' * 100, cond_video.shape)

        del self.depth_estimater
        del self.caption_processor
        del self.captioner
        gc.collect()
        torch.cuda.empty_cache()
        with torch.no_grad():
            sample = self.pipeline(
                prompt,
                num_frames=opts.video_length,
                negative_prompt=opts.negative_prompt,
                height=opts.sample_size[0],
                width=opts.sample_size[1],
                generator=generator,
                guidance_scale=opts.diffusion_guidance_scale,
                num_inference_steps=opts.diffusion_inference_steps,
                video=cond_video,
                mask_video=cond_masks,
                reference=frames_ref,
            ).videos
        save_video(
            sample[0].permute(1, 2, 3, 0),
            os.path.join(opts.save_dir, 'gen.mp4'),
            fps=opts.fps,
        )

        viz = True
        if viz:
            tensor_left = frames[0].to(opts.device)
            tensor_right = sample[0].to(opts.device)
            interval = torch.ones(3, 49, 384, 30).to(opts.device)
            result = torch.cat((tensor_left, interval, tensor_right), dim=3)
            result_reverse = torch.flip(result, dims=[1])
            final_result = torch.cat((result, result_reverse[:, 1:, :, :]), dim=1)
            save_video(
                final_result.permute(1, 2, 3, 0),
                os.path.join(opts.save_dir, 'viz.mp4'),
                fps=opts.fps * 2,
            )


    def _load_video_thwc(self, video_path, device, num_frames=-1, uniform=True):
        """Load native-resolution video as a (T,3,H,W) float tensor in [0,1].

        A positive num_frames limits sampling to uniform or leading frames.
        Shorter videos are returned in full.
        """
        vr = VideoReader(video_path, ctx=cpu(0))
        total = len(vr)
        if num_frames is not None and num_frames > 0 and total > num_frames:
            if uniform:
                idx = np.linspace(0, total - 1, num_frames).round().astype(int).tolist()
            else:
                idx = list(range(num_frames))
        else:
            idx = list(range(total))
        arr = vr.get_batch(idx).asnumpy().astype("float32") / 255.0  # (T,H,W,3)
        return torch.from_numpy(arr).permute(0, 3, 1, 2).to(device)   # (T,3,H,W)


    def infer_from_cond_videos(self, opts):
        prompt      = opts.prompt if getattr(opts, "prompt", None) else "a natural scene, high quality"
        render_path = opts.render_path
        mask_path   = opts.mask_path
        input_path  = getattr(opts, "input_path", None)  # Optional reference video.

        assert render_path and os.path.exists(render_path), f"render_path does not exist: {render_path}"
        assert mask_path and os.path.exists(mask_path),     f"mask_path does not exist: {mask_path}"

        os.makedirs(opts.save_dir, exist_ok=True)
        with open(os.path.join(opts.save_dir, "prompt.txt"), "w", encoding="utf-8") as f:
            f.write(prompt)

        # Limit decoded frames to video_length.
        cond_video = self._load_video_thwc(render_path, opts.device, num_frames=opts.video_length)  # (T,3,H,W) [0,1]
        cond_masks = self._load_video_thwc(mask_path,   opts.device, num_frames=opts.video_length)  # (T,3,H,W) [0,1]
        cond_masks = cond_masks[:, :1]  # Single-channel mask: (T,1,H,W).

        # Match the VAE temporal constraint: T = 4k + 1.
        T = cond_video.shape[0]
        assert cond_masks.shape[0] == T, "Render and mask frame counts do not match"
        if (T - 1) % 4 != 0:
            T = (T - 1) // 4 * 4 + 1
            print(f"[infer_from_cond_videos] Frame count aligned to 4k+1: T={T}")
        cond_video = cond_video[:T]
        cond_masks = cond_masks[:T]
        opts.video_length = T

        # Use the first ten clean input frames as reference, or fall back to conditioning frames.
        if input_path and os.path.exists(input_path):
            frames = self._load_video_thwc(input_path, opts.device, num_frames=opts.video_length)  # (T,3,H,W) [0,1]
            frames = frames[:T]
        else:
            frames = cond_video

        frames = F.interpolate(
            frames, size=opts.sample_size, mode="bilinear", align_corners=False
        )
        cond_video = F.interpolate(
            cond_video, size=opts.sample_size, mode="bilinear", align_corners=False
        )
        cond_masks = F.interpolate(
            cond_masks, size=opts.sample_size, mode="nearest"
        )
        cond_masks = (cond_masks > 0.5).float()

        frames     = frames.permute(1, 0, 2, 3).unsqueeze(0)                 # (1,3,T,H,W)
        frames_ref = frames[:, :, :10, :, :]                                 # (1,3,10,H,W)
        cond_video = cond_video.permute(1, 0, 2, 3).unsqueeze(0)             # (1,3,T,H,W)
        cond_masks = (1.0 - cond_masks.permute(1, 0, 2, 3).unsqueeze(0)) * 255.0  # (1,1,T,H,W)

        generator = torch.Generator(device=opts.device).manual_seed(opts.seed)

        # Release depth and caption models before diffusion.
        for name in ("depth_estimater", "caption_processor", "captioner"):
            if hasattr(self, name):
                delattr(self, name)
        gc.collect()
        torch.cuda.empty_cache()

        with torch.no_grad():
            sample = self.pipeline(
                prompt,
                num_frames=opts.video_length,
                negative_prompt=opts.negative_prompt,
                height=opts.sample_size[0],
                width=opts.sample_size[1],
                generator=generator,
                guidance_scale=opts.diffusion_guidance_scale,
                num_inference_steps=opts.diffusion_inference_steps,
                video=cond_video,
                mask_video=cond_masks,
                reference=frames_ref,
            ).videos

        save_video(
            sample[0].permute(1, 2, 3, 0),
            os.path.join(opts.save_dir, "gen.mp4"),
            fps=opts.fps,
        )

    def infer_depth_warping(self, opts, input_depth=False):


        frames = read_video_frames(
            opts.video_path, opts.video_length, opts.stride, opts.max_res
        )
        prompt = self.get_caption(opts, frames[opts.video_length // 2])

        os.makedirs(opts.save_dir, exist_ok=True)
        with open(os.path.join(opts.save_dir, "prompt.txt"), "w", encoding="utf-8") as f:
            f.write(prompt)

        depths = self.depth_estimater.infer(
            frames,
            opts.near,
            opts.far,
            opts.depth_inference_steps,
            opts.depth_guidance_scale,
            window_size=opts.window_size,
            overlap=opts.overlap,
        ).to(opts.device)   # expected: (T,1,H,W)


        

        frames = (
            torch.from_numpy(frames).permute(0, 3, 1, 2).to(opts.device) * 2.0 - 1.0
        )  # (T,3,H,W), [-1,1]

        assert frames.shape[0] == opts.video_length

        pose_s, pose_t, K = self.get_poses(
            opts, depths, num_frames=opts.video_length
        )

        print('*' * 100, depths.min(), depths.max())

        warped_images = []
        masks = []
        warped_depth = []

        for i in tqdm(range(opts.video_length)):
            warped_frame2, mask2, warped_depth2, _, flow12 = self.funwarp.forward_warp(
                frames[i:i+1],
                None,
                depths[i:i+1],
                pose_s[i:i+1],
                pose_t[i:i+1],
                K[i:i+1],
                None,
                opts.mask,
                twice=False,
            )

            warped_depth2 = warped_depth2.repeat(1,3,1,1)
            warped_depth2[warped_depth2 == 0] = 1e4
            warped_depth2 = 1 / (warped_depth2 + 1)

            warped_images.append(warped_frame2)   # (1,3,H,W)
            masks.append(mask2)                   # (1,1,H,W) or similar
            warped_depth.append(warped_depth2)    # keep single channel if possible

        cond_video = (torch.cat(warped_images, dim=0) + 1.0) / 2.0   # (T,3,H,W)
        cond_masks = torch.cat(masks, dim=0)                          # (T,1,H,W)
        cond_video_depth = torch.cat(warped_depth, dim=0)             # (T,1,H,W)

        # unify all saved outputs to this resolution
        opts.sample_size = [384, 672]

        frames = F.interpolate(
            frames, size=opts.sample_size, mode="bilinear", align_corners=False
        )
        cond_video = F.interpolate(
            cond_video, size=opts.sample_size, mode="bilinear", align_corners=False
        )
        cond_masks = F.interpolate(
            cond_masks, size=opts.sample_size, mode="nearest"
        )
        depths = F.interpolate(
            depths, size=opts.sample_size, mode="bilinear", align_corners=False
        )
        cond_video_depth = F.interpolate(
            cond_video_depth, size=opts.sample_size, mode="bilinear", align_corners=False
        )

        cond_masks = (cond_masks > 0.5).float()

        # save resized raw depth: (T,H,W)
        depth_np = depths[:, 0].detach().cpu().numpy().astype(np.float16)
        np.savez_compressed(
            os.path.join(opts.save_dir, "depth.npz"),
            depth=depth_np,
        )

        # save videos, all in sample_size
        save_video(
            (frames.permute(0, 2, 3, 1) + 1.0) / 2.0,
            os.path.join(opts.save_dir, "input.mp4"),
            fps=opts.fps,
        )

        save_video(
            cond_video.permute(0, 2, 3, 1),
            os.path.join(opts.save_dir, "render.mp4"),
            fps=opts.fps,
        )

        # RGB conditioning is already normalized to [0, 1].
        render_np = (
            cond_video.detach()
                    .permute(0, 2, 3, 1)        # (T, H, W, 3)
                    .clamp_(0.0, 1.0)
                    .cpu()
                    .numpy()
                    .astype(np.float16)  # Store half-precision arrays.
        )
        np.savez_compressed(
            os.path.join(opts.save_dir, "render_rgb.npz"),
            rgb=render_np,
            fps=int(opts.fps),
            dtype=str(render_np.dtype),
        )

        save_video(
            cond_masks.repeat(1, 3, 1, 1).permute(0, 2, 3, 1),
            os.path.join(opts.save_dir, "mask.mp4"),
            fps=opts.fps,
        )

        # masked render depth, also at sample_size
        render_depth = cond_video_depth.detach().cpu()  # (T,1,H,W)

        render_depth_np = render_depth[:, 0].numpy().astype(np.float16)
        np.savez_compressed(
            os.path.join(opts.save_dir, "render_depth.npz"),
            depth=render_depth_np,
        )


        frames = (frames.permute(1, 0, 2, 3).unsqueeze(0) + 1.0) / 2.0
        frames_ref = frames[:, :, :10, :, :]
        cond_video = cond_video.permute(1, 0, 2, 3).unsqueeze(0)
        cond_masks = (1.0 - cond_masks.permute(1, 0, 2, 3).unsqueeze(0)) * 255.0
        generator = torch.Generator(device=opts.device).manual_seed(opts.seed)



        del self.depth_estimater
        del self.caption_processor
        del self.captioner
        gc.collect()
        torch.cuda.empty_cache()
        with torch.no_grad():
            sample = self.pipeline(
                prompt,
                num_frames=opts.video_length,
                negative_prompt=opts.negative_prompt,
                height=opts.sample_size[0],
                width=opts.sample_size[1],
                generator=generator,
                guidance_scale=opts.diffusion_guidance_scale,
                num_inference_steps=opts.diffusion_inference_steps,
                video=cond_video,
                mask_video=cond_masks,
                reference=frames_ref,
            ).videos
        save_video(
            sample[0].permute(1, 2, 3, 0),
            os.path.join(opts.save_dir, 'gen.mp4'),
            fps=opts.fps,
        )


    def export_target_pose_to_recammaster(
        self,
        opts,
        json_path,
        cam_name="cam11",
        depths=None,
        mode="step",              # "step" | "static" | "moving"
        align_frame0_to="[1 0 0 0] [-0 1 0 0] [0 -0 1 0] [3390 1380 240 1] ",  # None disables alignment.
    ):
        """Write target poses to cam_name in ReCamMaster camera-extrinsics JSON.

        Modes: step keeps the initial first frame; static repeats the target;
        moving interpolates from the initial pose to the target. An optional
        frame-zero alignment applies one rigid transform to the whole trajectory.
        """
        import json
        import numpy as np
        import torch

        _warp_tp = getattr(opts, 'warp_target_pose', [0.0, 20, -0.2, 0, 0.0])
        opts.target_pose = list(_warp_tp)
        if getattr(opts, 'camera', None) != 'target':
            opts.camera = 'target'

        # Depth is a placeholder because the radius is fixed at one.
        if depths is None:
            depths = torch.zeros(1, 1, 576, 1024, device=opts.device)

        num_frames = opts.video_length
        radius = 1

        if mode == "moving":
            dtheta, dphi, dr, dx, dy = _warp_tp
            c2w_init = torch.tensor(
                [[-1., 0., 0., 0.], [0., 1., 0., 0.], [0., 0., -1., 0.], [0., 0., 0., 1.]],
                device=opts.device,
            ).unsqueeze(0)
            traj = generate_traj_specified(
                c2w_init, dtheta, dphi, dr * radius, dx, dy, num_frames, opts.device
            )
            traj[:, 2, 3] += radius
            traj = traj.detach().cpu().numpy()
        else:
            _, poses2, _ = self.get_poses_fixed(opts, depths, num_frames=num_frames)
            traj = poses2.detach().cpu().numpy()  # Repeat the target pose for all frames.

            if mode == "step":
                init = np.array(
                    [[-1., 0., 0., 0.], [0., 1., 0., 0.], [0., 0., -1., 0.], [0., 0., 0., 1.]],
                    dtype=np.float64,
                )
                init[2, 3] += radius                       # z -> 1m(=100cm)
                traj = traj.copy()
                traj[0] = init  # Initial pose at frame zero; target pose thereafter.

        # Align frame zero with a global rigid transform, preserving relative poses.
        if align_frame0_to is not None:
            C_target = self._recammaster_string_to_c2w(align_frame0_to)
            G = C_target @ np.linalg.inv(traj[0])
            traj = np.einsum('ij,njk->nik', G, traj)

        with open(json_path, "r") as f:
            data = json.load(f)

        assert len(data) == traj.shape[0], (
            f"Frame count mismatch: json={len(data)}, traj={traj.shape[0]} "
            f"(opts.video_length must match the JSON frame count, normally 81)"
        )
        for idx in range(traj.shape[0]):
            data[f"frame{idx}"][cam_name] = self._c2w_to_recammaster_string(traj[idx])

        with open(json_path, "w") as f:
            json.dump(data, f, indent=4)

        print(f"[export] Wrote {cam_name} -> {json_path} "
            f"({traj.shape[0]} frames, mode={mode}, aligned={align_frame0_to is not None})")
        return traj


    @staticmethod
    def _c2w_to_recammaster_string(c2w):
        """Convert OpenCV c2w in meters to ReCamMaster row vectors in centimeters."""
        import numpy as np
        A = np.array(c2w, dtype=np.float64).copy()
        A[:3, 3] *= 100.0  # Meters to centimeters.
        A[:3, 1] *= -1.0  # Flip the Y axis.
        A = A[:, [2, 0, 1, 3]]  # Undo the column permutation.
        M = A.T  # Convert to row vectors.
        def fmt(v):
            if abs(v) < 1e-9:  # Suppress near-zero numerical residuals.
                v = 0.0
            return f"{v:g}"
        return " ".join("[" + " ".join(fmt(v) for v in M[i]) + "]" for i in range(4)) + " "


    @staticmethod
    def _recammaster_string_to_c2w(s):
        """Convert ReCamMaster row vectors in centimeters to OpenCV c2w in meters."""
        import numpy as np
        M = [list(map(float, r.replace('[', '').replace(']', '').split()))
            for r in s.strip().split('] [')]
        c2w = np.array(M, dtype=np.float64).T  # Convert to column vectors.
        c2w = c2w[:, [1, 2, 0, 3]]  # Restore column order.
        c2w[:3, 1] *= -1.0  # Flip the Y axis.
        c2w[:3, 3] /= 100.0  # Centimeters to meters.
        return c2w


            
    def infer_depth_warping_2(self, opts, input_depth=False):
        self.funwarp = Warper(device=opts.device)
        frames = read_video_frames(
            opts.video_path, opts.video_length, opts.stride, opts.max_res
        )
        prompt = self.get_caption(opts, frames[opts.video_length // 2])
        depths = self.depth_estimater.infer(
            frames,
            opts.near,
            opts.far,
            opts.depth_inference_steps,
            opts.depth_guidance_scale,
            window_size=opts.window_size,
            overlap=opts.overlap,
        ).to(opts.device)
        frames = (
            torch.from_numpy(frames).permute(0, 3, 1, 2).to(opts.device) * 2.0 - 1.0
        )
        assert frames.shape[0] == opts.video_length
        pose_s, pose_t, K = self.get_poses(opts, depths, num_frames=opts.video_length)
        dtheta, dphi, dr, dx, dy = opts.target_pose

        pose_s_1, _, K = self.get_poses(opts, depths, num_frames=opts.video_length)
        depth_1 = depths
        frame_1 = frames

        import json as _json
        _src_cfg_raw = getattr(opts, 'source_configs', '[]')
        if isinstance(_src_cfg_raw, str):
            _src_cfg_parsed = _json.loads(_src_cfg_raw) if _src_cfg_raw else []
        else:
            _src_cfg_parsed = list(_src_cfg_raw)
        source_configs = [
            (item[0], item[1], item[2], list(item[3])) for item in _src_cfg_parsed
        ]
        base_path = opts.source_base_path
        if len(source_configs) > 0 and not base_path:
            raise ValueError(
                "--source_base_path is required when source_configs is not empty"
            )

        frames_list   = []
        depths_list   = []
        poses_list    = []
        raw_mask_list = []  # None denotes the main source, with an all-valid mask.

        frames_list.append(frame_1.cpu())
        depths_list.append(depth_1.cpu())
        _, pose_s_1, K = self.get_poses(opts, depths, num_frames=opts.video_length)
        poses_list.append(pose_s_1.cpu())
        raw_mask_list.append(None)

        for npz_name, video_name, mask_path, src_target_pose in source_configs:
            mask_frames_raw = read_video_frames(
                os.path.join(base_path, mask_path), opts.video_length, opts.stride, opts.max_res
            )
            mf = torch.from_numpy(mask_frames_raw).permute(0, 3, 1, 2).to(opts.device)
            mf = mf[:, 0:1, :, :]
            mf = (mf >= 0.5).to(mf.dtype)
            raw_mask_list.append(mf.cpu())

            depth = np.load(os.path.join(base_path, npz_name))['depth']
            depth = torch.from_numpy(depth).to(opts.device)
            depth = depth.unsqueeze(1)
            depth = 1 / (depth + 1e-5) - 1
            depth = F.interpolate(depth, size=(576, 1024), mode='bilinear', align_corners=False)
            depths_list.append(depth.cpu())

            video_frames = read_video_frames(
                os.path.join(base_path, video_name), opts.video_length, opts.stride, opts.max_res
            )
            video_frames = torch.from_numpy(video_frames).permute(0, 3, 1, 2).to(opts.device) * 2 - 1
            frames_list.append(video_frames.cpu())

            opts.target_pose = list(src_target_pose)
            _, pose_s, _ = self.get_poses(opts, depths, num_frames=opts.video_length)
            poses_list.append(pose_s.cpu())

        _warp_tp = getattr(opts, 'warp_target_pose', [0.0, 20, -0.2, 0, 0.0])
        opts.target_pose = list(_warp_tp)
        _, pose_t, K = self.get_poses(opts, depths, num_frames=opts.video_length)

        def _build_masks_list(force_zero_source_mask: bool):
            masks_list = []
            masks_list.append(torch.ones_like(depth_1).cpu())
            for raw_m in raw_mask_list[1:]:
                mf = raw_m.to(opts.device).clone()
                if force_zero_source_mask:
                    mf = torch.zeros_like(mf)
                DILATE_KSIZE = 3
                DILATE_ITERS = 1
                for _ in range(DILATE_ITERS):
                    mf = F.max_pool2d(
                        mf, kernel_size=DILATE_KSIZE, stride=1,
                        padding=DILATE_KSIZE // 2,
                    )
                mf = (mf > 0.5).to(mf.dtype)
                mf = 1 - mf
                masks_list.append(mf.cpu())
            return masks_list

        # Reject pixels with near-zero depth, high gradients, or high local variance.
        def _do_unreliable_clean(cond_video, cond_masks, cond_video_depth):
            depth_for_check = cond_video_depth[:, 0:1, :, :].clone()
            d_min, d_max = depth_for_check.min(), depth_for_check.max()

            mask_zero = depth_for_check < 1e-5

            dy_, dx_ = torch.gradient(depth_for_check, dim=[2, 3])
            grad_mag = torch.sqrt(dx_ ** 2 + dy_ ** 2)
            mask_grad = grad_mag > 0.03

            kernel = torch.ones(
                1, 1, 3, 3,
                device=depth_for_check.device, dtype=depth_for_check.dtype,
            ) / 9.0
            local_mean    = F.conv2d(depth_for_check,        kernel, padding=1)
            local_sq_mean = F.conv2d(depth_for_check ** 2,   kernel, padding=1)
            local_var = local_sq_mean - local_mean ** 2
            mask_local_var = local_var > 1e-4

            final_mask = mask_zero | mask_grad | mask_local_var

            cond_video_depth = cond_video_depth.clone()
            cond_video_depth[final_mask.expand_as(cond_video_depth)] = 0
            cond_video_depth = torch.clamp(cond_video_depth, d_min, d_max)

            cond_video = cond_video.clone()
            cond_video[final_mask.expand_as(cond_video)] = 0

            cond_masks = cond_masks.clone()
            cond_masks[final_mask] = 0

            return cond_video, cond_masks, cond_video_depth, final_mask

        # Warp sources, merge with a z-buffer, then clean unreliable pixels.
        def _run_warp(masks_list, n_clean_iters: int = 1):
            T = opts.video_length
            N = len(poses_list)
            H, W = frames_list[0].shape[-2:]

            T_CHUNK = 16
            N_CHUNK = 8
            DEPTH_VALID_THR = 1e-5

            warped_images, masks_out, warped_depth = [], [], []

            for s in tqdm(range(0, T, T_CHUNK)):
                e = min(s + T_CHUNK, T)
                t = e - s

                _dev = pose_t.device
                merged_frame = torch.full(
                    (t, 3, H, W), -1.0, device=_dev, dtype=frames_list[0].dtype,
                )
                merged_mask = torch.zeros(
                    (t, 1, H, W), device=_dev, dtype=frames_list[0].dtype,
                )
                BIG_DEPTH = torch.finfo(depths_list[0].dtype).max
                merged_depth = torch.full(
                    (t, 1, H, W), BIG_DEPTH, device=_dev, dtype=depths_list[0].dtype,
                )

                for ns in range(0, N, N_CHUNK):
                    ne = min(ns + N_CHUNK, N)
                    n = ne - ns

                    frames_b = torch.cat(
                        [f[s:e].to(_dev, non_blocking=True) for f in frames_list[ns:ne]], dim=0
                    )
                    depths_b = torch.cat(
                        [d[s:e].to(_dev, non_blocking=True) for d in depths_list[ns:ne]], dim=0
                    )
                    masks_b  = torch.cat(
                        [m[s:e].to(_dev, non_blocking=True) for m in masks_list[ns:ne]], dim=0
                    )
                    poses_b  = torch.cat(
                        [p[s:e].to(_dev, non_blocking=True) for p in poses_list[ns:ne]], dim=0
                    )

                    K_src_b  = K[s:e].repeat(n, 1, 1)
                    pose_t_b = pose_t[s:e].repeat(n, 1, 1)
                    K_tgt_b  = K[s:e].repeat(n, 1, 1)

                    warped_frame_b, warped_mask_b, warped_depth_b, _, _ = self.funwarp.forward_warp(
                        frames_b, masks_b, depths_b, poses_b, pose_t_b, K_src_b, K_tgt_b,
                        mask=opts.mask, twice=False,
                    )

                    wf = warped_frame_b.view(n, t, 3, H, W)
                    wm = warped_mask_b.view(n, t, 1, H, W)
                    wd = warped_depth_b.view(n, t, 1, H, W)

                    valid = (wm > 0.5) & (wd > DEPTH_VALID_THR)
                    depth_for_z = torch.where(valid, wd, torch.full_like(wd, BIG_DEPTH))

                    local_best = depth_for_z.argmin(dim=0, keepdim=True)
                    local_any_valid = valid.any(dim=0, keepdim=True)

                    def _gather_local(x, c):
                        idx = local_best.expand(1, t, c, H, W)
                        return torch.gather(x, dim=0, index=idx).squeeze(0)

                    local_frame = _gather_local(wf, 3)
                    local_depth = _gather_local(wd, 1)
                    local_mask  = _gather_local(wm, 1)
                    local_valid = local_any_valid.squeeze(0)

                    update = local_valid & (local_depth < merged_depth)

                    merged_frame = torch.where(
                        update.expand_as(merged_frame), local_frame, merged_frame,
                    )
                    merged_depth = torch.where(update, local_depth, merged_depth)
                    merged_mask  = torch.where(update, local_mask,  merged_mask)

                final_invalid = ~(
                    (merged_mask > 0.5)
                    & (merged_depth > DEPTH_VALID_THR)
                    & (merged_depth < BIG_DEPTH)
                )
                merged_frame = torch.where(
                    final_invalid.expand_as(merged_frame),
                    torch.full_like(merged_frame, -1.0),
                    merged_frame,
                )
                merged_mask = torch.where(
                    final_invalid, torch.zeros_like(merged_mask), merged_mask,
                )
                merged_depth = torch.where(
                    final_invalid, torch.zeros_like(merged_depth), merged_depth,
                )

                bad = merged_depth <= 1e-5
                merged_depth = torch.where(
                    bad, torch.full_like(merged_depth, 10000.0), merged_depth,
                )
                merged_mask = torch.where(bad, torch.zeros_like(merged_mask), merged_mask)
                merged_frame = torch.where(
                    bad.expand_as(merged_frame),
                    torch.full_like(merged_frame, -1.0),
                    merged_frame,
                )

                merged_depth = merged_depth.repeat(1, 3, 1, 1)
                merged_depth[merged_depth == 0] = 10000
                merged_depth = 1 / (merged_depth + 1)

                warped_images.append(merged_frame)
                masks_out.append(merged_mask)
                warped_depth.append(merged_depth)

            cond_video       = (torch.cat(warped_images) + 1.0) / 2.0
            cond_masks       = torch.cat(masks_out)
            cond_video_depth = torch.cat(warped_depth)

            opts.sample_size = [480, 832]
            cond_video       = F.interpolate(cond_video,       size=opts.sample_size, mode='bilinear', align_corners=False)
            cond_masks       = F.interpolate(cond_masks,       size=opts.sample_size, mode='nearest')
            cond_video_depth = F.interpolate(cond_video_depth, size=opts.sample_size, mode='bilinear', align_corners=False)

            cond_masks_raw = cond_masks.clone()

            ratios = []
            for it in range(n_clean_iters):
                cond_video, cond_masks, cond_video_depth, fmask = _do_unreliable_clean(
                    cond_video, cond_masks, cond_video_depth
                )
                r = float(fmask.float().mean().item())
                ratios.append(r)
                print(f"[clean] iter {it+1}/{n_clean_iters}  unreliable ratio = {r:.4f}")
            unrel_total = ratios[0] if ratios else 0.0

            return cond_video, cond_masks, cond_video_depth, cond_masks_raw, unrel_total

        def _save_outputs(cond_video, cond_masks, cond_masks_raw, cond_video_depth, suffix=""):
            sfx = suffix
            save_video(
                cond_video.permute(0, 2, 3, 1),
                os.path.join(opts.save_dir, f'render{sfx}.mp4'),
                fps=opts.fps,
            )
            save_video(
                cond_masks.repeat(1, 3, 1, 1).permute(0, 2, 3, 1),
                os.path.join(opts.save_dir, f'mask{sfx}.mp4'),
                fps=opts.fps,
            )
            save_video(
                cond_masks_raw.repeat(1, 3, 1, 1).permute(0, 2, 3, 1),
                os.path.join(opts.save_dir, f'mask_raw{sfx}.mp4'),
                fps=opts.fps,
            )
            save_video(
                cond_video_depth.permute(0, 2, 3, 1),
                os.path.join(opts.save_dir, f'render_depth{sfx}.mp4'),
                fps=opts.fps,
            )

            render_depth = cond_video_depth[:, 0:1, :, :].detach().cpu()
            render_depth_np = render_depth[:, 0].numpy().astype(np.float16)
            np.savez_compressed(
                os.path.join(opts.save_dir, f"render_depth{sfx}.npz"),
                depth=render_depth_np,
            )
            print(f"render_depth{sfx}.npz saved:",
                  render_depth_np.shape, render_depth_np.dtype,
                  render_depth_np.min(), render_depth_np.max())

            render_rgb = cond_video.detach().cpu()
            render_rgb_np = render_rgb.numpy().astype(np.float16)
            np.savez_compressed(
                os.path.join(opts.save_dir, f"render_rgb{sfx}.npz"),
                video=render_rgb_np,
            )
            print(f"render_rgb{sfx}.npz saved:",
                  render_rgb_np.shape, render_rgb_np.dtype,
                  render_rgb_np.min(), render_rgb_np.max())

        def _empty_cache():
            try:
                torch.npu.empty_cache()
            except Exception:
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass

        # Run A: force_zero_source_mask = False
        print("=" * 80)
        print("[Run A] force_zero_source_mask = False, n_clean_iters = 1")
        print("=" * 80)
        masks_list_A = _build_masks_list(force_zero_source_mask=False)
        rgb_A, mask_A, depth_A, mask_raw_A, unrel_A = _run_warp(masks_list_A, n_clean_iters=1)
        print(f"[Run A] mask ratio = {mask_A.mean().item():.4f},  unreliable(iter1) = {unrel_A:.4f}")

        _save_outputs(rgb_A, mask_A, mask_raw_A, depth_A, suffix="_A")

        del masks_list_A
        _empty_cache()


        # Run B-aux: force_zero_source_mask = True
        print("=" * 80)
        print("[Run B-aux] force_zero_source_mask = True, n_clean_iters = 1")
        print("=" * 80)
        masks_list_B = _build_masks_list(force_zero_source_mask=True)
        rgb_Baux, mask_Baux, depth_Baux, mask_raw_Baux, unrel_Baux = _run_warp(masks_list_B, n_clean_iters=1)
        print(f"[Run B-aux] mask ratio = {mask_Baux.mean().item():.4f},  unreliable(iter1) = {unrel_Baux:.4f}")

        _save_outputs(rgb_Baux, mask_Baux, mask_raw_Baux, depth_Baux, suffix="_B")

        del masks_list_B
        _empty_cache()

        # Save the union of both raw validity masks.
        mask_raw_A_dev    = mask_raw_A.to(opts.device)
        mask_raw_Baux_dev = mask_raw_Baux.to(opts.device)
        mask_raw_C = ((mask_raw_A_dev > 0.5) | (mask_raw_Baux_dev > 0.5)).to(mask_raw_A_dev.dtype)
        save_video(
            mask_raw_C.repeat(1, 3, 1, 1).permute(0, 2, 3, 1).cpu(),
            os.path.join(opts.save_dir, 'mask_raw_C.mp4'),
            fps=opts.fps,
        )
        print(f"[mask_raw_C] = mask_raw_A OR mask_raw_B  ratio = {mask_raw_C.float().mean().item():.4f}")
        del mask_raw_A_dev, mask_raw_Baux_dev, mask_raw_C
        _empty_cache()

        # Compare rounded depth values on the 0-255 scale.
        DIFF_THRESHOLD_255 = float(getattr(opts, 'diff_depth_threshold', 15.0))  # Threshold on the 0-255 scale.

        dA01 = depth_A[:, 0:1, :, :].to(opts.device).clamp(0.0, 1.0)
        dB01 = depth_Baux[:, 0:1, :, :].to(opts.device).clamp(0.0, 1.0)
        dA_u8 = (dA01 * 255.0).round()
        dB_u8 = (dB01 * 255.0).round()
        diff = (dA_u8 - dB_u8).abs()
        diff_mask = (diff > DIFF_THRESHOLD_255)               # bool, (T,1,H,W)

        diff_mask_vis = diff_mask.to(torch.float32)
        save_video(
            diff_mask_vis.repeat(1, 3, 1, 1).permute(0, 2, 3, 1).cpu(),
            os.path.join(opts.save_dir, 'diff_mask.mp4'),
            fps=opts.fps,
        )
        print(f"[diff] threshold = {DIFF_THRESHOLD_255} (0~255 scale, aligned with test2.py)")
        print(f"[diff] diff(0~255): min={diff.min().item():.2f}, "
              f"max={diff.max().item():.2f}, mean={diff.mean().item():.4f}, "
              f"white_ratio={diff_mask.float().mean().item():.4f}")

        del dA01, dB01, dA_u8, dB_u8, diff, diff_mask_vis
        _empty_cache()


        rgb_A      = rgb_A.to(opts.device)
        mask_A     = mask_A.to(opts.device)
        mask_raw_A = mask_raw_A.to(opts.device)
        depth_A    = depth_A.to(opts.device)
        rgb_Baux      = rgb_Baux.to(opts.device)
        mask_Baux     = mask_Baux.to(opts.device)
        mask_raw_Baux = mask_raw_Baux.to(opts.device)
        depth_Baux    = depth_Baux.to(opts.device)

        # Replace inconsistent pixels with the auxiliary warp for the main output.
        rgb_final      = torch.where(diff_mask.expand_as(rgb_A), rgb_Baux,      rgb_A)
        mask_final     = torch.where(diff_mask,                  mask_Baux,     mask_A)
        mask_raw_final = torch.where(diff_mask,                  mask_raw_Baux, mask_raw_A)
        depth_final    = torch.where(diff_mask.expand_as(depth_A), depth_Baux,  depth_A)

        # For the D output, replace inconsistent pixels with zero-valued holes.
        DEPTH_INVALID_PLACEHOLDER = 0
        rgb_final_D = torch.where(
            diff_mask.expand_as(rgb_A),
            torch.zeros_like(rgb_A),
            rgb_A,
        )
        mask_final_D = torch.where(
            diff_mask,
            torch.zeros_like(mask_A),
            mask_A,
        )
        mask_raw_final_D = torch.where(
            diff_mask,
            torch.zeros_like(mask_raw_A),
            mask_raw_A,
        )
        depth_final_D = torch.where(
            diff_mask.expand_as(depth_A),
            torch.full_like(depth_A, DEPTH_INVALID_PLACEHOLDER),
            depth_A,
        )

        _save_outputs(rgb_final_D, mask_final_D, mask_raw_final_D, depth_final_D, suffix="_D")
        print(f"[Run D] (zero replacement) mask mean = {mask_final_D.mean().item():.4f}, "
              f"rgb mean = {rgb_final_D.mean().item():.4f}")
        del rgb_final_D, mask_final_D, mask_raw_final_D, depth_final_D
        _empty_cache()

        # Report the intersection mask; it does not modify the output.
        gate_AB = ((mask_A > 0.5) & (mask_Baux > 0.5)).to(rgb_final.dtype)   # (T,1,H,W) 0/1
        print(f"[gate] (mask_A * mask_Baux) ratio = {gate_AB.float().mean().item():.4f}")
        print(f"[gate] rgb mean = {rgb_final.mean().item():.4f}, "
              f"mask mean = {mask_final.mean().item():.4f}")

        del (rgb_A, mask_A, depth_A, mask_raw_A,
             rgb_Baux, mask_Baux, depth_Baux, mask_raw_Baux,
             diff_mask, gate_AB)
        _empty_cache()


        # Optionally clean unreliable pixels in the final output.
        N_FINAL_CLEAN = int(getattr(opts, 'final_clean_iters', 0))  # Zero disables final cleanup.
        if N_FINAL_CLEAN <= 0:
            print("[final-clean] skipped (final_clean_iters <= 0)")
        else:
            print(f"[final-clean] running {N_FINAL_CLEAN} iter(s) of unreliable-pixel cleaning on final output")
            for it in range(N_FINAL_CLEAN):
                rgb_final, mask_final, depth_final, fmask_final = _do_unreliable_clean(
                    rgb_final, mask_final, depth_final,
                )
                mask_raw_final = mask_raw_final.clone()
                mask_raw_final[fmask_final] = 0

                ratio = float(fmask_final.float().mean().item())
                print(f"[final-clean] iter {it+1}/{N_FINAL_CLEAN}  unreliable ratio = {ratio:.4f}, "
                      f"mask mean after = {mask_final.mean().item():.4f}")
            del fmask_final
        _empty_cache()


        cond_video       = rgb_final
        cond_masks       = mask_final
        cond_video_depth = depth_final
        cond_masks_raw   = mask_raw_final


        _save_outputs(cond_video, cond_masks, cond_masks_raw, cond_video_depth, suffix="")




    def infer_direct(self, opts):
        opts.cut = 20
        frames = read_video_frames(
            opts.video_path, opts.video_length, opts.stride, opts.max_res
        )
        prompt = self.get_caption(opts, frames[opts.video_length // 2])
        depths = self.depth_estimater.infer(
            frames,
            opts.near,
            opts.far,
            opts.depth_inference_steps,
            opts.depth_guidance_scale,
            window_size=opts.window_size,
            overlap=opts.overlap,
        ).to(opts.device)
        frames = (
            torch.from_numpy(frames).permute(0, 3, 1, 2).to(opts.device) * 2.0 - 1.0
        )  # 49 576 1024 3 -> 49 3 576 1024, [-1,1]
        assert frames.shape[0] == opts.video_length
        pose_s, pose_t, K = self.get_poses_fixed(opts, depths, num_frames=opts.cut)

        warped_images = []
        masks = []
        for i in tqdm(range(opts.video_length)):
            if i < opts.cut:
                warped_frame2, mask2, warped_depth2, flow12 = self.funwarp.forward_warp(
                    frames[0:1],
                    None,
                    depths[0:1],
                    pose_s[0:1],
                    pose_t[i : i + 1],
                    K[0:1],
                    None,
                    opts.mask,
                    twice=False,
                )
                warped_images.append(warped_frame2)
                masks.append(mask2)
            else:
                warped_frame2, mask2, warped_depth2, flow12 = self.funwarp.forward_warp(
                    frames[i - opts.cut : i - opts.cut + 1],
                    None,
                    depths[i - opts.cut : i - opts.cut + 1],
                    pose_s[0:1],
                    pose_t[-1:],
                    K[0:1],
                    None,
                    opts.mask,
                    twice=False,
                )
                warped_images.append(warped_frame2)
                masks.append(mask2)
        cond_video = (torch.cat(warped_images) + 1.0) / 2.0
        cond_masks = torch.cat(masks)
        frames = F.interpolate(
            frames, size=opts.sample_size, mode='bilinear', align_corners=False
        )
        cond_video = F.interpolate(
            cond_video, size=opts.sample_size, mode='bilinear', align_corners=False
        )
        cond_masks = F.interpolate(cond_masks, size=opts.sample_size, mode='nearest')
        save_video(
            (frames[: opts.video_length - opts.cut].permute(0, 2, 3, 1) + 1.0) / 2.0,
            os.path.join(opts.save_dir, 'input.mp4'),
            fps=opts.fps,
        )
        save_video(
            cond_video[opts.cut :].permute(0, 2, 3, 1),
            os.path.join(opts.save_dir, 'render.mp4'),
            fps=opts.fps,
        )
        save_video(
            cond_masks[opts.cut :].repeat(1, 3, 1, 1).permute(0, 2, 3, 1),
            os.path.join(opts.save_dir, 'mask.mp4'),
            fps=opts.fps,
        )
        frames = (frames.permute(1, 0, 2, 3).unsqueeze(0) + 1.0) / 2.0
        frames_ref = frames[:, :, :10, :, :]
        cond_video = cond_video.permute(1, 0, 2, 3).unsqueeze(0)
        cond_masks = (1.0 - cond_masks.permute(1, 0, 2, 3).unsqueeze(0)) * 255.0
        generator = torch.Generator(device=opts.device).manual_seed(opts.seed)

        del self.depth_estimater
        del self.caption_processor
        del self.captioner
        gc.collect()
        _empty_cache()
        with torch.no_grad():
            sample = self.pipeline(
                prompt,
                num_frames=opts.video_length,
                negative_prompt=opts.negative_prompt,
                height=opts.sample_size[0],
                width=opts.sample_size[1],
                generator=generator,
                guidance_scale=opts.diffusion_guidance_scale,
                num_inference_steps=opts.diffusion_inference_steps,
                video=cond_video,
                mask_video=cond_masks,
                reference=frames_ref,
            ).videos
        save_video(
            sample[0].permute(1, 2, 3, 0)[opts.cut :],
            os.path.join(opts.save_dir, 'gen.mp4'),
            fps=opts.fps,
        )

        viz = True
        if viz:
            tensor_left = frames[0][:, : opts.video_length - opts.cut, ...].to(
                opts.device
            )
            tensor_right = sample[0][:, opts.cut :, ...].to(opts.device)
            interval = torch.ones(3, opts.video_length - opts.cut, 384, 30).to(
                opts.device
            )
            result = torch.cat((tensor_left, interval, tensor_right), dim=3)
            result_reverse = torch.flip(result, dims=[1])
            final_result = torch.cat((result, result_reverse[:, 1:, :, :]), dim=1)
            save_video(
                final_result.permute(1, 2, 3, 0),
                os.path.join(opts.save_dir, 'viz.mp4'),
                fps=opts.fps * 2,
            )

    def infer_bullet(self, opts):
        frames = read_video_frames(
            opts.video_path, opts.video_length, opts.stride, opts.max_res
        )
        prompt = self.get_caption(opts, frames[opts.video_length // 2])
        depths = self.depth_estimater.infer(
            frames,
            opts.near,
            opts.far,
            opts.depth_inference_steps,
            opts.depth_guidance_scale,
            window_size=opts.window_size,
            overlap=opts.overlap,
        ).to(opts.device)

        frames = (
            torch.from_numpy(frames).permute(0, 3, 1, 2).to(opts.device) * 2.0 - 1.0
        )  # 49 576 1024 3 -> 49 3 576 1024, [-1,1]
        assert frames.shape[0] == opts.video_length
        pose_s, pose_t, K = self.get_poses_fixed(opts, depths, num_frames=opts.video_length)

        warped_images = []
        masks = []
        for i in tqdm(range(opts.video_length)):
            warped_frame2, mask2, warped_depth2, flow12 = self.funwarp.forward_warp(
                frames[-1:],
                None,
                depths[-1:],
                pose_s[0:1],
                pose_t[i : i + 1],
                K[0:1],
                None,
                opts.mask,
                twice=False,
            )
            warped_images.append(warped_frame2)
            masks.append(mask2)
        cond_video = (torch.cat(warped_images) + 1.0) / 2.0
        cond_masks = torch.cat(masks)
        frames = F.interpolate(
            frames, size=opts.sample_size, mode='bilinear', align_corners=False
        )
        cond_video = F.interpolate(
            cond_video, size=opts.sample_size, mode='bilinear', align_corners=False
        )
        cond_masks = F.interpolate(cond_masks, size=opts.sample_size, mode='nearest')
        save_video(
            (frames.permute(0, 2, 3, 1) + 1.0) / 2.0,
            os.path.join(opts.save_dir, 'input.mp4'),
            fps=opts.fps,
        )
        save_video(
            cond_video.permute(0, 2, 3, 1),
            os.path.join(opts.save_dir, 'render.mp4'),
            fps=opts.fps,
        )
        save_video(
            cond_masks.repeat(1, 3, 1, 1).permute(0, 2, 3, 1),
            os.path.join(opts.save_dir, 'mask.mp4'),
            fps=opts.fps,
        )
        frames = (frames.permute(1, 0, 2, 3).unsqueeze(0) + 1.0) / 2.0
        frames_ref = frames[:, :, -10:, :, :]
        cond_video = cond_video.permute(1, 0, 2, 3).unsqueeze(0)
        cond_masks = (1.0 - cond_masks.permute(1, 0, 2, 3).unsqueeze(0)) * 255.0
        generator = torch.Generator(device=opts.device).manual_seed(opts.seed)

        del self.depth_estimater
        del self.caption_processor
        del self.captioner
        gc.collect()
        _empty_cache()
        with torch.no_grad():
            sample = self.pipeline(
                prompt,
                num_frames=opts.video_length,
                negative_prompt=opts.negative_prompt,
                height=opts.sample_size[0],
                width=opts.sample_size[1],
                generator=generator,
                guidance_scale=opts.diffusion_guidance_scale,
                num_inference_steps=opts.diffusion_inference_steps,
                video=cond_video,
                mask_video=cond_masks,
                reference=frames_ref,
            ).videos
        save_video(
            sample[0].permute(1, 2, 3, 0),
            os.path.join(opts.save_dir, 'gen.mp4'),
            fps=opts.fps,
        )

        viz = True
        if viz:
            tensor_left = frames[0].to(opts.device)
            tensor_left_full = torch.cat(
                [tensor_left, tensor_left[:, -1:, :, :].repeat(1, 48, 1, 1)], dim=1
            )
            tensor_right = sample[0].to(opts.device)
            tensor_right_full = torch.cat(
                [tensor_left, tensor_right[:, 1:, :, :]], dim=1
            )
            interval = torch.ones(3, 49 * 2 - 1, 384, 30).to(opts.device)
            result = torch.cat((tensor_left_full, interval, tensor_right_full), dim=3)
            result_reverse = torch.flip(result, dims=[1])
            final_result = torch.cat((result, result_reverse[:, 1:, :, :]), dim=1)
            save_video(
                final_result.permute(1, 2, 3, 0),
                os.path.join(opts.save_dir, 'viz.mp4'),
                fps=opts.fps * 4,
            )

    def infer_zoom(self, opts):
        frames = read_video_frames(
            opts.video_path, opts.video_length, opts.stride, opts.max_res
        )
        prompt = self.get_caption(opts, frames[opts.video_length // 2])
        depths = self.depth_estimater.infer(
            frames,
            opts.near,
            opts.far,
            opts.depth_inference_steps,
            opts.depth_guidance_scale,
            window_size=opts.window_size,
            overlap=opts.overlap,
        ).to(opts.device)
        frames = (
            torch.from_numpy(frames).permute(0, 3, 1, 2).to(opts.device) * 2.0 - 1.0
        )  # 49 576 1024 3 -> 49 3 576 1024, [-1,1]
        assert frames.shape[0] == opts.video_length
        pose_s, pose_t, K = self.get_poses_fixed(opts, depths, num_frames=opts.video_length, f_new=250)

        warped_images = []
        masks = []
        for i in tqdm(range(opts.video_length)):
            warped_frame2, mask2, warped_depth2, flow12 = self.funwarp.forward_warp(
                frames[i : i + 1],
                None,
                depths[i : i + 1],
                pose_s[i : i + 1],
                pose_t[i : i + 1],
                K[0 : 1],
                K[i : i + 1],
                opts.mask,
                twice=False,
            )
            warped_images.append(warped_frame2)
            masks.append(mask2)
        cond_video = (torch.cat(warped_images) + 1.0) / 2.0
        cond_masks = torch.cat(masks)

        frames = F.interpolate(
            frames, size=opts.sample_size, mode='bilinear', align_corners=False
        )
        cond_video = F.interpolate(
            cond_video, size=opts.sample_size, mode='bilinear', align_corners=False
        )
        cond_masks = F.interpolate(cond_masks, size=opts.sample_size, mode='nearest')
        save_video(
            (frames.permute(0, 2, 3, 1) + 1.0) / 2.0,
            os.path.join(opts.save_dir, 'input.mp4'),
            fps=opts.fps,
        )
        save_video(
            cond_video.permute(0, 2, 3, 1),
            os.path.join(opts.save_dir, 'render.mp4'),
            fps=opts.fps,
        )
        save_video(
            cond_masks.repeat(1, 3, 1, 1).permute(0, 2, 3, 1),
            os.path.join(opts.save_dir, 'mask.mp4'),
            fps=opts.fps,
        )

        frames = (frames.permute(1, 0, 2, 3).unsqueeze(0) + 1.0) / 2.0
        frames_ref = frames[:, :, :10, :, :]
        cond_video = cond_video.permute(1, 0, 2, 3).unsqueeze(0)
        cond_masks = (1.0 - cond_masks.permute(1, 0, 2, 3).unsqueeze(0)) * 255.0
        generator = torch.Generator(device=opts.device).manual_seed(opts.seed)

        del self.depth_estimater
        del self.caption_processor
        del self.captioner
        gc.collect()
        _empty_cache()
        with torch.no_grad():
            sample = self.pipeline(
                prompt,
                num_frames=opts.video_length,
                negative_prompt=opts.negative_prompt,
                height=opts.sample_size[0],
                width=opts.sample_size[1],
                generator=generator,
                guidance_scale=opts.diffusion_guidance_scale,
                num_inference_steps=opts.diffusion_inference_steps,
                video=cond_video,
                mask_video=cond_masks,
                reference=frames_ref,
            ).videos
        save_video(
            sample[0].permute(1, 2, 3, 0),
            os.path.join(opts.save_dir, 'gen.mp4'),
            fps=opts.fps,
        )

        viz = True
        if viz:
            tensor_left = frames[0].to(opts.device)
            tensor_right = sample[0].to(opts.device)
            interval = torch.ones(3, 49, 384, 30).to(opts.device)
            result = torch.cat((tensor_left, interval, tensor_right), dim=3)
            result_reverse = torch.flip(result, dims=[1])
            final_result = torch.cat((result, result_reverse[:, 1:, :, :]), dim=1)
            save_video(
                final_result.permute(1, 2, 3, 0),
                os.path.join(opts.save_dir, 'viz.mp4'),
                fps=opts.fps * 2,
            )

    def get_caption(self, opts, image):
        image_array = (image * 255).astype(np.uint8)
        pil_image = Image.fromarray(image_array)
        inputs = self.caption_processor(images=pil_image, return_tensors="pt").to(
            opts.device, torch.float16
        )
        generated_ids = self.captioner.generate(**inputs)
        generated_text = self.caption_processor.batch_decode(
            generated_ids, skip_special_tokens=True
        )[0].strip()
        # return generated_text + opts.refine_prompt
        return generated_text 

    def get_poses(self, opts, depths, num_frames):
        radius = (
            depths[0, 0, depths.shape[-2] // 2, depths.shape[-1] // 2].cpu()
            * opts.radius_scale
        )
        radius = min(radius, 5)

        radius = 1
        
        cx = 512.0  # depths.shape[-1]//2
        cy = 288.0  # depths.shape[-2]//2
        f = 500  # 500.
        K = (
            torch.tensor([[f, 0.0, cx], [0.0, f, cy], [0.0, 0.0, 1.0]])
            .repeat(num_frames, 1, 1)
            .to(opts.device)
        )
        c2w_init = (
            torch.tensor(
                [
                    [-1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, -1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ]
            )
            .to(opts.device)
            .unsqueeze(0)
        )
        if opts.camera == 'target':
            dtheta, dphi, dr, dx, dy = opts.target_pose
            poses = generate_traj_specified(
                c2w_init, dtheta, dphi, dr * radius, dx, dy, num_frames, opts.device
            )
        elif opts.camera == 'traj':
            with open(opts.traj_txt, 'r') as file:
                lines = file.readlines()
                theta = [float(i) for i in lines[0].split()]
                phi = [float(i) for i in lines[1].split()]
                r = [float(i) * radius for i in lines[2].split()]
            poses = generate_traj_txt(c2w_init, phi, theta, r, num_frames, opts.device)
        poses[:, 2, 3] = poses[:, 2, 3] + radius
        pose_s = poses[opts.anchor_idx : opts.anchor_idx + 1].repeat(num_frames, 1, 1)
        pose_t = poses
        return pose_s, pose_t, K

    def get_poses_fixed(self, opts, depths, num_frames, c2w_init=None):
        radius = (
            depths[0, 0, depths.shape[-2] // 2, depths.shape[-1] // 2].cpu()
            * opts.radius_scale
        )
        radius = min(radius, 5)

        radius = 1
        
        cx = 512.0  # depths.shape[-1]//2
        cy = 288.0  # depths.shape[-2]//2
        f = 500  # 500.
        K = (
            torch.tensor([[f, 0.0, cx], [0.0, f, cy], [0.0, 0.0, 1.0]])
            .repeat(num_frames, 1, 1)
            .to(opts.device)
        )
        if c2w_init == None:
            c2w_init = (
                torch.tensor(
                    [
                        [-1.0, 0.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0, 0.0],
                        [0.0, 0.0, -1.0, 0.0],
                        [0.0, 0.0, 0.0, 1.0],
                    ]
                )
                .to(opts.device)
                .unsqueeze(0)
            )
        if opts.camera == 'target':
            dtheta, dphi, dr, dx, dy = opts.target_pose
            poses1 = generate_traj_specified(
                c2w_init, dtheta, dphi, dr * radius, dx, dy, num_frames, opts.device
            )
            poses2 = generate_traj_specified_fixed(
                c2w_init, dtheta, dphi, dr * radius, dx, dy, num_frames, opts.device
            )
        elif opts.camera == 'traj':
            with open(opts.traj_txt, 'r') as file:
                lines = file.readlines()
                theta = [float(i) for i in lines[0].split()]
                phi = [float(i) for i in lines[1].split()]
                r = [float(i) * radius for i in lines[2].split()]
            poses = generate_traj_txt(c2w_init, phi, theta, r, num_frames, opts.device)
        poses1[:, 2, 3] = poses1[:, 2, 3] + radius
        poses2[:, 2, 3] = poses2[:, 2, 3] + radius
        pose_s = poses1[opts.anchor_idx : opts.anchor_idx + 1].repeat(num_frames, 1, 1)
        pose_t = poses2
        return pose_s, pose_t, K


    def get_poses_panorama(self, opts, depths, num_frames, c2w_init=None):
        """Rotate a fixed-position camera using dtheta/dphi; ignore dr, dx, and dy."""
        # Retain intrinsics for compatibility with forward_warp.
        cx, cy, f = 512.0, 288.0, 500
        K = (
            torch.tensor([[f, 0.0, cx], [0.0, f, cy], [0.0, 0.0, 1.0]])
            .repeat(num_frames, 1, 1)
            .to(opts.device)
        )

        if c2w_init is None:
            # Keep the camera position fixed; do not add a radius offset.
            c2w_init = (
                torch.tensor(
                    [
                        [-1.0, 0.0, 0.0, 0.0],
                        [ 0.0, 1.0, 0.0, 0.0],
                        [ 0.0, 0.0,-1.0, 0.0],
                        [ 0.0, 0.0, 0.0, 1.0],
                    ]
                )
                .to(opts.device)
                .unsqueeze(0)
            )

        if opts.camera == 'target':
            dtheta, dphi, _, _, _ = opts.target_pose
            poses1 = generate_traj_panorama(
                c2w_init, dtheta, dphi, num_frames, opts.device
            )
            poses2 = generate_traj_panorama_fixed(
                c2w_init, dtheta, dphi, num_frames, opts.device
            )
        elif opts.camera == 'traj':
            with open(opts.traj_txt, 'r') as file:
                lines = file.readlines()
                theta_list = [float(i) for i in lines[0].split()]
                phi_list   = [float(i) for i in lines[1].split()]
            # Interpolate rotation only; ignore the radius in panorama mode.
            from models.utils import txt_interpolation
            ths = txt_interpolation(theta_list, num_frames,
                                    'smooth' if len(theta_list) > 3 else 'linear')
            phs = txt_interpolation(phi_list,   num_frames,
                                    'smooth' if len(phi_list)   > 3 else 'linear')
            poses1 = torch.cat(
                [sphere2pose_panorama(c2w_init, float(t), float(p), opts.device)
                 for t, p in zip(ths, phs)], dim=0
            )
            poses2 = poses1.clone()

        # Panorama mode does not apply translation.
        pose_s = poses1[opts.anchor_idx : opts.anchor_idx + 1].repeat(num_frames, 1, 1)
        pose_t = poses2
        return pose_s, pose_t, K


    def get_poses_lookfix(self, opts, depths, num_frames, c2w_init=None):
        """Orbit with dtheta/dphi, then translate in camera coordinates.

        Camera-local dx/dy/dr translations preserve the viewing orientation.
        """
        from models.utils import (
            generate_traj_specified_lookfix,
            generate_traj_specified_fixed_lookfix,
            generate_traj_txt,
        )

        radius = (
            depths[0, 0, depths.shape[-2] // 2, depths.shape[-1] // 2].cpu()
            * opts.radius_scale
        )
        radius = min(radius, 5)

        radius = 1

        cx, cy, f = 512.0, 288.0, 500
        K = (
            torch.tensor([[f, 0.0, cx], [0.0, f, cy], [0.0, 0.0, 1.0]])
            .repeat(num_frames, 1, 1).to(opts.device)
        )

        if c2w_init is None:
            c2w_init = torch.tensor(
                [[-1.0, 0.0,  0.0, 0.0],
                [ 0.0, 1.0,  0.0, 0.0],
                [ 0.0, 0.0, -1.0, 0.0],
                [ 0.0, 0.0,  0.0, 1.0]]
            ).to(opts.device).unsqueeze(0)

        if opts.camera == 'target':
            dtheta, dphi, dr, dx, dy = opts.target_pose
            poses1 = generate_traj_specified_lookfix(
                c2w_init, dtheta, dphi, dr * radius, dx, dy, num_frames, opts.device
            )
            poses2 = generate_traj_specified_fixed_lookfix(
                c2w_init, dtheta, dphi, dr * radius, dx, dy, num_frames, opts.device
            )
        elif opts.camera == 'traj':
            with open(opts.traj_txt, 'r') as file:
                lines = file.readlines()
                theta = [float(i) for i in lines[0].split()]
                phi   = [float(i) for i in lines[1].split()]
                r     = [float(i) * radius for i in lines[2].split()]
            # Trajectory-file mode retains the original orbit semantics.
            poses1 = generate_traj_txt(c2w_init, phi, theta, r, num_frames, opts.device)
            poses2 = poses1.clone()

        # Match get_poses_fixed by offsetting the camera from the scene center.
        poses1[:, 2, 3] = poses1[:, 2, 3] + radius
        poses2[:, 2, 3] = poses2[:, 2, 3] + radius

        pose_s = poses1[opts.anchor_idx : opts.anchor_idx + 1].repeat(num_frames, 1, 1)
        pose_t = poses2
        return pose_s, pose_t, K



    def get_poses_f(self, opts, depths, num_frames, f_new):
        radius = (
            depths[0, 0, depths.shape[-2] // 2, depths.shape[-1] // 2].cpu()
            * opts.radius_scale
        )
        radius = min(radius, 5)
        cx = 512.0  
        cy = 288.0  
        f = 500
        # f_new,d_r: 250,0.5; 1000,-0.9
        f_values = torch.linspace(f, f_new, num_frames, device=opts.device)
        K = torch.zeros((num_frames, 3, 3), device=opts.device)
        K[:, 0, 0] = f_values
        K[:, 1, 1] = f_values
        K[:, 0, 2] = cx
        K[:, 1, 2] = cy
        K[:, 2, 2] = 1.0
        c2w_init = (
            torch.tensor(
                [
                    [-1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, -1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ]
            )
            .to(opts.device)
            .unsqueeze(0)
        )
        if opts.camera == 'target':
            dtheta, dphi, dr, dx, dy = opts.target_pose
            poses = generate_traj_specified(
                c2w_init, dtheta, dphi, dr * radius, dx, dy, num_frames, opts.device
            )
        elif opts.camera == 'traj':
            with open(opts.traj_txt, 'r') as file:
                lines = file.readlines()
                theta = [float(i) for i in lines[0].split()]
                phi = [float(i) for i in lines[1].split()]
                r = [float(i) * radius for i in lines[2].split()]
            poses = generate_traj_txt(c2w_init, phi, theta, r, num_frames, opts.device)
        poses[:, 2, 3] = poses[:, 2, 3] + radius
        pose_s = poses[opts.anchor_idx : opts.anchor_idx + 1].repeat(num_frames, 1, 1)
        pose_t = poses
        return pose_s, pose_t, K

    def setup_diffusion(self, opts):
        # transformer = CrossTransformer3DModel.from_pretrained_cus(opts.transformer_path).to(opts.weight_dtype)
        transformer = CrossTransformer3DModel.from_pretrained(opts.transformer_path).to(
            opts.weight_dtype
        )
        # transformer = transformer.to(opts.weight_dtype)
        vae = AutoencoderKLCogVideoX.from_pretrained(
            opts.model_name, subfolder="vae"
        ).to(opts.weight_dtype)
        text_encoder = T5EncoderModel.from_pretrained(
            opts.model_name, subfolder="text_encoder", torch_dtype=opts.weight_dtype
        )
        # Get Scheduler
        Choosen_Scheduler = {
            "Euler": EulerDiscreteScheduler,
            "Euler A": EulerAncestralDiscreteScheduler,
            "DPM++": DPMSolverMultistepScheduler,
            "PNDM": PNDMScheduler,
            "DDIM_Cog": CogVideoXDDIMScheduler,
            "DDIM_Origin": DDIMScheduler,
        }[opts.sampler_name]
        scheduler = Choosen_Scheduler.from_pretrained(
            opts.model_name, subfolder="scheduler"
        )

        self.pipeline = TrajCrafter_Pipeline.from_pretrained(
            opts.model_name,
            vae=vae,
            text_encoder=text_encoder,
            transformer=transformer,
            scheduler=scheduler,
            torch_dtype=opts.weight_dtype,
        )

        if opts.low_gpu_memory_mode:
            self.pipeline.enable_sequential_cpu_offload()
        else:
            self.pipeline.enable_model_cpu_offload()

    def run_gradio(self, input_video, stride, radius_scale, pose, steps, seed):
        frames = read_video_frames(
            input_video, self.opts.video_length, stride, self.opts.max_res
        )
        prompt = self.get_caption(self.opts, frames[self.opts.video_length // 2])
        depths = self.depth_estimater.infer(
            frames,
            self.opts.near,
            self.opts.far,
            self.opts.depth_inference_steps,
            self.opts.depth_guidance_scale,
            window_size=self.opts.window_size,
            overlap=self.opts.overlap,
        ).to(self.opts.device)
        frames = (
            torch.from_numpy(frames).permute(0, 3, 1, 2).to(self.opts.device) * 2.0
            - 1.0
        )  # 49 576 1024 3 -> 49 3 576 1024, [-1,1]
        num_frames = frames.shape[0]
        assert num_frames == self.opts.video_length
        radius_scale = float(radius_scale)
        radius = (
            depths[0, 0, depths.shape[-2] // 2, depths.shape[-1] // 2].cpu()
            * radius_scale
        )
        radius = min(radius, 5)
        cx = 512.0  # depths.shape[-1]//2
        cy = 288.0  # depths.shape[-2]//2
        f = 500  # 500.
        K = (
            torch.tensor([[f, 0.0, cx], [0.0, f, cy], [0.0, 0.0, 1.0]])
            .repeat(num_frames, 1, 1)
            .to(self.opts.device)
        )
        c2w_init = (
            torch.tensor(
                [
                    [-1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, -1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ]
            )
            .to(self.opts.device)
            .unsqueeze(0)
        )

        # import pdb
        # pdb.set_trace()
        theta, phi, r, x, y = [float(i) for i in pose.split(';')]
        # theta,phi,r,x,y = [float(i) for i in theta.split()],[float(i)
        # for i in phi.split()],[float(i)
        # for i in r.split()],[float(i) for i in x.split()],[float(i) for i in y.split()]
        # target mode
        poses = generate_traj_specified(
            c2w_init, theta, phi, r * radius, x, y, num_frames, self.opts.device
        )
        poses[:, 2, 3] = poses[:, 2, 3] + radius
        pose_s = poses[self.opts.anchor_idx : self.opts.anchor_idx + 1].repeat(
            num_frames, 1, 1
        )
        pose_t = poses

        warped_images = []
        masks = []
        for i in tqdm(range(self.opts.video_length)):
            warped_frame2, mask2, warped_depth2, flow12 = self.funwarp.forward_warp(
                frames[i : i + 1],
                None,
                depths[i : i + 1],
                pose_s[i : i + 1],
                pose_t[i : i + 1],
                K[i : i + 1],
                None,
                self.opts.mask,
                twice=False,
            )
            warped_images.append(warped_frame2)
            masks.append(mask2)
        cond_video = (torch.cat(warped_images) + 1.0) / 2.0
        cond_masks = torch.cat(masks)

        frames = F.interpolate(
            frames, size=self.opts.sample_size, mode='bilinear', align_corners=False
        )
        cond_video = F.interpolate(
            cond_video, size=self.opts.sample_size, mode='bilinear', align_corners=False
        )
        cond_masks = F.interpolate(
            cond_masks, size=self.opts.sample_size, mode='nearest'
        )
        save_video(
            (frames.permute(0, 2, 3, 1) + 1.0) / 2.0,
            os.path.join(self.opts.save_dir, 'input.mp4'),
            fps=self.opts.fps,
        )
        save_video(
            cond_video.permute(0, 2, 3, 1),
            os.path.join(self.opts.save_dir, 'render.mp4'),
            fps=self.opts.fps,
        )
        save_video(
            cond_masks.repeat(1, 3, 1, 1).permute(0, 2, 3, 1),
            os.path.join(self.opts.save_dir, 'mask.mp4'),
            fps=self.opts.fps,
        )

        frames = (frames.permute(1, 0, 2, 3).unsqueeze(0) + 1.0) / 2.0
        frames_ref = frames[:, :, :10, :, :]
        cond_video = cond_video.permute(1, 0, 2, 3).unsqueeze(0)
        cond_masks = (1.0 - cond_masks.permute(1, 0, 2, 3).unsqueeze(0)) * 255.0
        generator = torch.Generator(device=self.opts.device).manual_seed(seed)

        # del self.depth_estimater
        # del self.caption_processor
        # del self.captioner
        # gc.collect()
        _empty_cache()
        with torch.no_grad():
            sample = self.pipeline(
                prompt,
                num_frames=self.opts.video_length,
                negative_prompt=self.opts.negative_prompt,
                height=self.opts.sample_size[0],
                width=self.opts.sample_size[1],
                generator=generator,
                guidance_scale=self.opts.diffusion_guidance_scale,
                num_inference_steps=steps,
                video=cond_video,
                mask_video=cond_masks,
                reference=frames_ref,
            ).videos
        save_video(
            sample[0].permute(1, 2, 3, 0),
            os.path.join(self.opts.save_dir, 'gen.mp4'),
            fps=self.opts.fps,
        )

        viz = True
        if viz:
            tensor_left = frames[0].to(self.opts.device)
            tensor_right = sample[0].to(self.opts.device)
            interval = torch.ones(3, 49, 384, 30).to(self.opts.device)
            result = torch.cat((tensor_left, interval, tensor_right), dim=3)
            result_reverse = torch.flip(result, dims=[1])
            final_result = torch.cat((result, result_reverse[:, 1:, :, :]), dim=1)
            save_video(
                final_result.permute(1, 2, 3, 0),
                os.path.join(self.opts.save_dir, 'viz.mp4'),
                fps=self.opts.fps * 2,
            )
        return os.path.join(self.opts.save_dir, 'viz.mp4')
