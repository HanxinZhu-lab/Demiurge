# Modified for Demiurge; see THIRD_PARTY_NOTICES.md.
import numpy as np
import cv2
import PIL
from PIL import Image
import os
from datetime import datetime
import pdb
import torch.nn.functional as F
import numpy as np
import os
import cv2
import copy
from scipy.interpolate import UnivariateSpline, interp1d
import numpy as np
import PIL.Image
import torch
import torchvision
from tqdm import tqdm
from pathlib import Path
from typing import Tuple, Optional
import cv2
import PIL
import numpy
import skimage.io
import torch
import torch.nn.functional as F
from decord import VideoReader, cpu


import os
import numpy as np
from decord import VideoReader, cpu


def _load_rgb_from_npz(path):
    """Load an RGB NPZ sequence with common key, layout, and dtype variants.

    Returns float32 frames of shape (T, H, W, 3) in [0, 1].
    """
    data = np.load(path)
    files = list(data.files)

    candidate_keys = ("video", "rgb", "frames", "data", "arr_0")
    key = None
    for k in candidate_keys:
        if k in files:
            key = k
            break

    # Fall back to a unique 4D array, excluding metadata such as fps and dtype.
    if key is None:
        array_like_keys = [
            k for k in files
            if isinstance(data[k], np.ndarray) and data[k].ndim == 4
        ]
        if len(array_like_keys) == 1:
            key = array_like_keys[0]
        else:
            raise KeyError(
                f"{path} has no RGB array key; candidates={candidate_keys}, "
                f"available keys={files}"
            )

    arr = data[key]
    if arr.ndim != 4:
        raise ValueError(f"{path} key='{key}' expected 4D, got shape={arr.shape}")

    # Channel order: (T,3,H,W) -> (T,H,W,3).
    if arr.shape[1] == 3 and arr.shape[-1] != 3:
        arr = np.transpose(arr, (0, 2, 3, 1))
    elif arr.shape[-1] == 3:
        pass
    else:
        raise ValueError(
            f"{path} key='{key}' has an unrecognized channel dimension, shape={arr.shape}"
        )

    # Normalize uint8 and floating-point values in [0, 255] to [0, 1].
    if arr.dtype == np.uint8:
        arr = arr.astype(np.float32) / 255.0
    else:
        arr = arr.astype(np.float32)
        if arr.max() > 1.5:
            arr = arr / 255.0

    return np.clip(arr, 0.0, 1.0)


def _resize_video_bilinear(frames_thwc, out_h, out_w):
    """
    frames_thwc: (T, H, W, 3) float32, [0,1]
    return: (T, out_h, out_w, 3) float32, [0,1]
    """
    if frames_thwc.shape[1] == out_h and frames_thwc.shape[2] == out_w:
        return frames_thwc

    import torch
    import torch.nn.functional as F

    t = torch.from_numpy(frames_thwc).permute(0, 3, 1, 2)  # (T,3,H,W)
    t = F.interpolate(t, size=(out_h, out_w), mode="bilinear", align_corners=False)
    out = t.permute(0, 2, 3, 1).contiguous().numpy()       # (T,oh,ow,3)
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def read_video_frames(video_path, process_length, stride, max_res, dataset="open"):
    ext = os.path.splitext(video_path)[1].lower()

    # NPZ input.
    if ext == ".npz":
        print("==> processing npz: ", video_path)
        frames = _load_rgb_from_npz(video_path)  # (T, H, W, 3) float32 [0,1]
        print("==> original npz shape: ", frames.shape)

        # FIXME: Fixed resolution, matching the video input branch.
        width = 1024
        height = 576
        frames = _resize_video_bilinear(frames, height, width)
        print(f"==> resized npz shape: {frames.shape}")

        # Match the video branch's stride and process_length behavior.
        frames_idx = list(range(0, frames.shape[0], stride))
        print(
            f"==> downsampled shape: {(len(frames_idx), *frames.shape[1:])}, "
            f"with stride: {stride}"
        )
        if process_length != -1 and process_length < len(frames_idx):
            frames_idx = frames_idx[:process_length]

        frames = frames[frames_idx]
        print(f"==> final processing shape: {frames.shape}")
        return frames.astype(np.float32)

    # Video input.
    if dataset == "open":
        print("==> processing video: ", video_path)
        vid = VideoReader(video_path, ctx=cpu(0))
        print("==> original video shape: ", (len(vid), *vid.get_batch([0]).shape[1:]))

        # FIXME: hard coded
        width = 1024
        height = 576

    vid = VideoReader(video_path, ctx=cpu(0), width=width, height=height)

    frames_idx = list(range(0, len(vid), stride))
    print(
        f"==> downsampled shape: {len(frames_idx), *vid.get_batch([0]).shape[1:]}, with stride: {stride}"
    )
    if process_length != -1 and process_length < len(frames_idx):
        frames_idx = frames_idx[:process_length]
    print(
        f"==> final processing shape: {len(frames_idx), *vid.get_batch([0]).shape[1:]}"
    )
    frames = vid.get_batch(frames_idx).asnumpy().astype("float32") / 255.0

    return frames



def save_video(data, images_path, folder=None, fps=8):
    if isinstance(data, np.ndarray):
        tensor_data = (torch.from_numpy(data) * 255).to(torch.uint8)
    elif isinstance(data, torch.Tensor):
        tensor_data = (data.detach().cpu() * 255).to(torch.uint8)
    elif isinstance(data, list):
        folder = [folder] * len(data)
        images = [
            np.array(Image.open(os.path.join(folder_name, path)))
            for folder_name, path in zip(folder, data)
        ]
        stacked_images = np.stack(images, axis=0)
        tensor_data = torch.from_numpy(stacked_images).to(torch.uint8)
    torchvision.io.write_video(
        images_path, tensor_data, fps=fps, video_codec='h264', options={'crf': '10'}
    )


def sphere2pose(c2ws_input, theta, phi, r, device, x=None, y=None):
    c2ws = copy.deepcopy(c2ws_input)
    # Translate along the world Z axis before rotating.
    c2ws[:, 2, 3] -= r
    if x is not None:
        c2ws[:, 1, 3] += y
    if y is not None:
        c2ws[:, 0, 3] -= x

    theta = torch.deg2rad(torch.tensor(theta)).to(device)
    sin_value_x = torch.sin(theta)
    cos_value_x = torch.cos(theta)
    rot_mat_x = (
        torch.tensor(
            [
                [1, 0, 0, 0],
                [0, cos_value_x, -sin_value_x, 0],
                [0, sin_value_x, cos_value_x, 0],
                [0, 0, 0, 1],
            ]
        )
        .unsqueeze(0)
        .repeat(c2ws.shape[0], 1, 1)
        .to(device)
    )

    phi = torch.deg2rad(torch.tensor(phi)).to(device)
    sin_value_y = torch.sin(phi)
    cos_value_y = torch.cos(phi)
    rot_mat_y = (
        torch.tensor(
            [
                [cos_value_y, 0, sin_value_y, 0],
                [0, 1, 0, 0],
                [-sin_value_y, 0, cos_value_y, 0],
                [0, 0, 0, 1],
            ]
        )
        .unsqueeze(0)
        .repeat(c2ws.shape[0], 1, 1)
        .to(device)
    )

    c2ws = torch.matmul(rot_mat_x, c2ws)
    c2ws = torch.matmul(rot_mat_y, c2ws)
    return c2ws


def generate_traj_specified(c2ws_anchor, theta, phi, d_r, d_x, d_y, frame, device):
    # Initialize a camera.
    thetas = np.linspace(0, theta, frame)
    phis = np.linspace(0, phi, frame)
    rs = np.linspace(0, d_r, frame)
    xs = np.linspace(0, d_x, frame)
    ys = np.linspace(0, d_y, frame)
    c2ws_list = []
    for th, ph, r, x, y in zip(thetas, phis, rs, xs, ys):
        c2w_new = sphere2pose(
            c2ws_anchor,
            np.float32(th),
            np.float32(ph),
            np.float32(r),
            device,
            np.float32(x),
            np.float32(y),
        )
        c2ws_list.append(c2w_new)
    c2ws = torch.cat(c2ws_list, dim=0)
    return c2ws


def generate_traj_specified_fixed(c2ws_anchor, theta, phi, d_r, d_x, d_y, frame, device):
    # Initialize a camera.
    thetas = np.linspace(theta, theta, frame)
    phis = np.linspace(phi, phi, frame)
    rs = np.linspace(d_r, d_r, frame)
    xs = np.linspace(d_x, d_x, frame)
    ys = np.linspace(d_y, d_y, frame)

    c2ws_list = []
    for th, ph, r, x, y in zip(thetas, phis, rs, xs, ys):
        c2w_new = sphere2pose(
            c2ws_anchor,
            np.float32(th),
            np.float32(ph),
            np.float32(r),
            device,
            np.float32(x),
            np.float32(y),
        )
        c2ws_list.append(c2w_new)
    c2ws = torch.cat(c2ws_list, dim=0)
    return c2ws


def sphere2pose_panorama(c2ws_input, theta, phi, device):
    """Rotate in camera coordinates without changing the camera position.

    Right-multiply c2w by rotations about the local X and Y axes.
    theta is pitch and phi is yaw, both in degrees.
    """
    c2ws = copy.deepcopy(c2ws_input)

    theta = torch.deg2rad(torch.tensor(theta)).to(device)
    sin_x, cos_x = torch.sin(theta), torch.cos(theta)
    rot_mat_x = (
        torch.tensor(
            [
                [1, 0, 0, 0],
                [0, cos_x, -sin_x, 0],
                [0, sin_x,  cos_x, 0],
                [0, 0, 0, 1],
            ]
        )
        .unsqueeze(0)
        .repeat(c2ws.shape[0], 1, 1)
        .to(device)
    )

    phi = torch.deg2rad(torch.tensor(phi)).to(device)
    sin_y, cos_y = torch.sin(phi), torch.cos(phi)
    rot_mat_y = (
        torch.tensor(
            [
                [ cos_y, 0, sin_y, 0],
                [     0, 1,     0, 0],
                [-sin_y, 0, cos_y, 0],
                [     0, 0,     0, 1],
            ]
        )
        .unsqueeze(0)
        .repeat(c2ws.shape[0], 1, 1)
        .to(device)
    )

    # Right multiplication changes orientation, not position.
    c2ws = torch.matmul(c2ws, rot_mat_y)
    c2ws = torch.matmul(c2ws, rot_mat_x)
    return c2ws


def generate_traj_panorama(c2ws_anchor, theta, phi, frame, device):
    """Generate a target trajectory rotating in place from (0, 0) to (theta, phi)."""
    thetas = np.linspace(0, theta, frame)
    phis   = np.linspace(0, phi,   frame)
    c2ws_list = []
    for th, ph in zip(thetas, phis):
        c2w_new = sphere2pose_panorama(
            c2ws_anchor, np.float32(th), np.float32(ph), device
        )
        c2ws_list.append(c2w_new)
    return torch.cat(c2ws_list, dim=0)


def generate_traj_panorama_fixed(c2ws_anchor, theta, phi, frame, device):
    """Generate a source trajectory with every frame fixed at (theta, phi)."""
    c2ws_list = []
    for _ in range(frame):
        c2w_new = sphere2pose_panorama(
            c2ws_anchor, np.float32(theta), np.float32(phi), device
        )
        c2ws_list.append(c2w_new)
    return torch.cat(c2ws_list, dim=0)



def txt_interpolation(input_list, n, mode='smooth'):
    x = np.linspace(0, 1, len(input_list))
    if mode == 'smooth':
        f = UnivariateSpline(x, input_list, k=3)
    elif mode == 'linear':
        f = interp1d(x, input_list)
    else:
        raise KeyError(f"Invalid txt interpolation mode: {mode}")
    xnew = np.linspace(0, 1, n)
    ynew = f(xnew)
    return ynew


def generate_traj_txt(c2ws_anchor, phi, theta, r, frame, device):
    # Initialize a camera.
    """
    The camera coordinate sysmte in COLMAP is right-down-forward
    Pytorch3D is left-up-forward
    """

    if len(phi) > 3:
        phis = txt_interpolation(phi, frame, mode='smooth')
        phis[0] = phi[0]
        phis[-1] = phi[-1]
    else:
        phis = txt_interpolation(phi, frame, mode='linear')

    if len(theta) > 3:
        thetas = txt_interpolation(theta, frame, mode='smooth')
        thetas[0] = theta[0]
        thetas[-1] = theta[-1]
    else:
        thetas = txt_interpolation(theta, frame, mode='linear')

    if len(r) > 3:
        rs = txt_interpolation(r, frame, mode='smooth')
        rs[0] = r[0]
        rs[-1] = r[-1]
    else:
        rs = txt_interpolation(r, frame, mode='linear')

    c2ws_list = []
    for th, ph, r in zip(thetas, phis, rs):
        c2w_new = sphere2pose(
            c2ws_anchor, np.float32(th), np.float32(ph), np.float32(r), device
        )
        c2ws_list.append(c2w_new)
    c2ws = torch.cat(c2ws_list, dim=0)
    return c2ws



def sphere2pose_then_local_translate(c2ws_input, theta, phi, r, device, x=None, y=None):
    """Rotate about world axes, then translate in camera coordinates.

    Left multiplication sets the orientation, as in sphere2pose. The local
    translation (x, y, r) is right-multiplied and preserves that orientation.
    """
    c2ws = copy.deepcopy(c2ws_input)

    # World-axis rotation.
    theta_t = torch.deg2rad(torch.tensor(theta)).to(device)
    sx, cx = torch.sin(theta_t), torch.cos(theta_t)
    rot_x = torch.tensor(
        [[1, 0, 0, 0],
         [0, cx, -sx, 0],
         [0, sx,  cx, 0],
         [0, 0, 0, 1]]
    ).unsqueeze(0).repeat(c2ws.shape[0], 1, 1).to(device)

    phi_t = torch.deg2rad(torch.tensor(phi)).to(device)
    sy, cy = torch.sin(phi_t), torch.cos(phi_t)
    rot_y = torch.tensor(
        [[ cy, 0, sy, 0],
         [  0, 1,  0, 0],
         [-sy, 0, cy, 0],
         [  0, 0,  0, 1]]
    ).unsqueeze(0).repeat(c2ws.shape[0], 1, 1).to(device)

    c2ws = torch.matmul(rot_x, c2ws)
    c2ws = torch.matmul(rot_y, c2ws)

    # Right-multiply the local translation, preserving the homogeneous entry (3,3)=1.
    if x is None: x = 0.0
    if y is None: y = 0.0
    T_local = torch.eye(4, device=device, dtype=c2ws.dtype) \
        .unsqueeze(0).repeat(c2ws.shape[0], 1, 1).clone()
    T_local[:, 0, 3] = -float(x)
    T_local[:, 1, 3] =  float(y)
    T_local[:, 2, 3] = -float(r)
    c2ws = torch.matmul(c2ws, T_local)

    return c2ws

def generate_traj_specified_lookfix(c2ws_anchor, theta, phi, d_r, d_x, d_y, frame, device):
    """Interpolate orbital orientation and camera-local translation independently."""
    thetas = np.linspace(0, theta, frame)
    phis   = np.linspace(0, phi,   frame)
    rs = np.linspace(0, d_r, frame)
    xs = np.linspace(0, d_x, frame)
    ys = np.linspace(0, d_y, frame)
    c2ws_list = []
    for th, ph, r, x, y in zip(thetas, phis, rs, xs, ys):
        c2w_new = sphere2pose_then_local_translate(
            c2ws_anchor,
            np.float32(th), np.float32(ph), np.float32(r),
            device,
            np.float32(x), np.float32(y),
        )
        c2ws_list.append(c2w_new)
    return torch.cat(c2ws_list, dim=0)


def generate_traj_specified_fixed_lookfix(c2ws_anchor, theta, phi, d_r, d_x, d_y, frame, device):
    """Return the fixed final pose, matching generate_traj_specified_fixed's interface."""
    thetas = np.linspace(theta, theta, frame)
    phis   = np.linspace(phi,   phi,   frame)
    rs = np.linspace(d_r, d_r, frame)
    xs = np.linspace(d_x, d_x, frame)
    ys = np.linspace(d_y, d_y, frame)
    c2ws_list = []
    for th, ph, r, x, y in zip(thetas, phis, rs, xs, ys):
        c2w_new = sphere2pose_then_local_translate(
            c2ws_anchor,
            np.float32(th), np.float32(ph), np.float32(r),
            device,
            np.float32(x), np.float32(y),
        )
        c2ws_list.append(c2w_new)
    return torch.cat(c2ws_list, dim=0)



class Warper:
    def __init__(self, resolution: tuple = None, device: str = 'gpu0'):
        self.resolution = resolution
        self.device = self.get_device(device)
        self.dtype = torch.float32
        return

    def forward_warp(
        self,
        frame1: torch.Tensor,
        mask1: Optional[torch.Tensor],
        depth1: torch.Tensor,
        transformation1: torch.Tensor,
        transformation2: torch.Tensor,
        intrinsic1: torch.Tensor,
        intrinsic2: Optional[torch.Tensor],
        mask=False,
        twice=False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Given a frame1 and global transformations transformation1 and transformation2, warps frame1 to next view using
        bilinear splatting.
        All arrays should be torch tensors with batch dimension and channel first
        :param frame1: (b, 3, h, w). If frame1 is not in the range [-1, 1], either set is_image=False when calling
                        bilinear_splatting on frame within this function, or modify clipping in bilinear_splatting()
                        method accordingly.
        :param mask1: (b, 1, h, w) - 1 for known, 0 for unknown. Optional
        :param depth1: (b, 1, h, w)
        :param transformation1: (b, 4, 4) extrinsic transformation matrix of first view: [R, t; 0, 1]
        :param transformation2: (b, 4, 4) extrinsic transformation matrix of second view: [R, t; 0, 1]
        :param intrinsic1: (b, 3, 3) camera intrinsic matrix
        :param intrinsic2: (b, 3, 3) camera intrinsic matrix. Optional
        """
        if self.resolution is not None:
            assert frame1.shape[2:4] == self.resolution
        b, c, h, w = frame1.shape
        if mask1 is None:
            mask1 = torch.ones(size=(b, 1, h, w)).to(frame1)
        if intrinsic2 is None:
            intrinsic2 = intrinsic1.clone()

        assert frame1.shape == (b, 3, h, w)
        assert mask1.shape == (b, 1, h, w)
        assert depth1.shape == (b, 1, h, w)
        assert transformation1.shape == (b, 4, 4)
        assert transformation2.shape == (b, 4, 4)
        assert intrinsic1.shape == (b, 3, 3)
        assert intrinsic2.shape == (b, 3, 3)

        frame1 = frame1.to(self.device).to(self.dtype)
        mask1 = mask1.to(self.device).to(self.dtype)
        depth1 = depth1.to(self.device).to(self.dtype)
        transformation1 = transformation1.to(self.device).to(self.dtype)
        transformation2 = transformation2.to(self.device).to(self.dtype)
        intrinsic1 = intrinsic1.to(self.device).to(self.dtype)
        intrinsic2 = intrinsic2.to(self.device).to(self.dtype)

        trans_points1 = self.compute_transformed_points(
            depth1, transformation1, transformation2, intrinsic1, intrinsic2
        )
        trans_coordinates = (
            trans_points1[:, :, :, :2, 0] / trans_points1[:, :, :, 2:3, 0]
        )
        trans_depth1 = trans_points1[:, :, :, 2, 0]
        grid = self.create_grid(b, h, w).to(trans_coordinates)
        flow12 = trans_coordinates.permute(0, 3, 1, 2) - grid

        warped_depth2, depth_mask2 = self.bilinear_splatting(
            frame1=trans_depth1.unsqueeze(1),   # (b, 1, h, w)
            mask1=mask1,
            depth1=trans_depth1,                # Depth used for splatting weights.
            flow12=flow12,
            flow12_mask=None,
            is_image=False,
        )
        # warped_depth2: (b, 1, h, w)

        
        
        if not twice:
            warped_frame2, mask2 = self.bilinear_splatting(
                frame1, mask1, trans_depth1, flow12, None, is_image=True
            )
            mask3 = mask2
            if mask:
                warped_depth2 = warped_depth2
                warped_frame2, mask2 = self.clean_points_fast(warped_frame2, mask2)
                warped_depth2 = warped_depth2 * mask2
                warped_depth2 = warped_depth2[:,0:1,:,:]
            return warped_frame2, mask2, warped_depth2, None, flow12

        else:
            warped_frame2, mask2 = self.bilinear_splatting(
                frame1, mask1, trans_depth1, flow12, None, is_image=True
            )
            warped_frame2, mask2 = self.clean_points(warped_frame2, mask2)
            warped_flow, _ = self.bilinear_splatting(
                flow12, mask1, trans_depth1, flow12, None, is_image=False
            )
            twice_warped_frame1, twice_warped_mask1 = self.bilinear_splatting(
                warped_frame2,
                mask2,
                warped_depth2.squeeze(1),
                -warped_flow,
                None,
                is_image=True,
            )

            twice_warped_depth1, _ = self.bilinear_splatting(
                warped_depth2,
                depth_mask2,
                warped_depth2.squeeze(1),
                -warped_flow,
                None,
                is_image=False,
            )
            return twice_warped_frame1, twice_warped_mask1, twice_warped_depth1, None, flow12


    def forward_warp_exact_fusion(
        self,
        frames,
        masks,
        depths,
        transformations_src,
        transformation_tgt,
        intrinsics_src,
        intrinsic_tgt,
        mask=False,
        twice=False,
    ):
        assert isinstance(frames, (list, tuple)) and len(frames) > 0
        num_src = len(frames)
        assert len(depths) == num_src
        assert len(transformations_src) == num_src
        assert len(intrinsics_src) == num_src
        if masks is None:
            masks = [None] * num_src
        else:
            assert len(masks) == num_src

        b, c, h, w = frames[0].shape
        device = frames[0].device
        dtype = frames[0].dtype

        merged_frame = torch.full((b, 3, h, w), -1.0, device=device, dtype=dtype)
        merged_mask  = torch.zeros((b, 1, h, w), device=device, dtype=dtype)
        merged_depth = torch.zeros((b, 1, h, w), device=device, dtype=depths[0].dtype)

        flows = []
        DEPTH_VALID_THR = 1e-5   # Reject nonpositive depths, holes, and numerical noise.

        for frame_i, mask_i, depth_i, trans_i, K_i in zip(
            frames, masks, depths, transformations_src, intrinsics_src
        ):
            warped_frame_i, warped_mask_i, warped_depth_i, _, flow_i = self.forward_warp(
                frame_i, mask_i, depth_i,
                trans_i, transformation_tgt,
                K_i, intrinsic_tgt,
                mask=True, twice=twice,
            )

            warped_frame_i = warped_frame_i.to(merged_frame.device, dtype=merged_frame.dtype)
            warped_mask_i  = warped_mask_i.to(merged_mask.device,   dtype=merged_mask.dtype)
            warped_depth_i = warped_depth_i.to(merged_depth.device, dtype=merged_depth.dtype)
            if flow_i is not None:
                flow_i = flow_i.to(merged_frame.device)
            flows.append(flow_i)

            # Validate the current source and the accumulated output.
            valid_i = (warped_mask_i > 0.5) & (warped_depth_i > DEPTH_VALID_THR)   # (b,1,h,w) bool

            valid_merged = (merged_mask > 0.5) & (merged_depth > DEPTH_VALID_THR)  # (b,1,h,w) bool

            # Fill only valid source pixels not already covered by a valid result.
            fill_region = valid_i & (~valid_merged)

            merged_frame = torch.where(
                fill_region.expand_as(merged_frame),
                warped_frame_i,
                merged_frame,
            )
            merged_depth = torch.where(
                fill_region,
                warped_depth_i,
                merged_depth,
            )
            merged_mask = torch.where(
                fill_region,
                warped_mask_i,
                merged_mask,
            )

        # Use consistent output values for any remaining invalid pixels.
        final_invalid = ~((merged_mask > 0.5) & (merged_depth > DEPTH_VALID_THR))
        merged_frame = torch.where(
            final_invalid.expand_as(merged_frame),
            torch.full_like(merged_frame, -1.0),
            merged_frame,
        )
        merged_mask  = torch.where(final_invalid, torch.zeros_like(merged_mask), merged_mask)
        merged_depth = torch.where(final_invalid, torch.zeros_like(merged_depth), merged_depth)

        return merged_frame, merged_mask, merged_depth, None, flows
    
    def forward_warp_exact_fusion_zbuffer(
        self,
        frames,
        masks,
        depths,
        transformations_src,
        transformation_tgt,
        intrinsics_src,
        intrinsic_tgt,
        mask=False,
        twice=False,
        depth_eps: float = 0.0,
    ):
        """Fuse forward-warped sources using nearest-depth-wins z-buffering.

        Unlike first-valid fusion, overlapping pixels use the smallest valid
        Z value in target camera coordinates.

        Args:
            frames, masks, depths, transformations_src, intrinsics_src:
                Lists of N sources with batch dimensions matching the target.
            transformation_tgt: Target transforms of shape (b, 4, 4).
            intrinsic_tgt: Target intrinsics of shape (b, 3, 3).
            mask: Clean pixels within each forward warp.
            twice: Warp back to the original view.
            depth_eps: Minimum depth improvement required to replace a pixel.

        Returns:
            merged_frame (b, 3, h, w), merged_mask (b, 1, h, w),
            merged_depth (b, 1, h, w), None, and a list of N source flows.
            Uncovered pixels have frame value -1, mask 0, and depth 0.
        """
        assert isinstance(frames, (list, tuple)) and len(frames) > 0
        num_src = len(frames)
        assert len(depths) == num_src
        assert len(transformations_src) == num_src
        assert len(intrinsics_src) == num_src
        if masks is None:
            masks = [None] * num_src
        else:
            assert len(masks) == num_src

        b, c, h, w = frames[0].shape
        device = frames[0].device
        dtype = frames[0].dtype
        ddtype = depths[0].dtype

        DEPTH_VALID_THR = 1e-5
        # A large sentinel allows the first valid source pixel to be accepted.
        BIG_DEPTH = (
            torch.finfo(ddtype).max if ddtype.is_floating_point else 1e9
        )

        merged_frame = torch.full((b, 3, h, w), -1.0, device=device, dtype=dtype)
        merged_mask  = torch.zeros((b, 1, h, w), device=device, dtype=dtype)
        merged_depth = torch.full((b, 1, h, w), BIG_DEPTH,
                                  device=device, dtype=ddtype)

        flows = []

        for frame_i, mask_i, depth_i, trans_i, K_i in zip(
            frames, masks, depths, transformations_src, intrinsics_src
        ):
            warped_frame_i, warped_mask_i, warped_depth_i, _, flow_i = self.forward_warp(
                frame_i, mask_i, depth_i,
                trans_i, transformation_tgt,
                K_i, intrinsic_tgt,
                mask=mask, twice=twice,
            )

            warped_frame_i = warped_frame_i.to(device=device, dtype=dtype)
            warped_mask_i  = warped_mask_i.to(device=device, dtype=dtype)
            warped_depth_i = warped_depth_i.to(device=device, dtype=ddtype)
            flows.append(flow_i)

            # Valid pixels from the current source.
            valid_i = (warped_mask_i > 0.5) & (warped_depth_i > DEPTH_VALID_THR)  # (b,1,h,w)

            # Fill empty pixels, or replace existing pixels with a closer source.
            merged_empty = (merged_mask <= 0.5) | (merged_depth >= BIG_DEPTH)
            closer = warped_depth_i + depth_eps < merged_depth
            fill_region = valid_i & (merged_empty | closer)

            merged_frame = torch.where(
                fill_region.expand_as(merged_frame),
                warped_frame_i,
                merged_frame,
            )
            merged_depth = torch.where(
                fill_region,
                warped_depth_i,
                merged_depth,
            )
            merged_mask = torch.where(
                fill_region,
                warped_mask_i,
                merged_mask,
            )

        # Clear masks and depths for pixels not covered by any valid source.
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
        merged_mask  = torch.where(final_invalid, torch.zeros_like(merged_mask),  merged_mask)
        merged_depth = torch.where(final_invalid, torch.zeros_like(merged_depth), merged_depth)

        return merged_frame, merged_mask, merged_depth, None, flows


    def forward_warp_exact_fusion_zbuffer_batched(
        self,
        frames,
        masks,
        depths,
        transformations_src,
        transformation_tgt,
        intrinsics_src,
        intrinsic_tgt,
        mask=False,
        twice=False,
        depth_eps: float = 0.0,
    ):
        """Batch N sources into one forward warp, then fuse by nearest depth.

        Each source must have batch size 1. Returns merged_frame, merged_mask,
        merged_depth, None, None.
        """
        assert isinstance(frames, (list, tuple)) and len(frames) > 0
        N = len(frames)
        assert frames[0].shape[0] == 1, \
            "Batched fusion requires batch size 1 per source; use frame chunks for larger batches"

        device = frames[0].device
        dtype  = frames[0].dtype
        ddtype = depths[0].dtype
        H, W = frames[0].shape[-2:]

        # Concatenate N sources along the batch dimension.
        frames_b = torch.cat(frames, dim=0)                              # (N, 3, H, W)
        depths_b = torch.cat(depths, dim=0)                              # (N, 1, H, W)
        if masks is None or all(m is None for m in masks):
            masks_b = torch.ones((N, 1, H, W), device=device, dtype=dtype)
        else:
            masks_b = torch.cat(
                [m if m is not None else torch.ones((1, 1, H, W),
                                                    device=device, dtype=dtype)
                 for m in masks], dim=0,
            )                                                             # (N, 1, H, W)
        trans_s_b = torch.cat(transformations_src, dim=0)                # (N, 4, 4)
        K_s_b     = torch.cat(intrinsics_src,      dim=0)                # (N, 3, 3)
        trans_t_b = transformation_tgt.expand(N, -1, -1).contiguous()    # (N, 4, 4)
        K_t_b     = intrinsic_tgt.expand(N, -1, -1).contiguous()         # (N, 3, 3)

        # Warp all sources in one call.
        warped_frame, warped_mask, warped_depth, _, _ = self.forward_warp(
            frames_b, masks_b, depths_b,
            trans_s_b, trans_t_b,
            K_s_b, K_t_b,
            mask=mask, twice=twice,
        )
        # warped_frame: (N, 3, H, W); warped_mask/depth: (N, 1, H, W)

        # Fuse the z-buffer with a single argmin.
        DEPTH_VALID_THR = 1e-5
        BIG_DEPTH = (torch.finfo(warped_depth.dtype).max
                     if warped_depth.dtype.is_floating_point else 1e9)

        valid = (warped_mask > 0.5) & (warped_depth > DEPTH_VALID_THR)   # (N,1,H,W)
        # Raise invalid depths to a sentinel so valid sources take priority.
        depth_for_z = torch.where(
            valid, warped_depth + depth_eps,
            torch.full_like(warped_depth, BIG_DEPTH),
        )
        best = depth_for_z.argmin(dim=0, keepdim=True)                    # (1, 1, H, W)
        any_valid = valid.any(dim=0, keepdim=True)                        # (1, 1, H, W)

        def _gather(x):
            idx = best.expand(1, x.shape[1], H, W)
            return torch.gather(x, dim=0, index=idx)

        merged_frame = _gather(warped_frame)
        merged_depth = _gather(warped_depth)
        merged_mask  = _gather(warped_mask)

        # Fill pixels not covered by any valid source with invalid output values.
        merged_frame = torch.where(
            any_valid.expand_as(merged_frame), merged_frame,
            torch.full_like(merged_frame, -1.0),
        )
        merged_mask  = torch.where(any_valid, merged_mask,
                                   torch.zeros_like(merged_mask))
        merged_depth = torch.where(any_valid, merged_depth,
                                   torch.zeros_like(merged_depth))

        return merged_frame, merged_mask, merged_depth, None, None




    def compute_transformed_points(
        self,
        depth1: torch.Tensor,
        transformation1: torch.Tensor,
        transformation2: torch.Tensor,
        intrinsic1: torch.Tensor,
        intrinsic2: Optional[torch.Tensor],
    ):
        """
        Computes transformed position for each pixel location
        """
        if self.resolution is not None:
            assert depth1.shape[2:4] == self.resolution
        b, _, h, w = depth1.shape
        if intrinsic2 is None:
            intrinsic2 = intrinsic1.clone()
        transformation = torch.bmm(
            transformation2, torch.linalg.inv(transformation1)
        )  # (b, 4, 4)

        x1d = torch.arange(0, w)[None]
        y1d = torch.arange(0, h)[:, None]
        x2d = x1d.repeat([h, 1]).to(depth1)  # (h, w)
        y2d = y1d.repeat([1, w]).to(depth1)  # (h, w)
        ones_2d = torch.ones(size=(h, w)).to(depth1)  # (h, w)
        ones_4d = ones_2d[None, :, :, None, None].repeat(
            [b, 1, 1, 1, 1]
        )  # (b, h, w, 1, 1)
        pos_vectors_homo = torch.stack([x2d, y2d, ones_2d], dim=2)[
            None, :, :, :, None
        ]  # (1, h, w, 3, 1)

        intrinsic1_inv = torch.linalg.inv(intrinsic1)  # (b, 3, 3)
        intrinsic1_inv_4d = intrinsic1_inv[:, None, None]  # (b, 1, 1, 3, 3)
        intrinsic2_4d = intrinsic2[:, None, None]  # (b, 1, 1, 3, 3)
        depth_4d = depth1[:, 0][:, :, :, None, None]  # (b, h, w, 1, 1)
        trans_4d = transformation[:, None, None]  # (b, 1, 1, 4, 4)

        unnormalized_pos = torch.matmul(
            intrinsic1_inv_4d, pos_vectors_homo
        )  # (b, h, w, 3, 1)
        world_points = depth_4d * unnormalized_pos  # (b, h, w, 3, 1)
        world_points_homo = torch.cat([world_points, ones_4d], dim=3)  # (b, h, w, 4, 1)
        trans_world_homo = torch.matmul(trans_4d, world_points_homo)  # (b, h, w, 4, 1)
        trans_world = trans_world_homo[:, :, :, :3]  # (b, h, w, 3, 1)
        trans_norm_points = torch.matmul(intrinsic2_4d, trans_world)  # (b, h, w, 3, 1)
        return trans_norm_points

    def bilinear_splatting(
        self,
        frame1: torch.Tensor,
        mask1: Optional[torch.Tensor],
        depth1: torch.Tensor,
        flow12: torch.Tensor,
        flow12_mask: Optional[torch.Tensor],
        is_image: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Bilinear splatting
        :param frame1: (b,c,h,w)
        :param mask1: (b,1,h,w): 1 for known, 0 for unknown. Optional
        :param depth1: (b,1,h,w)
        :param flow12: (b,2,h,w)
        :param flow12_mask: (b,1,h,w): 1 for valid flow, 0 for invalid flow. Optional
        :param is_image: if true, output will be clipped to (-1,1) range
        :return: warped_frame2: (b,c,h,w)
                 mask2: (b,1,h,w): 1 for known and 0 for unknown
        """
        if self.resolution is not None:
            assert frame1.shape[2:4] == self.resolution
        b, c, h, w = frame1.shape
        if mask1 is None:
            mask1 = torch.ones(size=(b, 1, h, w)).to(frame1)
        if flow12_mask is None:
            flow12_mask = torch.ones(size=(b, 1, h, w)).to(flow12)
        grid = self.create_grid(b, h, w).to(frame1)
        trans_pos = flow12 + grid

        trans_pos_offset = trans_pos + 1
        trans_pos_floor = torch.floor(trans_pos_offset).long()
        trans_pos_ceil = torch.ceil(trans_pos_offset).long()
        trans_pos_offset = torch.stack(
            [
                torch.clamp(trans_pos_offset[:, 0], min=0, max=w + 1),
                torch.clamp(trans_pos_offset[:, 1], min=0, max=h + 1),
            ],
            dim=1,
        )
        trans_pos_floor = torch.stack(
            [
                torch.clamp(trans_pos_floor[:, 0], min=0, max=w + 1),
                torch.clamp(trans_pos_floor[:, 1], min=0, max=h + 1),
            ],
            dim=1,
        )
        trans_pos_ceil = torch.stack(
            [
                torch.clamp(trans_pos_ceil[:, 0], min=0, max=w + 1),
                torch.clamp(trans_pos_ceil[:, 1], min=0, max=h + 1),
            ],
            dim=1,
        )

        prox_weight_nw = (1 - (trans_pos_offset[:, 1:2] - trans_pos_floor[:, 1:2])) * (
            1 - (trans_pos_offset[:, 0:1] - trans_pos_floor[:, 0:1])
        )
        prox_weight_sw = (1 - (trans_pos_ceil[:, 1:2] - trans_pos_offset[:, 1:2])) * (
            1 - (trans_pos_offset[:, 0:1] - trans_pos_floor[:, 0:1])
        )
        prox_weight_ne = (1 - (trans_pos_offset[:, 1:2] - trans_pos_floor[:, 1:2])) * (
            1 - (trans_pos_ceil[:, 0:1] - trans_pos_offset[:, 0:1])
        )
        prox_weight_se = (1 - (trans_pos_ceil[:, 1:2] - trans_pos_offset[:, 1:2])) * (
            1 - (trans_pos_ceil[:, 0:1] - trans_pos_offset[:, 0:1])
        )

        sat_depth1 = torch.clamp(depth1, min=0, max=1000)
        log_depth1 = torch.log(1 + sat_depth1)
        depth_weights = torch.exp(log_depth1 / log_depth1.max() * 50)

        weight_nw = torch.moveaxis(
            prox_weight_nw * mask1 * flow12_mask / depth_weights.unsqueeze(1),
            [0, 1, 2, 3],
            [0, 3, 1, 2],
        )
        weight_sw = torch.moveaxis(
            prox_weight_sw * mask1 * flow12_mask / depth_weights.unsqueeze(1),
            [0, 1, 2, 3],
            [0, 3, 1, 2],
        )
        weight_ne = torch.moveaxis(
            prox_weight_ne * mask1 * flow12_mask / depth_weights.unsqueeze(1),
            [0, 1, 2, 3],
            [0, 3, 1, 2],
        )
        weight_se = torch.moveaxis(
            prox_weight_se * mask1 * flow12_mask / depth_weights.unsqueeze(1),
            [0, 1, 2, 3],
            [0, 3, 1, 2],
        )

        warped_frame = torch.zeros(size=(b, h + 2, w + 2, c), dtype=torch.float32).to(
            frame1
        )
        warped_weights = torch.zeros(size=(b, h + 2, w + 2, 1), dtype=torch.float32).to(
            frame1
        )

        frame1_cl = torch.moveaxis(frame1, [0, 1, 2, 3], [0, 3, 1, 2])
        batch_indices = torch.arange(b)[:, None, None].to(frame1.device)
        warped_frame.index_put_(
            (batch_indices, trans_pos_floor[:, 1], trans_pos_floor[:, 0]),
            frame1_cl * weight_nw,
            accumulate=True,
        )
        warped_frame.index_put_(
            (batch_indices, trans_pos_ceil[:, 1], trans_pos_floor[:, 0]),
            frame1_cl * weight_sw,
            accumulate=True,
        )
        warped_frame.index_put_(
            (batch_indices, trans_pos_floor[:, 1], trans_pos_ceil[:, 0]),
            frame1_cl * weight_ne,
            accumulate=True,
        )
        warped_frame.index_put_(
            (batch_indices, trans_pos_ceil[:, 1], trans_pos_ceil[:, 0]),
            frame1_cl * weight_se,
            accumulate=True,
        )

        warped_weights.index_put_(
            (batch_indices, trans_pos_floor[:, 1], trans_pos_floor[:, 0]),
            weight_nw,
            accumulate=True,
        )
        warped_weights.index_put_(
            (batch_indices, trans_pos_ceil[:, 1], trans_pos_floor[:, 0]),
            weight_sw,
            accumulate=True,
        )
        warped_weights.index_put_(
            (batch_indices, trans_pos_floor[:, 1], trans_pos_ceil[:, 0]),
            weight_ne,
            accumulate=True,
        )
        warped_weights.index_put_(
            (batch_indices, trans_pos_ceil[:, 1], trans_pos_ceil[:, 0]),
            weight_se,
            accumulate=True,
        )

        warped_frame_cf = torch.moveaxis(warped_frame, [0, 1, 2, 3], [0, 2, 3, 1])
        warped_weights_cf = torch.moveaxis(warped_weights, [0, 1, 2, 3], [0, 2, 3, 1])
        cropped_warped_frame = warped_frame_cf[:, :, 1:-1, 1:-1]
        cropped_weights = warped_weights_cf[:, :, 1:-1, 1:-1]

        mask = cropped_weights > 0
        zero_value = -1 if is_image else 0
        zero_tensor = torch.tensor(zero_value, dtype=frame1.dtype, device=frame1.device)
        warped_frame2 = torch.where(
            mask, cropped_warped_frame / cropped_weights, zero_tensor
        )
        mask2 = mask.to(frame1)

        if is_image:
            assert warped_frame2.min() >= -1.1  # Allow for rounding errors
            assert warped_frame2.max() <= 1.1
            warped_frame2 = torch.clamp(warped_frame2, min=-1, max=1)
        return warped_frame2, mask2


    def clean_points_fast(self, warped_frame2, mask2, kernel=5, iters=1):
        """Dilate invalid mask regions on-device using max pooling.

        Inputs are warped_frame2 in [-1, 1] and mask2 in [0, 1]. Returns
        cleaned frames and a validity mask, matching clean_points.
        """
        pad = kernel // 2
        inv = 1.0 - mask2                             # 1 denotes a hole.
        for _ in range(iters):
            inv = F.max_pool2d(inv, kernel_size=kernel, stride=1, padding=pad)
        new_invalid = (inv >= 0.5).to(warped_frame2.dtype)   # (b,1,h,w)
        new_valid   = 1.0 - new_invalid

        # Fill holes with -1, the black value in [-1, 1].
        warped_frame2 = warped_frame2 * new_valid + (-1.0) * new_invalid
        return warped_frame2, new_valid


    def clean_points(self, warped_frame2, mask2):
        warped_frame2 = (warped_frame2 + 1.0) / 2.0
        mask = 1 - mask2
        mask[mask < 0.5] = 0
        mask[mask >= 0.5] = 1
        mask = mask.squeeze(0).repeat(3, 1, 1).permute(1, 2, 0) * 255.0
        mask = mask.cpu().numpy()
        kernel = numpy.ones((5, 5), numpy.uint8)
        mask_erosion = cv2.dilate(numpy.array(mask), kernel, iterations=1)
        mask_erosion = PIL.Image.fromarray(numpy.uint8(mask_erosion))
        mask_erosion_ = numpy.array(mask_erosion) / 255.0
        mask_erosion_[mask_erosion_ < 0.5] = 0
        mask_erosion_[mask_erosion_ >= 0.5] = 1
        mask_new = (
            torch.from_numpy(mask_erosion_)
            .permute(2, 0, 1)
            .unsqueeze(0)
            .to(self.device)
        )
        warped_frame2 = warped_frame2 * (1 - mask_new)
        return warped_frame2 * 2.0 - 1.0, 1 - mask_new[:, 0:1, :, :]

    @staticmethod
    def create_grid(b, h, w):
        x_1d = torch.arange(0, w)[None]
        y_1d = torch.arange(0, h)[:, None]
        x_2d = x_1d.repeat([h, 1])
        y_2d = y_1d.repeat([1, w])
        grid = torch.stack([x_2d, y_2d], dim=0)
        batch_grid = grid[None].repeat([b, 1, 1, 1])
        return batch_grid

    @staticmethod
    def read_image(path: Path) -> torch.Tensor:
        image = skimage.io.imread(path.as_posix())
        return image

    @staticmethod
    def read_depth(path: Path) -> torch.Tensor:
        if path.suffix == '.png':
            depth = skimage.io.imread(path.as_posix())
        elif path.suffix == '.npy':
            depth = numpy.load(path.as_posix())
        elif path.suffix == '.npz':
            with numpy.load(path.as_posix()) as depth_data:
                depth = depth_data['depth']
        else:
            raise RuntimeError(f'Unknown depth format: {path.suffix}')
        return depth

    @staticmethod
    def camera_intrinsic_transform(
        capture_width=1920, capture_height=1080, patch_start_point: tuple = (0, 0)
    ):
        start_y, start_x = patch_start_point
        camera_intrinsics = numpy.eye(4)
        camera_intrinsics[0, 0] = 2100
        camera_intrinsics[0, 2] = capture_width / 2.0 - start_x
        camera_intrinsics[1, 1] = 2100
        camera_intrinsics[1, 2] = capture_height / 2.0 - start_y
        return camera_intrinsics

    @staticmethod
    def get_device(device: str):
        """
        Returns torch device object
        :param device: cpu / gpu0 / gpu1 / npu0 / npu:0 / cuda:0
        :return:
        """
        if device == 'cpu':
            return torch.device('cpu')

        # Pass explicit 'npu:N' or 'cuda:N' device names directly to torch.
        if device.startswith('npu:') or device.startswith('cuda:'):
            return torch.device(device)

        # Accept 'npuN', or 'npu' without an index.
        if device.startswith('npu') and hasattr(torch, 'npu') and torch.npu.is_available():
            suffix = device[3:]
            npu_num = int(suffix) if suffix.isdigit() else 0
            return torch.device(f'npu:{npu_num}')

        # 'gpuN'
        if device.startswith('gpu') and torch.cuda.is_available():
            suffix = device[3:]
            gpu_num = int(suffix) if suffix.isdigit() else 0
            return torch.device(f'cuda:{gpu_num}')

        return torch.device('cpu')
