# Modified for Demiurge; see THIRD_PARTY_NOTICES.md.
"""
Naive RGB alignment using global least squares independently per channel.
When HARD_REPLACE_KNOWN is enabled, partial replaces the fit at known pixels.
No Poisson blending, comparison outputs, or backups are produced.

Outputs:
  - poisson_blend_naive.mp4    aligned RGB, with optional known-pixel replacement
  - poisson_blend_naive.npz    same data, key='video', (T,3,H,W), float16, [0,1]

Baseline mode:
  - BYPASS_ALIGN=True skips least squares and sets naive_blend = diff.
    Known-pixel replacement is also skipped in this mode.
"""

import os
import numpy as np
import imageio.v2 as imageio


# Shared output root supplied by the workflow environment.
OUT_DIR     = os.environ["OUT_DIR"]
TC_EXP_NAME = os.environ.get("TC_EXP_NAME", "test")
TC_EXP_DIR  = os.path.join(OUT_DIR, TC_EXP_NAME)
os.makedirs(OUT_DIR, exist_ok=True)
print(f"[rgb_align] OUT_DIR    = {OUT_DIR}")
print(f"[rgb_align] TC_EXP_DIR = {TC_EXP_DIR}")


# DiffSynth RGB output.
DIFF_PATH    = os.path.join(OUT_DIR,    "video_Wan2.1-Fun-1.3B-InP-rgb.npz")
# TrajectoryCrafter partial RGB and mask.
PARTIAL_PATH = os.path.join(TC_EXP_DIR, "render_rgb.npz")
MASK_PATH    = os.path.join(TC_EXP_DIR, "mask.mp4")
FPS          = 10

# Alignment settings.
RIDGE       = 1e-4

# True passes through diff without LSQ or known-pixel replacement.
BYPASS_ALIGN = False

# Optionally replace fitted RGB with partial at known pixels.
HARD_REPLACE_KNOWN = False

# The refined mask selects LSQ samples and optional replacement pixels.
MASK_THRESHOLD = 10        # Grayscale mask values above this threshold are known.
MASK_EROSION_PIXELS = 0    # 0 disables erosion; larger values remove more boundary pixels.

# Optionally exclude masked pixels whose partial RGB sum is near zero.
ENABLE_PARTIAL_VALID_BY_BLACK = False
PARTIAL_VALID_BLACK_EPS = 1e-3


# ===================== utils =====================
def to_uint8_rgb(x):
    """ x: [...,3] in [0,1] float -> uint8 """
    return (np.clip(x, 0.0, 1.0) * 255.0).round().astype(np.uint8)


def resize_bilinear_2d(img, out_h, out_w):
    h, w = img.shape
    if h == out_h and w == out_w:
        return img.astype(np.float32, copy=False)
    ys = (np.arange(out_h, dtype=np.float32) + 0.5) * (h / out_h) - 0.5
    xs = (np.arange(out_w, dtype=np.float32) + 0.5) * (w / out_w) - 0.5
    ys = np.clip(ys, 0, h - 1); xs = np.clip(xs, 0, w - 1)
    y0 = np.floor(ys).astype(np.int32); x0 = np.floor(xs).astype(np.int32)
    y1 = np.clip(y0 + 1, 0, h - 1);     x1 = np.clip(x0 + 1, 0, w - 1)
    wy = (ys - y0)[:, None]; wx = (xs - x0)[None, :]
    Ia = img[y0[:, None], x0[None, :]]; Ib = img[y0[:, None], x1[None, :]]
    Ic = img[y1[:, None], x0[None, :]]; Id = img[y1[:, None], x1[None, :]]
    top = Ia * (1 - wx) + Ib * wx; bot = Ic * (1 - wx) + Id * wx
    return (top * (1 - wy) + bot * wy).astype(np.float32)


def resize_bilinear_rgb(img, out_h, out_w):
    h, w, c = img.shape
    if h == out_h and w == out_w:
        return img.astype(np.float32, copy=False)
    out = np.zeros((out_h, out_w, c), dtype=np.float32)
    for k in range(c):
        out[..., k] = resize_bilinear_2d(img[..., k], out_h, out_w)
    return out


def resize_nearest_2d(img, out_h, out_w):
    h, w = img.shape
    if h == out_h and w == out_w:
        return img.astype(np.float32, copy=False)
    ys = np.minimum((np.arange(out_h) * h / out_h).astype(np.int32), h - 1)
    xs = np.minimum((np.arange(out_w) * w / out_w).astype(np.int32), w - 1)
    return img[ys[:, None], xs[None, :]].astype(np.float32)


def erode_binary(mask_bool, n_pixels=1):
    if n_pixels <= 0:
        return mask_bool
    m = mask_bool.copy()
    for _ in range(n_pixels):
        eroded = m.copy()
        eroded[..., :, 1:]  &= m[..., :, :-1]
        eroded[..., :, :-1] &= m[..., :, 1:]
        eroded[..., 1:, :]  &= m[..., :-1, :]
        eroded[..., :-1, :] &= m[..., 1:, :]
        m = eroded
    return m


# ===================== loaders =====================
def load_rgb_mp4(path):
    reader = imageio.get_reader(path)
    frames = []
    for f in reader:
        a = np.asarray(f)
        if a.ndim == 2:
            a = np.stack([a, a, a], axis=-1)
        elif a.shape[-1] == 4:
            a = a[..., :3]
        frames.append(a)
    reader.close()
    arr = np.stack(frames, axis=0).astype(np.float32) / 255.0
    arr = np.clip(arr, 0.0, 1.0)
    print(f"[load rgb]  {os.path.basename(path)}  shape={arr.shape}  "
          f"range=({arr.min():.4f},{arr.max():.4f})")
    return arr


def load_rgb_npz(path, candidate_keys=("video", "rgb", "frames", "data", "arr_0")):
    """
    Load an RGB NPZ sequence in (T,3,H,W) or (T,H,W,3), uint8 or float.
    Return float32 data in (T,H,W,3), clipped to [0,1].
    """
    data = np.load(path)
    files = list(data.files)

    key = None
    for k in candidate_keys:
        if k in files:
            key = k
            break

    if key is None:
        array_like_keys = [
            k for k in files
            if isinstance(data[k], np.ndarray) and data[k].ndim == 4
        ]
        if len(array_like_keys) == 1:
            key = array_like_keys[0]
        else:
            raise KeyError(
                f"{path}: no RGB array key found; candidates={candidate_keys}, "
                f"available keys={files}"
            )

    arr = data[key]
    print(f"[load npz]  {os.path.basename(path)}  using key='{key}'  "
          f"raw shape={arr.shape}  raw dtype={arr.dtype}")

    if arr.ndim != 4:
        raise ValueError(f"{path} key='{key}': expected 4D, got shape={arr.shape}")

    if arr.shape[1] == 3 and arr.shape[-1] != 3:
        arr = np.transpose(arr, (0, 2, 3, 1))
    elif arr.shape[-1] == 3:
        pass
    else:
        raise ValueError(
            f"{path} key='{key}': unrecognized channel dimension, shape={arr.shape}"
        )

    if arr.dtype == np.uint8:
        arr = arr.astype(np.float32) / 255.0
    else:
        arr = arr.astype(np.float32)
        if arr.max() > 1.5:
            print(f"[load npz]  detected float in 0~255 range, /255")
            arr = arr / 255.0

    arr = np.clip(arr, 0.0, 1.0)
    print(f"[load npz]  {os.path.basename(path)}  -> shape={arr.shape}  "
          f"range=({arr.min():.4f},{arr.max():.4f})")
    return arr


def load_rgb_auto(path):
    ext = os.path.splitext(path)[1].lower()
    if ext == ".npz":
        return load_rgb_npz(path)
    return load_rgb_mp4(path)


def load_mask_imageio(path, threshold=10):
    reader = imageio.get_reader(path)
    frames = []
    for frame in reader:
        f = np.asarray(frame)
        if f.ndim == 3:
            gray = (0.299 * f[..., 0] + 0.587 * f[..., 1] + 0.114 * f[..., 2])
        else:
            gray = f
        frames.append((gray > threshold).astype(np.float32))
    reader.close()
    arr = np.stack(frames, axis=0)
    print(f"[mask]      shape={arr.shape}, known_ratio={arr.mean():.4f}")
    return arr


# Global least squares for one channel.
def solve_global_ab(diff, partial, known, ridge=1e-4, trim=(0.02, 0.98)):
    """Fit partial ~= a*diff + b on trimmed known pixels without modifying inputs."""
    pv = partial[known]
    if pv.size < 100:
        return 1.0, 0.0, int(pv.size)
    lo = np.quantile(pv, trim[0]); hi = np.quantile(pv, trim[1])
    inlier = known & (partial >= lo) & (partial <= hi)
    d = diff[inlier].astype(np.float64)
    p = partial[inlier].astype(np.float64)
    n = d.size
    if n < 100:
        return 1.0, 0.0, n
    A = np.array([[(d * d).sum() + ridge, d.sum()],
                  [d.sum(),                n + ridge]])
    rhs = np.array([(d * p).sum(), p.sum()])
    a, b = np.linalg.solve(A, rhs)
    return float(a), float(b), n


# ===================== main =====================
def main():
    diff = load_rgb_auto(DIFF_PATH)            # [T,H,W,3]

    # Baseline passthrough mode: output equals input.
    if BYPASS_ALIGN:
        print("[BYPASS_ALIGN=True] skipping LSQ, naive_blend = diff (no hard replacement)")

        naive_blend = diff.astype(np.float32, copy=False)
        print(f"[bypass] naive_blend shape={naive_blend.shape}  "
              f"range=({naive_blend.min():.4f},{naive_blend.max():.4f})")

        os.makedirs(OUT_DIR, exist_ok=True)

        out_path_mp4 = os.path.join(OUT_DIR, "poisson_blend_naive.mp4")
        imageio.mimsave(out_path_mp4, to_uint8_rgb(naive_blend),
                        fps=FPS, macro_block_size=1)

        out_path_npz = os.path.join(OUT_DIR, "poisson_blend_naive.npz")
        blend_for_npz = np.clip(naive_blend, 0.0, 1.0)
        blend_for_npz = np.transpose(blend_for_npz, (0, 3, 1, 2))
        blend_for_npz = blend_for_npz.astype(np.float16)
        np.savez_compressed(out_path_npz, video=blend_for_npz)
        print(f"[save npz]  {os.path.basename(out_path_npz)}  shape={blend_for_npz.shape}  "
              f"dtype={blend_for_npz.dtype}  "
              f"range=({blend_for_npz.min()},{blend_for_npz.max()})")

        print("Done (bypass)")
        print("  -", out_path_mp4)
        print("  -", out_path_npz)
        return

    # Normal mode: LSQ and optional known-pixel replacement.
    partial = load_rgb_auto(PARTIAL_PATH)      # [T,H,W,3]
    mask    = load_mask_imageio(MASK_PATH, threshold=MASK_THRESHOLD)  # [T,H,W]

    # Match sequence lengths.
    T = min(diff.shape[0], partial.shape[0], mask.shape[0])
    diff = diff[:T]; partial = partial[:T]; mask = mask[:T]
    print(f"[align T] T={T}")

    # Match the partial-RGB resolution.
    H, W = partial.shape[1:3]
    if diff.shape[1:3] != (H, W):
        diff = np.stack([resize_bilinear_rgb(diff[t], H, W) for t in range(T)], 0)
        print(f"[diffusion] resized to ({H},{W})")
    if mask.shape[1:] != (H, W):
        mask = np.stack([resize_nearest_2d(mask[t], H, W) for t in range(T)], 0)
        print(f"[mask] resized to ({H},{W})")

    # Refine the LSQ sampling mask and optional replacement region.
    mask_bin = (mask > 0.5)
    print(f"[refine] mask_bin     ratio = {mask_bin.mean():.4f}")

    if ENABLE_PARTIAL_VALID_BY_BLACK:
        partial_valid = partial.sum(axis=-1) > PARTIAL_VALID_BLACK_EPS
        known = mask_bin & partial_valid
        print(f"[refine] mask & p>0   ratio = {known.mean():.4f}")
    else:
        known = mask_bin

    if MASK_EROSION_PIXELS > 0:
        before = known.mean()
        known = erode_binary(known, n_pixels=MASK_EROSION_PIXELS)
        print(f"[refine] erode {MASK_EROSION_PIXELS}px    ratio = {known.mean():.4f}  "
              f"(eroded fraction: {(before - known.mean()):.4f})")

    print(f"[final known overlap ratio] {known.mean():.4f}")

    # Step 1: independent global LSQ for each RGB channel.
    diff_aligned = np.zeros_like(diff)
    for c, name in enumerate("RGB"):
        a, b, n_in = solve_global_ab(diff[..., c], partial[..., c],
                                     known, ridge=RIDGE)
        diff_aligned[..., c] = np.clip(a * diff[..., c] + b, 0.0, 1.0)
        err_before = float(np.abs(diff[..., c]         - partial[..., c])[known].mean())
        err_lsq    = float(np.abs(diff_aligned[..., c] - partial[..., c])[known].mean())
        print(f"[global LSQ {name}] a={a:.4f} b={b:.4f} n_inlier={n_in}  "
              f"L1 raw={err_before:.4f} -> LSQ={err_lsq:.4f}")

    # Step 2: optionally restore partial RGB at known pixels.
    naive_blend = diff_aligned.astype(np.float32, copy=False)
    if HARD_REPLACE_KNOWN and known.any():
        before_diff = float(np.abs(naive_blend - partial)[known].mean())
        # Expand (T,H,W) to (T,H,W,1) for broadcasting across RGB.
        known3 = known[..., None]
        naive_blend = np.where(known3, partial, naive_blend).astype(np.float32)
        after_diff_max = float(np.abs(naive_blend - partial)[known].max())
        print(f"[hard-replace] replaced known pixels with partial  "
              f"(before mean|delta|={before_diff:.4e}, after max|delta|={after_diff_max:.4e})")
    elif not HARD_REPLACE_KNOWN:
        print("[hard-replace] skipped (HARD_REPLACE_KNOWN=False)")
    else:
        print("[hard-replace] skipped (known is empty)")

    # ----- verify -----
    if known.any():
        err_known = float(np.abs(naive_blend - partial)[known].mean())
        print(f"[verify] L1 in known = {err_known:.6f}  "
              f"({'expected near zero after hard replacement' if HARD_REPLACE_KNOWN else 'determined by LSQ residuals'})")

    # Save aligned RGB outputs.
    os.makedirs(OUT_DIR, exist_ok=True)

    # 1) mp4
    out_path_mp4 = os.path.join(OUT_DIR, "poisson_blend_naive.mp4")
    imageio.mimsave(out_path_mp4, to_uint8_rgb(naive_blend),
                    fps=FPS, macro_block_size=1)

    # 2) npz (key='video', (T,3,H,W), float16, [0,1])
    out_path_npz = os.path.join(OUT_DIR, "poisson_blend_naive.npz")
    blend_for_npz = np.clip(naive_blend, 0.0, 1.0)                 # (T, H, W, 3)
    blend_for_npz = np.transpose(blend_for_npz, (0, 3, 1, 2))      # (T, 3, H, W)
    blend_for_npz = blend_for_npz.astype(np.float16)
    np.savez_compressed(out_path_npz, video=blend_for_npz)
    print(f"[save npz]  {os.path.basename(out_path_npz)}  shape={blend_for_npz.shape}  "
          f"dtype={blend_for_npz.dtype}  "
          f"range=({blend_for_npz.min()},{blend_for_npz.max()})")

    print("Done")
    print("  -", out_path_mp4)
    print("  -", out_path_npz)


if __name__ == "__main__":
    main()
