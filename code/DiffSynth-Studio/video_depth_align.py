# Modified for Demiurge; see THIRD_PARTY_NOTICES.md.
"""
Naive depth alignment using global least squares (a, b), then metric rescaling.
No Poisson blending, comparison outputs, or backups are produced.
When HARD_REPLACE_KNOWN is enabled, partial_raw replaces the fitted metric depth
at known pixels; other pixels retain the least-squares result.

Outputs:
  - poisson_aligned_metric.npz   metric depth, mask, and fitted coefficients
  - poisson_aligned_metric.mp4   metric grayscale visualization (clip(d,0,1)*255)
"""

import os
import numpy as np
import imageio.v2 as imageio


# Shared output root supplied by the workflow environment.
OUT_DIR     = os.environ["OUT_DIR"]
TC_EXP_NAME = os.environ.get("TC_EXP_NAME", "test")
TC_EXP_DIR  = os.path.join(OUT_DIR, TC_EXP_NAME)
os.makedirs(OUT_DIR, exist_ok=True)
print(f"[depth_align] OUT_DIR    = {OUT_DIR}")
print(f"[depth_align] TC_EXP_DIR = {TC_EXP_DIR}")


# DiffSynth depth output.
DIFF_PATH    = os.path.join(OUT_DIR,    "video_single_channel_expanded.npz")
# TrajectoryCrafter partial depth and mask.
PARTIAL_PATH = os.path.join(TC_EXP_DIR, "render_depth.npz")
MASK_PATH    = os.path.join(TC_EXP_DIR, "mask_B.mp4")

SAVE_NPZ     = os.path.join(OUT_DIR, "poisson_aligned_metric.npz")
SAVE_MP4     = os.path.join(OUT_DIR, "poisson_aligned_metric.mp4")
FPS          = 10

# Alignment settings.
PARTIAL_RENORM = True       # Repeat min-max normalization over valid pixels.
RIDGE          = 1e-4

# The refined mask selects LSQ samples and optional replacement pixels.
MASK_REFINE_BY_PARTIAL = True
PARTIAL_VALID_THRESH   = 1e-6   # Partial depth must exceed this value to be valid.
MASK_EROSION_PIXELS    = 0      # 0 disables erosion; 1-2 pixels are usually enough.

# Optionally replace fitted depth with partial_raw at known pixels.
HARD_REPLACE_KNOWN     = False


# ===================== utils =====================
def to_uint8_gray3(x01):
    """ x01: [...] in [0,1] -> [...,3] uint8 """
    g = (np.clip(x01, 0.0, 1.0) * 255.0).round().astype(np.uint8)
    return np.stack([g, g, g], axis=-1)


def resize_bilinear(img, out_h, out_w):
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


def resize_nearest(img, out_h, out_w):
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
def load_diffusion(path):
    """Load diffusion depth at any scale and min-max normalize to [0,1]."""
    data = np.load(path)
    key = "video" if "video" in data.files else data.files[0]
    arr = data[key].astype(np.float32)
    arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)
    print(f"[diffusion] raw shape={arr.shape}, range=({arr.min():.4f},{arr.max():.4f})")
    if arr.ndim == 4:
        c0, c1 = arr.shape[0], arr.shape[1]
        if c0 <= 4 and c1 > 4:   arr = arr[0]
        elif c1 <= 4 and c0 > 4: arr = arr[:, 0]
        else:                    arr = arr[0]
    elif arr.ndim != 3:
        raise RuntimeError(f"Unsupported diffusion shape: {arr.shape}")
    if arr.max() > 1.5:
        arr = arr / 255.0
    return np.clip(arr, 0.0, 1.0)


def load_partial(path, renorm=True, valid_thresh=1e-6):
    """
    Return (partial_norm, partial_raw, valid).
      partial_norm: [0,1], LSQ input on the same scale as diffusion depth.
      partial_raw : original metric depth for stage 2 and optional replacement.
      valid       : bool [T,H,W], partial > valid_thresh.
    """
    data = np.load(path)
    key = "depth" if "depth" in data.files else data.files[0]
    arr = data[key].astype(np.float32)
    print(f"[partial]   raw shape={arr.shape}, range=({arr.min():.4f},{arr.max():.4f})")
    valid = arr > valid_thresh
    raw = arr.copy()                       # Metric depth for stage 2 and optional replacement.

    out = arr.copy()
    if valid.any() and renorm:
        lo = out[valid].min(); hi = out[valid].max()
        out = (out - lo) / (hi - lo + 1e-8)
    out = np.where(valid, np.clip(out, 0.0, 1.0), 0.0)
    return out.astype(np.float32), raw.astype(np.float32), valid


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


# Global least squares.
def solve_global_ab(diff, partial, known, ridge=1e-4, trim=(0.02, 0.98)):
    """Fit partial ~= a * diff + b on known pixels with quantile trimming."""
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


def solve_unnormalize_ac(x_norm, y_metric, known):
    """Fit y_metric ~= a * x_norm + c on known pixels without trimming."""
    x = x_norm[known].astype(np.float64)
    y = y_metric[known].astype(np.float64)
    n = x.size
    if n < 2:
        raise RuntimeError(f"too few anchor pixels: {n}")
    sx  = x.sum();    sy  = y.sum()
    sxx = (x * x).sum(); sxy = (x * y).sum()
    denom = n * sxx - sx * sx
    if denom < 1e-12:
        a = 0.0
        c = float(y.mean())
    else:
        a = float((n * sxy - sx * sy) / denom)
        c = float((sy - a * sx) / n)
    return a, c, n


# ===================== main =====================
def main():
    diff = load_diffusion(DIFF_PATH)
    partial_norm, partial_raw, partial_valid = load_partial(
        PARTIAL_PATH, renorm=PARTIAL_RENORM, valid_thresh=PARTIAL_VALID_THRESH)
    mask = load_mask_imageio(MASK_PATH)

    # Match sequence lengths.
    T = min(diff.shape[0], partial_norm.shape[0], mask.shape[0])
    diff          = diff[:T]
    partial_norm  = partial_norm[:T]
    partial_raw   = partial_raw[:T]
    partial_valid = partial_valid[:T]
    mask          = mask[:T]
    print(f"[align T] T={T}")

    # Match the partial-depth resolution.
    H, W = partial_norm.shape[1:]
    if diff.shape[1:] != (H, W):
        diff = np.stack([resize_bilinear(diff[t], H, W) for t in range(T)], 0)
        print(f"[diffusion] resized to ({H},{W})")
    if mask.shape[1:] != (H, W):
        mask = np.stack([resize_nearest(mask[t], H, W) for t in range(T)], 0)
        print(f"[mask] resized to ({H},{W})")

    # Refine the LSQ sampling mask and optional replacement region.
    mask_bin = (mask > 0.5)
    print(f"[refine] mask_bin     ratio = {mask_bin.mean():.4f}")

    if MASK_REFINE_BY_PARTIAL:
        known = mask_bin & partial_valid
        print(f"[refine] mask & p>0   ratio = {known.mean():.4f}  "
              f"(removed masked pixels with invalid partial depth: {(mask_bin & ~partial_valid).mean():.4f})")
    else:
        known = mask_bin

    if MASK_EROSION_PIXELS > 0:
        before = known.mean()
        known = erode_binary(known, n_pixels=MASK_EROSION_PIXELS)
        print(f"[refine] erode {MASK_EROSION_PIXELS}px    ratio = {known.mean():.4f}  "
              f"(eroded fraction: {(before - known.mean()):.4f})")

    print(f"[final known overlap ratio] {known.mean():.4f}")

    # Stage 1: globally fit diff to the partial_norm scale.
    a1, b1, n_in = solve_global_ab(diff, partial_norm, known, ridge=RIDGE)
    print(f"[Stage1 global LSQ] a={a1:.4f}, b={b1:.4f}, n_inlier={n_in}")
    diff_aligned = np.clip(a1 * diff + b1, 0.0, 1.0)

    err_before = float(np.abs(diff         - partial_norm)[known].mean())
    err_lsq    = float(np.abs(diff_aligned - partial_norm)[known].mean())
    print(f"[L1 in known] raw={err_before:.4f}  after-LSQ={err_lsq:.4f}")

    # Stage 2: restore metric depth, partial_raw ~= a2 * diff_aligned + c2.
    a2, c2, n2 = solve_unnormalize_ac(diff_aligned, partial_raw, known)
    print(f"[Stage2 unnormalize] a={a2:.6f}, c={c2:.6f}, anchors={n2}")
    depth_metric = (a2 * diff_aligned + c2).astype(np.float32)

    # Optionally restore partial_raw at known pixels.
    if HARD_REPLACE_KNOWN and known.any():
        before_diff = float(np.abs(depth_metric - partial_raw)[known].mean())
        depth_metric = np.where(known, partial_raw, depth_metric).astype(np.float32)
        after_diff = float(np.abs(depth_metric - partial_raw)[known].max())
        print(f"[hard-replace] replaced known pixels with partial_raw  "
              f"(before mean|delta|={before_diff:.4e}, after max|delta|={after_diff:.4e})")
    else:
        print("[hard-replace] skipped (HARD_REPLACE_KNOWN=False or known is empty)")

    # ----- verify -----
    if known.any():
        err = np.abs(depth_metric - partial_raw)[known]
        print(f"[verify] |depth_metric - partial_raw| at known: "
              f"max={err.max():.3e}  mean={err.mean():.3e}")
    print(f"[range] depth_metric: ({depth_metric.min():.4f}, {depth_metric.max():.4f})")
    print(f"[range] partial_raw : ({partial_raw.min():.4f}, {partial_raw.max():.4f})")

    # Save aligned depth outputs.
    os.makedirs(OUT_DIR, exist_ok=True)

    np.savez_compressed(
        SAVE_NPZ,
        depth=depth_metric,                     # [T,H,W] float32, metric scale
        mask=known.astype(np.uint8),
        a=np.float32(a2),                       # Stage2 a
        c=np.float32(c2),                       # Stage2 c
        a_stage1=np.float32(a1),                # Stage1 a for reference
        b_stage1=np.float32(b1),                # Stage1 b for reference
    )
    print(f"[saved] {SAVE_NPZ}")

    # Visualize metric depth as clip(metric, 0, 1) * 255.
    imageio.mimsave(SAVE_MP4,
                    to_uint8_gray3(depth_metric),
                    fps=FPS, macro_block_size=1)
    print(f"[saved] {SAVE_MP4}")

    print("Done")


if __name__ == "__main__":
    main()
