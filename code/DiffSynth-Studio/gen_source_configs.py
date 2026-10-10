# Modified for Demiurge; see THIRD_PARTY_NOTICES.md.
"""
Generate SOURCE_CONFIGS and WARP_TARGET_POSE for the TC warp iteration IDX.

Input: environment variable IDX (iteration number, 1..N).
Output: stdout line 1 = SOURCE_CONFIGS (single-line JSON).
        stdout line 2 = WARP_TARGET_POSE (five space-separated numbers).

The trajectory is defined by make_pose_for_idx.
"""
import os
import json


def make_pose_for_idx(idx: int):
    if idx <= 4:
        return [0.0, -10 * idx , 0, 0, 0.0] 
    if idx > 4 and idx <= 8:
        return [0.0, 10 * (idx - 4), 0, 0, 0.0] 
    if idx > 8 and idx <= 12:
        return [10 * (idx - 8), 0 , 0, 0, 0.0] 
    if idx > 12 and idx <= 16:
        return [-10 * (idx - 12), 0 , 0, 0, 0.0]
    if idx > 16 and idx <= 20:
        return [0, 0 , 0, 0.3 * (idx - 16), 0.0]
    if idx > 20 and idx <= 24:
        return [0, 0 , 0, -0.3 * (idx - 20), 0.0]   
    if idx > 24 and idx <= 28:
        return [0,0,0,0, 0.3 * (idx - 24)]
    if idx > 28 and idx <= 32:
        return [0,0,0,0, -0.3 * (idx - 32)]
    if idx > 32 and idx <= 36:
        return [0, 0 , -0.3 * (idx - 32), 0, 0.0] 
    if idx > 36 and idx <= 40:
        return [0, 0 , 0.3 * (idx - 36), 0, 0.0]


# Earlier iterations supply sources with their original warp poses in world space.
def make_sources_for_idx(idx: int):
    sources = []
    for i in range(1, idx):                      # 1..idx-1
        npz   = f"poisson_aligned_metric-{i}.npz"
        video = f"poisson_blend_naive-{i}.npz"
        mask  = f"mask-{i}.mp4"
        pose  = make_pose_for_idx(i)
        sources.append([npz, video, mask, pose])
    return sources


def main():
    idx = int(os.environ.get("IDX", "1"))
    sources = make_sources_for_idx(idx)
    pose    = make_pose_for_idx(idx)

    print(json.dumps(sources, separators=(",", ":")))
    print(" ".join(str(x) for x in pose))


if __name__ == "__main__":
    main()
