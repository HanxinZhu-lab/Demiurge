# -*- coding: utf-8 -*-
# Modified for Demiurge; see THIRD_PARTY_NOTICES.md.
"""Select the default device and provide NPU-compatible device helpers.

Import after torch and before using CUDA/device APIs. When available,
transfer_to_npu redirects CUDA calls to NPU; the helpers provide fallbacks.
"""

import os
import sys
import warnings

import torch

# Suppress dependency deprecation warnings.
warnings.filterwarnings("ignore", category=FutureWarning)

_HAS_NPU = False
_HAS_CUDA = False

try:
    import torch_npu  # noqa: F401
    try:
        # Redirect CUDA calls to NPU.
        from torch_npu.contrib import transfer_to_npu  # noqa: F401
    except Exception as _e:
        print(f"[npu_patch] Failed to import transfer_to_npu: {_e}", file=sys.stderr)
    _HAS_NPU = torch.npu.is_available() if hasattr(torch, "npu") else False
except ImportError:
    _HAS_NPU = False

_HAS_CUDA = torch.cuda.is_available()

# Default device.
if _HAS_NPU:
    DEFAULT_DEVICE = "npu:0"
elif _HAS_CUDA:
    DEFAULT_DEVICE = "cuda:0"
else:
    DEFAULT_DEVICE = "cpu"

# Shared device helpers.
def empty_cache():
    try:
        if _HAS_NPU:
            torch.npu.empty_cache()
        elif _HAS_CUDA:
            torch.cuda.empty_cache()
    except Exception:
        pass


def synchronize():
    try:
        if _HAS_NPU:
            torch.npu.synchronize()
        elif _HAS_CUDA:
            torch.cuda.synchronize()
    except Exception:
        pass


def device_count():
    if _HAS_NPU:
        return torch.npu.device_count()
    if _HAS_CUDA:
        return torch.cuda.device_count()
    return 0


def print_summary():
    print("=" * 60)
    print("[npu_patch] NPU available :", _HAS_NPU)
    print("[npu_patch] CUDA available:", _HAS_CUDA)
    print("[npu_patch] Default device:", DEFAULT_DEVICE)
    print("[npu_patch] Device count  :", device_count())
    print("=" * 60)


# Print the device summary on import when enabled.
if os.environ.get("NPU_PATCH_VERBOSE", "1") == "1":
    print_summary()
