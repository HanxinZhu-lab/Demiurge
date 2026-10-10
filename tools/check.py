#!/usr/bin/env python3
"""Static preflight only: no project imports, model loading, or GPU queries."""
import argparse
import ast
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
errors = []


def require_file(path):
    path = Path(path)
    if not path.is_file() or path.stat().st_size == 0:
        errors.append(f"Missing or empty file: {path}")


def number(name, minimum, maximum=None):
    value = os.environ.get(name, "")
    if not re.fullmatch(r"0|[1-9][0-9]*", value):
        errors.append(f"{name} must be a nonnegative decimal integer (no leading zeros)")
        return minimum
    value = int(value)
    if value < minimum or (maximum is not None and value > maximum):
        errors.append(f"{name} must be in {minimum}..{maximum or 'unlimited'}")
    return value


def source_files():
    roots = [ROOT / "code", ROOT / "tools", ROOT / "environments"]
    candidates = list(ROOT.iterdir())
    for directory in roots:
        candidates.extend(directory.rglob("*"))
    for path in sorted(candidates):
        relative = path.relative_to(ROOT)
        if not path.is_file() or any(part.startswith(".") for part in relative.parts[:-1]):
            continue
        if "__pycache__" in relative.parts or relative == Path("tools/assemble.py"):
            continue
        if relative.parts[0] == "environments" and path.suffix == ".json":
            continue
        legal = path.name.lower().startswith(("license", "notice", "copying"))
        if legal or path.suffix in {".py", ".sh", ".md", ".txt", ".toml", ".json"} or path.name in {".gitignore", ".gitattributes"}:
            yield path, legal


def check_source(release=False):
    python_count = shell_count = 0
    han = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\U00020000-\U0002fa1f]")
    private_path = re.compile(r"/(?:mnt|dockerdata|home|root)/[A-Za-z0-9_.-]+")
    secret = re.compile(r"(?:AKIA[0-9A-Z]{16}|gh[pousr]_[A-Za-z0-9]{36,}|-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----)")
    for path, legal in source_files():
        relative = path.relative_to(ROOT)
        try:
            text = path.read_text(encoding="utf-8")
            if path.suffix == ".py":
                ast.parse(text, filename=str(relative))
                python_count += 1
            elif path.suffix == ".sh":
                result = subprocess.run(["bash", "-n", str(path)], capture_output=True, text=True)
                if result.returncode:
                    errors.append(f"Bash syntax: {relative}: {result.stderr.strip()}")
                shell_count += 1
            if release and not legal:
                for lineno, line in enumerate(text.splitlines(), 1):
                    for label, pattern in [("Non-English source text", han), ("Machine-specific path", private_path), ("Potential secret", secret)]:
                        if pattern.search(line):
                            errors.append(f"{label}: {relative}:{lineno}")
        except (OSError, UnicodeError, SyntaxError) as exc:
            errors.append(f"Source check failed: {relative}: {exc}")
    print(f"Source syntax: {python_count} Python files, {shell_count} Bash files; no project imports or inference")
    if release:
        print("Release scan covers source and documentation; upstream legal texts are preserved.")
        print("This is a heuristic working-tree scan, not a Git-history or license clearance.")


def require_component(directory):
    require_file(directory / "config.json")
    weights = sorted(directory.glob("*.safetensors"))
    if not weights:
        errors.append(f"Missing safetensors weights: {directory}")
    for path in weights:
        require_file(path)
    for index in directory.glob("*.index.json"):
        try:
            data = json.loads(index.read_text(encoding="utf-8"))
            mapping = data["weight_map"]
            if not isinstance(mapping, dict) or not mapping:
                raise ValueError("weight_map must be a nonempty object")
            for shard in set(mapping.values()):
                if not isinstance(shard, str) or Path(shard).name != shard:
                    raise ValueError("shard names must be plain filenames")
                require_file(directory / shard)
        except (OSError, UnicodeError, ValueError, KeyError, TypeError) as exc:
            errors.append(f"Invalid weight index: {index}: {exc}")


def check_models(skip):
    ds = Path(os.environ.get("DIFFSYNTH_MODEL_BASE_PATH", ""))
    for relative in [
        "PAI/Wan2.1-Fun-1.3B-InP/diffusion_pytorch_model.safetensors",
        "DiffSynth-Studio/Wan-Series-Converted-Safetensors/models_t5_umt5-xxl-enc-bf16.safetensors",
        "DiffSynth-Studio/Wan-Series-Converted-Safetensors/Wan2.1_VAE.safetensors",
        "DiffSynth-Studio/Wan-Series-Converted-Safetensors/models_clip_open-clip-xlm-roberta-large-vit-huge-14.safetensors",
        "Wan-AI/Wan2.1-T2V-1.3B/google/umt5-xxl/tokenizer_config.json",
        "Wan-AI/Wan2.1-T2V-1.3B/google/umt5-xxl/spiece.model",
    ]:
        require_file(ds / relative)
    if skip:
        return
    tc = Path(os.environ.get("TC_CKPT", ""))
    for relative in [
        "TrajectoryCrafter", "DepthCrafter", "blip2-opt-2.7b",
        "CogVideoX-Fun-V1.1-5b-InP/text_encoder", "CogVideoX-Fun-V1.1-5b-InP/vae",
        "stable-video-diffusion-img2vid/vae", "stable-video-diffusion-img2vid/image_encoder",
    ]:
        require_component(tc / relative)
    for relative in [
        "blip2-opt-2.7b/tokenizer_config.json", "blip2-opt-2.7b/tokenizer.json",
        "blip2-opt-2.7b/preprocessor_config.json",
        "CogVideoX-Fun-V1.1-5b-InP/model_index.json",
        "CogVideoX-Fun-V1.1-5b-InP/tokenizer/tokenizer_config.json",
        "CogVideoX-Fun-V1.1-5b-InP/tokenizer/spiece.model",
        "CogVideoX-Fun-V1.1-5b-InP/scheduler/scheduler_config.json",
        "stable-video-diffusion-img2vid/model_index.json",
        "stable-video-diffusion-img2vid/scheduler/scheduler_config.json",
        "stable-video-diffusion-img2vid/feature_extractor/preprocessor_config.json",
        "stable-video-diffusion-img2vid/vae/diffusion_pytorch_model.fp16.safetensors",
        "stable-video-diffusion-img2vid/image_encoder/model.fp16.safetensors",
    ]:
        require_file(tc / relative)
    print("Model files checked at configured paths; tensor contents and hashes were not verified.")


def finish():
    if errors:
        for message in errors:
            print(f"[ERROR] {message}", file=sys.stderr)
        return 1
    print("Static checks passed. GPU compatibility and end-to-end inference were not tested by this check.")
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-only", action="store_true", help="Check source syntax and publication hygiene without models or Conda")
    args = parser.parse_args()
    errors.clear()
    check_source(release=args.source_only)
    if args.source_only:
        return finish()
    start = number("START_IDX", 1, 40)
    end = number("N", 1, 40)
    refine = number("REFINE_ITERS", 0)
    skip = number("SKIP_TC", 0, 1)
    if start > end:
        errors.append("START_IDX must not exceed N (N is the last round, not the count)")
    for name in ["OUT_DIR", "LOG_DIR", "TC_VIDEO_PATH", "DIFFSYNTH_MODEL_BASE_PATH", "TC_CKPT", "DS_RGB_GEN_CKPT", "DS_RGB_REFINE_CKPT", "DS_DEPTH_CKPT"]:
        if not Path(os.environ.get(name, "")).is_absolute():
            errors.append(f"{name} must be an absolute path")
    output = Path(os.environ.get("OUT_DIR", ROOT / "results")).resolve()
    experiment = os.environ.get("TC_EXP_NAME", "test")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", experiment):
        errors.append("TC_EXP_NAME must be a plain directory name (letters/digits/._-)")
    protected = [ROOT / name for name in ["code", "models", "assets", "tools", "environments", "manifests"]]
    for name in ["CONDA_BASE", "TC_CKPT", "DIFFSYNTH_MODEL_BASE_PATH", "TC_ENV", "DS_ENV"]:
        value = os.environ.get(name, "")
        if value and Path(value).is_absolute():
            protected.append(Path(value))
    protected = [path.resolve() for path in protected]
    for name in ["OUT_DIR", "LOG_DIR"]:
        target = Path(os.environ.get(name, ROOT)).resolve()
        if target == ROOT or target == Path("/") or any(target == p or p in target.parents or target in p.parents for p in protected):
            errors.append(f"{name} would overlap a protected directory: {target}")
    require_file(os.environ.get("TC_VIDEO_PATH", ROOT / "assets/p7.mp4"))
    base = Path(os.environ.get("CONDA_BASE", ""))
    require_file(base / "etc/profile.d/conda.sh")
    for name in (["DS_ENV"] if skip else ["TC_ENV", "DS_ENV"]):
        env = os.environ.get(name, "")
        prefix = Path(env) if Path(env).is_absolute() else base / "envs" / env
        require_file(prefix / "bin/python")
        print(f"{name}: {prefix} (existence only; not executed)")
    for tool in ["bash", "tee", "sed", "flock"]:
        if shutil.which(tool) is None:
            errors.append(f"Required system command not found: {tool}")
    if (ROOT / "manifests/.assemble.lock").exists():
        errors.append("Local asset assembly is still in progress")
    check_models(skip)
    for name in ["DS_RGB_GEN_CKPT", "DS_DEPTH_CKPT"] + (["DS_RGB_REFINE_CKPT"] if refine else []):
        require_file(os.environ.get(name, ""))
    if start > 1:
        for idx in range(1, start):
            for name in [f"poisson_aligned_metric-{idx}.npz", f"poisson_blend_naive-{idx}.npz", f"mask-{idx}.mp4"]:
                require_file(output / name)
    if skip:
        names = ["render_rgb.npz", "render_depth.npz", "mask.mp4", "mask_B.mp4", "mask_raw_B.mp4"]
        if refine:
            names += ["render_rgb_D.npz", "mask_D.mp4"]
        for name in names:
            require_file(output / experiment / name)
    return finish()


if __name__ == "__main__":
    sys.exit(main())
