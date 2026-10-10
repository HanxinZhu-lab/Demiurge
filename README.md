# Demiurge

### FORGING NAVIGABLE 4D WORLDS VIA AUTOREGRESSIVE VIDEO RERENDERING

**One input video. Multiple camera trajectories. A shared, evolving 4D world.**

[Hugging Face Weights](https://huggingface.co/ingzzzz/Demiurge) · [Quick Start](#quick-start) · [Model Preparation](#model-preparation) · [Outputs](#outputs) · [Advanced Usage](#advanced-usage) · [License](#license)

## Overview

Video diffusion models can synthesize photorealistic novel-view videos from a single input video. However, rendering each camera trajectory independently makes it difficult to maintain a coherent world across viewpoints. **Demiurge** addresses this problem with an autoregressive video rendering pipeline built around a dynamic 4D point-cloud memory cache: previously observed and generated content guides subsequent trajectories, promoting consistency in both appearance and geometry.

Demiurge combines four components:

1. **Appearance video diffusion:** renders photorealistic novel-view videos along target camera trajectories, conditioned on the shared 4D cache.
2. **Cache-consistent video depth completion:** aligns known-region depths with cached geometry while completing missing regions, reducing scale ambiguity and discontinuities during cache updates.
3. **Memory-aware dual-stream warping:** supplies visual guidance and occlusion cues for plausible novel-view synthesis.
4. **Geometry-aware video refinement:** uses geometric guidance to improve the perceptual quality of generated videos.

The method targets coherent spatiotemporal scene construction across dense multi-view trajectories, large-parallax rendering, and geometry-faithful outputs suitable for downstream 3D reconstruction. This repository provides the **inference workflow**: warping, RGB/depth generation, optional refinement, alignment, and autoregressive reuse of earlier results. It does not currently provide an end-to-end training recipe or a 3D Gaussian Splatting optimization pipeline.

> **Code and weights:** this repository contains inference code; the three fine-tuned checkpoints are publicly available at [ingzzzz/Demiurge on Hugging Face](https://huggingface.co/ingzzzz/Demiurge). No login or access request is required to download these checkpoints. Upstream base models are still required and may have separate access conditions. See [Demiurge checkpoints](#demiurge-checkpoints) for download and configuration instructions.

## Quick Start

### 1. Requirements

- Linux, Bash, Conda, and an NVIDIA GPU with a driver compatible with CUDA 12.1 PyTorch wheels.
- Python 3.10 and **two separate environments** for warping and diffusion.
- Approximately **65 GiB of disk space for the selected model assets**, plus outputs and caches. This is not a VRAM requirement; minimum VRAM has not been established.
- A decodable input video with **at least 81 frames**. With the default stride of 1, the first 81 consecutive frames are used; shorter videos are not padded.

The loader resizes input frames to 1024 × 576. The main generation path produces 81-frame videos at 832 × 480. Use a 16:9 source to avoid aspect-ratio distortion. Input frame rate is not automatically normalized.

Clone or download this repository, enter its root directory, and keep the following variables in your shell:

```bash
cd /path/to/Demiurge
export REPO_ROOT="$(pwd)"
export CONDA_BASE="$(conda info --base)"
source "$CONDA_BASE/etc/profile.d/conda.sh"
```

### 2. Prepare the environments

If you already have compatible environments, skip installation and set `TC_ENV` and `DS_ENV` to their names or absolute prefixes.

The following **reference installation recipe** uses the core versions recorded in the original local environments. It has not yet been validated as a complete clean-machine installation. The files in `environments/*.freeze.txt` provide additional version references; they are not portable lockfiles and include unrelated packages.

| Component | Warping: `trajcrafter_new` | Diffusion: `diffsynth_cuda` |
| --- | --- | --- |
| Python | 3.10 | 3.10 |
| PyTorch | 2.4.0, CUDA 12.1 | 2.5.1, CUDA 12.1 |
| torchvision | 0.19.0 | 0.20.1 |
| transformers | 4.47.0 | 5.15.1 |
| diffusers | 0.31.0 | Not used by the main DiffSynth pipeline |
| xformers | 0.0.27.post2 | Not required by this recipe |

**Warping environment:**

```bash
conda create -y -n trajcrafter_new python=3.10
conda activate trajcrafter_new
python -m pip install torch==2.4.0 torchvision==0.19.0 \
  --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r "$REPO_ROOT/code/TrajectoryCrafter/requirements.txt" \
  "torchvision==0.19.0" "diffusers==0.31.0" \
  "transformers==4.47.0" "xformers==0.0.27.post2" scipy tqdm
python -m pip check
```

**Diffusion environment:**

```bash
conda create -y -n diffsynth_cuda python=3.10
conda activate diffsynth_cuda
python -m pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
  --index-url https://download.pytorch.org/whl/cu121
python -m pip install -e "$REPO_ROOT/code/DiffSynth-Studio" \
  "torch==2.5.1" "torchvision==0.20.1" "torchaudio==2.5.1" \
  "transformers==5.15.1" "peft==0.20.0"
python -m pip check
```

Install the **bundled** DiffSynth package, not a stock upstream release: custom model layers and RGB/depth conditioning are required. `torchaudio` is included because the package imports it even for this video workflow. TrajectoryCrafter and DepthCrafter are imported directly from the bundled source; do not separately install the nested DepthCrafter package metadata, which targets a different dependency stack.

If dependency resolution fails, retain the error and check the corresponding environment snapshot rather than upgrading both environments together. The runner activates each environment automatically and does not install or upgrade packages during inference.

### 3. Prepare models and input

Complete [Model Preparation](#model-preparation), then set your input and a prompt describing its scene. The local sample `assets/p7.mp4` is not distributed with the source release.

```bash
export TC_ENV=trajcrafter_new
export DS_ENV=diffsynth_cuda
export TC_VIDEO_PATH=/absolute/path/to/input.mp4
export DEMIURGE_RGB_PROMPT="A description of the scene in your input video."
export OUT_DIR="$REPO_ROOT/results/first-run"
```

**Set the prompt for your own video.** The fallback RGB prompt describes a blue bird beside a red apple; changing the input does not automatically change it. TrajectoryCrafter's captioning step does not replace this DiffSynth prompt.

### 4. Check the setup and render the first trajectory

```bash
bash "$REPO_ROOT/run.sh" --check
GPU_ID=0 N=1 REFINE_ITERS=0 bash "$REPO_ROOT/run.sh"
```

`--check` validates configuration, source syntax, model-file presence, and referenced weight shards. It does **not** load models, decode the input, query the GPU, or verify package compatibility.

Each round runs:

**Dual-stream warping → RGB generation → depth completion → optional refinement/depth regeneration → depth/RGB alignment → numbered outputs.**

For the first result, open:

```text
results/first-run/poisson_blend_naive-1.mp4
```

Logs are written under `logs/`. A successful run ends with `ALL ROUNDS DONE`. To check source syntax and publication hygiene without any models, input video, or Conda environments, use `bash run.sh --source-only`.

## Model Preparation

All inference weights are read from local files. **Automatic downloads are disabled during a run.** Prepare weights in advance and retain the applicable model licenses. Access-controlled repositories may require accepting their terms and authenticating with the download provider.

### Base models

Run downloads in the diffusion environment, which provides the `modelscope` and `hf` command-line clients. These commands select the components used by this workflow; they are not a substitute for the custom checkpoints below.

```bash
conda activate diffsynth_cuda
export DS_MODELS="$REPO_ROOT/models/diffsynth"
export TC_MODELS="$REPO_ROOT/models/trajectorycrafter"

modelscope download --model PAI/Wan2.1-Fun-1.3B-InP \
  --include diffusion_pytorch_model.safetensors \
  --local_dir "$DS_MODELS/PAI/Wan2.1-Fun-1.3B-InP"

modelscope download --model DiffSynth-Studio/Wan-Series-Converted-Safetensors \
  --include models_t5_umt5-xxl-enc-bf16.safetensors Wan2.1_VAE.safetensors \
  models_clip_open-clip-xlm-roberta-large-vit-huge-14.safetensors \
  --local_dir "$DS_MODELS/DiffSynth-Studio/Wan-Series-Converted-Safetensors"

hf download Wan-AI/Wan2.1-T2V-1.3B --include "google/umt5-xxl/*" \
  --local-dir "$DS_MODELS/Wan-AI/Wan2.1-T2V-1.3B"

hf download TrajectoryCrafter/TrajectoryCrafter \
  --local-dir "$TC_MODELS/TrajectoryCrafter"

hf download Salesforce/blip2-opt-2.7b --exclude "*.bin" \
  --local-dir "$TC_MODELS/blip2-opt-2.7b"

hf download alibaba-pai/CogVideoX-Fun-V1.1-5b-InP \
  --include model_index.json configuration.json LICENSE \
  "vae/*" "text_encoder/*" "tokenizer/*" "scheduler/*" \
  --local-dir "$TC_MODELS/CogVideoX-Fun-V1.1-5b-InP"

hf download stabilityai/stable-video-diffusion-img2vid \
  --include model_index.json LICENSE.md "scheduler/*" "feature_extractor/*" \
  vae/config.json vae/diffusion_pytorch_model.fp16.safetensors \
  image_encoder/config.json image_encoder/model.fp16.safetensors \
  --local-dir "$TC_MODELS/stable-video-diffusion-img2vid"
```

Obtain the pretrained **DepthCrafter UNet** through the [official DepthCrafter repository](https://github.com/Tencent/DepthCrafter) and place its `config.json` and `diffusion_pytorch_model.safetensors` in `$TC_MODELS/DepthCrafter/`, retaining its license and notices. Do not substitute the original SVD UNet.

The expected top-level layout is:

```text
models/
├── diffsynth/
│   ├── PAI/Wan2.1-Fun-1.3B-InP/
│   ├── DiffSynth-Studio/Wan-Series-Converted-Safetensors/
│   ├── Wan-AI/Wan2.1-T2V-1.3B/google/umt5-xxl/
│   └── Demiurge/
│       ├── rgb_generation.safetensors
│       ├── depth_completion.safetensors
│       └── rgb_refinement.safetensors
└── trajectorycrafter/
    ├── TrajectoryCrafter/
    ├── DepthCrafter/
    ├── blip2-opt-2.7b/
    ├── CogVideoX-Fun-V1.1-5b-InP/
    └── stable-video-diffusion-img2vid/
```

Keep tokenizer/processor files, model configurations, schedulers, safetensors shards, and their indexes together. TrajectoryCrafter replaces the CogVideoX transformer; DepthCrafter replaces the SVD UNet. The other listed components are still initialized during warping and cannot be omitted.

Download availability and provider access policies can change. The commands above have not been revalidated by downloading the full models for this documentation update.

### Demiurge checkpoints

Download the three fine-tuned checkpoints from **[ingzzzz/Demiurge](https://huggingface.co/ingzzzz/Demiurge/tree/main)**. The files total approximately 11.7 GB, in addition to the upstream base models. These are custom DiT checkpoints for the bundled pipeline, not complete standalone Diffusers pipelines.

| Checkpoint | Hugging Face filename | Required for |
| --- | --- | --- |
| Appearance generation | `rgb_generation.safetensors` | Every run |
| Depth completion | `depth_completion.safetensors` | Every run |
| Geometry-aware refinement | `rgb_refinement.safetensors` | `REFINE_ITERS > 0` |

**Access:** the model repository is public. You can download all three checkpoints without logging in or requesting access. Optional authentication with `hf auth login` may help with Hub rate limits, but is not a prerequisite. Never put a token in the README, a shell command literal, or version-controlled configuration.

Public availability does not change the applicable model licenses or grant additional usage rights. Upstream base models may still require accepting their own terms or authenticating separately.

From the project root, download directly to the directory expected by `config.sh`:

```bash
export REPO_ROOT="$(pwd)"
export DS_MODELS="${DIFFSYNTH_MODEL_BASE_PATH:-$REPO_ROOT/models/diffsynth}"
hf download ingzzzz/Demiurge \
  rgb_generation.safetensors depth_completion.safetensors rgb_refinement.safetensors \
  --local-dir "$DS_MODELS/Demiurge"
```

No renaming or per-checkpoint configuration is needed for a fresh checkout. Set the input and environments from Quick Start, then run `bash run.sh --check` before inference. This command checks local files only; it does not verify the remote files or their hashes.

For reproducible experiments, add `--revision FULL_COMMIT_HASH` to the download command, replacing `FULL_COMMIT_HASH` with the desired full commit ID from the model repository's history, and record it with your results.

If your checkpoints are stored elsewhere, override the paths instead of copying them:

```bash
export DS_RGB_GEN_CKPT=/absolute/checkpoints/rgb_generation.safetensors
export DS_DEPTH_CKPT=/absolute/checkpoints/depth_completion.safetensors
export DS_RGB_REFINE_CKPT=/absolute/checkpoints/rgb_refinement.safetensors
```

Checkpoint selection is performed independently for each stage:

1. A nonempty explicit `DS_*_CKPT` environment variable takes precedence.
2. Otherwise, `config.sh` selects the corresponding file in `DIFFSYNTH_MODEL_BASE_PATH/Demiurge/`.
3. If that release path is absent, an existing legacy checkpoint under `train/` is used. If neither exists, preflight reports the missing release path.

Existing installations using `step-27000`, `step-10000`, and `step-24000` therefore continue to work without moving or duplicating large files. When both layouts exist, the release path wins unless explicitly overridden. Unset any previously exported `DS_RGB_GEN_CKPT`, `DS_DEPTH_CKPT`, or `DS_RGB_REFINE_CKPT` values to return to automatic selection.

`DIFFSYNTH_MODEL_BASE_PATH` and `TC_CKPT` override the two base-model roots. All custom filesystem paths must be absolute. Local assembly manifests and the original author's workspace layout are not required.

## Outputs

Each trajectory round `i` writes these numbered results to `OUT_DIR`:

| File | Contents |
| --- | --- |
| `poisson_blend_naive-i.mp4` | Final aligned RGB visualization; start here to inspect the result. |
| `poisson_blend_naive-i.npz` | RGB array used by subsequent rounds: key `video`, float16, `[T, 3, H, W]`, values in `[0, 1]`. |
| `poisson_aligned_metric-i.npz` | Aligned depth: key `depth`, float32, `[T, H, W]`; also contains a known-region `mask` and alignment coefficients. |
| `poisson_aligned_metric-i.mp4` | Visualization of the aligned depth. |
| `mask-i.mp4` | Raw B-stream warp-validity mask: white = valid, black = hole; not semantic segmentation. |
| `video_Wan2.1-Fun-1.3B-InP-i.mp4` | RGB generation/refinement output before final alignment. |
| `depth_gray_video-i.mp4` | Generated-depth visualization before final alignment. |

Here `T=81`, `H=480`, and `W=832`. RGB/depth generation MP4s use 15 FPS; aligned visualizations and collected masks use 10 FPS. The different playback durations do not indicate different frame counts. MP4s are lossy; use NPZ arrays for downstream processing.

Despite the inherited filenames, the alignment scripts perform global least-squares alignment, not a Poisson solve. Saved depth follows the warping representation `1 / (z + 1)` and is **not calibrated metric depth in meters**. Subsequent rounds invert this representation before warping.

Intermediate files without a round suffix and the warp directory `OUT_DIR/TC_EXP_NAME/` are reused. Keep all earlier numbered RGB NPZ, depth NPZ, and mask MP4 files if you intend to resume or extend a sequence. Use a fresh output directory for independent experiments: rerunning an existing index replaces its numbered outputs.

## Advanced Usage

The examples below reuse the exported input, prompt, environment, and model settings from Quick Start.

### Render multiple trajectories

```bash
OUT_DIR="$REPO_ROOT/results/multi-view" GPU_ID=0 N=8 \
  bash "$REPO_ROOT/run.sh"
```

This renders rounds 1–8 autoregressively, using earlier results as sources for later rounds. `N` is the **inclusive final trajectory index**, not a denoising-step count or video length. The default trajectory schedule supports indices 1–40.

### Enable geometry-aware refinement

```bash
OUT_DIR="$REPO_ROOT/results/refined" N=1 REFINE_ITERS=2 \
  bash "$REPO_ROOT/run.sh"
```

Each extra pass refines RGB and regenerates depth. This requires the refinement checkpoint. The default `REFINE_ITERS=0` still performs RGB/depth generation and alignment, but skips these extra passes.

### Resume or extend a previous sequence

```bash
OUT_DIR="$REPO_ROOT/results/multi-view" START_IDX=9 N=12 \
  bash "$REPO_ROOT/run.sh"
```

Rounds 1–8 must already exist in that directory. Resume continues from saved outputs; it does not restore random-number-generator state or partially completed model execution.

### Customize camera motion

Edit `make_pose_for_idx()` in `code/DiffSynth-Studio/gen_source_configs.py`. It returns `[dtheta, dphi, dr, dx, dy]` for each round. Angles are in degrees; radial offsets and translations use the internal scene scale, not calibrated physical distances.

The generator also assigns each previous source its original pose. Keep these poses unchanged when resuming existing results, or begin a new output directory. The default runner uses this generator, not a trajectory TXT file. Supporting more than 40 rounds also requires updating the index bounds in `tools/check.py`.

### Configuration reference

Environment variables override the defaults in `config.sh`.

| Variable | Default | Purpose |
| --- | --- | --- |
| `TC_VIDEO_PATH` | `assets/p7.mp4` under the project root | Input video; supply your own in a public checkout. |
| `DEMIURGE_RGB_PROMPT` | English bird-and-apple description | Positive prompt shared by RGB generation and refinement. |
| `DEMIURGE_NEGATIVE_PROMPT` | English quality/artifact exclusions | Negative prompt shared by all three generation stages; an empty value is accepted. |
| `GPU_ID` | `0` | Physical CUDA device exposed to each stage. |
| `N` / `START_IDX` | `1` / `1` | Inclusive final/starting trajectory indices. |
| `REFINE_ITERS` | `0` | Additional refinement/depth passes per round. |
| `OUT_DIR` / `LOG_DIR` | Project-root `results/` / `logs/` | Results and run logs. |
| `TC_EXP_NAME` | `test` | Warping subdirectory inside `OUT_DIR`. |
| `CONDA_BASE` | Auto-detected | Conda installation root. |
| `TC_ENV` / `DS_ENV` | `trajcrafter_new` / `diffsynth_cuda` | Environment names or absolute prefixes. |
| `DIFFSYNTH_MODEL_BASE_PATH` | Project-root `models/diffsynth/` | DiffSynth base-model root; fine-tuned release weights are stored in its `Demiurge/` subdirectory. |
| `TC_CKPT` | Project-root `models/trajectorycrafter/` | TrajectoryCrafter base-model root. |
| `DS_RGB_GEN_CKPT` | `DIFFSYNTH_MODEL_BASE_PATH/Demiurge/rgb_generation.safetensors` | Appearance checkpoint; falls back to the existing legacy path when the release path is absent. |
| `DS_DEPTH_CKPT` | `DIFFSYNTH_MODEL_BASE_PATH/Demiurge/depth_completion.safetensors` | Depth checkpoint; same legacy fallback behavior. |
| `DS_RGB_REFINE_CKPT` | `DIFFSYNTH_MODEL_BASE_PATH/Demiurge/rgb_refinement.safetensors` | Refinement checkpoint; same legacy fallback behavior. |

`--skip-tc` / `SKIP_TC=1` is intended for reusing existing warp outputs. Required files include `render_rgb.npz`, `render_depth.npz`, `mask.mp4`, `mask_B.mp4`, and `mask_raw_B.mp4`; refinement additionally needs `render_rgb_D.npz` and `mask_D.mp4`. Skipping warping over multiple rounds reuses the same warp inputs and is not equivalent to normal autoregressive rendering.

## Implementation Guide

| Component | Entry point |
| --- | --- |
| Stage orchestration and environment switching | `run.sh`, `config.sh` |
| Target/source camera poses | `code/DiffSynth-Studio/gen_source_configs.py` |
| Multi-source dual-stream warping | `code/TrajectoryCrafter/inference_depth_warping_2.py`, `code/TrajectoryCrafter/demo.py` |
| Appearance, depth, and refinement entrypoints | `code/DiffSynth-Studio/examples/wanvideo/model_training/validate_full/` |
| Customized video diffusion pipeline | `code/DiffSynth-Studio/diffsynth/pipelines/wan_video.py` |
| RGB/depth alignment | `code/DiffSynth-Studio/video_rgb_align.py`, `code/DiffSynth-Studio/video_depth_align.py` |
| Numbered result collection | `code/DiffSynth-Studio/copy_file.sh` |
| Static preflight and source checks | `tools/check.py` |

The generation scripts reside in an inherited `model_training/validate_full/` directory, but are inference entrypoints. The runner passes the required environment variables; invoking these scripts directly without that setup is not supported by this guide.

## Troubleshooting and Reproducibility

- **Hugging Face 401/403 or repository not found:** verify the repository ID, filename, and revision. The three checkpoints in `ingzzzz/Demiurge` are public; check for an invalid saved token or `HF_TOKEN` if authentication fails. Upstream access-controlled models may still require accepting their terms and using your own authorized account. Never share the maintainer's upload token.
- **Missing or empty file:** check the exact path printed by `--check`, including model shards, tokenizer files, and custom checkpoints. Release checkpoints belong in `models/diffsynth/Demiurge/` by default, not the project root. Downloads do not occur during inference.
- **Environment not found:** set `CONDA_BASE`, `TC_ENV`, and `DS_ENV`. A preflight pass checks that Python exists, not that all imports are compatible.
- **Index errors while reading video:** provide at least 81 decodable frames with the default stride. Frame padding is not implemented.
- **CUDA out of memory:** reduce `N` for an initial test and avoid concurrent GPU workloads. Fewer rounds reduce accumulated sources, but do not reduce one round's fixed frame count or resolution; no minimum-VRAM guarantee is provided.
- **Another run is using OUT_DIR:** use a different output directory or wait for the current run to finish. Do not remove a live run's lock to bypass it.
- **Different results on repeated runs:** RGB generation/refinement select random seeds. The English prompt defaults also differ from the original Chinese defaults. Exact numerical reproduction is not guaranteed.

The pre-cleanup pipeline was reported working on the maintainer's GPU. Source syntax, static model-file checks, and the isolated source-only check have passed; a fresh installation and post-cleanup end-to-end GPU run have not yet been independently validated. CUDA compatibility workarounds are retained. The fine-tuned checkpoints are publicly available at the repository linked above. Licensing documentation, exact upstream revisions, and a fully tested installation lockfile remain outstanding release items.

## Acknowledgements

This implementation builds on [DiffSynth-Studio](https://github.com/modelscope/DiffSynth-Studio), [TrajectoryCrafter](https://github.com/TrajectoryCrafter/TrajectoryCrafter), [DepthCrafter](https://github.com/Tencent/DepthCrafter), and their underlying Wan, CogVideoX, BLIP2, and Stable Video Diffusion components. The bundled packages include local modifications and should not be treated as pristine upstream releases.

## License

This is a research code release with **mixed third-party terms**, not a blanket MIT/Apache-licensed project. The included TrajectoryCrafter and DepthCrafter licenses restrict use to academic, research, and education purposes and prohibit commercial or production use. Weights and media have separate terms. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) and the original license files before use or redistribution.

Model weights, input media, outputs, caches, and private environment/provenance records are excluded by `.gitignore`. Original copyright notices and bilingual legal texts are retained. A license for independently authored Demiurge code and permission to redistribute its custom checkpoints must be established before final publication.
