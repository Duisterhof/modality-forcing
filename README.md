<div align="center">

<h1>Modality Forcing for Scalable<br>Spatial Generation</h1>

[![Project Page](https://img.shields.io/badge/Project_Page-Modality_Forcing-blue?logo=googlechrome&logoColor=white)](https://modality-forcing.github.io/)
[![Demo](https://img.shields.io/badge/Demo-Hugging_Face_Space-yellow?logo=huggingface)](https://huggingface.co/spaces/bartduis/modality_forcing)
[![Model](https://img.shields.io/badge/Model-modality__forcing-yellow?logo=huggingface)](https://huggingface.co/bartduis/modality_forcing)
[![arXiv](https://img.shields.io/badge/arXiv-2606.13676-b31b1b?logo=arxiv&logoColor=white)](https://arxiv.org/abs/2606.13676)

**[Bardienus Pieter Duisterhof](https://bart-ai.com)<sup>1,2</sup> · [Deva Ramanan](https://www.cs.cmu.edu/~deva/)<sup>1</sup> · [Jeffrey Ichnowski](https://ichnow.ski)<sup>1</sup> · [Justin Johnson](https://web.eecs.umich.edu/~justincj/)<sup>2</sup> · [Keunhong Park](https://keunhong.com)<sup>2</sup>**

**<sup>1</sup> Carnegie Mellon University &nbsp;&nbsp; <sup>2</sup> World Labs**

*Preprint, 2026*

<p>
  <img src="assets/cmu_logo.png" alt="Carnegie Mellon University" height="72">
  &nbsp;&nbsp;&nbsp;&nbsp;
  <img src="assets/world_labs_logo.jpg" alt="World Labs" height="72">
</p>

</div>

## Abstract

Text-to-image (T2I) models contain rich spatial priors.
Synthesizing photorealistic, cluttered scenes requires an understanding of geometry, including perspective and relative scale.
Prior works adapt T2I models to leverage this prior for depth prediction, but they require dense depth data and involve complex recipes.
We propose **Modality Forcing**, a simple, scalable post-training recipe for joint image-depth generation using a single DiT trained on sparse depth data.
Modality Forcing enables conditional and joint generation of image and depth in any permutation by assigning separate noise levels per modality.
Per-modality decoders let us train on sparse, real-world depth and achieve strong, generalizable depth prediction.
We further show that Modality Forcing inherits the scalability of T2I pre-training: by training a set of T2I models from scratch (300M to 3B parameters), we find that larger models trained on more image data produce more accurate depth.
Our strongest model is competitive with state-of-the-art monocular depth estimators and reduces AbsRel by **57%** relative to existing joint image-depth generative models.
These results provide strong evidence that image generation is a scalable pre-training objective for spatial perception.

## Overview

**Text → RGB + depth**, built on FLUX.2. Modality Forcing turns a
pretrained text-to-image diffusion transformer into a joint image–depth
generator with a simple post-training recipe: each modality is noised
independently during training, so at inference any permutation of
conditional and joint generation falls out of a single model.

The released model is a single diffusion transformer with three streams —
text, image, and depth — that supports three generation modes:

| Mode | Script | Input | Output |
|------|--------|-------|--------|
| **joint** | `scripts/joint.py` | text | RGB + depth + 3D point cloud |
| **i2d** | `scripts/i2d.py` | text + image | depth + 3D point cloud |
| **d2i** | `scripts/d2i.py` | text + depth | RGB |

## Table of Contents

- [Abstract](#abstract)
- [Overview](#overview)
- [Installation](#installation)
- [Model Weights](#model-weights)
- [Quick Start](#quick-start)
- [Usage](#usage)
  - [Joint — text → RGB + depth](#joint--text--rgb--depth)
  - [Image-to-depth — text + image → depth](#image-to-depth--text--image--depth)
  - [Depth-to-image — text + depth → RGB](#depth-to-image--text--depth--rgb)
- [Interactive Demo](#interactive-demo)
- [How It Works](#how-it-works)
- [License](#license)
- [Citation](#citation)

## Installation

Requires Python 3.10+ and a CUDA GPU with **at least 48 GB of memory**
(in bf16 the DiT is ~24 GB and the Qwen3-8B text encoder ~16 GB, plus
activations — an A100/H100-class card).

**Option A — [uv](https://docs.astral.sh/uv/) (recommended):**

```bash
git clone https://github.com/Duisterhof/modality-forcing.git
cd modality-forcing

# Pick the torch extra matching your driver (`nvidia-smi` → "CUDA Version")
uv sync --extra cu128
```

| Driver CUDA (`nvidia-smi`) | Command                 |
|----------------------------|-------------------------|
| 13.0+                      | `uv sync --extra cu130` |
| 12.8 – 12.9                | `uv sync --extra cu128` |
| 12.6 – 12.7                | `uv sync --extra cu126` |
| ≤ 12.5 / 11.x              | use Option B (cu121/cu118 indexes) |
| no GPU                     | `uv sync --extra cpu`   |

This creates `.venv/` from the committed `uv.lock`, so you get the exact
dependency set the release was tested with. Run scripts through `uv run`
(no activation needed) or activate the env:

```bash
uv run scripts/joint.py --prompt "a cozy sunlit kitchen with wooden cabinets"
# or: source .venv/bin/activate && python scripts/joint.py ...
```

> [!NOTE]
> A bare `uv sync` (no `--extra`) installs everything **except** PyTorch — if
> you hit `ModuleNotFoundError: torch`, re-run with an extra from the table.

**Option B — conda / pip:**

```bash
git clone https://github.com/Duisterhof/modality-forcing.git
cd modality-forcing

conda create -n mofo python=3.12 -y && conda activate mofo
# (a plain venv works the same: python -m venv .venv && source .venv/bin/activate)

# Install a PyTorch build matching your driver's CUDA FIRST — pick the index
# for your CUDA: cu130, cu128, cu126, cu121, cu118. A cu12x wheel runs on any
# >= that 12.x driver, so it need not match exactly.
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

In both cases, verify the GPU is visible — this must print `True`:

```bash
python -c "import torch; print(torch.cuda.is_available())"   # uv: prefix with `uv run`
```

<details>
<summary>Troubleshooting</summary>
<br>

- **`torch.cuda.is_available()` is `False` with a GPU present** — the installed
  torch build targets a newer CUDA than your driver supports. Reinstall from
  the index matching `nvidia-smi`'s "CUDA Version": with uv, re-run
  `uv sync --extra <cuXXX>` from the table; with pip,
  `pip install torch --index-url https://download.pytorch.org/whl/cuXXX`.
- **`ModuleNotFoundError: torch` after `uv sync`** — you skipped the `--extra`;
  PyTorch only installs via one of the `cpu`/`cu126`/`cu128`/`cu130` extras.
- **uv says the extras are incompatible** — the torch extras conflict by
  design; pick exactly one.
- **Driver older than CUDA 12.6** — the uv extras don't cover the legacy
  cu121/cu118 indexes; use Option B with the matching `--index-url`.
- **macOS** — the `cu*` extras are Linux-only (PyTorch publishes no CUDA
  builds for macOS); use `uv sync --extra cpu` (Apple silicon only).
</details>

## Model Weights

The model weights and the FLUX.2 autoencoder are downloaded automatically from
the Hugging Face Hub on first run. By default the scripts pull
[`bartduis/modality_forcing`](https://huggingface.co/bartduis/modality_forcing)
and the Qwen3-8B text encoder [`Qwen/Qwen3-8B`](https://huggingface.co/Qwen/Qwen3-8B).
Point `--model` at a different repo id or a local checkpoint directory
(containing `config.json` and `model.safetensors`) to override.

## Quick Start

```bash
python scripts/joint.py --prompt "a cozy sunlit kitchen with wooden cabinets"
```

This generates an RGB image, a depth map, and a colored 3D point cloud
from a single text prompt, and writes them to `./outputs/`. (uv users can
run any command in this README without activating the env: `uv run
scripts/joint.py …`.)

## Usage

All three scripts share these options: `--prompt`, `--model`, `--text-encoder`,
`--num-steps` (default 50), `--seed`, `--device`, `--resolution` (default 512;
must match the checkpoint's training resolution), `--output-dir` (default
`./outputs`), and `--compile`. Inference runs in bfloat16 (the depth head in
fp32).

`--compile` runs the DiT under `torch.compile(mode="reduce-overhead")` (CUDA
graphs) for a ~1.4× faster sampling loop. The first run spends a few minutes
compiling (about a minute once Triton's kernel cache is warm); kernels are
cached at `~/.cache/modality-forcing/torchinductor` (override with
`TORCHINDUCTOR_CACHE_DIR`), so subsequent runs warm-start in seconds. Worth it
for repeated generations; a single one-off run is faster without it.

Each run writes a timestamped subdirectory containing `rgb.png`,
`depth_raw.npy` (raw depth, relative scale — unit-mean normalized),
`depth_magma.png` (disparity visualization, near = bright), and
`metadata.json`.

### Joint — text → RGB + depth

```bash
python scripts/joint.py --prompt "a cozy sunlit kitchen with wooden cabinets"
```

Every run also exports a colored point cloud as `cloud.glb` (for 3D viewers)
and `cloud.ply` (for point-cloud tools).

- `--cfg-scale` (default `4.0`) — classifier-free guidance for the RGB stream.
- `--log2-alpha` (default `5.0`) — tilts the RGB/depth denoising trajectory.
  `> 0` is rgb-first (RGB resolves before depth, giving cleaner depth), `0` is
  the diagonal joint schedule, `< 0` is depth-first.
- `--refine-depth` — run a second image→depth pass on the generated RGB for
  sharper, more RGB-consistent geometry (matches the online demo).
- `--sor` — also apply statistical outlier removal to the point cloud
  (off by default; can over-trim fine structures).
- `--edge-rtol` (default `0.04`) — depth-edge mask for the point cloud: drops
  pixels at depth jumps larger than this fraction (removes occlusion-boundary
  floaters). Lower = more aggressive; `0` disables it.
- `--fov-deg` (default `65`) — vertical field of view used to back-project the
  cloud.

### Image-to-depth — text + image → depth

```bash
python scripts/i2d.py --prompt "" --image photo.jpg
```

The RGB is held fixed and resized to 512×512 (non-square inputs are
stretched; the hosted demo letterboxes instead), and guidance is disabled
(CFG = 1.0). Leave `--prompt` empty unless you want to nudge the
depth with a caption. It also writes the point cloud (GLB + PLY); `--edge-rtol`
and `--fov-deg` work the same as in joint mode.

### Depth-to-image — text + depth → RGB

```bash
python scripts/d2i.py \
    --prompt "a cozy sunlit kitchen with wooden cabinets" \
    --depth depth.npy
```

`--depth` accepts a `.npy` float array or a 16-bit single-channel PNG/TIFF —
e.g. the `depth_raw.npy` written by a previous joint run. The
depth is normalized scale-invariantly before conditioning, so it can be metric
or relative. `--cfg-scale` (default `4.0`) guides the RGB stream.

## Interactive Demo

Try the model without installing anything in the
[online demo](https://huggingface.co/spaces/bartduis/modality_forcing).
`app.py` is the same Gradio app; to run it locally:

```bash
python app.py        # or: uv run app.py
```

For a long-lived local demo, `COMPILE=1 python app.py` torch.compiles the DiT
(one-time cost on the first generation, then every later generation benefits).

On HF Spaces, ZeroGPU cannot use torch.compile (it forks a fresh process per
GPU call); the app instead compiles the DiT ahead of time with the `spaces`
AoT path, on by default. The compiled package is cached on disk (`/data` if
the Space has persistent storage) and reloads in milliseconds on later boots;
set `ZEROGPU_AOTI=0` to opt out, or `ZEROGPU_AOTI=1` to exercise the same
path locally.

## How It Works

- **512×512** generation; depth is a peer stream patchified into the model's
  depth tokens (16×16 patches), decoded back to a depth map.
- The text encoder draws multi-layer hidden states from Qwen3-8B (mirroring the
  FLUX.2 [klein] recipe).
- Depth normalization (`unit_mean` + mip-NeRF 360 `contract`) is invertible up
  to the per-sample scale, which is why `i2d`/`d2i` depth is relative.

## License

- **Code** is licensed under **Apache-2.0** (see [`LICENSE`](LICENSE)). Files
  derived from the [FLUX.2 reference implementation](https://github.com/black-forest-labs/flux2)
  (`flux_rgbd/_flux2/`, `flux_rgbd/dit.py`) are Apache-2.0, Copyright Black
  Forest Labs, with modifications by World Labs.
- **Model weights** are licensed under **CC BY-NC 4.0** (see
  [`LICENSE-WEIGHTS`](LICENSE-WEIGHTS)) — non-commercial use only.

## Citation

If you find Modality Forcing useful, please consider giving the repo a star and citing our paper:

```bibtex
@article{duisterhof2026mofo,
  title   = {Modality Forcing for Scalable Spatial Generation},
  author  = {Duisterhof, Bardienus Pieter and Ramanan, Deva and Ichnowski, Jeffrey and Johnson, Justin and Park, Keunhong},
  journal = {arXiv preprint arXiv:2606.13676},
  year    = {2026}
}
```
