# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 World Labs.
"""HuggingFace ZeroGPU Space for flux_rgbd.

Wraps the same generate-and-render path used by the local demo with
``@spaces.GPU`` so HF ZeroGPU can attach a GPU per call. The model
loads at module-level (CPU), then moves to CUDA inside the first
decorated call.
"""

import os
import sys
import tempfile
import time
import uuid
from pathlib import Path

# Persistent torch.compile kernel cache. Must be set before torch/transformers
# are imported -- inductor caches the first cache-dir lookup.
os.environ.setdefault(
    "TORCHINDUCTOR_CACHE_DIR",
    str(Path("~/.cache/modality-forcing/torchinductor").expanduser()),
)

# The vendored flux_rgbd package lives next to this file.
sys.path.insert(0, str(Path(__file__).resolve().parent))

import gradio as gr
import numpy as np
import spaces

try:
    import torch
except ModuleNotFoundError as e:  # a bare `uv sync` installs everything but torch
    raise ModuleNotFoundError(
        "torch is not installed — run `uv sync --extra <cpu|cu126|cu128|cu130>` "
        "matching your driver (see README → Installation). The cu* extras are "
        "Linux-only; on macOS use `--extra cpu`."
    ) from e

from flux_rgbd import FluxRGBDRunner
from flux_rgbd.pointcloud import depth_edge_mask, statistical_outlier_mask

# BF16 checkpoint, downloaded from the Hub on first use. Override with the
# WEIGHTS_REPO env var (repo id or local path).
WEIGHTS_REPO = os.environ.get("WEIGHTS_REPO", "bartduis/modality_forcing")

DEFAULT_PROMPT = (
    "A warm, inviting kitchen with a rustic-modern feel, where soft morning "
    "light filters through a small window above the sink."
)

# Lazy-loaded runner. On ZeroGPU the model is loaded inside the first
# @spaces.GPU call so the import path costs nothing.
_runner: FluxRGBDRunner | None = None


def _ensure_runner() -> FluxRGBDRunner:
    global _runner
    if _runner is None:
        # Use BF16 Qwen3-8B instead of Qwen3-8B-FP8 to avoid the
        # finegrained-fp8 / deep-gemm kernel dependency, which currently
        # hits a metadata.json parse bug on HF's kernels-community.
        text_encoder = os.environ.get("TEXT_ENCODER_REPO", "Qwen/Qwen3-8B")
        # Generation resolution must match the checkpoint's training resolution
        # (512 for the default model, 1024 for the 1024 checkpoint). Set
        # IMG_RESOLUTION=1024 alongside WEIGHTS_REPO when using the 1024 ckpt.
        res = int(os.environ.get("IMG_RESOLUTION", "512"))
        # Never compile on a Space: ZeroGPU runs each @spaces.GPU call in a
        # fresh process, which would recompile on every generation.
        want_compile = os.environ.get("COMPILE") == "1"
        compile_model = want_compile and not os.environ.get("SPACE_ID")
        if want_compile and not compile_model:
            print(
                "[boot] COMPILE=1 ignored: torch.compile is unsupported "
                "on ZeroGPU Spaces.",
                flush=True,
            )
        print(
            f"[boot] loading {WEIGHTS_REPO} @ {res}px (text encoder: {text_encoder})"
            f"{' [torch.compile]' if compile_model else ''}…",
            flush=True,
        )
        _runner = FluxRGBDRunner.from_pretrained(
            WEIGHTS_REPO,
            device="cuda",
            dtype=torch.bfloat16,
            head_dtype=torch.float32,
            text_encoder=text_encoder,
            img_hw=(res, res),
            compile_model=compile_model,
        )
        print("[boot] runner ready.", flush=True)
    return _runner


# --- helpers (kept inline so the Space repo only needs app.py and flux_rgbd) ---


def _letterbox(img: np.ndarray, target: int = 512):
    """Resize so long side = target, then zero-pad to (target, target).

    Returns the canvas and the (top, left, h, w) box of the resized content.
    """
    import cv2

    h_in, w_in = img.shape[:2]
    if h_in >= w_in:
        h_out, w_out = target, max(1, round(w_in * target / h_in))
    else:
        w_out, h_out = target, max(1, round(h_in * target / w_in))
    resized = cv2.resize(img, (w_out, h_out), interpolation=cv2.INTER_AREA)
    canvas = np.zeros(
        (target, target, img.shape[2] if img.ndim == 3 else 1), dtype=img.dtype
    )
    if img.ndim == 2:
        canvas = canvas[..., 0]
    top = (target - h_out) // 2
    left = (target - w_out) // 2
    canvas[top : top + h_out, left : left + w_out] = resized
    return canvas, (top, left, h_out, w_out)


def _depth_to_pointcloud(
    rgb_u8, depth, *, fov_deg=65.0, max_points=1_200_000, edge_rtol=0.04, sor=False
):
    """Back-project depth into a median-centered cloud; returns (points, colors)."""
    h, w = depth.shape
    fx = w / (2.0 * np.tan(np.deg2rad(fov_deg) / 2.0))
    cx, cy = w * 0.5, h * 0.5
    # Keep every valid pixel -- no percentile clip. Clipping to e.g. [1, 99]
    # carves a hole in the closest surface (the geometry users care most
    # about) and drops the far background; the i2d depth is clean enough
    # that it isn't needed.
    valid = (depth > 0) & np.isfinite(depth)
    # Depth-edge mask: drop occlusion-boundary "veil" pixels (MoGe-style).
    if edge_rtol and edge_rtol > 0:
        valid &= ~depth_edge_mask(depth, rtol=float(edge_rtol))
    v_idx, u_idx = np.where(valid)
    z = depth[v_idx, u_idx]
    x = (u_idx + 0.5 - cx) * z / fx
    y = (v_idx + 0.5 - cy) * z / fx
    # glTF / Three.js: +Y up, camera looks down -Z. Flip image-y
    # (which points down) and depth (which points into the scene).
    pts = np.stack([x, -y, -z], axis=-1).astype(np.float32)
    cols = rgb_u8[v_idx, u_idx]
    if sor:
        # Statistical outlier rejection: drops isolated floaters, but can
        # over-trim fine structures -- opt-in (the edge mask above is the
        # default cleanup).
        inliers = statistical_outlier_mask(pts)
        pts, cols = pts[inliers], cols[inliers]
    if pts.shape[0] > max_points:
        idx = np.random.default_rng(0).choice(pts.shape[0], max_points, replace=False)
        pts, cols = pts[idx], cols[idx]
    if pts.shape[0]:
        pts -= np.median(pts, axis=0, keepdims=True)
    return pts, cols


def _save_glb(path, points, colors):
    """Colored point cloud -> binary glTF, the format gr.Model3D handles cleanly."""
    import trimesh

    cloud = trimesh.PointCloud(vertices=points, colors=colors)
    scene = trimesh.Scene()
    scene.add_geometry(cloud)
    scene.export(str(path))


def _depth_to_magma(depth: np.ndarray) -> np.ndarray:
    """Depth -> magma-colormapped disparity image (uint8 RGB).

    Visualizes 1/depth (so near = bright) robustly normalized to the 5-95th
    percentile.
    """
    from matplotlib import cm

    valid = (depth > 0) & np.isfinite(depth)
    disparity = np.zeros_like(depth, dtype=np.float32)
    if valid.any():
        disparity[valid] = 1.0 / np.maximum(depth[valid], 1e-8)
        lo, hi = np.percentile(disparity[valid], [5, 95])
        disparity = np.clip((disparity - lo) / max(hi - lo, 1e-8), 0, 1)
        disparity[~valid] = 0.0
    return (cm.magma(disparity)[..., :3] * 255).astype(np.uint8)


# Must live under tempfile.gettempdir(): Gradio only serves files from the
# system temp dir (or cwd), and gettempdir() honors TMPDIR, which clusters
# often point away from /tmp -- a hardcoded /tmp breaks serving there. On HF
# Spaces it still resolves to /tmp, the writable mount. We write the GLB here
# from the parent process (i.e. NOT inside the @spaces.GPU subprocess) so
# Gradio's file route can read it. Unique filename per call so Gradio's
# content-hashed cache always serves fresh bytes.
_ARTIFACT_DIR = Path(tempfile.gettempdir()) / "flux_rgbd_artifacts"

# 2h is comfortably longer than any viewer session; keeps a busy Space's
# /tmp bounded since nothing else ever deletes these.
_ARTIFACT_TTL_S = 2 * 3600.0


def _prune_old_artifacts() -> None:
    now = time.time()
    for f in _ARTIFACT_DIR.glob("cloud_*.glb"):
        try:
            if now - f.stat().st_mtime > _ARTIFACT_TTL_S:
                f.unlink()
        except OSError:
            pass  # concurrent delete / fs hiccup -- never fail a generation


_ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)


@spaces.GPU(duration=120)
def _sample_on_gpu(
    prompt: str,
    input_image,
    num_steps: int,
    cfg_scale: float,
    seed: int,
    refine_depth: bool = True,
    log2_alpha: float = 5.0,
):
    """GPU-only step: text encode + diffusion sample + VAE decode.

    Returns plain numpy arrays so the parent process (which is what
    serves Gradio files) can do the rest. Writing the GLB here would
    leave it in the subprocess's filesystem view where the parent's
    Gradio file route can't find it (returns 404).
    """
    runner = _ensure_runner()
    mode = "i2d" if input_image is not None else "joint"

    target = runner.img_hw[0]
    if mode == "i2d":
        letterboxed, (top, left, vh, vw) = _letterbox(input_image, target)
        model_input = letterboxed
    else:
        letterboxed = None
        top = left = 0
        vh = vw = target
        model_input = None

    t0 = time.time()
    if mode == "i2d":
        # Image given: single image->depth pass at CFG 1.0 (no guidance -- the
        # RGB is fixed, so there is nothing for CFG to steer).
        result = runner.generate(
            prompt.strip() if prompt else "",
            mode="i2d",
            num_steps=int(num_steps),
            cfg_scale=1.0,
            seed=int(seed),
            clean_rgb_image=model_input,
        )
    else:
        # Text->RGBD. Stage 1 joint at the requested CFG (default 4.0), rgb-first
        # trajectory (log2_alpha=5) for cleaner depth. When `refine_depth` is on,
        # a stage 2 re-derives depth via i2d on that RGB at CFG 1.0 for sharper,
        # RGB-consistent geometry; otherwise the single joint pass is used.
        result = runner.generate(
            prompt.strip() if prompt else "",
            mode="joint",
            num_steps=int(num_steps),
            cfg_scale=float(cfg_scale),
            seed=int(seed),
            log2_alpha=float(log2_alpha),
            refine_depth_i2d=bool(refine_depth),
            i2d_cfg_scale=1.0,
        )
    elapsed = time.time() - t0

    rgb_for_pc = (
        letterboxed[top : top + vh, left : left + vw]
        if mode == "i2d"
        else result["rgb"]
    )
    depth = result["depth"]
    if mode == "i2d":
        depth = depth[top : top + vh, left : left + vw]
    return rgb_for_pc, depth, mode, elapsed


def generate(
    prompt: str,
    input_image,
    num_steps: int,
    cfg_scale: float,
    seed: int,
    refine_depth: bool = True,
    log2_alpha: float = 5.0,
    edge_rtol: float = 0.04,
    sor: bool = False,
):
    """Public Gradio handler.

    Runs the GPU step, then writes the GLB here in the parent process so
    the file persists for Gradio.
    """
    rgb_for_pc, depth, mode, elapsed = _sample_on_gpu(
        prompt,
        input_image,
        num_steps,
        cfg_scale,
        seed,
        refine_depth,
        log2_alpha,
    )

    pts, cols = _depth_to_pointcloud(
        rgb_for_pc, depth, edge_rtol=edge_rtol, sor=bool(sor)
    )
    _prune_old_artifacts()
    cloud_path = str(_ARTIFACT_DIR / f"cloud_{uuid.uuid4().hex[:12]}.glb")
    _save_glb(cloud_path, pts, cols)

    valid = (depth > 0) & np.isfinite(depth)
    if valid.any():
        d = depth[valid]
        depth_summary = (
            f"depth median={float(np.median(d)):.2f}  "
            f"p5={float(np.percentile(d, 5)):.2f}  "
            f"p95={float(np.percentile(d, 95)):.2f}"
        )
    else:
        depth_summary = "depth has no valid pixels"
    status = f"{mode} · {elapsed:.1f} s · {depth_summary} · {pts.shape[0]:,} points"
    return rgb_for_pc, _depth_to_magma(depth), cloud_path, status


# --- Presentation layer ----------------------------------------------------
# Only the Gradio UI definition lives below.

_WORLD_LABS_URL = "https://www.worldlabs.ai"
_PROJECT_URL = "https://modality-forcing.github.io/"
_ARXIV_URL = "https://arxiv.org/abs/2606.13676"
_CODE_URL = "https://github.com/Duisterhof/modality-forcing"

# Editorial monochrome: a fully neutral palette, the system UI stack for body
# text, the system mono stack for the uppercase "eyebrow" labels. The serif
# display face for the title (Gilda Display) is pulled in via @import in the
# CSS below.
_THEME = gr.themes.Default(
    # System fonts only -- no Google-fetched web fonts for the body/mono, which
    # were loading unreliably (falling back to Arial and looking cheap).
    font=(
        "system-ui",
        "-apple-system",
        "Segoe UI",
        "Helvetica Neue",
        "Arial",
        "sans-serif",
    ),
    font_mono=(
        "ui-monospace",
        "SFMono-Regular",
        "Menlo",
        "Consolas",
        "monospace",
    ),
    primary_hue=gr.themes.colors.neutral,
    secondary_hue=gr.themes.colors.neutral,
    neutral_hue=gr.themes.colors.neutral,
).set(
    # Hairline, low-contrast borders; no heavy shadows or filled labels.
    block_border_width="1px",
    block_border_color="*neutral_200",
    block_background_fill="white",
    block_shadow="none",
    block_label_background_fill="transparent",
    block_label_border_width="0px",
    block_label_text_weight="500",
    input_border_width="1px",
    input_border_color="*neutral_200",
    input_shadow="none",
    # Near-black, fully-rounded primary button (pill); white secondary.
    button_large_radius="*radius_xxl",
    button_small_radius="*radius_xxl",
    button_primary_background_fill="#111111",
    button_primary_background_fill_hover="#1f1f1f",
    button_primary_text_color="white",
    button_primary_border_color="#111111",
    button_secondary_background_fill="white",
    button_secondary_border_color="rgba(0,0,0,0.16)",
)

_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Gilda+Display&display=swap');

/* Warm "paper" canvas everywhere (page + app + container) so the white
   component cards lift off the background and the layout reads premium
   rather than flat white-on-white. Body uses the system UI font. */
html, body, gradio-app, .gradio-container, .gradio-container .gap {
    background: #f4f3ef !important;
}
.gradio-container {
    color: #141414 !important;
    max-width: 1120px !important;
    margin: 0 auto !important;
    padding: 28px 24px 12px !important;
    font-family: system-ui, -apple-system, "Segoe UI", "Helvetica Neue",
        Arial, sans-serif !important;
}

/* Components become quiet white cards: hairline edge, soft round corners,
   and a whisper of shadow for depth. (Also overrides this Gradio build's
   hardcoded 3px black .block border.) */
.gradio-container .block {
    border: 1px solid rgba(20,20,20,0.07) !important;
    border-radius: 16px !important;
    background: #ffffff !important;
    box-shadow: 0 1px 2px rgba(20,20,20,0.04),
                0 12px 28px -18px rgba(20,20,20,0.18) !important;
}

/* Text/HTML blocks float on the page — no card border, fill, or shadow. */
.gradio-container .mf-bare {
    border: 0 !important; background: transparent !important;
    box-shadow: none !important; padding: 0 !important;
}

/* Hairline rule separating the masthead from the workspace. */
.mf-rule {
    height: 1px; border: 0; background: rgba(20,20,20,0.08);
    max-width: 1080px; margin: 0.75rem auto 1.5rem;
}

/* ---- Publication header (mirrors the project page) ---- */
/* Everything in the masthead is centered. Forced with !important because
   Gradio's prose CSS otherwise left-aligns <p>/<div> inside gr.HTML. */
.mf-pub, .mf-pub *, .mf-intro, .mf-intro * { text-align: center !important; }
.mf-pub { margin: 0.25rem auto 0.25rem; }
.mf-pub-title {
    font-weight: 600;
    font-size: clamp(1.9rem, 4.2vw, 3rem);
    line-height: 1.13; letter-spacing: -0.01em;
    color: #363636; margin: 0 auto 0.7em; max-width: 900px;
}
.mf-authors {
    font-size: clamp(1rem, 1.4vw, 1.25rem); line-height: 1.5;
    color: #363636; margin: 0 auto;
}
.mf-authors a { color: #3273dc !important; text-decoration: none !important; }
.mf-authors a:hover { text-decoration: underline !important; }
.mf-authors .ab { white-space: nowrap; margin: 0 0.15em; }
.mf-affil {
    font-size: clamp(0.95rem, 1.3vw, 1.2rem); color: #363636;
    margin: 0.5em auto 0;
}
.mf-affil .ab { margin: 0 0.6em; }
.mf-logos {
    display: flex; justify-content: center; align-items: center;
    gap: 40px; flex-wrap: wrap; margin: 1.4em auto 0;
}
.mf-logos img { height: 78px; }
.mf-venue { font-weight: 700; color: #363636; margin: 1em auto 0; }
.mf-links {
    display: flex; justify-content: center; gap: 12px;
    flex-wrap: wrap; margin: 1.1em auto 0.25rem;
}
.mf-btn {
    display: inline-flex; align-items: center; gap: 7px;
    padding: 7px 18px; border-radius: 9999px;
    background: #363636; color: #ffffff !important;
    font-size: 14px; font-weight: 500; text-decoration: none !important;
    transition: background .15s ease;
}
.mf-btn:hover { background: #4a4a4a; }
.mf-pub-sub {
    font-size: clamp(1rem, 1.3vw, 1.2rem); line-height: 1.5;
    color: #4a4a4a; max-width: 720px; margin: 1.2em auto 0; font-weight: 400;
}

/* ---- Quiet helper line + section eyebrows ---- */
.mf-intro {
    text-align: center; max-width: 640px; margin: 0.25rem auto 1rem;
    font-size: 14px; line-height: 1.6; color: #6b6b6b;
}
.mf-intro b { color: #111111; font-weight: 600; }
.mf-sec {
    font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
    font-size: 11px; font-weight: 500; letter-spacing: 0.18em;
    text-transform: uppercase; color: #9a9a9a; margin: 4px 2px 2px;
}

/* ---- Primary button: full pill, ink-black ---- */
.gradio-container button.primary,
.gradio-container button.lg.primary {
    background: #111111 !important; color: #ffffff !important;
    border: 1px solid #111111 !important; border-radius: 9999px !important;
    font-weight: 500 !important; letter-spacing: -0.005em !important;
}
.gradio-container button.primary:hover { background: #1f1f1f !important; }

/* ---- Examples: borderless, quiet ---- */
.gradio-container .examples table,
.gradio-container .examples .tr-head { border: 0 !important; }
.gradio-container .examples td {
    border-color: rgba(0,0,0,0.06) !important; font-size: 13px !important;
}

/* ---- Footer ---- */
.mf-footer {
    text-align: center; margin-top: 2rem; padding-top: 1.1rem;
    border-top: 1px solid rgba(0,0,0,0.08);
}
.mf-footer .mf-cta { font-size: 14px; color: #3d3d3d; }
.mf-footer .mf-cta a {
    color: #111111 !important; text-decoration: none !important;
    border-bottom: 1px solid rgba(0,0,0,0.25);
}
.mf-footer .mf-credit {
    font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
    font-size: 11px; letter-spacing: 0.08em; text-transform: uppercase;
    color: #9a9a9a; margin-top: 6px;
}
"""

_HEADER_HTML = (
    '<div class="mf-pub">'
    '<h1 class="mf-pub-title">Modality Forcing for Scalable Spatial Generation</h1>'
    '<div class="mf-authors">'
    '<span class="ab"><a href="https://bart-ai.com" target="_blank" '
    'rel="noreferrer">Bardienus Pieter Duisterhof</a><sup>1,2</sup>,</span>'
    '<span class="ab"><a href="https://www.cs.cmu.edu/~deva/" target="_blank" '
    'rel="noreferrer">Deva Ramanan</a><sup>1</sup>,</span>'
    '<span class="ab"><a href="https://ichnow.ski" target="_blank" '
    'rel="noreferrer">Jeffrey Ichnowski</a><sup>1</sup>,</span>'
    '<span class="ab"><a href="https://web.eecs.umich.edu/~justincj/" '
    'target="_blank" rel="noreferrer">Justin Johnson</a><sup>2</sup>,</span>'
    '<span class="ab"><a href="https://keunhong.com" target="_blank" '
    'rel="noreferrer">Keunhong Park</a><sup>2</sup></span>'
    "</div>"
    '<div class="mf-affil">'
    '<span class="ab"><sup>1</sup>Carnegie Mellon University</span>'
    '<span class="ab"><sup>2</sup>World Labs</span>'
    "</div>"
    '<div class="mf-logos">'
    '<img alt="Carnegie Mellon University" '
    'src="https://modality-forcing.github.io/static/images/cmu_logo.png">'
    '<img alt="World Labs" '
    'src="https://modality-forcing.github.io/static/images/world_labs_logo.jpg" '
    'style="border-radius:12px;">'
    "</div>"
    '<div class="mf-venue">Preprint, 2026</div>'
    '<div class="mf-links">'
    f'<a class="mf-btn" href="{_PROJECT_URL}" target="_blank" '
    'rel="noopener">📄 Project Page</a>'
    f'<a class="mf-btn" href="{_ARXIV_URL}" target="_blank" rel="noopener">📚 arXiv</a>'
    f'<a class="mf-btn" href="{_CODE_URL}" target="_blank" rel="noopener">⌨ Code</a>'
    "</div>"
    '<div class="mf-pub-sub" style="text-align:center !important;">Modality '
    "Forcing turns a pretrained text-to-image diffusion transformer into a "
    "joint image-depth generator with a simple post-training recipe.</div>"
    "</div>"
)

_INTRO_HTML = (
    '<div class="mf-intro">Type a scene and press <b>Generate</b>, or upload an '
    "image to run <b>image→depth</b> mode instead.</div>"
)

_EXAMPLE_IMAGES = [
    ["assets/examples/alley.png"],
    ["assets/examples/yosemite.png"],
    ["assets/examples/castle.png"],
]

_EXAMPLE_PROMPTS = [
    [DEFAULT_PROMPT],
    [
        "A sunlit Scandinavian living room with a linen sofa, a low oak coffee "
        "table, and tall windows opening onto a snowy courtyard."
    ],
    [
        "A misty pine forest at dawn, shafts of golden light cutting between the "
        "trunks and a narrow dirt trail winding into the distance."
    ],
    [
        "A cozy bookshop interior with floor-to-ceiling wooden shelves, a rolling "
        "ladder, warm pendant lighting, and a worn leather reading chair."
    ],
    [
        "A still life on a marble countertop: a bowl of ripe lemons, a ceramic "
        "pitcher, and a sprig of rosemary lit by soft side light."
    ],
    [
        "A coastal cliffside at golden hour overlooking a turquoise bay, with wild "
        "grass in the foreground and distant sailboats on the water."
    ],
]

with gr.Blocks(title="Modality Forcing — World Labs") as demo:
    gr.HTML(_HEADER_HTML, elem_classes="mf-bare")
    gr.HTML(_INTRO_HTML, elem_classes="mf-bare")
    gr.HTML('<hr class="mf-rule">', elem_classes="mf-bare")

    with gr.Row(equal_height=False):
        with gr.Column(scale=2, min_width=320) as left_col:
            gr.HTML('<div class="mf-sec">Input</div>', elem_classes="mf-bare")
            prompt = gr.Textbox(
                value=DEFAULT_PROMPT,
                lines=4,
                label="Scene prompt",
                placeholder="Describe a scene to generate…",
            )
            input_image = gr.Image(
                label="Optional input image (switches to image→depth mode)",
                type="numpy",
                height=200,
                sources=("upload", "clipboard"),
            )
            btn = gr.Button("Generate", variant="primary", size="lg")

            with gr.Accordion("Advanced settings", open=False):
                with gr.Row():
                    num_steps = gr.Slider(
                        1, 80, value=50, step=1, label="Sampling steps"
                    )
                    cfg_scale = gr.Slider(
                        1.0,
                        8.0,
                        value=4.0,
                        step=0.1,
                        label="Guidance (CFG) — text mode only",
                    )
                log2_alpha = gr.Slider(
                    -5.0,
                    5.0,
                    value=5.0,
                    step=0.5,
                    label="log2(alpha) — depth trajectory (>0 rgb-first, "
                    "cleaner depth; 0 diagonal; <0 depth-first) — text mode",
                )
                edge_rtol = gr.Slider(
                    0.0,
                    0.25,
                    value=0.04,
                    step=0.005,
                    label="Point-cloud depth-edge mask (rtol) — drop pixels at "
                    "depth jumps > this; lower = more aggressive, 0 = off",
                )
                sor_toggle = gr.Checkbox(
                    value=False,
                    label="Statistical outlier removal (point cloud) — drops "
                    "isolated floaters; can over-trim fine structures",
                )
                seed = gr.Number(value=0, precision=0, label="Seed")
                refine_depth = gr.Checkbox(
                    value=True,
                    label="Refine depth: joint (CFG) → image→depth (CFG 1) "
                    "— text mode only",
                )

            status = gr.Textbox(label="Status", interactive=False)

        with gr.Column(scale=3, min_width=380):
            gr.HTML('<div class="mf-sec">Output</div>', elem_classes="mf-bare")
            with gr.Row():
                rgb_out = gr.Image(
                    label="RGB image", type="numpy", height=320, format="png"
                )
                depth_out = gr.Image(
                    label="Depth (disparity, magma)",
                    type="numpy",
                    height=320,
                    format="png",
                )
            with gr.Group():
                cloud_out = gr.Model3D(
                    label="Interactive 3D point cloud",
                    clear_color=(0.12, 0.12, 0.14, 1.0),
                    zoom_speed=0.5,
                    pan_speed=0.5,
                    height=520,
                )

    def _run_text_example(prompt_text):
        # Cached example runs pin the UI defaults so the cache stays valid.
        return generate(
            prompt_text,
            None,
            num_steps=50,
            cfg_scale=4.0,
            seed=0,
            refine_depth=True,
            log2_alpha=5.0,
            edge_rtol=0.04,
            sor=False,
        )

    def _run_image_example(image):
        return generate(
            "",
            image,
            num_steps=50,
            cfg_scale=4.0,
            seed=0,
            refine_depth=True,
            log2_alpha=5.0,
            edge_rtol=0.04,
            sor=False,
        )

    # Defined after the output components exist, rendered back into the left
    # column. Cached: clicking an example serves precomputed results instead
    # of spending GPU time ("lazy" = no cache rebuild at startup; the
    # committed cache ships with the Space repo).
    _CACHE_MODE = os.environ.get("EXAMPLES_CACHE_MODE", "lazy")
    with left_col:
        gr.Examples(
            examples=_EXAMPLE_PROMPTS,
            inputs=[prompt],
            outputs=[rgb_out, depth_out, cloud_out, status],
            fn=_run_text_example,
            cache_examples=True,
            cache_mode=_CACHE_MODE,
            label="Example prompts",
        )
        gr.Examples(
            examples=_EXAMPLE_IMAGES,
            inputs=[input_image],
            outputs=[rgb_out, depth_out, cloud_out, status],
            fn=_run_image_example,
            cache_examples=True,
            cache_mode=_CACHE_MODE,
            label="Example images (image → depth)",
        )

    gr.HTML(
        '<div class="mf-footer">'
        f'<div class="mf-cta">Built by <a href="{_WORLD_LABS_URL}" '
        'target="_blank" rel="noopener noreferrer">World Labs</a></div>'
        '<div class="mf-credit">worldlabs.ai · Modality Forcing</div>'
        "</div>",
        elem_classes="mf-bare",
    )

    btn.click(
        generate,
        inputs=[
            prompt,
            input_image,
            num_steps,
            cfg_scale,
            seed,
            refine_depth,
            log2_alpha,
            edge_rtol,
            sor_toggle,
        ],
        outputs=[rgb_out, depth_out, cloud_out, status],
    )

    # Adding an image switches the demo into image->depth mode, where the prompt
    # is optional -- clear it so it defaults to empty. The user can retype one.
    input_image.change(
        lambda img: "" if img is not None else gr.update(),
        inputs=[input_image],
        outputs=[prompt],
    )

# The UI is designed light-only (paper-white cards, explicit light CSS). A
# visitor whose OS is in dark mode otherwise gets gradio's dark text colors on
# our light backgrounds -- an unreadable mix. Redirect to ?__theme=light in
# <head>, before gradio boots, so every visitor gets the light theme.
_FORCE_LIGHT_HEAD = """
<script>
  (function () {
    const url = new URL(window.location);
    if (url.searchParams.get("__theme") !== "light") {
      url.searchParams.set("__theme", "light");
      window.location.replace(url);
    }
  })();
</script>
"""

if __name__ == "__main__":
    # Gradio 6 takes theme + css on launch() (not the Blocks constructor),
    # so they must be passed here to actually apply on the Space.
    demo.queue(max_size=4).launch(theme=_THEME, css=_CSS, head=_FORCE_LIGHT_HEAD)
