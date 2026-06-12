# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 World Labs.
"""Shared helpers for the joint / i2d / d2i generation scripts.

Keeps the per-mode scripts short and readable: each one parses its own
arguments and calls the model, while the model loading, point-cloud export,
and a couple of small I/O helpers live here.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Persistent torch.compile kernel cache. Must be set before torch/transformers
# are imported — inductor caches the first cache-dir lookup.
os.environ.setdefault(
    "TORCHINDUCTOR_CACHE_DIR",
    str(Path("~/.cache/modality-forcing/torchinductor").expanduser()))

import numpy as np

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

DEFAULT_MODEL = "bartduis/modality_forcing"
DEFAULT_TEXT_ENCODER = "Qwen/Qwen3-8B"


def add_shared_args(parser: argparse.ArgumentParser) -> None:
    """Arguments common to every generation mode."""
    parser.add_argument("--prompt", required=True, help="Text prompt.")
    parser.add_argument("--model", default=DEFAULT_MODEL,
                        help="HuggingFace repo id or local path to the weights.")
    parser.add_argument("--text-encoder", default=DEFAULT_TEXT_ENCODER,
                        help="HuggingFace repo id of the Qwen3 text encoder.")
    parser.add_argument("--num-steps", type=int, default=50,
                        help="Number of flow-matching sampling steps.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resolution", type=int, default=512,
                        help="Generation resolution (square). Must match the "
                             "checkpoint's training resolution: 512 for the "
                             "default model, 1024 for the 1024 checkpoint.")
    parser.add_argument("--output-dir", default="./outputs",
                        help="Directory to write rgb / depth / metadata into.")
    parser.add_argument("--compile", action="store_true",
                        help="torch.compile the DiT (reduce-overhead). The "
                             "first run compiles for a few minutes (seconds "
                             "once the on-disk cache is warm). Pays off for "
                             "repeated generations.")


def load_runner(args: argparse.Namespace) -> FluxRGBDRunner:
    """Build the runner from parsed args. The DiT runs in bfloat16 (fp16
    overflows to NaN in this model); the depth head is kept in fp32, which
    avoids banding artifacts in the depth. The whole pipeline (sampling grid,
    VAE, depth) runs at ``--resolution``."""
    res = int(getattr(args, "resolution", 512))
    if res % 16 != 0:
        raise ValueError(f"--resolution must be a multiple of 16; got {res}")
    if str(args.device).startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit(
            "CUDA was requested but torch.cuda.is_available() is False — the "
            "installed torch build does not match your GPU driver. Reinstall "
            "a matching build (see README → Installation → Troubleshooting), "
            "or pass --device cpu.")
    return FluxRGBDRunner.from_pretrained(
        args.model,
        device=args.device,
        dtype=torch.bfloat16,
        head_dtype=torch.float32,
        text_encoder=args.text_encoder,
        img_hw=(res, res),
        compile_model=bool(getattr(args, "compile", False)),
    )


def read_image_rgb(path: str) -> np.ndarray:
    """Load an image as uint8 RGB (H, W, 3)."""
    import cv2
    bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"could not read image: {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def read_depth_map(path: str) -> np.ndarray:
    """Load a depth map as float32 (H, W).

    Supports ``.npy`` (raw float depth) and 16-bit single-channel PNG/TIFF
    (read as-is — only the *relative* structure matters, see ``encode_depth``).
    """
    if path.endswith(".npy"):
        depth = np.load(path)
    else:
        import cv2
        depth = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if depth is None:
            raise FileNotFoundError(f"could not read depth: {path}")
        if depth.dtype == np.uint8:
            print("[depth] WARNING: 8-bit input — depth is heavily quantized. "
                  "Prefer .npy or a 16-bit PNG/TIFF.", file=sys.stderr)
    depth = np.asarray(depth, dtype=np.float32)
    if depth.ndim == 3:
        print("[depth] WARNING: multi-channel input — using channel 0. If this "
              "is a colormapped depth visualization the result will be wrong; "
              "pass the raw depth (.npy or 16-bit PNG/TIFF).", file=sys.stderr)
        depth = depth[..., 0]
    return depth


def write_point_cloud(rgb_u8: np.ndarray, depth: np.ndarray, out_path: str,
                      *, fov_deg: float = 65.0, max_points: int = 1_200_000,
                      edge_rtol: float = 0.04, sor: bool = False) -> tuple[int, list[str]]:
    """Back-project (rgb, depth) into a colored point cloud; save GLB + PLY.

    Writes ``out_path`` (GLB, for 3D viewers) and a sibling ``.ply`` (for
    point-cloud tools). Returns ``(num_points, [glb_path, ply_path])``. Assumes
    a centered pinhole camera with the given vertical field of view.
    ``edge_rtol`` drops depth-edge (occlusion-boundary) pixels before
    back-projection; 0 disables it.
    """
    import trimesh
    h, w = depth.shape
    fx = w / (2.0 * np.tan(np.deg2rad(fov_deg) / 2.0))
    cx, cy = w * 0.5, h * 0.5
    valid = (depth > 0) & np.isfinite(depth)
    if edge_rtol and edge_rtol > 0:
        valid &= ~depth_edge_mask(depth, rtol=edge_rtol)
    v_idx, u_idx = np.where(valid)
    z = depth[v_idx, u_idx]
    x = (u_idx + 0.5 - cx) * z / fx
    y = (v_idx + 0.5 - cy) * z / fx
    # glTF convention: +Y up, camera looks down -Z.
    pts = np.stack([x, -y, -z], axis=-1).astype(np.float32)
    cols = rgb_u8[v_idx, u_idx]
    if sor:
        # Statistical outlier rejection: drops isolated floaters, but can
        # over-trim fine structures — opt-in via --sor (the edge mask above
        # is the default cleanup).
        inliers = statistical_outlier_mask(pts)
        pts, cols = pts[inliers], cols[inliers]
    if pts.shape[0] > max_points:
        idx = np.random.default_rng(0).choice(pts.shape[0], max_points, replace=False)
        pts, cols = pts[idx], cols[idx]
    if pts.shape[0]:
        pts -= np.median(pts, axis=0, keepdims=True)
    cloud = trimesh.PointCloud(vertices=pts, colors=cols)
    glb_path = str(out_path)
    scene = trimesh.Scene()
    scene.add_geometry(cloud)
    scene.export(glb_path)              # GLB for 3D viewers
    ply_path = str(Path(out_path).with_suffix(".ply"))
    cloud.export(ply_path)             # PLY for point-cloud tools
    return int(pts.shape[0]), [glb_path, ply_path]
