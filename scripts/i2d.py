#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 World Labs.
"""Image-to-depth: text prompt + RGB image -> depth.

The RGB is held fixed, so there is nothing for guidance to steer -- CFG is 1.0.

Example:
    python scripts/i2d.py --prompt "" --image photo.jpg
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts._common import (
    add_shared_args,
    load_runner,
    read_image_rgb,
    write_point_cloud,
)


def main() -> int:
    p = argparse.ArgumentParser(description="Text + image -> depth (i2d mode).")
    add_shared_args(p)
    p.add_argument("--image", required=True, help="Path to the input RGB image.")
    p.add_argument(
        "--fov-deg",
        type=float,
        default=65.0,
        help="Vertical field of view used to back-project the cloud.",
    )
    p.add_argument(
        "--sor",
        action="store_true",
        help="Also apply statistical outlier removal to the point "
        "cloud (off by default; can over-trim fine structures).",
    )
    p.add_argument(
        "--edge-rtol",
        type=float,
        default=0.04,
        help="Point-cloud depth-edge mask: drop pixels at depth jumps "
        "> this fraction (removes occlusion-boundary floaters). "
        "Lower = more aggressive; 0 = off.",
    )
    args = p.parse_args()

    image = read_image_rgb(args.image)
    runner = load_runner(args)
    result = runner.generate(
        args.prompt,
        mode="i2d",
        num_steps=args.num_steps,
        cfg_scale=1.0,
        seed=args.seed,
        clean_rgb_image=image,
    )

    paths = runner.save(result, args.output_dir)
    for key, path in paths.items():
        print(f"  {key}: {path}")
    glb = os.path.join(paths["run_dir"], "cloud.glb")
    n, cloud_paths = write_point_cloud(
        result["rgb"],
        result["depth"],
        glb,
        fov_deg=args.fov_deg,
        edge_rtol=args.edge_rtol,
        sor=args.sor,
    )
    for cp in cloud_paths:
        print(f"  point_cloud: {cp} ({n:,} points)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
