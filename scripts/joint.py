#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 World Labs.
"""Joint generation: text prompt -> RGB image + depth.

Example:
    python scripts/joint.py --prompt "a cozy sunlit kitchen"
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts._common import (
    add_shared_args,
    load_runner,
    write_point_cloud,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Text -> RGB + depth (joint mode).")
    add_shared_args(parser)
    parser.add_argument(
        "--cfg-scale",
        type=float,
        default=4.0,
        help="Classifier-free guidance scale for the RGB stream.",
    )
    parser.add_argument(
        "--log2-alpha",
        type=float,
        default=5.0,
        help="log2 of the RGB/depth trajectory tilt. >0 is rgb-first "
        "(cleaner depth), 0 is the diagonal joint schedule, <0 "
        "is depth-first.",
    )
    parser.add_argument(
        "--refine-depth",
        action="store_true",
        help="Sharpen depth with a second image->depth pass on the "
        "generated RGB (matches the online demo).",
    )
    parser.add_argument(
        "--fov-deg",
        type=float,
        default=65.0,
        help="Horizontal field of view used to back-project the cloud.",
    )
    parser.add_argument(
        "--sor",
        action="store_true",
        help="Also apply statistical outlier removal to the point "
        "cloud (off by default; can over-trim fine structures).",
    )
    parser.add_argument(
        "--edge-rtol",
        type=float,
        default=0.04,
        help="Point-cloud depth-edge mask: drop pixels at depth jumps "
        "> this fraction (removes occlusion-boundary floaters). "
        "Lower = more aggressive; 0 = off.",
    )
    args = parser.parse_args()

    runner = load_runner(args)
    result = runner.generate(
        args.prompt,
        mode="joint",
        num_steps=args.num_steps,
        cfg_scale=args.cfg_scale,
        seed=args.seed,
        log2_alpha=args.log2_alpha,
        refine_depth_i2d=args.refine_depth,
        i2d_cfg_scale=1.0,
    )

    paths = runner.save(result, args.output_dir)
    for key, path in paths.items():
        print(f"  {key}: {path}")
    glb = os.path.join(paths["run_dir"], "cloud.glb")
    num_points, cloud_paths = write_point_cloud(
        result["rgb"],
        result["depth"],
        glb,
        fov_deg=args.fov_deg,
        edge_rtol=args.edge_rtol,
        sor=args.sor,
    )
    for cloud_path in cloud_paths:
        print(f"  point_cloud: {cloud_path} ({num_points:,} points)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
