#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 World Labs.
"""Depth-to-image: text prompt + depth map -> RGB image.

The depth map is encoded into the model's depth stream (see
``flux_rgbd.depth.preprocess.encode_depth``) and held fixed while the RGB is
sampled. The normalization is scale-invariant, so the input depth can be
metric or relative.

Example:
    python scripts/d2i.py --prompt "a cozy kitchen" --depth depth.npy
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts._common import (
    add_shared_args,
    load_runner,
    read_depth_map,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Text + depth -> RGB (d2i mode).")
    add_shared_args(parser)
    parser.add_argument(
        "--depth",
        required=True,
        help="Path to the input depth map (.npy or 16-bit PNG/TIFF).",
    )
    parser.add_argument(
        "--cfg-scale",
        type=float,
        default=4.0,
        help="Classifier-free guidance scale for the RGB stream.",
    )
    args = parser.parse_args()

    depth_map = read_depth_map(args.depth)
    runner = load_runner(args)
    depth_tokens = runner.encode_depth_map(depth_map)
    result = runner.generate(
        args.prompt,
        mode="d2i",
        num_steps=args.num_steps,
        cfg_scale=args.cfg_scale,
        seed=args.seed,
        clean_depth=depth_tokens,
    )

    paths = runner.save(result, args.output_dir)
    for key, path in paths.items():
        print(f"  {key}: {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
