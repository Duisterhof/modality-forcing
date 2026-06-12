# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 World Labs.
"""Point-cloud post-processing utilities."""

from __future__ import annotations

import numpy as np


def statistical_outlier_mask(
    points: np.ndarray, *, num_neighbors: int = 5, std_ratio: float = 1.0
) -> np.ndarray:
    """Boolean inlier mask via statistical outlier rejection.

    For each point, the mean distance to its ``num_neighbors`` nearest
    neighbors is computed; a point is an outlier when that mean exceeds
    ``global_mean + std_ratio * global_std``. This removes the flying-pixel
    floaters that back-projected depth produces at depth discontinuities.
    Mirrors Open3D's ``remove_statistical_outlier``.

    Args:
        points: ``(N, 3)`` float array.
        num_neighbors: neighbors used to estimate each point's local spacing.
        std_ratio: smaller is more aggressive (keeps fewer points).

    Returns:
        ``(N,)`` boolean mask, ``True`` for inliers.
    """
    n = len(points)
    if n <= num_neighbors:
        return np.ones(n, dtype=bool)
    # Lazy import: only point-cloud export needs scipy.
    from scipy.spatial import cKDTree

    tree = cKDTree(points)
    # k + 1: the nearest neighbor is the point itself (distance 0); drop it.
    dists, _ = tree.query(points, k=num_neighbors + 1, workers=-1)
    mean_dist = dists[:, 1:].mean(axis=1)
    threshold = mean_dist.mean() + std_ratio * float(mean_dist.std())
    return mean_dist <= threshold


def depth_edge_mask(
    depth: np.ndarray, *, rtol: float = 0.04, kernel_size: int = 3
) -> np.ndarray:
    """Boolean mask, ``True`` on depth-discontinuity pixels.

    A pixel is an edge when the local depth range (max - min over a
    ``kernel_size`` window) exceeds ``rtol`` of its depth. Removing these
    pixels *before* back-projection deletes the occlusion-boundary "veils"
    that bridge foreground->background -- the connected floaters that
    statistical outlier rejection cannot catch (they have close neighbors).
    Mirrors MoGe / utils3d ``depth_edge`` (``rtol`` path).
    """
    from scipy.ndimage import maximum_filter, minimum_filter

    local_range = maximum_filter(depth, size=kernel_size) - minimum_filter(
        depth, size=kernel_size
    )
    return local_range > rtol * np.maximum(depth, 1e-6)
