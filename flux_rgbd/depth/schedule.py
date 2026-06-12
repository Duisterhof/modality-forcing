# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 World Labs.
"""Per-modality flow-matching timestep schedules for RGB + depth inference.

Each modality walks from t = t_max down to t = t_min, warped by the
SD3 / FLUX shift transform

    t' = exp(mu) / (exp(mu) + (1 / t - 1) ** sigma)

so larger mu pushes the schedule toward noisier (later) timesteps.

In asymmetric modes one stream is held at the boundary of its noise
range:
  "joint" -- both schedules are warped linear walks.
  "i2d"   -- RGB held at t = 0 (clean RGB conditioning).
  "d2i"   -- depth held at t = 0; RGB optionally held at t = 1.
"""

from __future__ import annotations

import dataclasses
import math
from typing import Literal

import torch
from torch import Tensor

Mode = Literal["joint", "i2d", "d2i"]


@dataclasses.dataclass
class ScheduleConfig:
    rgb_shift_mu: float = 0.0
    rgb_shift_sigma: float = 1.0
    depth_shift_mu: float = 0.0
    depth_shift_sigma: float = 1.0
    # Some d2i checkpoints were trained with RGB pinned at full noise; this
    # flag mirrors that conditioning at sample time.
    d2i_full_noise_rgb: bool = False


def _shift(mu: float, sigma: float, t: Tensor) -> Tensor:
    return math.exp(mu) / (math.exp(mu) + (1.0 / t - 1.0) ** sigma)


def _f_alpha(t: Tensor, alpha: float) -> Tensor:
    """Latent-forcing Möbius warp f(t) = alpha*t / (1 + (alpha-1)*t).

    Fixes 0 and 1.
    """
    return (alpha * t) / (1.0 + (alpha - 1.0) * t)


def rollout_timesteps(
    config: ScheduleConfig,
    num_steps: int,
    *,
    mode: Mode = "joint",
    log2_alpha: float | None = None,
    t_max: float = 1.0,
    t_min: float = 0.0,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
) -> tuple[Tensor, Tensor]:
    """Return per-modality timestep tensors of shape `(num_steps + 1,)`.

    ``log2_alpha`` (joint mode only) tilts the RGB/depth denoising trajectory:
    the depth schedule becomes ``f_alpha(t_rgb)`` with ``alpha = 2 **
    log2_alpha`` on top of the training time-shift density. ``alpha > 1``
    keeps depth noisier for longer so RGB resolves first (rgb-first ->
    cleaner depth); ``alpha < 1`` is depth-first; ``alpha = 1`` (or ``None``)
    is the diagonal joint schedule.
    """
    if num_steps < 1:
        raise ValueError(f"num_steps must be >= 1; got {num_steps}")

    linear = torch.linspace(t_max, t_min, num_steps + 1, device=device, dtype=dtype)
    t_rgb = _shift(config.rgb_shift_mu, config.rgb_shift_sigma, linear)
    t_depth = _shift(config.depth_shift_mu, config.depth_shift_sigma, linear)

    if mode == "i2d":
        t_rgb = torch.zeros_like(t_rgb)
    elif mode == "d2i":
        if config.d2i_full_noise_rgb:
            t_rgb = torch.ones_like(t_rgb)
        t_depth = torch.zeros_like(t_depth)
    elif mode == "joint":
        if log2_alpha is not None:
            # Warp the depth schedule off the (time-shifted) RGB grid, matching
            # the final-paper time-shift trajectory: nodes follow the training
            # sigmoid density, then the Möbius warp tilts depth vs RGB.
            t_depth = _f_alpha(t_rgb, 2.0**log2_alpha)
    else:
        raise ValueError(f"unknown mode {mode!r}; expected joint / i2d / d2i")

    return t_rgb, t_depth
