# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 World Labs.
"""Depth-side decoding: model tokens -> depth pixel map.

Decoding steps:
  1. Unpatchify the (tok_h, tok_w, patch_h * patch_w) token grid.
  2. Divide out the post-normalization scalar `depth_value_scale`.
  3. Apply the inverse of every recorded normalization stage, in reverse.

Only `"contract"` (the mip-NeRF 360 radial squash, arXiv 2111.12077
Eq. 10) is genuinely invertible -- the other stages discard per-sample
statistics during encoding and pass through unchanged on decode.
"""

from __future__ import annotations

import dataclasses

import einops
import torch
from torch import Tensor

_KNOWN_STAGES = frozenset({"contract", "normal", "mean", "unit_mean", "none"})


@dataclasses.dataclass(frozen=True)
class DepthConfig:
    """How depth was normalized at training time. Decoder reads these."""

    depth_normalize_mode: tuple[str, ...] | str = ("unit_mean", "contract")
    patch_size: int = 16
    depth_value_scale: float = 1.0

    def __post_init__(self) -> None:
        stages = self.depth_normalize_mode
        if isinstance(stages, str):
            object.__setattr__(self, "depth_normalize_mode", (stages,))
        unknown = set(self.depth_normalize_mode) - _KNOWN_STAGES
        if unknown:
            raise ValueError(
                f"unknown stages {sorted(unknown)}; expected {sorted(_KNOWN_STAGES)}"
            )
        if self.depth_value_scale <= 0:
            raise ValueError(
                f"depth_value_scale must be > 0; got {self.depth_value_scale}"
            )
        if self.patch_size <= 0:
            raise ValueError(f"patch_size must be > 0; got {self.patch_size}")


def _inv_contract(z: Tensor) -> Tensor:
    """Inverse of the mip-NeRF 360 radial contract bijection (operates on last dim).

    Contract maps R^n into the ball of radius 2; this undoes it. For |z| <= 1
    the contract was identity, so this is also identity in that region.
    """
    mag_sq = torch.clamp(z.square().sum(dim=-1, keepdim=True), min=1.0)
    return z / (2.0 * mag_sq.sqrt() - mag_sq)


def _undo_stage(depth: Tensor, stage: str) -> Tensor:
    if stage != "contract":  # "normal"/"mean"/"unit_mean"/"none" are pass-through.
        return depth
    if depth.shape[-1] == 1:
        return _inv_contract(depth)
    return _inv_contract(depth.unsqueeze(-1)).squeeze(-1)


def decode_depth(tokens: Tensor, config: DepthConfig) -> Tensor:
    """Token grid -> depth pixel map.

    Input  shape: (..., tok_h, tok_w, patch_h * patch_w)
    Output shape: (..., tok_h * patch_size, tok_w * patch_size, 1)
    """
    p = config.patch_size
    depth = einops.rearrange(
        tokens, "... h w (p1 p2 c) -> ... (h p1) (w p2) c", p1=p, p2=p
    )
    if config.depth_value_scale != 1.0:
        depth = depth / config.depth_value_scale
    for stage in reversed(config.depth_normalize_mode):
        depth = _undo_stage(depth, stage)
    return depth


def _contract(z: Tensor) -> Tensor:
    """Forward mip-NeRF 360 radial contract (inverse of `_inv_contract`).

    Identity for |z| <= 1; otherwise squashes radius r into (2 - 1/r).
    """
    mag = z.norm(dim=-1, keepdim=True)
    safe = mag.clamp(min=1e-12)
    scale = torch.where(mag <= 1.0, torch.ones_like(mag), (2.0 - 1.0 / safe) / safe)
    return z * scale


def _apply_stage(depth: Tensor, stage: str) -> Tensor:
    """Forward of `_undo_stage`. Only `contract` and `unit_mean` are active.

    `unit_mean` divides by the per-map mean so the depth is scale-normalised
    the way the model was trained; it is pass-through on decode (the absolute
    scale is not recoverable), so encode/decode round-trips up to that scale.
    """
    if stage == "unit_mean":
        return depth / depth.mean().clamp(min=1e-12)
    if stage != "contract":  # "normal"/"mean"/"none" are pass-through.
        return depth
    if depth.shape[-1] == 1:
        return _contract(depth)
    return _contract(depth.unsqueeze(-1)).squeeze(-1)


def encode_depth(depth_map: Tensor, config: DepthConfig) -> Tensor:
    """Depth pixel map -> token grid (inverse of `decode_depth`).

    Input  shape: (..., H, W, 1)
    Output shape: (..., H/patch_size, W/patch_size, patch_size**2)

    Used for ``mode="d2i"`` to turn a depth map into the depth-stream tokens
    the model conditions on. Because `unit_mean` is scale-normalising, only the
    relative depth structure matters -- the input need not be metric.
    """
    p = config.patch_size
    depth = depth_map
    for stage in config.depth_normalize_mode:
        depth = _apply_stage(depth, stage)
    if config.depth_value_scale != 1.0:
        depth = depth * config.depth_value_scale
    return einops.rearrange(
        depth, "... (h p1) (w p2) c -> ... h w (p1 p2 c)", p1=p, p2=p
    )
