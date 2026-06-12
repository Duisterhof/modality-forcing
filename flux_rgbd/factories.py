# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 World Labs.
"""Model variant builders. One function per published checkpoint."""

from __future__ import annotations

from flux_rgbd.depth.preprocess import DepthConfig
from flux_rgbd.depth.schedule import ScheduleConfig
from flux_rgbd.dit import FluxRGBDDiT
from flux_rgbd.model import FluxRGBD


def flux_rgbd_9b_v1() -> tuple[FluxRGBD, DepthConfig, ScheduleConfig]:
    """v1 9B checkpoint: 8 dual + 24 single + 4 depth-decoder blocks."""
    dit = FluxRGBDDiT(
        in_channels=128,
        depth_channels=256,
        context_in_dim=12_288,
        hidden_size=4096,
        num_heads=32,
        depth_double=8,
        depth_single=24,
        depth_decoder_num_layers=4,
        axes_dim=(32, 32, 32, 32),
        theta=2000,
        mlp_ratio=3.0,
    )
    return (
        FluxRGBD(dit),
        DepthConfig(depth_normalize_mode=("unit_mean", "contract"), patch_size=16),
        ScheduleConfig(rgb_shift_mu=1.1, depth_shift_mu=1.1),
    )


def flux_rgbd_9b_v2() -> tuple[FluxRGBD, DepthConfig, ScheduleConfig]:
    """v2 9B checkpoint: v1 architecture + cross-stream timestep mixing.

    Same backbone as v1 (8 dual + 24 single + 4 depth-decoder blocks; ~12B
    params total -- "9B" names the FLUX.2 backbone class), plus a cross-stream
    timestep-mixing path (``cross_alpha_*`` +
    ``time_in_{rgb_to_depth,depth_to_img}``): each stream's modulation vector
    sees a gated embedding of the *other* stream's timestep. Training-time
    depth-visibility gating was configured as a no-op in the released
    checkpoint and is not modelled at inference.
    """
    dit = FluxRGBDDiT(
        in_channels=128,
        depth_channels=256,
        context_in_dim=12_288,
        hidden_size=4096,
        num_heads=32,
        depth_double=8,
        depth_single=24,
        depth_decoder_num_layers=4,
        axes_dim=(32, 32, 32, 32),
        theta=2000,
        mlp_ratio=3.0,
        cross_stream_timestep_mixing=True,
    )
    return (
        FluxRGBD(dit),
        DepthConfig(depth_normalize_mode=("unit_mean", "contract"), patch_size=16),
        ScheduleConfig(rgb_shift_mu=1.1, depth_shift_mu=1.1),
    )


VARIANTS = {
    "flux_rgbd_9b_v1": flux_rgbd_9b_v1,
    "flux_rgbd_9b_v2": flux_rgbd_9b_v2,
}
