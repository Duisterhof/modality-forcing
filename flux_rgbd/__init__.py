# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 World Labs.
"""flux_rgbd -- joint text -> RGB + depth diffusion on top of FLUX.2."""

__version__ = "0.1.0"
__all__ = ["FluxRGBDRunner"]


def __getattr__(name):
    if name == "FluxRGBDRunner":
        from flux_rgbd.runner import FluxRGBDRunner

        return FluxRGBDRunner
    raise AttributeError(name)
