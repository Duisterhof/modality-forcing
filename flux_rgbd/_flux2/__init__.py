# SPDX-License-Identifier: Apache-2.0
# Copyright 2025 Black Forest Labs.
"""Vendored FLUX.2 reference implementation.

This subpackage holds the FLUX.2 architecture (Apache-2.0, Black Forest
Labs upstream) with light local modifications (autoencoder weight loading
and latent normalization). Keep edits minimal so these files stay easy to
diff against upstream. Higher-level RGB+D inference code lives in the
parent ``flux_rgbd`` package.

Upstream: https://github.com/black-forest-labs/flux2
"""
