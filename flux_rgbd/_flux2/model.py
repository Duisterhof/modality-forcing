# SPDX-License-Identifier: Apache-2.0
# Copyright 2025 Black Forest Labs.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Adapted from the FLUX.2 codebase:
#     https://github.com/black-forest-labs/flux2
"""
FLUX.2 diffusion transformer architecture for image generation and editing.

This module implements the core transformer architecture for FLUX.2 models from
Black Forest Labs. The architecture uses dual-stream and single-stream transformer
blocks to process text and image latents for text-to-image and image-to-image tasks.
"""

import math

import einops
import torch
from torch import Tensor, nn


class Flux2(nn.Module):
    """
    FLUX.2 diffusion transformer for image generation and editing.

    This is a flow-matching diffusion model that uses a stack of dual-stream and
    single-stream transformer blocks for text context and image latents. The
    model supports text-to-image and image-to-image generation tasks.

    Default parameter values match the FLUX.2 [klein] 4B architecture, which is
    optimized for fast inference. For other model variants (klein-9B or dev),
    use the parameters from :mod:`flux_rgbd._flux2.constants`.
    """

    def __init__(
        self,
        in_channels: int = 128,
        context_in_dim: int = 7680,
        hidden_size: int = 3072,
        num_heads: int = 24,
        depth: int = 5,
        depth_single_blocks: int = 20,
        axes_dim: tuple[int, int, int, int] = (32, 32, 32, 32),
        theta: int = 2000,
        mlp_ratio: float = 3.0,
        use_guidance_embed: bool = False,
    ):
        """
        Args:
            in_channels: Number of input channels for image latents. Matches the
                output dimension of the autoencoder used to encode images.
            context_in_dim: Dimension of text context embeddings from the text
                encoder. This should match the concatenated output dimension of
                the text encoder being used (7680 for Qwen3-4B, 12288 for Qwen3-8B,
                15360 for Mistral-Small).
            hidden_size: Hidden dimension size for transformer blocks. All
                attention and MLP operations use this dimension internally.
            num_heads: Number of attention heads in multi-head attention layers.
                Must evenly divide `hidden_size`.
            depth: Number of dual-stream transformer blocks. These blocks process
                text and image streams separately with cross-attention.
            depth_single_blocks: Number of single-stream transformer blocks. These
                blocks process the concatenated text+image sequence.
            axes_dim: Tuple of 4 integers specifying the dimensionality for each
                axis in rotary embeddings. Must sum to `hidden_size / num_heads`.
            theta: Base frequency for rotary position embeddings (RoPE). Higher
                values result in slower position encoding rotation.
            mlp_ratio: Expansion ratio for MLP hidden dimension relative to
                `hidden_size`. MLP hidden dim = `hidden_size * mlp_ratio`.
            use_guidance_embed: Whether to include guidance scale embeddings for
                classifier-free guidance. Set to False for distilled models where
                guidance is baked into weights.
        """
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = in_channels
        if hidden_size % num_heads != 0:
            raise ValueError(
                f"Hidden size {hidden_size} must be divisible by num_heads {num_heads}"
            )
        pe_dim = hidden_size // num_heads
        if sum(axes_dim) != pe_dim:
            raise ValueError(f"Got {axes_dim} but expected positional dim {pe_dim}")
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.pe_embedder = EmbedND(dim=pe_dim, theta=theta, axes_dim=axes_dim)
        self.img_in = nn.Linear(self.in_channels, self.hidden_size, bias=False)
        self.time_in = MLPEmbedder(
            in_dim=256, hidden_dim=self.hidden_size, disable_bias=True
        )
        self.txt_in = nn.Linear(context_in_dim, self.hidden_size, bias=False)

        self.use_guidance_embed = use_guidance_embed
        if self.use_guidance_embed:
            self.guidance_in = MLPEmbedder(
                in_dim=256, hidden_dim=self.hidden_size, disable_bias=True
            )

        double_blocks = [
            DoubleStreamBlock(self.hidden_size, self.num_heads, mlp_ratio=mlp_ratio)
            for _ in range(depth)
        ]
        self.double_blocks = nn.ModuleList(double_blocks)

        single_blocks = [
            SingleStreamBlock(self.hidden_size, self.num_heads, mlp_ratio=mlp_ratio)
            for _ in range(depth_single_blocks)
        ]
        self.single_blocks = nn.ModuleList(single_blocks)

        self.double_stream_modulation_img = Modulation(
            self.hidden_size, double=True, disable_bias=True
        )
        self.double_stream_modulation_txt = Modulation(
            self.hidden_size, double=True, disable_bias=True
        )
        self.single_stream_modulation = Modulation(
            self.hidden_size, double=False, disable_bias=True
        )

        self.final_layer = LastLayer(self.hidden_size, self.out_channels)

    def forward(
        self,
        x: Tensor,
        x_ids: Tensor,
        timesteps: Tensor,
        ctx: Tensor,
        ctx_ids: Tensor,
        guidance: Tensor | None,
    ):
        num_txt_tokens = ctx.shape[1]

        timestep_emb = timestep_embedding(timesteps, 256)
        vec = self.time_in(timestep_emb)
        if self.use_guidance_embed:
            guidance_emb = timestep_embedding(guidance, 256)
            vec = vec + self.guidance_in(guidance_emb)

        double_block_mod_img = self.double_stream_modulation_img(vec)
        double_block_mod_txt = self.double_stream_modulation_txt(vec)
        single_block_mod, _ = self.single_stream_modulation(vec)

        img = self.img_in(x)
        txt = self.txt_in(ctx)

        pe_x = self.pe_embedder(x_ids)
        pe_ctx = self.pe_embedder(ctx_ids)

        for block in self.double_blocks:
            img, txt = block(
                img,
                txt,
                pe_x,
                pe_ctx,
                double_block_mod_img,
                double_block_mod_txt,
            )

        img = torch.cat((txt, img), dim=1)
        pe = torch.cat((pe_ctx, pe_x), dim=2)

        for block in self.single_blocks:
            img = block(
                img,
                pe,
                single_block_mod,
            )

        img = img[:, num_txt_tokens:, ...]

        img = self.final_layer(img, vec)
        return img


class SelfAttention(nn.Module):
    """
    Projection layers for multi-head attention with QK normalization.

    This module holds a fused QKV projection, RMS normalization for queries and
    keys, and an output projection. It has no forward method; DoubleStreamBlock
    calls the submodules directly around a joint text+image attention.
    """

    def __init__(self, dim: int, num_heads: int = 8):
        """
        Args:
            dim: Hidden dimension size. Must be divisible by `num_heads`.
            num_heads: Number of parallel attention heads.
        """
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=False)

        self.norm = QKNorm(head_dim)
        self.proj = nn.Linear(dim, dim, bias=False)


class SiLUActivation(nn.Module):
    """
    Gated activation using SiLU (Swish) function.

    This module splits the input tensor along the last dimension, applies SiLU to one
    half, and element-wise multiplies it with the other half. This is commonly known
    as SwiGLU when used in MLP layers.
    """

    def __init__(self):
        super().__init__()
        self.gate_fn = nn.SiLU()

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Input tensor of shape `(..., 2 * dim)` where the last dimension will
                be split into two equal parts for gating.

        Returns:
            Gated output of shape `(..., dim)`.
        """
        x1, x2 = x.chunk(2, dim=-1)
        return self.gate_fn(x1) * x2


class Modulation(nn.Module):
    """
    Adaptive layer normalization (AdaLN) modulation layer.

    This module generates scale, shift, and gate parameters for adaptive normalization
    from timestep/guidance embeddings. For double-stream blocks, it produces two sets
    of modulation parameters (one for each stream).
    """

    def __init__(self, dim: int, double: bool, disable_bias: bool = False):
        """
        Args:
            dim: Hidden dimension size matching the transformer blocks.
            double: If True, generates parameters for dual-stream blocks (6 params:
                shift, scale, gate for each stream). If False, generates for
                single-stream blocks (3 params: shift, scale, gate).
            disable_bias: If True, the linear layer has no bias term.
        """
        super().__init__()
        self.is_double = double
        self.multiplier = 6 if double else 3
        self.lin = nn.Linear(dim, self.multiplier * dim, bias=not disable_bias)

    def forward(self, vec: Tensor):
        """
        Args:
            vec: Timestep/guidance embedding of shape `(batch_size, dim)` or
                `(batch_size, seq_len, dim)`.

        Returns:
            Tuple of modulation parameters. For single-stream: `(mod, None)` where
            `mod` is a 3-tuple of (shift, scale, gate). For double-stream:
            `(mod1, mod2)` where each is a 3-tuple for different streams.
        """
        out = self.lin(nn.functional.silu(vec))
        if out.ndim == 2:
            out = out[:, None, :]
        out = out.chunk(self.multiplier, dim=-1)
        return out[:3], out[3:] if self.is_double else None


class LastLayer(nn.Module):
    """
    Final output layer with adaptive layer normalization.

    This module applies AdaLN-modulated normalization followed by a linear projection
    to map transformer hidden states back to the output space (image latent channels).
    """

    def __init__(self, hidden_size: int, out_channels: int):
        """
        Args:
            hidden_size: Hidden dimension of transformer blocks.
            out_channels: Output dimension (number of latent channels).
        """
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, out_channels, bias=False)
        self.adaLN_modulation = nn.Sequential(  # pylint: disable=invalid-name
            nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size, bias=False)
        )

    def forward(self, x: Tensor, vec: Tensor) -> Tensor:
        """
        Args:
            x: Hidden states from transformer as a tensor of shape
                `(batch_size, seq_len, hidden_size)`.
            vec: Timestep embedding of shape `(batch_size, hidden_size)`.

        Returns:
            Output tensor of shape `(batch_size, seq_len, out_channels)`.
        """
        mod = self.adaLN_modulation(vec)
        shift, scale = mod.chunk(2, dim=-1)
        if shift.ndim == 2:
            shift = shift[:, None, :]
            scale = scale[:, None, :]
        x = (1 + scale) * self.norm_final(x) + shift
        x = self.linear(x)
        return x


class SingleStreamBlock(nn.Module):
    """
    Single-stream transformer block processing concatenated text+image tokens.

    This block applies self-attention and MLP operations to a unified sequence of
    text and image tokens. Both operations share pre-normalization and use adaptive
    modulation from timestep embeddings.
    """

    def __init__(self, hidden_size: int, num_heads: int, mlp_ratio: float = 4.0):
        """
        Args:
            hidden_size: Hidden dimension size for all linear projections.
            num_heads: Number of attention heads. Must divide `hidden_size` evenly.
            mlp_ratio: Ratio of MLP hidden dimension to `hidden_size`. The actual
                MLP hidden dim is `int(hidden_size * mlp_ratio)`.
        """
        super().__init__()

        self.hidden_dim = hidden_size
        self.num_heads = num_heads
        head_dim = hidden_size // num_heads
        self.scale = head_dim**-0.5
        self.mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.mlp_mult_factor = 2

        self.linear1 = nn.Linear(
            hidden_size,
            hidden_size * 3 + self.mlp_hidden_dim * self.mlp_mult_factor,
            bias=False,
        )
        self.linear2 = nn.Linear(
            hidden_size + self.mlp_hidden_dim, hidden_size, bias=False
        )
        self.norm = QKNorm(head_dim)

        self.hidden_size = hidden_size
        self.pre_norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)

        self.mlp_act = SiLUActivation()

    def forward(self, x: Tensor, pe: Tensor, mod: tuple[Tensor, Tensor]) -> Tensor:
        mod_shift, mod_scale, mod_gate = mod
        x_mod = (1 + mod_scale) * self.pre_norm(x) + mod_shift

        qkv, mlp = torch.split(
            self.linear1(x_mod),
            [3 * self.hidden_size, self.mlp_hidden_dim * self.mlp_mult_factor],
            dim=-1,
        )

        q, k, v = einops.rearrange(
            qkv, "B L (K H D) -> K B H L D", K=3, H=self.num_heads
        )
        q, k = self.norm(q, k, v)

        attn = attention(q, k, v, pe)

        # Compute activation in mlp stream, cat again and run second linear layer.
        output = self.linear2(torch.cat((attn, self.mlp_act(mlp)), 2))
        return x + mod_gate * output


class DoubleStreamBlock(nn.Module):
    """
    Dual-stream transformer block processing text and image tokens separately.

    This block maintains separate streams for text and image tokens, each with their
    own self-attention and MLP sublayers. Cross-stream information exchange happens
    through joint attention where Q, K, V from both streams are concatenated.
    """

    def __init__(self, hidden_size: int, num_heads: int, mlp_ratio: float):
        """
        Args:
            hidden_size: Hidden dimension size for all linear projections.
            num_heads: Number of attention heads. Must divide `hidden_size` evenly.
            mlp_ratio: Ratio of MLP hidden dimension to `hidden_size`.
        """
        super().__init__()
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.num_heads = num_heads
        assert hidden_size % num_heads == 0, (
            f"{hidden_size=} must be divisible by {num_heads=}"
        )

        self.hidden_size = hidden_size
        self.img_norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.mlp_mult_factor = 2

        self.img_attn = SelfAttention(dim=hidden_size, num_heads=num_heads)

        self.img_norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.img_mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden_dim * self.mlp_mult_factor, bias=False),
            SiLUActivation(),
            nn.Linear(mlp_hidden_dim, hidden_size, bias=False),
        )

        self.txt_norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.txt_attn = SelfAttention(dim=hidden_size, num_heads=num_heads)

        self.txt_norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.txt_mlp = nn.Sequential(
            nn.Linear(
                hidden_size,
                mlp_hidden_dim * self.mlp_mult_factor,
                bias=False,
            ),
            SiLUActivation(),
            nn.Linear(mlp_hidden_dim, hidden_size, bias=False),
        )

    def forward(
        self,
        img: Tensor,
        txt: Tensor,
        pe: Tensor,
        pe_ctx: Tensor,
        mod_img: tuple[Tensor, Tensor],
        mod_txt: tuple[Tensor, Tensor],
    ) -> tuple[Tensor, Tensor]:
        img_mod1, img_mod2 = mod_img
        txt_mod1, txt_mod2 = mod_txt

        img_mod1_shift, img_mod1_scale, img_mod1_gate = img_mod1
        img_mod2_shift, img_mod2_scale, img_mod2_gate = img_mod2
        txt_mod1_shift, txt_mod1_scale, txt_mod1_gate = txt_mod1
        txt_mod2_shift, txt_mod2_scale, txt_mod2_gate = txt_mod2

        # Prepare image for attention.
        img_modulated = self.img_norm1(img)
        img_modulated = (1 + img_mod1_scale) * img_modulated + img_mod1_shift

        img_qkv = self.img_attn.qkv(img_modulated)
        img_q, img_k, img_v = einops.rearrange(
            img_qkv, "B L (K H D) -> K B H L D", K=3, H=self.num_heads
        )
        img_q, img_k = self.img_attn.norm(img_q, img_k, img_v)

        # Prepare txt for attention.
        txt_modulated = self.txt_norm1(txt)
        txt_modulated = (1 + txt_mod1_scale) * txt_modulated + txt_mod1_shift

        txt_qkv = self.txt_attn.qkv(txt_modulated)
        txt_q, txt_k, txt_v = einops.rearrange(
            txt_qkv, "B L (K H D) -> K B H L D", K=3, H=self.num_heads
        )
        txt_q, txt_k = self.txt_attn.norm(txt_q, txt_k, txt_v)

        q = torch.cat((txt_q, img_q), dim=2)
        k = torch.cat((txt_k, img_k), dim=2)
        v = torch.cat((txt_v, img_v), dim=2)

        pe = torch.cat((pe_ctx, pe), dim=2)
        attn = attention(q, k, v, pe)
        txt_attn, img_attn = attn[:, : txt_q.shape[2]], attn[:, txt_q.shape[2] :]

        # Calculate the img blocks.
        img = img + img_mod1_gate * self.img_attn.proj(img_attn)
        img = img + img_mod2_gate * self.img_mlp(
            (1 + img_mod2_scale) * (self.img_norm2(img)) + img_mod2_shift
        )

        # Calculate the txt blocks.
        txt = txt + txt_mod1_gate * self.txt_attn.proj(txt_attn)
        txt = txt + txt_mod2_gate * self.txt_mlp(
            (1 + txt_mod2_scale) * (self.txt_norm2(txt)) + txt_mod2_shift
        )
        return img, txt


class MLPEmbedder(nn.Module):
    """
    Two-layer MLP for embedding timestep and guidance values.

    This simple MLP transforms scalar timestep or guidance embeddings (after
    sinusoidal encoding) into the transformer's hidden dimension space.
    """

    def __init__(self, in_dim: int, hidden_dim: int, disable_bias: bool = False):
        """
        Args:
            in_dim: Input dimension (typically 256 for sinusoidal embeddings).
            hidden_dim: Output hidden dimension matching transformer blocks.
            disable_bias: If True, linear layers have no bias terms.
        """
        super().__init__()
        self.in_layer = nn.Linear(in_dim, hidden_dim, bias=not disable_bias)
        self.silu = nn.SiLU()
        self.out_layer = nn.Linear(hidden_dim, hidden_dim, bias=not disable_bias)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Input embeddings of shape `(batch_size, in_dim)`.

        Returns:
            Projected embeddings of shape `(batch_size, hidden_dim)`.
        """
        return self.out_layer(self.silu(self.in_layer(x)))


class EmbedND(nn.Module):
    """
    N-dimensional rotary position embeddings (RoPE) for spatial-temporal tokens.

    This module creates rotary embeddings for multi-dimensional position indices
    (e.g., time, height, width, sequence). Each dimension gets its own embedding
    component with configurable dimensions.
    """

    def __init__(self, dim: int, theta: int, axes_dim: list[int]):
        """
        Args:
            dim: Total position embedding dimension. Should equal `sum(axes_dim)`.
            theta: Base frequency for RoPE. Higher values give slower rotation.
            axes_dim: Dimension allocation for each position axis.
        """
        super().__init__()
        self.dim = dim
        self.theta = theta
        self.axes_dim = axes_dim

    def forward(self, ids: Tensor) -> Tensor:
        """
        Args:
            ids: Position indices of shape `(..., num_axes)` where `num_axes`
                matches `len(axes_dim)`.

        Returns:
            Rotary embeddings of shape `(..., 1, sum(axes_dim), 2, 2)` suitable
            for applying rotation to query and key tensors.
        """
        emb = torch.cat(
            [
                rope(ids[..., i], self.axes_dim[i], self.theta)
                for i in range(len(self.axes_dim))
            ],
            dim=-3,
        )

        return emb.unsqueeze(1)


def timestep_embedding(
    t: Tensor, dim: int, max_period: int = 10000, time_factor: float = 1000.0
) -> Tensor:
    """
    Create sinusoidal timestep embeddings.

    Args:
        t: A 1-D tensor of N indices, one per batch element. These may be
            fractional.
        dim: The dimension of the output.
        max_period: Controls the minimum frequency of the embeddings.
        time_factor: Scale applied to `t` before the sinusoidal encoding.

    Returns:
        An (N, D) tensor of positional embeddings.
    """
    t = time_factor * t
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period)
        * torch.arange(start=0, end=half, device=t.device, dtype=torch.float32)
        / half
    )

    args = t[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    if torch.is_floating_point(t):
        embedding = embedding.to(t)
    return embedding


class RMSNorm(torch.nn.Module):
    """
    Root Mean Square Layer Normalization.

    RMSNorm normalizes using only the variance (RMS) without centering by mean.
    """

    def __init__(self, dim: int):
        """
        Args:
            dim: Dimension to normalize over (last dimension of input).
        """
        super().__init__()
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x: Tensor):
        """
        Args:
            x: Input tensor of shape `(..., dim)`.

        Returns:
            Normalized tensor of same shape as input.
        """
        x_dtype = x.dtype
        x = x.float()
        rrms = torch.rsqrt(torch.mean(x**2, dim=-1, keepdim=True) + 1e-6)
        return (x * rrms).to(dtype=x_dtype) * self.scale


class QKNorm(torch.nn.Module):
    """
    Separate RMSNorm for query and key tensors in attention.

    Normalizing queries and keys independently before attention computation
    improves training stability.
    """

    def __init__(self, dim: int):
        """
        Args:
            dim: Head dimension for queries and keys.
        """
        super().__init__()
        self.query_norm = RMSNorm(dim)
        self.key_norm = RMSNorm(dim)

    def forward(self, q: Tensor, k: Tensor, v: Tensor) -> tuple[Tensor, Tensor]:
        """
        Args:
            q: Query tensor of shape `(..., head_dim)`.
            k: Key tensor of shape `(..., head_dim)`.
            v: Value tensor (used only for dtype matching).

        Returns:
            Tuple of normalized `(query, key)` tensors with dtype matching `v`.
        """
        q = self.query_norm(q)
        k = self.key_norm(k)
        return q.to(v), k.to(v)


def attention(q: Tensor, k: Tensor, v: Tensor, pe: Tensor) -> Tensor:
    q, k = apply_rope(q, k, pe)

    x = torch.nn.functional.scaled_dot_product_attention(q, k, v)
    x = einops.rearrange(x, "B H L D -> B L (H D)")

    return x


def rope(pos: Tensor, dim: int, theta: int) -> Tensor:
    assert dim % 2 == 0
    scale = torch.arange(0, dim, 2, dtype=pos.dtype, device=pos.device) / dim
    omega = 1.0 / (theta**scale)
    out = torch.einsum("...n,d->...nd", pos, omega)
    out = torch.stack(
        [torch.cos(out), -torch.sin(out), torch.sin(out), torch.cos(out)], dim=-1
    )
    out = einops.rearrange(out, "b n d (i j) -> b n d i j", i=2, j=2)
    return out.float()


def apply_rope(xq: Tensor, xk: Tensor, freqs_cis: Tensor) -> tuple[Tensor, Tensor]:
    xq_ = xq.float().reshape(*xq.shape[:-1], -1, 1, 2)
    xk_ = xk.float().reshape(*xk.shape[:-1], -1, 1, 2)
    xq_out = freqs_cis[..., 0] * xq_[..., 0] + freqs_cis[..., 1] * xq_[..., 1]
    xk_out = freqs_cis[..., 0] * xk_[..., 0] + freqs_cis[..., 1] * xk_[..., 1]
    return xq_out.reshape(*xq.shape).type_as(xq), xk_out.reshape(*xk.shape).type_as(xk)
