# SPDX-License-Identifier: Apache-2.0
# Copyright 2025 Black Forest Labs.
# Copyright (c) 2026 World Labs. Modifications: depth stream, joint RGBD
# attention, depth decoder blocks, cross-stream timestep mixing.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""FluxRGBDDiT — FLUX.2 dual+single-stream transformer extended for joint RGB+depth.

Derived from the FLUX.2 reference implementation:
    https://github.com/black-forest-labs/flux2

Architecture, inference-only:
  * Depth is a peer stream alongside image and text — each gets its own
    pre-norm, Q/K/V, MLP inside every dual-stream block. Joint attention
    runs over the concatenated `[txt, img, depth]` sequence.
  * 8 dual-stream blocks + 24 single-stream blocks (FLUX.2 backbone).
  * 4-block depth-only decoder before the depth output head.
  * 4D RoPE position ids (axes_dim=(32,32,32,32), theta=2000) shared via
    EmbedND. RGB tokens use time_id=0, depth uses time_id=1 so RoPE can
    distinguish them.

Reuses the FLUX.2 reference building blocks unchanged so img/txt-side
weights lifted from an RGB-only FLUX.2 [klein-9B] base load 1:1.
"""

from __future__ import annotations

import einops
import torch
from torch import Tensor, nn

from flux_rgbd._flux2 import model as flux2_model
from flux_rgbd._flux2.model import (
    EmbedND,
    LastLayer,
    MLPEmbedder,
    Modulation,
    SelfAttention,
    SiLUActivation,
    SingleStreamBlock,
    timestep_embedding,
)


class _TripleStreamBlock(nn.Module):
    """Joint-attention block over (img, text, depth) with per-stream AdaLN.

    Submodule names on the img/txt halves match FLUX.2's DoubleStreamBlock
    so RGB-only base weights load 1:1; the depth half adds new submodules
    with the same shapes as img.
    """

    def __init__(self, hidden_size: int, num_heads: int, mlp_ratio: float):
        super().__init__()
        if hidden_size % num_heads != 0:
            raise ValueError(f"hidden_size {hidden_size} not divisible by num_heads {num_heads}")
        mlp_hidden = int(hidden_size * mlp_ratio)
        self.num_heads = num_heads

        # Same shapes per stream: pre-norm + (Q,K,V,proj) + post-norm + 2-up SiLU MLP.
        # The `* 2` in the first Linear is the gated-SiLU activation packing FLUX.2 uses.
        for prefix in ("img", "txt", "depth"):
            setattr(self, f"{prefix}_norm1", nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6))
            setattr(self, f"{prefix}_attn", SelfAttention(dim=hidden_size, num_heads=num_heads))
            setattr(self, f"{prefix}_norm2", nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6))
            setattr(self, f"{prefix}_mlp", nn.Sequential(
                nn.Linear(hidden_size, mlp_hidden * 2, bias=False),
                SiLUActivation(),
                nn.Linear(mlp_hidden, hidden_size, bias=False),
            ))

    def _qkv(self, x: Tensor, norm: nn.LayerNorm, attn: SelfAttention,
             mod1: tuple[Tensor, Tensor, Tensor]):
        """Pre-norm + AdaLN + Q/K/V projection. Returns (q, k, v, gate)."""
        shift, scale, gate = mod1
        qkv = attn.qkv((1 + scale) * norm(x) + shift)
        q, k, v = einops.rearrange(qkv, "B L (K H D) -> K B H L D",
                                   K=3, H=self.num_heads)
        q, k = attn.norm(q, k, v)
        return q, k, v, gate

    @staticmethod
    def _residual(x, attn_out, proj, gate1, mod2, norm2, mlp):
        """attn residual then MLP residual, both gated. Standard DiT pattern."""
        x = x + gate1 * proj(attn_out)
        shift, scale, gate2 = mod2
        return x + gate2 * mlp((1 + scale) * norm2(x) + shift)

    def forward(self, img, txt, depth, pe_img, pe_txt, pe_depth,
                mod_img, mod_txt, mod_depth):
        q_img, k_img, v_img, g_img = self._qkv(img,   self.img_norm1,   self.img_attn,   mod_img[0])
        q_txt, k_txt, v_txt, g_txt = self._qkv(txt,   self.txt_norm1,   self.txt_attn,   mod_txt[0])
        q_d,   k_d,   v_d,   g_d   = self._qkv(depth, self.depth_norm1, self.depth_attn, mod_depth[0])

        # Joint attention over [txt, img, depth]; BFL's `attention` does
        # RoPE + scaled_dot_product_attention + rearrange.
        q = torch.cat([q_txt, q_img, q_d], dim=2)
        k = torch.cat([k_txt, k_img, k_d], dim=2)
        v = torch.cat([v_txt, v_img, v_d], dim=2)
        pe = torch.cat([pe_txt, pe_img, pe_depth], dim=2)
        out = flux2_model.attention(q, k, v, pe)

        n_txt, n_img = q_txt.shape[2], q_img.shape[2]
        txt_out = out[:, :n_txt]
        img_out = out[:, n_txt : n_txt + n_img]
        depth_out = out[:, n_txt + n_img :]

        img = self._residual(img, img_out, self.img_attn.proj, g_img,
                             mod_img[1], self.img_norm2, self.img_mlp)
        txt = self._residual(txt, txt_out, self.txt_attn.proj, g_txt,
                             mod_txt[1], self.txt_norm2, self.txt_mlp)
        depth = self._residual(depth, depth_out, self.depth_attn.proj, g_d,
                               mod_depth[1], self.depth_norm2, self.depth_mlp)
        return img, txt, depth


def _stack_per_token_mod(
    mod_img: tuple[Tensor, Tensor, Tensor],
    mod_depth: tuple[Tensor, Tensor, Tensor],
    n_txt: int, n_img: int, n_depth: int,
) -> tuple[Tensor, Tensor, Tensor]:
    """Build the per-token modulation triple consumed by SingleStreamBlock.

    Each ``Modulation`` output is broadcast over the sequence axis. For the
    triple-stream single stack we need different (shift, scale, gate) for
    the depth tokens vs the txt+img tokens — so we expand each entry along
    seq and ``cat``.
    """
    n_txt_img = n_txt + n_img

    def _build(slot: int) -> Tensor:
        head = mod_img[slot].expand(-1, n_txt_img, -1)
        tail = mod_depth[slot].expand(-1, n_depth, -1)
        return torch.cat([head, tail], dim=1)

    return _build(0), _build(1), _build(2)


class FluxRGBDDiT(nn.Module):
    """FLUX.2 Klein-9B DiT with peer depth stream.

    Forward: (img, depth, text, RGB timestep, depth timestep, *_ids)
    → (rgb_latent_out, depth_latent_out).

    Architecture (v1 defaults match the published checkpoint):
        depth_double=8 triple-stream blocks  →  full-token concat  →
        depth_single=24 single-stream blocks  →  split  →
        depth_decoder_num_layers=4 single blocks over depth only  →
        final_layer / depth_final_layer.
    """

    def __init__(
        self,
        *,
        in_channels: int = 128,
        depth_channels: int = 256,
        context_in_dim: int = 12288,
        hidden_size: int = 4096,
        num_heads: int = 32,
        depth_double: int = 8,
        depth_single: int = 24,
        depth_decoder_num_layers: int = 4,
        axes_dim: tuple[int, ...] = (32, 32, 32, 32),
        theta: int = 2000,
        mlp_ratio: float = 3.0,
        use_guidance_embed: bool = False,
        cross_stream_timestep_mixing: bool = False,
    ):
        super().__init__()

        pe_dim = hidden_size // num_heads
        if sum(axes_dim) != pe_dim:
            raise ValueError(
                f"axes_dim sum {sum(axes_dim)} != hidden_size/num_heads {pe_dim}"
            )

        self.in_channels = in_channels
        self.depth_channels = depth_channels
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.depth_double = depth_double
        self.depth_single = depth_single
        self.depth_decoder_num_layers = depth_decoder_num_layers

        # Stream-in projections.
        self.img_in = nn.Linear(in_channels, hidden_size, bias=False)
        self.txt_in = nn.Linear(context_in_dim, hidden_size, bias=False)
        self.depth_in = nn.Linear(depth_channels, hidden_size, bias=False)

        # Timestep embedders. `time_in` carries the RGB timestep (plus optional
        # guidance scale); `time_in_depth` carries the depth timestep.
        self.time_in = MLPEmbedder(
            in_dim=256, hidden_dim=hidden_size, disable_bias=True
        )
        self.time_in_depth = MLPEmbedder(
            in_dim=256, hidden_dim=hidden_size, disable_bias=True
        )
        self.use_guidance_embed = use_guidance_embed
        if use_guidance_embed:
            self.guidance_in = MLPEmbedder(
                in_dim=256, hidden_dim=hidden_size, disable_bias=True
            )

        # Cross-stream timestep mixing: each stream's modulation vector also
        # sees the *other* stream's noise level, projected through a small MLP
        # and gated by a learnable scalar (trained from a zero init, so the
        # path is identity at step 0 and a learned delta thereafter):
        #     vec_img   += cross_alpha_img   * time_in_depth_to_img(emb(t_depth))
        #     vec_depth += cross_alpha_depth * time_in_rgb_to_depth(emb(t_rgb))
        # This lets the var-AR checkpoint condition each modality on how noisy
        # the other one is (e.g. clean RGB in i2d, clean depth in d2i).
        self.cross_stream_timestep_mixing = cross_stream_timestep_mixing
        if cross_stream_timestep_mixing:
            self.time_in_depth_to_img = MLPEmbedder(
                in_dim=256, hidden_dim=hidden_size, disable_bias=True
            )
            self.time_in_rgb_to_depth = MLPEmbedder(
                in_dim=256, hidden_dim=hidden_size, disable_bias=True
            )
            # Shape (1,) not () to mirror the training checkpoint's parameter
            # layout (kept 1-D for FSDP2's no-scalar-parameter rule).
            self.cross_alpha_img = nn.Parameter(torch.zeros(1))
            self.cross_alpha_depth = nn.Parameter(torch.zeros(1))

        # Position embedder shared across streams.
        self.pe_embedder = EmbedND(dim=pe_dim, theta=theta, axes_dim=axes_dim)

        # Dual-stream stack.
        self.double_blocks = nn.ModuleList(
            [_TripleStreamBlock(hidden_size, num_heads, mlp_ratio)
             for _ in range(depth_double)]
        )

        # Single-stream stack — reuses the FLUX.2 SingleStreamBlock with
        # per-token modulation built on the fly.
        self.single_blocks = nn.ModuleList(
            [SingleStreamBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio)
             for _ in range(depth_single)]
        )

        # Modulation heads.
        self.double_stream_modulation_img = Modulation(
            hidden_size, double=True, disable_bias=True
        )
        self.double_stream_modulation_txt = Modulation(
            hidden_size, double=True, disable_bias=True
        )
        self.double_stream_modulation_depth = Modulation(
            hidden_size, double=True, disable_bias=True
        )
        self.single_stream_modulation = Modulation(
            hidden_size, double=False, disable_bias=True
        )
        self.single_stream_modulation_depth = Modulation(
            hidden_size, double=False, disable_bias=True
        )

        # Optional depth-only decoder stack (4 SingleStreamBlocks for v1).
        if depth_decoder_num_layers > 0:
            self.depth_decoder_blocks = nn.ModuleList(
                [SingleStreamBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio)
                 for _ in range(depth_decoder_num_layers)]
            )
            self.depth_decoder_modulation = Modulation(
                hidden_size, double=False, disable_bias=True
            )
        else:
            self.depth_decoder_blocks = nn.ModuleList()
            self.depth_decoder_modulation = None

        # Final unpatchifiers.
        self.final_layer = LastLayer(hidden_size, in_channels)
        self.depth_final_layer = LastLayer(hidden_size, depth_channels)

    def forward(
        self,
        img: Tensor,
        img_ids: Tensor,
        timesteps: Tensor,
        ctx: Tensor,
        ctx_ids: Tensor,
        depth: Tensor,
        depth_ids: Tensor,
        depth_timesteps: Tensor,
        guidance: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Triple-stream forward.

        Args:
            img:       `(B, n_img, in_channels)`        RGB latent tokens
            img_ids:   `(B, n_img, 4)`                  4D position ids
            timesteps: `(B,)`                           RGB diffusion time
            ctx:       `(B, n_txt, context_in_dim)`     text embeddings
            ctx_ids:   `(B, n_txt, 4)`                  text position ids
            depth:     `(B, n_depth, depth_channels)`   depth latent tokens
            depth_ids: `(B, n_depth, 4)`                depth position ids
            depth_timesteps: `(B,)`                     depth diffusion time
            guidance:  `(B,)`                           optional guidance scale

        Returns:
            `(rgb_out, depth_out)` patches.
        """
        # Body dtype (set by the parameters of the input projections). Inputs
        # are cast to this so mixed-precision callers (BF16 body + FP32
        # sampling state) work transparently.
        body_dtype = self.img_in.weight.dtype
        img = img.to(body_dtype)
        ctx = ctx.to(body_dtype)
        depth = depth.to(body_dtype)
        timesteps = timesteps.to(body_dtype)
        depth_timesteps = depth_timesteps.to(body_dtype)
        if guidance is not None:
            guidance = guidance.to(body_dtype)

        # Timestep embeddings (separate per modality). `timestep_embedding`
        # always returns FP32; cast before the BF16 MLPEmbedder weights.
        rgb_t_emb = timestep_embedding(timesteps, 256).to(body_dtype)
        depth_t_emb = timestep_embedding(depth_timesteps, 256).to(body_dtype)
        vec_img = self.time_in(rgb_t_emb)
        if self.use_guidance_embed and guidance is not None:
            vec_img = vec_img + self.guidance_in(
                timestep_embedding(guidance, 256).to(body_dtype)
            )
        vec_depth = self.time_in_depth(depth_t_emb)

        # Cross-stream timestep mixing (see __init__): fold the other stream's
        # noise level into each modulation vector. The scalar gate is part of
        # the body dtype after `.to(dtype=…)`, so no extra cast is needed.
        if self.cross_stream_timestep_mixing:
            vec_img = vec_img + self.cross_alpha_img * self.time_in_depth_to_img(
                depth_t_emb
            )
            vec_depth = vec_depth + self.cross_alpha_depth * self.time_in_rgb_to_depth(
                rgb_t_emb
            )

        # Per-stream modulation coefficients.
        mod_img_double = self.double_stream_modulation_img(vec_img)
        mod_txt_double = self.double_stream_modulation_txt(vec_img)
        mod_depth_double = self.double_stream_modulation_depth(vec_depth)
        mod_img_single, _ = self.single_stream_modulation(vec_img)
        mod_depth_single, _ = self.single_stream_modulation_depth(vec_depth)

        # Project each stream into the hidden dim.
        img = self.img_in(img)
        txt = self.txt_in(ctx)
        depth = self.depth_in(depth)

        # Position embeddings.
        pe_img = self.pe_embedder(img_ids)
        pe_txt = self.pe_embedder(ctx_ids)
        pe_depth = self.pe_embedder(depth_ids)

        # Dual-stream stack.
        for block in self.double_blocks:
            img, txt, depth = block(
                img, txt, depth,
                pe_img, pe_txt, pe_depth,
                mod_img_double, mod_txt_double, mod_depth_double,
            )

        # Single-stream stack over the concatenated [txt, img, depth] sequence
        # with per-token modulation.
        n_txt = txt.shape[1]
        n_img = img.shape[1]
        n_depth = depth.shape[1]
        joint = torch.cat([txt, img, depth], dim=1)
        pe_joint = torch.cat([pe_txt, pe_img, pe_depth], dim=2)
        per_token_mod = _stack_per_token_mod(
            mod_img_single, mod_depth_single, n_txt, n_img, n_depth
        )
        for block in self.single_blocks:
            joint = block(joint, pe_joint, per_token_mod)

        # Split back per stream.
        rgb_tokens = joint[:, n_txt : n_txt + n_img]
        depth_tokens = joint[:, n_txt + n_img :]

        # Optional depth-only decoder.
        if self.depth_decoder_num_layers > 0:
            depth_decoder_mod, _ = self.depth_decoder_modulation(vec_depth)
            for block in self.depth_decoder_blocks:
                depth_tokens = block(depth_tokens, pe_depth, depth_decoder_mod)

        # Each output head may be in a different dtype than the body
        # (e.g. FP32 depth head over a BF16 body); cast inputs to the
        # head's own dtype at the boundary.
        rgb_head_dtype = self.final_layer.linear.weight.dtype
        depth_head_dtype = self.depth_final_layer.linear.weight.dtype
        rgb_out = self.final_layer(
            rgb_tokens.to(rgb_head_dtype), vec_img.to(rgb_head_dtype)
        )
        depth_out = self.depth_final_layer(
            depth_tokens.to(depth_head_dtype), vec_depth.to(depth_head_dtype)
        )
        return rgb_out, depth_out
