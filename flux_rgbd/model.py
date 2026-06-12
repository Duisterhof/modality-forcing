# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 World Labs.
"""FluxRGBD -- pairs the depth-extended DiT with a joint flow-matching sampler.

Public surface:
  * `model.forward(...)` -- one denoiser call.
  * `model.sample(...)`  -- N-step Euler rollout over RGB + depth. Three
    modes: `"joint"` (text -> RGBD), `"i2d"` (text+RGB -> depth),
    `"d2i"` (text+depth -> RGB).
"""

from __future__ import annotations

import einops
import torch
from torch import Tensor, nn

from flux_rgbd.depth.schedule import Mode, ScheduleConfig, rollout_timesteps
from flux_rgbd.dit import FluxRGBDDiT


class FluxRGBD(nn.Module):
    def __init__(self, dit: FluxRGBDDiT) -> None:
        super().__init__()
        self.dit = dit
        self.in_channels = dit.in_channels
        self.depth_channels = dit.depth_channels

    # ------------------------------------------------------------------ ids
    # FLUX.2's 4D position ids: (time_axis, row, col, seq_idx).
    # Image tokens use (0, row, col, 0); depth tokens use (1, row, col, 0)
    # so RoPE can distinguish them. Text tokens use (0, 0, 0, seq_idx).

    @staticmethod
    def _img_ids(batch: int, h: int, w: int, *, device, time_id: float = 0.0) -> Tensor:
        ids = torch.zeros(h, w, 4, device=device)
        ids[..., 0] = time_id
        ids[..., 1] = torch.arange(h, device=device)[:, None]
        ids[..., 2] = torch.arange(w, device=device)[None, :]
        return einops.repeat(ids, "h w d -> b (h w) d", b=batch)

    @staticmethod
    def _txt_ids(batch: int, seq_len: int, *, device) -> Tensor:
        ids = torch.zeros(batch, seq_len, 4, device=device)
        ids[..., 3] = torch.arange(seq_len, device=device)
        return ids

    # ----------------------------------------------------------------- step

    def forward(
        self,
        img,
        timesteps,
        ctx,
        img_height,
        img_width,
        depth,
        depth_timesteps,
        guidance=None,
    ):
        b, device = img.shape[0], img.device
        return self.dit(
            img=img,
            img_ids=self._img_ids(b, img_height, img_width, device=device),
            timesteps=timesteps,
            ctx=ctx,
            ctx_ids=self._txt_ids(b, ctx.shape[1], device=device),
            depth=depth,
            depth_ids=self._img_ids(
                b, img_height, img_width, device=device, time_id=1.0
            ),
            depth_timesteps=depth_timesteps,
            guidance=guidance,
        )

    # --------------------------------------------------------------- sample

    @torch.no_grad()
    def sample(
        self,
        *,
        ctx: Tensor,
        img_height: int,
        img_width: int,
        num_steps: int = 50,
        mode: Mode = "joint",
        schedule_config: ScheduleConfig | None = None,
        cfg_scale: float = 1.0,
        guidance: float = 1.0,
        seed: int | None = None,
        rgb_use_x_prediction: bool = False,
        depth_use_x_prediction: bool = True,
        x_prediction_t_min: float = 0.05,
        clean_rgb: Tensor | None = None,
        clean_depth: Tensor | None = None,
        null_text_embed: Tensor | None = None,
        log2_alpha: float | None = None,
    ) -> tuple[Tensor, Tensor]:
        """N-step Euler rollout; returns the final (rgb, depth) latent tokens.

        `mode` picks what is denoised: "joint" denoises both, "i2d" freezes
        `clean_rgb` and denoises depth, "d2i" freezes `clean_depth` and
        denoises RGB. CFG runs when `cfg_scale` > 1.0, against
        `null_text_embed` (zeros_like(ctx) if omitted). `*_use_x_prediction`
        converts x-prediction outputs to velocities before each Euler step.
        """
        if mode == "i2d" and clean_rgb is None:
            raise ValueError("clean_rgb is required for mode='i2d'")
        if mode == "d2i" and clean_depth is None:
            raise ValueError("clean_depth is required for mode='d2i'")

        device, dtype, batch = ctx.device, ctx.dtype, ctx.shape[0]
        num_tokens = img_height * img_width

        gen = (
            torch.Generator(device=device).manual_seed(seed)
            if seed is not None
            else None
        )

        def randn(*shape):
            return torch.randn(*shape, device=device, dtype=dtype, generator=gen)

        if mode == "i2d":
            rgb = clean_rgb.to(dtype=dtype, device=device)
            depth = randn(batch, num_tokens, self.depth_channels)
        elif mode == "d2i":
            rgb = randn(batch, num_tokens, self.in_channels)
            depth = clean_depth.to(dtype=dtype, device=device)
        else:  # joint
            rgb = randn(batch, num_tokens, self.in_channels)
            depth = randn(batch, num_tokens, self.depth_channels)

        if schedule_config is None:
            schedule_config = ScheduleConfig()
        t_rgb_sched, t_depth_sched = rollout_timesteps(
            schedule_config,
            num_steps,
            mode=mode,
            log2_alpha=log2_alpha,
            device=device,
            dtype=dtype,
        )

        guidance_tensor = (
            torch.full((batch,), guidance, device=device, dtype=dtype)
            if self.dit.use_guidance_embed
            else None
        )

        use_cfg = cfg_scale > 1.0
        if use_cfg:
            uncond = (
                null_text_embed
                if null_text_embed is not None
                else torch.zeros_like(ctx)
            )
            if uncond.ndim == 2:
                uncond = uncond.unsqueeze(0)
            if uncond.shape[0] == 1 and batch > 1:
                uncond = uncond.expand(batch, *uncond.shape[1:])
            uncond = uncond.to(device=device, dtype=dtype)

        for step in range(num_steps):
            t_rgb_b = t_rgb_sched[step].expand(batch)
            t_depth_b = t_depth_sched[step].expand(batch)
            dt_rgb = t_rgb_sched[step + 1] - t_rgb_sched[step]
            dt_depth = t_depth_sched[step + 1] - t_depth_sched[step]

            pred_rgb, pred_depth = self.forward(
                img=rgb,
                timesteps=t_rgb_b,
                ctx=ctx,
                img_height=img_height,
                img_width=img_width,
                depth=depth,
                depth_timesteps=t_depth_b,
                guidance=guidance_tensor,
            )
            if use_cfg:
                # The uncond CUDA-graph replay below overwrites these output
                # buffers (torch.compile reduce-overhead) -- clone before reuse.
                pred_rgb, pred_depth = pred_rgb.clone(), pred_depth.clone()
                pred_rgb_u, pred_depth_u = self.forward(
                    img=rgb,
                    timesteps=t_rgb_b,
                    ctx=uncond,
                    img_height=img_height,
                    img_width=img_width,
                    depth=depth,
                    depth_timesteps=t_depth_b,
                    guidance=guidance_tensor,
                )
                pred_rgb = pred_rgb_u + cfg_scale * (pred_rgb - pred_rgb_u)
                pred_depth = pred_depth_u + cfg_scale * (pred_depth - pred_depth_u)

            # x-prediction -> velocity: when the head was trained to predict
            # the clean x rather than the velocity, convert it before Euler.
            if rgb_use_x_prediction:
                clamp = t_rgb_b.view(batch, 1, 1).clamp(min=x_prediction_t_min)
                pred_rgb = (rgb - pred_rgb) / clamp
            if depth_use_x_prediction:
                clamp = t_depth_b.view(batch, 1, 1).clamp(min=x_prediction_t_min)
                pred_depth = (depth - pred_depth) / clamp

            # Euler step. dt=0 for the frozen modality in i2d/d2i is a no-op.
            rgb = rgb + dt_rgb * pred_rgb
            depth = depth + dt_depth * pred_depth

        return rgb, depth
