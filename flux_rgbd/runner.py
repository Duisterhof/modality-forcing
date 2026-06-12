# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 World Labs.
"""Public inference API: FluxRGBDRunner.

Bundles the diffusion model, the FLUX.2 VAE decoder, and the Qwen3
text encoder so callers can go from a text prompt to a numpy RGB image
plus a depth map in one call.
"""

from __future__ import annotations

import datetime as _dt
import json
import re
import sys
from pathlib import Path

import cv2
import einops
import numpy as np
import torch
from matplotlib import cm
from safetensors.torch import load_file as load_safetensors
from torch import Tensor

from flux_rgbd._flux2.autoencoder import Flux2Decoder, Flux2Encoder
from flux_rgbd.depth.preprocess import decode_depth, encode_depth
from flux_rgbd.depth.schedule import Mode
from flux_rgbd.factories import VARIANTS
from flux_rgbd.text_encoder import Qwen3Embedder


def _load_local_or_hub(repo_id_or_path: str) -> tuple[Path, Path]:
    """Resolve `config.json` and `model.safetensors` to local paths.

    Accepts a local directory or a HuggingFace repo id.
    """
    local = Path(repo_id_or_path)
    if local.is_dir():
        return local / "config.json", local / "model.safetensors"
    from huggingface_hub import hf_hub_download
    return (
        Path(hf_hub_download(repo_id=repo_id_or_path, filename="config.json")),
        Path(hf_hub_download(repo_id=repo_id_or_path, filename="model.safetensors")),
    )


def _materialize_meta_tensors(model: torch.nn.Module, device) -> list[str]:
    """Give a real (uninitialized) tensor to any parameter/buffer still on the
    meta device after an ``assign=True`` load.

    A meta-device build defers allocation until weights are assigned, so any
    tensor the checkpoint does *not* supply remains on `meta` and would crash
    at first use. A complete checkpoint leaves none — this is a safety net for
    drift between the model definition and the checkpoint (returns the names it
    had to materialize so callers can surface the gap).
    """
    materialized: list[str] = []
    for name, tensor in [*model.named_parameters(), *model.named_buffers()]:
        if not tensor.is_meta:
            continue
        *parents, leaf = name.split(".")
        owner = model
        for part in parents:
            owner = getattr(owner, part)
        real = torch.empty(tensor.shape, dtype=tensor.dtype, device=device)
        if isinstance(tensor, torch.nn.Parameter):
            owner._parameters[leaf] = torch.nn.Parameter(real, requires_grad=False)
        else:
            owner._buffers[leaf] = real
        materialized.append(name)
    return materialized


class FluxRGBDRunner:
    """High-level inference wrapper."""

    def __init__(self, model, decoder, embedder, depth_config, schedule_config,
                 *, device="cuda", img_hw=(512, 512), latent_compression=16):
        self.model = model
        self.decoder = decoder
        self.embedder = embedder
        self.depth_config = depth_config
        self.schedule_config = schedule_config
        self.device = torch.device(device)
        self.img_hw = img_hw
        self.latent_compression = latent_compression
        self.latent_hw = (img_hw[0] // latent_compression, img_hw[1] // latent_compression)
        self._encoder: Flux2Encoder | None = None  # lazy: only needed for i2d
        self._null_text_embed: Tensor | None = None  # lazy CFG uncond, see _null_embed

    @classmethod
    def from_pretrained(cls, repo_id_or_path, *, device="cuda",
                        dtype=torch.float32, head_dtype: torch.dtype | None = None,
                        text_encoder="Qwen/Qwen3-8B",
                        img_hw=(512, 512), latent_compression=16,
                        compile_model: bool = False):
        """Load model + decoder + embedder. Path or HuggingFace Hub repo id.

        ``head_dtype`` (optional) overrides the dtype of the depth output head.
        Setting ``dtype=torch.bfloat16, head_dtype=torch.float32`` runs the
        DiT body in BF16 (fast) but keeps the depth final layer in FP32,
        which avoids BF16 quantization artifacts under x-prediction.

        ``compile_model`` torch.compiles the DiT with CUDA graphs
        (``mode="reduce-overhead"``). The first forward pays the compile cost
        (minutes on a fresh machine); kernels are cached on disk so later
        processes warm-start in seconds. CUDA only; not supported on ZeroGPU
        Spaces (fresh process per call).
        """
        config_path, weights_path = _load_local_or_hub(repo_id_or_path)
        config = json.loads(config_path.read_text())
        builder = VARIANTS[config["variant"]]
        # Build on the meta device, then assign the checkpoint tensors straight
        # onto `device`. This skips ~45 s of random initialization for the 9B
        # model (every parameter is overwritten by the checkpoint anyway) and
        # the H2D copy of a throwaway randinit model — cutting DiT load from
        # ~60 s to ~4 s. Bit-identical to the eager construct+load path.
        with torch.device("meta"):
            model, depth_cfg, schedule_cfg = builder()
        state_dict = load_safetensors(str(weights_path), device=str(device))
        if dtype is not None:
            state_dict = {k: v.to(dtype) for k, v in state_dict.items()}
        model.load_state_dict(state_dict, strict=False, assign=True)
        leftover = _materialize_meta_tensors(model, device)
        if leftover:
            raise RuntimeError(
                f"checkpoint is missing {len(leftover)} tensors (e.g. "
                f"{leftover[:4]}); refusing to run with uninitialized weights. "
                f"Re-download the checkpoint or check --model.")
        model.eval()
        if head_dtype is not None and head_dtype != dtype:
            model.dit.depth_final_layer.to(head_dtype)
        if compile_model and torch.device(device).type != "cuda":
            print(f"[compile] torch.compile (reduce-overhead) needs CUDA; "
                  f"ignoring it on device {device}.", file=sys.stderr)
            compile_model = False
        if compile_model:
            # Compile the inner DiT: the sampling loop calls model.forward()
            # directly, bypassing the __call__ hook that nn.Module.compile()
            # wraps; self.dit(...) is reached through __call__. Shapes are
            # static (fixed text padding + token grid), so CUDA graphs hold.
            model.dit.compile(mode="reduce-overhead")

        decoder = Flux2Decoder().to(device).eval()
        decoder.load_weights()
        embedder = Qwen3Embedder(model_spec=text_encoder, device=device).eval()
        return cls(model, decoder, embedder, depth_cfg, schedule_cfg,
                   device=device, img_hw=img_hw,
                   latent_compression=latent_compression)

    def _ensure_encoder(self) -> Flux2Encoder:
        if self._encoder is None:
            enc = Flux2Encoder(filename="ae_encoder.safetensors").to(self.device).eval()
            enc.load_weights()
            self._encoder = enc
        return self._encoder

    @torch.no_grad()
    def encode_image(self, image_chw_uint8: np.ndarray) -> Tensor:
        """uint8 (H, W, 3) image → latent tokens (1, num_tokens, in_channels).

        Resizes to ``self.img_hw`` and normalises to [-1, 1] before encoding.
        Output dtype matches the diffusion model's parameter dtype.
        """
        h, w = self.img_hw
        rgb = cv2.resize(image_chw_uint8, (w, h), interpolation=cv2.INTER_AREA)
        x = torch.from_numpy(rgb).to(self.device).float() / 255.0  # (H, W, 3)
        x = x * 2 - 1  # [-1, 1]
        x = x.unsqueeze(0)  # (1, H, W, 3)
        encoder = self._ensure_encoder()
        latents_nhwc = encoder(x)  # (1, lat_h, lat_w, in_channels)
        b, lh, lw, c = latents_nhwc.shape
        # Match the model body dtype so the DiT can consume directly.
        body_dtype = next(self.model.dit.img_in.parameters()).dtype
        return latents_nhwc.reshape(b, lh * lw, c).to(body_dtype)

    @torch.no_grad()
    def encode_depth_map(self, depth_hw: np.ndarray) -> Tensor:
        """(H, W) depth map → depth-stream tokens (1, num_tokens, depth_channels).

        For ``mode="d2i"``. Resizes to ``self.img_hw`` then patchifies via the
        same normalisation the model was trained with (see ``encode_depth``).
        Because that normalisation is scale-invariant (``unit_mean``), the input
        can be metric or relative depth. Output dtype matches the depth stream.
        """
        h, w = self.img_hw
        d = cv2.resize(depth_hw.astype(np.float32), (w, h), interpolation=cv2.INTER_NEAREST)
        x = torch.from_numpy(d).to(self.device)[None, ..., None]  # (1, H, W, 1)
        tokens = encode_depth(x, self.depth_config)  # (1, lat_h, lat_w, depth_channels)
        b, lh, lw, c = tokens.shape
        body_dtype = next(self.model.dit.depth_in.parameters()).dtype
        return tokens.reshape(b, lh * lw, c).to(body_dtype)

    @torch.no_grad()
    def _null_embed(self) -> Tensor:
        """Cached Qwen3 embedding of the empty string, used as the CFG uncond.

        This checkpoint family was trained with the encoder's ``""`` embedding
        as the unconditional. Letting ``model.sample`` fall back to
        ``zeros_like(ctx)`` instead gives an out-of-distribution uncond that
        CFG amplifies by ``(cfg_scale - 1)``, softening the RGB and corrupting
        depth — so we encode ``""`` with the same embedder used for the prompt
        (keeping cond/uncond in the same space) and reuse it across calls.
        """
        if self._null_text_embed is None:
            null = self.embedder.forward([""]).to(self.device, torch.float32)
            if null.ndim == 2:
                null = null.unsqueeze(0)
            self._null_text_embed = null
        return self._null_text_embed

    @torch.no_grad()
    def generate(self, prompt: str, *, mode: Mode = "joint", num_steps: int = 50,
                 cfg_scale: float = 2.5, seed: int = 0,
                 clean_rgb_image: np.ndarray | None = None,
                 clean_rgb: Tensor | None = None,
                 clean_depth: Tensor | None = None,
                 refine_depth_i2d: bool = False,
                 i2d_cfg_scale: float = 1.0,
                 log2_alpha: float | None = None) -> dict:
        """Sample one (RGB, depth) pair from `prompt`. Returns a dict with:

            rgb:   (H, W, 3) uint8
            depth: (H, W)    float32 depth, relative scale (positive = farther)
            rgb_latent / depth_latent: raw model outputs (for debugging)
            metadata: dict of the call parameters

        ``refine_depth_i2d`` (joint mode only) turns the call into a two-stage
        pipeline: stage 1 joint-samples RGB+depth at ``cfg_scale``, then stage 2
        re-derives depth via image→depth on the stage-1 RGB *latent* at
        ``i2d_cfg_scale``. The final RGB is stage 1's; the final depth is stage
        2's. Conditioning depth on a fully-formed RGB (instead of a co-evolving
        noisy one) gives sharper, more RGB-consistent geometry.

        ``log2_alpha`` (joint mode only) tilts the RGB/depth denoising trajectory
        toward rgb-first (``> 0``) for cleaner depth; see ``rollout_timesteps``.
        """
        text_embed = self.embedder.forward([prompt]).to(self.device, torch.float32)
        if text_embed.ndim == 2:
            text_embed = text_embed.unsqueeze(0)

        # Convenience: if the caller supplies a clean image instead of token
        # latents, encode it here.
        if clean_rgb is None and clean_rgb_image is not None:
            clean_rgb = self.encode_image(clean_rgb_image)

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        lh, lw = self.latent_hw
        rgb_lat, depth_lat = self.model.sample(
            ctx=text_embed, img_height=lh, img_width=lw,
            num_steps=num_steps, mode=mode,
            schedule_config=self.schedule_config,
            cfg_scale=cfg_scale, seed=seed,
            rgb_use_x_prediction=False, depth_use_x_prediction=True,
            clean_rgb=clean_rgb, clean_depth=clean_depth,
            null_text_embed=self._null_embed() if cfg_scale > 1.0 else None,
            log2_alpha=log2_alpha,
        )

        # Stage 2 (optional): re-derive depth from the stage-1 RGB latent via
        # i2d. cfg_scale=1.0 here means no guidance, so no null embed is needed.
        if refine_depth_i2d and mode == "joint":
            _, depth_lat = self.model.sample(
                ctx=text_embed, img_height=lh, img_width=lw,
                num_steps=num_steps, mode="i2d",
                schedule_config=self.schedule_config,
                cfg_scale=i2d_cfg_scale, seed=seed,
                rgb_use_x_prediction=False, depth_use_x_prediction=True,
                clean_rgb=rgb_lat,
                null_text_embed=self._null_embed() if i2d_cfg_scale > 1.0 else None,
            )

        # VAE decode → uint8 RGB.
        rgb_spatial = einops.rearrange(rgb_lat, "b (h w) c -> b h w c", h=lh, w=lw)
        rgb_hw3 = (self.decoder(rgb_spatial.float())[0] * 0.5 + 0.5).clamp(0, 1)
        rgb = (rgb_hw3.float().cpu().numpy() * 255).clip(0, 255).astype(np.uint8)

        # Contract-bijection decode → depth.
        depth_spatial = einops.rearrange(depth_lat, "b (h w) c -> b 1 h w c", h=lh, w=lw)
        depth = decode_depth(depth_spatial, self.depth_config) \
            .squeeze(0).squeeze(0).squeeze(-1).float().cpu().numpy()

        return {
            "rgb": rgb, "depth": depth,
            "rgb_latent": rgb_lat.detach(), "depth_latent": depth_lat.detach(),
            "metadata": {"prompt": prompt, "mode": mode, "seed": seed,
                         "num_steps": num_steps, "cfg_scale": cfg_scale,
                         "img_hw": self.img_hw},
        }

    def save(self, result: dict, output_root: str | Path) -> dict[str, str]:
        """Save a generation into a timestamped subdir of `output_root`.

        Writes: rgb.png, depth_raw.npy (raw depth, relative scale), depth_magma.png
        (disparity visualization, near = bright), metadata.json.
        """
        slug = re.sub(r"[^a-z0-9]+", "_", result["metadata"]["prompt"].lower()).strip("_")[:48]
        ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        d = Path(output_root) / f"{ts}_{slug}"
        d.mkdir(parents=True, exist_ok=True)

        rgb, depth = result["rgb"], result["depth"]
        cv2.imwrite(str(d / "rgb.png"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        np.save(d / "depth_raw.npy", depth)

        # Disparity (1/depth) magma, robustly normalized to the 5–95th
        # percentile so near surfaces read bright and far ones dark.
        valid = (depth > 0) & np.isfinite(depth)
        if valid.any():
            disparity = np.zeros_like(depth, dtype=np.float32)
            disparity[valid] = 1.0 / depth[valid]
            lo, hi = np.percentile(disparity[valid], [5, 95])
            disparity = np.clip((disparity - lo) / max(hi - lo, 1e-8), 0, 1)
            disparity[~valid] = 0.0
            magma = (cm.magma(disparity)[..., :3] * 255).astype(np.uint8)
            cv2.imwrite(str(d / "depth_magma.png"), cv2.cvtColor(magma, cv2.COLOR_RGB2BGR))

        (d / "metadata.json").write_text(json.dumps(result["metadata"], indent=2))
        return {"run_dir": str(d), "rgb": str(d / "rgb.png"),
                "depth_raw": str(d / "depth_raw.npy"),
                "depth_magma": str(d / "depth_magma.png"),
                "metadata": str(d / "metadata.json")}
