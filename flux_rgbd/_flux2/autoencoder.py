# SPDX-License-Identifier: Apache-2.0
# Copyright 2025 Black Forest Labs.
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

"""FLUX.2 autoencoder.

Adapted from the FLUX.2 codebase:
    https://github.com/black-forest-labs/flux2
"""

import math
from collections.abc import Sequence

import torch
from einops import rearrange
from jaxtyping import Float
from safetensors.torch import load_file as load_safetensors
from torch import Tensor, nn

__all__ = ["Flux2Decoder", "Flux2Encoder"]

# FLUX.2-dev autoencoder weights, pre-split into encoder + decoder
# safetensors. (BFL's canonical `ae.safetensors` is a combined file
# we'd have to filter by key prefix -- easier to host the split files
# inside the model repo. Override via the FLUX_RGBD_LOCAL_AE_{...}
# env vars if you want to skip the Hub fetch entirely.)
DEFAULT_AE_REPO = "bartduis/modality_forcing"
DEFAULT_ENCODER_FILE = "ae_encoder.safetensors"
DEFAULT_DECODER_FILE = "ae_decoder.safetensors"

DEFAULT_RESOLUTION = 256
DEFAULT_IN_CHANNELS = 3
DEFAULT_CH = 128
DEFAULT_OUT_CHANNELS = 3
DEFAULT_CH_MULT = (1, 2, 4, 4)
DEFAULT_NUM_RES_BLOCKS = 2
DEFAULT_Z_CHANNELS = 32


def _hf_download(repo_id: str, filename: str) -> str:
    """Resolve a HuggingFace Hub file path. Lazy-import hf_hub to keep the
    base package importable in environments that pin huggingface_hub via
    extras."""
    from huggingface_hub import hf_hub_download

    return hf_hub_download(repo_id=repo_id, filename=filename)


def _load_weights(repo_id: str, filename: str) -> dict[str, Tensor]:
    # Allow callers to point at a local file via env var. Useful for tests
    # and air-gapped environments.
    import os

    env_key = f"FLUX_RGBD_LOCAL_{filename.upper().replace('.', '_')}"
    override = os.environ.get(env_key)
    if override:
        if not os.path.isfile(override):
            raise FileNotFoundError(f"{env_key}={override} is set but not a file")
        return load_safetensors(override)
    path = _hf_download(repo_id, filename)
    return load_safetensors(path)


def _swish(x: Tensor) -> Tensor:
    return x * torch.sigmoid(x)


class AttnBlock(nn.Module):
    """Attention block for FLUX.2 autoencoder."""

    def __init__(self, in_channels: int):
        super().__init__()
        self.in_channels = in_channels

        self.norm = nn.GroupNorm(
            num_groups=32, num_channels=in_channels, eps=1e-6, affine=True
        )

        self.q = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.k = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.v = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.proj_out = nn.Conv2d(in_channels, in_channels, kernel_size=1)

    def attention(self, h_: Tensor) -> Tensor:
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        b, c, h, w = q.shape
        q = rearrange(q, "b c h w -> b 1 (h w) c").contiguous()
        k = rearrange(k, "b c h w -> b 1 (h w) c").contiguous()
        v = rearrange(v, "b c h w -> b 1 (h w) c").contiguous()
        h_ = nn.functional.scaled_dot_product_attention(q, k, v)

        return rearrange(h_, "b 1 (h w) c -> b c h w", h=h, w=w, c=c, b=b)

    def forward(self, x: Tensor) -> Tensor:
        return x + self.proj_out(self.attention(x))


class ResnetBlock(nn.Module):
    """Resnet block for FLUX.2 autoencoder."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels

        self.norm1 = nn.GroupNorm(
            num_groups=32, num_channels=in_channels, eps=1e-6, affine=True
        )
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=1, padding=1
        )
        self.norm2 = nn.GroupNorm(
            num_groups=32, num_channels=out_channels, eps=1e-6, affine=True
        )
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1
        )
        if self.in_channels != self.out_channels:
            self.nin_shortcut = nn.Conv2d(
                in_channels, out_channels, kernel_size=1, stride=1, padding=0
            )

    def forward(self, x):
        h = x
        h = self.norm1(h)
        h = _swish(h)
        h = self.conv1(h)

        h = self.norm2(h)
        h = _swish(h)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            x = self.nin_shortcut(x)

        return x + h


class Downsample(nn.Module):
    """Downsample block for FLUX.2 autoencoder."""

    def __init__(self, in_channels: int):
        super().__init__()
        # no asymmetric padding in torch conv, must do it ourselves
        self.conv = nn.Conv2d(
            in_channels, in_channels, kernel_size=3, stride=2, padding=0
        )

    def forward(self, x: Tensor):
        pad = (0, 1, 0, 1)
        x = nn.functional.pad(x, pad, mode="constant", value=0)
        x = self.conv(x)
        return x


class Upsample(nn.Module):
    """Upsample block for FLUX.2 autoencoder."""

    def __init__(self, in_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, in_channels, kernel_size=3, stride=1, padding=1
        )

    def forward(self, x: Tensor):
        x = nn.functional.interpolate(x, scale_factor=2.0, mode="nearest")
        x = self.conv(x)
        return x


class Flux2Encoder(nn.Module):
    """Encoder for FLUX.2 autoencoder."""

    def __init__(
        self,
        resolution: int = DEFAULT_RESOLUTION,
        in_channels: int = DEFAULT_IN_CHANNELS,
        ch: int = DEFAULT_CH,
        ch_mult: Sequence[int] = DEFAULT_CH_MULT,
        num_res_blocks: int = DEFAULT_NUM_RES_BLOCKS,
        z_channels: int = DEFAULT_Z_CHANNELS,
        repo_id: str = DEFAULT_AE_REPO,
        filename: str = DEFAULT_ENCODER_FILE,
    ):
        """Initialize the FLUX.2 encoder.

        Args:
            resolution: The resolution of the input images.
            in_channels: The number of channels in the input images.
            ch: The number of channels in the encoder.
            ch_mult: The number of channels in the encoder at each resolution.
            num_res_blocks: The number of ResNet blocks in each downsampling block.
            z_channels: The number of channels in the latent space.
            repo_id: HuggingFace Hub repository id from which to load weights.
            filename: Filename of the safetensors weights within `repo_id`.
        """
        super().__init__()
        self._repo_id = repo_id
        self._filename = filename

        self.quant_conv = torch.nn.Conv2d(2 * z_channels, 2 * z_channels, 1)
        self.ch = ch
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels
        # downsampling
        self.conv_in = nn.Conv2d(
            in_channels, self.ch, kernel_size=3, stride=1, padding=1
        )

        curr_res = resolution
        in_ch_mult = (1, *tuple(ch_mult))
        self.in_ch_mult = in_ch_mult
        self.down = nn.ModuleList()
        block_in = self.ch
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = ch * in_ch_mult[i_level]
            block_out = ch * ch_mult[i_level]
            for _ in range(self.num_res_blocks):
                block.append(ResnetBlock(in_channels=block_in, out_channels=block_out))
                block_in = block_out
            down = nn.Module()
            down.block = block
            down.attn = attn
            if i_level != self.num_resolutions - 1:
                down.downsample = Downsample(block_in)
                curr_res = curr_res // 2
            self.down.append(down)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in, out_channels=block_in)
        self.mid.attn_1 = AttnBlock(block_in)
        self.mid.block_2 = ResnetBlock(in_channels=block_in, out_channels=block_in)

        # end
        self.norm_out = nn.GroupNorm(
            num_groups=32, num_channels=block_in, eps=1e-6, affine=True
        )
        self.conv_out = nn.Conv2d(
            block_in, 2 * z_channels, kernel_size=3, stride=1, padding=1
        )

        self.bn_eps = 1e-4
        self.bn_momentum = 0.1
        self.ps = [2, 2]
        self.bn = torch.nn.BatchNorm2d(
            math.prod(self.ps) * z_channels,
            eps=self.bn_eps,
            momentum=self.bn_momentum,
            affine=False,
            track_running_stats=True,
        )

    def load_weights(self):
        """Fetch and load weights from HuggingFace Hub."""
        self.load_state_dict(_load_weights(self._repo_id, self._filename))

    def _forward(self, x: Float[Tensor, "n cx hx wx"]) -> Float[Tensor, "n cz hz wz"]:
        # downsampling
        hs = [self.conv_in(x)]
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](hs[-1])
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)
                hs.append(h)
            if i_level != self.num_resolutions - 1:
                hs.append(self.down[i_level].downsample(hs[-1]))

        # middle
        h = hs[-1]
        h = self.mid.block_1(h)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h)
        # end
        h = self.norm_out(h)
        h = _swish(h)
        h = self.conv_out(h)
        h = self.quant_conv(h)
        return h

    def _normalize(self, z: Float[Tensor, "n cz hz wz"]) -> Float[Tensor, "n cz hz wz"]:
        self.bn.eval()
        return self.bn(z)

    def forward(
        self, x: Float[Tensor, "... hx wx cx"]
    ) -> Float[Tensor, "... hz wz cz"]:
        """Encode images to latents.

        Args:
            x: Images to encode.

        Returns:
            The encoded latents. The latents will be normalized using the running
                statistics from training.
        """
        *batch_shape, _, _, _ = x.shape
        # Convert to NCHW.
        x = rearrange(x, "... h w c -> (...) c h w")

        # Compute latents.
        moments = self._forward(x)
        mean = torch.chunk(moments, 2, dim=1)[0]
        z = rearrange(
            mean,
            "... c (i pi) (j pj)  -> ... (c pi pj) i j",
            pi=self.ps[0],
            pj=self.ps[1],
        )
        z = self._normalize(z)

        # Convert back to NHWC and restore batch shape.
        z = rearrange(z, "... c h w -> ... h w c")
        return z.reshape(*batch_shape, *z.shape[-3:])


class Flux2Decoder(nn.Module):
    """Decoder for FLUX.2 autoencoder."""

    def __init__(
        self,
        ch: int = DEFAULT_CH,
        out_ch: int = DEFAULT_OUT_CHANNELS,
        ch_mult: Sequence[int] = DEFAULT_CH_MULT,
        num_res_blocks: int = DEFAULT_NUM_RES_BLOCKS,
        in_channels: int = DEFAULT_IN_CHANNELS,
        resolution: int = DEFAULT_RESOLUTION,
        z_channels: int = DEFAULT_Z_CHANNELS,
        repo_id: str = DEFAULT_AE_REPO,
        filename: str = DEFAULT_DECODER_FILE,
    ):
        """Initialize the FLUX.2 decoder.

        Args:
            ch: The number of channels in the decoder.
            out_ch: The number of channels in the output.
            ch_mult: The number of channels in the decoder at each resolution.
            num_res_blocks: The number of ResNet blocks in each upsampling block.
            in_channels: The number of channels in the input images.
            resolution: The resolution of the input images.
            z_channels: The number of channels in the latent space.
            repo_id: HuggingFace Hub repository id from which to load weights.
            filename: Filename of the safetensors weights within `repo_id`.
        """
        super().__init__()
        self._repo_id = repo_id
        self._filename = filename

        self.post_quant_conv = torch.nn.Conv2d(z_channels, z_channels, 1)
        self.ch = ch
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels
        self.ffactor = 2 ** (self.num_resolutions - 1)

        # compute in_ch_mult, block_in and curr_res at lowest res
        block_in = ch * ch_mult[self.num_resolutions - 1]
        curr_res = resolution // 2 ** (self.num_resolutions - 1)
        self.z_shape = (1, z_channels, curr_res, curr_res)

        # z to block_in
        self.conv_in = nn.Conv2d(
            z_channels, block_in, kernel_size=3, stride=1, padding=1
        )

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in, out_channels=block_in)
        self.mid.attn_1 = AttnBlock(block_in)
        self.mid.block_2 = ResnetBlock(in_channels=block_in, out_channels=block_in)

        # upsampling
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = ch * ch_mult[i_level]
            for _ in range(self.num_res_blocks + 1):
                block.append(ResnetBlock(in_channels=block_in, out_channels=block_out))
                block_in = block_out
            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level != 0:
                up.upsample = Upsample(block_in)
                curr_res = curr_res * 2
            self.up.insert(0, up)  # prepend to get consistent order

        # end
        self.norm_out = nn.GroupNorm(
            num_groups=32, num_channels=block_in, eps=1e-6, affine=True
        )
        self.conv_out = nn.Conv2d(block_in, out_ch, kernel_size=3, stride=1, padding=1)

        self.bn_eps = 1e-4
        self.bn_momentum = 0.1
        self.ps = [2, 2]
        self.bn = torch.nn.BatchNorm2d(
            math.prod(self.ps) * z_channels,
            eps=self.bn_eps,
            momentum=self.bn_momentum,
            affine=False,
            track_running_stats=True,
        )

    def load_weights(self):
        """Fetch and load weights from HuggingFace Hub."""
        self.load_state_dict(_load_weights(self._repo_id, self._filename))

    def _forward(self, z: Float[Tensor, "n cz hz wz"]) -> Float[Tensor, "n cx hx wx"]:
        z = self.post_quant_conv(z)

        # get dtype for proper tracing
        upscale_dtype = next(self.up.parameters()).dtype

        # z to block_in
        h = self.conv_in(z)

        # middle
        h = self.mid.block_1(h)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h)

        # cast to proper dtype
        h = h.to(upscale_dtype)
        # upsampling
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks + 1):
                h = self.up[i_level].block[i_block](h)
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)
            if i_level != 0:
                h = self.up[i_level].upsample(h)

        # end
        h = self.norm_out(h)
        h = _swish(h)
        h = self.conv_out(h)
        return h

    def _inv_normalize(
        self, z: Float[Tensor, "n cz hz wz"]
    ) -> Float[Tensor, "n cz hz wz"]:
        self.bn.eval()
        s = torch.sqrt(self.bn.running_var.view(1, -1, 1, 1) + self.bn_eps)
        m = self.bn.running_mean.view(1, -1, 1, 1)
        return z * s + m

    def forward(
        self, z: Float[Tensor, "... cz hz wz"]
    ) -> Float[Tensor, "... cx hx wx"]:
        """Decode latents to images.

        Args:
            z: Image latents normalized to have zero mean and unit variance.

        Returns:
            Decoded images.
        """
        *batch_shape, _, _, _ = z.shape

        # Convert to NCHW.
        z = rearrange(z, "... h w c -> (...) c h w")

        # Denormalize and evaluate decoder.
        z = self._inv_normalize(z)
        z = rearrange(
            z,
            "... (c pi pj) i j -> ... c (i pi) (j pj)",
            pi=self.ps[0],
            pj=self.ps[1],
        )
        dec = self._forward(z)

        # Convert back to NHWC and restore batch shape.
        dec = rearrange(dec, "... c h w -> ... h w c")
        dec = dec.reshape(*batch_shape, *dec.shape[-3:])
        return dec
