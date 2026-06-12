# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 World Labs.
"""Qwen3-based text encoder for the FLUX-RGBD model family.

The encoder turns text prompts into a stack of hidden states drawn from
three intermediate layers of Qwen3-8B-FP8 (layers 9, 18, 27, mirroring
the FLUX.2 [klein] recipe). Output shape is

    (batch, MAX_LENGTH=512, 3 * hidden_size = 12288)

Pads to the full ``MAX_LENGTH`` (no stripping) because the FLUX.2 DiT was
trained against the padded text stream produced by Black Forest Labs.

Loads everything from HuggingFace Hub -- no internal mirrors, no auth.
"""

from __future__ import annotations

import einops
import torch
from torch import Tensor, nn
from transformers import AutoModelForCausalLM, AutoTokenizer

from flux_rgbd._flux2.constants import MAX_LENGTH, OUTPUT_LAYERS_QWEN3

DEFAULT_MODEL_SPEC = "Qwen/Qwen3-8B-FP8"


class Qwen3Embedder(nn.Module):
    """Wrap a Qwen3 causal LM, surface multi-layer hidden states.

    Inference-only. Chat-template policy matches the upstream FLUX.2
    [klein] recipe: user role, generation-prompt appended, thinking
    disabled.
    """

    def __init__(
        self,
        model_spec: str = DEFAULT_MODEL_SPEC,
        *,
        device: str | torch.device = "cuda",
        max_length: int = MAX_LENGTH,
        output_layers: tuple[int, ...] = OUTPUT_LAYERS_QWEN3,
    ) -> None:
        super().__init__()
        self.model_spec = model_spec
        self.device_ = torch.device(device)
        self.max_length = max_length
        self.output_layers = tuple(output_layers)

        self._tokenizer = AutoTokenizer.from_pretrained(model_spec)
        # Load straight onto the target device. `dtype=None` lets
        # transformers honor whatever the checkpoint advertises (FP8 for
        # the Qwen3-8B-FP8 we use by default).
        self._model = AutoModelForCausalLM.from_pretrained(
            model_spec, dtype=None, device_map=str(self.device_)
        )
        self._model.eval()
        self._pad_id = self._tokenizer.pad_token_id or 0

    @staticmethod
    def _apply_chat_template(tok, text: str) -> str:
        return tok.apply_chat_template(
            [{"role": "user", "content": text}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )

    def _tokenize(self, prompts: list[str]) -> tuple[Tensor, Tensor]:
        formatted = [self._apply_chat_template(self._tokenizer, p) for p in prompts]
        # First tokenize without padding to control truncation; we'll pad
        # ourselves to `max_length` so the output sequence length is fixed
        # regardless of the prompt batch.
        encoded = self._tokenizer(
            formatted,
            padding=False,
            truncation=True,
            max_length=self.max_length,
        )
        rows = encoded["input_ids"]
        b = len(rows)
        input_ids = torch.full(
            (b, self.max_length), fill_value=self._pad_id, dtype=torch.long
        )
        attn = torch.zeros((b, self.max_length), dtype=torch.long)
        for i, row in enumerate(rows):
            n = min(len(row), self.max_length)
            input_ids[i, :n] = torch.tensor(row[:n], dtype=torch.long)
            attn[i, :n] = 1
        return input_ids.to(self.device_), attn.to(self.device_)

    @torch.inference_mode()
    def forward(self, prompts: list[str]) -> Tensor:
        """Encode a batch of prompts.

        Returns:
            Tensor of shape ``(batch, max_length, len(output_layers) * hidden_size)``.
        """
        input_ids, attention_mask = self._tokenize(list(prompts))
        out = self._model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False,
        )
        # `hidden_states` is a tuple of length num_hidden_layers + 1.
        layers = torch.stack(
            [out.hidden_states[k] for k in self.output_layers], dim=1
        )  # (b, len(output_layers), seq, hidden)
        return einops.rearrange(layers, "b c l d -> b l (c d)")
