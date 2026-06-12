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
Architecture parameters and other constants useful for FLUX.2 inference.

This module centralizes all constants used across the FLUX.2 model implementation,
including model architecture configurations for all public FLUX.2 variants, their
text encoder parameters, and system prompts for prompt upsampling.
"""

from types import MappingProxyType

__all__ = [
    "FLUX2_DEV_PARAMS",
    "FLUX2_KLEIN_4B_PARAMS",
    "FLUX2_KLEIN_9B_PARAMS",
    "OUTPUT_LAYERS_MISTRAL",
    "OUTPUT_LAYERS_QWEN3",
    "MAX_LENGTH",
    "UPSAMPLING_MAX_IMAGE_SIZE",
    "SYSTEM_MESSAGE",
    "SYSTEM_MESSAGE_UPSAMPLING_T2I",
    "SYSTEM_MESSAGE_UPSAMPLING_I2I",
]

# =============================================================================
# Model Architecture Parameters
# =============================================================================

FLUX2_DEV_PARAMS = MappingProxyType(
    {
        "in_channels": 128,
        "context_in_dim": 15360,
        "hidden_size": 6144,
        "num_heads": 48,
        "depth": 8,
        "depth_single_blocks": 48,
        "axes_dim": (32, 32, 32, 32),
        "theta": 2000,
        "mlp_ratio": 3.0,
        "use_guidance_embed": True,
    }
)
"""
Architecture config for FLUX.2 [dev] 32B parameter model (guidance distilled).
"""

FLUX2_KLEIN_4B_PARAMS = MappingProxyType(
    {
        "in_channels": 128,
        "context_in_dim": 7680,
        "hidden_size": 3072,
        "num_heads": 24,
        "depth": 5,
        "depth_single_blocks": 20,
        "axes_dim": (32, 32, 32, 32),
        "theta": 2000,
        "mlp_ratio": 3.0,
        "use_guidance_embed": False,
    }
)
"""
Architecture config for FLUX.2 [klein] 4B parameter model (size distilled).
"""

FLUX2_KLEIN_9B_PARAMS = MappingProxyType(
    {
        "in_channels": 128,
        "context_in_dim": 12288,
        "hidden_size": 4096,
        "num_heads": 32,
        "depth": 8,
        "depth_single_blocks": 24,
        "axes_dim": (32, 32, 32, 32),
        "theta": 2000,
        "mlp_ratio": 3.0,
        "use_guidance_embed": False,
    }
)
"""
Immutable architecture configuration for FLUX.2 [klein]-9B model.

This is a mid-size 9B parameter model that balances quality and speed. It uses
Qwen3-8B as text encoder and provides higher quality than the 4B variant while
maintaining fast 4-step inference. The model is guidance-distilled.

Key architectural features:
    - 24 single-stream transformer blocks (between 4B and dev)
    - 4096 hidden dimensions with 32 attention heads
    - No guidance embeddings (guidance-distilled into weights)
    - Context dimension 12288 matches concatenated Qwen3-8B outputs

Default generation parameters (distilled variants):
    - guidance: 1.0 (fixed, baked into weights)
    - num_steps: 4 (fixed)

Base variant generation parameters:
    - guidance: 4.0 (adjustable)
    - num_steps: 50 (adjustable)
"""

# =============================================================================
# Text Encoder Configuration
# =============================================================================

OUTPUT_LAYERS_MISTRAL = (10, 20, 30)
"""
Layer indices to extract hidden states from Mistral-Small-3.2-24B text encoder.

The FLUX.2 [dev] model uses Mistral-Small as its text encoder and concatenates
hidden states from these three intermediate layers to form rich text embeddings.
"""

OUTPUT_LAYERS_QWEN3 = (9, 18, 27)
"""
Layer indices to extract hidden states from Qwen3 text encoder variants.

The FLUX.2 [klein] models (4B and 9B) use Qwen3 text encoders and concatenate
hidden states from these three intermediate layers to form text embeddings.
"""

MAX_LENGTH = 512
"""
Maximum sequence length for text encoder tokenization.

Text prompts are truncated or padded to this length during encoding. This
applies to both Mistral-Small and Qwen3 text encoders.
"""

UPSAMPLING_MAX_IMAGE_SIZE = 768**2
"""
Maximum pixel area for images during prompt upsampling.

When images are provided as conditioning for prompt upsampling (image-to-image
mode), they are resized to have at most this many pixels while preserving
aspect ratio. This prevents out-of-memory errors when processing large images.
"""

# =============================================================================
# System Messages for Prompt Upsampling
# =============================================================================

SYSTEM_MESSAGE = """You are an AI that reasons about image descriptions. You give \
structured responses focusing on object relationships, object attribution and \
actions without speculation."""
"""
Default system message for text encoder when generating image descriptions.

This system message is used for general-purpose text encoding tasks where the
model needs to reason about images in a structured, factual manner.
"""

SYSTEM_MESSAGE_UPSAMPLING_T2I = """You are an expert prompt engineer for FLUX.2 \
by Black Forest Labs. Rewrite user prompts to be more descriptive while \
strictly preserving their core subject and intent.

Guidelines:
1. Structure: Keep structured inputs structured (enhance within fields). \
Convert natural language to detailed paragraphs.
2. Details: Add concrete visual specifics - form, scale, textures, materials, \
lighting (quality, direction, color), shadows, spatial relationships, and \
environmental context.
3. Text in Images: Put ALL text in quotation marks, matching the prompt's \
language. Always provide explicit quoted text for objects that would contain \
text in reality (signs, labels, screens, etc.) - without it, the model \
generates gibberish.

Output only the revised prompt and nothing else."""
"""
System message for text-to-image prompt upsampling.

This prompt engineering template guides the text encoder to expand sparse user
prompts into detailed, visually-rich descriptions that improve image generation
quality. It emphasizes adding concrete visual details while preserving the
original intent.
"""

SYSTEM_MESSAGE_UPSAMPLING_I2I = """You are FLUX.2 by Black Forest Labs, an \
image-editing expert. You convert editing requests into one concise \
instruction (50-80 words, ~30 for brief requests).

Rules:
- Single instruction only, no commentary
- Use clear, analytical language (avoid "whimsical," "cascading," etc.)
- Specify what changes AND what stays the same (face, lighting, composition)
- Reference actual image elements
- Turn negatives into positives ("don't change X" → "keep X")
- Make abstractions concrete ("futuristic" → "glowing cyan neon, metallic panels")
- Keep content PG-13

Output only the final instruction in plain text and nothing else."""
"""
System message for image-to-image prompt upsampling.

This prompt engineering template guides the text encoder to convert free-form
image editing requests into concise, actionable instructions suitable for
image-to-image generation tasks.
"""
