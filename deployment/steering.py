"""Add layer increments of raw persona vectors at Qwen decoder-block outputs."""

from __future__ import annotations

from contextlib import contextmanager
import math

from deployment.core import AXES, MAX_MOOD_COEFFICIENT, MOOD_COEFFICIENTS


def zero_direction_metadata(vectors):
    """Flag exact zeros without pruning, replacing, or rescaling raw vectors."""
    zero = (vectors == 0).all(dim=-1).cpu()
    return {
        "mood_vectors_zero_layer_trait_count": int(zero.sum().item()),
        "mood_vectors_zero_layers": {
            trait: [layer + 1 for layer in range(zero.shape[0]) if bool(zero[layer, index])]
            for index, trait in enumerate(AXES)
        },
        "mood_vectors_entirely_zero_traits": [
            trait for index, trait in enumerate(AXES) if bool(zero[:, index].all())
        ],
        "mood_vectors_entirely_zero_bank": bool(zero.all()),
    }


class MoodSteering:
    """A required frozen [layer, mood, hidden] bank, without rescaling.

    Hooks belong to one serialized generation and never edit model weights.
    Each layer adds its raw vector minus the previous layer's raw vector, before
    final RMSNorm. The first layer uses a zero predecessor; the bank stays raw.
    """

    def __init__(self, model, vectors=None, provenance=None, excluded_content_token_ids=(), thinking_token_ids=(None, None)):
        import torch

        self.torch = torch
        self.model = model
        self.excluded_content_token_ids = tuple(excluded_content_token_ids)
        self.thinking_token_ids = tuple(thinking_token_ids)
        self.layers = model.model.language_model.layers
        config = model.config.text_config
        shape = (int(config.num_hidden_layers), len(AXES), int(config.hidden_size))
        if len(self.layers) != shape[0] or shape[0] < 1 or shape[2] < 1:
            raise ValueError("Mood bank dimensions disagree with decoder configuration")
        if vectors is None:
            raise ValueError("A real extracted persona vector bank is required")
        if not isinstance(vectors, torch.Tensor) or tuple(vectors.shape) != shape:
            raise ValueError(f"Mood vectors must have shape {shape} in layer/axis/hidden order")
        if vectors.dtype != torch.float32 or not bool(torch.isfinite(vectors).all()):
            raise ValueError("Mood vectors must be finite raw float32 values")
        bank = vectors.detach().to(device=model.device, dtype=torch.float32).clone()
        self.vectors = bank
        self.incremental_vectors = bank.clone()
        self.incremental_vectors[1:] = bank[1:] - bank[:-1]
        self.zero_metadata = zero_direction_metadata(bank)
        self.provenance = dict(provenance or {})
        self.source = self.provenance.get("mood_vectors_source", "provided")

    def metadata(self):
        return {
            **self.provenance,
            **self.zero_metadata,
            "mood_vectors_available": True,
            "steering_available": True,
            "mood_vectors_source": self.source,
            "mood_vectors_validated": False,
            "steering_method": "paper_incremental_all_layers",
            "steering_incremental_definition": "raw_layer_vector_minus_previous_layer_vector",
            "steering_first_layer_previous_vector": "zero",
            "steering_activation_boundary": "decoder_block_output_pre_final_global_rmsnorm",
            "steering_token_scope": "final_formatted_prompt_token_then_generated_content_tokens",
            "mood_coefficients": list(MOOD_COEFFICIENTS),
        }

    def offsets(self, coefficients):
        if len(coefficients) != len(AXES) or any(
            type(level) not in (int, float)
            or not -MAX_MOOD_COEFFICIENT <= level <= MAX_MOOD_COEFFICIENT
            or not math.isfinite(level) for level in coefficients
        ):
            raise ValueError(f"Expected six finite mood coefficients between {-MAX_MOOD_COEFFICIENT} and {MAX_MOOD_COEFFICIENT}")
        alpha = self.torch.tensor(coefficients, dtype=self.torch.float32, device=self.vectors.device)
        return self.torch.einsum("m,lmh->lh", alpha, self.incremental_vectors)

    @contextmanager
    def apply(self, coefficients, final_prompt_token):
        offsets = self.offsets(coefficients)
        if not any(coefficients):
            # Balanced requests use the original model without even a zero add.
            yield False
            return
        if type(final_prompt_token) is not int or final_prompt_token < 0:
            raise ValueError("Expected a nonnegative final formatted prompt token index")
        torch = self.torch
        handles = []
        current_tokens = {"prefilled": False, "thinking": False}

        def capture_content_tokens(module, args, kwargs):
            ids = args[0] if args else kwargs.get("input_ids")
            if ids is None or ids.ndim != 2 or ids.shape[0] != 1:
                raise ValueError("Content-token steering requires batch-one input IDs")
            excluded = torch.tensor(self.excluded_content_token_ids, device=ids.device, dtype=ids.dtype)
            content = ~torch.isin(ids, excluded)
            # The disabled-thinking chat prefix belongs to the prompt, not to a
            # generated reasoning span. Read tags only after the initial prefill.
            if current_tokens["prefilled"] and any(token is not None for token in self.thinking_token_ids):
                start, end = self.thinking_token_ids
                for index, token in enumerate(ids[0].tolist()):
                    if token == start:
                        current_tokens["thinking"] = True
                        content[0, index] = False
                    elif token == end:
                        current_tokens["thinking"] = False
                        content[0, index] = False
                    elif current_tokens["thinking"]:
                        content[0, index] = False
            current_tokens["prefilled"] = True
            current_tokens["mask"] = content

        def make_hook(offset):
            def hook(module, args, kwargs, output):
                hidden = output[0] if isinstance(output, tuple) else output
                positions = kwargs.get("position_ids")
                # Pinned Qwen passes its 2D text-position plane to each block;
                # the other three MRoPE planes must not determine this mask.
                if hidden is None or hidden.ndim != 3 or hidden.shape[0] != 1:
                    raise ValueError("Mood steering supports batch-one decoder outputs")
                if positions is None or positions.ndim != 2 or positions.shape != hidden.shape[:2]:
                    raise ValueError("Decoder text positions must match the current input tokens")
                added = (hidden.float() + offset.to(hidden.device)).to(hidden.dtype)
                mask = (positions.to(hidden.device) >= final_prompt_token).unsqueeze(-1)
                if self.excluded_content_token_ids:
                    content = current_tokens.get("mask")
                    if content is None or content.shape != positions.shape:
                        raise ValueError("Content token IDs must match decoder text positions")
                    eligible = (positions == final_prompt_token) | (
                        (positions > final_prompt_token) & content.to(positions.device)
                    )
                    mask = eligible.to(hidden.device).unsqueeze(-1)
                steered = torch.where(mask, added, hidden)
                return (steered, *output[1:]) if isinstance(output, tuple) else steered
            return hook

        try:
            if self.excluded_content_token_ids:
                handles.append(self.model.register_forward_pre_hook(capture_content_tokens, with_kwargs=True))
            for layer, offset in zip(self.layers, offsets):
                handles.append(layer.register_forward_hook(make_hook(offset), with_kwargs=True))
            yield True
        finally:
            for handle in handles:
                handle.remove()
