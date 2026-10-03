"""Add already-normalized mood vectors to Qwen's decoder-layer inputs."""

from __future__ import annotations

from contextlib import contextmanager

from deployment.core import AXES, PLACEHOLDER_SEED, PLACEHOLDER_NORM


class MoodSteering:
    """A frozen [layer, mood, hidden] bank; supplied magnitudes are preserved.

    The default deterministic bank is an integration fixture, not learned mood
    directions. Hooks belong to one serialized generation and never edit weights.
    """

    def __init__(self, model, vectors=None):
        import torch

        self.torch = torch
        self.layers = model.model.language_model.layers
        config = model.config.text_config
        shape = (int(config.num_hidden_layers), len(AXES), int(config.hidden_size))
        if len(self.layers) != shape[0] or shape[0] < 1 or shape[2] < 1:
            raise ValueError("Mood bank dimensions disagree with decoder configuration")
        self.source = "random_placeholder" if vectors is None else "provided"
        if vectors is None:
            generator = torch.Generator(device="cpu").manual_seed(PLACEHOLDER_SEED)
            vectors = torch.randn(shape, generator=generator, dtype=torch.float32)
            vectors = vectors * (PLACEHOLDER_NORM / vectors.norm(dim=-1, keepdim=True))
        if not isinstance(vectors, torch.Tensor) or tuple(vectors.shape) != shape:
            raise ValueError(f"Mood vectors must have shape {shape} in layer/axis/hidden order")
        if not vectors.is_floating_point() or not bool(torch.isfinite(vectors).all()):
            raise ValueError("Mood vectors must be finite floating-point values")
        bank = vectors.detach().to(device=model.device, dtype=torch.float32).clone()
        norms = bank.norm(dim=-1)
        if not bool(torch.isfinite(norms).all()) or bool((norms == 0).any()):
            raise ValueError("Every provided mood vector must have a finite nonzero norm")
        self.vectors = bank

    def metadata(self):
        return {
            "mood_vectors_available": True,
            "steering_available": True,
            "mood_vectors_source": self.source,
            "mood_vectors_validated": False,
        }

    def offsets(self, coefficients):
        if len(coefficients) != len(AXES) or any(
            type(level) is not int or not -2 <= level <= 2 for level in coefficients
        ):
            raise ValueError("Expected six mood coefficients between -2 and 2")
        alpha = self.torch.tensor(coefficients, dtype=self.torch.float32, device=self.vectors.device)
        return self.torch.einsum("m,lmh->lh", alpha, self.vectors)

    @contextmanager
    def apply(self, coefficients, post_instruction_start):
        offsets = self.offsets(coefficients)
        if not any(coefficients):
            # Balanced requests use the original model without even a zero add.
            yield False
            return
        if type(post_instruction_start) is not int or post_instruction_start < 0:
            raise ValueError("Expected a nonnegative post-instruction token boundary")
        torch = self.torch
        handles = []

        def make_hook(offset):
            def hook(module, args, kwargs):
                hidden = args[0] if args else kwargs.get("hidden_states")
                positions = kwargs.get("position_ids")
                # Pinned Qwen passes its 2D text-position plane to each block;
                # the other three MRoPE planes must not determine this mask.
                if hidden is None or hidden.ndim != 3 or hidden.shape[0] != 1:
                    raise ValueError("Mood steering supports batch-one decoder inputs")
                if positions is None or positions.ndim != 2 or positions.shape != hidden.shape[:2]:
                    raise ValueError("Decoder text positions must match the current input tokens")
                added = (hidden.float() + offset.to(hidden.device)).to(hidden.dtype)
                mask = (positions.to(hidden.device) >= post_instruction_start).unsqueeze(-1)
                steered = torch.where(mask, added, hidden)
                if args:
                    return (steered, *args[1:]), kwargs
                return args, {**kwargs, "hidden_states": steered}
            return hook

        try:
            for layer, offset in zip(self.layers, offsets):
                handles.append(layer.register_forward_pre_hook(make_hook(offset), with_kwargs=True))
            yield True
        finally:
            for handle in handles:
                handle.remove()
