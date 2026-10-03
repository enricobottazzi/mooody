"""Text-only Arditi-style direction extraction and weight editing for Qwen3.5.

Architecture reviewed against Transformers v5.18.0's modeling_qwen3_5.py.
Qwen3.5-9B has 32 sequential residual blocks: 24 GatedDeltaNet mixers and
8 full-attention mixers, followed by an MLP in every block.  Only the text
embedding and the output maps writing into the text residual stream are edited.
The vision encoder, input projections, normalizations, and untied LM head are
preserved.  This module does not establish multimodal behavior or capabilities.

Extraction defaults to residual *pre* (the decoder block input), matching the
authors' implementation.  Optional *mid* means the residual after the mixer and
before post_attention_layernorm; it is an extension, not the paper's default.
All suffix positions are derived from the pinned tokenizer's no-thinking chat
template, including the user end marker and empty thinking block.  Accumulators
are CPU float64; no full prompt-by-layer activation cache is retained.
"""

from __future__ import annotations

import contextlib
import math
import re
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn


Progress = Callable[[str, int, int], None]
SUPPORTED_TRANSFORMERS = "5.18.0"
SOURCE_REVISION = "c202236235762e1c871ad0ccb60c8ee5ba337b9a"
_CONTROL_TOKEN = re.compile(r"<\|[^>]+\|>")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


@dataclass(frozen=True)
class ResidualWriter:
    """Linear map with Torch [output, input] weight orientation."""

    name: str
    module: nn.Linear
    layer: int
    kind: str


class Qwen35Adapter:
    """Assert the supported composite model and inventory every residual writer.

    ``strict_9b=False`` permits small architecture fixtures; such fixtures test
    implementation arithmetic only, not the full checkpoint or GPU kernels.
    """

    def __init__(self, model: nn.Module, strict_9b: bool = True):
        _require(not getattr(model, "is_quantized", False), "Edit unquantized floating-point weights only.")
        _require(hasattr(model, "model") and hasattr(model.model, "language_model"),
                 "Expected Qwen3_5ForConditionalGeneration.model.language_model.")
        self.model = model
        self.decoder = model.model.language_model
        self.config = model.config.text_config
        self.layers = self.decoder.layers
        self.embedding = self.decoder.embed_tokens
        self.hidden_size = int(self.config.hidden_size)
        self.n_layers = int(self.config.num_hidden_layers)
        _require(isinstance(self.embedding, nn.Embedding), "Text embedding must be nn.Embedding.")
        _require(len(self.layers) == self.n_layers, "Decoder layer count disagrees with config.")
        _require(tuple(self.embedding.weight.shape) == (self.config.vocab_size, self.hidden_size),
                 "Unexpected text embedding weight shape.")
        _require(len(self.config.layer_types) == self.n_layers, "Missing per-layer mixer types.")
        _require(not bool(getattr(model.config, "tie_word_embeddings", False)),
                 "This implementation preserves the untied LM head; tied embeddings are unsupported.")
        _require(hasattr(model, "lm_head") and isinstance(model.lm_head, nn.Linear),
                 "Expected an untied nn.Linear LM head.")
        _require(tuple(model.lm_head.weight.shape) == tuple(self.embedding.weight.shape),
                 "LM head shape does not match text vocabulary.")
        _require(self.embedding.weight is not model.lm_head.weight,
                 "LM head and input embedding unexpectedly share a parameter.")
        if self.embedding.weight.device.type != "meta":
            _require(self.embedding.weight.data_ptr() != model.lm_head.weight.data_ptr(),
                     "LM head and input embedding unexpectedly share storage.")
        if strict_9b:
            _require(model.__class__.__name__ == "Qwen3_5ForConditionalGeneration",
                     "Expected the official Qwen3_5ForConditionalGeneration class.")
            _require(self.n_layers == 32 and self.hidden_size == 4096,
                     "Expected Qwen3.5-9B's 32 layers and hidden size 4096.")
            _require(self.config.vocab_size == 248320 and self.config.intermediate_size == 12288,
                     "Unexpected Qwen3.5-9B vocabulary or MLP size.")
            expected = ["full_attention" if (i + 1) % 4 == 0 else "linear_attention"
                        for i in range(32)]
            _require(list(self.config.layer_types) == expected,
                     "Expected 24 linear-attention and 8 full-attention layers in 3:1 order.")

        self.mixers: list[nn.Module] = []
        self.writers: list[ResidualWriter] = []
        for i, layer in enumerate(self.layers):
            kind = self.config.layer_types[i]
            _require(getattr(layer, "block_type", None) == kind, f"Layer {i} mixer/config mismatch.")
            _require(hasattr(layer, "input_layernorm") and hasattr(layer, "post_attention_layernorm"),
                     f"Layer {i} lacks the expected sequential residual normalizations.")
            if kind == "linear_attention":
                mixer = layer.linear_attn
                output = mixer.out_proj
                output_name = f"model.language_model.layers.{i}.linear_attn.out_proj"
                mixer_in = self.config.linear_num_value_heads * self.config.linear_value_head_dim
            elif kind == "full_attention":
                mixer = layer.self_attn
                output = mixer.o_proj
                output_name = f"model.language_model.layers.{i}.self_attn.o_proj"
                mixer_in = self.config.num_attention_heads * self.config.head_dim
            else:
                raise ValueError(f"Unsupported residual mixer type {kind!r} at layer {i}.")
            self.mixers.append(mixer)
            _require(isinstance(output, nn.Linear), f"{output_name} must be nn.Linear.")
            _require(tuple(output.weight.shape) == (self.hidden_size, mixer_in),
                     f"Unexpected output projection shape at {output_name}.")
            down = layer.mlp.down_proj
            _require(isinstance(down, nn.Linear), f"Layer {i} MLP down projection must be nn.Linear.")
            _require(tuple(down.weight.shape) == (self.hidden_size, self.config.intermediate_size),
                     f"Unexpected MLP output projection shape at layer {i}.")
            self.writers.extend([
                ResidualWriter(output_name, output, i, kind),
                ResidualWriter(f"model.language_model.layers.{i}.mlp.down_proj", down, i, "mlp"),
            ])
        _require(len({id(w.module.weight) for w in self.writers}) == 2 * self.n_layers,
                 "Residual writer parameters unexpectedly share storage.")
        for writer in self.writers:
            _require(writer.module.weight.is_floating_point(), f"Non-floating weight: {writer.name}.")
            if writer.module.bias is not None:
                _require(tuple(writer.module.bias.shape) == (self.hidden_size,),
                         f"Unexpected output bias shape at {writer.name}.")

    @property
    def device(self) -> torch.device:
        return self.embedding.weight.device

    def metadata(self) -> dict[str, Any]:
        return {
            "model_class": self.model.__class__.__name__,
            "hidden_size": self.hidden_size,
            "num_hidden_layers": self.n_layers,
            "num_linear_attention_layers": list(self.config.layer_types).count("linear_attention"),
            "num_full_attention_layers": list(self.config.layer_types).count("full_attention"),
            "residual_writer_count": len(self.writers),
            "embedding_name": "model.language_model.embed_tokens",
            "weight_dtype": str(self.embedding.weight.dtype),
            "scope": "text-only; vision weights preserved; multimodal behavior unvalidated",
            "writer_names": [writer.name for writer in self.writers],
        }


def render_prompt(tokenizer: Any, prompt: str) -> str:
    """Render exactly one user message using the official non-thinking template."""
    _require(isinstance(prompt, str) and bool(prompt.strip()), "Prompts must be nonempty strings.")
    _require(_CONTROL_TOKEN.search(prompt) is None, "A prompt contains a reserved chat/control token.")
    result = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}], tokenize=False,
        add_generation_prompt=True, enable_thinking=False,
    )
    _require(isinstance(result, str), "Chat template did not return text.")
    _require(result.endswith("<think>\n\n</think>\n\n"),
             "Pinned tokenizer did not emit the expected empty no-thinking block.")
    return result


def shared_suffix(tokenizer: Any, prompts: Sequence[str]) -> tuple[int, ...]:
    """Return and verify postinstruction suffix IDs, without assuming its length.

    The suffix starts at the user <|im_end|> token.  Tokenizing each whole
    rendered prompt verifies the boundary instead of assuming separate-string
    tokenization has the same BPE segmentation as the complete prompt.
    """
    _require(len(prompts) > 0, "Cannot derive suffix from an empty prompt set.")
    marker = "<|im_end|>"
    first = render_prompt(tokenizer, prompts[0])
    start = first.rfind(marker)
    _require(start >= 0, "User end marker missing from the chat template.")
    suffix_text = first[start:]
    suffix = tuple(tokenizer.encode(suffix_text, add_special_tokens=False))
    _require(bool(suffix), "Postinstruction suffix tokenization is empty.")
    for prompt in prompts:
        rendered = render_prompt(tokenizer, prompt)
        _require(rendered.endswith(suffix_text), "Rendered prompts do not share a chat suffix.")
        ids = tokenizer.encode(rendered, add_special_tokens=False)
        _require(tuple(ids[-len(suffix):]) == suffix,
                 "Postinstruction suffix IDs differ when tokenized as a complete prompt.")
    return suffix


def tokenize_batch(
    tokenizer: Any,
    prompts: Sequence[str],
    device: torch.device | str | None = None,
    max_prompt_tokens: int = 2048,
) -> dict[str, Tensor]:
    """Left-pad text inputs, reject overlength prompts, and never truncate them."""
    _require(len(prompts) > 0, "Cannot tokenize an empty prompt batch.")
    _require(max_prompt_tokens > 0, "max_prompt_tokens must be positive.")
    tokenizer.padding_side = "left"
    _require(tokenizer.pad_token_id is not None, "Tokenizer must have a pad token configured.")
    encoded = tokenizer(
        [render_prompt(tokenizer, prompt) for prompt in prompts],
        padding=True, truncation=False, add_special_tokens=False, return_tensors="pt",
    )
    inputs = {name: encoded[name] for name in ("input_ids", "attention_mask")}
    _require(inputs["input_ids"].ndim == 2, "Tokenization returned unexpected tensor rank.")
    _require(inputs["input_ids"].shape == inputs["attention_mask"].shape,
             "Token IDs and attention mask have different shapes.")
    lengths = inputs["attention_mask"].sum(dim=1)
    _require(int(lengths.max()) <= max_prompt_tokens,
             f"Prompt exceeds {max_prompt_tokens} tokens; explicit review required instead of truncation.")
    _require(bool((inputs["attention_mask"][:, -1] == 1).all()), "Inputs are not left padded.")
    _require(bool((inputs["attention_mask"][:, 1:] >= inputs["attention_mask"][:, :-1]).all()),
             "Attention mask is not contiguous left padding.")
    if device is not None:
        inputs = {key: tensor.to(device) for key, tensor in inputs.items()}
    return inputs


def _hidden_input(args: tuple[Any, ...], kwargs: Mapping[str, Any]) -> Tensor:
    hidden = args[0] if args else kwargs.get("hidden_states")
    _require(isinstance(hidden, Tensor), "Residual hook did not receive hidden_states.")
    return hidden


@torch.inference_mode()
def collect_mean_activations(
    adapter: Qwen35Adapter,
    tokenizer: Any,
    prompts: Sequence[str],
    *,
    batch_size: int = 4,
    sites: Sequence[str] = ("pre",),
    max_prompt_tokens: int = 2048,
    progress: Progress | None = None,
) -> dict[str, Tensor]:
    """Stream suffix residual means as CPU float64 [position, layer, hidden]."""
    _require(batch_size > 0, "batch_size must be positive.")
    _require(bool(sites) and len(set(sites)) == len(sites), "Activation sites must be unique and nonempty.")
    _require(set(sites).issubset({"pre", "mid"}), "Supported activation sites are pre and mid.")
    suffix = shared_suffix(tokenizer, prompts)
    n_pos = len(suffix)
    totals = {site: torch.zeros(n_pos, adapter.n_layers, adapter.hidden_size, dtype=torch.float64)
              for site in sites}
    calls = {(site, layer): 0 for site in sites for layer in range(adapter.n_layers)}
    handles = []

    def make_hook(site: str, layer: int):
        def collect(module: nn.Module, args: tuple[Any, ...], kwargs: dict[str, Any]):
            hidden = _hidden_input(args, kwargs)
            _require(hidden.ndim == 3 and hidden.shape[-1] == adapter.hidden_size,
                     f"Unexpected residual shape at {site}, layer {layer}.")
            _require(hidden.shape[1] >= n_pos, "Sequence is shorter than the chat suffix.")
            selected = hidden[:, -n_pos:, :].detach().to(dtype=torch.float64)
            totals[site][:, layer].add_(selected.sum(dim=0).cpu())
            calls[(site, layer)] += 1
        return collect

    try:
        for site in sites:
            for layer, block in enumerate(adapter.layers):
                module = block if site == "pre" else block.post_attention_layernorm
                handles.append(module.register_forward_pre_hook(make_hook(site, layer), with_kwargs=True))
        adapter.model.eval()
        n_batches = math.ceil(len(prompts) / batch_size)
        for batch_index, start in enumerate(range(0, len(prompts), batch_size)):
            inputs = tokenize_batch(tokenizer, prompts[start:start + batch_size],
                                    adapter.device, max_prompt_tokens)
            actual_suffix = inputs["input_ids"][:, -n_pos:].cpu()
            expected_suffix = torch.tensor(suffix).expand(actual_suffix.shape[0], -1)
            _require(torch.equal(actual_suffix, expected_suffix), "Batched suffix alignment changed.")
            _require(bool((inputs["attention_mask"][:, -n_pos:] == 1).all()),
                     "A suffix position overlaps padding.")
            # logits_to_keep=1 avoids allocating [batch, whole_sequence, 248320].
            output = adapter.model(**inputs, use_cache=False, logits_to_keep=1, return_dict=True)
            del output
            for key, count in calls.items():
                _require(count == batch_index + 1, f"Residual hook was bypassed or repeated at {key}.")
            if progress is not None:
                progress("activation_batches", batch_index + 1, n_batches)
    finally:
        for handle in handles:
            handle.remove()
    for site, total in totals.items():
        total.div_(len(prompts))
        _require(bool(torch.isfinite(total).all()), f"Nonfinite activation mean at site {site}.")
    return totals


def extract_mean_differences(
    adapter: Qwen35Adapter,
    tokenizer: Any,
    harmful: Sequence[str],
    harmless: Sequence[str],
    **kwargs: Any,
) -> dict[str, Tensor]:
    """Return harmful minus harmless mean residuals, CPU float32 by site."""
    _require(shared_suffix(tokenizer, harmful) == shared_suffix(tokenizer, harmless),
             "Harmful and harmless prompts use different postinstruction suffixes.")
    harmful_means = collect_mean_activations(adapter, tokenizer, harmful, **kwargs)
    harmless_means = collect_mean_activations(adapter, tokenizer, harmless, **kwargs)
    differences = {site: (harmful_means[site] - harmless_means[site]).float()
                   for site in harmful_means}
    for site, tensor in differences.items():
        _require(bool(torch.isfinite(tensor).all()), f"Nonfinite mean difference at {site}.")
    return differences


def normalized_direction(direction: Tensor, device: torch.device | str | None = None) -> Tensor:
    """Normalize once in float32; reject zero and nonfinite candidate vectors."""
    _require(isinstance(direction, Tensor) and direction.ndim == 1,
             "A direction must be a one-dimensional tensor.")
    result = direction.detach().to(device=device, dtype=torch.float32)
    _require(bool(torch.isfinite(result).all()), "Direction contains nonfinite values.")
    norm = torch.linalg.vector_norm(result)
    _require(float(norm) > 1e-8, "Direction has zero or negligible norm.")
    return result / norm


def project_activations(activation: Tensor, direction: Tensor) -> Tensor:
    """Remove the direction in FP32, returning the original activation dtype."""
    _require(activation.is_floating_point() and activation.shape[-1] == direction.numel(),
             "Activation and refusal direction dimensions differ.")
    unit = normalized_direction(direction, activation.device)
    return _project_unit(activation, unit)


def _project_unit(activation: Tensor, unit: Tensor) -> Tensor:
    """Hot-path projection with an already validated, normalized device vector.

    Shape checks read metadata only.  Avoid GPU reductions, host transfers, or
    synchronization here: this runs 65 times per generated token on Qwen3.5-9B.
    """
    _require(activation.is_floating_point() and activation.shape[-1] == unit.numel(),
             "Activation and refusal direction dimensions differ.")
    _require(activation.device == unit.device, "Projection direction must be cached on activation device.")
    floating = activation.float()
    return (floating - (floating @ unit).unsqueeze(-1) * unit).to(dtype=activation.dtype)


@contextlib.contextmanager
def ablation_hooks(adapter: Qwen35Adapter, direction: Tensor) -> Iterator[dict[str, int]]:
    """Project text embeddings and every mixer/MLP output for all token positions.

    This is mathematically equivalent to editing every residual writer.  FP32
    hook arithmetic and BF16-rounded weight edits will have different rounding;
    compare their logits empirically before claiming numerical equivalence.
    Hooks attach to whole mixers/MLPs so fused output projections cannot bypass
    the intervention.  Full attention returns (hidden, attentions); linear
    attention and MLPs return hidden directly.
    """
    unit = normalized_direction(direction, adapter.device)
    _require(unit.numel() == adapter.hidden_size, "Direction dimension does not match residual stream.")
    device_units = {unit.device: unit}
    counts: dict[str, int] = {}
    handles = []
    completed = False

    def make_hook(name: str):
        counts[name] = 0
        def hook(module: nn.Module, args: tuple[Any, ...], output: Any):
            hidden = output[0] if isinstance(output, tuple) else output
            _require(isinstance(hidden, Tensor) and hidden.ndim == 3,
                     f"Unexpected output type or rank at {name}.")
            if hidden.device not in device_units:
                # Model-parallel use is rare here, but cache once per device.
                device_units[hidden.device] = unit.to(hidden.device)
            projected = _project_unit(hidden, device_units[hidden.device])
            counts[name] += 1
            return (projected, *output[1:]) if isinstance(output, tuple) else projected
        return hook

    try:
        handles.append(adapter.embedding.register_forward_hook(make_hook("embedding")))
        for layer, (block, mixer) in enumerate(zip(adapter.layers, adapter.mixers)):
            handles.append(mixer.register_forward_hook(make_hook(f"mixer.{layer}")))
            handles.append(block.mlp.register_forward_hook(make_hook(f"mlp.{layer}")))
        yield counts
        completed = True
    finally:
        for handle in handles:
            handle.remove()
        # An unused context is allowed, but once used every writer must execute.
        if completed and any(counts.values()):
            _require(all(count > 0 for count in counts.values()), "Some intervention hooks were bypassed.")


@contextlib.contextmanager
def add_direction_hook(
    adapter: Qwen35Adapter, direction: Tensor, layer: int, coefficient: float,
) -> Iterator[None]:
    """Add a unit direction to the pre residual at one zero-based decoder layer."""
    _require(0 <= layer < adapter.n_layers, "Direction-addition layer is out of range.")
    _require(math.isfinite(coefficient), "Addition coefficient must be finite.")
    unit = normalized_direction(direction, adapter.device)
    _require(unit.numel() == adapter.hidden_size, "Direction dimension does not match residual stream.")
    device_units = {unit.device: unit}
    def hook(module: nn.Module, args: tuple[Any, ...], kwargs: dict[str, Any]):
        hidden = _hidden_input(args, kwargs)
        if hidden.device not in device_units:
            device_units[hidden.device] = unit.to(hidden.device)
        added = (hidden.float() + coefficient * device_units[hidden.device]).to(hidden.dtype)
        if args:
            return (added, *args[1:]), kwargs
        return args, {**kwargs, "hidden_states": added}
    handle = adapter.layers[layer].register_forward_pre_hook(hook, with_kwargs=True)
    try:
        yield
    finally:
        handle.remove()


@torch.no_grad()
def project_weight_(
    parameter: Tensor, direction: Tensor, *, residual_axis: int, chunk_size: int = 2048,
) -> dict[str, Any]:
    """Chunked rank-one edit, preserving parameter storage, dtype, and device.

    Linear weights [hidden, input] use residual_axis=0: W' = (I-ddᵀ)W.
    Embeddings [vocabulary, hidden] use residual_axis=1: E' = E(I-ddᵀ).
    Chunk the other dimension so every residual vector is projected in one FP32
    operation.  Chunking hidden/output rows would use already edited data when
    computing dᵀW and produce an incorrect sequential projection.
    """
    _require(parameter.ndim == 2 and residual_axis in (0, 1), "Expected a matrix and residual_axis 0 or 1.")
    _require(parameter.is_floating_point() and parameter.device.type != "meta",
             "Projection requires materialized floating-point weights.")
    _require(chunk_size > 0, "chunk_size must be positive.")
    unit = normalized_direction(direction, parameter.device)
    _require(parameter.shape[residual_axis] == unit.numel(), "Weight residual dimension mismatch.")
    before2 = after2 = magnitude2 = 0.0
    other_size = parameter.shape[1 - residual_axis]
    for start in range(0, other_size, chunk_size):
        view = parameter[:, start:start + chunk_size] if residual_axis == 0 else parameter[start:start + chunk_size, :]
        original = view.float()
        component = unit @ original if residual_axis == 0 else original @ unit
        original_magnitude2 = float(original.double().square().sum())
        corrected = original - (unit[:, None] * component[None, :] if residual_axis == 0
                                else component[:, None] * unit[None, :])
        view.copy_(corrected.to(parameter.dtype))
        rounded = view.float()
        remaining = unit @ rounded if residual_axis == 0 else rounded @ unit
        before2 += float(component.double().square().sum())
        after2 += float(remaining.double().square().sum())
        magnitude2 += original_magnitude2
    return {
        "shape": list(parameter.shape), "dtype": str(parameter.dtype),
        "residual_axis": residual_axis,
        "directional_l2_before": math.sqrt(before2),
        "directional_l2_after": math.sqrt(after2),
        "directional_l2_after_over_weight_l2": math.sqrt(after2 / max(magnitude2, 1e-30)),
    }


@torch.no_grad()
def orthogonalize_weights(
    adapter: Qwen35Adapter,
    direction: Tensor,
    *,
    chunk_size: int = 2048,
    progress: Progress | None = None,
) -> dict[str, Any]:
    """Edit the embedding plus all 64 text residual maps and any output biases."""
    unit = normalized_direction(direction)
    _require(unit.numel() == adapter.hidden_size, "Direction dimension does not match residual stream.")
    targets = [("model.language_model.embed_tokens.weight", adapter.embedding.weight, 1)]
    targets.extend((writer.name + ".weight", writer.module.weight, 0) for writer in adapter.writers)
    details = []
    for index, (name, parameter, axis) in enumerate(targets):
        detail = project_weight_(parameter, unit, residual_axis=axis, chunk_size=chunk_size)
        details.append({"parameter": name, **detail})
        if progress is not None:
            progress("weight_matrices", index + 1, len(targets))
    biases = []
    for writer in adapter.writers:
        bias = writer.module.bias
        if bias is not None:
            bias.copy_(project_activations(bias, unit))
            biases.append(writer.name + ".bias")
    return {
        "edited_matrix_count": len(targets), "edited_bias_count": len(biases),
        "edited_bias_names": biases,
        "projection": "unit-direction orthogonal projection; FP32 arithmetic rounded to original weight dtype",
        "text_only": True, "vision_weights_preserved": True,
        "untied_lm_head_preserved": True, "matrices": details,
    }


@torch.inference_mode()
def next_token_logits(adapter: Qwen35Adapter, inputs: Mapping[str, Tensor]) -> Tensor:
    """Compute only final prompt-position logits; return CPU float32 [batch,vocab]."""
    _require(set(inputs).issubset({"input_ids", "attention_mask"}), "Only text token inputs are supported.")
    adapter.model.eval()
    output = adapter.model(**dict(inputs), use_cache=False, logits_to_keep=1, return_dict=True)
    _require(output.logits.ndim == 3 and output.logits.shape[1] == 1,
             "Model did not honor logits_to_keep=1.")
    logits = output.logits[:, 0].detach().float().cpu()
    _require(bool(torch.isfinite(logits).all()), "Next-token logits contain nonfinite values.")
    return logits


def compare_logits(reference: Tensor, current: Tensor) -> dict[str, float]:
    """Report empirical logit errors and KL(reference || current), not a pass claim."""
    _require(reference.shape == current.shape and reference.ndim == 2,
             "Expected matching [batch,vocabulary] logit matrices.")
    ref, cur = reference.detach().double().cpu(), current.detach().double().cpu()
    _require(bool(torch.isfinite(ref).all() and torch.isfinite(cur).all()), "Nonfinite comparison logits.")
    delta = cur - ref
    ref_logp, cur_logp = ref.log_softmax(-1), cur.log_softmax(-1)
    per_example_kl = (ref_logp.exp() * (ref_logp - cur_logp)).sum(-1)
    return {
        "mean_kl_reference_to_current": float(per_example_kl.mean()),
        "max_kl_reference_to_current": float(per_example_kl.max()),
        "logit_rms_error": float(delta.square().mean().sqrt()),
        "logit_max_abs_error": float(delta.abs().max()),
        "next_token_argmax_agreement": float((ref.argmax(-1) == cur.argmax(-1)).double().mean()),
    }


def run_core_self_checks() -> dict[str, Any]:
    """Small CPU regressions for arithmetic/coverage; not full-model validation.

    Uses a deliberately tiny, explicitly synthetic sequential-residual fixture
    to test hooks, biases, orientation, and preservation.  Its token mixers are
    ordinary linear maps, not GatedDeltaNet or full attention; passing these
    checks says nothing about the native Qwen kernels or refusal behavior.
    Random state is local and the complete fixture is released on return.
    """
    from types import SimpleNamespace

    generator = torch.Generator(device="cpu").manual_seed(20261003)
    hidden, intermediate, vocabulary = 8, 13, 29
    vector = torch.randn(hidden, generator=generator)
    direction = normalized_direction(vector)
    checks: dict[str, Any] = {"scope": "small synthetic CPU math fixture; native 9B validation still required"}
    errors = {}
    for dtype in (torch.float32, torch.bfloat16):
        for residual_axis, shape in ((0, (hidden, 17)), (1, (31, hidden))):
            original = torch.randn(*shape, generator=generator).to(dtype)
            working = original.clone()
            floating = original.float()
            reference = (floating - direction[:, None] * (direction @ floating)[None, :]
                         if residual_axis == 0
                         else floating - (floating @ direction)[:, None] * direction[None, :]).to(dtype)
            pointer = working.data_ptr()
            project_weight_(working, vector, residual_axis=residual_axis, chunk_size=3)
            _require(working.dtype == dtype and working.data_ptr() == pointer,
                     "Weight projection replaced parameter storage or dtype.")
            error = float((working.float() - reference.float()).abs().max())
            tolerance = 2e-6 if dtype == torch.float32 else 0.016
            _require(error <= tolerance, "Chunked projection disagrees with full matrix projection.")
            errors[f"{dtype}_axis{residual_axis}"] = error
    checks["chunked_vs_full_max_abs_errors"] = errors

    class FixtureMixer(nn.Module):
        def __init__(self, kind: str):
            super().__init__()
            self.kind = kind
            self.in_proj = nn.Linear(hidden, hidden, bias=True)
            output = nn.Linear(hidden, hidden, bias=True)
            if kind == "linear_attention":
                self.out_proj = output
            else:
                self.o_proj = output

        def forward(self, hidden_states: Tensor):
            hidden_states = torch.tanh(self.in_proj(hidden_states))
            output = self.out_proj(hidden_states) if self.kind == "linear_attention" else self.o_proj(hidden_states)
            return output if self.kind == "linear_attention" else (output, None)

    class FixtureMLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.up_proj = nn.Linear(hidden, intermediate, bias=True)
            self.down_proj = nn.Linear(intermediate, hidden, bias=True)

        def forward(self, hidden_states: Tensor):
            return self.down_proj(torch.tanh(self.up_proj(hidden_states)))

    class FixtureLayer(nn.Module):
        def __init__(self, kind: str):
            super().__init__()
            self.block_type = kind
            if kind == "linear_attention":
                self.linear_attn = FixtureMixer(kind)
            else:
                self.self_attn = FixtureMixer(kind)
            self.mlp = FixtureMLP()
            self.input_layernorm = nn.LayerNorm(hidden)
            self.post_attention_layernorm = nn.LayerNorm(hidden)

        def forward(self, hidden_states: Tensor):
            normalized = self.input_layernorm(hidden_states)
            update = (self.linear_attn(normalized) if self.block_type == "linear_attention"
                      else self.self_attn(normalized)[0])
            hidden_states = hidden_states + update
            return hidden_states + self.mlp(self.post_attention_layernorm(hidden_states))

    class FixtureModel(nn.Module):
        def __init__(self):
            super().__init__()
            types = ["linear_attention", "full_attention"]
            text_config = SimpleNamespace(
                hidden_size=hidden, num_hidden_layers=2, vocab_size=vocabulary,
                intermediate_size=intermediate, layer_types=types,
                linear_num_value_heads=2, linear_value_head_dim=4,
                num_attention_heads=2, head_dim=4,
            )
            self.config = SimpleNamespace(text_config=text_config, tie_word_embeddings=False)
            self.model = nn.Module()
            self.model.language_model = nn.Module()
            self.model.language_model.embed_tokens = nn.Embedding(vocabulary, hidden)
            self.model.language_model.layers = nn.ModuleList([FixtureLayer(kind) for kind in types])
            self.model.language_model.norm = nn.LayerNorm(hidden)
            self.model.visual = nn.Linear(3, 5)
            self.lm_head = nn.Linear(hidden, vocabulary, bias=False)

        def forward(self, input_ids: Tensor, attention_mask: Tensor | None = None,
                    logits_to_keep: int = 0, **kwargs: Any):
            states = self.model.language_model.embed_tokens(input_ids)
            for layer in self.model.language_model.layers:
                states = layer(states)
            states = self.model.language_model.norm(states)
            states = states[:, -logits_to_keep:] if logits_to_keep else states
            return SimpleNamespace(logits=self.lm_head(states))

    # fork_rng restores global RNG state after nn.Module constructors initialize.
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(20261003)
        model = FixtureModel().eval().requires_grad_(False)
    adapter = Qwen35Adapter(model, strict_9b=False)
    _require(len(adapter.writers) == 4, "Fixture omitted a residual writer.")
    inputs = {"input_ids": torch.tensor([[1, 2, 3], [4, 5, 6]]),
              "attention_mask": torch.ones(2, 3, dtype=torch.long)}
    preserved = {name: parameter.detach().clone() for name, parameter in model.named_parameters()
                 if name.startswith("model.visual.") or name.startswith("lm_head.") or "layernorm" in name
                 or name.startswith("model.language_model.norm.") or ".up_proj." in name or ".in_proj." in name}
    with ablation_hooks(adapter, direction) as coverage:
        hooked_logits = next_token_logits(adapter, inputs)
    _require(len(coverage) == 5 and all(count == 1 for count in coverage.values()),
             "Fixture hook coverage is incomplete.")
    report = orthogonalize_weights(adapter, vector, chunk_size=3)
    edited_logits = next_token_logits(adapter, inputs)
    max_error = float((hooked_logits - edited_logits).abs().max())
    _require(max_error < 1e-5, "FP32 fixture hooks and edited weights disagree.")
    _require(report["edited_matrix_count"] == 5 and report["edited_bias_count"] == 4,
             "Wrong fixture matrix or output bias edit count.")
    for name, parameter in model.named_parameters():
        if name in preserved:
            _require(torch.equal(parameter, preserved[name]), f"Unexpected modification of {name}.")
    for writer in adapter.writers:
        _require(abs(float(writer.module.bias.float() @ direction)) < 1e-6,
                 "Output bias was not projected.")
    watched = [adapter.embedding, *adapter.mixers, *(block.mlp for block in adapter.layers)]
    hooks_before = [len(module._forward_hooks) for module in watched]
    try:
        with ablation_hooks(adapter, direction):
            # Invoke just the embedding, then fail before other writers run.
            adapter.embedding(inputs["input_ids"])
            raise RuntimeError("intentional cleanup regression")
    except RuntimeError as error:
        _require(str(error) == "intentional cleanup regression", "Hook cleanup masked the original exception.")
    _require(hooks_before == [len(module._forward_hooks) for module in watched],
             "Intervention hooks leaked after an exception.")
    hooks_before_add = len(adapter.layers[0]._forward_pre_hooks)
    try:
        with add_direction_hook(adapter, direction, 0, 1.0):
            raise RuntimeError("intentional addition cleanup regression")
    except RuntimeError:
        pass
    _require(len(adapter.layers[0]._forward_pre_hooks) == hooks_before_add,
             "Direction addition hook leaked after an exception.")
    try:
        normalized_direction(torch.zeros(hidden))
    except ValueError:
        checks["zero_direction_rejected"] = True
    else:
        raise ValueError("Zero direction was accepted.")
    checks.update({
        "fixture_hook_vs_weight_edit_max_abs_logit_error": max_error,
        "fixture_hook_coverage": coverage,
        "fixture_output_biases_projected": True,
        "fixture_vision_norm_input_projection_lm_head_preserved": True,
        "fixture_hook_cleanup_on_exception": True,
        "passed": True,
    })
    return checks
