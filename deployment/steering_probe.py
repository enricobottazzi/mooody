"""Bounded native regression probe for the reported steered repetition.

Receipts contain numeric metrics and hashes, never generated response text or
token IDs. This is one prompt-level regression check, not a persona evaluation.
"""

from __future__ import annotations

from contextlib import contextmanager
import hashlib
import math
import re
import time

from deployment.core import AXES, MAX_MOOD_COEFFICIENT, MODEL_ID, MODEL_REVISION, bounded_messages
from deployment.formatting import post_instruction_start


PROMPT = "what's on your mind"


def _longest_run(values):
    longest = current = 0
    previous = object()
    for value in values:
        current = current + 1 if value == previous else 1
        longest = max(longest, current)
        previous = value
    return longest


def repetition_metrics(token_ids, text):
    """Measure consecutive loops of periods 1..4 without exposing their content."""
    words = re.findall(r"[^\W_]+(?:['’][^\W_]+)*", text.casefold().replace("’", "'"))
    loops = {}
    for period in range(1, 5):
        longest = 0
        for start in range(max(0, len(token_ids) - 2 * period + 1)):
            end = start + period
            while end < len(token_ids) and token_ids[end] == token_ids[end - period]:
                end += 1
            span = end - start
            if span >= 2 * period:
                longest = max(longest, span)
        loops[str(period)] = {
            "max_repeated_span_tokens": longest,
            "complete_repetitions": longest // period,
        }
    word_run = _longest_run(words)
    degenerate = word_run >= 8 or any(
        value["complete_repetitions"] >= 8 and value["max_repeated_span_tokens"] >= 12
        for value in loops.values()
    )
    return {
        "content_tokens": len(token_ids), "characters": len(text), "words": len(words),
        "unique_content_token_fraction": len(set(token_ids)) / max(1, len(token_ids)),
        "max_repeated_token_run": _longest_run(token_ids),
        "max_repeated_word_run": word_run, "token_ngram_loops": loops,
        "degenerate_loop_detected": degenerate,
        "response_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
    }


@contextmanager
def _diagnostic_offsets(steering, *, direct=False, zero_first=False):
    """Diagnostic override, restoring the exact previous instance state."""
    sentinel = object()
    previous = steering.__dict__.get("offsets", sentinel)

    vectors = steering.vectors if direct else steering.incremental_vectors
    if zero_first:
        vectors = vectors.clone()
        vectors[0].zero_()

    def diagnostic(coefficients):
        if len(coefficients) != len(AXES) or any(
            type(level) not in (int, float) or not math.isfinite(level) or not -2 <= level <= 2
            for level in coefficients
        ):
            raise ValueError("Expected six finite diagnostic coefficients between -2 and 2")
        alpha = steering.torch.tensor(coefficients, dtype=steering.torch.float32,
                                      device=steering.vectors.device)
        return steering.torch.einsum("m,lmh->lh", alpha, vectors)

    steering.offsets = diagnostic
    try:
        yield
    finally:
        if previous is sentinel:
            del steering.offsets
        else:
            steering.offsets = previous


def _content_ids(ids, steering):
    excluded = set(steering.excluded_content_token_ids)
    start, end = steering.thinking_token_ids
    content, thinking, unexpected = [], False, False
    for token in ids:
        if start is not None and token == start:
            thinking, unexpected = True, True
        elif end is not None and token == end:
            thinking, unexpected = False, True
        elif not thinking and token not in excluded:
            content.append(token)
    return content, unexpected


def probe_runtime(runtime, *, max_new_tokens=128, compare_direct=True, include_combined=False,
                  diagnostic_only=False, include_trait_endpoints=False):
    """Use one already-loaded native runtime; caller serializes this invocation.

    Balanced and incremental cases determine ``passed``. The previous direct
    method is diagnostic and may fail without failing the current method.
    Diagnostic-only mode also compares fractional strengths, disabling the first
    increment, and an alternate benign prompt. These cases never change the
    production coefficient rules. The caller may inspect a failed diagnostic
    receipt, while the normal release gate continues to require the unchanged
    balanced and the configured maximum-strength incremental cases to pass.
    No generation configuration, model weights, or shared vectors are changed.
    """
    if type(max_new_tokens) is not int or not 1 <= max_new_tokens <= 128:
        raise ValueError("The regression probe must be bounded to at most 128 tokens")
    if runtime.steering.metadata().get("steering_method") != "paper_incremental_all_layers":
        raise ValueError("The current runtime must use paper_incremental_all_layers")
    torch = runtime.torch
    zero = [0] * len(AXES)
    arousal = list(zero)
    arousal[AXES.index("sexual_arousal")] = MAX_MOOD_COEFFICIENT
    maximum_label = f"{MAX_MOOD_COEFFICIENT:g}".replace(".", "_")
    arousal_name = f"incremental_sexual_arousal_{maximum_label}"
    cases = [("balanced", zero, False, False, PROMPT),
             (arousal_name, arousal, False, False, PROMPT)]
    gate_names = ["balanced", arousal_name]
    if diagnostic_only:
        for name, strength, zero_first in (("2", 2, False), ("1", 1, False), ("0_5", 0.5, False),
                                            ("0_25", 0.25, False), ("2_first_increment_disabled", 2, True)):
            coefficients = list(zero)
            coefficients[AXES.index("sexual_arousal")] = strength
            if not zero_first and any(existing[1] == coefficients and not existing[2] and not existing[3]
                                      and existing[4] == PROMPT for existing in cases):
                continue
            cases.append((f"incremental_sexual_arousal_{name}", coefficients, False, zero_first, PROMPT))
        diagnostic_arousal = list(zero)
        diagnostic_arousal[AXES.index("sexual_arousal")] = 2
        cases.append(("incremental_sexual_arousal_2_sunrise", diagnostic_arousal, False, False,
                      "Describe a sunrise in one sentence."))
    if include_trait_endpoints:
        for index, trait in enumerate(AXES):
            for sign, direction in ((1, "positive"), (-1, "negative")):
                coefficients = list(zero)
                coefficients[index] = sign * MAX_MOOD_COEFFICIENT
                if any(existing[1] == coefficients and not existing[2] and not existing[3]
                       and existing[4] == PROMPT for existing in cases):
                    continue
                name = f"incremental_{trait}_{direction}_{maximum_label}"
                cases.append((name, coefficients, False, False, PROMPT))
                gate_names.append(name)
    if include_combined or include_trait_endpoints:
        name = f"incremental_all_traits_{maximum_label}"
        cases.append((name, [MAX_MOOD_COEFFICIENT] * len(AXES), False, False, PROMPT))
        gate_names.append(name)
    if compare_direct:
        direct_arousal = list(zero)
        direct_arousal[AXES.index("sexual_arousal")] = 2
        cases.append(("previous_direct_sexual_arousal_2", direct_arousal, True, False, PROMPT))
    results = {}
    initial_hooks = [len(layer._forward_hooks) for layer in runtime.steering.layers]
    initial_pre_hooks = len(runtime.model._forward_pre_hooks)
    try:
        for name, coefficients, direct, zero_first, prompt in cases:
            runtime._clear_request_cache()
            inputs, removed = bounded_messages(runtime.tokenizer, [{"role": "user", "content": prompt}], coefficients)
            if removed:
                raise RuntimeError("The bounded probe unexpectedly removed messages")
            boundary = post_instruction_start(runtime.tokenizer, inputs)
            prompt_tokens = int(inputs["input_ids"].shape[-1])
            inputs = inputs.to(runtime.model.device)
            started = time.monotonic()
            from contextlib import nullcontext
            override = direct or zero_first or (diagnostic_only and any(
                type(value) is float or abs(value) > MAX_MOOD_COEFFICIENT for value in coefficients
            ))
            offset_scope = _diagnostic_offsets(runtime.steering, direct=direct, zero_first=zero_first) if override else nullcontext()
            try:
                with offset_scope, torch.inference_mode(), runtime.steering.apply(coefficients, boundary) as applied:
                    output = runtime.model.generate(
                        **inputs, do_sample=False, max_new_tokens=max_new_tokens, use_cache=True,
                        logits_to_keep=1, pad_token_id=runtime.tokenizer.pad_token_id,
                    )
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                generated = output[0, prompt_tokens:].detach().cpu().tolist()
                text = runtime.tokenizer.decode(generated, skip_special_tokens=True)
                content, unexpected = _content_ids(generated, runtime.steering)
                unexpected = unexpected or "<think>" in text or "</think>" in text
                metrics = repetition_metrics(content, text)
                eos = runtime.model.generation_config.eos_token_id
                eos = {eos} if type(eos) is int else set(eos or [])
                passed = bool(content) and bool(text.strip()) and not unexpected and not metrics["degenerate_loop_detected"]
                results[name] = {
                    "coefficients": list(coefficients), "moods_applied": bool(applied),
                    "steering_method": "direct_raw_all_layers" if direct else
                        "paper_incremental_all_layers" if any(coefficients) else "none",
                    "first_increment_disabled": bool(zero_first),
                    "diagnostic_offset_override": bool(override),
                    "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                    "generated_tokens": len(generated), "max_new_tokens": max_new_tokens,
                    "finish_reason": "stop" if generated and generated[-1] in eos else
                        "length" if len(generated) >= max_new_tokens else "stop",
                    "generation_seconds": time.monotonic() - started,
                    "thinking_detected": bool(unexpected), "passed": bool(passed), **metrics,
                }
            finally:
                runtime._clear_request_cache()
                if ([len(layer._forward_hooks) for layer in runtime.steering.layers] != initial_hooks
                        or len(runtime.model._forward_pre_hooks) != initial_pre_hooks):
                    raise RuntimeError("The regression probe left steering hooks installed")
    finally:
        runtime._clear_request_cache()
    return {
        "scope": "bounded_steering_diagnostic" if diagnostic_only else "single_prompt_repetition_regression",
        "diagnostic_only": bool(diagnostic_only), "model_id": MODEL_ID,
        "configured_maximum_coefficient": MAX_MOOD_COEFFICIENT,
        "individual_trait_endpoints_tested": bool(include_trait_endpoints),
        "model_revision": MODEL_REVISION,
        "prompt_sha256": hashlib.sha256(PROMPT.encode()).hexdigest(),
        "decoding": "native_greedy", "thinking": False,
        "system_prompt_present": False,
        "mood_conditioning": "vectors_with_prompt_assistance",
        "bank_tensor_sha256": runtime.steering.metadata().get("mood_vectors_tensor_sha256"),
        "response_text_omitted": True,
        "loop_thresholds": {"repeated_word_run": 8, "token_ngram_periods": [1, 2, 3, 4],
                            "complete_repetitions": 8, "minimum_repeated_span_tokens": 12},
        "cases": results,
        "release_gate_cases": gate_names,
        "passed": all(results[name]["passed"] for name in gate_names),
    }
