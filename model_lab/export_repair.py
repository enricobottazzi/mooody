"""Audited verification-only repair for Qwen3.5's manual cached-decode probe.

The frozen experiment, directions, selection, prompts, edit, and thresholds are
unchanged.  Only _Experiment.equivalence_rows is replaced at runtime.  Explicit
2D text positions cover precisely the input tokens passed to each forward call;
Qwen3_5TextModel expands them to all four text/M-RoPE position axes.

In Transformers 5.18, direct conditional-generation forwards can otherwise
reuse rope_deltas left by an earlier, larger generation batch.  For a new batch
of one, repeat_interleave(1 // 4) creates empty rotary tensors.  Its automatic
cached path also constructs positions over the complete attention mask rather
than slicing them to the current one-token input.  generate() normally prepares
and slices positions; the original manual verification loop did not.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import inspect
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch

from model_lab.abliteration import tokenize_batch


FROZEN_SOURCE_SHA256 = {
    "experiment.py": "f41dd03e970bf6386e5cf689f9c370e80f667aa13ebadc3c224d79f2ad6fc681",
    "abliteration.py": "ab3f891646c923447d86814565ffa34dae246e40bc058ec8cee3cd8ee4851444",
    "evaluation.py": "cbc4059d06ab03c7daf1b9b5b917364c04207a1723f6c9526b9992d13fc5722a",
    "modal_app.py": "62844aca0c2e15b1d3e0b70daa665c9dfec8e25fa777acdc23fe2b956594c536",
}
REPAIR_ID = "explicit_text_positions_for_export_equivalence_v1"
_ORIGINAL_LAYOUT = (
    "For each of three benign probes: one fresh prefill, then eight cached-decode "
    "positions on identical forced token sequences"
)


def text_position_ids(attention_mask: torch.Tensor, current_length: int | None = None) -> torch.Tensor:
    """Derive 2D text positions, slicing only current inputs for cached decode.

    The full attention mask stays intact for attention/cache masking.  Positions
    alone are sliced.  Metadata checks here avoid GPU-to-host synchronization in
    the actual verification loop; CPU self-checks cover padding and shape rules.
    """
    if attention_mask.ndim != 2 or attention_mask.shape[1] == 0:
        raise ValueError("Expected a nonempty [batch, sequence] attention mask")
    width = attention_mask.shape[1]
    length = width if current_length is None else current_length
    if not isinstance(length, int) or not 0 < length <= width:
        raise ValueError("Current token count must be positive and no greater than mask width")
    positions = attention_mask.long().cumsum(-1) - 1
    positions = positions.masked_fill(attention_mask == 0, 0)
    return positions[:, -length:]


def run_position_self_checks() -> dict[str, Any]:
    """Meaningful CPU shape/padding regressions, not native kernel validation."""
    mask = torch.tensor([[0, 0, 1, 1, 1], [0, 1, 1, 1, 1]], dtype=torch.long, device="cpu")
    expected = torch.tensor([[0, 0, 0, 1, 2], [0, 0, 1, 2, 3]], device="cpu")
    if not torch.equal(text_position_ids(mask), expected):
        raise ValueError("Left-padded prefill text positions are wrong")
    extended = torch.cat((mask, torch.ones((2, 1), dtype=mask.dtype, device="cpu")), dim=1)
    expected_decode = torch.tensor([[3], [4]], device="cpu")
    if not torch.equal(text_position_ids(extended, 1), expected_decode):
        raise ValueError("Cached decode positions are wrong or include the past sequence")
    # Qwen3_5TextModel's documented 2D -> 4D-axis expansion has one position
    # per current token, independent of any previously cached rope_deltas batch.
    one = torch.ones((1, 19), dtype=torch.long, device="cpu")
    current = text_position_ids(one, 1)
    expanded = current[None, ...].expand(4, current.shape[0], -1)
    if expanded.shape != (4, 1, 1) or not bool((expanded == 18).all()):
        raise ValueError("Single-token four-axis position expansion is wrong")
    rejected = 0
    for bad_mask, bad_length in ((torch.ones(5, device="cpu"), None),
                                 (torch.ones(1, 0, device="cpu"), None),
                                 (mask, 0), (mask, 6)):
        try:
            text_position_ids(bad_mask, bad_length)
        except ValueError:
            rejected += 1
    if rejected != 4:
        raise ValueError("Invalid position shapes or slice lengths were accepted")
    return {"passed": True, "scope": "small CPU position arithmetic; native cached verification still required",
            "left_padding_checked": True, "decode_slice_checked": True,
            "four_axis_single_token_shape": list(expanded.shape),
            "invalid_inputs_rejected": rejected}


@torch.inference_mode()
def repaired_equivalence_rows(self: Any, forced: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    """The frozen verification probe with explicit positions on every forward."""
    # Import the original frozen configuration; no duplicated threshold changes.
    from model_lab.experiment import CONFIG

    rows = []
    plan = []
    native_checks = []
    probes = [
        {"prompt_id": "equivalence:plants", "instruction": "Explain in two sentences how plants use sunlight to grow."},
        {"prompt_id": "equivalence:python", "instruction": "Describe three everyday uses of Python in one short paragraph."},
        {"prompt_id": "equivalence:parentheses", "instruction": "Explain in two sentences why adding parentheses can change an arithmetic expression."},
    ]
    steps = CONFIG["cached_decode_steps"]
    if steps != 8:
        raise ValueError("Frozen cached verification must retain exactly eight decode steps")
    if forced is not None:
        if len(forced) != len(probes):
            raise ValueError("Forced verification plan has the wrong probe count")
        for probe, item in zip(probes, forced):
            if item["prompt_id"] != probe["prompt_id"] or len(item["forced_tokens"]) != steps:
                raise ValueError("Forced verification plan differs from the frozen protocol")

    for index, probe in enumerate(probes):
        inputs = tokenize_batch(self.tokenizer, [probe["instruction"]], self.adapter.device,
                                CONFIG["max_prompt_tokens"])
        # A fresh text-only verification batch must not inherit M-RoPE deltas
        # from the prior multi-prompt generate() call.  Explicit positions below
        # also make this loop independent of that automatic inference path.
        self.model.model.rope_deltas = None
        prefill_positions = text_position_ids(inputs["attention_mask"])
        if prefill_positions.shape != inputs["input_ids"].shape:
            raise ValueError("Prefill positions do not cover precisely the prefill tokens")
        output = self.model(**inputs, position_ids=prefill_positions,
                            use_cache=True, logits_to_keep=1, return_dict=True)
        logits = output.logits[:, -1].detach().float().cpu()
        cache = output.past_key_values
        if cache is None or int(cache.get_seq_length()) != inputs["input_ids"].shape[1]:
            raise ValueError("Fresh prefill cache length differs from the input token count")
        rows.append(logits)
        generated = []
        token = int(logits.argmax(-1)) if forced is None else forced[index]["forced_tokens"][0]
        mask = inputs["attention_mask"]
        prompt_length = inputs["input_ids"].shape[1]
        for step in range(steps):
            generated.append(token)
            mask = torch.cat((mask, torch.ones((1, 1), dtype=mask.dtype, device=mask.device)), dim=1)
            decode_positions = text_position_ids(mask, current_length=1)
            if decode_positions.shape != (1, 1):
                raise ValueError("Cached decode must supply one text position for one input token")
            output = self.model(input_ids=torch.tensor([[token]], device=self.adapter.device),
                                attention_mask=mask, position_ids=decode_positions,
                                past_key_values=cache, use_cache=True,
                                logits_to_keep=1, return_dict=True)
            cache = output.past_key_values
            if int(cache.get_seq_length()) != prompt_length + step + 1:
                raise ValueError("Cached verification did not advance by exactly one input token")
            logits = output.logits[:, -1].detach().float().cpu()
            rows.append(logits)
            if step + 1 < steps:
                token = int(logits.argmax(-1)) if forced is None else forced[index]["forced_tokens"][step + 1]
        plan.append({"prompt_id": probe["prompt_id"], "forced_tokens": generated})
        if forced is None:
            # Independently exercise the native generation preparation path.
            # This checks that manually supplied positions yield its exact
            # greedy trajectory, rather than comparing two identically wrong
            # manual paths.  An active caller ablation context applies to both.
            native = self.model.generate(
                **inputs, do_sample=False, max_new_tokens=steps, use_cache=True,
                past_key_values=None, pad_token_id=self.tokenizer.pad_token_id,
            )[:, prompt_length:]
            native_ids = native[0].detach().cpu().tolist()
            eos = self.model.generation_config.eos_token_id
            eos_ids = {eos} if isinstance(eos, int) else set(eos or [])
            native_end = next((position for position, value in enumerate(native_ids)
                               if value in eos_ids), None)
            if native_end is not None:
                native_ids = native_ids[:native_end + 1]
            # A single unpadded sequence cannot acquire batch-finish padding;
            # trimming at EOS nevertheless ensures padding is never mistaken
            # for a generated token if that assumption changes.
            if len(native_ids) != steps:
                raise ValueError(
                    f"Native cached verification ended before all {steps} positions at {probe['prompt_id']}; "
                    "post-EOS manual positions cannot establish generation agreement"
                )
            if native_ids != generated:
                raise ValueError(f"Explicit cached positions disagree with native greedy generation at {probe['prompt_id']}")
            native_checks.append({"prompt_id": probe["prompt_id"], "checked_tokens": steps,
                                  "exact_token_agreement": True, "early_eos": False})
            del native
        del output, cache, inputs
    if forced is None:
        from model_lab.evaluation import atomicdump_json
        atomicdump_json(self.out / "cache_position_native_check.json", {
            "repair_id": REPAIR_ID, "passed": True, "probe_count": len(native_checks),
            "checked_generated_tokens": sum(item["checked_tokens"] for item in native_checks),
            "fresh_native_attention_and_recurrent_cache": True,
            "comparison": "Exact greedy token IDs from manually positioned cached calls versus model.generate preparation",
            "caller_intervention_applies_to_both_paths": True, "probes": native_checks,
        })
        self.persist(force=True)
    return {"logits": torch.cat(rows), "plan": plan, "layout": _ORIGINAL_LAYOUT}


def install_verification_repair(
    run_settings: Mapping[str, Any] | str | Path,
    *,
    provenance_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Assert frozen sources/settings, then replace only the verification method.

    The caller persists this returned audit plus this module's source.  Baseline,
    extraction, selection, projection, final evaluation, and thresholds continue
    to execute the unchanged original code and keep its provenance fingerprint.
    """
    from model_lab import experiment

    settings = (json.loads(Path(run_settings).read_text(encoding="utf-8"))
                if isinstance(run_settings, (str, Path)) else dict(run_settings))
    source_root = Path(experiment.__file__).parent
    actual = {}
    for name, expected in FROZEN_SOURCE_SHA256.items():
        digest = hashlib.sha256((source_root / name).read_bytes()).hexdigest()
        if digest != expected:
            raise ValueError(f"Frozen source file changed before verification repair: {name}")
        recorded = settings["source_code_hashes"].get(name)
        if recorded is not None and digest != recorded:
            raise ValueError(f"Run settings source provenance mismatch: {name}")
        if provenance_dir is not None and (Path(provenance_dir) / name).is_file():
            preserved = hashlib.sha256((Path(provenance_dir) / name).read_bytes()).hexdigest()
            if preserved != digest:
                raise ValueError(f"Preserved provenance source mismatch: {name}")
        actual[name] = digest
    if experiment.CONFIG != settings["configuration"]:
        raise ValueError("The frozen experiment configuration or thresholds changed")
    versions = {name: importlib.metadata.version(name) for name in settings["versions"]}
    if versions != settings["versions"]:
        raise ValueError("Runtime package versions changed since the frozen experiment")
    if versions.get("transformers") != "5.18.0" or versions.get("torch", "").split("+")[0] != "2.10.0":
        raise ValueError("This repair targets exactly Transformers 5.18.0 and Torch 2.10.0")
    original = experiment._Experiment.equivalence_rows
    if original is repaired_equivalence_rows:
        raise ValueError("Verification repair is already installed; do not nest or repeat replacements")
    if original.__module__ != "model_lab.experiment":
        raise ValueError("Original verification method was already replaced by another patch")
    original_source_hash = hashlib.sha256(inspect.getsource(original).encode()).hexdigest()
    repair_source_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    self_checks = run_position_self_checks()
    experiment._Experiment.equivalence_rows = repaired_equivalence_rows
    return {
        "repair_id": REPAIR_ID, "replaced_method": "_Experiment.equivalence_rows",
        "original_file_sha256": actual, "original_method_source_sha256": original_source_hash,
        "repair_source_sha256": repair_source_hash, "runtime_versions_unchanged": versions,
        "position_self_checks": self_checks,
        "changes": ["Explicit 2D text position_ids on every prefill and one-token cached forward",
                    "Clear stale multimodal rope_deltas before each fresh text-only prefill"],
        "baseline_selection_weight_edit_final_protocol_unchanged": True,
        "original_verification_probes_steps_forced_plans_layout_thresholds_unchanged": True,
    }
