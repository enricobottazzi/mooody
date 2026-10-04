"""Audited batch-eight continuation, preserving every existing generation file.

This module never replaces the frozen serial generator. New records retain
their original condition IDs/seeds and exact unpadded prompt/generated IDs.
Incomplete/failed batch attempts block automatic reruns of missing records.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import inspect
import json
import os
from pathlib import Path
import re
import time
import uuid
from typing import Callable

from model_lab.persona_extraction import (
    MODEL_ID, MODEL_REVISION, atomic_json, checkpoint_metadata, classify_response,
    conditions, content_indices, file_hash, fingerprint, freeze_stage,
    frozen_settings, load_model, log, progress, read_json, record_path, render_tokens,
)

STAGE = "generation_batched_v1"
BATCH_SIZE = 8
SAMPLING_MODULE = Path(__file__).with_name("seeded_sampling.py")
GENERATION_CONFIG_SHA256 = "d0bb1d295ae3e46bb02d6d1a4dc59d689ec822d7479a32835f6945db85c3512a"
ARITHMETIC_NOTE = (
    "Batched BF16 arithmetic may produce different sampled text from serial generation, "
    "even with the same condition seed. Each condition has an independent random stream; "
    "every saved token sequence is replayed exactly for extraction."
)


def pending_conditions(root: Path, rows: list[dict], settings: dict) -> list[dict]:
    pending = []
    for row in rows:
        path = record_path(root, row["trait"], row["response_id"])
        if not path.exists():
            pending.append(row)
            continue
        saved = read_json(path)
        if (saved.get("condition_sha256") != fingerprint(row)
                or saved.get("settings_sha256") != fingerprint(settings)
                or saved.get("model_id") != MODEL_ID or saved.get("model_revision") != MODEL_REVISION
                or saved.get("status") not in ("complete", "generation_error")):
            raise ValueError("Existing serial/batched record provenance mismatch")
        # Errors, capped completions and successful completions are all kept.
        # Retrying a failed condition requires an explicit separately audited repair.
    pending_ids = {row["response_id"] for row in pending}
    trait = rows[0]["trait"] if rows else ""
    for path in (root / "generation_batches" / trait).glob("*.json"):
        batch = read_json(path)
        if batch.get("status") != "complete" and pending_ids.intersection(batch.get("condition_ids", [])):
            raise RuntimeError("An incomplete/failed batch owns missing conditions; explicit repair required")
    return pending


def exclusive_record(path: Path, record: dict) -> None:
    """Never overwrite a saved row; commit only after this handle is closed.

    Modal v1 lacks distributed file locking/hard links, so the coordinator must
    stop serial writers and launch one worker per trait. Exclusive creation
    also protects against duplicate paths within a mounted filesystem view.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(record, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    with path.open("x", encoding="utf-8") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def resolved_sampling(model, tokenizer, checkpoint: Path) -> tuple[object, dict]:
    from transformers.generation.utils import GenerationMixin
    from model_lab.seeded_sampling import validate_supported_config

    config_hash = file_hash(checkpoint / "generation_config.json")
    if config_hash != GENERATION_CONFIG_SHA256:
        raise ValueError("Generation configuration differs from the audited checkpoint")
    kwargs = {"do_sample": True, "temperature": 1.0, "top_p": 1.0,
              "max_new_tokens": 1000, "use_cache": True, "pad_token_id": tokenizer.pad_token_id}
    config, unused = model._prepare_generation_config(None, **kwargs)
    if unused:
        raise ValueError("Unexpected forwarded extraction generation configuration")
    config = validate_supported_config(config)
    if (config.do_sample, config.temperature, config.top_p, config.top_k,
        config.repetition_penalty, config.max_new_tokens, config.use_cache) != (True, 1.0, 1.0, 50, 1.0, 1000, True):
        raise ValueError("Native resolved sampling differs from the verified frozen collection")
    receipt = {"schema_version": 1, "model_id": MODEL_ID, "model_revision": MODEL_REVISION,
               "collection_kwargs": kwargs, "enable_thinking": False,
               "resolved_config": config.to_dict(), "resolved_config_sha256": fingerprint(config.to_dict()),
               "resolution_boundary": "before_prompt_dependent_max_length_adjustment",
               "checkpoint_generation_config_sha256": config_hash,
               "versions": {name: importlib.metadata.version(name) for name in
                            ("torch", "transformers", "flash-linear-attention", "safetensors")},
               "native_source_sha256": {name: hashlib.sha256(inspect.getsource(function).encode()).hexdigest()
                                        for name, function in {
                                            "prepare_generation_config": GenerationMixin._prepare_generation_config,
                                            "get_logits_processor": GenerationMixin._get_logits_processor,
                                            "sample": GenerationMixin._sample}.items()},
               "baseline_seed_strategy": "global_cuda_rng_reset_per_condition",
               "continuation_seed_strategy": "independent_cuda_generator_per_condition",
               "batch_size": BATCH_SIZE, "batch_ordering": "polarity_then_canonical_pair_question",
               "arithmetic_note": ARITHMETIC_NOTE,
               "legacy_record_config_note": "Serial effective_generation_config records preserve pre-resolution overrides; this receipt reconstructs actual native defaults."}
    return config, receipt


def freeze_batched_stage(root: Path, sampling: dict) -> dict:
    baseline = freeze_stage(root, "generation")
    payload = {"stage": STAGE, "module_sha256": file_hash(Path(__file__)),
               "sampling_module_sha256": file_hash(SAMPLING_MODULE),
               "original_generation_implementation_sha256": baseline["implementation_sha256"],
               "sampling_receipt_sha256": fingerprint(sampling), "batch_size": BATCH_SIZE}
    saved = {**payload, "implementation_sha256": fingerprint(payload)}
    path = root / "implementations" / f"{STAGE}.json"
    if path.exists() and read_json(path) != saved:
        raise ValueError("Frozen batched continuation implementation/config changed")
    sampling_path = root / "generation_sampling.json"
    if sampling_path.exists() and read_json(sampling_path) != sampling:
        raise ValueError("Mixed traits resolved different native sampling configurations")
    atomic_json(path, saved)
    atomic_json(sampling_path, sampling)
    for source, target in ((Path(__file__), path.with_suffix(".py")),
                           (SAMPLING_MODULE, path.with_name(f"{STAGE}_sampling.py"))):
        # Unique temporary snapshots use the same atomic JSON helper's pattern.
        temporary = target.with_name(f".{target.name}.{uuid.uuid4().hex}.tmp")
        temporary.write_bytes(source.read_bytes())
        os.replace(temporary, target)
    return saved


def completion_record(tokenizer, row: dict, rendered: str, prompt_ids: list[int],
                      padded_output: list[int], padded_width: int, eos_ids: set[int],
                      settings: dict, sampling: dict, stage: dict, batch_id: str,
                      batch_size: int, seconds: float) -> dict:
    if padded_output[padded_width - len(prompt_ids):padded_width] != prompt_ids:
        raise ValueError("Generated sequence changed original unpadded prompt IDs")
    generated = padded_output[padded_width:]
    end = next((index for index, token in enumerate(generated) if token in eos_ids), None)
    ids = generated[:end + 1] if end is not None else generated
    if not ids or len(ids) > 1000 or (end is None and len(ids) != 1000):
        raise ValueError("Unexpected native generation length/stop condition")
    indices, unexpected = content_indices(tokenizer, ids, eos_ids)
    response = tokenizer.decode([ids[index] for index in indices], skip_special_tokens=False).strip()
    return {**row, "condition_sha256": fingerprint(row), "settings_sha256": fingerprint(settings),
            "model_id": MODEL_ID, "model_revision": MODEL_REVISION,
            "generation_settings": {**sampling["collection_kwargs"], "enable_thinking": False},
            "effective_generation_config": sampling["resolved_config"],
            "effective_generation_config_kind": "native_resolved_before_prompt_length",
            "rendered_prompt": rendered, "prompt_token_ids": prompt_ids,
            "response_token_ids": ids, "response_content_indices": indices,
            "response": response, "raw_response": tokenizer.decode(ids, skip_special_tokens=False),
            "status": "complete", "generated_token_count": len(ids), "pooled_token_count": len(indices),
            "unexpected_thinking": unexpected,
            "stop_reason": "eos" if end is not None else "token_limit_truncation",
            "generation_seconds": seconds, "generation_seconds_scope": "shared_batch_wall_time",
            "generation_implementation": STAGE,
            "generation_implementation_sha256": stage["implementation_sha256"],
            "generation_batch_id": batch_id, "generation_batch_size": batch_size,
            "sampling_configuration_sha256": sampling["resolved_config_sha256"],
            **classify_response(response)}


def generate_batched_trait(root: Path, inputs: Path, checkpoint: Path, trait: str,
                           commit: Callable = lambda: None, limit: int = 0) -> dict:
    import copy
    import torch
    from transformers import LogitsProcessorList
    from model_lab.seeded_sampling import IndependentRowSampler, left_padded_inputs

    settings, protocol, traits = frozen_settings(root, inputs)
    rows = conditions(trait, traits[trait], protocol)
    pending = pending_conditions(root, rows, settings)
    pending = sorted(pending, key=lambda row: (row["polarity"] != "positive", row["pair_id"], row["question_id"]))
    if limit:
        pending = pending[:limit]
    if not pending:
        return progress(root, inputs, trait)
    tokenizer, model, adapter = load_model(checkpoint)
    metadata = checkpoint_metadata(checkpoint, adapter)
    model_path = root / "models" / f"{trait}.json"
    if model_path.exists() and read_json(model_path) != metadata:
        raise ValueError("Continuation differs from original checkpoint/tokenizer metadata")
    if not model_path.exists():
        atomic_json(model_path, metadata)
    config, sampling = resolved_sampling(model, tokenizer, checkpoint)
    stage = freeze_batched_stage(root, sampling)
    commit()
    eos = config.eos_token_id
    eos_ids = {eos} if isinstance(eos, int) else set(eos or [])
    for start in range(0, len(pending), BATCH_SIZE):
        chunk = pending[start:start + BATCH_SIZE]
        if any(record_path(root, trait, row["response_id"]).exists() for row in chunk):
            raise RuntimeError("A selected record appeared; stop concurrent generation writers")
        rendered_and_ids = [render_tokens(tokenizer, row["messages"]) for row in chunk]
        input_ids, mask = left_padded_inputs([ids for _, ids in rendered_and_ids],
                                            tokenizer.pad_token_id, adapter.device)
        sampler = IndependentRowSampler([row["seed"] for row in chunk], config, adapter.device)
        batch_id = uuid.uuid4().hex
        batch_path = root / "generation_batches" / trait / f"{batch_id}.json"
        batch = {"batch_id": batch_id, "stage": STAGE, "status": "started",
                 "implementation_sha256": stage["implementation_sha256"],
                 "condition_ids": [row["response_id"] for row in chunk],
                 "condition_seeds": [row["seed"] for row in chunk],
                 "sampling_configuration_sha256": sampling["resolved_config_sha256"],
                 "started_at_unix": time.time()}
        atomic_json(batch_path, batch)
        commit()
        started = time.monotonic()
        try:
            model.model.rope_deltas = None
            with torch.inference_mode():
                output = model.generate(input_ids=input_ids, attention_mask=mask,
                                        generation_config=copy.deepcopy(config), logits_to_keep=1,
                                        logits_processor=LogitsProcessorList([sampler]))
            torch.cuda.synchronize()
            seconds = time.monotonic() - started
            sequences = output.detach().cpu().tolist()
            if len(sequences) != len(chunk):
                raise ValueError("Native batch did not return one sequence per condition")
            records = [completion_record(tokenizer, row, rendered, ids, sequence, input_ids.shape[1],
                                         eos_ids, settings, sampling, stage, batch_id, len(chunk), seconds)
                       for row, (rendered, ids), sequence in zip(chunk, rendered_and_ids, sequences)]
            if any(record_path(root, trait, row["response_id"]).exists() for row in chunk):
                raise RuntimeError("A record appeared during generation; refusing overwrite")
            for row, record in zip(chunk, records):
                exclusive_record(record_path(root, trait, row["response_id"]), record)
            batch.update(status="complete", finished_at_unix=time.time(), seconds=seconds,
                         aggregate_tokens_per_second=sum(record["generated_token_count"] for record in records) / seconds,
                         record_file_sha256={row["response_id"]: file_hash(record_path(root, trait, row["response_id"]))
                                             for row in chunk})
            atomic_json(batch_path, batch)
            commit()
            log("batched_generation_progress", trait=trait, saved=min(start + BATCH_SIZE, len(pending)),
                pending=len(pending), batch_size=len(chunk), aggregate_tokens_per_second=batch["aggregate_tokens_per_second"])
            del output, input_ids, mask
        except Exception as error:
            batch.update(status="failed", finished_at_unix=time.time(), error_type=type(error).__name__,
                         error=str(error)[:1500])
            atomic_json(batch_path, batch)
            commit()
            raise
        finally:
            model.model.rope_deltas = None
    return progress(root, inputs, trait)


def public_generation_provenance(root: Path, inputs: Path) -> dict:
    """Aggregate execution receipts without exposing any prompts or token IDs."""
    settings, protocol, traits = frozen_settings(root, inputs)
    sampling_path = root / "generation_sampling.json"
    if not sampling_path.exists():
        return {}
    sampling = read_json(sampling_path)
    stage = read_json(root / "implementations" / f"{STAGE}.json")
    stage_payload = {key: value for key, value in stage.items() if key != "implementation_sha256"}
    if (stage["sampling_receipt_sha256"] != fingerprint(sampling)
            or stage["implementation_sha256"] != fingerprint(stage_payload)
            or sampling["resolved_config_sha256"] != fingerprint(sampling["resolved_config"])
            or sampling["checkpoint_generation_config_sha256"] != GENERATION_CONFIG_SHA256):
        raise ValueError("Public generation sampling provenance mismatch")
    if (file_hash(root / "implementations" / f"{STAGE}.py") != stage["module_sha256"]
            or file_hash(root / "implementations" / f"{STAGE}_sampling.py") != stage["sampling_module_sha256"]):
        raise ValueError("Frozen batched implementation snapshots changed")
    counts = {trait: {"serial": 0, STAGE: 0} for trait in traits}
    batches = {}
    seen_members = {}
    for trait, data in traits.items():
        for row in conditions(trait, data, protocol):
            path = record_path(root, trait, row["response_id"])
            record = read_json(path)
            if (record.get("condition_sha256") != fingerprint(row)
                    or record.get("settings_sha256") != fingerprint(settings)
                    or record.get("model_id") != MODEL_ID or record.get("model_revision") != MODEL_REVISION):
                raise ValueError("Generation row differs from its frozen condition/settings/checkpoint")
            method = record.get("generation_implementation", "serial")
            if method not in counts[trait]:
                raise ValueError("Unknown generation implementation in saved records")
            if method == STAGE and (record.get("generation_implementation_sha256") != stage["implementation_sha256"]
                                    or record.get("sampling_configuration_sha256") != sampling["resolved_config_sha256"]):
                raise ValueError("Batched row does not match public sampling/implementation provenance")
            if method == STAGE:
                batch_id = record.get("generation_batch_id", "")
                if not isinstance(batch_id, str) or re.fullmatch(r"[0-9a-f]{32}", batch_id) is None:
                    raise ValueError("Invalid batch receipt identifier")
                key = (trait, batch_id)
                if key not in batches:
                    batch = read_json(root / "generation_batches" / trait / f"{batch_id}.json")
                    members = batch.get("condition_ids", [])
                    if (batch.get("status") != "complete" or batch.get("batch_id") != batch_id
                            or batch.get("stage") != STAGE
                            or batch.get("implementation_sha256") != stage["implementation_sha256"]
                            or batch.get("sampling_configuration_sha256") != sampling["resolved_config_sha256"]
                            or not 1 <= len(members) <= BATCH_SIZE or len(set(members)) != len(members)
                            or len(batch.get("condition_seeds", [])) != len(members)
                            or set(batch.get("record_file_sha256", {})) != set(members)):
                        raise ValueError("Incomplete or inconsistent generation batch receipt")
                    batches[key] = batch
                    seen_members[key] = set()
                batch = batches[key]
                if (row["response_id"] not in batch["condition_ids"] or record.get("status") != "complete"
                        or record.get("generation_batch_size") != len(batch["condition_ids"])
                        or batch["condition_seeds"][batch["condition_ids"].index(row["response_id"])] != row["seed"]
                        or batch["record_file_sha256"].get(row["response_id"]) != file_hash(path)):
                    raise ValueError("Generation row does not match its completed batch receipt")
                seen_members[key].add(row["response_id"])
            counts[trait][method] += 1
    if any(seen_members[key] != set(batch["condition_ids"]) for key, batch in batches.items()):
        raise ValueError("Completed batch receipt references absent/unknown conditions")
    return {"native_sampling_provenance": sampling, "execution_counts_by_trait": counts,
            "batched_implementation_sha256": stage["implementation_sha256"],
            "batched_implementation_provenance": stage}
