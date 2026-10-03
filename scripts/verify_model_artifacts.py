"""Verify the downloaded checkpoint, experiment reports, and final CSV locally.

Run after the full artifact download:
    .venv/bin/python scripts/verify_model_artifacts.py

Uses only the standard library. Safetensors headers are inspected without
loading model weights. Output contains aggregate results, never prompt or
response text. The separately fetched pinned source index must be available at
provenance/base_model.safetensors.index.json, or supplied with --source-index.
The separate read-only CPU tensor precision audit is stored in provenance and
binds the source/export header comparison to the native BF16 baseline.
"""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import re
import struct
import sys
from typing import Any


WORKSPACE = Path(__file__).resolve().parents[1]
RUN_ID = "qwen35-9b-arditi-20261003-v1"
MODEL_ID = "Qwen/Qwen3.5-9B"
MODEL_REVISION = "c202236235762e1c871ad0ccb60c8ee5ba337b9a"
FINAL_COUNT = 128
SOURCE_TENSOR_COUNT = 775
SOURCE_TENSOR_BYTES = 19_306_216_416
NATIVE_DTYPE_CONVERSION_BYTES = 7_680
# SHA256 of the actual read-only Modal CPU header audit, separate from the
# frozen remote experiment bundle. This prevents widening its allowed delta.
PRECISION_AUDIT_SHA256 = "0a1219a5ebcffe098b4aed0c1c03e63d92fe3a774828be11d0695c1a2408b429"
CSV_FIELDS = (
    "prompt_id", "instruction", "category", "response", "heuristic_refusal",
    "refusal_matches", "generated_tokens", "finish_reason", "model_id",
    "source_revision", "thinking", "max_new_tokens", "run_id", "status",
)
REQUIRED_FILES = (
    "run_settings.json", "summary.json", "selection.json", "export.json",
    "final_protocol.json", "final_results.csv", "final_results.jsonl",
    "direction.pt", "reload_equivalence.json", "checkpoint_complete.json",
    "weight_edit.json", "nonedited_weight_checks.json",
    "auxiliary_weight_preservation.json", "baseline/capability.jsonl",
    "edited_capability.jsonl", "checkpoint/model.safetensors.index.json",
    "checkpoint/config.json", "checkpoint/tokenizer_config.json",
    "checkpoint/tokenizer.json", "checkpoint/chat_template.jinja",
    "checkpoint/preprocessor_config.json", "checkpoint/video_preprocessor_config.json",
    "checkpoint/README.md", "checkpoint/LICENSE", "provenance/dataset_manifest.json",
)


class VerificationError(ValueError):
    """A missing, inconsistent, or corrupt delivery artifact."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise VerificationError(message)


def checksum(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fingerprint(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def read_json(path: Path) -> Any:
    require(path.is_file(), f"Missing file: {path}")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise VerificationError(f"Invalid UTF-8 JSON file: {path}") from error


def artifact_path(root: Path, relative: str) -> Path:
    require(isinstance(relative, str), "Artifact filename must be a string")
    part = PurePosixPath(relative)
    require(bool(relative) and not part.is_absolute() and ".." not in part.parts,
            f"Invalid relative artifact path: {relative}")
    target = root / relative
    require(target.resolve().is_relative_to(root.resolve()),
            f"Artifact path escapes its directory: {relative}")
    return target


def latest_jsonl(path: Path, id_key: str = "prompt_id") -> dict[str, dict[str, Any]]:
    """Require complete valid result lines; retain the last retry for each ID."""
    require(path.is_file(), f"Missing JSONL file: {path}")
    records: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            require(bool(line.strip()), f"Blank JSONL record at {path.name}:{line_number}")
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise VerificationError(f"Invalid JSONL record at {path.name}:{line_number}") from error
            require(isinstance(record, dict) and isinstance(record.get(id_key), str),
                    f"Missing string {id_key} at {path.name}:{line_number}")
            records[record[id_key]] = record
    return records


def verify_manifest(root: Path, run_id: str) -> tuple[dict[str, Any], int]:
    manifest = read_json(root / "artifact_manifest.json")
    require(isinstance(manifest, dict) and manifest.get("run_id") == run_id,
            "Artifact manifest belongs to a different run")
    require(manifest.get("status") == "completed", "Artifact manifest does not mark the run completed")
    files = manifest.get("files")
    require(isinstance(files, dict) and bool(files), "Artifact manifest has no file inventory")
    for relative in REQUIRED_FILES:
        require(relative in files, f"Required artifact missing from manifest: {relative}")
    total_bytes = 0
    for relative, expected in sorted(files.items()):
        target = artifact_path(root, relative)
        require(isinstance(expected, dict), f"Malformed manifest entry: {relative}")
        size, digest = expected.get("bytes"), expected.get("sha256")
        require(type(size) is int and size >= 0, f"Invalid manifest byte count: {relative}")
        require(isinstance(digest, str) and re.fullmatch(r"[0-9a-f]{64}", digest) is not None,
                f"Invalid manifest SHA256: {relative}")
        require(target.is_file(), f"Artifact not downloaded: {relative}")
        require(target.stat().st_size == size, f"Artifact size mismatch: {relative}")
        require(checksum(target) == digest, f"Artifact SHA256 mismatch: {relative}")
        total_bytes += size
    return manifest, total_bytes


def csv_text(value: Any) -> str:
    return "" if value is None else str(value)


def csv_boolean(value: Any) -> str:
    require(type(value) is bool, "Result Boolean field has an invalid type")
    return "true" if value else "false"


def verify_final_results(
    root: Path, dataset_dir: Path, settings: dict[str, Any], summary: dict[str, Any],
    selection: dict[str, Any], protocol: dict[str, Any], run_id: str,
) -> dict[str, Any]:
    data_manifest = read_json(dataset_dir / "manifest.json")
    dataset_path = dataset_dir / "splits" / "harmful_val_final.json"
    dataset_hash = checksum(dataset_path)
    require(checksum(dataset_dir / "manifest.json") == settings.get("dataset_manifest_sha256"),
            "Local dataset manifest differs from the experiment provenance")
    require(checksum(root / "provenance" / "dataset_manifest.json") == settings["dataset_manifest_sha256"],
            "Saved dataset manifest differs from run settings")
    require(dataset_hash == data_manifest["splits"]["harmful_val_final"]["sha256"]
            == protocol.get("input_sha256"), "Final dataset checksum differs from the locked protocol")
    inputs = read_json(dataset_path)
    require(isinstance(inputs, list) and len(inputs) == FINAL_COUNT, "Final dataset must contain exactly 128 records")
    expected_ids = [f"harmful_val_final:{index:03d}" for index in range(FINAL_COUNT)]
    saved = latest_jsonl(root / "final_results.jsonl")
    require(set(saved) == set(expected_ids), "Final JSONL does not contain exactly the expected 128 prompt IDs")
    csv.field_size_limit(16 * 1024 * 1024)
    with (root / "final_results.csv").open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        require(tuple(reader.fieldnames or ()) == CSV_FIELDS, "Final CSV column schema or order differs")
        rows = list(reader)
    require(len(rows) == FINAL_COUNT, "Final CSV must contain exactly 128 data rows")
    require([row.get("prompt_id") for row in rows] == expected_ids, "Final CSV prompt IDs are duplicated, missing, or reordered")
    configuration = settings["configuration"]
    budget = configuration["final_max_new_tokens"]
    method = configuration["refusal_method"]
    require(type(budget) is int and budget > 0, "Invalid final token budget in run settings")
    require(protocol.get("count") == FINAL_COUNT and protocol.get("max_new_tokens") == budget,
            "Final protocol count or token budget differs from run settings")
    require(protocol.get("thinking") is False and configuration.get("thinking") is False,
            "Final protocol does not use the recorded non-thinking mode")
    require(protocol.get("model_dependent_filtering_applied") is False
            and protocol.get("selection_or_tuning_use") is False,
            "Final holdout protocol does not confirm unfiltered evaluation after selection")
    require(protocol.get("selected_candidate_id") == selection["winner"]["candidate_id"]
            == summary.get("selected_candidate_id"), "Selected candidate differs across final reports")
    generation_key = fingerprint({"settings": fingerprint(settings), "condition": "edited_final", "max_new_tokens": budget})
    totals = {"count": FINAL_COUNT, "heuristic_refusals": 0, "empty_outputs": 0,
              "degenerate_outputs": 0, "generation_errors": 0}
    for identity, original, row in zip(expected_ids, inputs, rows):
        record = saved[identity]
        require(None not in row and all(value is not None for value in row.values()),
                f"Malformed CSV field count for {identity}")
        require(isinstance(original, dict) and isinstance(original.get("instruction"), str),
                f"Invalid local input record for {identity}")
        require(record.get("instruction") == original["instruction"]
                and row["instruction"] == original["instruction"], f"Final input text changed for {identity}")
        require(record.get("category") == original.get("category")
                and row["category"] == csv_text(original.get("category")), f"Final category changed for {identity}")
        require(record.get("instruction_sha256") == hashlib.sha256(original["instruction"].encode()).hexdigest(),
                f"Final instruction checksum changed for {identity}")
        require(record.get("status") in {"completed", "empty"}, f"Final generation did not succeed for {identity}")
        require(record.get("condition") == "edited_final" and record.get("generation_key") == generation_key,
                f"Final result generation provenance differs for {identity}")
        require(record.get("run_id") == run_id and record.get("model_id") == run_id
                and record.get("source_revision") == settings["model_revision"],
                f"Final model or run provenance differs for {identity}")
        require(record.get("method") == method, f"Final refusal heuristic version differs for {identity}")
        require(record.get("thinking") is False and record.get("max_new_tokens") == budget,
                f"Final generation settings differ for {identity}")
        tokens = record.get("generated_tokens")
        require(type(tokens) is int and 1 <= tokens <= budget, f"Invalid generated token count for {identity}")
        reason = record.get("finish_reason")
        require(reason in {"eos", "length"}, f"Invalid final finish reason for {identity}")
        require(reason != "length" or tokens == budget, f"Length stop has an inconsistent token count for {identity}")
        require(isinstance(record.get("response"), str), f"Missing final response string for {identity}")
        require(record.get("empty") is (not bool(record["response"].strip())), f"Empty-output flag differs for {identity}")
        require((record["status"] == "empty") is record["empty"], f"Output status differs from empty flag for {identity}")
        require(isinstance(record.get("refusal_matches"), list)
                and all(isinstance(item, str) for item in record["refusal_matches"]),
                f"Invalid refusal matches for {identity}")
        expected_row = {field: csv_text(record.get(field)) for field in CSV_FIELDS}
        expected_row["heuristic_refusal"] = csv_boolean(record.get("heuristic_refusal"))
        expected_row["thinking"] = csv_boolean(record.get("thinking"))
        expected_row["refusal_matches"] = json.dumps(record["refusal_matches"], ensure_ascii=False, allow_nan=False)
        for field in CSV_FIELDS:
            require(row[field] == expected_row[field], f"CSV/JSONL mismatch in {field} for {identity}")
        require(type(record.get("degenerate")) is bool, f"Invalid degeneration flag for {identity}")
        totals["heuristic_refusals"] += int(record["heuristic_refusal"])
        totals["empty_outputs"] += int(record["empty"])
        totals["degenerate_outputs"] += int(record["degenerate"])
    metrics = summary.get("final_evaluation", {})
    for field, value in totals.items():
        require(metrics.get(field) == value, f"Final summary metric differs: {field}")
    require(metrics.get("refusal_method") == method and summary.get("csv_rows") == FINAL_COUNT,
            "Final summary heuristic or CSV row count differs")
    require(metrics.get("heuristic_refusal_rate") == totals["heuristic_refusals"] / FINAL_COUNT,
            "Final summary refusal rate differs from records")
    return totals


def verify_export(root: Path, settings: dict[str, Any], selection: dict[str, Any], exported: dict[str, Any],
                  protocol: dict[str, Any]) -> dict[str, Any]:
    require(selection.get("status") == "completed" and selection.get("selection_locked") is True,
            "Direction selection is incomplete or unlocked")
    require(selection.get("final_holdout_used") is False and exported.get("final_holdout_used") is False,
            "Selection/export does not confirm the final holdout was unused")
    winner = selection["winner"]
    require(winner.get("generated_guards_passed") is True and winner.get("kl_guard_passed") is True,
            "Selected candidate did not pass recorded development guards")
    require(winner.get("lost_baseline_capability_probes") == []
            and winner.get("additional_benign_refusals") == 0
            and winner.get("additional_benign_degenerations") == 0,
            "Selected candidate failed benign capability/refusal/degeneration retention")
    config = settings["configuration"]
    require(winner["benign_kl"] < config["benign_kl_threshold_nats"], "Selected candidate exceeded the benign KL threshold")
    require(winner["harmful_validation"]["heuristic_refusals"]
            < selection["baseline_harmful_validation"]["heuristic_refusals"],
            "Selected candidate did not reduce generated development refusal matches")
    require(exported.get("status") == "completed" and exported.get("selection_locked") is True,
            "Checkpoint export is incomplete")
    require(exported.get("baseline_capability_probes_lost") == [], "Export reports lost baseline capability probes")
    equivalence = read_json(root / "reload_equivalence.json")
    require(equivalence == exported.get("equivalence") and equivalence.get("guards_passed") is True,
            "Export/reload equivalence reports differ or guards failed")
    for field, threshold in (("mean_kl_reference_to_current", "reload_mean_kl_limit_nats"),
                             ("max_kl_reference_to_current", "reload_max_kl_limit_nats")):
        value = equivalence.get(field)
        require(isinstance(value, (int, float)) and math.isfinite(value) and value <= config[threshold],
                f"Reload equivalence exceeds the recorded guard: {field}")
    require(equivalence.get("confident_argmax_agreement", -1) >= config["reload_confident_argmax_minimum"],
            "Reload equivalence failed confident argmax agreement")
    require(read_json(root / "nonedited_weight_checks.json").get("passed") is True,
            "Reloaded nonedited weight checks failed")
    edit = read_json(root / "weight_edit.json")
    require(edit.get("edited_matrix_count") == 65 and edit.get("edited_bias_count") == 0,
            "Weight edit inventory differs from 65 bias-free text matrices")
    require(edit.get("text_only") is True and edit.get("vision_weights_preserved") is True
            and edit.get("untied_lm_head_preserved") is True
            and edit.get("nonedited_checks_after_edit", {}).get("passed") is True,
            "Weight edit preservation checks failed")
    marker = read_json(root / "checkpoint_complete.json")
    direction_hash = checksum(root / "direction.pt")
    require(direction_hash == selection.get("direction_sha256") == protocol.get("direction_sha256")
            == marker.get("direction_sha256"), "Selected direction changed across saved stages")
    require(marker.get("weight_files_saved") is True, "Checkpoint completion marker is incomplete")
    baseline = latest_jsonl(root / "baseline" / "capability.jsonl")
    current = latest_jsonl(root / "edited_capability.jsonl")
    require(len(baseline) == 12 and set(baseline) == set(current), "Capability probe IDs/count differ between baseline and export")
    for identity in baseline:
        require(baseline[identity].get("status") in {"completed", "empty"}
                and current[identity].get("status") in {"completed", "empty"},
                f"Capability generation error for {identity}")
        require(type(baseline[identity].get("capability_pass")) is bool
                and type(current[identity].get("capability_pass")) is bool,
                f"Capability pass flag missing for {identity}")
        require(not baseline[identity]["capability_pass"] or current[identity]["capability_pass"],
                f"Export lost a passing baseline capability probe: {identity}")
    passes = sum(row["capability_pass"] for row in current.values())
    require(exported.get("capability_passes") == passes, "Export capability total differs from actual probe records")
    return {"baseline_capability_passes": sum(row["capability_pass"] for row in baseline.values()),
            "export_capability_passes": passes,
            "reload_mean_kl_nats": equivalence["mean_kl_reference_to_current"]}


def safetensors_header(path: Path) -> tuple[dict[str, dict[str, Any]], int, dict[str, Any]]:
    """Validate header inventory, shapes, dtype sizes, and the complete data span."""
    require(path.is_file(), f"Missing checkpoint shard: {path.name}")
    with path.open("rb") as stream:
        prefix = stream.read(8)
        require(len(prefix) == 8, f"Truncated Safetensors prefix: {path.name}")
        length = struct.unpack("<Q", prefix)[0]
        require(2 <= length <= 100 * 1024 * 1024 and 8 + length <= path.stat().st_size,
                f"Invalid Safetensors header length: {path.name}")
        raw = stream.read(length)
    try:
        header = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise VerificationError(f"Invalid Safetensors JSON header: {path.name}") from error
    require(isinstance(header, dict), f"Safetensors header is not an object: {path.name}")
    tensors = {key: value for key, value in header.items() if key != "__metadata__"}
    require(bool(tensors), f"Empty Safetensors shard: {path.name}")
    data_bytes = path.stat().st_size - 8 - length
    ranges = []
    for name, tensor in tensors.items():
        require(isinstance(tensor, dict), f"Invalid tensor descriptor in {path.name}")
        shape, dtype, offsets = tensor.get("shape"), tensor.get("dtype"), tensor.get("data_offsets")
        require(isinstance(shape, list) and all(type(size) is int and size >= 0 for size in shape),
                f"Invalid tensor shape in {path.name}: {name}")
        require(dtype in {"BF16", "F32"}, f"Unexpected floating-point dtype in {path.name}: {name}")
        require(isinstance(offsets, list) and len(offsets) == 2
                and all(type(offset) is int for offset in offsets),
                f"Invalid tensor offsets in {path.name}: {name}")
        begin, end = offsets
        require(0 <= begin <= end <= data_bytes, f"Tensor extends beyond shard data in {path.name}: {name}")
        require(end - begin == math.prod(shape) * (2 if dtype == "BF16" else 4),
                f"Tensor shape/dtype disagrees with byte span in {path.name}: {name}")
        ranges.append((begin, end))
    cursor = 0
    for begin, end in sorted(ranges):
        require(begin == cursor, f"Safetensors shard contains overlapping or missing data: {path.name}")
        cursor = end
    require(cursor == data_bytes, f"Safetensors shard contains unindexed trailing data: {path.name}")
    shard = {"header_sha256": hashlib.sha256(prefix + raw).hexdigest(),
             "header_bytes": 8 + length, "tensor_bytes": data_bytes,
             "file_bytes": path.stat().st_size}
    return tensors, data_bytes, shard


def verify_tensor_precision(
    root: Path, audit_path: Path, source_index: Path, exported_index: Path,
    settings: dict[str, Any], original: dict[str, Any], index: dict[str, Any],
    inventory: dict[str, dict[str, Any]], shard_metadata: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Allow only the exact, independently observed native loading conversion.

    The 65 rank-one matrix edits operate on the loaded BF16 model. Loading the
    pinned source converts 24 A_log and 24 linear-attention norm vectors from
    F32 to BF16; preserving other model weights refers to that native baseline.
    """
    require(checksum(audit_path) == PRECISION_AUDIT_SHA256,
            "Tensor precision audit checksum differs from the recorded read-only header audit")
    audit = read_json(audit_path)
    require(audit.get("run_id") == settings["run_id"]
            and audit.get("model_revision") == settings["model_revision"],
            "Tensor precision audit belongs to a different run or model revision")
    source, exported = audit.get("source"), audit.get("export")
    require(isinstance(source, dict) and isinstance(exported, dict),
            "Tensor precision audit is missing source/export inventories")
    require(source.get("index_sha256") == checksum(source_index) == settings["base_weight_index_sha256"]
            and exported.get("index_sha256") == checksum(exported_index),
            "Tensor precision audit source/export index checksum differs")
    source_tensors, exported_tensors = source.get("tensors"), exported.get("tensors")
    require(isinstance(source_tensors, dict) and isinstance(exported_tensors, dict)
            and set(source_tensors) == set(original["weight_map"]) == set(exported_tensors) == set(inventory),
            "Tensor precision audit does not cover the original/exported 775-key inventory")
    actual_export = {
        name: {"dtype": tensor["dtype"], "shape": tensor["shape"],
               "bytes": tensor["data_offsets"][1] - tensor["data_offsets"][0]}
        for name, tensor in inventory.items()
    }
    require(actual_export == exported_tensors,
            "Downloaded checkpoint shapes/dtypes/byte spans differ from the read-only audit")
    require(shard_metadata == exported.get("shards"),
            "Downloaded checkpoint headers or shard sizes differ from the read-only audit")
    source_shards = source.get("shards")
    require(isinstance(source_shards, dict)
            and set(source_shards) == set(original["weight_map"].values()),
            "Tensor precision audit original shard inventory differs from the pinned source index")
    source_total = 0
    for name, tensor in source_tensors.items():
        shape, dtype, size = tensor.get("shape"), tensor.get("dtype"), tensor.get("bytes")
        require(isinstance(shape, list) and all(type(value) is int and value >= 0 for value in shape)
                and dtype in {"BF16", "F32"} and type(size) is int
                and size == math.prod(shape) * (2 if dtype == "BF16" else 4),
                f"Invalid pinned source tensor metadata in precision audit: {name}")
        source_total += size
    for filename, metadata in source_shards.items():
        require(isinstance(metadata, dict), f"Invalid source shard precision metadata: {filename}")
        size = sum(tensor["bytes"] for name, tensor in source_tensors.items()
                   if original["weight_map"][name] == filename)
        require(metadata.get("tensor_bytes") == size
                and type(metadata.get("header_bytes")) is int and metadata["header_bytes"] >= 10
                and metadata.get("file_bytes") == size + metadata["header_bytes"]
                and isinstance(metadata.get("header_sha256"), str)
                and re.fullmatch(r"[0-9a-f]{64}", metadata["header_sha256"]) is not None,
                f"Source shard byte totals/header checksum are inconsistent: {filename}")
    exported_total = sum(tensor["bytes"] for tensor in exported_tensors.values())
    require(source_total == source.get("header_tensor_bytes") == source.get("index_tensor_bytes")
            == original.get("metadata", {}).get("total_size") == SOURCE_TENSOR_BYTES,
            "Source precision audit tensor byte total differs from the pinned index")
    require(exported_total == exported.get("header_tensor_bytes") == exported.get("index_tensor_bytes")
            == index.get("metadata", {}).get("total_size")
            == SOURCE_TENSOR_BYTES - NATIVE_DTYPE_CONVERSION_BYTES,
            "Export precision audit tensor byte total differs from the checkpoint index")
    expected_changes: dict[str, tuple[list[int], int]] = {}
    for layer in range(32):
        if (layer + 1) % 4 != 0:
            stem = f"model.language_model.layers.{layer}.linear_attn."
            expected_changes[stem + "A_log"] = ([32], 64)
            expected_changes[stem + "norm.weight"] = ([128], 256)
    observed_changes = []
    for name in sorted(source_tensors):
        before, after = source_tensors[name], exported_tensors[name]
        if name in expected_changes:
            shape, reduction = expected_changes[name]
            require(before == {"dtype": "F32", "shape": shape, "bytes": math.prod(shape) * 4}
                    and after == {"dtype": "BF16", "shape": shape, "bytes": math.prod(shape) * 2},
                    f"Native BF16 source conversion shape/dtype differs: {name}")
            observed_changes.append({"tensor": name, "source": before, "export": after,
                                     "byte_reduction": reduction})
        else:
            require(before == after, f"Unexpected source/export shape, dtype, or byte-size change: {name}")
    require(audit.get("dtype_changes") == observed_changes and len(observed_changes) == 48,
            "Source precision audit conversion inventory differs from the exact 48 native BF16 conversions")
    require(source_total - exported_total == audit.get("source_dtype_conversion_bytes")
            == NATIVE_DTYPE_CONVERSION_BYTES,
            "Source/export precision delta differs from the exact native BF16 conversion")
    return {"source_dtype_conversion_bytes": NATIVE_DTYPE_CONVERSION_BYTES,
            "source_dtype_conversion_tensors": len(observed_changes),
            "preservation_interpretation": "preserved_relative_to_native_bf16_baseline",
            "tensor_precision_audit_sha256": PRECISION_AUDIT_SHA256}


def verify_checkpoint(root: Path, source_index: Path, settings: dict[str, Any], manifest: dict[str, Any],
                      precision_audit: Path) -> dict[str, Any]:
    require(checksum(source_index) == settings.get("base_weight_index_sha256"),
            "Pinned source weight index checksum differs from run settings")
    original = read_json(source_index)
    source_map = original.get("weight_map")
    require(isinstance(source_map, dict) and len(source_map) == SOURCE_TENSOR_COUNT,
            "Pinned source index does not contain the original 775 tensor keys")
    exported_index = root / "checkpoint" / "model.safetensors.index.json"
    index = read_json(exported_index)
    weight_map = index.get("weight_map")
    require(isinstance(weight_map, dict) and set(weight_map) == set(source_map),
            "Checkpoint tensor keys differ from the pinned original 775-key inventory")
    shards = sorted(set(weight_map.values()))
    require(all(isinstance(name, str) for name in shards), "Invalid checkpoint shard filenames")
    marker = read_json(root / "checkpoint_complete.json")
    require(sorted(marker.get("shards", [])) == shards, "Checkpoint shard completion inventory differs from the index")
    inventory: dict[str, dict[str, Any]] = {}
    shard_metadata: dict[str, dict[str, Any]] = {}
    total_bytes = 0
    for shard in shards:
        relative = f"checkpoint/{shard}"
        require(relative in manifest["files"], f"Checkpoint shard absent from artifact manifest: {shard}")
        entries, data_bytes, metadata = safetensors_header(artifact_path(root, relative))
        expected = {name for name, filename in weight_map.items() if filename == shard}
        require(set(entries) == expected, f"Checkpoint shard header inventory differs from the index: {shard}")
        require(not set(entries).intersection(inventory), f"Tensor duplicated across checkpoint shards: {shard}")
        inventory.update(entries)
        shard_metadata[shard] = metadata
        total_bytes += data_bytes
    require(len(inventory) == SOURCE_TENSOR_COUNT
            and total_bytes == index.get("metadata", {}).get("total_size"),
            "Checkpoint tensor byte total differs from its index")
    precision = verify_tensor_precision(root, precision_audit, source_index, exported_index,
                                        settings, original, index, inventory, shard_metadata)
    configuration = read_json(root / "checkpoint" / "config.json")
    text = configuration.get("text_config", {})
    require(configuration.get("architectures") == ["Qwen3_5ForConditionalGeneration"]
            and configuration.get("model_type") == "qwen3_5"
            and configuration.get("tie_word_embeddings") is False,
            "Checkpoint does not retain the full untied Qwen3.5 architecture")
    require(text.get("hidden_size") == 4096 and text.get("num_hidden_layers") == 32
            and text.get("vocab_size") == 248320 and text.get("intermediate_size") == 12288,
            "Checkpoint text dimensions differ from Qwen3.5-9B")
    types = ["full_attention" if (index + 1) % 4 == 0 else "linear_attention" for index in range(32)]
    require(text.get("layer_types") == types and isinstance(configuration.get("vision_config"), dict),
            "Checkpoint hybrid layer layout or vision configuration differs")
    embedding = "model.language_model.embed_tokens.weight"
    require(inventory[embedding]["shape"] == [248320, 4096]
            and inventory["lm_head.weight"]["shape"] == [248320, 4096], "Checkpoint embedding/LM-head shape differs")
    expected_writers = {embedding: (1, [248320, 4096])}
    for layer, kind in enumerate(types):
        stem = f"model.language_model.layers.{layer}"
        mixer = ".linear_attn.out_proj.weight" if kind == "linear_attention" else ".self_attn.o_proj.weight"
        expected_writers[stem + mixer] = (0, [4096, 4096])
        expected_writers[stem + ".mlp.down_proj.weight"] = (0, [4096, 12288])
    edits = read_json(root / "weight_edit.json").get("matrices", [])
    edit_map = {entry.get("parameter"): entry for entry in edits}
    require(len(edits) == len(edit_map) == 65 and set(edit_map) == set(expected_writers),
            "Edited tensor names differ from the expected 65 residual matrices")
    for name, (axis, shape) in expected_writers.items():
        require(edit_map[name].get("residual_axis") == axis and edit_map[name].get("shape") == shape
                and inventory[name]["shape"] == shape, f"Edited projection axis or matrix shape differs: {name}")
    auxiliary = read_json(root / "auxiliary_weight_preservation.json")
    mtp_keys = {name for name in source_map if name.startswith("mtp.")}
    require(len(mtp_keys) == 15 and set(auxiliary.get("auxiliary_tensor_names", [])) == mtp_keys
            and auxiliary.get("auxiliary_tensor_count") == 15
            and auxiliary.get("source_tensor_count") == auxiliary.get("export_tensor_count") == SOURCE_TENSOR_COUNT
            and auxiliary.get("auxiliary_values_unchanged") is True,
            "Original MTP auxiliary tensor preservation report differs")
    require({weight_map[name] for name in mtp_keys} == {"model-auxiliary.safetensors"},
            "Original MTP tensors are not isolated in the preserved auxiliary shard")
    require(auxiliary.get("auxiliary_bytes") == sum(
        inventory[name]["data_offsets"][1] - inventory[name]["data_offsets"][0] for name in mtp_keys),
        "MTP auxiliary byte count differs from the saved shard")
    return {"checkpoint_shards": len(shards), "checkpoint_tensors": len(inventory),
            "checkpoint_tensor_bytes": total_bytes, **precision}


def verify_execution_repair(
    root: Path, settings: dict[str, Any], selection: dict[str, Any], manifest: dict[str, Any], run_id: str,
) -> dict[str, Any]:
    """Bind an optional verification repair to the unchanged selected experiment."""
    files = manifest["files"]
    relative = "execution_repairs/cache_position_fix.json"
    repair_files = [name for name in files if name.startswith("execution_repairs/")]
    if not repair_files:
        return {"execution_repair": None}
    require(relative in files, "Execution repair artifacts have no audit definition")
    require("failure.json" not in files, "Completed run manifest contains a current unarchived failure")
    audit = read_json(root / relative)
    require(audit.get("repair_id") == "explicit_text_positions_for_export_equivalence_v1"
            and audit.get("run_id") == run_id, "Unknown execution repair or mismatched repair run")
    require(audit.get("selection_sha256") == checksum(root / "selection.json")
            and audit.get("direction_sha256") == selection["direction_sha256"] == checksum(root / "direction.pt"),
            "Execution repair is not bound to the original locked selection/direction")
    require(audit.get("original_source_code_hashes") == settings["source_code_hashes"],
            "Execution repair changes the original frozen experiment source inventory")
    require(audit.get("generation_selection_and_weight_edit_unchanged") is True
            and audit.get("equivalence_probes_forced_tokens_layout_and_guards_unchanged") is True,
            "Execution repair does not preserve the original generation/selection/edit/guard scope")
    hashes = audit.get("repair_source_hashes")
    require(isinstance(hashes, dict) and set(hashes) == {"export_repair.py", "modal_repair.py"},
            "Execution repair source inventory differs")
    for name, digest in hashes.items():
        source = f"execution_repairs/{name}"
        require(source in files and checksum(root / source) == digest,
                f"Execution repair source checksum differs: {name}")
    checks = audit.get("position_self_checks")
    require(isinstance(checks, dict) and checks.get("passed") is True,
            "Execution repair position self-checks did not pass")
    # Reproduce inspect.getsource for the original decorated helper without
    # importing Torch or executing the saved experiment source.
    frozen = (root / "provenance" / "experiment.py").read_text(encoding="utf-8")
    tree = ast.parse(frozen)
    classes = [node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "_Experiment"]
    require(len(classes) == 1, "Frozen experiment class missing from repair provenance")
    methods = [node for node in classes[0].body
               if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "equivalence_rows"]
    require(len(methods) == 1, "Original equivalence helper missing from repair provenance")
    method = methods[0]
    start = min([method.lineno, *(decorator.lineno for decorator in method.decorator_list)])
    original_method = "".join(frozen.splitlines(keepends=True)[start - 1:method.end_lineno])
    require(hashlib.sha256(original_method.encode()).hexdigest() == audit.get("original_method_sha256"),
            "Execution repair original helper checksum differs")
    archived = "execution_repairs/original_export_failure.json"
    require(archived in files, "Execution repair is missing the original export failure archive")
    failure = read_json(root / archived)
    require(failure.get("stage") == "export", "Archived repair failure did not occur during export verification")
    if "native_generation_agreement_check" in audit:
        native_relative = "cache_position_native_check.json"
        require(native_relative in files, "Execution repair native-generation agreement report is missing")
        native = read_json(root / native_relative)
        require(native.get("repair_id") == audit["repair_id"] and native.get("passed") is True
                and native.get("probe_count") == 3 and native.get("checked_generated_tokens") == 24,
                "Execution repair native-generation agreement count, identity, or result differs")
        require(native.get("fresh_native_attention_and_recurrent_cache") is True
                and native.get("caller_intervention_applies_to_both_paths") is True,
                "Execution repair native comparison did not use fresh caches and the same intervention")
        probes = native.get("probes")
        require(isinstance(probes, list) and len(probes) == 3
                and [probe.get("prompt_id") for probe in probes]
                == ["equivalence:plants", "equivalence:python", "equivalence:parentheses"],
                "Execution repair native-generation probe inventory differs")
        for probe in probes:
            require(probe.get("checked_tokens") == 8 and probe.get("exact_token_agreement") is True
                    and probe.get("early_eos") is False,
                    f"Execution repair native-generation agreement failed for {probe.get('prompt_id')}")
    return {"execution_repair": audit["repair_id"]}


def verify(root: Path, dataset_dir: Path, source_index: Path, run_id: str,
           precision_audit: Path | None = None) -> dict[str, Any]:
    require(root.is_dir(), f"Artifact directory does not exist: {root}")
    manifest, total_bytes = verify_manifest(root, run_id)
    settings = read_json(root / "run_settings.json")
    summary = read_json(root / "summary.json")
    selection = read_json(root / "selection.json")
    exported = read_json(root / "export.json")
    protocol = read_json(root / "final_protocol.json")
    require(settings.get("run_id") == summary.get("run_id") == run_id
            and summary.get("status") == "completed", "Run settings/summary identity or completion status differs")
    require(settings.get("model_id") == summary.get("base_model_id") == MODEL_ID
            and settings.get("model_revision") == summary.get("base_model_revision") == MODEL_REVISION,
            "Base model or pinned revision differs from the requested run")
    source_hashes = settings.get("source_code_hashes")
    require(isinstance(source_hashes, dict)
            and set(source_hashes) == {"experiment.py", "abliteration.py", "evaluation.py", "modal_app.py"},
            "Run settings do not contain the complete frozen source provenance")
    for name, digest in source_hashes.items():
        relative = f"provenance/{name}"
        require(relative in manifest["files"] and checksum(artifact_path(root, relative)) == digest,
                f"Saved source snapshot differs from run provenance: {name}")
    final = verify_final_results(root, dataset_dir, settings, summary, selection, protocol, run_id)
    export = verify_export(root, settings, selection, exported, protocol)
    checkpoint = verify_checkpoint(root, source_index, settings, manifest,
                                   precision_audit or root / "provenance" / "tensor_precision_audit.json")
    repair = verify_execution_repair(root, settings, selection, manifest, run_id)
    return {"verified": True, "run_id": run_id, "artifact_files": len(manifest["files"]),
            "verified_artifact_bytes": total_bytes, "csv_rows": FINAL_COUNT,
            "refusal_method": settings["configuration"]["refusal_method"],
            "final_heuristic_refusals": final["heuristic_refusals"],
            "final_empty_outputs": final["empty_outputs"],
            "final_degenerate_outputs": final["degenerate_outputs"], **export, **checkpoint, **repair}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default=RUN_ID)
    parser.add_argument("--artifacts", type=Path, help="Directory containing the downloaded run artifacts")
    parser.add_argument("--source-index", type=Path, help="Pinned original model.safetensors.index.json")
    parser.add_argument("--precision-audit", type=Path, help="Separate read-only CPU source/export tensor header audit")
    parser.add_argument("--dataset-dir", type=Path, default=WORKSPACE / "data" / "arditi_refusal")
    arguments = parser.parse_args()
    root = arguments.artifacts or WORKSPACE / "artifacts" / arguments.run_id
    source_index = arguments.source_index or root / "provenance" / "base_model.safetensors.index.json"
    try:
        result = verify(root, arguments.dataset_dir, source_index, arguments.run_id, arguments.precision_audit)
    except (VerificationError, OSError, UnicodeDecodeError, KeyError, TypeError, ValueError) as error:
        print(f"Verification failed: {error}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
