"""Integrity and publication packaging for the real all-layer persona bank.

Uses the standard library so publication checks do not need a GPU or torch.
The public payload is an explicit allowlist; private response/judge logs are
never traversed or copied into the release.
"""

from __future__ import annotations

from array import array
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import re
import shutil
import struct
import sys


TRAITS = ("depression", "curiosity", "paranoia", "sexual_arousal", "narcissism", "euphoria")
MODEL_ID = "demivoleegaston/Qwen3.5-9B-mooody"
MODEL_REVISION = "705afd95bced3ac0424d7e68b1299d8fcdffb858"
HIDDEN_SIZE = 4096
CONFIG_SHA256 = "5bfc82e1be6c5eb0cefd957b87f8f3a96120c5ea15530d1cfa85d7d082018ca8"
GENERATION_CONFIG_SHA256 = "d0bb1d295ae3e46bb02d6d1a4dc59d689ec822d7479a32835f6945db85c3512a"
BATCHED_STAGE = "generation_batched_v1"
TOKENIZER_HASHES = {
    "tokenizer.json": {"bytes": 19989506, "sha256": "2f4dd754486e96054d0c841ddde816748be0116cbbc8ba8b5311ae4853432a21"},
    "tokenizer_config.json": {"bytes": 1464, "sha256": "6fc30221b0ec773811dfa2a04ebf0eec1a7e36a3bb1698888e28b38eabc1126c"},
    "chat_template.jinja": {"bytes": 7756, "sha256": "a4aee8afcf2e0711942cf848899be66016f8d14a889ff9ede07bca099c28f715"},
}
BANK_FILENAME = "persona_vectors.safetensors"
MANIFEST_FILENAME = "persona_manifest.json"
PUBLICATION_FILENAME = "publication_manifest.json"
POLICY_SOURCE = "data/persona_traits/extraction_policy.json"
POLICY_SHA256 = "fcec10d4f4b6443b8fd9d7e408ad8853fc2d3c798ebce7e022f91d2a9d3c89b4"
POLICY_BYTES = 1350
FILTERING_POLICY = {
    "schema_version": 1,
    "policy_id": "persona-vectors-filtering-v2",
    "generation_protocol_unchanged": True,
    "positive_trait_threshold": ">50",
    "negative_trait_threshold": "<50",
    "minimum_coherence_score_both_sides": 50,
    "matched_pairs": True,
    "allow_judged_capped_responses": True,
    "require_all_generated_questions": True,
    "require_all_generated_system_prompt_pairs": True,
    "require_all_accepted_questions": False,
    "require_all_accepted_system_prompt_pairs": False,
    "minimum_accepted_pairs_per_trait": 1,
    "supersedes": [
        "The frozen protocol's blanket exclusion of target responses at the token limit",
        "The frozen protocol's requirement that every question and system pair have an accepted contrast",
    ],
    "rationale": "Match the Persona Vectors extraction implementation: judge capped answers with the same trait and coherence gates, and report accepted coverage as a diagnostic. Generate every planned condition, keep matched contrasts, and require nonempty groups for the difference of means. Preserve the original collection protocol and all generated records.",
    "references": [
        "https://github.com/safety-research/persona_vectors/blob/main/eval/eval_persona.py",
        "https://github.com/safety-research/persona_vectors/blob/main/generate_vec.py",
        "https://arxiv.org/html/2507.21509v1#S2.SS2",
    ],
}
RECOVERY_STAGE = "judging_recovery_v1"
RECOVERY_POLICY_SOURCE = "data/persona_traits/judging_recovery_policy.json"
RECOVERY_POLICY_SHA256 = "f9fa377240ac3e44903b0cf2d3ed312c1a9e770a2c02b3af2b7f12c0d5799d72"
RECOVERY_POLICY_BYTES = 1129
RECOVERY_POLICY = {
    "schema_version": 1, "stage": RECOVERY_STAGE,
    "reason": "Recover exhausted HTTP429 transport failures without changing any generated response, rubric, request identity, valid score, or filtering threshold.",
    "eligible_score": "null", "original_attempt_count": 3,
    "original_attempts_must_all_be_completed_http_429": True,
    "maximum_additional_attempts_per_score": 3, "global_recovery_workers": 1,
    "request_concurrency": 1, "minimum_delay_after_request_completion_seconds": 0.5,
    "http_429_backoff_seconds": [2, 4], "retry_during_recovery_only_http_429": True,
    "do_not_retry_malformed_refusal_or_other_nontransport_failures": True,
    "preserve_original_attempt_prefix_and_valid_scores": True,
    "journal_and_commit_each_attempt_before_http": True,
    "unknown_interrupted_attempts_require_explicit_repair": True,
    "request_payload_model_provider_rubrics_and_response_fingerprints": "unchanged",
    "require_original_judging_workers_finished_before_recovery": True,
    "refilter_and_reassemble_each_affected_trait_after_recovery": True,
    "generation_rerolls": 0, "filtering_policy_change": False,
}
BOUNDARY = "decoder_block_output_pre_final_global_rmsnorm"
POOLING = "response_content_mean_then_equal_response_group_mean"
PUBLIC_FILES = {BANK_FILENAME, MANIFEST_FILENAME, "README.md", "LICENSE", PUBLICATION_FILENAME}
SOURCE_FILES = {
    "data/persona_traits/protocol.json", "data/persona_traits/manifest.json",
    "data/persona_traits/coherence_evaluation_prompt.txt", POLICY_SOURCE,
    *(f"data/persona_traits/source/{trait}.json" for trait in TRAITS),
}
SHA256 = re.compile(r"[0-9a-f]{64}\Z")
SECRET_TEXT = re.compile(r"hf_[A-Za-z0-9]{24,}|sk-or-v1-[A-Za-z0-9_-]{16,}|-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----")
PRIVATE_KEYS = {"response", "responses", "raw_response", "raw_responses", "messages", "question",
                "prompt", "input_ids", "raw_judge_output", "raw_judge_outputs", "raw", "api_key",
                "hf_token", "openrouter_api_key", "authorization"}


def digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def json_bytes(value: object) -> bytes:
    return (json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")


def metadata_fingerprint(value: object) -> str:
    """Match extraction's canonical fingerprints for public stage receipts."""
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    allow_nan=False).encode()).hexdigest()


def validate_recovery(manifest: dict, sources: dict) -> int:
    """Bind the separate transport policy and complete all-journal audit."""
    judging = manifest["judging"]
    recovery = judging.get("transport_recovery")
    if recovery is None:
        require(RECOVERY_POLICY_SOURCE not in sources, "Recovery policy source requires its transport metadata")
        return 0
    require(isinstance(recovery, dict) and recovery.get("stage") == RECOVERY_STAGE
            and recovery.get("policy") == RECOVERY_POLICY
            and recovery.get("policy_source") == RECOVERY_POLICY_SOURCE
            and recovery.get("policy_sha256") == RECOVERY_POLICY_SHA256
            and sources.get(RECOVERY_POLICY_SOURCE) == {"sha256": RECOVERY_POLICY_SHA256, "bytes": RECOVERY_POLICY_BYTES},
            "Missing or different transport recovery policy/hash/source")
    stage = recovery.get("implementation", {})
    require(stage == manifest.get("implementations", {}).get(RECOVERY_STAGE)
            and stage.get("stage") == RECOVERY_STAGE and stage.get("policy") == RECOVERY_POLICY
            and stage.get("policy_source") == RECOVERY_POLICY_SOURCE
            and stage.get("policy_sha256") == RECOVERY_POLICY_SHA256
            and stage.get("policy_bytes") == RECOVERY_POLICY_BYTES
            and stage.get("original_judging_implementation_sha256") == manifest.get("implementations", {}).get("judging", {}).get("implementation_sha256")
            and stage.get("implementation_sha256") == metadata_fingerprint({key: value for key, value in stage.items()
                                                                           if key != "implementation_sha256"}),
            "Recovery implementation receipt/hash mismatch")
    for field in ("module_sha256", "original_judging_implementation_sha256"):
        require(isinstance(stage.get(field), str) and SHA256.fullmatch(stage[field]) is not None,
                "Missing recovery implementation source hash")
    audit = recovery.get("audit", {})
    require(isinstance(audit, dict) and audit.get("status") == "complete"
            and audit.get("stage") == RECOVERY_STAGE
            and audit.get("policy_sha256") == RECOVERY_POLICY_SHA256
            and audit.get("implementation_sha256") == stage["implementation_sha256"]
            and recovery.get("audit_sha256") == hashlib.sha256(json_bytes(audit)).hexdigest(),
            "Recovery all-journal audit is incomplete or its hash changed")
    fields = ("present", "valid_scores", "missing", "in_progress_or_interrupted", "eligible_http429_none",
              "other_none", "actual_attempts", "original_actual_attempts", "recovery_actual_attempts")
    snapshots = []
    for label in ("before", "after"):
        snapshot = audit.get(label, {})
        by_trait = snapshot.get("by_trait", {})
        require(snapshot.get("planned_score_slots") == 4800 and isinstance(by_trait, dict)
                and set(by_trait) == set(TRAITS), "Recovery audit must cover all 4800 slots and six traits")
        for row in by_trait.values():
            require(all(type(row.get(key)) is int and row[key] >= 0 for key in fields)
                    and row["present"] == 800 and row["missing"] == 0 and row["in_progress_or_interrupted"] == 0
                    and row["valid_scores"] + row["eligible_http429_none"] + row["other_none"] == 800
                    and row["actual_attempts"] == row["original_actual_attempts"] + row["recovery_actual_attempts"]
                    and row["original_actual_attempts"] <= 2400,
                    "Invalid per-trait recovery audit counts")
        require(all(type(snapshot.get(key)) is int
                    and snapshot[key] == sum(row[key] for row in by_trait.values()) for key in fields),
                "Recovery audit aggregate counts do not match the six traits")
        snapshots.append(snapshot)
    before, after = snapshots
    for trait in TRAITS:
        original, current = before["by_trait"][trait], after["by_trait"][trait]
        recovered_trait = current["valid_scores"] - original["valid_scores"]
        require(0 <= recovered_trait <= original["eligible_http429_none"]
                and current["eligible_http429_none"] == original["eligible_http429_none"] - recovered_trait
                and current["other_none"] == original["other_none"]
                and current["original_actual_attempts"] == original["original_actual_attempts"]
                and original["recovery_actual_attempts"] == 0
                and original["eligible_http429_none"] <= current["recovery_actual_attempts"] <= 3 * original["eligible_http429_none"],
                "Per-trait recovery changed untouched journals or exceeded transport budgets")
    counts = recovery.get("counts", {})
    names = ("records_touched", "additional_attempts", "recovered_scores", "remaining_none",
             "original_http429_attempts", "http429_recovery_attempts")
    require(all(type(counts.get(key)) is int and counts[key] >= 0 for key in names), "Missing recovery call counts")
    touched, extra, recovered, remaining = (counts[key] for key in names[:4])
    require(touched == before["eligible_http429_none"] and recovered + remaining == touched
            and touched <= extra <= 3 * touched and counts["original_http429_attempts"] == 3 * touched
            and counts["http429_recovery_attempts"] <= extra - recovered
            and after["valid_scores"] == before["valid_scores"] + recovered
            and after["eligible_http429_none"] == remaining and after["other_none"] == before["other_none"]
            and before["recovery_actual_attempts"] == 0 and after["recovery_actual_attempts"] == extra
            and after["original_actual_attempts"] == before["original_actual_attempts"]
            and judging.get("score_records") == 4800 and judging.get("valid_scores") == after["valid_scores"]
            and judging.get("actual_attempts") == after["actual_attempts"],
            "Recovery scores/attempts differ from preserved originals and the complete audit")
    require(recovery.get("valid_and_ineligible_original_journals_preserved") is True,
            "Recovery must preserve valid and ineligible original journals")
    for field in ("original_journal_hashes_sha256", "baseline_journal_hashes_sha256"):
        require(isinstance(recovery.get(field), str) and SHA256.fullmatch(recovery[field]) is not None,
                "Missing recovery original journal fingerprint")
    return extra


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def load_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_unique_object,
                       parse_constant=lambda text: (_ for _ in ()).throw(ValueError(f"Invalid JSON number: {text}")))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path.name}")
    return value


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _record(record: dict, label: str) -> None:
    require(isinstance(record, dict) and isinstance(record.get("sha256"), str)
            and SHA256.fullmatch(record["sha256"]) is not None, f"Missing SHA-256: {label}")
    require(type(record.get("bytes")) is int and record["bytes"] > 0, f"Invalid byte count: {label}")


def _public_metadata(value: object) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            require(str(key).lower() not in PRIVATE_KEYS, f"Private transcript/credential field in public manifest: {key}")
            _public_metadata(child)
    elif isinstance(value, list):
        for child in value:
            _public_metadata(child)
    elif isinstance(value, str):
        require(SECRET_TEXT.search(value) is None, "Possible credential in public manifest")


def read_vectors(path: Path, shape: list[int]) -> tuple[list[list[float]], dict]:
    """Check the sole Safetensors tensor, all values and per-layer/trait norms."""
    require(len(shape) == 3 and shape[:2] == [32, 6] and type(shape[2]) is int and 0 < shape[2] <= 65536,
            "Expected [32,6,hidden_size] bank")
    with path.open("rb") as handle:
        prefix = handle.read(8)
        require(len(prefix) == 8, "Truncated Safetensors length")
        header_size = struct.unpack("<Q", prefix)[0]
        require(0 < header_size <= 1024 * 1024, "Invalid Safetensors header length")
        header = json.loads(handle.read(header_size), object_pairs_hook=_unique_object)
        require(isinstance(header, dict) and set(header) - {"__metadata__"} == {"vectors"},
                "Safetensors bank must contain exactly the vectors tensor")
        _public_metadata(header.get("__metadata__", {}))
        tensor = header["vectors"]
        expected_bytes = math.prod(shape) * 4
        require(tensor == {"dtype": "F32", "shape": shape, "data_offsets": [0, expected_bytes]},
                "Safetensors key/dtype/shape/offset mismatch")
        payload = handle.read(expected_bytes + 1)
        require(len(payload) == expected_bytes, "Safetensors payload is truncated or has trailing data")
    values = array("f")
    values.frombytes(payload)
    require(values.itemsize == 4, "Unsupported native float representation")
    if sys.byteorder != "little":
        values.byteswap()
    require(all(math.isfinite(value) for value in values), "Vector bank contains nonfinite values")
    hidden = shape[2]
    norms = []
    for layer in range(32):
        row = []
        for trait in range(6):
            start = (layer * 6 + trait) * hidden
            norm = math.sqrt(math.fsum(float(value) ** 2 for value in values[start:start + hidden]))
            require(math.isfinite(norm), f"Invalid vector norm at layer {layer}, trait {TRAITS[trait]}")
            row.append(norm)
        norms.append(row)
    zeros = [[layer, trait] for layer in range(32) for trait in range(6) if norms[layer][trait] == 0]
    zero_traits = [trait for trait in range(6) if all(norms[layer][trait] == 0 for layer in range(32))]
    return norms, {"shape": shape, "dtype": "float32", "all_finite": True,
                   "nonzero_vectors": 32 * 6 - len(zeros), "zero_vector_count": len(zeros),
                   "zero_vector_indices": zeros, "entire_zero_trait_indices": zero_traits,
                   "entire_zero_bank": len(zeros) == 32 * 6,
                   "sha256": digest(path), "bytes": path.stat().st_size}


def validate_bank(root: Path, *, source_root: Path | None = None) -> dict:
    manifest = load_json(root / MANIFEST_FILENAME)
    _public_metadata(manifest)
    require(manifest.get("schema_version") == 1 and manifest.get("artifact_type") == "mooody_persona_vector_bank",
            "Unexpected persona manifest schema")
    require(manifest.get("status") == "extracted_not_behaviorally_validated", "Bank must distinguish extraction from behavioral validation")
    require(tuple(manifest.get("trait_order", [])) == TRAITS, "Incorrect trait order")
    checkpoint, tokenizer = manifest.get("checkpoint", {}), manifest.get("tokenizer", {})
    for label, record in (("checkpoint", checkpoint), ("tokenizer", tokenizer)):
        require(record.get("model_id") == MODEL_ID and record.get("revision") == MODEL_REVISION,
                f"{label} is not the pinned abliterated checkpoint")
    require(checkpoint.get("decoder_layers") == 32 and checkpoint.get("hidden_size") == HIDDEN_SIZE
            and checkpoint.get("config_sha256") == CONFIG_SHA256,
            "Missing checkpoint architecture/hash provenance")
    file_hashes = tokenizer.get("file_hashes", {})
    require(isinstance(file_hashes, dict) and {"tokenizer.json", "tokenizer_config.json", "chat_template.jinja"} <= set(file_hashes),
            "Missing tokenizer/chat-template hashes")
    for name, record in file_hashes.items():
        _record(record, f"tokenizer/{name}")
    require(all(file_hashes[name] == expected for name, expected in TOKENIZER_HASHES.items()),
            "Tokenizer hashes differ from the pinned audited checkpoint")
    sources = manifest.get("sources", {})
    require(isinstance(sources, dict) and SOURCE_FILES <= set(sources), "Missing frozen data/protocol hashes")
    for relative, expected in sources.items():
        path = PurePosixPath(relative)
        require(not path.is_absolute() and ".." not in path.parts and str(path) == relative, "Unsafe source path")
        _record(expected, relative)
        if source_root is not None:
            source = source_root / relative
            require(source.is_file() and source.stat().st_size == expected["bytes"] and digest(source) == expected["sha256"],
                    f"Current source no longer matches extraction: {relative}")
    extraction = manifest.get("extraction", {})
    require(extraction.get("activation_boundary") == BOUNDARY and extraction.get("pooling") == POOLING
            and extraction.get("raw_normalization") == "none" and extraction.get("retain_layers") == "all_decoder_layers"
            and extraction.get("select_best_layer") is False and extraction.get("position_axis") is False,
            "Wrong extraction boundary/pooling/layer retention")
    inference = manifest.get("inference", {})
    require(inference.get("method") == "direct_raw_all_layers" and inference.get("activation_boundary") == BOUNDARY
            and inference.get("coefficients") == [-2, -1, 0, 1, 2]
            and inference.get("token_scope") == "final_formatted_prompt_then_generated_content"
            and not any(key in inference for key in ("gain", "gains", "incremental_definition")),
            "Inference must use direct raw addition without gains or incremental directions")
    generation = manifest.get("generation", {})
    require(generation.get("total_responses_before_filtering") == 2400, "Wrong generated response budget")
    if "native_sampling_provenance" in generation:
        sampling = generation["native_sampling_provenance"]
        counts = generation.get("execution_counts_by_trait", {})
        require(isinstance(counts, dict) and set(counts) == set(TRAITS)
                and all(isinstance(row, dict) and set(row) == {"serial", BATCHED_STAGE}
                        and all(type(number) is int and 0 <= number <= 400 for number in row.values())
                        and sum(row.values()) == 400 for row in counts.values()),
                "Inconsistent serial/batched condition counts")
        require(isinstance(sampling, dict) and sampling.get("model_id") == MODEL_ID
                and sampling.get("model_revision") == MODEL_REVISION
                and sampling.get("checkpoint_generation_config_sha256") == GENERATION_CONFIG_SHA256
                and sampling.get("batch_size") == 8 and sampling.get("enable_thinking") is False
                and isinstance(sampling.get("arithmetic_note"), str) and bool(sampling["arithmetic_note"]),
                "Missing pinned native/batched generation provenance")
        resolved = sampling.get("resolved_config", {})
        require(isinstance(resolved, dict) and all(resolved.get(key) == value for key, value in {
            "do_sample": True, "temperature": 1.0, "top_p": 1.0, "top_k": 50,
            "repetition_penalty": 1.0, "max_new_tokens": 1000, "use_cache": True}.items()),
            "Batched native sampling differs from the fixed collection")
        stage = manifest.get("implementations", {}).get(BATCHED_STAGE, {})
        require(sampling.get("resolved_config_sha256") == metadata_fingerprint(resolved)
                and stage.get("sampling_receipt_sha256") == metadata_fingerprint(sampling)
                and generation.get("batched_implementation_sha256") == stage.get("implementation_sha256")
                and isinstance(stage.get("implementation_sha256"), str)
                and SHA256.fullmatch(stage["implementation_sha256"]) is not None
                and stage["implementation_sha256"] == metadata_fingerprint({key: value for key, value in stage.items()
                                                                            if key != "implementation_sha256"}),
                "Batched generation receipt/implementation hash mismatch")
    judging = manifest.get("judging", {})
    require(judging.get("gateway") == "openrouter" and judging.get("model") == "google/gemini-3.8-flash"
            and judging.get("planned_scoring_calls_before_retries") == 4800, "Missing actual judge configuration/counts")
    require(judging.get("provider") == {"order": ["google-ai-studio"], "only": ["google-ai-studio"],
                                       "allow_fallbacks": False, "require_parameters": True}
            and judging.get("reasoning") == {"effort": "low", "exclude": True},
            "Judge model/provider/reasoning differs from the frozen collection protocol")
    require(bool(judging.get("provenance")), "Missing judge provenance summary/hash references")
    filtering = manifest.get("filtering", {})
    require(filtering.get("policy") == FILTERING_POLICY
            and filtering.get("policy_source") == POLICY_SOURCE
            and filtering.get("policy_sha256") == POLICY_SHA256
            and sources[POLICY_SOURCE] == {"sha256": POLICY_SHA256, "bytes": POLICY_BYTES},
            "Missing or different explicit filtering policy/hash/source")
    records = filtering.get("traits", {})
    require(set(records) == set(TRAITS), "Filtering counts must cover all six traits")
    accepted = {}
    for trait, record in records.items():
        count = record.get("accepted_pairs")
        require(type(count) is int and 1 <= count <= 200 and record.get("generated_positive") == 200
                and record.get("generated_negative") == 200 and record.get("accepted_positive") == count
                and record.get("accepted_negative") == count, f"Invalid matched response counts: {trait}")
        missing = {}
        for key, size in (("accepted_by_question", 40), ("accepted_by_system_prompt_pair", 5)):
            coverage = record.get(key, {})
            require(isinstance(coverage, dict) and set(coverage) == {f"{i:02d}" for i in range(1, size + 1)}
                    and all(type(number) is int and 0 <= number <= 200 // size for number in coverage.values())
                    and sum(coverage.values()) == count, f"Incomplete/inconsistent extraction coverage: {trait}/{key}")
            missing[key] = sorted(identity for identity, number in coverage.items() if number == 0)
        require(record.get("missing_questions") == missing["accepted_by_question"]
                and record.get("missing_system_prompt_pairs") == missing["accepted_by_system_prompt_pair"]
                and record.get("coverage_passed") is (not any(missing.values()))
                and record.get("extraction_usable") is True and record.get("policy_sha256") == POLICY_SHA256,
                f"Inconsistent accepted coverage/usability diagnostics: {trait}")
        for polarity in ("positive", "negative"):
            generated_capped = record.get(f"generated_capped_{polarity}")
            accepted_capped = record.get(f"accepted_capped_{polarity}")
            require(type(generated_capped) is int and type(accepted_capped) is int
                    and 0 <= accepted_capped <= min(generated_capped, count) <= 200
                    and 0 <= generated_capped <= 200,
                    f"Inconsistent capped-response counts: {trait}/{polarity}")
        capped_pairs = record.get("accepted_capped_pairs")
        capped_positive, capped_negative = record["accepted_capped_positive"], record["accepted_capped_negative"]
        require(type(capped_pairs) is int
                and max(capped_positive, capped_negative) <= capped_pairs <= min(count, capped_positive + capped_negative),
                f"Inconsistent capped matched-pair count: {trait}")
        accepted[trait] = count
    attempts = judging.get("actual_attempts")
    valid_scores = judging.get("valid_scores")
    recovery_extra = validate_recovery(manifest, sources)
    require(type(attempts) is int and type(valid_scores) is int
            and 4 * sum(accepted.values()) <= valid_scores <= 4800
            and valid_scores <= attempts and 0 <= attempts - recovery_extra <= 14400,
            "Judge score/attempt counts cannot support the accepted contrasts")
    tensor = manifest.get("tensor", {})
    require(tensor.get("filename") == BANK_FILENAME and tensor.get("key") == "vectors"
            and tensor.get("dtype") == "float32" and tensor.get("all_finite") is True,
            "Wrong vector artifact metadata")
    _record(tensor, "vector bank")
    require(tensor.get("shape") == [32, 6, checkpoint.get("hidden_size")], "Bank shape differs from pinned text architecture")
    norms, actual = read_vectors(root / BANK_FILENAME, tensor["shape"])
    require(actual["sha256"] == tensor["sha256"] and actual["bytes"] == tensor["bytes"], "Vector bank content hash/size mismatch")
    declared = tensor.get("l2_norms", [])
    require(isinstance(declared, list) and len(declared) == 32 and all(isinstance(row, list) and len(row) == 6 for row in declared),
            "Missing per-layer/trait norms")
    for layer, row in enumerate(declared):
        for trait, value in enumerate(row):
            require(type(value) in (int, float) and math.isfinite(value)
                    and math.isclose(value, norms[layer][trait], rel_tol=1e-5, abs_tol=1e-8), "Declared vector norm mismatch")
    if actual["zero_vector_indices"] or "zero_vector_indices" in tensor:
        require(tensor.get("zero_vector_indices") == actual["zero_vector_indices"], "Missing or incorrect zero-vector flags")
    return {"run_id": manifest.get("run_id"), "checkpoint": checkpoint,
            "trait_order": list(TRAITS), "tensor": actual, "accepted_pairs_by_trait": accepted,
            "manifest_sha256": digest(root / MANIFEST_FILENAME), "all_sources_verified": source_root is not None}


def model_card(manifest: dict) -> str:
    tensor = manifest["tensor"]
    zeros = tensor.get("zero_vector_indices", [])
    zero_traits = [TRAITS[trait] for trait in range(6) if all([layer, trait] in zeros for layer in range(32))]
    records = manifest["filtering"]["traits"]
    rows = "\n".join(
        f"| {trait} | {records[trait]['accepted_pairs']} | {records[trait]['accepted_capped_pairs']} | "
        f"{40 - len(records[trait]['missing_questions'])}/40 | {5 - len(records[trait]['missing_system_prompt_pairs'])}/5 |"
        for trait in TRAITS)
    generation = manifest["generation"]
    batching_note = ""
    if "native_sampling_provenance" in generation:
        execution = generation["execution_counts_by_trait"]
        serial = sum(row["serial"] for row in execution.values())
        batched = sum(row[BATCHED_STAGE] for row in execution.values())
        batching_note = f"""
Recorded execution counts are {serial} serial and {batched} batched conditions.
The continuation uses batch size up to eight, an independent random stream per
condition, and the same fixed sampling settings, including native top-k 50.
Batched BF16 arithmetic can change sampled text despite the same seed; bitwise
equivalence with serial generation is not claimed. The manifest's native sampling
receipt and implementation hashes record this mixed execution.
"""
    recovery_note = ""
    if "transport_recovery" in manifest["judging"]:
        counts = manifest["judging"]["transport_recovery"]["counts"]
        recovery_note = f"""
An independently recorded HTTP429 transport recovery made {counts['additional_attempts']}
additional judge attempts across {counts['records_touched']} previously unscored
journals, recovering {counts['recovered_scores']} scores; {counts['remaining_none']}
remain unscored. Only journals with three completed original HTTP429 failures
were eligible, with at most three extra attempts per score under the unchanged
request and rubrics. Existing valid scores and original journals were preserved.
This adds no target-model rollouts and no logical score slots; updated scores
were used for fresh filtering and affected vector means. It does not validate
the judge or steering behavior.
"""
    return f"""---
license: apache-2.0
base_model: {MODEL_ID}
base_model_relation: adapter
tags:
- activation-steering
- persona-vectors
- qwen3.5
- experimental
---

# Mooody all-layer persona vectors

Six real contrastive activation directions extracted from the pinned abliterated
[{MODEL_ID}](https://huggingface.co/{MODEL_ID}/tree/{MODEL_REVISION}) checkpoint at
revision `{MODEL_REVISION}`. These are steering vectors, not model weights.

`{BANK_FILENAME}` contains the sole tensor `vectors`, raw FP32 with shape
`{tensor['shape']}` and axes `(decoder layer, trait, hidden coordinate)`.
Trait order is `{', '.join(TRAITS)}`. All 32 decoder layers are retained, with no
normalization, layer selection, incremental differencing or gain rescaling.
Direct additions use coefficients -2, -1, 0, 1 or 2 at each decoder output before
the final global RMSNorm, beginning at the final formatted prompt token and
continuing through generated assistant content.
This inference rule differs from the paper's layer selection and Appendix J.3
incremental layer differences; its cumulative effects remain unvalidated.

This bank contains {32 * 6 - len(zeros)} nonzero layer/trait vectors and
{len(zeros)} finite zero vectors. Zero coordinates `[layer, trait]` are `{zeros}`;
entirely zero traits are `{zero_traits}`. All coordinates are zero-based.
These raw results are retained and flagged; no direction is pruned or fabricated.

## Extraction and provenance

We adapt the [Persona Vectors](https://arxiv.org/abs/2507.21509) response-difference
method using five contrastive system-prompt pairs and all 40 authored questions
per trait, with one saved response at most per condition. All 2,400 planned
conditions are attempted; existing saved responses are retained unchanged.
Generation failures and unusable responses are excluded.
All 40 questions belong to extraction; there is no held-out question set.
Gemini 3.8 Flash through OpenRouter supplies target-trait and coherence
scores. Matched pairs require positive trait score >50, negative score <50,
and coherence >=50 on both sides. Per-response content activations are averaged
first, then responses receive equal weight within each polarity.
{batching_note}
{recovery_note}

Filtering follows the explicit `{FILTERING_POLICY['policy_id']}` policy recorded
in the manifest, with SHA-256 `{POLICY_SHA256}`. The original collection protocol
and generated responses are preserved. Responses that reach the 1,000-token cap
are eligible only if the same blind trait and coherence gates pass; the cap does
not automatically accept them. Missing accepted questions or system pairs are
reported as coverage diagnostics. Each trait requires nonempty matched groups;
complete accepted coverage is not a release requirement or a quality claim.

| Trait | Accepted matched pairs | Accepted pairs with a capped response | Accepted question coverage | Accepted system-pair coverage |
| --- | --- | --- | --- | --- |
{rows}

[{MANIFEST_FILENAME}]({MANIFEST_FILENAME}) records frozen checkpoint/tokenizer
revisions and hashes, input/protocol hashes, counts and coverage, judge provenance,
activation boundaries, shape, checksum and all layer/trait norms.
[{PUBLICATION_FILENAME}]({PUBLICATION_FILENAME}) hashes the public payload.
Generated responses, raw judge transcripts, secrets and private run logs are
excluded from this repository.

## Validation status

Tensor shape, finiteness, checksums and recorded provenance have been checked.
No independent behavioral evaluation or held-out validation has been run.
Behavioral effectiveness, cross-trait isolation, task quality and judge reliability
have not been established.
The names describe operationally expressed personas, not clinical diagnoses or
claims about a model's subjective feelings. Tone and verbosity can confound these
directions. Use only with the exact pinned checkpoint and matching hook boundary;
do not interpret extraction or successful loading as behavioral validation.

The base model's Apache-2.0 license is retained in [LICENSE](LICENSE).
"""


def stage_release(run_dir: Path, destination: Path, *, license_path: Path, source_root: Path | None = None) -> dict:
    result = validate_bank(run_dir, source_root=source_root)
    require(not destination.exists() or not any(destination.iterdir()), "Release destination must be empty")
    require(license_path.is_file() and "Apache License" in license_path.read_text(encoding="utf-8"), "Missing inherited Apache license")
    destination.mkdir(parents=True, exist_ok=True)
    for name in (BANK_FILENAME, MANIFEST_FILENAME):
        shutil.copyfile(run_dir / name, destination / name)
    shutil.copyfile(license_path, destination / "LICENSE")
    (destination / "README.md").write_text(model_card(load_json(destination / MANIFEST_FILENAME)), encoding="utf-8")
    files = {name: {"sha256": digest(destination / name), "bytes": (destination / name).stat().st_size}
             for name in sorted(PUBLIC_FILES - {PUBLICATION_FILENAME})}
    publication = {"schema_version": 1, "artifact_type": "mooody_persona_vectors_publication",
                   "run_id": result["run_id"], "files": files,
                   "raw_transcripts_included": False,
                   "manifest_scope": "All public files except this publication manifest itself"}
    (destination / PUBLICATION_FILENAME).write_bytes(json_bytes(publication))
    return verify_release(destination)


def verify_release(root: Path) -> dict:
    actual_names = {path.name for path in root.iterdir() if path.is_file()}
    require(actual_names == PUBLIC_FILES, "Public release has missing or unlisted payload files")
    # HF download bookkeeping is allowed; no other directories or symlinks are payload.
    require(all(not path.is_symlink() and (path.is_file() or path.name == ".cache") for path in root.iterdir()),
            "Unexpected directory/symlink in release")
    publication = load_json(root / PUBLICATION_FILENAME)
    require(publication.get("raw_transcripts_included") is False
            and set(publication.get("files", {})) == PUBLIC_FILES - {PUBLICATION_FILENAME}, "Invalid publication file allowlist")
    _public_metadata(publication)
    for name, expected in publication["files"].items():
        _record(expected, name)
        require((root / name).stat().st_size == expected["bytes"] and digest(root / name) == expected["sha256"],
                f"Publication payload hash mismatch: {name}")
    for name in ("README.md", "LICENSE"):
        require(SECRET_TEXT.search((root / name).read_text(encoding="utf-8")) is None, "Possible credential in public text")
    result = validate_bank(root)
    result["publication_manifest_sha256"] = digest(root / PUBLICATION_FILENAME)
    result["files_verified"] = len(PUBLIC_FILES)
    return result
