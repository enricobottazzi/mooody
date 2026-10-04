"""Load only integrity-checked files from the pinned Mooody release."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re

MANIFEST = Path(__file__).with_name("checkpoint_manifest.json")
EXISTING_CHECKPOINT = Path("/artifacts/runs/qwen35-9b-arditi-20261003-v1/checkpoint")
PERSONA_RELEASE = Path(__file__).with_name("persona_release.json")
PERSONA_AXES = ("depression", "curiosity", "paranoia", "sexual_arousal", "narcissism", "euphoria")
PERSONA_BOUNDARY = "decoder_block_output_pre_final_global_rmsnorm"
PERSONA_HIDDEN_SIZE = 4096
PERSONA_POLICY_SOURCE = "data/persona_traits/extraction_policy.json"
PERSONA_POLICY_SHA256 = "fcec10d4f4b6443b8fd9d7e408ad8853fc2d3c798ebce7e022f91d2a9d3c89b4"
PERSONA_POLICY_BYTES = 1350
PERSONA_RECOVERY_STAGE = "judging_recovery_v1"
PERSONA_RECOVERY_POLICY_SOURCE = "data/persona_traits/judging_recovery_policy.json"
PERSONA_RECOVERY_POLICY_SHA256 = "f9fa377240ac3e44903b0cf2d3ed312c1a9e770a2c02b3af2b7f12c0d5799d72"
PERSONA_RECOVERY_POLICY_BYTES = 1129
PERSONA_RECOVERY_POLICY_FINGERPRINT = "a8eacc81a955f948abcfb61798a440cb51353cde4604b98a128031f94329fc27"
# Serving images carry deployment code only. Pin the approved analytical policy
# here as well as its exact source-file hash; the collection protocol is unchanged.
PERSONA_FILTERING_POLICY = {
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


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_checkpoint(root: Path, manifest: dict) -> bool:
    for relative, expected in manifest["files"].items():
        path = root / relative
        if not path.is_file() or path.stat().st_size != expected["bytes"] or sha256(path) != expected["sha256"]:
            return False
    index = json.loads((root / "model.safetensors.index.json").read_text())
    if len(index["weight_map"]) != 775 or len(set(index["weight_map"].values())) != 7:
        return False
    return all(shard in manifest["files"] for shard in set(index["weight_map"].values()))


def get_checkpoint(commit=lambda: None) -> Path:
    from deployment.core import MODEL_ID, MODEL_REVISION

    cache_checkpoint = Path("/artifacts/serving") / MODEL_REVISION
    manifest = json.loads(MANIFEST.read_text())
    if manifest["model_id"] != MODEL_ID or manifest["revision"] != MODEL_REVISION:
        raise RuntimeError("The checkpoint manifest does not describe the configured pinned release")
    # The published weights are identical to the already audited experiment.
    # Compare every required file, including all complete shard hashes.
    if EXISTING_CHECKPOINT.exists() and verify_checkpoint(EXISTING_CHECKPOINT, manifest):
        return EXISTING_CHECKPOINT
    if cache_checkpoint.exists() and verify_checkpoint(cache_checkpoint, manifest):
        return cache_checkpoint
    from huggingface_hub import snapshot_download

    snapshot_download(
        MODEL_ID, revision=MODEL_REVISION, local_dir=cache_checkpoint,
        allow_patterns=list(manifest["files"]), max_workers=4,
    )
    if not verify_checkpoint(cache_checkpoint, manifest):
        raise RuntimeError("Downloaded model files do not match the audited publication hashes")
    commit()
    return cache_checkpoint


def persona_release() -> dict:
    """Read a complete immutable locator; never substitute main/latest/random."""
    release = json.loads(PERSONA_RELEASE.read_text()) if PERSONA_RELEASE.is_file() else {}
    if not isinstance(release, dict):
        raise RuntimeError("Persona release locator must be a JSON object")
    overrides = {
        "repo_id": "MOOODY_PERSONA_REPO_ID", "revision": "MOOODY_PERSONA_REVISION",
        "manifest_filename": "MOOODY_PERSONA_MANIFEST_FILENAME",
    }
    for field, variable in overrides.items():
        if variable in os.environ:
            release[field] = os.environ[variable]
    release.setdefault("manifest_filename", "persona_manifest.json")
    release.setdefault("repo_type", "model")
    if not isinstance(release.get("repo_id"), str) or not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", release["repo_id"]):
        raise RuntimeError("A published persona bank repository must be configured")
    if not isinstance(release.get("revision"), str) or not re.fullmatch(r"[0-9a-f]{40}", release["revision"]):
        raise RuntimeError("Persona bank revision must be an exact pinned commit SHA")
    _safe_artifact_filename(release["manifest_filename"])
    if release["repo_type"] not in ("model", "dataset"):
        raise RuntimeError("Persona bank repository type must be model or dataset")
    return release


def _safe_artifact_filename(value):
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", value) or value in (".", "..") or Path(value).name != value:
        raise RuntimeError("Persona bank artifact filenames must be safe basenames")


def configured_persona_metadata() -> dict:
    """Dependency-free public configuration, without downloading or loading tensors."""
    from deployment.core import MOOD_COEFFICIENTS

    metadata = {
        "mood_vectors_available": False, "steering_available": False,
        "mood_vectors_source": "unavailable", "mood_vectors_validated": False,
        "steering_method": "paper_incremental_all_layers",
        "steering_incremental_definition": "raw_layer_vector_minus_previous_layer_vector",
        "steering_first_layer_previous_vector": "zero",
        "steering_activation_boundary": PERSONA_BOUNDARY,
        "steering_token_scope": "final_formatted_prompt_token_then_generated_content_tokens",
        "mood_coefficients": list(MOOD_COEFFICIENTS),
    }
    try:
        release = persona_release()
    except (OSError, ValueError, TypeError, RuntimeError):
        return metadata
    return {
        **metadata, "mood_vectors_available": True, "steering_available": True,
        "mood_vectors_source": "persona_vectors",
        "mood_vectors_published_inference_method": "direct_raw_all_layers",
        "mood_vectors_repo_id": release["repo_id"],
        "mood_vectors_revision": release["revision"],
        "mood_vectors_manifest": release["manifest_filename"],
    }


def _metadata_fingerprint(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     allow_nan=False).encode()).hexdigest()


def _validate_transport_recovery(manifest: dict, sources: dict) -> None:
    """Check the optional transport receipt; full journal audit belongs to publication."""
    judging = manifest["judging"]
    recovery = judging.get("transport_recovery")
    if recovery is None:
        if PERSONA_RECOVERY_POLICY_SOURCE in sources:
            raise RuntimeError("Persona recovery policy source requires transport recovery provenance")
        return
    expected_source = {"sha256": PERSONA_RECOVERY_POLICY_SHA256, "bytes": PERSONA_RECOVERY_POLICY_BYTES}
    if (not isinstance(recovery, dict) or recovery.get("stage") != PERSONA_RECOVERY_STAGE
            or recovery.get("policy_source") != PERSONA_RECOVERY_POLICY_SOURCE
            or recovery.get("policy_sha256") != PERSONA_RECOVERY_POLICY_SHA256
            or sources.get(PERSONA_RECOVERY_POLICY_SOURCE) != expected_source
            or _metadata_fingerprint(recovery.get("policy")) != PERSONA_RECOVERY_POLICY_FINGERPRINT):
        raise RuntimeError("Persona transport recovery policy/source differs from the approved repair")
    stage = recovery.get("implementation", {})
    implementations = manifest.get("implementations", {})
    if (not isinstance(stage, dict) or stage != implementations.get(PERSONA_RECOVERY_STAGE)
            or stage.get("stage") != PERSONA_RECOVERY_STAGE
            or stage.get("policy_source") != PERSONA_RECOVERY_POLICY_SOURCE
            or stage.get("policy_sha256") != PERSONA_RECOVERY_POLICY_SHA256
            or stage.get("policy_bytes") != PERSONA_RECOVERY_POLICY_BYTES
            or stage.get("policy") != recovery["policy"]
            or stage.get("original_judging_implementation_sha256") != implementations.get("judging", {}).get("implementation_sha256")
            or any(not isinstance(stage.get(field), str) or not re.fullmatch(r"[0-9a-f]{64}", stage[field])
                   for field in ("module_sha256", "original_judging_implementation_sha256"))
            or stage.get("implementation_sha256") != _metadata_fingerprint({key: value for key, value in stage.items() if key != "implementation_sha256"})):
        raise RuntimeError("Persona transport recovery implementation receipt differs from its provenance")
    audit = recovery.get("audit", {})
    audit_hash = hashlib.sha256((json.dumps(audit, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode()).hexdigest()
    if (not isinstance(audit, dict) or audit.get("status") != "complete"
            or audit.get("stage") != PERSONA_RECOVERY_STAGE
            or audit.get("policy_sha256") != PERSONA_RECOVERY_POLICY_SHA256
            or audit.get("implementation_sha256") != stage["implementation_sha256"]
            or recovery.get("audit_sha256") != audit_hash
            or recovery.get("valid_and_ineligible_original_journals_preserved") is not True
            or any(not isinstance(recovery.get(field), str) or not re.fullmatch(r"[0-9a-f]{64}", recovery[field])
                   for field in ("original_journal_hashes_sha256", "baseline_journal_hashes_sha256"))):
        raise RuntimeError("Persona transport recovery audit is incomplete or its fingerprint changed")
    counts = recovery.get("counts", {})
    fields = ("records_touched", "additional_attempts", "recovered_scores", "remaining_none",
              "original_http429_attempts", "http429_recovery_attempts")
    if not isinstance(counts, dict) or any(type(counts.get(field)) is not int or counts[field] < 0 for field in fields):
        raise RuntimeError("Persona transport recovery aggregate counts are missing")
    touched, extra, recovered, remaining = (counts[field] for field in fields[:4])
    before, after = audit.get("before", {}), audit.get("after", {})
    if (not 0 <= touched <= 4800 or not touched <= extra <= 3 * touched or recovered + remaining != touched
            or counts["original_http429_attempts"] != 3 * touched or counts["http429_recovery_attempts"] > extra - recovered
            or before.get("planned_score_slots") != 4800 or after.get("planned_score_slots") != 4800
            or before.get("eligible_http429_none") != touched or after.get("eligible_http429_none") != remaining
            or before.get("recovery_actual_attempts") != 0 or after.get("recovery_actual_attempts") != extra
            or before.get("original_actual_attempts") != after.get("original_actual_attempts")
            or judging.get("score_records") != 4800 or judging.get("valid_scores") != after.get("valid_scores")
            or judging.get("actual_attempts") != after.get("actual_attempts")):
        raise RuntimeError("Persona transport recovery counts contradict the unchanged score budget")


def validate_persona_manifest(manifest: dict) -> dict:
    """Validate the immutable raw bank and its original publication provenance."""
    from deployment.core import MODEL_ID, MODEL_REVISION

    if not isinstance(manifest, dict) or manifest.get("schema_version") != 1:
        raise RuntimeError("Unsupported persona bank manifest")
    if manifest.get("artifact_type") != "mooody_persona_vector_bank" or manifest.get("status") != "extracted_not_behaviorally_validated":
        raise RuntimeError("Serving requires an extracted persona bank with honest validation status")
    if manifest.get("trait_order") != list(PERSONA_AXES) or not manifest.get("run_id"):
        raise RuntimeError("Persona bank must contain all six ordered traits and extraction run provenance")
    for name in ("checkpoint", "tokenizer"):
        info = manifest.get(name, {})
        if info.get("model_id") != MODEL_ID or info.get("revision") != MODEL_REVISION:
            raise RuntimeError(f"Persona bank {name} differs from the pinned serving checkpoint")
    audited = json.loads(MANIFEST.read_text())
    checkpoint = manifest["checkpoint"]
    if (checkpoint.get("decoder_layers") != 32 or checkpoint.get("hidden_size") != PERSONA_HIDDEN_SIZE
            or checkpoint.get("config_sha256") != audited["files"]["config.json"]["sha256"]):
        raise RuntimeError("Persona bank architecture/config provenance differs from the audited checkpoint")
    tokenizer_files = manifest["tokenizer"].get("file_hashes", {})
    if not isinstance(tokenizer_files, dict) or any(
        tokenizer_files.get(name) != audited["files"][name]
        for name in ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja")
    ):
        raise RuntimeError("Persona bank tokenizer/template hashes differ from the audited checkpoint")
    extraction = manifest.get("extraction", {})
    expected = {
        "activation_boundary": PERSONA_BOUNDARY,
        "pooling": "response_content_mean_then_equal_response_group_mean",
        "raw_normalization": "none", "retain_layers": "all_decoder_layers",
        "select_best_layer": False, "position_axis": False,
    }
    if any(extraction.get(key) != value for key, value in expected.items()):
        raise RuntimeError("Persona extraction semantics differ from raw block-output vectors")
    inference = manifest.get("inference", {})
    # This historical field belongs to the immutable publication. Current
    # serving derives layer increments without rewriting that bank or manifest.
    expected_inference = {
        "method": "direct_raw_all_layers", "activation_boundary": PERSONA_BOUNDARY,
        "coefficients": [-2, -1, 0, 1, 2],
        "token_scope": "final_formatted_prompt_then_generated_content",
    }
    if any(inference.get(key) != value for key, value in expected_inference.items()) or any(
        key in inference for key in ("gain", "gains", "incremental_definition")
    ):
        raise RuntimeError("Persona bank original inference provenance is incompatible")
    sources = manifest.get("sources", {})
    required_sources = [
        "data/persona_traits/protocol.json", "data/persona_traits/manifest.json",
        "data/persona_traits/coherence_evaluation_prompt.txt", PERSONA_POLICY_SOURCE, *[
        f"data/persona_traits/source/{trait}.json" for trait in PERSONA_AXES
    ]]
    if not isinstance(sources, dict) or any(path not in sources for path in required_sources):
        raise RuntimeError("Persona bank source hashes are incomplete")
    for info in sources.values():
        if not isinstance(info, dict) or not re.fullmatch(r"[0-9a-f]{64}", str(info.get("sha256", ""))) or type(info.get("bytes")) is not int or info["bytes"] <= 0:
            raise RuntimeError("Persona bank source integrity provenance is invalid")
    filtering = manifest.get("filtering", {})
    if not isinstance(filtering, dict):
        raise RuntimeError("Persona bank filtering policy provenance is missing")
    policy = filtering.get("policy")
    if not isinstance(policy, dict) or json.dumps(policy, sort_keys=True) != json.dumps(PERSONA_FILTERING_POLICY, sort_keys=True):
        raise RuntimeError("Persona bank filtering policy differs from the approved analytical policy")
    if (filtering.get("policy_source") != PERSONA_POLICY_SOURCE
            or filtering.get("policy_sha256") != PERSONA_POLICY_SHA256
            or sources[PERSONA_POLICY_SOURCE] != {"sha256": PERSONA_POLICY_SHA256, "bytes": PERSONA_POLICY_BYTES}):
        raise RuntimeError("Persona bank filtering policy source integrity is invalid")
    judging = manifest.get("judging", {})
    if not isinstance(judging, dict) or judging.get("gateway") != "openrouter" or (
        judging.get("model") != "google/gemini-3.8-flash"
        or judging.get("planned_scoring_calls_before_retries") != 4800
        or not judging.get("provenance")
    ):
        raise RuntimeError("Persona bank judge provenance is missing")
    _validate_transport_recovery(manifest, sources)
    traits = filtering.get("traits", {})
    for trait in PERSONA_AXES:
        info = traits.get(trait, {})
        count = info.get("accepted_pairs")
        expected_counts = {"generated_positive": 200, "generated_negative": 200,
                           "accepted_positive": count, "accepted_negative": count}
        if type(count) is not int or not 1 <= count <= 200 or any(
            type(info.get(key)) is not int or info[key] != value for key, value in expected_counts.items()
        ):
            raise RuntimeError(f"Persona bank has invalid matched accepted coverage for {trait}")
        if info.get("policy_sha256") != PERSONA_POLICY_SHA256:
            raise RuntimeError(f"Persona bank trait policy provenance differs for {trait}")
        missing = {}
        for key, width in (("accepted_by_question", 40), ("accepted_by_system_prompt_pair", 5)):
            coverage = info.get(key, {})
            if not isinstance(coverage, dict) or any(type(coverage.get(f"{i:02}")) is not int or not 0 <= coverage[f"{i:02}"] <= 200 // width for i in range(1, width + 1)):
                raise RuntimeError(f"Persona bank is missing extraction coverage for {trait}")
            if set(coverage) != {f"{i:02}" for i in range(1, width + 1)} or sum(coverage.values()) != count:
                raise RuntimeError(f"Persona bank extraction coverage counts disagree for {trait}")
            missing[key] = [key for key, value in coverage.items() if value == 0]
        coverage_passed = not any(missing.values())
        if info.get("coverage_passed") is not coverage_passed or info.get("extraction_usable") is not True:
            raise RuntimeError(f"Persona bank extraction diagnostic flags disagree for {trait}")
        for field, key in (("missing_questions", "accepted_by_question"),
                           ("missing_system_prompt_pairs", "accepted_by_system_prompt_pair")):
            if info.get(field) != sorted(missing[key]):
                raise RuntimeError(f"Persona bank missing-coverage diagnostics disagree for {trait}")
        capped_fields = ("generated_capped_positive", "generated_capped_negative",
                         "accepted_capped_positive", "accepted_capped_negative", "accepted_capped_pairs")
        if any(type(info.get(field)) is not int or info[field] < 0 for field in capped_fields):
            raise RuntimeError(f"Persona bank capped-response diagnostics are invalid for {trait}")
        p, n = info["accepted_capped_positive"], info["accepted_capped_negative"]
        if (info["generated_capped_positive"] > 200 or info["generated_capped_negative"] > 200
                or p > min(count, info["generated_capped_positive"])
                or n > min(count, info["generated_capped_negative"])
                or not max(p, n) <= info["accepted_capped_pairs"] <= min(count, p + n)):
            raise RuntimeError(f"Persona bank capped-response diagnostics disagree for {trait}")
    tensor = manifest.get("tensor", {})
    _safe_artifact_filename(tensor.get("filename"))
    if (tensor.get("key") != "vectors" or tensor.get("dtype") != "float32"
            or tensor.get("shape") != [32, 6, PERSONA_HIDDEN_SIZE]
            or tensor.get("all_finite") is not True):
        raise RuntimeError("Persona bank tensor must be finite raw float32 [32,6,4096]")
    if not re.fullmatch(r"[0-9a-f]{64}", str(tensor.get("sha256", ""))) or type(tensor.get("bytes")) is not int or tensor["bytes"] < 1:
        raise RuntimeError("Persona bank tensor integrity hashes are missing")
    return tensor


def load_persona_bank(root: Path, release: dict):
    """Load safe tensors only after checking publication and extraction provenance."""
    import torch
    from safetensors.torch import load_file

    manifest_path = root / release["manifest_filename"]
    manifest = json.loads(manifest_path.read_text())
    tensor = validate_persona_manifest(manifest)
    path = root / tensor["filename"]
    if not path.is_file() or path.stat().st_size != tensor["bytes"] or sha256(path) != tensor["sha256"]:
        raise RuntimeError("Published persona tensor does not match its integrity manifest")
    payload = load_file(str(path), device="cpu")
    if set(payload) != {tensor["key"]}:
        raise RuntimeError("Persona bank safetensors keys differ from the manifest")
    vectors = payload[tensor["key"]]
    if vectors.dtype != torch.float32 or list(vectors.shape) != tensor["shape"] or not bool(torch.isfinite(vectors).all()):
        raise RuntimeError("Published persona vectors are not finite raw FP32 tensors of the expected shape")
    # Verification uses FP64 to avoid underflow/overflow in magnitude reporting;
    # the retained bank stays raw FP32 and runtime differences use FP32 too.
    norms = vectors.double().norm(dim=-1)
    recorded = torch.tensor(tensor.get("l2_norms", []), dtype=torch.float64)
    if tuple(recorded.shape) != tuple(norms.shape) or not torch.allclose(recorded, norms, rtol=1e-5, atol=1e-6):
        raise RuntimeError("Persona vector magnitudes differ from recorded raw L2 norms")
    zero_indices = (vectors == 0).all(dim=-1).nonzero().tolist()
    if (zero_indices or "zero_vector_indices" in tensor) and tensor.get("zero_vector_indices") != zero_indices:
        raise RuntimeError("Persona bank zero-direction flags disagree with the raw tensor")
    metadata = {
        "mood_vectors_source": "persona_vectors", "mood_vectors_validated": False,
        "mood_vectors_published_inference_method": manifest["inference"]["method"],
        "mood_vectors_repo_id": release["repo_id"], "mood_vectors_revision": release["revision"],
        "mood_vectors_manifest": release["manifest_filename"],
        "mood_vectors_manifest_sha256": sha256(manifest_path),
        "mood_vectors_tensor_sha256": tensor["sha256"],
        "mood_vectors_run_id": manifest["run_id"],
        "mood_vectors_integrity_verified": True,
        "mood_vectors_shape": tensor["shape"],
        "mood_vectors_zero_vector_indices": zero_indices,
    }
    from deployment.steering import zero_direction_metadata

    metadata.update(zero_direction_metadata(vectors))
    return vectors, metadata


def get_persona_bank(commit=lambda: None):
    """Fetch the required manifest and bank from one immutable HF publication."""
    from huggingface_hub import snapshot_download

    release = persona_release()
    cache = Path("/artifacts/persona_banks") / release["revision"]
    if (cache / release["manifest_filename"]).is_file():
        try:
            return load_persona_bank(cache, release)
        except (OSError, ValueError, TypeError, KeyError, RuntimeError):
            pass
    # Read the pinned manifest first, then download only its declared safe tensor.
    snapshot_download(
        release["repo_id"], repo_type=release["repo_type"], revision=release["revision"],
        local_dir=cache, allow_patterns=[release["manifest_filename"]], max_workers=1,
    )
    manifest = json.loads((cache / release["manifest_filename"]).read_text())
    tensor = validate_persona_manifest(manifest)
    snapshot_download(
        release["repo_id"], repo_type=release["repo_type"], revision=release["revision"],
        local_dir=cache, allow_patterns=[tensor["filename"]], max_workers=1,
    )
    result = load_persona_bank(cache, release)
    commit()
    return result
