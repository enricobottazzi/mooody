import copy
import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest
import subprocess
import sys
from unittest.mock import patch

from deployment.checkpoint import (
    MANIFEST, PERSONA_AXES, PERSONA_BOUNDARY, PERSONA_FILTERING_POLICY, PERSONA_HIDDEN_SIZE,
    PERSONA_POLICY_BYTES, PERSONA_POLICY_SHA256, PERSONA_POLICY_SOURCE,
    PERSONA_RECOVERY_STAGE, PERSONA_RECOVERY_POLICY_SOURCE, PERSONA_RECOVERY_POLICY_SHA256,
    PERSONA_RECOVERY_POLICY_BYTES, PERSONA_RECOVERY_POLICY_FINGERPRINT,
    configured_persona_metadata,
    load_persona_bank, persona_release, sha256, validate_persona_manifest,
)
from deployment.core import MODEL_ID, MODEL_REVISION, MOOD_COEFFICIENTS
from deployment.worker import ModelRuntime

try:
    import torch
    from safetensors.torch import save_file
except ImportError:
    torch = None


def manifest_fixture():
    audited = json.loads(MANIFEST.read_text())
    source_names = [
        "data/persona_traits/protocol.json", "data/persona_traits/manifest.json",
        "data/persona_traits/coherence_evaluation_prompt.txt", PERSONA_POLICY_SOURCE, *[
        f"data/persona_traits/source/{trait}.json" for trait in PERSONA_AXES
    ]]
    coverage = {
        "generated_positive": 200, "generated_negative": 200,
        "accepted_pairs": 40, "accepted_positive": 40, "accepted_negative": 40,
        "accepted_by_question": {f"{i:02}": 1 for i in range(1, 41)},
        "accepted_by_system_prompt_pair": {f"{i:02}": 8 for i in range(1, 6)},
        "coverage_passed": True, "extraction_usable": True,
        "missing_questions": [], "missing_system_prompt_pairs": [],
        "policy_sha256": PERSONA_POLICY_SHA256,
        "generated_capped_positive": 1, "generated_capped_negative": 2,
        "accepted_capped_positive": 1, "accepted_capped_negative": 1, "accepted_capped_pairs": 1,
    }
    manifest = {
        "schema_version": 1, "artifact_type": "mooody_persona_vector_bank",
        "status": "extracted_not_behaviorally_validated", "run_id": "unit-fixture-only",
        "trait_order": list(PERSONA_AXES),
        "checkpoint": {
            "model_id": MODEL_ID, "revision": MODEL_REVISION,
            "decoder_layers": 32, "hidden_size": PERSONA_HIDDEN_SIZE,
            "config_sha256": audited["files"]["config.json"]["sha256"],
        },
        "tokenizer": {
            "model_id": MODEL_ID, "revision": MODEL_REVISION,
            "file_hashes": {name: copy.deepcopy(audited["files"][name]) for name in
                            ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja")},
        },
        "sources": {name: {"sha256": "1" * 64, "bytes": 1} for name in source_names},
        "judging": {
            "model": "google/gemini-3.8-flash", "gateway": "openrouter",
            "planned_scoring_calls_before_retries": 4800, "provenance": "unit-fixture",
        },
        "filtering": {
            "policy": copy.deepcopy(PERSONA_FILTERING_POLICY),
            "policy_source": PERSONA_POLICY_SOURCE, "policy_sha256": PERSONA_POLICY_SHA256,
            "traits": {trait: copy.deepcopy(coverage) for trait in PERSONA_AXES},
        },
        "extraction": {
            "activation_boundary": PERSONA_BOUNDARY,
            "pooling": "response_content_mean_then_equal_response_group_mean",
            "raw_normalization": "none", "retain_layers": "all_decoder_layers",
            "select_best_layer": False, "position_axis": False,
        },
        "inference": {
            "method": "direct_raw_all_layers", "activation_boundary": PERSONA_BOUNDARY,
            "coefficients": [-2, -1, 0, 1, 2],
            "token_scope": "final_formatted_prompt_then_generated_content",
        },
        "tensor": {
            "filename": "persona_vectors.safetensors", "key": "vectors", "dtype": "float32",
            "shape": [32, 6, PERSONA_HIDDEN_SIZE], "all_finite": True, "sha256": "2" * 64, "bytes": 1,
        },
    }
    manifest["sources"][PERSONA_POLICY_SOURCE] = {"sha256": PERSONA_POLICY_SHA256, "bytes": PERSONA_POLICY_BYTES}
    return manifest


def add_transport_recovery(manifest):
    def fingerprint(value):
        return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                         allow_nan=False).encode()).hexdigest()
    policy = json.loads((Path(__file__).resolve().parents[2] / PERSONA_RECOVERY_POLICY_SOURCE).read_text())
    stage = {
        "stage": PERSONA_RECOVERY_STAGE, "module_sha256": "a" * 64,
        "original_judging_implementation_sha256": "b" * 64,
        "policy_source": PERSONA_RECOVERY_POLICY_SOURCE,
        "policy_sha256": PERSONA_RECOVERY_POLICY_SHA256,
        "policy_bytes": PERSONA_RECOVERY_POLICY_BYTES, "policy": policy,
    }
    stage["implementation_sha256"] = fingerprint(stage)
    manifest["implementations"] = {PERSONA_RECOVERY_STAGE: stage, "judging": {"implementation_sha256": "b" * 64}}
    manifest["sources"][PERSONA_RECOVERY_POLICY_SOURCE] = {
        "sha256": PERSONA_RECOVERY_POLICY_SHA256, "bytes": PERSONA_RECOVERY_POLICY_BYTES,
    }
    def snapshot(after):
        rows = {}
        for trait in PERSONA_AXES:
            eligible = int(trait in ("paranoia", "euphoria"))
            rows[trait] = {
                "present": 800, "valid_scores": 800 if after else 800 - eligible,
                "missing": 0, "in_progress_or_interrupted": 0,
                "eligible_http429_none": 0 if after else eligible, "other_none": 0,
                "original_actual_attempts": 800 + 2 * eligible,
                "recovery_actual_attempts": eligible if after else 0,
                "actual_attempts": 800 + (3 if after else 2) * eligible,
            }
        return {"planned_score_slots": 4800, "by_trait": rows,
                **{key: sum(row[key] for row in rows.values()) for key in next(iter(rows.values()))}}
    audit = {
        "stage": PERSONA_RECOVERY_STAGE, "policy_sha256": PERSONA_RECOVERY_POLICY_SHA256,
        "implementation_sha256": stage["implementation_sha256"], "settings_sha256": "c" * 64,
        "before": snapshot(False), "after": snapshot(True), "status": "complete",
    }
    recovery = {
        "stage": PERSONA_RECOVERY_STAGE, "policy_source": PERSONA_RECOVERY_POLICY_SOURCE,
        "policy_sha256": PERSONA_RECOVERY_POLICY_SHA256, "policy": policy,
        "implementation": stage, "audit": audit,
        "audit_sha256": hashlib.sha256((json.dumps(audit, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode()).hexdigest(),
        "counts": {"records_touched": 2, "additional_attempts": 2, "recovered_scores": 2,
                   "remaining_none": 0, "original_http429_attempts": 6, "http429_recovery_attempts": 0},
        "original_journal_hashes_sha256": "d" * 64, "baseline_journal_hashes_sha256": "e" * 64,
        "valid_and_ineligible_original_journals_preserved": True,
    }
    manifest["judging"].update(transport_recovery=recovery, score_records=4800, actual_attempts=4806, valid_scores=4800)
    return recovery


class PersonaManifestTests(unittest.TestCase):
    def test_transport_recovery_policy_pin_matches_the_separate_approved_source(self):
        path = Path(__file__).resolve().parents[2] / PERSONA_RECOVERY_POLICY_SOURCE
        self.assertEqual(sha256(path), PERSONA_RECOVERY_POLICY_SHA256)
        self.assertEqual(path.stat().st_size, PERSONA_RECOVERY_POLICY_BYTES)
        policy = json.loads(path.read_text())
        canonical = json.dumps(policy, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()
        self.assertEqual(hashlib.sha256(canonical).hexdigest(), PERSONA_RECOVERY_POLICY_FINGERPRINT)

    def test_conditional_transport_recovery_requires_its_policy_receipt_and_audit(self):
        manifest = manifest_fixture()
        add_transport_recovery(manifest)
        validate_persona_manifest(manifest)
        mutations = [
            lambda m: m["judging"].pop("transport_recovery"),
            lambda m: m["sources"].pop(PERSONA_RECOVERY_POLICY_SOURCE),
            lambda m: m["judging"]["transport_recovery"]["policy"].update(generation_rerolls=1),
            lambda m: m["judging"]["transport_recovery"].update(policy_sha256="0" * 64),
            lambda m: m["judging"]["transport_recovery"]["implementation"].update(module_sha256="0" * 64),
            lambda m: m["implementations"]["judging"].update(implementation_sha256="0" * 64),
            lambda m: m["judging"]["transport_recovery"].update(audit_sha256="0" * 64),
            lambda m: m["judging"]["transport_recovery"]["audit"].update(status="in_progress"),
            lambda m: m["judging"]["transport_recovery"].update(valid_and_ineligible_original_journals_preserved=False),
            lambda m: m["judging"]["transport_recovery"]["counts"].update(additional_attempts=7),
            lambda m: m["judging"].update(score_records=4801),
            lambda m: m["judging"].update(actual_attempts=4807),
            lambda m: m["judging"].update(valid_scores=4799),
        ]
        for mutate in mutations:
            changed = copy.deepcopy(manifest)
            mutate(changed)
            with self.subTest(mutation=mutate), self.assertRaises(RuntimeError):
                validate_persona_manifest(changed)

    def test_cpu_configuration_imports_no_model_or_network_dependencies_and_exposes_no_secrets(self):
        script = """
import json, os, sys
os.environ['HF_TOKEN'] = 'PRIVATE_HF_SENTINEL'
os.environ['OPENROUTER_API_KEY'] = 'PRIVATE_JUDGE_SENTINEL'
os.environ['MOOODY_PROXY_TOKEN'] = 'PRIVATE_PROXY_SENTINEL'
from deployment.core import configuration
result = configuration()
assert not {'torch', 'transformers', 'huggingface_hub', 'safetensors'} & set(sys.modules)
assert 'PRIVATE_' not in json.dumps(result)
"""
        subprocess.run([sys.executable, "-c", script], check=True, capture_output=True,
                       cwd=Path(__file__).resolve().parents[2])

    def test_pinned_analytical_policy_matches_approved_source_file_exactly(self):
        path = Path(__file__).resolve().parents[2] / PERSONA_POLICY_SOURCE
        self.assertEqual(json.loads(path.read_text()), PERSONA_FILTERING_POLICY)
        self.assertEqual(sha256(path), PERSONA_POLICY_SHA256)
        self.assertEqual(path.stat().st_size, PERSONA_POLICY_BYTES)

    def test_public_configuration_fails_closed_without_immutable_publication(self):
        with tempfile.TemporaryDirectory() as temporary, patch.dict(os.environ, {}, clear=True), patch(
            "deployment.checkpoint.PERSONA_RELEASE", Path(temporary) / "not-published.json"
        ):
            metadata = configured_persona_metadata()
            self.assertFalse(metadata["mood_vectors_available"])
            self.assertFalse(metadata["steering_available"])
            self.assertFalse(metadata["mood_vectors_validated"])
            self.assertEqual(metadata["mood_coefficients"], list(MOOD_COEFFICIENTS))
            self.assertEqual(metadata["mood_vectors_source"], "unavailable")
            with self.assertRaises(RuntimeError):
                persona_release()

    def test_mutable_branch_or_missing_revision_is_never_accepted(self):
        for revision in ("main", "latest", "", "a" * 39):
            with self.subTest(revision=revision), tempfile.TemporaryDirectory() as temporary, patch(
                "deployment.checkpoint.PERSONA_RELEASE", Path(temporary) / "missing.json"
            ), patch.dict(os.environ, {
                "MOOODY_PERSONA_REPO_ID": "owner/bank", "MOOODY_PERSONA_REVISION": revision
            }, clear=True):
                with self.assertRaises(RuntimeError):
                    persona_release()
                self.assertFalse(configured_persona_metadata()["steering_available"])

    def test_public_config_reports_pin_without_claiming_behavioral_validation(self):
        with tempfile.TemporaryDirectory() as temporary, patch(
            "deployment.checkpoint.PERSONA_RELEASE", Path(temporary) / "missing.json"
        ), patch.dict(os.environ, {
            "MOOODY_PERSONA_REPO_ID": "owner/bank", "MOOODY_PERSONA_REVISION": "a" * 40
        }, clear=True):
            metadata = configured_persona_metadata()
            self.assertEqual(metadata["mood_vectors_source"], "persona_vectors")
            self.assertEqual(metadata["mood_vectors_revision"], "a" * 40)
            self.assertEqual(metadata["mood_vectors_published_inference_method"], "direct_raw_all_layers")
            self.assertEqual(metadata["steering_method"], "paper_incremental_all_layers")
            self.assertEqual(metadata["steering_incremental_definition"], "raw_layer_vector_minus_previous_layer_vector")
            self.assertEqual(metadata["steering_first_layer_previous_vector"], "zero")
            self.assertEqual(metadata["mood_coefficients"], [-0.25, -0.125, 0, 0.125, 0.25])
            self.assertTrue(metadata["steering_available"])
            self.assertFalse(metadata["mood_vectors_validated"])

    def test_incompatible_or_incomplete_extraction_provenance_is_rejected(self):
        valid = manifest_fixture()
        self.assertEqual(validate_persona_manifest(valid)["shape"], [32, 6, PERSONA_HIDDEN_SIZE])
        mutations = [
            lambda m: m.update(trait_order=list(reversed(PERSONA_AXES))),
            lambda m: m["checkpoint"].update(revision="main"),
            lambda m: m["checkpoint"].update(decoder_layers=31),
            lambda m: m["checkpoint"].update(hidden_size=4095),
            lambda m: m["checkpoint"].update(config_sha256="2" * 64),
            lambda m: m["tokenizer"].update(revision="a" * 40),
            lambda m: m["tokenizer"].pop("file_hashes"),
            lambda m: m["tokenizer"]["file_hashes"]["chat_template.jinja"].update(sha256="2" * 64),
            lambda m: m["extraction"].update(raw_normalization="unit_norm"),
            lambda m: m["extraction"].update(activation_boundary="block_input"),
            lambda m: m["inference"].update(activation_boundary="block_input"),
            lambda m: m["inference"].update(method="paper_incremental_all_layers"),
            lambda m: m["inference"].update(token_scope="all_prompt_tokens"),
            lambda m: m["inference"].update(coefficients=[0, 1, 2]),
            lambda m: m["inference"].update(coefficients=list(MOOD_COEFFICIENTS)),
            lambda m: m["inference"].update(gain=1),
            lambda m: m["tensor"].update(dtype="bfloat16"),
            lambda m: m["tensor"].update(shape=[31, 6, 4]),
            lambda m: m["tensor"].update(shape=[32, 6, 4095]),
            lambda m: m["tensor"].update(all_finite=False),
            lambda m: m["tensor"].update(filename="../vectors.safetensors"),
            lambda m: m["sources"].pop("data/persona_traits/protocol.json"),
            lambda m: m["sources"].pop("data/persona_traits/coherence_evaluation_prompt.txt"),
            lambda m: m["sources"].pop(PERSONA_POLICY_SOURCE),
            lambda m: m["sources"][PERSONA_POLICY_SOURCE].update(sha256="2" * 64),
            lambda m: m["sources"][PERSONA_POLICY_SOURCE].update(bytes=PERSONA_POLICY_BYTES + 1),
            lambda m: m["filtering"].update(policy_source="other-policy.json"),
            lambda m: m["filtering"].update(policy_sha256="2" * 64),
            lambda m: m["filtering"]["policy"].update(minimum_coherence_score_both_sides=40),
            lambda m: m["filtering"]["policy"].update(positive_trait_threshold=">=50"),
            lambda m: m["filtering"]["policy"].update(negative_trait_threshold="<=50"),
            lambda m: m["filtering"]["policy"].update(generation_protocol_unchanged=1),
            lambda m: m["filtering"]["policy"].update(require_all_generated_questions=False),
            lambda m: m["filtering"]["policy"].update(require_all_accepted_questions=True),
            lambda m: m["filtering"]["traits"]["depression"].update(policy_sha256="2" * 64),
            lambda m: m["filtering"]["traits"]["depression"].update(generated_positive=199),
            lambda m: m["filtering"]["traits"]["depression"].update(accepted_negative=39),
            lambda m: m["filtering"]["traits"]["depression"].update(coverage_passed=False),
            lambda m: m["filtering"]["traits"]["depression"].update(extraction_usable=False),
            lambda m: m["filtering"]["traits"]["depression"].update(extraction_usable=1),
            lambda m: m["filtering"]["traits"]["depression"].update(missing_questions=["01"]),
            lambda m: m["filtering"]["traits"]["depression"].pop("missing_questions"),
            lambda m: m["filtering"]["traits"]["depression"].pop("accepted_capped_pairs"),
            lambda m: m["filtering"]["traits"]["depression"].update(accepted_capped_pairs=0),
            lambda m: m["filtering"]["traits"]["depression"].update(accepted_capped_positive=2),
            lambda m: m["filtering"]["traits"]["depression"].update(generated_capped_negative=201),
            lambda m: m["filtering"]["traits"]["depression"]["accepted_by_question"].update({"01": 0}),
            lambda m: m["filtering"]["traits"]["depression"]["accepted_by_question"].update({"01": 2}),
            lambda m: m.update(judging={}),
            lambda m: m["judging"].update(model="unrecorded-judge"),
        ]
        for mutate in mutations:
            manifest = copy.deepcopy(valid)
            mutate(manifest)
            with self.subTest(manifest=manifest), self.assertRaises(RuntimeError):
                validate_persona_manifest(manifest)

    def test_nonempty_sparse_accepted_coverage_is_valid_without_false_coverage_claims(self):
        manifest = manifest_fixture()
        info = manifest["filtering"]["traits"]["depression"]
        info.update(
            accepted_pairs=1, accepted_positive=1, accepted_negative=1,
            accepted_by_question={f"{i:02}": int(i == 1) for i in range(1, 41)},
            accepted_by_system_prompt_pair={f"{i:02}": int(i == 1) for i in range(1, 6)},
            missing_questions=[f"{i:02}" for i in range(2, 41)],
            missing_system_prompt_pairs=[f"{i:02}" for i in range(2, 6)],
            coverage_passed=False,
        )
        validate_persona_manifest(manifest)
        for field, value in (("coverage_passed", True), ("extraction_usable", False),
                             ("accepted_pairs", 0), ("missing_questions", [])):
            changed = copy.deepcopy(manifest)
            changed["filtering"]["traits"]["depression"][field] = value
            with self.subTest(field=field), self.assertRaises(RuntimeError):
                validate_persona_manifest(changed)

    def test_complete_generated_and_accepted_counts_have_consistent_bucket_limits(self):
        manifest = manifest_fixture()
        info = manifest["filtering"]["traits"]["depression"]
        info.update(accepted_pairs=200, accepted_positive=200, accepted_negative=200,
                    accepted_by_question={f"{i:02}": 5 for i in range(1, 41)},
                    accepted_by_system_prompt_pair={f"{i:02}": 40 for i in range(1, 6)})
        validate_persona_manifest(manifest)
        info["accepted_by_question"].update({"01": 6, "02": 4})
        with self.assertRaises(RuntimeError):
            validate_persona_manifest(manifest)

    def test_worker_rejects_missing_or_unverified_bank_before_loading_model(self):
        for vectors, metadata in ((None, {}), (object(), {}), (object(), {"mood_vectors_source": "random_placeholder"})):
            with self.subTest(metadata=metadata), self.assertRaisesRegex(ValueError, "integrity-verified"):
                ModelRuntime("unused-checkpoint", vectors, metadata)


@unittest.skipIf(torch is None, "Tensor loader verification requires torch and safetensors")
class PersonaTensorLoaderTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.bank = torch.arange(1, 32 * 6 * PERSONA_HIDDEN_SIZE + 1, dtype=torch.float32).reshape(32, 6, PERSONA_HIDDEN_SIZE) / 11
        self.path = self.root / "persona_vectors.safetensors"
        save_file({"vectors": self.bank}, str(self.path))
        self.manifest = manifest_fixture()
        self.manifest["tensor"].update(
            sha256=sha256(self.path), bytes=self.path.stat().st_size,
            l2_norms=self.bank.norm(dim=-1).tolist(),
        )
        self.release = {"repo_id": "owner/test-only-bank", "revision": "a" * 40, "manifest_filename": "persona_manifest.json"}

    def tearDown(self):
        self.temporary.cleanup()

    def load(self):
        (self.root / "persona_manifest.json").write_text(json.dumps(self.manifest))
        return load_persona_bank(self.root, self.release)

    def test_exact_raw_tensor_and_verified_pin_are_loaded_without_normalization(self):
        vectors, metadata = self.load()
        self.assertTrue(torch.equal(vectors, self.bank))
        self.assertEqual(vectors.dtype, torch.float32)
        self.assertEqual(metadata["mood_vectors_source"], "persona_vectors")
        self.assertEqual(metadata["mood_vectors_revision"], "a" * 40)
        self.assertEqual(metadata["mood_vectors_published_inference_method"], "direct_raw_all_layers")
        self.assertTrue(metadata["mood_vectors_integrity_verified"])
        self.assertFalse(metadata["mood_vectors_validated"])

    def test_separately_audited_transport_recovery_keeps_exact_raw_tensor_and_public_metadata(self):
        add_transport_recovery(self.manifest)
        vectors, metadata = self.load()
        self.assertTrue(torch.equal(vectors, self.bank))
        self.assertTrue(metadata["mood_vectors_integrity_verified"])
        self.assertFalse(metadata["mood_vectors_validated"])
        self.assertNotIn("transport_recovery", metadata)

    def test_changed_payload_or_recorded_raw_magnitudes_fail_closed(self):
        self.manifest["tensor"]["sha256"] = "0" * 64
        with self.assertRaisesRegex(RuntimeError, "integrity manifest"):
            self.load()
        self.manifest["tensor"]["sha256"] = sha256(self.path)
        self.manifest["tensor"]["l2_norms"][0][0] += 1
        with self.assertRaisesRegex(RuntimeError, "raw L2 norms"):
            self.load()

    def test_zero_vectors_are_preserved_at_all_layers_and_reported_truthfully(self):
        self.bank[0, 0] = 0
        self.bank[:, 1] = 0
        save_file({"vectors": self.bank}, str(self.path))
        self.manifest["tensor"].update(
            sha256=sha256(self.path), bytes=self.path.stat().st_size,
            l2_norms=self.bank.double().norm(dim=-1).tolist(),
            zero_vector_indices=(self.bank == 0).all(dim=-1).nonzero().tolist(),
        )
        vectors, metadata = self.load()
        self.assertTrue(torch.equal(vectors, self.bank))
        self.assertEqual(tuple(vectors.shape), (32, 6, PERSONA_HIDDEN_SIZE))
        self.assertEqual(metadata["mood_vectors_zero_layer_trait_count"], 33)
        self.assertEqual(metadata["mood_vectors_entirely_zero_traits"], [PERSONA_AXES[1]])
        self.assertFalse(metadata["mood_vectors_entirely_zero_bank"])
        self.assertFalse(metadata["mood_vectors_validated"])
        self.manifest["tensor"]["zero_vector_indices"] = []
        with self.assertRaisesRegex(RuntimeError, "zero-direction flags"):
            self.load()


if __name__ == "__main__":
    unittest.main()
