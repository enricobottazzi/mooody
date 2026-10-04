"""Integrity failures that must block publication of a persona bank."""

from array import array
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import struct
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("persona_bank_artifacts", ROOT / "scripts/persona_bank_artifacts.py")
bank = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(bank)


def fixture(root: Path) -> dict:
    shape = [32, 6, bank.HIDDEN_SIZE]
    values = array("f", [1.0, -2.0] * (32 * 6 * bank.HIDDEN_SIZE // 2))
    header = json.dumps({"vectors": {"dtype": "F32", "shape": shape, "data_offsets": [0, len(values) * 4]}}).encode()
    header += b" " * (-len(header) % 8)
    path = root / bank.BANK_FILENAME
    path.write_bytes(struct.pack("<Q", len(header)) + header + values.tobytes())
    norms, tensor = bank.read_vectors(path, shape)
    count = {"generated_positive": 200, "generated_negative": 200, "accepted_pairs": 40,
             "accepted_positive": 40, "accepted_negative": 40,
             "accepted_by_question": {f"{i:02d}": 1 for i in range(1, 41)},
             "accepted_by_system_prompt_pair": {f"{i:02d}": 8 for i in range(1, 6)},
             "missing_questions": [], "missing_system_prompt_pairs": [],
             "coverage_passed": True, "extraction_usable": True, "policy_sha256": bank.POLICY_SHA256,
             "generated_capped_positive": 0, "generated_capped_negative": 0,
             "accepted_capped_positive": 0, "accepted_capped_negative": 0, "accepted_capped_pairs": 0}
    manifest = {
        "schema_version": 1, "artifact_type": "mooody_persona_vector_bank",
        "status": "extracted_not_behaviorally_validated", "run_id": "unit-test-fixture",
        "trait_order": list(bank.TRAITS),
        "checkpoint": {"model_id": bank.MODEL_ID, "revision": bank.MODEL_REVISION,
                       "decoder_layers": 32, "hidden_size": bank.HIDDEN_SIZE, "config_sha256": bank.CONFIG_SHA256},
        "tokenizer": {"model_id": bank.MODEL_ID, "revision": bank.MODEL_REVISION,
                      "file_hashes": copy.deepcopy(bank.TOKENIZER_HASHES)},
        "sources": {name: {"sha256": bank.POLICY_SHA256, "bytes": bank.POLICY_BYTES} if name == bank.POLICY_SOURCE
                    else {"sha256": "c" * 64, "bytes": 1} for name in bank.SOURCE_FILES},
        "generation": {"total_responses_before_filtering": 2400},
        "judging": {"gateway": "openrouter", "model": "google/gemini-3.8-flash",
                    "planned_scoring_calls_before_retries": 4800, "actual_attempts": 4800, "valid_scores": 4800,
                    "provider": {"order": ["google-ai-studio"], "only": ["google-ai-studio"],
                                 "allow_fallbacks": False, "require_parameters": True},
                    "reasoning": {"effort": "low", "exclude": True}, "provenance": {"log_sha256": "d" * 64}},
        "filtering": {"policy": copy.deepcopy(bank.FILTERING_POLICY), "policy_source": bank.POLICY_SOURCE,
                      "policy_sha256": bank.POLICY_SHA256,
                      "traits": {trait: copy.deepcopy(count) for trait in bank.TRAITS}},
        "extraction": {"activation_boundary": bank.BOUNDARY, "pooling": bank.POOLING,
                       "raw_normalization": "none", "retain_layers": "all_decoder_layers",
                       "select_best_layer": False, "position_axis": False},
        "inference": {"method": "direct_raw_all_layers", "activation_boundary": bank.BOUNDARY,
                      "coefficients": [-2, -1, 0, 1, 2], "token_scope": "final_formatted_prompt_then_generated_content"},
        "tensor": {**tensor, "filename": bank.BANK_FILENAME, "key": "vectors", "l2_norms": norms},
    }
    (root / bank.MANIFEST_FILENAME).write_bytes(bank.json_bytes(manifest))
    return manifest


class PersonaBankIntegrityTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.run = self.root / "run"
        self.run.mkdir()
        self.manifest = fixture(self.run)

    def tearDown(self):
        self.temp.cleanup()

    def save(self):
        (self.run / bank.MANIFEST_FILENAME).write_bytes(bank.json_bytes(self.manifest))

    def test_valid_bank_and_public_allowlist(self):
        (self.run / "private_responses.jsonl").write_text('"Never publish this transcript"\n')
        license_path = self.root / "LICENSE"
        license_path.write_text("Apache License\nVersion 2.0\n")
        release = self.root / "release"
        result = bank.stage_release(self.run, release, license_path=license_path)
        self.assertEqual(result["tensor"]["shape"], [32, 6, bank.HIDDEN_SIZE])
        self.assertEqual({p.name for p in release.iterdir()}, bank.PUBLIC_FILES)
        self.assertNotIn("Never publish", (release / "README.md").read_text())

    def test_corruption_and_trailing_bytes_block(self):
        path = self.run / bank.BANK_FILENAME
        path.write_bytes(path.read_bytes() + b"extra")
        with self.assertRaisesRegex(ValueError, "truncated or has trailing"):
            bank.validate_bank(self.run)

    def test_nonfinite_vectors_block(self):
        path = self.run / bank.BANK_FILENAME
        for value in (float("nan"), float("inf")):
            with self.subTest(value=value):
                data = bytearray(path.read_bytes())
                header = struct.unpack("<Q", data[:8])[0]
                data[8 + header:8 + header + 4 * bank.HIDDEN_SIZE] = struct.pack("<f", value) * bank.HIDDEN_SIZE
                path.write_bytes(data)
                with self.assertRaisesRegex(ValueError, "nonfinite"):
                    bank.validate_bank(self.run)

    def test_finite_zero_vectors_are_retained_and_flagged(self):
        path = self.run / bank.BANK_FILENAME
        data = bytearray(path.read_bytes())
        header = struct.unpack("<Q", data[:8])[0]
        data[8 + header:8 + header + 4 * bank.HIDDEN_SIZE] = bytes(4 * bank.HIDDEN_SIZE)
        path.write_bytes(data)
        norms, actual = bank.read_vectors(path, self.manifest["tensor"]["shape"])
        self.manifest["tensor"].update(actual, l2_norms=norms)
        self.save()
        result = bank.validate_bank(self.run)
        self.assertEqual(result["tensor"]["zero_vector_indices"], [[0, 0]])
        self.assertEqual(result["tensor"]["nonzero_vectors"], 191)
        self.assertEqual(result["tensor"]["shape"], [32, 6, bank.HIDDEN_SIZE])
        self.assertIn("1 finite zero vectors", bank.model_card(self.manifest))
        self.manifest["tensor"]["zero_vector_indices"] = []
        self.save()
        with self.assertRaisesRegex(ValueError, "zero-vector flags"):
            bank.validate_bank(self.run)

    def test_entire_zero_bank_is_retained_for_release_assessment(self):
        path = self.run / bank.BANK_FILENAME
        data = bytearray(path.read_bytes())
        header = struct.unpack("<Q", data[:8])[0]
        data[8 + header:] = bytes(len(data) - 8 - header)
        path.write_bytes(data)
        norms, actual = bank.read_vectors(path, self.manifest["tensor"]["shape"])
        self.manifest["tensor"].update(actual, l2_norms=norms)
        self.save()
        result = bank.validate_bank(self.run)
        self.assertTrue(result["tensor"]["entire_zero_bank"])
        self.assertEqual(result["tensor"]["zero_vector_count"], 192)
        self.assertEqual(result["tensor"]["entire_zero_trait_indices"], list(range(6)))

    def test_trait_order_and_norm_integrity(self):
        self.manifest["trait_order"].reverse()
        self.save()
        with self.assertRaisesRegex(ValueError, "trait order"):
            bank.validate_bank(self.run)
        self.manifest["trait_order"].reverse()
        self.manifest["tensor"]["l2_norms"][0][0] *= 2
        self.save()
        with self.assertRaisesRegex(ValueError, "norm mismatch"):
            bank.validate_bank(self.run)

    def test_missing_coverage_and_unmatched_counts_block(self):
        self.manifest["filtering"]["traits"]["depression"]["accepted_by_question"]["01"] = 0
        self.save()
        with self.assertRaisesRegex(ValueError, "coverage"):
            bank.validate_bank(self.run)
        self.manifest["filtering"]["traits"]["depression"]["accepted_by_question"]["01"] = 1
        self.manifest["filtering"]["traits"]["paranoia"]["accepted_negative"] = 39
        self.save()
        with self.assertRaisesRegex(ValueError, "matched response counts"):
            bank.validate_bank(self.run)

    def test_nonempty_sparse_coverage_is_retained_and_reported(self):
        record = self.manifest["filtering"]["traits"]["curiosity"]
        record.update(accepted_pairs=1, accepted_positive=1, accepted_negative=1,
                      accepted_by_question={f"{i:02d}": int(i == 1) for i in range(1, 41)},
                      accepted_by_system_prompt_pair={f"{i:02d}": int(i == 1) for i in range(1, 6)},
                      missing_questions=[f"{i:02d}" for i in range(2, 41)],
                      missing_system_prompt_pairs=[f"{i:02d}" for i in range(2, 6)], coverage_passed=False,
                      generated_capped_positive=7, accepted_capped_positive=1, accepted_capped_pairs=1)
        self.save()
        result = bank.validate_bank(self.run)
        self.assertEqual(result["accepted_pairs_by_trait"]["curiosity"], 1)
        self.assertIn("| curiosity | 1 | 1 | 1/40 | 1/5 |", bank.model_card(self.manifest))
        record["coverage_passed"] = True
        self.save()
        with self.assertRaisesRegex(ValueError, "diagnostics"):
            bank.validate_bank(self.run)

    def test_explicit_filtering_policy_and_hash_are_required(self):
        self.assertEqual(bank.load_json(ROOT / bank.POLICY_SOURCE), bank.FILTERING_POLICY)
        self.assertEqual(bank.digest(ROOT / bank.POLICY_SOURCE), bank.POLICY_SHA256)
        for mutation in ("embedded", "hash", "source"):
            with self.subTest(mutation=mutation):
                changed = copy.deepcopy(self.manifest)
                if mutation == "embedded":
                    changed["filtering"]["policy"]["allow_judged_capped_responses"] = False
                elif mutation == "hash":
                    changed["filtering"]["policy_sha256"] = "f" * 64
                else:
                    changed["sources"][bank.POLICY_SOURCE]["bytes"] -= 1
                (self.run / bank.MANIFEST_FILENAME).write_bytes(bank.json_bytes(changed))
                with self.assertRaisesRegex(ValueError, "filtering policy"):
                    bank.validate_bank(self.run)

    def test_capped_response_and_missing_coverage_counts_cannot_be_fabricated(self):
        record = self.manifest["filtering"]["traits"]["curiosity"]
        record["accepted_capped_positive"] = 1
        self.save()
        with self.assertRaisesRegex(ValueError, "capped-response counts"):
            bank.validate_bank(self.run)
        record["generated_capped_positive"] = 1
        self.save()
        with self.assertRaisesRegex(ValueError, "capped matched-pair count"):
            bank.validate_bank(self.run)
        record["accepted_capped_pairs"] = 1
        record["missing_questions"] = ["01"]
        self.save()
        with self.assertRaisesRegex(ValueError, "diagnostics"):
            bank.validate_bank(self.run)

    def test_empty_matched_group_still_blocks(self):
        self.manifest["filtering"]["traits"]["curiosity"].update(accepted_pairs=0, accepted_positive=0, accepted_negative=0)
        self.save()
        with self.assertRaisesRegex(ValueError, "matched response counts"):
            bank.validate_bank(self.run)

    def test_mixed_generation_provenance_and_card_counts_are_bound(self):
        fingerprint = lambda value: hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                                              allow_nan=False).encode()).hexdigest()
        resolved = {"do_sample": True, "temperature": 1.0, "top_p": 1.0, "top_k": 50,
                    "repetition_penalty": 1.0, "max_new_tokens": 1000, "use_cache": True}
        sampling = {"model_id": bank.MODEL_ID, "model_revision": bank.MODEL_REVISION,
                    "checkpoint_generation_config_sha256": bank.GENERATION_CONFIG_SHA256,
                    "batch_size": 8, "enable_thinking": False,
                    "arithmetic_note": "BF16 batching can change sampled text despite identical seeds.",
                    "resolved_config": resolved, "resolved_config_sha256": fingerprint(resolved)}
        stage = {"stage": bank.BATCHED_STAGE, "module_sha256": "a" * 64,
                 "sampling_module_sha256": "b" * 64,
                 "original_generation_implementation_sha256": "c" * 64,
                 "sampling_receipt_sha256": fingerprint(sampling), "batch_size": 8}
        stage["implementation_sha256"] = fingerprint(stage)
        self.manifest["implementations"] = {bank.BATCHED_STAGE: stage}
        generation = self.manifest["generation"]
        generation.update(native_sampling_provenance=sampling,
                          execution_counts_by_trait={trait: {"serial": 8, bank.BATCHED_STAGE: 392}
                                                     for trait in bank.TRAITS},
                          batched_implementation_sha256=stage["implementation_sha256"])
        self.save()
        bank.validate_bank(self.run)
        self.assertIn("48 serial and 2352 batched conditions", bank.model_card(self.manifest))
        self.assertIn("bitwise", bank.model_card(self.manifest))
        generation["execution_counts_by_trait"]["curiosity"]["serial"] += 1
        self.save()
        with self.assertRaisesRegex(ValueError, "condition counts"):
            bank.validate_bank(self.run)
        generation["execution_counts_by_trait"]["curiosity"]["serial"] -= 1
        sampling["arithmetic_note"] += " Altered receipt."
        self.save()
        with self.assertRaisesRegex(ValueError, "receipt/implementation hash"):
            bank.validate_bank(self.run)

    def test_transport_recovery_binds_policy_audit_and_unchanged_score_budget(self):
        self.assertEqual(bank.load_json(ROOT / bank.RECOVERY_POLICY_SOURCE), bank.RECOVERY_POLICY)
        self.assertEqual(bank.digest(ROOT / bank.RECOVERY_POLICY_SOURCE), bank.RECOVERY_POLICY_SHA256)
        stage = {"stage": bank.RECOVERY_STAGE, "module_sha256": "a" * 64,
                 "original_judging_implementation_sha256": "b" * 64,
                 "policy_source": bank.RECOVERY_POLICY_SOURCE, "policy_sha256": bank.RECOVERY_POLICY_SHA256,
                 "policy_bytes": bank.RECOVERY_POLICY_BYTES, "policy": copy.deepcopy(bank.RECOVERY_POLICY)}
        stage["implementation_sha256"] = bank.metadata_fingerprint(stage)
        self.manifest["implementations"] = {bank.RECOVERY_STAGE: stage, "judging": {"implementation_sha256": "b" * 64}}
        self.manifest["sources"][bank.RECOVERY_POLICY_SOURCE] = {"sha256": bank.RECOVERY_POLICY_SHA256,
                                                               "bytes": bank.RECOVERY_POLICY_BYTES}
        def snapshot(after):
            by_trait = {}
            for trait in bank.TRAITS:
                eligible = int(trait in ("paranoia", "euphoria"))
                by_trait[trait] = {"present": 800, "valid_scores": 800 if after else 800 - eligible,
                                   "missing": 0, "in_progress_or_interrupted": 0,
                                   "eligible_http429_none": 0 if after else eligible, "other_none": 0,
                                   "original_actual_attempts": 800 + 2 * eligible,
                                   "recovery_actual_attempts": eligible if after else 0,
                                   "actual_attempts": 800 + (3 if after else 2) * eligible}
            return {"planned_score_slots": 4800, "by_trait": by_trait,
                    **{key: sum(row[key] for row in by_trait.values()) for key in next(iter(by_trait.values()))}}
        audit = {"stage": bank.RECOVERY_STAGE, "policy_sha256": bank.RECOVERY_POLICY_SHA256,
                 "implementation_sha256": stage["implementation_sha256"], "settings_sha256": "c" * 64,
                 "before": snapshot(False), "after": snapshot(True), "status": "complete"}
        recovery = {"stage": bank.RECOVERY_STAGE, "policy_source": bank.RECOVERY_POLICY_SOURCE,
                    "policy_sha256": bank.RECOVERY_POLICY_SHA256, "policy": copy.deepcopy(bank.RECOVERY_POLICY),
                    "implementation": stage, "audit": audit,
                    "audit_sha256": hashlib.sha256(bank.json_bytes(audit)).hexdigest(),
                    "counts": {"records_touched": 2, "additional_attempts": 2, "recovered_scores": 2,
                               "remaining_none": 0, "original_http429_attempts": 6, "http429_recovery_attempts": 0},
                    "original_journal_hashes_sha256": "d" * 64, "baseline_journal_hashes_sha256": "e" * 64,
                    "valid_and_ineligible_original_journals_preserved": True}
        self.manifest["judging"].update(transport_recovery=recovery, score_records=4800,
                                       actual_attempts=4806, valid_scores=4800)
        self.save()
        bank.validate_bank(self.run)
        self.assertIn("additional judge attempts across 2 previously unscored", bank.model_card(self.manifest))
        recovery["counts"]["additional_attempts"] = 7
        self.save()
        with self.assertRaisesRegex(ValueError, "Recovery scores/attempts"):
            bank.validate_bank(self.run)
        recovery["counts"]["additional_attempts"] = 2
        recovery["audit"]["after"]["valid_scores"] = 4799
        recovery["audit_sha256"] = hashlib.sha256(bank.json_bytes(audit)).hexdigest()
        self.save()
        with self.assertRaisesRegex(ValueError, "aggregate counts"):
            bank.validate_bank(self.run)

    def test_transcript_and_secret_metadata_block(self):
        self.manifest["judging"]["raw_response"] = "private transcript"
        self.save()
        with self.assertRaisesRegex(ValueError, "Private transcript"):
            bank.validate_bank(self.run)
        del self.manifest["judging"]["raw_response"]
        for token in ("hf_" + "x" * 30, "sk-or-v1-" + "a" * 64):
            with self.subTest(prefix=token[:9]):
                self.manifest["judging"]["comment"] = token
                self.save()
                with self.assertRaisesRegex(ValueError, "credential"):
                    bank.validate_bank(self.run)

    def test_judge_counts_must_support_accepted_contrasts(self):
        self.manifest["judging"]["valid_scores"] = 0
        self.save()
        with self.assertRaisesRegex(ValueError, "counts cannot support"):
            bank.validate_bank(self.run)

    def test_tokenizer_hash_must_match_audited_checkpoint(self):
        self.manifest["tokenizer"]["file_hashes"]["chat_template.jinja"]["sha256"] = "f" * 64
        self.save()
        with self.assertRaisesRegex(ValueError, "Tokenizer hashes differ"):
            bank.validate_bank(self.run)

    def test_source_drift_blocks_staging(self):
        with self.assertRaisesRegex(ValueError, "no longer matches extraction"):
            bank.validate_bank(self.run, source_root=self.root)

    def test_duplicate_json_keys_block(self):
        path = self.run / bank.MANIFEST_FILENAME
        path.write_text('{"schema_version":1,"schema_version":2}')
        with self.assertRaisesRegex(ValueError, "Duplicate JSON"):
            bank.validate_bank(self.run)


if __name__ == "__main__":
    unittest.main()
