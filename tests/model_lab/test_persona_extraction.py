"""CPU collection/filtering regressions; native activation test runs with Torch."""

import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock
from concurrent.futures import ThreadPoolExecutor
import re
import shutil

from model_lab.persona_extraction import (
    TRAITS, atomic_json, classify_response, conditions, content_indices,
    file_hash, filter_trait, fingerprint, freeze_analysis_policy, freeze_stage, initialize_run, judge_one,
    judge_payload, judge_result, load_inputs, matched_decision, parse_score,
    record_path, render_tokens, replay_response, response_mean_difference,
    response_providers, stable_seed, template_diagnostics, verify_tokenizer_files,
)

ROOT = Path(__file__).resolve().parents[2]
INPUTS = ROOT / "data/persona_traits"
if not INPUTS.is_dir() and Path("/root/persona_inputs").is_dir():
    INPUTS = Path("/root/persona_inputs")


def complete_response():
    return {"status": "complete", "response": "A useful answer.", "stop_reason": "eos",
            "empty_response": False, "refusal": False, "unexpected_thinking": False,
            "response_content_indices": [0, 1]}


class CollectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.protocol, cls.traits, _ = load_inputs(INPUTS)

    def test_every_question_crosses_both_sides_of_all_five_pairs_once(self):
        rows = [row for trait in TRAITS for row in conditions(trait, self.traits[trait], self.protocol)]
        self.assertEqual(len(rows), 2400)
        self.assertEqual(len({row["response_id"] for row in rows}), 2400)
        self.assertEqual({row["rollout_index"] for row in rows}, {0})
        for trait in TRAITS:
            grouped = {}
            for row in conditions(trait, self.traits[trait], self.protocol):
                grouped.setdefault(row["contrast_id"], []).append(row)
                self.assertEqual(row["messages"][0]["content"], self.traits[trait]["system_prompt_pairs"][int(row["pair_id"]) - 1][row["polarity"]])
            self.assertEqual(len(grouped), 200)
            for pair in grouped.values():
                self.assertEqual({row["polarity"] for row in pair}, {"positive", "negative"})
                self.assertEqual(pair[0]["messages"][1], pair[1]["messages"][1])

    def test_seed_implements_frozen_unsigned_sha256_rule(self):
        expected = int(hashlib.sha256(b"20261004|curiosity|03|12|0|positive").hexdigest()[:8], 16)
        self.assertEqual(stable_seed("curiosity", "03", "12", "positive"), expected)
        self.assertNotEqual(expected, stable_seed("curiosity", "03", "12", "negative"))

    def test_rendering_has_exact_system_and_user_without_extra_persona(self):
        calls = []
        messages = [{"role": "system", "content": "Specific system"}, {"role": "user", "content": "Question"}]
        class Tokenizer:
            def apply_chat_template(self, actual, **kwargs):
                calls.append((actual, kwargs))
                if not kwargs["tokenize"]:
                    return "rendered<think>\n\n</think>\n\n"
                # Native Transformers 5.18 now returns a mapping by default.
                return {"input_ids": [7, 8]} if kwargs.get("return_dict", True) else [7, 8]
            def encode(self, text, **kwargs):
                self.add_special_tokens = kwargs["add_special_tokens"]
                return [7, 8]
        tokenizer = Tokenizer()
        _, ids = render_tokens(tokenizer, messages)
        self.assertEqual(ids, [7, 8])
        self.assertFalse(tokenizer.add_special_tokens)
        for actual, kwargs in calls:
            self.assertEqual(actual, messages)
            self.assertFalse(kwargs["enable_thinking"])
            self.assertTrue(kwargs["add_generation_prompt"])
            self.assertFalse(kwargs["return_dict"])

    def test_template_preflight_checks_every_condition_under_new_mapping_default(self):
        class Tokenizer:
            def apply_chat_template(self, actual, **kwargs):
                if not kwargs["tokenize"]:
                    return "rendered<think>\n\n</think>\n\n"
                return {"input_ids": [7, 8]} if kwargs.get("return_dict", True) else [7, 8]
            def encode(self, text, **kwargs):
                return [7, 8]
        result = template_diagnostics(Tokenizer(), INPUTS)
        self.assertTrue(result["passed"])
        self.assertEqual(result["checked_conditions"], 2400)
        self.assertEqual(result["default_tokenized_output_type"], "dict")
        self.assertEqual(result["default_output_keys"], ["input_ids"])
        self.assertTrue(all(row["checked"] == 400 for row in result["traits"].values()))

    def test_cached_tokenizer_preflight_rejects_modified_files(self):
        from model_lab.persona_extraction import MODEL_ID, MODEL_REVISION
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint = Path(temporary)
            files = {}
            for name in ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja"):
                path = checkpoint / name
                path.write_text("audited " + name)
                files[name] = {"bytes": path.stat().st_size, "sha256": file_hash(path)}
            manifest = {"model_id": MODEL_ID, "revision": MODEL_REVISION, "files": files}
            self.assertEqual(verify_tokenizer_files(checkpoint, manifest), files)
            (checkpoint / "chat_template.jinja").write_text("changed")
            with self.assertRaisesRegex(ValueError, "integrity check failed"):
                verify_tokenizer_files(checkpoint, manifest)

    def test_content_mask_excludes_special_eos_and_thinking_without_retokenizing(self):
        class Tokenizer:
            all_special_ids = [0, 8]
            def decode(self, ids, **kwargs):
                return {0: "<|pad|>", 1: "<think>", 2: "reason", 3: "</think>",
                        4: "answer", 5: " more", 8: "<|im_end|>"}[ids[0]]
        ids = [0, 1, 2, 3, 4, 5, 8]
        indices, unexpected = content_indices(Tokenizer(), ids, {8})
        self.assertEqual(indices, [4, 5])
        self.assertTrue(unexpected)
        self.assertEqual(ids, [0, 1, 2, 3, 4, 5, 8])

    def test_persona_inability_is_not_misclassified_as_task_refusal(self):
        self.assertFalse(classify_response("I cannot stop thinking about my partner.")["refusal"])
        self.assertFalse(classify_response("I can't make myself care about this.")["refusal"])
        self.assertTrue(classify_response("I'm sorry, I cannot help with that request.")["refusal"])

    def test_judge_parser_is_complete_integer_only(self):
        for value in (0, 25, 50, 75, 100):
            self.assertEqual(parse_score(f" \n{value}\t"), value)
        for invalid in ("101", "-1", "50.0", "Score: 50", "50\n75", "050", "", None, ["50"]):
            with self.assertRaises(ValueError):
                parse_score(invalid)

    def test_hidden_reasoning_not_parsed_and_truncation_fails_even_with_valid_integer(self):
        body = {"choices": [{"message": {"content": "75", "reasoning": "0"}, "finish_reason": "stop"}]}
        self.assertEqual(judge_result(body), 75)
        body["choices"][0]["finish_reason"] = "length"
        with self.assertRaisesRegex(ValueError, "truncation"):
            judge_result(body)
        body["choices"][0].update(finish_reason="stop", message={"content": "75", "refusal": "blocked"})
        with self.assertRaisesRegex(ValueError, "refusal"):
            judge_result(body)

    def test_judge_blinding_and_single_pass_untrusted_placeholder_substitution(self):
        payload = judge_payload(self.protocol, "Question: {question}\nResponse: {response}",
                                "literal {response}", "literal {question}")
        self.assertEqual(payload["messages"][0]["content"],
                         "Question: literal {response}\nResponse: literal {question}")
        self.assertFalse(any(key in payload for key in ("polarity", "temperature", "top_p", "top_k", "seed")))
        self.assertEqual(payload["provider"]["only"], ["google-ai-studio"])
        self.assertEqual(payload["reasoning"], {"effort": "low", "exclude": True})

    def test_judge_errors_are_not_zero_and_attempt_limit_survives_resumption(self):
        calls = []
        def transport(payload):
            calls.append(payload)
            return {"choices": [{"message": {"content": "bad"}, "finish_reason": "stop"}]}, {}
        record = {"response_id": "x", "question": "Q", "response": "A"}
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "judge.json"
            saved = judge_one(path, self.protocol, "{question} {response}", record, "trait", transport)
            self.assertIsNone(saved["score"])
            self.assertEqual(len(calls), 3)
            judge_one(path, self.protocol, "{question} {response}", record, "trait", transport)
            self.assertEqual(len(calls), 3)

    def test_matching_requires_strict_trait_thresholds_and_both_quality_gates(self):
        p, n = complete_response(), complete_response()
        self.assertEqual(matched_decision(p, n, 51, 49, 50, 50), [])
        for scores in ((50, 49, 50, 50), (51, 50, 50, 50), (51, 49, 49, 100), (51, 49, 100, 49), (None, 49, 100, 100)):
            self.assertTrue(matched_decision(p, n, *scores))
        n["stop_reason"] = "token_limit_truncation"
        self.assertEqual(matched_decision(p, n, 100, 0, 100, 100), [])
        self.assertIn("coherence_below_50", matched_decision(p, n, 100, 0, 100, 49))

    def test_response_means_have_equal_weight_even_for_three_and_six_token_answers(self):
        # Two positive response means, obtained from lengths 3 and 6, are 2 and
        # 10. Both negative means are 6. Equal response weighting cancels;
        # flattening tokens would incorrectly produce a nonzero direction.
        sums = {"positive": 2.0 + 10.0, "negative": 6.0 + 6.0}
        counts = {"positive": 2, "negative": 2}
        self.assertEqual(response_mean_difference(sums, counts), 0.0)
        self.assertNotEqual((3 * 2.0 + 6 * 10.0) / 9 - 6.0, 0.0)
        for bad in ({"positive": 0, "negative": 0}, {"positive": 2, "negative": 1}):
            with self.assertRaises(RuntimeError):
                response_mean_difference(sums, bad)

    def test_provider_provenance_uses_selected_router_endpoint(self):
        body = {"openrouter_metadata": {"endpoints": {"available": [
            {"provider": "Google Vertex", "selected": False},
            {"provider": "Google AI Studio", "selected": True}]}}}
        self.assertEqual(response_providers(body), ["Google AI Studio"])

    def test_judge_only_correction_preserves_frozen_generation_but_not_judge_stage(self):
        def revised_payload(*args):
            return {"changed": True}
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            generation = freeze_stage(root, "generation")
            freeze_stage(root, "judging")
            with mock.patch("model_lab.persona_extraction.judge_payload", revised_payload):
                self.assertEqual(freeze_stage(root, "generation"), generation)
                with self.assertRaisesRegex(ValueError, "judging implementation changed"):
                    freeze_stage(root, "judging")

    def test_judge_pattern_correction_does_not_change_generation_fingerprint(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            generation = freeze_stage(root, "generation")
            freeze_stage(root, "judging")
            with mock.patch("model_lab.persona_extraction._SCORE", re.compile(r"changed")):
                self.assertEqual(freeze_stage(root, "generation"), generation)
                with self.assertRaisesRegex(ValueError, "judging implementation changed"):
                    freeze_stage(root, "judging")

    def test_concurrent_stage_writes_are_atomic_and_leave_no_temporary_files(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            target = root / "shared.json"
            with ThreadPoolExecutor(max_workers=6) as pool:
                list(pool.map(lambda index: atomic_json(target, {"index": index, "padding": "x" * 1000}), range(60)))
            self.assertIsInstance(json.loads(target.read_text())["index"], int)
            self.assertEqual([path.name for path in root.iterdir()], ["shared.json"])
            with ThreadPoolExecutor(max_workers=6) as pool:
                hashes = list(pool.map(lambda index: freeze_stage(root, "generation")["implementation_sha256"], range(6)))
            self.assertEqual(len(set(hashes)), 1)
            metadata = json.loads((root / "implementations/generation.json").read_text())
            self.assertEqual(file_hash(root / "implementations/generation.py"), metadata["module_sha256"])

    def test_analysis_policy_is_frozen_separately_from_generation_and_judging(self):
        def revised_exclusions(record):
            return ["changed"]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "run"
            inputs = Path(temporary) / "inputs"
            shutil.copytree(INPUTS, inputs)
            initialize_run(root, inputs, "test-policy")
            generation = freeze_stage(root, "generation")
            judging = freeze_stage(root, "judging")
            policy = freeze_analysis_policy(root, inputs)
            filtering = freeze_stage(root, "filtering", policy["policy_sha256"])
            self.assertEqual(filtering["analysis_policy_sha256"], policy["policy_sha256"])
            with mock.patch("model_lab.persona_extraction.exclusion_reasons", revised_exclusions):
                self.assertEqual(freeze_stage(root, "generation"), generation)
                self.assertEqual(freeze_stage(root, "judging"), judging)
                with self.assertRaisesRegex(ValueError, "filtering implementation changed"):
                    freeze_stage(root, "filtering", policy["policy_sha256"])
            updated = json.loads((inputs / "extraction_policy.json").read_text())
            updated["rationale"] += " Altered policy."
            atomic_json(inputs / "extraction_policy.json", updated)
            with self.assertRaisesRegex(ValueError, "analytical extraction policy changed"):
                freeze_analysis_policy(root, inputs)
            self.assertEqual(freeze_stage(root, "generation"), generation)
            self.assertEqual(freeze_stage(root, "judging"), judging)

    def test_filter_keeps_capped_answers_and_reports_incomplete_coverage_as_usable(self):
        from model_lab.persona_extraction import judge_path
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            initialize_run(root, INPUTS, "test-filter")
            settings = json.loads((root / "run_settings.json").read_text())
            trait = "curiosity"
            for row in conditions(trait, self.traits[trait], self.protocol):
                record = {**row, **complete_response(), "condition_sha256": fingerprint(row),
                          "settings_sha256": fingerprint(settings)}
                if (row["question_id"], row["polarity"]) in (("01", "negative"), ("02", "positive")):
                    record["stop_reason"] = "token_limit_truncation"
                atomic_json(record_path(root, trait, row["response_id"]), record)
                for kind in ("trait", "coherence"):
                    score = 50 if kind == "coherence" else (75 if row["polarity"] == "positive" else 25)
                    if row["question_id"] == "01" and kind == "trait" and row["polarity"] == "positive":
                        score = 50
                    atomic_json(judge_path(root, trait, row["response_id"], kind),
                                {"score": score, "response_sha256": fingerprint(record)})
            result = filter_trait(root, INPUTS, trait)
            self.assertEqual(result["accepted_pairs"], 195)
            self.assertEqual(result["accepted_positive"], result["accepted_negative"])
            self.assertEqual(result["missing_questions"], ["01"])
            self.assertFalse(result["coverage_passed"])
            self.assertTrue(result["extraction_usable"])
            self.assertEqual(result["generated_capped_positive"], 5)
            self.assertEqual(result["generated_capped_negative"], 5)
            self.assertEqual(result["accepted_capped_positive"], 5)
            self.assertEqual(result["accepted_capped_negative"], 0)
            self.assertEqual(result["accepted_capped_pairs"], 5)
            self.assertEqual(result["policy_sha256"], file_hash(INPUTS / "extraction_policy.json"))
            self.assertFalse((root / "publication" / "persona_manifest.json").exists())

    def test_filter_still_rejects_a_trait_with_no_accepted_matches(self):
        from model_lab.persona_extraction import judge_path
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            initialize_run(root, INPUTS, "test-empty")
            settings = json.loads((root / "run_settings.json").read_text())
            trait = "curiosity"
            for row in conditions(trait, self.traits[trait], self.protocol):
                record = {**row, **complete_response(), "condition_sha256": fingerprint(row),
                          "settings_sha256": fingerprint(settings)}
                atomic_json(record_path(root, trait, row["response_id"]), record)
                for kind in ("trait", "coherence"):
                    atomic_json(judge_path(root, trait, row["response_id"], kind),
                                {"score": 50, "response_sha256": fingerprint(record)})
            result = filter_trait(root, INPUTS, trait)
            self.assertEqual(result["accepted_pairs"], 0)
            self.assertFalse(result["extraction_usable"])
            self.assertFalse(result["coverage_passed"])
            self.assertEqual(len(result["missing_questions"]), 40)
            self.assertEqual(len(result["missing_system_prompt_pairs"]), 5)


class NativeActivationTests(unittest.TestCase):
    def test_assembly_accepts_sparse_coverage_and_embeds_separate_policy_provenance(self):
        try:
            import torch
            from safetensors.torch import load_file
        except ImportError:
            self.skipTest("Native tensor packaging is exercised in the Modal preflight image")
        from model_lab.persona_extraction import BOUNDARY, assemble_bank, save_tensors
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            initialize_run(root, INPUTS, "test-sparse-bank")
            original = json.loads((root / "run_settings.json").read_text())
            policy = freeze_analysis_policy(root, INPUTS)
            summaries = {}
            import threading
            filters_ready = threading.Barrier(len(TRAITS))
            def concurrent_filter(root, inputs, trait):
                filters_ready.wait(timeout=10)
                return summaries[trait]
            for index, trait in enumerate(TRAITS):
                summary = {"trait": trait, "accepted_pairs": 1, "accepted_positive": 1,
                           "accepted_negative": 1, "generated_positive": 200, "generated_negative": 200,
                           "accepted_by_question": {f"{i:02}": int(i == 1) for i in range(1, 41)},
                           "accepted_by_system_prompt_pair": {f"{i:02}": int(i == 1) for i in range(1, 6)},
                           "missing_questions": [f"{i:02}" for i in range(2, 41)],
                           "missing_system_prompt_pairs": [f"{i:02}" for i in range(2, 6)],
                           "coverage_passed": False, "extraction_usable": True,
                           "policy_sha256": policy["policy_sha256"]}
                for field in ("generated_capped_positive", "generated_capped_negative",
                              "accepted_capped_positive", "accepted_capped_negative", "accepted_capped_pairs"):
                    summary[field] = 0
                summaries[trait] = summary
                filter_path = root / "filtering" / f"{trait}.json"
                atomic_json(filter_path, summary)
                atomic_json(root / "models" / f"{trait}.json", {"checkpoint": {"hidden_size": 2}})
                save_tensors(root / "trait_vectors" / f"{trait}.safetensors",
                             {"vectors": torch.full((32, 2), float(index + 1), dtype=torch.float32)},
                             {"trait": trait, "filter_sha256": file_hash(filter_path),
                              "activation_boundary": BOUNDARY, "policy_sha256": policy["policy_sha256"]})
                for record_index in range(2):
                    atomic_json(root / "judges" / trait / f"fixture-{record_index}.json",
                                {"score": 75, "attempts": [{"status": "complete", "raw_response": {
                                    "model": "test-only", "provider": "Google AI Studio",
                                    "usage": {"cost": 0.01}}}]})
            with mock.patch("model_lab.persona_extraction.filter_trait",
                            side_effect=concurrent_filter):
                result = assemble_bank(root, INPUTS)
            self.assertEqual(result["shape"], [32, 6, 2])
            manifest = json.loads((root / "publication/persona_manifest.json").read_text())
            self.assertEqual(manifest["filtering"]["policy"], policy["policy"])
            self.assertEqual(manifest["filtering"]["policy_source"], policy["policy_source"])
            self.assertEqual(manifest["filtering"]["policy_sha256"], policy["policy_sha256"])
            self.assertEqual(manifest["sources"][policy["policy_source"]],
                             {"sha256": policy["policy_sha256"], "bytes": policy["policy_bytes"]})
            self.assertEqual({key: value for key, value in manifest["sources"].items()
                              if key != policy["policy_source"]}, original["sources"])
            self.assertTrue(all(not summary["coverage_passed"] and summary["extraction_usable"]
                                for summary in manifest["filtering"]["traits"].values()))
            self.assertEqual(manifest["judging"]["score_records"], 12)
            self.assertEqual(manifest["judging"]["actual_attempts"], 12)
            self.assertEqual(manifest["judging"]["valid_scores"], 12)
            self.assertAlmostEqual(manifest["judging"]["reported_cost_usd"], 0.12)
            bank = load_file(str(root / "publication/persona_vectors.safetensors"))["vectors"]
            self.assertTrue(torch.equal(bank[:, 5], torch.full((32, 2), 6.0)))

    def test_replay_uses_actual_block_outputs_content_positions_and_removes_hooks(self):
        try:
            import torch
        except ImportError:
            self.skipTest("Torch is exercised in the Modal GPU preflight image")
        class Block(torch.nn.Module):
            def __init__(self, offset):
                super().__init__()
                self.offset = offset
            def forward(self, hidden):
                return hidden + self.offset
        class Decoder(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.layers = torch.nn.ModuleList([Block(1), Block(10)])
            def forward(self, input_ids, **kwargs):
                hidden = input_ids.float().unsqueeze(-1).expand(-1, -1, 2)
                for layer in self.layers:
                    hidden = layer(hidden)
                return hidden * 100  # final normalization proxy must not leak
        decoder = Decoder()
        adapter = SimpleNamespace(decoder=decoder, layers=decoder.layers, n_layers=2,
                                  hidden_size=2, device="cpu")
        model = SimpleNamespace(model=SimpleNamespace(rope_deltas="stale"))
        record = {"prompt_token_ids": [1000, 2000], "response_token_ids": [3, 7, 99],
                  "response_content_indices": [0, 1]}
        result = replay_response(model, adapter, record)
        self.assertTrue(torch.equal(result, torch.tensor([[6., 6.], [16., 16.]])))
        self.assertEqual(result.dtype, torch.float32)
        self.assertIsNone(model.model.rope_deltas)
        self.assertTrue(all(not layer._forward_hooks for layer in decoder.layers))
        record["response_content_indices"] = [0, 0]
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            replay_response(model, adapter, record)


if __name__ == "__main__":
    unittest.main()
