"""Resume, token fidelity and provenance checks for batched continuation."""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

from model_lab import persona_batched as batched
from model_lab.persona_extraction import (
    MODEL_ID, MODEL_REVISION, atomic_json, conditions, fingerprint, frozen_settings,
    initialize_run, load_inputs, record_path,
)

ROOT = Path(__file__).resolve().parents[2]
INPUTS = ROOT / "data/persona_traits"


def saved_record(row, settings, **extra):
    return {**row, "condition_sha256": fingerprint(row), "settings_sha256": fingerprint(settings),
            "model_id": MODEL_ID, "model_revision": MODEL_REVISION, "status": "complete", **extra}


def sampling_fixture():
    config = {"do_sample": True, "temperature": 1.0, "top_p": 1.0, "top_k": 50,
              "repetition_penalty": 1.0, "max_new_tokens": 1000, "use_cache": True}
    return {"collection_kwargs": {**config, "pad_token_id": 99}, "resolved_config": config,
            "resolved_config_sha256": fingerprint(config),
            "checkpoint_generation_config_sha256": batched.GENERATION_CONFIG_SHA256}


class ResumeTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        initialize_run(self.root, INPUTS, "test-batched")
        self.settings, self.protocol, self.traits = frozen_settings(self.root, INPUTS)
        self.rows = conditions("curiosity", self.traits["curiosity"], self.protocol)

    def test_existing_success_error_and_cap_are_preserved_and_mismatch_blocks(self):
        for row, status, stop in zip(self.rows, ("complete", "generation_error", "complete"),
                                     ("eos", None, "token_limit_truncation")):
            atomic_json(record_path(self.root, row["trait"], row["response_id"]),
                        saved_record(row, self.settings, status=status, stop_reason=stop))
        pending = batched.pending_conditions(self.root, self.rows, self.settings)
        self.assertEqual(pending, self.rows[3:])
        path = record_path(self.root, "curiosity", self.rows[0]["response_id"])
        changed = json.loads(path.read_text())
        changed["condition_sha256"] = "changed"
        atomic_json(path, changed)
        with self.assertRaisesRegex(ValueError, "provenance mismatch"):
            batched.pending_conditions(self.root, self.rows, self.settings)

    def test_failed_batch_blocks_missing_conditions_without_automatic_reroll(self):
        path = self.root / "generation_batches/curiosity/failed.json"
        atomic_json(path, {"status": "failed", "condition_ids": [self.rows[0]["response_id"]]})
        with self.assertRaisesRegex(RuntimeError, "explicit repair"):
            batched.pending_conditions(self.root, self.rows, self.settings)

    def test_exclusive_creation_cannot_replace_existing_bytes(self):
        path = self.root / "exclusive.json"
        batched.exclusive_record(path, {"one": 1})
        before = path.read_bytes()
        with self.assertRaises(FileExistsError):
            batched.exclusive_record(path, {"one": 2})
        self.assertEqual(path.read_bytes(), before)

    def test_public_provenance_requires_completed_receipt_and_exact_file_hash(self):
        rows = self.rows[:2]
        sampling = sampling_fixture()
        stage = batched.freeze_batched_stage(self.root, sampling)
        batch_id = "a" * 32
        paths = []
        for row in rows:
            path = record_path(self.root, row["trait"], row["response_id"])
            atomic_json(path, saved_record(row, self.settings, generation_implementation=batched.STAGE,
                        generation_implementation_sha256=stage["implementation_sha256"],
                        generation_batch_id=batch_id, generation_batch_size=2,
                        sampling_configuration_sha256=sampling["resolved_config_sha256"]))
            paths.append(path)
        receipt = {"batch_id": batch_id, "stage": batched.STAGE, "status": "complete",
                   "implementation_sha256": stage["implementation_sha256"],
                   "sampling_configuration_sha256": sampling["resolved_config_sha256"],
                   "condition_ids": [row["response_id"] for row in rows],
                   "condition_seeds": [row["seed"] for row in rows],
                   "record_file_sha256": {row["response_id"]: batched.file_hash(path)
                                           for row, path in zip(rows, paths)}}
        batch_path = self.root / f"generation_batches/curiosity/{batch_id}.json"
        atomic_json(batch_path, receipt)
        with mock.patch.object(batched, "frozen_settings", return_value=(self.settings, self.protocol, {"curiosity": {}})), \
                mock.patch.object(batched, "conditions", return_value=rows):
            result = batched.public_generation_provenance(self.root, INPUTS)
            self.assertEqual(result["execution_counts_by_trait"]["curiosity"][batched.STAGE], 2)
            self.assertNotIn("condition_ids", json.dumps(result))
            atomic_json(batch_path, {**receipt, "status": "failed"})
            with self.assertRaisesRegex(ValueError, "batch receipt"):
                batched.public_generation_provenance(self.root, INPUTS)
            atomic_json(batch_path, receipt)
            atomic_json(paths[0], {**json.loads(paths[0].read_text()), "changed": True})
            with self.assertRaisesRegex(ValueError, "completed batch receipt"):
                batched.public_generation_provenance(self.root, INPUTS)


class NativeBatchTests(ResumeTests):
    def setUp(self):
        super().setUp()
        try:
            import torch
            from transformers import GenerationConfig
        except ImportError:
            self.skipTest("Native batch checks run in the isolated Torch environment")
        self.torch = torch
        self.config = GenerationConfig(do_sample=True, temperature=1.0, top_p=1.0, top_k=50,
                                       max_new_tokens=1000, pad_token_id=99, eos_token_id=99)

        class Tokenizer:
            all_special_ids = [99]
            pad_token_id = 99
            calls = []
            def apply_chat_template(inner, messages, **kwargs):
                inner.calls.append(kwargs)
                rendered = str(len(messages[0]["content"])) + "<think>\n\n</think>\n\n"
                return inner.encode(rendered) if kwargs["tokenize"] else rendered
            def encode(inner, rendered, **kwargs):
                return [31] * (2 + int(rendered.split("<", 1)[0]) % 5)
            def decode(inner, ids, **kwargs):
                return "".join({99: "<|im_end|>", 13: "<think>", 14: "</think>"}.get(i, "word ") for i in ids)
        self.tokenizer = Tokenizer()

    def record(self, tokens, prompt=(31, 32), padded_width=4):
        output = [99] * (padded_width - len(prompt)) + list(prompt) + tokens
        return batched.completion_record(self.tokenizer, self.rows[0], "formatted", list(prompt),
                output, padded_width, {99}, self.settings, sampling_fixture(),
                {"implementation_sha256": "stage"}, "b" * 32, 2, 1.0)

    def test_first_eos_is_retained_and_post_eos_padding_removed(self):
        record = self.record([11, 12, 99, 99, 99])
        self.assertEqual(record["prompt_token_ids"], [31, 32])
        self.assertEqual(record["response_token_ids"], [11, 12, 99])
        self.assertEqual(record["response_content_indices"], [0, 1])
        self.assertEqual(record["stop_reason"], "eos")
        self.assertEqual(record["seed"], self.rows[0]["seed"])
        self.assertEqual(record["effective_generation_config"]["top_k"], 50)

    def test_cap_and_thinking_masks_are_honest_and_short_non_eos_fails(self):
        record = self.record([11] * 1000)
        self.assertEqual(record["stop_reason"], "token_limit_truncation")
        self.assertEqual(len(record["response_token_ids"]), 1000)
        thinking = self.record([13, 11, 14, 12, 99])
        self.assertTrue(thinking["unexpected_thinking"])
        self.assertEqual(thinking["response_content_indices"], [3])
        with self.assertRaisesRegex(ValueError, "length/stop"):
            self.record([11, 12])
        with self.assertRaisesRegex(ValueError, "unpadded prompt"):
            batched.completion_record(self.tokenizer, self.rows[0], "formatted", [31, 32],
                [99, 99, 31, 33, 11, 99], 4, {99}, self.settings, sampling_fixture(),
                {"implementation_sha256": "stage"}, "b" * 32, 1, 1.0)

    def run_fake_batch(self, fail=False, limit=8):
        torch = self.torch
        calls = []
        def generate(**kwargs):
            calls.append(kwargs)
            if fail:
                raise RuntimeError("native test failure")
            ids, mask = kwargs["input_ids"], kwargs["attention_mask"]
            self.assertEqual(kwargs["logits_to_keep"], 1)
            self.assertTrue(all(mask[i].tolist() == [0] * int((mask[i] == 0).sum()) +
                                [1] * int(mask[i].sum()) for i in range(len(ids))))
            suffix = torch.tensor([[11, 99, 99] if i % 2 == 0 else [11, 12, 99]
                                   for i in range(len(ids))])
            return torch.cat([ids, suffix], 1)
        model = SimpleNamespace(model=SimpleNamespace(rope_deltas="stale"), generate=generate)
        adapter = SimpleNamespace(device="cpu")
        with mock.patch.object(batched, "load_model", return_value=(self.tokenizer, model, adapter)), \
                mock.patch.object(batched, "checkpoint_metadata", return_value={"same": True}), \
                mock.patch.object(batched, "resolved_sampling", return_value=(self.config, sampling_fixture())), \
                mock.patch.object(torch.cuda, "synchronize"):
            result = batched.generate_batched_trait(self.root, INPUTS, Path("unused"), "curiosity", limit=limit)
        return result, calls

    def test_mixed_serial_resume_preserves_bytes_and_exact_batch_records(self):
        first = self.rows[0]
        path = record_path(self.root, "curiosity", first["response_id"])
        atomic_json(path, saved_record(first, self.settings, stop_reason="token_limit_truncation"))
        before = path.read_bytes()
        _, calls = self.run_fake_batch()
        self.assertEqual(len(calls), 1)
        self.assertEqual(path.read_bytes(), before)
        batches = list((self.root / "generation_batches/curiosity").glob("*.json"))
        self.assertEqual(len(batches), 1)
        receipt = json.loads(batches[0].read_text())
        self.assertEqual(receipt["status"], "complete")
        rows_by_id = {row["response_id"]: row for row in self.rows}
        self.assertTrue(all(rows_by_id[identity]["polarity"] == "positive" for identity in receipt["condition_ids"]))
        for identity in receipt["condition_ids"]:
            record = json.loads(record_path(self.root, "curiosity", identity).read_text())
            self.assertEqual(record["condition_sha256"], fingerprint(rows_by_id[identity]))
            _, original_ids = batched.render_tokens(self.tokenizer, rows_by_id[identity]["messages"])
            self.assertEqual(record["prompt_token_ids"], original_ids)
            self.assertIn(record["response_token_ids"], ([11, 99], [11, 12, 99]))
        self.assertTrue(all(not call["enable_thinking"] for call in self.tokenizer.calls))
        # A completed run loads no model, even when collection contains mixed methods.
        for row in batched.pending_conditions(self.root, self.rows, self.settings):
            atomic_json(record_path(self.root, "curiosity", row["response_id"]), saved_record(row, self.settings))
        with mock.patch.object(batched, "load_model", side_effect=AssertionError("must not load")):
            batched.generate_batched_trait(self.root, INPUTS, Path("unused"), "curiosity")

    def test_failed_generation_saves_no_bogus_rows_and_blocks_retry(self):
        with self.assertRaisesRegex(RuntimeError, "native test failure"):
            self.run_fake_batch(fail=True)
        self.assertEqual(list((self.root / "responses/curiosity").glob("*.json")), [])
        receipt = json.loads(next((self.root / "generation_batches/curiosity").glob("*.json")).read_text())
        self.assertEqual(receipt["status"], "failed")
        with self.assertRaisesRegex(RuntimeError, "explicit repair"):
            batched.pending_conditions(self.root, self.rows, self.settings)

    def test_audited_native_config_resolution_and_config_hash_gate(self):
        from transformers.generation.utils import GenerationMixin
        from transformers import PretrainedConfig
        model = SimpleNamespace(generation_config=self.config, config=PretrainedConfig(is_encoder_decoder=False))
        model._prepare_generation_config = lambda config, **kwargs: GenerationMixin._prepare_generation_config(model, config, **kwargs)
        checkpoint = ROOT / "artifacts/huggingface/Qwen3.5-9B-mooody"
        with mock.patch("importlib.metadata.version", return_value="test-version"):
            config, receipt = batched.resolved_sampling(model, self.tokenizer, checkpoint)
        self.assertEqual((config.top_k, config.max_new_tokens, config.use_cache), (50, 1000, True))
        self.assertEqual(receipt["checkpoint_generation_config_sha256"], batched.GENERATION_CONFIG_SHA256)
        with tempfile.TemporaryDirectory() as temporary:
            changed = Path(temporary)
            (changed / "generation_config.json").write_text("{}")
            with self.assertRaisesRegex(ValueError, "audited checkpoint"):
                batched.resolved_sampling(model, self.tokenizer, changed)

    def test_production_sampler_native_top_k_and_finished_row_stream_independence(self):
        from model_lab.seeded_sampling import IndependentRowSampler
        from transformers.generation.logits_process import TopKLogitsWarper
        torch = self.torch
        scores = torch.arange(64, dtype=torch.float32).repeat(2, 1) / 12
        sampler = IndependentRowSampler([101, 202], self.config, "cpu")
        alone = IndependentRowSampler([202], self.config, "cpu")
        native_generator = torch.Generator(device="cpu").manual_seed(202)
        for step in range(12):
            ids = torch.full((2, step + 1), 99, dtype=torch.long)
            # First row represents an already-ended row: its draws cannot
            # consume the still-active second row's independently seeded RNG.
            actual = sampler(ids, scores.clone()).argmax(-1)
            single = alone(ids[1:], scores[1:].clone()).argmax(-1)
            expected = torch.multinomial(torch.softmax(TopKLogitsWarper(50)(ids[1:], scores[1:].clone()), -1),
                                         1, generator=native_generator).squeeze(-1)
            self.assertTrue(torch.equal(actual[1:], single))
            self.assertTrue(torch.equal(single, expected))


if __name__ == "__main__":
    unittest.main()
