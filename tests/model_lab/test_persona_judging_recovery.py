"""Transport recovery preserves scores, request identity and original journals."""
import io
import json
from pathlib import Path
import tempfile
import unittest
from urllib.error import HTTPError

from model_lab import persona_judging_recovery as recovery
from model_lab.persona_extraction import (
    TRAITS, atomic_json, conditions, fingerprint, frozen_settings, initialize_run,
    judge_path, record_path,
)

ROOT = Path(__file__).resolve().parents[2]
INPUTS = ROOT / "data/persona_traits"


def failed_attempts(code=429):
    return [{"attempt": number, "status": "judge_error", "http_status": code,
             "error_type": "HTTPError", "finished_at_unix": float(number)} for number in range(1, 4)]


def successful_body(score=75):
    return {"provider": "Google AI Studio", "choices": [{"finish_reason": "stop",
            "message": {"content": str(score)}}]}


class RecoveryTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        initialize_run(self.root, INPUTS, "test-recovery")
        self.settings, self.protocol, self.traits = frozen_settings(self.root, INPUTS)
        self.row = conditions("paranoia", self.traits["paranoia"], self.protocol)[0]
        self.response = {**self.row, "status": "complete", "response": "Test-only fixture answer."}
        self.rubric = self.traits["paranoia"]["evaluation_prompt"]
        self.identity, self.payload = recovery.recovery_identity(self.protocol, self.rubric, self.response, "trait")
        self.path = judge_path(self.root, "paranoia", self.row["response_id"], "trait")
        self.stage = recovery.freeze_recovery(self.root, INPUTS)

    def save(self, score=None, attempts=None):
        atomic_json(self.path, {**self.identity, "score": score,
                               "attempts": failed_attempts() if attempts is None else attempts})

    def test_only_three_exhausted_http429_attempts_are_eligible(self):
        self.assertTrue(recovery.eligible({"score": None, "attempts": failed_attempts()}))
        for invalid in ({"score": 75, "attempts": failed_attempts()},
                        {"score": None, "attempts": failed_attempts(401)},
                        {"score": None, "attempts": failed_attempts()[:2]},
                        {"score": None, "attempts": [{"status": "judge_error", "error_type": "ValueError"}] * 3},
                        {"score": None, "attempts": failed_attempts()[:2] + [{"status": "request_started"}]}):
            self.assertFalse(recovery.eligible(invalid))

    def test_valid_score_and_ineligible_journal_bytes_never_change(self):
        for score, attempts in ((75, failed_attempts()), (None, failed_attempts(401))):
            self.save(score, attempts)
            original = self.path.read_bytes()
            recovery.recover_one(self.root, self.path, self.protocol, self.rubric, self.response, "trait",
                                 self.stage, recovery.Pacer(sleep=lambda _: None),
                                 transport=lambda _: self.fail("must not call"))
            self.assertEqual(self.path.read_bytes(), original)

    def test_exact_request_durable_before_http_and_prefix_immutable(self):
        self.save()
        original = self.path.read_bytes()
        prefix = json.loads(original)["attempts"]
        commits = []
        def transport(payload):
            self.assertEqual(payload, self.payload)
            journal = json.loads(self.path.read_text())
            self.assertEqual(journal["attempts"][-1]["status"], "request_started")
            self.assertGreaterEqual(len(commits), 2)
            return successful_body(), {}
        saved = recovery.recover_one(self.root, self.path, self.protocol, self.rubric, self.response, "trait",
                self.stage, recovery.Pacer(sleep=lambda _: None), commit=lambda: commits.append(1), transport=transport)
        self.assertEqual(saved["score"], 75)
        self.assertEqual(saved["attempts"][:3], prefix)
        snapshot = self.root / recovery.STAGE / "originals/paranoia" / self.path.name
        self.assertEqual(snapshot.read_bytes(), original)
        self.assertEqual(saved["attempts"][3]["attempt"], 4)
        self.assertEqual(saved["attempts"][3]["recovery_attempt"], 1)
        before = self.path.read_bytes()
        recovery.recover_one(self.root, self.path, self.protocol, self.rubric, self.response, "trait",
                             self.stage, recovery.Pacer(), transport=lambda _: self.fail("no valid-score reroll"))
        self.assertEqual(self.path.read_bytes(), before)

    def test_additional_http429_budget_is_three_and_survives_resume(self):
        self.save()
        calls = []
        def transport(payload):
            calls.append(payload)
            raise HTTPError("https://example.invalid", 429, "rate limited", {}, io.BytesIO(b'{"error":"rate"}'))
        pacer = recovery.Pacer(sleep=lambda _: None)
        result = recovery.recover_one(self.root, self.path, self.protocol, self.rubric, self.response, "trait",
                                     self.stage, pacer, transport=transport)
        self.assertIsNone(result["score"])
        self.assertEqual(len(calls), 3)
        self.assertEqual(len(result["attempts"]), 6)
        recovery.recover_one(self.root, self.path, self.protocol, self.rubric, self.response, "trait",
                             self.stage, pacer, transport=transport)
        self.assertEqual(len(calls), 3)

    def test_new_malformed_or_refusal_failure_is_not_retried(self):
        for body in ({"choices": [{"finish_reason": "stop", "message": {"content": "Score:75"}}]},
                     {"choices": [{"finish_reason": "stop", "message": {"content": "75", "refusal": "blocked"}}]}):
            with tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                initialize_run(root, INPUTS, "test-malformed")
                stage = recovery.freeze_recovery(root, INPUTS)
                path = judge_path(root, "paranoia", self.row["response_id"], "trait")
                atomic_json(path, {**self.identity, "score": None, "attempts": failed_attempts()})
                calls = []
                result = recovery.recover_one(root, path, self.protocol, self.rubric, self.response, "trait",
                        stage, recovery.Pacer(sleep=lambda _: None), transport=lambda payload: (calls.append(payload) or body, {}))
                self.assertIsNone(result["score"])
                self.assertEqual(len(calls), 1)
                recovery.recover_one(root, path, self.protocol, self.rubric, self.response, "trait",
                        stage, recovery.Pacer(sleep=lambda _: None), transport=lambda _: self.fail("terminal parse failure"))

    def test_unknown_interrupted_request_blocks_silent_reroll(self):
        self.save()
        def transport(_):
            raise KeyboardInterrupt()
        with self.assertRaises(KeyboardInterrupt):
            recovery.recover_one(self.root, self.path, self.protocol, self.rubric, self.response, "trait",
                                 self.stage, recovery.Pacer(), transport=transport)
        with self.assertRaisesRegex(RuntimeError, "Unknown interrupted"):
            recovery.recover_one(self.root, self.path, self.protocol, self.rubric, self.response, "trait",
                                 self.stage, recovery.Pacer(), transport=lambda _: self.fail("no reroll"))

    def test_global_pacing_waits_half_second_after_completion(self):
        clock = [0.0]
        sleeps = []
        def sleep(seconds):
            sleeps.append(seconds)
            clock[0] += seconds
        pacer = recovery.Pacer(clock=lambda: clock[0], sleep=sleep)
        pacer.wait()
        self.assertEqual(sleeps, [])
        pacer.finish()
        pacer.wait()
        self.assertEqual(sleeps, [0.5])
        pacer.finish()
        clock[0] += 0.2
        pacer.wait()
        self.assertAlmostEqual(sleeps[-1], 0.3)

    def test_snapshot_tampering_fails_closed(self):
        path = self.root / "implementations" / f"{recovery.STAGE}.py"
        path.write_text("changed")
        with self.assertRaisesRegex(ValueError, "snapshot changed"):
            recovery.freeze_recovery(self.root, INPUTS)

    def test_full_all_trait_audit_and_public_metadata_preserve_valid_journals(self):
        target = self.path
        valid_path = None
        coherence = (INPUTS / "coherence_evaluation_prompt.txt").read_text()
        for trait in TRAITS:
            for row in conditions(trait, self.traits[trait], self.protocol):
                response = {**row, "status": "complete", "response": "Explicit test-only fixture answer."}
                atomic_json(record_path(self.root, trait, row["response_id"]), response)
                for kind, rubric in (("trait", self.traits[trait]["evaluation_prompt"]), ("coherence", coherence)):
                    identity, _ = recovery.recovery_identity(self.protocol, rubric, response, kind)
                    path = judge_path(self.root, trait, row["response_id"], kind)
                    attempts = failed_attempts() if path == target else [{"attempt": 1, "status": "complete", "score": 50}]
                    atomic_json(path, {**identity, "score": None if path == target else 50, "attempts": attempts})
                    if valid_path is None and path != target:
                        valid_path = path
        before = recovery.audit_judges(self.root, INPUTS)
        self.assertEqual((before["present"], before["valid_scores"], before["eligible_http429_none"]), (4800, 4799, 1))
        self.assertEqual(before["by_trait"]["paranoia"]["eligible_http429_none"], 1)
        original = valid_path.read_bytes()
        outcome = recovery.recover_judges(self.root, INPUTS, transport=lambda _: (successful_body(), {}))
        self.assertEqual(outcome["after"]["valid_scores"], 4800)
        self.assertEqual(valid_path.read_bytes(), original)
        public, sources = recovery.public_recovery_provenance(self.root, INPUTS)
        self.assertEqual(public["counts"]["records_touched"], 1)
        self.assertEqual(public["counts"]["additional_attempts"], 1)
        self.assertEqual(public["counts"]["recovered_scores"], 1)
        self.assertEqual(sources[recovery.POLICY_SOURCE]["sha256"], self.stage["policy_sha256"])
        self.assertNotIn("raw_response", json.dumps(public))
        changed = json.loads(valid_path.read_text())
        changed["extra"] = "changed"
        atomic_json(valid_path, changed)
        with self.assertRaisesRegex(ValueError, "original score journal was changed"):
            recovery.public_recovery_provenance(self.root, INPUTS)


if __name__ == "__main__":
    unittest.main()
