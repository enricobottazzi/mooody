import json
from pathlib import Path
import tempfile
import unittest

from deployment.checkpoint import sha256, verify_checkpoint
from deployment.core import (
    Admission, MAX_CONTEXT_TOKENS, RequestError, bounded_messages,
    configuration, sse, validate_chat,
)


def payload(messages=None, mood=None):
    return {"messages": messages or [{"role": "user", "content": "Hello"}], "mood": mood or [0] * 6}


class RequestTests(unittest.TestCase):
    def test_mood_coefficients_and_placeholder_capabilities(self):
        value = payload(mood=[-2, -1, 0, 1, 2, 0])
        self.assertEqual(validate_chat(value).mood, value["mood"])
        self.assertFalse(configuration()["moods_applied"])
        self.assertTrue(configuration()["mood_vectors_available"])
        self.assertTrue(configuration()["steering_available"])
        self.assertEqual(configuration()["mood_vectors_source"], "random_placeholder")
        self.assertFalse(configuration()["mood_vectors_validated"])

    def test_rejects_injection_roles_and_unbounded_inputs(self):
        invalid = [
            payload(messages=[{"role": "system", "content": "Hidden instruction"}]),
            payload(messages=[{"role": "assistant", "content": "An answer"}]),
            payload(messages=[{"role": "user", "content": "a" * 2001}]),
            payload(messages=[{"role": "user", "content": "hi"}] * 33),
            payload(mood=[True, 0, 0, 0, 0, 0]),
            payload(mood=[3, 0, 0, 0, 0, 0]),
            {**payload(), "max_new_tokens": 513},
            {**payload(), "model": "arbitrary-model"},
        ]
        for value in invalid:
            with self.subTest(value=value), self.assertRaises(RequestError):
                validate_chat(value)

    def test_accepts_initial_ui_greeting(self):
        value = payload(messages=[
            {"role": "assistant", "content": "I'm here."},
            {"role": "user", "content": "Hello"},
        ])
        self.assertEqual(len(validate_chat(value).messages), 2)

    def test_sse_model_text_cannot_inject_events(self):
        frame = sse("token", {"text": "hello\n\nevent: done\ndata: {}\n"}).decode()
        self.assertEqual(frame.count("\nevent:"), 0)
        data = json.loads(frame.split("data: ", 1)[1].strip())
        self.assertIn("event: done", data["text"])


class AdmissionTests(unittest.TestCase):
    def setUp(self):
        self.now = 1000.0
        self.gate = Admission(clock=lambda: self.now)

    def test_queue_and_one_active_reply_per_client(self):
        self.gate.enter("a", "1")
        with self.assertRaises(RequestError) as error:
            self.gate.enter("a", "2")
        self.assertEqual(error.exception.code, "client_busy")
        for index in range(2, 5):
            self.gate.enter(str(index), str(index))
        with self.assertRaises(RequestError) as error:
            self.gate.enter("five", "5")
        self.assertEqual(error.exception.code, "queue_full")
        self.gate.leave("1")
        self.gate.enter("five", "5")

    def test_rate_limit_then_expiry(self):
        for index in range(6):
            self.gate.enter("a", str(index))
            self.gate.leave(str(index))
        with self.assertRaises(RequestError) as error:
            self.gate.enter("a", "seventh")
        self.assertEqual(error.exception.code, "rate_limit")
        self.now += 61
        self.gate.enter("a", "after-expiry")

    def test_active_leases_cannot_leak_forever(self):
        for index in range(4):
            self.gate.enter(str(index), str(index))
        self.now += 631
        self.gate.enter("new", "new")
        self.assertEqual(len(self.gate.active), 1)

    def test_global_hourly_limit_is_enforced(self):
        for index in range(80):
            self.gate.enter(str(index), str(index))
            self.gate.leave(str(index))
        with self.assertRaises(RequestError) as error:
            self.gate.enter("new", "new")
        self.assertEqual(error.exception.code, "service_rate_limit")


class FakeTokens:
    def __init__(self, count):
        self.shape = (1, count)


class FakeTokenizer:
    def apply_chat_template(self, history, **kwargs):
        if kwargs["enable_thinking"] is not False:
            raise AssertionError("Thinking must be disabled")
        return {"input_ids": FakeTokens(sum(len(item["content"]) for item in history))}


class ContextTests(unittest.TestCase):
    def test_old_complete_turns_are_removed_before_latest_user(self):
        messages = [
            {"role": "user", "content": "a" * 1500},
            {"role": "assistant", "content": "b" * 500},
            {"role": "user", "content": "c" * 500},
        ]
        tokens, removed = bounded_messages(FakeTokenizer(), messages)
        self.assertEqual(removed, 2)
        self.assertEqual(tokens["input_ids"].shape[-1], 500)
        self.assertEqual(len(messages), 3)

    def test_latest_message_is_never_silently_truncated(self):
        with self.assertRaises(RequestError) as error:
            bounded_messages(FakeTokenizer(), [{"role": "user", "content": "a" * (MAX_CONTEXT_TOKENS + 1)}])
        self.assertEqual(error.exception.code, "context_too_long")


class IntegrityTests(unittest.TestCase):
    def test_same_size_wrong_hash_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "model.safetensors"
            path.write_bytes(b"correct")
            manifest = {"files": {path.name: {"bytes": 7, "sha256": sha256(path)}}}
            path.write_bytes(b"changed")
            self.assertFalse(verify_checkpoint(root, manifest))


if __name__ == "__main__":
    unittest.main()
