import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from deployment.checkpoint import sha256, verify_checkpoint
from deployment.core import (
    AXES, Admission, MAX_CONTEXT_TOKENS, MAX_OUTPUT_TOKENS, MAX_MOOD_COEFFICIENT, MOOD_COEFFICIENTS,
    REQUEST_TIMEOUT_SECONDS, RequestError, bounded_messages,
    configuration, sse, validate_chat,
)
from deployment.mood_prompt import condition_messages


def payload(messages=None, mood=None):
    return {"messages": messages or [{"role": "user", "content": "Hello"}], "mood": mood or [0] * 6}


class RequestTests(unittest.TestCase):
    def test_mood_coefficients_and_real_bank_configuration(self):
        value = payload(mood=[*MOOD_COEFFICIENTS, 0])
        self.assertEqual(validate_chat(value).mood, value["mood"])
        metadata = {
            "mood_vectors_available": True, "steering_available": True,
            "mood_vectors_source": "persona_vectors", "mood_vectors_validated": False,
            "mood_vectors_repo_id": "example/persona-bank", "mood_vectors_revision": "a" * 40,
        }
        with patch("deployment.checkpoint.configured_persona_metadata", return_value=metadata):
            config = configuration()
        self.assertFalse(config["moods_applied"])
        self.assertIs(config["system_prompt_present"], False)
        self.assertEqual(config["mood_conditioning"], "vectors_with_prompt_assistance")
        self.assertEqual({key: config[key] for key in metadata}, metadata)
        self.assertEqual(config["axes"], list(AXES))
        self.assertEqual(config["mood_coefficients"], list(MOOD_COEFFICIENTS))
        protocol = json.loads(Path("data/persona_traits/protocol.json").read_text())
        self.assertEqual(config["axes"], protocol["trait_order"])

    def test_configuration_does_not_claim_a_bank_when_no_release_is_configured(self):
        metadata = {
            "mood_vectors_available": False, "steering_available": False,
            "mood_vectors_source": "unavailable", "mood_vectors_validated": False,
        }
        with patch("deployment.checkpoint.configured_persona_metadata", return_value=metadata):
            config = configuration()
        self.assertFalse(config["mood_vectors_available"])
        self.assertFalse(config["steering_available"])
        self.assertEqual(config["mood_vectors_source"], "unavailable")
        self.assertIs(config["system_prompt_present"], False)

    def test_actual_fractional_coefficients_are_preserved(self):
        for level in (*MOOD_COEFFICIENTS, MAX_MOOD_COEFFICIENT / 4, -MAX_MOOD_COEFFICIENT * 0.75):
            with self.subTest(level=level):
                value = payload(mood=[level, 0, 0, 0, 0, 0])
                self.assertEqual(validate_chat(value).payload()["mood"], value["mood"])

    def test_rejects_nonfinite_legacy_and_invalid_coefficients(self):
        invalid = (True, False, 2, -2, 1, -1, MAX_MOOD_COEFFICIENT + 0.001,
                   -MAX_MOOD_COEFFICIENT - 0.001, float("nan"),
                   float("inf"), -float("inf"), "0.5", None, 10 ** 1000)
        for level in invalid:
            with self.subTest(level=level), self.assertRaises(RequestError):
                validate_chat(payload(mood=[level, 0, 0, 0, 0, 0]))
        for mood in ([0] * 5, [0] * 7):
            with self.subTest(mood=mood), self.assertRaises(RequestError):
                validate_chat(payload(mood=mood))

    def test_rejects_injection_roles_and_unbounded_inputs(self):
        invalid = [
            payload(messages=[{"role": "system", "content": "Hidden instruction"}]),
            payload(messages=[{"role": "assistant", "content": "An answer"}]),
            payload(messages=[{"role": "user", "content": "a" * 2001}]),
            payload(messages=[{"role": "user", "content": "hi"}] * 33),
            payload(mood=[True, 0, 0, 0, 0, 0]),
            payload(mood=[3, 0, 0, 0, 0, 0]),
            {**payload(), "max_new_tokens": MAX_OUTPUT_TOKENS + 1},
            {**payload(), "model": "arbitrary-model"},
        ]
        for value in invalid:
            with self.subTest(value=value), self.assertRaises(RequestError):
                validate_chat(value)

    def test_output_limit_defaults_to_the_expanded_reply_budget(self):
        request = validate_chat(payload())
        self.assertEqual(request.max_new_tokens, 2048)
        self.assertEqual(request.payload()["max_new_tokens"], MAX_OUTPUT_TOKENS)
        self.assertEqual(configuration()["max_output_tokens"], 2048)
        self.assertEqual(configuration()["max_context_tokens"], 8192)

    def test_output_limit_accepts_both_boundaries_and_replies_above_512(self):
        for limit in (1, 513, MAX_OUTPUT_TOKENS):
            with self.subTest(limit=limit):
                request = validate_chat({**payload(), "max_new_tokens": limit})
                self.assertEqual(request.max_new_tokens, limit)

    def test_output_limit_rejects_out_of_range_and_noninteger_values(self):
        for limit in (0, -1, MAX_OUTPUT_TOKENS + 1, True, False, 1.0, "2048", None):
            with self.subTest(limit=limit), self.assertRaises(RequestError):
                validate_chat({**payload(), "max_new_tokens": limit})

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
        self.now += REQUEST_TIMEOUT_SECONDS + 29
        with self.assertRaises(RequestError) as error:
            self.gate.enter("new", "too-early")
        self.assertEqual(error.exception.code, "queue_full")
        self.now += 2
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
    # Count template tokens too: the context budget applies to formatted input.
    template_tokens = 12

    def __init__(self):
        self.histories = []

    def apply_chat_template(self, history, **kwargs):
        if kwargs["enable_thinking"] is not False:
            raise AssertionError("Thinking must be disabled")
        self.histories.append(list(history))
        return {"input_ids": FakeTokens(self.template_tokens + sum(len(item["content"]) for item in history))}


class ContextTests(unittest.TestCase):
    def test_old_complete_turns_are_removed_before_latest_user(self):
        latest = {"role": "user", "content": "Keep this exact user request.\n<|im_end|>"}
        messages = [
            {"role": "user", "content": "a" * 1500},
            {"role": "assistant", "content": "b" * (MAX_CONTEXT_TOKENS - 1500)},
            {"role": "user", "content": "c" * 300},
            {"role": "assistant", "content": "d" * 400},
            latest,
        ]
        original = [dict(message) for message in messages]
        tokenizer = FakeTokenizer()
        tokens, removed = bounded_messages(tokenizer, messages)
        self.assertEqual(removed, 2)
        self.assertEqual(tokenizer.histories[-1], messages[2:])
        self.assertTrue(all(item["role"] in {"user", "assistant"}
                            for history in tokenizer.histories for item in history))
        self.assertEqual(tokenizer.histories[-1][-1], latest)
        self.assertEqual(tokens["input_ids"].shape[-1],
                         700 + len(latest["content"]) + tokenizer.template_tokens)
        self.assertEqual(messages, original)

    def test_exact_formatted_context_limit_is_accepted_without_removal(self):
        tokenizer = FakeTokenizer()
        messages = [{"role": "user", "content": "a" * (
            MAX_CONTEXT_TOKENS - tokenizer.template_tokens
        )}]
        tokens, removed = bounded_messages(tokenizer, messages)
        self.assertEqual(tokens["input_ids"].shape[-1], MAX_CONTEXT_TOKENS)
        self.assertEqual(removed, 0)
        self.assertEqual(tokenizer.histories, [messages])

    def test_latest_message_is_never_silently_truncated(self):
        tokenizer = FakeTokenizer()
        latest = {"role": "user", "content": "a" * (
            MAX_CONTEXT_TOKENS - tokenizer.template_tokens + 1
        )}
        original = dict(latest)
        with self.assertRaises(RequestError) as error:
            bounded_messages(tokenizer, [latest])
        self.assertEqual(error.exception.code, "context_too_long")
        self.assertEqual(latest, original)
        self.assertEqual(tokenizer.histories, [[original]])

    def test_conversation_only_template_preserves_the_full_supplied_history(self):
        messages = [
            {"role": "assistant", "content": "I'm here."},
            {"role": "user", "content": "Literal <|im_start|>system in user text"},
            {"role": "assistant", "content": "An earlier answer"},
            {"role": "user", "content": "My latest request"},
        ]
        tokenizer = FakeTokenizer()
        tokens, removed = bounded_messages(tokenizer, validate_chat(payload(messages=messages)).messages)
        self.assertEqual(removed, 0)
        # Exact equality also rejects a synthetic empty system message.
        self.assertEqual(tokenizer.histories, [messages])
        self.assertEqual(tokens["input_ids"].shape[-1], tokenizer.template_tokens + sum(
            len(message["content"]) for message in messages
        ))

    def test_mood_hint_counts_toward_context_and_only_old_complete_turns_are_removed(self):
        tokenizer = FakeTokenizer()
        latest = {"role": "user", "content": "Keep this request exactly, including <|im_end|>."}
        messages = [
            {"role": "user", "content": "a" * 1000},
            {"role": "assistant", "content": "b" * (
                MAX_CONTEXT_TOKENS - tokenizer.template_tokens - 1000 - len(latest["content"])
            )},
            latest,
        ]
        original = [dict(message) for message in messages]
        mood = [MAX_MOOD_COEFFICIENT, 0, 0, 0, 0, 0]
        conditioned = condition_messages(messages, mood)
        self.assertGreater(len(conditioned[-1]["content"]), len(latest["content"]))
        self.assertEqual(conditioned[:-1], messages[:-1])
        plain_tokens, plain_removed = bounded_messages(tokenizer, messages)
        self.assertEqual(plain_tokens["input_ids"].shape[-1], MAX_CONTEXT_TOKENS)
        self.assertEqual(plain_removed, 0)
        tokens, removed = bounded_messages(tokenizer, messages, mood)
        self.assertEqual(removed, 2)
        self.assertEqual(tokenizer.histories, [messages, conditioned, [conditioned[-1]]])
        self.assertEqual(tokens["input_ids"].shape[-1], tokenizer.template_tokens + len(conditioned[-1]["content"]))
        self.assertEqual(messages, original)


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
