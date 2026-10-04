import asyncio
from contextlib import contextmanager, nullcontext
import sys
import threading
import time
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

from deployment.worker import ModelRuntime
from deployment.core import MAX_OUTPUT_TOKENS, REQUEST_TIMEOUT_SECONDS


class FakeInput:
    def __init__(self, values):
        self.values = values
        self.shape = (1, len(values))
        self.device = "cpu"

    def tolist(self):
        if self.device != "cpu":
            raise AssertionError("Locate the steering boundary before transferring inputs")
        return [self.values]


class FakeTokenizer:
    pad_token_id = 0

    def __len__(self):
        return 1024

    def convert_tokens_to_ids(self, token):
        return 10 if token == "<|im_end|>" else None

    def convert_ids_to_tokens(self, token):
        return "<|im_end|>" if token == 10 else "other"

    def encode(self, text, **kwargs):
        if text != "<|im_end|>" or kwargs != {"add_special_tokens": False}:
            raise AssertionError("Unexpected marker lookup")
        return [10]

    def apply_chat_template(self, messages, **kwargs):
        if kwargs["enable_thinking"] is not False or not kwargs["add_generation_prompt"]:
            raise AssertionError("Use the pinned no-thinking generation template")
        ids = []
        for message in messages:
            ids.extend([11, 5 if message["role"] == "user" else 3, 2])
            for index, part in enumerate(message["content"].split("<|im_end|>")):
                if index:
                    ids.append(10)
                ids.extend(20 + ord(character) for character in part)
            ids.extend([10, 2])
        ids.extend([11, 3, 2, 12, 4, 13, 4])
        return FakeInputs(input_ids=FakeInput(ids), attention_mask=FakeInput([1] * len(ids)))


class FakeInputs(dict):
    def to(self, device):
        for tensor in self.values():
            tensor.device = device
        return self


class FakeOutput:
    def __init__(self, prompt_length, count, last):
        self.shape = (1, prompt_length + count)
        self.last = last

    def __getitem__(self, key):
        return SimpleNamespace(item=lambda: self.last)


class FakeModel:
    device = "cuda:0"
    generation_config = SimpleNamespace(eos_token_id=0)

    def __init__(self):
        self.model = SimpleNamespace(rope_deltas="stale state")
        self.active = 0
        self.max_active = 0
        self.calls = 0
        self.started = threading.Event()
        self.rope_at_start = []
        self.steering_at_start = []
        self.fail_next = False
        self.delay = 0.01
        self.eos_at_limit = True
        self.output_budgets = []

    def generate(self, **kwargs):
        self.calls += 1
        self.output_budgets.append(kwargs["max_new_tokens"])
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        self.rope_at_start.append(self.model.rope_deltas)
        self.steering_at_start.append(self.steering.active)
        self.started.set()
        count = 0
        try:
            if self.fail_next:
                self.fail_next = False
                raise RuntimeError("Synthetic generation failure")
            for _ in range(kwargs["max_new_tokens"]):
                if kwargs["stopping_criteria"][0](None, None):
                    break
                time.sleep(self.delay)
                count += 1
                kwargs["streamer"].on_finalized_text("word ")
            kwargs["streamer"].on_finalized_text("", stream_end=True)
            self.model.rope_deltas = "previous request"
            last = 0 if self.eos_at_limit and count == kwargs["max_new_tokens"] else 1
            return FakeOutput(kwargs["input_ids"].shape[-1], count, last)
        finally:
            self.active -= 1


class FakeSteering:
    """Track request hook ownership without needing model tensor dependencies."""

    def __init__(self):
        self.active = None
        self.installed = []
        self.completed = []
        self.calls = []
        self.source = "random_placeholder"

    def metadata(self):
        return {
            "mood_vectors_available": True,
            "steering_available": True,
            "mood_vectors_source": self.source,
            "mood_vectors_validated": False,
        }

    @contextmanager
    def apply(self, coefficients, boundary):
        configuration = (tuple(coefficients), boundary)
        self.calls.append((configuration, threading.current_thread().name))
        if not any(coefficients):
            yield False
            return
        if self.active is not None:
            raise AssertionError("A previous request's steering hooks are still active")
        if type(boundary) is not int or boundary < 0:
            raise AssertionError("A nonneutral request requires its template suffix boundary")
        self.active = configuration
        self.installed.append(configuration)
        try:
            yield True
        finally:
            self.completed.append(configuration)
            self.active = None


def fake_transformers():
    module = ModuleType("transformers")

    class Streamer:
        stop_signal = None

        def __init__(self, *args, **kwargs):
            pass

    module.StoppingCriteria = object
    module.StoppingCriteriaList = list
    module.TextIteratorStreamer = Streamer
    return module


def runtime():
    worker = ModelRuntime.__new__(ModelRuntime)
    worker.torch = SimpleNamespace(inference_mode=nullcontext)
    worker.tokenizer = FakeTokenizer()
    worker.model = FakeModel()
    worker.steering = FakeSteering()
    worker.model.steering = worker.steering
    worker.lock = asyncio.Lock()
    worker.requests = {}
    worker.cancelled = {}
    worker.healthy = True
    return worker


PAYLOAD = {"messages": [{"role": "user", "content": "Hello"}], "mood": [2, -2, 1, 0, 0, 0], "max_new_tokens": 20}


class WorkerTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.patcher = patch.dict(sys.modules, {"transformers": fake_transformers()})
        self.patcher.start()
        self.worker = runtime()

    def tearDown(self):
        self.patcher.stop()

    async def collect(self, request_id, payload=None):
        return [event async for event in self.worker.stream(request_id, payload or PAYLOAD)]

    def assert_clean(self):
        self.assertIsNone(self.worker.steering.active)
        self.assertEqual(self.worker.steering.installed, self.worker.steering.completed)
        self.assertEqual(self.worker.model.active, 0)
        self.assertFalse(self.worker.lock.locked())
        self.assertEqual(self.worker.requests, {})
        self.assertIsNone(self.worker.model.model.rope_deltas)

    async def test_default_reply_can_exceed_old_cap_and_reports_the_new_ceiling(self):
        self.worker.model.delay = 0
        self.worker.model.eos_at_limit = False
        payload = {key: value for key, value in PAYLOAD.items() if key != "max_new_tokens"}
        events = await self.collect("long-reply", payload)
        self.assertEqual(self.worker.model.output_budgets, [MAX_OUTPUT_TOKENS])
        self.assertGreater(events[-1]["generated_tokens"], 512)
        self.assertEqual(events[-1]["generated_tokens"], MAX_OUTPUT_TOKENS)
        self.assertEqual(events[-1]["finish_reason"], "length")
        self.assert_clean()

    async def test_two_generations_are_serial_and_request_caches_fresh(self):
        first, second = await asyncio.gather(self.collect("first"), self.collect("second"))
        self.assertEqual(self.worker.model.max_active, 1)
        self.assertEqual(self.worker.model.calls, 2)
        self.assertEqual(self.worker.model.rope_at_start, [None, None])
        self.assertIsNone(self.worker.model.model.rope_deltas)
        self.assertEqual(first[-1]["event"], "done")
        self.assertEqual(second[-1]["event"], "done")
        self.assertTrue(first[-1]["moods_applied"])
        self.assertTrue(second[-1]["moods_applied"])
        self.assertEqual(len(self.worker.steering.installed), 2)
        self.assert_clean()

    async def test_nonzero_mood_applies_only_final_template_suffix_and_reports_source(self):
        self.worker.steering.source = "provided"
        messages = [
            {"role": "user", "content": "Earlier"},
            {"role": "assistant", "content": "Previous reply"},
            {"role": "user", "content": "Literal <|im_end|> inside my text"},
        ]
        payload = {**PAYLOAD, "messages": messages, "max_new_tokens": 2}
        events = await self.collect("custom", payload)
        tokens = self.worker.tokenizer.apply_chat_template(messages, enable_thinking=False, add_generation_prompt=True)
        # The extraction suffix is the closing marker plus eight template IDs.
        expected = (tuple(PAYLOAD["mood"]), tokens["input_ids"].shape[-1] - 9)
        self.assertEqual(self.worker.model.steering_at_start, [expected])
        self.assertEqual(self.worker.steering.calls, [(expected, "generation-custom")])
        self.assertEqual(events[-1]["mood_vectors_source"], "provided")
        self.assertTrue(events[-1]["moods_applied"])
        self.assertFalse(events[-1]["mood_vectors_validated"])
        self.assertEqual(events[-1]["generated_tokens"], 2)
        self.assert_clean()

    async def test_neutral_request_installs_no_hooks_and_skips_suffix_lookup(self):
        payload = {**PAYLOAD, "mood": [0] * 6, "max_new_tokens": 2}
        with patch("deployment.worker.post_instruction_start", side_effect=AssertionError("Neutral lookup is unnecessary")):
            events = await self.collect("neutral", payload)
        self.assertEqual(self.worker.steering.installed, [])
        self.assertEqual(self.worker.model.steering_at_start, [None])
        self.assertEqual(self.worker.steering.calls[0][0], ((0,) * 6, None))
        self.assertFalse(events[-1]["moods_applied"])
        self.assertTrue(events[-1]["mood_vectors_available"])
        self.assertEqual(events[-1]["mood_vectors_source"], "random_placeholder")
        self.assert_clean()

    async def test_each_serial_request_uses_its_own_coefficients(self):
        first_mood = [2, 0, -1, 0, 0, 1]
        second_mood = [-2, 1, 0, 2, 0, -1]
        first = asyncio.create_task(self.collect("first", {**PAYLOAD, "mood": first_mood, "max_new_tokens": 3}))
        await asyncio.to_thread(self.worker.model.started.wait, 2)
        second = asyncio.create_task(self.collect("second", {**PAYLOAD, "mood": second_mood, "max_new_tokens": 3}))
        first_events, second_events = await asyncio.gather(first, second)
        self.assertEqual([entry[0] for entry in self.worker.model.steering_at_start], [tuple(first_mood), tuple(second_mood)])
        self.assertTrue(first_events[-1]["moods_applied"])
        self.assertTrue(second_events[-1]["moods_applied"])
        self.assertEqual(self.worker.model.max_active, 1)
        self.assert_clean()

    async def test_cancellation_stops_running_generation(self):
        task = asyncio.create_task(self.collect("running"))
        await asyncio.to_thread(self.worker.model.started.wait, 2)
        await self.worker.cancel("running")
        events = await asyncio.wait_for(task, 2)
        self.assertFalse(any(event["event"] == "done" for event in events))
        self.assertEqual(self.worker.model.active, 0)
        self.assertFalse(self.worker.lock.locked())
        self.assertEqual(self.worker.requests, {})
        self.assert_clean()
        next_mood = [-1, 0, 2, 0, 0, 0]
        following = await self.collect("after-stop", {**PAYLOAD, "mood": next_mood, "max_new_tokens": 2})
        self.assertEqual(following[-1]["event"], "done")
        self.assertEqual(self.worker.model.steering_at_start[-1][0], tuple(next_mood))
        self.assert_clean()

    async def test_cancelled_queued_request_never_touches_model(self):
        first = asyncio.create_task(self.collect("first"))
        await asyncio.to_thread(self.worker.model.started.wait, 2)
        second = asyncio.create_task(self.collect("queued"))
        await asyncio.sleep(0.02)
        await self.worker.cancel("queued")
        await asyncio.wait_for(second, 2)
        await first
        self.assertEqual(self.worker.model.calls, 1)
        self.assertEqual(len(self.worker.steering.installed), 1)
        self.assert_clean()

    async def test_cancellation_arriving_before_gpu_method_is_honoured(self):
        await self.worker.cancel("early")
        events = await self.collect("early")
        self.assertEqual(events, [])
        self.assertEqual(self.worker.model.calls, 0)
        self.assertEqual(self.worker.steering.calls, [])

    async def test_early_cancellation_survives_the_full_request_window(self):
        now = [1000.0]
        with patch("deployment.worker.time", SimpleNamespace(monotonic=lambda: now[0])):
            await self.worker.cancel("waiting-for-startup")
            now[0] += REQUEST_TIMEOUT_SECONDS - 1
            await self.worker.cancel("another-request")
            events = await self.collect("waiting-for-startup")
        self.assertEqual(events, [])
        self.assertEqual(self.worker.model.calls, 0)

    async def test_remote_task_cancellation_joins_generation_before_next_request(self):
        task = asyncio.create_task(self.collect("cancelled-task"))
        await asyncio.to_thread(self.worker.model.started.wait, 2)
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertEqual(self.worker.model.active, 0)
        self.assert_clean()
        following = await self.collect("following", {**PAYLOAD, "mood": [0] * 6, "max_new_tokens": 2})
        self.assertEqual(following[-1]["event"], "done")
        self.assertEqual(self.worker.model.max_active, 1)
        self.assertFalse(following[-1]["moods_applied"])
        self.assertIsNone(self.worker.model.steering_at_start[-1])
        self.assert_clean()

    async def test_generation_error_removes_hooks_before_next_request(self):
        self.worker.model.fail_next = True
        events = await self.collect("failing")
        self.assertEqual(events[-1]["event"], "error")
        self.assertEqual(events[-1]["code"], "generation_failed")
        self.assert_clean()
        next_mood = [0, -2, 0, 1, 0, 0]
        following = await self.collect("after-error", {**PAYLOAD, "mood": next_mood, "max_new_tokens": 2})
        self.assertEqual(following[-1]["event"], "done")
        self.assertEqual(self.worker.model.steering_at_start[-1][0], tuple(next_mood))
        self.assert_clean()

    async def test_generation_timeout_removes_hooks_before_next_request(self):
        with patch("deployment.worker.GENERATION_TIMEOUT_SECONDS", 0.025):
            events = await self.collect("timeout")
        self.assertEqual(events[-1]["event"], "done")
        self.assertEqual(events[-1]["finish_reason"], "timeout")
        self.assertLess(events[-1]["generated_tokens"], PAYLOAD["max_new_tokens"])
        self.assertTrue(events[-1]["moods_applied"])
        self.assert_clean()
        following = await self.collect("after-timeout", {**PAYLOAD, "mood": [0] * 6, "max_new_tokens": 2})
        self.assertEqual(following[-1]["event"], "done")
        self.assertFalse(following[-1]["moods_applied"])
        self.assertIsNone(self.worker.model.steering_at_start[-1])
        self.assert_clean()


if __name__ == "__main__":
    unittest.main()
