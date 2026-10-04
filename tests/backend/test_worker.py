import asyncio
from contextlib import contextmanager, nullcontext
import sys
import threading
import time
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

from deployment.worker import ModelRuntime, native_smoke_replies_valid
from deployment.core import MAX_OUTPUT_TOKENS, REQUEST_TIMEOUT_SECONDS
from deployment.mood_prompt import condition_messages


class NativeSmokeReplyTests(unittest.TestCase):
    def test_accepts_correct_numeric_or_spelled_answer(self):
        for answer in ("4.", "Four.", "Two plus two is four. Cheers."):
            self.assertTrue(native_smoke_replies_valid(["Ready.\n", answer]))

    def test_rejects_wrong_answers_and_unexpected_thinking(self):
        for replies in (["Ready.", "14"], ["Ready.", "Fourteen."],
                        ["Already.", "4"], ["Ready.", "<think>4</think>"]):
            self.assertFalse(native_smoke_replies_valid(replies))


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

    def __init__(self):
        self.histories = []

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
        self.histories.append([dict(message) for message in messages])
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
        self.prefill_release = None
        self.prefill_waiting = threading.Event()
        self.generation_error = None

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
            if self.prefill_release is not None:
                self.prefill_waiting.set()
                if not self.prefill_release.wait(5):
                    raise RuntimeError("Test did not release the blocked native prefill")
            if self.generation_error is not None:
                raise self.generation_error
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
        self.source = "persona_vectors"

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
    worker._retirements = set()
    return worker


PAYLOAD = {"messages": [{"role": "user", "content": "Hello"}], "mood": [0.25, -0.25, 0.125, 0, 0, 0], "max_new_tokens": 20}


class WorkerTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.patcher = patch.dict(sys.modules, {"transformers": fake_transformers()})
        self.patcher.start()
        self.worker = runtime()

    def tearDown(self):
        self.patcher.stop()

    async def collect(self, request_id, payload=None):
        return [event async for event in self.worker.stream(request_id, payload or PAYLOAD)]

    async def wait_for_retirement(self):
        tasks = tuple(self.worker._retirements)
        if tasks:
            await asyncio.wait_for(asyncio.gather(*(asyncio.shield(task) for task in tasks)), 2)

    async def blocked_generation(self, request_id):
        release = threading.Event()
        self.worker.model.prefill_release = release
        task = asyncio.create_task(self.collect(request_id))
        self.assertTrue(await asyncio.to_thread(self.worker.model.prefill_waiting.wait, 2))
        await self.worker.cancel(request_id)
        return task, release

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

    async def test_nonzero_mood_starts_at_final_formatted_prompt_token_and_reports_source(self):
        messages = [
            {"role": "user", "content": "Earlier"},
            {"role": "assistant", "content": "Previous reply"},
            {"role": "user", "content": "Literal <|im_end|> inside my text"},
        ]
        payload = {**PAYLOAD, "messages": messages, "max_new_tokens": 2}
        events = await self.collect("custom", payload)
        conditioned = condition_messages(messages, payload["mood"])
        tokens = self.worker.tokenizer.apply_chat_template(
            conditioned,
            enable_thinking=False, add_generation_prompt=True,
        )
        self.assertEqual(self.worker.tokenizer.histories[0], conditioned)
        self.assertEqual(self.worker.tokenizer.histories[0][:-1], messages[:-1])
        self.assertGreater(len(conditioned[-1]["content"]), len(messages[-1]["content"]))
        # Only the last formatted prompt position predicts the first response.
        expected = (tuple(PAYLOAD["mood"]), tokens["input_ids"].shape[-1] - 1)
        self.assertEqual(self.worker.model.steering_at_start, [expected])
        self.assertEqual(self.worker.steering.calls, [(expected, "generation-custom")])
        self.assertEqual(events[-1]["mood_vectors_source"], "persona_vectors")
        self.assertEqual(events[-1]["mood_coefficients_applied"], payload["mood"])
        self.assertIs(events[-1]["system_prompt_present"], False)
        self.assertEqual(events[-1]["mood_conditioning"], "vectors_with_prompt_assistance")
        self.assertIs(events[-1]["mood_prompt_assistance_applied"], True)
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
        self.assertEqual(events[-1]["mood_coefficients_applied"], [0] * 6)
        self.assertIs(events[-1]["system_prompt_present"], False)
        self.assertIs(events[-1]["mood_prompt_assistance_applied"], False)
        self.assertEqual(self.worker.tokenizer.histories, [payload["messages"]])
        self.assertTrue(events[-1]["mood_vectors_available"])
        self.assertEqual(events[-1]["mood_vectors_source"], "persona_vectors")
        self.assert_clean()

    async def test_each_serial_request_uses_its_own_coefficients(self):
        first_mood = [0.25, 0, -0.125, 0, 0, 0.125]
        second_mood = [-0.25, 0.125, 0, 0.25, 0, -0.125]
        first = asyncio.create_task(self.collect("first", {**PAYLOAD, "mood": first_mood, "max_new_tokens": 3}))
        await asyncio.to_thread(self.worker.model.started.wait, 2)
        second = asyncio.create_task(self.collect("second", {**PAYLOAD, "mood": second_mood, "max_new_tokens": 3}))
        first_events, second_events = await asyncio.gather(first, second)
        self.assertEqual([entry[0] for entry in self.worker.model.steering_at_start], [tuple(first_mood), tuple(second_mood)])
        self.assertTrue(first_events[-1]["moods_applied"])
        self.assertTrue(second_events[-1]["moods_applied"])
        self.assertEqual(first_events[-1]["mood_coefficients_applied"], first_mood)
        self.assertEqual(second_events[-1]["mood_coefficients_applied"], second_mood)
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
        next_mood = [-0.125, 0, 0.25, 0, 0, 0]
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
        next_mood = [0, -0.25, 0, 0.125, 0, 0]
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

    async def test_late_completion_during_second_join_restores_health(self):
        with patch("deployment.worker.GENERATION_JOIN_GRACE_SECONDS", 0.01):
            task, release = await self.blocked_generation("late-second-join")
            original = self.worker._start_retirement

            def begin_retirement(*args, **kwargs):
                self.assertFalse(self.worker.healthy)
                self.assertTrue(self.worker.lock.locked())
                release.set()
                return original(*args, **kwargs)

            try:
                with patch.object(self.worker, "_start_retirement", side_effect=begin_retirement):
                    events = await asyncio.wait_for(task, 2)
            finally:
                release.set()
                await self.wait_for_retirement()
        self.assertEqual(events[-1]["code"], "worker_unavailable")
        self.assertTrue(self.worker.healthy)
        self.assert_clean()
        self.worker.model.prefill_release = None
        following = await self.collect("after-late", {**PAYLOAD, "max_new_tokens": 1})
        self.assertEqual(following[-1]["event"], "done")

    async def test_completion_after_stream_cleanup_keeps_lease_and_recovers(self):
        with patch("deployment.worker.GENERATION_JOIN_GRACE_SECONDS", 0.01), patch(
            "deployment.worker.STREAM_RETIREMENT_GRACE_SECONDS", 0.01,
        ):
            task, release = await self.blocked_generation("background-retirement")
            try:
                events = await asyncio.wait_for(task, 2)
                self.assertEqual(events[-1]["code"], "worker_unavailable")
                self.assertFalse(self.worker.healthy)
                self.assertTrue(self.worker.lock.locked())
                self.assertIsNotNone(self.worker.steering.active)
                self.assertEqual(len(self.worker._retirements), 1)
                self.assertIn("background-retirement", self.worker.requests)
                unavailable = await self.collect("too-early")
                self.assertEqual(unavailable[-1]["code"], "worker_unavailable")
                with patch.object(self.worker, "_preflight", side_effect=AssertionError("Premature model access")):
                    with self.assertRaisesRegex(RuntimeError, "temporarily unavailable"):
                        await self.worker.preflight()
                self.assertEqual(self.worker.model.calls, 1)
                observer = asyncio.create_task(self.worker.lock.acquire())
                await asyncio.sleep(0)
                self.assertFalse(observer.done())
            finally:
                release.set()
                await self.wait_for_retirement()
            self.assertTrue(await asyncio.wait_for(observer, 2))
            self.worker.lock.release()
        self.assertTrue(self.worker.healthy)
        self.assert_clean()
        self.worker.model.prefill_release = None
        following = await self.collect("after-retirement", {**PAYLOAD, "max_new_tokens": 1})
        self.assertEqual(following[-1]["event"], "done")
        self.assertEqual(self.worker.model.max_active, 1)

    async def test_repeated_caller_cancellation_cannot_release_native_lease(self):
        release = threading.Event()
        self.worker.model.prefill_release = release
        task = asyncio.create_task(self.collect("repeated-cancellation"))
        self.assertTrue(await asyncio.to_thread(self.worker.model.prefill_waiting.wait, 2))
        try:
            task.cancel()
            for _ in range(100):
                if self.worker._retirements:
                    break
                await asyncio.sleep(0.001)
            self.assertEqual(len(self.worker._retirements), 1)
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task
            self.assertFalse(self.worker.healthy)
            self.assertTrue(self.worker.lock.locked())
            self.assertIsNotNone(self.worker.steering.active)
        finally:
            release.set()
            await self.wait_for_retirement()
        self.assertTrue(self.worker.healthy)
        self.assert_clean()

    async def test_cancelled_preflight_retains_exact_native_work_and_recovers(self):
        started, release = threading.Event(), threading.Event()

        def probe():
            with self.worker.steering.apply(PAYLOAD["mood"], 1):
                started.set()
                if not release.wait(5):
                    raise RuntimeError("Test did not release its maintenance probe")
                self.worker.model.model.rope_deltas = "probe cache"
            return {"passed": True}

        with patch.object(self.worker, "_preflight", side_effect=probe):
            task = asyncio.create_task(self.worker.preflight())
            self.assertTrue(await asyncio.to_thread(started.wait, 2))
            try:
                task.cancel()
                with self.assertRaises(asyncio.CancelledError):
                    await task
                self.assertFalse(self.worker.healthy)
                self.assertTrue(self.worker.lock.locked())
                self.assertIsNotNone(self.worker.steering.active)
                self.assertEqual((await self.collect("during-probe"))[-1]["code"], "worker_unavailable")
                with self.assertRaisesRegex(RuntimeError, "temporarily unavailable"):
                    await self.worker.preflight()
            finally:
                release.set()
                await self.wait_for_retirement()
        self.assertTrue(self.worker.healthy)
        self.assert_clean()
        self.assertEqual((await self.collect("after-probe", {**PAYLOAD, "max_new_tokens": 1}))[-1]["event"], "done")

    async def test_diagnose_uses_serialized_maintenance_owner(self):
        module = ModuleType("deployment.steering_probe")
        calls = []
        module.probe_runtime = lambda worker, **kwargs: calls.append((worker, kwargs)) or {"passed": True}
        with patch.dict(sys.modules, {"deployment.steering_probe": module}):
            self.assertEqual(await self.worker.diagnose(), {"passed": True})
        self.assertEqual(calls, [(self.worker, {"diagnostic_only": True, "compare_direct": False})])
        self.assert_clean()

    async def test_cuda_failure_stays_quarantined_even_after_thread_ends(self):
        self.worker.model.generation_error = RuntimeError("CUDA error: device-side assert triggered")
        events = await self.collect("unsafe-cuda")
        self.assertEqual(events[-1]["code"], "generation_failed")
        self.assertFalse(self.worker.healthy)
        self.assert_clean()
        self.assertEqual((await self.collect("after-cuda-failure"))[-1]["code"], "worker_unavailable")
        self.assertEqual(self.worker.model.calls, 1)

    async def test_cleanup_sync_failure_does_not_restore_health(self):
        def fail_sync(device):
            raise RuntimeError("Synthetic synchronization failure")

        self.worker.torch.cuda = SimpleNamespace(synchronize=fail_sync)
        await self.collect("failed-sync", {**PAYLOAD, "max_new_tokens": 1})
        self.assertFalse(self.worker.healthy)
        self.assertFalse(self.worker.lock.locked())
        with self.assertRaisesRegex(RuntimeError, "temporarily unavailable"):
            await self.worker.preflight()
        self.assertEqual((await self.collect("after-sync-failure"))[-1]["code"], "worker_unavailable")

    async def test_leaked_hook_registry_fails_closed(self):
        self.worker.model._forward_pre_hooks = {}
        generate = self.worker.model.generate

        def leaking_generate(**kwargs):
            self.worker.model._forward_pre_hooks[99] = lambda *args: None
            return generate(**kwargs)

        with patch.object(self.worker.model, "generate", side_effect=leaking_generate):
            await self.collect("leaking-hook", {**PAYLOAD, "max_new_tokens": 1})
        self.assertFalse(self.worker.healthy)
        self.assertFalse(self.worker.lock.locked())
        self.assertEqual((await self.collect("after-hook-leak"))[-1]["code"], "worker_unavailable")

    async def test_cancel_at_queued_lock_release_does_not_leak_lease(self):
        await self.worker.lock.acquire()
        task = asyncio.create_task(self.collect("queued-handoff"))
        await asyncio.sleep(0)
        self.worker.lock.release()
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertEqual(self.worker.model.calls, 0)
        self.assertEqual(self.worker.requests, {})
        self.assertFalse(self.worker.lock.locked())

    async def test_dead_generation_is_not_ready_until_cuda_cleanup_finishes(self):
        syncing, synced = threading.Event(), threading.Event()

        def sync(device):
            syncing.set()
            if not synced.wait(5):
                raise RuntimeError("Test did not finish its CUDA cleanup")

        self.worker.torch.cuda = SimpleNamespace(synchronize=sync)
        with patch("deployment.worker.GENERATION_JOIN_GRACE_SECONDS", 0.01), patch(
            "deployment.worker.STREAM_RETIREMENT_GRACE_SECONDS", 0.01,
        ):
            task, release = await self.blocked_generation("slow-cleanup")
            try:
                await asyncio.wait_for(task, 2)
                release.set()
                self.assertTrue(await asyncio.to_thread(syncing.wait, 2))
                self.assertEqual(self.worker.model.active, 0)
                self.assertIsNone(self.worker.steering.active)
                self.assertFalse(self.worker.healthy)
                self.assertTrue(self.worker.lock.locked())
                self.assertEqual((await self.collect("before-sync"))[-1]["code"], "worker_unavailable")
            finally:
                release.set()
                synced.set()
                await self.wait_for_retirement()
        self.assertTrue(self.worker.healthy)
        self.assert_clean()

    async def test_cancelled_join_child_fails_closed_without_spinning(self):
        await self.worker.lock.acquire()
        async def cancelled_work(*args):
            raise asyncio.CancelledError

        with patch("deployment.worker.asyncio.to_thread", new=cancelled_work):
            # Force cancellation of the child task itself, as during shutdown.
            task = asyncio.create_task(self.worker._retire_model_work(None, {}, self.worker._hook_snapshot(), None))
            with self.assertRaises(asyncio.CancelledError):
                await asyncio.wait_for(task, 2)
        self.assertFalse(self.worker.healthy)
        self.assertTrue(self.worker.lock.locked())
        self.worker.lock.release()


if __name__ == "__main__":
    unittest.main()
