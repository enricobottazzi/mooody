import asyncio
import json
from pathlib import Path
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import httpx
from synchronicity import Synchronizer

from deployment.api import _close_remote_stream, create_web_app


TOKEN = "test-private-proxy-secret"
HEADERS = {"x-mooody-proxy-token": TOKEN, "x-mooody-client-ip": "192.0.2.1"}
PAYLOAD = {"messages": [{"role": "user", "content": "Hello"}], "mood": [0] * 6}


class FakeWorker:
    def __init__(self, mode="complete"):
        self.mode = mode
        self.calls = []
        self.cancellations = []
        self.stopped = asyncio.Event()
        self.stream = SimpleNamespace(remote_gen=SimpleNamespace(aio=self.events))
        self.cancel = SimpleNamespace(remote=SimpleNamespace(aio=self.cancel_call))

    async def cancel_call(self, request_id):
        self.cancellations.append(request_id)
        self.stopped.set()

    async def events(self, request_id, value):
        self.calls.append((request_id, value))
        yield {"event": "token", "text": "Hello world"}
        if self.mode == "wait":
            await self.stopped.wait()
            return
        elif self.mode == "failure":
            raise RuntimeError("Private cloud failure detail")
        yield {"event": "done", "finish_reason": "stop", "generated_tokens": 2,
               "moods_applied": any(value["mood"]), "mood_vectors_source": "random_placeholder"}


class BridgeWorker(FakeWorker):
    """Use Modal's actual SDK bridge, including delayed inner finalization."""

    def __init__(self):
        super().__init__("wait")
        self.bridge = Synchronizer()
        self.stopping = threading.Event()
        self.finalized = threading.Event()
        self.waiting = threading.Event()

        async def bridged(request_id, value):
            async for event in self.bridge_events(request_id, value):
                yield event

        self.stream = SimpleNamespace(remote_gen=self.bridge.create_blocking(bridged))

    async def cancel_call(self, request_id):
        self.cancellations.append(request_id)
        self.stopping.set()

    async def bridge_events(self, request_id, value):
        self.calls.append((request_id, value))
        try:
            yield {"event": "token", "text": "Hello world"}
            self.waiting.set()
            while not self.stopping.is_set():
                await asyncio.sleep(0.01)
        finally:
            # Model/network cleanup finishes after its cancellation is requested.
            await asyncio.sleep(0.05)
            self.finalized.set()


class ApiTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.directory = tempfile.TemporaryDirectory()
        Path(self.directory.name, "index.html").write_text("mooody test page")
        self.worker = FakeWorker()
        self.app = create_web_app(self.worker, self.directory.name, TOKEN)
        self.client = httpx.AsyncClient(transport=httpx.ASGITransport(app=self.app), base_url="https://modal-origin.example")

    async def asyncTearDown(self):
        await self.client.aclose()
        self.directory.cleanup()

    async def test_origin_cannot_bypass_proxy_and_health_does_not_start_gpu(self):
        denied = await self.client.get("/api/config")
        self.assertEqual(denied.status_code, 403)
        health = await self.client.get("/api/health", headers=HEADERS)
        self.assertEqual(health.status_code, 200)
        self.assertEqual(self.worker.calls, [])
        page = await self.client.get("/", headers=HEADERS)
        self.assertIn("mooody test page", page.text)
        self.assertIn("frame-ancestors 'self'", page.headers["content-security-policy"])

    async def test_cross_origin_is_rejected(self):
        denied = await self.client.post("/api/chat", json=PAYLOAD, headers={**HEADERS, "origin": "https://attacker.example"})
        self.assertEqual(denied.status_code, 403)
        self.assertEqual(self.worker.calls, [])

    async def test_validation_and_body_size_do_not_start_gpu(self):
        invalid = await self.client.post("/api/chat", json={**PAYLOAD, "mood": [3] * 6}, headers=HEADERS)
        self.assertEqual(invalid.status_code, 422)
        huge = await self.client.post("/api/chat", content=b"x" * 65537, headers={**HEADERS, "content-type": "application/json"})
        self.assertEqual(huge.status_code, 413)
        self.assertEqual(self.worker.calls, [])

    async def test_real_sse_contract_and_normal_completion(self):
        response = await self.client.post("/api/chat", json=PAYLOAD, headers={**HEADERS, "origin": "https://mooody.ai"})
        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.headers["content-type"].startswith("text/event-stream"))
        self.assertIn('event: meta\n', response.text)
        self.assertIn('"moods_applied":false', response.text)
        self.assertIn('event: token\ndata: {"text":"Hello world"}', response.text)
        self.assertIn('event: done\n', response.text)
        self.assertEqual(self.worker.calls[0][1]["mood"], [0] * 6)

    async def test_steering_configuration_and_nonzero_mood_reply_metadata(self):
        config = await self.client.get("/api/config", headers=HEADERS)
        self.assertTrue(config.json()["steering_available"])
        self.assertEqual(config.json()["mood_vectors_source"], "random_placeholder")
        self.assertFalse(config.json()["mood_vectors_validated"])
        self.assertEqual(self.worker.calls, [])
        mood = [-2, -1, 0, 1, 2, 0]
        response = await self.client.post("/api/chat", json={**PAYLOAD, "mood": mood}, headers=HEADERS)
        self.assertEqual(self.worker.calls[0][1]["mood"], mood)
        frames = [frame for frame in response.text.split("\n\n") if frame.startswith("event: done\n")]
        done = json.loads(frames[0].split("data: ", 1)[1])
        self.assertTrue(done["moods_applied"])
        self.assertEqual(done["mood_vectors_source"], "random_placeholder")
        self.assertEqual(self.worker.cancellations, [])

    async def test_upstream_error_is_sanitized_and_cancelled(self):
        self.worker.mode = "failure"
        response = await self.client.post("/api/chat", json=PAYLOAD, headers=HEADERS)
        self.assertIn('event: error\n', response.text)
        self.assertNotIn("Private cloud failure", response.text)
        self.assertEqual(len(self.worker.cancellations), 1)

    async def test_client_disconnect_cancels_gpu_request(self):
        self.worker.mode = "wait"
        await self.disconnect_after_token()
        self.assertEqual(len(self.worker.cancellations), 1)

    async def disconnect_after_token(self, before_disconnect=None):
        incoming = asyncio.Queue()
        await incoming.put({"type": "http.request", "body": json.dumps(PAYLOAD).encode(), "more_body": False})
        sent = []

        async def receive():
            return await incoming.get()

        async def send(message):
            sent.append(message)
            if message["type"] == "http.response.body" and b"event: token" in message.get("body", b""):
                if before_disconnect:
                    await before_disconnect()
                await incoming.put({"type": "http.disconnect"})

        scope = {
            "type": "http", "asgi": {"version": "3.0", "spec_version": "2.3"},
            "http_version": "1.1", "method": "POST", "scheme": "https",
            "path": "/api/chat", "raw_path": b"/api/chat", "query_string": b"",
            "root_path": "", "client": ("192.0.2.1", 1234), "server": ("modal-origin.example", 443),
            "headers": [(key.encode(), value.encode()) for key, value in {
                **HEADERS, "content-type": "application/json", "host": "modal-origin.example",
            }.items()],
        }
        await asyncio.wait_for(self.app(scope, receive, send), timeout=3)
        # The shielded cancellation task may finish after ASGI disconnect returns.
        for _ in range(20):
            if self.worker.cancellations:
                break
            await asyncio.sleep(0.01)
        self.assertEqual(len(self.worker.cancellations), 1)
        self.assertTrue(any(item.get("status") == 200 for item in sent))

    async def test_disconnected_asgi_scope_waits_for_actual_modal_bridge_finalization(self):
        self.worker = BridgeWorker()
        self.app = create_web_app(self.worker, self.directory.name, TOKEN)
        errors = []
        bridge_loop = await self.worker.bridge._get_loop_async()
        bridge_loop.call_soon_threadsafe(bridge_loop.set_exception_handler, lambda loop, context: errors.append(context))

        async def wait_until_inner_anext_is_running():
            self.assertTrue(await asyncio.to_thread(self.worker.waiting.wait, 2))

        try:
            await self.disconnect_after_token(wait_until_inner_anext_is_running)
            self.assertTrue(self.worker.finalized.is_set())
            self.assertFalse(any(
                task.get_name().startswith("mooody-next-event-") and not task.done()
                for task in asyncio.all_tasks()
            ))
            await asyncio.sleep(0.05)
            self.assertEqual(errors, [])
        finally:
            self.worker.bridge._close_loop()


class CleanupTests(unittest.IsolatedAsyncioTestCase):
    async def test_cooperative_cancel_settles_iteration_before_close(self):
        release = asyncio.Event()
        order = []

        class Iterator:
            def __init__(self):
                self.running = False

            async def __anext__(self):
                self.running = True
                await release.wait()
                await asyncio.sleep(0.02)
                self.running = False
                order.append("iteration_finished")
                return {"event": "token", "text": "last buffered fragment"}

            async def aclose(self):
                if self.running:
                    raise AssertionError("aclose must not run concurrently with anext")
                order.append("closed")

        remote = Iterator()
        pending = asyncio.create_task(anext(remote))
        await asyncio.sleep(0)

        async def cancel():
            order.append("worker_cancelled")
            release.set()

        await _close_remote_stream(cancel, remote, pending, complete=False)
        self.assertTrue(pending.done())
        self.assertEqual(order, ["worker_cancelled", "iteration_finished", "closed"])

    async def test_failed_iteration_owns_finalization_and_is_not_closed_again(self):
        class Iterator:
            async def __anext__(self):
                raise RuntimeError("SDK iterator already finalized")

            async def aclose(self):
                raise AssertionError("failed anext already owns close")

        remote = Iterator()
        pending = asyncio.create_task(anext(remote))
        await asyncio.sleep(0)

        async def cancel():
            pass

        await _close_remote_stream(cancel, remote, pending, complete=False)
        self.assertTrue(pending.done())

    async def test_timeout_fallback_joins_cancelled_iteration_without_second_close(self):
        finalized = asyncio.Event()

        class Iterator:
            async def __anext__(self):
                try:
                    await asyncio.sleep(60)
                finally:
                    await asyncio.sleep(0.01)
                    finalized.set()

            async def aclose(self):
                raise AssertionError("cancelled anext already owns finalization")

        remote = Iterator()
        pending = asyncio.create_task(anext(remote))
        await asyncio.sleep(0)

        async def cancel():
            pass

        actual_wait = asyncio.wait

        async def short_wait(tasks, timeout):
            return await actual_wait(tasks, timeout=min(timeout, 0.03))

        with patch("deployment.api.asyncio.wait", short_wait):
            await _close_remote_stream(cancel, remote, pending, complete=False)
        self.assertTrue(pending.cancelled())
        self.assertTrue(finalized.is_set())


if __name__ == "__main__":
    unittest.main()
