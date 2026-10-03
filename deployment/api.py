"""Same-origin static site and anonymous, bounded streaming chat API."""

import asyncio
import contextlib
import hmac
import json
import os
import time
from pathlib import Path
from urllib.parse import urlsplit
import uuid

from anyio import CancelScope

from deployment.core import (
    Admission, MAX_BODY_BYTES, MODEL_ID, MODEL_REVISION,
    REQUEST_TIMEOUT_SECONDS, RequestError, configuration, sse, validate_chat,
)


def _consume_task_result(task):
    with contextlib.suppress(asyncio.CancelledError, Exception):
        task.result()


async def _close_remote_stream(cancel_worker, remote, pending, complete):
    """Finish the current SDK iteration before closing its generator.

    Modal's synchronous/asynchronous bridge shields its inner generator task.
    Cancelling our anext task and immediately calling aclose can therefore race
    that inner iteration. Cooperative GPU cancellation normally ends anext;
    explicit task cancellation is only the bounded fallback. A cancelled or
    failed anext owns the SDK generator's finalization and must not be closed
    concurrently from here.
    """
    if not complete:
        with contextlib.suppress(Exception):
            await asyncio.wait_for(cancel_worker(), timeout=10)
    if pending is not None:
        if not pending.done():
            await asyncio.wait({pending}, timeout=5)
        if not pending.done():
            pending.cancel()
            # The SDK finalizer has its own ten-second grace period. Wait for
            # it under the caller's shield instead of cancelling it repeatedly.
            await asyncio.wait({pending}, timeout=10)
        if not pending.done():
            # A broken remote connection may outlive the cleanup deadline. The
            # already-cancelled SDK task still owns close; retrieve its eventual
            # result and never start a concurrent aclose operation.
            pending.add_done_callback(_consume_task_result)
            print("Mooody remote cleanup reached its deadline", flush=True)
            return
        try:
            pending.result()
        except (asyncio.CancelledError, Exception):
            return
    if hasattr(remote, "aclose"):
        with contextlib.suppress(Exception):
            await asyncio.wait_for(remote.aclose(), timeout=5)


def create_web_app(worker, static_dir: str | Path = "/www", proxy_token: str | None = None):
    from fastapi import FastAPI, Request
    from fastapi.responses import JSONResponse, StreamingResponse
    from fastapi.staticfiles import StaticFiles

    web = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)
    gate = Admission()
    secret = proxy_token if proxy_token is not None else os.environ.get("MOOODY_PROXY_TOKEN", "")
    if not secret:
        raise RuntimeError("MOOODY_PROXY_TOKEN must be configured before serving the website")

    @web.middleware("http")
    async def protect(request: Request, call_next):
        token = request.headers.get("x-mooody-proxy-token", "")
        if not hmac.compare_digest(token.encode(), secret.encode()):
            return JSONResponse({"message": "This endpoint is available through mooody.ai."}, status_code=403)
        if request.method not in {"GET", "HEAD", "POST"}:
            return JSONResponse({"message": "Method not allowed."}, status_code=405)
        origin = request.headers.get("origin")
        if origin:
            parsed = urlsplit(origin)
            if parsed.scheme != "https" or parsed.netloc not in {"mooody.ai", "www.mooody.ai", request.url.netloc}:
                return JSONResponse({"message": "This request must come from mooody.ai."}, status_code=403)
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["Permissions-Policy"] = "camera=(), microphone=(), geolocation=()"
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline' https://fonts.googleapis.com; "
            "font-src 'self' https://fonts.gstatic.com; img-src 'self' data:; connect-src 'self'; "
            "frame-ancestors 'self'; object-src 'none'; base-uri 'self'"
        )
        if request.url.path.startswith("/api/"):
            response.headers["Cache-Control"] = "no-store"
        return response

    @web.get("/api/health")
    async def health():
        # Does not invoke or start the GPU. Readiness is checked at release time.
        return {"ok": True, "service": "mooody", "model_id": MODEL_ID, "revision": MODEL_REVISION}

    @web.get("/api/config")
    async def config():
        return configuration()

    @web.post("/api/chat")
    async def chat(request: Request):
        try:
            content_length = request.headers.get("content-length")
            if content_length and (not content_length.isdigit() or int(content_length) > MAX_BODY_BYTES):
                raise RequestError("The request is too large.", "request_too_large", 413)
            if request.headers.get("content-type", "").split(";", 1)[0].lower() != "application/json":
                raise RequestError("Send this request as application/json.", "invalid_content_type", 415)
            body = bytearray()
            async for chunk in request.stream():
                if len(body) + len(chunk) > MAX_BODY_BYTES:
                    raise RequestError("The request is too large.", "request_too_large", 413)
                body.extend(chunk)
            try:
                payload = validate_chat(json.loads(body))
            except (json.JSONDecodeError, UnicodeDecodeError):
                raise RequestError("The request is not valid JSON.") from None
            request_id = uuid.uuid4().hex
            # The Worker replaces this header with Cloudflare's observed client IP.
            client = request.headers.get("x-mooody-client-ip") or (request.client.host if request.client else "unknown")
            gate.enter(client, request_id)
        except RequestError as error:
            headers = {"Retry-After": "60"} if error.status in (429, 503) else None
            return JSONResponse({"message": str(error), "code": error.code}, status_code=error.status, headers=headers)

        async def cancel_worker():
            await worker.cancel.remote.aio(request_id)

        async def events():
            remote = None
            pending = None
            complete = False
            started = time.monotonic()
            try:
                yield sse("meta", {"request_id": request_id, **configuration()})
                yield sse("status", {"message": "Getting mooody ready. The first reply after a break can take a few minutes."})
                remote = worker.stream.remote_gen.aio(request_id, payload.payload())
                pending = asyncio.create_task(anext(remote), name=f"mooody-next-event-{request_id}")
                while True:
                    if await request.is_disconnected():
                        break
                    if time.monotonic() - started > REQUEST_TIMEOUT_SECONDS:
                        yield sse("error", {"code": "request_timeout", "message": "Mooody took too long to reply. Please try again."})
                        break
                    ready, _ = await asyncio.wait({pending}, timeout=5)
                    if not ready:
                        yield b": keep-alive\n\n"
                        continue
                    try:
                        event = pending.result()
                    except StopAsyncIteration:
                        if not complete:
                            yield sse("error", {"code": "incomplete_response", "message": "The reply was interrupted. Please try again."})
                        break
                    name = event.pop("event")
                    yield sse(name, event)
                    if name == "done":
                        complete = True
                        break
                    if name == "error":
                        break
                    pending = asyncio.create_task(anext(remote), name=f"mooody-next-event-{request_id}")
            except asyncio.CancelledError:
                raise
            except Exception:
                # Do not expose credentials, tracebacks, prompt content, or cloud internals.
                yield sse("error", {"code": "inference_unavailable", "message": "Mooody could not finish this reply. Please try again."})
            finally:
                try:
                    if remote is not None:
                        # ASGI uses level cancellation: asyncio.shield alone
                        # protects a child task but interrupts each later await.
                        # Shield the entire ordered, bounded cleanup operation.
                        with CancelScope(shield=True):
                            await _close_remote_stream(cancel_worker, remote, pending, complete)
                finally:
                    gate.leave(request_id)

        return StreamingResponse(events(), media_type="text/event-stream", headers={
            "Cache-Control": "no-store, no-transform", "X-Accel-Buffering": "no",
        })

    web.mount("/", StaticFiles(directory=str(static_dir), html=True), name="site")
    return web
