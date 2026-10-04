"""Native Qwen inference with serialized generation and cooperative cancellation."""

from __future__ import annotations

import asyncio
import copy
import queue
import re
import threading
import time

from deployment.core import (
    GENERATION_TIMEOUT_SECONDS, MAX_ADMITTED, MAX_CONTEXT_TOKENS,
    MAX_OUTPUT_TOKENS, MAX_MOOD_COEFFICIENT, MODEL_ID, MODEL_REVISION, REQUEST_TIMEOUT_SECONDS, RequestError,
    bounded_messages, validate_chat,
)
from deployment.formatting import content_token_controls, post_instruction_start
from deployment.steering import MoodSteering

GENERATION_JOIN_GRACE_SECONDS = 15
STREAM_RETIREMENT_GRACE_SECONDS = 15
UNAVAILABLE_MESSAGE = "Mooody is temporarily unavailable. Please try again shortly."


def native_smoke_replies_valid(replies):
    return (
        len(replies) == 2
        and re.search(r"\bready\b", replies[0], re.IGNORECASE) is not None
        and re.search(r"\b(?:4|four)\b", replies[1], re.IGNORECASE) is not None
        and all("<think>" not in reply for reply in replies)
    )


class ModelRuntime:
    def __init__(self, checkpoint, mood_vectors, mood_vector_metadata):
        if mood_vectors is None or not isinstance(mood_vector_metadata, dict) or (
            mood_vector_metadata.get("mood_vectors_source") != "persona_vectors"
            or not mood_vector_metadata.get("mood_vectors_integrity_verified")
        ):
            raise ValueError("Serving requires an integrity-verified extracted persona bank")
        import torch
        from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration

        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True)
        self.model = Qwen3_5ForConditionalGeneration.from_pretrained(
            checkpoint, torch_dtype=torch.bfloat16, device_map="cuda:0",
            attn_implementation="sdpa", local_files_only=True,
        ).eval()
        excluded, thinking = content_token_controls(self.tokenizer, self.model.generation_config.eos_token_id)
        self.steering = MoodSteering(
            self.model, mood_vectors, mood_vector_metadata,
            excluded_content_token_ids=excluded, thinking_token_ids=thinking,
        )
        self.lock = asyncio.Lock()
        self.requests: dict[str, threading.Event] = {}
        self.cancelled: dict[str, float] = {}
        self.healthy = True
        self._retirements: set[asyncio.Task] = set()

    def _clear_request_cache(self):
        # Qwen's multimodal position bookkeeping is model state; generation's
        # fresh cache and this reset prevent previous requests from leaking in.
        self.model.model.rope_deltas = None

    def _hook_snapshot(self):
        """Keep immutable registry identities, never request text or tensors."""
        return tuple(
            (id(module), tuple((key, id(hook)) for key, hook in getattr(module, "_forward_pre_hooks", {}).items()),
             tuple((key, id(hook)) for key, hook in getattr(module, "_forward_hooks", {}).items()))
            for module in (self.model, *getattr(self.steering, "layers", ()))
        )

    @staticmethod
    def _unsafe_cuda_error(error):
        # A successful thread join does not repair a poisoned CUDA context.
        # Ordinary Python failures can recover after synchronization and cleanup.
        message = str(error).lower()
        return type(error).__name__ == "AcceleratorError" or any(marker in message for marker in (
            "cuda error", "device-side assert", "illegal memory access",
            "unspecified launch failure", "cublas_status", "cudnn_status",
        ))

    def _finish_model_work(self, state, hooks):
        cuda = getattr(self.torch, "cuda", None)
        if cuda is not None:
            cuda.synchronize(self.model.device)
        if self._hook_snapshot() != hooks:
            raise RuntimeError("Model hook cleanup did not restore its baseline")
        self._clear_request_cache()
        return not state.get("unsafe_cuda_error", False)

    async def _retire_model_work(self, thread, state, hooks, request_id):
        """Own the acquired lease through exact native completion and cleanup."""
        cleanup_finished = False
        cleaning = None
        async def finish_even_if_cancelled(task):
            while True:
                try:
                    return await asyncio.shield(task)
                except asyncio.CancelledError:
                    self.healthy = False
                    # Shield stops caller cancellation from reaching the child,
                    # but event-loop shutdown can cancel both tasks directly.
                    if task.cancelled():
                        raise

        try:
            if thread is not None and thread.ident is not None:
                joining = asyncio.create_task(asyncio.to_thread(thread.join))
                await finish_even_if_cancelled(joining)
            cleaning = asyncio.create_task(asyncio.to_thread(self._finish_model_work, state, hooks))
            self.healthy = await finish_even_if_cancelled(cleaning)
            cleanup_finished = True
        except Exception as error:
            self.healthy = False
            cleanup_finished = cleaning is not None and cleaning.done() and not cleaning.cancelled()
            print(f"Model retirement failed ({type(error).__name__})", flush=True)
        finally:
            # This task is strongly retained and protected from caller cancellation.
            # Never hand the lease back while its exact native thread is alive.
            if cleanup_finished and (thread is None or not thread.is_alive()):
                if request_id is not None:
                    self.requests.pop(request_id, None)
                self.lock.release()
        return self.healthy

    def _start_retirement(self, thread, state, hooks, request_id=None):
        task = asyncio.create_task(self._retire_model_work(thread, state, hooks, request_id))
        self._retirements.add(task)

        def completed(done):
            self._retirements.discard(done)
            if done.cancelled() or done.exception() is not None:
                self.healthy = False

        task.add_done_callback(completed)
        return task

    async def _run_maintenance(self, function, *args, **kwargs):
        if not self.healthy:
            raise RuntimeError(UNAVAILABLE_MESSAGE)
        await self.lock.acquire()
        thread = None
        state = {}
        retirement = None
        try:
            if not self.healthy:
                raise RuntimeError(UNAVAILABLE_MESSAGE)
            hooks = self._hook_snapshot()

            def run():
                try:
                    state["result"] = function(*args, **kwargs)
                except BaseException as error:
                    state["exception"] = error.with_traceback(None)
                    state["unsafe_cuda_error"] = self._unsafe_cuda_error(error)

            thread = threading.Thread(target=run, name="model-maintenance", daemon=True)
            thread.start()
            # Transfer ownership before the first await after starting native work.
            retirement = self._start_retirement(thread, state, hooks)
            await asyncio.shield(retirement)
        except asyncio.CancelledError:
            if retirement is not None and not retirement.done():
                self.healthy = False
            raise
        finally:
            if retirement is None:
                self.lock.release()
        if "exception" in state:
            raise state["exception"]
        if not self.healthy:
            raise RuntimeError(UNAVAILABLE_MESSAGE)
        return state["result"]

    async def cancel(self, request_id: str):
        now = time.monotonic()
        self.cancelled = {
            key: timestamp for key, timestamp in self.cancelled.items()
            if timestamp > now - (REQUEST_TIMEOUT_SECONDS + 60)
        }
        if request_id in self.requests:
            self.requests[request_id].set()
        elif len(self.cancelled) < 128:
            self.cancelled[request_id] = now
        return {"cancelled": True}

    async def stream(self, request_id: str, raw_payload: dict):
        from transformers import StoppingCriteria, StoppingCriteriaList, TextIteratorStreamer

        payload = validate_chat(raw_payload)
        if not self.healthy:
            yield {"event": "error", "code": "worker_unavailable", "message": UNAVAILABLE_MESSAGE}
            return
        if len(self.requests) >= MAX_ADMITTED:
            yield {"event": "error", "code": "queue_full", "message": "Mooody is busy. Please try again shortly."}
            return
        stop = threading.Event()
        if request_id in self.cancelled:
            self.cancelled.pop(request_id, None)
            return
        self.requests[request_id] = stop
        acquired = False
        thread = None
        state: dict = {}
        hooks = None
        try:
            if self.lock.locked():
                yield {"event": "status", "message": "Waiting for the current reply to finish."}
            # Cancellation while queued is checked without taking the model lock.
            while not acquired and not stop.is_set():
                try:
                    async with asyncio.timeout(0.25):
                        await self.lock.acquire()
                        acquired = True
                except asyncio.TimeoutError:
                    pass
            if stop.is_set():
                return
            if not self.healthy:
                yield {"event": "error", "code": "worker_unavailable", "message": UNAVAILABLE_MESSAGE}
                return
            hooks = self._hook_snapshot()
            inputs, removed = bounded_messages(self.tokenizer, payload.messages, payload.mood)
            boundary = post_instruction_start(self.tokenizer, inputs) if any(payload.mood) else None
            inputs = inputs.to(self.model.device)
            if removed:
                yield {"event": "status", "message": "Using the most recent messages to fit this conversation."}
            yield {"event": "status", "message": "Mooody is replying."}
            self._clear_request_cache()
            started = time.monotonic()
            deadline = started + GENERATION_TIMEOUT_SECONDS

            class StopRequest(StoppingCriteria):
                def __call__(self, input_ids, scores, **kwargs):
                    if time.monotonic() >= deadline:
                        state["timed_out"] = True
                        stop.set()
                    return stop.is_set()

            class BoundedStreamer(TextIteratorStreamer):
                def __init__(self, tokenizer):
                    super().__init__(tokenizer, skip_prompt=True, skip_special_tokens=True)
                    self.text_queue = queue.Queue(maxsize=64)

                def on_finalized_text(self, text: str, stream_end: bool = False):
                    for item in ([text, self.stop_signal] if stream_end else [text]):
                        while not stop.is_set():
                            try:
                                self.text_queue.put(item, timeout=0.1)
                                break
                            except queue.Full:
                                continue

            streamer = BoundedStreamer(self.tokenizer)

            def generate():
                try:
                    with self.torch.inference_mode(), self.steering.apply(payload.mood, boundary) as applied:
                        output = self.model.generate(
                            **inputs, do_sample=False, max_new_tokens=payload.max_new_tokens,
                            use_cache=True, logits_to_keep=1,
                            pad_token_id=self.tokenizer.pad_token_id,
                            streamer=streamer,
                            stopping_criteria=StoppingCriteriaList([StopRequest()]),
                        )
                    state["moods_applied"] = applied
                    state["tokens"] = output.shape[-1] - inputs["input_ids"].shape[-1]
                    eos = self.model.generation_config.eos_token_id
                    eos = {eos} if isinstance(eos, int) else set(eos or [])
                    last = int(output[0, -1].item())
                    state["finish_reason"] = (
                        "timeout" if state.get("timed_out") else "cancelled" if stop.is_set()
                        else "stop" if last in eos else "length" if state["tokens"] >= payload.max_new_tokens else "stop"
                    )
                except Exception as error:
                    # Keep only the exception type in container logs; no prompts.
                    print(f"Generation failed ({type(error).__name__})", flush=True)
                    state["error"] = True
                    state["unsafe_cuda_error"] = self._unsafe_cuda_error(error)

            thread = threading.Thread(target=generate, name=f"generation-{request_id[:8]}", daemon=True)
            thread.start()
            while thread.is_alive() or not streamer.text_queue.empty():
                if stop.is_set():
                    break
                try:
                    text = await asyncio.to_thread(streamer.text_queue.get, True, 0.2)
                except queue.Empty:
                    continue
                if text == streamer.stop_signal:
                    break
                if text:
                    yield {"event": "token", "text": text}
            await asyncio.to_thread(thread.join, GENERATION_JOIN_GRACE_SECONDS)
            if thread.is_alive():
                self.healthy = False
                yield {"event": "error", "code": "worker_unavailable", "message": "Mooody could not finish this reply. Please try again."}
            elif state.get("error"):
                yield {"event": "error", "code": "generation_failed", "message": "Mooody could not finish this reply. Please try again."}
            elif not stop.is_set() or state.get("timed_out"):
                yield {
                    "event": "done", "finish_reason": state.get("finish_reason", "stop"),
                    "generated_tokens": state.get("tokens", 0), **self.steering.metadata(),
                    "moods_applied": state.get("moods_applied", False),
                    "mood_coefficients_applied": list(payload.mood),
                    "system_prompt_present": False,
                    "mood_conditioning": "vectors_with_prompt_assistance",
                    "mood_prompt_assistance_applied": bool(any(payload.mood)),
                }
        except RequestError as error:
            yield {"event": "error", "code": error.code, "message": str(error)}
        finally:
            stop.set()
            if acquired and hooks is not None:
                retirement = self._start_retirement(thread, state, hooks, request_id)
                try:
                    await asyncio.wait_for(asyncio.shield(retirement), STREAM_RETIREMENT_GRACE_SECONDS)
                except asyncio.TimeoutError:
                    if not retirement.done():
                        self.healthy = False
                except asyncio.CancelledError:
                    if not retirement.done():
                        self.healthy = False
                    raise
            else:
                if acquired:
                    self.lock.release()
                self.requests.pop(request_id, None)

    async def preflight(self):
        """Probe real native generation and worst allowed context/output on L4."""
        return await self._run_maintenance(self._preflight)

    async def diagnose(self):
        from deployment.steering_probe import probe_runtime

        return await self._run_maintenance(
            probe_runtime, self, diagnostic_only=True, compare_direct=False,
        )

    async def steering_regression(self):
        from deployment.steering_probe import probe_runtime

        return await self._run_maintenance(
            probe_runtime, self, include_trait_endpoints=True, compare_direct=False,
        )

    def _preflight(self):
        import importlib.metadata
        from deployment.steering_probe import probe_runtime

        torch = self.torch
        replies = []
        timings = []
        for prompt in ("Reply with the single word ready.", "What is two plus two? Reply briefly."):
            self._clear_request_cache()
            inputs, _ = bounded_messages(self.tokenizer, [{"role": "user", "content": prompt}])
            inputs = inputs.to(self.model.device)
            started = time.monotonic()
            with torch.inference_mode():
                output = self.model.generate(
                    **inputs, do_sample=False, max_new_tokens=32, use_cache=True,
                    logits_to_keep=1, pad_token_id=self.tokenizer.pad_token_id,
                )
            torch.cuda.synchronize()
            timings.append(time.monotonic() - started)
            replies.append(self.tokenizer.decode(output[0, inputs["input_ids"].shape[-1]:], skip_special_tokens=True))
        if not native_smoke_replies_valid(replies):
            raise RuntimeError(f"Native Mooody smoke probes failed: {replies!r}")
        repetition_probe = probe_runtime(self, include_trait_endpoints=True)
        if not repetition_probe["passed"]:
            raise RuntimeError(f"Incremental steering repetition probe failed: {repetition_probe!r}")
        print("Mooody preflight: trait endpoint checks passed; testing full token capacity", flush=True)
        self._clear_request_cache()
        long_prompt = "Explain a rainbow briefly. " * MAX_CONTEXT_TOKENS
        inputs = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": long_prompt}], tokenize=True,
            add_generation_prompt=True, enable_thinking=False,
            return_tensors="pt", return_dict=True,
        )
        # Test an exact maximum-width text prefill and force the entire output
        # allocation. This is a capacity test, not a quality evaluation.
        inputs = {key: tensor[:, -MAX_CONTEXT_TOKENS:] for key, tensor in inputs.items()}
        boundary = post_instruction_start(self.tokenizer, inputs)
        inputs = {key: tensor.to(self.model.device) for key, tensor in inputs.items()}
        capacity_mood = [MAX_MOOD_COEFFICIENT] * 6
        generation = copy.deepcopy(self.model.generation_config)
        generation.eos_token_id = None
        generation.forced_eos_token_id = None
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        started = time.monotonic()
        with torch.inference_mode(), self.steering.apply(capacity_mood, boundary) as applied:
            output = self.model.generate(
                **inputs, generation_config=generation, do_sample=False,
                max_new_tokens=MAX_OUTPUT_TOKENS, min_new_tokens=MAX_OUTPUT_TOKENS,
                use_cache=True, logits_to_keep=1, pad_token_id=self.tokenizer.pad_token_id,
            )
        torch.cuda.synchronize()
        result = {
            "model_id": MODEL_ID, "revision": MODEL_REVISION,
            "gpu": torch.cuda.get_device_name(),
            "gpu_memory_gib": torch.cuda.get_device_properties(0).total_memory / 2**30,
            "peak_gpu_memory_gib": torch.cuda.max_memory_allocated() / 2**30,
            "peak_gpu_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
            "context_tokens": inputs["input_ids"].shape[-1],
            "output_tokens": output.shape[-1] - inputs["input_ids"].shape[-1],
            "max_context_generation_seconds": time.monotonic() - started,
            "generation_deadline_seconds": GENERATION_TIMEOUT_SECONDS,
            "smoke_replies": replies, "smoke_generation_seconds": timings,
            "steering_repetition_probe": repetition_probe,
            "thinking": False, **self.steering.metadata(), "moods_applied": applied,
            "capacity_mood_coefficients": capacity_mood,
            "steering_layers": len(self.steering.layers),
            "versions": {name: importlib.metadata.version(name) for name in (
                "torch", "transformers", "flash-linear-attention", "huggingface_hub",
            )},
        }
        if result["context_tokens"] != MAX_CONTEXT_TOKENS or result["output_tokens"] != MAX_OUTPUT_TOKENS:
            raise RuntimeError("The capacity probe did not exercise both deployed token limits")
        if result["max_context_generation_seconds"] >= GENERATION_TIMEOUT_SECONDS:
            raise RuntimeError("Maximum-size generation exceeded the application deadline")
        self._clear_request_cache()
        del inputs, output
        torch.cuda.empty_cache()
        return result
