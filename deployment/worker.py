"""Native Qwen inference with serialized generation and cooperative cancellation."""

from __future__ import annotations

import asyncio
import copy
import contextlib
import queue
import threading
import time

from deployment.core import (
    GENERATION_TIMEOUT_SECONDS, MAX_ADMITTED, MAX_CONTEXT_TOKENS,
    MAX_OUTPUT_TOKENS, MODEL_ID, MODEL_REVISION, RequestError,
    bounded_messages, validate_chat,
)
from deployment.formatting import post_instruction_start
from deployment.steering import MoodSteering


class ModelRuntime:
    def __init__(self, checkpoint, mood_vectors=None):
        import torch
        from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration

        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True)
        self.model = Qwen3_5ForConditionalGeneration.from_pretrained(
            checkpoint, torch_dtype=torch.bfloat16, device_map="cuda:0",
            attn_implementation="sdpa", local_files_only=True,
        ).eval()
        self.steering = MoodSteering(self.model, mood_vectors)
        self.lock = asyncio.Lock()
        self.requests: dict[str, threading.Event] = {}
        self.cancelled: dict[str, float] = {}
        self.healthy = True

    def _clear_request_cache(self):
        # Qwen's multimodal position bookkeeping is model state; generation's
        # fresh cache and this reset prevent previous requests from leaking in.
        self.model.model.rope_deltas = None

    async def cancel(self, request_id: str):
        now = time.monotonic()
        self.cancelled = {key: timestamp for key, timestamp in self.cancelled.items() if timestamp > now - 660}
        if request_id in self.requests:
            self.requests[request_id].set()
        elif len(self.cancelled) < 128:
            self.cancelled[request_id] = now
        return {"cancelled": True}

    async def stream(self, request_id: str, raw_payload: dict):
        from transformers import StoppingCriteria, StoppingCriteriaList, TextIteratorStreamer

        payload = validate_chat(raw_payload)
        if not self.healthy:
            yield {"event": "error", "code": "worker_unavailable", "message": "Mooody is restarting. Please try again."}
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
        try:
            if self.lock.locked():
                yield {"event": "status", "message": "Waiting for the current reply to finish."}
            # Cancellation while queued is checked without taking the model lock.
            while not acquired and not stop.is_set():
                try:
                    await asyncio.wait_for(self.lock.acquire(), timeout=0.25)
                    acquired = True
                except asyncio.TimeoutError:
                    pass
            if stop.is_set():
                return
            if not self.healthy:
                yield {"event": "error", "code": "worker_unavailable", "message": "Mooody is restarting. Please try again."}
                return
            inputs, removed = bounded_messages(self.tokenizer, payload.messages)
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
            await asyncio.to_thread(thread.join, 15)
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
                }
        except RequestError as error:
            yield {"event": "error", "code": error.code, "message": str(error)}
        finally:
            stop.set()
            if thread is not None and thread.is_alive():
                # Do not release the model to another caller while cancelled CUDA
                # work is still finishing. A fresh cache starts only after join.
                try:
                    await asyncio.shield(asyncio.to_thread(thread.join, 15))
                except asyncio.CancelledError:
                    await asyncio.to_thread(thread.join, 15)
                if thread.is_alive():
                    self.healthy = False
            if acquired:
                if thread is None or not thread.is_alive():
                    self._clear_request_cache()
                self.lock.release()
            self.requests.pop(request_id, None)

    async def preflight(self):
        """Probe real native generation and worst allowed context/output on L4."""
        async with self.lock:
            return await asyncio.to_thread(self._preflight)

    def _preflight(self):
        import importlib.metadata

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
        if "ready" not in replies[0].lower() or "4" not in replies[1] or any("<think>" in reply for reply in replies):
            raise RuntimeError(f"Native Mooody smoke probes failed: {replies!r}")
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
        capacity_mood = [2] * 6
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
            "smoke_replies": replies, "smoke_generation_seconds": timings,
            "thinking": False, **self.steering.metadata(), "moods_applied": applied,
            "capacity_mood_coefficients": capacity_mood,
            "steering_layers": len(self.steering.layers),
            "versions": {name: importlib.metadata.version(name) for name in (
                "torch", "transformers", "flash-linear-attention", "huggingface_hub",
            )},
        }
        if result["context_tokens"] != MAX_CONTEXT_TOKENS or result["output_tokens"] != MAX_OUTPUT_TOKENS:
            raise RuntimeError("The capacity probe did not exercise both deployed token limits")
        self._clear_request_cache()
        del inputs, output
        torch.cuda.empty_cache()
        return result
