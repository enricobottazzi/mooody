"""Dependency-free request limits and streaming helpers."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import json
import time
from typing import Any

MODEL_ID = "demivoleegaston/Qwen3.5-9B-mooody"
MODEL_REVISION = "705afd95bced3ac0424d7e68b1299d8fcdffb858"
AXES = ("warmth", "patience", "playfulness", "optimism", "energy", "curiosity")
PLACEHOLDER_SEED = 20261003
PLACEHOLDER_NORM = 1.0
MAX_INPUT_CHARS = 2000
MAX_ASSISTANT_CHARS = 10000
MAX_HISTORY_CHARS = 32000
MAX_MESSAGES = 32
MAX_BODY_BYTES = 65536
MAX_CONTEXT_TOKENS = 8192
MAX_OUTPUT_TOKENS = 2048
MAX_ADMITTED = 4  # One active generation, at most three waiting.
REQUEST_TIMEOUT_SECONDS = 1800
GENERATION_TIMEOUT_SECONDS = 300


class RequestError(ValueError):
    def __init__(self, message: str, code: str = "invalid_request", status: int = 422):
        super().__init__(message)
        self.code = code
        self.status = status


@dataclass(frozen=True)
class ChatInput:
    messages: list[dict[str, str]]
    mood: list[int]
    max_new_tokens: int = MAX_OUTPUT_TOKENS

    def payload(self) -> dict[str, Any]:
        return {"messages": self.messages, "mood": self.mood, "max_new_tokens": self.max_new_tokens}


def validate_chat(value: Any) -> ChatInput:
    if not isinstance(value, dict):
        raise RequestError("Send a JSON object containing messages and mood.")
    if set(value) - {"messages", "mood", "max_new_tokens"}:
        raise RequestError("The request contains unsupported fields.")
    messages = value.get("messages")
    if not isinstance(messages, list) or not 1 <= len(messages) <= MAX_MESSAGES:
        raise RequestError(f"Send between 1 and {MAX_MESSAGES} messages.")
    cleaned: list[dict[str, str]] = []
    for message in messages:
        if not isinstance(message, dict) or set(message) != {"role", "content"}:
            raise RequestError("Each message must contain only role and content.")
        role, content = message["role"], message["content"]
        if role not in ("user", "assistant"):
            raise RequestError("Only user and assistant messages are accepted.")
        if not isinstance(content, str) or not content.strip():
            raise RequestError("Messages must contain text.")
        try:
            content.encode("utf-8")
        except UnicodeEncodeError:
            raise RequestError("Messages must contain valid Unicode text.") from None
        limit = MAX_INPUT_CHARS if role == "user" else MAX_ASSISTANT_CHARS
        if len(content) > limit:
            raise RequestError(f"A {role} message exceeds the {limit} character limit.")
        if cleaned and cleaned[-1]["role"] == role:
            raise RequestError("User and assistant messages must alternate.")
        cleaned.append({"role": role, "content": content})
    if cleaned[-1]["role"] != "user":
        raise RequestError("The final message must be from the user.")
    if sum(len(message["content"]) for message in cleaned) > MAX_HISTORY_CHARS:
        raise RequestError("This conversation is too long. Start a new chat.")
    mood = value.get("mood", [0] * len(AXES))
    if not isinstance(mood, list) or len(mood) != len(AXES) or any(
        type(level) is not int or not -2 <= level <= 2 for level in mood
    ):
        raise RequestError("Mood must contain six integer levels between -2 and 2.")
    output_limit = value.get("max_new_tokens", MAX_OUTPUT_TOKENS)
    if type(output_limit) is not int or not 1 <= output_limit <= MAX_OUTPUT_TOKENS:
        raise RequestError(f"Output length must be between 1 and {MAX_OUTPUT_TOKENS} tokens.")
    return ChatInput(cleaned, list(mood), output_limit)


def configuration() -> dict[str, Any]:
    return {
        "model_id": MODEL_ID,
        "revision": MODEL_REVISION,
        "mood_vectors_available": True,
        "steering_available": True,
        "mood_vectors_source": "random_placeholder",
        "mood_vectors_validated": False,
        "moods_applied": False,
        "axes": list(AXES),
        "mood_levels": [-2, -1, 0, 1, 2],
        "max_input_chars": MAX_INPUT_CHARS,
        "max_messages": MAX_MESSAGES,
        "max_context_tokens": MAX_CONTEXT_TOKENS,
        "max_output_tokens": MAX_OUTPUT_TOKENS,
        "thinking": False,
        "max_waiting_requests": MAX_ADMITTED - 1,
    }


def sse(event: str, data: dict[str, Any]) -> bytes:
    # JSON escapes embedded newlines; user/model text cannot inject SSE events.
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False, separators=(',', ':'))}\n\n".encode()


class Admission:
    """Single CPU-container admission and ephemeral anonymous rate limits.

    No prompts or transcripts are retained. Keys expire, and the key count is
    bounded even when a public caller rotates addresses. Limits reset on restart.
    """

    def __init__(self, clock=time.monotonic):
        self.clock = clock
        self.active: dict[str, tuple[str, float]] = {}
        self.clients: dict[str, deque[float]] = {}
        self.global_requests: deque[float] = deque()

    def enter(self, client: str, request_id: str) -> None:
        now = self.clock()
        self.active = {key: item for key, item in self.active.items() if item[1] > now}
        self.clients = {key: times for key, times in self.clients.items() if times and times[-1] > now - 60}
        if any(item[0] == client for item in self.active.values()):
            raise RequestError("You already have a reply in progress.", "client_busy", 429)
        if len(self.active) >= MAX_ADMITTED:
            raise RequestError("Mooody is busy. Please try again shortly.", "queue_full", 503)
        while self.global_requests and self.global_requests[0] <= now - 3600:
            self.global_requests.popleft()
        if len(self.global_requests) >= 80:
            raise RequestError("Mooody has reached its hourly limit. Please try again later.", "service_rate_limit", 429)
        times = self.clients.setdefault(client, deque())
        while times and times[0] <= now - 60:
            times.popleft()
        if len(times) >= 6 or len(self.clients) > 4096:
            raise RequestError("Please wait a minute before sending more messages.", "rate_limit", 429)
        times.append(now)
        self.global_requests.append(now)
        self.active[request_id] = (client, now + REQUEST_TIMEOUT_SECONDS + 30)

    def leave(self, request_id: str) -> None:
        self.active.pop(request_id, None)


def bounded_messages(tokenizer: Any, messages: list[dict[str, str]]) -> tuple[Any, int]:
    """Keep the newest complete turns, then bound actual formatted tokens."""
    history = list(messages)
    removed = 0
    while history:
        inputs = tokenizer.apply_chat_template(
            history, tokenize=True, add_generation_prompt=True,
            enable_thinking=False, return_tensors="pt", return_dict=True,
        )
        if inputs["input_ids"].shape[-1] <= MAX_CONTEXT_TOKENS:
            return inputs, removed
        if len(history) == 1:
            raise RequestError("Your message is too long for the model context. Please shorten it.", "context_too_long")
        history.pop(0)
        removed += 1
        # A cutoff should not leave an old reply without its user prompt.
        if history and history[0]["role"] == "assistant":
            history.pop(0)
            removed += 1
    raise RequestError("The conversation contains no user prompt.")
