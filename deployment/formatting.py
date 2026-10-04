"""Locate the final formatted prompt token for first-response prediction."""

from __future__ import annotations

from typing import Any, Mapping
import re


USER_END_MARKER = "<|im_end|>"


def content_token_controls(tokenizer: Any, eos_token_id: Any) -> tuple[tuple[int, ...], tuple[int | None, int | None]]:
    """Match extraction's exclusion of chat/EOS controls and thinking spans."""
    excluded = set(tokenizer.all_special_ids)
    excluded.update([eos_token_id] if type(eos_token_id) is int else (eos_token_id or []))
    for token, token_id in tokenizer.get_vocab().items():
        if re.fullmatch(r"<\|.*\|>", token):
            excluded.add(token_id)
    thinking = []
    for marker in ("<think>", "</think>"):
        ids = tokenizer.encode(marker, add_special_tokens=False)
        token_id = ids[0] if len(ids) == 1 and tokenizer.decode(ids, skip_special_tokens=False).strip() == marker else None
        thinking.append(token_id)
        if token_id is not None:
            excluded.add(token_id)
    return tuple(sorted(excluded)), tuple(thinking)


def _single_unpadded_row(value: Any, name: str) -> list[Any]:
    """Read one CPU token row without importing the model runtime dependencies."""
    shape = getattr(value, "shape", ())
    if len(shape) != 2 or shape[0] != 1 or shape[1] < 1:
        raise ValueError(f"{name} must be a nonempty batch-one tensor.")
    if not callable(getattr(value, "tolist", None)):
        raise ValueError(f"{name} must expose its token row through tolist().")
    rows = value.tolist()
    if (
        not isinstance(rows, list) or len(rows) != 1
        or not isinstance(rows[0], list) or len(rows[0]) != shape[1]
    ):
        raise ValueError(f"{name} values do not match its batch-one shape.")
    return rows[0]


def post_instruction_start(tokenizer: Any, inputs: Mapping[str, Any]) -> int:
    """Return the final fully formatted prompt token's zero-based index.

    Call this on the already bounded, unpadded, batch-one inputs before moving
    them to the GPU. The pinned Qwen no-thinking template closes the final user
    message with ``<|im_end|>`` and emits an assistant generation prefix with no
    further end marker. The rightmost marker therefore excludes all current
    user text, including literal control markers, and all older messages.

    The closing marker validates the native assistant prefix, but is not the
    steering boundary. Earlier prefix tokens are left untouched. Steering starts
    only at the last formatted prompt token, which predicts the first response
    token, then continues at generated content positions during cached decoding.
    Unsupported formatting raises ValueError instead of steering user content.
    """
    if "input_ids" not in inputs:
        raise ValueError("Chat inputs are missing input_ids.")
    ids = _single_unpadded_row(inputs["input_ids"], "input_ids")
    vocabulary_size = len(tokenizer)
    if vocabulary_size < 1 or any(type(token) is not int or not 0 <= token < vocabulary_size for token in ids):
        raise ValueError("Chat inputs contain invalid token IDs.")

    if "attention_mask" in inputs:
        mask = _single_unpadded_row(inputs["attention_mask"], "attention_mask")
        if len(mask) != len(ids) or any(value != 1 for value in mask):
            raise ValueError("Steering suffix lookup requires unpadded chat inputs.")

    marker_id = tokenizer.convert_tokens_to_ids(USER_END_MARKER)
    if (
        type(marker_id) is not int or not 0 <= marker_id < vocabulary_size
        or tokenizer.convert_ids_to_tokens(marker_id) != USER_END_MARKER
        or tokenizer.encode(USER_END_MARKER, add_special_tokens=False) != [marker_id]
    ):
        raise ValueError("Tokenizer does not support the user end marker as one known token.")

    for index in range(len(ids) - 1, -1, -1):
        if ids[index] == marker_id:
            if index == len(ids) - 1:
                raise ValueError("Chat template is missing the assistant generation suffix.")
            return len(ids) - 1
    raise ValueError("Chat template is missing the final user end marker.")
