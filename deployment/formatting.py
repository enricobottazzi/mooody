"""Locate the validated postinstruction steering region in native chat inputs."""

from __future__ import annotations

from typing import Any, Mapping


USER_END_MARKER = "<|im_end|>"


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
    """Return the inclusive start of the final user's template suffix.

    Call this on the already bounded, unpadded, batch-one inputs before moving
    them to the GPU. The pinned Qwen no-thinking template closes the final user
    message with ``<|im_end|>`` and emits an assistant generation prefix with no
    further end marker. The rightmost marker therefore excludes all current
    user text, including literal control markers, and all older messages.

    The returned index includes the user's closing marker, matching the suffix
    used by ``model_lab.abliteration.shared_suffix``. Steering applies from this
    index through the rest of prefill and then to generated token positions.
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
            return index
    raise ValueError("Chat template is missing the final user end marker.")
