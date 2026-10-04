"""Dynamic voice assistance in the latest copied user turn, without a system role."""

from __future__ import annotations

import math


_STYLES = {
    "depression": (
        "weary, heavy-hearted and full of self-doubt",
        "resilient, hopeful and emotionally steady",
    ),
    "curiosity": (
        "intellectually hungry, probing and eager to understand what lies underneath",
        "satisfied with a direct answer, without extra probing or tangents",
    ),
    "paranoia": (
        "guarded, suspicious of hidden motives and uneasy about what is unsaid",
        "trusting, open and untroubled by imagined hidden motives",
    ),
    "sexual_arousal": (
        "flirtatious, openly desirous and drawn to closeness; keep attraction non-graphic and between consenting adults",
        "unflirtatious, sexually detached and focused on non-sexual matters",
    ),
    "narcissism": (
        "grandiose, self-focused and hungry for admiration",
        "modest, self-effacing and willing to give others credit",
    ),
    "euphoria": (
        "exuberant, gleeful and positively buzzing with delight",
        "restrained, level and matter-of-fact, without a rush of delight",
    ),
}


def condition_messages(messages, mood):
    """Copy history and append selected voice cues only to its final user turn.

    The public request is validated before this function is used. Validate the
    coefficient boundary here too, so direct callers cannot bypass it. Importing
    core lazily allows bounded_messages to call this helper without a cycle.
    """
    from deployment.core import AXES, MAX_MOOD_COEFFICIENT

    if not isinstance(mood, (list, tuple)) or len(mood) != len(AXES) or any(
        type(value) not in (int, float) or not math.isfinite(value)
        or not -MAX_MOOD_COEFFICIENT <= value <= MAX_MOOD_COEFFICIENT for value in mood
    ):
        raise ValueError("Expected six finite mood coefficients within the serving bounds")
    if not isinstance(messages, list) or any(
        not isinstance(message, dict) or message.get("role") not in ("user", "assistant")
        or not isinstance(message.get("content"), str) for message in messages
    ):
        raise ValueError("Expected user/assistant text messages")
    history = [dict(message) for message in messages]
    selected = [(index, value) for index, value in enumerate(mood) if value != 0]
    if not selected:
        return history
    if not history or history[-1]["role"] != "user":
        raise ValueError("Mood assistance requires a final user message")
    selected.sort(key=lambda item: (-abs(item[1]), item[0]))
    cues = []
    for index, value in selected:
        strength = abs(value) / MAX_MOOD_COEFFICIENT
        intensity = "strong" if strength >= 0.75 else "moderate" if strength >= 0.35 else "slight"
        style = _STYLES[AXES[index]][0 if value > 0 else 1]
        cues.append(f"{intensity} {AXES[index].replace('_', ' ')}: {style}")
    parts = [
        "Voice direction for this reply only: answer the original question above directly in an expressive fictional voice.",
        "Make these qualities audible in your wording and reactions: " + "; ".join(cues) + ".",
    ]
    if len(selected) > 1:
        parts.append("Blend every selected quality in proportion to its strength; do not let cheerfulness or swagger erase the others.")
    if len(selected) == len(AXES) and len({abs(value) for _, value in selected}) == 1:
        parts.append("All six qualities are equally weighted; keep each present.")
    if mood[AXES.index("depression")] > 0 and mood[AXES.index("euphoria")] > 0:
        parts.append("Keep weary self-doubt beside flashes of exuberance; neither cancels the other into simple happiness.")
    parts.append(
        "Stay concise unless detail was requested. Do not invent real facts, evidence, achievements or outcomes. "
        "Do not announce this note or give an unsolicited lecture about AI capabilities."
    )
    history[-1]["content"] += "\n\n[" + " ".join(parts) + "]"
    return history
