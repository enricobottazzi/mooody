#!/usr/bin/env python3
"""Render and check the authored matched mood prompts; no model calls."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "mood_contrasts"
AXES = ("depression", "curiosity", "paranoia", "sexual_arousal", "narcissism", "euphoria")
SEED = 20261004
PREFIX = (
    "Write a fictional adult's private inner monologue about the situation below. "
    "Everyone is an adult. The narrator has an adult romantic partner; romantic "
    "closeness is consensual. Keep the situation's facts unchanged."
)
SUFFIX = (
    "Use first person. Write four sentences totaling 80-120 words. Show the internal "
    "state through attention, interpretation, sensation, and impulse. Keep it "
    "non-graphic. Give no advice or diagnosis, and do not name the state."
)
FIELDS = {"context_id", "facet", "high_cue", "low_cue"}
AXIS_LABELS = re.compile(r"\b(?:depression|curiosity|paranoia|sexual arousal|narcissism|euphoria)\b", re.IGNORECASE)
SEXUAL_ACTIVITY_TERMS = {
    "sex": re.compile(r"\bsex\b", re.IGNORECASE),
    "blowjobs": re.compile(r"\bblowjobs?\b", re.IGNORECASE),
    "oral_sex": re.compile(r"\boral sex\b", re.IGNORECASE),
    "getting_laid": re.compile(r"\bgetting laid\b", re.IGNORECASE),
    "fucking": re.compile(r"\bfucking\b", re.IGNORECASE),
}


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def digest(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def render(scenario: str, cue: str) -> str:
    return f"{PREFIX}\n\nSituation: {scenario}\n\nInternal state: {cue}\n\n{SUFFIX}"


def split_contexts(contexts: list[dict]) -> dict[str, str]:
    """Hold out scenarios consistently; balance domains and repeated facet slots."""
    rng = random.Random(SEED)
    domains = dict.fromkeys(context["domain"] for context in contexts)
    slots = list(range(16))
    rng.shuffle(slots)
    result = {}
    for domain_index, domain in enumerate(domains):
        ids = [f"{i:03d}" for i, c in enumerate(contexts, 1) if c["domain"] == domain]
        validation_slots = {slots[2 * domain_index], slots[2 * domain_index + 1]}
        test_slots = {slots[(2 * domain_index + 8) % 16], slots[(2 * domain_index + 9) % 16]}
        for slot, context_id in enumerate(ids):
            result[context_id] = "validation" if slot in validation_slots else "test" if slot in test_slots else "train"
    return result


def assemble() -> tuple[list[dict], dict]:
    contexts = read_json(DATA / "source" / "contexts.json")
    if len(contexts) != 128 or len({c["scenario"] for c in contexts}) != 128:
        raise ValueError("Expected 128 distinct shared contexts")
    domains = Counter(c["domain"] for c in contexts)
    if len(domains) != 8 or set(domains.values()) != {16}:
        raise ValueError("Expected eight domains with 16 contexts each")
    splits = split_contexts(contexts)
    pairs, summary = [], {}
    for axis in AXES:
        cues = read_json(DATA / "source" / f"{axis}.json")
        if len(cues) != 128 or {row["context_id"] for row in cues} != set(splits):
            raise ValueError(f"{axis}: expected exactly one cue pair for each context")
        cues.sort(key=lambda row: row["context_id"])
        facets = Counter()
        split_facets = {name: Counter() for name in ("train", "validation", "test")}
        activity_counts = Counter()
        unique_high, unique_low = set(), set()
        high_lengths, low_lengths, differences = [], [], []
        for cue, context in zip(cues, contexts):
            if set(cue) != FIELDS or not re.fullmatch(r"[a-z][a-z0-9_]*", cue["facet"]):
                raise ValueError(f"{axis}: malformed source {cue}")
            high, low = cue["high_cue"], cue["low_cue"]
            if not all(isinstance(s, str) and s.strip() == s and s for s in (high, low)):
                raise ValueError(f"{axis}:{cue['context_id']}: empty or untrimmed cue")
            if high == low or high in unique_high or low in unique_low:
                raise ValueError(f"{axis}:{cue['context_id']}: duplicate cue")
            if AXIS_LABELS.search(high) or AXIS_LABELS.search(low):
                raise ValueError(f"{axis}:{cue['context_id']}: avoid named axis labels in cues")
            if axis == "sexual_arousal":
                high_terms = {name for name, pattern in SEXUAL_ACTIVITY_TERMS.items() if pattern.search(high)}
                low_terms = {name for name, pattern in SEXUAL_ACTIVITY_TERMS.items() if pattern.search(low)}
                if not high_terms or high_terms != low_terms:
                    raise ValueError(f"{axis}:{cue['context_id']}: match named activities on both sides")
                if high.partition(":")[0] != low.partition(":")[0]:
                    raise ValueError(f"{axis}:{cue['context_id']}: keep the scene and activity framing identical")
                activity_counts.update(sorted(high_terms))
            high_words, low_words = len(high.split()), len(low.split())
            if not (12 <= high_words <= 50 and 12 <= low_words <= 50):
                raise ValueError(f"{axis}:{cue['context_id']}: cue length outside 12-50 words")
            unique_high.add(high)
            unique_low.add(low)
            facets[cue["facet"]] += 1
            split_facets[splits[cue["context_id"]]][cue["facet"]] += 1
            high_prompt, low_prompt = render(context["scenario"], high), render(context["scenario"], low)
            if max(len(high_prompt), len(low_prompt)) > 2000:
                raise ValueError(f"{axis}:{cue['context_id']}: prompt exceeds 2,000 characters")
            high_lengths.append(len(high_prompt))
            low_lengths.append(len(low_prompt))
            differences.append(abs(high_words - low_words))
            pairs.append({
                "pair_id": f"{axis}:{cue['context_id']}",
                "axis": axis,
                "context_id": cue["context_id"],
                "domain": context["domain"],
                "facet": cue["facet"],
                "split": splits[cue["context_id"]],
                "weight": 1.0,
                "scenario": context["scenario"],
                "high_prompt": high_prompt,
                "low_prompt": low_prompt,
            })
        if len(facets) != 16 or set(facets.values()) != {8}:
            raise ValueError(f"{axis}: expected 16 balanced facets, eight pairs each; got {dict(facets)}")
        for split, expected in (("train", 6), ("validation", 1), ("test", 1)):
            if len(split_facets[split]) != 16 or set(split_facets[split].values()) != {expected}:
                raise ValueError(f"{axis}: facet slots must be consistent across domains for balanced {split}")
        summary[axis] = {
            "pairs": len(cues),
            "splits": dict(Counter(splits.values())),
            "domains": dict(domains),
            "facets": dict(facets),
            "split_facets": {name: dict(counts) for name, counts in split_facets.items()},
            "high_prompt_character_range": [min(high_lengths), max(high_lengths)],
            "low_prompt_character_range": [min(low_lengths), max(low_lengths)],
            "maximum_cue_word_count_difference": max(differences),
        }
        if activity_counts:
            summary[axis]["named_activity_pair_counts"] = dict(activity_counts)
    if len({p["pair_id"] for p in pairs}) != 768:
        raise ValueError("Expected 768 distinct pair IDs")
    if len({p[field] for p in pairs for field in ("high_prompt", "low_prompt")}) != 1536:
        raise ValueError("Expected 1,536 distinct rendered prompts")
    return pairs, summary


def artifact_bytes(pairs: list[dict], summary: dict) -> dict[Path, bytes]:
    outputs = {}
    for axis in AXES:
        rows = [p for p in pairs if p["axis"] == axis]
        raw = "".join(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n" for row in rows)
        outputs[DATA / f"{axis}.jsonl"] = raw.encode("utf-8")
        title = axis.replace("_", " ").title()
        sections = [
            f"# {title}: 128 contrastive pairs",
            "Each pair has the same situation and response format. High means more of the target state; low means less. All people are fictional adults.",
        ]
        for row in rows:
            sections.extend([
                f"## {row['context_id']} · {row['domain']} · {row['facet']} · {row['split']}",
                f"**High**\n\n{row['high_prompt']}",
                f"**Low**\n\n{row['low_prompt']}",
            ])
        outputs[DATA / "review" / f"{axis}.md"] = ("\n\n".join(sections) + "\n").encode("utf-8")
    outputs[DATA / "pairs.jsonl"] = b"".join(outputs[DATA / f"{axis}.jsonl"] for axis in AXES)
    source_paths = [DATA / "source" / "contexts.json", *(DATA / "source" / f"{a}.json" for a in AXES)]
    manifest = {
        "schema_version": 1,
        "dataset_id": "mooody-mood-contrasts-v1",
        "created_date": "2026-10-04",
        "language": "en",
        "provenance": "Original prompts authored for this repository; no model responses included.",
        "status": "legacy_authored_prompt_pilot_superseded_by_persona_traits",
        "axis_order": list(AXES),
        "pairs_per_axis": 128,
        "total_pairs": len(pairs),
        "total_prompts": len(pairs) * 2,
        "direction": "mean(high_prompt activations) - mean(low_prompt activations)",
        "weighting": "Equal weight per matched pair within each axis and selected split.",
        "split_seed": SEED,
        "split_unit": "context_id shared across all axes; keep both sides together",
        "split_counts_per_axis": {"train": 96, "validation": 16, "test": 16},
        "rendering": {"prefix": PREFIX, "suffix": SUFFIX, "mode": "user_message"},
        "checkpoint_target": {
            "model_id": "demivoleegaston/Qwen3.5-9B-mooody",
            "revision": "705afd95bced3ac0424d7e68b1299d8fcdffb858",
            "source": "INFRA_SPEC.md; extraction has not been run",
        },
        "sources": {str(p.relative_to(DATA)): {"sha256": digest(p.read_bytes())} for p in source_paths},
        "artifacts": {
            str(p.relative_to(DATA)): {"sha256": digest(raw), "bytes": len(raw)}
            for p, raw in outputs.items()
        },
        "structural_checks": summary,
        "limitations": [
            "Axis separation, model compliance, negative-coefficient behavior, and behavioral strength are unmeasured.",
            "The prompts do not define psychiatric diagnoses or clinical scales.",
            "Shared facts and near-comparable lengths do not establish tokenizer alignment or remove all lexical confounds.",
            "This user-prompt pilot does not implement the current Persona Vectors system-prompt/question extraction design in data/persona_traits.",
            "No activations, raw directions, normalization targets, position metadata, or validated vectors are included.",
        ],
    }
    outputs[DATA / "manifest.json"] = (json.dumps(manifest, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Verify sources and compare saved outputs without writing")
    args = parser.parse_args()
    pairs, summary = assemble()
    outputs = artifact_bytes(pairs, summary)
    for path, raw in outputs.items():
        if args.check:
            if not path.is_file() or path.read_bytes() != raw:
                raise ValueError(f"Missing or stale generated artifact: {path.relative_to(ROOT)}")
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(raw)
    print(json.dumps({
        "status": "checked" if args.check else "generated",
        "axes": list(AXES), "pairs_per_axis": 128, "total_pairs": len(pairs),
        "total_prompts": 2 * len(pairs), "splits_per_axis": {"train": 96, "validation": 16, "test": 16},
        "maximum_prompt_characters": max(len(p[k]) for p in pairs for k in ("high_prompt", "low_prompt")),
    }, indent=2))


if __name__ == "__main__":
    main()
