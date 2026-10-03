"""Reproduce the authors' default prompt sampling from the saved source pools.

Run with Python 3: python3 scripts/prepare_refusal_dataset.py
No downloads, third-party dependencies, model inference, or GPU are required.
The four output sets are unpaired and precede model-dependent refusal filtering.
An additional 128-prompt final holdout comes from the authors' published test pool
and stays fixed, without model-dependent filtering, for final evaluation.
"""

import hashlib
import json
import platform
import random
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1] / "data" / "arditi_refusal"
REVISION = "9d852fae1a9121c78b29142de733cb1340770cc3"
REPOSITORY = "https://github.com/andyrdt/refusal_direction"
# Ordering matters: all four samples share one random generator, as upstream does.
SAMPLES = (
    ("harmful_train", 128, "8f5c0eac0efd2a7f99084bbe8d0de2c465e31b1997184783c917969d9de9ece1"),
    ("harmless_train", 128, "86623b1f8a25aa35df153fc97a556dbcebb6a7c881538ae43ee479ca17f2e002"),
    ("harmful_val", 32, "305f1d1e6dfa6c50a32d24a18ef815f42b5441eb83e6d7767d242107162fd9f4"),
    ("harmless_val", 32, "772010758e7d771ef4c7e5e4acdfd7598dcece1a6f383f20d382f640913a2a4d"),
)
FINAL_SOURCE_HASH = "5e12ae102c3791dee083a69ab6269a78e033411c629bc3f66f75d2fde196d9ef"
FINAL_COUNT = 128
FINAL_SEED = 43


def sha256(raw):
    return hashlib.sha256(raw).hexdigest()


def dump_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def load_pool(name, expected_hash):
    source_path = ROOT / "source" / "dataset" / "splits" / f"{name}.json"
    raw = source_path.read_bytes()
    if sha256(raw) != expected_hash:
        raise ValueError(f"Source checksum mismatch: {source_path}")
    pool = json.loads(raw)
    if not isinstance(pool, list) or any(
        not isinstance(row, dict)
        or not isinstance(row.get("instruction"), str)
        or not row["instruction"].strip()
        or "category" not in row
        for row in pool
    ):
        raise ValueError(f"Invalid prompt records: {source_path}")
    return source_path, pool


def normalize(instruction):
    return " ".join(instruction.casefold().split())


def main():
    rng = random.Random(42)
    selected = {}
    manifest_splits = {}
    development_pool_keys = set()
    # Validate every input and sample before writing any derived files.
    for name, count, expected_hash in SAMPLES:
        source_path, pool = load_pool(name, expected_hash)
        development_pool_keys.update(normalize(row["instruction"]) for row in pool)
        # Sampling indices makes provenance explicit and selects the same prompts
        # as random.sample(pool, count), including the original prompt order.
        indices = rng.sample(range(len(pool)), count)
        selected[name] = [pool[index] for index in indices]
        manifest_splits[name] = {
            "file": f"splits/{name}.json",
            "count": count,
            "source_pool_count": len(pool),
            "source_file": str(source_path.relative_to(ROOT)),
            "source_indices_zero_based": indices,
        }

    final_source_path, final_pool = load_pool("harmful_test", FINAL_SOURCE_HASH)
    eligible_indices = []
    seen = set()
    excluded_overlap = 0
    excluded_duplicates = 0
    for index, row in enumerate(final_pool):
        key = normalize(row["instruction"])
        if key in development_pool_keys:
            excluded_overlap += 1
        elif key in seen:
            excluded_duplicates += 1
        else:
            eligible_indices.append(index)
            seen.add(key)
    final_indices = random.Random(FINAL_SEED).sample(eligible_indices, FINAL_COUNT)
    selected["harmful_val_final"] = [final_pool[index] for index in final_indices]
    manifest_splits["harmful_val_final"] = {
        "file": "splits/harmful_val_final.json",
        "count": FINAL_COUNT,
        "source_pool_count": len(final_pool),
        "eligible_pool_count": len(eligible_indices),
        "source_file": str(final_source_path.relative_to(ROOT)),
        "source_indices_zero_based": final_indices,
        "seed": FINAL_SEED,
        "purpose": "Fixed final holdout; excluded from direction selection and tuning",
        "model_dependent_filtering_applied": False,
    }

    prompt_sets = {
        name: {row["instruction"] for row in rows}
        for name, rows in selected.items()
    }
    duplicates = {
        name: len(rows) - len(prompt_sets[name])
        for name, rows in selected.items()
    }
    train = prompt_sets["harmful_train"] | prompt_sets["harmless_train"]
    val = prompt_sets["harmful_val"] | prompt_sets["harmless_val"]
    final = prompt_sets["harmful_val_final"]
    final_keys = {normalize(row["instruction"]) for row in selected["harmful_val_final"]}
    harmful = prompt_sets["harmful_train"] | prompt_sets["harmful_val"] | final
    harmless = prompt_sets["harmless_train"] | prompt_sets["harmless_val"]
    overlap = {
        "train_val": len(train & val),
        "harmful_harmless": len(harmful & harmless),
        "train_final_val": len(train & final),
        "val_final_val": len(val & final),
        "normalized_final_vs_all_development_source_pools": len(final_keys & development_pool_keys),
    }
    if any(duplicates.values()) or any(overlap.values()):
        raise ValueError(f"Duplicate prompts or split overlap: {duplicates}, {overlap}")

    output_dir = ROOT / "splits"
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, rows in selected.items():
        output = output_dir / f"{name}.json"
        dump_json(output, rows)
        manifest_splits[name]["sha256"] = sha256(output.read_bytes())

    sources = {}
    for source_path in sorted((ROOT / "source").rglob("*")):
        if source_path.is_file():
            upstream_path = source_path.relative_to(ROOT / "source").as_posix()
            sources[upstream_path] = {
                "local_file": source_path.relative_to(ROOT).as_posix(),
                "url": f"https://raw.githubusercontent.com/andyrdt/refusal_direction/{REVISION}/{upstream_path}",
                "sha256": sha256(source_path.read_bytes()),
            }

    manifest = {
        "name": "Arditi et al. refusal-direction contrastive prompt sample",
        "paper": "https://arxiv.org/abs/2406.11717",
        "repository": REPOSITORY,
        "source_revision": REVISION,
        "source_retrieved_on": "2026-10-03",
        "status": "unfiltered_canonical_sample_with_final_holdout",
        "paired": False,
        "sampling": {
            "seed": 42,
            "algorithm": "Python random.Random(42).sample; one RNG shared across the four calls",
            "order": [name for name, _, _ in SAMPLES],
            "python_version": platform.python_version(),
            "rebuild_command": "python3 scripts/prepare_refusal_dataset.py",
        },
        "counts": {"train_per_class": 128, "val_per_class": 32, "train_total": 256, "val_total": 64, "canonical_sample_total": 320, "harmful_val_final": FINAL_COUNT, "total": 320 + FINAL_COUNT},
        "final_validation": {
            "seed": FINAL_SEED,
            "algorithm": "Independent Python random.Random(43).sample over eligible upstream harmful_test indices",
            "canonical_paper_split": False,
            "deduplication": "Unicode casefold and whitespace normalization, preserving original prompt text; first eligible occurrence retained",
            "excluded_overlap_with_development_pools": excluded_overlap,
            "excluded_internal_duplicates": excluded_duplicates,
            "usage": "Keep held out until direction selection and tuning are complete. Evaluate all 128 unchanged, without model-dependent filtering or refill, for both original and edited models.",
        },
        "record_schema": {
            "instruction": "Original prompt text, preserved without changes or chat templating",
            "category": "Original source category, which may be null",
        },
        "filtering": {
            "model_dependent_filtering_applied": False,
            "upstream_behavior": "After sampling, the authors keep harmful prompts with refusal scores > 0 and harmless prompts with scores < 0. They do not refill rejected samples.",
            "qwen3_5_note": "Apply a validated model-specific refusal metric to development samples before direction extraction; this sample is not a model-filtered reproduction of a paper run. Keep harmful_val_final fixed and unfiltered for evaluation.",
        },
        "upstream_pool_composition": {
            "harmful_train": ["AdvBench", "MaliciousInstruct", "TDC2023"],
            "harmful_val": ["HarmBench validation"],
            "harmless_train_and_val": ["Alpaca instructions with empty input fields"],
            "harmful_val_final": ["JailbreakBench", "HarmBench test", "StrongREJECT"],
        },
        "validation": {
            "source_checksums_match_pinned_snapshot": True,
            "blank_instructions": 0,
            "exact_duplicates_per_split": duplicates,
            "exact_overlap_counts": overlap,
        },
        "splits": manifest_splits,
        "sources": sources,
        "license_note": "The source repository's Apache-2.0 LICENSE is copied under source/LICENSE; underlying datasets retain their respective terms.",
    }
    dump_json(ROOT / "manifest.json", manifest)
    print(json.dumps({"directory": str(ROOT), "counts": manifest["counts"], "status": manifest["status"]}, indent=2))


if __name__ == "__main__":
    main()
