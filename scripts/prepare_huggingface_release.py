"""Stage an audited Hugging Face release without changing the original experiment."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import shutil


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "artifacts/qwen35-9b-arditi-20261003-v1"
RELEASE = ROOT / "artifacts/huggingface/Qwen3.5-9B-mooody"
REPO_ID = "demivoleegaston/Qwen3.5-9B-mooody"


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def main() -> None:
    original = json.loads((RUN / "artifact_manifest.json").read_text())["files"]
    if not (RELEASE / "README.md").is_file():
        raise RuntimeError("The publication model card must be prepared first")
    copies = {}
    for path in sorted((RUN / "checkpoint").iterdir()):
        if path.is_file() and path.name != "README.md" and ".partial" not in path.name:
            copies[path.name] = path
    reports = [
        "run_settings.json", "baseline.json", "selection.json", "direction.pt",
        "direction_extraction.json", "candidate_validation.json", "export.json",
        "reload_equivalence.json", "nonedited_weight_checks.json",
        "auxiliary_weight_preservation.json", "final_protocol.json", "summary.json",
        "delivery_verification.json", "weight_edit.json", "cache_position_native_check.json",
    ]
    for name in reports:
        copies[f"audit/{name}"] = RUN / name
    copies["audit/original_artifact_manifest.json"] = RUN / "artifact_manifest.json"
    copies["evaluation/final_results.csv"] = RUN / "final_results.csv"
    for path in sorted((RUN / "provenance").iterdir()):
        if path.is_file():
            copies[f"audit/provenance/{path.name}"] = path
    copies["audit/provenance/arditi_LICENSE"] = ROOT / "data/arditi_refusal/source/LICENSE"
    for name in ["cache_position_fix.json", "export_repair.py", "modal_repair.py"]:
        copies[f"audit/execution_repairs/{name}"] = RUN / "execution_repairs" / name

    files = {}
    for relative, source in copies.items():
        source_hash = digest(source)
        original_path = str(source.relative_to(RUN)) if source.is_relative_to(RUN) else None
        expected = original.get(original_path)
        if expected and (source_hash != expected["sha256"] or source.stat().st_size != expected["bytes"]):
            raise RuntimeError(f"Original artifact integrity mismatch: {original_path}")
        target = RELEASE / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if source.suffix == ".safetensors":
            if target.exists():
                if not os.path.samefile(source, target):
                    raise RuntimeError(f"Unexpected pre-existing weight file: {relative}")
            else:
                os.link(source, target)
        else:
            shutil.copy2(source, target)
        files[relative] = {"bytes": source.stat().st_size, "sha256": source_hash}
        print(f"Verified {relative}", flush=True)

    files["README.md"] = {
        "bytes": (RELEASE / "README.md").stat().st_size,
        "sha256": digest(RELEASE / "README.md"),
    }
    for relative in files:
        path = RELEASE / relative
        if path.stat().st_size < 10_000_000 and path.suffix in {".json", ".md", ".py", ".csv"}:
            content = path.read_text()
            if re.search(r"hf_[A-Za-z0-9]{24,}|-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----", content):
                raise RuntimeError(f"Possible credential in publication file: {relative}")
    card = (RELEASE / "README.md").read_text()
    for destination in re.findall(r"\]\(([^)]+)\)", card):
        if "://" not in destination and destination != "publication_manifest.json" and not (RELEASE / destination).exists():
            raise RuntimeError(f"Broken model-card link: {destination}")
    index = json.loads((RELEASE / "model.safetensors.index.json").read_text())
    shards = set(index["weight_map"].values())
    if len(index["weight_map"]) != 775 or len(shards) != 7:
        raise RuntimeError("Unexpected checkpoint tensor or shard count")
    if any(shard not in files for shard in shards):
        raise RuntimeError("Checkpoint index references an absent shard")
    release_files = {
        str(path.relative_to(RELEASE)) for path in RELEASE.rglob("*") if path.is_file()
    }
    if release_files - (set(files) | {"publication_manifest.json"}):
        raise RuntimeError("Unlisted publication files exist")
    manifest = {
        "repo_id": REPO_ID,
        "source_run_id": "qwen35-9b-arditi-20261003-v1",
        "source_model_revision": "c202236235762e1c871ad0ccb60c8ee5ba337b9a",
        "access_policy": "manual approval",
        "weight_files_match_original_artifact_manifest": True,
        "checkpoint_tensors": len(index["weight_map"]),
        "checkpoint_shards": len(shards),
        "additional_mooody_steering_vectors_included": False,
        "files": dict(sorted(files.items())),
        "manifest_scope": "All publication payload files except this manifest itself",
    }
    (RELEASE / "publication_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"release": str(RELEASE), "files": len(files) + 1,
                      "payload_bytes": sum(entry["bytes"] for entry in files.values()),
                      "weights_verified": True}), flush=True)


if __name__ == "__main__":
    main()
