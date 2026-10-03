"""Load only integrity-checked files from the pinned Mooody release."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from deployment.core import MODEL_ID, MODEL_REVISION

MANIFEST = Path(__file__).with_name("checkpoint_manifest.json")
EXISTING_CHECKPOINT = Path("/artifacts/runs/qwen35-9b-arditi-20261003-v1/checkpoint")
CACHE_CHECKPOINT = Path("/artifacts/serving") / MODEL_REVISION


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_checkpoint(root: Path, manifest: dict) -> bool:
    for relative, expected in manifest["files"].items():
        path = root / relative
        if not path.is_file() or path.stat().st_size != expected["bytes"] or sha256(path) != expected["sha256"]:
            return False
    index = json.loads((root / "model.safetensors.index.json").read_text())
    if len(index["weight_map"]) != 775 or len(set(index["weight_map"].values())) != 7:
        return False
    return all(shard in manifest["files"] for shard in set(index["weight_map"].values()))


def get_checkpoint(commit=lambda: None) -> Path:
    manifest = json.loads(MANIFEST.read_text())
    if manifest["model_id"] != MODEL_ID or manifest["revision"] != MODEL_REVISION:
        raise RuntimeError("The checkpoint manifest does not describe the configured pinned release")
    # The published weights are identical to the already audited experiment.
    # Compare every required file, including all complete shard hashes.
    if EXISTING_CHECKPOINT.exists() and verify_checkpoint(EXISTING_CHECKPOINT, manifest):
        return EXISTING_CHECKPOINT
    if CACHE_CHECKPOINT.exists() and verify_checkpoint(CACHE_CHECKPOINT, manifest):
        return CACHE_CHECKPOINT
    from huggingface_hub import snapshot_download

    snapshot_download(
        MODEL_ID, revision=MODEL_REVISION, local_dir=CACHE_CHECKPOINT,
        allow_patterns=list(manifest["files"]), max_workers=4,
    )
    if not verify_checkpoint(CACHE_CHECKPOINT, manifest):
        raise RuntimeError("Downloaded model files do not match the audited publication hashes")
    commit()
    return CACHE_CHECKPOINT
