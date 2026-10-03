"""Publish verified existing Modal-volume weights without a GPU or local weight transfer."""

from __future__ import annotations

import json
from pathlib import Path
import shutil

import modal
from huggingface_hub import get_token


ROOT = Path(__file__).resolve().parents[1]
RELEASE = ROOT / "artifacts/huggingface/Qwen3.5-9B-mooody"
OVERLAY = ROOT / "artifacts/huggingface/modal_publication_overlay"


def publish() -> dict:
    import hashlib
    import json
    from pathlib import Path
    import shutil

    from huggingface_hub import HfApi

    repo_id = "demivoleegaston/Qwen3.5-9B-mooody"
    api = HfApi()
    if api.whoami()["name"] != "demivoleegaston":
        raise RuntimeError("Unexpected publishing account")
    info = api.model_info(repo_id)
    if info.gated != "manual" or not info.private:
        raise RuntimeError("Publication must begin in a private, manually gated repository")
    target = Path("/tmp/mooody-publication")
    shutil.copytree("/publication_overlay", target)
    manifest = json.loads((target / "publication_manifest.json").read_text())
    source = Path("/artifacts/runs/qwen35-9b-arditi-20261003-v1/checkpoint")
    for relative, expected in manifest["files"].items():
        path = target / relative
        if relative.endswith(".safetensors"):
            path.symlink_to(source / relative)
        sha = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(16 * 1024 * 1024), b""):
                sha.update(chunk)
        if path.stat().st_size != expected["bytes"] or sha.hexdigest() != expected["sha256"]:
            raise RuntimeError(f"Cloud publication integrity mismatch: {relative}")
        print(f"Cloud verified {relative}", flush=True)
    print("All cloud sources match the publication manifest; starting upload", flush=True)
    commit = api.upload_folder(
        repo_id=repo_id,
        repo_type="model",
        folder_path=target,
        ignore_patterns=[".cache/*"],
        commit_message="Publish Qwen3.5-9B-mooody checkpoint and evaluation",
    )
    after = api.model_info(repo_id, files_metadata=True)
    if after.gated != "manual" or not after.private:
        raise RuntimeError("Access settings changed during upload")
    return {"repo_id": repo_id, "commit": commit.oid, "gated": after.gated,
            "private": after.private, "cloud_payload_hashes_verified": True,
            "release_files": len(manifest["files"]) + 1,
            "method": "CPU upload from existing verified Modal volume"}


def main() -> None:
    token = get_token()
    if token is None:
        raise RuntimeError("A local Hugging Face login is required")
    OVERLAY.mkdir(parents=True, exist_ok=True)
    for source in RELEASE.rglob("*"):
        if not source.is_file() or source.suffix == ".safetensors" or ".cache" in source.parts:
            continue
        target = OVERLAY / source.relative_to(RELEASE)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    app = modal.App("mooody-huggingface-publication")
    image = (
        modal.Image.debian_slim(python_version="3.12")
        .pip_install("huggingface_hub==2.1.1")
        .env({"HF_XET_HIGH_PERFORMANCE": "1", "HF_HUB_DISABLE_PROGRESS_BARS": "1"})
        .add_local_dir(OVERLAY, remote_path="/publication_overlay", copy=True)
    )
    function = app.function(
        image=image,
        volumes={"/artifacts": modal.Volume.from_name("mooody-model-lab")},
        secrets=[modal.Secret.from_dict({"HF_TOKEN": token})],
        cpu=8, memory=16384, timeout=7200, max_containers=1, scaledown_window=10,
    )(publish)
    del token
    with modal.enable_output(), app.run():
        result = function.remote()
    output = ROOT / "artifacts/huggingface/cloud_upload_receipt.json"
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
