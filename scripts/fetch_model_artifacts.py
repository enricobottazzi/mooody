"""Download and verify completed experiment artifacts from the Modal Volume.

Run: .venv/bin/python scripts/fetch_model_artifacts.py
Use --reports-only to omit the approximately 19 GB model checkpoint.
No credentials or model completions are printed.
"""

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath

import modal


RUN_ID = "qwen35-9b-arditi-20261003-v1"
ROOT = Path(__file__).resolve().parents[1] / "artifacts" / RUN_ID
REMOTE_ROOT = f"/runs/{RUN_ID}"


def checksum(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(volume, relative, expected=None):
    path = PurePosixPath(relative)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"Invalid artifact path: {relative}")
    target = ROOT / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    if expected and target.is_file() and target.stat().st_size == expected["bytes"] and checksum(target) == expected["sha256"]:
        print(f"Verified existing {relative}", flush=True)
        return target
    temporary = target.with_name(target.name + ".partial")
    # The SDK's file path writes parallel blocks at explicit offsets and
    # verifies advertised block digests. Verify the resulting disk file too.
    with temporary.open("wb+") as stream:
        volume.read_file_into_fileobj(f"{REMOTE_ROOT}/{relative}", stream)
    size = temporary.stat().st_size
    digest = checksum(temporary)
    if expected and (size != expected["bytes"] or digest != expected["sha256"]):
        from volume_transfer_recovery import recover_file

        recover_file(temporary, f"{REMOTE_ROOT}/{relative}", expected, RUN_ID)
        size = temporary.stat().st_size
        digest = checksum(temporary)
        if size != expected["bytes"] or digest != expected["sha256"]:
            raise RuntimeError(f"Downloaded artifact failed checksum or size verification: {relative}")
    temporary.replace(target)
    print(f"Downloaded {relative}: {size:,} bytes", flush=True)
    return target


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reports-only", action="store_true")
    arguments = parser.parse_args()
    volume = modal.Volume.from_name("mooody-model-lab")
    manifest_path = download(volume, "artifact_manifest.json")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("run_id") != RUN_ID or manifest.get("status") != "completed":
        raise ValueError("Artifact manifest does not describe the completed requested run")
    files = manifest["files"]
    if not isinstance(files, dict):
        raise ValueError("Artifact manifest must map relative filenames to bytes and sha256")
    for relative, expected in sorted(files.items()):
        if arguments.reports_only and relative.startswith("checkpoint/"):
            continue
        download(volume, relative, expected)
    print(f"Artifacts verified in {ROOT}", flush=True)


if __name__ == "__main__":
    main()
