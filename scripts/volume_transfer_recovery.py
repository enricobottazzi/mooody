"""Repair corrupt local download blocks using read-only Modal CPU functions.

The completed remote manifest remains authoritative. Each mounted-file source
is hashed in full before any recovery; only mismatched 8 MiB blocks are returned
through function results, bypassing the Volume file-download endpoint.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
from typing import Any

import modal


BLOCK_BYTES = 8 * 1024 * 1024
app = modal.App("mooody-volume-transfer-recovery")
volume = modal.Volume.from_name("mooody-model-lab")
cpu_image = modal.Image.debian_slim(python_version="3.12")


def _relative(remote_path: str, run_id: str) -> str:
    if not isinstance(run_id, str) or re.fullmatch(r"[A-Za-z0-9_-]+", run_id) is None:
        raise ValueError("Invalid transfer recovery run ID")
    if not isinstance(remote_path, str):
        raise ValueError("Transfer recovery remote path must be a string")
    prefix = f"/runs/{run_id}/"
    relative = remote_path[len(prefix):] if remote_path.startswith(prefix) else remote_path
    path = PurePosixPath(relative)
    if not relative or path.is_absolute() or ".." in path.parts:
        raise ValueError("Transfer recovery path is outside the requested run")
    return str(path)


def _inventory(path: Path) -> dict[str, Any]:
    before = path.stat()
    digest = hashlib.sha256()
    blocks = []
    offset = 0
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(BLOCK_BYTES), b""):
            digest.update(chunk)
            blocks.append({"index": len(blocks), "offset": offset, "bytes": len(chunk),
                           "sha256": hashlib.sha256(chunk).hexdigest()})
            offset += len(chunk)
    after = path.stat()
    identity = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
    if identity != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns) or offset != after.st_size:
        raise ValueError("Transfer recovery input changed while checksums were computed")
    return {"bytes": offset, "sha256": digest.hexdigest(), "blocks": blocks,
            "file_identity": identity}


@app.function(image=cpu_image, volumes={"/artifacts": volume}, cpu=1, memory=512, timeout=600)
def _source_inventory(run_id: str, remote_path: str, expected: dict[str, Any]) -> dict[str, Any]:
    relative = _relative(remote_path, run_id)
    root = Path("/artifacts/runs") / run_id
    raw_manifest = (root / "artifact_manifest.json").read_bytes()
    manifest = json.loads(raw_manifest)
    entry = manifest.get("files", {}).get(relative)
    if manifest.get("run_id") != run_id or manifest.get("status") != "completed" or not isinstance(entry, dict):
        raise ValueError("Remote transfer source has no completed run manifest entry")
    if entry.get("bytes") != expected["bytes"] or entry.get("sha256") != expected["sha256"]:
        raise ValueError("Remote transfer manifest differs from expected bytes/SHA256")
    result = _inventory(root / relative)
    if result["bytes"] != expected["bytes"] or result["sha256"] != expected["sha256"]:
        raise ValueError("Mounted remote source does not match the completed manifest")
    del result["file_identity"]
    result["manifest_sha256"] = hashlib.sha256(raw_manifest).hexdigest()
    return result


@app.function(image=cpu_image, volumes={"/artifacts": volume}, cpu=1, memory=512, timeout=120)
def _source_block(run_id: str, remote_path: str, block: dict[str, Any]) -> bytes:
    relative = _relative(remote_path, run_id)
    index, offset, size = block["index"], block["offset"], block["bytes"]
    if type(index) is not int or index < 0 or offset != index * BLOCK_BYTES or not 0 < size <= BLOCK_BYTES:
        raise ValueError("Invalid audited transfer block request")
    with (Path("/artifacts/runs") / run_id / relative).open("rb") as stream:
        stream.seek(offset)
        data = stream.read(size)
    if len(data) != size or hashlib.sha256(data).hexdigest() != block["sha256"]:
        raise ValueError("Mounted source block differs from the verified source inventory")
    return data


def _save_audit(path: Path, audit: dict[str, Any]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(audit, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def recover_file(
    local_partial: Path | str, relative_remote_path: str, expected: dict[str, Any], run_id: str,
) -> dict[str, Any]:
    """Patch a local partial, fsync it, and verify its complete manifest SHA256.

    Accepts either a run-relative path or /runs/<run_id>/<relative> as used by
    the downloader. The caller publishes the partial only after this succeeds.
    The audit sidecar is separate from the original completed artifact manifest.
    """
    relative = _relative(relative_remote_path, run_id)
    if (not isinstance(expected, dict) or type(expected.get("bytes")) is not int
            or expected["bytes"] < 0 or not isinstance(expected.get("sha256"), str)
            or re.fullmatch(r"[0-9a-f]{64}", expected["sha256"]) is None):
        raise ValueError("Transfer recovery requires manifest byte count and SHA256")
    target = Path(local_partial)
    if not target.is_file():
        raise ValueError("Transfer recovery requires an existing local partial file")
    audit_path = Path(str(target) + ".transfer_recovery.json")
    audit: dict[str, Any] = {
        "run_id": run_id, "relative_file": relative, "block_bytes": BLOCK_BYTES,
        "scope": "Local transfer repair from read-only mounted CPU function results; remote run unchanged",
        "expected_bytes": expected["bytes"], "expected_sha256": expected["sha256"],
        "status": "started", "recovered_blocks": [],
    }
    try:
        with app.run():
            source_call = _source_inventory.spawn(run_id, relative, expected)
            local = _inventory(target)
            source = source_call.get()
            audit.update({"source_manifest_sha256": source["manifest_sha256"],
                          "source_bytes": source["bytes"], "source_sha256": source["sha256"],
                          "local_before_bytes": local["bytes"], "local_before_sha256": local["sha256"]})
            blocks = source["blocks"]
            bad = [block for block in blocks
                   if block["index"] >= len(local["blocks"])
                   or local["blocks"][block["index"]] != block]
            audit["mismatched_blocks"] = [block["index"] for block in bad]
            current = target.stat()
            if local["file_identity"] != (current.st_dev, current.st_ino, current.st_size, current.st_mtime_ns):
                raise ValueError("Local partial changed before transfer recovery patching")
            with target.open("r+b") as stream:
                for block in bad:
                    data = _source_block.remote(run_id, relative, block)
                    if len(data) != block["bytes"] or hashlib.sha256(data).hexdigest() != block["sha256"]:
                        raise ValueError(f"Returned recovery block failed verification: {block['index']}")
                    stream.seek(block["offset"])
                    if stream.write(data) != len(data):
                        raise OSError("Recovery write did not write the entire verified block")
                    audit["recovered_blocks"].append(block)
                stream.truncate(expected["bytes"])
                stream.flush()
                os.fsync(stream.fileno())
        after = _inventory(target)
        audit.update({"local_after_bytes": after["bytes"], "local_after_sha256": after["sha256"]})
        if after["bytes"] != expected["bytes"] or after["sha256"] != expected["sha256"]:
            raise ValueError("Recovered local file still differs from the completed manifest")
        audit["status"] = "completed"
        _save_audit(audit_path, audit)
        print(f"Recovered {len(audit['recovered_blocks'])} transfer blocks; full file SHA256 verified: {relative}", flush=True)
        return audit
    except Exception as error:
        audit.update({"status": "failed", "error_type": type(error).__name__, "error": str(error)})
        _save_audit(audit_path, audit)
        raise
