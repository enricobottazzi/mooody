"""Verify a publication against the local release manifest and live access settings."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from huggingface_hub import HfApi, hf_hub_download


ROOT = Path(__file__).resolve().parents[1]
RELEASE = ROOT / "artifacts/huggingface/Qwen3.5-9B-mooody"
REPO_ID = "demivoleegaston/Qwen3.5-9B-mooody"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--expect-private", action="store_true")
    args = parser.parse_args()
    api = HfApi()
    identity = api.whoami()
    if identity["name"] != "demivoleegaston":
        raise RuntimeError("Publication verification authenticated as the wrong account")
    info = api.model_info(REPO_ID, files_metadata=True)
    if info.gated != "manual" or info.private != args.expect_private:
        raise RuntimeError(f"Unexpected access policy: gated={info.gated}, private={info.private}")
    manifest = json.loads((RELEASE / "publication_manifest.json").read_text())
    expected = dict(manifest["files"])
    manifest_bytes = (RELEASE / "publication_manifest.json").read_bytes()
    expected["publication_manifest.json"] = {
        "bytes": len(manifest_bytes), "sha256": hashlib.sha256(manifest_bytes).hexdigest()
    }
    remote = {entry.rfilename: entry for entry in info.siblings}
    missing = set(expected) - set(remote)
    extra = set(remote) - set(expected) - {".gitattributes"}
    if missing or extra:
        raise RuntimeError(f"Repository file mismatch: missing={sorted(missing)}, extra={sorted(extra)}")
    verified = {}
    downloads = ROOT / "artifacts/huggingface/verification_downloads" / info.sha
    for name, local in expected.items():
        entry = remote[name]
        if entry.size != local["bytes"]:
            raise RuntimeError(f"Remote size mismatch: {name}")
        if entry.lfs is not None:
            remote_sha = entry.lfs.sha256
            method = "server_content_sha256"
        else:
            path = hf_hub_download(REPO_ID, name, revision=info.sha, local_dir=downloads)
            remote_sha = hashlib.sha256(Path(path).read_bytes()).hexdigest()
            method = "downloaded_content_sha256"
        if remote_sha != local["sha256"]:
            raise RuntimeError(f"Remote SHA-256 mismatch: {name}")
        verified[name] = {"bytes": entry.size, "sha256": remote_sha, "method": method}
    card = Path(hf_hub_download(REPO_ID, "README.md", revision=info.sha, local_dir=downloads)).read_text()
    if "# Qwen3.5-9B-mooody" not in card or f'checkpoint = "{REPO_ID}"' not in card:
        raise RuntimeError("Published model card has the wrong model name")
    index = json.loads(Path(hf_hub_download(
        REPO_ID, "model.safetensors.index.json", revision=info.sha, local_dir=downloads
    )).read_text())
    shards = set(index["weight_map"].values())
    if len(index["weight_map"]) != 775 or len(shards) != 7 or not shards <= set(verified):
        raise RuntimeError("Published checkpoint index is incomplete")

    public_info = None
    anonymous_download_statuses = {}
    if not args.expect_private:
        public_info = HfApi(token=False).model_info(REPO_ID)
        if public_info.gated != "manual" or public_info.private:
            raise RuntimeError("Anonymous repository metadata does not show a public gated model")
        for filename in ["config.json", sorted(shards)[0]]:
            request = Request(
                f"https://huggingface.co/{REPO_ID}/resolve/{info.sha}/{filename}", method="HEAD"
            )
            try:
                with urlopen(request, timeout=30) as response:
                    status = response.status
            except HTTPError as error:
                status = error.code
            anonymous_download_statuses[filename] = status
            if status not in {401, 403}:
                raise RuntimeError(f"Anonymous checkpoint access was not denied: {filename}={status}")
    record = {
        "repo_id": REPO_ID,
        "url": f"https://huggingface.co/{REPO_ID}",
        "commit": info.sha,
        "verified_at": datetime.now(timezone.utc).isoformat(),
        "gated": info.gated,
        "private": info.private,
        "release_files_verified": len(verified),
        "checkpoint_tensors": len(index["weight_map"]),
        "checkpoint_shards": len(shards),
        "all_payload_sha256_match": True,
        "anonymous_checkpoint_http_statuses": anonymous_download_statuses,
        "files": dict(sorted(verified.items())),
    }
    name = "upload_verification.json" if args.expect_private else "publication_verification.json"
    output = ROOT / "artifacts/huggingface" / name
    output.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps({key: value for key, value in record.items() if key != "files"}, indent=2))


if __name__ == "__main__":
    main()
