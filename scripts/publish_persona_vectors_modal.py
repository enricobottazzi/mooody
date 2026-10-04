"""Publish and verify the vector-only bank from Modal, using its HF secret.

The default creates a public model repository under the authenticated HF user's
account. An existing nonempty repository is accepted only as an exact idempotent
retry; it is never overwritten. No model weights or raw transcripts are uploaded.

Example (after extraction has committed its two public artifacts to the volume):
    .venv/bin/python scripts/publish_persona_vectors_modal.py \
        --run-dir /artifacts/persona_runs/<run_id>/publication

Verification only:
    .venv/bin/python scripts/publish_persona_vectors_modal.py --verify \
        --repo-id <owner>/Qwen3.5-9B-mooody-persona-vectors --revision <40-hex-commit>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re

import modal

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPO_NAME = "Qwen3.5-9B-mooody-persona-vectors"


def identity_remote() -> dict:
    """Read account identity and token role; never return the auth payload."""
    from huggingface_hub import HfApi

    identity = HfApi().whoami()
    owner = identity.get("name")
    if not isinstance(owner, str) or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", owner) is None:
        raise ValueError("HF identity returned an invalid account name")
    auth = identity.get("auth")
    auth = auth if isinstance(auth, dict) else {}
    token = auth.get("accessToken")
    token = token if isinstance(token, dict) else {}
    # Whitelist enum labels rather than serializing arbitrary authentication data.
    auth_type = auth.get("type")
    auth_type = auth_type if auth_type in ("oauth", "access_token") else None
    role = token.get("role")
    role = role if role in ("read", "write", "admin", "fineGrained", "fine-grained") else None
    return {"owner": owner, "auth_type": auth_type, "auth_role": role,
            "write_role_reported": True if role in ("write", "admin") else False if role == "read" else None}


def verify_remote(repo_id: str, revision: str, private: bool, expected_owner: str | None = None) -> dict:
    from datetime import datetime, timezone
    from pathlib import Path
    import re
    import shutil
    import tempfile

    from huggingface_hub import HfApi, hf_hub_download
    from persona_bank_artifacts import PUBLIC_FILES, digest, require, verify_release

    require(re.fullmatch(r"[0-9a-f]{40}", revision) is not None, "Verification requires an exact 40-hex HF commit")
    api = HfApi()
    owner = api.whoami()["name"]
    require(repo_id.split("/")[0] == owner and (expected_owner is None or owner == expected_owner),
            "Publication verification authenticated as an unexpected owner")
    info = api.model_info(repo_id, revision=revision, files_metadata=True)
    require(info.sha == revision and bool(info.private) == private and info.gated in (False, None),
            "Unexpected pinned revision or vector repository access policy")
    remote_names = {item.rfilename for item in info.siblings}
    require(remote_names - {".gitattributes"} == PUBLIC_FILES, "Remote repository contains missing or unlisted payload files")
    # Download every byte at the immutable commit, rather than trusting server LFS metadata alone.
    download_root = Path(tempfile.mkdtemp(prefix="mooody-persona-verify-"))
    try:
        for name in sorted(PUBLIC_FILES):
            cached = hf_hub_download(repo_id, name, revision=revision, force_download=True)
            shutil.copyfile(cached, download_root / name)
        result = verify_release(download_root)
        if not private:
            anonymous = HfApi(token=False).model_info(repo_id, revision=revision)
            require(not anonymous.private and anonymous.gated in (False, None), "Public vector metadata is not anonymously accessible")
            cached = hf_hub_download(repo_id, "persona_vectors.safetensors", revision=revision,
                                     token=False, force_download=True)
            require(digest(Path(cached)) == result["tensor"]["sha256"], "Anonymous pinned bank download failed integrity check")
        result.update({"repo_id": repo_id, "revision": revision, "commit": revision,
                       "url": f"https://huggingface.co/{repo_id}/tree/{revision}",
                       "private": private, "gated": False,
                       "verified_at": datetime.now(timezone.utc).isoformat(),
                       "pinned_downloads_verified": True,
                       "anonymous_bank_download_verified": not private,
                       "runtime_locator": {"repo_id": repo_id, "revision": revision,
                                           "manifest_filename": "persona_manifest.json"}})
        return result
    finally:
        shutil.rmtree(download_root)


def publish_remote(run_dir: str, repo_name: str, private: bool, expected_owner: str | None = None) -> dict:
    from pathlib import Path
    import re
    import shutil
    import tempfile

    from huggingface_hub import CommitOperationAdd, HfApi
    from huggingface_hub.errors import RepositoryNotFoundError
    from persona_bank_artifacts import PUBLIC_FILES, require, stage_release

    require(re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,95}", repo_name) is not None, "Use a repository name without an owner or slash")
    source = Path(run_dir).resolve()
    require(source.is_relative_to(Path("/artifacts").resolve()), "Run must be stored in the mounted artifact volume")
    api = HfApi()
    owner = api.whoami()["name"]
    require(isinstance(owner, str) and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", owner) is not None,
            "Authenticated HF user has an invalid account name")
    require(expected_owner is None or owner == expected_owner, "Authenticated HF user differs from requested owner")
    repo_id = f"{owner}/{repo_name}"
    require(repo_id != "demivoleegaston/Qwen3.5-9B-mooody", "Do not modify the base checkpoint repository")
    staging_parent = Path(tempfile.mkdtemp(prefix="mooody-persona-release-"))
    target = staging_parent / "release"
    try:
        staged = stage_release(source, target, license_path=Path("/root/persona_base_LICENSE"),
                               source_root=Path("/root"))
        try:
            existing = api.model_info(repo_id, files_metadata=True)
        except RepositoryNotFoundError:
            # exist_ok=False prevents silently taking over an independently created repository.
            api.create_repo(repo_id=repo_id, repo_type="model", private=private, exist_ok=False)
            existing = api.model_info(repo_id, files_metadata=True)
        require(bool(existing.private) == private and existing.gated in (False, None),
                "Existing vector repository has a different access policy")
        remote_names = {item.rfilename for item in existing.siblings} - {".gitattributes"}
        if remote_names:
            # Recovery after an unknown upload outcome: exact existing release only.
            verified = verify_remote(repo_id, existing.sha, private, owner)
            require(verified["publication_manifest_sha256"] == staged["publication_manifest_sha256"],
                    "Repository already contains a different release; choose a new repository name")
            verified["publication_method"] = "exact_payload_idempotent_retry"
            return verified
        operations = [CommitOperationAdd(path_in_repo=name, path_or_fileobj=target / name)
                      for name in sorted(PUBLIC_FILES)]
        commit = api.create_commit(repo_id=repo_id, repo_type="model", operations=operations,
                                   parent_commit=existing.sha,
                                   commit_message="Publish six real all-layer Mooody persona vectors")
        result = verify_remote(repo_id, commit.oid, private, owner)
        require(result["publication_manifest_sha256"] == staged["publication_manifest_sha256"],
                "Pinned published release differs from staged payload")
        result["publication_method"] = "single_commit_from_verified_modal_volume"
        return result
    finally:
        shutil.rmtree(staging_parent)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", help="Finalized directory containing the bank and persona manifest on mooody-model-lab volume")
    parser.add_argument("--repo-name", default=DEFAULT_REPO_NAME)
    parser.add_argument("--expected-owner", help="Optional account identity guard; never a credential")
    parser.add_argument("--private", action="store_true", help="Create/verify a private rather than public vector repository")
    parser.add_argument("--hf-secret", default="mooody-hf", help="Existing Modal secret exposing HF_TOKEN")
    parser.add_argument("--volume", default="mooody-model-lab")
    parser.add_argument("--license", type=Path, default=ROOT / "artifacts/huggingface/Qwen3.5-9B-mooody/LICENSE")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--verify", action="store_true", help="Read-only pinned remote verification")
    mode.add_argument("--identity", action="store_true", help="CPU-only account/role preflight; no repository changes")
    parser.add_argument("--repo-id")
    parser.add_argument("--revision")
    parser.add_argument("--output", type=Path, help="Sanitized identity, verification or publication receipt path")
    args = parser.parse_args()
    if args.identity:
        pass
    elif args.verify:
        if not args.repo_id or not args.revision or re.fullmatch(r"[0-9a-f]{40}", args.revision) is None:
            parser.error("--verify requires --repo-id and exact 40-hex --revision")
    elif not args.run_dir:
        parser.error("--run-dir is required for publication")
    if not args.identity and not args.license.is_file():
        parser.error("The inherited Apache license file is required")
    app = modal.App("mooody-persona-vector-publication")
    image = (modal.Image.debian_slim(python_version="3.12")
             .uv_pip_install("huggingface_hub==2.1.1")
             .env({"HF_HUB_DISABLE_PROGRESS_BARS": "1"}))
    volumes = {}
    if not args.identity:
        image = (image.add_local_file(ROOT / "scripts/persona_bank_artifacts.py", remote_path="/root/persona_bank_artifacts.py", copy=True)
                 .add_local_file(args.license, remote_path="/root/persona_base_LICENSE", copy=True)
                 .add_local_dir(ROOT / "data/persona_traits", remote_path="/root/data/persona_traits", copy=True))
        volumes = {"/artifacts": modal.Volume.from_name(args.volume, create_if_missing=False)}
    remote = app.function(image=image, volumes=volumes,
                          secrets=[modal.Secret.from_name(args.hf_secret, required_keys=["HF_TOKEN"])],
                          cpu=2, memory=4096, timeout=1800, max_containers=1, scaledown_window=10)(
                              identity_remote if args.identity else verify_remote if args.verify else publish_remote)
    with modal.enable_output(), app.run():
        if args.identity:
            result = remote.remote()
        elif args.verify:
            result = remote.remote(args.repo_id, args.revision, args.private, args.expected_owner)
        else:
            result = remote.remote(args.run_dir, args.repo_name, args.private, args.expected_owner)
    output = args.output or ROOT / "artifacts/persona_vectors" / (
        "identity_preflight.json" if args.identity else "publication_verification.json" if args.verify else "publication_receipt.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
