"""Stage or verify a vector-only release locally; no network calls."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from persona_bank_artifacts import stage_release, verify_release

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--release-dir", type=Path, required=True)
    parser.add_argument("--license", type=Path, default=ROOT / "artifacts/huggingface/Qwen3.5-9B-mooody/LICENSE")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if args.check:
        result = verify_release(args.release_dir)
    else:
        if args.run_dir is None:
            parser.error("--run-dir is required when staging")
        result = stage_release(args.run_dir, args.release_dir, license_path=args.license, source_root=ROOT)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
