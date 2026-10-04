"""New Modal app for the pinned Mooody persona-vector extraction run.

Each GPU stage shards by trait (six H100 workers); generation is independent of
the OpenRouter secret. Examples, from the repository root:
    .venv/bin/modal run --detach model_lab/modal_persona.py --phase generate
    .venv/bin/modal run --detach model_lab/modal_persona.py --phase judge
    .venv/bin/modal run model_lab/modal_persona.py --phase judge-pilot
    .venv/bin/modal run --detach model_lab/modal_persona.py --phase replay
    .venv/bin/modal run model_lab/modal_persona.py --phase template-check
    .venv/bin/modal run model_lab/modal_persona.py --phase assemble
    .venv/bin/modal run model_lab/modal_persona.py --phase progress
    .venv/bin/modal run model_lab/modal_persona.py --phase download

Use --trait curiosity to resume only that shard. --limit sets the response count
for generation or judge-pilot; judging defaults to the first two pilot responses.
"""

import json
from pathlib import Path

import modal

from model_lab.persona_extraction import TRAITS

WORKSPACE = Path(__file__).resolve().parents[1]
RUN_ID = "qwen35-mooody-persona-20261004-v2"
RUNS = Path("/artifacts/persona_runs")
INPUTS = Path("/root/persona_inputs")
GPU_TYPE = "H100"  # Modal may supply a compatible H200 at the H100 rate.
SOURCE_IGNORE = ["**/__pycache__/**", "**/*.pyc"]
app = modal.App("mooody-persona-extraction")
volume = modal.Volume.from_name("mooody-model-lab", create_if_missing=False)
hf_secret = modal.Secret.from_name("mooody-hf", required_keys=["HF_TOKEN"])
judge_secret = modal.Secret.from_name("mooody-openrouter", required_keys=["OPENROUTER_API_KEY"])

cpu_image = (
    modal.Image.debian_slim(python_version="3.12")
    .add_local_dir(WORKSPACE / "model_lab", remote_path="/root/model_lab", copy=True, ignore=SOURCE_IGNORE)
    .add_local_dir(WORKSPACE / "data/persona_traits", remote_path=str(INPUTS), copy=True)
)
gpu_image = (
    modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu22.04", add_python="3.12")
    .entrypoint([])
    .apt_install("git", "build-essential")
    .uv_pip_install("torch==2.10.0", index_url="https://download.pytorch.org/whl/cu128")
    .uv_pip_install("transformers==5.18.0", "accelerate==1.15.0",
                    "flash-linear-attention==0.5.2", "huggingface_hub==1.33.0",
                    "safetensors==0.8.0")
    .env({"HF_HUB_DISABLE_PROGRESS_BARS": "1", "TOKENIZERS_PARALLELISM": "false"})
    .add_local_dir(WORKSPACE / "model_lab", remote_path="/root/model_lab", copy=True, ignore=SOURCE_IGNORE)
    .add_local_dir(WORKSPACE / "data/persona_traits", remote_path=str(INPUTS), copy=True)
    .add_local_dir(WORKSPACE / "deployment", remote_path="/root/deployment", copy=True, ignore=SOURCE_IGNORE)
    .add_local_dir(WORKSPACE / "tests/model_lab", remote_path="/root/persona_tests", copy=True, ignore=SOURCE_IGNORE)
)


@app.function(image=cpu_image, volumes={"/artifacts": volume}, timeout=300)
def initialize(run_id: str = RUN_ID):
    from model_lab.persona_extraction import initialize_run
    volume.reload()
    return initialize_run(RUNS / run_id, INPUTS, run_id, volume.commit)


@app.function(image=gpu_image, cpu=2, memory=4096, timeout=600)
def activation_checks():
    import importlib.util
    import unittest
    spec = importlib.util.spec_from_file_location("persona_checks", "/root/persona_tests/test_persona_extraction.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    result = unittest.TextTestRunner().run(unittest.defaultTestLoader.loadTestsFromTestCase(module.NativeActivationTests))
    if not result.wasSuccessful() or result.skipped:
        raise RuntimeError("Native activation self-checks did not pass")
    return {"passed": True, "tests": result.testsRun, "scope": "small native Torch fixtures"}


@app.function(image=gpu_image, cpu=2, memory=4096, timeout=600,
              volumes={"/artifacts": volume})
def template_checks(run_id: str = RUN_ID):
    """Use the exact native tokenizer and runtime, without allocating a GPU."""
    import importlib.metadata
    from transformers import AutoTokenizer
    from deployment.checkpoint import EXISTING_CHECKPOINT, MANIFEST
    from model_lab.persona_extraction import (
        MODEL_REVISION, atomic_json, read_json, template_diagnostics, verify_tokenizer_files,
    )
    volume.reload()
    manifest = read_json(MANIFEST)
    candidates = (EXISTING_CHECKPOINT, Path("/artifacts/serving") / MODEL_REVISION)
    checkpoint = next((path for path in candidates if path.is_dir()), None)
    if checkpoint is None:
        raise RuntimeError("No cached audited tokenizer; generation must acquire the pinned checkpoint first")
    hashes = verify_tokenizer_files(checkpoint, manifest)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True)
    result = template_diagnostics(tokenizer, INPUTS)
    result.update({"transformers_version": importlib.metadata.version("transformers"),
                   "tokenizer_file_hashes": hashes, "gpu_allocated": False,
                   "model_weights_loaded": False})
    atomic_json(RUNS / run_id / "preflight" / "template_check.json", result)
    volume.commit()
    return result


def record_gpu_hardware(run_id: str, trait: str, stage: str) -> dict:
    """Private operational audit; not part of frozen generation/replay math."""
    import datetime
    import uuid
    import torch
    from model_lab.persona_extraction import atomic_json

    device = torch.cuda.get_device_properties(0)
    receipt = {"schema_version": 1, "run_id": run_id, "trait": trait, "stage": stage,
               "invocation_id": uuid.uuid4().hex,
               "started_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
               "requested_gpu": GPU_TYPE, "actual_gpu": device.name,
               "compute_capability": [device.major, device.minor],
               "gpu_memory_bytes": device.total_memory,
               "torch_version": torch.__version__, "cuda_runtime_version": torch.version.cuda}
    atomic_json(RUNS / run_id / "hardware" / stage / trait / f"{receipt['invocation_id']}.json", receipt)
    volume.commit()
    return receipt


@app.function(image=gpu_image, gpu=GPU_TYPE, cpu=4, memory=65536,
              volumes={"/artifacts": volume}, secrets=[hf_secret], timeout=86400,
              max_containers=6, scaledown_window=10)
def generate_trait_job(run_id: str, trait: str, limit: int = 0):
    from deployment.checkpoint import get_checkpoint
    from model_lab.persona_extraction import generate_trait
    volume.reload()
    record_gpu_hardware(run_id, trait, "generation")
    checkpoint = get_checkpoint(commit=volume.commit)
    return generate_trait(RUNS / run_id, INPUTS, checkpoint, trait, volume.commit, limit)


@app.function(image=cpu_image, cpu=2, memory=2048,
              volumes={"/artifacts": volume}, secrets=[judge_secret], timeout=86400,
              max_containers=6, scaledown_window=10)
def judge_trait_job(run_id: str, trait: str, concurrency: int = 4):
    from model_lab.persona_extraction import judge_trait
    volume.reload()
    return judge_trait(RUNS / run_id, INPUTS, trait, volume.commit, concurrency)


@app.function(image=cpu_image, cpu=2, memory=2048,
              volumes={"/artifacts": volume}, secrets=[judge_secret], timeout=1800,
              max_containers=6, scaledown_window=10)
def judge_pilot_job(run_id: str, trait: str, limit: int = 2):
    """Score initial pilot responses, reusing normal score files."""
    import time
    from model_lab.persona_extraction import (
        conditions, freeze_stage, frozen_settings, judge_one, judge_path,
        read_json, record_path, response_providers,
    )

    volume.reload()
    root = RUNS / run_id
    _, protocol, traits = frozen_settings(root, INPUTS)
    freeze_stage(root, "judging")
    if limit < 1:
        raise ValueError("Judge pilot limit must be positive")
    rows = conditions(trait, traits[trait], protocol)[:limit]
    records = []
    for row in rows:
        path = record_path(root, trait, row["response_id"])
        if not path.is_file():
            raise RuntimeError("Generate the requested initial pilot responses before the judge pilot")
        record = read_json(path)
        if record.get("status") != "complete" or not record.get("response", "").strip():
            raise RuntimeError("The initial pilot response is not valid for scoring")
        records.append(record)
    coherence = (INPUTS / "coherence_evaluation_prompt.txt").read_text()
    volume.commit()
    started = time.monotonic()
    results = []
    new_attempts = 0
    providers, models = set(), set()
    judge_seconds = 0.0
    known_errors = {
        "malformed_or_out_of_range_score", "judge_refusal", "judge_token_limit_truncation",
        "content_filter_or_safety_block", "empty_judge_content", "http_or_api_error",
        "unexpected_judge_provider",
    }
    for record in records:
        scores, statuses, attempts, errors = {}, {}, {}, {}
        for kind, rubric in (("trait", traits[trait]["evaluation_prompt"]), ("coherence", coherence)):
            path = judge_path(root, trait, record["response_id"], kind)
            old_attempts = len(read_json(path).get("attempts", [])) if path.is_file() else 0
            saved = judge_one(path, protocol, rubric, record, kind)
            volume.commit()
            scores[kind] = saved["score"]
            statuses[kind] = "scored" if saved["score"] is not None else "judge_error"
            attempts[kind] = len(saved["attempts"])
            new_attempts += len(saved["attempts"]) - old_attempts
            errors[kind] = sorted({
                attempt.get("error") if attempt.get("error") in known_errors else "judge_request_error"
                for attempt in saved["attempts"] if attempt.get("status") == "judge_error"
            })
            for attempt in saved["attempts"]:
                if "finished_at_unix" in attempt:
                    judge_seconds += max(0.0, attempt["finished_at_unix"] - attempt["started_at_unix"])
                body = attempt.get("raw_response", {})
                providers.update(response_providers(body))
                if isinstance(body.get("model"), str):
                    models.add(body["model"])
        results.append({"response_id": record["response_id"], "polarity": record["polarity"],
                        "scores": scores, "statuses": statuses, "attempts": attempts,
                        "judge_error_categories": errors})
    return {
        "run_id": run_id, "trait": trait, "phase": "judge-pilot",
        "status": "scored" if all(item == "scored" for row in results for item in row["statuses"].values()) else "judge_error",
        "responses": results, "score_records": 2 * len(records), "new_judge_attempts": new_attempts,
        "actual_providers": sorted(providers), "actual_response_models": sorted(models),
        "saved_attempt_seconds": judge_seconds, "elapsed_seconds": time.monotonic() - started,
        "full_filtering_performed": False, "gpu_allocated": False,
    }


@app.function(image=cpu_image, volumes={"/artifacts": volume}, timeout=600)
def filter_report(run_id: str, trait: str):
    from model_lab.persona_extraction import filter_trait
    volume.reload()
    return filter_trait(RUNS / run_id, INPUTS, trait, volume.commit)


@app.function(image=gpu_image, gpu=GPU_TYPE, cpu=4, memory=65536,
              volumes={"/artifacts": volume}, secrets=[hf_secret], timeout=86400,
              max_containers=6, scaledown_window=10)
def replay_trait_job(run_id: str, trait: str):
    from deployment.checkpoint import get_checkpoint
    from model_lab.persona_extraction import replay_trait
    volume.reload()
    record_gpu_hardware(run_id, trait, "replay")
    checkpoint = get_checkpoint(commit=volume.commit)
    return replay_trait(RUNS / run_id, INPUTS, checkpoint, trait, volume.commit)


@app.function(image=gpu_image, cpu=4, memory=8192, volumes={"/artifacts": volume}, timeout=1800)
def assemble(run_id: str = RUN_ID):
    from model_lab.persona_extraction import assemble_bank
    volume.reload()
    return assemble_bank(RUNS / run_id, INPUTS, volume.commit)


@app.function(image=cpu_image, volumes={"/artifacts": volume}, timeout=600)
def report(run_id: str = RUN_ID):
    from model_lab.persona_extraction import progress
    volume.reload()
    return progress(RUNS / run_id, INPUTS)


@app.function(image=cpu_image, volumes={"/artifacts": volume}, timeout=600)
def publication_files(run_id: str = RUN_ID):
    volume.reload()
    root = RUNS / run_id / "publication"
    # Raw response/score directories are deliberately not returned for public
    # packaging. Root can separately download private audit files via Volume.
    return {name: (root / name).read_bytes() for name in
            ("persona_vectors.safetensors", "persona_manifest.json")}


@app.local_entrypoint()
def main(phase: str = "progress", run_id: str = RUN_ID, trait: str = "all",
         limit: int = 0, judge_concurrency: int = 4):
    if trait != "all" and trait not in TRAITS:
        raise ValueError("Unknown trait")
    if limit < 0:
        raise ValueError("Response limit cannot be negative")
    selected = list(TRAITS) if trait == "all" else [trait]
    if phase in ("initialize", "generate", "template-check"):
        initialize.remote(run_id)
    if phase == "checks":
        result = activation_checks.remote()
    elif phase == "template-check":
        result = template_checks.remote(run_id)
    elif phase == "initialize":
        result = {"run_id": run_id, "initialized": True}
    elif phase in ("generate", "judge", "judge-pilot", "replay", "filter"):
        function = {"generate": generate_trait_job, "judge": judge_trait_job,
                    "judge-pilot": judge_pilot_job, "replay": replay_trait_job,
                    "filter": filter_report}[phase]
        calls = []
        for name in selected:
            args = [run_id, name]
            if phase == "generate":
                args.append(limit)
            elif phase == "judge-pilot":
                args.append(limit or 2)
            elif phase == "judge":
                args.append(judge_concurrency)
            calls.append(function.spawn(*args))
        # Spawn every trait before waiting: all six GPU shards run concurrently.
        result = {name: call.get() for name, call in zip(selected, calls)}
    elif phase == "assemble":
        result = assemble.remote(run_id)
    elif phase == "progress":
        result = report.remote(run_id)
    elif phase == "download":
        files = publication_files.remote(run_id)
        target = WORKSPACE / "artifacts" / run_id / "publication"
        target.mkdir(parents=True, exist_ok=True)
        for name, content in files.items():
            (target / name).write_bytes(content)
        result = {"downloaded": str(target), "files": list(files)}
    else:
        raise ValueError("Unknown phase")
    target = WORKSPACE / "artifacts" / run_id
    target.mkdir(parents=True, exist_ok=True)
    (target / f"{phase}_result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
