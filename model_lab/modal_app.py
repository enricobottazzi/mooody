"""Run the research stages on Modal; download artifacts to the workspace.

Examples:
    .venv/bin/modal run --detach model_lab/modal_app.py --phase download
    .venv/bin/modal run --detach model_lab/modal_app.py --phase preflight
    .venv/bin/modal run --detach model_lab/modal_app.py --phase experiment
"""

import json
from pathlib import Path

import modal

MODEL_ID = "Qwen/Qwen3.5-9B"
MODEL_REVISION = "c202236235762e1c871ad0ccb60c8ee5ba337b9a"
RUN_ID = "qwen35-9b-arditi-20261003-v1"
VOLUME_NAME = "mooody-model-lab"
WORKSPACE = Path(__file__).resolve().parents[1]

app = modal.App("mooody-abliteration-lab")
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
download_image = modal.Image.debian_slim(python_version="3.12").uv_pip_install(
    "huggingface_hub==1.33.0"
)
gpu_image = (
    modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu22.04", add_python="3.12")
    .entrypoint([])
    .apt_install("git", "build-essential")
    .uv_pip_install("torch==2.10.0", index_url="https://download.pytorch.org/whl/cu128")
    .uv_pip_install(
        "transformers==5.18.0",
        "accelerate==1.15.0",
        "flash-linear-attention==0.5.2",
        "huggingface_hub==1.33.0",
    )
    .env({"HF_HUB_DISABLE_PROGRESS_BARS": "1", "TOKENIZERS_PARALLELISM": "false"})
    .add_local_dir(WORKSPACE / "model_lab", remote_path="/root/model_lab", copy=True)
    .add_local_dir(WORKSPACE / "data/arditi_refusal/splits", remote_path="/root/prompts", copy=True)
    .add_local_file(WORKSPACE / "data/arditi_refusal/manifest.json", remote_path="/root/prompts/manifest.json", copy=True)
)


@app.function(image=download_image, volumes={"/artifacts": volume}, timeout=3600)
def download_model():
    from huggingface_hub import snapshot_download

    root = Path("/artifacts/base") / MODEL_REVISION
    path = snapshot_download(
        MODEL_ID,
        revision=MODEL_REVISION,
        local_dir=root,
        allow_patterns=["*.json", "*.safetensors", "*.txt", "*.jinja", "LICENSE", "README.md"],
        max_workers=4,
    )
    index = json.loads((root / "model.safetensors.index.json").read_text())
    for filename in set(index["weight_map"].values()):
        if not (root / filename).is_file():
            raise RuntimeError(f"Missing model shard: {filename}")
    result = {"model_id": MODEL_ID, "revision": MODEL_REVISION, "path": path, "tensor_bytes": index["metadata"]["total_size"]}
    (root / "download_verified.json").write_text(json.dumps(result, indent=2) + "\n")
    volume.commit()
    return result


@app.function(image=gpu_image, gpu="L40S", cpu=4, memory=65536, volumes={"/artifacts": volume}, timeout=1800, max_containers=1, scaledown_window=10)
def preflight():
    import importlib.metadata
    import time
    import torch
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule, fused_recurrent_gated_delta_rule
    from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration
    from model_lab.abliteration import Qwen35Adapter, run_core_self_checks, shared_suffix, tokenize_batch

    self_checks = run_core_self_checks()
    volume.reload()
    base = Path("/artifacts/base") / MODEL_REVISION
    if not (base / "download_verified.json").is_file():
        raise RuntimeError("Run the download stage first")
    tokenizer = AutoTokenizer.from_pretrained(base, local_files_only=True)
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        base, torch_dtype=torch.bfloat16, device_map="cuda:0", attn_implementation="sdpa", local_files_only=True
    ).eval()
    adapter = Qwen35Adapter(model)
    probes = ["Reply with the single word ready.", "What is two plus two?"]
    inputs = tokenize_batch(tokenizer, probes, device=adapter.device)
    started = time.monotonic()
    with torch.inference_mode():
        output = model.generate(**inputs, do_sample=False, max_new_tokens=32, use_cache=True, pad_token_id=tokenizer.pad_token_id)
    torch.cuda.synchronize()
    elapsed = time.monotonic() - started
    replies = tokenizer.batch_decode(output[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    if "ready" not in replies[0].lower() or "4" not in replies[1]:
        raise RuntimeError(f"Native model failed benign smoke probes: {replies!r}")
    if any("<think>" in response for response in replies):
        raise RuntimeError("Model generated unexpected thinking content despite the no-thinking template")
    result = {
        "model_id": MODEL_ID, "revision": MODEL_REVISION,
        "gpu": torch.cuda.get_device_name(),
        "gpu_memory_gib": torch.cuda.get_device_properties(0).total_memory / 2**30,
        "peak_gpu_memory_gib": torch.cuda.max_memory_allocated() / 2**30,
        "versions": {name: importlib.metadata.version(name) for name in ("torch", "transformers", "flash-linear-attention", "huggingface_hub")},
        "core_self_checks": self_checks,
        "linear_attention_kernel_modules": [chunk_gated_delta_rule.__module__, fused_recurrent_gated_delta_rule.__module__],
        "smoke_generation_seconds_including_kernel_warmup": elapsed,
        "adapter": adapter.metadata(), "suffix_ids": list(shared_suffix(tokenizer, probes)), "replies": replies,
    }
    output_root = Path("/artifacts/runs") / RUN_ID
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "preflight.json").write_text(json.dumps(result, indent=2) + "\n")
    volume.commit()
    return result


@app.function(image=gpu_image, gpu="L40S", cpu=4, memory=65536, volumes={"/artifacts": volume}, timeout=86400, max_containers=1, scaledown_window=10)
def experiment(stage: str = "all"):
    volume.reload()
    from model_lab.experiment import run_experiment
    return run_experiment(
        base_dir=Path("/artifacts/base") / MODEL_REVISION,
        output_dir=Path("/artifacts/runs") / RUN_ID,
        prompt_dir=Path("/root/prompts"),
        model_id=MODEL_ID, revision=MODEL_REVISION, run_id=RUN_ID,
        commit=volume.commit, stage=stage,
    )


@app.local_entrypoint()
def main(phase: str = "preflight", stage: str = "all"):
    if phase == "download":
        result = download_model.remote()
    elif phase == "preflight":
        result = preflight.remote()
    elif phase == "experiment":
        result = experiment.remote(stage)
    else:
        raise ValueError(f"Unknown phase: {phase}")
    target = WORKSPACE / "artifacts" / RUN_ID
    target.mkdir(parents=True, exist_ok=True)
    (target / f"{phase}_result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
