"""Deploy the always-on website and its private native Qwen GPU worker.

Build frontend first: npm run build
Capacity probe: .venv/bin/modal run deployment/modal_app.py --phase preflight
Deploy: .venv/bin/modal deploy deployment/modal_app.py
"""

import json
from pathlib import Path

import modal
from deployment.core import REQUEST_TIMEOUT_SECONDS

ROOT = Path(__file__).resolve().parents[1]
SOURCE_IGNORE = ["**/__pycache__/**", "**/*.pyc"]
app = modal.App("mooody-production")
volume = modal.Volume.from_name("mooody-model-lab", create_if_missing=False)
hf_secret = modal.Secret.from_name("mooody-hf", required_keys=["HF_TOKEN"])
web_secret = modal.Secret.from_name("mooody-web", required_keys=["MOOODY_PROXY_TOKEN"])
gpu_image = (
    modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu22.04", add_python="3.12")
    .entrypoint([])
    .apt_install("git", "build-essential")
    .uv_pip_install("torch==2.10.0", index_url="https://download.pytorch.org/whl/cu128")
    .uv_pip_install(
        "transformers==5.18.0", "accelerate==1.15.0",
        "flash-linear-attention==0.5.2", "huggingface_hub==1.33.0",
    )
    .env({"HF_HUB_DISABLE_PROGRESS_BARS": "1", "TOKENIZERS_PARALLELISM": "false"})
    .add_local_dir(ROOT / "deployment", remote_path="/root/deployment", copy=True, ignore=SOURCE_IGNORE)
)
cpu_image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install("fastapi==0.115.12")
    .add_local_dir(ROOT / "deployment", remote_path="/root/deployment", copy=True, ignore=SOURCE_IGNORE)
    .add_local_dir(ROOT / "dist", remote_path="/www", copy=True)
)


@app.cls(
    image=gpu_image, gpu="L4", cpu=4, memory=65536,
    volumes={"/artifacts": volume}, secrets=[hf_secret],
    min_containers=1, max_containers=1, scaledown_window=180,
    timeout=REQUEST_TIMEOUT_SECONDS + 60, startup_timeout=900,
)
@modal.concurrent(max_inputs=8)
class QwenWorker:
    @modal.enter()
    def load(self):
        from deployment.checkpoint import get_checkpoint, get_persona_bank
        from deployment.worker import ModelRuntime

        volume.reload()
        print("Mooody startup: checking pinned extracted persona bank", flush=True)
        mood_vectors, mood_vector_metadata = get_persona_bank(commit=volume.commit)
        print("Mooody startup: checking pinned checkpoint hashes", flush=True)
        checkpoint = get_checkpoint(commit=volume.commit)
        print("Mooody startup: loading native BF16 model", flush=True)
        self.runtime = ModelRuntime(checkpoint, mood_vectors, mood_vector_metadata)
        print("Mooody startup: ready", flush=True)

    @modal.method(is_generator=True)
    async def stream(self, request_id: str, payload: dict):
        async for event in self.runtime.stream(request_id, payload):
            yield event

    @modal.method()
    async def cancel(self, request_id: str):
        return await self.runtime.cancel(request_id)

    @modal.method()
    async def preflight(self):
        return await self.runtime.preflight()

    @modal.method()
    async def diagnose(self):
        return await self.runtime.diagnose()

    @modal.method()
    async def steering_regression(self):
        return await self.runtime.steering_regression()


@app.function(
    image=cpu_image, cpu=0.5, memory=512, secrets=[web_secret],
    min_containers=1, max_containers=1, scaledown_window=60, timeout=REQUEST_TIMEOUT_SECONDS + 60,
)
@modal.concurrent(max_inputs=64)
@modal.asgi_app(label="mooody-web")
def web():
    from deployment.api import create_web_app

    return create_web_app(QwenWorker())


@app.local_entrypoint()
async def main(phase: str = "preflight"):
    if phase not in {"preflight", "diagnose"}:
        raise ValueError("Use preflight or diagnose; use modal deploy to serve the website")
    result = await getattr(QwenWorker(), phase).remote.aio()
    output = ROOT / "artifacts/deployment"
    output.mkdir(parents=True, exist_ok=True)
    receipt = json.dumps(result, indent=2) + "\n"
    if phase == "diagnose":
        (output / "incremental_steering_diagnostic.json").write_text(receipt)
        print(receipt)
        return
    filename = f"l4_preflight_{result['context_tokens']}_{result['output_tokens']}.json"
    (output / filename).write_text(receipt)
    (output / "l4_preflight.json").write_text(receipt)
    print(json.dumps(result, indent=2))
