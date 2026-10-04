"""Launch only after the coordinator has stopped every serial generation job."""
import json
from pathlib import Path
import modal
from model_lab.modal_persona import GPU_TYPE, INPUTS, RUN_ID, RUNS, gpu_image, hf_secret, record_gpu_hardware, volume
from model_lab.persona_extraction import TRAITS

WORKSPACE = Path(__file__).resolve().parents[1]
app = modal.App("mooody-persona-batched-continuation")


@app.function(image=gpu_image, gpu=GPU_TYPE, cpu=4, memory=65536,
              volumes={"/artifacts": volume}, secrets=[hf_secret], timeout=86400,
              max_containers=6, scaledown_window=10)
def generate_batched_job(run_id: str, trait: str, limit: int = 0):
    from deployment.checkpoint import get_checkpoint
    from model_lab.persona_batched import generate_batched_trait
    volume.reload()
    record_gpu_hardware(run_id, trait, "generation_batched_v1")
    checkpoint = get_checkpoint(commit=volume.commit)
    return generate_batched_trait(RUNS / run_id, INPUTS, checkpoint, trait, volume.commit, limit)


@app.local_entrypoint()
def main(run_id: str = RUN_ID, trait: str = "all", limit: int = 0):
    if trait != "all" and trait not in TRAITS or limit < 0:
        raise ValueError("Invalid trait/limit")
    selected = list(TRAITS) if trait == "all" else [trait]
    calls = [generate_batched_job.spawn(run_id, name, limit) for name in selected]
    result = {name: call.get() for name, call in zip(selected, calls)}
    target = WORKSPACE / "artifacts" / run_id
    target.mkdir(parents=True, exist_ok=True)
    (target / "generate_batched_result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
