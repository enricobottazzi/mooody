"""One global recovery worker; original judging/replay writers must be finished."""
import json
from pathlib import Path
import modal
from model_lab.modal_persona import INPUTS, RUN_ID, RUNS, cpu_image, judge_secret, volume

app = modal.App("mooody-persona-judge-recovery")


@app.function(image=cpu_image, volumes={"/artifacts": volume}, secrets=[judge_secret],
              cpu=2, memory=4096, timeout=86400, max_containers=1)
def recovery_job(run_id: str, audit_only: bool = False):
    from model_lab.persona_judging_recovery import audit_judges, recover_judges
    volume.reload()
    if audit_only:
        return audit_judges(RUNS / run_id, INPUTS)
    return recover_judges(RUNS / run_id, INPUTS, volume.commit)


@app.local_entrypoint()
def main(run_id: str = RUN_ID, phase: str = "audit"):
    if phase not in ("audit", "recover"):
        raise ValueError("Use audit or recover")
    result = recovery_job.remote(run_id, phase == "audit")
    target = Path(__file__).resolve().parents[1] / "artifacts" / "persona"
    target.mkdir(parents=True, exist_ok=True)
    (target / f"judge_recovery_{phase}.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
