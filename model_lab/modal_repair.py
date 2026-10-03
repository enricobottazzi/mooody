"""Resume the frozen run with an audited cached-position verification repair.

Only equivalence_rows is replaced. The original generation, extraction,
selection, editing and export guards remain unchanged.

Run: .venv/bin/modal run --detach model_lab/modal_repair.py
"""

import hashlib
import json
from pathlib import Path
import shutil

import modal

from model_lab.modal_app import (
    MODEL_ID, MODEL_REVISION, RUN_ID, WORKSPACE, gpu_image, volume,
)

app = modal.App("mooody-abliteration-repair")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@app.function(image=gpu_image, gpu="L40S", cpu=4, memory=65536,
              volumes={"/artifacts": volume}, timeout=86400,
              max_containers=1, scaledown_window=10)
def resume():
    import inspect
    from model_lab import experiment
    from model_lab.export_repair import repaired_equivalence_rows, run_position_self_checks

    volume.reload()
    root = Path("/artifacts/runs") / RUN_ID
    settings = json.loads((root / "run_settings.json").read_text())
    selection_path = root / "selection.json"
    selection = json.loads(selection_path.read_text())
    if not selection.get("selection_locked"):
        raise RuntimeError("Repair requires the completed, locked original selection")
    if digest(root / "direction.pt") != selection["direction_sha256"]:
        raise RuntimeError("Locked direction changed before repair")
    source = Path(experiment.__file__).parent
    for name, expected in settings["source_code_hashes"].items():
        if digest(source / name) != expected or digest(root / "provenance" / name) != expected:
            raise RuntimeError(f"Original frozen source changed: {name}")

    repair_dir = root / "execution_repairs"
    repair_dir.mkdir(exist_ok=True)
    for name in ("export_repair.py", "modal_repair.py"):
        shutil.copyfile(source / name, repair_dir / name)
    audit = {
        "repair_id": "explicit_text_positions_for_export_equivalence_v1",
        "run_id": RUN_ID,
        "scope": "Only the export hook-versus-checkpoint equivalence helper",
        "cause": "Manual batch-one cached decode inferred rotary positions from stale batch-four rope_deltas",
        "correction": "Explicit 2D text positions for prefill/current cached token; clear rotary delta state before each fresh probe",
        "selection_sha256": digest(selection_path),
        "direction_sha256": selection["direction_sha256"],
        "original_source_code_hashes": settings["source_code_hashes"],
        "original_method_sha256": hashlib.sha256(inspect.getsource(experiment._Experiment.equivalence_rows).encode()).hexdigest(),
        "repair_source_hashes": {name: digest(source / name) for name in ("export_repair.py", "modal_repair.py")},
        "position_self_checks": run_position_self_checks(),
        "native_generation_agreement_check": "Three benign probes; repaired manual cached token plans checked against native greedy generation before weight editing",
        "generation_selection_and_weight_edit_unchanged": True,
        "equivalence_probes_forced_tokens_layout_and_guards_unchanged": True,
    }
    audit_path = repair_dir / "cache_position_fix.json"
    if audit_path.exists() and json.loads(audit_path.read_text()) != audit:
        raise RuntimeError("Verification repair definition changed; preserve a separate repair audit")
    audit_path.write_text(json.dumps(audit, indent=2) + "\n")
    failure = root / "failure.json"
    if failure.exists():
        archived = repair_dir / "original_export_failure.json"
        if archived.exists() and archived.read_bytes() != failure.read_bytes():
            raise RuntimeError("A new failure needs independent diagnosis before retry")
        if not archived.exists():
            shutil.copyfile(failure, archived)
        failure.unlink()
    volume.commit()
    experiment._Experiment.equivalence_rows = repaired_equivalence_rows
    print(json.dumps({"event": "verification_repair_installed", "repair_id": audit["repair_id"]}), flush=True)
    return experiment.run_experiment(
        base_dir=Path("/artifacts/base") / MODEL_REVISION,
        output_dir=root, prompt_dir=Path("/root/prompts"), model_id=MODEL_ID,
        revision=MODEL_REVISION, run_id=RUN_ID, commit=volume.commit, stage="all",
    )


@app.local_entrypoint()
def main():
    result = resume.remote()
    target = WORKSPACE / "artifacts" / RUN_ID
    target.mkdir(parents=True, exist_ok=True)
    (target / "experiment_result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
