"""Single-worker transport-only recovery, preserving original judge journals."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import time
from concurrent.futures import ThreadPoolExecutor
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from model_lab.persona_extraction import (
    TRAITS, atomic_json, conditions, file_hash, fingerprint, freeze_stage,
    frozen_settings, judge_path, judge_payload, judge_result, log, read_json,
    record_path, response_providers,
)

STAGE = "judging_recovery_v1"
POLICY_SOURCE = "data/persona_traits/judging_recovery_policy.json"
POLICY_FILENAME = "judging_recovery_policy.json"


def recovery_identity(protocol, rubric, response, kind):
    payload = judge_payload(protocol, rubric, response["question"], response["response"])
    identity = {"response_id": response["response_id"], "response_sha256": fingerprint(response),
                "judge_kind": kind, "judge_prompt_sha256": hashlib.sha256(rubric.encode()).hexdigest(),
                "request_sha256": fingerprint(payload)}
    return identity, payload


def original_attempts(saved):
    return saved["attempts"][:saved.get("transport_recovery", {}).get("original_attempts_count", len(saved["attempts"]))]


def eligible(saved):
    attempts = original_attempts(saved)
    return (saved.get("score") is None and len(attempts) == 3
            and all(attempt.get("status") == "judge_error" and attempt.get("http_status") == 429
                    and attempt.get("error_type") == "HTTPError" for attempt in attempts))


def read_json_sha(path):
    raw = path.read_bytes()
    return json.loads(raw), hashlib.sha256(raw).hexdigest()


def journal_items(root, protocol, traits):
    jobs = [{"trait": trait, "row": row, "kind": kind,
             "path": judge_path(root, trait, row["response_id"], kind)}
            for trait in TRAITS for row in conditions(trait, traits[trait], protocol)
            for kind in ("trait", "coherence")]
    def read(item):
        try:
            saved, digest = read_json_sha(item["path"])
            return {**item, "saved": saved, "sha256": digest}
        except FileNotFoundError:
            return {**item, "saved": None, "sha256": None}
    # Only immutable read-only IO is parallel. All HTTP remains single-worker.
    with ThreadPoolExecutor(max_workers=8) as pool:
        return list(pool.map(read, jobs))


def audit_items(items):
    summary = {"planned_score_slots": 4800, "present": 0, "valid_scores": 0, "missing": 0,
               "in_progress_or_interrupted": 0, "eligible_http429_none": 0,
               "other_none": 0, "actual_attempts": 0, "original_actual_attempts": 0,
               "recovery_actual_attempts": 0, "by_trait": {}}
    fields = tuple(key for key in summary if key not in ("planned_score_slots", "by_trait"))
    summary["by_trait"] = {trait: {key: 0 for key in fields} for trait in TRAITS}
    for item in items:
        counts = summary["by_trait"][item["trait"]]
        saved = item["saved"]
        if saved is None:
            counts["missing"] += 1
            continue
        counts["present"] += 1
        counts["actual_attempts"] += len(saved["attempts"])
        counts["original_actual_attempts"] += len(original_attempts(saved))
        counts["recovery_actual_attempts"] += len(saved["attempts"]) - len(original_attempts(saved))
        if type(saved.get("score")) is int and 0 <= saved["score"] <= 100:
            counts["valid_scores"] += 1
        elif saved["attempts"] and saved["attempts"][-1].get("status") == "request_started":
            counts["in_progress_or_interrupted"] += 1
        elif eligible(saved):
            counts["eligible_http429_none"] += 1
        else:
            counts["other_none"] += 1
    for counts in summary["by_trait"].values():
        for key, count in counts.items():
            summary[key] += count
    return summary


def audit_judges(root: Path, inputs: Path) -> dict:
    """Read each planned journal once with eight IO threads, without HTTP."""
    _, protocol, traits = frozen_settings(root, inputs)
    return audit_items(journal_items(root, protocol, traits))


def freeze_recovery(root: Path, inputs: Path) -> dict:
    policy_path = inputs / POLICY_FILENAME
    policy = read_json(policy_path)
    if (policy.get("stage") != STAGE or policy.get("original_attempt_count") != 3
            or policy.get("maximum_additional_attempts_per_score") != 3
            or policy.get("global_recovery_workers") != 1 or policy.get("request_concurrency") != 1
            or policy.get("minimum_delay_after_request_completion_seconds") != 0.5
            or policy.get("http_429_backoff_seconds") != [2, 4]
            or not all(policy.get(key) is True for key in (
                "original_attempts_must_all_be_completed_http_429", "retry_during_recovery_only_http_429",
                "do_not_retry_malformed_refusal_or_other_nontransport_failures",
                "preserve_original_attempt_prefix_and_valid_scores", "journal_and_commit_each_attempt_before_http",
                "unknown_interrupted_attempts_require_explicit_repair",
                "require_original_judging_workers_finished_before_recovery"))):
        raise ValueError("Unexpected transport-only recovery policy")
    baseline = freeze_stage(root, "judging")
    payload = {"stage": STAGE, "module_sha256": file_hash(Path(__file__)),
               "original_judging_implementation_sha256": baseline["implementation_sha256"],
               "policy_source": POLICY_SOURCE, "policy_sha256": file_hash(policy_path),
               "policy_bytes": policy_path.stat().st_size, "policy": policy}
    stage = {**payload, "implementation_sha256": fingerprint(payload)}
    path = root / "implementations" / f"{STAGE}.json"
    if path.exists() and read_json(path) != stage:
        raise ValueError("Frozen transport-only recovery implementation/policy changed")
    if not path.exists():
        atomic_json(path, stage)
        shutil.copyfile(Path(__file__), path.with_suffix(".py"))
    if file_hash(path.with_suffix(".py")) != stage["module_sha256"]:
        raise ValueError("Frozen recovery implementation snapshot changed")
    return stage


class Pacer:
    def __init__(self, delay=0.5, clock=time.monotonic, sleep=time.sleep):
        self.delay, self.clock, self.sleep = delay, clock, sleep
        self.finished = None

    def wait(self):
        if self.finished is not None:
            self.sleep(max(0.0, self.delay - (self.clock() - self.finished)))

    def finish(self):
        self.finished = self.clock()


def request(protocol, payload, key):
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json",
               **protocol["judging"]["request_headers"]}
    native = Request(protocol["judging"]["base_url"] + protocol["judging"]["endpoint"],
                     data=json.dumps(payload).encode(), headers=headers, method="POST")
    with urlopen(native, timeout=120) as reply:
        body = json.load(reply)
        selected_headers = {name: value for name, value in reply.headers.items()
                            if name.lower() in ("x-request-id", "x-openrouter-request-id", "date")}
    return body, selected_headers


def recover_one(root: Path, path: Path, protocol: dict, rubric: str, response: dict, kind: str,
                stage: dict, pacer: Pacer, commit=lambda: None, transport=None) -> dict:
    identity, payload = recovery_identity(protocol, rubric, response, kind)
    saved = read_json(path)
    if any(saved.get(key) != value for key, value in identity.items()):
        raise ValueError("Recovery request identity differs from the frozen judge")
    # In particular, never reserialize or replace an existing valid score.
    if saved.get("score") is not None:
        return saved
    if not eligible(saved):
        return saved
    key = os.environ.get(protocol["judging"]["api_key_env"])
    if transport is None and not key:
        raise RuntimeError("OPENROUTER_API_KEY is unavailable")
    original_path = root / STAGE / "originals" / response["trait"] / path.name
    if "transport_recovery" not in saved:
        if original_path.exists() and original_path.read_bytes() != path.read_bytes():
            raise ValueError("Original journal snapshot changed")
        original_path.parent.mkdir(parents=True, exist_ok=True)
        if not original_path.exists():
            with original_path.open("xb") as handle:
                handle.write(path.read_bytes())
        saved["transport_recovery"] = {"stage": STAGE,
                "policy_sha256": stage["policy_sha256"], "implementation_sha256": stage["implementation_sha256"],
                "original_file_sha256": file_hash(original_path),
                "original_attempts_sha256": fingerprint(saved["attempts"]), "original_attempts_count": 3}
        atomic_json(path, saved)
        commit()
    meta = saved["transport_recovery"]
    original = read_json(original_path)
    if (meta["policy_sha256"] != stage["policy_sha256"]
            or meta["implementation_sha256"] != stage["implementation_sha256"]
            or meta["original_file_sha256"] != file_hash(original_path)
            or meta["original_attempts_sha256"] != fingerprint(original["attempts"])
            or saved["attempts"][:3] != original["attempts"]):
        raise ValueError("Recovery original journal prefix/provenance changed")
    additions = saved["attempts"][3:]
    if any(attempt.get("stage") != STAGE or attempt.get("recovery_attempt") != index + 1
           for index, attempt in enumerate(additions)) or len(additions) > 3:
        raise ValueError("Invalid recovery attempt history")
    if additions and additions[-1]["status"] == "request_started":
        raise RuntimeError("Unknown interrupted recovery request; explicit repair required")
    if additions and additions[-1].get("http_status") != 429:
        return saved  # A malformed/refusal/non429 failure ends this recovery.
    while saved["score"] is None and len(saved["attempts"]) < 6:
        index = len(saved["attempts"]) - 3
        pacer.wait()
        attempt = {"attempt": len(saved["attempts"]) + 1, "stage": STAGE,
                   "recovery_attempt": index + 1, "status": "request_started", "started_at_unix": time.time()}
        saved["attempts"].append(attempt)
        atomic_json(path, saved)
        commit()  # The journal is durable before any HTTP request is sent.
        try:
            body, headers = transport(payload) if transport else request(protocol, payload, key)
            attempt.update(raw_response=body, response_headers=headers)
            score = judge_result(body)
            providers = response_providers(body)
            if any("".join(character for character in provider.lower() if character.isalpha()) != "googleaistudio"
                   for provider in providers):
                raise ValueError("unexpected_judge_provider")
            saved["score"] = score
            attempt.update(status="complete", score=score)
        except (HTTPError, URLError, ValueError, KeyError, IndexError, TypeError, TimeoutError) as error:
            description = str(error)[:2000]
            attempt.update(status="judge_error", error_type=type(error).__name__,
                           error=description.replace(key, "[redacted]") if key else description)
            if isinstance(error, HTTPError):
                attempt["http_status"] = error.code
                body = error.read().decode(errors="replace")[:2000]
                attempt["http_error_body"] = body.replace(key, "[redacted]") if key else body
        finally:
            pacer.finish()
        attempt["finished_at_unix"] = time.time()
        atomic_json(path, saved)
        commit()
        if saved["score"] is not None or attempt.get("http_status") != 429:
            break
        if len(saved["attempts"]) < 6:
            pacer.sleep(stage["policy"]["http_429_backoff_seconds"][index])
    return saved


def recover_judges(root: Path, inputs: Path, commit=lambda: None, transport=None) -> dict:
    settings, protocol, traits = frozen_settings(root, inputs)
    items = journal_items(root, protocol, traits)
    before = audit_items(items)
    if before["missing"] or before["in_progress_or_interrupted"]:
        raise RuntimeError("Finish all original judging workers before transport recovery")
    stage = freeze_recovery(root, inputs)
    receipt_path = root / STAGE / "audit.json"
    baseline_path = root / STAGE / "baseline_journal_hashes.json"
    if not baseline_path.exists():
        # Valid scores and ineligible failures must retain these exact bytes.
        baseline = {f"{item['trait']}/{item['path'].name}": item["sha256"] for item in items}
        if len(baseline) != 4800:
            raise ValueError("Baseline must describe all4800 original journals")
        atomic_json(baseline_path, baseline)
    receipt = read_json(receipt_path) if receipt_path.exists() else {
        "stage": STAGE, "settings_sha256": fingerprint(settings),
        "policy_sha256": stage["policy_sha256"], "implementation_sha256": stage["implementation_sha256"],
        "before": before, "baseline_journal_hashes_sha256": file_hash(baseline_path),
        "started_at_unix": time.time()}
    if (receipt["settings_sha256"] != fingerprint(settings) or receipt["policy_sha256"] != stage["policy_sha256"]
            or receipt["implementation_sha256"] != stage["implementation_sha256"]
            or receipt["baseline_journal_hashes_sha256"] != file_hash(baseline_path)):
        raise ValueError("Recovery run receipt changed")
    atomic_json(receipt_path, receipt)
    commit()
    pacer = Pacer()
    coherence = (inputs / "coherence_evaluation_prompt.txt").read_text()
    for trait in TRAITS:
        touched, recovered = 0, 0
        for item in items:
            if item["trait"] != trait or not eligible(item["saved"]):
                continue
            row, kind = item["row"], item["kind"]
            rubric = traits[trait]["evaluation_prompt"] if kind == "trait" else coherence
            response = read_json(record_path(root, trait, row["response_id"]))
            result = recover_one(root, item["path"], protocol, rubric, response, kind, stage, pacer, commit, transport)
            touched += 1
            recovered += result["score"] is not None
        log("recovery_progress", trait=trait, touched=touched, recovered=recovered)
    receipt.update(after=audit_judges(root, inputs), finished_at_unix=time.time(), status="complete")
    atomic_json(receipt_path, receipt)
    commit()
    return receipt


def public_recovery_provenance(root: Path, inputs: Path) -> tuple[dict, dict]:
    path = root / "implementations" / f"{STAGE}.json"
    if not path.exists():
        return {}, {}
    stage = freeze_recovery(root, inputs)
    audit = read_json(root / STAGE / "audit.json")
    _, protocol, traits = frozen_settings(root, inputs)
    items = journal_items(root, protocol, traits)
    if audit.get("status") != "complete" or audit["after"] != audit_items(items):
        raise ValueError("Transport recovery is incomplete or scores changed after its final audit")
    counts = {"records_touched": 0, "additional_attempts": 0, "recovered_scores": 0,
              "remaining_none": 0, "original_http429_attempts": 0, "http429_recovery_attempts": 0}
    baseline_path = root / STAGE / "baseline_journal_hashes.json"
    if audit.get("baseline_journal_hashes_sha256") != file_hash(baseline_path):
        raise ValueError("Recovery baseline journal hash receipt changed")
    baseline = read_json(baseline_path)
    if len(baseline) != 4800:
        raise ValueError("Recovery baseline does not cover all original journals")
    originals = {}
    reconstructed = {trait: {"valid_scores": 0, "eligible_http429_none": 0, "other_none": 0,
                             "actual_attempts": 0} for trait in TRAITS}
    response_jobs = [(trait, row["response_id"], record_path(root, trait, row["response_id"]))
                     for trait in TRAITS for row in conditions(trait, traits[trait], protocol)]
    def read_response(job):
        trait, identity, source = job
        return (trait, identity), read_json(source)
    with ThreadPoolExecutor(max_workers=8) as pool:
        responses = dict(pool.map(read_response, response_jobs))
    original_jobs = [(item["trait"], item["path"].name) for item in items
                     if "transport_recovery" in item["saved"]]
    def read_original(job):
        trait, name = job
        return job, read_json_sha(root / STAGE / "originals" / trait / name)
    with ThreadPoolExecutor(max_workers=8) as pool:
        original_cache = dict(pool.map(read_original, original_jobs))
    item_cache = {(item["trait"], item["path"].name): item for item in items}
    coherence = (inputs / "coherence_evaluation_prompt.txt").read_text()
    for trait in TRAITS:
        for row in conditions(trait, traits[trait], protocol):
            for kind in ("trait", "coherence"):
                target = judge_path(root, trait, row["response_id"], kind)
                item = item_cache[trait, target.name]
                saved = item["saved"]
                response = responses[trait, row["response_id"]]
                rubric = traits[trait]["evaluation_prompt"] if kind == "trait" else coherence
                identity, _ = recovery_identity(protocol, rubric, response, kind)
                if any(saved.get(key) != value for key, value in identity.items()):
                    raise ValueError("Recovery changed the frozen request/response/rubric identity")
                if "transport_recovery" not in saved:
                    if baseline.get(f"{trait}/{target.name}") != item["sha256"]:
                        raise ValueError("A valid or ineligible original score journal was changed")
                    reconstructed[trait]["actual_attempts"] += len(saved["attempts"])
                    reconstructed[trait]["valid_scores" if saved["score"] is not None else
                                         "eligible_http429_none" if eligible(saved) else "other_none"] += 1
                    continue
                original_path = root / STAGE / "originals" / trait / target.name
                original, original_digest = original_cache[trait, target.name]
                meta = saved["transport_recovery"]
                additions = saved["attempts"][3:]
                if (not eligible(original) or original["score"] is not None
                        or any(original.get(key) != value for key, value in identity.items())
                        or original["attempts"] != saved["attempts"][:3]
                        or meta["original_file_sha256"] != original_digest
                        or baseline.get(f"{trait}/{target.name}") != original_digest
                        or meta["original_attempts_sha256"] != fingerprint(original["attempts"])
                        or meta["policy_sha256"] != stage["policy_sha256"]
                        or meta["implementation_sha256"] != stage["implementation_sha256"]
                        or meta.get("stage") != STAGE or meta.get("original_attempts_count") != 3
                        or not 1 <= len(additions) <= 3
                        or any(attempt.get("stage") != STAGE or attempt.get("attempt") != index + 4
                               or attempt.get("recovery_attempt") != index + 1
                               or attempt.get("status") not in ("complete", "judge_error")
                               for index, attempt in enumerate(additions))
                        or any(attempt.get("http_status") != 429 or attempt.get("status") != "judge_error"
                               for attempt in additions[:-1])
                        or ((saved["score"] is not None) != (additions[-1].get("status") == "complete"))
                        or (saved["score"] is not None and additions[-1].get("score") != saved["score"])):
                    raise ValueError("Transport recovery violated the immutable original journal/budget")
                reconstructed[trait]["actual_attempts"] += 3
                reconstructed[trait]["eligible_http429_none"] += 1
                originals[f"{trait}/{target.name}"] = original_digest
                counts["records_touched"] += 1
                counts["additional_attempts"] += len(additions)
                counts["recovered_scores"] += saved["score"] is not None
                counts["remaining_none"] += saved["score"] is None
                counts["original_http429_attempts"] += 3
                counts["http429_recovery_attempts"] += sum(attempt.get("http_status") == 429 for attempt in additions)
    if any(reconstructed[trait][key] != audit["before"]["by_trait"][trait][key]
           for trait in TRAITS for key in reconstructed[trait]):
        raise ValueError("Recovery before-audit does not match the original journals")
    if (counts["records_touched"] != audit["before"]["eligible_http429_none"]
            or audit["after"]["valid_scores"] != audit["before"]["valid_scores"] + counts["recovered_scores"]
            or audit["after"]["actual_attempts"] != audit["before"]["actual_attempts"] + counts["additional_attempts"]
            or audit["after"]["original_actual_attempts"] != audit["before"]["original_actual_attempts"]
            or audit["after"]["eligible_http429_none"] != counts["remaining_none"]):
        raise ValueError("Recovery score/attempt counts are inconsistent")
    public = {"stage": STAGE, "policy_source": POLICY_SOURCE, "policy_sha256": stage["policy_sha256"],
              "policy": stage["policy"], "implementation": stage, "audit": audit,
              "audit_sha256": file_hash(root / STAGE / "audit.json"), "counts": counts,
              "original_journal_hashes_sha256": fingerprint(originals),
              "baseline_journal_hashes_sha256": file_hash(baseline_path),
              "valid_and_ineligible_original_journals_preserved": True}
    return public, {POLICY_SOURCE: {"sha256": stage["policy_sha256"], "bytes": stage["policy_bytes"]}}
