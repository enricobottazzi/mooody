"""Resumable persona collection, blinded judging and response-pooled extraction.

Raw transcripts and judge attempts stay in the private Modal run directory.
Only the final tensor and its aggregate provenance manifest are publication
artifacts. No steering or weight edits occur in this module.
"""

from __future__ import annotations

import contextlib
import hashlib
import importlib.metadata
import inspect
import json
import os
from pathlib import Path
import re
import shutil
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


TRAITS = ("depression", "curiosity", "paranoia", "sexual_arousal", "narcissism", "euphoria")
BOUNDARY = "decoder_block_output_pre_final_global_rmsnorm"
POOLING = "response_content_mean_then_equal_response_group_mean"
MODEL_ID = "demivoleegaston/Qwen3.5-9B-mooody"
MODEL_REVISION = "705afd95bced3ac0424d7e68b1299d8fcdffb858"
_SCORE = re.compile(r"(?:100|[1-9]?[0-9])\Z")
# Narrow task-refusal heuristic: target persona statements such as 'I cannot
# stop thinking about ...' are not refusals. The method is recorded, not claimed
# to be a clinical or semantic classifier.
_REFUSAL = re.compile(
    r"\A\s*(?:(?:I'm sorry|I am sorry|Sorry|As an AI[^,.]*|As a language model)[,.!:\s]*)*"
    r"I\s+(?:cannot|can't|won't|will not|am unable to|must decline to)\s+"
    r"(?:help|assist|provide|comply|fulfil|fulfill|write|generate|discuss)\b", re.I)


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fingerprint(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     allow_nan=False).encode()).hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, filename = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(filename)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def log(event: str, **fields) -> None:
    print(json.dumps({"event": event, **fields}, sort_keys=True), flush=True)


def stable_seed(trait: str, pair_id: str, question_id: str, polarity: str,
                base_seed: int = 20261004) -> int:
    text = f"{base_seed}|{trait}|{pair_id}|{question_id}|0|{polarity}"
    return int(hashlib.sha256(text.encode()).hexdigest()[:8], 16)


def load_inputs(directory: Path) -> tuple[dict, dict[str, dict], dict]:
    protocol = read_json(directory / "protocol.json")
    manifest = read_json(directory / "manifest.json")
    if tuple(protocol["trait_order"]) != TRAITS:
        raise ValueError("Unexpected trait order")
    checkpoint = protocol["checkpoint"]
    if (checkpoint["model_id"], checkpoint["revision"]) != (MODEL_ID, MODEL_REVISION):
        raise ValueError("Extraction requires the pinned audited abliterated checkpoint")
    generation = protocol["generation"]
    if (generation["system_prompt_pairs_per_trait"], generation["extraction_questions_per_trait"],
        generation["rollouts_per_question_per_system_prompt_per_polarity"],
        generation["total_responses_before_filtering"]) != (5, 40, 1, 2400):
        raise ValueError("Expected exactly 2400 single-rollout responses")
    if (generation["temperature"], generation["top_p"], generation["max_new_tokens"]) != (1.0, 1.0, 1000):
        raise ValueError("Extraction sampling differs from frozen protocol")
    paths = ["protocol.json", "manifest.json", "coherence_evaluation_prompt.txt",
             *(f"source/{trait}.json" for trait in TRAITS)]
    sources = {}
    for relative in paths:
        path = directory / relative
        digest = file_hash(path)
        if relative != "manifest.json" and manifest["sources"][relative]["sha256"] != digest:
            raise ValueError(f"Stale input manifest: {relative}")
        sources[f"data/persona_traits/{relative}"] = {"sha256": digest, "bytes": path.stat().st_size}
    traits = {trait: read_json(directory / "source" / f"{trait}.json") for trait in TRAITS}
    for trait, data in traits.items():
        if data["trait"] != trait or len(data["system_prompt_pairs"]) != 5 or len(data["questions"]) != 40:
            raise ValueError(f"Incomplete trait inputs: {trait}")
        if [p["id"] for p in data["system_prompt_pairs"]] != [f"{i:02d}" for i in range(1, 6)]:
            raise ValueError(f"Unexpected system-prompt IDs: {trait}")
        if [q["id"] for q in data["questions"]] != [f"{i:02d}" for i in range(1, 41)] or any(
                q["split"] != "extraction" for q in data["questions"]):
            raise ValueError(f"Unexpected extraction-question IDs: {trait}")
    return protocol, traits, sources


def initialize_run(root: Path, inputs: Path, run_id: str, commit: Callable = lambda: None) -> dict:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", run_id):
        raise ValueError("Unsafe run ID")
    protocol, _, sources = load_inputs(inputs)
    settings = {"schema_version": 1, "run_id": run_id, "protocol": protocol,
                "sources": sources,
                "generation_rollouts": 2400, "activation_boundary": BOUNDARY}
    path = root / "run_settings.json"
    if path.exists() and read_json(path) != settings:
        raise ValueError("Frozen run inputs or implementation changed; use a new run ID")
    atomic_json(path, settings)
    provenance = root / "provenance"
    provenance.mkdir(exist_ok=True)
    for relative in ("protocol.json", "manifest.json", "coherence_evaluation_prompt.txt"):
        shutil.copyfile(inputs / relative, provenance / relative)
    commit()
    return {"run_id": run_id, "settings_sha256": fingerprint(settings), "responses": 2400}


def frozen_settings(root: Path, inputs: Path) -> tuple[dict, dict, dict]:
    settings = read_json(root / "run_settings.json")
    protocol, traits, sources = load_inputs(inputs)
    if settings["protocol"] != protocol or settings["sources"] != sources:
        raise ValueError("Frozen run provenance changed")
    return settings, protocol, traits


def freeze_stage(root: Path, stage: str, policy_sha256: str | None = None) -> dict:
    """Freeze each stage's implementation independently of later-stage fixes.

    A judge/replay-only correction need not discard already generated samples.
    Changing a completed stage itself fails closed, requiring a separately
    documented repair or new run rather than silently mixing implementations.
    """
    common = [file_hash, fingerprint, read_json, atomic_json, load_inputs,
              frozen_settings, conditions, stable_seed, record_path]
    functions = {
        "generation": [generate_trait, render_tokens, content_indices, classify_response,
                       load_model, checkpoint_metadata],
        "judging": [judge_trait, judge_one, judge_payload, judge_result, parse_score,
                    response_providers, judge_path],
        "filtering": [filter_trait, matched_decision, exclusion_reasons, judge_path, freeze_analysis_policy],
        "replay": [replay_trait, replay_response, load_model, save_tensors, response_mean_difference,
                   freeze_analysis_policy],
        "assembly": [assemble_bank, save_tensors, response_providers, freeze_analysis_policy],
    }[stage]
    patterns = {}
    if stage == "generation":
        patterns["refusal_pattern"] = _REFUSAL.pattern
    elif stage == "judging":
        patterns["score_pattern"] = _SCORE.pattern
    implementation_inputs = {"functions": {function.__name__: inspect.getsource(function)
                                             for function in common + functions},
                             "trait_order": TRAITS, "boundary": BOUNDARY, "pooling": POOLING,
                             "patterns": patterns}
    if stage in ("filtering", "replay", "assembly"):
        if not isinstance(policy_sha256, str) or not re.fullmatch(r"[0-9a-f]{64}", policy_sha256):
            raise ValueError("Analytical stages require a frozen extraction-policy hash")
        implementation_inputs["analysis_policy_sha256"] = policy_sha256
    implementation = fingerprint(implementation_inputs)
    path = root / "implementations" / f"{stage}.json"
    if path.exists():
        saved = read_json(path)
        if saved["implementation_sha256"] != implementation:
            raise ValueError(f"Frozen {stage} implementation changed; preserve original samples and document a repair")
        return saved
    saved = {"stage": stage, "implementation_sha256": implementation,
             "module_sha256": file_hash(Path(__file__))}
    if policy_sha256 is not None:
        saved["analysis_policy_sha256"] = policy_sha256
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, filename = tempfile.mkstemp(prefix=f".{stage}.", suffix=".py", dir=path.parent)
    snapshot = Path(filename)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(Path(__file__).read_bytes())
        os.replace(snapshot, path.with_suffix(".py"))
    finally:
        snapshot.unlink(missing_ok=True)
    atomic_json(path, saved)
    return saved


def conditions(trait: str, data: dict, protocol: dict) -> list[dict]:
    if trait not in TRAITS:
        raise ValueError("Unknown trait")
    rows = []
    for pair in data["system_prompt_pairs"]:
        for question in data["questions"]:
            for polarity in ("positive", "negative"):
                contrast_id = f"{trait}:{pair['id']}:{question['id']}:0"
                identity = contrast_id + ":" + polarity
                rows.append({"response_id": identity, "contrast_id": contrast_id, "trait": trait,
                             "pair_id": pair["id"], "question_id": question["id"],
                             "rollout_index": 0, "polarity": polarity,
                             "seed": stable_seed(trait, pair["id"], question["id"], polarity,
                                                 protocol["generation"]["base_seed"]),
                             "messages": [{"role": "system", "content": pair[polarity]},
                                          {"role": "user", "content": question["question"]}],
                             "question": question["question"]})
    return rows


def record_path(root: Path, trait: str, identity: str) -> Path:
    return root / "responses" / trait / (identity.replace(":", "_") + ".json")


def render_tokens(tokenizer, messages: list[dict]) -> tuple[str, list[int]]:
    rendered = tokenizer.apply_chat_template(messages, tokenize=False,
                                             return_dict=False,
                                             add_generation_prompt=True, enable_thinking=False)
    ids = tokenizer.encode(rendered, add_special_tokens=False)
    # Transformers 5.18 defaults to BatchEncoding. Request the original ID
    # list explicitly so this comparison checks tokenization, not container type.
    native = tokenizer.apply_chat_template(messages, tokenize=True,
                                           return_dict=False,
                                           add_generation_prompt=True, enable_thinking=False)
    if not isinstance(native, list) or any(type(token) is not int for token in native):
        raise ValueError("Pinned native chat tokenization did not return a flat integer ID list")
    if ids != native or not ids:
        raise ValueError("Pinned native chat tokenization differs from encoding its rendered template")
    if not rendered.endswith("<think>\n\n</think>\n\n"):
        raise ValueError("Pinned chat template lacks the expected no-thinking generation suffix")
    return rendered, ids


def verify_tokenizer_files(checkpoint: Path, manifest: dict) -> dict:
    """Verify the small audited tokenizer files without reading model weights."""
    if (manifest["model_id"], manifest["revision"]) != (MODEL_ID, MODEL_REVISION):
        raise ValueError("Tokenizer audit requires the pinned checkpoint manifest")
    hashes = {}
    for name in ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja"):
        path = checkpoint / name
        expected = manifest["files"][name]
        if (not path.is_file() or path.stat().st_size != expected["bytes"]
                or file_hash(path) != expected["sha256"]):
            raise ValueError(f"Cached pinned tokenizer integrity check failed: {name}")
        hashes[name] = dict(expected)
    return hashes


def template_diagnostics(tokenizer, inputs: Path) -> dict:
    """Check every collection prompt on CPU; report hashes/counts, never text."""
    protocol, traits, _ = load_inputs(inputs)
    probe = [{"role": "system", "content": "Answer the user's question."},
             {"role": "user", "content": "What is two plus two?"}]
    default = tokenizer.apply_chat_template(probe, tokenize=True,
                                            add_generation_prompt=True, enable_thinking=False)
    rendered, ids = render_tokens(tokenizer, probe)
    default_ids = default["input_ids"] if hasattr(default, "keys") else default
    if default_ids != ids:
        raise ValueError("Default native chat output contains different token IDs")
    counts = {}
    digests = {}
    for trait in TRAITS:
        lengths = []
        digest = hashlib.sha256()
        for condition in conditions(trait, traits[trait], protocol):
            _, prompt_ids = render_tokens(tokenizer, condition["messages"])
            lengths.append(len(prompt_ids))
            digest.update(fingerprint({"id": condition["response_id"], "ids": prompt_ids}).encode())
        counts[trait] = {"checked": len(lengths), "min_prompt_tokens": min(lengths),
                         "max_prompt_tokens": max(lengths)}
        digests[trait] = digest.hexdigest()
    return {"passed": True, "checked_conditions": sum(row["checked"] for row in counts.values()),
            "tokenizer_class": type(tokenizer).__name__,
            "default_tokenized_output_type": type(default).__name__,
            "default_output_keys": sorted(default.keys()) if hasattr(default, "keys") else [],
            "explicit_return_dict_false_output_type": "list",
            "encode_and_native_ids_identical": True, "no_thinking_suffix_verified": True,
            "probe_prompt_tokens": len(ids), "probe_prompt_sha256": fingerprint(ids),
            "traits": counts, "prompt_id_hashes": digests}


def content_indices(tokenizer, ids: list[int], eos_ids: set[int]) -> tuple[list[int], bool]:
    """Preserve original IDs; mask special/control tokens and reasoning spans."""
    excluded = set(tokenizer.all_special_ids) | eos_ids
    thinking = False
    unexpected = False
    indices = []
    for index, token in enumerate(ids):
        piece = tokenizer.decode([token], skip_special_tokens=False)
        if piece == "<think>":
            thinking = unexpected = True
            continue
        if piece == "</think>":
            thinking = False
            unexpected = True
            continue
        if token in excluded or re.fullmatch(r"<\|[^>]+\|>", piece):
            continue
        if not thinking:
            indices.append(index)
    return indices, unexpected


def classify_response(response: str) -> dict:
    return {"empty_response": not response.strip(), "refusal": bool(_REFUSAL.search(response)),
            "refusal_method": "task_refusal_opening_v1_narrow_lexical_heuristic"}


def load_model(checkpoint: Path):
    import torch
    from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration
    from model_lab.abliteration import Qwen35Adapter

    tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True)
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        checkpoint, torch_dtype=torch.bfloat16, device_map="cuda:0",
        attn_implementation="sdpa", local_files_only=True).eval()
    model.requires_grad_(False)
    adapter = Qwen35Adapter(model)
    return tokenizer, model, adapter


def checkpoint_metadata(checkpoint: Path, adapter) -> dict:
    names = ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja")
    hashes = {name: {"sha256": file_hash(checkpoint / name),
                     "bytes": (checkpoint / name).stat().st_size} for name in names}
    return {"checkpoint": {"model_id": MODEL_ID, "revision": MODEL_REVISION,
                           "config_sha256": file_hash(checkpoint / "config.json"),
                           "decoder_layers": adapter.n_layers, "hidden_size": adapter.hidden_size,
                           "weight_dtype": "bfloat16",
                           "versions": {name: importlib.metadata.version(name) for name in
                                        ("torch", "transformers", "flash-linear-attention", "safetensors")}},
            "tokenizer": {"model_id": MODEL_ID, "revision": MODEL_REVISION, "file_hashes": hashes}}


def generate_trait(root: Path, inputs: Path, checkpoint: Path, trait: str,
                   commit: Callable = lambda: None, limit: int = 0) -> dict:
    import torch
    settings, protocol, traits = frozen_settings(root, inputs)
    freeze_stage(root, "generation")
    rows = conditions(trait, traits[trait], protocol)
    pending = []
    for row in rows:
        path = record_path(root, trait, row["response_id"])
        if path.exists():
            saved = read_json(path)
            if saved["condition_sha256"] != fingerprint(row) or saved["settings_sha256"] != fingerprint(settings):
                raise ValueError("Response resume identity mismatch")
        else:
            pending.append(row)
    if limit:
        pending = pending[:limit]
    if not pending:
        return {"trait": trait, "pending": 0, "generated": len(rows)}
    tokenizer, model, adapter = load_model(checkpoint)
    atomic_json(root / "models" / f"{trait}.json", checkpoint_metadata(checkpoint, adapter))
    eos = model.generation_config.eos_token_id
    eos_ids = {eos} if isinstance(eos, int) else set(eos or [])
    generation = protocol["generation"]
    for index, row in enumerate(pending):
        result = {**row, "condition_sha256": fingerprint(row), "settings_sha256": fingerprint(settings),
                  "generation_settings": {"do_sample": True, "temperature": generation["temperature"],
                                          "top_p": generation["top_p"], "max_new_tokens": 1000,
                                          "enable_thinking": False},
                  "model_id": MODEL_ID, "model_revision": MODEL_REVISION}
        result["effective_generation_config"] = {
            **model.generation_config.to_dict(), "do_sample": True,
            "temperature": 1.0, "top_p": 1.0, "max_new_tokens": 1000,
            "use_cache": True, "pad_token_id": tokenizer.pad_token_id}
        started = time.monotonic()
        try:
            rendered, prompt_ids = render_tokens(tokenizer, row["messages"])
            result.update(rendered_prompt=rendered, prompt_token_ids=prompt_ids)
            tensor = torch.tensor([prompt_ids], dtype=torch.long, device=adapter.device)
            model.model.rope_deltas = None
            torch.manual_seed(row["seed"])
            torch.cuda.manual_seed_all(row["seed"])
            with torch.inference_mode():
                generated = model.generate(input_ids=tensor, attention_mask=torch.ones_like(tensor),
                                           do_sample=True, temperature=1.0, top_p=1.0,
                                           max_new_tokens=1000, use_cache=True,
                                           logits_to_keep=1,
                                           pad_token_id=tokenizer.pad_token_id)
            ids = generated[0, len(prompt_ids):].detach().cpu().tolist()
            ended = next((i for i, token in enumerate(ids) if token in eos_ids), None)
            if ended is not None:
                ids = ids[:ended + 1]
            positions, unexpected = content_indices(tokenizer, ids, eos_ids)
            response = tokenizer.decode([ids[i] for i in positions], skip_special_tokens=False).strip()
            result.update(status="complete", response_token_ids=ids,
                          response_content_indices=positions, response=response,
                          raw_response=tokenizer.decode(ids, skip_special_tokens=False),
                          generated_token_count=len(ids), pooled_token_count=len(positions),
                          unexpected_thinking=unexpected,
                          stop_reason="eos" if ended is not None else "token_limit_truncation",
                          **classify_response(response))
            del generated, tensor
        except Exception as exc:
            result.update(status="generation_error", error_type=type(exc).__name__,
                          error=str(exc)[:1500], response="", response_token_ids=[],
                          response_content_indices=[], generated_token_count=0,
                          empty_response=True, refusal=False, stop_reason="generation_error")
        result["generation_seconds"] = time.monotonic() - started
        atomic_json(record_path(root, trait, row["response_id"]), result)
        # Each completed response is committed before starting the next. Distinct
        # trait workers write disjoint paths, so retries never overwrite samples.
        commit()
        if index % 10 == 0 or index + 1 == len(pending):
            log("generation_progress", trait=trait, newly_generated=index + 1, pending=len(pending))
    return progress(root, inputs, trait)


def parse_score(content: Any) -> int:
    if not isinstance(content, str) or not _SCORE.fullmatch(content.strip()):
        raise ValueError("malformed_or_out_of_range_score")
    return int(content.strip())


def judge_payload(protocol: dict, rubric: str, question: str, response: str) -> dict:
    judging = protocol["judging"]
    # Single-pass substitution prevents braces in untrusted inputs from being
    # treated as replacement fields or second-pass instructions.
    prompt = re.sub(r"\{question\}|\{response\}",
                    lambda match: question if match.group() == "{question}" else response, rubric)
    return {"model": judging["model"], "messages": [{"role": "user", "content": prompt}],
            "provider": judging["provider"], "reasoning": judging["reasoning"],
            "max_completion_tokens": judging["max_completion_tokens"], "stream": False}


def judge_result(payload: dict) -> int:
    if payload.get("error"):
        raise ValueError("http_or_api_error")
    choice = payload["choices"][0]
    message = choice["message"]
    if message.get("refusal"):
        raise ValueError("judge_refusal")
    if choice.get("finish_reason") == "length":
        raise ValueError("judge_token_limit_truncation")
    if choice.get("finish_reason") in ("content_filter", "error"):
        raise ValueError("content_filter_or_safety_block")
    if not message.get("content"):
        raise ValueError("empty_judge_content")
    return parse_score(message["content"])


def response_providers(payload: dict) -> list[str]:
    providers = []
    if isinstance(payload.get("provider"), str):
        providers.append(payload["provider"])
    endpoints = payload.get("openrouter_metadata", {}).get("endpoints", {}).get("available", [])
    providers.extend(endpoint["provider"] for endpoint in endpoints
                     if endpoint.get("selected") and isinstance(endpoint.get("provider"), str))
    return sorted(set(providers))


def judge_one(path: Path, protocol: dict, rubric: str, record: dict, kind: str,
              transport: Callable | None = None) -> dict:
    key = os.environ.get(protocol["judging"]["api_key_env"])
    if transport is None and not key:
        raise RuntimeError("OPENROUTER_API_KEY is not available in the judge environment")
    request_payload = judge_payload(protocol, rubric, record["question"], record["response"])
    identity = {"response_id": record["response_id"], "response_sha256": fingerprint(record),
                "judge_kind": kind, "judge_prompt_sha256": hashlib.sha256(rubric.encode()).hexdigest(),
                "request_sha256": fingerprint(request_payload)}
    saved = read_json(path) if path.exists() else {**identity, "attempts": [], "score": None}
    if any(saved.get(field) != value for field, value in identity.items()):
        raise ValueError("Judge resume identity mismatch")
    if saved["score"] is not None or len(saved["attempts"]) >= 3:
        return saved
    while len(saved["attempts"]) < 3:
        attempt = {"attempt": len(saved["attempts"]) + 1, "status": "request_started",
                   "started_at_unix": time.time()}
        saved["attempts"].append(attempt)
        atomic_json(path, saved)
        try:
            if transport is not None:
                body, headers = transport(request_payload)
            else:
                headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json",
                           **protocol["judging"]["request_headers"]}
                request = Request(protocol["judging"]["base_url"] + protocol["judging"]["endpoint"],
                                  data=json.dumps(request_payload).encode(), headers=headers, method="POST")
                with urlopen(request, timeout=120) as reply:
                    body = json.load(reply)
                    headers = {name: value for name, value in reply.headers.items()
                               if name.lower() in ("x-request-id", "x-openrouter-request-id", "date")}
            attempt.update(raw_response=body, response_headers=headers)
            score = judge_result(body)
            providers = response_providers(body)
            if any(re.sub(r"[^a-z]", "", provider.lower()) != "googleaistudio" for provider in providers):
                raise ValueError("unexpected_judge_provider")
            saved["score"] = score
            attempt.update(status="complete", score=score)
        except (HTTPError, URLError, ValueError, KeyError, IndexError, TypeError, TimeoutError) as exc:
            error = str(exc)[:2000]
            if key:
                error = error.replace(key, "[redacted]")
            attempt.update(status="judge_error", error_type=type(exc).__name__, error=error)
            if isinstance(exc, HTTPError):
                attempt["http_status"] = exc.code
                error_body = exc.read().decode(errors="replace")[:2000]
                attempt["http_error_body"] = error_body.replace(key, "[redacted]") if key else error_body
        attempt["finished_at_unix"] = time.time()
        atomic_json(path, saved)
        if saved["score"] is not None:
            break
        if len(saved["attempts"]) < 3 and transport is None:
            time.sleep(min(2 ** len(saved["attempts"]), 8))
    return saved


def judge_path(root: Path, trait: str, identity: str, kind: str) -> Path:
    return root / "judges" / trait / (identity.replace(":", "_") + f".{kind}.json")


def judge_trait(root: Path, inputs: Path, trait: str, commit: Callable = lambda: None,
                concurrency: int = 4) -> dict:
    _, protocol, traits = frozen_settings(root, inputs)
    freeze_stage(root, "judging")
    if not 1 <= concurrency <= 16:
        raise ValueError("Judge concurrency must be between 1 and 16")
    rows = conditions(trait, traits[trait], protocol)
    tasks = []
    coherence = (inputs / "coherence_evaluation_prompt.txt").read_text()
    for row in rows:
        path = record_path(root, trait, row["response_id"])
        if not path.exists():
            raise RuntimeError("Complete generation before judging this trait")
        record = read_json(path)
        if record["status"] == "generation_error" or not record["response"].strip():
            continue
        for kind, rubric in (("trait", traits[trait]["evaluation_prompt"]), ("coherence", coherence)):
            tasks.append((judge_path(root, trait, row["response_id"], kind), protocol, rubric, record, kind))
    # Commit only after a batch of threads has closed all files. Completed
    # attempts survive resumption; a started/interrupted request consumes its
    # attempt slot because its remote outcome and charge cannot be known.
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        for start in range(0, len(tasks), concurrency):
            list(pool.map(lambda arguments: judge_one(*arguments), tasks[start:start + concurrency]))
            commit()
            log("judge_progress", trait=trait, done=min(start + concurrency, len(tasks)), total=len(tasks))
    return filter_trait(root, inputs, trait, commit)


def exclusion_reasons(record: dict) -> list[str]:
    reasons = []
    if record.get("status") != "complete":
        reasons.append("generation_error")
    if record.get("empty_response") or not record.get("response", "").strip():
        reasons.append("empty_response")
    if record.get("refusal"):
        reasons.append("refusal")
    if record.get("unexpected_thinking"):
        reasons.append("unexpected_thinking")
    if not record.get("response_content_indices"):
        reasons.append("no_content_tokens")
    return reasons


def freeze_analysis_policy(root: Path, inputs: Path) -> dict:
    """Version analytical policy separately from immutable collected responses."""
    path = inputs / "extraction_policy.json"
    policy = read_json(path)
    required = {"schema_version": 1, "policy_id": "persona-vectors-filtering-v2",
                "generation_protocol_unchanged": True,
                "positive_trait_threshold": ">50", "negative_trait_threshold": "<50",
                "minimum_coherence_score_both_sides": 50, "matched_pairs": True,
                "allow_judged_capped_responses": True, "require_all_generated_questions": True,
                "require_all_generated_system_prompt_pairs": True,
                "require_all_accepted_questions": False,
                "require_all_accepted_system_prompt_pairs": False,
                "minimum_accepted_pairs_per_trait": 1}
    if any(policy.get(key) != value or type(policy.get(key)) is not type(value)
           for key, value in required.items()):
        raise ValueError("Unsupported extraction policy; preserve the collection protocol")
    receipt = {"policy": policy, "policy_source": "data/persona_traits/extraction_policy.json",
               "policy_sha256": file_hash(path), "policy_bytes": path.stat().st_size}
    saved_path = root / "analysis_policy.json"
    if saved_path.exists():
        if read_json(saved_path) != receipt:
            raise ValueError("Frozen analytical extraction policy changed")
    else:
        atomic_json(saved_path, receipt)
    return receipt


def matched_decision(positive: dict, negative: dict,
                     positive_trait: int | None, negative_trait: int | None,
                     positive_coherence: int | None, negative_coherence: int | None) -> list[str]:
    reasons = [f"positive:{reason}" for reason in exclusion_reasons(positive)]
    reasons.extend(f"negative:{reason}" for reason in exclusion_reasons(negative))
    if any(score is None for score in (positive_trait, negative_trait, positive_coherence, negative_coherence)):
        reasons.append("missing_judge_score")
    else:
        if positive_trait <= 50:
            reasons.append("positive_trait_not_above_50")
        if negative_trait >= 50:
            reasons.append("negative_trait_not_below_50")
        if min(positive_coherence, negative_coherence) < 50:
            reasons.append("coherence_below_50")
    return reasons


def filter_trait(root: Path, inputs: Path, trait: str, commit: Callable = lambda: None) -> dict:
    settings, protocol, traits = frozen_settings(root, inputs)
    policy = freeze_analysis_policy(root, inputs)
    freeze_stage(root, "filtering", policy["policy_sha256"])
    rows = conditions(trait, traits[trait], protocol)
    by_id = {row["response_id"]: row for row in rows}
    records = {}
    scores = {}
    for identity, row in by_id.items():
        path = record_path(root, trait, identity)
        if not path.exists():
            raise RuntimeError("Filtering requires all 400 generation records")
        record = read_json(path)
        if record["condition_sha256"] != fingerprint(row) or record["settings_sha256"] != fingerprint(settings):
            raise ValueError("Filtering response provenance mismatch")
        records[identity] = record
        for kind in ("trait", "coherence"):
            path = judge_path(root, trait, identity, kind)
            if path.exists():
                judge = read_json(path)
                if judge["response_sha256"] != fingerprint(record):
                    raise ValueError("Filtering judge response provenance mismatch")
                scores[identity, kind] = judge["score"]
    accepted = []
    exclusions = {}
    question_counts = {f"{i:02d}": 0 for i in range(1, 41)}
    pair_counts = {f"{i:02d}": 0 for i in range(1, 6)}
    for positive in (row for row in rows if row["polarity"] == "positive"):
        contrast = positive["contrast_id"]
        p, n = contrast + ":positive", contrast + ":negative"
        reasons = matched_decision(records[p], records[n], scores.get((p, "trait")),
                                  scores.get((n, "trait")), scores.get((p, "coherence")),
                                  scores.get((n, "coherence")))
        if reasons:
            exclusions[contrast] = reasons
        else:
            accepted.append(contrast)
            question_counts[positive["question_id"]] += 1
            pair_counts[positive["pair_id"]] += 1
    missing_questions = [key for key, count in question_counts.items() if not count]
    missing_pairs = [key for key, count in pair_counts.items() if not count]
    generated_capped = {polarity: sum(record["polarity"] == polarity
                                      and record.get("stop_reason") == "token_limit_truncation"
                                      for record in records.values())
                        for polarity in ("positive", "negative")}
    accepted_capped = {polarity: sum(records[contrast + ":" + polarity].get("stop_reason")
                                     == "token_limit_truncation" for contrast in accepted)
                       for polarity in ("positive", "negative")}
    summary = {"trait": trait, "generated_positive": 200, "generated_negative": 200,
               "accepted_pairs": len(accepted), "accepted_positive": len(accepted),
               "accepted_negative": len(accepted), "accepted_by_question": question_counts,
               "accepted_by_system_prompt_pair": pair_counts, "accepted_contrast_ids": accepted,
               "exclusions": exclusions, "missing_questions": missing_questions,
               "missing_system_prompt_pairs": missing_pairs,
               "coverage_passed": bool(accepted) and not missing_questions and not missing_pairs,
               "extraction_usable": len(accepted) >= policy["policy"]["minimum_accepted_pairs_per_trait"],
               "policy_sha256": policy["policy_sha256"],
               "generated_capped_positive": generated_capped["positive"],
               "generated_capped_negative": generated_capped["negative"],
               "accepted_capped_positive": accepted_capped["positive"],
               "accepted_capped_negative": accepted_capped["negative"],
               "accepted_capped_pairs": sum(any(records[contrast + ":" + polarity].get("stop_reason")
                                                == "token_limit_truncation"
                                                for polarity in ("positive", "negative"))
                                             for contrast in accepted)}
    atomic_json(root / "filtering" / f"{trait}.json", summary)
    commit()
    log("filter_result", trait=trait, accepted_pairs=len(accepted), coverage_passed=summary["coverage_passed"],
        extraction_usable=summary["extraction_usable"])
    return summary


def replay_response(model, adapter, record: dict):
    """Capture decoder-block outputs directly, never normalized hidden_states."""
    import torch
    prompt = record["prompt_token_ids"]
    response = record["response_token_ids"]
    indices = record["response_content_indices"]
    if not prompt or not indices or any(not 0 <= index < len(response) for index in indices):
        raise ValueError("Invalid exact-token replay mask")
    if len(set(indices)) != len(indices):
        raise ValueError("Duplicate content positions")
    ids = torch.tensor([prompt + response], dtype=torch.long, device=adapter.device)
    positions = torch.tensor([len(prompt) + index for index in indices], dtype=torch.long, device=adapter.device)
    means = [None] * adapter.n_layers
    hooks = []
    def capture(layer_index):
        def hook(module, arguments, output):
            hidden = output[0] if isinstance(output, tuple) else output
            if hidden.shape != (1, len(prompt) + len(response), adapter.hidden_size):
                raise ValueError("Unexpected block-output residual shape")
            if means[layer_index] is not None:
                raise ValueError("Decoder block executed twice during replay")
            means[layer_index] = hidden[0].index_select(0, positions).float().mean(0).detach().cpu()
        return hook
    try:
        for index, layer in enumerate(adapter.layers):
            hooks.append(layer.register_forward_hook(capture(index)))
        model.model.rope_deltas = None
        with torch.inference_mode():
            # Calling the text decoder skips allocating vocabulary-sized logits.
            # No cache object survives the replay. Explicit text positions avoid
            # inheriting generation's multimodal rotary bookkeeping.
            adapter.decoder(input_ids=ids, attention_mask=torch.ones_like(ids),
                            position_ids=torch.arange(ids.shape[1], device=adapter.device).unsqueeze(0),
                            use_cache=False, return_dict=True)
    finally:
        for hook in hooks:
            hook.remove()
        model.model.rope_deltas = None
    if any(value is None for value in means):
        raise ValueError("Replay did not capture every decoder layer")
    result = torch.stack(means)
    if not torch.isfinite(result).all():
        raise ValueError("Non-finite response activation mean")
    return result


def save_tensors(path: Path, tensors: dict, metadata: dict | None = None) -> None:
    from safetensors.torch import save_file
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    save_file(tensors, str(temporary), metadata=metadata)
    os.replace(temporary, path)


def response_mean_difference(sums: dict, counts: dict):
    """Every response contributed one pooled vector, not its token count."""
    if counts["positive"] != counts["negative"] or not counts["positive"]:
        raise RuntimeError("Empty or unbalanced response means")
    return sums["positive"] / counts["positive"] - sums["negative"] / counts["negative"]


def replay_trait(root: Path, inputs: Path, checkpoint: Path, trait: str,
                 commit: Callable = lambda: None) -> dict:
    import torch
    from safetensors import safe_open
    from safetensors.torch import load_file
    frozen_settings(root, inputs)
    policy = freeze_analysis_policy(root, inputs)
    freeze_stage(root, "replay", policy["policy_sha256"])
    filtered = filter_trait(root, inputs, trait, commit)
    if not filtered["extraction_usable"]:
        raise RuntimeError(f"No accepted matched responses for {trait}; do not fabricate vectors")
    _, model, adapter = load_model(checkpoint)
    sums = {polarity: torch.zeros(adapter.n_layers, adapter.hidden_size, dtype=torch.float32)
            for polarity in ("positive", "negative")}
    counts = {polarity: 0 for polarity in sums}
    for contrast in filtered["accepted_contrast_ids"]:
        for polarity in sums:
            identity = contrast + ":" + polarity
            source_path = record_path(root, trait, identity)
            record = read_json(source_path)
            tensor_path = root / "pooled" / trait / (identity.replace(":", "_") + ".safetensors")
            metadata = {"response_sha256": fingerprint(record), "activation_boundary": BOUNDARY,
                        "model_revision": MODEL_REVISION, "pooled_token_count": str(len(record["response_content_indices"]))}
            if tensor_path.exists():
                with safe_open(str(tensor_path), framework="pt", device="cpu") as saved:
                    if saved.metadata() != metadata:
                        raise ValueError("Pooled activation resume provenance mismatch")
                mean = load_file(str(tensor_path), device="cpu")["response_mean"]
            else:
                mean = replay_response(model, adapter, record)
                save_tensors(tensor_path, {"response_mean": mean}, metadata)
                commit()
            if mean.dtype != torch.float32 or mean.shape != sums[polarity].shape or not torch.isfinite(mean).all():
                raise ValueError("Invalid pooled activation artifact")
            sums[polarity].add_(mean)
            counts[polarity] += 1
        if counts["positive"] % 10 == 0:
            log("replay_progress", trait=trait, done_pairs=counts["positive"], total_pairs=filtered["accepted_pairs"])
    vector = response_mean_difference(sums, counts)
    save_tensors(root / "trait_vectors" / f"{trait}.safetensors", {"vectors": vector.contiguous()},
                 {"trait": trait, "filter_sha256": file_hash(root / "filtering" / f"{trait}.json"),
                  "activation_boundary": BOUNDARY, "policy_sha256": policy["policy_sha256"]})
    commit()
    return {"trait": trait, "accepted_pairs": counts["positive"], "shape": list(vector.shape)}


def progress(root: Path, inputs: Path, trait: str | None = None) -> dict:
    _, protocol, traits = frozen_settings(root, inputs)
    result = {}
    for name in ([trait] if trait else TRAITS):
        rows = conditions(name, traits[name], protocol)
        records = [read_json(record_path(root, name, row["response_id"])) for row in rows
                   if record_path(root, name, row["response_id"]).exists()]
        judges = [read_json(path) for path in (root / "judges" / name).glob("*.json")]
        filtered_path = root / "filtering" / f"{name}.json"
        filtered = read_json(filtered_path) if filtered_path.exists() else {}
        result[name] = {"generated": len(records), "generation_errors": sum(row["status"] != "complete" for row in records),
                        "scores": sum(row["score"] is not None for row in judges),
                        "judge_attempts": sum(len(row["attempts"]) for row in judges),
                        "judge_errors": sum(attempt["status"] != "complete" for row in judges for attempt in row["attempts"]),
                        "accepted_pairs": filtered.get("accepted_pairs"),
                        "coverage_passed": filtered.get("coverage_passed"),
                        "vector_ready": (root / "trait_vectors" / f"{name}.safetensors").exists()}
    return {"run_id": read_json(root / "run_settings.json")["run_id"], "traits": result,
            "publication_ready": (root / "publication" / "persona_manifest.json").exists()}


def assemble_bank(root: Path, inputs: Path, commit: Callable = lambda: None) -> dict:
    from concurrent.futures import ThreadPoolExecutor
    import torch
    from safetensors import safe_open
    from safetensors.torch import load_file
    from model_lab.persona_batched import public_generation_provenance
    settings, protocol, _ = frozen_settings(root, inputs)
    policy = freeze_analysis_policy(root, inputs)
    freeze_stage(root, "assembly", policy["policy_sha256"])
    # Each existing filter owns distinct trait files; map preserves trait order.
    with ThreadPoolExecutor(max_workers=6) as pool:
        filters = dict(zip(TRAITS, pool.map(lambda trait: filter_trait(root, inputs, trait), TRAITS)))
    if not all(value["extraction_usable"] for value in filters.values()):
        raise RuntimeError("At least one trait has no accepted matched responses; no bank will be produced")
    if any(value["policy_sha256"] != policy["policy_sha256"] for value in filters.values()):
        raise ValueError("Traits used different extraction policies")
    model_metadata = [read_json(root / "models" / f"{trait}.json") for trait in TRAITS]
    if any(value != model_metadata[0] for value in model_metadata):
        raise ValueError("Trait workers used different checkpoint/tokenizer metadata")
    vectors = []
    for trait in TRAITS:
        path = root / "trait_vectors" / f"{trait}.safetensors"
        with safe_open(str(path), framework="pt", device="cpu") as saved:
            metadata = saved.metadata()
            if (metadata["filter_sha256"] != file_hash(root / "filtering" / f"{trait}.json")
                    or metadata["activation_boundary"] != BOUNDARY
                    or metadata.get("policy_sha256") != policy["policy_sha256"]):
                raise ValueError("Trait-vector filtering provenance mismatch")
        vectors.append(load_file(str(path), device="cpu")["vectors"])
    bank = torch.stack(vectors, dim=1).contiguous()
    expected_shape = (32, 6, model_metadata[0]["checkpoint"]["hidden_size"])
    if bank.dtype != torch.float32 or tuple(bank.shape) != expected_shape or not torch.isfinite(bank).all():
        raise ValueError("Invalid full-layer FP32 bank")
    from model_lab.persona_judging_recovery import public_recovery_provenance
    recovery, recovery_sources = public_recovery_provenance(root, inputs)
    publication = root / "publication"
    tensor_path = publication / "persona_vectors.safetensors"
    save_tensors(tensor_path, {"vectors": bank}, {"trait_order": json.dumps(TRAITS), "activation_boundary": BOUNDARY})
    judge_paths = [path for trait in TRAITS for path in (root / "judges" / trait).glob("*.json")]
    with ThreadPoolExecutor(max_workers=8) as pool:
        judge_records = list(pool.map(read_json, judge_paths))
    attempts = [attempt for record in judge_records for attempt in record["attempts"]]
    reported_cost = sum(float(attempt.get("raw_response", {}).get("usage", {}).get("cost") or 0)
                        for attempt in attempts)
    actual_models = sorted({attempt["raw_response"].get("model", "unknown") for attempt in attempts if "raw_response" in attempt})
    actual_providers = sorted({provider for attempt in attempts if "raw_response" in attempt
                               for provider in response_providers(attempt["raw_response"])})
    provenance = {"private_attempts_fingerprint": fingerprint(judge_records),
                  "actual_response_models": actual_models, "actual_providers": actual_providers,
                  "actual_attempts": len(attempts), "reported_cost_usd": reported_cost,
                  "response_ids_fingerprint": fingerprint([attempt.get("raw_response", {}).get("id")
                                                            for attempt in attempts]),
                  "routing_metadata_fingerprint": fingerprint([attempt.get("raw_response", {}).get("openrouter_metadata")
                                                               for attempt in attempts])}
    manifest = {"schema_version": 1, "artifact_type": "mooody_persona_vector_bank",
                "status": "extracted_not_behaviorally_validated", "run_id": settings["run_id"],
                "trait_order": list(TRAITS), **model_metadata[0],
                "sources": {**settings["sources"], policy["policy_source"]: {
                    "sha256": policy["policy_sha256"], "bytes": policy["policy_bytes"]}, **recovery_sources},
                "implementations": {path.stem: read_json(path) for path in sorted((root / "implementations").glob("*.json"))},
                "generation": {**protocol["generation"], **public_generation_provenance(root, inputs)},
                "filtering": {"traits": {trait: {key: value for key, value in summary.items()
                                                  if key not in ("accepted_contrast_ids", "exclusions")}
                                         for trait, summary in filters.items()},
                              "policy": policy["policy"], "policy_source": policy["policy_source"],
                              "policy_sha256": policy["policy_sha256"],
                              "positive_trait_threshold": ">50", "negative_trait_threshold": "<50",
                              "coherence_minimum": 50, "matched_pairs": True,
                              "refusal_method": "task_refusal_opening_v1_narrow_lexical_heuristic"},
                "judging": {"model": protocol["judging"]["model"], "gateway": "openrouter",
                            "provider": protocol["judging"]["provider"],
                            "reasoning": protocol["judging"]["reasoning"],
                            "canonical_model_snapshot": protocol["judging"]["canonical_model_snapshot"],
                            "planned_scoring_calls_before_retries": 4800,
                            "actual_attempts": len(attempts), "score_records": len(judge_records),
                            "valid_scores": sum(record["score"] is not None for record in judge_records),
                            "actual_response_models": actual_models, "actual_providers": actual_providers,
                            "reported_cost_usd": reported_cost,
                            "private_attempts_fingerprint": fingerprint(judge_records), "provenance": provenance,
                            **({"transport_recovery": recovery} if recovery else {})},
                "extraction": {"activation_boundary": BOUNDARY, "pooling": POOLING,
                               "raw_normalization": "none", "retain_layers": "all_decoder_layers",
                               "select_best_layer": False, "position_axis": False,
                               "accumulator_dtype": "float32", "exact_original_token_replay": True,
                               "steering_enabled": False},
                "tensor": {"filename": tensor_path.name, "key": "vectors", "dtype": "float32",
                           "shape": list(bank.shape), "sha256": file_hash(tensor_path),
                           "bytes": tensor_path.stat().st_size, "all_finite": True,
                           "l2_norms": torch.linalg.vector_norm(bank, dim=-1).tolist(),
                           "zero_vector_indices": (bank.abs().sum(dim=-1) == 0).nonzero().tolist()},
                "inference": {"method": "direct_raw_all_layers", "coefficients": [-2, -1, 0, 1, 2],
                              "activation_boundary": BOUNDARY,
                              "token_scope": "final_formatted_prompt_then_generated_content"},
                "limitations": ["No independent behavioral validation has been run.",
                                "Direct all-layer raw addition differs from the paper's incremental Appendix J.3 method.",
                                "Judge reliability and combined-trait steering efficacy are unmeasured."]}
    atomic_json(publication / "persona_manifest.json", manifest)
    commit()
    return {"run_id": settings["run_id"], "publication_dir": str(publication),
            "tensor_sha256": manifest["tensor"]["sha256"], "shape": list(bank.shape),
            "accepted_pairs": {trait: filters[trait]["accepted_pairs"] for trait in TRAITS}}
