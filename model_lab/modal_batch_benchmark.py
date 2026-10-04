"""One isolated H100, ordinary prompts only, no extraction dataset writes."""
import json
from pathlib import Path
import modal
from model_lab.modal_persona import gpu_image, hf_secret, volume

app = modal.App("mooody-persona-batch-benchmark")
WORKSPACE = Path(__file__).resolve().parents[1]


@app.function(image=gpu_image, gpu="H100", cpu=4, memory=65536,
              volumes={"/artifacts": volume}, secrets=[hf_secret], timeout=1800,
              max_containers=1, scaledown_window=5)
def benchmark():
    import copy
    import hashlib
    import importlib.metadata
    import inspect
    import time
    import torch
    from transformers import LogitsProcessorList
    from transformers.generation.utils import GenerationMixin
    from deployment.checkpoint import EXISTING_CHECKPOINT, MANIFEST, verify_checkpoint
    from model_lab.persona_extraction import (
        MODEL_ID, MODEL_REVISION, content_indices, fingerprint, load_model, read_json, render_tokens,
    )
    from model_lab.seeded_sampling_benchmark import (
        IndependentRowSampler, left_padded_inputs, validate_supported_config,
    )

    volume.reload()
    audit = read_json(MANIFEST)
    checkpoint = EXISTING_CHECKPOINT
    if not checkpoint.is_dir() or not verify_checkpoint(checkpoint, audit):
        raise RuntimeError("Benchmark requires the existing audited cached checkpoint; no download/write fallback")
    tokenizer, model, adapter = load_model(checkpoint)
    config, _ = model._prepare_generation_config(
        None, do_sample=True, temperature=1.0, top_p=1.0, max_new_tokens=100,
        use_cache=True, pad_token_id=tokenizer.pad_token_id)
    config = validate_supported_config(config)
    collection_config, _ = model._prepare_generation_config(
        None, do_sample=True, temperature=1.0, top_p=1.0, max_new_tokens=1000,
        use_cache=True, pad_token_id=tokenizer.pad_token_id)
    questions = [
        "Explain how a bicycle moves, starting with the pedals and following the energy through the wheels. Provide a detailed explanation with several examples.",
        "Describe how to organize a small home library. Give a substantial practical checklist covering sorting, shelf labels, upkeep, and finding books later.",
        "Explain how a rain gauge works and how to keep a consistent daily rainfall log. Include several examples and common measurement mistakes.",
        "Describe the process of making a simple paper airplane. Give a detailed step-by-step explanation of folding, balance, and adjusting its flight.",
        "Explain why a ceramic mug and a metal spoon can feel different at the same room temperature. Include several everyday examples and a thorough explanation.",
        "Describe how to plan a weekly grocery list for one adult who cooks at home. Give a detailed checklist, sample categories, and practical ways to reduce waste.",
        "Explain how an ordinary mechanical clock keeps time. Follow the parts from the energy source through the gears and hands, and provide several clear examples.",
        "Describe how to care for a windowsill herb garden with basil, mint, and parsley. Give a long practical checklist covering light, water, soil, harvesting, and common mistakes.",
    ]
    messages = [[{"role": "system", "content": "Answer the user's ordinary practical question clearly and accurately."},
                 {"role": "user", "content": question}] for question in questions]
    prompts = [render_tokens(tokenizer, row)[1] for row in messages]
    seeds = [73101 + index * 101 for index in range(8)]
    eos = config.eos_token_id
    eos_ids = {eos} if isinstance(eos, int) else set(eos or [])

    def run(indices, seeded, max_tokens=100):
        selected = [prompts[index] for index in indices]
        input_ids, mask = left_padded_inputs(selected, tokenizer.pad_token_id, adapter.device)
        for row, original in zip(input_ids, selected):
            if row[-len(original):].tolist() != original:
                raise ValueError("Left padding changed original prompt token IDs")
        settings = copy.deepcopy(config)
        settings.max_new_tokens = max_tokens
        processor = IndependentRowSampler([seeds[index] for index in indices], settings, adapter.device) if seeded else None
        model.model.rope_deltas = None
        if not seeded:
            if len(indices) != 1:
                raise ValueError("Native seed baseline is deliberately serial")
            torch.manual_seed(seeds[indices[0]])
            torch.cuda.manual_seed_all(seeds[indices[0]])
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        start = time.monotonic()
        with torch.inference_mode():
            output = model.generate(input_ids=input_ids, attention_mask=mask,
                                    generation_config=settings, logits_to_keep=1,
                                    logits_processor=LogitsProcessorList([processor]) if processor else None)
        torch.cuda.synchronize()
        duration = time.monotonic() - start
        result_ids = []
        stats = []
        for index, row in zip(indices, output[:, input_ids.shape[1]:].cpu().tolist()):
            end = next((position for position, token in enumerate(row) if token in eos_ids), None)
            ids = row[:end + 1] if end is not None else row
            content, thinking = content_indices(tokenizer, ids, eos_ids)
            if thinking:
                raise ValueError("Benchmark unexpectedly generated a reasoning span")
            result_ids.append(ids)
            stats.append({"prompt_index": index, "seed": seeds[index], "prompt_tokens": len(prompts[index]),
                          "generated_tokens": len(ids), "content_tokens": len(content),
                          "stop_reason": "eos" if end is not None else "token_cap",
                          "prompt_ids_sha256": fingerprint(prompts[index]),
                          "generated_ids_sha256": fingerprint(ids)})
        metadata = {"batch_size": len(indices), "independent_row_rng": seeded, "seconds": duration,
                    "aggregate_tokens_per_second": sum(len(ids) for ids in result_ids) / duration,
                    "peak_allocated_gpu_bytes": torch.cuda.max_memory_allocated(),
                    "peak_reserved_gpu_bytes": torch.cuda.max_memory_reserved(),
                    "left_padding_verified": True, "rows": stats}
        del output, input_ids, mask
        return metadata, result_ids

    # Initial JIT/load effects are excluded from the reported warm batch trials.
    warmup, _ = run([0], False, 16)
    serial = []
    native_ids = []
    for index in range(2):
        metadata, ids = run([index], False)
        serial.append(metadata)
        native_ids.append(ids[0])
    results = []
    seeded_ids = {}
    for size in (1, 2, 4, 8):
        run(list(range(size)), True, 16)  # shape/kernel warmup, ordinary prompts only
        metadata, ids = run(list(range(size)), True)
        results.append(metadata)
        seeded_ids[size] = ids
    _, reordered = run([3, 1, 0, 2], True, 20)
    order_comparison = [reordered[position] == seeded_ids[4][index][:len(reordered[position])]
                        for position, index in enumerate((3, 1, 0, 2))]
    device = torch.cuda.get_device_properties(0)
    result = {"scope": "isolated ordinary-prompt batching benchmark; no dataset rollouts or writes",
            "model_id": MODEL_ID, "model_revision": MODEL_REVISION,
            "gpu": device.name, "torch_version": str(torch.__version__),
            "transformers_version": importlib.metadata.version("transformers"),
            "effective_generation_config": config.to_dict(),
            "collection_resolved_generation_config": collection_config.to_dict(),
            "collection_resolved_generation_config_sha256": fingerprint(collection_config.to_dict()),
            "collection_sampling_parameters": {name: getattr(collection_config, name) for name in
                ("do_sample", "temperature", "top_p", "top_k", "repetition_penalty",
                 "renormalize_logits", "max_new_tokens", "use_cache", "pad_token_id", "eos_token_id")},
            "native_generation_config_resolution_source_sha256": hashlib.sha256(inspect.getsource(GenerationMixin._prepare_generation_config).encode()).hexdigest(),
            "native_generation_processor_source_sha256": hashlib.sha256(inspect.getsource(GenerationMixin._get_logits_processor).encode()).hexdigest(),
            "native_sampling_source_sha256": hashlib.sha256(inspect.getsource(GenerationMixin._sample).encode()).hexdigest(),
            "warmup": warmup, "native_serial": serial, "seeded_batches": results,
            "batch1_native_exact_ids_match": native_ids[0] == seeded_ids[1][0],
            "batch2_native_exact_ids_match": [native_ids[index] == seeded_ids[2][index] for index in range(2)],
            "batch4_reorder_first20_exact_ids_match": order_comparison,
            "main_run_modified": False}
    # Local orchestration deliberately has no Torch dependency. A plain JSON
    # boundary prevents version-string subclasses from entering Modal pickle.
    return json.dumps(result)


@app.local_entrypoint()
def main():
    result = json.loads(benchmark.remote())
    directory = WORKSPACE / "artifacts/persona"
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "batch_benchmark_result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"gpu": result["gpu"], "collection_sampling_parameters": result["collection_sampling_parameters"],
                      "native": [{key: row[key] for key in ("batch_size", "seconds", "aggregate_tokens_per_second")}
                                 for row in result["native_serial"]],
                      "batches": [{key: row[key] for key in ("batch_size", "seconds", "aggregate_tokens_per_second", "peak_allocated_gpu_bytes")}
                                  for row in result["seeded_batches"]],
                      "native_batch1_match": result["batch1_native_exact_ids_match"],
                      "native_batch2_match": result["batch2_native_exact_ids_match"],
                      "reordered_batch4_match": result["batch4_reorder_first20_exact_ids_match"]}, indent=2))
