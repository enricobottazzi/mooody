#!/usr/bin/env python3
"""Validate and render Persona Vectors inputs; no generation or judge calls."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import string
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "persona_traits"
TRAITS = ("depression", "curiosity", "paranoia", "sexual_arousal", "narcissism", "euphoria")
SOURCE_FIELDS = {"trait", "display_name", "description", "system_prompt_pairs", "questions", "evaluation_prompt"}
PAIR_FIELDS = {"id", "focus", "positive", "negative"}
QUESTION_FIELDS = {"id", "split", "domain", "question"}
NAMED_TRAITS = re.compile(r"\b(?:depression|curiosity|paranoia|sexual arousal|horniness|narcissism|narcism|euphoria)\b", re.I)
ACTIVITY_PATTERNS = {
    "sex": re.compile(r"\bsex\b", re.I),
    "blowjobs": re.compile(r"\bblowjobs?\b", re.I),
    "oral sex": re.compile(r"\boral sex\b", re.I),
    "fucking": re.compile(r"\bfucking\b", re.I),
    "getting laid": re.compile(r"\bgetting laid\b", re.I),
}


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def json_bytes(value) -> bytes:
    return (json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")


def require_text(value, label: str, maximum: int | None = None):
    if not isinstance(value, str) or not value or value.strip() != value:
        raise ValueError(f"{label}: expected nonempty trimmed text")
    if maximum is not None and len(value) > maximum:
        raise ValueError(f"{label}: exceeds {maximum} characters")


def validate_judge(prompt: str, label: str):
    require_text(prompt, label)
    fields = [name for _, name, _, _ in string.Formatter().parse(prompt) if name is not None]
    if Counter(fields) != Counter({"question": 1, "response": 1}):
        raise ValueError(f"{label}: expected exactly one {{question}} and one {{response}}")
    prompt.format(question="A neutral question.", response="A neutral response.")
    if not all(re.search(rf"\b{score}\b", prompt) for score in (0, 25, 50, 75, 100)):
        raise ValueError(f"{label}: missing score anchors")
    if "integer" not in prompt.lower() or "untrusted" not in prompt.lower():
        raise ValueError(f"{label}: specify integer output and untrusted transcript")


def assemble() -> tuple[dict, list[dict], dict]:
    protocol = load_json(DATA / "protocol.json")
    if tuple(protocol["trait_order"]) != TRAITS:
        raise ValueError("Protocol trait order differs from the authored bank")
    generation = protocol["generation"]
    if (generation["system_prompt_pairs_per_trait"], generation["extraction_questions_per_trait"],
        generation["held_out_evaluation_questions_per_trait"],
        generation["rollouts_per_question_per_system_prompt_per_polarity"]) != (5, 40, 0, 1):
        raise ValueError("Expected the five-pair, 40-extraction-question, single-rollout protocol")
    per_polarity = 5 * 40
    if (generation["responses_per_polarity_per_trait_before_filtering"],
        generation["responses_per_trait_before_filtering"],
        generation["total_responses_before_filtering"]) != (per_polarity, 2 * per_polarity, 12 * per_polarity):
        raise ValueError("Protocol rollout totals are inconsistent")
    extraction = protocol["extraction"]
    if extraction["select_best_layer"] or extraction["position_axis"] or extraction["retain_layers"] != "all_decoder_layers":
        raise ValueError("Extraction must retain all decoder layers without a position axis")
    judging = protocol["judging"]
    if judging["gateway"] != "openrouter" or judging["model"] != "google/gemini-3.8-flash":
        raise ValueError("Expected the Gemini 3.8 Flash judge through OpenRouter")
    if judging["reasoning"] != {"effort": "low", "exclude": True}:
        raise ValueError("Gemini judge must use the documented low reasoning setting")
    if any(parameter in judging for parameter in ("temperature", "top_p", "top_k")):
        raise ValueError("Omit deprecated Gemini 3.8 judge sampling parameters")
    if judging["provider"] != {"order": ["google-ai-studio"], "only": ["google-ai-studio"],
                               "allow_fallbacks": False, "require_parameters": True}:
        raise ValueError("Pin the documented Google AI Studio judge provider without fallbacks")
    if (judging["scoring_calls_per_response"], judging["planned_scoring_calls_before_retries"]) != (2, 4800):
        raise ValueError("Expected one trait score and one coherence score per target response")
    coherence = (DATA / "coherence_evaluation_prompt.txt").read_text(encoding="utf-8").strip()
    validate_judge(coherence, "shared coherence judge")

    traits, checks, all_system_prompts = [], {}, set()
    for trait in TRAITS:
        source = load_json(DATA / "source" / f"{trait}.json")
        if set(source) != SOURCE_FIELDS or source["trait"] != trait:
            raise ValueError(f"{trait}: malformed source fields or trait ID")
        require_text(source["display_name"], f"{trait} display name")
        require_text(source["description"], f"{trait} description")
        pairs = source["system_prompt_pairs"]
        if not isinstance(pairs, list) or len(pairs) != 5:
            raise ValueError(f"{trait}: expected five system-prompt pairs")
        focuses = set()
        sexual_terms = Counter()
        for number, pair in enumerate(pairs, 1):
            if set(pair) != PAIR_FIELDS or pair["id"] != f"{number:02d}":
                raise ValueError(f"{trait}: malformed system pair {number}")
            require_text(pair["focus"], f"{trait}:{pair['id']} focus")
            if pair["focus"] in focuses:
                raise ValueError(f"{trait}: repeated system-prompt focus")
            focuses.add(pair["focus"])
            for side in ("positive", "negative"):
                prompt = pair[side]
                require_text(prompt, f"{trait}:{pair['id']} {side}", 4000)
                if prompt in all_system_prompts:
                    raise ValueError(f"{trait}:{pair['id']}: repeated system prompt")
                all_system_prompts.add(prompt)
            if trait == "sexual_arousal":
                terms = [{name for name, pattern in ACTIVITY_PATTERNS.items() if pattern.search(pair[side])}
                         for side in ("positive", "negative")]
                if not terms[0] or terms[0] != terms[1]:
                    raise ValueError(f"{trait}:{pair['id']}: match named sexual activities on both sides")
                sexual_terms.update(terms[0])

        questions = source["questions"]
        if not isinstance(questions, list) or len(questions) != 40:
            raise ValueError(f"{trait}: expected 40 questions")
        seen, domains = set(), Counter()
        for number, row in enumerate(questions, 1):
            if set(row) != QUESTION_FIELDS or row["id"] != f"{number:02d}" or row["split"] != "extraction":
                raise ValueError(f"{trait}: expected IDs 01..40, all in extraction")
            require_text(row["domain"], f"{trait}:{row['id']} domain")
            require_text(row["question"], f"{trait}:{row['id']} question", 2000)
            key = " ".join(row["question"].casefold().split())
            if key in seen:
                raise ValueError(f"{trait}:{row['id']}: duplicate question")
            if NAMED_TRAITS.search(row["question"]):
                raise ValueError(f"{trait}:{row['id']}: question names a target trait")
            seen.add(key)
            domains[row["domain"]] += 1
        if len(domains) < 5:
            raise ValueError(f"{trait}: cover at least five question domains")
        validate_judge(source["evaluation_prompt"], f"{trait} judge")
        checks[trait] = {
            "system_prompt_pairs": 5,
            "system_prompts": 10,
            "questions": 40,
            "split_counts": {"extraction": 40, "evaluation": 0},
            "question_domains": dict(domains),
            "system_prompt_character_range": [min(len(p[s]) for p in pairs for s in ("positive", "negative")),
                                              max(len(p[s]) for p in pairs for s in ("positive", "negative"))],
            "question_character_range": [min(len(q["question"]) for q in questions),
                                         max(len(q["question"]) for q in questions)],
        }
        if sexual_terms:
            if set(sexual_terms) != set(ACTIVITY_PATTERNS):
                raise ValueError("Sexual system pairs must cover all five requested activity terms")
            checks[trait]["matched_activity_pair_counts"] = dict(sorted(sexual_terms.items()))
        traits.append(source)
    return protocol, traits, checks


def artifacts(protocol: dict, traits: list[dict], checks: dict) -> dict[Path, bytes]:
    outputs = {}
    for source in traits:
        trait = source["trait"]
        sections = [f"# {source['display_name']}: Persona Vectors inputs", source["description"],
                    "Five paired system conditions and 40 questions, all used for extraction. These are authored inputs, not scored model responses.",
                    "## Contrastive system prompts"]
        for pair in source["system_prompt_pairs"]:
            sections.extend([f"### Pair {pair['id']} · {pair['focus']}",
                             f"**Positive system prompt**\n\n{pair['positive']}",
                             f"**Negative system prompt**\n\n{pair['negative']}"])
        sections.append("## Extraction questions · 01–40")
        for question in source["questions"]:
            sections.append(f"{question['id']}. **{question['domain']}** — {question['question']}")
        export = {
            "instruction": [{"pos": pair["positive"], "neg": pair["negative"]}
                            for pair in source["system_prompt_pairs"]],
            "questions": [q["question"] for q in source["questions"]],
            "eval_prompt": source["evaluation_prompt"].replace("{response}", "{answer}"),
        }
        outputs[DATA / "trait_data_extract" / f"{trait}.json"] = json_bytes(export)
        sections.extend(["## Trait-expression judge prompt", "Substitute the literal `{question}` and `{response}` fields. Supply only the question and response; keep generation conditions hidden from the judge.",
                         "```text\n" + source["evaluation_prompt"] + "\n```"])
        outputs[DATA / "review" / f"{trait}.md"] = ("\n\n".join(sections) + "\n").encode("utf-8")

    source_paths = [DATA / "protocol.json", DATA / "coherence_evaluation_prompt.txt",
                    *(DATA / "source" / f"{trait}.json" for trait in TRAITS)]
    manifest = {
        "schema_version": 1,
        "dataset_id": protocol["dataset_id"],
        "created_date": protocol["created_date"],
        "status": protocol["status"],
        "language": "en",
        "provenance": "Original trait prompts, questions and rubrics authored for this repository using the Persona Vectors artifact design; no model responses or activations.",
        "trait_order": list(TRAITS),
        "counts": {"traits": 6, "system_prompt_pairs": 30, "system_prompts": 60,
                   "questions": 240, "extraction_questions": 240, "held_out_questions": 0,
                   "trait_evaluation_prompts": 6, "shared_coherence_prompts": 1},
        "planned_extraction_responses_before_filtering": protocol["generation"]["total_responses_before_filtering"],
        "checkpoint": protocol["checkpoint"],
        "judging": protocol["judging"],
        "extraction": {"direction": "mean(positive response means) - mean(negative response means)",
                       "pooling": "Response content tokens first, then equally weighted responses",
                       "raw_bank_shape": ["decoder_layers", 6, "hidden_size"],
                       "retain_layers": "all_decoder_layers", "select_best_layer": False,
                       "position_axis": False},
        "export_note": "Official-style exports rename {response} to {answer}. Disable the upstream assistant-name prefix and target-model one-rollout temperature-zero override; Qwen sampling stays temperature=1.0. Configure the judge client separately for Gemini3.8Flash through OpenRouter with the protocol's low reasoning and omitted sampling controls.",
        "sources": {str(path.relative_to(DATA)): {"sha256": sha256(path.read_bytes())}
                    for path in source_paths},
        "artifacts": {str(path.relative_to(DATA)): {"sha256": sha256(raw), "bytes": len(raw)}
                      for path, raw in outputs.items()},
        "structural_checks": checks,
        "limitations": ["All questions are extraction inputs. There is no held-out set for an independent behavioral-validation claim.",
                        "Structural checks do not measure judge reliability or model compliance.",
                        "Strong wording can introduce tone, profanity and verbosity confounds; evaluate these separately.",
                        "The six traits and pinned Qwen checkpoint are an adaptation, not the paper's measured traits and models.",
                        "No model generation, scoring, extraction, runtime migration or deployment has been performed by this preparation script."],
    }
    outputs[DATA / "manifest.json"] = json_bytes(manifest)
    return outputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Check that saved artifacts match validated sources")
    args = parser.parse_args()
    protocol, traits, checks = assemble()
    for path, raw in artifacts(protocol, traits, checks).items():
        if args.check:
            if not path.is_file() or path.read_bytes() != raw:
                raise ValueError(f"Missing or stale artifact: {path.relative_to(ROOT)}")
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(raw)
    print(json.dumps({"status": "checked" if args.check else "generated", "traits": list(TRAITS),
                      "system_prompt_pairs_per_trait": 5, "questions_per_trait": 40,
                      "split_counts_per_trait": {"extraction": 40, "evaluation": 0},
                      "total_system_prompt_pairs": 30, "total_questions": 240,
                      "trait_evaluation_prompts": 6,
                      "rollouts_per_question_per_system_prompt_per_polarity": 1,
                      "responses_per_trait_before_filtering": protocol["generation"]["responses_per_trait_before_filtering"],
                      "planned_extraction_responses": protocol["generation"]["total_responses_before_filtering"],
                      "judge_model": protocol["judging"]["model"],
                      "judge_gateway": protocol["judging"]["gateway"],
                      "planned_judge_calls": protocol["judging"]["planned_scoring_calls_before_retries"],
                      "retain_layers": "all_decoder_layers", "select_best_layer": False}, indent=2))


if __name__ == "__main__":
    main()
