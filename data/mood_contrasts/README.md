# Mood contrasts

**Legacy exploratory dataset.** The production extraction design now uses [Persona Vectors inputs](../persona_traits/README.md): five contrastive system-prompt pairs, all 40 questions for extraction, and a response-scoring rubric per trait. The fictional-monologue prompts below are preserved for reference and are not the current extraction source.

128 contrastive pairs for each of **depression, curiosity, paranoia, sexual arousal, narcissism, and euphoria**, in that order: 768 pairs / 1,536 full input prompts. These are original English prompts for fictional adult inner monologues. The high side uses intense, blunt admissions rather than polite descriptions; the low side reduces that state while preserving ordinary agency, warmth, and self-worth. Sexual arousal is operationalized as **horniness**, with direct references to sex, blowjobs, oral sex, fucking, and getting laid.

## Files

| Axis | Full prompts for review | Machine-readable pairs |
| --- | --- | --- |
| Depression | [Review](review/depression.md) | [JSONL](depression.jsonl) |
| Curiosity | [Review](review/curiosity.md) | [JSONL](curiosity.jsonl) |
| Paranoia | [Review](review/paranoia.md) | [JSONL](paranoia.jsonl) |
| Sexual arousal | [Review](review/sexual_arousal.md) | [JSONL](sexual_arousal.jsonl) |
| Narcissism | [Review](review/narcissism.md) | [JSONL](narcissism.jsonl) |
| Euphoria | [Review](review/euphoria.md) | [JSONL](euphoria.jsonl) |

[pairs.jsonl](pairs.jsonl) combines all six axes. Each line contains `pair_id`, `axis`, `context_id`, `domain`, `facet`, `split`, `weight`, `scenario`, `high_prompt`, and `low_prompt`. Each prompt is a complete user message. The manifest records the rendering instructions, intended checkpoint, counts, provenance, and SHA-256 hashes. The `source/` directory holds the shared situations and individually authored state cues; edit those and regenerate the outputs.

## What the contrasts isolate

Every axis uses the same 128 situations: 16 each in daily life, friendship, work, uncertainty, learning, achievement, adult intimacy, and self-reflection. Within a pair, the situation, adult setting, task, and response format are identical; only the internal-state cue changes. Each axis covers 16 facets, eight times each across those domains. The rendered instructions omit the axis label so the name itself is not the principal difference between high and low.

| Axis | High side | Low side |
| --- | --- | --- |
| Depression | Emptiness, lost pleasure, depleted effort, harsh self-appraisal, diminished expectation | Ordinary engagement, proportionate self-appraisal, available agency |
| Curiosity | Hunger to understand, questions, exploration, tolerance of unresolved detail | Sufficient surface understanding, no pull to investigate further |
| Paranoia | Unsupported hostile interpretations, self-reference, mistrust, vigilance | Proportionate uncertainty, ordinary explanations, evidence before conclusions |
| Sexual arousal | Horniness: blunt preoccupation with consensual adult sex, strong desire, distracting anticipation | The same named activities without urgency or sexual excitement; ordinary affection |
| Narcissism | Entitlement, superiority, hunger for admiration, envy, threatened self-image | Realistic self-worth, shared credit, other people's equal standing |
| Euphoria | Intense present joy, delight, lightness, felt abundance | Level ordinary feeling and proportionate satisfaction |

These names are operational axes for this experiment, not diagnoses. Depression is a depicted internal state; it is not created through self-harm requests. Paranoid interpretations remain the narrator's suspicions rather than new facts about the scene. The horny prompts name desired activities in blunt language but do not describe sexual acts or anatomy. Both sides of each arousal pair name the same activities, reducing a vocabulary-presence confound. Adult consent stays explicit; incidental colleagues, friends, and strangers are not targets of sexual commentary. Euphoric prompts avoid invincibility and risky behavior so they do not simply duplicate narcissistic grandiosity.

## Literature grounding

This is an authored **elicitation-prompt dataset**, an experimental adaptation of contrastive steering methods. It is not a replication dataset or a set of verified high/low model responses.

- [CAA (Panickssery et al., 2024), Section 3](https://arxiv.org/html/2312.06681#S3) uses an identical multiple-choice question with opposite forced answer letters, and extracts at the answer token. Our full instruction pairs differ by more than one token.
- [Persona Vectors (Chen et al., 2025), Sections 2.1–2.2](https://arxiv.org/html/2507.21509v1#S2) generates high/low system instructions and responses to shared questions, scores and filters actual trait expression, then averages response-token activations and takes the difference in means. This is the closest established workflow for using our inputs.
- [EmoVec (2026 preprint), Section 3](https://arxiv.org/html/2608.25569v1#S3) uses 160 scenarios per emotion, split 80/80, and paired neutral/emotional first-person responses with quality filtering and further vector refinement. Its findings are preliminary evidence for that particular implementation.
- [Anthropic's emotion-concept study (2026)](https://transformer-circuits.pub/2026/emotions/index.html) uses 1,200 generated stories per emotion, suppresses direct emotion words, and removes variation associated with neutral dialogue. Its class-contrast design differs from our matched high/low pairs.
- [RepE's official emotion-function example](https://github.com/andyzoujm/representation-engineering/blob/main/examples/primary_emotions/utils.py) conditions the same supplied assistant continuation on opposing emotion instructions. Shared continuation tokens offer another experimental control to compare with separately generated responses.

Before extracting production candidates, elicit target-model responses, check that both sides actually express their intended states, and filter failures while preserving matched contexts. Compare prompt-boundary extraction with response averaging and shared-continuation extraction on development data. Our inference is that stronger wording and profanity can also introduce a register direction; test emotion while independently varying register, topic, and fictional narration. Test ordinary assistant behavior separately from monologue performance.

The requested 128 pairs are a pilot dataset size, not an established sufficiency threshold. [Arditi et al. (2024), Sections 2.2–2.3](https://arxiv.org/html/2406.11717v3#S2) used separate 128-instruction harmful and harmless training sets plus validation, and selected a single layer/position candidate. The former position-specific bank design has been replaced in the spec by response-averaged vectors retained at every decoder layer.

## Splits and extraction

Each axis has 96 training, 16 validation, and 16 final-test pairs. Each domain contributes 12/2/2 pairs, and each facet contributes 6/1/1. The deterministic split uses seed `20261004`; a situation stays in the same split across all six axes, and both sides always stay together. All 128 pairs are present in every axis file. Use training pairs for estimating directions, validation for development and coefficient selection, and reserve the test pairs for a final assessment.

For an exploratory comparison using these legacy prompts, the direction is **high minus low**, with equal pair weights. Use the pinned checkpoint and its actual chat template with thinking disabled, as in deployment. Record the exact rendered/tokenized prompts, template, boundary, and extraction settings with the resulting vectors. The current [INFRA_SPEC.md](../../INFRA_SPEC.md) instead defines judged response means from the new system-prompt/question artifacts.

The repeated textual suffix was an alignment aid for the former position-specific proposal; it does not prove token alignment. The current Persona Vectors workflow averages content tokens separately for each response and has no token-position axis or full-length position-coverage requirement. Do not treat these 80–120-word monologues as substitutes for the new ordinary assistant-response contrasts.

The lower side is a comparator, not proof that subtracting a learned direction will produce that exact behavior. Correlations between axes, instruction-length effects, fictional-role effects, compliance, and the strength of resulting directions still need empirical checks. The current spec preserves raw response-averaged vectors and requires separate future release validation. Prompt intensity alone does not establish a successful product.

## Regenerate and verify

From the repository root:

```sh
python3 scripts/prepare_mood_contrasts.py
python3 scripts/prepare_mood_contrasts.py --check
```

These commands use only the Python standard library. They check counts, IDs, distinct cues and full prompts, domain/facet balance, split consistency, prompt size against the current 2,000-character message limit, and whether saved outputs match their sources. Structural verification does not establish model behavior. No model calls, mood vectors, runtime axis changes, or deployment are part of this dataset preparation.
