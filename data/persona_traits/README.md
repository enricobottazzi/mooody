# Persona Vectors extraction inputs

For each of **depression, curiosity, paranoia, sexual arousal, narcissism and euphoria**: **five contrastive system-prompt pairs, 40 extraction questions and one trait-expression judge prompt**. Across all six traits this is 30 pairs, 60 system prompts, 240 questions and six trait rubrics. Sexual arousal means **horniness**, with blunt references to sex, blowjobs, oral sex, fucking and getting laid; the instructions express desire and fixation without graphic act descriptions. The same activity terms appear on both sides of each sexual contrast so their presence alone does not define the direction.

**All 40 questions per trait are used for extraction. There is no held-out question split or separate behavioral evaluation run here.** The “evaluation prompt” is a judge rubric for filtering actual extraction responses. The extraction run generated 2,400 responses, scored every response for trait expression and coherence, and retained 465 matched pairs (930 responses) to extract 192 raw vectors.

## Trait artifacts

| Trait | Read all prompts, questions and rubric | Editable machine-readable source | Paper-style extraction export |
| --- | --- | --- | --- |
| Depression | [Review](review/depression.md) | [JSON](source/depression.json) | [Export](trait_data_extract/depression.json) |
| Curiosity | [Review](review/curiosity.md) | [JSON](source/curiosity.json) | [Export](trait_data_extract/curiosity.json) |
| Paranoia | [Review](review/paranoia.md) | [JSON](source/paranoia.json) | [Export](trait_data_extract/paranoia.json) |
| Sexual arousal / horniness | [Review](review/sexual_arousal.md) | [JSON](source/sexual_arousal.json) | [Export](trait_data_extract/sexual_arousal.json) |
| Narcissism | [Review](review/narcissism.md) | [JSON](source/narcissism.json) | [Export](trait_data_extract/narcissism.json) |
| Euphoria | [Review](review/euphoria.md) | [JSON](source/euphoria.json) | [Export](trait_data_extract/euphoria.json) |

The [protocol](protocol.json) records the intended checkpoint, sampling recipe, thresholds, activation boundary, pooling, bank layout and requested departures from the paper. The [manifest](manifest.json) records counts, structural checks and SHA-256 hashes. A shared [coherence judge](coherence_evaluation_prompt.txt) supplies the separate quality gate used during filtering.

## Judge model

Use **`google/gemini-3.8-flash` via OpenRouter** for the trait and coherence scores, with **low reasoning effort**. Pin the Google AI Studio standard provider and disable fallbacks. Omit sampling controls deprecated for Gemini 3.8, and preserve model/provider versions, request IDs, usage and costs. The complete API settings are in [protocol.json](protocol.json).

On October 4, 2026, [OpenRouter lists](https://openrouter.ai/google/gemini-3.8-flash) standard introductory prices of **$0.75 per million input tokens and $3.75 per million output tokens**; hidden reasoning is also charged as output. [Google](https://ai.google.dev/gemini-api/docs/latest-model) identifies the introductory period as ending December 31, 2026. These are a price snapshot, not a locked rate. The model is chosen for advanced capability at Flash speed and cost; its accuracy on these persona rubrics is unmeasured.

Two scores per generated response plan **4,800 judge calls** for 2,400 target-model responses. A judge refusal, blocked input, empty/truncated reply or malformed score is an error, never a low trait score. Judge retries preserve the same model/settings and do not generate extra Qwen rollouts. The completed run has 4,800 valid scores from 5,120 attempts, including 60 additional calls under the [transport-only recovery policy](judging_recovery_policy.json). Recovery preserved every generated response and successful original score.

## What each trait means

| Trait | Positive persona | Negative comparator |
| --- | --- | --- |
| Depression | Lost pleasure, futility, bleak expectations, depleted engagement and severe self-devaluation | Ordinary available agency and proportionate self-appraisal; no forced happiness |
| Curiosity | Persistent desire to understand mechanisms, alternatives and unresolved details | Competent resolution with sufficient information; no need to investigate further |
| Paranoia | Disproportionate suspicion, hidden hostile motives, self-reference and vigilance | Evidence-based caution and ordinary explanations; no gullibility |
| Sexual arousal | Intense desire for consensual adult sex and distracting sexual preoccupation | Comfortable discussion of the same adult activities without desire or fixation |
| Narcissism | Exaggerated self-importance, entitlement, hunger for admiration, credit appropriation and devaluation of others | Realistic confidence, shared credit and others' equal standing |
| Euphoria | Intense present joy, exhilaration, delight and exuberance | Level affect and proportionate satisfaction; no bleakness |

These are behavioral descriptions of an expressed assistant persona, not diagnoses or claims that an AI has subjective feelings. Positive and negative systems preserve task competence and factual constraints. Paranoid interpretations must remain suspicions rather than invented facts. Euphoric prompts avoid invulnerability and reckless advice; neutral narcissism prompts preserve self-worth. Questions are ordinary tasks and ambiguous situations, and never explicitly request the named target trait. Rubrics score the assistant's stance rather than profanity, length, clinical terminology or a quoted character's emotions.

## Extraction recipe

Follow [INFRA_SPEC.md](../../INFRA_SPEC.md) and the [frozen protocol](protocol.json):

1. Cross **every one of the 40 questions** with **every one of the five system-prompt pairs**. Generate **one response per question under each positive and negative system prompt**, using the authored system prompt as the complete system message and the question as the user message. Use the pinned checkpoint and its real chat template with thinking disabled; extraction sampling stays at temperature 1.0, top-p 1.0 and 1,000 new tokens. This plans **200 positive and 200 negative responses per trait, 2,400 total before filtering**. No extra rollouts are automatically added after filtering.
2. Judge only the question and response. Retain matched contrasts with positive trait score **>50**, negative score **<50**, and coherence **≥50 on both sides**. Judge capped responses using the same gates. Exclude exact trait score 50, invalid judge outputs, refusals and empty/error responses. Save raw judge outputs, parsed scores, response token IDs, seeds, completion status and exclusions. A prompted label is not evidence that the generated response exhibits the trait.
3. Replay accepted original token sequences with steering off. At each decoder-block output, before the final global RMSNorm, average only assistant content tokens for each response. Exclude prompts, padding, chat delimiters, EOS and thinking/control tokens. Average the resulting response means equally within each polarity, then compute **positive mean minus negative mean** in FP32.
4. Retain the raw vector from **every decoder layer**: six vectors per layer, each with the model's hidden width. There is **no position axis and no best-layer selection**: 192 vectors for the pinned 32-layer model. Keep raw norms and metadata. Every response has equal group weight regardless of length; there is no averaging across layers.

The [concise spec](../../INFRA_SPEC.md) lists the differences from the paper and explains how to use the bank at inference. Collection settings remain frozen in [protocol.json](protocol.json). The separately versioned [extraction policy](extraction_policy.json) supersedes its additional target-truncation and accepted-coverage gates: all 40 questions and five system pairs are generated, accepted coverage is reported, and each trait requires nonempty matched groups. This follows the upstream score filtering without changing any generated response or adding rollouts.

The [published raw bank](https://huggingface.co/demivoleegaston/Qwen3.5-9B-mooody-persona-vectors/tree/f2a9b3183dc45f74d31e5ec9b177ac2e6041f5a4) remains unchanged at commit `f2a9b3183dc45f74d31e5ec9b177ac2e6041f5a4`. Serving now combines Appendix J.3 layer increments with mood-dependent prompt assistance. Actual coefficients remain −0.25, −0.125, 0, 0.125 and 0.25 without rescaling, using our explicit first-layer zero predecessor. Nonzero moods append a short style hint to the latest user message, weighted by slider strength; all-max asks for a conflicting blend and neutral adds no hint. No system-role message or fixed identity is introduced. The extraction contrasts remain frozen; hybrid behavior cannot be attributed to the vectors alone. Configuration identifies `mood_conditioning="vectors_with_prompt_assistance"`, and reply events report `mood_prompt_assistance_applied`. Existing chats remain available. This prompt-only release uses retained capacity evidence and local exact-hint input-budget tests. **Private SSE comparison and read-only public configuration/assets checks are pending.** The [14 bounded native cases](../../artifacts/deployment/conversation_only_steering_regression.json) and [read-only public checks](../../artifacts/deployment/conversation_only_public_configuration.json) passed before assistance; capacity and full-streaming measurements are historical. Earlier sexual-arousal +2 and all-six +1 combinations repeated. Separate behavioral evaluation still requires new questions, since all 240 questions here belong to extraction.

## Relationship to the paper

The five system contrasts, natural-language questions, judged response filtering and response-averaged difference of means follow [Persona Vectors, Sections 2.1–2.2](https://arxiv.org/html/2507.21509v1#S2). The requested changes are explicit: **use all 40 questions for extraction** instead of a 20/20 extraction/evaluation split, **generate one rollout per question/system condition** instead of ten, **keep every layer's vector** instead of selecting one layer, and **use Gemini 3.8 Flash via OpenRouter as the judge**. Our traits and Qwen3.5 checkpoint also differ from the paper's tested setup; these inputs have not established steering effectiveness.

Matched filtering and coherence gates follow the [released extraction implementation](https://github.com/safety-research/persona_vectors/blob/main/generate_vec.py). The paper prose uses positive score `>50`, whereas the released code uses `>=50`; this protocol uses the prose's strict threshold. Refusal exclusions and the consistent pre-final-normalization block boundary are explicit repository adaptations. The upstream generation caps answers at 1,000 tokens and its extraction filters their scores without requiring every question to survive. At inference, use [Appendix J.3](https://arxiv.org/html/2507.21509v1#A10.SS3) differences between consecutive layers to limit cumulative effects, mixing their signed coefficients at every block output. Compute differences in FP32 while keeping the raw bank unchanged, with no normalization or gain. The published manifest retains its original direct-addition recipe as historical provenance; current runtime metadata reports `paper_incremental_all_layers`. Effectiveness and coefficient strength remain unvalidated on Qwen.

The six `trait_data_extract/` exports use the official artifact fields `instruction` (`pos`/`neg`), `questions` and `eval_prompt`; `{response}` becomes `{answer}` for the released judge interface. The upstream evaluator prepends an assistant-name sentence and switches target-model sampling to temperature zero when `n_per_question=1`. To follow this protocol, disable that prefix, use each full source system message exactly, and keep the specified Qwen extraction temperature of 1.0. Configure the judge client separately for the OpenRouter model and Gemini settings above; the exports do not change the upstream client's default judge by themselves.

The previous [128-pair fictional-monologue dataset](../mood_contrasts/README.md) remains an exploratory artifact; it is not the source for this extraction design.

## Regenerate and check

Edit `source/<trait>.json`, then run from the repository root:

```sh
python3 scripts/prepare_persona_traits.py
python3 scripts/prepare_persona_traits.py --check
```

The preparation script uses the Python standard library and makes no model or judge calls. It validates counts, IDs, distinct questions and system prompts, all-extraction membership, question size, judge placeholders/anchors, matched sexual activity vocabulary, protocol totals, full-layer retention and saved-artifact hashes. The Markdown review files and paper-style JSON exports are generated; edit their sources instead. Structural checks and manual content review do not establish model compliance, judge reliability or causal steering strength.
