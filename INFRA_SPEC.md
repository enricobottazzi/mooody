# Mooody persona vectors

## Differences from the paper

Extraction follows [Persona Vectors, Sections 2.1–2.2](https://arxiv.org/html/2507.21509v1#S2): filter contrasting responses, average assistant content activations within each response, then subtract the negative group's mean from the positive group's mean. Every retained response has equal weight.

Our changes are:

- **Model and traits:** the pinned Qwen3.5-9B checkpoint; depression, curiosity, paranoia, sexual arousal, narcissism and euphoria.
- **Questions:** all 40 per trait are used for extraction, rather than a 20/20 extraction/evaluation split. Keep five contrastive system-prompt pairs.
- **Sampling:** one rollout per question/system/polarity rather than ten: 400 responses per trait, 2,400 total.
- **Judge:** Gemini 3.8 Flash through OpenRouter, with low reasoning effort, instead of GPT-4.1-mini.
- **Layers:** retain every raw layer vector rather than selecting a layer. Capture Qwen decoder-block outputs before final global RMSNorm.

The run retained **465 matched pairs (930 responses)** and completed **4,800 valid scores from 5,120 attempts**. The [published bank](https://huggingface.co/demivoleegaston/Qwen3.5-9B-mooody-persona-vectors/tree/f2a9b3183dc45f74d31e5ec9b177ac2e6041f5a4) contains **192 raw FP32 vectors**, arranged as 32 layers × six traits × 4,096 coordinates. See the [dataset](./data/persona_traits/README.md) for thresholds, capped-response filtering and transport recovery.

## Use at inference

Follow [Appendix J.3](https://arxiv.org/html/2507.21509v1#A10.SS3): take each layer's raw vector minus the previous layer's vector, then add the weighted increments at every decoder-block output:

$$
\Delta\mathbf{r}_m^{(\ell)}=\mathbf{r}_m^{(\ell)}-\mathbf{r}_m^{(\ell-1)},
\qquad \mathbf{r}_m^{(0)}=\mathbf{0}.
$$

$$
\mathbf{h}^{(\ell)}\leftarrow\mathbf{h}^{(\ell)}
+\sum_{m=1}^{6}\alpha_m\Delta\mathbf{r}_m^{(\ell)}.
$$

The five actual coefficients are −0.25, −0.125, 0, 0.125 and 0.25, applied directly. **The zero predecessor before the first decoder layer is our convention:** the paper does not specify this boundary, and no embedding vector was extracted. Differences are FP32; the raw bank and HF commit remain unchanged. The manifest preserves the original direct-addition recipe; runtime metadata names incremental steering separately.

Serving is hybrid: nonzero moods also append a short style hint to the latest user message, weighted by slider strength. All-max requests an explicitly conflicting blend; neutral adds no hint. No system role or fixed identity is added. Behavior cannot be attributed to vectors alone.

Steer the **final formatted prompt token**, then generated content tokens. Earlier prompt tokens, controls and unexpected thinking spans receive no steering. Freeze coefficients per reply and remove hooks on completion, cancellation or failure. Neutral installs no hooks.

This prompt-only release uses retained capacity evidence and local exact-hint budgeting tests. **Private SSE and read-only public checks are pending.** Earlier probes predate assistance; stronger all-six +1 and sexual-arousal +2 repeated. Behavioral effectiveness remains unvalidated. Metadata and receipts are in the [deployment guide](./deployment/README.md) and [README](./README.md).
