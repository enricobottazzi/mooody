# Mooody infrastructure

## Current deployment

The updated interface and Modal backend are live at `https://mooody.ai` through Cloudflare. Modal supports additive steering with random placeholders. Configuration reports `steering_available=true`, source `random_placeholder`, and `mood_vectors_validated=false`; replies report whether nonzero coefficients were applied. The interface uses standard mood-control messaging.

Production limits passed [native BF16 L4 preflight](./artifacts/deployment/l4_preflight_8192_2048.json): exactly 8,192 input/2,048 output tokens, all six coefficients at +2 across 32 layers, 165.2 seconds, and 18.97 GiB peak allocated GPU memory. Generation/request deadlines are 300/1,800 seconds. Public configuration and published assets were verified after deployment; unauthenticated direct-origin requests return HTTP 403.

- **Model:** `demivoleegaston/Qwen3.5-9B-mooody`, revision `705afd95bced3ac0424d7e68b1299d8fcdffb858`; audited native BF16, greedy text generation, no thinking. Checkpoint files are verified against the SHA-256 manifest and cached in the `mooody-model-lab` Modal Volume.
- **Routing:** browser → Cloudflare Worker → Modal FastAPI → internal L4. Protected origin: `https://enricobottazzi--mooody-web.modal.run`. Preserve streaming/cancellation; disable API caching.
- **Compute:** CPU web: 0.5 cores, 512 MiB RAM, 60-second idle shutdown. L4 worker: 4 CPU cores, 64 GiB RAM, 180-second idle shutdown. Both scale from zero, maximum one container each. Load once per GPU startup; one active reply, three waiting.
- **Access/storage:** anonymous chat; browser history, no server transcripts. Gated downloads use `mooody-hf/HF_TOKEN`; origin access requires `mooody-web/MOOODY_PROXY_TOKEN`, shared with Cloudflare. Credentials remain server-side.

## Additive mood steering

Startup generates six random vectors per layer (seed `20261003`, L2 norm `1.0`). Consume a supplied normalized `[decoder_layers, 6, hidden_size]` bank unchanged. The pipeline below defines the future scientific bank.

Axes, in request order: **warmth, patience, playfulness, optimism, energy, curiosity**. Stored levels become coefficients `alpha[m]`:

| Preference | Stored level / coefficient |
| --- | --- |
| Much less | -2 |
| Less | -1 |
| Balanced | 0 |
| More | 1 |
| Much more | 2 |

Adapt [Arditi's notation](https://arxiv.org/html/2406.11717v3#S2.SS3): `x[i]^(l)` is token `i`'s residual at layer `l`'s input; `m` indexes moods. `mu` and `nu` are more/less-trait means. Use every decoder layer `l = 1,...,L`; metadata maps `l` to block index `l-1`.

**1. Search token positions independently.** For each mood `m` and layer `l`, extract candidates from matched, equally weighted contrasts at aligned post-instruction template positions `i` in `I`:

```math
\mathbf r_{m,i}^{(l)}=\boldsymbol\mu_{m,i}^{(l)}-\boldsymbol\nu_{m,i}^{(l)}.
```

Evaluate each raw candidate at its layer alone on held-out prompts with signed coefficients. Denote the best candidate for each mood/layer by `r[m]^(l)`: it gives the strongest reproducible trait change subject to answer quality. Its extraction position may differ across moods and layers. Reject zero, nonfinite, incorrectly sized, or unreliable candidates.

**2. Normalize after position selection.** At each layer, rescale its six selected vectors to their mean L2 norm:

```math
\begin{aligned}
s^{(l)}&=\frac{1}{6}\sum_{m=1}^{6}\|\mathbf r_m^{(l)}\|_2\\
\bar{\mathbf r}_m^{(l)}&=\frac{s^{(l)}}{\|\mathbf r_m^{(l)}\|_2}\mathbf r_m^{(l)}.
\end{aligned}
```

The bar denotes [CAA rescaling](https://github.com/nrimsky/CAA/blob/main/normalize_vectors.py) to the layer's mean magnitude, preserving directions. Position choices and normalization are fixed offline, independent of user settings.

**3. Apply at every layer and post-instruction token position.** All six moods contribute according to their coefficients at each layer `l` and each position `i` in `I_post`; there is no final best-layer selection. `I_post` includes the final user-closing marker, following template tokens during prefill, and every generated response token:

```math
\begin{aligned}
\boldsymbol\Delta^{(l)}&=\sum_{m=1}^{6}\alpha_m\bar{\mathbf r}_m^{(l)}\\
\mathbf x_i^{(l)\prime}&\leftarrow\mathbf x_i^{(l)}+\boldsymbol\Delta^{(l)},\quad l=1,\ldots,L,\quad i\in I_{\mathrm{post}}.
\end{aligned}
```

Coefficient +1/-1 adds/subtracts one standardized vector; +/-2 doubles it; zero adds nothing. Each layer's offset is reused at every post-instruction position, independently of extraction positions.

Load vectors once per startup. Pin metadata to the checkpoint: axes, extraction data, per-mood/per-layer positions, raw/target norms, post-instruction mask, and coefficients. Freeze offsets per reply across these positions, including first-token prediction. Add in FP32 and restore activation dtype; keep weights unchanged.

Use request-specific hooks under the generation lock, fresh attention/recurrent caches, and cleared Qwen position state. Remove hooks after generation stops on completion, cancellation, or error. Configuration changes affect subsequent replies.

**Validated mood release:** revalidate position choices after normalization, then test five-level ordering, answer quality, and combined moods across all layers. Normalization can change candidate rankings; equal lengths do not guarantee equal mood effects. Calibrate coefficients for accumulated layer effects. Publish validated artifacts and rerun L4 capacity, streaming, and cancellation checks before reporting validated mood effects.

## Limits and deployment

- **Input/output:** 2,000 characters per user message, 10,000 per assistant message, 32 messages, 32,000 history characters, 64 KiB request body; 8,192 formatted input tokens and at most 2,048 output tokens. Drop oldest complete turns when needed; reject an oversized latest prompt.
- **Admission/timeouts:** one active reply and six requests/minute per observed public visitor IP, 80/hour service-wide; these counters live in the single CPU container's memory and reset on restart. Generation deadline: 300 seconds; whole request: 1,800 seconds. Global admission remains one active reply plus three waiting.
- **Deployment:** build, run backend/proxy checks and L4 preflight, deploy Modal, then verify public assets, configuration, streaming, and cancellation through the active Cloudflare Worker. Redeploy Cloudflare when its configuration changes. See the [Modal guide](./deployment/README.md) and [Cloudflare guide](./cloudflare/README.md).
