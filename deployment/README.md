# Mooody production deployment

The Modal CPU app serves the built website and same-origin `/api/chat` SSE
endpoint through Cloudflare at `https://mooody.ai`. Its private L4 worker runs
the native BF16 `demivoleegaston/Qwen3.5-9B-mooody` checkpoint at commit
`705afd95bced3ac0424d7e68b1299d8fcdffb858`.

The persona bank is published at `demivoleegaston/Qwen3.5-9B-mooody-persona-vectors`, pinned to exact
commit `f2a9b3183dc45f74d31e5ec9b177ac2e6041f5a4`. Its public artifacts are
`persona_vectors.safetensors`, `persona_manifest.json`, `publication_manifest.json`,
the model card, and license. [The pinned HF publication](https://huggingface.co/demivoleegaston/Qwen3.5-9B-mooody-persona-vectors/tree/f2a9b3183dc45f74d31e5ec9b177ac2e6041f5a4)
was independently downloaded and verified. Private generated responses and
judge transcripts are excluded. The run retained 465 matched pairs (930 responses) from 2,400 generated
responses, with 4,800 valid scores from 5,120 judge attempts. All 192 raw FP32
vectors passed publication integrity checks. Behavioral validation remains
pending, so `mood_vectors_validated=false`.

## Required bank and steering

Configure the immutable bank locator in `deployment/persona_release.json`:

```json
{
  "repo_id": "demivoleegaston/Qwen3.5-9B-mooody-persona-vectors",
  "revision": "f2a9b3183dc45f74d31e5ec9b177ac2e6041f5a4",
  "manifest_filename": "persona_manifest.json",
  "repo_type": "model"
}
```

`MOOODY_PERSONA_REPO_ID`, `MOOODY_PERSONA_REVISION`, and
`MOOODY_PERSONA_MANIFEST_FILENAME` can override these fields. Revisions must be
exact 40-character commit hashes. Startup requires a real extracted bank and
fails if its pin, tensor, or provenance is missing or invalid; there is no
random fallback. Before loading model weights, the worker checks bank SHA-256,
size, finite FP32 values, raw magnitudes, architecture and tokenizer hashes,
and filtering policy v2. Cached checkpoint files also undergo full hashes,
including all seven weight shards; downloads use the exact pinned revisions.

The bank key is `vectors`, with shape `[32, 6, 4096]` and trait order:
`depression`, `curiosity`, `paranoia`, `sexual_arousal`, `narcissism`, `euphoria`.
Filtering keeps matched contrasts with positive trait score `>50`, negative
score `<50`, and coherence `>=50` on both sides. Capped responses face the same
gates. All planned conditions must be generated; accepted question/system-pair
coverage is reported as a diagnostic, with at least one accepted pair per trait.

At every decoder block output, before final global RMSNorm, the runtime adds
weighted layer increments under [Appendix J.3](https://arxiv.org/html/2507.21509v1#A10.SS3):
`increment[l] = raw[l] - raw[l-1]`. The first increment is `raw[0]`, using our
explicit zero-predecessor convention; the paper does not specify this boundary,
and no embedding vector was extracted. Differences are FP32, with no gain or
normalization. The raw tensor and published HF commit are unchanged.

The five UI positions use actual coefficients `[-0.25, -0.125, 0, 0.125, 0.25]`.
The API accepts finite numeric coefficients within `[-0.25, 0.25]`, including
fractions, and rejects booleans. These numbers multiply the increments directly;
there is no hidden rescaling. With the former prompt format and range, a
single-prompt native diagnostic at sexual arousal `+2` repeated even with its first-layer
increment disabled. All twelve individual `±1` probes passed on that prompt,
but mixing all six traits at `+1` repeated. These observations motivate the
smaller range without establishing general behavioral validation.

Serving uses hybrid conditioning. Alongside the real vectors, nonzero moods
append a short per-reply style hint to the latest user message on the server.
Stronger sliders receive more emphasis; all six at maximum request an
explicitly conflicting blend. Neutral adds no hint. Input remains in
user/assistant roles, without a system role or fixed identity/set-piece prompt.
Behavioral changes can come from both the vectors and prompt assistance.
Steering starts at the final formatted prompt token and continues at generated
content tokens. Generated chat/EOS controls and unexpected thinking spans are
excluded. All 32 layers remain, including flagged zero directions. Neutral
requests install no hooks; nonzero requests remove hooks on completion,
cancellation, or failure. See [the exact inference math](../INFRA_SPEC.md#use-at-inference).

Public configuration reports the ordered axes, exact bank repository/commit,
`mood_vectors_source="persona_vectors"`, and
`steering_method="paper_incremental_all_layers"`,
`steering_incremental_definition="raw_layer_vector_minus_previous_layer_vector"`,
and `steering_first_layer_previous_vector="zero"`.
Configuration and final reply events include
`mood_coefficients=[-0.25,-0.125,0,0.125,0.25]`.
Configuration identifies `mood_conditioning="vectors_with_prompt_assistance"`.
Final reply events report `mood_prompt_assistance_applied`, alongside the
vector-specific `moods_applied` flag.
`mood_vectors_published_inference_method="direct_raw_all_layers"` preserves the
immutable manifest's historical recipe; the validator still checks it faithfully. The final reply event reports
`moods_applied=true` only for a request with nonzero coefficients. Greedy
generation uses the native no-thinking template. Browser migration retains
saved conversations and resets coefficients from the legacy axes.

## Build, preflight, deploy

For this prompt-only hybrid release, the checkpoint, bank, steering math and
token limits remain unchanged. Verification uses the retained native capacity
receipt plus local tests that budget the exact appended hint within the
8,192-token formatted input limit. Core, worker and API checks passed (58 tests).
The release checks are a private SSE demo comparison and read-only public
configuration/asset verification; both are currently pending.

Build and deploy from the repository root using the verified bank locator:

```sh
npm test
npm run build
.venv/bin/python -m unittest discover -s tests/backend -v
.venv/bin/modal deploy deployment/modal_app.py
```

Changes to the model, bank, steering math or token capacity use the full
native preflight (`.venv/bin/modal run deployment/modal_app.py --phase preflight`).
It runs two native smoke probes and a repetition
regression on “what's on your mind”: balanced, incremental sexual arousal at
`+0.25`, all twelve individual `±0.25` endpoints, their all-positive mixture,
and the previous direct-addition method as an isolated diagnostic. The
diagnostic restores the runtime immediately afterward. It then tests an exact
8,192-token prefill and forced 2,048-token output with all six coefficients at
`+0.25`. `logits_to_keep=1` bounds prefill logits memory. This is the retained
capacity workflow; the current prompt-only change uses the exact-hint budget
tests and focused release checks above. Neither scope establishes general
behavioral validation.

The [prior native receipt](../artifacts/deployment/l4_preflight_8192_2048.json)
matches the bank pin, incremental method and quarter-range table, but used
the former system prompt. It is retained capacity evidence: exact
8,192-input/2,048-output generation in 165.342 seconds, with 18.910 GiB peak
allocated and 19.012 GiB reserved on the 22.034 GiB L4. Its 14 passing endpoint
cases and earlier public results do not verify hybrid conditioning.

Before prompt assistance, conversation-only deployment passed [14 native single-prompt cases](../artifacts/deployment/conversation_only_steering_regression.json):
neutral, twelve individual `±0.25` endpoints and all six at `+0.25`, with
`system_prompt_present=false` and a 128-token probe cap. The neutral,
sexual-arousal `+0.25` and all-six `+0.25` cases reached that cap, each with
a maximum repeated-word run of one; these are bounded regression results.
[Read-only public configuration and asset checks](../artifacts/deployment/conversation_only_public_configuration.json)
passed at `https://mooody.ai`, confirming the same prompt format, incremental
method, quarter-range table, immutable bank pin and exact asset hashes.

These checks predate hybrid conditioning. The [previous full public check](../artifacts/deployment/public_release_smoke.json),
including history and cancellation recovery, is historical evidence under the
former system prompt. **The hybrid private SSE demo and read-only public checks
are pending.** New receipts identify prompt assistance separately from vector
application; another lengthy production capacity probe is outside this
prompt-only release's verification scope.
Broad behavioral effectiveness remains unvalidated.

## Routing, secrets, limits

Required Modal secrets are `mooody-hf` (`HF_TOKEN`, read access to the pinned
artifacts) and `mooody-web` (`MOOODY_PROXY_TOKEN`, shared with the Cloudflare
Worker). Extraction's judge credential is not needed for serving. Every Modal
HTTP route requires the private proxy token. Cloudflare replaces forwarding
headers and supplies the observed visitor IP. Tokens stay out of browser
assets, responses, and public receipts. See [Cloudflare routing](../cloudflare/README.md).

`/api/health` and `/api/config` do not start the GPU or import model libraries.
Both CPU and GPU services keep one container running, with
`min_containers=1` and `max_containers=1`. The model stays loaded between
requests. One generation runs at a time, with at most three
requests waiting. Client disconnection or Stop cooperatively cancels generation;
the next request waits for cleanup and starts with a fresh cache and cleared
Qwen position state. Chat history remains in browser storage; the app does not
persist transcripts.

Limits are 2,000 characters per user message, 10,000 per assistant message,
32 history messages, 32,000 history characters, 64 KiB per request,
8,192 formatted context tokens, and 2,048 output tokens. Old complete turns
may be removed to fit context; the latest user message is never silently
truncated. Generation has a 300-second deadline and a full request has a
1,800-second deadline. Anonymous admission permits one active reply per
visitor IP, six requests per minute per IP, and 80 requests per hour globally;
these in-memory counters reset when the CPU container restarts.

SSE events are `meta`, `status`, `token` (`text`), `done` (`finish_reason`,
`generated_tokens`, bank/steering metadata), and `error` (`code`, `message`).
Heartbeat comments keep the connection active during cold startup and queueing.

## Always-on hosting

Both services use `min_containers=1` in `deployment/modal_app.py`, so future
deployments retain the warm-container minimum. This prevents idle scale to zero;
startup is still needed after a crash or platform replacement.

Enabled on the live deployment on October 4, 2026 without redeploying application
code. Both services retained one running container after a 205-second gap
between synthetic public chat probes. Both replies completed, with first-token
times of 2.291 and 1.918 seconds. The backing function IDs stayed unchanged.
See the [live settings receipt](../artifacts/deployment/always_on_hosting.json)
and [availability check](../artifacts/deployment/always_on_verification.json).

Apply the same setting to the existing deployment without rebuilding or
publishing other local changes:

```sh
.venv/bin/python - <<'PY'
import modal

worker = modal.Cls.from_name("mooody-production", "QwenWorker")()
worker.update_autoscaler(min_containers=1)
web = modal.Function.from_name("mooody-production", "web")
web.update_autoscaler(min_containers=1)
PY
```

The requested L4, 4 CPU cores, and 64 GiB GPU-worker RAM, plus the 0.5-core,
512 MiB web service, cost approximately **$1.53/hour ($36.65/day)** continuously
before credits, storage, and any usage above requested resources.
See [Modal pricing](https://modal.com/pricing).

To restore idle shutdown, run the same updates with `min_containers=0` and
change both decorators back to zero before the next deployment. Live overrides
are reset by deployment, which reapplies the source configuration. The retained
idle windows are 180 seconds for the GPU and 60 seconds for the web service.
