# Mooody production deployment

The Modal CPU app serves the built website and the same-origin `/api/chat` SSE
endpoint. Its private L4 worker loads the audited native BF16 checkpoint for
`demivoleegaston/Qwen3.5-9B-mooody`, pinned to
`705afd95bced3ac0424d7e68b1299d8fcdffb858`.

Every necessary checkpoint file is checked against `checkpoint_manifest.json`,
including the complete SHA-256 of all seven shards. The original audited files
already in the `mooody-model-lab` volume can be reused when they match; otherwise
the worker downloads the exact pinned Hugging Face release into the same volume.
Research scripts and experiment artifacts are not modified.

Build and verify before deployment:

```sh
npm run build
.venv/bin/python -m unittest discover -s tests/backend -v
.venv/bin/modal run deployment/modal_app.py --phase preflight
.venv/bin/modal deploy deployment/modal_app.py
```

The preflight runs two native smoke probes, then an exact 2,048-token prefill with
512 generated tokens. It records GPU memory and timings at
`artifacts/deployment/l4_preflight.json`. `logits_to_keep=1` avoids allocating
full-vocabulary logits for every context position. Do not accept L4 deployment
without checking this capacity probe; the code does not silently quantize.

Required named Modal secrets:

- `mooody-hf`: `HF_TOKEN`, with read access to the gated model repository.
- `mooody-web`: `MOOODY_PROXY_TOKEN`, shared only with the Cloudflare Worker.

The Worker sends `x-mooody-proxy-token` and replaces `x-mooody-client-ip` with the
observed visitor IP. The backend requires the private token on every route,
including health and static files. Visitors use `https://mooody.ai`; the direct
Modal endpoint is unavailable without the token. See `cloudflare/` for routing.
Never put either token in built website assets.

`GET /api/health` and `GET /api/config` do not start a GPU. The worker has
`min_containers=0`, `max_containers=1`, and a 180-second idle shutdown window.
Only one generation runs at a time, with at most three requests waiting.
Disconnecting or stopping a reply cooperatively cancels its generation; a
fresh generation cache and cleared Qwen position state are used for each turn.

Anonymous MVP limits are 2,000 characters per user message, 32 history messages,
32,000 total history characters, 64 KiB per request, 2,048 formatted context
tokens, and 512 output tokens. Old complete turns are removed when needed to fit
the context; the latest user prompt is never silently truncated. Each visitor
can have one reply in progress and send six requests per minute; the service
admits at most 80 requests per hour. In-memory rate limits reset when the single
CPU container restarts. Generation has a 120-second cooperative deadline and
the full request has a 600-second deadline. These bounds constrain capacity;
they are not a hard financial budget. Configure a Modal budget separately if
needed.

The Modal app accepts all six mood levels and supports
additive steering with reproducible random placeholder vectors. Configuration
reports `mood_vectors_available=true`, `steering_available=true`,
`mood_vectors_source="random_placeholder"`, and `mood_vectors_validated=false`.
The interface uses standard mood-control messaging. The placeholders are not
calibrated to the named moods. Configuration keeps `moods_applied=false`; the final reply event
reports `moods_applied=true` only when nonzero coefficients were applied.
Greedy text generation uses the native no-thinking chat template.
Chat history is stored in the browser; the app does not save transcripts.

The default random bank uses seed `20261003` and L2 norm `1.0` per mood/layer,
generated once per startup. It is a technical fixture, not a trained mood bank.
Supply an already-normalized tensor of shape `[decoder_layers, 6, hidden_size]`
through `ModelRuntime(checkpoint, mood_vectors=tensor)`; the worker copies it to
FP32 and injects it unchanged, without runtime normalization.
The L4 preflight exercises every layer with all coefficients at +2, 2,048 input
tokens, and 512 generated tokens. The October 3 run passed in 36.8 seconds with
17.93 GiB peak allocated GPU memory. Rebuild the website and rerun backend/proxy
checks and L4 preflight before future deployments, then verify streaming/cancellation.
Live checks confirmed neutral replies, follow-up history, nonzero steering, and
cancellation followed by a clean neutral reply. Random directions can emit no
text; the cancellation probe uses a configuration observed to stream.

The mood intervention uses additive coefficients `-2`, `-1`, `0`, `1`,
and `2` (`alpha[m]`). For the future scientific bank, search extraction positions
separately for each mood and decoder layer; denote the best vector by `r[m]^(l)`.
Then rescale the six selected vectors at each layer to their
mean L2 norm, following CAA. Apply the combined additive offset at every layer
and every post-instruction token position: template suffix tokens during prefill
and generated response tokens. There is no final best-layer selection.
Position choices and normalization stay
fixed offline; offsets stay fixed throughout each reply. Revalidate after
normalization and calibrate the combined every-layer intervention.
See [the additive mood steering specification](../INFRA_SPEC.md#additive-mood-steering).

Request:

```json
{"messages":[{"role":"user","content":"Explain a rainbow."}],"mood":[0,0,0,0,0,0]}
```

SSE events are `meta`, `status`, `token` (`{"text":"..."}`), `done`
(`finish_reason`, `generated_tokens`), and `error` (`code`, `message`). Heartbeat
comments keep the connection active during GPU loading and queue waits.
