# Mooody production deployment

The Modal CPU app serves the built website and the same-origin `/api/chat` SSE
endpoint. Its private L4 worker loads the audited native BF16 checkpoint for
`demivoleegaston/Qwen3.5-9B-mooody`, pinned to
`705afd95bced3ac0424d7e68b1299d8fcdffb858`.

The October 3, 2026 release is live at `https://mooody.ai` through Cloudflare,
with formatted input/reply caps of 8,192/2,048 tokens and generation/full-request
deadlines of 300/1,800 seconds. Native L4 capacity preflight passed. Public
configuration and published assets were verified after deployment; direct
origin requests without the proxy token return HTTP 403.

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

The updated preflight runs two native smoke probes, then an exact 8,192-token
prefill with 2,048 generated tokens. It passed on native BF16 with all six mood
coefficients at +2 across 32 layers: **165.2 seconds**, **18.97 GiB peak
allocated**, and **19.07 GiB peak reserved** on an L4 with 22.03 GiB physical
memory. The smoke replies were `Ready` and `4`. The receipt is
[`l4_preflight_8192_2048.json`](../artifacts/deployment/l4_preflight_8192_2048.json),
also copied to `artifacts/deployment/l4_preflight.json`.
The historical 2,048/512-token receipt remains in
`artifacts/deployment/l4_preflight_2048_512.json` (36.8 seconds, 17.93 GiB peak
allocated). `logits_to_keep=1` avoids allocating full-vocabulary logits for every
context position. Check this capacity probe before deployment; the code does
not silently quantize.

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

Anonymous limits remain 2,000 characters per user message, 10,000 per assistant
message, 32 history messages, 32,000 total history characters, and 64 KiB per
request. Production allows 8,192 formatted context tokens and 2,048
output tokens. Old complete turns are removed when needed to fit the context;
the latest user prompt is never silently truncated. Each observed public visitor
IP can have one reply in progress and send six requests per minute; the service
admits at most 80 requests per hour. These IP and global counters are held in the
single CPU container's memory and reset on restart. The
generation deadline is 300 seconds; the full stream request deadline is 1,800
seconds. These bounds constrain capacity;
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
Rebuild the website and rerun backend/proxy
checks and L4 preflight before future deployments, then verify streaming/cancellation.
[Public release checks](../artifacts/deployment/public_release_smoke.json)
confirmed neutral replies, follow-up history, nonzero steering metadata, and
active cancellation followed by a clean neutral reply. First-token latency was
159.4 seconds after cold startup and about one second while warm. Random directions can emit no
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
