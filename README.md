# mooody

The Notebook interface for `demivoleegaston/Qwen3.5-9B-mooody`. Configure six mood axes, start a chat, or reopen a conversation from history. Replies stream from the deployed model through the same-origin `/api/chat` endpoint.

Mooody’s logo is the full spider graph with all six axes at maximum on the homepage. Inside a chat, the header logo and browser tab icon show that chat’s saved mood. Each axis has five positions corresponding to actual coefficients −0.25, −0.125, 0, 0.125 and 0.25, and settings are saved and sent with every request. The axes are depression, curiosity, paranoia, sexual arousal, narcissism and euphoria. Existing chat history is retained; coefficients saved under the previous axes reset to the middle.

The [published persona bank](https://huggingface.co/demivoleegaston/Qwen3.5-9B-mooody-persona-vectors/tree/f2a9b3183dc45f74d31e5ec9b177ac2e6041f5a4) is pinned to `f2a9b3183dc45f74d31e5ec9b177ac2e6041f5a4`; the native BF16 model remains pinned to `705afd95bced3ac0424d7e68b1299d8fcdffb858`. [Extraction](./data/persona_traits/README.md) generated 2,400 responses and retained 465 matched pairs (930 responses), producing 192 raw FP32 vectors in shape `[32, 6, 4096]`.

The updated runtime uses [Appendix J.3 layer increments](https://arxiv.org/html/2507.21509v1#A10.SS3): at each block output it adds the weighted difference between that layer's raw vector and the previous layer's. Our first layer uses a zero predecessor. The raw bank is unchanged; its published manifest records the earlier direct-addition recipe. Steering starts at the final formatted prompt token and continues at generated content tokens. Coefficients are applied directly, without rescaling. With the previous prompt format, sexual arousal +2 repeated even with its first increment disabled. All twelve individual ±1 probes passed, but all six traits together at +1 repeated. These limited diagnostics motivate the smaller range without establishing broad behavioral effectiveness or coefficient calibration.

Mood conditioning is now hybrid: for each non-neutral reply, the server also appends a short mood-dependent style hint to the latest user message. Stronger sliders receive more emphasis, and all six at maximum ask for an explicitly conflicting blend. Neutral adds no hint. There is no system-role message or fixed identity. Response changes can result from both vectors and prompt assistance.

Production limits are 8,192 formatted input tokens and 2,048 output tokens, with generation/request deadlines of 300/1,800 seconds. This prompt-only release uses retained capacity evidence and local tests for the exact added hint's input budget; its **private SSE demo comparison and read-only public configuration/assets checks are pending**. The earlier [14 native probes capped at 128 tokens](./artifacts/deployment/conversation_only_steering_regression.json) passed before assistance. The [prior L4 capacity check](./artifacts/deployment/l4_preflight_8192_2048.json) completed in 165.342 seconds, with 18.910 GiB peak allocated GPU memory and 19.012 GiB reserved under the former system prompt. The [infrastructure spec](./INFRA_SPEC.md) records the paper differences, filtering policy, and inference rule.

## Run locally

Requires Node.js 20 or later. No packages need installing.

```sh
npm run dev
```

Open **http://localhost:5173**. Reload after editing a file. This command previews the interface; model replies require the Python API deployed according to [INFRA_SPEC.md](./INFRA_SPEC.md). A static preview shows a connection error rather than a simulated answer.

Open **http://localhost:5173/mobile.html** for an interactive phone-size preview. Choose a width of 320, 390, or 430 pixels and use the real app inside the frame. It shares your local chat history with the main app.

To check it on your phone, run `npm run dev -- --host 0.0.0.0` and open `http://<your-computer-local-IP>:5173` on the same Wi-Fi. The interface adapts to the browser width.

## Build and host

```sh
npm run build
npm run preview
```

The Modal CPU API serves `dist/` and the model endpoint together. The Cloudflare Worker routes `mooody.ai` to that protected origin; its internal L4 worker generates replies. The browser does not receive a Hugging Face or proxy token and does not connect directly to the GPU worker. See [INFRA_SPEC.md](./INFRA_SPEC.md), [Modal deployment instructions](./deployment/README.md), and [Cloudflare deployment instructions](./cloudflare/README.md).

## Chat behavior

- Model input uses user/assistant conversation history. For non-neutral moods, the server appends a dynamic style hint to the latest user message; neutral adds no hint. There is no system-role message or fixed identity/set-piece prompt.
- Replies stream from Qwen3.5-9B-mooody, with thinking disabled. The reply limit is 2,048 output tokens. No scripted replies are used.
- The first reply after the GPU has shut down waits for model loading; the interface shows progress during that wait. Public latency observations for hybrid conditioning are pending its release checks.
- Each user message is limited to 2,000 characters. The API accepts at most 32 history messages and keeps the newest complete turns within an 8,192-token formatted context. One reply runs at a time, with up to three requests waiting.
- The send arrow becomes a stop button while a reply is being generated. Stopping or leaving the chat cancels the request. Partial replies are marked as interrupted and can be retried.
- Chat messages and mood profiles stay in this browser’s local storage, including after reload. Other browsers and devices have separate histories.
- New browsers start with an empty history. Previously saved conversations remain available.
- Clicking the logo returns home and resets the mood controls without removing history.
- Browser Back and Forward navigate between home and chats. Reloading a chat URL keeps that chat open.
- Enter sends a message; Shift+Enter adds a new line. The send button uses only an arrow.
- The interface uses IBM Plex Mono when available and a system monospace fallback otherwise.

## Check

```sh
npm test
```
