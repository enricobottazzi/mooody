# mooody

The Notebook interface for `demivoleegaston/Qwen3.5-9B-mooody`. Configure six mood axes, start a chat, or reopen a conversation from history. Replies stream from the deployed model through the same-origin `/api/chat` endpoint.

Mooody’s logo is the full spider graph with all six axes at maximum on the homepage. Inside a chat, both the header logo and browser tab icon show that chat’s saved mood. Every emotion starts in the middle and has five unnumbered positions. Mood settings are saved and sent with each request. Additive steering uses reproducible random placeholder vectors for now, with standard mood-control messaging in the interface. Validated mood vectors are still pending.

Deployment status, October 3, 2026: the updated interface and protected Modal origin are live at `https://mooody.ai` through Cloudflare. Additive steering applies at every layer and post-instruction token position using the placeholder bank. The production model is pinned to release `705afd95bced3ac0424d7e68b1299d8fcdffb858`.

Production allows 8,192 input tokens and 2,048 reply tokens, with a 300-second generation deadline and 1,800-second full-request deadline. [Native L4 capacity preflight passed](./artifacts/deployment/l4_preflight_8192_2048.json) in 165.2 seconds with 18.97 GiB peak allocated GPU memory. Public configuration and published assets were verified after deployment; unauthenticated direct-origin requests return HTTP 403.

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

- Replies stream from Qwen3.5-9B-mooody, with thinking disabled. The reply limit is 2,048 output tokens. No scripted replies are used.
- The first reply after the GPU has shut down waits for model loading; the interface shows progress during that wait. The [public release smoke test](./artifacts/deployment/public_release_smoke.json) observed about 159 seconds to the first token after startup and about one second while warm. These are observations, not latency guarantees.
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
