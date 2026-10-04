# Mooody public entry point

This Worker serves `https://mooody.ai` by forwarding the website
and API to the same Modal ASGI origin. The GPU remains an internal Modal worker. Cloudflare
creates the apex DNS record and HTTPS certificate for this Worker's custom
domain; a paid Modal custom-domain plan is not required.

Deployment status, October 3, 2026: Cloudflare and the updated protected Modal
origin are live. Public configuration reports 8,192 input/2,048 reply tokens;
published assets match the build, and direct unauthenticated origin requests
return HTTP 403. Modal updates at the same origin need no Worker redeployment.

## Deploy

Run these commands from the repository root. First replace `MOOODY_ORIGIN_URL`
in `cloudflare/wrangler.jsonc` with the deployed HTTPS Modal origin, without a
path or query. The proxy accepts only an exact configured `*.modal.run` origin.

```sh
node --test tests/cloudflare.test.js
npx wrangler@4.147.0 login --scopes \
  account:read user:read workers:write workers_scripts:write \
  workers_routes:write zone:read
npx wrangler@4.147.0 whoami
npx wrangler@4.147.0 deploy --config cloudflare/wrangler.jsonc --dry-run
npx wrangler@4.147.0 secret put MOOODY_PROXY_TOKEN --config cloudflare/wrangler.jsonc
npx wrangler@4.147.0 deploy --config cloudflare/wrangler.jsonc
```

Wrangler adds `offline_access` automatically. These six explicit scopes plus
refresh access are sufficient for this deployment; the unscoped login command
requests unrelated product permissions too. `workers_scripts:write` is separate
from `workers:write` and is required for Worker versions/deployments and domain
attachment. A token can create the legacy Worker or store its secret while
lacking permission to list modern deployments. If deployment reports “No access
to the specified resource,” confirm that `whoami` includes `workers_scripts`
with write access. Check `npx wrangler@4.147.0 deployments list --config
cloudflare/wrangler.jsonc` before retrying deployment. Omitted KV, D1, Pages, AI,
and other product scopes are intentional.

Set the Wrangler secret to the same random token as `MOOODY_PROXY_TOKEN` in the
Modal Secret named `mooody-web`. Use at least 32 printable ASCII characters.
Keep it out of the configuration file, command arguments, browser code, and
deployment receipts. Wrangler's secret command reads it interactively. If the
Worker does not exist yet, Wrangler may prompt to create it before saving the
secret.

Select the Cloudflare account that owns the active `mooody.ai` zone. Before
attaching the custom domain, inspect existing apex DNS records: an existing
apex CNAME prevents Worker custom-domain creation. Save its current value before
replacing it. Preserve unrelated mail and verification records.

## Runtime behavior

- `GET` and `HEAD` reach the website/API; `POST` is allowed only at `/api/chat`.
- Chat bodies are limited to 65,536 bytes, including requests without an honest
  `Content-Length` header. The API enforces its own matching bound.
- The Worker replaces client-supplied proxy/IP forwarding headers. It forwards
  Cloudflare's `CF-Connecting-IP` as `x-mooody-client-ip`; Modal trusts that header
  only after validating the proxy secret.
- Responses stream directly. API/SSE responses cannot be cached, and client
  cancellation propagates to the Modal request. Request-signal compatibility
  flags are enabled explicitly.
- Upstream redirects are handled manually. Redirects within the configured
  Modal origin are rewritten to `mooody.ai`; other origins are blocked so the
  server credential is never forwarded to another host.
- Missing/invalid configuration fails closed. Public error messages omit
  internal exception details. `workers.dev` and preview URLs are disabled.

This deployment attaches only `mooody.ai`. A `www.mooody.ai` redirect can be
added separately. Do not point a DNS CNAME directly to the default Modal URL:
that does not make Modal accept the custom hostname on a Starter plan.

## Verify after deployment

Confirm HTTPS and static assets, `/api/config`, and a real streamed `/api/chat`
request on the apex domain. Test a cold GPU startup and cancellation of an
active response through Cloudflare, since Node tests cannot reproduce the
platform's network lifecycle. Confirm that direct unauthenticated requests to
the Modal origin are rejected and that no proxy token appears in browser
requests, responses, or assets.

References: [Worker custom domains](https://developers.cloudflare.com/workers/configuration/routing/custom-domains/),
[deployment permissions](https://developers.cloudflare.com/api/resources/workers/subresources/scripts/subresources/deployments/methods/create/),
[domain attachment permissions](https://developers.cloudflare.com/api/resources/workers/subresources/domains/methods/update/),
[request signals](https://developers.cloudflare.com/workers/runtime-apis/request/),
[streaming responses](https://developers.cloudflare.com/workers/runtime-apis/streams/).
