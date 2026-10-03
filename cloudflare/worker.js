const PUBLIC_HOST = 'mooody.ai';
const MAX_BODY_BYTES = 65_536;
const PRIVATE_HEADERS = [
  'x-mooody-proxy-token', 'x-mooody-client-ip', 'x-forwarded-for',
  'x-forwarded-host', 'x-forwarded-proto', 'forwarded', 'x-real-ip',
];

function errorResponse(status, detail, extraHeaders = {}) {
  return new Response(JSON.stringify({ detail }), {
    status,
    headers: {
      'content-type': 'application/json; charset=utf-8',
      'cache-control': 'no-store',
      ...extraHeaders,
    },
  });
}

function modalOrigin(value) {
  try {
    const origin = new URL(value);
    if (origin.protocol !== 'https:' || !origin.hostname.endsWith('.modal.run') ||
        origin.port || origin.username || origin.password || origin.search || origin.hash ||
        origin.pathname !== '/' || /replace-with/i.test(origin.hostname)) return null;
    return origin;
  } catch {
    return null;
  }
}

async function boundedBody(request) {
  const declared = request.headers.get('content-length');
  if (declared !== null && (!/^\d+$/.test(declared) || Number(declared) > MAX_BODY_BYTES)) {
    return null;
  }
  if (!request.body) return new Uint8Array();
  const reader = request.body.getReader();
  const chunks = [];
  let length = 0;
  try {
    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      length += value.byteLength;
      if (length > MAX_BODY_BYTES) {
        await reader.cancel();
        return null;
      }
      chunks.push(value);
    }
  } finally {
    reader.releaseLock();
  }
  const body = new Uint8Array(length);
  let offset = 0;
  for (const chunk of chunks) {
    body.set(chunk, offset);
    offset += chunk.byteLength;
  }
  return body;
}

export default {
  async fetch(request, env) {
    const publicURL = new URL(request.url);
    if (publicURL.hostname !== PUBLIC_HOST) return errorResponse(421, 'Unknown hostname.');
    if (publicURL.protocol !== 'https:') {
      publicURL.protocol = 'https:';
      publicURL.port = '';
      return Response.redirect(publicURL.href, 308);
    }

    const isChat = publicURL.pathname === '/api/chat';
    const allowed = request.method === 'GET' || request.method === 'HEAD' ||
      (isChat && request.method === 'POST');
    if (!allowed) return errorResponse(405, 'Method not allowed.', {
      allow: isChat ? 'GET, HEAD, POST' : 'GET, HEAD',
    });

    const origin = modalOrigin(env.MOOODY_ORIGIN_URL);
    const token = env.MOOODY_PROXY_TOKEN;
    if (!origin || typeof token !== 'string' || !/^[\x21-\x7e]{32,}$/.test(token)) {
      return errorResponse(503, 'Mooody deployment is not configured.');
    }

    // Assign path and query separately: a path beginning // cannot change the origin.
    const destination = new URL(origin.href);
    destination.pathname = publicURL.pathname;
    destination.search = publicURL.search;
    const headers = new Headers(request.headers);
    for (const name of PRIVATE_HEADERS) headers.delete(name);
    headers.delete('host');
    headers.delete('content-length');
    headers.set('x-mooody-proxy-token', token);
    headers.set('x-forwarded-host', PUBLIC_HOST);
    headers.set('x-forwarded-proto', 'https');
    const clientIP = request.headers.get('cf-connecting-ip');
    if (clientIP) headers.set('x-mooody-client-ip', clientIP);

    try {
      const body = request.method === 'POST' ? await boundedBody(request) : undefined;
      if (body === null) return errorResponse(413, 'Request body is too large.');
      if (request.signal.aborted) return errorResponse(499, 'Request canceled.');
      const upstreamRequest = new Request(destination.href, {
        method: request.method, headers, body, signal: request.signal,
        // Never forward the server secret to a redirect destination automatically.
        redirect: 'manual',
      });
      const upstream = await fetch(upstreamRequest, {
        cf: { cacheTtl: 0, cacheEverything: false },
      });
      const responseHeaders = new Headers(upstream.headers);
      for (const name of PRIVATE_HEADERS) responseHeaders.delete(name);

      const location = responseHeaders.get('location');
      if (location) {
        const redirect = new URL(location, destination);
        if (redirect.origin !== origin.origin) {
          await upstream.body?.cancel();
          return errorResponse(502, 'Unexpected redirect from the model service.');
        }
        redirect.protocol = 'https:';
        redirect.host = PUBLIC_HOST;
        responseHeaders.set('location', redirect.href);
      }

      const isAPI = publicURL.pathname.startsWith('/api/');
      const isSSE = responseHeaders.get('content-type')?.startsWith('text/event-stream');
      if (isAPI || isSSE) {
        responseHeaders.set('cache-control', 'no-store, no-cache, no-transform');
        responseHeaders.set('pragma', 'no-cache');
        responseHeaders.set('expires', '0');
      }
      if (isSSE) responseHeaders.set('x-accel-buffering', 'no');

      // Pass through the stream without reading it or extending work with waitUntil.
      // Browser cancellation propagates to the upstream signal and response body.
      return new Response(upstream.body, {
        status: upstream.status, statusText: upstream.statusText, headers: responseHeaders,
      });
    } catch {
      // Fetch errors can contain request headers. Return only a fixed public message.
      return request.signal.aborted
        ? errorResponse(499, 'Request canceled.')
        : errorResponse(502, 'The model service is temporarily unavailable.');
    }
  },
};
