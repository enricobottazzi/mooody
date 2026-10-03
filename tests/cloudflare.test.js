import test from 'node:test';
import assert from 'node:assert/strict';
import worker from '../cloudflare/worker.js';

const ENV = {
  MOOODY_ORIGIN_URL: 'https://owner--mooody-web-web.modal.run',
  MOOODY_PROXY_TOKEN: 'test-only-secret-0123456789abcdef0123456789abcdef',
};

test('proxy keeps the exact Modal origin, forwards the body, and replaces forged trust headers', async t => {
  const body = JSON.stringify({ messages: [{ role: 'user', content: 'Hello' }], mood: [0, 0, 0, 0, 0, 0] });
  let received;
  let options;
  t.mock.method(globalThis, 'fetch', async (request, init) => {
    received = request;
    options = init;
    assert.equal(await request.text(), body);
    return new Response('okay', { headers: { 'x-mooody-proxy-token': ENV.MOOODY_PROXY_TOKEN } });
  });
  const response = await worker.fetch(new Request('https://mooody.ai/api/chat?session=x%2Fy', {
    method: 'POST', body,
    headers: {
      'content-type': 'application/json', origin: 'https://mooody.ai',
      'cf-connecting-ip': '203.0.113.4', 'x-mooody-client-ip': 'forged',
      'x-mooody-proxy-token': 'forged', 'x-forwarded-for': 'forged',
      'x-forwarded-host': 'evil.example',
    },
  }), ENV);
  assert.equal(received.url, `${ENV.MOOODY_ORIGIN_URL}/api/chat?session=x%2Fy`);
  assert.equal(received.headers.get('origin'), 'https://mooody.ai');
  assert.equal(received.headers.get('x-mooody-client-ip'), '203.0.113.4');
  assert.equal(received.headers.get('x-mooody-proxy-token'), ENV.MOOODY_PROXY_TOKEN);
  assert.equal(received.headers.get('x-forwarded-host'), 'mooody.ai');
  assert.equal(received.headers.get('x-forwarded-for'), null);
  assert.equal(received.redirect, 'manual');
  assert.equal(options.cf.cacheTtl, 0);
  assert.equal(response.headers.get('x-mooody-proxy-token'), null);
  assert.match(response.headers.get('cache-control'), /no-store/);
});

test('path tricks and missing Cloudflare client IP cannot override the origin or trusted IP', async t => {
  let received;
  t.mock.method(globalThis, 'fetch', async request => {
    received = request;
    return new Response('okay');
  });
  await worker.fetch(new Request('https://mooody.ai//evil.example/assets/site.css?x=1', {
    headers: { 'x-mooody-client-ip': 'forged' },
  }), ENV);
  assert.equal(received.url, `${ENV.MOOODY_ORIGIN_URL}//evil.example/assets/site.css?x=1`);
  assert.equal(received.headers.get('x-mooody-client-ip'), null);
});

test('SSE reaches the browser before completion and cancellation reaches the upstream body', async t => {
  let source;
  let cancellation;
  const stream = new ReadableStream({
    start(controller) { source = controller; },
    cancel(reason) { cancellation = reason; },
  });
  t.mock.method(globalThis, 'fetch', async () => new Response(stream, {
    headers: { 'content-type': 'text/event-stream', 'cache-control': 'public, max-age=3600' },
  }));
  const response = await worker.fetch(new Request('https://mooody.ai/api/chat', {
    method: 'POST', body: '{}',
  }), ENV);
  const reader = response.body.getReader();
  source.enqueue(new TextEncoder().encode('data: {"type":"status"}\n\n'));
  const first = await reader.read();
  assert.equal(new TextDecoder().decode(first.value), 'data: {"type":"status"}\n\n');
  assert.equal(first.done, false);
  assert.match(response.headers.get('cache-control'), /no-store.*no-transform/);
  assert.equal(response.headers.get('x-accel-buffering'), 'no');
  await reader.cancel('user stopped');
  assert.equal(cancellation, 'user stopped');
});

test('incoming abort also aborts the forwarded request signal', async t => {
  const controller = new AbortController();
  let received;
  t.mock.method(globalThis, 'fetch', async request => {
    received = request;
    return new Response('okay');
  });
  await worker.fetch(new Request('https://mooody.ai/', { signal: controller.signal }), ENV);
  assert.equal(received.signal.aborted, false);
  controller.abort();
  assert.equal(received.signal.aborted, true);
});

test('aborting an in-flight request stops the upstream fetch without exposing its error', async t => {
  const controller = new AbortController();
  let started;
  const upstreamStarted = new Promise(resolve => { started = resolve; });
  t.mock.method(globalThis, 'fetch', request => new Promise((resolve, reject) => {
    request.signal.addEventListener('abort', () => reject(new Error(ENV.MOOODY_PROXY_TOKEN)), { once: true });
    started();
  }));
  const pending = worker.fetch(new Request('https://mooody.ai/api/chat', {
    method: 'POST', body: '{}', signal: controller.signal,
  }), ENV);
  await upstreamStarted;
  controller.abort();
  const response = await pending;
  assert.equal(response.status, 499);
  assert.doesNotMatch(await response.text(), /test-only-secret/);
});

test('rejects actual oversized streaming bodies before invoking the GPU service', async t => {
  let calls = 0;
  t.mock.method(globalThis, 'fetch', async () => { calls++; return new Response('unexpected'); });
  const body = new ReadableStream({
    start(controller) {
      controller.enqueue(new Uint8Array(32_768));
      controller.enqueue(new Uint8Array(32_769));
      controller.close();
    },
  });
  const response = await worker.fetch(new Request('https://mooody.ai/api/chat', {
    method: 'POST', body, duplex: 'half', headers: { 'content-length': '2' },
  }), ENV);
  assert.equal(response.status, 413);
  assert.equal(calls, 0);
});

test('upstream redirects never follow to another host and same-origin redirects remain on mooody.ai', async t => {
  let calls = 0;
  let location = 'https://evil.example/collect';
  t.mock.method(globalThis, 'fetch', async request => {
    calls++;
    assert.equal(request.redirect, 'manual');
    return new Response(null, { status: 303, headers: { location } });
  });
  const rejected = await worker.fetch(new Request('https://mooody.ai/'), ENV);
  assert.equal(rejected.status, 502);
  assert.equal(calls, 1);
  location = `${ENV.MOOODY_ORIGIN_URL}/api/chat?result=123`;
  const redirected = await worker.fetch(new Request('https://mooody.ai/'), ENV);
  assert.equal(redirected.status, 303);
  assert.equal(redirected.headers.get('location'), 'https://mooody.ai/api/chat?result=123');
});

test('configuration and upstream failures return fixed messages without leaking credentials', async t => {
  let calls = 0;
  t.mock.method(globalThis, 'fetch', async () => {
    calls++;
    throw new Error(`Authorization failed for ${ENV.MOOODY_PROXY_TOKEN}`);
  });
  for (const env of [
    { ...ENV, MOOODY_PROXY_TOKEN: undefined },
    { ...ENV, MOOODY_ORIGIN_URL: 'https://evil.example/' },
    { ...ENV, MOOODY_ORIGIN_URL: `${ENV.MOOODY_ORIGIN_URL}/nested` },
  ]) {
    const response = await worker.fetch(new Request('https://mooody.ai/'), env);
    assert.equal(response.status, 503);
    assert.doesNotMatch(await response.text(), /test-only-secret/);
  }
  assert.equal(calls, 0);
  const response = await worker.fetch(new Request('https://mooody.ai/'), ENV);
  assert.equal(response.status, 502);
  assert.doesNotMatch(await response.text(), /test-only-secret/);
});

test('unexpected hostnames and unsupported methods do not reach Modal', async t => {
  let calls = 0;
  t.mock.method(globalThis, 'fetch', async () => { calls++; return new Response('unexpected'); });
  assert.equal((await worker.fetch(new Request('https://evil.example/'), ENV)).status, 421);
  assert.equal((await worker.fetch(new Request('https://mooody.ai/api/chat', { method: 'DELETE' }), ENV)).status, 405);
  assert.equal((await worker.fetch(new Request('https://mooody.ai/styles.css', { method: 'POST' }), ENV)).status, 405);
  assert.equal(calls, 0);
});
