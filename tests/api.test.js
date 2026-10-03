import test from 'node:test';
import assert from 'node:assert/strict';
import { streamChat, ChatError } from '../src/api.js';

const encoder = new TextEncoder();
const payload = { messages: [{ role: 'user', content: 'hello' }], mood: [0, 0, 0, 0, 0, 0] };
const event = (type, body) => `event: ${type}\r\ndata: ${JSON.stringify(body)}\r\n\r\n`;

function streamResponse(text, chunkSize = 7) {
  const bytes = encoder.encode(text);
  return new Response(new ReadableStream({
    start(controller) {
      for (let position = 0; position < bytes.length; position += chunkSize) {
        controller.enqueue(bytes.slice(position, position + chunkSize));
      }
      controller.close();
    }
  }), { headers: { 'Content-Type': 'text/event-stream; charset=utf-8' } });
}

test('streaming survives arbitrary UTF-8 and SSE boundaries and sends the correct same-origin request', async () => {
  const events = [];
  const result = await streamChat(payload, {
    fetchImpl: async (url, options) => {
      assert.equal(url, '/api/chat');
      assert.equal(options.method, 'POST');
      assert.deepEqual(JSON.parse(options.body), payload);
      return streamResponse(': heartbeat\r\n\r\n'
        + event('status', { message: 'Loading the model.' })
        + event('meta', { model: 'demivoleegaston/Qwen3.5-9B-mooody', steering_available: false })
        + event('token', { text: 'Caffè ☕ ' })
        + event('token', { text: 'is ready.' })
        + event('done', { finish_reason: 'eos', generated_tokens: 7 }), 1);
    },
    onEvent: value => events.push(value)
  });
  assert.equal(events.filter(value => value.type === 'token').map(value => value.text).join(''), 'Caffè ☕ is ready.');
  assert.equal(events[0].type, 'status');
  assert.equal(result.finish_reason, 'eos');
});

test('early disconnect retains already delivered text and reports an interrupted stream', async () => {
  const received = [];
  await assert.rejects(streamChat(payload, {
    fetchImpl: async () => streamResponse(event('token', { text: 'Partial answer' })),
    onEvent: value => received.push(value)
  }), error => error instanceof ChatError && error.code === 'interrupted_stream');
  assert.equal(received[0].text, 'Partial answer');
});

test('generation errors and malformed token frames never count as successful replies', async () => {
  await assert.rejects(streamChat(payload, {
    fetchImpl: async () => streamResponse(event('error', { code: 'queue_full', message: 'The queue is full. Try again shortly.' }))
  }), error => error.code === 'queue_full' && error.message.includes('queue is full'));
  await assert.rejects(streamChat(payload, {
    fetchImpl: async () => streamResponse('event: token\ndata: {broken}\n\n')
  }), error => error.code === 'invalid_stream');
  await assert.rejects(streamChat(payload, {
    fetchImpl: async () => streamResponse(event('token', { text: null }))
  }), error => error.code === 'invalid_stream');
});

test('HTTP admission errors show the server message and proxy HTML produces a usable fallback', async () => {
  await assert.rejects(streamChat(payload, {
    fetchImpl: async () => Response.json({ code: 'busy', message: 'Please wait before sending again.' }, { status: 429 })
  }), error => error.code === 'busy' && error.message === 'Please wait before sending again.');
  await assert.rejects(streamChat(payload, {
    fetchImpl: async () => new Response('<html>Bad gateway</html>', { status: 502 })
  }), error => error instanceof ChatError && error.code === 'http_502' && !error.message.includes('<html>'));
});

test('stop passes AbortController to fetch and cancels the response reader', async () => {
  const controller = new AbortController();
  let cancelled = false;
  let suppliedSignal;
  const received = [];
  const fetchImpl = async (_, options) => {
    suppliedSignal = options.signal;
    return new Response(new ReadableStream({
      start(stream) {
        stream.enqueue(encoder.encode(event('token', { text: 'Start' })));
      },
      cancel() { cancelled = true; }
    }), { headers: { 'Content-Type': 'text/event-stream' } });
  };
  await assert.rejects(streamChat(payload, {
    signal: controller.signal, fetchImpl,
    onEvent(value) {
      received.push(value);
      controller.abort();
    }
  }), error => error.name === 'AbortError');
  assert.equal(suppliedSignal, controller.signal);
  assert.equal(received[0].text, 'Start');
  assert.equal(cancelled, true);
});

test('completion cancels further input and accepts a final frame without a blank line', async () => {
  const received = [];
  const done = await streamChat(payload, {
    fetchImpl: async () => streamResponse(event('token', { text: 'Done' })
      + 'event: done\ndata: {"finish_reason":"length"}'),
    onEvent: value => received.push(value)
  });
  assert.equal(done.finish_reason, 'length');
  assert.deepEqual(received.map(value => value.type), ['token', 'done']);
});
