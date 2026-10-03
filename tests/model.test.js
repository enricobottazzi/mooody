import test from 'node:test';
import assert from 'node:assert/strict';
import {
  initialState, createChat, appendMessage, decodeState, normalizeMood, profilePoints, requestMessages, retryChat
} from '../src/model.js';

const NOW = 1_800_000_000_000;

test('fresh states begin at neutral and do not share editable data', () => {
  const first = initialState(NOW);
  const second = initialState(NOW);
  assert.deepEqual(first.mood, [0, 0, 0, 0, 0, 0]);
  assert.equal(first.activeId, null);
  assert.deepEqual(first.conversations, []);
  first.mood[0] = 2;
  first.conversations.push(createChat([2, 0, 0, 0, 0, 0], 'first', NOW));
  assert.equal(second.mood[0], 0);
  assert.deepEqual(second.conversations, []);
});

test('each new chat saves an independent copy of its mood', () => {
  const draft = [2, 1, 2, 0, 0, -1];
  const first = createChat(draft, 'first', NOW);
  draft[0] = -2;
  draft[2] = -2;
  const second = createChat(draft, 'second', NOW + 1);
  draft.fill(0);
  assert.deepEqual(first.mood, [2, 1, 2, 0, 0, -1]);
  assert.deepEqual(second.mood, [-2, 1, -2, 0, 0, -1]);
  first.mood[1] = -1;
  assert.equal(second.mood[1], 1);
});

test('new chats work on HTTP mobile previews where randomUUID is unavailable', { concurrency: false }, t => {
  const descriptor = Object.getOwnPropertyDescriptor(globalThis, 'crypto');
  let random = 0;
  t.mock.method(Date, 'now', () => NOW);
  t.mock.method(Math, 'random', () => ++random / 10);
  try {
    for (const unavailableCrypto of [undefined, {}]) {
      Object.defineProperty(globalThis, 'crypto', { configurable: true, value: unavailableCrypto });
      const first = createChat([1, 0, 0, 0, 0, 0]);
      const second = createChat([0, 0, 0, 0, 0, 0]);
      assert.equal(typeof first.id, 'string');
      assert.ok(first.id.length > 0);
      assert.notEqual(first.id, second.id);
      assert.deepEqual(first.mood, [1, 0, 0, 0, 0, 0]);
      assert.equal(appendMessage(first, 'hello').messages.at(-1).text, 'hello');
    }
  } finally {
    if (descriptor) Object.defineProperty(globalThis, 'crypto', descriptor);
    else delete globalThis.crypto;
  }
});

test('new chats contain no fabricated replies; sending adds only the real user message', () => {
  const chat = createChat([1, 0, 2, 0, 0, -2], 'new', NOW);
  assert.equal(appendMessage(chat, ' \n\t '), chat);
  const sent = appendMessage(chat, '  I need names for my project.  ');
  assert.equal(sent.title, 'I need names for my project.');
  assert.equal(chat.title, '');
  assert.equal(chat.messages.length, 0);
  assert.deepEqual(sent.messages, [{ who: 'you', text: 'I need names for my project.' }]);
  const continued = appendMessage(sent, 'another thought');
  assert.equal(continued.title, sent.title);
  assert.equal(continued.messages.length, sent.messages.length + 1);
});

test('interrupted streaming history survives reload without becoming a completed model answer', () => {
  const chat = appendMessage(createChat([1, 0, 2, 0, 0, -2], 'stream', NOW), 'hello');
  chat.messages.push({ who: 'mooody', text: 'Part of a reply', status: 'streaming' });
  const state = { ...initialState(), conversations: [chat], activeId: chat.id };
  const restored = decodeState(JSON.stringify(state));
  assert.deepEqual(restored.conversations[0].messages.at(-1), {
    who: 'mooody', text: 'Part of a reply', status: 'interrupted'
  });
  assert.deepEqual(requestMessages(restored.conversations[0]), [{ role: 'user', content: 'hello' }]);
  assert.deepEqual(retryChat(restored.conversations[0]).messages, [{ who: 'you', text: 'hello' }]);
  assert.equal(chat.messages.at(-1).status, 'streaming');
});

test('requests keep recent real history, skip initial greeting, and end with the latest user turn', () => {
  const chat = createChat([], 'long', NOW);
  chat.messages.push({ who: 'mooody', text: 'An old greeting.' });
  for (let index = 0; index < 20; index++) {
    chat.messages.push({ who: 'you', text: `Question ${index}` }, { who: 'mooody', text: `Answer ${index}` });
  }
  chat.messages.push({ who: 'you', text: 'Latest question' });
  const messages = requestMessages(chat, 32);
  assert.equal(messages.length, 31);
  assert.equal(messages[0].role, 'user');
  assert.deepEqual(messages.at(-1), { role: 'user', content: 'Latest question' });
  assert.ok(messages.every(message => !message.content.includes('greeting')));
  assert.equal(retryChat({ ...chat, messages: chat.messages.slice(0, -1) }), null);
});

test('failed turns and long context preserve the latest user request without concatenating or clipping it', () => {
  const chat = createChat([], 'bounds', NOW);
  for (let index = 0; index < 8; index++) {
    chat.messages.push({ who: 'you', text: 'q'.repeat(2000) }, { who: 'mooody', text: 'a'.repeat(8000) });
  }
  chat.messages.push({ who: 'you', text: 'f'.repeat(2000) }, { who: 'mooody', text: 'unfinished', status: 'interrupted' });
  const latest = 'l'.repeat(2000);
  chat.messages.push({ who: 'you', text: latest });
  const messages = requestMessages(chat);
  assert.deepEqual(messages.at(-1), { role: 'user', content: latest });
  assert.ok(messages.reduce((sum, message) => sum + message.content.length, 0) <= 32000);
  assert.ok(messages.every((message, index) => message.role === (index % 2 ? 'assistant' : 'user')));
  assert.ok(messages.every(message => !message.content.includes('unfinished') && !message.content.includes('f')));
});

test('saved conversations, draft mood, and the open chat survive a persistence roundtrip', () => {
  const state = initialState(NOW);
  const chat = appendMessage(createChat([-2, 1, 0, 2, -1, 2], 'saved', NOW), 'An idea for a small app');
  state.mood = [2, 0, -1, 1, 0, -2];
  state.conversations.unshift(chat);
  state.activeId = chat.id;
  const restored = decodeState(JSON.stringify(state), NOW);
  assert.deepEqual(restored, state);
  restored.conversations[0].messages[0].text = 'changed';
  assert.notEqual(state.conversations[0].messages[0].text, 'changed');
});

test('malformed or incompatible saved data falls back to a usable fresh state', () => {
  for (const raw of ['{broken', 'null', '{"version":2,"conversations":[]}', '{"version":1,"conversations":{}}']) {
    const state = decodeState(raw, NOW);
    assert.equal(state.version, 1);
    assert.deepEqual(state.mood, [0, 0, 0, 0, 0, 0]);
    assert.equal(state.activeId, null);
    assert.deepEqual(state.conversations, []);
  }
});

test('restoration discards corrupt and duplicate chats, sanitizes fields, and clears an orphan active chat', () => {
  const valid = {
    id: 'valid', title: 'kept', createdAt: NOW, mood: [2, 9, -2, 1.5, '1', null],
    messages: [{ who: 'you', text: 'hello', unexpected: 'drop me' }], unexpected: true
  };
  const raw = JSON.stringify({
    version: 1, mood: [1, -1, 0, 2, -2, 0], activeId: 'missing',
    conversations: [
      null, valid, { ...valid, title: 'duplicate' }, { ...valid, id: '' },
      { ...valid, id: 'bad-title', title: 42 },
      { ...valid, id: 'bad-date', createdAt: null },
      { ...valid, id: 'bad-messages', messages: [{ who: 'system', text: 'wrong role' }] },
      { ...valid, id: 'bad-text', messages: [{ who: 'you', text: null }] },
      { ...valid, id: 'no-messages', messages: null }
    ]
  });
  const state = decodeState(raw, NOW);
  assert.equal(state.conversations.length, 1);
  assert.equal(state.conversations[0].title, 'kept');
  assert.deepEqual(state.conversations[0].mood, [2, 0, -2, 0, 0, 0]);
  assert.deepEqual(state.conversations[0].messages, [{ who: 'you', text: 'hello' }]);
  assert.equal('unexpected' in state.conversations[0], false);
  assert.equal(state.activeId, null);
});

test('restoration rejects finite timestamps outside the supported date range', () => {
  const valid = createChat([0, 0, 0, 0, 0, 0], 'valid', NOW);
  const invalid = { ...valid, id: 'invalid-date', createdAt: 1e30 };
  const restored = decodeState(JSON.stringify({
    version: 1, mood: [], activeId: invalid.id, conversations: [invalid, valid]
  }), NOW);
  assert.deepEqual(restored.conversations.map(chat => chat.id), ['valid']);
  assert.equal(restored.activeId, null);
  assert.equal(new Date(restored.conversations[0].createdAt).toISOString(), new Date(NOW).toISOString());
});

test('moods allow exactly the five integer stops and default missing or invalid axes to neutral', () => {
  assert.deepEqual(normalizeMood([-2, -1, 0, 1, 2, 0]), [-2, -1, 0, 1, 2, 0]);
  assert.deepEqual(normalizeMood([-3, 3, 0.5, '2', NaN, Infinity]), [0, 0, 0, 0, 0, 0]);
  assert.deepEqual(normalizeMood([1, -1]), [1, -1, 0, 0, 0, 0]);
  assert.deepEqual(normalizeMood(undefined), [0, 0, 0, 0, 0, 0]);
});

test('radar profiles have six points and extreme moods occupy the inner and outer radii', () => {
  const center = [200, 200];
  const parse = text => text.split(' ').map(pair => pair.split(',').map(Number));
  const minimum = parse(profilePoints(Array(6).fill(-2), 100, ...center));
  const maximum = parse(profilePoints(Array(6).fill(2), 100, ...center));
  assert.equal(minimum.length, 6);
  assert.equal(maximum.length, 6);
  for (const [points, expectedRadius] of [[minimum, 25], [maximum, 100]]) {
    for (const [x, y] of points) {
      assert.ok(Math.abs(Math.hypot(x - center[0], y - center[1]) - expectedRadius) < 0.01);
    }
  }
  assert.deepEqual(maximum[0], [200, 100]);
  assert.deepEqual(maximum[3], [200, 300]);
  assert.ok(maximum[1][0] > center[0]);
  assert.ok(maximum[5][0] < center[0]);
});
