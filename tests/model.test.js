import test from 'node:test';
import assert from 'node:assert/strict';
import {
  AXES, AXIS_LABELS, describeMood, initialState, createChat, appendMessage, decodeState, normalizeMood, profilePoints, requestMessages, retryChat,
  moodCoefficientTable, moodCoefficients, configuredMoodCoefficients
} from '../src/model.js';

const NOW = 1_800_000_000_000;

test('public moods use the extracted bank order with readable labels', () => {
  assert.deepEqual(AXES, ['depression', 'curiosity', 'paranoia', 'sexual_arousal', 'narcissism', 'euphoria']);
  assert.equal(AXIS_LABELS[3], 'sexual arousal');
  assert.match(describeMood([0, 0, 0, 2, 0, 0]), /sexual arousal much more/);
});

test('legacy chats survive the axis migration while unrelated old mood levels reset', () => {
  const chat = appendMessage(createChat([2, 1, -1, 0, 2, -2], 'legacy', NOW), 'Keep this conversation');
  const restored = decodeState(JSON.stringify({
    version: 1, mood: [-2, 0, 1, 2, -1, 2], activeId: chat.id, conversations: [chat]
  }));
  assert.equal(restored.version, 2);
  assert.equal(restored.activeId, chat.id);
  assert.deepEqual(restored.mood, [0, 0, 0, 0, 0, 0]);
  assert.deepEqual(restored.conversations[0], { ...chat, mood: [0, 0, 0, 0, 0, 0] });
});

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
  for (const raw of ['{broken', 'null', '{"version":3,"conversations":[]}', '{"version":2,"conversations":{}}']) {
    const state = decodeState(raw, NOW);
    assert.equal(state.version, 2);
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
    version: 2, mood: [1, -1, 0, 2, -2, 0], activeId: 'missing',
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
    version: 2, mood: [], activeId: invalid.id, conversations: [invalid, valid]
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

const coefficientConfig = (table = [-1, -0.5, 0, 0.5, 1]) => ({
  axes: [...AXES], mood_levels: [-2, -1, 0, 1, 2], mood_coefficients: table
});

test('signed saved slider levels map directly to advertised actual coefficients without changing storage or radar', () => {
  const levels = [-2, -1, 0, 1, 2, -1];
  const state = initialState();
  state.mood = [...levels];
  state.conversations = [createChat(levels, 'mapped', NOW)];
  state.activeId = 'mapped';
  const stored = JSON.stringify(state);
  const radar = profilePoints(state.conversations[0].mood);
  assert.deepEqual(moodCoefficients(levels, coefficientConfig()), [-1, -0.5, 0, 0.5, 1, -0.5]);
  assert.deepEqual(moodCoefficients(levels, coefficientConfig([-0.5, -0.25, 0, 0.25, 0.5])),
    [-0.5, -0.25, 0, 0.25, 0.5, -0.25]);
  assert.deepEqual(moodCoefficients([0, 0, 0, 0, 0, 0], coefficientConfig()), [0, 0, 0, 0, 0, 0]);
  assert.equal(JSON.stringify(state), stored);
  assert.deepEqual(decodeState(stored).conversations[0].mood, levels);
  assert.equal(profilePoints(decodeState(stored).conversations[0].mood), radar);
  const table = moodCoefficientTable(coefficientConfig());
  table[0] = 0;
  assert.deepEqual(moodCoefficientTable(coefficientConfig()), [-1, -0.5, 0, 0.5, 1]);
});

test('missing, legacy, unsafe or mismatched coefficient configuration never falls back to ordinal coefficients', () => {
  const invalid = [
    null, {}, { axes: AXES, mood_levels: [-2, -1, 0, 1, 2] },
    coefficientConfig([-2, -1, 0, 1, 2]), coefficientConfig([-1, -0.5, 0, 0.5]),
    coefficientConfig([-1, -0.5, 0, 0.5, 0.5]), coefficientConfig([-1, -0.5, 0.1, 0.5, 1]),
    coefficientConfig([-1, -0.5, 0, 0.25, 1]), coefficientConfig([-1, 0.5, 0, -0.5, 1]),
    coefficientConfig([-1, '-0.5', 0, 0.5, 1]), coefficientConfig([-1, false, 0, 0.5, 1]),
    coefficientConfig([-1, NaN, 0, 0.5, 1]), coefficientConfig([-Infinity, -0.5, 0, 0.5, Infinity]),
    coefficientConfig(Array(5)), { ...coefficientConfig(), mood_levels: [-1, -0.5, 0, 0.5, 1] },
    { ...coefficientConfig(), axes: [...AXES].reverse() }
  ];
  for (const config of invalid) {
    assert.throws(() => moodCoefficients([2, 0, 0, 0, 0, 0], config), /Refresh this page/);
    assert.throws(() => moodCoefficients([0, 0, 0, 0, 0, 0], config), /Refresh this page/);
  }
  for (const levels of [[1, 0, 0, 0, 0], [0.5, 0, 0, 0, 0, 0], ['1', 0, 0, 0, 0, 0], Array(6)]) {
    assert.throws(() => moodCoefficients(levels, coefficientConfig()), /Choose a valid mood/);
  }
});

test('a pending configuration gates coefficient requests until its advertised table is validated', async () => {
  let resolveConfig;
  const pendingConfig = new Promise(resolve => { resolveConfig = resolve; });
  let sent = null;
  const request = configuredMoodCoefficients([2, -1, 0, 1, -2, 0], pendingConfig)
    .then(coefficients => { sent = coefficients; });
  await Promise.resolve();
  assert.equal(sent, null);
  resolveConfig(coefficientConfig());
  await request;
  assert.deepEqual(sent, [1, -0.5, 0, 0.5, -1, 0]);
  for (const config of [null, {}, coefficientConfig([-2, -1, 0, 1, 2])]) {
    await assert.rejects(configuredMoodCoefficients([2, 0, 0, 0, 0, 0], Promise.resolve(config)), /Refresh this page/);
  }
  await assert.rejects(configuredMoodCoefficients([0, 0, 0, 0, 0, 0], Promise.reject(new Error('network failure'))),
    /Refresh this page/);
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
