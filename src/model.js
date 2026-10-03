export const AXES = ['warmth', 'patience', 'playfulness', 'optimism', 'energy', 'curiosity'];
export const LEVELS = ['much less', 'less', 'balanced', 'more', 'much more'];
export const NEUTRAL = Object.freeze([0, 0, 0, 0, 0, 0]);
export const BRAND_MOOD = Object.freeze([2, -2, 1, -1, 2, 0]);
export const STORAGE_KEY = 'mooody.notebook.v1';

export function normalizeMood(value) {
  return AXES.map((_, index) =>
    Number.isInteger(value?.[index]) && value[index] >= -2 && value[index] <= 2
      ? value[index]
      : 0
  );
}

export function point(index, radius, cx = 215, cy = 188) {
  const angle = -Math.PI / 2 + index * Math.PI / 3;
  return [cx + Math.cos(angle) * radius, cy + Math.sin(angle) * radius];
}

export function profilePoints(mood, radius = 132, cx = 215, cy = 188) {
  return normalizeMood(mood).map((value, index) =>
    point(index, radius * (.25 + (value + 2) * .1875), cx, cy)
      .map(number => number.toFixed(2)).join(',')
  ).join(' ');
}

export function ringPoints(radius, cx = 215, cy = 188) {
  return AXES.map((_, index) => point(index, radius, cx, cy)
    .map(number => number.toFixed(2)).join(',')).join(' ');
}

export function describeMood(mood) {
  return normalizeMood(mood).map((value, index) => `${AXES[index]} ${LEVELS[value + 2]}`).join(', ');
}

export function initialState() {
  return {
    version: 1,
    mood: [...NEUTRAL],
    activeId: null,
    conversations: []
  };
}

export function decodeState(raw, now = Date.now()) {
  try {
    const saved = JSON.parse(raw);
    if (saved?.version !== 1 || !Array.isArray(saved.conversations)) return initialState(now);
    const ids = new Set();
    const conversations = saved.conversations.filter(chat => {
      if (!chat || typeof chat.id !== 'string' || !chat.id || ids.has(chat.id)
        || typeof chat.title !== 'string' || !Number.isFinite(chat.createdAt)
        || !Number.isFinite(new Date(chat.createdAt).getTime())
        || !Array.isArray(chat.messages)
        || !chat.messages.every(message => message && ['you', 'mooody'].includes(message.who)
          && typeof message.text === 'string')) return false;
      ids.add(chat.id);
      return true;
    }).map(chat => ({
      id: chat.id, title: chat.title, createdAt: chat.createdAt,
      mood: normalizeMood(chat.mood),
      messages: chat.messages.map(({ who, text, status }) => ({
        who, text,
        ...(['streaming', 'interrupted', 'error'].includes(status)
          ? { status: status === 'streaming' ? 'interrupted' : status }
          : {})
      }))
    }));
    return {
      version: 1,
      mood: normalizeMood(saved.mood),
      activeId: conversations.some(chat => chat.id === saved.activeId) ? saved.activeId : null,
      conversations
    };
  } catch {
    return initialState(now);
  }
}

function chatId() {
  return globalThis.crypto?.randomUUID?.()
    ?? `chat-${Date.now().toString(36)}-${Math.random().toString(36).slice(2)}`;
}

export function createChat(mood, id = chatId(), now = Date.now()) {
  return {
    id, title: '', createdAt: now, mood: normalizeMood(mood),
    messages: []
  };
}

export function appendMessage(chat, input) {
  const text = String(input).trim().slice(0, 2000);
  if (!text) return chat;
  return {
    ...chat,
    title: chat.title || (text.length > 38 ? text.slice(0, 35) + '…' : text),
    messages: [...chat.messages, { who: 'you', text }]
  };
}

export function requestMessages(chat, maxMessages = 32) {
  const messages = [];
  for (const message of chat.messages) {
    if (!message.text.trim() || message.status) continue;
    const role = message.who === 'you' ? 'user' : 'assistant';
    if (!messages.length && role === 'assistant') continue;
    if (messages.at(-1)?.role === role) {
      // After an interrupted turn, preserve the latest request verbatim.
      messages[messages.length - 1] = { role, content: message.text };
    } else {
      messages.push({ role, content: message.text });
    }
  }
  const recent = messages.slice(-Math.max(1, maxMessages));
  if (recent[0]?.role === 'assistant') recent.shift();
  while (recent.length > 1 && (recent.reduce((total, message) => total + message.content.length, 0) > 32000
    || recent.some(message => message.role === 'assistant' && message.content.length > 10000))) {
    recent.splice(0, 2);
  }
  return recent;
}

export function retryChat(chat) {
  const messages = [...chat.messages];
  while (messages.at(-1)?.who === 'mooody' && messages.at(-1)?.status) messages.pop();
  return messages.at(-1)?.who === 'you' ? { ...chat, messages } : null;
}
