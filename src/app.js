import {
  AXES, LEVELS, NEUTRAL, BRAND_MOOD, STORAGE_KEY, initialState, decodeState,
  createChat, appendMessage, requestMessages, retryChat, point, profilePoints, ringPoints, describeMood
} from './model.js';
import { streamChat, ChatError } from './api.js';

const app = document.getElementById('app');
function readStoredState() {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    return raw ? decodeState(raw) : null;
  } catch {
    return null;
  }
}
let state = readStoredState() ?? initialState();
let pending = null;
const notices = new Map();
let capabilities = {
  mood_vectors_available: false,
  steering_available: false,
  mood_vectors_source: null,
  mood_vectors_validated: false,
  max_messages: 32
};

const escape = value => String(value).replace(/[&<>"']/g, character => ({
  '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'
})[character]);

function persist(changedChat = null) {
  try {
    const saved = readStoredState();
    if (saved) {
      state.conversations = saved.conversations;
      if (changedChat) {
        const index = state.conversations.findIndex(chat => chat.id === changedChat.id);
        if (index < 0) state.conversations.unshift(changedChat);
        else state.conversations[index] = changedChat;
      }
    }
    localStorage.setItem(STORAGE_KEY, JSON.stringify(state));
  } catch {
    // Chat remains usable for this session if browser storage is unavailable.
  }
}

function logoArtwork(mood) {
  const rings = [4.25, 7.4375, 10.625, 13.8125, 17].map(radius =>
    `<polygon class="logo-ring" points="${ringPoints(radius, 20, 20)}"/>`
  ).join('');
  const spokes = AXES.map((_, index) => {
    const [x, y] = point(index, 17, 20, 20);
    return `<line class="logo-axis" x1="20" y1="20" x2="${x}" y2="${y}"/>`;
  }).join('');
  return `${rings}${spokes}<polygon class="logo-profile" points="${profilePoints(mood, 17, 20, 20)}"/>`;
}

function logo(chatMood = null) {
  const mood = chatMood ?? BRAND_MOOD;
  const description = chatMood ? `Mooody mood: ${describeMood(mood)}` : 'Mooody radar logo';
  return `<svg class="brand-mark" viewBox="0 0 40 40" role="img" aria-label="${description}">${logoArtwork(mood)}</svg>`;
}

function updateTabIcon(chatMood = null) {
  const favicon = document.querySelector('link[rel="icon"]');
  let href = './favicon.svg';
  if (chatMood) {
    const svg = `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 40 40"><style>
      .logo-ring, .logo-axis { fill: none; stroke: #dddacf; stroke-width: .45; }
      .logo-profile { fill: #62775318; stroke: #627753; stroke-width: 1.3; stroke-linejoin: round; }
      @media (prefers-color-scheme: dark) {
        .logo-ring, .logo-axis { stroke: #37362f; }
        .logo-profile { fill: #9cb58918; stroke: #9cb589; }
      }
    </style>${logoArtwork(chatMood)}</svg>`;
    href = `data:image/svg+xml,${encodeURIComponent(svg)}`;
  }
  if (favicon.getAttribute('href') !== href) favicon.setAttribute('href', href);
}

function graph(mood) {
  const rings = [33, 57.75, 82.5, 107.25, 132].map(radius =>
    `<polygon class="graph-ring" points="${ringPoints(radius)}"/>`
  ).join('');
  const spokes = AXES.map((_, index) => {
    const [x, y] = point(index, 132);
    return `<line class="graph-axis" x1="215" y1="188" x2="${x}" y2="${y}"/>`;
  }).join('');
  return `<svg class="mood-graph" viewBox="65 38 300 300" role="img" aria-label="Mood profile: ${describeMood(mood)}">${rings}${spokes}<polygon class="graph-profile" points="${profilePoints(mood)}"/></svg>`;
}

function miniature(mood) {
  return `<svg class="mini-profile" viewBox="0 0 40 40" role="img" aria-label="Mood profile: ${describeMood(mood)}"><polygon class="logo-outline" points="${ringPoints(17, 20, 20)}"/><polygon class="logo-profile" points="${profilePoints(mood, 17, 20, 20)}"/></svg>`;
}

function dateLabel(timestamp) {
  const date = new Date(timestamp);
  const today = new Date();
  const yesterday = new Date();
  yesterday.setDate(today.getDate() - 1);
  if (date.toDateString() === today.toDateString()) return 'today';
  if (date.toDateString() === yesterday.toDateString()) return 'yesterday';
  return new Intl.DateTimeFormat('en-GB', { day: 'numeric', month: 'short' }).format(date).toLowerCase();
}

function setup() {
  const controls = AXES.map((name, axis) => `<div class="control-row">
    <span class="emotion" id="mood-${name}">${name}</span>
    <div class="mood-stops" role="radiogroup" aria-labelledby="mood-${name}">
      ${LEVELS.map((level, index) => `<label class="mood-stop"><input type="radio" name="mood-${name}" data-axis="${axis}" value="${index - 2}" aria-label="${name}: ${level}" ${state.mood[axis] === index - 2 ? 'checked' : ''}></label>`).join('')}
    </div>
  </div>`).join('');
  const history = state.conversations.map(chat => `<button type="button" class="history-item" data-chat-id="${escape(chat.id)}">
    <span class="history-title">${escape(chat.title || 'untitled chat')}</span>
    <span class="history-meta"><time datetime="${new Date(chat.createdAt).toISOString()}">${dateLabel(chat.createdAt)}</time>${miniature(chat.mood)}</span>
  </button>`).join('') || '<p class="empty-history">Your conversations will appear here.<br>History is saved in this browser.</p>';
  return `<section class="setup"><div class="setup-layout">
    <div class="mood-editor">
      <h1>Set the mood.</h1>
      <div class="mood-controls"><div class="control-list">${controls}</div></div>
      <button class="start-chat" type="button" data-action="start" aria-label="Chat with mooody using your selected mood"><span class="start-chat-label">chat with</span><span class="graph-holder">${graph(state.mood)}</span><span class="start-chat-name">mooody</span></button>
    </div>
    <section class="history" aria-labelledby="history-heading"><h2 id="history-heading">history</h2><div class="history-list">${history}</div></section>
  </div></section>`;
}

function messageMarkup(message) {
  const partial = ['interrupted', 'error'].includes(message.status);
  return `<article class="message"><span class="speaker">${message.who}</span><div><p class="message-text">${escape(message.text)}</p>${partial ? '<span class="message-note">reply interrupted</span>' : ''}</div></article>`;
}

function noticeFor(chat) {
  if (pending?.chat.id === chat.id) return { text: pending.status, busy: true };
  if (notices.has(chat.id)) return notices.get(chat.id);
  if (chat.messages.at(-1)?.status || chat.messages.at(-1)?.who === 'you') {
    return { text: 'This reply was interrupted. You can retry it.', retry: true };
  }
  return { text: '' };
}

function composerButton(busy) {
  return busy
    ? '<svg viewBox="0 0 24 24" aria-hidden="true"><rect x="6" y="6" width="12" height="12"/></svg>'
    : '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M12 19V5m-6 6 6-6 6 6" stroke-linecap="round" stroke-linejoin="round"/></svg>';
}

function conversation(chat) {
  const notice = noticeFor(chat);
  const busy = Boolean(notice.busy);
  return `<section class="chat-view" aria-label="${escape(chat.title || 'Chat with mooody')}">
    <div class="transcript" role="log" aria-label="Conversation" aria-live="polite" aria-relevant="additions">${chat.messages.map(messageMarkup).join('')}</div>
    <div class="reply-status" role="status"><span>${escape(notice.text)}</span><button type="button" class="retry-button" data-action="retry" ${notice.retry ? '' : 'hidden'}>retry reply</button></div>
    <form class="composer">
      <label class="sr-only" for="message">Message mooody</label>
      <textarea id="message" name="message" rows="1" maxlength="2000" placeholder="say something..." required ${busy ? 'disabled' : ''}></textarea>
      <button type="${busy ? 'button' : 'submit'}" class="send-button" ${busy ? 'data-action="stop"' : ''} aria-label="${busy ? 'Stop reply' : 'Send message'}" ${busy ? '' : 'disabled'}>${composerButton(busy)}</button>
    </form>
  </section>`;
}

function activeChat() {
  return state.conversations.find(chat => chat.id === state.activeId);
}

function chatIdFromLocation() {
  if (!window.location.hash.startsWith('#chat=')) return null;
  try {
    const id = decodeURIComponent(window.location.hash.slice(6));
    return state.conversations.some(chat => chat.id === id) ? id : null;
  } catch {
    return null;
  }
}

function viewUrl(chatId) {
  const url = new URL(window.location.href);
  url.hash = chatId ? `chat=${encodeURIComponent(chatId)}` : '';
  return url;
}

function writeHistory(chatId, replace = false) {
  window.history[replace ? 'replaceState' : 'pushState'](
    { app: 'mooody', chatId }, '', viewUrl(chatId)
  );
}

function navigate(chatId, changedChat = null) {
  if (pending) stopReply();
  state.activeId = chatId;
  if (!chatId) state.mood = [...NEUTRAL];
  persist(changedChat);
  if (window.location.href !== viewUrl(chatId).href) writeHistory(chatId);
  render({ focusMessage: Boolean(chatId) });
  if (!chatId) app.querySelector('.brand').focus({ preventScroll: true });
}

function saveChat(chat) {
  state.conversations = state.conversations.map(item => item.id === chat.id ? chat : item);
  persist(chat);
}

function refreshReplyUI(chat, { focus = false } = {}) {
  if (state.activeId !== chat.id) return;
  const notice = noticeFor(chat);
  const textarea = app.querySelector('textarea');
  const button = app.querySelector('.send-button');
  const status = app.querySelector('.reply-status');
  if (!textarea || !button || !status) return;
  status.querySelector('span').textContent = notice.text;
  status.querySelector('.retry-button').hidden = !notice.retry;
  textarea.disabled = Boolean(notice.busy);
  button.type = notice.busy ? 'button' : 'submit';
  button.innerHTML = composerButton(notice.busy);
  button.setAttribute('aria-label', notice.busy ? 'Stop reply' : 'Send message');
  if (notice.busy) button.dataset.action = 'stop';
  else delete button.dataset.action;
  button.disabled = !notice.busy && !textarea.value.trim();
  if (focus && !notice.busy) textarea.focus({ preventScroll: true });
}

function refreshTranscript(chat) {
  if (state.activeId !== chat.id) return;
  app.querySelector('.transcript').innerHTML = chat.messages.map(messageMarkup).join('');
}

function stopReply() {
  const request = pending;
  if (!request) return;
  request.stopped = true;
  request.controller.abort();
  clearTimeout(request.saveTimer);
  clearTimeout(request.loadingTimer);
  const message = request.chat.messages.at(-1);
  if (message.text) message.status = 'interrupted';
  else request.chat.messages.pop();
  notices.set(request.chat.id, { text: 'Reply stopped.', retry: true });
  pending = null;
  saveChat(request.chat);
  refreshTranscript(request.chat);
  refreshReplyUI(request.chat, { focus: true });
}

async function beginReply(chat) {
  if (pending) return;
  const request = {
    chat: { ...chat, messages: [...chat.messages, { who: 'mooody', text: '', status: 'streaming' }] },
    controller: new AbortController(),
    status: 'Starting mooody. The first reply may take a few minutes.',
    stopped: false,
    received: false
  };
  pending = request;
  notices.delete(chat.id);
  saveChat(request.chat);
  refreshTranscript(request.chat);
  refreshReplyUI(request.chat);
  request.loadingTimer = setTimeout(() => {
    if (pending !== request || request.received) return;
    request.status = 'Loading the model. Your first reply can take a few minutes.';
    refreshReplyUI(request.chat);
  }, 25000);
  try {
    const result = await streamChat({
      messages: requestMessages(chat, capabilities.max_messages), mood: [...chat.mood]
    }, {
      signal: request.controller.signal,
      onEvent(event) {
        if (request.stopped) return;
        if (event.type === 'status' && !request.received && typeof event.message === 'string') {
          request.status = event.message;
          refreshReplyUI(request.chat);
        }
        if (event.type !== 'token' || !event.text) return;
        request.received = true;
        clearTimeout(request.loadingTimer);
        request.status = 'Replying…';
        request.chat.messages.at(-1).text += event.text;
        if (state.activeId === request.chat.id) {
          app.querySelector('.transcript .message:last-child .message-text').textContent = request.chat.messages.at(-1).text;
          refreshReplyUI(request.chat);
          // Follow output while the reader is already near the bottom.
          if (window.innerHeight + window.scrollY >= document.documentElement.scrollHeight - 180) {
            app.querySelector('.composer').scrollIntoView({ block: 'end', behavior: 'instant' });
          }
        }
        if (!request.saveTimer) request.saveTimer = setTimeout(() => {
          request.saveTimer = null;
          if (!request.stopped) saveChat(request.chat);
        }, 250);
      }
    });
    if (request.stopped) return;
    if (!request.chat.messages.at(-1).text.trim()) {
      throw new ChatError('Mooody returned an empty reply. Please try again.', 'empty_reply');
    }
    delete request.chat.messages.at(-1).status;
    notices.set(chat.id, { text: result.finish_reason === 'length' ? 'Reply reached its length limit.' : '' });
  } catch (error) {
    if (request.stopped) return;
    const message = request.chat.messages.at(-1);
    if (message.text) message.status = 'interrupted';
    else request.chat.messages.pop();
    notices.set(chat.id, {
      text: error.name === 'AbortError' ? 'Reply stopped.' : error.message || 'The reply could not finish. Please try again.',
      retry: true
    });
  } finally {
    clearTimeout(request.saveTimer);
    clearTimeout(request.loadingTimer);
    if (!request.stopped) {
      if (pending === request) pending = null;
      saveChat(request.chat);
      refreshTranscript(request.chat);
      refreshReplyUI(request.chat, { focus: true });
    }
  }
}

function render({ focusMessage = false } = {}) {
  const chat = activeChat();
  document.body.classList.toggle('is-chat', Boolean(chat));
  document.title = chat?.title ? `${chat.title} · mooody` : 'mooody';
  updateTabIcon(chat?.mood);
  app.innerHTML = `<header class="topbar"><button type="button" class="brand" data-action="home" aria-label="Mooody home">${logo(chat?.mood)}<span>mooody</span></button>${chat?.title ? `<h1 class="chat-title">${escape(chat.title)}</h1>` : ''}</header><main id="main">${chat ? conversation(chat) : setup()}</main>`;
  if (focusMessage) app.querySelector('textarea')?.focus({ preventScroll: true });
  window.scrollTo({ top: 0, behavior: 'instant' });
}

app.addEventListener('change', event => {
  const input = event.target.closest('input[data-axis]');
  if (!input) return;
  state.mood[Number(input.dataset.axis)] = Number(input.value);
  const chart = app.querySelector('.mood-graph');
  chart.querySelector('.graph-profile').setAttribute('points', profilePoints(state.mood));
  chart.setAttribute('aria-label', `Mood profile: ${describeMood(state.mood)}`);
  persist();
});

app.addEventListener('click', event => {
  const button = event.target.closest('button');
  if (!button) return;
  if (button.dataset.action === 'home') {
    navigate(null);
  } else if (button.dataset.action === 'start') {
    const chat = createChat(state.mood);
    state.conversations.unshift(chat);
    navigate(chat.id, chat);
  } else if (button.dataset.chatId) {
    navigate(button.dataset.chatId);
  } else if (button.dataset.action === 'stop') {
    stopReply();
  } else if (button.dataset.action === 'retry') {
    const chat = retryChat(activeChat());
    if (chat) beginReply(chat);
  }
});

app.addEventListener('input', event => {
  const textarea = event.target.closest('textarea');
  if (!textarea || pending) return;
  textarea.style.height = 'auto';
  textarea.style.height = `${Math.min(textarea.scrollHeight, 180)}px`;
  app.querySelector('.send-button').disabled = !textarea.value.trim();
});

app.addEventListener('keydown', event => {
  if (event.target.matches('textarea') && event.key === 'Enter' && !event.shiftKey && !event.isComposing) {
    event.preventDefault();
    if (!pending && event.target.value.trim()) event.target.closest('form').requestSubmit();
  }
});

app.addEventListener('submit', event => {
  if (!event.target.matches('.composer')) return;
  event.preventDefault();
  if (pending) return;
  const localChat = activeChat();
  const chat = readStoredState()?.conversations.find(item => item.id === state.activeId) ?? localChat;
  const textarea = event.target.querySelector('textarea');
  if (!chat || !textarea.value.trim()) return;
  const updated = appendMessage(chat, textarea.value);
  state.conversations = state.conversations.map(item => item.id === chat.id ? updated : item);
  persist(updated);
  refreshTranscript(updated);
  let heading = app.querySelector('.topbar .chat-title');
  if (!heading) {
    heading = document.createElement('h1');
    heading.className = 'chat-title';
    app.querySelector('.topbar').append(heading);
  }
  heading.textContent = updated.title;
  app.querySelector('.chat-view').setAttribute('aria-label', updated.title);
  document.title = `${updated.title} · mooody`;
  textarea.value = '';
  textarea.style.height = 'auto';
  app.querySelector('.send-button').disabled = true;
  event.target.scrollIntoView({ block: 'end', behavior: 'instant' });
  beginReply(updated);
});

window.addEventListener('storage', event => {
  if (event.key !== STORAGE_KEY && event.key !== null) return;
  if (pending) return;
  if (app.querySelector('textarea')?.value.trim()) return;
  state.conversations = decodeState(event.newValue).conversations;
  if (!activeChat()) {
    state.activeId = null;
    writeHistory(null, true);
  }
  render();
});

window.addEventListener('popstate', () => {
  if (pending) stopReply();
  const saved = readStoredState();
  if (saved) state.conversations = saved.conversations;
  state.activeId = chatIdFromLocation();
  if (!state.activeId) state.mood = [...NEUTRAL];
  if (window.location.href !== viewUrl(state.activeId).href) writeHistory(state.activeId, true);
  persist();
  render();
  app.querySelector('.brand').focus({ preventScroll: true });
});

window.addEventListener('pagehide', () => { if (pending) stopReply(); });

state.activeId = chatIdFromLocation();
if (state.activeId && window.history.state?.app !== 'mooody') {
  // A newly opened chat link still has a home view to return to with Back.
  const chatId = state.activeId;
  writeHistory(null, true);
  writeHistory(chatId);
} else {
  writeHistory(state.activeId, true);
}
render();

fetch('/api/config', { headers: { Accept: 'application/json' } })
  .then(response => response.ok ? response.json() : null)
  .then(config => {
    if (!config || typeof config !== 'object') return;
    capabilities = {
      ...capabilities,
      mood_vectors_available: config.mood_vectors_available === true,
      steering_available: config.steering_available === true,
      mood_vectors_source: typeof config.mood_vectors_source === 'string' ? config.mood_vectors_source : null,
      mood_vectors_validated: config.mood_vectors_validated === true,
      max_messages: Number.isInteger(config.max_messages) && config.max_messages > 0 ? config.max_messages : 32
    };
  })
  .catch(() => { /* Chat requests show a useful connection error if the API is unavailable. */ });
