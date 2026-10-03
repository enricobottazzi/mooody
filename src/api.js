export class ChatError extends Error {
  constructor(message, code = 'connection_error') {
    super(message);
    this.name = 'ChatError';
    this.code = code;
  }
}

function readEvent(frame) {
  let type = 'message';
  const data = [];
  for (const line of frame.split(/\r\n|\n|\r/)) {
    if (line.startsWith('event:')) type = line.slice(6).trim();
    if (line.startsWith('data:')) data.push(line.slice(5).replace(/^ /, ''));
  }
  if (!data.length || !['status', 'meta', 'token', 'done', 'error'].includes(type)) return null;
  try {
    const value = JSON.parse(data.join('\n'));
    if (!value || typeof value !== 'object' || Array.isArray(value)) throw new Error();
    return { ...value, type };
  } catch {
    throw new ChatError('The reply could not be read. Please try again.', 'invalid_stream');
  }
}

export async function streamChat(payload, { signal, onEvent = () => {}, fetchImpl = globalThis.fetch } = {}) {
  let response;
  try {
    response = await fetchImpl('/api/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', Accept: 'text/event-stream' },
      body: JSON.stringify(payload),
      signal
    });
  } catch (error) {
    if (signal?.aborted || error.name === 'AbortError') throw error;
    throw new ChatError('Could not connect to mooody. Check your connection and try again.');
  }
  if (!response.ok) {
    let body;
    try { body = await response.json(); } catch { /* A proxy may return HTML. */ }
    const message = typeof body?.message === 'string' ? body.message
      : typeof body?.detail === 'string' ? body.detail
      : response.status === 429 ? 'Mooody is busy. Please try again in a moment.'
      : 'Mooody could not start this reply. Please try again.';
    throw new ChatError(message, body?.code ?? `http_${response.status}`);
  }
  if (!response.body || !response.headers.get('content-type')?.includes('text/event-stream')) {
    throw new ChatError('Mooody returned an unreadable reply. Please try again.', 'invalid_stream');
  }
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = '';
  const dispatch = frame => {
    const event = readEvent(frame);
    if (!event) return null;
    if (event.type === 'error') {
      throw new ChatError(typeof event.message === 'string' ? event.message : 'The reply stopped. Please try again.', event.code);
    }
    if (event.type === 'token' && typeof event.text !== 'string') {
      throw new ChatError('The reply could not be read. Please try again.', 'invalid_stream');
    }
    onEvent(event);
    if (event.type === 'done') {
      return event;
    }
    return null;
  };
  try {
    while (true) {
      if (signal?.aborted) throw new DOMException('The reply was stopped.', 'AbortError');
      const { value, done } = await reader.read();
      buffer += done ? decoder.decode() : decoder.decode(value, { stream: true });
      let boundary;
      while ((boundary = /\r\n\r\n|\n\n|\r\r/.exec(buffer))) {
        const frame = buffer.slice(0, boundary.index);
        buffer = buffer.slice(boundary.index + boundary[0].length);
        const result = dispatch(frame);
        if (result) return result;
      }
      if (done) {
        if (buffer.trim()) {
          const result = dispatch(buffer);
          if (result) return result;
        }
        throw new ChatError('The connection ended before the reply finished. You can retry it.', 'interrupted_stream');
      }
    }
  } catch (error) {
    if (signal?.aborted || error.name === 'AbortError' || error instanceof ChatError) throw error;
    throw new ChatError('The connection was interrupted. You can retry the reply.', 'interrupted_stream');
  } finally {
    // Stop consuming after completion, failure, or navigation away from the chat.
    try { await reader.cancel(); } catch { /* The fetch may already be aborted. */ }
    reader.releaseLock();
  }
}
