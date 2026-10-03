import { createServer } from 'node:http';
import { readFile, stat } from 'node:fs/promises';
import { resolve, extname, sep } from 'node:path';
import { fileURLToPath } from 'node:url';

const args = process.argv.slice(2);
const option = (name, fallback) => {
  const index = args.indexOf(name);
  return index >= 0 && args[index + 1] ? args[index + 1] : fallback;
};
const projectRoot = fileURLToPath(new URL('../', import.meta.url));
const staticRoot = resolve(projectRoot, option('--dir', '.'));
const port = Number(option('--port', process.env.PORT || '5173'));
const host = option('--host', '127.0.0.1');
const types = { '.html': 'text/html; charset=utf-8', '.css': 'text/css; charset=utf-8', '.js': 'text/javascript; charset=utf-8', '.svg': 'image/svg+xml' };

if (!Number.isInteger(port) || port < 0 || port > 65535) throw new Error('Choose a port between 0 and 65535.');
await stat(resolve(staticRoot, 'index.html')).catch(() => {
  throw new Error('No index.html found. Run npm run build before previewing dist.');
});

const server = createServer(async (request, response) => {
  if (!['GET', 'HEAD'].includes(request.method)) {
    response.writeHead(405, { Allow: 'GET, HEAD' }).end();
    return;
  }
  try {
    const pathname = decodeURIComponent(new URL(request.url, 'http://localhost').pathname);
    const relative = pathname === '/' ? '/index.html' : pathname;
    const filename = resolve(staticRoot, '.' + relative);
    if (!filename.startsWith(staticRoot + sep) || relative.split('/').some(part => part.startsWith('.'))
      || !['.html', '.css', '.js', '.svg'].includes(extname(filename))) {
      response.writeHead(404).end('Not found');
      return;
    }
    const content = await readFile(filename);
    response.writeHead(200, { 'Content-Type': types[extname(filename)], 'Cache-Control': 'no-store', 'X-Content-Type-Options': 'nosniff' });
    response.end(request.method === 'HEAD' ? undefined : content);
  } catch {
    response.writeHead(404).end('Not found');
  }
});

server.on('error', error => {
  console.error(error.code === 'EADDRINUSE' ? `Port ${port} is busy. Try npm run dev -- --port ${port + 1}.` : error.message);
  process.exitCode = 1;
});
server.listen(port, host, () => {
  const boundPort = server.address().port;
  console.log(`mooody → http://${host === '0.0.0.0' ? 'localhost' : host}:${boundPort}`);
});
for (const signal of ['SIGINT', 'SIGTERM']) process.on(signal, () => server.close());
