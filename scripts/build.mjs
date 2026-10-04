import { cp, mkdir, rm } from 'node:fs/promises';
import { fileURLToPath } from 'node:url';

const projectRoot = new URL('../', import.meta.url);
const output = new URL('dist/', projectRoot);
await rm(output, { recursive: true, force: true });
await mkdir(output, { recursive: true });
for (const filename of ['index.html', 'favicon.svg', 'styles.css', 'mobile.html', 'mobile.css', 'src']) {
  await cp(new URL(filename, projectRoot), new URL(filename, output), { recursive: true });
}
console.log(`Built static app in ${fileURLToPath(output)}`);
