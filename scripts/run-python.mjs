import { existsSync } from 'node:fs';
import { spawnSync } from 'node:child_process';
import path from 'node:path';

const candidates = process.platform === 'win32'
  ? ['python/venv/Scripts/python.exe']
  : ['python/venv/bin/python', 'python/venv/bin/python3'];
const python = candidates.find(existsSync);

if (!python) {
  console.error('Project Python virtual environment not found. Create python/venv and install python/requirements.txt.');
  process.exit(1);
}

const [, , ...args] = process.argv;
if (args.length === 0) {
  console.error('Usage: node scripts/run-python.mjs <script-or-python-args>');
  process.exit(1);
}

const result = spawnSync(path.resolve(python), args, { stdio: 'inherit' });
if (result.error) {
  console.error(`Unable to start project Python: ${result.error.message}`);
  process.exit(1);
}
process.exit(result.status ?? 1);
