const backend = process.env.PYTHON_BACKEND_URL ?? 'http://localhost:8000';

export {};

async function checkEndpoint(path: string) {
  const response = await fetch(`${backend}${path}`);
  if (!response.ok) {
    throw new Error(`${path} returned HTTP ${response.status}`);
  }
  return response.json();
}

async function main() {
  try {
    const health = await checkEndpoint('/health');
    const documents = await checkEndpoint('/api/python/documents');
    const count = Array.isArray(documents.documents) ? documents.documents.length : 0;
    console.log(`Backend status: ${health.status ?? 'unknown'}`);
    console.log(`Indexed documents: ${count}`);
  } catch (error) {
    console.error(`Database check failed: ${error instanceof Error ? error.message : String(error)}`);
    process.exitCode = 1;
  }
}

void main();
