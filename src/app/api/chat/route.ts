/**
 * Chat API proxy — forwards to the Python FastAPI backend.
 * All RAG logic runs in Python; this route just proxies the streaming response.
 */

export const maxDuration = 60;

const PYTHON_BACKEND = process.env.PYTHON_BACKEND_URL ?? 'http://localhost:8000';

export async function POST(req: Request) {
  const body = await req.text();
  console.log("NEXTJS SENDING BODY:", body);

  const upstream = await fetch(`${PYTHON_BACKEND}/api/python/chat`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body,
  });

  // Stream the response body directly back to the client
  return new Response(upstream.body, {
    status: upstream.status,
    headers: {
      'Content-Type': upstream.headers.get('Content-Type') ?? 'text/plain; charset=utf-8',
      'X-Vercel-AI-Data-Stream': upstream.headers.get('X-Vercel-AI-Data-Stream') ?? 'v1',
      'Cache-Control': 'no-cache',
    },
  });
}
