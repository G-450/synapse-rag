import { NextResponse } from 'next/server';

export const maxDuration = 120;

const PYTHON_BACKEND = process.env.PYTHON_BACKEND_URL ?? 'http://localhost:8000';

export async function POST(request: Request) {
  try {
    const formData = await request.formData();
    const upstream = await fetch(`${PYTHON_BACKEND}/api/python/upload`, {
      method: 'POST',
      body: formData,
    });
    const data = await upstream.json();
    return NextResponse.json(data, { status: upstream.status });
  } catch (error) {
    console.error('[Upload Proxy Error]', error);
    return NextResponse.json({ detail: 'Python backend unavailable' }, { status: 502 });
  }
}
