/**
 * Documents API proxy — forwards to the Python FastAPI backend.
 */

import { NextResponse } from 'next/server';

const PYTHON_BACKEND = process.env.PYTHON_BACKEND_URL ?? 'http://localhost:8001';

export async function GET() {
  try {
    const upstream = await fetch(`${PYTHON_BACKEND}/api/python/documents`);
    const data = await upstream.json();
    return NextResponse.json(data, { status: upstream.status });
  } catch (error) {
    console.error('[Documents Proxy Error]', error);
    return NextResponse.json({ error: 'Python backend unavailable' }, { status: 502 });
  }
}

export async function DELETE(request: Request) {
  try {
    const body = await request.text();
    const upstream = await fetch(`${PYTHON_BACKEND}/api/python/documents`, {
      method: 'DELETE',
      headers: { 'Content-Type': 'application/json' },
      body,
    });
    const data = await upstream.json();
    return NextResponse.json(data, { status: upstream.status });
  } catch (error) {
    console.error('[Documents Delete Proxy Error]', error);
    return NextResponse.json({ error: 'Python backend unavailable' }, { status: 502 });
  }
}
