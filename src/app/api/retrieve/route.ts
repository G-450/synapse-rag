import { NextResponse } from 'next/server';
import { retrieveChunks } from '@/lib/rag';

export async function POST(request: Request) {
  try {
    const { query, documentId, limit = 5, rerank = true } = await request.json();

    if (!query) {
      return NextResponse.json({ error: 'Query is required' }, { status: 400 });
    }

    const chunks = await retrieveChunks(query, { documentId, limit, rerank });

    return NextResponse.json({ chunks });
  } catch (error) {
    console.error('[Retrieve API Error]', error);
    return NextResponse.json(
      { error: 'Internal Server Error' },
      { status: 500 }
    );
  }
}
