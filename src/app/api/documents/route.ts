import { NextResponse } from 'next/server';
import { neon } from '@neondatabase/serverless';

export async function GET() {
  try {
    const sql = neon(process.env.DATABASE_URL!);

    const documents = await sql`
      SELECT 
        d.id, 
        d.filename, 
        d.title, 
        d.source_corpus, 
        d.category,
        d."createdAt",
        COUNT(c.id)::int as chunk_count
      FROM "Document" d
      LEFT JOIN "Chunk" c ON c.document_id = d.id
      GROUP BY d.id, d.filename, d.title, d.source_corpus, d.category, d."createdAt"
      ORDER BY d.source_corpus, d.filename
    `;

    // Group by corpus for the sidebar
    const grouped: Record<string, typeof documents> = {};
    for (const doc of documents) {
      const corpus = doc.source_corpus;
      if (!grouped[corpus]) grouped[corpus] = [];
      grouped[corpus].push(doc);
    }

    return NextResponse.json({ documents, grouped });
  } catch (error) {
    console.error('[Documents API Error]', error);
    return NextResponse.json(
      { error: 'Failed to fetch documents' },
      { status: 500 }
    );
  }
}
