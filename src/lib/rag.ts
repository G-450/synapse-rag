import { neon } from '@neondatabase/serverless';
import { generateEmbedding } from './embeddings';
import { rankChunks } from './cross-encoder';

export interface RetrievedChunk {
  id: string;
  content: string;
  document_id: string;
  similarity: number;
  filename?: string;
  source_corpus?: string;
  cross_score?: number;
}

export interface DocumentWithChunks {
  document_id: string;
  filename: string;
  source_corpus: string;
  chunks: RetrievedChunk[];
}

const getSql = () => neon(process.env.DATABASE_URL!);

/**
 * Retrieve the most relevant chunks for a query using cosine similarity.
 * Optionally scoped to a single document.
 */
export async function retrieveChunks(
  query: string,
  options: { documentId?: string; limit?: number; rerank?: boolean } = {}
): Promise<RetrievedChunk[]> {
  const { documentId, limit = 5, rerank = false } = options;
  const fetchLimit = rerank ? Math.max(limit * 4, 20) : limit;
  const embeddingArray = await generateEmbedding(query);
  const embeddingString = JSON.stringify(embeddingArray);
  const sql = getSql();

  let chunks: RetrievedChunk[];

  if (documentId) {
    // Single-document retrieval
    chunks = await sql`
      SELECT c.id, c.content, c.document_id,
             1 - (c.embedding <=> ${embeddingString}::vector) as similarity,
             d.filename, d.source_corpus
      FROM "Chunk" c
      JOIN "Document" d ON d.id = c.document_id
      WHERE c.document_id = ${documentId}
        AND c.embedding IS NOT NULL
      ORDER BY c.embedding <=> ${embeddingString}::vector
      LIMIT ${fetchLimit}
    `;
  } else {
    // Global retrieval across all documents
    chunks = await sql`
      SELECT c.id, c.content, c.document_id,
             1 - (c.embedding <=> ${embeddingString}::vector) as similarity,
             d.filename, d.source_corpus
      FROM "Chunk" c
      JOIN "Document" d ON d.id = c.document_id
      WHERE c.embedding IS NOT NULL
      ORDER BY c.embedding <=> ${embeddingString}::vector
      LIMIT ${fetchLimit}
    `;
  }

  if (rerank && chunks.length > 0) {
    chunks = await rankChunks(query, chunks);
    chunks = chunks.slice(0, limit);
  }

  return chunks;
}

/**
 * Fan-out multi-document retrieval.
 * 1. Find the top documents by best chunk similarity.
 * 2. Retrieve the best chunks per document independently.
 * Returns chunks grouped by document.
 */
export async function fanOutRetrieve(
  query: string,
  options: { topDocs?: number; chunksPerDoc?: number } = {}
): Promise<DocumentWithChunks[]> {
  const { topDocs = 3, chunksPerDoc = 3 } = options;
  const embeddingArray = await generateEmbedding(query);
  const embeddingString = JSON.stringify(embeddingArray);
  const sql = getSql();

  // Step 1: Find the top documents by their best-matching chunk
  const topDocuments = await sql`
    SELECT DISTINCT ON (c.document_id) 
      c.document_id, d.filename, d.source_corpus,
      1 - (c.embedding <=> ${embeddingString}::vector) as best_similarity
    FROM "Chunk" c
    JOIN "Document" d ON d.id = c.document_id
    WHERE c.embedding IS NOT NULL
    ORDER BY c.document_id, c.embedding <=> ${embeddingString}::vector
  `;

  // Sort by best similarity and take top N
  const sortedDocs = topDocuments
    .sort((a: any, b: any) => b.best_similarity - a.best_similarity)
    .slice(0, topDocs);

  // Step 2: For each document, retrieve the best chunks
  const results: DocumentWithChunks[] = [];
  for (const doc of sortedDocs) {
    const docChunks = await sql`
      SELECT c.id, c.content, c.document_id,
             1 - (c.embedding <=> ${embeddingString}::vector) as similarity,
             d.filename, d.source_corpus
      FROM "Chunk" c
      JOIN "Document" d ON d.id = c.document_id
      WHERE c.document_id = ${doc.document_id}
        AND c.embedding IS NOT NULL
      ORDER BY c.embedding <=> ${embeddingString}::vector
      LIMIT ${chunksPerDoc}
    `;

    results.push({
      document_id: doc.document_id,
      filename: doc.filename,
      source_corpus: doc.source_corpus,
      chunks: docChunks,
    });
  }

  return results;
}

/**
 * Format retrieved chunks into a context string for the LLM prompt.
 */
export function formatContext(chunks: RetrievedChunk[]): string {
  return chunks
    .map(
      (chunk, i) =>
        `[Source: ${chunk.filename || chunk.document_id} | Relevance: ${(chunk.similarity * 100).toFixed(1)}%]\n${chunk.content}`
    )
    .join('\n\n---\n\n');
}

/**
 * Format fan-out results grouped by document.
 */
export function formatFanOutContext(docGroups: DocumentWithChunks[]): string {
  return docGroups
    .map((group) => {
      const header = `=== Document: ${group.filename} (${group.source_corpus}) ===`;
      const chunks = group.chunks
        .map(
          (chunk, i) =>
            `[Snippet ${i + 1} | Relevance: ${(chunk.similarity * 100).toFixed(1)}%]\n${chunk.content}`
        )
        .join('\n\n');
      return `${header}\n\n${chunks}`;
    })
    .join('\n\n' + '='.repeat(60) + '\n\n');
}
