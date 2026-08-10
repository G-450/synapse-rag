/**
 * Synapse RAG — Retrieval Evaluation Script
 * 
 * Evaluates retrieval quality using the LegalBench-RAG dataset.
 * Metrics: Precision, Recall, F1 at document level, plus Document Retrieval Match (DRM).
 * 
 * Usage: npm run evaluate
 */

import 'dotenv/config';
import fs from 'fs';
import path from 'path';
import { neon } from '@neondatabase/serverless';
import { generateEmbedding } from '../src/lib/embeddings';

interface QAPair {
  id: string;
  text: string;
  metadata: {
    id: string;
    query: string;
    answer: string;
    corpus_file: string;
    num_snippets: number;
    snippets: Array<{
      answer: string;
      file_path: string;
      span: [number, number];
    }>;
    dataset_source: string;
  };
}

interface EvalResult {
  query_id: string;
  query: string;
  expected_corpus_file: string;
  retrieved_files: string[];
  document_match: boolean;
  content_overlap_score: number;
}

async function run() {
  console.log('='.repeat(60));
  console.log('  Synapse RAG — Retrieval Evaluation');
  console.log('='.repeat(60));

  // 1. Load dataset
  const dataPath = path.join(__dirname, '../data/legalbench.json');
  if (!fs.existsSync(dataPath)) {
    console.error('Dataset not found at', dataPath);
    process.exit(1);
  }

  const qaPairs: QAPair[] = JSON.parse(fs.readFileSync(dataPath, 'utf-8'));
  console.log(`\nLoaded ${qaPairs.length} QA pairs for evaluation.\n`);

  const sql = neon(process.env.DATABASE_URL!);
  const results: EvalResult[] = [];

  let documentsMatched = 0;
  let totalOverlap = 0;
  let totalPrecision = 0;
  let totalRecall = 0;

  for (let i = 0; i < qaPairs.length; i++) {
    const qa = qaPairs[i];
    const query = qa.metadata.query;
    const expectedFile = qa.metadata.corpus_file;
    const expectedAnswer = qa.metadata.answer;

    process.stdout.write(`[${i + 1}/${qaPairs.length}] Evaluating: ${qa.id}...`);

    try {
      // Generate query embedding
      const embeddingArray = await generateEmbedding(query);
      const embeddingString = JSON.stringify(embeddingArray);

      // Retrieve top-5 chunks
      const chunks = await sql`
        SELECT c.id, c.content, c.document_id, d.filename, d.source_corpus,
               1 - (c.embedding <=> ${embeddingString}::vector) as similarity
        FROM "Chunk" c
        JOIN "Document" d ON d.id = c.document_id
        WHERE c.embedding IS NOT NULL
        ORDER BY c.embedding <=> ${embeddingString}::vector
        LIMIT 5
      `;

      // Document Retrieval Match: Did we find the right document?
      const retrievedFiles = [...new Set(chunks.map((c: any) => c.filename))];
      const expectedFilename = expectedFile.split('/').pop() || expectedFile;
      const docMatch = retrievedFiles.some(
        (f: string) => f === expectedFilename || expectedFilename.includes(f.replace('.txt', ''))
      );
      if (docMatch) documentsMatched++;

      // Content overlap: How much of the expected answer appears in retrieved chunks?
      const allRetrievedText = chunks.map((c: any) => c.content).join(' ');
      const overlapScore = computeContentOverlap(expectedAnswer, allRetrievedText);
      totalOverlap += overlapScore;

      // Precision: What fraction of retrieved text is relevant?
      const precision = computeContentOverlap(allRetrievedText, expectedAnswer);
      totalPrecision += precision;

      // Recall: What fraction of the expected answer was retrieved?
      const recall = overlapScore;
      totalRecall += recall;

      results.push({
        query_id: qa.id,
        query: query.substring(0, 80),
        expected_corpus_file: expectedFilename,
        retrieved_files: retrievedFiles as string[],
        document_match: docMatch,
        content_overlap_score: overlapScore,
      });

      console.log(` ${docMatch ? '✓' : '✗'} DRM | Overlap: ${(overlapScore * 100).toFixed(1)}%`);
    } catch (e: any) {
      console.log(` ERROR: ${e.message}`);
    }
  }

  // Compute aggregate metrics
  const n = results.length;
  const drm = documentsMatched / n;
  const avgPrecision = totalPrecision / n;
  const avgRecall = totalRecall / n;
  const f1 = avgPrecision + avgRecall > 0
    ? (2 * avgPrecision * avgRecall) / (avgPrecision + avgRecall)
    : 0;

  console.log('\n' + '='.repeat(60));
  console.log('  EVALUATION RESULTS');
  console.log('='.repeat(60));
  console.log(`  Total QA pairs evaluated:   ${n}`);
  console.log(`  Document Retrieval Match:    ${(drm * 100).toFixed(1)}% (${documentsMatched}/${n})`);
  console.log(`  Avg Precision:              ${(avgPrecision * 100).toFixed(1)}%`);
  console.log(`  Avg Recall:                 ${(avgRecall * 100).toFixed(1)}%`);
  console.log(`  F1 Score:                   ${(f1 * 100).toFixed(1)}%`);
  console.log('='.repeat(60));

  // Save results to file
  const outputPath = path.join(__dirname, '../data/eval_results.json');
  fs.writeFileSync(
    outputPath,
    JSON.stringify(
      {
        timestamp: new Date().toISOString(),
        config: {
          embedding_model: 'Xenova/all-MiniLM-L6-v2',
          dimensions: 384,
          top_k: 5,
          chunk_size: 1000,
          chunk_overlap: 200,
        },
        metrics: {
          document_retrieval_match: drm,
          precision: avgPrecision,
          recall: avgRecall,
          f1: f1,
        },
        results,
      },
      null,
      2
    )
  );
  console.log(`\nDetailed results saved to: ${outputPath}`);
}

/**
 * Compute word-level overlap between text A and reference text B.
 * Returns the fraction of words in A that also appear in B.
 */
function computeContentOverlap(textA: string, textB: string): number {
  const wordsA = new Set(
    textA.toLowerCase().split(/\s+/).filter((w) => w.length > 2)
  );
  const wordsB = new Set(
    textB.toLowerCase().split(/\s+/).filter((w) => w.length > 2)
  );

  if (wordsA.size === 0) return 0;

  let overlap = 0;
  for (const word of wordsA) {
    if (wordsB.has(word)) overlap++;
  }

  return overlap / wordsA.size;
}

run()
  .catch(console.error)
  .finally(() => process.exit(0));
