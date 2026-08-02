import 'dotenv/config';
import fs from 'fs';
import path from 'path';
import { generateEmbedding } from '../src/lib/embeddings';
import { Pool, neonConfig } from '@neondatabase/serverless';
import ws from 'ws';

neonConfig.webSocketConstructor = ws;

async function main() {
  const pool = new Pool({ connectionString: process.env.DATABASE_URL });
  
  console.log('Starting ingestion pipeline...');
  const dataPath = path.join(__dirname, '../data/legalbench.json');
  if (!fs.existsSync(dataPath)) {
    console.error('Dataset not found at', dataPath);
    return;
  }
  
  const qaPairs = JSON.parse(fs.readFileSync(dataPath, 'utf-8'));
  console.log(`Loaded ${qaPairs.length} QA pairs from LegalBench-RAG. Process...`);

  console.log('Altering Chunk table to accept 384-dimensional vectors (Transformers.js)...');
  await pool.query(`ALTER TABLE "Chunk" ALTER COLUMN "embedding" TYPE vector(384);`);

  // Group snippets by document (corpus_file)
  const docsMap = new Map<string, Set<string>>();
  
  for (const pair of qaPairs) {
    const meta = pair.metadata;
    const file = meta.corpus_file;
    if (!docsMap.has(file)) docsMap.set(file, new Set());
    
    // Some QA pairs have multiple snippets
    for (const snippet of meta.snippets) {
       docsMap.get(file)!.add(snippet.answer.trim());
    }
  }
  
  console.log(`Found ${docsMap.size} unique contracts in the dataset sample.`);

  let totalChunks = 0;
  for (const [corpusFile, uniqueSnippets] of docsMap.entries()) {
    console.log(`Processing Document: ${corpusFile}`);
    
    // 1. Create or Find Document
    const documentId = crypto.randomUUID();
    await pool.query(`
      INSERT INTO "Document" ("id", "filename", "title", "source_corpus", "createdAt") 
      VALUES ($1, $2, $3, $4, NOW())
    `, [documentId, corpusFile.split('/').pop() || corpusFile, corpusFile, 'legalbench-rag']);

    // 2. Generate Embeddings & Save Chunks
    for (const text of uniqueSnippets) {
      if (!text) continue;
      
      console.log(` Generating embedding for snippet (${text.length} chars)...`);
      const embeddingArray = await generateEmbedding(text);
      
      await pool.query(`
        INSERT INTO "Chunk" ("id", "document_id", "content", "embedding", "char_start", "char_end") 
        VALUES (gen_random_uuid(), $1, $2, $3::vector, 0, 0)
      `, [documentId, text, JSON.stringify(embeddingArray)]);
      totalChunks++;
    }
  }
  
  console.log(`\n✅ Ingestion Complete! Inserted ${docsMap.size} Documents and ${totalChunks} Chunks.`);
}

main().catch(e => {
  console.error(e);
  process.exit(1);
}).finally(() => {
  process.exit(0);
});
