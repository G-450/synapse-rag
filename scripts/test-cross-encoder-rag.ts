import 'dotenv/config';
import { retrieveChunks } from '../src/lib/rag';

async function main() {
  const query = "What is the definition of Intervening Event?";

  console.log('============================================================');
  console.log(`QUERY: "${query}"`);
  console.log('============================================================\n');

  console.log('[1/2] Retrieving context chunks from vector database (NO RERANK)...');
  const startRetrieveNoRerank = Date.now();
  
  const chunksNoRerank = await retrieveChunks(query, { limit: 5, rerank: false });
  const msNoRerank = Date.now() - startRetrieveNoRerank;
  
  console.log(`[✓] Retrieved ${chunksNoRerank.length} chunks in ${msNoRerank}ms.`);
  chunksNoRerank.forEach((c, i) => {
    console.log(`  - Chunk ${i+1}: ${c.similarity.toFixed(4)} similarity | ${c.content.substring(0, 50)}...`);
  });

  console.log('\n[2/2] Retrieving context chunks from vector database (WITH RERANK)...');
  const startRetrieveRerank = Date.now();
  
  const chunksRerank = await retrieveChunks(query, { limit: 5, rerank: true });
  const msRerank = Date.now() - startRetrieveRerank;
  
  console.log(`[✓] Retrieved & Reranked ${chunksRerank.length} chunks in ${msRerank}ms.`);
  chunksRerank.forEach((c, i) => {
    console.log(`  - Chunk ${i+1}: ${c.cross_score?.toFixed(4)} cross_score | ${c.content.substring(0, 50)}...`);
  });
}

main().catch(console.error);
