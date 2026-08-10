import 'dotenv/config';
import { createGroq } from '@ai-sdk/groq';
import { generateText } from 'ai';
import { retrieveChunks, fanOutRetrieve, formatContext, formatFanOutContext } from '../src/lib/rag';

const groq = createGroq({
  apiKey: process.env.GROQ_API_KEY,
});

const SYSTEM_PROMPT = `You are Synapse RAG, an expert legal contract analyst.
You will be given relevant excerpts from legal contracts (the "Context") and a user question.

CRITICAL RULES:
1. Answer ONLY using information found in the Context. Do NOT use outside knowledge.
2. If the Context does not contain the information needed, respond with: "I cannot answer this question based on the provided contract excerpts."
3. Be precise. Quote exact clauses or passages when relevant, wrapping quotes in quotation marks.
4. When citing information, mention the source document name.
5. For multi-document queries, structure your response clearly by document.
6. Use professional legal analysis tone.`;

async function main() {
  const query = process.argv.slice(2).join(' ');

  if (!query) {
    console.log('Please provide a question to test.');
    console.log('Example: npx tsx scripts/test-rag.ts "What is the definition of Intervening Event?"');
    process.exit(1);
  }

  console.log('============================================================');
  console.log(`QUERY: "${query}"`);
  console.log('============================================================\n');

  console.log('[1/3] Retrieving context chunks from vector database...');
  const startRetrieve = Date.now();
  
  // Running the fan-out retrieval (multi-document mode)
  const docGroups = await fanOutRetrieve(query, {
    topDocs: 3,
    chunksPerDoc: 3,
  });

  const retrieveMs = Date.now() - startRetrieve;
  console.log(`[✓] Retrieved from ${docGroups.length} documents in ${retrieveMs}ms.\n`);

  // Log what chunks we actually found
  docGroups.forEach(group => {
    console.log(`  📄 ${group.filename} (${group.chunks.length} chunks)`);
    group.chunks.forEach((chunk, i) => {
      console.log(`     - Snippet ${i+1}: ${(chunk.similarity * 100).toFixed(1)}% match`);
    });
  });

  console.log('\n[2/3] Generating response with Groq (llama-3.1-8b-instant)...');
  
  const contextText = formatFanOutContext(docGroups);
  const augmentedSystem =
    SYSTEM_PROMPT +
    '\n\n=== CONTEXT (Retrieved Contract Excerpts) ===\n\n' +
    contextText;

  const startGen = Date.now();
  
  try {
    const { text } = await generateText({
      model: groq('llama-3.1-8b-instant'),
      system: augmentedSystem,
      prompt: query,
    });
    
    const genMs = Date.now() - startGen;
    console.log(`[✓] Response generated in ${genMs}ms.\n`);
    
    console.log('====================== RESPONSE ============================');
    console.log(text);
    console.log('============================================================\n');
  } catch (err: any) {
    console.error('\n[X] Error generating response:', err.message);
  }
}

main().catch(console.error);
