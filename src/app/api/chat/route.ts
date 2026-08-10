import {
  retrieveChunks,
  fanOutRetrieve,
  formatContext,
  formatFanOutContext,
} from '@/lib/rag';
import { streamText } from 'ai';
import { createGroq } from '@ai-sdk/groq';

// Allow streaming responses up to 60 seconds
export const maxDuration = 60;

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

export async function POST(req: Request) {
  try {
    const { messages, documentId } = await req.json();

    // Get the latest user message
    const latestMessage = messages[messages.length - 1];
    const userQuery = latestMessage.content || 
      (latestMessage.parts && latestMessage.parts
        .filter((p: any) => p.type === 'text')
        .map((p: any) => p.text)
        .join('')) || 
      '';
    
    if (!userQuery) {
      return new Response(JSON.stringify({ error: 'Query is empty' }), { status: 400 });
    }

    // Retrieve context based on mode
    let contextText: string;
    let citations: Array<{
      chunk_id: string;
      document_id: string;
      filename: string;
      content: string;
      similarity: number;
    }> = [];

    if (documentId) {
      // Single-document mode
      const chunks = await retrieveChunks(userQuery, {
        documentId,
        limit: 5,
      });
      contextText = formatContext(chunks);
      citations = chunks.map((c) => ({
        chunk_id: c.id,
        document_id: c.document_id,
        filename: c.filename || '',
        content: c.content,
        similarity: c.similarity,
      }));
    } else {
      // Multi-document fan-out mode
      const docGroups = await fanOutRetrieve(userQuery, {
        topDocs: 3,
        chunksPerDoc: 3,
      });
      contextText = formatFanOutContext(docGroups);
      citations = docGroups.flatMap((group) =>
        group.chunks.map((c) => ({
          chunk_id: c.id,
          document_id: c.document_id,
          filename: group.filename,
          content: c.content,
          similarity: c.similarity,
        }))
      );
    }

    // Build the augmented system prompt
    const augmentedSystem =
      SYSTEM_PROMPT +
      '\n\n=== CONTEXT (Retrieved Contract Excerpts) ===\n\n' +
      contextText;

    // Convert UI messages to model messages
    const modelMessages = messages.map((m: any) => {
      let content = m.content || '';
      if (!content && m.parts) {
        content = m.parts
          .filter((p: any) => p.type === 'text')
          .map((p: any) => p.text)
          .join('');
      }
      return {
        role: m.role,
        content,
      };
    });

    // Stream the response
    const result = streamText({
      model: groq('llama-3.1-8b-instant'),
      system: augmentedSystem,
      messages: modelMessages,
      // Attach citations as metadata in the response headers
      headers: {
        'X-Citations': encodeURIComponent(JSON.stringify(citations)),
      },
    });

    return result.toTextStreamResponse({
      headers: {
        'X-Citations': encodeURIComponent(JSON.stringify(citations)),
      },
    });
  } catch (error) {
    console.error('[Chat API Error]', error);
    return new Response(
      JSON.stringify({ error: 'Internal Server Error' }),
      { status: 500, headers: { 'Content-Type': 'application/json' } }
    );
  }
}
