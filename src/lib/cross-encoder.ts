import { AutoTokenizer, AutoModelForSequenceClassification } from '@xenova/transformers';
import { RetrievedChunk } from './rag';

class CrossEncoderSingleton {
  static model_id = 'Xenova/ms-marco-MiniLM-L-6-v2';
  static tokenizer: any = null;
  static model: any = null;
  static isInitializing = false;
  static initPromise: Promise<void> | null = null;

  static async getInstance() {
    if (this.tokenizer && this.model) {
      return { tokenizer: this.tokenizer, model: this.model };
    }
    
    if (this.initPromise) {
      await this.initPromise;
      return { tokenizer: this.tokenizer, model: this.model };
    }

    this.isInitializing = true;
    this.initPromise = (async () => {
      this.tokenizer = await AutoTokenizer.from_pretrained(this.model_id);
      this.model = await AutoModelForSequenceClassification.from_pretrained(this.model_id);
    })();
    
    await this.initPromise;
    this.isInitializing = false;
    return { tokenizer: this.tokenizer, model: this.model };
  }
}

/**
 * Re-ranks a list of retrieved chunks against a query using a cross-encoder model.
 * @param query The search query string
 * @param chunks The candidate chunks retrieved by the bi-encoder
 * @returns The chunks sorted by their cross-encoder relevance score (descending)
 */
export async function rankChunks(query: string, chunks: RetrievedChunk[]): Promise<(RetrievedChunk & { cross_score?: number })[]> {
  if (!chunks || chunks.length === 0) return [];

  const { tokenizer, model } = await CrossEncoderSingleton.getInstance();
  
  // Rank each chunk against the query
  const rankedChunks = await Promise.all(
    chunks.map(async (chunk) => {
      // The tokenizer needs { text_pair: ... } for cross-encoder sequence classification
      const inputs = await tokenizer(query, {
        text_pair: chunk.content,
        padding: true,
        truncation: true
      });
      
      const output = await model(inputs);
      // MS MARCO cross-encoders output a single logit representing relevance
      const score = output.logits.data[0];
      
      return {
        ...chunk,
        cross_score: score
      };
    })
  );

  // Sort descending by cross_score
  return rankedChunks.sort((a, b) => (b.cross_score ?? 0) - (a.cross_score ?? 0));
}
