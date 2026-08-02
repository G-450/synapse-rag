import { pipeline } from '@xenova/transformers';

// We use the Singleton pattern so the model is only loaded once in memory
class PipelineSingleton {
  static task = 'feature-extraction';
  static model = 'Xenova/all-MiniLM-L6-v2';
  static instance: any = null;

  static async getInstance(progress_callback: any = null) {
    if (this.instance === null) {
      this.instance = pipeline(this.task, this.model, { progress_callback });
    }
    return this.instance;
  }
}

export async function generateEmbedding(text: string): Promise<number[]> {
  const embedder = await PipelineSingleton.getInstance();
  const output = await embedder(text, { pooling: 'mean', normalize: true });
  // Convert Float32Array to standard JS Array of numbers
  return Array.from(output.data);
}
