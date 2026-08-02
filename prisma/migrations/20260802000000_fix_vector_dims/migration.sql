-- Fix vector dimensions from 1536 to 384 (matching Xenova/all-MiniLM-L6-v2)
ALTER TABLE "Chunk" ALTER COLUMN "embedding" TYPE vector(384);

-- Create HNSW index for fast cosine similarity search
CREATE INDEX IF NOT EXISTS "Chunk_embedding_hnsw_idx" ON "Chunk" 
USING hnsw ("embedding" vector_cosine_ops)
WITH (m = 16, ef_construction = 64);
