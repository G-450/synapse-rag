# Synapse RAG - Project Context & Memory

## Project Overview
Synapse RAG is a Retrieval-Augmented Generation application for legal contracts (using the LegalBench-RAG corpus). It handles single-document lookups and multi-document fan-out (comparative) queries.

## Tech Stack
- **Frontend/Backend:** Next.js App Router, Tailwind CSS, shadcn/ui
- **Database:** PostgreSQL (Supabase/Neon) + `pgvector`
- **ORM:** Prisma
- **AI/LLM:** OpenAI `text-embedding-3-small` (embeddings) and Groq API (fast generation)
- **Evaluation:** LegalBench-RAG (Retrieval) and Ragas (Generation)

## Current Progress & State
- **Phase 1 (Foundation):** 
  - Next.js successfully initialized in `c:\Users\Rithan\Academics\synapse-rag`.
  - Python evaluation environment (`eval-env`) initialized with `ragas` and `datasets`.
  - Prisma schema is defined with `Document`, `Chunk`, `QueryLog`, and `pgvector` configuration.
  - User has provided Supabase and Groq credentials in `.env`.
  - **Currently:** We are running the first Prisma migration (`npx prisma migrate dev --name init`) to push the schema to the database.

## Critical Notes
- **Multi-document Fan-out:** We do not blend chunks from different documents. We retrieve independently per document, generate an answer per document, and synthesize.
- **Evaluation:** Must report both retrieval (Precision/Recall/F1) via LegalBench-RAG and end-to-end (Faithfulness/Answer Relevancy) via Ragas.
- **Agents:** Currently executing long-running setup scripts as background tasks to maximize efficiency.
