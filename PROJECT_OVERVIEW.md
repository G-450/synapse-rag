# Synapse RAG: Project & Approach Overview

Synapse RAG is an intelligent question-answering application built for legal contracts. It allows users to ask plain-English questions about a specific contract, or compare information across a whole collection of contracts. The system guarantees that its answers are provably grounded in actual contract text, preventing LLM hallucination.

## 1. Core Architecture

The system is built on **Retrieval-Augmented Generation (RAG)**:
1. **Ingestion**: Contracts are chunked, converted to vector embeddings, and stored.
2. **Retrieval**: User queries are embedded, and semantic similarity search finds the most relevant passages.
3. **Generation**: An LLM (powered by Groq) generates an answer strictly using the retrieved text, and will refuse to answer if the context does not support it.

## 2. The Dataset: LegalBench-RAG

Instead of generic text or the limited CUAD v1 dataset, this project uses **LegalBench-RAG**.
This dataset provides expert-verified ground truth exact character spans across four distinct legal sub-corpora:
- **CUAD**: Commercial contracts (clause types).
- **ContractNLI**: NDAs (hypothesis checking).
- **MAUD**: Merger agreements.
- **PrivacyQA**: Privacy policies.

This guarantees we can evaluate accuracy against a true legal baseline.

## 3. Query Routing: Single vs. Multi-Document

Synapse RAG strictly separates two types of searches:

### Single-Document Mode
- **Use Case:** *"What is the termination clause in this agreement?"*
- **Action:** Searches only within the specified document and grounds the answer locally.

### Multi-Document (Fan-Out) Mode
- **Use Case:** *"Which of these agreements have an uncapped liability clause?"*
- **Action:** 
  1. Searches each relevant document independently.
  2. Generates an answer per document.
  3. Combines the results into a single aggregated comparison (e.g., a table).

*Blending chunks from unrelated documents into a single search often yields incorrect conclusions. Our fan-out approach solves this.*

## 4. Evaluation Strategy

To prove the system works, we measure performance in two independent stages:
- **Retrieval Quality (LegalBench-RAG):** Measures precision, recall, and F1 score at the exact character span level to verify we retrieved the right text.
- **Answer Quality (Ragas):** Evaluates faithfulness, answer relevancy, context precision, and context recall using an LLM-as-a-judge to ensure the generated answer faithfully represents the retrieved text without hallucinations.

## 5. Technology Stack Summary

- **Frontend & API**: Next.js App Router (TypeScript), Tailwind CSS, shadcn/ui.
- **Database**: PostgreSQL with `pgvector`.
- **ORM**: Prisma.
- **Auth**: NextAuth.js.
- **Embedding Model**: OpenAI `text-embedding-3-small` (or BGE/e5).
- **Generation LLM**: Groq (LPU inference for high-speed multi-document parallel generation).
- **Orchestration**: Custom TypeScript pipelines (no LangChain overhead).
