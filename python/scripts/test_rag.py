"""
Synapse RAG — RAG Query Test (Python)
Equivalent of scripts/test-cross-encoder-rag.ts

Tests retrieval with and without cross-encoder reranking.

Usage:
    python scripts/test_rag.py
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv(dotenv_path=Path(__file__).parent.parent.parent / ".env")
load_dotenv()

from app.rag import retrieve_chunks


def main():
    query = "What is the definition of Intervening Event?"

    print("=" * 60)
    print(f'QUERY: "{query}"')
    print("=" * 60 + "\n")

    print("[1/2] Retrieving context chunks (NO RERANK)...")
    t0 = time.time()
    chunks_no_rerank = retrieve_chunks(query, limit=5, rerank=False)
    ms_no_rerank = int((time.time() - t0) * 1000)

    print(f"[✓] Retrieved {len(chunks_no_rerank)} chunks in {ms_no_rerank}ms.")
    for i, c in enumerate(chunks_no_rerank, 1):
        sim = c.get("similarity", 0)
        preview = c["content"][:50]
        print(f"  - Chunk {i}: {sim:.4f} similarity | {preview}...")

    print("\n[2/2] Retrieving context chunks (WITH RERANK)...")
    t0 = time.time()
    chunks_rerank = retrieve_chunks(query, limit=5, rerank=True)
    ms_rerank = int((time.time() - t0) * 1000)

    print(f"[✓] Retrieved & Reranked {len(chunks_rerank)} chunks in {ms_rerank}ms.")
    for i, c in enumerate(chunks_rerank, 1):
        score = c.get("cross_score", 0)
        preview = c["content"][:50]
        print(f"  - Chunk {i}: {score:.4f} cross_score | {preview}...")


if __name__ == "__main__":
    main()
