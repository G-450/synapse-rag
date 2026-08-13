"""
Synapse RAG — Retrieval Evaluation Script (Python)
Equivalent of scripts/evaluate.ts

Evaluates retrieval quality using the LegalBench-RAG dataset.
Metrics: Precision, Recall, F1 at document level, plus Document Retrieval Match (DRM).

Usage:
    python scripts/evaluate.py
"""

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv(dotenv_path=Path(__file__).parent.parent.parent / ".env")
load_dotenv()

import psycopg2
import psycopg2.extras
from pgvector.psycopg2 import register_vector
from app.embeddings import generate_embedding


def compute_content_overlap(text_a: str, text_b: str) -> float:
    """Word-level overlap: fraction of words in text_a that appear in text_b."""
    words_a = {w for w in text_a.lower().split() if len(w) > 2}
    words_b = {w for w in text_b.lower().split() if len(w) > 2}
    if not words_a:
        return 0.0
    return len(words_a & words_b) / len(words_a)


def main():
    sep = "=" * 60
    print(sep)
    print("  Synapse RAG — Retrieval Evaluation")
    print(sep)

    data_path = Path(__file__).parent.parent.parent / "data" / "legalbench.json"
    if not data_path.exists():
        print(f"ERROR: Dataset not found at {data_path}")
        sys.exit(1)

    with open(data_path, "r", encoding="utf-8") as f:
        qa_pairs = json.load(f)

    print(f"\nLoaded {len(qa_pairs)} QA pairs for evaluation.\n")

    conn = psycopg2.connect(os.environ["DATABASE_URL"])
    register_vector(conn)

    results = []
    documents_matched = 0
    total_overlap = 0.0
    total_precision = 0.0
    total_recall = 0.0

    for i, qa in enumerate(qa_pairs):
        query = qa["metadata"]["query"]
        expected_file = qa["metadata"]["corpus_file"]
        expected_answer = qa["metadata"]["answer"]

        print(f"[{i + 1}/{len(qa_pairs)}] Evaluating: {qa['id']}...", end="", flush=True)

        try:
            embedding = generate_embedding(query)
            embedding_str = json.dumps(embedding)

            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(
                    """
                    SELECT c.id, c.content, c.document_id, d.filename, d.source_corpus,
                           1 - (c.embedding <=> %s::vector) AS similarity
                    FROM "Chunk" c
                    JOIN "Document" d ON d.id = c.document_id
                    WHERE c.embedding IS NOT NULL
                    ORDER BY c.embedding <=> %s::vector
                    LIMIT 5
                    """,
                    (embedding_str, embedding_str),
                )
                chunks = [dict(r) for r in cur.fetchall()]

            retrieved_files = list({c["filename"] for c in chunks})
            expected_filename = expected_file.split("/")[-1] or expected_file
            doc_match = any(
                f == expected_filename or expected_filename in f.replace(".txt", "")
                for f in retrieved_files
            )
            if doc_match:
                documents_matched += 1

            all_retrieved_text = " ".join(c["content"] for c in chunks)
            overlap_score = compute_content_overlap(expected_answer, all_retrieved_text)
            total_overlap += overlap_score

            precision = compute_content_overlap(all_retrieved_text, expected_answer)
            total_precision += precision
            total_recall += overlap_score

            results.append(
                {
                    "query_id": qa["id"],
                    "query": query[:80],
                    "expected_corpus_file": expected_filename,
                    "retrieved_files": retrieved_files,
                    "document_match": doc_match,
                    "content_overlap_score": overlap_score,
                }
            )

            mark = "✓" if doc_match else "✗"
            print(f" {mark} DRM | Overlap: {overlap_score * 100:.1f}%")

        except Exception as e:
            print(f" ERROR: {e}")

    conn.close()

    n = len(results)
    drm = documents_matched / n if n else 0
    avg_precision = total_precision / n if n else 0
    avg_recall = total_recall / n if n else 0
    f1 = (
        (2 * avg_precision * avg_recall) / (avg_precision + avg_recall)
        if (avg_precision + avg_recall) > 0
        else 0
    )

    print("\n" + sep)
    print("  EVALUATION RESULTS")
    print(sep)
    print(f"  Total QA pairs evaluated:   {n}")
    print(f"  Document Retrieval Match:    {drm * 100:.1f}% ({documents_matched}/{n})")
    print(f"  Avg Precision:              {avg_precision * 100:.1f}%")
    print(f"  Avg Recall:                 {avg_recall * 100:.1f}%")
    print(f"  F1 Score:                   {f1 * 100:.1f}%")
    print(sep)

    output_path = Path(__file__).parent.parent.parent / "data" / "eval_results.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "timestamp": __import__("datetime").datetime.utcnow().isoformat(),
                "config": {
                    "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
                    "dimensions": 384,
                    "top_k": 5,
                },
                "metrics": {
                    "document_retrieval_match": drm,
                    "precision": avg_precision,
                    "recall": avg_recall,
                    "f1": f1,
                },
                "results": results,
            },
            f,
            indent=2,
        )
    print(f"\nDetailed results saved to: {output_path}")


if __name__ == "__main__":
    main()
