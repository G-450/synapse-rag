"""
Synapse RAG — Retrieval Quality Evaluation

Measures retrieval-only metrics (no generation) against LegalBench-RAG gold snippets:
  - Document Retrieval Match (DRM) at k=1, 5, 20
  - Snippet-level Recall at k=1, 5, 20
  - Mean Reciprocal Rank (MRR) at document and snippet level
  - Re-ranking lift (bi-encoder only vs bi-encoder + cross-encoder)

Usage:
    python scripts/evaluate_retrieval.py --limit 200
    python scripts/evaluate_retrieval.py --limit 200 --seed 42
"""

import argparse
import json
import random
import re
import sys
import time
from pathlib import Path

current_dir = Path(__file__).resolve().parent
while current_dir.parent != current_dir:
    if (current_dir / "data").exists() and (current_dir / "python" / "app").exists():
        break
    if (current_dir / "app" / "rag.py").exists():
        current_dir = current_dir.parent
        break
    current_dir = current_dir.parent

root_dir = current_dir
python_dir = root_dir / "python" if (root_dir / "python").exists() else root_dir

for p in [str(python_dir), str(root_dir)]:
    if p not in sys.path:
        sys.path.insert(0, p)

# Patch Qdrant client to use a separate DB copy (avoids lock conflicts)
from qdrant_client import QdrantClient
import app.rag as _rag

_eval_client = None
def _patched_get_qdrant_client():
    global _eval_client
    if _eval_client is None:
        eval_db = python_dir / "qdrant_db_eval2"
        if not eval_db.exists():
            eval_db = python_dir / "qdrant_db"
        _eval_client = QdrantClient(path=str(eval_db))
    return _eval_client

_rag.get_qdrant_client = _patched_get_qdrant_client
_rag._qdrant_client = None
_rag._vectorstore = None

from app.rag import retrieve_chunks


def normalise(text: str) -> str:
    return re.sub(r"\W+", " ", text.lower()).strip()


def evaluate_query(item: dict, limit: int, rerank: bool) -> dict:
    meta = item.get("metadata", {})
    query = meta.get("query", "")
    corpus_file = meta.get("corpus_file", "")
    expected_file = corpus_file.split("/")[-1].replace(".txt", "").lower().strip()

    snippets_raw = meta.get("snippets", [])
    if isinstance(snippets_raw, str):
        import ast
        snippets_raw = ast.literal_eval(snippets_raw)
    gold_texts = {normalise(s.get("answer", s.get("text", ""))) for s in snippets_raw if isinstance(s, dict)}

    chunks = retrieve_chunks(query=query, document_id=None, limit=limit, rerank=rerank)

    doc_hits = []
    snippet_hits = []
    for c in chunks:
        fname = (c.get("filename", "") or "").replace(".txt", "").lower().strip()
        doc_match = expected_file in fname or fname in expected_file
        doc_hits.append(doc_match)

        chunk_text = normalise(c.get("content", c.get("page_content", "")))
        snip_match = any(g in chunk_text or chunk_text in g for g in gold_texts) if gold_texts else False
        snippet_hits.append(snip_match)

    def recall_at_k(hits, k):
        return 1.0 if any(hits[:k]) else 0.0

    def mrr(hits):
        for i, h in enumerate(hits):
            if h:
                return 1.0 / (i + 1)
        return 0.0

    return {
        "id": item.get("id", ""),
        "query": query[:100],
        "expected_file": expected_file,
        "doc_recall_1": recall_at_k(doc_hits, 1),
        "doc_recall_5": recall_at_k(doc_hits, 5),
        "doc_recall_20": recall_at_k(doc_hits, 20),
        "doc_mrr": mrr(doc_hits),
        "snip_recall_1": recall_at_k(snippet_hits, 1),
        "snip_recall_5": recall_at_k(snippet_hits, 5),
        "snip_recall_20": recall_at_k(snippet_hits, 20),
        "snip_mrr": mrr(snippet_hits),
    }


def main():
    parser = argparse.ArgumentParser(description="Synapse RAG — Retrieval Quality Evaluation")
    parser.add_argument("--limit", type=int, default=200, help="Number of queries to sample")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for sampling")
    parser.add_argument("--k", type=int, default=20, help="Number of chunks to retrieve per query")
    args = parser.parse_args()

    data_path = root_dir / "data" / "legalbench.json"
    if not data_path.exists():
        print(f"Error: {data_path} not found.")
        sys.exit(1)

    with open(data_path, "r", encoding="utf-8") as f:
        full_dataset = json.load(f)

    random.seed(args.seed)
    dataset = random.sample(full_dataset, min(args.limit, len(full_dataset)))

    configs = [
        ("bi-encoder only", False),
        ("bi-encoder + cross-encoder", True),
    ]

    all_results = {}

    for config_name, rerank in configs:
        print(f"\n{'=' * 70}")
        print(f"  RETRIEVAL EVALUATION — {config_name.upper()}")
        print(f"  Samples: {len(dataset)} | k={args.k} | seed={args.seed}")
        print(f"{'=' * 70}")

        results = []
        for i, item in enumerate(dataset, 1):
            try:
                res = evaluate_query(item, limit=args.k, rerank=rerank)
                results.append(res)
                if i % 25 == 0 or i == len(dataset):
                    print(f"  [{i}/{len(dataset)}] processed...")
            except Exception as e:
                print(f"  [{i}/{len(dataset)}] ERROR: {e}")

        n = len(results)
        if n == 0:
            continue

        metrics = {
            "config": config_name,
            "n": n,
            "k": args.k,
            "seed": args.seed,
            "doc_recall_at_1": round(sum(r["doc_recall_1"] for r in results) / n * 100, 1),
            "doc_recall_at_5": round(sum(r["doc_recall_5"] for r in results) / n * 100, 1),
            "doc_recall_at_20": round(sum(r["doc_recall_20"] for r in results) / n * 100, 1),
            "doc_mrr": round(sum(r["doc_mrr"] for r in results) / n, 3),
            "snip_recall_at_1": round(sum(r["snip_recall_1"] for r in results) / n * 100, 1),
            "snip_recall_at_5": round(sum(r["snip_recall_5"] for r in results) / n * 100, 1),
            "snip_recall_at_20": round(sum(r["snip_recall_20"] for r in results) / n * 100, 1),
            "snip_mrr": round(sum(r["snip_mrr"] for r in results) / n, 3),
        }

        print(f"\n  Document-level:")
        print(f"    Recall@1:  {metrics['doc_recall_at_1']:5.1f}%")
        print(f"    Recall@5:  {metrics['doc_recall_at_5']:5.1f}%")
        print(f"    Recall@20: {metrics['doc_recall_at_20']:5.1f}%")
        print(f"    MRR:       {metrics['doc_mrr']:.3f}")
        print(f"  Snippet-level:")
        print(f"    Recall@1:  {metrics['snip_recall_at_1']:5.1f}%")
        print(f"    Recall@5:  {metrics['snip_recall_at_5']:5.1f}%")
        print(f"    Recall@20: {metrics['snip_recall_at_20']:5.1f}%")
        print(f"    MRR:       {metrics['snip_mrr']:.3f}")

        all_results[config_name] = {"metrics": metrics, "per_query": results}

    if len(all_results) == 2:
        bi_only = all_results["bi-encoder only"]["metrics"]
        bi_ce = all_results["bi-encoder + cross-encoder"]["metrics"]
        print(f"\n{'=' * 70}")
        print(f"  RE-RANKING LIFT")
        print(f"{'=' * 70}")
        print(f"  Doc Recall@1:  {bi_only['doc_recall_at_1']:5.1f}% → {bi_ce['doc_recall_at_1']:5.1f}%  ({bi_ce['doc_recall_at_1'] - bi_only['doc_recall_at_1']:+.1f})")
        print(f"  Doc MRR:       {bi_only['doc_mrr']:.3f} → {bi_ce['doc_mrr']:.3f}  ({bi_ce['doc_mrr'] - bi_only['doc_mrr']:+.3f})")
        print(f"  Snip Recall@1: {bi_only['snip_recall_at_1']:5.1f}% → {bi_ce['snip_recall_at_1']:5.1f}%  ({bi_ce['snip_recall_at_1'] - bi_only['snip_recall_at_1']:+.1f})")
        print(f"  Snip MRR:      {bi_only['snip_mrr']:.3f} → {bi_ce['snip_mrr']:.3f}  ({bi_ce['snip_mrr'] - bi_only['snip_mrr']:+.3f})")

    out_file = root_dir / "results" / "retrieval_metrics.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n  Saved to: {out_file}")


if __name__ == "__main__":
    main()
