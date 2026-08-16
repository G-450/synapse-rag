"""
Synapse RAG — End-to-End Faithfulness & Accuracy Evaluation Suite

Measures:
  1. Retrieval Hit Rate (Document Retrieval Match / DRM)
  2. Answer Accuracy (LLM-as-a-Judge against LegalBench ground truth)
  3. Faithfulness Score (LLM-as-a-Judge against retrieved contract citations)
  4. Hallucination Rate (Percentage of answers with unsupported claims)

Usage:
    python scripts/evaluate_faithfulness.py --limit 10
    python scripts/evaluate_faithfulness.py --limit 20 --rerank
    python scripts/evaluate_faithfulness.py --limit 10 --unscoped
"""

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from dotenv import load_dotenv

# Robust root and python path resolution
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

load_dotenv(root_dir / ".env")
load_dotenv(python_dir / ".env")
load_dotenv()

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq
from app.rag import (
    SYSTEM_PROMPT,
    format_context,
    get_llm,
    get_qdrant_client,
    retrieve_chunks,
)


def get_judge_llm() -> ChatGroq:
    """Returns a deterministic LLM for evaluation."""
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY environment variable is required.")
    return ChatGroq(
        model="llama-3.1-8b-instant",
        api_key=api_key,
        temperature=0.0,
        max_tokens=512,
    )


def call_llm_with_retry(llm, messages, retries: int = 5) -> str:
    """Invoke LLM with exponential backoff for rate limits."""
    for attempt in range(retries):
        try:
            res = llm.invoke(messages)
            return res.content if hasattr(res, "content") else str(res)
        except Exception as e:
            err_str = str(e).lower()
            if "429" in err_str or "rate limit" in err_str:
                wait_sec = 3 * (attempt + 1)
                time.sleep(wait_sec)
            elif attempt == retries - 1:
                return f"Error: {e}"
            else:
                time.sleep(1.5)
    return ""


def build_docid_map() -> dict[str, str]:
    """Pre-indexes filename -> document_id from Qdrant for fast scoped lookups."""
    for attempt in range(3):
        try:
            client = get_qdrant_client()
            mapping = {}
            offset = None
            while True:
                records, next_offset = client.scroll(
                    collection_name="synapse_rag",
                    limit=1000,
                    with_payload=True,
                    with_vectors=False,
                    offset=offset,
                )
                for r in records:
                    payload = r.payload or {}
                    meta = payload.get("metadata", payload)
                    doc_id = meta.get("document_id")
                    filename = meta.get("filename")
                    if doc_id and filename:
                        clean_name = filename.replace(".txt", "").lower().strip()
                        mapping[clean_name] = doc_id
                        mapping[clean_name.replace(" ", "_")] = doc_id
                        mapping[clean_name.replace("_", " ")] = doc_id
                if next_offset is None:
                    break
                offset = next_offset
            return mapping
        except Exception as e:
            if attempt < 2:
                time.sleep(1.0)
            else:
                print(f"[Warning] Could not build doc_id map from Qdrant: {e}")
                return {}
    return {}


def parse_judge_json(content: str) -> dict:
    """Safely extracts structured JSON from LLM judge responses."""
    try:
        cleaned = content.strip()
        if "```json" in cleaned:
            cleaned = cleaned.split("```json")[1].split("```")[0].strip()
        elif "```" in cleaned:
            cleaned = cleaned.split("```")[1].split("```")[0].strip()

        match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if match:
            return json.loads(match.group(0))
        return json.loads(cleaned)
    except Exception:
        score = 1 if "1" in content and "0" not in content[:15] else 0
        return {"score": score, "reason": content[:100]}


def evaluate_query(
    item: dict,
    judge_llm: ChatGroq,
    docid_map: dict[str, str],
    scoped: bool = True,
    limit_chunks: int = 8,
    rerank: bool = False,
) -> dict:
    meta = item.get("metadata", {})
    query = meta.get("query", item.get("text", ""))
    ground_truth = meta.get("answer", "")
    corpus_file = meta.get("corpus_file", "")
    expected_clean = corpus_file.split("/")[-1].replace(".txt", "").lower().strip()

    # 1. Resolve target document_id if scoped
    target_doc_id = None
    if scoped and expected_clean:
        target_doc_id = (
            docid_map.get(expected_clean)
            or docid_map.get(expected_clean.replace(" ", "_"))
            or docid_map.get(expected_clean.replace("_", " "))
        )

    # 2. Retrieve relevant chunks
    chunks = retrieve_chunks(
        query=query,
        document_id=target_doc_id,
        limit=limit_chunks,
        rerank=rerank,
    )
    retrieved_filenames = [c.get("filename", "") for c in chunks]
    context_text = format_context(chunks)

    # 3. Generate RAG Answer with structured prompt
    gen_llm = get_llm(streaming=False, temperature=0.1)
    user_prompt = f"""Based strictly on the provided contract excerpts, answer the following legal query with precision:

QUERY: {query}

INSTRUCTIONS:
1. Identify and state the exact contractual clause, definition, condition, or section number that directly answers the query.
2. Quote relevant definitions, section numbers, conditions, and thresholds verbatim where applicable.
3. If the specific requested clause (e.g. Specific Performance, Regulatory Closing Condition, or Covenants) is NOT present in the provided excerpts, do NOT attempt to guess or infer other sections. State clearly: "I cannot answer this question based on the provided contract excerpts."
"""

    prompt = [
        SystemMessage(content=f"{SYSTEM_PROMPT}\n\n=== CONTEXT (Retrieved Contract Excerpts) ===\n\n{context_text}"),
        HumanMessage(content=user_prompt),
    ]
    answer = call_llm_with_retry(gen_llm, prompt)

    # 4. Metric 1: Retrieval Hit Rate
    if target_doc_id or not expected_clean:
        hit = len(chunks) > 0
    else:
        hit = any(
            expected_clean in (f or "").lower() or (f or "").lower() in expected_clean
            for f in retrieved_filenames
        )

    # 5. Metric 2: Answer Accuracy (LLM Judge against Ground Truth)
    is_refusal = "cannot answer this question based on the provided" in answer.lower()
    if is_refusal:
        acc_score = 0
        acc_reason = "Model refrained from answering because the required clause was absent from context."
    else:
        acc_prompt = f"""You are an expert legal AI evaluation judge.
Your task is to determine whether the GENERATED ANSWER accurately conveys the correct legal information contained in the GROUND TRUTH ANSWER for the given question.

QUESTION:
{query}

GROUND TRUTH LEGAL ANSWER:
{ground_truth}

GENERATED ANSWER:
{answer}

EVALUATION CRITERIA:
- Score 1 (Accurate): The generated answer accurately identifies the key legal terms, definitions, section numbers, conditions, or values matching the ground truth. Minor phrasing variations are acceptable if legally equivalent.
- Score 0 (Inaccurate): The generated answer makes contradictory claims, cites incorrect section numbers, misses the essential required term, or states it cannot find the information.

Respond strictly in valid JSON format:
{{"score": 1, "reason": "Detailed 1-2 sentence explanation of accuracy"}} or {{"score": 0, "reason": "Detailed 1-2 sentence explanation of inaccuracy"}}"""
        acc_resp = call_llm_with_retry(judge_llm, [HumanMessage(content=acc_prompt)])
        acc_data = parse_judge_json(acc_resp)
        acc_score = 1 if int(acc_data.get("score", 0)) == 1 else 0
        acc_reason = acc_data.get("reason", "")

    # 6. Metric 3: Faithfulness & Hallucination (LLM Judge against Context)
    if is_refusal:
        faith_score = 1
        faith_reason = "Model faithfully abstained from hallucinating when context was absent."
    else:
        faith_prompt = f"""You are an expert legal evaluation judge measuring FAITHFULNESS (grounding) in a Retrieval-Augmented Generation (RAG) system.

USER QUESTION:
{query}

RETRIEVED CONTRACT CONTEXT:
{context_text}

GENERATED ANSWER:
{answer}

EVALUATION RUBRIC:
- Score 1 (Faithful): The substantive legal facts, contractual obligations, and quoted provisions in the answer are accurately derived from the retrieved context. High-level summaries or reasonable restatements of provided clauses are completely valid.
- Score 0 (Unfaithful / Hallucination): The answer introduces major external factual claims not found in any excerpt, invents non-existent contract sections, or fabricates contract terms.

Respond strictly in valid JSON format:
{{"score": 1, "reason": "Detailed 1-2 sentence explanation of grounding"}} or {{"score": 0, "reason": "Detailed 1-2 sentence explanation of unsupported claims"}}"""
        faith_resp = call_llm_with_retry(judge_llm, [HumanMessage(content=faith_prompt)])
        faith_data = parse_judge_json(faith_resp)
        faith_score = 1 if int(faith_data.get("score", 0)) == 1 else 0
        faith_reason = faith_data.get("reason", "")

    return {
        "id": item.get("id", ""),
        "query": query,
        "expected_file": expected_clean,
        "retrieved_files": retrieved_filenames,
        "generated_answer": answer,
        "retrieval_hit": hit,
        "accuracy": acc_score,
        "accuracy_reason": acc_reason,
        "faithfulness": faith_score,
        "faithfulness_reason": faith_reason,
        "hallucination": 1 - faith_score,
    }


def main():
    parser = argparse.ArgumentParser(description="Synapse RAG — Faithfulness & Accuracy Evaluation")
    parser.add_argument("--limit", type=int, default=10, help="Number of samples to evaluate")
    parser.add_argument("--offset", type=int, default=0, help="Starting dataset index offset")
    parser.add_argument("--limit-chunks", type=int, default=8, help="Number of chunks retrieved per query (default: 8)")
    parser.add_argument("--rerank", action="store_true", help="Enable cross-encoder reranking")
    parser.add_argument("--unscoped", action="store_true", help="Disable single-document scoping")
    args = parser.parse_args()

    data_path = root_dir / "data" / "legalbench.json"
    if not data_path.exists():
        print(f"Error: {data_path} not found.")
        sys.exit(1)

    with open(data_path, "r", encoding="utf-8") as f:
        full_dataset = json.load(f)

    dataset = full_dataset[args.offset : args.offset + args.limit]
    scoped_mode = not args.unscoped

    print("=" * 70)
    print(f"  SYNAPSE RAG — FAITHFULNESS & ACCURACY EVALUATION")
    print(f"  Samples: {len(dataset)} | Scoped: {scoped_mode} | Chunks: {args.limit_chunks} | Rerank: {args.rerank}")
    print("=" * 70)

    docid_map = build_docid_map() if scoped_mode else {}
    judge_llm = get_judge_llm()
    results = []

    for i, item in enumerate(dataset, 1):
        res = evaluate_query(
            item=item,
            judge_llm=judge_llm,
            docid_map=docid_map,
            scoped=scoped_mode,
            limit_chunks=args.limit_chunks,
            rerank=args.rerank,
        )
        results.append(res)
        h = "Y" if res["retrieval_hit"] else "N"
        a = "Y" if res["accuracy"] else "N"
        f = "Y" if res["faithfulness"] else "N"
        print(f"[{i}/{len(dataset)}] QA: {res['id'][:15]:<15} | Hit: {h} | Acc: {a} | Faith: {f}")
        time.sleep(1.5)

    n = len(results)
    if n == 0:
        print("No evaluations completed.")
        return

    hit_rate = (sum(r["retrieval_hit"] for r in results) / n) * 100
    acc_rate = (sum(r["accuracy"] for r in results) / n) * 100
    faith_rate = (sum(r["faithfulness"] for r in results) / n) * 100
    halluc_rate = 100.0 - faith_rate

    print("\n" + "=" * 70)
    print(f"  EVALUATION SUMMARY RESULTS ({n} Queries)")
    print("=" * 70)
    print(f"  Retrieval Hit Rate:  {hit_rate:5.1f}% ({sum(r['retrieval_hit'] for r in results)}/{n})")
    print(f"  Answer Accuracy:     {acc_rate:5.1f}% ({sum(r['accuracy'] for r in results)}/{n})")
    print(f"  Faithfulness Score:  {faith_rate:5.1f}% ({sum(r['faithfulness'] for r in results)}/{n})")
    print(f"  Hallucination Rate:  {halluc_rate:5.1f}% ({sum(r['hallucination'] for r in results)}/{n})")
    print("=" * 70)

    out_file = root_dir / "data" / "eval_faithfulness_results.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(
            {
                "metrics": {
                    "hit_rate": round(hit_rate, 2),
                    "accuracy": round(acc_rate, 2),
                    "faithfulness": round(faith_rate, 2),
                    "hallucination_rate": round(halluc_rate, 2),
                },
                "results": results,
            },
            f,
            indent=2,
        )
    print(f"Saved results to: {out_file}\n")


if __name__ == "__main__":
    main()
