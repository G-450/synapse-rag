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


# The judge is deliberately a different, larger model than the generator (GROQ_MODEL), so that it is
# not grading its own output. Override with JUDGE_MODEL when the default is retired.
JUDGE_MODEL = os.getenv("JUDGE_MODEL", "openai/gpt-oss-120b")


def get_judge_llm() -> ChatGroq:
    """Returns a deterministic LLM for evaluation."""
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY environment variable is required.")
    return ChatGroq(
        model=JUDGE_MODEL,
        api_key=api_key,
        temperature=0.0,
        # The gpt-oss judges spend part of their completion budget on reasoning tokens before they
        # emit the JSON verdict. At 512 tokens a long context could consume the whole budget and
        # return empty content, which the aggregation then had to discard; low reasoning effort plus
        # a slightly larger ceiling keeps the verdict inside the budget and the token spend down.
        max_tokens=768,
        model_kwargs={"reasoning_effort": "low"},
    )


def call_llm_with_retry(llm, messages, retries: int = 5) -> str:
    """Invoke LLM with exponential backoff for rate limits.

    Every failure path returns a string starting with "Error:" so that parse_judge_json records an
    unscored verdict with a readable cause. Returning a bare empty string here used to make an
    exhausted rate-limit budget look like an empty model response.
    """
    last_err = "retries exhausted"
    for attempt in range(retries):
        try:
            res = llm.invoke(messages)
            return res.content if hasattr(res, "content") else str(res)
        except Exception as e:
            last_err = str(e)
            err_str = last_err.lower()
            if "429" in err_str or "rate limit" in err_str:
                wait_sec = 3 * (attempt + 1)
                time.sleep(wait_sec)
            elif attempt == retries - 1:
                return f"Error: {e}"
            else:
                time.sleep(1.5)
    return f"Error: {last_err}"


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
    """
    Extracts a structured verdict from an LLM judge response.

    A failed or unparseable judge call returns score=None, which the aggregation below excludes
    from the metrics and counts separately. It must never be coerced to a pass: silently scoring
    transport errors as 1 is what invalidated the earlier runs of this benchmark.
    """
    if not content or content.startswith("Error:"):
        return {"score": None, "reason": (content or "empty judge response")[:200]}

    cleaned = content.strip()
    if "```json" in cleaned:
        cleaned = cleaned.split("```json")[1].split("```")[0].strip()
    elif "```" in cleaned:
        cleaned = cleaned.split("```")[1].split("```")[0].strip()

    for candidate in (re.search(r"\{.*\}", cleaned, re.DOTALL), None):
        try:
            data = json.loads(candidate.group(0) if candidate else cleaned)
            score = data.get("score")
            if score in (0, 1, "0", "1"):
                return {"score": int(score), "reason": str(data.get("reason", ""))[:300]}
        except Exception:
            continue

    # No JSON, but an unambiguous bare verdict is still usable.
    bare = re.fullmatch(r"\s*([01])\s*", cleaned)
    if bare:
        return {"score": int(bare.group(1)), "reason": "bare score, no JSON"}

    return {"score": None, "reason": f"unparseable judge response: {cleaned[:160]}"}


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
    # A generation that failed outright (rate limit, API error, no text) is not a wrong answer: it
    # is a missing measurement. Judging it would score the error string itself, so the item is left
    # unscored and counted separately.
    gen_failed = answer.startswith("Error:") or not answer.strip()
    is_refusal = (
        not gen_failed
        and "cannot answer this question based on the provided" in answer.lower()
    )
    if gen_failed:
        reason = f"generation failed, not judged: {answer.strip()[:160] or 'empty response'}"
        return {
            "id": item.get("id", ""),
            "query": query,
            "expected_file": expected_clean,
            "retrieved_files": retrieved_filenames,
            "generated_answer": answer,
            "retrieval_hit": hit,
            "generation_failed": True,
            "refused": False,
            "accuracy": None,
            "accuracy_reason": reason,
            "faithfulness": None,
            "faithfulness_reason": reason,
            "hallucination": None,
        }
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
        acc_score = acc_data["score"]
        acc_reason = acc_data.get("reason", "")

    # 6. Metric 3: Faithfulness & Hallucination (LLM Judge against Context)
    ref_app_score = None
    ref_app_reason = ""
    if is_refusal:
        refusal_prompt = f"""You are an expert legal evaluation judge measuring FAITHFULNESS in a Retrieval-Augmented Generation (RAG) system.
The model REFUSED to answer the user question based on the retrieved context.
Your task is to determine whether that refusal was appropriate — i.e., does the retrieved context genuinely lack the necessary information to answer the question?

USER QUESTION:
{query}

RETRIEVED CONTRACT CONTEXT:
{context_text}

EVALUATION RUBRIC:
- Score 1 (Appropriate Refusal): The retrieved context does NOT contain enough information or the necessary clause to answer the question substantively. Refusing to answer was correct and faithful to the context.
- Score 0 (Unwarranted Refusal): The retrieved context DOES contain the required legal information/clause to answer the question, but the model incorrectly refused.

Respond strictly in valid JSON format:
{{"score": 1, "reason": "Detailed 1-2 sentence explanation of why refusal was appropriate"}} or {{"score": 0, "reason": "Detailed 1-2 sentence explanation of why refusal was unwarranted"}}"""
        faith_resp = call_llm_with_retry(judge_llm, [HumanMessage(content=refusal_prompt)])
        faith_data = parse_judge_json(faith_resp)
        ref_app_score = faith_data["score"]
        ref_app_reason = faith_data.get("reason", "")
        # faithfulness / hallucination are defined only for affirmative answers that make
        # factual claims; a refusal fabricates nothing, so it must not enter the grounding
        # denominator. Keeping faith_score=None here lets the rate() helper filter refusals
        # out by predicate (not by treating them as failed judge calls).
        faith_score = None
        faith_reason = ""
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
        faith_score = faith_data["score"]
        faith_reason = faith_data.get("reason", "")

    return {
        "id": item.get("id", ""),
        "query": query,
        "expected_file": expected_clean,
        "retrieved_files": retrieved_filenames,
        "generated_answer": answer,
        "retrieval_hit": hit,
        "refused": is_refusal,
        "accuracy": acc_score,
        "generation_failed": False,
        "accuracy_reason": acc_reason,
        "faithfulness": faith_score,
        "faithfulness_reason": faith_reason,
        "hallucination": None if faith_score is None else 1 - faith_score,
        "refusal_appropriate": ref_app_score,
        "refusal_appropriate_reason": ref_app_reason,
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
        if res.get("generation_failed"):
            print(f"[{i}/{len(dataset)}] QA: {res['id'][:15]:<15} | Hit: {h} | GENERATION FAILED")
            time.sleep(1.5)
            continue
        a = {1: "Y", 0: "N", None: "?"}[res["accuracy"]]
        f = {1: "Y", 0: "N", None: "?"}[res["faithfulness"]]
        print(f"[{i}/{len(dataset)}] QA: {res['id'][:15]:<15} | Hit: {h} | Acc: {a} | Faith: {f}")
        time.sleep(1.5)

    n = len(results)
    if n == 0:
        print("No evaluations completed.")
        return

    refusals = sum(1 for r in results if r.get("refused"))
    gen_failures = sum(1 for r in results if r.get("generation_failed"))
    n_affirmative = n - refusals - gen_failures

    def rate(key, predicate=None):
        """Percentage over items the judge actually scored that satisfy predicate.

        predicate: optional callable(result) -> bool that restricts the denominator.
        Items excluded by predicate are not counted as unscored — they are structurally
        outside the metric's scope (e.g. refusals are outside faithfulness' scope).
        Items included by predicate but with key=None are genuinely unscored judge calls.
        """
        pool = results if predicate is None else [r for r in results if predicate(r)]
        pool_n = len(pool)
        scored = [r[key] for r in pool if r[key] is not None]
        unscored = pool_n - len(scored)
        pct = (sum(scored) / len(scored)) * 100 if scored else None
        return pct, len(scored), sum(scored), unscored

    is_affirmative = lambda r: not r.get("refused") and not r.get("generation_failed")
    is_refusal_row = lambda r: r.get("refused") and not r.get("generation_failed")

    hits = sum(r["retrieval_hit"] for r in results)
    hit_rate = (hits / n) * 100
    acc_rate, acc_n, acc_pass, acc_unscored = rate("accuracy")
    # faithfulness is defined only over affirmative answers — refusals are excluded by
    # construction, not treated as missing judge calls.
    faith_rate, faith_n, faith_pass, faith_unscored = rate("faithfulness", predicate=is_affirmative)
    halluc_rate = None if faith_rate is None else 100.0 - faith_rate

    def fmt(pct):
        return "  n/a" if pct is None else f"{pct:5.1f}%"

    print("\n" + "=" * 70)
    print(f"  EVALUATION SUMMARY RESULTS ({n} Queries)")
    print(f"  Judge model: {JUDGE_MODEL}")
    print("=" * 70)
    print(f"  Retrieval Hit Rate:  {hit_rate:5.1f}% ({hits}/{n})")
    print(f"  Answer Accuracy:     {fmt(acc_rate)} ({acc_pass}/{acc_n} scored)")
    print(f"  Faithfulness Score:  {fmt(faith_rate)} ({faith_pass}/{faith_n} scored, affirmative answers only)")
    print(f"  Hallucination Rate:  {fmt(halluc_rate)}  (1 - faithfulness over affirmative answers)")
    print(f"  Explicit refusals:   {refusals}/{n} (scored as accuracy 0; refusal appropriateness judged separately)")
    if refusals > 0:
        ref_rate, ref_n, ref_pass, ref_unscored = rate("refusal_appropriate", predicate=is_refusal_row)
        print(f"  Refusal Appropriate: {fmt(ref_rate)} ({ref_pass}/{ref_n} scored)")
    else:
        ref_rate, ref_n, ref_unscored = None, 0, 0
    print(f"  Generations failed:  {gen_failures}/{n} (not judged, excluded from rates)")
    print(f"  Judge calls unscored: accuracy {acc_unscored}/{n} | faithfulness {faith_unscored}/{n_affirmative}")
    if acc_unscored or faith_unscored:
        print("  WARNING: unscored calls are EXCLUDED from the rates above, not counted as passes.")
        print("  Check accuracy_reason/faithfulness_reason in the saved JSON for the cause; a daily")
        print("  Groq token limit (429 TPD) will silently shrink the sample these rates rest on.")
    print("=" * 70)

    out_file = root_dir / "data" / "eval_faithfulness_results.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(
            {
                "config": {
                    "judge_model": JUDGE_MODEL,
                    "generator_model": os.getenv("GROQ_MODEL", "openai/gpt-oss-20b"),
                    "n": n,
                    "scoped": scoped_mode,
                    "rerank": args.rerank,
                    "limit_chunks": args.limit_chunks,
                },
                "metrics": {
                    "hit_rate": round(hit_rate, 2),
                    "accuracy": None if acc_rate is None else round(acc_rate, 2),
                    "faithfulness": None if faith_rate is None else round(faith_rate, 2),
                    "hallucination_rate": None if halluc_rate is None else round(halluc_rate, 2),
                    "refusal_rate": round((refusals / n) * 100, 2),
                    "refusal_appropriate_rate": None if ref_rate is None else round(ref_rate, 2),
                    "refusal_appropriate_scored_n": ref_n,
                    "accuracy_scored_n": acc_n,
                    "faithfulness_scored_n": faith_n,
                    "faithfulness_denominator": n_affirmative,
                    "generation_failures": gen_failures,
                    "judge_unscored_accuracy": acc_unscored,
                    "judge_unscored_faithfulness": faith_unscored,
                },
                "results": results,
            },
            f,
            indent=2,
        )
    print(f"Saved results to: {out_file}\n")


if __name__ == "__main__":
    main()
