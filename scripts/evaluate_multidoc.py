"""
Synapse RAG — Multi-Document Reasoning & Fan-Out Benchmark Suite

Measures:
  1. Citation Diversity (Retrieval fan-out across >= 2 distinct contract documents)
  2. Cross-Document Synthesis Accuracy (LLM-as-a-Judge evaluating comparative reasoning)
  3. Attribution Faithfulness (LLM-as-a-Judge validating factual grounding per document)

Usage:
    python scripts/evaluate_multidoc.py --limit 5
    python scripts/evaluate_multidoc.py --limit 10 --top-docs 3 --chunks-per-doc 3
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
from app.rag import SYSTEM_PROMPT, fan_out_retrieve, format_fan_out_context, get_llm


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


def evaluate_multidoc_query(
    item: dict,
    judge_llm: ChatGroq,
    top_docs: int = 3,
    chunks_per_doc: int = 3,
) -> dict:
    query = item.get("query", "")
    description = item.get("description", "")
    expected_min = item.get("expected_sources_min", 2)

    # 1. Multi-Document Fan-Out Retrieval
    doc_groups = fan_out_retrieve(query=query, top_docs=top_docs, chunks_per_doc=chunks_per_doc)
    distinct_docs = [g.get("filename", "") for g in doc_groups if g.get("filename")]
    context_text = format_fan_out_context(doc_groups)

    # 2. Generate Multi-Document Comparative Answer
    gen_llm = get_llm(streaming=False, temperature=0.1)
    user_prompt = f"""You are analyzing excerpts from multiple legal contracts.
Answer the following comparative query using ONLY the provided excerpts:

COMPARATIVE QUERY: {query}

INSTRUCTIONS:
1. Provide a clear, structured comparison across the provided documents.
2. For each document, cite its document name and quote or accurately summarize the relevant provisions.
3. Compare the terms across documents based strictly on what is stated in the excerpts.
4. If a document does not contain information on a specific point, explicitly note that it is not addressed in that document's excerpt.
"""

    prompt = [
        SystemMessage(content=f"{SYSTEM_PROMPT}\n\n=== CONTEXT (Multi-Contract Excerpts) ===\n\n{context_text}"),
        HumanMessage(content=user_prompt),
    ]
    answer = call_llm_with_retry(gen_llm, prompt)

    # 3. Metric 1: Citation Diversity
    diversity_pass = len(distinct_docs) >= expected_min

    # 4. Metric 2: Cross-Contract Synthesis Accuracy (LLM Judge)
    synth_prompt = f"""You are an expert legal AI evaluation judge measuring MULTI-DOCUMENT REASONING AND SYNTHESIS.

USER QUERY:
{query}

RETRIEVED CONTRACT DOCUMENTS:
{distinct_docs}

GENERATED ANSWER:
{answer}

EVALUATION TASK:
Evaluate whether the GENERATED ANSWER successfully performs relational/comparative synthesis across the retrieved documents:
1. Did the answer contrast, compare, or structure information from multiple contracts rather than only summarizing a single document?
2. Are specific clauses, terms, or distinctions attributed properly to their respective agreements?

Respond strictly in valid JSON format:
{{"score": 1, "reason": "Detailed 1-2 sentence explanation of synthesis"}} or {{"score": 0, "reason": "Detailed 1-2 sentence explanation of failure to synthesize"}}"""
    synth_resp = call_llm_with_retry(judge_llm, [HumanMessage(content=synth_prompt)])
    synth_data = parse_judge_json(synth_resp)
    synthesis_score = 1 if int(synth_data.get("score", 0)) == 1 else 0
    synthesis_reason = synth_data.get("reason", "")

    # 5. Metric 3: Multi-Doc Faithfulness (LLM Judge)
    is_refusal = "cannot answer this question based on the provided" in answer.lower()
    if is_refusal:
        faith_score = 1
        faith_reason = "Model faithfully abstained from hallucinating when excerpts were missing."
    else:
        faith_prompt = f"""You are an expert legal evaluation judge measuring FAITHFULNESS (grounding) in a Retrieval-Augmented Generation (RAG) system.

RETRIEVED MULTI-DOCUMENT CONTEXT:
{context_text}

GENERATED ANSWER:
{answer}

EVALUATION RUBRIC:
- Score 1 (Faithful): The substantive legal facts, contractual obligations, and quoted provisions in the answer are accurately derived from the retrieved context. High-level summaries, synthesis of commonalities/differences across the provided excerpts, or reasonable restatements of provided clauses are completely valid.
- Score 0 (Unfaithful / Hallucination): The answer introduces major external factual claims not found in any excerpt, invents entirely fictitious contract parties, or attributes specific terms to a document when the excerpts contradict that attribution.

Respond strictly in valid JSON format:
{{"score": 1, "reason": "Detailed 1-2 sentence explanation of grounding"}} or {{"score": 0, "reason": "Detailed 1-2 sentence explanation of unsupported claims"}}"""
        faith_resp = call_llm_with_retry(judge_llm, [HumanMessage(content=faith_prompt)])
        faith_data = parse_judge_json(faith_resp)
        faith_score = 1 if int(faith_data.get("score", 0)) == 1 else 0
        faith_reason = faith_data.get("reason", "")

    return {
        "id": item.get("id", ""),
        "query": query,
        "description": description,
        "retrieved_documents": distinct_docs,
        "distinct_docs_count": len(distinct_docs),
        "generated_answer": answer,
        "diversity_pass": diversity_pass,
        "synthesis_score": synthesis_score,
        "synthesis_reason": synthesis_reason,
        "faithfulness": faith_score,
        "faithfulness_reason": faith_reason,
        "hallucination": 1 - faith_score,
    }


def main():
    parser = argparse.ArgumentParser(description="Synapse RAG — Multi-Document Benchmark")
    parser.add_argument("--limit", type=int, default=10, help="Number of benchmark queries to evaluate")
    parser.add_argument("--top-docs", type=int, default=3, help="Number of distinct documents (default: 3)")
    parser.add_argument("--chunks-per-doc", type=int, default=3, help="Chunks per document (default: 3)")
    args = parser.parse_args()

    bench_path = root_dir / "data" / "multidoc_benchmark.json"
    if not bench_path.exists():
        print(f"Error: {bench_path} not found.")
        sys.exit(1)

    with open(bench_path, "r", encoding="utf-8") as f:
        benchmark = json.load(f)[: args.limit]

    judge_llm = get_judge_llm()
    results = []

    print("=" * 70)
    print(f"  SYNAPSE RAG — MULTI-DOCUMENT REASONING BENCHMARK")
    print(f"  Queries: {len(benchmark)} | Fan-Out: {args.top_docs} docs x {args.chunks_per_doc} chunks")
    print("=" * 70)

    for i, item in enumerate(benchmark, 1):
        res = evaluate_multidoc_query(
            item=item,
            judge_llm=judge_llm,
            top_docs=args.top_docs,
            chunks_per_doc=args.chunks_per_doc,
        )
        results.append(res)
        d = f"{res['distinct_docs_count']} docs"
        s = "Y" if res["synthesis_score"] else "N"
        f = "Y" if res["faithfulness"] else "N"
        print(f"[{i}/{len(benchmark)}] Query: {res['id']:<12} | Diversity: {d:<7} | Synth: {s} | Faith: {f}")
        time.sleep(1.5)

    n = len(results)
    if n == 0:
        print("No evaluations completed.")
        return

    div_rate = (sum(r["diversity_pass"] for r in results) / n) * 100
    synth_rate = (sum(r["synthesis_score"] for r in results) / n) * 100
    faith_rate = (sum(r["faithfulness"] for r in results) / n) * 100
    avg_docs = sum(r["distinct_docs_count"] for r in results) / n

    print("\n" + "=" * 70)
    print(f"  MULTI-DOCUMENT BENCHMARK RESULTS ({n} Queries)")
    print("=" * 70)
    print(f"  Multi-Doc Diversity Rate (>=2 docs): {div_rate:5.1f}% ({sum(r['diversity_pass'] for r in results)}/{n})")
    print(f"  Avg Distinct Documents Retrieved:    {avg_docs:5.2f} docs / query")
    print(f"  Synthesis Accuracy Score:            {synth_rate:5.1f}% ({sum(r['synthesis_score'] for r in results)}/{n})")
    print(f"  Attribution Faithfulness Score:      {faith_rate:5.1f}% ({sum(r['faithfulness'] for r in results)}/{n})")
    print("=" * 70)

    out_file = root_dir / "data" / "eval_multidoc_results.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(
            {
                "metrics": {
                    "diversity_rate": round(div_rate, 2),
                    "avg_docs": round(avg_docs, 2),
                    "synthesis_rate": round(synth_rate, 2),
                    "faithfulness": round(faith_rate, 2),
                },
                "results": results,
            },
            f,
            indent=2,
        )
    print(f"Saved results to: {out_file}\n")


if __name__ == "__main__":
    main()
