"""
Core RAG retrieval logic for Synapse RAG (LangChain + Qdrant version).
"""

import os
import re
import threading
from pathlib import Path
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.http import models
from langchain_qdrant import QdrantVectorStore
from langchain_groq import ChatGroq

from app.embeddings import get_embeddings
from app.cross_encoder import get_reranker

load_dotenv()

# ---------------------------------------------------------------------------
# Qdrant infrastructure
# ---------------------------------------------------------------------------

# Global instance of QdrantClient to avoid RocksDB lock conflicts
_qdrant_client = None
# FastAPI runs the sync route handlers in a threadpool, so concurrent requests can
# reach get_qdrant_client() at the same time. Without this lock each of them sees an
# empty _qdrant_client and opens its own embedded client, and every one after the
# first fails because the storage folder is already locked on disk.
_qdrant_client_lock = threading.Lock()


def get_qdrant_client():
    global _qdrant_client
    if _qdrant_client is None:
        with _qdrant_client_lock:
            if _qdrant_client is None:
                qdrant_dir = Path(__file__).parent.parent / "qdrant_db"
                _qdrant_client = QdrantClient(path=str(qdrant_dir))
    return _qdrant_client


def get_vectorstore() -> QdrantVectorStore:
    client = get_qdrant_client()
    collection_name = "synapse_rag"

    # Check if the collection exists, and create it if missing to prevent startup errors
    from qdrant_client.http.exceptions import UnexpectedResponse
    try:
        client.get_collection(collection_name=collection_name)
    except (ValueError, UnexpectedResponse):
        client.create_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(
                size=384,  # all-MiniLM-L6-v2 vector dimension
                distance=models.Distance.COSINE
            )
        )

    return QdrantVectorStore(
        client=client,
        collection_name=collection_name,
        embedding=get_embeddings(),
    )


# ---------------------------------------------------------------------------
# Retrieval functions
# ---------------------------------------------------------------------------

def build_document_filter(document_ids: list[str] | None) -> models.Filter | None:
    """Restrict a search to an explicit set of documents, or everything when empty."""
    scoped = [doc_id for doc_id in (document_ids or []) if doc_id]
    if not scoped:
        return None
    match = (
        models.MatchValue(value=scoped[0])
        if len(scoped) == 1
        else models.MatchAny(any=scoped)
    )
    return models.Filter(
        must=[models.FieldCondition(key="metadata.document_id", match=match)]
    )


# A chunk whose text is only a clause heading ("Section 8.2. Effect of Termination.")
# answers nothing on its own, and the index holds one per heading in the benchmark
# corpora. Those stubs match heading-shaped questions better than the clauses that
# actually contain the answer, so without this they fill every context slot and the
# model is left with nothing to ground an answer in.
MIN_CONTEXT_WORDS = max(1, int(os.getenv("MIN_CONTEXT_WORDS", "15")))

_BREADCRUMB_RE = re.compile(r"^\[[^\]]*\]\s*")


def has_substantive_text(text: str) -> bool:
    """True when a chunk carries enough prose to support an answer."""
    body = _BREADCRUMB_RE.sub("", (text or "").strip())
    return len(body.split()) >= MIN_CONTEXT_WORDS


def drop_heading_stubs(scored_docs: list) -> list:
    """Keep only chunks with real clause text, unless that would leave nothing."""
    substantive = [
        pair for pair in scored_docs
        if has_substantive_text(pair[0].metadata.get("child_content") or pair[0].page_content)
    ]
    return substantive or scored_docs


def retrieve_chunks(
    query: str,
    document_id: str | None = None,
    limit: int = 5,
    rerank: bool = True,
    document_ids: list[str] | None = None,
) -> list[dict]:
    """
    Retrieve the most relevant chunks using Qdrant similarity search.
    Optionally scoped to one or more documents and/or reranked with cross-encoder.
    """
    # Over-fetch: heading stubs are dropped below, so the pool has to be deep
    # enough to still hold `limit` real clauses once they are gone.
    fetch_limit = max(limit * 8, 60) if rerank else max(limit * 4, 40)
    vectorstore = get_vectorstore()

    scope = list(document_ids) if document_ids else ([document_id] if document_id else [])
    document_filter = build_document_filter(scope)

    filter_kwargs = {}
    if document_filter is not None:
        filter_kwargs["filter"] = document_filter

    # Perform similarity search with score
    docs_with_scores = vectorstore.similarity_search_with_score(
        query=query,
        k=fetch_limit,
        **filter_kwargs
    )

    documents = []
    for doc, score in drop_heading_stubs(docs_with_scores):
        doc.metadata["similarity"] = score
        documents.append(doc)

    if rerank and documents:
        reranker = get_reranker(top_n=limit)
        documents = reranker.compress_documents(documents, query)
    else:
        documents = documents[:limit]

    # Convert LangChain Documents back to dict format expected by our endpoints
    return [
        {
            "id": getattr(doc, "id", None) or doc.metadata.get("chunk_id", ""),
            "document_id": doc.metadata.get("document_id", ""),
            "content": doc.metadata.get("child_content", doc.page_content),
            "embedded_content": doc.page_content,
            "filename": doc.metadata.get("filename", ""),
            "source_corpus": doc.metadata.get("source_corpus", ""),
            "similarity": float(doc.metadata.get("similarity", 0.0)),
            "parent_id": doc.metadata.get("parent_id", ""),
            "parent_header": doc.metadata.get("parent_header", ""),
            "parent_content": doc.metadata.get("parent_content", doc.page_content),
            "parent_article": doc.metadata.get("parent_article", ""),
            "section_number": doc.metadata.get("section_number", ""),
            "section_title": doc.metadata.get("section_title", ""),
            "section_kind": doc.metadata.get("section_kind", ""),
            "chunk_type": doc.metadata.get("chunk_type", "child"),
            "start_char": doc.metadata.get("start_char"),
            "end_char": doc.metadata.get("end_char"),
        }
        for doc in documents
    ]


def fan_out_retrieve(
    query: str,
    top_docs: int = 3,
    chunks_per_doc: int = 3,
    document_ids: list[str] | None = None,
) -> list[dict]:
    """
    Fan-out multi-document retrieval.
    Fetches a large pool of chunks, groups by document, and selects top docs/chunks.
    When document_ids is given, the fan-out is restricted to that selection.
    """
    vectorstore = get_vectorstore()

    document_filter = build_document_filter(document_ids)
    filter_kwargs = {"filter": document_filter} if document_filter is not None else {}

    # Fetch a large pool so every document in scope gets a fair chance of surfacing,
    # with room to spare for the heading stubs dropped below.
    pool_size = max(90, top_docs * chunks_per_doc * 12)

    docs_with_scores = vectorstore.similarity_search_with_score(
        query=query,
        k=pool_size,
        **filter_kwargs
    )

    # Group by document_id
    grouped = {}
    for doc, score in drop_heading_stubs(docs_with_scores):
        doc_id = doc.metadata.get("document_id")
        if not doc_id:
            continue

        if doc_id not in grouped:
            grouped[doc_id] = {
                "document_id": doc_id,
                "filename": doc.metadata.get("filename", ""),
                "source_corpus": doc.metadata.get("source_corpus", ""),
                "chunks": [],
                "best_similarity": 0.0
            }

        chunk_dict = {
            "id": getattr(doc, "id", None) or doc.metadata.get("chunk_id", ""),
            "document_id": doc_id,
            "content": doc.metadata.get("child_content", doc.page_content),
            "embedded_content": doc.page_content,
            "filename": doc.metadata.get("filename", ""),
            "source_corpus": doc.metadata.get("source_corpus", ""),
            "similarity": float(score),
            "parent_id": doc.metadata.get("parent_id", ""),
            "parent_header": doc.metadata.get("parent_header", ""),
            "parent_content": doc.metadata.get("parent_content", doc.page_content),
            "parent_article": doc.metadata.get("parent_article", ""),
            "section_number": doc.metadata.get("section_number", ""),
            "section_title": doc.metadata.get("section_title", ""),
            "section_kind": doc.metadata.get("section_kind", ""),
            "chunk_type": doc.metadata.get("chunk_type", "child"),
            "start_char": doc.metadata.get("start_char"),
            "end_char": doc.metadata.get("end_char"),
        }

        grouped[doc_id]["chunks"].append(chunk_dict)
        # Update best similarity
        if score > grouped[doc_id]["best_similarity"]:
            grouped[doc_id]["best_similarity"] = float(score)

    # Sort documents by their best similarity and take top N
    sorted_docs = sorted(list(grouped.values()), key=lambda x: x["best_similarity"], reverse=True)
    top_docs_list = sorted_docs[:top_docs]

    # For each document, limit to chunks_per_doc
    for doc_group in top_docs_list:
        doc_group["chunks"] = doc_group["chunks"][:chunks_per_doc]

    return top_docs_list


# ---------------------------------------------------------------------------
# Context Formatting & Prompts
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are Synapse RAG, an expert legal contract analyst specializing in corporate M&A agreements, commercial contracts, and regulatory filings.

Your primary directive is to provide rigorous, accurate, and completely grounded legal analysis based EXCLUSIVELY on the provided contract excerpts.

CORE OPERATING PRINCIPLES:
1. STRICT FACTUAL GROUNDING:
   - Base your answer ONLY on the provided Context excerpts.
   - Do NOT assume, extrapolate, or introduce outside legal terms or clauses not present in the excerpts.
   - If the Context does not contain the necessary information to answer the question, state clearly: "I cannot answer this question based on the provided contract excerpts."

2. PRECISE LEGAL CITATION & QUOTATION:
   - Directly cite the specific section numbers, article headings, and document names whenever available (e.g., "Under Section 6.2(a) of [Document]...").
   - Quote operative contractual language verbatim using quotation marks for key definitions, conditions, thresholds, and covenants.

3. STRUCTURED & THOROUGH ANALYSIS:
   - Provide a direct answer upfront.
   - Break down complex multi-part clauses (e.g., conditions, exceptions, carve-outs, triggers) using clear bullet points or numbered lists.
   - For multi-document comparisons, contrast the terms agreement-by-agreement, highlighting material differences in definitions, liability caps, governing laws, or closing conditions.

4. PROFESSIONAL TONE:
   - Maintain an objective, authoritative legal counsel advisory tone."""


def format_context(chunks: list[dict], max_parent_clauses: int | None = None) -> str:
    """Expand children to unique parent clauses within a bounded prompt budget."""
    if not chunks:
        return "[NO RELEVANT CONTRACT EXCERPTS FOUND]"
    if max_parent_clauses is None:
        max_parent_clauses = max(1, int(os.getenv("MAX_PARENT_CLAUSES", "6")))
    parts = []
    seen: set[tuple[str, str]] = set()
    for chunk in chunks:
        parent_key = chunk.get("parent_id") or chunk.get("id") or chunk.get("content", "")
        key = (chunk.get("document_id", ""), parent_key)
        if key in seen:
            continue
        seen.add(key)
        source = chunk.get("filename") or chunk.get("document_id", "Unknown Document")
        relevance = float(chunk.get("similarity", 0))
        section = chunk.get("parent_header") or (
            f"Section {chunk['section_number']}" if chunk.get("section_number") else "Unnumbered clause"
        )
        span = ""
        if chunk.get("start_char") is not None and chunk.get("end_char") is not None:
            span = f" | Matching span: {chunk['start_char']}-{chunk['end_char']}"
        parts.append(
            f"--- [Parent Clause {len(parts) + 1} | Document: {source} | {section} | "
            f"Relevance Score: {relevance:.3f}{span}] ---\n"
            f"{chunk.get('parent_content', chunk['content']).strip()}"
        )
        if len(parts) >= max_parent_clauses:
            break
    return "\n\n".join(parts)


def format_fan_out_context(doc_groups: list[dict]) -> str:
    """Format fan-out multi-document results grouped distinctly by contract."""
    if not doc_groups:
        return "[NO RELEVANT CONTRACT EXCERPTS FOUND]"
    sections = []
    for group in doc_groups:
        corpus = group.get("source_corpus", "legalbench-rag")
        header = f"================================================================================\n=== CONTRACT DOCUMENT: {group['filename']} ({corpus}) ===\n================================================================================"
        sections.append(header + "\n\n" + format_context(group.get("chunks", [])))
    return "\n\n".join(sections)


# ---------------------------------------------------------------------------
# LLM Initialization (Lazy / Function Factory)
# ---------------------------------------------------------------------------

def get_llm(
    streaming: bool = True,
    temperature: float = 0.2,
    max_tokens: int = 2048,
):
    """
    Lazy initialization of ChatGroq to prevent startup crashes when GROQ_API_KEY
    is loaded at runtime or from a specific dotenv file path.
    """
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError(
            "GROQ_API_KEY environment variable is not set. "
            "Please configure it in your .env file."
        )
    configured = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")
    # Both former defaults (qwen/qwen3.6-27b, llama-3.1-8b-instant) have been decommissioned on Groq,
    # which left the fallback chain dead: every fallback returned 404. Keep this list checked against
    # the live catalogue (GET https://api.groq.com/openai/v1/models).
    fallback_names = os.getenv("GROQ_FALLBACK_MODELS", "openai/gpt-oss-120b,qwen/qwen3.8-27b")
    model_names = list(dict.fromkeys([configured, *[m.strip() for m in fallback_names.split(",") if m.strip()]]))

    def create(model_name: str) -> ChatGroq:
        return ChatGroq(
            model=model_name,
            api_key=api_key,
            temperature=temperature,
            max_tokens=max_tokens,
            streaming=streaming,
        )

    primary = create(model_names[0])
    fallbacks = [create(name) for name in model_names[1:]]
    return primary.with_fallbacks(fallbacks) if fallbacks else primary
