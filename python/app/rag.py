"""
Core RAG retrieval logic for Synapse RAG (LangChain + Qdrant version).
"""

import os
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


def get_qdrant_client():
    global _qdrant_client
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

def retrieve_chunks(
    query: str,
    document_id: str | None = None,
    limit: int = 5,
    rerank: bool = False,
) -> list[dict]:
    """
    Retrieve the most relevant chunks using Qdrant similarity search.
    Optionally scoped to a single document and/or reranked with cross-encoder.
    """
    fetch_limit = max(limit * 4, 20) if rerank else limit
    vectorstore = get_vectorstore()

    filter_kwargs = {}
    if document_id:
        filter_kwargs["filter"] = models.Filter(
            must=[
                models.FieldCondition(
                    key="metadata.document_id",
                    match=models.MatchValue(value=document_id),
                )
            ]
        )

    # Perform similarity search with score
    docs_with_scores = vectorstore.similarity_search_with_score(
        query=query,
        k=fetch_limit,
        **filter_kwargs
    )

    documents = []
    for doc, score in docs_with_scores:
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
            "content": doc.page_content,
            "filename": doc.metadata.get("filename", ""),
            "source_corpus": doc.metadata.get("source_corpus", ""),
            "similarity": float(doc.metadata.get("similarity", 0.0))
        }
        for doc in documents
    ]


def fan_out_retrieve(
    query: str,
    top_docs: int = 3,
    chunks_per_doc: int = 3,
) -> list[dict]:
    """
    Fan-out multi-document retrieval.
    Fetches a large pool of chunks, groups by document, and selects top docs/chunks.
    """
    vectorstore = get_vectorstore()

    # Fetch a large pool to ensure we hit multiple documents
    docs_with_scores = vectorstore.similarity_search_with_score(
        query=query,
        k=30
    )

    # Group by document_id
    grouped = {}
    for doc, score in docs_with_scores:
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
            "content": doc.page_content,
            "filename": doc.metadata.get("filename", ""),
            "source_corpus": doc.metadata.get("source_corpus", ""),
            "similarity": float(score)
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

SYSTEM_PROMPT = """You are Synapse RAG, an expert legal contract analyst.
You will be given relevant excerpts from legal contracts (the "Context") and a user question.

CRITICAL RULES:
1. Answer ONLY using information found in the Context. Do NOT use outside knowledge.
2. If the Context does not contain the information needed, respond with: "I cannot answer this question based on the provided contract excerpts."
3. Be precise. Quote exact clauses or passages when relevant, wrapping quotes in quotation marks.
4. When citing information, mention the source document name.
5. For multi-document queries, structure your response clearly by document.
6. Use professional legal analysis tone."""


def format_context(chunks: list[dict]) -> str:
    """Format retrieved chunks into a context string for the LLM prompt."""
    parts = []
    for chunk in chunks:
        source = chunk.get("filename") or chunk.get("document_id", "")
        relevance = float(chunk.get("similarity", 0))
        parts.append(
            f"[Source: {source} | Score: {relevance:.3f}]\n{chunk['content']}"
        )
    return "\n\n---\n\n".join(parts)


def format_fan_out_context(doc_groups: list[dict]) -> str:
    """Format fan-out results grouped by document."""
    sections = []
    for group in doc_groups:
        header = f"=== Document: {group['filename']} ({group['source_corpus']}) ==="
        chunk_texts = []
        for i, chunk in enumerate(group["chunks"], 1):
            relevance = float(chunk.get("similarity", 0))
            chunk_texts.append(
                f"[Snippet {i} | Score: {relevance:.3f}]\n{chunk['content']}"
            )
        sections.append(header + "\n\n" + "\n\n".join(chunk_texts))
    return ("\n\n" + "=" * 60 + "\n\n").join(sections)


# ---------------------------------------------------------------------------
# LLM Initialization (Lazy / Function Factory)
# ---------------------------------------------------------------------------

def get_llm(
    streaming: bool = True,
    temperature: float = 0.2,
    max_tokens: int = 2048,
) -> ChatGroq:
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
    return ChatGroq(
        model="llama-3.1-8b-instant",
        api_key=api_key,
        temperature=temperature,
        max_tokens=max_tokens,
        streaming=streaming,
    )
