"""
Core RAG retrieval logic for Synapse RAG (LangChain + Qdrant version).

Implements a conversational RAG chain with:
  - Question contextualization using chat history
  - Session-based memory via RunnableWithMessageHistory
  - Document-scoped retrieval support
"""

import os
from pathlib import Path
from typing import Sequence

from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.http import models
from langchain_qdrant import QdrantVectorStore
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import InMemoryChatMessageHistory

from app.embeddings import get_embeddings
from app.cross_encoder import get_reranker

load_dotenv()

# ---------------------------------------------------------------------------
# Qdrant infrastructure (unchanged)
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
# Direct retrieval (still used by /api/python/retrieve endpoint)
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


# ---------------------------------------------------------------------------
# LLM & Retriever setup
# ---------------------------------------------------------------------------

llm = ChatGroq(
    model="llama-3.1-8b-instant",
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0.2,
    max_tokens=2048,
)

retriever = get_vectorstore().as_retriever(
    search_type="similarity",
    search_kwargs={"k": 5},
)


def format_doc(docs):
    """Format retrieved LangChain Documents into a single context string."""
    return "\n".join(doc.page_content for doc in docs)


# ---------------------------------------------------------------------------
# Basic RAG chain (no chat history)
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are Synapse RAG, an expert legal contract analyst.\n"
    "You will be given relevant excerpts from legal contracts (the \"Context\") "
    "and a user question.\n\n"
    "CRITICAL RULES:\n"
    "1. Answer ONLY using information found in the Context. Do NOT use outside knowledge.\n"
    "2. If the Context does not contain the information needed, respond with: "
    "\"I cannot answer this question based on the provided contract excerpts.\"\n"
    "3. Be precise. Quote exact clauses or passages when relevant, wrapping quotes in quotation marks.\n"
    "4. When citing information, mention the source document name.\n"
    "5. For multi-document queries, structure your response clearly by document.\n"
    "6. Use professional legal analysis tone."
)

prompt = ChatPromptTemplate.from_template(
    SYSTEM_PROMPT + "\n\nContext:{context}\n\nQuestion:{question}"
)

rag_chain = (
    {"context": retriever | format_doc, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)


# ---------------------------------------------------------------------------
# Conversational RAG chain (with chat history)
# ---------------------------------------------------------------------------

# Step 1: Contextualize the user's question using chat history
context_prompt = (
    "Given a chat history and the latest user question "
    "which might reference context in the chat history, "
    "formulate a standalone question which can be understood "
    "without the chat history. Do NOT answer the question, "
    "just reformulate it if needed and otherwise return it as is."
)

contextualize_prompt = ChatPromptTemplate.from_messages([
    ("system", context_prompt),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])


def get_context(inputs):
    """
    Reformulates the question if chat history is present, then retrieves
    relevant documents from the vector store.
    """
    if inputs.get("chat_history"):
        rewrite_chain = contextualize_prompt | llm | StrOutputParser()
        question = rewrite_chain.invoke(inputs)
    else:
        question = inputs["input"]
    docs = retriever.invoke(question)
    return format_doc(docs)


# Step 2: QA prompt with chat history support
qa_prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT + "\n\nContext: {context}"),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])

# Step 3: Full conversational chain
conversation_chain = (
    RunnablePassthrough.assign(context=RunnableLambda(get_context))
    | qa_prompt
    | llm
    | StrOutputParser()
)

# Step 4: Session-based chat history store
store = {}


def get_session_history(session_id: str):
    """Returns (or creates) an InMemoryChatMessageHistory for the given session."""
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]


conversational_rag_chain = RunnableWithMessageHistory(
    conversation_chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
)
