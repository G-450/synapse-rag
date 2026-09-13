"""
LangChain Embeddings for Synapse RAG.
Uses HuggingFaceEmbeddings for local sentence-transformers generation.
"""

from functools import lru_cache

from langchain_huggingface import HuggingFaceEmbeddings

@lru_cache(maxsize=1)
def get_embeddings():
    """Lazily load and reuse the 384-dimensional embedding model."""
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        encode_kwargs={"batch_size": 64, "normalize_embeddings": True},
    )
