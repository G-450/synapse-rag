"""
LangChain Embeddings for Synapse RAG.
Uses HuggingFaceEmbeddings for local sentence-transformers generation.
"""

from langchain_huggingface import HuggingFaceEmbeddings

# Singleton instance of the embedding model
embeddings_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

def get_embeddings():
    """Returns the LangChain embeddings instance."""
    return embeddings_model
