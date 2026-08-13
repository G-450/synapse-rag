"""
LangChain Document Compressor for Cross-Encoder reranking.
Wraps the sentence-transformers CrossEncoder into a LangChain BaseDocumentCompressor.
"""

from typing import Sequence
from langchain_core.documents import Document
from langchain_core.callbacks.manager import Callbacks
from langchain_core.documents.compressor import BaseDocumentCompressor
from sentence_transformers import CrossEncoder

# Singleton instance of the reranker model
_cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')


class CrossEncoderReranker(BaseDocumentCompressor):
    """
    Reranks documents using a CrossEncoder.
    Replaces the original similarity score with the cross-encoder score.
    """
    top_n: int = 5

    def compress_documents(
        self,
        documents: Sequence[Document],
        query: str,
        callbacks: Callbacks | None = None,
    ) -> Sequence[Document]:
        if not documents:
            return []

        # Prepare pairs for cross-encoder
        pairs = [[query, doc.page_content] for doc in documents]
        
        # Predict scores
        scores = _cross_encoder.predict(pairs)
        
        import math
        
        # Zip documents with scores
        doc_score_pairs = list(zip(documents, scores))
        
        # Sort by score descending
        doc_score_pairs.sort(key=lambda x: x[1], reverse=True)
        
        # Take top N and inject new score into metadata
        results = []
        for doc, score in doc_score_pairs[:self.top_n]:
            # Apply sigmoid to convert logit to [0, 1] probability
            prob = 1 / (1 + math.exp(-score))
            # Convert to standard float for JSON serialization
            doc.metadata["similarity"] = float(prob)
            results.append(doc)
            
        return results

def get_reranker(top_n: int = 5):
    """Returns a configured CrossEncoderReranker."""
    return CrossEncoderReranker(top_n=top_n)
