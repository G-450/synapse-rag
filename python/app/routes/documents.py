"""
Documents API route — GET /api/python/documents
Fetches all unique documents from Qdrant and groups them by source corpus.
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from pathlib import Path
from qdrant_client import QdrantClient

router = APIRouter()

class DocumentMetadata(BaseModel):
    id: str
    filename: str
    source_corpus: str
    chunk_count: int


class DocumentResponse(BaseModel):
    documents: list[DocumentMetadata]
    grouped: dict[str, list[DocumentMetadata]]


@router.get("/documents", response_model=DocumentResponse)
def get_documents():
    """
    Retrieve all chunks from Qdrant using the scroll API,
    extract document metadata, deduplicate by document_id, 
    and group them by source_corpus.
    """
    try:
        from app.rag import get_qdrant_client
        client = get_qdrant_client()
        
        # We need to scroll through all points to aggregate metadata
        # In a real production system, this data would ideally live in a SQL db or a metadata cache.
        docs_map = {}
        
        offset = None
        while True:
            records, next_page_offset = client.scroll(
                collection_name="synapse_rag",
                limit=1000,
                with_payload=True,
                with_vectors=False,
                offset=offset,
            )
            
            for record in records:
                payload = record.payload or {}
                # In langchain-qdrant, metadata is usually stored under "metadata" 
                # or flattened into the payload depending on the version.
                metadata = payload.get("metadata", payload)
                
                doc_id = metadata.get("document_id")
                if not doc_id:
                    continue
                    
                if doc_id not in docs_map:
                    docs_map[doc_id] = {
                        "id": doc_id,
                        "filename": metadata.get("filename", "Unknown"),
                        "source_corpus": metadata.get("source_corpus", "Unknown"),
                        "chunk_count": 0
                    }
                
                docs_map[doc_id]["chunk_count"] += 1
                
            if next_page_offset is None:
                break
            offset = next_page_offset
            
        # Group by source_corpus
        corpora = {}
        for doc in docs_map.values():
            corpus = doc["source_corpus"]
            if corpus not in corpora:
                corpora[corpus] = []
            corpora[corpus].append(DocumentMetadata(**doc))
            
        # Sort alphabetically
        for corpus, docs in corpora.items():
            docs.sort(key=lambda d: d.filename)
            
        all_docs = list(docs_map.values())
        all_docs.sort(key=lambda d: d["filename"])
            
        return DocumentResponse(
            documents=[DocumentMetadata(**doc) for doc in all_docs],
            grouped=corpora
        )
        
    except Exception as e:
        print(f"Error fetching documents from Qdrant: {e}")
        return DocumentResponse(documents=[], grouped={})
