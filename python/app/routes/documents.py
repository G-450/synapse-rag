"""
Documents API route — GET /api/python/documents
Fetches all unique documents from Qdrant and groups them by source corpus.
"""

import logging

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from pathlib import Path
from qdrant_client import QdrantClient

router = APIRouter()
logger = logging.getLogger(__name__)

class DocumentMetadata(BaseModel):
    id: str
    filename: str
    title: str
    source_corpus: str
    chunk_count: int
    section_count: int = 0


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
                        "title": metadata.get("title") or Path(metadata.get("filename", "Unknown")).stem,
                        "source_corpus": metadata.get("source_corpus", "Unknown"),
                        "chunk_count": 0,
                        "parent_ids": set(),
                    }
                
                docs_map[doc_id]["chunk_count"] += 1
                if metadata.get("parent_id"):
                    docs_map[doc_id]["parent_ids"].add(metadata["parent_id"])
                
            if next_page_offset is None:
                break
            offset = next_page_offset
            
        # Group by source_corpus
        corpora = {}
        normalized_docs = []
        for raw_doc in docs_map.values():
            parent_ids = raw_doc.pop("parent_ids")
            doc = {**raw_doc, "section_count": len(parent_ids)}
            normalized_docs.append(doc)
            corpus = doc["source_corpus"]
            if corpus not in corpora:
                corpora[corpus] = []
            corpora[corpus].append(DocumentMetadata(**doc))
            
        # Sort alphabetically
        for corpus, docs in corpora.items():
            docs.sort(key=lambda d: d.filename)
            
        all_docs = normalized_docs
        all_docs.sort(key=lambda d: d["filename"])
            
        return DocumentResponse(
            documents=[DocumentMetadata(**doc) for doc in all_docs],
            grouped=corpora
        )
        
    except Exception as e:
        logger.exception("Unable to fetch documents from Qdrant")
        raise HTTPException(status_code=503, detail="Document index unavailable.") from e
