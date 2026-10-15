"""
Documents API route — GET /api/python/documents
Fetches all unique documents from Qdrant and groups them by source corpus.
"""

import logging

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from pathlib import Path
from qdrant_client import QdrantClient
from qdrant_client.http import models

router = APIRouter()
logger = logging.getLogger(__name__)

COLLECTION_NAME = "synapse_rag"
# Only user uploads are removable; benchmark corpora require a full re-ingest to restore.
DELETABLE_CORPORA = {"user-upload"}

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


class DeleteDocumentsRequest(BaseModel):
    documentIds: list[str] = []
    corpus: str | None = None


class DeleteDocumentsResponse(BaseModel):
    deleted: list[str]
    chunks_removed: int


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
                collection_name=COLLECTION_NAME,
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


def _collect_document_corpora(client: QdrantClient, document_ids: set[str]) -> dict[str, str]:
    """Map each requested document_id to its source corpus by scanning the index."""
    corpora: dict[str, str] = {}
    offset = None
    while True:
        records, next_page_offset = client.scroll(
            collection_name=COLLECTION_NAME,
            limit=1000,
            with_payload=True,
            with_vectors=False,
            offset=offset,
        )
        for record in records:
            payload = record.payload or {}
            metadata = payload.get("metadata", payload)
            doc_id = metadata.get("document_id")
            if doc_id in document_ids and doc_id not in corpora:
                corpora[doc_id] = metadata.get("source_corpus", "Unknown")
        if next_page_offset is None or len(corpora) == len(document_ids):
            break
        offset = next_page_offset
    return corpora


def _count_chunks(client: QdrantClient, document_ids: list[str]) -> int:
    from app.rag import build_document_filter

    document_filter = build_document_filter(document_ids)
    if document_filter is None:
        return 0
    return client.count(
        collection_name=COLLECTION_NAME,
        count_filter=document_filter,
        exact=True,
    ).count


def _remove_upload_files(document_ids: list[str]) -> None:
    """Drop the stored originals for deleted uploads; a leftover file is not fatal."""
    from app.routes.upload import UPLOAD_DIR

    for doc_id in document_ids:
        for stored in UPLOAD_DIR.glob(f"{doc_id}_*"):
            try:
                stored.unlink(missing_ok=True)
            except OSError:
                logger.warning("Could not remove the stored upload %s", stored)


@router.delete("/documents", response_model=DeleteDocumentsResponse)
def delete_documents(body: DeleteDocumentsRequest):
    """
    Remove uploaded documents from the index: every chunk is dropped from Qdrant
    and the stored original file is deleted. Pass documentIds to delete a specific
    set, or corpus="user-upload" to clear every upload at once.
    """
    from app.rag import build_document_filter, get_qdrant_client

    requested = list(dict.fromkeys(doc_id for doc_id in body.documentIds if doc_id))
    if not requested and not body.corpus:
        raise HTTPException(status_code=400, detail="Specify documentIds or a corpus to delete.")
    if body.corpus and body.corpus not in DELETABLE_CORPORA:
        raise HTTPException(status_code=403, detail=f"The {body.corpus} corpus cannot be deleted.")

    try:
        client = get_qdrant_client()
        corpora = _collect_document_corpora(client, set(requested))

        if body.corpus:
            index = get_documents()
            requested = list(
                dict.fromkeys(
                    [*requested, *(doc.id for doc in index.grouped.get(body.corpus, []))]
                )
            )
            corpora.update({doc.id: body.corpus for doc in index.grouped.get(body.corpus, [])})

        unknown = [doc_id for doc_id in requested if doc_id not in corpora]
        if unknown:
            raise HTTPException(
                status_code=404,
                detail=f"No indexed document found for {unknown[0]}.",
            )

        protected = [doc_id for doc_id in requested if corpora[doc_id] not in DELETABLE_CORPORA]
        if protected:
            raise HTTPException(
                status_code=403,
                detail=(
                    f"{corpora[protected[0]]} is a benchmark corpus and cannot be deleted. "
                    "Only uploaded contracts can be removed."
                ),
            )

        if not requested:
            return DeleteDocumentsResponse(deleted=[], chunks_removed=0)

        chunks_removed = _count_chunks(client, requested)
        client.delete(
            collection_name=COLLECTION_NAME,
            points_selector=models.FilterSelector(filter=build_document_filter(requested)),
            wait=True,
        )
        _remove_upload_files(requested)
        return DeleteDocumentsResponse(deleted=requested, chunks_removed=chunks_removed)

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Unable to delete documents from Qdrant")
        raise HTTPException(status_code=503, detail="Document index unavailable.") from e
