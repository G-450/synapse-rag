"""
Retrieve API route — POST /api/python/retrieve
Direct chunk retrieval with optional cross-encoder reranking.
Equivalent of src/app/api/retrieve/route.ts.
"""

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from app.rag import retrieve_chunks

router = APIRouter()


class RetrieveRequest(BaseModel):
    query: str
    documentId: str | None = None
    limit: int = 5
    rerank: bool = True


@router.post("/retrieve")
def retrieve(body: RetrieveRequest):
    if not body.query:
        raise HTTPException(status_code=400, detail="Query is required")

    chunks = retrieve_chunks(
        query=body.query,
        document_id=body.documentId,
        limit=body.limit,
        rerank=body.rerank,
    )
    return JSONResponse({"chunks": chunks})
