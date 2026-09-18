"""Native contract upload and structural ingestion endpoint."""

from pathlib import Path
import logging
import re
import uuid

from fastapi import APIRouter, File, HTTPException, UploadFile
from langchain_core.documents import Document
from qdrant_client.http import models

from app.chunking import ParentChildSplitter
from app.parsers.base import DocumentExtractionError, extract_document
from app.rag import get_qdrant_client, get_vectorstore


router = APIRouter()
logger = logging.getLogger(__name__)
ALLOWED_EXTENSIONS = {".pdf", ".docx", ".txt", ".md"}
MAX_UPLOAD_BYTES = 25 * 1024 * 1024
UPLOAD_DIR = Path(__file__).resolve().parents[3] / "data" / "uploads"


def _safe_filename(filename: str) -> str:
    name = Path(filename).name
    return re.sub(r"[^A-Za-z0-9._ -]", "_", name).strip(" .") or "contract"


def index_children(children, source_corpus: str = "user-upload", batch_size: int = 64) -> int:
    """Embed and atomically insert child chunks in bounded batches."""
    vectorstore = get_vectorstore()
    inserted_ids: list[str] = []
    inserted = 0
    try:
        for start in range(0, len(children), batch_size):
            batch = children[start:start + batch_size]
            documents = [
                Document(
                    id=child.id,
                    page_content=child.content,
                    metadata=child.metadata(source_corpus=source_corpus),
                )
                for child in batch
            ]
            vectorstore.add_documents(documents=documents, ids=[child.id for child in batch])
            inserted += len(documents)
            inserted_ids.extend(child.id for child in batch)
    except Exception:
        if inserted_ids:
            try:
                get_qdrant_client().delete(
                    collection_name="synapse_rag",
                    points_selector=models.PointIdsList(points=inserted_ids),
                    wait=True,
                )
            except Exception:
                logger.exception("Failed to roll back a partial contract ingestion")
        raise
    return inserted


@router.post("/upload")
async def upload_contract(file: UploadFile = File(...)):
    original_name = _safe_filename(file.filename or "")
    extension = Path(original_name).suffix.lower()
    if extension not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=415, detail="Unsupported file type. Upload a PDF, DOCX, TXT, or MD file.")

    document_id = str(uuid.uuid4())
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    destination = UPLOAD_DIR / f"{document_id}_{original_name}"
    size = 0
    try:
        with destination.open("wb") as output:
            while chunk := await file.read(1024 * 1024):
                size += len(chunk)
                if size > MAX_UPLOAD_BYTES:
                    raise HTTPException(status_code=413, detail="File exceeds the 25 MB upload limit.")
                output.write(chunk)

        extracted = extract_document(destination)
        _, children = ParentChildSplitter().split(extracted.text, document_id, original_name)
        if not children:
            raise DocumentExtractionError("No indexable contract text was found.")
        chunks_created = index_children(children)
    except HTTPException:
        destination.unlink(missing_ok=True)
        raise
    except DocumentExtractionError as exc:
        destination.unlink(missing_ok=True)
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        destination.unlink(missing_ok=True)
        logger.exception("Contract ingestion failed for %s", original_name)
        raise HTTPException(status_code=500, detail="Contract ingestion failed.") from exc
    finally:
        await file.close()

    return {
        "status": "success",
        "document_id": document_id,
        "filename": original_name,
        "chunks_created": chunks_created,
    }
