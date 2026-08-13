"""
Synapse RAG — FastAPI Application Entry Point

Run with:
    uvicorn app.main:app --reload --port 8000

The Next.js frontend (localhost:3000) proxies RAG API calls to this service.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

from app.routes import chat, documents, retrieve

# Load .env from the project root (one level above python/)
load_dotenv(dotenv_path="../.env")
# Also try loading from cwd for flexibility
load_dotenv()

app = FastAPI(
    title="Synapse RAG Python Backend",
    description="RAG retrieval, embeddings, reranking, and LLM chat for legal contracts.",
    version="1.0.0",
)

# Allow Next.js dev server and production origin
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register all routes under /api/python prefix
app.include_router(chat.router, prefix="/api/python")
app.include_router(documents.router, prefix="/api/python")
app.include_router(retrieve.router, prefix="/api/python")


@app.get("/health")
def health():
    return {"status": "ok", "service": "synapse-rag-python"}
