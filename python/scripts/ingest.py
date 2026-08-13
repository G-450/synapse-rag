"""
Synapse RAG — Ingestion Pipeline (Python/LangChain + Qdrant)

Reads data/legalbench.json, creates LangChain Document objects,
and ingests them into a local Qdrant instance using HuggingFaceEmbeddings.

Usage:
    python scripts/ingest.py
"""

import json
import sys
import uuid
import shutil
from pathlib import Path

# Allow running from project root or from python/
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv

load_dotenv(dotenv_path=Path(__file__).parent.parent.parent / ".env")
load_dotenv()

from langchain_core.documents import Document
from langchain_qdrant import QdrantVectorStore
from app.embeddings import get_embeddings


def main():
    data_path = Path(__file__).parent.parent.parent / "data" / "legalbench.json"
    if not data_path.exists():
        print(f"ERROR: Dataset not found at {data_path}")
        sys.exit(1)

    with open(data_path, "r", encoding="utf-8") as f:
        qa_pairs = json.load(f)

    print(f"Loaded {len(qa_pairs)} QA pairs from LegalBench-RAG.")

    # Group unique snippets by document (corpus_file)
    docs_map: dict[str, set] = {}
    for pair in qa_pairs:
        meta = pair["metadata"]
        file = meta["corpus_file"]
        if file not in docs_map:
            docs_map[file] = set()
        for snippet in meta.get("snippets", []):
            text = snippet.get("answer", "").strip()
            if text:
                docs_map[file].add(text)

    print(f"Found {len(docs_map)} unique contracts in the dataset sample. Preparing documents...")

    langchain_docs = []
    
    for corpus_file, unique_snippets in docs_map.items():
        document_id = str(uuid.uuid4())
        filename = corpus_file.split("/")[-1] or corpus_file

        for text in unique_snippets:
            doc = Document(
                page_content=text,
                metadata={
                    "document_id": document_id,
                    "filename": filename,
                    "source_corpus": "legalbench-rag",
                },
                id=str(uuid.uuid4()),
            )
            langchain_docs.append(doc)

    print(f"Prepared {len(langchain_docs)} chunks for ingestion.")

    qdrant_dir = Path(__file__).parent.parent / "qdrant_db"
    
    # Clear existing DB to prevent duplicates on re-run
    if qdrant_dir.exists():
        print(f"Removing existing Qdrant directory: {qdrant_dir}")
        shutil.rmtree(qdrant_dir)

    print("Initializing Qdrant and computing embeddings (this may take a minute)...")
    embeddings = get_embeddings()
    
    QdrantVectorStore.from_documents(
        documents=langchain_docs,
        embedding=embeddings,
        path=str(qdrant_dir),
        collection_name="synapse_rag",
    )
    
    print(f"\n✅ Ingestion Complete! Inserted {len(langchain_docs)} chunks into Qdrant at {qdrant_dir}.")


if __name__ == "__main__":
    main()
