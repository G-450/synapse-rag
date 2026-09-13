"""
Chat API route — POST /api/python/chat
Conversational RAG with the stateless AI SDK v7 UI-message stream protocol.
"""

import json
import uuid
from typing import Any
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

from app.rag import (
    retrieve_chunks,
    fan_out_retrieve,
    format_context,
    format_fan_out_context,
    SYSTEM_PROMPT,
    get_llm,
)

router = APIRouter()


class Message(BaseModel):
    role: str
    content: Any = None
    parts: Any = None


class ChatRequest(BaseModel):
    messages: list[Message]
    documentId: str | None = None


def _extract_message_content(m: Message) -> str:
    """Extract plain text string from message content or AI SDK parts."""
    if isinstance(m.content, str):
        return m.content
    elif isinstance(m.content, list):
        return "".join(
            c.get("text", "") for c in m.content if isinstance(c, dict) and c.get("type") == "text"
        )
    elif isinstance(m.parts, list):
        return "".join(
            p.get("text", "") for p in m.parts if isinstance(p, dict) and p.get("type") == "text"
        )
    return ""


def _extract_query(messages: list[Message]) -> str:
    """Get the text of the latest user message."""
    for m in reversed(messages):
        if m.role == "user":
            content = _extract_message_content(m)
            if content.strip():
                return content
    return ""


async def _ai_sdk_stream(langchain_stream, citations: list[dict]):
    """Convert LangChain chunks to the AI SDK v7 SSE UI-message protocol."""
    message_id = str(uuid.uuid4())
    text_id = str(uuid.uuid4())
    yield f"data: {json.dumps({'type': 'start', 'messageId': message_id})}\n\n"
    yield f"data: {json.dumps({'type': 'text-start', 'id': text_id})}\n\n"
    yield f"data: {json.dumps({'type': 'data-citations', 'data': citations, 'transient': True}, default=str)}\n\n"

    async for chunk in langchain_stream:
        # chunk is an AIMessageChunk or string
        token = ""
        if hasattr(chunk, "content"):
            token = chunk.content
        elif isinstance(chunk, str):
            token = chunk

        if token:
            yield f"data: {json.dumps({'type': 'text-delta', 'id': text_id, 'delta': token})}\n\n"

    yield f"data: {json.dumps({'type': 'text-end', 'id': text_id})}\n\n"
    yield f"data: {json.dumps({'type': 'finish', 'finishReason': 'stop'})}\n\n"
    yield "data: [DONE]\n\n"


@router.post("/chat")
async def chat(body: ChatRequest):
    user_query = _extract_query(body.messages)
    if not user_query:
        raise HTTPException(status_code=400, detail="Query is empty")

    # Scoped retrieval vs multi-document fan-out retrieval
    citations: list[dict] = []
    if body.documentId:
        chunks = retrieve_chunks(user_query, document_id=body.documentId, limit=5, rerank=True)
        context_text = format_context(chunks)
        citations = [
            {
                "chunk_id": c["id"],
                "document_id": c["document_id"],
                "filename": c.get("filename", ""),
                "content": c["content"],
                "similarity": c.get("similarity", 0),
                "parent_id": c.get("parent_id", ""),
                "parent_header": c.get("parent_header", ""),
                "parent_content": c.get("parent_content", ""),
                "section_number": c.get("section_number", ""),
                "section_title": c.get("section_title", ""),
                "start_char": c.get("start_char"),
                "end_char": c.get("end_char"),
            }
            for c in chunks
        ]
    else:
        doc_groups = fan_out_retrieve(user_query, top_docs=3, chunks_per_doc=3)
        context_text = format_fan_out_context(doc_groups)
        citations = [
            {
                "chunk_id": c["id"],
                "document_id": c["document_id"],
                "filename": group["filename"],
                "content": c["content"],
                "similarity": c.get("similarity", 0),
                "parent_id": c.get("parent_id", ""),
                "parent_header": c.get("parent_header", ""),
                "parent_content": c.get("parent_content", ""),
                "section_number": c.get("section_number", ""),
                "section_title": c.get("section_title", ""),
                "start_char": c.get("start_char"),
                "end_char": c.get("end_char"),
            }
            for group in doc_groups
            for c in group["chunks"]
        ]

    augmented_system = (
        SYSTEM_PROMPT
        + "\n\n=== CONTEXT (Retrieved Contract Excerpts) ===\n\n"
        + context_text
    )

    # Reconstruct message history for LangChain dynamically from frontend payload
    model_messages = [SystemMessage(content=augmented_system)]
    for m in body.messages:
        text = _extract_message_content(m)
        if not text:
            continue

        if m.role == "user":
            model_messages.append(HumanMessage(content=text))
        elif m.role == "assistant":
            model_messages.append(AIMessage(content=text))
        elif m.role == "system":
            model_messages.append(SystemMessage(content=text))
        else:
            model_messages.append(HumanMessage(content=text))

    # Initialize LLM lazily
    try:
        llm = get_llm(streaming=True)
    except ValueError as e:
        raise HTTPException(status_code=500, detail=str(e))

    # Stream using async generator
    langchain_stream = llm.astream(model_messages)

    return StreamingResponse(
        _ai_sdk_stream(langchain_stream, citations),
        media_type="text/event-stream",
        headers={
            "X-Vercel-AI-UI-Message-Stream": "v1",
            "Cache-Control": "no-cache",
        },
    )
