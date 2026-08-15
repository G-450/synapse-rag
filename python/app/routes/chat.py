"""
Chat API route — POST /api/python/chat
Conversational RAG with stateless AI SDK Data Stream Protocol.
"""

import json
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
    """
    Generator that converts a LangChain astream into the
    AI SDK Data Stream Protocol expected by useChat.

    Protocol:
      0:"<escaped text>"  — text chunk
      2:[{...}]           — data array (custom data, citations go here)
      d:{...}             — finish delimiter
    """
    first_chunk_sent = False
    citations_sent = False
    citations_payload = json.dumps(citations, default=str)

    async for chunk in langchain_stream:
        # chunk is an AIMessageChunk or string
        token = ""
        if hasattr(chunk, "content"):
            token = chunk.content
        elif isinstance(chunk, str):
            token = chunk

        if token:
            encoded = json.dumps(token)
            yield f"0:{encoded}\n"

            # Send citations as a data event right after the first text token
            if not first_chunk_sent:
                first_chunk_sent = True
                yield f"2:[{{\"citations\":{citations_payload}}}]\n"
                citations_sent = True

    # If LLM returned nothing at all, still emit citations so sources show
    if not citations_sent:
        yield f"2:[{{\"citations\":{citations_payload}}}]\n"

    # Finish delimiter
    yield 'd:{"finishReason":"stop","usage":{}}\n'


@router.post("/chat")
async def chat(body: ChatRequest):
    user_query = _extract_query(body.messages)
    if not user_query:
        raise HTTPException(status_code=400, detail="Query is empty")

    # Scoped retrieval vs multi-document fan-out retrieval
    citations: list[dict] = []
    if body.documentId:
        chunks = retrieve_chunks(user_query, document_id=body.documentId, limit=5)
        context_text = format_context(chunks)
        citations = [
            {
                "chunk_id": c["id"],
                "document_id": c["document_id"],
                "filename": c.get("filename", ""),
                "content": c["content"],
                "similarity": c.get("similarity", 0),
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
        media_type="text/plain; charset=utf-8",
        headers={
            "X-Vercel-AI-Data-Stream": "v1",
            "Cache-Control": "no-cache",
        },
    )
