"""
Chat API route — POST /api/python/chat
RAG retrieval + LangChain ChatGroq streaming, implementing the AI SDK Data Stream Protocol.
"""

import json
import os
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

from app.rag import (
    retrieve_chunks,
    fan_out_retrieve,
    format_context,
    format_fan_out_context,
)

router = APIRouter()

SYSTEM_PROMPT = """You are Synapse RAG, an expert legal contract analyst.
You will be given relevant excerpts from legal contracts (the "Context") and a user question.

CRITICAL RULES:
1. Answer ONLY using information found in the Context. Do NOT use outside knowledge.
2. If the Context does not contain the information needed, respond with: "I cannot answer this question based on the provided contract excerpts."
3. Be precise. Quote exact clauses or passages when relevant, wrapping quotes in quotation marks.
4. When citing information, mention the source document name.
5. For multi-document queries, structure your response clearly by document.
6. Use professional legal analysis tone."""


from typing import Any

class Message(BaseModel):
    role: str
    content: Any


class ChatRequest(BaseModel):
    messages: list[Message]
    documentId: str | None = None


def _extract_query(messages: list[Message]) -> str:
    """Get the text of the last user message."""
    if not messages:
        return ""
    latest = messages[-1]
    
    if isinstance(latest.content, str):
        return latest.content
    elif isinstance(latest.content, list):
        return "".join(c.get("text", "") for c in latest.content if isinstance(c, dict) and c.get("type") == "text")
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
        # chunk is an AIMessageChunk
        if chunk.content:
            encoded = json.dumps(chunk.content)
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

    # Retrieve context
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

    # Build model messages
    model_messages = [SystemMessage(content=augmented_system)]
    for m in body.messages:
        content_text = m.content if isinstance(m.content, str) else "".join(c.get("text", "") for c in m.content if isinstance(c, dict) and c.get("type") == "text")
        if m.role == "user":
            model_messages.append(HumanMessage(content=content_text))
        elif m.role == "assistant":
            model_messages.append(AIMessage(content=content_text))
        else:
            # Fallback
            model_messages.append(HumanMessage(content=content_text))

    # Initialize ChatGroq
    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        api_key=os.environ["GROQ_API_KEY"],
        temperature=0.2,
        max_tokens=2048,
        streaming=True,
    )

    # We use astream() for async streaming with FastAPI
    langchain_stream = llm.astream(model_messages)

    return StreamingResponse(
        _ai_sdk_stream(langchain_stream, citations),
        media_type="text/plain; charset=utf-8",
        headers={
            "X-Vercel-AI-Data-Stream": "v1",
            "Cache-Control": "no-cache",
        },
    )
