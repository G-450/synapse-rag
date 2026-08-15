"""
Chat API route — POST /api/python/chat
Conversational RAG with chat history, implementing the AI SDK Data Stream Protocol.
"""

import json
import os
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import InMemoryChatMessageHistory

from app.rag import (
    retrieve_chunks,
    retriever,
    format_doc,
    llm,
    SYSTEM_PROMPT,
    get_session_history,
    conversational_rag_chain,
)

router = APIRouter()


from typing import Any

class Message(BaseModel):
    role: str
    content: Any = None
    parts: Any = None

class ChatRequest(BaseModel):
    messages: list[Message]
    documentId: str | None = None
    sessionId: str | None = None

def _extract_query(messages: list[Message]) -> str:
    """Get the text of the last user message."""
    if not messages:
        return ""
    latest = messages[-1]

    # Handle standard content string
    if isinstance(latest.content, str):
        return latest.content
    # Handle content array
    elif isinstance(latest.content, list):
        return "".join(c.get("text", "") for c in latest.content if isinstance(c, dict) and c.get("type") == "text")
    # Handle AI SDK v4 parts array
    elif isinstance(latest.parts, list):
        return "".join(p.get("text", "") for p in latest.parts if isinstance(p, dict) and p.get("type") == "text")
    
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

    # Use a session ID for chat history (from request, or default)
    session_id = body.sessionId or "default"

    # Retrieve chunks for citations (separate from the chain's internal retrieval)
    citations: list[dict] = []
    if body.documentId:
        chunks = retrieve_chunks(user_query, document_id=body.documentId, limit=5)
    else:
        chunks = retrieve_chunks(user_query, limit=5)

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

    # Stream using the conversational RAG chain
    config = {"configurable": {"session_id": session_id}}

    langchain_stream = conversational_rag_chain.astream(
        {"input": user_query},
        config=config,
    )

    return StreamingResponse(
        _ai_sdk_stream(langchain_stream, citations),
        media_type="text/plain; charset=utf-8",
        headers={
            "X-Vercel-AI-Data-Stream": "v1",
            "Cache-Control": "no-cache",
        },
    )
