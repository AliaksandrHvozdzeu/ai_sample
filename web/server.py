"""
FastAPI server: minimal chat UI + SSE streaming for RAG answers.

Run from project root:
  uvicorn web.server:app --host 127.0.0.1 --port 8000

Environment:
  RAG_MODEL_ID   — Hugging Face model id (default: same as app.DEFAULT_LLM_ID)
  RAG_CONFIG     — optional path to YAML config (default: config.yaml if present)
"""

from __future__ import annotations

import json
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# Import project RAG stack (expects working directory = repo root)
import app as rag


class ChatBody(BaseModel):
    message: str = Field(..., min_length=1, max_length=8000)
    show_sources: bool = False


class AppState:
    vectorstore: Any = None
    tokenizer: Any = None
    pipe: Any = None
    model_id: str = ""
    startup_error: str | None = None


state = AppState()


def _load_stack() -> None:
    cfg_raw = os.environ.get("RAG_CONFIG")
    cfg_path = Path(cfg_raw) if cfg_raw else rag.resolve_config_path(None)
    rag.apply_app_config(rag.load_app_config_file(cfg_path))
    if cfg_path and cfg_path.is_file():
        logger.info("Web: loaded config %s", cfg_path)

    model_id = os.environ.get("RAG_MODEL_ID", rag.DEFAULT_LLM_ID)
    if model_id not in (rag.DEFAULT_LLM_ID, rag.ALT_LLM_ID):
        raise ValueError(f"RAG_MODEL_ID must be one of: {rag.DEFAULT_LLM_ID}, {rag.ALT_LLM_ID}")

    state.vectorstore = rag.load_vectorstore(rag.CHROMA_DIR)
    logger.info("Web: vector store ready at %s", rag.CHROMA_DIR)
    state.tokenizer, state.pipe = rag.build_llm_pipeline(model_id)
    state.model_id = model_id
    logger.info("Web: LLM ready (%s)", model_id)


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        _load_stack()
    except Exception as e:
        state.startup_error = str(e)
        logger.exception("Startup failed: %s", e)
    yield


ROOT = Path(__file__).resolve().parent.parent
STATIC = Path(__file__).resolve().parent / "static"

app = FastAPI(title="Local RAG", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health() -> dict[str, Any]:
    idx_ok = rag.CHROMA_DIR.is_dir() and any(rag.CHROMA_DIR.iterdir())
    return {
        "ok": state.startup_error is None,
        "startup_error": state.startup_error,
        "model_id": state.model_id or None,
        "chroma_dir": str(rag.CHROMA_DIR),
        "index_present": idx_ok,
    }


def _sse_chat(body: ChatBody):
    if state.startup_error:
        raise HTTPException(status_code=503, detail=state.startup_error)
    if state.vectorstore is None or state.tokenizer is None or state.pipe is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    question = body.message.strip()
    docs = rag.retrieve_context_documents(state.vectorstore, question)
    ctx = rag.documents_context_string(docs)
    prompt_text = rag.build_chat_prompt_text(state.tokenizer, question, ctx)

    def events():
        if body.show_sources:
            payload = json.dumps(
                {"type": "sources", "items": rag.serialize_sources_for_ui(docs)},
                ensure_ascii=False,
            )
            yield f"data: {payload}\n\n"
        try:
            for fragment in rag.iter_answer_tokens_stream(state.tokenizer, state.pipe, prompt_text):
                if fragment:
                    piece = json.dumps(
                        {"type": "token", "text": fragment},
                        ensure_ascii=False,
                    )
                    yield f"data: {piece}\n\n"
        except Exception as e:
            err = json.dumps({"type": "error", "detail": str(e)}, ensure_ascii=False)
            yield f"data: {err}\n\n"
            return
        done = json.dumps({"type": "done"}, ensure_ascii=False)
        yield f"data: {done}\n\n"

    return StreamingResponse(events(), media_type="text/event-stream")


@app.post("/api/chat/stream")
def chat_stream(body: ChatBody):
    return _sse_chat(body)


@app.get("/")
def index_page():
    index = STATIC / "index.html"
    if not index.is_file():
        raise HTTPException(status_code=404, detail="Missing web/static/index.html")
    return FileResponse(index)

