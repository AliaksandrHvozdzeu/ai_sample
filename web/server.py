"""
FastAPI server: chat + 3D vault graph UI + RAG APIs in one process.

Run from project root:
  uvicorn web.server:app --host 127.0.0.1 --port 8000

URL layout:
  /rag/chat          — single page: chat + 3D graph (also default `/`)
  /vault/graph       — redirects to `/rag/chat` (same UI)
  /api/rag/health    — Chroma + LLM status
  /api/rag/stream    — SSE chat (legacy: /api/chat/stream, /api/health)

Lightweight graph-only server (no LLM): web.vault_server (default port 8001).

Environment:
  RAG_MODEL_ID   — Hugging Face model id (default: same as app.DEFAULT_LLM_ID)
  RAG_CONFIG     — optional path to YAML config (default: config.yaml if present)
"""

from __future__ import annotations

import json
import logging
import os
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, RedirectResponse, StreamingResponse
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# Import project RAG stack (expects working directory = repo root)
import app as rag


class ChatBody(BaseModel):
    message: str = Field(..., min_length=1, max_length=8000)
    show_sources: bool = False


class AppState:
    """Process-global handles set once at startup by _load_stack()."""

    vectorstore: Any = None
    tokenizer: Any = None
    pipe: Any = None
    model_id: str = ""
    startup_error: str | None = None


state = AppState()


def _load_stack() -> None:
    """Load YAML config, Chroma index, and quantized HF model into AppState."""
    cfg_raw = os.environ.get("RAG_CONFIG")
    cfg_path = Path(cfg_raw) if cfg_raw else rag.resolve_config_path(None)
    rag.apply_app_config(rag.load_app_config_file(cfg_path))
    if cfg_path and cfg_path.is_file():
        logger.info("Web: loaded config %s", cfg_path)
    logger.info(
        "Web: LLM generation max_new_tokens=%s do_sample=%s",
        rag.LLM_MAX_NEW_TOKENS,
        rag.LLM_DO_SAMPLE,
    )

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


# Served SPA and assets (single-page chat + 3D graph: index.html).
STATIC = Path(__file__).resolve().parent / "static"

app = FastAPI(title="Local RAG", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- JSON diagnostics for the browser / ops ---
@app.get("/api/rag/health")
@app.get("/api/health")
def rag_health() -> dict[str, Any]:
    idx_ok = rag.CHROMA_DIR.is_dir() and any(rag.CHROMA_DIR.iterdir())
    vault_ok = rag.VAULT_DIR is not None and rag.VAULT_DIR.is_dir()
    return {
        "ok": state.startup_error is None,
        "startup_error": state.startup_error,
        "model_id": state.model_id or None,
        "chroma_dir": str(rag.CHROMA_DIR),
        "index_present": idx_ok,
        "vault_configured": vault_ok,
    }


@app.get("/api/vault/graph")
def vault_graph():
    """Wikilink graph JSON for the 3D canvas (same payload as web.vault_server)."""
    return rag.vault_graph_api_payload()


@app.get("/api/vault/files")
def vault_files_list() -> dict[str, Any]:
    """Markdown paths currently under the vault directory (for the UI file panel)."""
    if rag.VAULT_DIR is None:
        return {
            "vault_configured": False,
            "files": [],
            "count": 0,
            "message": "paths.vault_dir is not set in config.yaml",
        }
    root = rag.VAULT_DIR.resolve()
    if not rag.VAULT_DIR.is_dir():
        return {
            "vault_configured": True,
            "vault_root": str(root),
            "files": [],
            "count": 0,
            "error": "vault_not_found",
            "detail": str(rag.VAULT_DIR),
        }
    paths = rag.list_vault_markdown_relpaths()
    return {
        "vault_configured": True,
        "vault_root": str(root),
        "files": paths,
        "count": len(paths),
    }


@app.post("/api/vault/reindex")
def vault_reindex() -> dict[str, Any]:
    """Rebuild Chroma from vault Markdown and reload the in-memory vector store."""
    if rag.VAULT_DIR is None:
        raise HTTPException(status_code=400, detail="paths.vault_dir is not set in config.yaml")
    if not rag.VAULT_DIR.is_dir():
        raise HTTPException(status_code=400, detail=f"Vault directory not found: {rag.VAULT_DIR}")
    if state.startup_error:
        raise HTTPException(status_code=503, detail=state.startup_error)

    docs = rag.load_all_documents_for_reindex()
    if not docs:
        raise HTTPException(
            status_code=400,
            detail=(
                "No documents to index. Set paths.vault_dir in config.yaml, "
                "add .md files with content, then reindex."
            ),
        )

    rag.release_vectorstore(state.vectorstore)
    state.vectorstore = None
    try:
        rag.build_vectorstore(docs, rag.CHROMA_DIR, clear=True)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    state.vectorstore = rag.load_vectorstore(rag.CHROMA_DIR)

    seen_files: set[str] = set()
    for d in docs:
        meta = d.metadata or {}
        p = meta.get("vault_rel_path") or meta.get("source_file")
        if p:
            seen_files.add(str(p))

    return {
        "ok": True,
        "chunks_indexed": len(docs),
        "files_indexed": len(seen_files),
        "indexed_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    }


@app.post("/api/vault/upload")
async def vault_upload_md(files: list[UploadFile] = File(...)) -> dict[str, Any]:
    """Save uploaded .md files into the vault root (basename only; safe names)."""
    if rag.VAULT_DIR is None:
        raise HTTPException(status_code=400, detail="paths.vault_dir is not set in config.yaml")
    vault = rag.VAULT_DIR
    if not vault.is_dir():
        raise HTTPException(status_code=400, detail=f"Vault directory not found: {vault}")

    vault_resolved = vault.resolve()
    saved: list[str] = []
    skipped: list[str] = []

    for f in files:
        raw = f.filename or ""
        base = Path(raw).name
        if not base or base in (".", ".."):
            skipped.append(raw or "(empty)")
            continue
        if not base.lower().endswith(".md"):
            skipped.append(base)
            continue
        if any(c in base for c in ("/", "\\")):
            skipped.append(base)
            continue

        target = (vault / base).resolve()
        try:
            target.relative_to(vault_resolved)
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid path: {base}") from None

        body = await f.read()
        target.write_bytes(body)
        saved.append(base)

    return {
        "ok": True,
        "saved": saved,
        "saved_count": len(saved),
        "skipped": skipped,
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
        # Always send retrieved sources first so the chat+graph UI can highlight vault nodes;
        # the client only shows the text list when the user enabled "Show sources".
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


# --- Browser chat: Server-Sent Events (sources event first, then token chunks) ---
@app.post("/api/rag/stream")
@app.post("/api/chat/stream")
def chat_stream(body: ChatBody):
    return _sse_chat(body)


# --- Static UI (index.html = chat panel + ForceGraph3D) and legacy URL redirects ---
@app.get("/")
def root_redirect():
    return RedirectResponse(url="/rag/chat", status_code=302)


@app.get("/graph")
def graph_legacy_redirect():
    return RedirectResponse(url="/rag/chat", status_code=302)


@app.get("/rag/chat")
def rag_chat_page():
    index = STATIC / "index.html"
    if not index.is_file():
        raise HTTPException(status_code=404, detail="Missing web/static/index.html")
    return FileResponse(index)


@app.get("/vault/graph")
def vault_graph_legacy_redirect():
    return RedirectResponse(url="/rag/chat", status_code=302)
