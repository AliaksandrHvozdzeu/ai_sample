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
  RAG_CHAT_DB    — optional absolute path to SQLite chat DB (overrides config chat.db_path)
"""

from __future__ import annotations

import json
import logging
import os
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, RedirectResponse, StreamingResponse
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# Import project RAG stack (expects working directory = repo root)
import app as rag
from web.chat_store import ChatStore

# Persistent chat (SQLite). Initialized in _load_stack() after config is applied.
chat_store: ChatStore | None = None


class ChatBody(BaseModel):
    message: str = Field(..., min_length=1, max_length=8000)
    show_sources: bool = False
    session_id: str | None = Field(
        default=None,
        max_length=64,
        description="Ongoing chat session (UUID). Omit to start a new session.",
    )
    # Obsidian-native retrieval filters (optional)
    filter_note: str | None = Field(
        default=None,
        max_length=512,
        description="Vault-relative path to a single .md — restrict retrieval to that note.",
    )
    filter_tag: str | None = Field(
        default=None,
        max_length=128,
        description="Substring match against YAML tags embedded in chunk metadata.",
    )
    expand_wikilinks: bool = Field(
        default=True,
        description="After vector retrieval, add one-hop [[wikilink]] neighbor chunks.",
    )
    suggest_vault_edit: bool = Field(
        default=False,
        description="Run a second short generation suggesting JSON path + markdown patch.",
    )


class ChatExportBody(BaseModel):
    session_id: str = Field(..., min_length=8, max_length=64)


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
    global chat_store
    cfg_raw = os.environ.get("RAG_CONFIG")
    cfg_path = Path(cfg_raw) if cfg_raw else rag.resolve_config_path(None)
    rag.apply_app_config(rag.load_app_config_file(cfg_path))
    chat_db_raw = os.environ.get("RAG_CHAT_DB")
    if chat_db_raw:
        rag.CHAT_DB_PATH = Path(chat_db_raw).expanduser().resolve()
    chat_store = ChatStore(rag.CHAT_DB_PATH)
    logger.info("Web: chat history database at %s", rag.CHAT_DB_PATH)
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
@app.get("/api/chat/history")
def chat_history(session_id: str = Query(..., min_length=8, max_length=64)) -> dict[str, Any]:
    """Return persisted messages for a session (used by the SPA on load)."""
    if chat_store is None:
        raise HTTPException(status_code=503, detail="Chat store not initialized")
    if not chat_store.has_session(session_id):
        raise HTTPException(status_code=404, detail="Unknown session")
    return {
        "session_id": session_id,
        "messages": chat_store.list_messages_api(session_id),
    }


@app.get("/api/chat/sessions")
def chat_sessions_list() -> dict[str, Any]:
    """Recent chat sessions (newest first) for ops / optional UI."""
    if chat_store is None:
        raise HTTPException(status_code=503, detail="Chat store not initialized")
    return {"sessions": chat_store.list_sessions(limit=50)}


@app.delete("/api/chat/sessions/{session_id}")
def chat_session_delete(session_id: str) -> dict[str, Any]:
    """Delete a session and all messages."""
    if chat_store is None:
        raise HTTPException(status_code=503, detail="Chat store not initialized")
    if not chat_store.delete_session(session_id):
        raise HTTPException(status_code=404, detail="Unknown session")
    return {"ok": True, "deleted": session_id}


@app.post("/api/chat/export")
def chat_export_markdown(body: ChatExportBody) -> dict[str, Any]:
    """Write the session transcript to vault_dir/Chat exports/<date>-chat-<id>.md"""
    if rag.VAULT_DIR is None:
        raise HTTPException(status_code=400, detail="paths.vault_dir is not set in config.yaml")
    if not rag.VAULT_DIR.is_dir():
        raise HTTPException(status_code=400, detail=f"Vault not found: {rag.VAULT_DIR}")
    if chat_store is None:
        raise HTTPException(status_code=503, detail="Chat store not initialized")
    if not chat_store.has_session(body.session_id):
        raise HTTPException(status_code=404, detail="Unknown session")

    rows = chat_store.list_messages_api(body.session_id, limit=2000)
    lines = [
        "# Chat export",
        "",
        f"- Session: `{body.session_id}`",
        f"- Exported (UTC): {datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')}",
        "",
    ]
    for m in rows:
        role = m.get("role", "?")
        lines.append(f"## {role}")
        lines.append("")
        lines.append(str(m.get("content", "")))
        lines.append("")

    out_dir = rag.VAULT_DIR / "Chat exports"
    out_dir.mkdir(parents=True, exist_ok=True)
    fname = datetime.now(timezone.utc).strftime("%Y-%m-%d") + f"-chat-{body.session_id[:8]}.md"
    target = out_dir / fname
    # Avoid accidental overwrite: bump suffix if exists
    if target.is_file():
        for i in range(2, 50):
            alt = out_dir / (fname[:-3] + f"-{i}.md")
            if not alt.is_file():
                target = alt
                break

    target.write_text("\n".join(lines), encoding="utf-8")
    rel = target.relative_to(rag.VAULT_DIR.resolve()).as_posix()
    return {"ok": True, "vault_rel_path": rel, "bytes": target.stat().st_size}


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
    if chat_store is None:
        raise HTTPException(status_code=503, detail="Chat store not initialized")

    question = body.message.strip()
    sid = chat_store.ensure_session(body.session_id)
    prior_linear = chat_store.list_linear_messages(sid)

    retrieval_q = rag.build_retrieval_query_from_history(
        question,
        prior_linear,
        retrieval_pairs=rag.CHAT_RETRIEVAL_HISTORY_PAIRS,
    )

    chroma_where: dict[str, Any] | None = None
    fn = (body.filter_note or "").strip().replace("\\", "/")
    if fn:
        chroma_where = {"vault_rel_path": fn}

    scored = rag.retrieve_context_documents_scored(
        state.vectorstore,
        retrieval_q,
        top_k=rag.RETRIEVAL_TOP_K,
        chroma_where=chroma_where,
        tag_filter=body.filter_tag,
    )
    primary_docs = [d for d, _ in scored]

    neighbor_docs: list[Any] = []
    if (
        body.expand_wikilinks
        and rag.VAULT_DIR is not None
        and rag.VAULT_DIR.is_dir()
        and rag.GRAPH_EXPAND_MAX_NOTES > 0
        and rag.GRAPH_EXPAND_MAX_CHUNKS > 0
    ):
        neighbor_docs = rag.expand_docs_via_wikilink_neighbors(
            rag.VAULT_DIR,
            primary_docs,
            max_neighbor_notes=rag.GRAPH_EXPAND_MAX_NOTES,
            max_neighbor_chunks=rag.GRAPH_EXPAND_MAX_CHUNKS,
            exclude_dir_names=rag.VAULT_EXCLUDE_DIR_NAMES,
        )

    ctx = rag.documents_context_with_neighbor_section(primary_docs, neighbor_docs)

    scored_for_ui: list[tuple[Any, float | None]] = [(d, s) for d, s in scored]
    for nd in neighbor_docs:
        scored_for_ui.append((nd, None))

    prior_pairs = rag.linear_roles_to_pairs(prior_linear)
    hist_pairs = rag.take_last_pairs(prior_pairs, rag.CHAT_HISTORY_MAX_PAIRS)
    history_block = rag.format_pairs_for_prompt(hist_pairs, rag.CHAT_MAX_HISTORY_CHARS)

    prompt_text = rag.build_chat_prompt_text_with_vault_context(
        state.tokenizer,
        question,
        ctx,
        history_block,
    )

    def events():
        yield f"data: {json.dumps({'type': 'meta', 'session_id': sid}, ensure_ascii=False)}\n\n"
        # Always send retrieved sources first so the chat+graph UI can highlight vault nodes;
        # the client only shows the text list when the user enabled "Show sources".
        payload = json.dumps(
            {
                "type": "sources",
                "items": rag.serialize_sources_for_ui(
                    scored_for_ui,
                    vault_root=rag.VAULT_DIR,
                ),
            },
            ensure_ascii=False,
        )
        yield f"data: {payload}\n\n"
        pieces: list[str] = []
        try:
            for fragment in rag.iter_answer_tokens_stream(state.tokenizer, state.pipe, prompt_text):
                if fragment:
                    pieces.append(fragment)
                    piece = json.dumps(
                        {"type": "token", "text": fragment},
                        ensure_ascii=False,
                    )
                    yield f"data: {piece}\n\n"
        except Exception as e:
            err = json.dumps({"type": "error", "detail": str(e)}, ensure_ascii=False)
            yield f"data: {err}\n\n"
            return
        answer_text = "".join(pieces).strip()
        if answer_text and body.suggest_vault_edit:
            try:
                edit = rag.generate_vault_edit_suggestion(
                    state.tokenizer,
                    state.pipe,
                    question=question,
                    vault_excerpts=ctx,
                    assistant_answer=answer_text,
                )
                if edit and (edit.get("markdown") or edit.get("path")):
                    yield f"data: {json.dumps({'type': 'vault_edit', **edit}, ensure_ascii=False)}\n\n"
            except Exception as e:
                logger.exception("Vault edit suggestion failed: %s", e)
        if answer_text:
            try:
                chat_store.append_message(sid, "user", question)
                chat_store.append_message(sid, "assistant", answer_text)
            except Exception as e:
                logger.exception("Failed to persist chat messages: %s", e)
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
