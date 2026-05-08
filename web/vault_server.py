"""
Lightweight server: GET /api/vault/graph + static index.html only (no Chroma / LLM loaded).

Same HTML as the full app, but POST /api/rag/stream is not defined here — use the main
server on port 8000 for chat. Typical use: graph preview on 8001 while 8000 runs the LLM.

  uvicorn web.vault_server:app --host 127.0.0.1 --port 8001

Environment:
  RAG_CONFIG — optional path to YAML (same as main server)
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, RedirectResponse

import app as rag

logger = logging.getLogger(__name__)

STATIC = Path(__file__).resolve().parent / "static"


def _load_config_only() -> None:
    cfg_raw = os.environ.get("RAG_CONFIG")
    cfg_path = Path(cfg_raw) if cfg_raw else rag.resolve_config_path(None)
    rag.apply_app_config(rag.load_app_config_file(cfg_path))
    if cfg_path and cfg_path.is_file():
        logger.info("Vault graph server: config %s", cfg_path)


@asynccontextmanager
async def lifespan(app: FastAPI):
    _load_config_only()
    yield


app = FastAPI(title="Vault graph", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root_redirect():
    return RedirectResponse(url="/rag/chat", status_code=302)


@app.get("/vault/graph")
def vault_graph_redirect():
    return RedirectResponse(url="/rag/chat", status_code=302)


@app.get("/rag/chat")
def rag_chat_page():
    index = STATIC / "index.html"
    if not index.is_file():
        raise HTTPException(status_code=404, detail="Missing web/static/index.html")
    return FileResponse(index)


@app.get("/api/vault/graph")
def vault_graph():
    return rag.vault_graph_api_payload()
