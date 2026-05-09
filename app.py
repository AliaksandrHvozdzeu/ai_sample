"""
Author: Hvozdzeu Aliaksandr, 2026, Vilnius.

Local RAG (Retrieval-Augmented Generation) CLI application.

What this script does (high level):
  1. Loads Markdown notes from paths.vault_dir in config.yaml (Obsidian-style vault; chunked by ##/###).
  2. Embeds each chunk with sentence-transformers and stores vectors in ChromaDB (./chroma_db/).
  3. At query time, retrieves similar chunks, injects them as context, and runs a small local LLM
     (4-bit quantized) to answer in English, grounded on that context only.

Code map (major sections, top to bottom):
  - Paths / defaults -> load_app_config_file / apply_app_config (YAML merges into module globals).
  - Vault indexing: load_obsidian_documents, chunking, build_vectorstore / load_vectorstore.
  - Wikilink graph JSON for the web UI: build_vault_link_graph, vault_graph_api_payload.
  - LLM: build_llm_pipeline (4-bit HF model), streaming iter_answer_tokens_stream for FastAPI.
  - Retrieval helpers: retrieve_context_documents, build_chat_prompt_text, serialize_sources_for_ui.

Run:
  python app.py --reindex   # rebuild the vector index from the vault
  python app.py             # chat using the existing index
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import os
import re
import sys
from pathlib import Path
from typing import Any, Iterable, Iterator
from urllib.parse import quote

# LangChain: classic package holds retrieval chains; fallback supports older langchain installs.
from langchain_core.messages import BaseMessage

try:
    from langchain_classic.chains import create_retrieval_chain
    from langchain_classic.chains.combine_documents import create_stuff_documents_chain
except ImportError:
    from langchain.chains import create_retrieval_chain
    from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    pipeline,
)

try:
    import yaml
except ImportError:
    yaml = None  # type: ignore

# -----------------------------------------------------------------------------
# Paths and model IDs (defaults; can be overridden by config.yaml)
# -----------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
CHROMA_DIR = BASE_DIR / "chroma_db"  # persisted Chroma vector database on disk
# Markdown vault (directory of *.md). Set paths.vault_dir in config.yaml (required for --reindex).
VAULT_DIR: Path | None = None

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # fast local embeddings for retrieval
DEFAULT_LLM_ID = "Qwen/Qwen2.5-1.5B-Instruct"  # default instruct model (lighter)
ALT_LLM_ID = "unsloth/Llama-3.2-3B-Instruct"  # alternative instruct model (heavier)

# Default count of text chunks retrieved from Chroma for each question (config retrieval.top_k or CLI --top-k).
RETRIEVAL_TOP_K = 4

# Text generation limits / sampling (config.yaml llm.*); used by build_llm_pipeline and streaming SSE.
LLM_MAX_NEW_TOKENS = 512
LLM_DO_SAMPLE = True
LLM_TEMPERATURE = 0.2
LLM_TOP_P = 0.9

# Instructions to the LLM: must stay grounded; exact phrase when context is insufficient.
# Web chat may include a "Recent conversation" block for pronouns/follow-ups only; factual answers must still
# come from the "Relevant excerpts from your Obsidian vault" section.
_SYSTEM_PROMPT_DEFAULT = """You are a precise assistant for an Obsidian vault. Answer questions STRICTLY using only \
the "Relevant excerpts from your Obsidian vault" section in the user message. \
A "Recent conversation" section may appear for follow-up phrasing only — it is NOT a verified source of facts. \
Do not invent facts or use outside knowledge. \
If the answer is not in the vault excerpts, reply exactly: I could not find this in the documents. \
If an excerpt includes a note title or path, mention it briefly at the end of your answer."""
SYSTEM_PROMPT = _SYSTEM_PROMPT_DEFAULT

# Web UI chat persistence / dialog context (overridden by config.yaml chat.*)
CHAT_HISTORY_MAX_PAIRS = 6  # max (user, assistant) pairs injected into the prompt
CHAT_RETRIEVAL_HISTORY_PAIRS = 2  # recent pairs used to augment the embedding search query
CHAT_MAX_HISTORY_CHARS = 8000  # trim "Recent conversation" block from the start if longer
CHAT_DB_PATH = BASE_DIR / ".rag_chat.sqlite"  # SQLite path for FastAPI (relative paths resolved vs BASE_DIR)

# Wikilink neighborhood expansion (after primary retrieval) — retrieval.graph_expand_* in config.yaml
GRAPH_EXPAND_MAX_NOTES = 3
GRAPH_EXPAND_MAX_CHUNKS = 8

# Obsidian deep links (obsidian://) — obsidian.vault_name in config.yaml; optional
OBSIDIAN_VAULT_NAME: str | None = None

# Second-pass JSON suggestion for vault edits (web UI checkbox)
VAULT_EDIT_MAX_TOKENS = 192

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def load_app_config_file(path: Path | None) -> dict[str, Any]:
    """Load optional YAML config. Returns {} if missing or PyYAML not installed."""
    if path is None or not path.is_file():
        return {}
    if yaml is None:
        logger.warning("Install PyYAML to use --config / config.yaml: pip install pyyaml")
        return {}
    with path.open(encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data if isinstance(data, dict) else {}


def apply_app_config(cfg: dict[str, Any]) -> None:
    """
    Merge YAML settings into module-level paths and prompts (no need to fork app.py for new folders).
    """
    global CHROMA_DIR, VAULT_DIR, EMBEDDING_MODEL, SYSTEM_PROMPT, RETRIEVAL_TOP_K
    global VAULT_EXCLUDE_DIR_NAMES
    global LLM_MAX_NEW_TOKENS, LLM_DO_SAMPLE, LLM_TEMPERATURE, LLM_TOP_P
    global CHAT_HISTORY_MAX_PAIRS, CHAT_RETRIEVAL_HISTORY_PAIRS, CHAT_MAX_HISTORY_CHARS, CHAT_DB_PATH
    global GRAPH_EXPAND_MAX_NOTES, GRAPH_EXPAND_MAX_CHUNKS, OBSIDIAN_VAULT_NAME, VAULT_EDIT_MAX_TOKENS

    paths = cfg.get("paths") or {}
    if isinstance(paths.get("chroma_db"), str) and paths["chroma_db"].strip():
        CHROMA_DIR = (BASE_DIR / paths["chroma_db"].strip()).resolve()
    if isinstance(paths.get("vault_dir"), str) and paths["vault_dir"].strip():
        VAULT_DIR = (BASE_DIR / paths["vault_dir"].strip()).resolve()
    else:
        VAULT_DIR = None

    models = cfg.get("models") or {}
    if isinstance(models.get("embedding"), str) and models["embedding"].strip():
        EMBEDDING_MODEL = models["embedding"].strip()

    retrieval = cfg.get("retrieval") or {}
    if retrieval.get("top_k") is not None:
        RETRIEVAL_TOP_K = max(1, int(retrieval["top_k"]))
    if retrieval.get("graph_expand_max_notes") is not None:
        GRAPH_EXPAND_MAX_NOTES = max(0, min(16, int(retrieval["graph_expand_max_notes"])))
    if retrieval.get("graph_expand_max_chunks") is not None:
        GRAPH_EXPAND_MAX_CHUNKS = max(0, min(48, int(retrieval["graph_expand_max_chunks"])))

    obsidian_cfg = cfg.get("obsidian") or {}
    if isinstance(obsidian_cfg, dict):
        vn = obsidian_cfg.get("vault_name")
        if isinstance(vn, str) and vn.strip():
            OBSIDIAN_VAULT_NAME = vn.strip()
        else:
            OBSIDIAN_VAULT_NAME = None

    llm_cfg = cfg.get("llm") or {}
    if isinstance(llm_cfg, dict):
        if llm_cfg.get("max_new_tokens") is not None:
            LLM_MAX_NEW_TOKENS = max(16, min(4096, int(llm_cfg["max_new_tokens"])))
        if "do_sample" in llm_cfg:
            LLM_DO_SAMPLE = bool(llm_cfg["do_sample"])
        if llm_cfg.get("temperature") is not None:
            LLM_TEMPERATURE = float(llm_cfg["temperature"])
        if llm_cfg.get("top_p") is not None:
            LLM_TOP_P = float(llm_cfg["top_p"])

    prompt = cfg.get("prompt") or {}
    if isinstance(prompt.get("system"), str) and prompt["system"].strip():
        SYSTEM_PROMPT = prompt["system"].strip()

    vault_section = cfg.get("vault") or {}
    if isinstance(vault_section.get("exclude_dir_names"), list):
        names = [str(x).strip() for x in vault_section["exclude_dir_names"] if str(x).strip()]
        if names:
            VAULT_EXCLUDE_DIR_NAMES = frozenset(names)

    chat_cfg = cfg.get("chat") or {}
    if isinstance(chat_cfg, dict):
        if chat_cfg.get("history_max_pairs") is not None:
            CHAT_HISTORY_MAX_PAIRS = max(0, min(32, int(chat_cfg["history_max_pairs"])))
        if chat_cfg.get("retrieval_history_pairs") is not None:
            CHAT_RETRIEVAL_HISTORY_PAIRS = max(0, min(16, int(chat_cfg["retrieval_history_pairs"])))
        if chat_cfg.get("max_history_chars") is not None:
            CHAT_MAX_HISTORY_CHARS = max(500, min(100_000, int(chat_cfg["max_history_chars"])))
        if isinstance(chat_cfg.get("db_path"), str) and chat_cfg["db_path"].strip():
            CHAT_DB_PATH = (BASE_DIR / chat_cfg["db_path"].strip()).resolve()
        if chat_cfg.get("suggest_edit_max_tokens") is not None:
            VAULT_EDIT_MAX_TOKENS = max(32, min(512, int(chat_cfg["suggest_edit_max_tokens"])))


def _venv_consistency_warning() -> None:
    """
    If ./.venv exists but the current interpreter is not inside it, warn once.
    Keeps installs isolated from the global Python when users forget to activate.
    """
    project_venv = BASE_DIR / ".venv"
    if not project_venv.is_dir():
        return
    exe = Path(sys.executable).resolve()
    try:
        inside_project_venv = exe.is_relative_to(project_venv.resolve())
    except AttributeError:
        inside_project_venv = str(exe).startswith(str(project_venv.resolve()))
    if inside_project_venv:
        return
    if os.environ.get("VIRTUAL_ENV"):
        logger.warning(
            "A virtualenv is active (%s), but it is not this project's .venv. "
            "Prefer: .venv\\Scripts\\python.exe or scripts\\run_app.ps1 (Windows).",
            os.environ["VIRTUAL_ENV"],
        )
        return
    logger.warning(
        "This interpreter is not the project's .venv (%s). "
        "Run scripts\\setup_venv.ps1 then scripts\\run_app.ps1 so packages stay inside .venv.",
        exe,
    )


def resolve_config_path(cli_path: Path | None) -> Path | None:
    """Use explicit --config, else config.yaml next to app.py if it exists."""
    if cli_path is not None:
        return cli_path
    default_yaml = BASE_DIR / "config.yaml"
    return default_yaml if default_yaml.is_file() else None


def _log_pytorch_cuda() -> None:
    """Print whether PyTorch was built with CUDA and whether a GPU is visible (common CPU-only wheel issue)."""
    import torch

    cuda_build = getattr(torch.version, "cuda", None)
    logger.info("PyTorch %s — CUDA build tag: %s", torch.__version__, cuda_build or "none (CPU-only wheel)")
    if torch.cuda.is_available():
        logger.info("CUDA runtime OK — GPU 0: %s", torch.cuda.get_device_name(0))
    else:
        logger.warning(
            "CUDA runtime NOT available: inference will use CPU (slow). "
            "Install a CUDA-enabled PyTorch build from https://pytorch.org/get-started/locally/ "
            "(pip CPU wheels do not use your NVIDIA GPU)."
        )


def _make_embeddings() -> HuggingFaceEmbeddings:
    """SentenceTransformer defaults to CPU unless device is set — align with GPU when PyTorch sees CUDA."""
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        logger.info("Embeddings: %s on GPU (%s)", EMBEDDING_MODEL, torch.cuda.get_device_name(0))
    else:
        logger.warning("Embeddings: %s on CPU — install CUDA PyTorch for faster retrieval", EMBEDDING_MODEL)
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL, model_kwargs={"device": device})


def _log_llm_placement(model) -> None:
    """Where the quantized LLM actually landed (GPU vs CPU)."""
    dm = getattr(model, "hf_device_map", None)
    if dm:
        logger.info("LLM hf_device_map: %s", dm)
    try:
        logger.info("LLM first tensor device: %s", next(model.parameters()).device)
    except StopIteration:
        pass


def _clean_metadata(meta: dict[str, Any]) -> dict[str, Any]:
    """
    Chroma metadata values must be scalar types it accepts (str, int, float, bool).
    Drop None; coerce other types (e.g. nested dicts) to string so storage never fails.
    """
    out: dict[str, Any] = {}
    for k, v in meta.items():
        if v is None:
            continue
        if isinstance(v, (str, int, float, bool)):
            out[k] = v
        else:
            out[k] = str(v)
    return out


# --- Markdown vault (index + graph) -------------------------------------------

DEFAULT_VAULT_EXCLUDE_DIR_NAMES = frozenset(
    {".obsidian", ".git", "node_modules", ".trash"}
)
# Overridden by vault.exclude_dir_names in config.yaml
VAULT_EXCLUDE_DIR_NAMES: frozenset[str] = DEFAULT_VAULT_EXCLUDE_DIR_NAMES

_WIKILINK_BRACKET_RE = re.compile(r"\[\[([^\]]+)\]\]")


def extract_wikilink_targets(raw: str) -> list[str]:
    """
    Collect Obsidian-style [[wikilink]] targets from raw Markdown (includes embeds ![[...]] body).
    Strips [[alias|display]], heading anchors Note#Heading, and dedupes in order of appearance.
    """
    out: list[str] = []
    seen: set[str] = set()
    for m in _WIKILINK_BRACKET_RE.finditer(raw):
        inner = m.group(1).strip()
        if "|" in inner:
            inner = inner.split("|", 1)[0].strip()
        if "#" in inner:
            inner = inner.split("#", 1)[0].strip()
        inner = inner.replace("\\", "/")
        if inner and inner not in seen:
            seen.add(inner)
            out.append(inner)
    return out


def _stem_index_for_vault(files_rel: list[str]) -> dict[str, list[str]]:
    """Map lowercase note stem -> vault-relative paths (multiple if duplicate names in folders)."""
    stem_to_ids: dict[str, list[str]] = {}
    for rel_id in files_rel:
        stem = Path(rel_id).stem.lower()
        stem_to_ids.setdefault(stem, []).append(rel_id)
    return stem_to_ids


def resolve_wikilink_to_note_ids(
    target: str,
    vault_dir: Path,
    allowed_ids: set[str],
    stem_to_ids: dict[str, list[str]],
) -> list[str]:
    """
    Map one wikilink path/name to existing note id(s) (vault-relative posix paths).
    Order: exact relative path match, then basename stem match (possibly multiple).
    """
    t = target.strip().replace("\\", "/")
    if not t:
        return []
    if t.lower().endswith(".md"):
        t = t[:-3]
    key_file = (vault_dir / t).with_suffix(".md")
    try:
        if key_file.is_file():
            rid = key_file.resolve().relative_to(vault_dir.resolve()).as_posix()
            if rid in allowed_ids:
                return [rid]
    except ValueError:
        pass
    stem = Path(t).stem.lower()
    return [rid for rid in stem_to_ids.get(stem, []) if rid in allowed_ids]


def build_vault_link_graph(
    vault_dir: Path,
    *,
    exclude_dir_names: frozenset[str] | set[str] | None = None,
) -> dict[str, Any]:
    """
    Nodes = *.md under vault_dir (respecting excludes); edges = [[wikilink]] from source to target note.
    Used by the web UI graph view (Obsidian-style overview).
    """
    excl = exclude_dir_names if exclude_dir_names is not None else VAULT_EXCLUDE_DIR_NAMES
    out: dict[str, Any] = {"nodes": [], "edges": [], "vault_root": str(vault_dir.resolve())}

    if not vault_dir.is_dir():
        out["error"] = "vault_not_found"
        out["detail"] = str(vault_dir)
        return out

    files_rel: list[str] = []
    abs_by_rel: dict[str, Path] = {}
    for path in sorted(vault_dir.rglob("*.md")):
        rel_parts = path.relative_to(vault_dir).parts
        if any(p in excl for p in rel_parts):
            continue
        rel_id = path.relative_to(vault_dir).as_posix()
        files_rel.append(rel_id)
        abs_by_rel[rel_id] = path

    allowed_ids = set(files_rel)
    stem_to_ids = _stem_index_for_vault(files_rel)

    for rel_id in files_rel:
        label = Path(rel_id).stem
        try:
            raw = abs_by_rel[rel_id].read_text(encoding="utf-8")
            _, body = _parse_markdown_frontmatter(raw)
            title = _extract_note_title(body, Path(rel_id).stem)
            if title:
                label = title
        except OSError:
            pass
        out["nodes"].append({"id": rel_id, "label": label})

    edge_seen: set[tuple[str, str]] = set()
    for rel_id in files_rel:
        try:
            raw = abs_by_rel[rel_id].read_text(encoding="utf-8")
        except OSError:
            continue
        for tgt in extract_wikilink_targets(raw):
            for dest_id in resolve_wikilink_to_note_ids(tgt, vault_dir, allowed_ids, stem_to_ids):
                if dest_id == rel_id:
                    continue
                key = (rel_id, dest_id)
                if key in edge_seen:
                    continue
                edge_seen.add(key)
                out["edges"].append({"from": rel_id, "to": dest_id})

    return out


def vault_graph_api_payload() -> dict[str, Any]:
    """JSON for GET /api/vault/graph (used by the full RAG server and vault-only graph server)."""
    if VAULT_DIR is None:
        return {
            "nodes": [],
            "edges": [],
            "vault_configured": False,
            "message": "paths.vault_dir is not set in config.yaml",
        }
    root = VAULT_DIR.resolve()
    if not VAULT_DIR.is_dir():
        return {
            "nodes": [],
            "edges": [],
            "vault_configured": True,
            "vault_root": str(root),
            "error": "vault_not_found",
            "detail": str(VAULT_DIR),
        }
    data = build_vault_link_graph(VAULT_DIR)
    data["vault_configured"] = True
    return data


def _tags_for_metadata(tags: Any) -> str | None:
    """Normalize frontmatter tags to a single string for Chroma metadata."""
    if tags is None:
        return None
    if isinstance(tags, str):
        return tags.strip() or None
    if isinstance(tags, list):
        flat = [str(t).strip() for t in tags if str(t).strip()]
        return ", ".join(flat) if flat else None
    return str(tags)


def _parse_markdown_frontmatter(raw: str) -> tuple[dict[str, Any], str]:
    """
    Split optional YAML frontmatter (--- ... ---) from the note body.
    If PyYAML is available, frontmatter is parsed as YAML; otherwise body is returned with empty meta.
    """
    m = re.match(r"^---\s*\r?\n(.*?)\r?\n---\s*\r?\n?(.*)$", raw, re.DOTALL)
    if not m:
        return {}, raw
    fm_text, body = m.group(1), m.group(2)
    meta: dict[str, Any] = {}
    if yaml is not None:
        try:
            loaded = yaml.safe_load(fm_text)
            if isinstance(loaded, dict):
                meta = loaded
        except Exception:
            logger.debug("Could not parse YAML frontmatter; indexing body only.")
    return meta, body


def _normalize_obsidian_markup(text: str) -> str:
    """
    Light cleanup for indexing: resolve [[wikilinks]] to readable labels and drop embed-only ![[...]].
    Does not parse full Markdown — keeps retrieval text closer to what humans read in Obsidian.
    """
    s = text
    s = re.sub(r"!\[\[([^\]]*)\]\]", "", s)
    s = re.sub(r"\[\[([^\]|]+)\|([^\]]+)\]\]", r"\2", s)
    s = re.sub(r"\[\[([^\]]+)\]\]", r"\1", s)
    return "\n".join(line.rstrip() for line in s.splitlines()).strip()


def _extract_note_title(body: str, fallback_stem: str) -> str:
    """First # heading (H1) wins; else use file stem."""
    for line in body.splitlines():
        s = line.strip()
        if s.startswith("# ") and not s.startswith("##"):
            return s[2:].strip() or fallback_stem
    return fallback_stem


def _split_body_by_h2_h3(body: str) -> list[tuple[str, str]]:
    """
    Split markdown body into (section_heading, section_text). heading '' = intro before first ##/###.
    Section text does not include the '##' line itself.
    """
    body = body.strip()
    if not body:
        return []
    segments = re.split(r"\n(?=#{2,3}\s+)", body)
    out: list[tuple[str, str]] = []
    for i, seg in enumerate(segments):
        seg = seg.strip()
        if not seg:
            continue
        if i == 0 and not re.match(r"^#{2,3}\s+", seg):
            out.append(("", seg))
            continue
        first_nl = seg.find("\n")
        first_line = seg if first_nl == -1 else seg[:first_nl]
        hm = re.match(r"^#{2,3}\s+(.+)$", first_line.strip())
        if hm:
            heading = hm.group(1).strip()
            rest = seg[first_nl + 1 :].strip() if first_nl != -1 else ""
            out.append((heading, rest))
        else:
            out.append(("", seg))
    return out


def _format_obsidian_chunk(
    vault_rel: str,
    note_title: str,
    section_heading: str,
    section_text: str,
    *,
    tags: str | None,
) -> tuple[str, dict[str, Any]]:
    """Build page_content and metadata for one vault chunk (before the trailing [Metadata: ...] line)."""
    sec_label = section_heading if section_heading else "intro"
    body = _normalize_obsidian_markup(section_text)
    text = f"Note: {note_title}\nPath: {vault_rel}\nSection: {sec_label}\n\n{body}"
    source_title = f"{note_title} — {section_heading}" if section_heading else note_title
    meta: dict[str, Any] = {
        "format": "obsidian",
        "title": note_title,
        "source_title": source_title,
        "vault_rel_path": vault_rel,
        "heading": section_heading,
        "tags": tags,
    }
    return text, meta


def _md_file_to_documents(path: Path, vault_root: Path) -> list[Document]:
    """Parse one .md file into one or more Documents (chunked by ##/###)."""
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as e:
        logger.warning("Could not read %s: %s", path, e)
        return []
    if not raw.strip():
        return []

    fm, body = _parse_markdown_frontmatter(raw)
    tags = _tags_for_metadata(fm.get("tags"))

    rel = path.relative_to(vault_root)
    vault_rel = rel.as_posix()
    note_title = _extract_note_title(body, path.stem)

    sections = _split_body_by_h2_h3(body)
    if not sections:
        return []

    docs: list[Document] = []
    for idx, (heading, sec_text) in enumerate(sections):
        if not sec_text.strip():
            continue
        page, meta = _format_obsidian_chunk(
            vault_rel, note_title, heading, sec_text, tags=tags
        )
        meta["source_file"] = vault_rel
        meta["chunk_index"] = idx
        meta_line_parts = [f"file={vault_rel}", f"title={note_title}"]
        if tags:
            meta_line_parts.append(f"tags={tags}")
        page += "\n[Metadata: " + "; ".join(meta_line_parts) + "]"
        docs.append(Document(page_content=page, metadata=_clean_metadata(meta)))
    return docs


def load_obsidian_documents(
    vault_dir: Path,
    *,
    exclude_dir_names: frozenset[str] | set[str] | None = None,
) -> list[Document]:
    """
    Recursively index *.md under vault_dir. Skips configured folder names in path components
    (default: .obsidian, .git, node_modules, .trash). Each file is split on ## / ### headings.
    """
    excl = frozenset(exclude_dir_names) if exclude_dir_names is not None else VAULT_EXCLUDE_DIR_NAMES
    if not vault_dir.is_dir():
        logger.error("Vault directory not found: %s", vault_dir)
        return []

    out: list[Document] = []
    for path in sorted(vault_dir.rglob("*.md")):
        rel_parts = path.relative_to(vault_dir).parts
        if any(p in excl for p in rel_parts):
            continue
        out.extend(_md_file_to_documents(path, vault_dir))

    if not out:
        logger.warning("No Markdown documents loaded from vault %s", vault_dir)
    else:
        logger.info("Loaded %s Obsidian/vault chunk(s) for indexing.", len(out))
    return out


def load_all_documents_for_reindex() -> list[Document]:
    """Load vault Markdown chunks for --reindex."""
    if VAULT_DIR is None:
        logger.error("Set paths.vault_dir in config.yaml to your Markdown folder.")
        return []
    if not VAULT_DIR.is_dir():
        logger.error("Vault directory not found: %s", VAULT_DIR)
        return []
    return load_obsidian_documents(VAULT_DIR)


def list_vault_markdown_relpaths() -> list[str]:
    """
    Sorted vault-relative paths (posix) for every *.md under VAULT_DIR.
    Skips the same directory name rules as indexing (VAULT_EXCLUDE_DIR_NAMES).
    Used by the web UI file list.
    """
    if VAULT_DIR is None or not VAULT_DIR.is_dir():
        return []
    out: list[str] = []
    for path in sorted(VAULT_DIR.rglob("*.md")):
        rel_parts = path.relative_to(VAULT_DIR).parts
        if any(p in VAULT_EXCLUDE_DIR_NAMES for p in rel_parts):
            continue
        out.append(path.relative_to(VAULT_DIR).as_posix())
    return out


def build_vectorstore(documents: Iterable[Document], persist_dir: Path, clear: bool) -> Chroma:
    """
    Create or overwrite Chroma with embedded documents.
    If clear=True and persist_dir exists, delete it first (--reindex path).
    """
    embeddings = _make_embeddings()
    persist_dir.mkdir(parents=True, exist_ok=True)

    docs_list = list(documents)
    if not docs_list:
        raise ValueError(
            "No documents to index. Set paths.vault_dir in config.yaml, add .md files with content, and run --reindex."
        )

    if clear and persist_dir.exists():
        import shutil

        shutil.rmtree(persist_dir)
        logger.info("Vector store directory cleared: %s", persist_dir)

    # from_documents: embeds each Document.page_content and stores vectors + metadata on disk
    vs = Chroma.from_documents(
        documents=docs_list,
        embedding=embeddings,
        persist_directory=str(persist_dir),
    )
    logger.info("Chroma index saved to %s", persist_dir)
    return vs


def load_vectorstore(persist_dir: Path) -> Chroma:
    """Open an existing Chroma store from disk (same embedding model must be used at query time)."""
    embeddings = _make_embeddings()
    if not persist_dir.exists():
        raise FileNotFoundError(
            f"Database not found at {persist_dir}. Run: python app.py --reindex"
        )
    return Chroma(persist_directory=str(persist_dir), embedding_function=embeddings)


def release_vectorstore(vectorstore: Chroma | None) -> None:
    """
    Close the underlying chromadb client so persist_dir can be deleted on Windows.
    The web server keeps a long-lived Chroma handle; without this, shutil.rmtree
    hits PermissionError on chroma.sqlite3 (WinError 32).
    """
    if vectorstore is None:
        return
    client = getattr(vectorstore, "_client", None)
    if client is None:
        return
    close = getattr(client, "close", None)
    if not callable(close):
        return
    try:
        close()
    except Exception as e:
        logger.warning("release_vectorstore: client.close() failed: %s", e)


def build_llm_pipeline(model_id: str):
    """
    Load an instruct causal LM with 4-bit quantization (BitsAndBytes) to reduce VRAM.
    Returns (tokenizer, transformers pipeline) for text generation.
    """
    import torch

    _log_pytorch_cuda()

    # BPE tokenizers (e.g. Qwen): suppress HF warning about clean_up_tokenization_spaces.
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        trust_remote_code=True,
        clean_up_tokenization_spaces=False,
    )
    # NF4 double-quant is a common preset for quality/size tradeoff on consumer GPUs.
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    # Pin all weights to GPU 0 when CUDA exists; with a CPU-only PyTorch wheel this still maps to CPU.
    device_map = {"": 0} if torch.cuda.is_available() else "auto"
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=True,
        quantization_config=bnb_config,
        device_map=device_map,
    )
    _log_llm_placement(model)

    # Some checkpoints omit pad_token_id; generation requires a valid pad token.
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Tune defaults on the model so TextGenerationPipeline does not receive a duplicate
    # `generation_config` kwarg (newer transformers raise TypeError if passed twice).
    # Prefer max_new_tokens only; clear max_length to avoid conflicting constraints.
    generation_config = copy.deepcopy(model.generation_config)
    generation_config.max_new_tokens = LLM_MAX_NEW_TOKENS
    generation_config.do_sample = LLM_DO_SAMPLE
    generation_config.temperature = LLM_TEMPERATURE
    generation_config.top_p = LLM_TOP_P
    generation_config.pad_token_id = tokenizer.pad_token_id
    generation_config.max_length = None
    model.generation_config = generation_config

    # return_full_text=False: pipeline returns only newly generated tokens (not the prompt).
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        return_full_text=False,
    )
    return tokenizer, pipe


def _messages_to_hf(messages: list[BaseMessage]) -> list[dict[str, str]]:
    """Map LangChain message types to Hugging Face chat roles for apply_chat_template."""
    role_map = {"system": "system", "human": "user", "ai": "assistant"}
    out: list[dict[str, str]] = []
    for m in messages:
        role = role_map.get(m.type, "user")
        out.append({"role": role, "content": str(m.content)})
    return out


def make_llm_runnable(tokenizer, pipe):
    """
    Wrap the HF pipeline as a LangChain Runnable used as the "LLM" after ChatPromptTemplate.

    create_stuff_documents_chain does: format retrieved docs -> fill {context} and {input}
    -> ChatPromptTemplate -> this runnable.

    The runnable receives either:
      - ChatPromptValue (has .to_messages()) from the prompt, or
      - a legacy dict with "context" and "input" keys.

    We apply the model's chat template so Qwen/Llama instruct formats are respected.
    """

    def _generate(input_data: Any) -> str:
        if hasattr(input_data, "to_messages"):
            msgs_hf = _messages_to_hf(input_data.to_messages())
        elif isinstance(input_data, dict) and "context" in input_data and "input" in input_data:
            msgs_hf = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": (
                        f"Document context:\n{input_data['context']}\n\n"
                        f"Question: {input_data['input']}"
                    ),
                },
            ]
        else:
            raise TypeError(f"Unexpected LLM input: {type(input_data)!r}")

        # Single string prompt compatible with the tokenizer's instruct template.
        prompt_text = tokenizer.apply_chat_template(
            msgs_hf,
            tokenize=False,
            add_generation_prompt=True,
        )
        out = pipe(prompt_text)
        if isinstance(out, list) and out:
            text = out[0].get("generated_text", "")
        else:
            text = str(out)
        return text.strip()

    return RunnableLambda(_generate)


def build_retrieval_chain(vectorstore: Chroma, tokenizer, pipe, *, top_k: int | None = None):
    """
    Wire retrieval + generation:
      1. Retriever: vector similarity search, top-k chunks.
      2. create_stuff_documents_chain: concatenate chunks into one {context} string.
      3. create_retrieval_chain: user question -> retrieve -> combine -> LLM -> answer.

    Invocation returns a dict; we read "answer" in run_chat().
    """
    k = top_k if top_k is not None else RETRIEVAL_TOP_K
    retriever = vectorstore.as_retriever(search_kwargs={"k": k})
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT),
            ("human", "Document context:\n{context}\n\nQuestion: {input}"),
        ]
    )
    combine_docs_chain = create_stuff_documents_chain(make_llm_runnable(tokenizer, pipe), prompt)
    return create_retrieval_chain(retriever, combine_docs_chain)


def retrieve_context_documents(
    vectorstore: Chroma,
    question: str,
    *,
    top_k: int | None = None,
) -> list[Document]:
    """Same retrieval policy as build_retrieval_chain (for API streaming without duplicating chain wiring)."""
    return [d for d, _ in retrieve_context_documents_scored(vectorstore, question, top_k=top_k)]


def retrieve_context_documents_scored(
    vectorstore: Chroma,
    question: str,
    *,
    top_k: int | None = None,
    chroma_where: dict[str, Any] | None = None,
    tag_filter: str | None = None,
) -> list[tuple[Document, float]]:
    """
    Vector search with optional Chroma metadata filter (e.g. vault_rel_path) and optional tag substring filter.
    Returns (document, distance) where distance is Chroma L2 — lower is a closer match.
    """
    k = top_k if top_k is not None else RETRIEVAL_TOP_K
    tag_f = (tag_filter or "").strip()
    fetch_k = min(k * 6, 80) if tag_f else k

    pairs: list[tuple[Document, float]]
    try:
        if chroma_where:
            pairs = vectorstore.similarity_search_with_score(
                question, k=fetch_k, filter=chroma_where
            )
        else:
            pairs = vectorstore.similarity_search_with_score(question, k=fetch_k)
    except TypeError:
        pairs = vectorstore.similarity_search_with_score(question, k=fetch_k)
        if chroma_where:

            def _meta_ok(doc: Document) -> bool:
                m = doc.metadata or {}
                for key, val in chroma_where.items():
                    if str(m.get(key)) != str(val):
                        return False
                return True

            pairs = [p for p in pairs if _meta_ok(p[0])][:fetch_k]

    if not tag_f:
        return pairs[:k]

    out: list[tuple[Document, float]] = []
    needle = tag_f.lower()
    for doc, dist in pairs:
        tags = (doc.metadata or {}).get("tags") or ""
        if needle in tags.lower():
            out.append((doc, dist))
        if len(out) >= k:
            break
    return out


def documents_context_string(documents: list[Document]) -> str:
    """Join retrieved chunk texts for the LLM context window."""
    return "\n\n".join(d.page_content for d in documents)


def documents_context_with_neighbor_section(primary: list[Document], neighbors: list[Document]) -> str:
    """Primary retrieval chunks plus optional one-hop wikilink neighbors (clearly labeled)."""
    base = documents_context_string(primary)
    if not neighbors:
        return base
    extra = documents_context_string(neighbors)
    return (
        base
        + "\n\n---\nLinked vault notes (one hop via [[wikilink]] from retrieved chunks; "
        "treat as supplementary context only):\n\n"
        + extra
    )


def expand_docs_via_wikilink_neighbors(
    vault_dir: Path,
    seed_docs: list[Document],
    *,
    max_neighbor_notes: int,
    max_neighbor_chunks: int,
    exclude_dir_names: frozenset[str],
) -> list[Document]:
    """
    Load extra chunks from notes linked by [[wikilinks]] to/from the primary retrieval hits
    (local graph neighborhood only — not the full vault).
    """
    if max_neighbor_notes <= 0 or max_neighbor_chunks <= 0:
        return []

    data = build_vault_link_graph(vault_dir, exclude_dir_names=exclude_dir_names)
    edges = data.get("edges") or []

    seed_paths: set[str] = set()
    for d in seed_docs:
        meta = d.metadata or {}
        p = meta.get("vault_rel_path") or meta.get("source_file") or _vault_rel_path_from_chunk_text(
            d.page_content
        )
        if p:
            seed_paths.add(str(p).replace("\\", "/"))

    if not seed_paths:
        return []

    neighbor_ids: list[str] = []
    seen: set[str] = set()
    for e in edges:
        fr = e.get("from")
        to = e.get("to")
        if fr in seed_paths and to not in seed_paths and to not in seen:
            neighbor_ids.append(to)
            seen.add(to)
        if to in seed_paths and fr not in seed_paths and fr not in seen:
            neighbor_ids.append(fr)
            seen.add(fr)
        if len(neighbor_ids) >= max_neighbor_notes * 4:
            break

    neighbor_ids = neighbor_ids[:max_neighbor_notes]

    out_docs: list[Document] = []
    vault_resolved = vault_dir.resolve()
    for rel in neighbor_ids:
        if len(out_docs) >= max_neighbor_chunks:
            break
        path = (vault_dir / rel).resolve()
        try:
            path.relative_to(vault_resolved)
        except ValueError:
            continue
        if not path.is_file():
            continue
        for ch in _md_file_to_documents(path, vault_dir):
            if len(out_docs) >= max_neighbor_chunks:
                break
            meta = dict(ch.metadata or {})
            meta["context_expand"] = "wikilink_neighbor"
            out_docs.append(
                Document(page_content=ch.page_content, metadata=_clean_metadata(meta))
            )

    return out_docs


def build_obsidian_open_uri(vault_root: Path | None, vault_rel_path: str | None) -> str | None:
    """Desktop Obsidian URI — set obsidian.vault_name in config, or falls back to absolute file path."""
    if not vault_rel_path or vault_root is None:
        return None
    rel = str(vault_rel_path).replace("\\", "/")
    if OBSIDIAN_VAULT_NAME:
        return "obsidian://open?vault=" + quote(OBSIDIAN_VAULT_NAME) + "&file=" + quote(rel)
    abs_path = (vault_root / rel).resolve()
    return "obsidian://open?path=" + quote(str(abs_path))


def linear_roles_to_pairs(linear: list[tuple[str, str]]) -> list[tuple[str, str]]:
    """Chronological (role, text) rows -> (user, assistant) pairs; skips orphan lines."""
    pairs: list[tuple[str, str]] = []
    i = 0
    while i < len(linear):
        role, text = linear[i]
        if role == "user" and i + 1 < len(linear) and linear[i + 1][0] == "assistant":
            pairs.append((text, linear[i + 1][1]))
            i += 2
        else:
            i += 1
    return pairs


def take_last_pairs(pairs: list[tuple[str, str]], n: int) -> list[tuple[str, str]]:
    if n <= 0 or not pairs:
        return []
    return pairs[-n:]


def format_pairs_for_prompt(pairs: list[tuple[str, str]], max_chars: int) -> str:
    """Plain-text block for the model (trimmed by character budget)."""
    if not pairs:
        return ""
    lines: list[str] = []
    for u, a in pairs:
        lines.append(f"User: {u.strip()}")
        lines.append(f"Assistant: {a.strip()}")
    text = "\n".join(lines)
    if len(text) <= max_chars:
        return text
    return text[-max_chars:]


def build_retrieval_query_from_history(
    current_question: str,
    prior_linear_messages: list[tuple[str, str]],
    *,
    retrieval_pairs: int,
) -> str:
    """
    Augment the Chroma query with recent dialog so short follow-ups ('that note', 'expand')
    still retrieve relevant vault chunks.
    """
    q = current_question.strip()
    if retrieval_pairs <= 0:
        return q
    pairs = linear_roles_to_pairs(prior_linear_messages)
    tail = take_last_pairs(pairs, retrieval_pairs)
    if not tail:
        return q
    conv = "\n".join(f"User: {u}\nAssistant: {a}" for u, a in tail)
    return f"{conv}\n\nFollow-up question:\n{q}"


def build_hf_chat_messages_with_vault_context(
    user_question: str,
    doc_context: str,
    history_block: str,
) -> list[dict[str, str]]:
    """Web/SSE prompt: optional dialog block + vault excerpts + current question."""
    parts: list[str] = []
    hb = (history_block or "").strip()
    if hb:
        parts.append(
            "Recent conversation (for pronouns and follow-ups only; not a verified knowledge source):\n" + hb
        )
    parts.append("Relevant excerpts from your Obsidian vault:\n" + (doc_context or "").strip())
    parts.append("Current question:\n" + (user_question or "").strip())
    user_content = "\n\n".join(parts)
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


def build_chat_prompt_text_with_vault_context(
    tokenizer: Any,
    user_question: str,
    doc_context: str,
    history_block: str,
) -> str:
    """Single prompt string for streaming when dialog history is available."""
    msgs = build_hf_chat_messages_with_vault_context(user_question, doc_context, history_block)
    return tokenizer.apply_chat_template(
        msgs,
        tokenize=False,
        add_generation_prompt=True,
    )


def build_hf_chat_messages(user_question: str, context: str) -> list[dict[str, str]]:
    """HF chat turns for apply_chat_template (must stay aligned with make_llm_runnable)."""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"Document context:\n{context}\n\n"
                f"Question: {user_question}"
            ),
        },
    ]


def build_chat_prompt_text(tokenizer: Any, user_question: str, context: str) -> str:
    """Single prompt string passed to the small LM (streaming + non-streaming)."""
    msgs = build_hf_chat_messages(user_question, context)
    return tokenizer.apply_chat_template(
        msgs,
        tokenize=False,
        add_generation_prompt=True,
    )


def _vault_rel_path_from_chunk_text(page_content: str | None) -> str | None:
    """Recover vault-relative path from indexed Markdown chunk text if metadata is missing."""
    if not page_content:
        return None
    m = re.search(r"(?m)^Path:\s*(.+)$", page_content)
    if not m:
        return None
    return m.group(1).strip().replace("\\", "/")


def serialize_sources_for_ui(
    scored_documents: list[Document] | list[tuple[Document, float | None]],
    *,
    vault_root: Path | None = None,
    include_scores: bool = True,
) -> list[dict[str, Any]]:
    """Compact JSON for the web UI / SSE \"sources\" event (optional Chroma distances + Obsidian URIs)."""
    rows: list[tuple[Document, float | None]] = []
    for item in scored_documents:
        if isinstance(item, tuple):
            rows.append((item[0], item[1]))
        else:
            rows.append((item, None))

    out: list[dict[str, Any]] = []
    for d, dist in rows:
        meta = d.metadata or {}
        graph_id = meta.get("vault_rel_path") or meta.get("source_file")
        if not graph_id:
            graph_id = _vault_rel_path_from_chunk_text(d.page_content)
        row: dict[str, Any] = {
            "graph_id": graph_id,
            "source_file": meta.get("source_file") or graph_id,
            "vault_rel_path": meta.get("vault_rel_path") or graph_id,
            "label": meta.get("source_title") or meta.get("title") or meta.get("question"),
            "preview": (d.page_content or "")[:240].replace("\n", " "),
            "context_expand": meta.get("context_expand"),
        }
        if include_scores and dist is not None:
            row["distance"] = float(dist)
            row["match_hint"] = "Chroma vector distance (L2) — lower values are closer semantic matches."
        uri = build_obsidian_open_uri(vault_root, graph_id)
        if uri:
            row["obsidian_uri"] = uri
        out.append(row)
    return out


def generate_text_non_stream(
    tokenizer: Any,
    pipe: Any,
    prompt_text: str,
    *,
    max_new_tokens: int,
) -> str:
    """Single-shot generation for short JSON / auxiliary outputs (no streaming)."""
    import torch

    model = pipe.model
    enc = tokenizer(prompt_text, return_tensors="pt")
    device = next(model.parameters()).device
    enc = {key: val.to(device) for key, val in enc.items()}
    input_len = enc["input_ids"].shape[1]
    with torch.inference_mode():
        out_ids = model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
    gen_ids = out_ids[0, input_len:]
    return tokenizer.decode(gen_ids, skip_special_tokens=True).strip()


def parse_vault_edit_json(raw: str) -> dict[str, str] | None:
    """Parse {\"path\": \"...\", \"markdown\": \"...\"} from model output."""
    text = raw.strip()
    if "```" in text:
        parts = text.split("```")
        for block in parts:
            b = block.strip()
            if b.startswith("{") or b.startswith("json"):
                if b.startswith("json"):
                    b = b.split("\n", 1)[-1] if "\n" in b else b
                text = b
                break
    try:
        data = json.loads(text)
        if isinstance(data, dict):
            path = str(data.get("path") or "").strip()
            md = str(data.get("markdown") or "").strip()
            return {"path": path, "markdown": md}
    except json.JSONDecodeError:
        return None
    return None


def generate_vault_edit_suggestion(
    tokenizer: Any,
    pipe: Any,
    *,
    question: str,
    vault_excerpts: str,
    assistant_answer: str,
) -> dict[str, str] | None:
    """Second short generation: structured suggestion for a vault patch (optional UX)."""
    vault_excerpts = (vault_excerpts or "")[:6000]
    assistant_answer = (assistant_answer or "")[:4000]
    sys_msg = (
        "Reply with one JSON object only (no markdown fences, no extra text). "
        'Keys: "path" (vault-relative .md path or ""), '
        '"markdown" (markdown snippet to append under a ## heading, or "").'
    )
    user_msg = (
        f"User question:\n{question}\n\nVault excerpts:\n{vault_excerpts}\n\n"
        f"Assistant draft answer:\n{assistant_answer}\n\n"
        "If a concrete vault edit helps, fill path and markdown; otherwise use empty strings."
    )
    msgs = [{"role": "system", "content": sys_msg}, {"role": "user", "content": user_msg}]
    prompt_text = tokenizer.apply_chat_template(
        msgs,
        tokenize=False,
        add_generation_prompt=True,
    )
    raw = generate_text_non_stream(
        tokenizer,
        pipe,
        prompt_text,
        max_new_tokens=VAULT_EDIT_MAX_TOKENS,
    )
    return parse_vault_edit_json(raw)


def _stream_generation_kw(tokenizer: Any, streamer: Any) -> dict[str, Any]:
    """Kwargs for model.generate — aligned with module-level LLM_* and faster greedy path."""
    kw: dict[str, Any] = {
        "max_new_tokens": LLM_MAX_NEW_TOKENS,
        "do_sample": LLM_DO_SAMPLE,
        "pad_token_id": tokenizer.pad_token_id,
        "streamer": streamer,
    }
    eos = getattr(tokenizer, "eos_token_id", None)
    if eos is not None:
        kw["eos_token_id"] = eos
    if LLM_DO_SAMPLE:
        kw["temperature"] = LLM_TEMPERATURE
        kw["top_p"] = LLM_TOP_P
    return kw


def iter_answer_tokens_stream(tokenizer: Any, pipe: Any, prompt_text: str) -> Iterator[str]:
    """
    Stream decoded fragments while the quantized model generates (for FastAPI SSE).
    Uses TextIteratorStreamer + background generate on the pipeline's underlying model.
    """
    import torch
    from threading import Thread

    from transformers import TextIteratorStreamer

    model = pipe.model
    enc = tokenizer(prompt_text, return_tensors="pt")
    device = next(model.parameters()).device
    enc = {k: v.to(device) for k, v in enc.items()}

    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    gen_kw = _stream_generation_kw(tokenizer, streamer)

    def _worker() -> None:
        try:
            with torch.inference_mode():
                model.generate(**enc, **gen_kw)
        except Exception:
            logger.exception("Streaming generation failed")
            raise

    Thread(target=_worker, daemon=True).start()
    yield from streamer


def _print_retrieved_sources(out: dict[str, Any]) -> None:
    """Portfolio-style transparency: show which chunks were retrieved (file / title hints)."""
    ctx = out.get("context")
    if not ctx:
        print("(No separate context list in chain output.)")
        return
    print("--- Retrieved sources (for transparency) ---")
    if isinstance(ctx, list):
        for i, doc in enumerate(ctx, 1):
            meta = getattr(doc, "metadata", None) or {}
            src = meta.get("source_file", "?")
            label = (
                meta.get("source_title")
                or meta.get("title")
                or meta.get("question")
                or ""
            )
            preview = (getattr(doc, "page_content", "") or "")[:120].replace("\n", " ")
            extra = f" — {preview}..." if preview else ""
            print(f"  [{i}] file={src} | {str(label)[:60]}{extra}")
    else:
        print(f"  {ctx!r}")
    print("---")


def run_chat(chain, *, show_sources: bool = False) -> None:
    """Simple REPL: send {"input": question} into the retrieval chain and print the answer."""
    print("Local RAG (documents). Type exit, quit, or Ctrl+C to leave.")
    if show_sources:
        print("(Showing retrieved sources before each answer.)")
    while True:
        try:
            q = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not q:
            continue
        if q.lower() in ("exit", "quit"):
            break
        try:
            out = chain.invoke({"input": q})
            if show_sources:
                _print_retrieved_sources(out)
            answer = out.get("answer", out)
            print("Assistant:", answer)
        except Exception as e:
            logger.exception("Error while generating answer: %s", e)


def run_single_query(chain, question: str, *, show_sources: bool) -> None:
    """One-shot mode for scripts, demos, and CI (no interactive loop)."""
    out = chain.invoke({"input": question})
    if show_sources:
        _print_retrieved_sources(out)
    answer = out.get("answer", out)
    print("Assistant:", answer)


def parse_args():
    """CLI: --reindex rebuilds vectors; --model selects which HF instruct checkpoint to load."""
    p = argparse.ArgumentParser(
        description="Local RAG over Markdown vault (paths.vault_dir in config.yaml)."
    )
    p.add_argument(
        "--config",
        type=Path,
        default=None,
        metavar="PATH",
        help="YAML config file (default: config.yaml next to app.py if that file exists)",
    )
    p.add_argument(
        "--reindex",
        action="store_true",
        help="Clear ChromaDB and rebuild the index from vault_dir (Markdown)",
    )
    p.add_argument(
        "--model",
        type=str,
        default=DEFAULT_LLM_ID,
        choices=(DEFAULT_LLM_ID, ALT_LLM_ID),
        help=f"Hugging Face model id (default: {DEFAULT_LLM_ID})",
    )
    p.add_argument(
        "--top-k",
        type=int,
        default=None,
        metavar="K",
        help="Override number of retrieved chunks (default: from config or 4)",
    )
    p.add_argument(
        "--query",
        type=str,
        default=None,
        metavar="TEXT",
        help="Ask one question and exit (non-interactive; good for demos)",
    )
    p.add_argument(
        "--show-sources",
        action="store_true",
        help="Print retrieved chunk summary before the assistant answer",
    )
    return p.parse_args()


def main() -> None:
    """
    Program flow:
      reindex: load vault Markdown -> embed -> persist Chroma
      else:    load Chroma from disk
      then:    load quantized LLM -> build retrieval chain -> interactive chat
    """
    global RETRIEVAL_TOP_K

    args = parse_args()
    _venv_consistency_warning()
    cfg_path = resolve_config_path(args.config)
    apply_app_config(load_app_config_file(cfg_path))
    if cfg_path:
        logger.info("Loaded config: %s", cfg_path)
    logger.info(
        "LLM generation: max_new_tokens=%s do_sample=%s",
        LLM_MAX_NEW_TOKENS,
        LLM_DO_SAMPLE,
    )

    if args.top_k is not None:
        RETRIEVAL_TOP_K = max(1, int(args.top_k))

    if args.reindex:
        docs = load_all_documents_for_reindex()
        try:
            vs = build_vectorstore(docs, CHROMA_DIR, clear=True)
        except ValueError as e:
            logger.error("%s", e)
            sys.exit(1)
    else:
        try:
            vs = load_vectorstore(CHROMA_DIR)
        except FileNotFoundError as e:
            logger.error("%s", e)
            sys.exit(1)

    logger.info("Loading LLM %s (4-bit)...", args.model)
    tokenizer, pipe = build_llm_pipeline(args.model)
    chain = build_retrieval_chain(vs, tokenizer, pipe, top_k=RETRIEVAL_TOP_K)

    if args.query:
        run_single_query(chain, args.query.strip(), show_sources=args.show_sources)
    else:
        run_chat(chain, show_sources=args.show_sources)


if __name__ == "__main__":
    main()
