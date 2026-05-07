"""
Author: Hvozdzeu Aliaksandr, 2026, Vilnius.

Local RAG (Retrieval-Augmented Generation) CLI application.

What this script does (high level):
  1. Loads structured records from JSON files in ./json/ (FAQ or article format), and optionally
     Markdown notes from an Obsidian vault path (set paths.vault_dir in config.yaml).
  2. Embeds each record with sentence-transformers and stores vectors in ChromaDB (./chroma_db/).
  3. At query time, retrieves the most similar chunks, injects them as "context", and runs a
     small local LLM (4-bit quantized) to answer in English, grounded on that context only.

Run:
  python app.py --reindex   # rebuild the vector index from JSON + optional vault
  python app.py             # chat using the existing index
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import re
import sys
from pathlib import Path
from typing import Any, Iterable, Iterator

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
JSON_DIR = BASE_DIR / "json"  # input: all *.json files here are indexed
CHROMA_DIR = BASE_DIR / "chroma_db"  # persisted Chroma vector database on disk
# Optional Obsidian / Markdown vault (directory of *.md). Set paths.vault_dir in config.yaml.
VAULT_DIR: Path | None = None

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # fast local embeddings for retrieval
DEFAULT_LLM_ID = "Qwen/Qwen2.5-1.5B-Instruct"  # default instruct model (lighter)
ALT_LLM_ID = "unsloth/Llama-3.2-3B-Instruct"  # alternative instruct model (heavier)

RETRIEVAL_TOP_K = 4  # chunks to retrieve; overridden by config.yaml or --top-k

# Instructions to the LLM: must stay grounded; exact phrase when context is insufficient.
_SYSTEM_PROMPT_DEFAULT = """You are a precise corporate assistant. Answer questions STRICTLY using only the provided document context below. \
Do not invent facts or use outside knowledge. \
If the answer is not in the context, reply exactly: I could not find this in the documents. \
If the fragment metadata includes a source (title, path, or URL), mention it briefly at the end of your answer."""
SYSTEM_PROMPT = _SYSTEM_PROMPT_DEFAULT

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
    global JSON_DIR, CHROMA_DIR, VAULT_DIR, EMBEDDING_MODEL, SYSTEM_PROMPT, RETRIEVAL_TOP_K

    paths = cfg.get("paths") or {}
    if isinstance(paths.get("json_dir"), str) and paths["json_dir"].strip():
        JSON_DIR = (BASE_DIR / paths["json_dir"].strip()).resolve()
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

    prompt = cfg.get("prompt") or {}
    if isinstance(prompt.get("system"), str) and prompt["system"].strip():
        SYSTEM_PROMPT = prompt["system"].strip()


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


def _format_faq_item(obj: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    """
    Format A (FAQ): one object has "question" and "answer".
    Returns (text_for_embedding, metadata_dict).
    """
    q = (obj.get("question") or "").strip()
    a = (obj.get("answer") or "").strip()
    text = f"Question: {q}\nAnswer: {a}"
    meta: dict[str, Any] = {
        "format": "faq",
        "question": q,
        "source_title": q[:200] if q else None,
    }
    return text, meta


def _format_article_item(obj: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    """
    Format B (article/page): "title", "content", optional "url".
    URL is both in the body text (for the model) and in metadata (for filtering/display).
    """
    title = (obj.get("title") or "").strip()
    content = (obj.get("content") or "").strip()
    url = obj.get("url")
    if url is not None:
        url = str(url).strip() or None
    text = f"Title: {title}\nContent: {content}"
    if url:
        text += f"\nURL: {url}"
    meta: dict[str, Any] = {
        "format": "article",
        "title": title,
        "url": url,
        "source_title": title or None,
    }
    return text, meta


def _normalize_records(raw: Any, path: Path) -> list[dict[str, Any]]:
    """
    Accept flexible JSON layouts:
      - top-level list of objects -> use as-is
      - top-level dict with one list value -> use that list
      - top-level dict with several lists -> concatenate all list items (with a warning)
    Returns only dict rows (other types ignored).
    """
    if isinstance(raw, list):
        return [x for x in raw if isinstance(x, dict)]
    if isinstance(raw, dict):
        lists = [v for v in raw.values() if isinstance(v, list)]
        if len(lists) == 1:
            return [x for x in lists[0] if isinstance(x, dict)]
        out: list[dict[str, Any]] = []
        for v in raw.values():
            if isinstance(v, list):
                out.extend(x for x in v if isinstance(x, dict))
        if out:
            logger.warning(
                "File %s: multiple arrays in root object; merged all records.",
                path.name,
            )
            return out
    logger.warning("File %s: no object array found (skipped).", path.name)
    return []


def _object_to_document(obj: dict[str, Any], path: Path, index: int) -> Document | None:
    """
    Turn one JSON object into a LangChain Document:
      - page_content: string fed to the embedder and shown as retrieval context
      - metadata: stored in Chroma for filtering; important fields are duplicated into page_content
        so the LLM always sees file/title/url in the retrieved snippet.
    """
    has_qa = "question" in obj and "answer" in obj
    has_article = "title" in obj and "content" in obj

    # Prefer a single format; if both field sets exist, FAQ wins (same as previous behavior).
    if has_qa and not has_article:
        page, meta = _format_faq_item(obj)
    elif has_article and not has_qa:
        page, meta = _format_article_item(obj)
    elif has_qa and has_article:
        page, meta = _format_faq_item(obj)
        logger.debug("Object %s[%s]: both FAQ and article fields present; treated as FAQ.", path.name, index)
    else:
        logger.warning(
            "File %s[%s]: unknown format (need question+answer or title+content); skipped.",
            path.name,
            index,
        )
        return None

    # Skip useless rows (empty strings or degenerate templates).
    if not page.strip() or page.strip() == "Question:\nAnswer:" or page.endswith("Content:"):
        logger.warning("File %s[%s]: empty content after merging fields; skipped.", path.name, index)
        return None

    meta["source_file"] = path.name
    meta["chunk_index"] = index
    # Append a short metadata line so retrieval context includes citation hints even if
    # the chain does not expose raw Document.metadata to the prompt by default.
    meta_line_parts = [f"file={path.name}"]
    if meta.get("url"):
        meta_line_parts.append(f"URL={meta['url']}")
    if meta.get("source_title"):
        meta_line_parts.append(f"title={meta['source_title']}")
    page += "\n[Metadata: " + "; ".join(meta_line_parts) + "]"

    return Document(page_content=page, metadata=_clean_metadata(meta))


def load_json_documents(json_dir: Path) -> list[Document]:
    """
    Scan json_dir for *.json files, parse each file, normalize to record lists, and build Documents.
    Malformed files are skipped with a warning; processing continues.
    """
    if not json_dir.is_dir():
        logger.error("Directory not found: %s", json_dir)
        return []

    docs: list[Document] = []
    for path in sorted(json_dir.glob("*.json")):
        try:
            text = path.read_text(encoding="utf-8")
            if not text.strip():
                logger.warning("Empty file: %s", path.name)
                continue
            raw = json.loads(text)
        except json.JSONDecodeError as e:
            logger.warning("Invalid JSON in %s: %s", path.name, e)
            continue
        except OSError as e:
            logger.warning("Could not read %s: %s", path.name, e)
            continue

        records = _normalize_records(raw, path)
        if not records:
            continue

        for i, obj in enumerate(records):
            doc = _object_to_document(obj, path, i)
            if doc:
                docs.append(doc)

    if not docs:
        logger.warning("No documents loaded from %s", json_dir)
    else:
        logger.info("Loaded %s document(s) for indexing.", len(docs))
    return docs


# --- Obsidian / Markdown vault -------------------------------------------------

DEFAULT_VAULT_EXCLUDE_DIR_NAMES = frozenset({".obsidian", ".git", "node_modules"})


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
    text = f"Note: {note_title}\nPath: {vault_rel}\nSection: {sec_label}\n\n{section_text.strip()}"
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
    Recursively index *.md under vault_dir. Skips .obsidian, .git, node_modules in path components.
    Each file is split on ## / ### headings into multiple chunks.
    """
    excl = frozenset(exclude_dir_names) if exclude_dir_names is not None else DEFAULT_VAULT_EXCLUDE_DIR_NAMES
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
    """Merge JSON knowledge base documents and optional Obsidian vault chunks for --reindex."""
    docs: list[Document] = []
    docs.extend(load_json_documents(JSON_DIR))
    if VAULT_DIR is not None:
        if VAULT_DIR.is_dir():
            docs.extend(load_obsidian_documents(VAULT_DIR))
        else:
            logger.warning("paths.vault_dir is set but not a directory: %s (skipped).", VAULT_DIR)
    return docs


def build_vectorstore(documents: Iterable[Document], persist_dir: Path, clear: bool) -> Chroma:
    """
    Create or overwrite Chroma with embedded documents.
    If clear=True and persist_dir exists, delete it first (--reindex path).
    """
    embeddings = _make_embeddings()
    persist_dir.mkdir(parents=True, exist_ok=True)

    if clear and persist_dir.exists():
        import shutil

        shutil.rmtree(persist_dir)
        logger.info("Vector store directory cleared: %s", persist_dir)

    docs_list = list(documents)
    if not docs_list:
        hint = f"JSON dir: {JSON_DIR}"
        if VAULT_DIR:
            hint += f"; vault: {VAULT_DIR}"
        raise ValueError(f"No documents to index. Check {hint}.")

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
    generation_config.max_new_tokens = 512
    generation_config.do_sample = True
    generation_config.temperature = 0.2
    generation_config.top_p = 0.9
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
    k = top_k if top_k is not None else RETRIEVAL_TOP_K
    retriever = vectorstore.as_retriever(search_kwargs={"k": k})
    return retriever.invoke(question)


def documents_context_string(documents: list[Document]) -> str:
    """Join retrieved chunk texts for the LLM context window."""
    return "\n\n".join(d.page_content for d in documents)


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


def serialize_sources_for_ui(documents: list[Document]) -> list[dict[str, Any]]:
    """Compact JSON for the web UI / SSE \"sources\" event."""
    out: list[dict[str, Any]] = []
    for d in documents:
        meta = d.metadata or {}
        out.append(
            {
                "source_file": meta.get("source_file"),
                "vault_rel_path": meta.get("vault_rel_path"),
                "label": meta.get("source_title")
                or meta.get("title")
                or meta.get("question"),
                "preview": (d.page_content or "")[:240].replace("\n", " "),
            }
        )
    return out


def iter_answer_tokens_stream(tokenizer: Any, pipe: Any, prompt_text: str) -> Iterator[str]:
    """
    Stream decoded fragments while the quantized model generates (for FastAPI SSE).
    Uses TextIteratorStreamer + background generate on the pipeline's underlying model.
    """
    from threading import Thread

    from transformers import TextIteratorStreamer

    model = pipe.model
    enc = tokenizer(prompt_text, return_tensors="pt")
    device = next(model.parameters()).device
    enc = {k: v.to(device) for k, v in enc.items()}

    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    def _worker() -> None:
        try:
            model.generate(**enc, streamer=streamer)
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
        description="RAG over JSON files and optional Obsidian Markdown vault (see config.yaml)."
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
        help="Clear ChromaDB and rebuild the index from JSON + optional vault_dir",
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
      reindex: load JSON + optional vault -> embed -> persist Chroma
      else:    load Chroma from disk
      then:    load quantized LLM -> build retrieval chain -> interactive chat
    """
    global RETRIEVAL_TOP_K

    args = parse_args()
    cfg_path = resolve_config_path(args.config)
    apply_app_config(load_app_config_file(cfg_path))
    if cfg_path:
        logger.info("Loaded config: %s", cfg_path)

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
