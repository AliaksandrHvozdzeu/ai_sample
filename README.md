# Local RAG chat (Markdown vault → Chroma → small LLM)

This project runs **on your machine**. It indexes **Markdown notes** from a vault (`paths.vault_dir` in `config.yaml`), stores vectors in **`chroma_db/`**, and answers using a **small local instruct model** (4-bit). Retrieval + generation are grounded on **your vault**, not the open web.

**Current layout:**

| Piece | Role |
|-------|------|
| `app.py` | CLI: `--reindex`, interactive chat, `--query`; loads YAML and builds/runs the stack. |
| `web/server.py` | FastAPI on port **8000**: serves **`web/static/index.html`** (chat + **3D** wikilink graph), **`POST /api/rag/stream`** (SSE), health + graph JSON APIs. |
| `web/vault_server.py` | Optional “light” server on port **8001**: **`GET /api/vault/graph`** + same HTML; **no LLM** loaded — **`POST /api/rag/stream`** is **not** registered here (use **8000** for chat). |

Indexing is **Markdown-only** (chunks split at `##` / `###`). There is **no** separate JSON knowledge-base pipeline.

---

## What you need first

- **Python 3.12** is strongly recommended ( **`scripts/setup_venv.ps1` uses `py -3.12`** when available ). On **Python 3.14**, official **CUDA wheels for PyTorch are often absent**, so `pip` may install **`torch` CPU-only** — the UI works but the LLM stays on CPU.
- **NVIDIA GPU** with enough VRAM (e.g. **8 GB**) for the default quantized models.
- If **`torch.version.cuda`** is **`None`**, inference uses **CPU** (much slower). **`setup_venv.ps1`** reinstalls **`torch`** from the **PyTorch CUDA 12.4** index after `requirements.txt` so the GPU can be used on supported Python versions.

### Optional: verify the GPU

After setup:

```powershell
.\scripts\check_gpu.ps1
```

You should see a line like `2.x.x+cu124 True` (CUDA build + `torch.cuda.is_available()`).

### Why the chat can feel slow

1. **GPU vs CPU** — The LLM is intended to run on **CUDA**. Startup logs from `app.py` / the web server say whether CUDA is available.
2. **Token streaming** — Answers are generated token-by-token; **`llm.max_new_tokens`** in `config.yaml` caps the longest reply (default **512**).
3. **Sampling** — **`do_sample: true`** (default) is slower than greedy **`do_sample: false`** for factual RAG.
4. **Context size** — Lower **`retrieval.top_k`** slightly shrinks the retrieved prompt.

**Faster preset** (optional, in `config.yaml`):

```yaml
llm:
  max_new_tokens: 256
  do_sample: false
```

Restart the web app or CLI after changing **`config.yaml`**.

---

## Project folders (short overview)

| Path | Purpose |
|------|---------|
| `vault/` | Your `.md` notes (or set `paths.vault_dir` elsewhere). Chunked by `##` / `###`. |
| `chroma_db/` | Persisted Chroma index. Rebuilt with **`python app.py --reindex`**. |
| `app.py` | CLI entry: index + chat. |
| `config.yaml` | Paths, **`retrieval.top_k`**, optional **`llm`** generation settings, system prompt. |
| `web/static/index.html` | Single-page UI: **chat (left)** + **3D graph (right)**; uses Three.js + **3d-force-graph**. |
| `requirements.txt` | Runtime dependencies. |
| `requirements-dev.txt` | Dev deps (e.g. **pytest**). |
| `scripts/setup_venv.ps1` | Creates **`.venv`** (prefers Python **3.12**), installs deps, then **`torch`** with **CUDA 12.4** from PyTorch’s index. |
| `scripts/check_gpu.ps1` | Quick **`torch` + CUDA** check. |
| `scripts/run_app.ps1` / `run_web.ps1` / `run_vault_graph.ps1` | Convenience wrappers around **`python app.py`** and **uvicorn**. |
| `package_gan_ai.py` | Optional release bundle. |
| `Dockerfile` / `docker-compose.yml` | Containerized web app; mounts **`vault/`** and Chroma data. |

---

## Markdown vault

1. Set **`paths.vault_dir`** in **`config.yaml`** (relative to project root), e.g. **`vault`**.
2. Add **`.md`** files. Optional YAML frontmatter (`tags`, etc.) when **PyYAML** is installed.
3. Chunking splits at **`##`** and **`###`**. Default skipped dirs: **`.obsidian`**, **`.git`**, **`node_modules`**, **`.trash`** (override with **`vault.exclude_dir_names`**).
4. Rebuild the index: **`python app.py --reindex`**.

**Wikilink graph:** open **`http://127.0.0.1:8000/rag/chat`** — notes are **spheres**, **`[[wikilinks]]`** are **edges**; retrieved chunks **pulse** on the graph during each answer.

---

## Step 1 — Virtual environment (Windows, recommended)

From the repo root:

```powershell
cd D:\path\to\AI_SAMPLE
.\scripts\setup_venv.ps1
```

This:

- Creates **`.venv`** with **`py -3.12`** when possible.
- Runs **`pip install -r requirements.txt`**.
- Uninstalls the CPU **`torch`** from PyPI and installs **`torch`** from **`https://download.pytorch.org/whl/cu124`** (CUDA **12.4** runtime bundled with the wheel; works with current NVIDIA drivers).

Optionally dot-source **`env_local.ps1`** (created once) so Hugging Face caches live under **`./.cache/`**:

```powershell
. .\env_local.ps1
```

**Manual venv (any OS):** `python -m venv .venv`, activate, then `pip install -r requirements.txt`, then install **CUDA `torch`** following [pytorch.org](https://pytorch.org/get-started/locally/) for your platform.

---

## Step 2 — Index and run (CLI)

```powershell
.\scripts\run_app.ps1 --reindex
.\scripts\run_app.ps1
```

Or:

```bash
python app.py --reindex
python app.py
```

- **`exit`** / **`quit`** or Ctrl+C leaves the CLI loop.
- Flags: **`--model`**, **`--query "..."`**, **`--show-sources`**, **`--top-k`**, **`--config`**.

If nothing in the retrieved context answers the question, the configured system prompt asks the model to reply exactly:

**`I could not find this in the documents.`**

---

## Web UI (browser + SSE)

After indexing, from project root:

```powershell
.\scripts\run_web.ps1
```

Or:

```bash
uvicorn web.server:app --host 127.0.0.1 --port 8000
```

Open **`http://127.0.0.1:8000/rag/chat`** (**`/`** redirects there). **One page:** **chat on the left**, **interactive 3D vault graph on the right** (orbit / zoom). Each answer sends an SSE **`sources`** event first so matching notes **pulse** (amber). **Show sources** only toggles the source list in the chat column.

The header toolbar includes **Upload .md** (writes files into the vault folder), **Reindex** (rebuilds Chroma and reloads the vector store — status text shows chunk/file counts and timestamp), and **Vault files** (expandable list of **`*.md`** paths under the vault).

| Path | Purpose |
|------|---------|
| `/rag/chat` | Main UI (**`index.html`**) — Chroma + LLM + graph. |
| `/vault/graph`, `/graph` | Redirect to **`/rag/chat`**. |
| `POST /api/rag/stream` | SSE chat (alias: **`/api/chat/stream`**). |
| `GET /api/rag/health` | JSON status (alias: **`/api/health`**). |
| `GET /api/vault/graph` | Wikilink graph JSON (no generation). |
| `GET /api/vault/files` | Sorted list of vault-relative **`*.md`** paths. |
| `POST /api/vault/upload` | Multipart upload of **`.md`** files (saved under vault root by basename). |
| `POST /api/vault/reindex` | Rebuild Chroma from vault + reload in-memory store (same as **`python app.py --reindex`**). |

### Lightweight server (port 8001)

```powershell
.\scripts\run_vault_graph.ps1
```

→ **`http://127.0.0.1:8001/rag/chat`** serves the **same HTML** and **`GET /api/vault/graph`**, but **no LLM** is loaded — use **`http://127.0.0.1:8000`** for real chat.

---

## Docker Compose

- **`docker-compose.yml`** exposes **8000** and mounts **`./vault`** read-only; Chroma can live in a named volume.
- Build the index on the host (**`python app.py --reindex`**) before first run, or mount **`chroma_db`** (see comments in **`docker-compose.yml`**).

---

## Packaging (`package_gan_ai.py`)

```bash
python package_gan_ai.py --family gan-ai --version 0.2.0 --include-app --include-vault --include-packager --zip
```

- **`--include-vault`** — includes **`vault/`**.
- **`--include-chroma`** — includes **`chroma_db/`** (large).

---

## Tests

```bash
pip install -r requirements.txt -r requirements-dev.txt
pytest
```

---

## Troubleshooting

| Problem | What to try |
|---------|-------------|
| **`Database not found`** | Run **`python app.py --reindex`** after adding notes and setting **`paths.vault_dir`**. |
| **`No documents to index`** | Ensure **`vault_dir`** exists and `.md` files have content under **`##`/`###`** sections. |
| **Always CPU / very slow** | Use Python **3.12** + rerun **`setup_venv.ps1`**; run **`check_gpu.ps1`**; install CUDA **`torch`** per [pytorch.org](https://pytorch.org/get-started/locally/). |
| **Chat 503 on port 8001** | The light server has **no** LLM — use port **8000** (`web.server`) for **`/api/rag/stream`**. |
| **Graph empty / wrong** | Check **`paths.vault_dir`** and that notes use **`[[wikilinks]]`** to existing `.md` paths. |

---

## Author

**Hvozdzeu Aliaksandr**, 2026, Vilnius — see **`app.py`** header.
