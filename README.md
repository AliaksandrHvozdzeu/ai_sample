# Local RAG chat (JSON → Chroma → small LLM)

This project runs **on your computer**. It reads text from JSON files, saves a search index, and answers questions using a **small local model**. The answers should follow **your documents**, not random internet facts.

---

## What you need first

- **Python** 3.10–3.12 works best (some newer Python versions can cause trouble with PyTorch).
- **NVIDIA GPU** with enough VRAM is recommended (for example 8 GB). The code uses a light model in 4-bit mode.
- If PyTorch is installed **without CUDA**, everything runs on **CPU** — it works, but it is **much slower**.

---

## Project folders (short overview)

| Folder / file | What it is |
|---------------|------------|
| `json/` | Put your `.json` files here. The app reads **all** `*.json` in this folder. |
| `chroma_db/` | The vector database is stored here after indexing. You can delete it and rebuild with `--reindex`. |
| `app.py` | Main program: index + chat. |
| `requirements.txt` | Python libraries to install. |
| `package_gan_ai.py` | Optional script to pack a **release bundle** (manifest + files). |
| `config.yaml` | **Optional.** Change folders, retrieval `top_k`, or system prompt **without editing Python**. |
| `web/server.py` | **FastAPI** app: browser UI + **streaming** answers (SSE). |
| `Dockerfile` / `docker-compose.yml` | Run the web app in Docker with a **volume for Chroma**. |

---

## Why this repo is a bit special (for your portfolio)

These ideas are simple, but interviewers like **clear settings** and **honest RAG** (you show what the model actually saw):

- **`config.yaml`** — paths, how many chunks to retrieve (`top_k`), and the system prompt in one place.
- **`--show-sources`** — before each answer, the app can print **which files / chunks** were retrieved (transparency).
- **`--query "..."`** — one question and exit (good for **demos**, scripts, or screen recordings).
- **`--top-k`** — quick experiment: more chunks = more context (but also more noise).
- **Web UI** — chat in the browser with **token streaming** (not only the terminal).
- **Docker Compose** — optional **one-command** setup with a **persistent Chroma volume**.

---

## JSON format (two types)

The app supports:

**A) FAQ style** — each item has `question` and `answer`.

**B) Article style** — each item has `title`, `content`, and optionally `url`.

The root JSON can be a **list** of objects, or an **object** that contains a list (see the code if your file looks different).

---

## Step 1 — Virtual environment (recommended)

Using a virtual environment keeps libraries inside your project folder (cleaner PC).

**Windows (PowerShell):**

```powershell
cd path\to\your\project
py -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**Linux / macOS:**

```bash
cd path/to/your/project
python3 -m venv .venv
source .venv/bin/activate
```

---

## Step 2 — Install libraries

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### GPU note (NVIDIA)

If you see that the GPU is **not** used, install **PyTorch with CUDA** from the official site:  
[https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/)

Pick the options that match your system, then follow their `pip` command (often you uninstall the CPU-only `torch` first).

---

## Step 3 — Put data and build the index

1. Copy your JSON files into the **`json`** folder.
2. Build or rebuild the index:

```bash
python app.py --reindex
```

This **clears** the old `chroma_db` folder (if it exists) and creates a new index from your JSON.

If there are no valid documents, the program will stop with an error — check your JSON paths and format.

---

## Step 4 — Chat with the assistant

Start the chat **without** `--reindex`:

```bash
python app.py
```

Wait until the model finishes loading. Then you will see a prompt like **`You:`**.

- Type your question in **English** (the system prompt is written for English answers).
- The assistant line starts with **`Assistant:`**.
- Type **`exit`** or **`quit`** to leave, or press **Ctrl+C**.

### Different LLM (optional)

Default model is **Qwen2.5 1.5B Instruct**. You can switch to the heavier option:

```bash
python app.py --model unsloth/Llama-3.2-3B-Instruct
```

### One question only (demo mode)

```bash
python app.py --query "What is the office Wi-Fi network name?"
```

### Show where the answer came from (retrieved chunks)

```bash
python app.py --show-sources
python app.py --query "Your question" --show-sources
```

### Custom config file

```bash
python app.py --config my_settings.yaml
```

If you **do not** pass `--config`, the app loads **`config.yaml`** automatically **when that file exists** next to `app.py`.

### More chunks from the index (optional)

```bash
python app.py --top-k 6
```

### Important behaviour

The assistant is told to answer **only** from the retrieved document text.  
If the answer is not there, it should say exactly:

**`I could not find this in the documents.`**

---

## Web UI (browser + streaming)

After you install libraries and build the index (`python app.py --reindex`), start the server:

```bash
uvicorn web.server:app --host 127.0.0.1 --port 8000
```

Open **http://127.0.0.1:8000** in your browser. Type a question and press **Send**. You should see the answer appear **word by word** (streaming).

- Turn on **Show sources** to see which JSON chunks were used (same idea as `--show-sources` in the terminal).
- Check **http://127.0.0.1:8000/api/health** if something fails (for example missing Chroma index).

**Environment variables (optional):**

| Variable | Meaning |
|----------|---------|
| `RAG_MODEL_ID` | Same choices as CLI (`Qwen/...` or `unsloth/Llama-3.2-3B-Instruct`). |
| `RAG_CONFIG` | Full path to a YAML config file (if you do not use the default `config.yaml`). |

**Note:** The first request can be slow while the model loads into VRAM.

---

## Docker Compose (one command)

You need **Docker** and **Docker Compose** installed.

### Basic idea

- **`docker-compose.yml`** starts the **web UI** on port **8000**.
- Folder **`json/`** on your PC is mounted into the container (**read-only**).
- Chroma is stored in a **named volume** called **`chroma_data`** so it **does not disappear** when you restart the container.

### Typical workflow

1. Put JSON files in **`json/`** on your machine (same as before).
2. On your machine (with GPU if possible), build the index once:

   ```bash
   python app.py --reindex
   ```

3. Start Docker:

   ```bash
   docker compose up --build
   ```

4. Open **http://localhost:8000**.

If the browser shows an error, open **http://localhost:8000/api/health**.  
Often the problem is: **the container has an empty Chroma folder**. The named volume starts **empty**. You can fix it in two ways:

- **Bind mount your host folder** (easy when you already ran `--reindex` on the host):

  ```bash
  cp docker-compose.override.example.yml docker-compose.override.yml
  ```

  Then edit paths if needed and run `docker compose up --build` again.  
  (`docker-compose.override.yml` is in `.gitignore` so it stays local.)

- Or **copy** your host `./chroma_db` into the volume (more advanced).

### GPU inside Docker

You need the **NVIDIA Container Toolkit** on Linux/WSL2 (or similar on Windows). Then run Compose with GPU access, for example:

```bash
docker compose up --build
```

and enable GPU for the service (exact flags depend on your Docker version).  
If the container runs on **CPU only**, loading the **4-bit** model may fail or be very slow — the README cannot replace your GPU driver docs.

---

## Packaging a release (`gan-ai` bundle)

This does **not** train a new AI model. It only packs **files + manifest** for sharing or versioning.

Examples:

```bash
python package_gan_ai.py
```

With version name and zip:

```bash
python package_gan_ai.py --family gan-ai --version 0.2.0 --include-app --include-json --include-packager --zip
```

Rough meaning of flags:

- **`--include-app`** — copies `app.py` and `requirements.txt`.
- **`--include-json`** — copies the `json/` folder.
- **`--include-chroma`** — copies `chroma_db/` (can be **large**).
- **`--include-packager`** — copies `package_gan_ai.py`.
- **`--zip`** — creates a zip file under **`dist/`**.

Output folders look like **`dist/gan-ai-<version>/`** plus **`manifest.json`** and **`MODEL_CARD.md`**.

---

## Tests (for developers)

```bash
pip install -r requirements.txt -r requirements-dev.txt
pytest
```

---

## Troubleshooting (quick)

| Problem | Idea |
|--------|------|
| “Database not found” | Run `python app.py --reindex` once after adding JSON. |
| Very slow answers | Check CUDA PyTorch; first answer after start is often slower. |
| BitsAndBytes warnings | Usually harmless; updating libraries later can remove them. |

---

## Author line (from source)

Author: **Hvozdzeu Aliaksandr**, 2026, Vilnius — see `app.py` header.
