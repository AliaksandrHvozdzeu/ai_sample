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

### Important behaviour

The assistant is told to answer **only** from the retrieved document text.  
If the answer is not there, it should say exactly:

**`I could not find this in the documents.`**

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
