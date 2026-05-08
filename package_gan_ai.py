"""
Build a versioned *release bundle* named <family>-<version> (default gan-ai-0.0.1).

IMPORTANT — what this is NOT:
  Training or “converting” JSON into new neural weights requires fine-tuning (LoRA/full fine-tune).
  This script does NOT produce a new Hugging Face weight checkpoint.

What this IS:
  A distributable folder (and optional .zip) with manifest.json, MODEL_CARD.md,
  and optional copies of vault/, chroma_db/, app.py, requirements.txt.

Usage:
  py package_gan_ai.py
  py package_gan_ai.py --version 0.1.42 --family gan-ai --zip --include-app --include-vault
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path

DEFAULT_FAMILY = "gan-ai"
DEFAULT_VERSION = "0.0.1"

PROJECT_ROOT = Path(__file__).resolve().parent
VAULT_DIR = PROJECT_ROOT / "vault"
CHROMA_DIR = PROJECT_ROOT / "chroma_db"
APP_ENTRY = "app.py"
REQUIREMENTS = "requirements.txt"
PACKAGER_SCRIPT = "package_gan_ai.py"

DEFAULT_LLM_ID = "Qwen/Qwen2.5-1.5B-Instruct"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def _write_model_card(
    out_dir: Path,
    *,
    release_full_name: str,
    release_family: str,
    release_version: str,
) -> None:
    text = f"""---
license: mit
tags:
  - rag
  - retrieval
  - local-llm
  - {release_family}
base_model: {DEFAULT_LLM_ID}
---

# {release_full_name}

**{release_full_name}** is a *RAG application bundle* (not a standalone fine-tuned weight file).

- **Retrieval**: Chroma + `{EMBEDDING_MODEL}`
- **Generation**: local instruct model `{DEFAULT_LLM_ID}` (4-bit in `app.py`)
- **Knowledge**: Markdown notes under `vault/` (as indexed by `app.py`)

Version: `{release_version}` · Family: `{release_family}`

This repository folder is produced by `package_gan_ai.py` for versioning and hand-off.  
To get **real** custom weights named after this product, you must **fine-tune** a base model on your data (e.g. LoRA with Hugging Face / Axolotl / Unsloth), then upload those weights separately.

## Run

```bash
python -m venv .venv
.venv\\Scripts\\activate
pip install -r requirements.txt
python app.py --reindex
python app.py
```

Replace `app.py` / paths if you relocate the bundle.
"""
    (out_dir / "MODEL_CARD.md").write_text(text, encoding="utf-8")


def _write_manifest(
    out_dir: Path,
    options: dict,
    *,
    release_full_name: str,
    release_family: str,
    release_version: str,
) -> None:
    manifest = {
        "release_id": release_full_name,
        "release_family": release_family,
        "version": release_version,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "type": "rag_application_bundle",
        "components": {
            "base_llm_hf_id": DEFAULT_LLM_ID,
            "embedding_model_hf_id": EMBEDDING_MODEL,
            "vector_store": "chromadb",
            "runtime": "python_local_rag_cli",
        },
        "package_options": options,
        "notes": (
            "This is not a trained model checkpoint; it is metadata + optional data copies. "
            "Weights are still downloaded from Hugging Face at runtime unless you ship a cache."
        ),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def _copy_if_exists(src: Path, dst: Path) -> bool:
    if not src.exists():
        return False
    if src.is_dir():
        shutil.copytree(src, dst, dirs_exist_ok=True)
    else:
        shutil.copy2(src, dst)
    return True


def _zip_dir(src_dir: Path, zip_path: Path, *, arc_prefix: str) -> None:
    """Zip folder so archive root contains arc_prefix/... (single top-level directory)."""
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for path in src_dir.rglob("*"):
            if path.is_file():
                arcname = Path(arc_prefix) / path.relative_to(src_dir)
                zf.write(path, arcname=str(arcname))


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Package RAG project as <family>-<version> (e.g. gan-ai-0.1.5)."
    )
    parser.add_argument(
        "--family",
        default=DEFAULT_FAMILY,
        help=f"Release family name (default: {DEFAULT_FAMILY})",
    )
    parser.add_argument(
        "--version",
        default=DEFAULT_VERSION,
        help=f"Semantic version string (default: {DEFAULT_VERSION}). CI often passes 0.1.${{ run_number }}.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output directory (default: ./dist/<family>-<version>)",
    )
    parser.add_argument("--zip", action="store_true", help="Also write <family>-<version>.zip under ./dist/")
    parser.add_argument(
        "--include-vault",
        action="store_true",
        help=f"Copy {VAULT_DIR.name}/ into the bundle",
    )
    parser.add_argument(
        "--include-chroma",
        action="store_true",
        help=f"Copy {CHROMA_DIR.name}/ (can be large)",
    )
    parser.add_argument(
        "--include-app",
        action="store_true",
        help="Copy app.py and requirements.txt into the bundle",
    )
    parser.add_argument(
        "--include-packager",
        action="store_true",
        help=f"Copy {PACKAGER_SCRIPT} into the bundle",
    )
    args = parser.parse_args()

    release_family: str = args.family.strip()
    release_version: str = args.version.strip()
    release_full_name = f"{release_family}-{release_version}"

    out_dir = args.out if args.out is not None else PROJECT_ROOT / "dist" / release_full_name
    out_dir.mkdir(parents=True, exist_ok=True)

    opts = {
        "include_vault": args.include_vault,
        "include_chroma": args.include_chroma,
        "include_app": args.include_app,
        "include_packager": args.include_packager,
    }
    _write_manifest(
        out_dir,
        opts,
        release_full_name=release_full_name,
        release_family=release_family,
        release_version=release_version,
    )
    _write_model_card(
        out_dir,
        release_full_name=release_full_name,
        release_family=release_family,
        release_version=release_version,
    )

    copied: list[str] = []
    if args.include_app:
        if _copy_if_exists(PROJECT_ROOT / APP_ENTRY, out_dir / APP_ENTRY):
            copied.append(APP_ENTRY)
        if _copy_if_exists(PROJECT_ROOT / REQUIREMENTS, out_dir / REQUIREMENTS):
            copied.append(REQUIREMENTS)

    if args.include_packager:
        if _copy_if_exists(PROJECT_ROOT / PACKAGER_SCRIPT, out_dir / PACKAGER_SCRIPT):
            copied.append(PACKAGER_SCRIPT)

    if args.include_vault:
        dst = out_dir / VAULT_DIR.name
        if VAULT_DIR.is_dir():
            shutil.copytree(VAULT_DIR, dst, dirs_exist_ok=True)
            copied.append(f"{VAULT_DIR.name}/")
        else:
            print(f"Warning: {VAULT_DIR} not found; skipped.", file=sys.stderr)

    if args.include_chroma:
        dst = out_dir / CHROMA_DIR.name
        if CHROMA_DIR.is_dir():
            shutil.copytree(CHROMA_DIR, dst, dirs_exist_ok=True)
            copied.append(f"{CHROMA_DIR.name}/")
        else:
            print(f"Warning: {CHROMA_DIR} not found; skipped.", file=sys.stderr)

    print(f"Created bundle: {out_dir}")
    print("  manifest.json, MODEL_CARD.md")
    if copied:
        print(f"  copied: {', '.join(copied)}")

    if args.zip:
        zip_path = (PROJECT_ROOT / "dist" / f"{release_full_name}.zip")
        zip_path.parent.mkdir(parents=True, exist_ok=True)
        _zip_dir(out_dir, zip_path, arc_prefix=release_full_name)
        print(f"ZIP: {zip_path}")

    print()
    print(
        "Reminder: this bundle versions your RAG *product* metadata and optional files. "
        "It does not create new neural weights. For a real custom LLM checkpoint, fine-tune a base model."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
