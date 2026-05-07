---
tags:
  - rag
  - demo
---

# Sample vault note for local RAG

This file exists to verify that Markdown from `vault/` is indexed together with JSON.

## Quick facts

The indexer splits notes at `##` and `###` headings. Frontmatter keys like `tags` are stored as metadata when PyYAML is installed.

## Where this lives

On disk this note is: `vault/TestNote.md` relative to the project root. After `python app.py --reindex`, questions about this content should retrieve chunks from this path.
