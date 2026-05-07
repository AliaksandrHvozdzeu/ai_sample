"""
Unit tests for app.py (JSON → Document pipeline and helpers).
LLM / Chroma integration is not run here (no GPU, no large downloads).
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

import app as rag


class TestCleanMetadata:
    def test_drops_none(self) -> None:
        assert rag._clean_metadata({"a": 1, "b": None}) == {"a": 1}

    def test_coerces_non_scalar_to_str(self) -> None:
        assert rag._clean_metadata({"x": {"nested": 1}}) == {"x": "{'nested': 1}"}


class TestFormatFaqItem:
    def test_basic(self) -> None:
        text, meta = rag._format_faq_item(
            {"question": " Q ", "answer": " A "}
        )
        assert "Question: Q" in text and "Answer: A" in text
        assert meta["format"] == "faq"
        assert meta["question"] == "Q"


class TestFormatArticleItem:
    def test_with_url(self) -> None:
        text, meta = rag._format_article_item(
            {"title": "T", "content": "C", "url": " https://x.test "}
        )
        assert "Title: T" in text and "Content: C" in text
        assert "https://x.test" in text
        assert meta["url"] == "https://x.test"

    def test_url_none(self) -> None:
        _, meta = rag._format_article_item({"title": "T", "content": "C"})
        assert meta.get("url") is None


class TestNormalizeRecords:
    def test_top_level_list(self) -> None:
        raw = [{"a": 1}, "skip", 42]
        out = rag._normalize_records(raw, Path("f.json"))
        assert out == [{"a": 1}]

    def test_single_list_in_dict(self) -> None:
        raw = {"items": [{"x": 1}]}
        out = rag._normalize_records(raw, Path("f.json"))
        assert out == [{"x": 1}]


class TestObjectToDocument:
    def test_faq(self) -> None:
        doc = rag._object_to_document(
            {"question": "Q?", "answer": "A."},
            Path("wiki.json"),
            0,
        )
        assert doc is not None
        assert "Q?" in doc.page_content
        assert "[Metadata:" in doc.page_content
        assert doc.metadata.get("format") == "faq"
        assert doc.metadata.get("source_file") == "wiki.json"

    def test_article(self) -> None:
        doc = rag._object_to_document(
            {"title": "Hi", "content": "Body", "url": "https://u"},
            Path("x.json"),
            1,
        )
        assert doc is not None
        assert "Hi" in doc.page_content and "Body" in doc.page_content
        assert doc.metadata.get("format") == "article"

    def test_unknown_skipped(self) -> None:
        assert rag._object_to_document({"foo": 1}, Path("x.json"), 0) is None


class TestLoadJsonDocuments:
    def test_missing_dir(self) -> None:
        assert rag.load_json_documents(Path("/nonexistent/path/xyz")) == []

    def test_loads_sample_files(self, tmp_path: Path) -> None:
        d = tmp_path / "json_in"
        d.mkdir()
        (d / "a.json").write_text(
            json.dumps(
                [
                    {
                        "title": "T1",
                        "content": "C1",
                    }
                ]
            ),
            encoding="utf-8",
        )
        docs = rag.load_json_documents(d)
        assert len(docs) == 1
        assert "T1" in docs[0].page_content

    def test_invalid_json_skipped(self, tmp_path: Path) -> None:
        d = tmp_path / "j"
        d.mkdir()
        (d / "bad.json").write_text("{not json", encoding="utf-8")
        assert rag.load_json_documents(d) == []


class TestSystemPrompt:
    def test_has_grounding_rules(self) -> None:
        assert "document context" in rag.SYSTEM_PROMPT.lower()
        assert "could not find this in the documents" in rag.SYSTEM_PROMPT.lower()


class TestParseArgs:
    def test_defaults(self) -> None:
        import sys
        from unittest import mock

        with mock.patch.object(sys, "argv", ["app.py"]):
            args = rag.parse_args()
            assert args.model == rag.DEFAULT_LLM_ID
            assert args.reindex is False
            assert args.query is None
            assert args.show_sources is False


class TestWebHelpers:
    def test_serialize_sources_shape(self) -> None:
        from langchain_core.documents import Document

        docs = [
            Document(
                page_content="hello world",
                metadata={"source_file": "a.json", "title": "T"},
            )
        ]
        ser = rag.serialize_sources_for_ui(docs)
        assert len(ser) == 1
        assert ser[0]["source_file"] == "a.json"


class TestConfigHelpers:
    def test_load_missing_config_is_empty(self, tmp_path: Path) -> None:
        assert rag.load_app_config_file(tmp_path / "does-not-exist.yaml") == {}

    def test_resolve_config_explicit(self) -> None:
        p = Path("/tmp/x.yaml")
        assert rag.resolve_config_path(p) == p

    def test_vault_dir_from_config(self) -> None:
        # paths.vault_dir is resolved relative to app.BASE_DIR (repo root), not cwd.
        tmp = tempfile.mkdtemp(prefix="pytest_vault_", dir=str(rag.BASE_DIR))
        rel = Path(tmp).name
        prev_vault = rag.VAULT_DIR
        try:
            rag.apply_app_config({"paths": {"vault_dir": rel}})
            assert rag.VAULT_DIR == Path(tmp).resolve()
        finally:
            rag.VAULT_DIR = prev_vault
            Path(tmp).rmdir()


class TestObsidianHelpers:
    def test_split_body_intro_and_sections(self) -> None:
        body = "# Title line ignored here\n\nIntro para.\n\n## First\nA.\n\n### Nested\nB.\n"
        parts = rag._split_body_by_h2_h3(body)
        headings = [h for h, _ in parts]
        assert "" in headings
        assert "First" in headings
        assert "Nested" in headings

    def test_tags_for_metadata(self) -> None:
        assert rag._tags_for_metadata(["a", "b"]) == "a, b"
        assert rag._tags_for_metadata("x") == "x"
        assert rag._tags_for_metadata(None) is None

    def test_load_obsidian_skips_dot_obsidian(self, tmp_path: Path) -> None:
        vault = tmp_path / "v"
        vault.mkdir()
        obs = vault / ".obsidian"
        obs.mkdir()
        (obs / "secret.md").write_text("## X\nY", encoding="utf-8")
        (vault / "Note.md").write_text(
            "---\ntags: [work]\n---\n\n## Section\nBody text.",
            encoding="utf-8",
        )
        docs = rag.load_obsidian_documents(vault)
        assert len(docs) == 1
        assert docs[0].metadata.get("format") == "obsidian"
        assert docs[0].metadata.get("vault_rel_path") == "Note.md"
        assert "Body text" in docs[0].page_content

    def test_load_all_documents_merges_json_and_vault(self, tmp_path: Path) -> None:
        jdir = tmp_path / "jin"
        jdir.mkdir()
        (jdir / "k.json").write_text(
            json.dumps([{"title": "J", "content": "C"}]),
            encoding="utf-8",
        )
        vault = tmp_path / "vault"
        vault.mkdir()
        (vault / "N.md").write_text("## Only\nText.", encoding="utf-8")

        prev_json, prev_vault = rag.JSON_DIR, rag.VAULT_DIR
        try:
            rag.JSON_DIR = jdir
            rag.VAULT_DIR = vault
            merged = rag.load_all_documents_for_reindex()
            assert len(merged) == 2
            fmts = {d.metadata.get("format") for d in merged}
            assert "article" in fmts and "obsidian" in fmts
        finally:
            rag.JSON_DIR = prev_json
            rag.VAULT_DIR = prev_vault
