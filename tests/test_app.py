"""
Unit tests for app.py (JSON → Document pipeline and helpers).
LLM / Chroma integration is not run here (no GPU, no large downloads).
"""

from __future__ import annotations

import json
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
