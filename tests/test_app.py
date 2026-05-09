"""
Unit tests for app.py: config helpers, vault parsing, wikilink graph, serialization.

Heavy paths (LLM inference, full Chroma queries) are not executed here.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import app as rag


class TestCleanMetadata:
    def test_drops_none(self) -> None:
        assert rag._clean_metadata({"a": 1, "b": None}) == {"a": 1}

    def test_coerces_non_scalar_to_str(self) -> None:
        assert rag._clean_metadata({"x": {"nested": 1}}) == {"x": "{'nested': 1}"}


class TestSystemPrompt:
    def test_has_grounding_rules(self) -> None:
        low = rag.SYSTEM_PROMPT.lower()
        assert "vault" in low or "excerpt" in low
        assert "could not find this in the documents" in low


class TestWikilinkExpand:
    def test_neighbor_chunks_from_graph(self, tmp_path: Path) -> None:
        from langchain_core.documents import Document

        v = tmp_path / "vault"
        v.mkdir()
        (v / "Alpha.md").write_text("## One\nSee [[Bravo]] for more.\n", encoding="utf-8")
        (v / "Bravo.md").write_text("## Two\nBody B.\n", encoding="utf-8")
        seed = [Document(page_content="x", metadata={"vault_rel_path": "Alpha.md"})]
        out = rag.expand_docs_via_wikilink_neighbors(
            v,
            seed,
            max_neighbor_notes=3,
            max_neighbor_chunks=6,
            exclude_dir_names=frozenset(),
        )
        paths = {(d.metadata or {}).get("vault_rel_path") for d in out}
        assert "Bravo.md" in paths


class TestChatHistoryHelpers:
    def test_linear_roles_to_pairs(self) -> None:
        linear = [
            ("user", "Hi"),
            ("assistant", "Hello"),
            ("user", "VPN?"),
        ]
        assert rag.linear_roles_to_pairs(linear) == [("Hi", "Hello")]

    def test_build_retrieval_query_from_history(self) -> None:
        prior = [
            ("user", "What is VPN?"),
            ("assistant", "Corporate VPN is described in the IT note."),
        ]
        q = rag.build_retrieval_query_from_history(
            "Tell me more about that.",
            prior,
            retrieval_pairs=2,
        )
        assert "Follow-up question:" in q
        assert "Tell me more" in q
        assert "VPN" in q

    def test_format_pairs_for_prompt_truncates(self) -> None:
        pairs = [("u" * 5000, "a" * 5000)]
        out = rag.format_pairs_for_prompt(pairs, max_chars=100)
        assert len(out) == 100


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
                metadata={
                    "source_file": "note.md",
                    "vault_rel_path": "note.md",
                    "title": "T",
                },
            )
        ]
        ser = rag.serialize_sources_for_ui(docs)
        assert len(ser) == 1
        assert ser[0]["source_file"] == "note.md"
        assert ser[0]["graph_id"] == "note.md"

    def test_serialize_sources_path_from_chunk_when_meta_missing(self) -> None:
        from langchain_core.documents import Document

        body = "Note: Hi\nPath: Sub/TestNote.md\nSection: x\n\nfoo"
        docs = [Document(page_content=body, metadata={})]
        ser = rag.serialize_sources_for_ui(docs)
        assert ser[0]["graph_id"] == "Sub/TestNote.md"
        assert ser[0]["source_file"] == "Sub/TestNote.md"


class TestVaultMarkdownPaths:
    def test_list_vault_markdown_relpaths_respects_exclude(self, tmp_path: Path) -> None:
        vault = tmp_path / "vault"
        vault.mkdir()
        (vault / "keep.md").write_text("# K\n\n## A\nx", encoding="utf-8")
        skip_dir = vault / ".obsidian"
        skip_dir.mkdir()
        (skip_dir / "hidden.md").write_text("# H", encoding="utf-8")

        prev_vault = rag.VAULT_DIR
        prev_excl = rag.VAULT_EXCLUDE_DIR_NAMES
        try:
            rag.VAULT_DIR = vault
            rag.VAULT_EXCLUDE_DIR_NAMES = frozenset({".obsidian"})
            assert rag.list_vault_markdown_relpaths() == ["keep.md"]
        finally:
            rag.VAULT_DIR = prev_vault
            rag.VAULT_EXCLUDE_DIR_NAMES = prev_excl


class TestConfigHelpers:
    def test_llm_generation_from_config(self, tmp_path: Path) -> None:
        prev = (
            rag.LLM_MAX_NEW_TOKENS,
            rag.LLM_DO_SAMPLE,
            rag.LLM_TEMPERATURE,
            rag.LLM_TOP_P,
        )
        try:
            rag.apply_app_config(
                {
                    "llm": {
                        "max_new_tokens": 128,
                        "do_sample": False,
                        "temperature": 0.7,
                        "top_p": 0.95,
                    }
                }
            )
            assert rag.LLM_MAX_NEW_TOKENS == 128
            assert rag.LLM_DO_SAMPLE is False
            assert rag.LLM_TEMPERATURE == 0.7
            assert rag.LLM_TOP_P == 0.95
        finally:
            (
                rag.LLM_MAX_NEW_TOKENS,
                rag.LLM_DO_SAMPLE,
                rag.LLM_TEMPERATURE,
                rag.LLM_TOP_P,
            ) = prev

    def test_load_missing_config_is_empty(self, tmp_path: Path) -> None:
        assert rag.load_app_config_file(tmp_path / "does-not-exist.yaml") == {}

    def test_resolve_config_explicit(self) -> None:
        p = Path("/tmp/x.yaml")
        assert rag.resolve_config_path(p) == p

    def test_vault_dir_from_config(self) -> None:
        tmp = tempfile.mkdtemp(prefix="pytest_vault_", dir=str(rag.BASE_DIR))
        rel = Path(tmp).name
        prev_vault = rag.VAULT_DIR
        try:
            rag.apply_app_config({"paths": {"vault_dir": rel}})
            assert rag.VAULT_DIR == Path(tmp).resolve()
        finally:
            rag.VAULT_DIR = prev_vault
            Path(tmp).rmdir()


class TestVaultGraphApiPayload:
    def test_payload_when_vault_not_configured(self) -> None:
        prev = rag.VAULT_DIR
        try:
            rag.VAULT_DIR = None
            p = rag.vault_graph_api_payload()
            assert p["vault_configured"] is False
            assert "message" in p
        finally:
            rag.VAULT_DIR = prev


class TestVaultLinkGraph:
    def test_extract_wikilink_targets(self) -> None:
        raw = "X [[A]] [[B|alias]] ![[C]] [[D#head]]"
        assert rag.extract_wikilink_targets(raw) == ["A", "B", "C", "D"]

    def test_build_vault_link_graph(self, tmp_path: Path) -> None:
        vault = tmp_path / "vault"
        vault.mkdir()
        (vault / "Alpha.md").write_text(
            "# Alpha\n\nLinks to [[Beta]] and [[Sub/Gamma]].\n",
            encoding="utf-8",
        )
        (vault / "Beta.md").write_text("# Beta\nStandalone.\n", encoding="utf-8")
        sub = vault / "Sub"
        sub.mkdir()
        (sub / "Gamma.md").write_text("# Gamma\n", encoding="utf-8")

        g = rag.build_vault_link_graph(vault, exclude_dir_names=frozenset())
        ids = {n["id"] for n in g["nodes"]}
        assert ids == {"Alpha.md", "Beta.md", "Sub/Gamma.md"}
        pairs = {(e["from"], e["to"]) for e in g["edges"]}
        assert ("Alpha.md", "Beta.md") in pairs
        assert ("Alpha.md", "Sub/Gamma.md") in pairs


class TestObsidianHelpers:
    def test_normalize_obsidian_markup(self) -> None:
        assert "Alias" in rag._normalize_obsidian_markup("See [[Note|Alias]] here.")
        assert "Note" in rag._normalize_obsidian_markup("Link [[Note]] done.")
        assert "![" not in rag._normalize_obsidian_markup("Embed ![[Other]] tail.")

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

    def test_load_obsidian_respects_config_exclude_dir(self, tmp_path: Path) -> None:
        vault = tmp_path / "v"
        vault.mkdir()
        skip_dir = vault / "SkipMe"
        skip_dir.mkdir()
        (skip_dir / "hidden.md").write_text("## X\nY", encoding="utf-8")
        (vault / "keep.md").write_text("## K\nVisible.", encoding="utf-8")
        prev_excl = rag.VAULT_EXCLUDE_DIR_NAMES
        try:
            rag.apply_app_config({"vault": {"exclude_dir_names": ["SkipMe"]}})
            docs = rag.load_obsidian_documents(vault)
            assert len(docs) == 1
            assert docs[0].metadata.get("vault_rel_path") == "keep.md"
        finally:
            rag.VAULT_EXCLUDE_DIR_NAMES = prev_excl

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

    def test_load_all_documents_only_vault(self, tmp_path: Path) -> None:
        vault = tmp_path / "vault"
        vault.mkdir()
        (vault / "N.md").write_text("## Only\nText.", encoding="utf-8")

        prev_vault = rag.VAULT_DIR
        try:
            rag.VAULT_DIR = vault
            merged = rag.load_all_documents_for_reindex()
            assert len(merged) == 1
            assert merged[0].metadata.get("format") == "obsidian"
        finally:
            rag.VAULT_DIR = prev_vault


class TestReindexRequiresVault:
    def test_no_vault_returns_empty(self) -> None:
        prev = rag.VAULT_DIR
        try:
            rag.VAULT_DIR = None
            assert rag.load_all_documents_for_reindex() == []
        finally:
            rag.VAULT_DIR = prev
