"""Unit tests for SplitterFactory provider routing."""

from __future__ import annotations

import pytest

from core.settings import Settings
from libs.splitter.base_splitter import BaseSplitter
from libs.splitter.splitter_factory import SplitterFactory


class _RecursiveSplitter(BaseSplitter):
    def split_text(self, text: str, trace: object | None = None) -> list[str]:
        return [f"recursive:{text}"]


class _SemanticSplitter(BaseSplitter):
    def split_text(self, text: str, trace: object | None = None) -> list[str]:
        return [f"semantic:{text}"]


class _FixedSplitter(BaseSplitter):
    def split_text(self, text: str, trace: object | None = None) -> list[str]:
        return [f"fixed:{text}"]


def _make_settings(splitter_type: str) -> Settings:
    raw = {
        "llm": {"provider": "azure"},
        "embedding": {"provider": "azure"},
        "splitter": {"type": splitter_type},
        "vector_store": {"backend": "chroma"},
        "retrieval": {"top_k": 5},
        "rerank": {"backend": "none"},
        "evaluation": {"backends": ["custom_metrics"]},
        "observability": {"enabled": True},
    }
    return Settings(
        llm=raw["llm"],
        embedding=raw["embedding"],
        vector_store=raw["vector_store"],
        retrieval=raw["retrieval"],
        rerank=raw["rerank"],
        evaluation=raw["evaluation"],
        observability=raw["observability"],
        raw=raw,
    )


def test_create_routes_to_registered_splitter_types(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        SplitterFactory,
        "_registry",
        {
            "recursive": lambda settings: _RecursiveSplitter(),
            "semantic": lambda settings: _SemanticSplitter(),
            "fixed": lambda settings: _FixedSplitter(),
        },
    )

    recursive = SplitterFactory.create(_make_settings("recursive"))
    semantic = SplitterFactory.create(_make_settings("semantic"))
    fixed = SplitterFactory.create(_make_settings("fixed"))

    assert recursive.split_text("x") == ["recursive:x"]
    assert semantic.split_text("x") == ["semantic:x"]
    assert fixed.split_text("x") == ["fixed:x"]


def test_create_with_missing_splitter_type_raises_value_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(SplitterFactory, "_registry", {"recursive": lambda settings: _RecursiveSplitter()})
    settings = _make_settings("")

    with pytest.raises(ValueError, match=r"splitter\.type"):
        SplitterFactory.create(settings)


def test_create_with_unknown_splitter_type_raises_value_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(SplitterFactory, "_registry", {"recursive": lambda settings: _RecursiveSplitter()})
    settings = _make_settings("unknown")

    with pytest.raises(ValueError, match=r"Unsupported splitter\.type"):
        SplitterFactory.create(settings)
