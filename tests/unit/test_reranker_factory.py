"""Unit tests for RerankerFactory backend routing."""

from __future__ import annotations

from typing import Any

import pytest

from core.settings import Settings
from libs.reranker.base_reranker import BaseReranker
from libs.reranker.reranker_factory import RerankerFactory


class _FakeReranker(BaseReranker):
    def rerank(
        self,
        query: str,
        candidates: list[dict[str, Any]],
        top_k: int | None = None,
    ) -> list[dict[str, Any]]:
        del query
        reversed_candidates = list(reversed(candidates))
        if top_k is None:
            return reversed_candidates
        return reversed_candidates[:top_k]


def _make_settings(backend: str) -> Settings:
    raw = {
        "llm": {"provider": "azure"},
        "embedding": {"provider": "azure"},
        "vector_store": {"backend": "chroma"},
        "retrieval": {"top_k": 5},
        "rerank": {"backend": backend},
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


def test_create_with_none_backend_returns_passthrough_reranker() -> None:
    settings = _make_settings("none")

    reranker = RerankerFactory.create(settings)
    candidates = [{"id": "a"}, {"id": "b"}, {"id": "c"}]

    assert reranker.rerank("query", candidates) == candidates
    assert reranker.rerank("query", candidates, top_k=2) == candidates[:2]


def test_create_with_registered_backend_returns_expected_reranker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_registry = dict(RerankerFactory._registry)
    monkeypatch.setattr(
        RerankerFactory,
        "_registry",
        {"none": original_registry["none"], "fake": lambda _: _FakeReranker()},
    )
    settings = _make_settings("fake")

    reranker = RerankerFactory.create(settings)
    candidates = [{"id": "a"}, {"id": "b"}, {"id": "c"}]

    assert isinstance(reranker, _FakeReranker)
    assert reranker.rerank("q", candidates, top_k=2) == [{"id": "c"}, {"id": "b"}]


def test_create_with_missing_backend_raises_value_error() -> None:
    settings = _make_settings("")

    with pytest.raises(ValueError, match=r"rerank\.backend"):
        RerankerFactory.create(settings)


def test_create_with_unknown_backend_raises_value_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(RerankerFactory, "_registry", {"none": lambda _: _FakeReranker()})
    settings = _make_settings("unknown")

    with pytest.raises(ValueError, match=r"Unsupported rerank\.backend"):
        RerankerFactory.create(settings)
