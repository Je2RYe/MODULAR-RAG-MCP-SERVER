"""Unit tests for EmbeddingFactory provider routing."""

from __future__ import annotations

import pytest

from core.settings import Settings
from libs.embedding.base_embedding import BaseEmbedding
from libs.embedding.embedding_factory import EmbeddingFactory


class _FakeEmbedding(BaseEmbedding):
    def __init__(self, settings: Settings) -> None:
        self._provider = str(settings.embedding.get("provider", "unknown"))

    def embed(self, texts: list[str], trace: object | None = None) -> list[list[float]]:
        return [[float(len(text)), float(len(self._provider))] for text in texts]


def _make_settings(provider: str) -> Settings:
    raw = {
        "llm": {"provider": "azure"},
        "embedding": {"provider": provider},
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


def test_create_with_registered_provider_returns_expected_embedding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(EmbeddingFactory, "_registry", {"fake": _FakeEmbedding})
    settings = _make_settings("fake")

    embedding = EmbeddingFactory.create(settings)

    assert isinstance(embedding, _FakeEmbedding)
    assert embedding.embed(["hi", "hello"]) == [[2.0, 4.0], [5.0, 4.0]]


def test_create_with_missing_provider_raises_value_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(EmbeddingFactory, "_registry", {"fake": _FakeEmbedding})
    settings = _make_settings("")

    with pytest.raises(ValueError, match=r"embedding\.provider"):
        EmbeddingFactory.create(settings)


def test_create_with_unknown_provider_raises_value_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(EmbeddingFactory, "_registry", {"fake": _FakeEmbedding})
    settings = _make_settings("unknown")

    with pytest.raises(ValueError, match=r"Unsupported embedding\.provider"):
        EmbeddingFactory.create(settings)
