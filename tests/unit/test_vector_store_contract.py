"""Contract tests for BaseVectorStore shape and VectorStoreFactory routing."""

from __future__ import annotations

from typing import Any

import pytest

from core.settings import Settings
from libs.vector_store.base_vector_store import BaseVectorStore
from libs.vector_store.vector_store_factory import VectorStoreFactory


class _FakeVectorStore(BaseVectorStore):
    def __init__(self, settings: Settings) -> None:
        self._backend = str(settings.vector_store.get("backend", "unknown"))
        self._records: list[dict[str, Any]] = []

    def upsert(self, records: list[dict[str, Any]], trace: object | None = None) -> None:
        for record in records:
            record_id = record["id"]
            self._records = [r for r in self._records if r["id"] != record_id]
            self._records.append(record)

    def query(
        self,
        vector: list[float],
        top_k: int,
        filters: dict[str, Any] | None = None,
        trace: object | None = None,
    ) -> list[dict[str, Any]]:
        del vector
        matched = self._records
        if filters:
            matched = [
                r
                for r in matched
                if all(r.get("metadata", {}).get(key) == value for key, value in filters.items())
            ]
        return [
            {
                "id": record["id"],
                "score": 1.0,
                "metadata": record.get("metadata", {}),
                "backend": self._backend,
            }
            for record in matched[:top_k]
        ]


def _make_settings(backend: str) -> Settings:
    raw = {
        "llm": {"provider": "azure"},
        "embedding": {"provider": "azure"},
        "vector_store": {"backend": backend},
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


def test_contract_upsert_and_query_shape(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(VectorStoreFactory, "_registry", {"fake": _FakeVectorStore})
    store = VectorStoreFactory.create(_make_settings("fake"))

    store.upsert(
        [
            {"id": "chunk-1", "vector": [0.1, 0.2], "metadata": {"source": "a"}},
            {"id": "chunk-2", "vector": [0.2, 0.3], "metadata": {"source": "b"}},
        ]
    )
    results = store.query(vector=[0.2, 0.3], top_k=1, filters={"source": "a"})

    assert isinstance(results, list)
    assert len(results) == 1
    assert set(results[0].keys()) >= {"id", "score", "metadata"}
    assert isinstance(results[0]["id"], str)
    assert isinstance(results[0]["score"], float)
    assert isinstance(results[0]["metadata"], dict)


def test_factory_with_missing_backend_raises_value_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(VectorStoreFactory, "_registry", {"fake": _FakeVectorStore})

    with pytest.raises(ValueError, match=r"vector_store\.backend"):
        VectorStoreFactory.create(_make_settings(""))


def test_factory_with_unknown_backend_raises_value_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(VectorStoreFactory, "_registry", {"fake": _FakeVectorStore})

    with pytest.raises(ValueError, match=r"Unsupported vector_store\.backend"):
        VectorStoreFactory.create(_make_settings("unknown"))
