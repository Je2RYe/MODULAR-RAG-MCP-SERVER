"""Unit tests for CustomEvaluator and EvaluatorFactory."""

from __future__ import annotations

import pytest

from core.settings import Settings
from libs.evaluator.custom_evaluator import CustomEvaluator
from libs.evaluator.evaluator_factory import EvaluatorFactory


def _make_settings(backends: list[str]) -> Settings:
    raw = {
        "llm": {"provider": "azure"},
        "embedding": {"provider": "azure"},
        "vector_store": {"backend": "chroma"},
        "retrieval": {"top_k": 5},
        "rerank": {"backend": "none"},
        "evaluation": {"backends": backends},
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


def test_custom_evaluator_returns_expected_hit_rate_and_mrr() -> None:
    evaluator = CustomEvaluator()

    metrics = evaluator.evaluate(
        query="what is rag",
        retrieved_ids=["d1", "d2", "d3"],
        golden_ids=["d2"],
    )

    assert metrics["hit_rate"] == 1.0
    assert metrics["mrr"] == 0.5


def test_custom_evaluator_returns_zero_metrics_when_no_hit() -> None:
    evaluator = CustomEvaluator()

    metrics = evaluator.evaluate(
        query="what is rag",
        retrieved_ids=["d1", "d2", "d3"],
        golden_ids=["x1", "x2"],
    )

    assert metrics == {"hit_rate": 0.0, "mrr": 0.0}


def test_factory_create_uses_settings_backend() -> None:
    evaluator = EvaluatorFactory.create(_make_settings(["custom_metrics"]))

    assert isinstance(evaluator, CustomEvaluator)


def test_factory_create_with_missing_backends_raises_value_error() -> None:
    with pytest.raises(ValueError, match=r"evaluation\.backends"):
        EvaluatorFactory.create(_make_settings([]))


def test_factory_create_with_unknown_backend_raises_value_error() -> None:
    with pytest.raises(ValueError, match=r"Unsupported evaluator backend"):
        EvaluatorFactory.create(_make_settings(["unknown"]))
