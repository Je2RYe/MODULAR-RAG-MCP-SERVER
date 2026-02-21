"""Unit tests for LLMFactory provider routing."""

from __future__ import annotations

import pytest

from core.settings import Settings
from libs.llm.base_llm import BaseLLM
from libs.llm.llm_factory import LLMFactory


class _FakeLLM(BaseLLM):
    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    def chat(self, messages: list[dict[str, str]]) -> str:
        return f"fake:{self._settings.llm.get('provider')}:{len(messages)}"


def _make_settings(provider: str) -> Settings:
    raw = {
        "llm": {"provider": provider},
        "embedding": {"provider": "azure"},
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


def test_create_with_registered_provider_returns_expected_llm(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(LLMFactory, "_registry", {"fake": _FakeLLM})
    settings = _make_settings("fake")

    llm = LLMFactory.create(settings)

    assert isinstance(llm, _FakeLLM)
    assert llm.chat([{"role": "user", "content": "hello"}]) == "fake:fake:1"


def test_create_with_missing_provider_raises_value_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(LLMFactory, "_registry", {"fake": _FakeLLM})
    settings = _make_settings("")

    with pytest.raises(ValueError, match=r"llm\.provider"):
        LLMFactory.create(settings)


def test_create_with_unknown_provider_raises_value_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(LLMFactory, "_registry", {"fake": _FakeLLM})
    settings = _make_settings("unknown")

    with pytest.raises(ValueError, match=r"Unsupported llm\.provider"):
        LLMFactory.create(settings)
