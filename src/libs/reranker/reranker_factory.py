"""Factory for creating reranker instances from settings."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from core.settings import Settings
from libs.reranker.base_reranker import BaseReranker


RerankerCreator = Callable[[Settings], BaseReranker]


class _NoneReranker(BaseReranker):
    """No-op reranker used when rerank backend is disabled."""

    def rerank(
        self,
        query: str,
        candidates: list[dict[str, Any]],
        top_k: int | None = None,
    ) -> list[dict[str, Any]]:
        del query
        if top_k is None:
            return list(candidates)
        return list(candidates[:top_k])


class RerankerFactory:
    """Factory that resolves reranker implementations by backend name."""

    _registry: dict[str, RerankerCreator] = {"none": lambda _: _NoneReranker()}

    @classmethod
    def register(cls, backend: str, creator: RerankerCreator) -> None:
        """Register a reranker constructor.

        Args:
            backend: Backend key (e.g. "cross_encoder", "llm", "none").
            creator: Callable that builds a reranker instance.
        """

        normalized = backend.strip().lower()
        if not normalized:
            raise ValueError("Reranker backend cannot be empty")
        cls._registry[normalized] = creator

    @classmethod
    def create(cls, settings: Settings) -> BaseReranker:
        """Create a reranker instance from settings.

        Args:
            settings: Global application settings.

        Returns:
            Configured reranker implementation.

        Raises:
            ValueError: If backend is missing or not registered.
        """

        backend_raw = settings.rerank.get("backend")
        if not isinstance(backend_raw, str) or not backend_raw.strip():
            raise ValueError("Missing required rerank backend: rerank.backend")

        backend = backend_raw.strip().lower()
        creator = cls._registry.get(backend)
        if creator is None:
            available = ", ".join(sorted(cls._registry)) or "<none>"
            raise ValueError(
                f"Unsupported rerank.backend: {backend}. Registered backends: {available}"
            )

        return creator(settings)
