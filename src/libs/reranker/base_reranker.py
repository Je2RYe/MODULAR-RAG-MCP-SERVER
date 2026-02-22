"""Base abstraction for reranker providers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class BaseReranker(ABC):
    """Abstract interface for reranking candidate retrieval results.

    Implementations should score and reorder candidates according to the
    query relevance while preserving the candidate data contract.
    """

    @abstractmethod
    def rerank(
        self,
        query: str,
        candidates: list[dict[str, Any]],
        top_k: int | None = None,
    ) -> list[dict[str, Any]]:
        """Rerank candidate chunks/documents for a user query.

        Args:
            query: User query string.
            candidates: Candidate retrieval results to reorder.
            top_k: Optional final result size after reranking.

        Returns:
            Reranked candidates list.
        """
