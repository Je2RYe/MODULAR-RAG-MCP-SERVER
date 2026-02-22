"""Base abstraction for evaluation backends."""

from __future__ import annotations

from abc import ABC, abstractmethod


class BaseEvaluator(ABC):
    """Abstract interface for retrieval quality evaluators."""

    @abstractmethod
    def evaluate(
        self,
        query: str,
        retrieved_ids: list[str],
        golden_ids: list[str],
    ) -> dict[str, float]:
        """Evaluate retrieval quality for one query.

        Args:
            query: User query string.
            retrieved_ids: Ranked retrieval result identifiers.
            golden_ids: Ground-truth relevant identifiers.

        Returns:
            A metrics dictionary, e.g. ``{"hit_rate": 1.0, "mrr": 0.5}``.
        """
