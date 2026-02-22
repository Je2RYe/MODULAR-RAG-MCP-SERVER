"""Lightweight retrieval evaluator with basic ranking metrics."""

from __future__ import annotations

from libs.evaluator.base_evaluator import BaseEvaluator


class CustomEvaluator(BaseEvaluator):
    """Compute lightweight retrieval metrics for a single query."""

    def evaluate(
        self,
        query: str,
        retrieved_ids: list[str],
        golden_ids: list[str],
    ) -> dict[str, float]:
        """Compute hit rate and MRR from ranked retrieval IDs.

        Args:
            query: User query string.
            retrieved_ids: Ranked retrieval result identifiers.
            golden_ids: Ground-truth relevant identifiers.

        Returns:
            Metrics dictionary with ``hit_rate`` and ``mrr``.
        """

        del query
        if not golden_ids:
            return {"hit_rate": 0.0, "mrr": 0.0}

        golden_set = set(golden_ids)
        first_relevant_rank: int | None = None

        for index, candidate_id in enumerate(retrieved_ids, start=1):
            if candidate_id in golden_set:
                first_relevant_rank = index
                break

        if first_relevant_rank is None:
            return {"hit_rate": 0.0, "mrr": 0.0}

        return {"hit_rate": 1.0, "mrr": 1.0 / float(first_relevant_rank)}
