"""Base abstraction for vector store backends."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from core.trace.trace_context import TraceContext


class BaseVectorStore(ABC):
    """Abstract interface for vector store implementations."""

    @abstractmethod
    def upsert(
        self,
        records: list[dict[str, Any]],
        trace: TraceContext | None = None,
    ) -> None:
        """Insert or update records in the vector store.

        Args:
            records: Vector records containing ids, vectors, and metadata.
            trace: Optional trace context object.
        """

    @abstractmethod
    def query(
        self,
        vector: list[float],
        top_k: int,
        filters: dict[str, Any] | None = None,
        trace: TraceContext | None = None,
    ) -> list[dict[str, Any]]:
        """Query top-k records by a query vector.

        Args:
            vector: Query embedding vector.
            top_k: Maximum number of results.
            filters: Optional metadata filters.
            trace: Optional trace context object.

        Returns:
            Ranked result records.
        """
