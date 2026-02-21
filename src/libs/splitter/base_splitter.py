"""Base abstraction for text splitter strategies."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from core.trace.trace_context import TraceContext


class BaseSplitter(ABC):
    """Abstract interface for splitter implementations."""

    @abstractmethod
    def split_text(self, text: str, trace: TraceContext | None = None) -> list[str]:
        """Split text into semantically useful chunks.

        Args:
            text: Source text to split.
            trace: Optional trace context object.

        Returns:
            List of text chunks in source order.
        """
