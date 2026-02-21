"""Base abstraction for embedding providers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class BaseEmbedding(ABC):
    """Abstract interface for embedding providers."""

    @abstractmethod
    def embed(self, texts: list[str], trace: Any | None = None) -> list[list[float]]:
        """Embed a batch of texts into dense vectors.

        Args:
            texts: Batch of input strings.
            trace: Optional trace context object.

        Returns:
            Dense vectors aligned with `texts` order.
        """
