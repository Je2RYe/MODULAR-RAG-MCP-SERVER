"""Factory for creating embedding provider instances from settings."""

from __future__ import annotations

from collections.abc import Callable

from core.settings import Settings
from libs.embedding.base_embedding import BaseEmbedding


EmbeddingCreator = Callable[[Settings], BaseEmbedding]


class EmbeddingFactory:
    """Factory that resolves embedding implementations by provider name."""

    _registry: dict[str, EmbeddingCreator] = {}

    @classmethod
    def register(cls, provider: str, creator: EmbeddingCreator) -> None:
        """Register an embedding provider constructor.

        Args:
            provider: Provider key (e.g. "openai", "azure", "ollama").
            creator: Callable that builds an embedding instance.
        """

        normalized = provider.strip().lower()
        if not normalized:
            raise ValueError("Provider name cannot be empty")
        cls._registry[normalized] = creator

    @classmethod
    def create(cls, settings: Settings) -> BaseEmbedding:
        """Create an embedding instance based on settings.

        Args:
            settings: Global application settings.

        Returns:
            A configured embedding implementation.

        Raises:
            ValueError: If provider is missing or not registered.
        """

        provider_raw = settings.embedding.get("provider")
        if not isinstance(provider_raw, str) or not provider_raw.strip():
            raise ValueError("Missing required embedding provider: embedding.provider")

        provider = provider_raw.strip().lower()
        creator = cls._registry.get(provider)
        if creator is None:
            available = ", ".join(sorted(cls._registry)) or "<none>"
            raise ValueError(
                "Unsupported embedding.provider: "
                f"{provider}. Registered providers: {available}"
            )

        return creator(settings)
