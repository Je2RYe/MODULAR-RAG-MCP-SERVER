"""Factory for creating LLM provider instances from settings."""

from __future__ import annotations

from collections.abc import Callable

from core.settings import Settings
from libs.llm.base_llm import BaseLLM


LLMCreator = Callable[[Settings], BaseLLM]


class LLMFactory:
    """Factory that resolves LLM implementations by provider name."""

    _registry: dict[str, LLMCreator] = {}

    @classmethod
    def register(cls, provider: str, creator: LLMCreator) -> None:
        """Register a provider constructor.

        Args:
            provider: Provider key (e.g. "azure", "openai", "ollama").
            creator: Callable that builds an LLM instance.
        """

        normalized = provider.strip().lower()
        if not normalized:
            raise ValueError("Provider name cannot be empty")
        cls._registry[normalized] = creator

    @classmethod
    def create(cls, settings: Settings) -> BaseLLM:
        """Create an LLM instance based on settings.

        Args:
            settings: Global application settings.

        Returns:
            A configured LLM implementation.

        Raises:
            ValueError: If provider is missing or not registered.
        """

        provider_raw = settings.llm.get("provider")
        if not isinstance(provider_raw, str) or not provider_raw.strip():
            raise ValueError("Missing required LLM provider: llm.provider")

        provider = provider_raw.strip().lower()
        creator = cls._registry.get(provider)
        if creator is None:
            available = ", ".join(sorted(cls._registry)) or "<none>"
            raise ValueError(
                f"Unsupported llm.provider: {provider}. Registered providers: {available}"
            )

        return creator(settings)
