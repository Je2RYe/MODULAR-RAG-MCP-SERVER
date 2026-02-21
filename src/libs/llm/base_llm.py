"""Base abstraction for chat-capable LLM providers."""

from __future__ import annotations

from abc import ABC, abstractmethod


class BaseLLM(ABC):
    """Abstract interface for all LLM providers.

    Implementations should adapt provider-specific request/response formats
    behind a unified `chat` API.
    """

    @abstractmethod
    def chat(self, messages: list[dict[str, str]]) -> str:
        """Generate a chat completion from message history.

        Args:
            messages: Chat message list, e.g. `[{"role": "user", "content": "..."}]`.

        Returns:
            Model response text.
        """
