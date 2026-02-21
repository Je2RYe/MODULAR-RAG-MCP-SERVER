"""Factory for creating splitter strategy instances from settings."""

from __future__ import annotations

from collections.abc import Callable

from core.settings import Settings
from libs.splitter.base_splitter import BaseSplitter


SplitterCreator = Callable[[Settings], BaseSplitter]


class SplitterFactory:
    """Factory that resolves splitter implementations by configured type."""

    _registry: dict[str, SplitterCreator] = {}

    @classmethod
    def register(cls, splitter_type: str, creator: SplitterCreator) -> None:
        """Register a splitter constructor.

        Args:
            splitter_type: Splitter type key (e.g. "recursive", "semantic").
            creator: Callable that builds a splitter instance.
        """

        normalized = splitter_type.strip().lower()
        if not normalized:
            raise ValueError("Splitter type cannot be empty")
        cls._registry[normalized] = creator

    @classmethod
    def create(cls, settings: Settings) -> BaseSplitter:
        """Create a splitter instance based on settings.

        Args:
            settings: Global application settings.

        Returns:
            A configured splitter implementation.

        Raises:
            ValueError: If splitter type is missing or not registered.
        """

        splitter_config = settings.raw.get("splitter", {})
        splitter_type_raw = splitter_config.get("type")
        if not isinstance(splitter_type_raw, str) or not splitter_type_raw.strip():
            raise ValueError("Missing required splitter type: splitter.type")

        splitter_type = splitter_type_raw.strip().lower()
        creator = cls._registry.get(splitter_type)
        if creator is None:
            available = ", ".join(sorted(cls._registry)) or "<none>"
            raise ValueError(
                "Unsupported splitter.type: "
                f"{splitter_type}. Registered splitters: {available}"
            )

        return creator(settings)
