"""Factory for creating vector store instances from settings."""

from __future__ import annotations

from collections.abc import Callable

from core.settings import Settings
from libs.vector_store.base_vector_store import BaseVectorStore


VectorStoreCreator = Callable[[Settings], BaseVectorStore]


class VectorStoreFactory:
    """Factory that resolves vector store implementations by backend name."""

    _registry: dict[str, VectorStoreCreator] = {}

    @classmethod
    def register(cls, backend: str, creator: VectorStoreCreator) -> None:
        """Register a vector store constructor.

        Args:
            backend: Backend key (e.g. "chroma", "qdrant").
            creator: Callable that builds a vector store instance.
        """

        normalized = backend.strip().lower()
        if not normalized:
            raise ValueError("Vector store backend cannot be empty")
        cls._registry[normalized] = creator

    @classmethod
    def create(cls, settings: Settings) -> BaseVectorStore:
        """Create a vector store instance from settings.

        Args:
            settings: Global application settings.

        Returns:
            Configured vector store implementation.

        Raises:
            ValueError: If backend is missing or not registered.
        """

        backend_raw = settings.vector_store.get("backend")
        if not isinstance(backend_raw, str) or not backend_raw.strip():
            raise ValueError("Missing required vector store backend: vector_store.backend")

        backend = backend_raw.strip().lower()
        creator = cls._registry.get(backend)
        if creator is None:
            available = ", ".join(sorted(cls._registry)) or "<none>"
            raise ValueError(
                "Unsupported vector_store.backend: "
                f"{backend}. Registered backends: {available}"
            )

        return creator(settings)
