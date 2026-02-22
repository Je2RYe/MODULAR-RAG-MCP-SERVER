"""Factory for creating evaluator instances from settings."""

from __future__ import annotations

from collections.abc import Callable

from core.settings import Settings
from libs.evaluator.base_evaluator import BaseEvaluator
from libs.evaluator.custom_evaluator import CustomEvaluator


EvaluatorCreator = Callable[[Settings], BaseEvaluator]


class EvaluatorFactory:
    """Factory that resolves evaluator implementations by backend name."""

    _registry: dict[str, EvaluatorCreator] = {
        "custom": lambda _: CustomEvaluator(),
        "custom_metrics": lambda _: CustomEvaluator(),
    }

    @classmethod
    def register(cls, backend: str, creator: EvaluatorCreator) -> None:
        """Register an evaluator constructor.

        Args:
            backend: Backend key (e.g. "custom_metrics", "ragas").
            creator: Callable that builds an evaluator instance.
        """

        normalized = backend.strip().lower()
        if not normalized:
            raise ValueError("Evaluator backend cannot be empty")
        cls._registry[normalized] = creator

    @classmethod
    def create(cls, settings: Settings, backend: str | None = None) -> BaseEvaluator:
        """Create an evaluator instance from settings.

        Args:
            settings: Global application settings.
            backend: Optional explicit backend override.

        Returns:
            Configured evaluator implementation.

        Raises:
            ValueError: If backend configuration is missing or unsupported.
        """

        if backend is None:
            backends_raw = settings.evaluation.get("backends")
            if not isinstance(backends_raw, list) or not backends_raw:
                raise ValueError(
                    "Missing required evaluator backends: evaluation.backends"
                )
            backend_raw = backends_raw[0]
        else:
            backend_raw = backend

        if not isinstance(backend_raw, str) or not backend_raw.strip():
            raise ValueError("Missing required evaluator backend")

        normalized = backend_raw.strip().lower()
        creator = cls._registry.get(normalized)
        if creator is None:
            available = ", ".join(sorted(cls._registry)) or "<none>"
            raise ValueError(
                f"Unsupported evaluator backend: {normalized}. "
                f"Registered backends: {available}"
            )

        return creator(settings)
