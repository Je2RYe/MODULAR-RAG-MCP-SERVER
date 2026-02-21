"""Settings loading and validation utilities."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(slots=True)
class Settings:
    """Application settings structure.

    Attributes:
        llm: LLM configuration dictionary.
        embedding: Embedding configuration dictionary.
        vector_store: Vector store configuration dictionary.
        retrieval: Retrieval configuration dictionary.
        rerank: Rerank configuration dictionary.
        evaluation: Evaluation configuration dictionary.
        observability: Observability configuration dictionary.
        raw: Original full settings dictionary.
    """

    llm: dict[str, Any]
    embedding: dict[str, Any]
    vector_store: dict[str, Any]
    retrieval: dict[str, Any]
    rerank: dict[str, Any]
    evaluation: dict[str, Any]
    observability: dict[str, Any]
    raw: dict[str, Any]


def _require_path(data: dict[str, Any], dotted_path: str) -> None:
    current: Any = data
    for key in dotted_path.split("."):
        if not isinstance(current, dict) or key not in current:
            raise ValueError(f"Missing required settings field: {dotted_path}")
        current = current[key]


def validate_settings(settings: Settings) -> None:
    """Validate required settings fields.

    Args:
        settings: Parsed settings object.

    Raises:
        ValueError: If any required field is missing.
    """

    required_paths = [
        "llm",
        "embedding",
        "embedding.provider",
        "vector_store",
        "retrieval",
        "rerank",
        "evaluation",
        "observability",
    ]

    for path in required_paths:
        _require_path(settings.raw, path)


def load_settings(path: str) -> Settings:
    """Load YAML settings from a file and validate required fields.

    Args:
        path: Path to the YAML settings file.

    Returns:
        Parsed and validated settings object.

    Raises:
        FileNotFoundError: If the path does not exist.
        ValueError: If YAML is invalid or required fields are missing.
    """

    settings_path = Path(path)
    if not settings_path.exists():
        raise FileNotFoundError(f"Settings file not found: {settings_path}")

    with settings_path.open("r", encoding="utf-8") as fp:
        parsed = yaml.safe_load(fp)

    if not isinstance(parsed, dict):
        raise ValueError("Settings file must contain a YAML mapping at top level")

    settings = Settings(
        llm=parsed.get("llm", {}),
        embedding=parsed.get("embedding", {}),
        vector_store=parsed.get("vector_store", {}),
        retrieval=parsed.get("retrieval", {}),
        rerank=parsed.get("rerank", {}),
        evaluation=parsed.get("evaluation", {}),
        observability=parsed.get("observability", {}),
        raw=parsed,
    )
    validate_settings(settings)
    return settings
