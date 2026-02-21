"""Tests for settings loading and validation."""

from __future__ import annotations

from pathlib import Path

import pytest

from core.settings import Settings, load_settings, validate_settings


_MINIMAL_VALID_SETTINGS = """
llm:
  provider: azure
embedding:
  provider: azure
vector_store:
  backend: chroma
retrieval:
  top_k: 5
rerank:
  backend: none
evaluation:
  backends: [custom_metrics]
observability:
  enabled: true
"""


def _write_yaml(path: Path, content: str) -> Path:
    path.write_text(content, encoding="utf-8")
    return path


def test_load_settings_valid_yaml_returns_settings(tmp_path: Path) -> None:
    config_path = _write_yaml(tmp_path / "settings.yaml", _MINIMAL_VALID_SETTINGS)

    settings = load_settings(str(config_path))

    assert isinstance(settings, Settings)
    assert settings.embedding["provider"] == "azure"


def test_load_settings_missing_required_field_raises_readable_error(tmp_path: Path) -> None:
    invalid_yaml = _MINIMAL_VALID_SETTINGS.replace(
        "embedding:\n  provider: azure\n",
        "embedding:\n",
        1,
    )
    config_path = _write_yaml(tmp_path / "settings.yaml", invalid_yaml)

    with pytest.raises(ValueError, match=r"embedding\.provider"):
        load_settings(str(config_path))


def test_validate_settings_missing_embedding_provider_raises() -> None:
    settings = Settings(
        llm={"provider": "azure"},
        embedding={},
        vector_store={"backend": "chroma"},
        retrieval={"top_k": 5},
        rerank={"backend": "none"},
        evaluation={"backends": ["custom_metrics"]},
        observability={"enabled": True},
        raw={
            "llm": {"provider": "azure"},
            "embedding": {},
            "vector_store": {"backend": "chroma"},
            "retrieval": {"top_k": 5},
            "rerank": {"backend": "none"},
            "evaluation": {"backends": ["custom_metrics"]},
            "observability": {"enabled": True},
        },
    )

    with pytest.raises(ValueError, match=r"embedding\.provider"):
        validate_settings(settings)
