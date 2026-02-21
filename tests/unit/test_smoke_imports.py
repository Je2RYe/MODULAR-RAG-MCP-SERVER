"""Smoke tests for basic package imports."""


def test_imports_smoke() -> None:
    import core  # noqa: F401
    import ingestion  # noqa: F401
    import libs  # noqa: F401
    import mcp_server  # noqa: F401
    import observability  # noqa: F401
