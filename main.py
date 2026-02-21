"""MCP Server 启动入口。

本模块为 Modular RAG MCP Server 的主入口点。
启动时加载配置并初始化 MCP Server（Stdio Transport）。
"""

from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from core.settings import load_settings
from observability.logger import get_logger


LOGGER = get_logger(__name__)


def main() -> None:
    """启动 MCP Server。"""
    try:
        settings = load_settings("config/settings.yaml")
    except (FileNotFoundError, ValueError) as exc:
        LOGGER.error("Failed to load settings: %s", exc)
        raise SystemExit(1) from exc

    LOGGER.info(
        "Settings loaded successfully (llm=%s, embedding=%s)",
        settings.llm.get("provider", "unknown"),
        settings.embedding.get("provider", "unknown"),
    )

    # E1 阶段将接入 MCP Server 启动逻辑
    print("Modular RAG MCP Server — ready.", file=sys.stderr)


if __name__ == "__main__":
    main()
