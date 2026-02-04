"""MCP Server entry point for stdio transport.

This module implements a minimal JSON-RPC 2.0 loop that supports
the MCP `initialize` request. It ensures stdout only contains
protocol messages while all logs go to stderr.
"""

from __future__ import annotations

import json
import sys
from typing import Any, Dict, Optional

from src.observability.logger import get_logger


DEFAULT_PROTOCOL_VERSION = "2025-06-18"
SERVER_NAME = "modular-rag-mcp-server"
SERVER_VERSION = "0.1.0"


def _build_initialize_result(params: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Build MCP initialize result payload.

    Args:
        params: Initialize request parameters.

    Returns:
        Result payload for the initialize response.
    """

    params = params or {}
    protocol_version = params.get("protocolVersion") or DEFAULT_PROTOCOL_VERSION
    return {
        "protocolVersion": protocol_version,
        "serverInfo": {"name": SERVER_NAME, "version": SERVER_VERSION},
        "capabilities": {"tools": {}},
    }


def _write_response(payload: Dict[str, Any]) -> None:
    """Write a JSON-RPC response to stdout.

    Args:
        payload: JSON-RPC response payload.
    """

    sys.stdout.write(json.dumps(payload, ensure_ascii=False) + "\n")
    sys.stdout.flush()


def _handle_request(request: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Handle a single JSON-RPC request.

    Args:
        request: Parsed JSON-RPC request.

    Returns:
        JSON-RPC response payload, or None for notifications.
    """

    method = request.get("method")
    request_id = request.get("id")
    if method == "initialize":
        result = _build_initialize_result(request.get("params"))
        return {"jsonrpc": "2.0", "id": request_id, "result": result}

    if request_id is None:
        return None

    return {
        "jsonrpc": "2.0",
        "id": request_id,
        "error": {"code": -32601, "message": "Method not found"},
    }


def run_stdio_server() -> int:
    """Run MCP server over stdio.

    Returns:
        Exit code.
    """

    logger = get_logger(log_level="INFO")
    logger.info("Starting MCP server (stdio transport).")

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)

    for line in sys.stdin:
        raw = line.strip()
        if not raw:
            continue
        try:
            request = json.loads(raw)
        except json.JSONDecodeError:
            logger.warning("Invalid JSON received on stdin.")
            continue

        response = _handle_request(request)
        if response is not None:
            _write_response(response)
            logger.info("Handled request: %s", request.get("method"))

    logger.info("MCP server shutting down.")
    return 0


def main() -> int:
    """Entry point for stdio MCP server."""

    return run_stdio_server()


if __name__ == "__main__":
    raise SystemExit(main())