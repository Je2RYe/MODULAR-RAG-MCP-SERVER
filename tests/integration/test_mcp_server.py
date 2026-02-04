"""Integration tests for MCP server stdio entrypoint."""

from __future__ import annotations

import json
import subprocess
import sys
import threading
from typing import Dict

import pytest


def _read_line(stream, buffer: Dict[str, str], key: str) -> None:
    buffer[key] = stream.readline().strip()


@pytest.mark.integration
def test_mcp_server_initialize_stdio() -> None:
    """Ensure initialize works and stdout is clean JSON-RPC output."""

    proc = subprocess.Popen(
        [sys.executable, "-m", "src.mcp_server.server"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    request = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": {
            "protocolVersion": "2025-06-18",
            "clientInfo": {"name": "pytest", "version": "0.0.0"},
        },
    }

    assert proc.stdin is not None
    proc.stdin.write(json.dumps(request) + "\n")
    proc.stdin.flush()

    output: Dict[str, str] = {}
    stdout_thread = threading.Thread(target=_read_line, args=(proc.stdout, output, "stdout"))
    stderr_thread = threading.Thread(target=_read_line, args=(proc.stderr, output, "stderr"))

    stdout_thread.start()
    stderr_thread.start()

    stdout_thread.join(timeout=5)
    stderr_thread.join(timeout=5)

    proc.terminate()
    proc.wait(timeout=5)

    if stdout_thread.is_alive():
        stdout_thread.join(timeout=1)
    if stderr_thread.is_alive():
        stderr_thread.join(timeout=1)

    assert "stdout" in output, "Server did not emit stdout response."
    response = json.loads(output["stdout"])

    assert response["jsonrpc"] == "2.0"
    assert response["id"] == 1
    assert "result" in response
    assert "serverInfo" in response["result"]
    assert "capabilities" in response["result"]

    assert "stderr" in output
    assert output["stderr"] != ""