"""手写 MCP Server：暴露仓库读写 / 测试 / git 工具。"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

# 脚本方式运行时要把项目根塞进 path
_SRC = Path(__file__).resolve().parent
_ROOT = _SRC.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.repo_tools import TOOL_DEFS, RepoTools, dispatch_tool  # noqa: E402


def list_tools() -> List[Dict[str, Any]]:
    return list(TOOL_DEFS)


def _repo_from_env() -> RepoTools:
    repo = os.environ.get("CODING_REPO") or os.environ.get("MCP_REPO_PATH") or "."
    return RepoTools(repo)


def call_tool(name: str, arguments: Dict[str, Any] | None = None) -> str:
    return dispatch_tool(_repo_from_env(), name, arguments or {})


def _run_fastmcp() -> None:
    from mcp.server.fastmcp import FastMCP

    mcp = FastMCP("mini-coding-agent-tools")
    # 每次调用读环境变量绑 repo，方便切换

    @mcp.tool(name="read_file", description="读取仓库内相对路径文件的全文")
    def read_file(path: str) -> str:
        return call_tool("read_file", {"path": path})

    @mcp.tool(name="write_file", description="写入仓库内相对路径文件（覆盖）")
    def write_file(path: str, content: str) -> str:
        return call_tool("write_file", {"path": path, "content": content})

    @mcp.tool(name="list_dir", description="列出仓库内某目录")
    def list_dir(path: str = ".") -> str:
        return call_tool("list_dir", {"path": path})

    @mcp.tool(name="run_tests", description="运行 pytest")
    def run_tests() -> str:
        return call_tool("run_tests", {})

    @mcp.tool(name="git_diff", description="查看工作区 diff")
    def git_diff() -> str:
        return call_tool("git_diff", {})

    @mcp.tool(name="git_apply", description="应用 unified diff")
    def git_apply(patch: str) -> str:
        return call_tool("git_apply", {"patch": patch})

    @mcp.tool(name="search_text", description="源码子串搜索")
    def search_text(pattern: str, glob: str = "*.py") -> str:
        return call_tool("search_text", {"pattern": pattern, "glob": glob})

    mcp.run(transport="stdio")


def _run_stdio_fallback() -> None:
    # FastMCP 不可用时的简易 JSON-RPC stdio 回环
    tools = list_tools()

    def reply(msg_id, result=None, error=None):
        payload = {"jsonrpc": "2.0", "id": msg_id}
        if error is not None:
            payload["error"] = error
        else:
            payload["result"] = result
        sys.stdout.write(json.dumps(payload, ensure_ascii=False) + "\n")
        sys.stdout.flush()

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except json.JSONDecodeError:
            continue
        method = req.get("method")
        msg_id = req.get("id")
        params = req.get("params") or {}

        if method == "initialize":
            reply(
                msg_id,
                {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {"tools": {}},
                    "serverInfo": {"name": "mini-coding-agent-tools", "version": "0.1"},
                },
            )
        elif method == "notifications/initialized":
            continue
        elif method == "tools/list":
            reply(
                msg_id,
                {
                    "tools": [
                        {
                            "name": t["name"],
                            "description": t.get("description", ""),
                            "inputSchema": t.get("input_schema", {}),
                        }
                        for t in tools
                    ]
                },
            )
        elif method == "tools/call":
            name = params.get("name")
            args = params.get("arguments") or {}
            text = call_tool(name, args)
            reply(msg_id, {"content": [{"type": "text", "text": text}]})
        elif method == "ping":
            reply(msg_id, {})
        else:
            if msg_id is not None:
                reply(msg_id, error={"code": -32601, "message": f"Method not found: {method}"})


if __name__ == "__main__":
    try:
        _run_fastmcp()
    except ImportError:
        print("[mcp_server] FastMCP 不可用，回退到简易 JSON-RPC stdio", file=sys.stderr)
        _run_stdio_fallback()
