import json
import sys
from typing import List, Dict, Any
from app.services.contextor import ContextorPipeline
from app.services.injection_guard import InjectionGuard
from app.services.compressor import ContextCompressor

pipeline = ContextorPipeline()
guard = InjectionGuard()
compressor = ContextCompressor()

MCP_TOOLS_MANIFEST = [
    {
        "name": "get_relevant_context",
        "description": "Scans, decays, and filters historical turn context to return relevant chunks.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Current turn search query or user intent"},
                "turn_history": {"type": "array", "items": {"type": "object"}, "description": "List of turn messages"},
                "current_turn": {"type": "integer", "description": "Current turn index"}
            },
            "required": ["query", "turn_history"]
        }
    },
    {
        "name": "check_injection",
        "description": "Scans tool output for prompt injection, prompt leakage, or instruction override attempts.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "text": {"type": "string", "description": "Tool result output text to audit"},
                "source": {"type": "string", "description": "Context source, default 'tool_output'"}
            },
            "required": ["text"]
        }
    },
    {
        "name": "compress_context",
        "description": "Summarizes low-relevance context chunks into a single condensed note.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "chunks": {"type": "array", "items": {"type": "object"}, "description": "List of low relevance chunks"}
            },
            "required": ["chunks"]
        }
    }
]


def handle_mcp_request(request: Dict[str, Any]) -> Dict[str, Any]:
    """
    Handles Model Context Protocol (MCP) JSON-RPC request frames.
    """
    req_id = request.get("id")
    method = request.get("method")
    params = request.get("params", {})

    if method == "tools/list":
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {"tools": MCP_TOOLS_MANIFEST}
        }

    elif method == "tools/call":
        name = params.get("name")
        args = params.get("arguments", {})

        if name == "get_relevant_context":
            query = args.get("query", "")
            turn_history = args.get("turn_history", [])
            current_turn = args.get("current_turn", len(turn_history))
            res = pipeline.process(current_query=query, turn_history=turn_history, current_turn=current_turn)
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {"content": [{"type": "text", "text": json.dumps(res, indent=2)}]}
            }

        elif name == "check_injection":
            text = args.get("text", "")
            source = args.get("source", "tool_output")
            res = guard.scan(text=text, source=source)
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {"content": [{"type": "text", "text": json.dumps(res, indent=2)}]}
            }

        elif name == "compress_context":
            chunks = args.get("chunks", [])
            summary = compressor.compress_chunks(chunks)
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {"content": [{"type": "text", "text": summary}]}
            }

        else:
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "error": {"code": -32601, "message": f"Method / Tool '{name}' not found"}
            }

    elif method == "initialize":
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {
                "protocolVersion": "2024-11-05",
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "Contextor MCP Server", "version": "0.1.0"}
            }
        }

    else:
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "error": {"code": -32601, "message": f"Unsupported method '{method}'"}
        }


def run_stdio_mcp_server():
    """Run STDIO loop for MCP server interaction."""
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
            resp = handle_mcp_request(req)
            sys.stdout.write(json.dumps(resp) + "\n")
            sys.stdout.flush()
        except Exception as e:
            err_resp = {
                "jsonrpc": "2.0",
                "id": None,
                "error": {"code": -32700, "message": f"Parse error: {str(e)}"}
            }
            sys.stdout.write(json.dumps(err_resp) + "\n")
            sys.stdout.flush()


if __name__ == "__main__":
    run_stdio_mcp_server()
