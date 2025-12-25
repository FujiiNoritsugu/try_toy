#!/usr/bin/env python3
"""
MCP Client for theme_to_vector tool
"""

import json
import subprocess
import sys


def send_request(process, method: str, params: dict = None, request_id: int = 1) -> dict:
    """Send a JSON-RPC request to the MCP server and return the response."""
    request = {
        "jsonrpc": "2.0",
        "id": request_id,
        "method": method,
        "params": params or {}
    }

    request_json = json.dumps(request) + "\n"
    process.stdin.write(request_json)
    process.stdin.flush()

    response_line = process.stdout.readline()
    return json.loads(response_line)


def main():
    # Default theme
    theme = "machine learning"
    model = None

    # Parse command line arguments
    if len(sys.argv) > 1:
        theme = sys.argv[1]
    if len(sys.argv) > 2:
        model = sys.argv[2]

    # Start the MCP server
    server_path = "./target/release/mcp-server"
    process = subprocess.Popen(
        [server_path],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1
    )

    try:
        # 1. Initialize
        print("Initializing MCP server...")
        response = send_request(process, "initialize", {}, 1)
        server_info = response.get("result", {}).get("serverInfo", {})
        print(f"Connected to: {server_info.get('name')} v{server_info.get('version')}")

        # 2. Send initialized notification
        notification = {
            "jsonrpc": "2.0",
            "method": "notifications/initialized",
            "params": {}
        }
        process.stdin.write(json.dumps(notification) + "\n")
        process.stdin.flush()

        # 3. Call theme_to_vector tool
        print(f"\nConverting theme to vector: '{theme}'")
        if model:
            print(f"Using model: {model}")

        arguments = {"theme": theme}
        if model:
            arguments["model"] = model

        response = send_request(
            process,
            "tools/call",
            {
                "name": "theme_to_vector",
                "arguments": arguments
            },
            2
        )

        # 4. Parse and display result
        if "result" in response:
            content = response["result"].get("content", [])
            if content:
                result_text = content[0].get("text", "")
                result_data = json.loads(result_text)

                print(f"\nResult:")
                print(f"  Theme: {result_data.get('theme')}")
                print(f"  Source: {result_data.get('source')}")
                print(f"  Dimensions: {result_data.get('dimensions')}")

                if "model" in result_data:
                    print(f"  Model: {result_data.get('model')}")

                if "usage" in result_data:
                    usage = result_data["usage"]
                    print(f"  Tokens: {usage.get('total_tokens')}")

                if "fallback_reason" in result_data:
                    print(f"  Fallback reason: {result_data.get('fallback_reason')}")

                vector = result_data.get("vector", [])
                print(f"\nVector (first 10 dimensions):")
                print(f"  {vector[:10]}")

        elif "error" in response:
            error = response["error"]
            print(f"Error: {error.get('message')}")

    finally:
        process.terminate()
        process.wait()


if __name__ == "__main__":
    main()
