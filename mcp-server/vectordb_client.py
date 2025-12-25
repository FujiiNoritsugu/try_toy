#!/usr/bin/env python3
"""
MCP Client for Vector Database operations
"""

import json
import subprocess
import sys
from typing import Optional


class VectorDBClient:
    """Client for interacting with the MCP Vector Database server."""

    def __init__(self, server_path: str = "./target/release/mcp-server"):
        self.server_path = server_path
        self.process = None
        self.request_id = 0

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

    def start(self):
        """Start the MCP server process."""
        self.process = subprocess.Popen(
            [self.server_path],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1
        )
        # Initialize
        self._send_request("initialize", {})
        # Send initialized notification
        self._send_notification("notifications/initialized", {})

    def stop(self):
        """Stop the MCP server process."""
        if self.process:
            self.process.terminate()
            self.process.wait()
            self.process = None

    def _send_request(self, method: str, params: dict) -> dict:
        """Send a JSON-RPC request and return the response."""
        self.request_id += 1
        request = {
            "jsonrpc": "2.0",
            "id": self.request_id,
            "method": method,
            "params": params
        }
        self.process.stdin.write(json.dumps(request) + "\n")
        self.process.stdin.flush()
        response_line = self.process.stdout.readline()
        return json.loads(response_line)

    def _send_notification(self, method: str, params: dict):
        """Send a JSON-RPC notification (no response expected)."""
        notification = {
            "jsonrpc": "2.0",
            "method": method,
            "params": params
        }
        self.process.stdin.write(json.dumps(notification) + "\n")
        self.process.stdin.flush()

    def _call_tool(self, tool_name: str, arguments: dict) -> dict:
        """Call a tool and return the parsed result."""
        response = self._send_request("tools/call", {
            "name": tool_name,
            "arguments": arguments
        })

        if "error" in response:
            raise Exception(response["error"].get("message", "Unknown error"))

        content = response.get("result", {}).get("content", [])
        if content:
            result_text = content[0].get("text", "{}")
            return json.loads(result_text)
        return {}

    # Vector Database Operations

    def store(self, text: str, id: Optional[str] = None, metadata: Optional[dict] = None) -> dict:
        """Store text with its vector embedding in the database."""
        arguments = {"text": text}
        if id:
            arguments["id"] = id
        if metadata:
            arguments["metadata"] = metadata
        return self._call_tool("vector_store", arguments)

    def search(self, query: str, top_k: int = 5) -> dict:
        """Search for similar entries using semantic similarity."""
        return self._call_tool("vector_search", {
            "query": query,
            "top_k": top_k
        })

    def get(self, id: str) -> dict:
        """Get a specific entry by ID."""
        return self._call_tool("vector_get", {"id": id})

    def delete(self, id: str) -> dict:
        """Delete an entry by ID."""
        return self._call_tool("vector_delete", {"id": id})

    def list(self, limit: int = 100) -> dict:
        """List all entries in the database."""
        return self._call_tool("vector_list", {"limit": limit})

    def embed(self, text: str, model: str = "text-embedding-3-small") -> dict:
        """Convert text to vector embedding."""
        return self._call_tool("theme_to_vector", {
            "theme": text,
            "model": model
        })


def interactive_mode(client: VectorDBClient):
    """Run an interactive session."""
    print("\nVector Database Interactive Mode")
    print("Commands: store, search, get, delete, list, embed, quit")
    print("-" * 50)

    while True:
        try:
            cmd = input("\n> ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if cmd == "quit" or cmd == "exit":
            print("Goodbye!")
            break

        elif cmd == "store":
            text = input("Text to store: ").strip()
            if not text:
                print("Error: Text cannot be empty")
                continue

            id_input = input("ID (press Enter for auto): ").strip()
            metadata_input = input("Metadata JSON (press Enter for none): ").strip()

            metadata = None
            if metadata_input:
                try:
                    metadata = json.loads(metadata_input)
                except json.JSONDecodeError:
                    print("Error: Invalid JSON for metadata")
                    continue

            result = client.store(text, id_input or None, metadata)
            print(f"\nStored successfully!")
            print(f"  ID: {result.get('id')}")

        elif cmd == "search":
            query = input("Search query: ").strip()
            if not query:
                print("Error: Query cannot be empty")
                continue

            top_k_input = input("Number of results (default 5): ").strip()
            top_k = int(top_k_input) if top_k_input else 5

            result = client.search(query, top_k)
            print(f"\nFound {result.get('count', 0)} results:")
            for i, entry in enumerate(result.get("results", []), 1):
                print(f"\n  {i}. [{entry.get('similarity'):.4f}] {entry.get('id')}")
                print(f"     {entry.get('text')[:80]}...")
                if entry.get("metadata"):
                    print(f"     Metadata: {entry.get('metadata')}")

        elif cmd == "get":
            id = input("Entry ID: ").strip()
            if not id:
                print("Error: ID cannot be empty")
                continue

            result = client.get(id)
            if result.get("found"):
                print(f"\nEntry found:")
                print(f"  ID: {result.get('id')}")
                print(f"  Text: {result.get('text')}")
                print(f"  Dimensions: {result.get('dimensions')}")
                print(f"  Metadata: {result.get('metadata')}")
            else:
                print(f"Entry not found: {result.get('message')}")

        elif cmd == "delete":
            id = input("Entry ID to delete: ").strip()
            if not id:
                print("Error: ID cannot be empty")
                continue

            result = client.delete(id)
            if result.get("success"):
                print(f"Deleted: {result.get('deleted_text')}")
            else:
                print(f"Failed: {result.get('message')}")

        elif cmd == "list":
            limit_input = input("Limit (default 100): ").strip()
            limit = int(limit_input) if limit_input else 100

            result = client.list(limit)
            print(f"\nTotal entries: {result.get('total_count', 0)}")
            for entry in result.get("entries", []):
                print(f"\n  [{entry.get('id')}]")
                print(f"    {entry.get('text')[:60]}...")
                print(f"    Dimensions: {entry.get('dimensions')}")

        elif cmd == "embed":
            text = input("Text to embed: ").strip()
            if not text:
                print("Error: Text cannot be empty")
                continue

            result = client.embed(text)
            print(f"\nEmbedding generated:")
            print(f"  Source: {result.get('source')}")
            print(f"  Dimensions: {result.get('dimensions')}")
            vector = result.get("vector", [])
            print(f"  Vector (first 5): {vector[:5]}")

        elif cmd == "help":
            print("\nCommands:")
            print("  store  - Store text with embedding")
            print("  search - Semantic similarity search")
            print("  get    - Get entry by ID")
            print("  delete - Delete entry by ID")
            print("  list   - List all entries")
            print("  embed  - Generate embedding for text")
            print("  quit   - Exit")

        else:
            print("Unknown command. Type 'help' for available commands.")


def demo_mode(client: VectorDBClient):
    """Run a demonstration of the vector database."""
    print("\n=== Vector Database Demo ===\n")

    # Store some sample data
    print("1. Storing sample documents...")
    samples = [
        ("Machine learning is a branch of artificial intelligence", {"category": "AI"}),
        ("Neural networks are inspired by biological neurons", {"category": "AI"}),
        ("Python is widely used for data science", {"category": "Programming"}),
        ("JavaScript runs in web browsers", {"category": "Programming"}),
        ("Deep learning requires large datasets and GPUs", {"category": "AI"}),
    ]

    stored_ids = []
    for text, metadata in samples:
        result = client.store(text, metadata=metadata)
        stored_ids.append(result.get("id"))
        print(f"  Stored: {text[:40]}... -> {result.get('id')}")

    # Search
    print("\n2. Searching for 'artificial intelligence algorithms'...")
    results = client.search("artificial intelligence algorithms", top_k=3)
    print(f"  Found {results.get('count')} results:")
    for entry in results.get("results", []):
        print(f"    [{entry.get('similarity'):.4f}] {entry.get('text')[:50]}...")

    # List all
    print("\n3. Listing all entries...")
    all_entries = client.list()
    print(f"  Total: {all_entries.get('total_count')} entries")

    # Get specific entry
    if stored_ids:
        print(f"\n4. Getting entry {stored_ids[0]}...")
        entry = client.get(stored_ids[0])
        if entry.get("found"):
            print(f"  Text: {entry.get('text')}")
            print(f"  Metadata: {entry.get('metadata')}")

    print("\n=== Demo Complete ===")


def main():
    """Main entry point."""
    if len(sys.argv) > 1:
        mode = sys.argv[1]
    else:
        mode = "interactive"

    print("Starting Vector Database Client...")

    with VectorDBClient() as client:
        if mode == "demo":
            demo_mode(client)
        elif mode == "interactive":
            interactive_mode(client)
        else:
            print(f"Unknown mode: {mode}")
            print("Usage: python vectordb_client.py [interactive|demo]")


if __name__ == "__main__":
    main()
