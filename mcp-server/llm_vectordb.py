#!/usr/bin/env python3
"""
LLM-powered Vector Database Client
Uses Claude API to interpret natural language commands and operate the vector database.
"""

import json
import os
import subprocess
from typing import Optional

from anthropic import Anthropic


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
        self.process = subprocess.Popen(
            [self.server_path],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1
        )
        self._send_request("initialize", {})
        self._send_notification("notifications/initialized", {})

    def stop(self):
        if self.process:
            self.process.terminate()
            self.process.wait()
            self.process = None

    def _send_request(self, method: str, params: dict) -> dict:
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
        notification = {
            "jsonrpc": "2.0",
            "method": method,
            "params": params
        }
        self.process.stdin.write(json.dumps(notification) + "\n")
        self.process.stdin.flush()

    def _call_tool(self, tool_name: str, arguments: dict) -> dict:
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

    def store(self, text: str, id: Optional[str] = None, metadata: Optional[dict] = None) -> dict:
        arguments = {"text": text}
        if id:
            arguments["id"] = id
        if metadata:
            arguments["metadata"] = metadata
        return self._call_tool("vector_store", arguments)

    def search(self, query: str, top_k: int = 5) -> dict:
        return self._call_tool("vector_search", {"query": query, "top_k": top_k})

    def get(self, id: str) -> dict:
        return self._call_tool("vector_get", {"id": id})

    def delete(self, id: str) -> dict:
        return self._call_tool("vector_delete", {"id": id})

    def list(self, limit: int = 100) -> dict:
        return self._call_tool("vector_list", {"limit": limit})

    def read_file(self, path: str) -> dict:
        return self._call_tool("read_file", {"path": path})

    def load_file_to_db(self, path: str, split_mode: str = "none", metadata: Optional[dict] = None) -> dict:
        arguments: dict = {"path": path, "split_mode": split_mode}
        if metadata:
            arguments["metadata"] = metadata
        return self._call_tool("load_file_to_db", arguments)


# Tool definitions for Claude
TOOLS = [
    {
        "name": "vector_store",
        "description": "Store text with its vector embedding in the database. Use this to save new documents or information.",
        "input_schema": {
            "type": "object",
            "properties": {
                "text": {
                    "type": "string",
                    "description": "The text content to store"
                },
                "id": {
                    "type": "string",
                    "description": "Optional unique identifier (auto-generated if not provided)"
                },
                "metadata": {
                    "type": "object",
                    "description": "Optional metadata like category, tags, source, etc."
                }
            },
            "required": ["text"]
        }
    },
    {
        "name": "vector_search",
        "description": "Search for similar documents using semantic similarity. Use this to find relevant information.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query"
                },
                "top_k": {
                    "type": "integer",
                    "description": "Number of results to return (default: 5)"
                }
            },
            "required": ["query"]
        }
    },
    {
        "name": "vector_get",
        "description": "Get a specific document by its ID.",
        "input_schema": {
            "type": "object",
            "properties": {
                "id": {
                    "type": "string",
                    "description": "The document ID"
                }
            },
            "required": ["id"]
        }
    },
    {
        "name": "vector_delete",
        "description": "Delete a document from the database by its ID.",
        "input_schema": {
            "type": "object",
            "properties": {
                "id": {
                    "type": "string",
                    "description": "The document ID to delete"
                }
            },
            "required": ["id"]
        }
    },
    {
        "name": "vector_list",
        "description": "List all documents in the database.",
        "input_schema": {
            "type": "object",
            "properties": {
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of documents to return (default: 100)"
                }
            }
        }
    },
    {
        "name": "read_file",
        "description": "Read the contents of a local text file. Use this to see what's in a file before loading it.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path to the text file to read"
                }
            },
            "required": ["path"]
        }
    },
    {
        "name": "load_file_to_db",
        "description": "Read a text file and store its content in the vector database. Can split the file by lines or paragraphs for better searchability.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path to the text file to load"
                },
                "split_mode": {
                    "type": "string",
                    "description": "How to split the file: 'none' (whole file as one entry), 'lines' (each line as separate entry), 'paragraphs' (split by empty lines)",
                    "enum": ["none", "lines", "paragraphs"]
                },
                "metadata": {
                    "type": "object",
                    "description": "Optional metadata to attach to all entries from this file"
                }
            },
            "required": ["path"]
        }
    }
]

SYSTEM_PROMPT = """You are a helpful assistant that manages a vector database. You can:
- Store new documents/text with optional metadata
- Search for similar documents using semantic search
- Get, delete, or list documents
- Read local text files
- Load text files into the database (with options to split by lines or paragraphs)

When users ask you to do something with the database, use the appropriate tools.
When showing search results, explain the relevance and similarity scores.
When loading files, suggest appropriate split modes based on the file content.
Be conversational and helpful. If the user's request is unclear, ask for clarification.

Always respond in the same language as the user's input."""


class LLMVectorDBAssistant:
    """LLM-powered assistant for vector database operations."""

    def __init__(self, db_client: VectorDBClient):
        self.db_client = db_client
        self.client = Anthropic()
        self.conversation_history = []

    def _execute_tool(self, tool_name: str, tool_input: dict) -> str:
        """Execute a tool and return the result as a string."""
        try:
            if tool_name == "vector_store":
                result = self.db_client.store(
                    text=tool_input["text"],
                    id=tool_input.get("id"),
                    metadata=tool_input.get("metadata")
                )
            elif tool_name == "vector_search":
                result = self.db_client.search(
                    query=tool_input["query"],
                    top_k=tool_input.get("top_k", 5)
                )
            elif tool_name == "vector_get":
                result = self.db_client.get(id=tool_input["id"])
            elif tool_name == "vector_delete":
                result = self.db_client.delete(id=tool_input["id"])
            elif tool_name == "vector_list":
                result = self.db_client.list(limit=tool_input.get("limit", 100))
            elif tool_name == "read_file":
                result = self.db_client.read_file(path=tool_input["path"])
            elif tool_name == "load_file_to_db":
                result = self.db_client.load_file_to_db(
                    path=tool_input["path"],
                    split_mode=tool_input.get("split_mode", "none"),
                    metadata=tool_input.get("metadata")
                )
            else:
                return json.dumps({"error": f"Unknown tool: {tool_name}"})

            return json.dumps(result, ensure_ascii=False, indent=2)
        except Exception as e:
            return json.dumps({"error": str(e)})

    def chat(self, user_message: str) -> str:
        """Process a user message and return the assistant's response."""
        self.conversation_history.append({
            "role": "user",
            "content": user_message
        })

        # Initial API call
        response = self.client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=4096,
            system=SYSTEM_PROMPT,
            tools=TOOLS,
            messages=self.conversation_history
        )

        # Process tool calls in a loop
        while response.stop_reason == "tool_use":
            # Find tool use blocks
            tool_uses = [block for block in response.content if block.type == "tool_use"]

            # Build assistant message with all content
            assistant_content = []
            for block in response.content:
                if block.type == "text":
                    assistant_content.append({"type": "text", "text": block.text})
                elif block.type == "tool_use":
                    assistant_content.append({
                        "type": "tool_use",
                        "id": block.id,
                        "name": block.name,
                        "input": block.input
                    })

            self.conversation_history.append({
                "role": "assistant",
                "content": assistant_content
            })

            # Execute tools and collect results
            tool_results = []
            for tool_use in tool_uses:
                result = self._execute_tool(tool_use.name, tool_use.input)
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": tool_use.id,
                    "content": result
                })

            self.conversation_history.append({
                "role": "user",
                "content": tool_results
            })

            # Continue the conversation
            response = self.client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=4096,
                system=SYSTEM_PROMPT,
                tools=TOOLS,
                messages=self.conversation_history
            )

        # Extract final text response
        final_response = ""
        for block in response.content:
            if hasattr(block, "text"):
                final_response += block.text

        self.conversation_history.append({
            "role": "assistant",
            "content": final_response
        })

        return final_response


def main():
    """Main entry point."""
    # Load .env file
    from dotenv import load_dotenv
    load_dotenv()

    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("Error: ANTHROPIC_API_KEY environment variable not set")
        print("Please set it in .env file or export it")
        return

    print("Starting LLM-powered Vector Database Assistant...")
    print("Type 'quit' to exit\n")

    with VectorDBClient() as db_client:
        assistant = LLMVectorDBAssistant(db_client)

        print("Assistant: Hello! I can help you manage your vector database.")
        print("           You can ask me to store, search, list, or delete documents.")
        print("           Just tell me what you'd like to do in natural language.\n")

        while True:
            try:
                user_input = input("You: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nGoodbye!")
                break

            if not user_input:
                continue

            if user_input.lower() in ["quit", "exit", "bye"]:
                print("Goodbye!")
                break

            try:
                response = assistant.chat(user_input)
                print(f"\nAssistant: {response}\n")
            except Exception as e:
                print(f"\nError: {e}\n")


if __name__ == "__main__":
    main()
