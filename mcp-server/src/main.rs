use reqwest::blocking::Client;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::collections::hash_map::DefaultHasher;
use std::env;
use std::hash::{Hash, Hasher};
use std::io::{self, BufRead, Write};

// JSON-RPC 2.0 Request
#[derive(Debug, Deserialize)]
struct JsonRpcRequest {
    #[allow(dead_code)]
    jsonrpc: String,
    id: Option<Value>,
    method: String,
    #[serde(default)]
    params: Value,
}

// JSON-RPC 2.0 Response
#[derive(Debug, Serialize)]
struct JsonRpcResponse {
    jsonrpc: String,
    id: Value,
    #[serde(skip_serializing_if = "Option::is_none")]
    result: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    error: Option<JsonRpcError>,
}

#[derive(Debug, Serialize)]
struct JsonRpcError {
    code: i32,
    message: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    data: Option<Value>,
}

// MCP Tool definition
#[derive(Debug, Serialize)]
struct Tool {
    name: String,
    description: String,
    #[serde(rename = "inputSchema")]
    input_schema: Value,
}

// OpenAI API types
#[derive(Debug, Serialize)]
struct OpenAIEmbeddingRequest {
    model: String,
    input: String,
}

#[derive(Debug, Deserialize)]
struct OpenAIEmbeddingResponse {
    data: Vec<EmbeddingData>,
    #[allow(dead_code)]
    model: String,
    usage: EmbeddingUsage,
}

#[derive(Debug, Deserialize)]
struct EmbeddingData {
    embedding: Vec<f64>,
    #[allow(dead_code)]
    index: usize,
}

#[derive(Debug, Deserialize)]
struct EmbeddingUsage {
    prompt_tokens: u32,
    total_tokens: u32,
}

struct McpServer {
    server_name: String,
    server_version: String,
    http_client: Client,
    openai_api_key: Option<String>,
}

impl McpServer {
    fn new() -> Self {
        // Load .env file if it exists
        if let Err(e) = dotenvy::dotenv() {
            eprintln!("Note: .env file not loaded ({})", e);
        }

        let openai_api_key = env::var("OPENAI_API_KEY").ok();
        if openai_api_key.is_some() {
            eprintln!("OpenAI API key found. Using OpenAI Embeddings API.");
        } else {
            eprintln!("No OPENAI_API_KEY found. Using fallback hash-based vectors.");
        }

        McpServer {
            server_name: "rust-mcp-server".to_string(),
            server_version: "0.1.0".to_string(),
            http_client: Client::new(),
            openai_api_key,
        }
    }

    fn handle_request(&self, request: &JsonRpcRequest) -> Option<JsonRpcResponse> {
        let id = request.id.clone().unwrap_or(Value::Null);

        // Handle notifications (no response needed)
        if request.id.is_none() && request.method.starts_with("notifications/") {
            eprintln!("Received notification: {}", request.method);
            return None;
        }

        let result = match request.method.as_str() {
            "initialize" => self.handle_initialize(),
            "tools/list" => self.handle_tools_list(),
            "tools/call" => self.handle_tools_call(&request.params),
            _ => Err(JsonRpcError {
                code: -32601,
                message: format!("Method not found: {}", request.method),
                data: None,
            }),
        };

        Some(match result {
            Ok(value) => self.make_success_response(id, value),
            Err(error) => self.make_error_response(id, error),
        })
    }

    fn handle_initialize(&self) -> Result<Value, JsonRpcError> {
        Ok(json!({
            "protocolVersion": "2024-11-05",
            "capabilities": {
                "tools": {
                    "listChanged": false
                }
            },
            "serverInfo": {
                "name": self.server_name,
                "version": self.server_version
            }
        }))
    }

    fn handle_tools_list(&self) -> Result<Value, JsonRpcError> {
        let tools = vec![
            Tool {
                name: "echo".to_string(),
                description: "Echoes back the input message".to_string(),
                input_schema: json!({
                    "type": "object",
                    "properties": {
                        "message": {
                            "type": "string",
                            "description": "The message to echo back"
                        }
                    },
                    "required": ["message"]
                }),
            },
            Tool {
                name: "add".to_string(),
                description: "Adds two numbers together".to_string(),
                input_schema: json!({
                    "type": "object",
                    "properties": {
                        "a": {
                            "type": "number",
                            "description": "First number"
                        },
                        "b": {
                            "type": "number",
                            "description": "Second number"
                        }
                    },
                    "required": ["a", "b"]
                }),
            },
            Tool {
                name: "get_time".to_string(),
                description: "Returns the current Unix timestamp".to_string(),
                input_schema: json!({
                    "type": "object",
                    "properties": {}
                }),
            },
            Tool {
                name: "theme_to_vector".to_string(),
                description: "Converts a theme name into a vector embedding using OpenAI Embeddings API (text-embedding-3-small). Falls back to hash-based vectors if API key is not set.".to_string(),
                input_schema: json!({
                    "type": "object",
                    "properties": {
                        "theme": {
                            "type": "string",
                            "description": "The theme name to convert to a vector"
                        },
                        "model": {
                            "type": "string",
                            "description": "OpenAI embedding model to use (default: text-embedding-3-small)",
                            "enum": ["text-embedding-3-small", "text-embedding-3-large", "text-embedding-ada-002"]
                        }
                    },
                    "required": ["theme"]
                }),
            },
        ];

        Ok(json!({ "tools": tools }))
    }

    fn handle_tools_call(&self, params: &Value) -> Result<Value, JsonRpcError> {
        let tool_name = params
            .get("name")
            .and_then(|v| v.as_str())
            .ok_or_else(|| JsonRpcError {
                code: -32602,
                message: "Missing tool name".to_string(),
                data: None,
            })?;

        let empty_args = json!({});
        let arguments = params.get("arguments").unwrap_or(&empty_args);

        let result = match tool_name {
            "echo" => self.tool_echo(arguments),
            "add" => self.tool_add(arguments),
            "get_time" => self.tool_get_time(),
            "theme_to_vector" => self.tool_theme_to_vector(arguments),
            _ => Err(format!("Unknown tool: {}", tool_name)),
        };

        match result {
            Ok(text) => Ok(json!({
                "content": [{
                    "type": "text",
                    "text": text
                }]
            })),
            Err(err_msg) => Ok(json!({
                "content": [{
                    "type": "text",
                    "text": err_msg
                }],
                "isError": true
            })),
        }
    }

    fn tool_echo(&self, args: &Value) -> Result<String, String> {
        let message = args
            .get("message")
            .and_then(|v| v.as_str())
            .ok_or("Missing 'message' argument")?;
        Ok(format!("Echo: {}", message))
    }

    fn tool_add(&self, args: &Value) -> Result<String, String> {
        let a = args
            .get("a")
            .and_then(|v| v.as_f64())
            .ok_or("Missing or invalid 'a' argument")?;
        let b = args
            .get("b")
            .and_then(|v| v.as_f64())
            .ok_or("Missing or invalid 'b' argument")?;
        Ok(format!("{} + {} = {}", a, b, a + b))
    }

    fn tool_get_time(&self) -> Result<String, String> {
        use std::time::{SystemTime, UNIX_EPOCH};
        let duration = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_err(|e| e.to_string())?;
        let secs = duration.as_secs();
        Ok(format!("Current Unix timestamp: {}", secs))
    }

    fn tool_theme_to_vector(&self, args: &Value) -> Result<String, String> {
        let theme = args
            .get("theme")
            .and_then(|v| v.as_str())
            .ok_or("Missing 'theme' argument")?;

        let model = args
            .get("model")
            .and_then(|v| v.as_str())
            .unwrap_or("text-embedding-3-small");

        // Try OpenAI API first, fall back to hash-based if unavailable
        match self.get_openai_embedding(theme, model) {
            Ok((vector, usage)) => {
                let result = json!({
                    "theme": theme,
                    "model": model,
                    "dimensions": vector.len(),
                    "source": "openai",
                    "usage": {
                        "prompt_tokens": usage.prompt_tokens,
                        "total_tokens": usage.total_tokens
                    },
                    "vector": vector
                });
                Ok(serde_json::to_string_pretty(&result).unwrap())
            }
            Err(api_error) => {
                eprintln!("OpenAI API error: {}. Falling back to hash-based vector.", api_error);
                let dimensions = 128;
                let vector = self.generate_hash_vector(theme, dimensions);
                let result = json!({
                    "theme": theme,
                    "dimensions": dimensions,
                    "source": "hash-based (fallback)",
                    "fallback_reason": api_error,
                    "vector": vector
                });
                Ok(serde_json::to_string_pretty(&result).unwrap())
            }
        }
    }

    fn get_openai_embedding(&self, text: &str, model: &str) -> Result<(Vec<f64>, EmbeddingUsage), String> {
        let api_key = self.openai_api_key.as_ref()
            .ok_or("OPENAI_API_KEY environment variable not set")?;

        let request_body = OpenAIEmbeddingRequest {
            model: model.to_string(),
            input: text.to_string(),
        };

        let response = self.http_client
            .post("https://api.openai.com/v1/embeddings")
            .header("Authorization", format!("Bearer {}", api_key))
            .header("Content-Type", "application/json")
            .json(&request_body)
            .send()
            .map_err(|e| format!("HTTP request failed: {}", e))?;

        if !response.status().is_success() {
            let status = response.status();
            let error_body = response.text().unwrap_or_default();
            return Err(format!("API returned error {}: {}", status, error_body));
        }

        let embedding_response: OpenAIEmbeddingResponse = response
            .json()
            .map_err(|e| format!("Failed to parse response: {}", e))?;

        let embedding = embedding_response.data
            .into_iter()
            .next()
            .ok_or("No embedding data in response")?
            .embedding;

        Ok((embedding, embedding_response.usage))
    }

    fn generate_hash_vector(&self, text: &str, dimensions: usize) -> Vec<f64> {
        let mut vector = Vec::with_capacity(dimensions);

        for i in 0..dimensions {
            let mut hasher = DefaultHasher::new();
            text.hash(&mut hasher);
            i.hash(&mut hasher);
            let hash = hasher.finish();

            // Convert hash to a value between -1.0 and 1.0
            let value = ((hash as f64) / (u64::MAX as f64)) * 2.0 - 1.0;
            vector.push((value * 10000.0).round() / 10000.0);
        }

        // Normalize the vector
        let magnitude: f64 = vector.iter().map(|x| x * x).sum::<f64>().sqrt();
        if magnitude > 0.0 {
            for v in &mut vector {
                *v = (*v / magnitude * 10000.0).round() / 10000.0;
            }
        }

        vector
    }

    fn make_success_response(&self, id: Value, result: Value) -> JsonRpcResponse {
        JsonRpcResponse {
            jsonrpc: "2.0".to_string(),
            id,
            result: Some(result),
            error: None,
        }
    }

    fn make_error_response(&self, id: Value, error: JsonRpcError) -> JsonRpcResponse {
        JsonRpcResponse {
            jsonrpc: "2.0".to_string(),
            id,
            result: None,
            error: Some(error),
        }
    }
}

fn main() {
    let server = McpServer::new();
    let stdin = io::stdin();
    let mut stdout = io::stdout();

    eprintln!("MCP Server started. Waiting for JSON-RPC messages on stdin...");

    for line in stdin.lock().lines() {
        let line = match line {
            Ok(l) => l,
            Err(e) => {
                eprintln!("Error reading line: {}", e);
                continue;
            }
        };

        if line.trim().is_empty() {
            continue;
        }

        let request: JsonRpcRequest = match serde_json::from_str(&line) {
            Ok(req) => req,
            Err(e) => {
                let error_response = json!({
                    "jsonrpc": "2.0",
                    "id": null,
                    "error": {
                        "code": -32700,
                        "message": format!("Parse error: {}", e)
                    }
                });
                writeln!(stdout, "{}", error_response).ok();
                stdout.flush().ok();
                continue;
            }
        };

        if let Some(response) = server.handle_request(&request) {
            let response_json = serde_json::to_string(&response).unwrap();
            writeln!(stdout, "{}", response_json).ok();
            stdout.flush().ok();
        }
    }
}
