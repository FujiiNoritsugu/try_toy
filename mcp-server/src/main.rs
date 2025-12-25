use reqwest::blocking::Client;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::collections::hash_map::DefaultHasher;
use std::collections::HashMap;
use std::env;
use std::fs;
use std::hash::{Hash, Hasher};
use std::io::{self, BufRead, Write};
use std::path::PathBuf;
use std::sync::{Arc, RwLock};

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

// Vector Database types
#[derive(Debug, Clone, Serialize, Deserialize)]
struct VectorEntry {
    id: String,
    text: String,
    vector: Vec<f64>,
    metadata: Option<Value>,
    created_at: u64,
}

#[derive(Debug, Default, Serialize, Deserialize)]
struct VectorDatabase {
    entries: HashMap<String, VectorEntry>,
}

impl VectorDatabase {
    fn load(path: &PathBuf) -> Self {
        if path.exists() {
            if let Ok(content) = fs::read_to_string(path) {
                if let Ok(db) = serde_json::from_str(&content) {
                    return db;
                }
            }
        }
        VectorDatabase::default()
    }

    fn save(&self, path: &PathBuf) -> Result<(), String> {
        let json = serde_json::to_string_pretty(self)
            .map_err(|e| format!("Failed to serialize: {}", e))?;
        fs::write(path, json)
            .map_err(|e| format!("Failed to write file: {}", e))?;
        Ok(())
    }

    fn insert(&mut self, entry: VectorEntry) {
        self.entries.insert(entry.id.clone(), entry);
    }

    fn delete(&mut self, id: &str) -> Option<VectorEntry> {
        self.entries.remove(id)
    }

    fn get(&self, id: &str) -> Option<&VectorEntry> {
        self.entries.get(id)
    }

    fn list(&self) -> Vec<&VectorEntry> {
        self.entries.values().collect()
    }

    fn search(&self, query_vector: &[f64], top_k: usize) -> Vec<(&VectorEntry, f64)> {
        let mut results: Vec<_> = self.entries
            .values()
            .map(|entry| {
                let similarity = cosine_similarity(&entry.vector, query_vector);
                (entry, similarity)
            })
            .collect();

        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        results.truncate(top_k);
        results
    }
}

fn cosine_similarity(a: &[f64], b: &[f64]) -> f64 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }

    let dot_product: f64 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let magnitude_a: f64 = a.iter().map(|x| x * x).sum::<f64>().sqrt();
    let magnitude_b: f64 = b.iter().map(|x| x * x).sum::<f64>().sqrt();

    if magnitude_a == 0.0 || magnitude_b == 0.0 {
        return 0.0;
    }

    dot_product / (magnitude_a * magnitude_b)
}

struct McpServer {
    server_name: String,
    server_version: String,
    http_client: Client,
    openai_api_key: Option<String>,
    vector_db: Arc<RwLock<VectorDatabase>>,
    vector_db_path: PathBuf,
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

        // Load vector database
        let vector_db_path = PathBuf::from("vector_db.json");
        let vector_db = VectorDatabase::load(&vector_db_path);
        eprintln!("Vector database loaded: {} entries", vector_db.entries.len());

        McpServer {
            server_name: "rust-mcp-server".to_string(),
            server_version: "0.2.0".to_string(),
            http_client: Client::new(),
            openai_api_key,
            vector_db: Arc::new(RwLock::new(vector_db)),
            vector_db_path,
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
            // Vector Database Tools
            Tool {
                name: "vector_store".to_string(),
                description: "Store text with its vector embedding in the database. Automatically generates embedding using OpenAI API.".to_string(),
                input_schema: json!({
                    "type": "object",
                    "properties": {
                        "id": {
                            "type": "string",
                            "description": "Unique identifier for the entry (auto-generated if not provided)"
                        },
                        "text": {
                            "type": "string",
                            "description": "The text to store and embed"
                        },
                        "metadata": {
                            "type": "object",
                            "description": "Optional metadata to store with the entry"
                        }
                    },
                    "required": ["text"]
                }),
            },
            Tool {
                name: "vector_search".to_string(),
                description: "Search for similar entries in the vector database using semantic similarity.".to_string(),
                input_schema: json!({
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The search query text"
                        },
                        "top_k": {
                            "type": "integer",
                            "description": "Number of results to return (default: 5)",
                            "default": 5
                        }
                    },
                    "required": ["query"]
                }),
            },
            Tool {
                name: "vector_get".to_string(),
                description: "Get a specific entry from the vector database by ID.".to_string(),
                input_schema: json!({
                    "type": "object",
                    "properties": {
                        "id": {
                            "type": "string",
                            "description": "The ID of the entry to retrieve"
                        }
                    },
                    "required": ["id"]
                }),
            },
            Tool {
                name: "vector_delete".to_string(),
                description: "Delete an entry from the vector database by ID.".to_string(),
                input_schema: json!({
                    "type": "object",
                    "properties": {
                        "id": {
                            "type": "string",
                            "description": "The ID of the entry to delete"
                        }
                    },
                    "required": ["id"]
                }),
            },
            Tool {
                name: "vector_list".to_string(),
                description: "List all entries in the vector database.".to_string(),
                input_schema: json!({
                    "type": "object",
                    "properties": {
                        "limit": {
                            "type": "integer",
                            "description": "Maximum number of entries to return (default: 100)",
                            "default": 100
                        }
                    }
                }),
            },
            // File Operations
            Tool {
                name: "read_file".to_string(),
                description: "Read the contents of a local text file.".to_string(),
                input_schema: json!({
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "Path to the text file to read"
                        }
                    },
                    "required": ["path"]
                }),
            },
            Tool {
                name: "load_file_to_db".to_string(),
                description: "Read a text file and store its content in the vector database. Can split by lines or paragraphs.".to_string(),
                input_schema: json!({
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "Path to the text file to load"
                        },
                        "split_mode": {
                            "type": "string",
                            "description": "How to split the file: 'none' (whole file), 'lines' (by line), 'paragraphs' (by empty lines)",
                            "enum": ["none", "lines", "paragraphs"],
                            "default": "none"
                        },
                        "metadata": {
                            "type": "object",
                            "description": "Optional metadata to attach to all entries"
                        }
                    },
                    "required": ["path"]
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
            "vector_store" => self.tool_vector_store(arguments),
            "vector_search" => self.tool_vector_search(arguments),
            "vector_get" => self.tool_vector_get(arguments),
            "vector_delete" => self.tool_vector_delete(arguments),
            "vector_list" => self.tool_vector_list(arguments),
            "read_file" => self.tool_read_file(arguments),
            "load_file_to_db" => self.tool_load_file_to_db(arguments),
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

    // Vector Database Tools

    fn tool_vector_store(&self, args: &Value) -> Result<String, String> {
        let text = args
            .get("text")
            .and_then(|v| v.as_str())
            .ok_or("Missing 'text' argument")?;

        let id = args
            .get("id")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string())
            .unwrap_or_else(|| generate_id());

        let metadata = args.get("metadata").cloned();

        // Generate embedding
        let vector = match self.get_openai_embedding(text, "text-embedding-3-small") {
            Ok((vec, _)) => vec,
            Err(_) => self.generate_hash_vector(text, 128),
        };

        let created_at = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();

        let entry = VectorEntry {
            id: id.clone(),
            text: text.to_string(),
            vector,
            metadata,
            created_at,
        };

        // Store in database
        {
            let mut db = self.vector_db.write().map_err(|e| e.to_string())?;
            db.insert(entry);
            db.save(&self.vector_db_path)?;
        }

        let result = json!({
            "success": true,
            "id": id,
            "message": format!("Stored entry with ID: {}", id)
        });
        Ok(serde_json::to_string_pretty(&result).unwrap())
    }

    fn tool_vector_search(&self, args: &Value) -> Result<String, String> {
        let query = args
            .get("query")
            .and_then(|v| v.as_str())
            .ok_or("Missing 'query' argument")?;

        let top_k = args
            .get("top_k")
            .and_then(|v| v.as_u64())
            .unwrap_or(5) as usize;

        // Generate query embedding
        let query_vector = match self.get_openai_embedding(query, "text-embedding-3-small") {
            Ok((vec, _)) => vec,
            Err(_) => self.generate_hash_vector(query, 128),
        };

        // Search in database
        let db = self.vector_db.read().map_err(|e| e.to_string())?;
        let results = db.search(&query_vector, top_k);

        let results_json: Vec<Value> = results
            .iter()
            .map(|(entry, similarity)| {
                json!({
                    "id": entry.id,
                    "text": entry.text,
                    "similarity": (similarity * 10000.0).round() / 10000.0,
                    "metadata": entry.metadata,
                    "created_at": entry.created_at
                })
            })
            .collect();

        let result = json!({
            "query": query,
            "count": results_json.len(),
            "results": results_json
        });
        Ok(serde_json::to_string_pretty(&result).unwrap())
    }

    fn tool_vector_get(&self, args: &Value) -> Result<String, String> {
        let id = args
            .get("id")
            .and_then(|v| v.as_str())
            .ok_or("Missing 'id' argument")?;

        let db = self.vector_db.read().map_err(|e| e.to_string())?;

        match db.get(id) {
            Some(entry) => {
                let result = json!({
                    "found": true,
                    "id": entry.id,
                    "text": entry.text,
                    "dimensions": entry.vector.len(),
                    "metadata": entry.metadata,
                    "created_at": entry.created_at
                });
                Ok(serde_json::to_string_pretty(&result).unwrap())
            }
            None => {
                let result = json!({
                    "found": false,
                    "message": format!("Entry with ID '{}' not found", id)
                });
                Ok(serde_json::to_string_pretty(&result).unwrap())
            }
        }
    }

    fn tool_vector_delete(&self, args: &Value) -> Result<String, String> {
        let id = args
            .get("id")
            .and_then(|v| v.as_str())
            .ok_or("Missing 'id' argument")?;

        let mut db = self.vector_db.write().map_err(|e| e.to_string())?;

        match db.delete(id) {
            Some(entry) => {
                db.save(&self.vector_db_path)?;
                let result = json!({
                    "success": true,
                    "deleted_id": entry.id,
                    "deleted_text": entry.text,
                    "message": format!("Deleted entry with ID: {}", id)
                });
                Ok(serde_json::to_string_pretty(&result).unwrap())
            }
            None => {
                let result = json!({
                    "success": false,
                    "message": format!("Entry with ID '{}' not found", id)
                });
                Ok(serde_json::to_string_pretty(&result).unwrap())
            }
        }
    }

    fn tool_vector_list(&self, args: &Value) -> Result<String, String> {
        let limit = args
            .get("limit")
            .and_then(|v| v.as_u64())
            .unwrap_or(100) as usize;

        let db = self.vector_db.read().map_err(|e| e.to_string())?;
        let entries = db.list();

        let entries_json: Vec<Value> = entries
            .iter()
            .take(limit)
            .map(|entry| {
                json!({
                    "id": entry.id,
                    "text": entry.text,
                    "dimensions": entry.vector.len(),
                    "metadata": entry.metadata,
                    "created_at": entry.created_at
                })
            })
            .collect();

        let result = json!({
            "total_count": entries.len(),
            "returned_count": entries_json.len(),
            "entries": entries_json
        });
        Ok(serde_json::to_string_pretty(&result).unwrap())
    }

    // File Operations

    fn tool_read_file(&self, args: &Value) -> Result<String, String> {
        let path = args
            .get("path")
            .and_then(|v| v.as_str())
            .ok_or("Missing 'path' argument")?;

        let file_path = PathBuf::from(path);

        if !file_path.exists() {
            return Err(format!("File not found: {}", path));
        }

        let content = fs::read_to_string(&file_path)
            .map_err(|e| format!("Failed to read file: {}", e))?;

        let result = json!({
            "path": path,
            "size_bytes": content.len(),
            "lines": content.lines().count(),
            "content": content
        });
        Ok(serde_json::to_string_pretty(&result).unwrap())
    }

    fn tool_load_file_to_db(&self, args: &Value) -> Result<String, String> {
        let path = args
            .get("path")
            .and_then(|v| v.as_str())
            .ok_or("Missing 'path' argument")?;

        let split_mode = args
            .get("split_mode")
            .and_then(|v| v.as_str())
            .unwrap_or("none");

        let base_metadata = args.get("metadata").cloned();

        let file_path = PathBuf::from(path);

        if !file_path.exists() {
            return Err(format!("File not found: {}", path));
        }

        let content = fs::read_to_string(&file_path)
            .map_err(|e| format!("Failed to read file: {}", e))?;

        // Split content based on mode
        let chunks: Vec<String> = match split_mode {
            "lines" => content
                .lines()
                .filter(|line| !line.trim().is_empty())
                .map(|s| s.to_string())
                .collect(),
            "paragraphs" => content
                .split("\n\n")
                .filter(|para| !para.trim().is_empty())
                .map(|s| s.trim().to_string())
                .collect(),
            _ => vec![content], // "none" - whole file
        };

        let file_name = file_path
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("unknown");

        let mut stored_ids = Vec::new();
        let created_at = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();

        {
            let mut db = self.vector_db.write().map_err(|e| e.to_string())?;

            for (i, chunk) in chunks.iter().enumerate() {
                if chunk.trim().is_empty() {
                    continue;
                }

                // Generate embedding
                let vector = match self.get_openai_embedding(chunk, "text-embedding-3-small") {
                    Ok((vec, _)) => vec,
                    Err(_) => self.generate_hash_vector(chunk, 128),
                };

                // Build metadata
                let mut metadata = base_metadata.clone().unwrap_or(json!({}));
                if let Some(obj) = metadata.as_object_mut() {
                    obj.insert("source_file".to_string(), json!(file_name));
                    obj.insert("chunk_index".to_string(), json!(i));
                    if split_mode != "none" {
                        obj.insert("split_mode".to_string(), json!(split_mode));
                    }
                }

                let id = generate_id();
                let entry = VectorEntry {
                    id: id.clone(),
                    text: chunk.clone(),
                    vector,
                    metadata: Some(metadata),
                    created_at,
                };

                db.insert(entry);
                stored_ids.push(id);
            }

            db.save(&self.vector_db_path)?;
        }

        let result = json!({
            "success": true,
            "file": file_name,
            "split_mode": split_mode,
            "chunks_stored": stored_ids.len(),
            "ids": stored_ids
        });
        Ok(serde_json::to_string_pretty(&result).unwrap())
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

fn generate_id() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let timestamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    format!("vec_{:x}", timestamp)
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
