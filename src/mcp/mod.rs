//! Minimal Model Context Protocol (MCP) server over stdio JSON-RPC 2.0.
//!
//! Exposes four agent-memory tools backed by a shared [`AgentMemory`]:
//! `remember` (store), `recall` (search, with optional bi-temporal `as_of`),
//! `reflect` (insight synthesis) and `forget` (retention-floor eviction).
//!
//! The protocol layer is deliberately small and dependency-free (plain
//! `serde_json` over the JSON-RPC 2.0 framing MCP uses); it implements the
//! `initialize` / `tools/list` / `tools/call` subset with a declared `tools`
//! capability. Request handling is factored into
//! [`McpServer::handle_request`] so the whole protocol surface is testable
//! in-process; the `synaptic_mcp` binary merely wires stdin/stdout to it via
//! [`run_stdio`]. Unknown methods, unknown tools and malformed parameters
//! fail closed with proper JSON-RPC error objects — never a fabricated
//! success.

use crate::error::{MemoryError, Result};
use crate::memory::forgetting::ForgettingPolicy;
use crate::AgentMemory;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::sync::Arc;
use tokio::sync::Mutex;

/// MCP protocol revision this server implements.
pub const PROTOCOL_VERSION: &str = "2024-11-05";

/// JSON-RPC 2.0 error code: invalid JSON was received.
pub const PARSE_ERROR: i64 = -32700;
/// JSON-RPC 2.0 error code: the request object is not a valid request.
pub const INVALID_REQUEST: i64 = -32600;
/// JSON-RPC 2.0 error code: the method does not exist.
pub const METHOD_NOT_FOUND: i64 = -32601;
/// JSON-RPC 2.0 error code: invalid method parameters (also used for
/// unknown tool names, per the MCP tools/call convention).
pub const INVALID_PARAMS: i64 = -32602;
/// JSON-RPC 2.0 error code: internal server error.
pub const INTERNAL_ERROR: i64 = -32603;

/// Default number of results `recall` returns when `limit` is omitted.
const DEFAULT_RECALL_LIMIT: usize = 10;

/// An incoming JSON-RPC 2.0 request or notification (no `id`).
#[derive(Debug, Clone, Deserialize)]
pub struct JsonRpcRequest {
    /// Protocol marker; must be exactly `"2.0"`.
    pub jsonrpc: String,
    /// Request id; absent for notifications, which receive no response.
    #[serde(default)]
    pub id: Option<Value>,
    /// Method name, e.g. `initialize`, `tools/list`, `tools/call`.
    pub method: String,
    /// Method parameters, if any.
    #[serde(default)]
    pub params: Option<Value>,
}

/// A JSON-RPC 2.0 error object.
#[derive(Debug, Clone, Serialize)]
pub struct JsonRpcError {
    /// Numeric error code (`-32700..-32603` for protocol errors).
    pub code: i64,
    /// Human-readable error description.
    pub message: String,
    /// Optional structured error detail.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub data: Option<Value>,
}

/// An outgoing JSON-RPC 2.0 response: exactly one of `result`/`error` is set.
#[derive(Debug, Clone, Serialize)]
pub struct JsonRpcResponse {
    /// Protocol marker, always `"2.0"`.
    pub jsonrpc: String,
    /// Mirrors the request id (`null` when the request id was unparseable).
    pub id: Value,
    /// Success payload; absent on error.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub result: Option<Value>,
    /// Error payload; absent on success.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<JsonRpcError>,
}

impl JsonRpcResponse {
    /// Build a success response mirroring `id`.
    pub fn success(id: Value, result: Value) -> Self {
        Self {
            jsonrpc: "2.0".to_string(),
            id,
            result: Some(result),
            error: None,
        }
    }

    /// Build an error response mirroring `id`.
    pub fn failure(id: Value, code: i64, message: impl Into<String>) -> Self {
        Self {
            jsonrpc: "2.0".to_string(),
            id,
            result: None,
            error: Some(JsonRpcError {
                code,
                message: message.into(),
                data: None,
            }),
        }
    }
}

/// Minimal MCP server dispatching the four memory tools to a shared
/// [`AgentMemory`] behind a mutex (tool calls that mutate — `remember`,
/// `reflect`, `forget` — take the same lock as reads, serializing access).
pub struct McpServer {
    memory: Arc<Mutex<AgentMemory>>,
}

impl McpServer {
    /// Wrap an [`AgentMemory`] in a server.
    pub fn new(memory: AgentMemory) -> Self {
        Self {
            memory: Arc::new(Mutex::new(memory)),
        }
    }

    /// Handle one JSON-RPC request. Returns `None` for notifications
    /// (requests without an `id`), which per JSON-RPC receive no response.
    pub async fn handle_request(&self, req: JsonRpcRequest) -> Option<JsonRpcResponse> {
        let id = req.id.clone()?;
        if req.jsonrpc != "2.0" {
            return Some(JsonRpcResponse::failure(
                id,
                INVALID_REQUEST,
                format!("unsupported jsonrpc version {:?}", req.jsonrpc),
            ));
        }
        let response = match req.method.as_str() {
            "initialize" => JsonRpcResponse::success(id, initialize_result()),
            "ping" => JsonRpcResponse::success(id, json!({})),
            "tools/list" => JsonRpcResponse::success(id, json!({ "tools": tool_definitions() })),
            "tools/call" => self.handle_tools_call(id, req.params).await,
            other => {
                JsonRpcResponse::failure(id, METHOD_NOT_FOUND, format!("method not found: {other}"))
            }
        };
        Some(response)
    }

    /// Dispatch `tools/call` to the named tool, failing closed on unknown
    /// tools or malformed arguments.
    async fn handle_tools_call(&self, id: Value, params: Option<Value>) -> JsonRpcResponse {
        let params = match params {
            Some(Value::Object(map)) => map,
            _ => {
                return JsonRpcResponse::failure(
                    id,
                    INVALID_PARAMS,
                    "tools/call requires an object with 'name' and 'arguments'",
                )
            }
        };
        let Some(name) = params.get("name").and_then(Value::as_str) else {
            return JsonRpcResponse::failure(id, INVALID_PARAMS, "missing tool 'name' (string)");
        };
        let args = match params.get("arguments") {
            None | Some(Value::Null) => serde_json::Map::new(),
            Some(Value::Object(map)) => map.clone(),
            Some(_) => {
                return JsonRpcResponse::failure(
                    id,
                    INVALID_PARAMS,
                    "'arguments' must be an object",
                )
            }
        };

        let outcome = match name {
            "remember" => self.tool_remember(&args).await,
            "recall" => self.tool_recall(&args).await,
            "reflect" => self.tool_reflect(&args).await,
            "forget" => self.tool_forget(&args).await,
            other => Err(ToolError::InvalidParams(format!("unknown tool: {other}"))),
        };

        match outcome {
            Ok(text) => JsonRpcResponse::success(
                id,
                json!({ "content": [{ "type": "text", "text": text }], "isError": false }),
            ),
            Err(ToolError::InvalidParams(message)) => {
                JsonRpcResponse::failure(id, INVALID_PARAMS, message)
            }
            Err(ToolError::Execution(err)) => {
                // Tool-execution failure: per MCP this is a tool result with
                // isError=true (the protocol call itself succeeded).
                tracing::warn!(tool = name, error = %err, "MCP tool execution failed");
                JsonRpcResponse::success(
                    id,
                    json!({
                        "content": [{ "type": "text", "text": format!("tool '{name}' failed: {err}") }],
                        "isError": true,
                    }),
                )
            }
        }
    }

    /// `remember{content, metadata?}` → [`AgentMemory::store_with_report`].
    ///
    /// Supported `metadata` fields: `key` (string) to choose the memory key.
    /// Any other metadata field is rejected rather than silently dropped.
    async fn tool_remember(&self, args: &serde_json::Map<String, Value>) -> ToolResult {
        let content = require_str(args, "content")?;
        let mut key: Option<String> = None;
        if let Some(metadata) = args.get("metadata") {
            let Value::Object(map) = metadata else {
                return Err(ToolError::InvalidParams(
                    "'metadata' must be an object".to_string(),
                ));
            };
            for (field, value) in map {
                match (field.as_str(), value) {
                    ("key", Value::String(k)) => key = Some(k.clone()),
                    ("key", _) => {
                        return Err(ToolError::InvalidParams(
                            "metadata.key must be a string".to_string(),
                        ))
                    }
                    (other, _) => {
                        return Err(ToolError::InvalidParams(format!(
                            "unsupported metadata field '{other}' (supported: key)"
                        )))
                    }
                }
            }
        }
        let key = key.unwrap_or_else(|| format!("mcp-{}", uuid::Uuid::new_v4()));

        let mut memory = self.memory.lock().await;
        let degradations = memory
            .store_with_report(&key, content)
            .await
            .map_err(ToolError::Execution)?;
        let degraded: Vec<&str> = [
            ("temporal", &degradations.temporal),
            ("analytics", &degradations.analytics),
            ("knowledge_graph", &degradations.knowledge_graph),
            ("advanced_management", &degradations.advanced_management),
            ("embeddings", &degradations.embeddings),
            ("reasoning", &degradations.reasoning),
        ]
        .into_iter()
        .filter_map(|(name, failure)| failure.as_ref().map(|_| name))
        .collect();
        Ok(json!({ "key": key, "degraded_subsystems": degraded }).to_string())
    }

    /// `recall{query, limit?, as_of?}` → [`AgentMemory::search`], filtered by
    /// bi-temporal validity when `as_of` is given: a result survives only if
    /// its backing knowledge-graph node is valid at `as_of` (event time and
    /// system time, via `Node::is_valid_at`), falling back to
    /// `created_at <= as_of` for memories without a graph node.
    async fn tool_recall(&self, args: &serde_json::Map<String, Value>) -> ToolResult {
        let query = require_str(args, "query")?;
        let limit = match args.get("limit") {
            None | Some(Value::Null) => DEFAULT_RECALL_LIMIT,
            Some(v) => v
                .as_u64()
                .and_then(|n| usize::try_from(n).ok())
                .filter(|n| *n >= 1)
                .ok_or_else(|| {
                    ToolError::InvalidParams("'limit' must be a positive integer".to_string())
                })?,
        };
        let as_of: Option<DateTime<Utc>> = match args.get("as_of") {
            None | Some(Value::Null) => None,
            Some(Value::String(s)) => Some(
                DateTime::parse_from_rfc3339(s)
                    .map(|t| t.with_timezone(&Utc))
                    .map_err(|e| {
                        ToolError::InvalidParams(format!("'as_of' is not RFC 3339: {e}"))
                    })?,
            ),
            Some(_) => {
                return Err(ToolError::InvalidParams(
                    "'as_of' must be an RFC 3339 string".to_string(),
                ))
            }
        };

        let memory = self.memory.lock().await;
        let mut fragments = memory
            .search(query, limit)
            .await
            .map_err(ToolError::Execution)?;

        if let Some(at) = as_of {
            let kg = memory.knowledge_graph_handle();
            let mut kept = Vec::with_capacity(fragments.len());
            for fragment in fragments {
                let node_validity = match &kg {
                    Some(handle) => handle
                        .read()
                        .await
                        .memory_node_valid_at(&fragment.entry.key, at)
                        .await
                        .map_err(ToolError::Execution)?,
                    None => None,
                };
                // Node validity is authoritative (both temporal axes); a
                // memory with no graph node falls back to ingestion time.
                let valid = node_validity.unwrap_or(fragment.entry.created_at() <= at);
                if valid {
                    kept.push(fragment);
                }
            }
            fragments = kept;
        }

        let results: Vec<Value> = fragments
            .iter()
            .map(|f| {
                json!({
                    "key": f.entry.key,
                    "content": f.entry.value,
                    "relevance": f.relevance_score,
                })
            })
            .collect();
        Ok(json!({ "results": results }).to_string())
    }

    /// `reflect{}` → [`AgentMemory::reflect`].
    async fn tool_reflect(&self, args: &serde_json::Map<String, Value>) -> ToolResult {
        if !args.is_empty() {
            return Err(ToolError::InvalidParams(
                "reflect takes no arguments".to_string(),
            ));
        }
        let mut memory = self.memory.lock().await;
        let insights = memory.reflect().await.map_err(ToolError::Execution)?;
        let insights = serde_json::to_value(&insights).map_err(|e| {
            ToolError::Execution(MemoryError::unexpected(format!(
                "insights failed to serialize: {e}"
            )))
        })?;
        Ok(json!({ "insights": insights }).to_string())
    }

    /// `forget{retention_floor?}` → [`AgentMemory::forget`] with the default
    /// [`ForgettingPolicy`], optionally overriding its retention floor.
    async fn tool_forget(&self, args: &serde_json::Map<String, Value>) -> ToolResult {
        let mut policy = ForgettingPolicy::default();
        match args.get("retention_floor") {
            None | Some(Value::Null) => {}
            Some(v) => {
                policy.retention_floor = v.as_f64().filter(|f| f.is_finite()).ok_or_else(|| {
                    ToolError::InvalidParams(
                        "'retention_floor' must be a finite number".to_string(),
                    )
                })?;
            }
        }
        let mut memory = self.memory.lock().await;
        let report = memory.forget(policy).await.map_err(ToolError::Execution)?;
        let report = serde_json::to_value(&report).map_err(|e| {
            ToolError::Execution(MemoryError::unexpected(format!(
                "forget report failed to serialize: {e}"
            )))
        })?;
        Ok(report.to_string())
    }
}

/// A tool dispatch outcome: the success payload is the text content returned
/// to the MCP client.
type ToolResult = std::result::Result<String, ToolError>;

/// Tool failure modes, kept distinct so parameter mistakes surface as
/// JSON-RPC `-32602` while runtime failures surface as `isError` results.
enum ToolError {
    /// Malformed or missing arguments (protocol-level failure).
    InvalidParams(String),
    /// The underlying memory operation failed.
    Execution(MemoryError),
}

/// Extract a required non-empty string argument.
fn require_str<'a>(
    args: &'a serde_json::Map<String, Value>,
    field: &str,
) -> std::result::Result<&'a str, ToolError> {
    args.get(field)
        .and_then(Value::as_str)
        .filter(|s| !s.trim().is_empty())
        .ok_or_else(|| {
            ToolError::InvalidParams(format!("missing required string argument '{field}'"))
        })
}

/// `initialize` result: protocol version, server identity, tools capability.
fn initialize_result() -> Value {
    json!({
        "protocolVersion": PROTOCOL_VERSION,
        "serverInfo": {
            "name": "synaptic-mcp",
            "version": env!("CARGO_PKG_VERSION"),
        },
        "capabilities": { "tools": {} },
    })
}

/// The four tool definitions advertised by `tools/list`.
fn tool_definitions() -> Vec<Value> {
    vec![
        json!({
            "name": "remember",
            "description": "Store a memory. Optional metadata: { key: string } to choose the memory key (otherwise one is generated).",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "content": { "type": "string", "description": "The content to remember" },
                    "metadata": {
                        "type": "object",
                        "description": "Optional metadata; supported field: key (string)",
                        "properties": { "key": { "type": "string" } },
                        "additionalProperties": false
                    }
                },
                "required": ["content"]
            }
        }),
        json!({
            "name": "recall",
            "description": "Search memories by content similarity. Optional as_of (RFC 3339) restricts results to memories bi-temporally valid at that instant.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "query": { "type": "string", "description": "Search query" },
                    "limit": { "type": "number", "description": "Maximum results (default 10)" },
                    "as_of": { "type": "string", "format": "date-time", "description": "RFC 3339 instant for bi-temporal filtering" }
                },
                "required": ["query"]
            }
        }),
        json!({
            "name": "reflect",
            "description": "Cluster recently stored memories and synthesize insight memories with provenance links. Fires only once accumulated importance crosses the reflection threshold.",
            "inputSchema": { "type": "object", "properties": {}, "additionalProperties": false }
        }),
        json!({
            "name": "forget",
            "description": "Run a forgetting pass: demote or evict memories whose retained strength (decay * importance * recency) is below the retention floor.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "retention_floor": { "type": "number", "description": "Override the policy's retention floor" }
                },
                "additionalProperties": false
            }
        }),
    ]
}

/// Serve MCP over stdio: read line-delimited JSON-RPC from stdin, write one
/// response per line to stdout. Logs go to `tracing` (stderr in the binary),
/// never stdout. Returns when stdin reaches EOF.
pub async fn run_stdio(server: &McpServer) -> Result<()> {
    use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};

    let stdin = BufReader::new(tokio::io::stdin());
    let mut lines = stdin.lines();
    let mut stdout = tokio::io::stdout();

    while let Some(line) = lines
        .next_line()
        .await
        .map_err(|e| MemoryError::unexpected(format!("failed reading stdin: {e}")))?
    {
        if line.trim().is_empty() {
            continue;
        }
        let response = match serde_json::from_str::<JsonRpcRequest>(&line) {
            Ok(request) => server.handle_request(request).await,
            Err(e) => {
                tracing::warn!(error = %e, "rejecting unparseable JSON-RPC line");
                Some(JsonRpcResponse::failure(
                    Value::Null,
                    PARSE_ERROR,
                    format!("parse error: {e}"),
                ))
            }
        };
        if let Some(response) = response {
            let mut bytes = serde_json::to_vec(&response)
                .map_err(|e| MemoryError::unexpected(format!("response serialization: {e}")))?;
            bytes.push(b'\n');
            stdout
                .write_all(&bytes)
                .await
                .map_err(|e| MemoryError::unexpected(format!("failed writing stdout: {e}")))?;
            stdout
                .flush()
                .await
                .map_err(|e| MemoryError::unexpected(format!("failed flushing stdout: {e}")))?;
        }
    }
    tracing::info!("stdin closed; MCP server shutting down");
    Ok(())
}
