//! Integration tests for the minimal MCP stdio server (Task 6.2).
//!
//! Exercises the server in-process (no real stdio): JSON-RPC requests go
//! straight through `McpServer::handle_request`, exactly as the stdio loop
//! in `src/bin/synaptic_mcp.rs` would deliver them.

#![cfg(feature = "mcp")]
#![allow(clippy::unwrap_used, clippy::panic)]

use serde_json::{json, Value};
use synaptic::mcp::{JsonRpcRequest, McpServer};
use synaptic::{AgentMemory, MemoryConfig};

async fn server() -> McpServer {
    let memory = AgentMemory::new(MemoryConfig::default())
        .await
        .expect("default AgentMemory construction must succeed");
    McpServer::new(memory)
}

fn request(id: u64, method: &str, params: Value) -> JsonRpcRequest {
    serde_json::from_value(json!({
        "jsonrpc": "2.0",
        "id": id,
        "method": method,
        "params": params,
    }))
    .expect("request literal is valid JSON-RPC")
}

/// Extract the concatenated text content of a successful tool call result.
fn tool_text(response: &Value) -> String {
    assert!(
        response.get("error").is_none(),
        "expected success, got error: {response}"
    );
    let result = &response["result"];
    assert_ne!(
        result.get("isError").and_then(Value::as_bool),
        Some(true),
        "tool result flagged isError: {result}"
    );
    result["content"]
        .as_array()
        .expect("tool result carries a content array")
        .iter()
        .map(|c| c["text"].as_str().unwrap_or_default())
        .collect::<Vec<_>>()
        .join("\n")
}

#[tokio::test]
async fn initialize_reports_server_info_and_tool_capability() {
    let server = server().await;
    let resp = server
        .handle_request(request(1, "initialize", json!({})))
        .await
        .expect("request with id gets a response");
    let resp = serde_json::to_value(&resp).unwrap();
    assert_eq!(resp["jsonrpc"], "2.0");
    assert_eq!(resp["id"], 1);
    assert!(resp["result"]["serverInfo"]["name"].is_string());
    assert!(resp["result"]["capabilities"]["tools"].is_object());
    assert!(resp["result"]["protocolVersion"].is_string());
}

#[tokio::test]
async fn tools_list_exposes_the_four_memory_tools() {
    let server = server().await;
    let resp = server
        .handle_request(request(1, "tools/list", json!({})))
        .await
        .expect("request with id gets a response");
    let resp = serde_json::to_value(&resp).unwrap();
    let tools = resp["result"]["tools"].as_array().expect("tools array");
    let names: Vec<&str> = tools
        .iter()
        .map(|t| t["name"].as_str().expect("tool has a name"))
        .collect();
    assert_eq!(tools.len(), 4, "exactly four tools: {names:?}");
    for expected in ["remember", "recall", "reflect", "forget"] {
        assert!(names.contains(&expected), "missing tool {expected}");
    }
    // Every tool must publish an input schema.
    for tool in tools {
        assert!(
            tool["inputSchema"]["type"] == "object",
            "tool {} lacks an object inputSchema",
            tool["name"]
        );
    }
}

#[tokio::test]
async fn remember_then_recall_round_trips_stored_content() {
    let server = server().await;

    let resp = server
        .handle_request(request(
            1,
            "tools/call",
            json!({"name": "remember", "arguments": {"content": "the deploy password is xyzzy"}}),
        ))
        .await
        .expect("response");
    let stored = tool_text(&serde_json::to_value(&resp).unwrap());
    assert!(stored.contains("key"), "remember reports the stored key");

    let resp = server
        .handle_request(request(
            2,
            "tools/call",
            json!({"name": "recall", "arguments": {"query": "deploy password", "limit": 5}}),
        ))
        .await
        .expect("response");
    let recalled = tool_text(&serde_json::to_value(&resp).unwrap());
    assert!(
        recalled.contains("xyzzy"),
        "recall must return the remembered content, got: {recalled}"
    );
}

#[tokio::test]
async fn recall_with_as_of_before_ingestion_excludes_the_memory() {
    let server = server().await;
    server
        .handle_request(request(
            1,
            "tools/call",
            json!({"name": "remember", "arguments": {"content": "quarterly revenue was 42 million"}}),
        ))
        .await
        .expect("response");

    // as_of one day before the memory was ingested: bi-temporally invalid.
    let before = (chrono::Utc::now() - chrono::Duration::days(1)).to_rfc3339();
    let resp = server
        .handle_request(request(
            2,
            "tools/call",
            json!({"name": "recall", "arguments": {"query": "quarterly revenue", "as_of": before}}),
        ))
        .await
        .expect("response");
    let recalled = tool_text(&serde_json::to_value(&resp).unwrap());
    assert!(
        !recalled.contains("42 million"),
        "memory not yet valid at as_of must be excluded, got: {recalled}"
    );

    // as_of now: valid, must come back.
    let now = chrono::Utc::now().to_rfc3339();
    let resp = server
        .handle_request(request(
            3,
            "tools/call",
            json!({"name": "recall", "arguments": {"query": "quarterly revenue", "as_of": now}}),
        ))
        .await
        .expect("response");
    let recalled = tool_text(&serde_json::to_value(&resp).unwrap());
    assert!(
        recalled.contains("42 million"),
        "memory valid at as_of must be returned, got: {recalled}"
    );
}

#[tokio::test]
async fn reflect_and_forget_tools_dispatch_and_return_structured_results() {
    let server = server().await;
    let resp = server
        .handle_request(request(
            1,
            "tools/call",
            json!({"name": "reflect", "arguments": {}}),
        ))
        .await
        .expect("response");
    let text = tool_text(&serde_json::to_value(&resp).unwrap());
    assert!(
        text.contains("insights"),
        "reflect reports insights: {text}"
    );

    let resp = server
        .handle_request(request(
            2,
            "tools/call",
            json!({"name": "forget", "arguments": {"retention_floor": 0.0}}),
        ))
        .await
        .expect("response");
    let text = tool_text(&serde_json::to_value(&resp).unwrap());
    assert!(
        text.contains("examined"),
        "forget returns a ForgetReport: {text}"
    );
}

#[tokio::test]
async fn unknown_method_and_unknown_tool_fail_closed_with_jsonrpc_errors() {
    let server = server().await;

    let resp = server
        .handle_request(request(1, "no/such/method", json!({})))
        .await
        .expect("response");
    let resp = serde_json::to_value(&resp).unwrap();
    assert!(resp.get("result").is_none());
    assert_eq!(resp["error"]["code"], -32601, "method not found: {resp}");

    let resp = server
        .handle_request(request(
            2,
            "tools/call",
            json!({"name": "hallucinate", "arguments": {}}),
        ))
        .await
        .expect("response");
    let resp = serde_json::to_value(&resp).unwrap();
    assert!(
        resp.get("result").is_none(),
        "unknown tool must not fake success"
    );
    assert_eq!(resp["error"]["code"], -32602, "invalid params: {resp}");

    // Bad params on a known tool (recall without query) also fail closed.
    let resp = server
        .handle_request(request(
            3,
            "tools/call",
            json!({"name": "recall", "arguments": {}}),
        ))
        .await
        .expect("response");
    let resp = serde_json::to_value(&resp).unwrap();
    assert!(resp.get("result").is_none());
    assert_eq!(resp["error"]["code"], -32602, "missing query: {resp}");
}

#[tokio::test]
async fn notifications_get_no_response() {
    let server = server().await;
    let req: JsonRpcRequest = serde_json::from_value(json!({
        "jsonrpc": "2.0",
        "method": "notifications/initialized",
    }))
    .expect("notification literal is valid");
    assert!(server.handle_request(req).await.is_none());
}
