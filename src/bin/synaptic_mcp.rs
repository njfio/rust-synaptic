//! `synaptic_mcp` — MCP stdio server for Synaptic agent memory.
//!
//! Speaks line-delimited JSON-RPC 2.0 on stdin/stdout (the MCP stdio
//! transport) and exposes the `remember`, `recall`, `reflect` and `forget`
//! tools over a single [`synaptic::AgentMemory`]. All logging goes to stderr
//! via `tracing` so stdout stays a clean protocol channel.
//!
//! Built only with `--features mcp` (`required-features` in Cargo.toml).

use synaptic::mcp::{run_stdio, McpServer};
use synaptic::{AgentMemory, MemoryConfig};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt()
        .with_writer(std::io::stderr)
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .init();

    let memory = AgentMemory::new(MemoryConfig::default()).await?;
    let server = McpServer::new(memory);
    tracing::info!("synaptic MCP server ready on stdio");
    run_stdio(&server).await?;
    Ok(())
}
