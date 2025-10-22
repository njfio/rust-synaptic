//! Context assembly for AI agents.
//!
//! This module provides turnkey retrieval and synthesis of memory context
//! for LLM agents, combining relevant memories, graph relationships, and
//! temporal information into a formatted context window.
//!
//! # Features
//!
//! - **ContextBuilder**: Fluent API for assembling agent context
//! - **AgentContext**: Structured context with core, related, and temporal memories
//! - **LLM Formatting**: Output formatting for different LLM providers
//! - **Token Management**: Automatic truncation to fit context windows
//! - **Multi-Source**: Combines semantic search, graph traversal, and temporal queries
//!
//! # Examples
//!
//! ```rust,no_run
//! use synaptic::memory::context::{ContextBuilder, LlmFormat};
//! use synaptic::memory::AgentMemory;
//! use chrono::Duration;
//!
//! # async fn example(memory: &AgentMemory) -> Result<(), Box<dyn std::error::Error>> {
//! // Build context for an agent query
//! let context = ContextBuilder::new(memory)
//!     .with_relevant_memories("user preferences", 5)?
//!     .with_graph_neighbors(2, None)?
//!     .with_temporal_slice(Duration::hours(24))?
//!     .with_summaries()?
//!     .build()
//!     .await?;
//!
//! // Format for OpenAI
//! let formatted = context.format(LlmFormat::OpenAI, 4000)?;
//! println!("{}", formatted);
//! # Ok(())
//! # }
//! ```

pub mod builder;

pub use builder::{
    AgentContext, ContextBuilder, ContextError, ContextResult, LlmFormat, MemorySection,
    RelationshipFilter, TemporalRange,
};
