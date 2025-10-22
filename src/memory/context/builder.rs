//! Context builder for assembling agent context
//!
//! This module provides a high-level API for building context from memories,
//! optimized for consumption by language models.

use super::{AgentContext, ContextFormat, TokenCounter};
use crate::error::{MemoryError, Result};
use crate::memory::knowledge_graph::MemoryKnowledgeGraph;
use crate::memory::storage::Storage;
use crate::memory::retrieval::MemoryRetriever;
use crate::memory::types::{MemoryEntry, MemoryType};
use chrono::{DateTime, Utc, Duration};
use std::sync::Arc;
use uuid::Uuid;

/// Builder for assembling agent context
///
/// Provides a fluent API for gathering relevant memories and formatting
/// them for consumption by language models.
pub struct ContextBuilder {
    storage: Arc<dyn Storage>,
    retriever: Option<Arc<MemoryRetriever>>,
    graph: Option<Arc<MemoryKnowledgeGraph>>,

    // Query parameters
    query: Option<String>,
    search_limit: usize,
    include_graph_context: bool,
    graph_depth: usize,

    // Temporal parameters
    time_range: Option<(DateTime<Utc>, DateTime<Utc>)>,
    recency_bias: f64,

    // Memory type filters
    memory_types: Vec<MemoryType>,

    // Formatting parameters
    format: ContextFormat,
    max_tokens: Option<usize>,
    include_metadata: bool,
    include_summaries: bool,

    // Collected memories
    core_memories: Vec<MemoryEntry>,
    related_memories: Vec<(MemoryEntry, f64)>,
    temporal_memories: Vec<MemoryEntry>,
}

impl ContextBuilder {
    /// Create a new context builder
    pub fn new(storage: Arc<dyn Storage>) -> Self {
        Self {
            storage,
            retriever: None,
            graph: None,
            query: None,
            search_limit: 10,
            include_graph_context: false,
            graph_depth: 2,
            time_range: None,
            recency_bias: 0.0,
            memory_types: vec![],
            format: ContextFormat::Markdown,
            max_tokens: None,
            include_metadata: false,
            include_summaries: true,
            core_memories: Vec::new(),
            related_memories: Vec::new(),
            temporal_memories: Vec::new(),
        }
    }

    /// Set the memory retriever
    pub fn with_retriever(mut self, retriever: Arc<MemoryRetriever>) -> Self {
        self.retriever = Some(retriever);
        self
    }

    /// Set the knowledge graph
    pub fn with_graph(mut self, graph: Arc<MemoryKnowledgeGraph>) -> Self {
        self.graph = Some(graph);
        self
    }

    /// Add relevant memories based on a query
    pub async fn with_relevant_memories(mut self, query: &str, limit: usize) -> Result<Self> {
        self.query = Some(query.to_string());
        self.search_limit = limit;

        // Use retriever if available
        if let Some(retriever) = &self.retriever {
            let results = retriever.retrieve_relevant(query, limit).await?;
            self.core_memories.extend(results);
        } else {
            // Fallback to simple storage search
            let all_entries = self.storage.list_all().await?;
            let mut scored: Vec<(MemoryEntry, f64)> = all_entries
                .into_iter()
                .filter(|entry| entry.value.to_lowercase().contains(&query.to_lowercase()))
                .map(|entry| {
                    let score = self.simple_relevance_score(&entry, query);
                    (entry, score)
                })
                .collect();

            scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            scored.truncate(limit);

            self.core_memories.extend(scored.into_iter().map(|(entry, _)| entry));
        }

        Ok(self)
    }

    /// Add graph neighbors for collected memories
    pub async fn with_graph_neighbors(
        mut self,
        depth: usize,
        relationship_types: Option<Vec<String>>,
    ) -> Result<Self> {
        self.include_graph_context = true;
        self.graph_depth = depth;

        if let Some(graph) = &self.graph {
            for memory in &self.core_memories {
                let memory_id = memory.id().to_string();

                // Get neighbors at specified depth
                let neighbors = if let Some(ref types) = relationship_types {
                    graph.get_related_memories(&memory_id, Some(types.clone()))
                } else {
                    graph.get_related_memories(&memory_id, None)
                };

                // Add neighbors with their relationship strength
                for (neighbor_id, strength) in neighbors {
                    if let Ok(neighbor_uuid) = Uuid::parse_str(&neighbor_id) {
                        if let Ok(Some(neighbor_entry)) = self.storage.get(&neighbor_uuid.to_string()).await {
                            self.related_memories.push((neighbor_entry, strength as f64));
                        }
                    }
                }
            }

            // Sort by relationship strength and remove duplicates
            self.related_memories.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            self.related_memories.dedup_by_key(|(entry, _)| entry.id());
        }

        Ok(self)
    }

    /// Add memories from a specific time range
    pub async fn with_temporal_slice(
        mut self,
        start: DateTime<Utc>,
        end: DateTime<Utc>,
    ) -> Result<Self> {
        self.time_range = Some((start, end));

        let all_entries = self.storage.list_all().await?;
        self.temporal_memories = all_entries
            .into_iter()
            .filter(|entry| {
                entry.created_at >= start && entry.created_at <= end
            })
            .collect();

        Ok(self)
    }

    /// Add recent memories (last N hours/days)
    pub async fn with_recent_memories(self, hours: i64) -> Result<Self> {
        let end = Utc::now();
        let start = end - Duration::hours(hours);
        self.with_temporal_slice(start, end).await
    }

    /// Filter by memory types
    pub fn with_memory_types(mut self, types: Vec<MemoryType>) -> Self {
        self.memory_types = types;
        self
    }

    /// Set output format
    pub fn with_format(mut self, format: ContextFormat) -> Self {
        self.format = format;
        self
    }

    /// Set maximum token limit
    pub fn with_max_tokens(mut self, max_tokens: usize) -> Self {
        self.max_tokens = Some(max_tokens);
        self
    }

    /// Include metadata in output
    pub fn with_metadata(mut self, include: bool) -> Self {
        self.include_metadata = include;
        self
    }

    /// Include summaries in output
    pub fn with_summaries(mut self, include: bool) -> Self {
        self.include_summaries = include;
        self
    }

    /// Build the final context
    pub fn build(mut self) -> Result<AgentContext> {
        // Apply memory type filters
        if !self.memory_types.is_empty() {
            self.core_memories.retain(|m| self.memory_types.contains(&m.memory_type));
            self.related_memories.retain(|(m, _)| self.memory_types.contains(&m.memory_type));
            self.temporal_memories.retain(|m| self.memory_types.contains(&m.memory_type));
        }

        // Create the agent context
        let mut context = AgentContext {
            query: self.query.clone(),
            core_memories: self.core_memories.clone(),
            related_memories: self.related_memories.iter().map(|(m, s)| (m.clone(), *s)).collect(),
            temporal_memories: self.temporal_memories.clone(),
            summaries: Vec::new(),
            metadata: std::collections::HashMap::new(),
            format: self.format,
        };

        // Generate summaries if requested
        if self.include_summaries {
            context.summaries = vec![
                format!("Core memories: {} items", context.core_memories.len()),
                format!("Related memories: {} items", context.related_memories.len()),
                format!("Temporal memories: {} items", context.temporal_memories.len()),
            ];
        }

        // Add metadata
        if self.include_metadata {
            context.metadata.insert("search_limit".to_string(), self.search_limit.to_string());
            context.metadata.insert("graph_depth".to_string(), self.graph_depth.to_string());
            if let Some(max_tokens) = self.max_tokens {
                context.metadata.insert("max_tokens".to_string(), max_tokens.to_string());
            }
        }

        // Apply token limit if specified
        if let Some(max_tokens) = self.max_tokens {
            context = self.truncate_to_tokens(context, max_tokens)?;
        }

        Ok(context)
    }

    /// Simple relevance scoring (fallback when no retriever is available)
    fn simple_relevance_score(&self, entry: &MemoryEntry, query: &str) -> f64 {
        let query_lower = query.to_lowercase();
        let value_lower = entry.value.to_lowercase();

        // Count query term occurrences
        let mut score = 0.0;
        for term in query_lower.split_whitespace() {
            if value_lower.contains(term) {
                score += 1.0;
            }
        }

        // Normalize by entry length
        if !entry.value.is_empty() {
            score / entry.value.split_whitespace().count() as f64
        } else {
            0.0
        }
    }

    /// Truncate context to fit within token limit
    fn truncate_to_tokens(&self, mut context: AgentContext, max_tokens: usize) -> Result<AgentContext> {
        let counter = TokenCounter::new();

        // Calculate current token count
        let mut current_tokens = counter.count_context(&context);

        // If already within limit, return as-is
        if current_tokens <= max_tokens {
            return Ok(context);
        }

        // Truncate in priority order: temporal -> related -> core
        while current_tokens > max_tokens && !context.temporal_memories.is_empty() {
            context.temporal_memories.pop();
            current_tokens = counter.count_context(&context);
        }

        while current_tokens > max_tokens && !context.related_memories.is_empty() {
            context.related_memories.pop();
            current_tokens = counter.count_context(&context);
        }

        while current_tokens > max_tokens && !context.core_memories.is_empty() {
            context.core_memories.pop();
            current_tokens = counter.count_context(&context);
        }

        Ok(context)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::storage::MemoryStorage;
    use crate::memory::types::MemoryType;

    #[tokio::test]
    async fn test_context_builder_creation() {
        let storage = Arc::new(MemoryStorage::new());
        let builder = ContextBuilder::new(storage);

        assert_eq!(builder.search_limit, 10);
        assert!(!builder.include_graph_context);
    }

    #[tokio::test]
    async fn test_context_builder_with_filters() {
        let storage = Arc::new(MemoryStorage::new());

        let builder = ContextBuilder::new(storage)
            .with_memory_types(vec![MemoryType::LongTerm])
            .with_format(ContextFormat::Json)
            .with_max_tokens(1000)
            .with_metadata(true);

        assert_eq!(builder.memory_types, vec![MemoryType::LongTerm]);
        assert_eq!(builder.max_tokens, Some(1000));
        assert!(builder.include_metadata);
    }

    #[tokio::test]
    async fn test_simple_relevance_scoring() {
        let storage = Arc::new(MemoryStorage::new());
        let builder = ContextBuilder::new(storage);

        let mut entry = MemoryEntry::new("machine learning algorithms".to_string(), MemoryType::ShortTerm);
        let score = builder.simple_relevance_score(&entry, "machine learning");

        assert!(score > 0.0);
    }
}
