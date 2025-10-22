//! Context builder for assembling agent memory context.

use crate::error::{MemoryError, Result};
use crate::memory::knowledge_graph::RelationshipType;
use crate::memory::types::{MemoryEntry, MemoryFragment};
use crate::AgentMemory;
use chrono::{DateTime, Duration, Utc};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use thiserror::Error;

/// Error types for context building
#[derive(Debug, Error)]
pub enum ContextError {
    /// Memory retrieval failed
    #[error("Memory retrieval failed: {0}")]
    RetrievalFailed(String),

    /// Graph query failed
    #[error("Graph query failed: {0}")]
    GraphQueryFailed(String),

    /// Context too large for token limit
    #[error("Context exceeds token limit: {current} > {limit}")]
    TokenLimitExceeded { current: usize, limit: usize },

    /// Invalid configuration
    #[error("Invalid configuration: {0}")]
    InvalidConfig(String),

    /// Formatting error
    #[error("Formatting error: {0}")]
    FormattingError(String),
}

/// Result type for context operations
pub type ContextResult<T> = std::result::Result<T, ContextError>;

/// LLM provider format for context output
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum LlmFormat {
    /// OpenAI format (ChatGPT, GPT-4)
    OpenAI,
    /// Anthropic format (Claude)
    Anthropic,
    /// Plain text format
    PlainText,
    /// JSON format
    Json,
    /// Markdown format
    Markdown,
}

/// Temporal range for filtering memories
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TemporalRange {
    /// Last N hours
    Hours(i64),
    /// Last N days
    Days(i64),
    /// Custom time range
    Range { start: DateTime<Utc>, end: DateTime<Utc> },
    /// Since a specific time
    Since(DateTime<Utc>),
}

impl TemporalRange {
    /// Check if a timestamp falls within this range
    pub fn contains(&self, timestamp: DateTime<Utc>) -> bool {
        let now = Utc::now();
        match self {
            Self::Hours(h) => timestamp > now - Duration::hours(*h),
            Self::Days(d) => timestamp > now - Duration::days(*d),
            Self::Range { start, end } => timestamp >= *start && timestamp <= *end,
            Self::Since(since) => timestamp >= *since,
        }
    }

    /// Get the start time for this range
    pub fn start_time(&self) -> DateTime<Utc> {
        let now = Utc::now();
        match self {
            Self::Hours(h) => now - Duration::hours(*h),
            Self::Days(d) => now - Duration::days(*d),
            Self::Range { start, .. } => *start,
            Self::Since(since) => *since,
        }
    }
}

/// Filter for relationship types in graph traversal
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RelationshipFilter {
    /// Include all relationship types
    All,
    /// Include only specific types
    Include(Vec<RelationshipType>),
    /// Exclude specific types
    Exclude(Vec<RelationshipType>),
}

impl RelationshipFilter {
    /// Check if a relationship type passes the filter
    pub fn matches(&self, rel_type: &RelationshipType) -> bool {
        match self {
            Self::All => true,
            Self::Include(types) => types.contains(rel_type),
            Self::Exclude(types) => !types.contains(rel_type),
        }
    }
}

/// Section of context (for organized output)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemorySection {
    /// Section title
    pub title: String,
    /// Section description
    pub description: String,
    /// Memories in this section
    pub memories: Vec<MemoryEntry>,
    /// Section importance (0.0 to 1.0)
    pub importance: f64,
}

impl MemorySection {
    /// Create a new section
    pub fn new(title: String, description: String) -> Self {
        Self {
            title,
            description,
            memories: Vec::new(),
            importance: 0.5,
        }
    }

    /// Add a memory to this section
    pub fn add_memory(&mut self, memory: MemoryEntry) {
        self.memories.push(memory);
    }

    /// Set section importance
    pub fn with_importance(mut self, importance: f64) -> Self {
        self.importance = importance.clamp(0.0, 1.0);
        self
    }

    /// Estimate token count for this section
    pub fn estimate_tokens(&self) -> usize {
        let header_tokens = (self.title.len() + self.description.len()) / 4;
        let memory_tokens: usize = self
            .memories
            .iter()
            .map(|m| m.value.len() / 4)
            .sum();
        header_tokens + memory_tokens
    }
}

/// Assembled context for an AI agent
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentContext {
    /// Core relevant memories (from semantic search)
    pub core_memories: Vec<MemoryEntry>,
    /// Related memories (from graph traversal)
    pub related_memories: Vec<MemoryEntry>,
    /// Temporal memories (recent/historical)
    pub temporal_memories: Vec<MemoryEntry>,
    /// Summary information
    pub summary: String,
    /// Metadata about the context
    pub metadata: HashMap<String, String>,
    /// Sections for organized output
    pub sections: Vec<MemorySection>,
    /// Total estimated token count
    pub estimated_tokens: usize,
}

impl AgentContext {
    /// Create a new empty context
    pub fn new() -> Self {
        Self {
            core_memories: Vec::new(),
            related_memories: Vec::new(),
            temporal_memories: Vec::new(),
            summary: String::new(),
            metadata: HashMap::new(),
            sections: Vec::new(),
            estimated_tokens: 0,
        }
    }

    /// Estimate total token count
    pub fn update_token_estimate(&mut self) {
        let core_tokens: usize = self.core_memories.iter().map(|m| m.value.len() / 4).sum();
        let related_tokens: usize = self.related_memories.iter().map(|m| m.value.len() / 4).sum();
        let temporal_tokens: usize = self.temporal_memories.iter().map(|m| m.value.len() / 4).sum();
        let summary_tokens = self.summary.len() / 4;
        let section_tokens: usize = self.sections.iter().map(|s| s.estimate_tokens()).sum();

        self.estimated_tokens = core_tokens + related_tokens + temporal_tokens + summary_tokens + section_tokens;
    }

    /// Format context for a specific LLM provider
    pub fn format(&self, format: LlmFormat, max_tokens: usize) -> ContextResult<String> {
        match format {
            LlmFormat::OpenAI => self.format_openai(max_tokens),
            LlmFormat::Anthropic => self.format_anthropic(max_tokens),
            LlmFormat::PlainText => self.format_plain_text(max_tokens),
            LlmFormat::Json => self.format_json(max_tokens),
            LlmFormat::Markdown => self.format_markdown(max_tokens),
        }
    }

    /// Format for OpenAI (system message style)
    fn format_openai(&self, max_tokens: usize) -> ContextResult<String> {
        let mut output = String::new();
        output.push_str("# Memory Context\n\n");

        if !self.summary.is_empty() {
            output.push_str("## Summary\n");
            output.push_str(&self.summary);
            output.push_str("\n\n");
        }

        if !self.core_memories.is_empty() {
            output.push_str("## Core Memories (Most Relevant)\n");
            for (i, memory) in self.core_memories.iter().enumerate() {
                output.push_str(&format!("{}. {}\n", i + 1, memory.value));
            }
            output.push_str("\n");
        }

        if !self.related_memories.is_empty() {
            output.push_str("## Related Context\n");
            for memory in &self.related_memories {
                output.push_str(&format!("- {}\n", memory.value));
            }
            output.push_str("\n");
        }

        if !self.temporal_memories.is_empty() {
            output.push_str("## Recent Memories\n");
            for memory in &self.temporal_memories {
                output.push_str(&format!("- {} ({})\n", memory.value, memory.created_at.format("%Y-%m-%d %H:%M")));
            }
        }

        self.truncate_to_tokens(&output, max_tokens)
    }

    /// Format for Anthropic (Claude)
    fn format_anthropic(&self, max_tokens: usize) -> ContextResult<String> {
        let mut output = String::new();

        if !self.summary.is_empty() {
            output.push_str(&self.summary);
            output.push_str("\n\n");
        }

        if !self.core_memories.is_empty() {
            output.push_str("Here are the most relevant memories:\n\n");
            for memory in &self.core_memories {
                output.push_str(&format!("<memory>\n{}\n</memory>\n\n", memory.value));
            }
        }

        if !self.related_memories.is_empty() {
            output.push_str("Related context:\n\n");
            for memory in &self.related_memories {
                output.push_str(&format!("<related>\n{}\n</related>\n\n", memory.value));
            }
        }

        self.truncate_to_tokens(&output, max_tokens)
    }

    /// Format as plain text
    fn format_plain_text(&self, max_tokens: usize) -> ContextResult<String> {
        let mut output = String::new();

        if !self.summary.is_empty() {
            output.push_str(&self.summary);
            output.push_str("\n\n");
        }

        for memory in &self.core_memories {
            output.push_str(&memory.value);
            output.push_str("\n");
        }

        for memory in &self.related_memories {
            output.push_str(&memory.value);
            output.push_str("\n");
        }

        for memory in &self.temporal_memories {
            output.push_str(&memory.value);
            output.push_str("\n");
        }

        self.truncate_to_tokens(&output, max_tokens)
    }

    /// Format as JSON
    fn format_json(&self, _max_tokens: usize) -> ContextResult<String> {
        serde_json::to_string_pretty(self)
            .map_err(|e| ContextError::FormattingError(format!("JSON serialization failed: {}", e)))
    }

    /// Format as Markdown
    fn format_markdown(&self, max_tokens: usize) -> ContextResult<String> {
        let mut output = String::new();
        output.push_str("# Memory Context\n\n");

        if !self.summary.is_empty() {
            output.push_str("## Summary\n\n");
            output.push_str(&self.summary);
            output.push_str("\n\n");
        }

        if !self.core_memories.is_empty() {
            output.push_str("## Core Memories\n\n");
            for memory in &self.core_memories {
                output.push_str(&format!("- **{}**: {}\n", memory.key, memory.value));
            }
            output.push_str("\n");
        }

        if !self.related_memories.is_empty() {
            output.push_str("## Related Memories\n\n");
            for memory in &self.related_memories {
                output.push_str(&format!("- {}\n", memory.value));
            }
            output.push_str("\n");
        }

        if !self.temporal_memories.is_empty() {
            output.push_str("## Recent Activity\n\n");
            for memory in &self.temporal_memories {
                output.push_str(&format!("- {} *({})*\n", memory.value, memory.created_at.format("%Y-%m-%d")));
            }
        }

        self.truncate_to_tokens(&output, max_tokens)
    }

    /// Truncate text to fit within token limit
    fn truncate_to_tokens(&self, text: &str, max_tokens: usize) -> ContextResult<String> {
        let estimated_tokens = text.len() / 4;

        if estimated_tokens <= max_tokens {
            Ok(text.to_string())
        } else {
            // Truncate to approximate token count
            let max_chars = max_tokens * 4;
            let truncated = if text.len() > max_chars {
                format!("{}...\n\n[Context truncated to fit token limit]", &text[..max_chars])
            } else {
                text.to_string()
            };
            Ok(truncated)
        }
    }
}

impl Default for AgentContext {
    fn default() -> Self {
        Self::new()
    }
}

/// Builder for assembling agent context from memory
pub struct ContextBuilder<'a> {
    memory: &'a AgentMemory,
    context: AgentContext,
    query: Option<String>,
    core_limit: usize,
    graph_depth: usize,
    relationship_filter: RelationshipFilter,
    temporal_range: Option<TemporalRange>,
    include_summaries: bool,
    deduplicate: bool,
}

impl<'a> ContextBuilder<'a> {
    /// Create a new context builder
    pub fn new(memory: &'a AgentMemory) -> Self {
        Self {
            memory,
            context: AgentContext::new(),
            query: None,
            core_limit: 10,
            graph_depth: 1,
            relationship_filter: RelationshipFilter::All,
            temporal_range: None,
            include_summaries: false,
            deduplicate: true,
        }
    }

    /// Add relevant memories based on semantic search
    pub fn with_relevant_memories(mut self, query: &str, limit: usize) -> ContextResult<Self> {
        self.query = Some(query.to_string());
        self.core_limit = limit;
        Ok(self)
    }

    /// Add graph neighbors (related memories via knowledge graph)
    pub fn with_graph_neighbors(
        mut self,
        depth: usize,
        filter: Option<RelationshipFilter>,
    ) -> ContextResult<Self> {
        self.graph_depth = depth;
        if let Some(f) = filter {
            self.relationship_filter = f;
        }
        Ok(self)
    }

    /// Add temporal slice (memories from a time range)
    pub fn with_temporal_slice(mut self, range: TemporalRange) -> ContextResult<Self> {
        self.temporal_range = Some(range);
        Ok(self)
    }

    /// Include memory summaries
    pub fn with_summaries(mut self) -> ContextResult<Self> {
        self.include_summaries = true;
        Ok(self)
    }

    /// Enable/disable deduplication
    pub fn with_deduplication(mut self, enabled: bool) -> Self {
        self.deduplicate = enabled;
        self
    }

    /// Build the final context
    pub async fn build(mut self) -> ContextResult<AgentContext> {
        // Step 1: Get core relevant memories
        if let Some(ref query) = self.query {
            self.retrieve_core_memories(query).await?;
        }

        // Step 2: Get graph-related memories
        if self.graph_depth > 0 && !self.context.core_memories.is_empty() {
            self.retrieve_graph_neighbors().await?;
        }

        // Step 3: Get temporal memories
        if let Some(ref range) = self.temporal_range {
            self.retrieve_temporal_memories(range).await?;
        }

        // Step 4: Deduplicate if enabled
        if self.deduplicate {
            self.deduplicate_memories();
        }

        // Step 5: Generate summary if requested
        if self.include_summaries {
            self.generate_summary();
        }

        // Step 6: Add metadata
        self.add_metadata();

        // Step 7: Update token estimate
        self.context.update_token_estimate();

        Ok(self.context)
    }

    /// Retrieve core memories via semantic search
    async fn retrieve_core_memories(&mut self, query: &str) -> ContextResult<()> {
        let fragments = self
            .memory
            .search(query, self.core_limit)
            .await
            .map_err(|e| ContextError::RetrievalFailed(format!("Search failed: {}", e)))?;

        self.context.core_memories = fragments.into_iter().map(|f| f.entry).collect();

        Ok(())
    }

    /// Retrieve graph neighbors for core memories
    async fn retrieve_graph_neighbors(&mut self) -> ContextResult<()> {
        let mut related = HashSet::new();

        // For each core memory, get graph neighbors
        for core_memory in &self.context.core_memories {
            if let Some(ref graph) = self.memory.knowledge_graph {
                // Get neighbors via graph traversal
                if let Some(node) = graph.get_node(&core_memory.key) {
                    for edge in graph.get_edges(&node.id) {
                        // Check relationship filter
                        if self.relationship_filter.matches(&edge.relationship_type) {
                            // Get the related node
                            let related_id = if edge.from_node == node.id {
                                &edge.to_node
                            } else {
                                &edge.from_node
                            };

                            if let Some(related_node) = graph.get_node_by_id(related_id) {
                                related.insert(related_node.memory_key.clone());
                            }
                        }
                    }
                }
            }
        }

        // Retrieve related memories
        for key in related {
            if let Ok(Some(entry)) = self.memory.retrieve(&key).await {
                self.context.related_memories.push(entry);
            }
        }

        Ok(())
    }

    /// Retrieve temporal memories
    async fn retrieve_temporal_memories(&mut self, range: &TemporalRange) -> ContextResult<()> {
        // Get all memories and filter by temporal range
        // This is a simplified implementation - in production, you'd query storage directly
        let all_keys = self
            .memory
            .storage
            .list_keys()
            .await
            .map_err(|e| ContextError::RetrievalFailed(format!("Failed to list keys: {}", e)))?;

        for key in all_keys {
            if let Ok(Some(entry)) = self.memory.retrieve(&key).await {
                if range.contains(entry.created_at) {
                    self.context.temporal_memories.push(entry);
                }
            }
        }

        // Sort by recency
        self.context.temporal_memories.sort_by(|a, b| b.created_at.cmp(&a.created_at));

        // Limit to reasonable size
        self.context.temporal_memories.truncate(20);

        Ok(())
    }

    /// Remove duplicate memories across sections
    fn deduplicate_memories(&mut self) {
        let mut seen_keys = HashSet::new();

        // Keep core memories (highest priority)
        self.context.core_memories.retain(|m| seen_keys.insert(m.key.clone()));

        // Deduplicate related memories
        self.context.related_memories.retain(|m| seen_keys.insert(m.key.clone()));

        // Deduplicate temporal memories
        self.context.temporal_memories.retain(|m| seen_keys.insert(m.key.clone()));
    }

    /// Generate summary of the context
    fn generate_summary(&mut self) {
        let total_memories = self.context.core_memories.len()
            + self.context.related_memories.len()
            + self.context.temporal_memories.len();

        self.context.summary = format!(
            "Context contains {} memories: {} core, {} related, {} temporal.",
            total_memories,
            self.context.core_memories.len(),
            self.context.related_memories.len(),
            self.context.temporal_memories.len()
        );

        if let Some(ref query) = self.query {
            self.context.summary.push_str(&format!(" Query: \"{}\"", query));
        }
    }

    /// Add metadata to context
    fn add_metadata(&mut self) {
        self.context.metadata.insert("generated_at".to_string(), Utc::now().to_rfc3339());
        self.context.metadata.insert("core_limit".to_string(), self.core_limit.to_string());
        self.context.metadata.insert("graph_depth".to_string(), self.graph_depth.to_string());

        if let Some(ref query) = self.query {
            self.context.metadata.insert("query".to_string(), query.clone());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_temporal_range_hours() {
        let range = TemporalRange::Hours(2);
        let now = Utc::now();
        let one_hour_ago = now - Duration::hours(1);
        let three_hours_ago = now - Duration::hours(3);

        assert!(range.contains(one_hour_ago));
        assert!(!range.contains(three_hours_ago));
    }

    #[test]
    fn test_temporal_range_days() {
        let range = TemporalRange::Days(7);
        let now = Utc::now();
        let yesterday = now - Duration::days(1);
        let two_weeks_ago = now - Duration::days(14);

        assert!(range.contains(yesterday));
        assert!(!range.contains(two_weeks_ago));
    }

    #[test]
    fn test_relationship_filter() {
        let filter = RelationshipFilter::Include(vec![RelationshipType::Prerequisite]);
        assert!(filter.matches(&RelationshipType::Prerequisite));
        assert!(!filter.matches(&RelationshipType::Similar));

        let exclude_filter = RelationshipFilter::Exclude(vec![RelationshipType::Causal]);
        assert!(!exclude_filter.matches(&RelationshipType::Causal));
        assert!(exclude_filter.matches(&RelationshipType::Similar));
    }

    #[test]
    fn test_memory_section_tokens() {
        let mut section = MemorySection::new("Test".to_string(), "Description".to_string());
        section.add_memory(MemoryEntry::new(
            "key1".to_string(),
            "content with some text".to_string(),
            crate::memory::types::MemoryType::ShortTerm,
        ));

        let tokens = section.estimate_tokens();
        assert!(tokens > 0);
    }

    #[test]
    fn test_agent_context_token_estimate() {
        let mut context = AgentContext::new();
        context.core_memories.push(MemoryEntry::new(
            "key1".to_string(),
            "test content".to_string(),
            crate::memory::types::MemoryType::ShortTerm,
        ));

        context.update_token_estimate();
        assert!(context.estimated_tokens > 0);
    }
}
