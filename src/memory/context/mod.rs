//! Context assembly for agent memory
//!
//! This module provides tools for assembling and formatting memory context
//! optimized for consumption by language models and AI agents.

pub mod builder;

use crate::memory::types::MemoryEntry;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

pub use builder::ContextBuilder;

/// Agent context assembled from memories
///
/// Contains all relevant memories organized by type (core, related, temporal)
/// with formatting options for different LLM providers.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentContext {
    /// Original query (if any)
    pub query: Option<String>,
    /// Core memories (most relevant to query)
    pub core_memories: Vec<MemoryEntry>,
    /// Related memories (from knowledge graph)
    pub related_memories: Vec<(MemoryEntry, f64)>, // (memory, relationship_strength)
    /// Temporal memories (time-based slice)
    pub temporal_memories: Vec<MemoryEntry>,
    /// Summaries and highlights
    pub summaries: Vec<String>,
    /// Additional metadata
    pub metadata: HashMap<String, String>,
    /// Output format
    pub format: ContextFormat,
}

/// Output format for context
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ContextFormat {
    /// Plain text format
    Plain,
    /// Markdown format
    Markdown,
    /// JSON format
    Json,
    /// XML format
    Xml,
    /// Custom format for specific LLM providers
    OpenAI,
    /// Anthropic Claude format
    Claude,
    /// Google formats
    Gemini,
}

impl AgentContext {
    /// Create a new empty context
    pub fn new() -> Self {
        Self {
            query: None,
            core_memories: Vec::new(),
            related_memories: Vec::new(),
            temporal_memories: Vec::new(),
            summaries: Vec::new(),
            metadata: HashMap::new(),
            format: ContextFormat::Markdown,
        }
    }

    /// Format the context as a string
    pub fn format_as_string(&self) -> String {
        match self.format {
            ContextFormat::Plain => self.format_plain(),
            ContextFormat::Markdown => self.format_markdown(),
            ContextFormat::Json => self.format_json(),
            ContextFormat::Xml => self.format_xml(),
            ContextFormat::OpenAI => self.format_openai(),
            ContextFormat::Claude => self.format_claude(),
            ContextFormat::Gemini => self.format_gemini(),
        }
    }

    /// Format as plain text
    fn format_plain(&self) -> String {
        let mut output = String::new();

        if let Some(ref query) = self.query {
            output.push_str(&format!("Query: {}\n\n", query));
        }

        if !self.summaries.is_empty() {
            output.push_str("Summary:\n");
            for summary in &self.summaries {
                output.push_str(&format!("- {}\n", summary));
            }
            output.push_str("\n");
        }

        if !self.core_memories.is_empty() {
            output.push_str("Core Memories:\n");
            for (i, memory) in self.core_memories.iter().enumerate() {
                output.push_str(&format!("{}. {}\n", i + 1, memory.value));
            }
            output.push_str("\n");
        }

        if !self.related_memories.is_empty() {
            output.push_str("Related Memories:\n");
            for (memory, strength) in &self.related_memories {
                output.push_str(&format!("- {} (relevance: {:.2})\n", memory.value, strength));
            }
            output.push_str("\n");
        }

        if !self.temporal_memories.is_empty() {
            output.push_str("Recent Context:\n");
            for memory in &self.temporal_memories {
                output.push_str(&format!("- {}\n", memory.value));
            }
        }

        output
    }

    /// Format as Markdown
    fn format_markdown(&self) -> String {
        let mut output = String::new();

        if let Some(ref query) = self.query {
            output.push_str(&format!("# Context for: {}\n\n", query));
        } else {
            output.push_str("# Agent Memory Context\n\n");
        }

        if !self.summaries.is_empty() {
            output.push_str("## Summary\n\n");
            for summary in &self.summaries {
                output.push_str(&format!("- {}\n", summary));
            }
            output.push_str("\n");
        }

        if !self.core_memories.is_empty() {
            output.push_str("## Core Memories\n\n");
            for (i, memory) in self.core_memories.iter().enumerate() {
                output.push_str(&format!("{}. **{}**\n   - Type: {:?}\n   - Created: {}\n\n",
                    i + 1,
                    memory.value,
                    memory.memory_type,
                    memory.created_at.format("%Y-%m-%d %H:%M")
                ));
            }
        }

        if !self.related_memories.is_empty() {
            output.push_str("## Related Memories\n\n");
            for (memory, strength) in &self.related_memories {
                output.push_str(&format!("- {} *(relevance: {:.0}%)*\n", memory.value, strength * 100.0));
            }
            output.push_str("\n");
        }

        if !self.temporal_memories.is_empty() {
            output.push_str("## Recent Context\n\n");
            for memory in &self.temporal_memories {
                output.push_str(&format!("- {} *({} ago)*\n",
                    memory.value,
                    format_time_ago(memory.created_at)
                ));
            }
        }

        output
    }

    /// Format as JSON
    fn format_json(&self) -> String {
        serde_json::to_string_pretty(self).unwrap_or_else(|_| "{}".to_string())
    }

    /// Format as XML
    fn format_xml(&self) -> String {
        let mut output = String::from("<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n<context>\n");

        if let Some(ref query) = self.query {
            output.push_str(&format!("  <query>{}</query>\n", escape_xml(query)));
        }

        output.push_str("  <core_memories>\n");
        for memory in &self.core_memories {
            output.push_str(&format!("    <memory type=\"{:?}\">{}</memory>\n",
                memory.memory_type, escape_xml(&memory.value)));
        }
        output.push_str("  </core_memories>\n");

        output.push_str("  <related_memories>\n");
        for (memory, strength) in &self.related_memories {
            output.push_str(&format!("    <memory strength=\"{}\">{}</memory>\n",
                strength, escape_xml(&memory.value)));
        }
        output.push_str("  </related_memories>\n");

        output.push_str("</context>");
        output
    }

    /// Format for OpenAI (structured for GPT models)
    fn format_openai(&self) -> String {
        let mut parts = Vec::new();

        if let Some(ref query) = self.query {
            parts.push(format!("User Query: {}", query));
        }

        if !self.core_memories.is_empty() {
            parts.push("Relevant Context:".to_string());
            for memory in &self.core_memories {
                parts.push(format!("- {}", memory.value));
            }
        }

        if !self.related_memories.is_empty() {
            parts.push("\nRelated Information:".to_string());
            for (memory, _) in &self.related_memories {
                parts.push(format!("- {}", memory.value));
            }
        }

        parts.join("\n")
    }

    /// Format for Anthropic Claude
    fn format_claude(&self) -> String {
        // Claude prefers clear structure with XML-like tags
        let mut output = String::new();

        if let Some(ref query) = self.query {
            output.push_str(&format!("<query>{}</query>\n\n", query));
        }

        if !self.core_memories.is_empty() {
            output.push_str("<relevant_memories>\n");
            for memory in &self.core_memories {
                output.push_str(&format!("<memory>{}</memory>\n", memory.value));
            }
            output.push_str("</relevant_memories>\n\n");
        }

        if !self.related_memories.is_empty() {
            output.push_str("<related_context>\n");
            for (memory, _) in &self.related_memories {
                output.push_str(&format!("<item>{}</item>\n", memory.value));
            }
            output.push_str("</related_context>");
        }

        output
    }

    /// Format for Google Gemini
    fn format_gemini(&self) -> String {
        // Gemini prefers clear, structured text
        self.format_markdown()
    }

    /// Get total memory count
    pub fn total_memories(&self) -> usize {
        self.core_memories.len() + self.related_memories.len() + self.temporal_memories.len()
    }

    /// Check if context is empty
    pub fn is_empty(&self) -> bool {
        self.core_memories.is_empty()
            && self.related_memories.is_empty()
            && self.temporal_memories.is_empty()
    }
}

impl Default for AgentContext {
    fn default() -> Self {
        Self::new()
    }
}

/// Token counter for estimating context size
pub struct TokenCounter {
    // Rough approximation: 4 characters ≈ 1 token
    chars_per_token: f64,
}

impl TokenCounter {
    pub fn new() -> Self {
        Self {
            chars_per_token: 4.0,
        }
    }

    /// Count tokens in a string (rough estimate)
    pub fn count(&self, text: &str) -> usize {
        (text.len() as f64 / self.chars_per_token).ceil() as usize
    }

    /// Count tokens in context
    pub fn count_context(&self, context: &AgentContext) -> usize {
        let text = context.format_as_string();
        self.count(&text)
    }
}

impl Default for TokenCounter {
    fn default() -> Self {
        Self::new()
    }
}

/// Format time ago string
fn format_time_ago(timestamp: chrono::DateTime<chrono::Utc>) -> String {
    let duration = chrono::Utc::now() - timestamp;

    let seconds = duration.num_seconds();
    if seconds < 60 {
        return format!("{}s", seconds);
    }

    let minutes = duration.num_minutes();
    if minutes < 60 {
        return format!("{}m", minutes);
    }

    let hours = duration.num_hours();
    if hours < 24 {
        return format!("{}h", hours);
    }

    let days = duration.num_days();
    format!("{}d", days)
}

/// Escape XML special characters
fn escape_xml(text: &str) -> String {
    text.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
        .replace('\'', "&apos;")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::types::MemoryType;

    #[test]
    fn test_agent_context_creation() {
        let context = AgentContext::new();
        assert!(context.is_empty());
        assert_eq!(context.total_memories(), 0);
    }

    #[test]
    fn test_context_formatting() {
        let mut context = AgentContext::new();
        context.query = Some("test query".to_string());

        let mut memory = MemoryEntry::new("test memory".to_string(), MemoryType::ShortTerm);
        context.core_memories.push(memory);

        let plain = context.format_plain();
        assert!(plain.contains("test query"));
        assert!(plain.contains("test memory"));

        let markdown = context.format_markdown();
        assert!(markdown.contains("# Context"));
        assert!(markdown.contains("test memory"));
    }

    #[test]
    fn test_token_counter() {
        let counter = TokenCounter::new();

        let text = "This is a test string with multiple words";
        let count = counter.count(text);
        assert!(count > 0);
        assert!(count < text.len()); // Should be less than character count
    }

    #[test]
    fn test_xml_escaping() {
        let text = "<test> & \"quotes\"";
        let escaped = escape_xml(text);
        assert!(escaped.contains("&lt;"));
        assert!(escaped.contains("&amp;"));
        assert!(escaped.contains("&quot;"));
    }

    #[test]
    fn test_format_time_ago() {
        use chrono::Duration;

        let now = chrono::Utc::now();
        let one_hour_ago = now - Duration::hours(1);

        let formatted = format_time_ago(one_hour_ago);
        assert!(formatted.contains("h") || formatted.contains("m"));
    }
}
