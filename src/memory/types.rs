//! Core memory types and data structures

use crate::error::{MemoryError, Result};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use uuid::Uuid;

/// Types of memory in the agent system
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum MemoryType {
    /// Short-term memory for current session context
    ShortTerm,
    /// Long-term memory for persistent knowledge
    LongTerm,
}

impl std::fmt::Display for MemoryType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MemoryType::ShortTerm => write!(f, "short_term"),
            MemoryType::LongTerm => write!(f, "long_term"),
        }
    }
}

impl std::str::FromStr for MemoryType {
    type Err = MemoryError;

    fn from_str(s: &str) -> Result<Self> {
        match s.to_lowercase().as_str() {
            "short_term" | "short" | "st" => Ok(MemoryType::ShortTerm),
            "long_term" | "long" | "lt" => Ok(MemoryType::LongTerm),
            _ => Err(MemoryError::InvalidMemoryType {
                memory_type: s.to_string(),
            }),
        }
    }
}

/// Metadata associated with memory entries
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryMetadata {
    /// Unique identifier for this memory entry
    pub id: Uuid,
    /// When this memory was created
    pub created_at: DateTime<Utc>,
    /// When this memory was last accessed
    pub last_accessed: DateTime<Utc>,
    /// When this memory was last modified
    pub last_modified: DateTime<Utc>,
    /// Number of times this memory has been accessed
    pub access_count: u64,
    /// Tags for categorization
    pub tags: Vec<String>,
    /// Custom metadata fields
    pub custom_fields: HashMap<String, String>,
    /// Memory importance score (0.0 to 1.0)
    pub importance: f64,
    /// Memory confidence score (0.0 to 1.0)
    pub confidence: f64,
}

impl MemoryMetadata {
    pub fn new() -> Self {
        let now = Utc::now();
        Self {
            id: Uuid::new_v4(),
            created_at: now,
            last_accessed: now,
            last_modified: now,
            access_count: 0,
            tags: Vec::new(),
            custom_fields: HashMap::new(),
            importance: 0.5,
            confidence: 1.0,
        }
    }

    pub fn with_tags(mut self, tags: Vec<String>) -> Self {
        self.tags = tags;
        self
    }

    pub fn with_importance(mut self, importance: f64) -> Self {
        self.importance = importance.clamp(0.0, 1.0);
        self
    }

    pub fn with_confidence(mut self, confidence: f64) -> Self {
        self.confidence = confidence.clamp(0.0, 1.0);
        self
    }

    pub fn add_tag(&mut self, tag: String) {
        if !self.tags.contains(&tag) {
            self.tags.push(tag);
        }
    }

    pub fn remove_tag(&mut self, tag: &str) {
        self.tags.retain(|t| t != tag);
    }

    pub fn set_custom_field(&mut self, key: String, value: String) {
        self.custom_fields.insert(key, value);
    }

    pub fn get_custom_field(&self, key: &str) -> Option<&String> {
        self.custom_fields.get(key)
    }

    pub fn mark_accessed(&mut self) {
        self.last_accessed = Utc::now();
        self.access_count += 1;
    }

    pub fn mark_modified(&mut self) {
        self.last_modified = Utc::now();
        self.mark_accessed();
    }
}

impl Default for MemoryMetadata {
    fn default() -> Self {
        Self::new()
    }
}

/// A complete memory entry in the agent's memory system
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryEntry {
    /// Unique key for this memory
    pub key: String,
    /// The actual memory content
    pub value: String,
    /// Type of memory (short-term or long-term)
    pub memory_type: MemoryType,
    /// Associated metadata
    pub metadata: MemoryMetadata,
    /// Optional vector embedding for similarity search
    pub embedding: Option<Vec<f32>>,
}

impl MemoryEntry {
    pub fn new(key: String, value: String, memory_type: MemoryType) -> Self {
        Self {
            key,
            value,
            memory_type,
            metadata: MemoryMetadata::new(),
            embedding: None,
        }
    }

    pub fn with_metadata(mut self, metadata: MemoryMetadata) -> Self {
        self.metadata = metadata;
        self
    }

    pub fn with_embedding(mut self, embedding: Vec<f32>) -> Self {
        self.embedding = Some(embedding);
        self
    }

    pub fn with_tags(mut self, tags: Vec<String>) -> Self {
        self.metadata = self.metadata.with_tags(tags);
        self
    }

    pub fn with_importance(mut self, importance: f64) -> Self {
        self.metadata = self.metadata.with_importance(importance);
        self
    }

    /// Get the unique ID of this memory entry
    pub fn id(&self) -> Uuid {
        self.metadata.id
    }

    /// Get when this memory was created
    pub fn created_at(&self) -> DateTime<Utc> {
        self.metadata.created_at
    }

    /// Get when this memory was last accessed
    pub fn last_accessed(&self) -> DateTime<Utc> {
        self.metadata.last_accessed
    }

    /// Get the access count
    pub fn access_count(&self) -> u64 {
        self.metadata.access_count
    }

    /// Mark this memory as accessed
    pub fn mark_accessed(&mut self) {
        self.metadata.mark_accessed();
    }

    /// Update the memory value
    pub fn update_value(&mut self, new_value: String) {
        self.value = new_value;
        self.metadata.mark_modified();
    }

    /// Check if this memory has a specific tag
    pub fn has_tag(&self, tag: &str) -> bool {
        self.metadata.tags.contains(&tag.to_string())
    }

    /// Get all tags
    pub fn tags(&self) -> &[String] {
        &self.metadata.tags
    }

    /// Estimate the size of this memory entry in bytes
    pub fn estimated_size(&self) -> usize {
        self.key.len()
            + self.value.len()
            + self.metadata.tags.iter().map(|t| t.len()).sum::<usize>()
            + self.metadata.custom_fields.iter()
                .map(|(k, v)| k.len() + v.len())
                .sum::<usize>()
            + self.embedding.as_ref().map_or(0, |e| e.len() * 4) // f32 = 4 bytes
            + 200 // Approximate overhead for other fields
    }

    /// Check if this memory is expired based on a time threshold
    pub fn is_expired(&self, max_age_hours: u64) -> bool {
        let max_age = chrono::Duration::hours(max_age_hours as i64);
        Utc::now() - self.metadata.created_at > max_age
    }

    /// Calculate similarity score with another memory entry
    pub fn similarity_score(&self, other: &MemoryEntry) -> f64 {
        if let (Some(embedding1), Some(embedding2)) = (&self.embedding, &other.embedding) {
            cosine_similarity(embedding1, embedding2)
        } else {
            // Fallback to simple text similarity
            text_similarity(&self.value, &other.value)
        }
    }
}

/// A fragment of memory returned from search operations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryFragment {
    /// The memory entry
    pub entry: MemoryEntry,
    /// Relevance score for this fragment (0.0 to 1.0)
    pub relevance_score: f64,
    /// Highlighted text snippets
    pub highlights: Vec<String>,
}

impl MemoryFragment {
    pub fn new(entry: MemoryEntry, relevance_score: f64) -> Self {
        Self {
            entry,
            relevance_score,
            highlights: Vec::new(),
        }
    }

    pub fn with_highlights(mut self, highlights: Vec<String>) -> Self {
        self.highlights = highlights;
        self
    }
}

/// Calculate cosine similarity between two vectors
fn cosine_similarity(a: &[f32], b: &[f32]) -> f64 {
    if a.len() != b.len() {
        return 0.0;
    }

    let dot_product: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

    if norm_a == 0.0 || norm_b == 0.0 {
        0.0
    } else {
        (dot_product / (norm_a * norm_b)) as f64
    }
}

/// Simple text similarity using Jaccard index
fn text_similarity(text1: &str, text2: &str) -> f64 {
    let words1: std::collections::HashSet<&str> = text1.split_whitespace().collect();
    let words2: std::collections::HashSet<&str> = text2.split_whitespace().collect();

    let intersection = words1.intersection(&words2).count();
    let union = words1.union(&words2).count();

    if union == 0 {
        0.0
    } else {
        intersection as f64 / union as f64
    }
}
