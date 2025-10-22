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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_memory_type_display() {
        assert_eq!(MemoryType::ShortTerm.to_string(), "short_term");
        assert_eq!(MemoryType::LongTerm.to_string(), "long_term");
    }

    #[test]
    fn test_memory_type_from_str() {
        assert_eq!("short_term".parse::<MemoryType>().unwrap(), MemoryType::ShortTerm);
        assert_eq!("short".parse::<MemoryType>().unwrap(), MemoryType::ShortTerm);
        assert_eq!("st".parse::<MemoryType>().unwrap(), MemoryType::ShortTerm);

        assert_eq!("long_term".parse::<MemoryType>().unwrap(), MemoryType::LongTerm);
        assert_eq!("long".parse::<MemoryType>().unwrap(), MemoryType::LongTerm);
        assert_eq!("lt".parse::<MemoryType>().unwrap(), MemoryType::LongTerm);

        assert!("invalid".parse::<MemoryType>().is_err());
    }

    #[test]
    fn test_memory_metadata_new() {
        let metadata = MemoryMetadata::new();

        assert_eq!(metadata.access_count, 0);
        assert!(metadata.tags.is_empty());
        assert!(metadata.custom_fields.is_empty());
        assert_eq!(metadata.importance, 0.5);
        assert_eq!(metadata.confidence, 1.0);
    }

    #[test]
    fn test_memory_metadata_builder_pattern() {
        let metadata = MemoryMetadata::new()
            .with_tags(vec!["test".to_string(), "example".to_string()])
            .with_importance(0.8)
            .with_confidence(0.9);

        assert_eq!(metadata.tags.len(), 2);
        assert_eq!(metadata.importance, 0.8);
        assert_eq!(metadata.confidence, 0.9);
    }

    #[test]
    fn test_memory_metadata_importance_clamping() {
        let metadata1 = MemoryMetadata::new().with_importance(1.5);
        assert_eq!(metadata1.importance, 1.0);

        let metadata2 = MemoryMetadata::new().with_importance(-0.5);
        assert_eq!(metadata2.importance, 0.0);
    }

    #[test]
    fn test_memory_metadata_tag_management() {
        let mut metadata = MemoryMetadata::new();

        // Add tags
        metadata.add_tag("test".to_string());
        metadata.add_tag("example".to_string());
        assert_eq!(metadata.tags.len(), 2);

        // Adding duplicate should not increase count
        metadata.add_tag("test".to_string());
        assert_eq!(metadata.tags.len(), 2);

        // Remove tag
        metadata.remove_tag("test");
        assert_eq!(metadata.tags.len(), 1);
        assert!(metadata.tags.contains(&"example".to_string()));
    }

    #[test]
    fn test_memory_metadata_custom_fields() {
        let mut metadata = MemoryMetadata::new();

        metadata.set_custom_field("author".to_string(), "Alice".to_string());
        metadata.set_custom_field("version".to_string(), "1.0".to_string());

        assert_eq!(metadata.get_custom_field("author"), Some(&"Alice".to_string()));
        assert_eq!(metadata.get_custom_field("version"), Some(&"1.0".to_string()));
        assert_eq!(metadata.get_custom_field("missing"), None);
    }

    #[test]
    fn test_memory_metadata_access_tracking() {
        let mut metadata = MemoryMetadata::new();
        let initial_accessed = metadata.last_accessed;

        std::thread::sleep(std::time::Duration::from_millis(10));
        metadata.mark_accessed();

        assert_eq!(metadata.access_count, 1);
        assert!(metadata.last_accessed > initial_accessed);
    }

    #[test]
    fn test_memory_metadata_modification_tracking() {
        let mut metadata = MemoryMetadata::new();
        let initial_modified = metadata.last_modified;
        let initial_count = metadata.access_count;

        std::thread::sleep(std::time::Duration::from_millis(10));
        metadata.mark_modified();

        assert!(metadata.last_modified > initial_modified);
        assert_eq!(metadata.access_count, initial_count + 1);
    }

    #[test]
    fn test_memory_entry_new() {
        let entry = MemoryEntry::new(
            "test_key".to_string(),
            "test_value".to_string(),
            MemoryType::ShortTerm,
        );

        assert_eq!(entry.key, "test_key");
        assert_eq!(entry.value, "test_value");
        assert_eq!(entry.memory_type, MemoryType::ShortTerm);
        assert!(entry.embedding.is_none());
    }

    #[test]
    fn test_memory_entry_builder_pattern() {
        let entry = MemoryEntry::new(
            "key".to_string(),
            "value".to_string(),
            MemoryType::LongTerm,
        )
        .with_tags(vec!["tag1".to_string()])
        .with_importance(0.9)
        .with_embedding(vec![1.0, 2.0, 3.0]);

        assert_eq!(entry.tags().len(), 1);
        assert_eq!(entry.metadata.importance, 0.9);
        assert!(entry.embedding.is_some());
    }

    #[test]
    fn test_memory_entry_update_value() {
        let mut entry = MemoryEntry::new(
            "key".to_string(),
            "old_value".to_string(),
            MemoryType::ShortTerm,
        );

        let initial_modified = entry.metadata.last_modified;
        std::thread::sleep(std::time::Duration::from_millis(10));

        entry.update_value("new_value".to_string());

        assert_eq!(entry.value, "new_value");
        assert!(entry.metadata.last_modified > initial_modified);
        assert_eq!(entry.metadata.access_count, 1);
    }

    #[test]
    fn test_memory_entry_tag_operations() {
        let entry = MemoryEntry::new(
            "key".to_string(),
            "value".to_string(),
            MemoryType::ShortTerm,
        )
        .with_tags(vec!["tag1".to_string(), "tag2".to_string()]);

        assert!(entry.has_tag("tag1"));
        assert!(entry.has_tag("tag2"));
        assert!(!entry.has_tag("tag3"));
        assert_eq!(entry.tags().len(), 2);
    }

    #[test]
    fn test_memory_entry_estimated_size() {
        let entry = MemoryEntry::new(
            "key".to_string(),
            "value".to_string(),
            MemoryType::ShortTerm,
        );

        let size = entry.estimated_size();
        assert!(size > 200); // Should be at least the overhead
        assert!(size < 1000); // Should be reasonable for small entry
    }

    #[test]
    fn test_memory_entry_expiration() {
        let entry = MemoryEntry::new(
            "key".to_string(),
            "value".to_string(),
            MemoryType::ShortTerm,
        );

        // Shouldn't be expired immediately
        assert!(!entry.is_expired(1));

        // Should definitely be expired after 0 hours
        assert!(entry.is_expired(0));
    }

    #[test]
    fn test_memory_fragment_new() {
        let entry = MemoryEntry::new(
            "key".to_string(),
            "value".to_string(),
            MemoryType::ShortTerm,
        );

        let fragment = MemoryFragment::new(entry.clone(), 0.95);

        assert_eq!(fragment.entry.key, "key");
        assert_eq!(fragment.relevance_score, 0.95);
        assert!(fragment.highlights.is_empty());
    }

    #[test]
    fn test_memory_fragment_with_highlights() {
        let entry = MemoryEntry::new(
            "key".to_string(),
            "value".to_string(),
            MemoryType::ShortTerm,
        );

        let fragment = MemoryFragment::new(entry, 0.8)
            .with_highlights(vec!["highlight1".to_string(), "highlight2".to_string()]);

        assert_eq!(fragment.highlights.len(), 2);
    }

    #[test]
    fn test_cosine_similarity_identical_vectors() {
        let vec1 = vec![1.0, 2.0, 3.0];
        let vec2 = vec![1.0, 2.0, 3.0];

        let similarity = cosine_similarity(&vec1, &vec2);
        assert!((similarity - 1.0).abs() < 0.0001);
    }

    #[test]
    fn test_cosine_similarity_orthogonal_vectors() {
        let vec1 = vec![1.0, 0.0];
        let vec2 = vec![0.0, 1.0];

        let similarity = cosine_similarity(&vec1, &vec2);
        assert!((similarity - 0.0).abs() < 0.0001);
    }

    #[test]
    fn test_cosine_similarity_different_lengths() {
        let vec1 = vec![1.0, 2.0, 3.0];
        let vec2 = vec![1.0, 2.0];

        let similarity = cosine_similarity(&vec1, &vec2);
        assert_eq!(similarity, 0.0);
    }

    #[test]
    fn test_cosine_similarity_zero_vectors() {
        let vec1 = vec![0.0, 0.0, 0.0];
        let vec2 = vec![1.0, 2.0, 3.0];

        let similarity = cosine_similarity(&vec1, &vec2);
        assert_eq!(similarity, 0.0);
    }

    #[test]
    fn test_text_similarity_identical_text() {
        let similarity = text_similarity("hello world", "hello world");
        assert_eq!(similarity, 1.0);
    }

    #[test]
    fn test_text_similarity_no_overlap() {
        let similarity = text_similarity("hello world", "foo bar");
        assert_eq!(similarity, 0.0);
    }

    #[test]
    fn test_text_similarity_partial_overlap() {
        let similarity = text_similarity("hello world test", "hello test example");
        // Intersection: {"hello", "test"} = 2
        // Union: {"hello", "world", "test", "example"} = 4
        assert!((similarity - 0.5).abs() < 0.0001);
    }

    #[test]
    fn test_text_similarity_empty_strings() {
        let similarity = text_similarity("", "");
        assert_eq!(similarity, 0.0);
    }

    #[test]
    fn test_memory_entry_similarity_with_embeddings() {
        let entry1 = MemoryEntry::new(
            "key1".to_string(),
            "value1".to_string(),
            MemoryType::ShortTerm,
        )
        .with_embedding(vec![1.0, 0.0]);

        let entry2 = MemoryEntry::new(
            "key2".to_string(),
            "value2".to_string(),
            MemoryType::ShortTerm,
        )
        .with_embedding(vec![0.0, 1.0]);

        let similarity = entry1.similarity_score(&entry2);
        assert!((similarity - 0.0).abs() < 0.0001);
    }

    #[test]
    fn test_memory_entry_similarity_without_embeddings() {
        let entry1 = MemoryEntry::new(
            "key1".to_string(),
            "hello world test".to_string(),
            MemoryType::ShortTerm,
        );

        let entry2 = MemoryEntry::new(
            "key2".to_string(),
            "hello world example".to_string(),
            MemoryType::ShortTerm,
        );

        let similarity = entry1.similarity_score(&entry2);
        assert!(similarity > 0.0);
        assert!(similarity < 1.0);
    }
}
