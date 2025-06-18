//! Agent state management for memory system

use crate::error::{MemoryError, Result};
use crate::memory::types::{MemoryEntry, MemoryType};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use uuid::Uuid;

/// The complete state of an AI agent's memory system
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentState {
    /// Unique session identifier
    session_id: Uuid,
    /// When this state was created
    created_at: DateTime<Utc>,
    /// When this state was last modified
    last_modified: DateTime<Utc>,
    /// Short-term memories (current session)
    short_term_memories: HashMap<String, MemoryEntry>,
    /// Long-term memories (persistent across sessions)
    long_term_memories: HashMap<String, MemoryEntry>,
    /// Memory access patterns for optimization
    access_patterns: HashMap<String, AccessPattern>,
    /// State version for conflict resolution
    version: u64,
}

/// Access pattern tracking for memory optimization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AccessPattern {
    pub key: String,
    pub access_count: u64,
    pub last_access: DateTime<Utc>,
    pub access_frequency: f64, // accesses per hour
    pub average_session_length: f64, // in minutes
}

impl AgentState {
    /// Create a new agent state with the given session ID
    pub fn new(session_id: Uuid) -> Self {
        let now = Utc::now();
        Self {
            session_id,
            created_at: now,
            last_modified: now,
            short_term_memories: HashMap::new(),
            long_term_memories: HashMap::new(),
            access_patterns: HashMap::new(),
            version: 1,
        }
    }

    /// Get the session ID
    pub fn session_id(&self) -> Uuid {
        self.session_id
    }

    /// Get when this state was created
    pub fn created_at(&self) -> DateTime<Utc> {
        self.created_at
    }

    /// Get when this state was last modified
    pub fn last_modified(&self) -> DateTime<Utc> {
        self.last_modified
    }

    /// Get the current version
    pub fn version(&self) -> u64 {
        self.version
    }

    /// Add a memory entry to the appropriate memory store
    pub fn add_memory(&mut self, mut entry: MemoryEntry) {
        entry.mark_accessed();
        
        match entry.memory_type {
            MemoryType::ShortTerm => {
                self.short_term_memories.insert(entry.key.clone(), entry.clone());
            }
            MemoryType::LongTerm => {
                self.long_term_memories.insert(entry.key.clone(), entry.clone());
            }
        }

        self.update_access_pattern(&entry.key);
        self.mark_modified();
    }

    /// Get a memory by key from either short-term or long-term storage
    pub fn get_memory(&mut self, key: &str) -> Option<MemoryEntry> {
        // First check short-term memory
        if let Some(entry) = self.short_term_memories.get(key).cloned() {
            self.update_access_pattern(key);
            return Some(entry);
        }

        // Then check long-term memory
        if let Some(entry) = self.long_term_memories.get(key).cloned() {
            self.update_access_pattern(key);
            return Some(entry);
        }

        None
    }

    /// Check if a memory exists
    pub fn has_memory(&self, key: &str) -> bool {
        self.short_term_memories.contains_key(key) || self.long_term_memories.contains_key(key)
    }

    /// Update an existing memory entry
    pub fn update_memory(&mut self, key: &str, new_value: String) -> Result<()> {
        if let Some(entry) = self.short_term_memories.get_mut(key) {
            entry.update_value(new_value);
            self.update_access_pattern(key);
            self.mark_modified();
            return Ok(());
        }

        if let Some(entry) = self.long_term_memories.get_mut(key) {
            entry.update_value(new_value);
            self.update_access_pattern(key);
            self.mark_modified();
            return Ok(());
        }

        Err(MemoryError::NotFound {
            key: key.to_string(),
        })
    }

    /// Remove a memory entry
    pub fn remove_memory(&mut self, key: &str) -> Option<MemoryEntry> {
        let removed = self.short_term_memories.remove(key)
            .or_else(|| self.long_term_memories.remove(key));

        if removed.is_some() {
            self.access_patterns.remove(key);
            self.mark_modified();
        }

        removed
    }

    /// Get all memory keys
    pub fn get_all_keys(&self) -> Vec<String> {
        let mut keys = Vec::new();
        keys.extend(self.short_term_memories.keys().cloned());
        keys.extend(self.long_term_memories.keys().cloned());
        keys
    }

    /// Get all short-term memory entries
    pub fn get_short_term_memories(&self) -> &HashMap<String, MemoryEntry> {
        &self.short_term_memories
    }

    /// Get all long-term memory entries
    pub fn get_long_term_memories(&self) -> &HashMap<String, MemoryEntry> {
        &self.long_term_memories
    }

    /// Get memory count by type
    pub fn short_term_memory_count(&self) -> usize {
        self.short_term_memories.len()
    }

    pub fn long_term_memory_count(&self) -> usize {
        self.long_term_memories.len()
    }

    /// Get total memory size estimation
    pub fn total_memory_size(&self) -> usize {
        let short_term_size: usize = self.short_term_memories
            .values()
            .map(|entry| entry.estimated_size())
            .sum();

        let long_term_size: usize = self.long_term_memories
            .values()
            .map(|entry| entry.estimated_size())
            .sum();

        short_term_size + long_term_size
    }

    /// Clear all memories
    pub fn clear(&mut self) {
        self.short_term_memories.clear();
        self.long_term_memories.clear();
        self.access_patterns.clear();
        self.mark_modified();
    }

    /// Clear only short-term memories
    pub fn clear_short_term(&mut self) {
        for key in self.short_term_memories.keys() {
            self.access_patterns.remove(key);
        }
        self.short_term_memories.clear();
        self.mark_modified();
    }

    /// Promote a short-term memory to long-term
    pub fn promote_to_long_term(&mut self, key: &str) -> Result<()> {
        if let Some(mut entry) = self.short_term_memories.remove(key) {
            entry.memory_type = MemoryType::LongTerm;
            entry.mark_accessed();
            self.long_term_memories.insert(key.to_string(), entry);
            self.mark_modified();
            Ok(())
        } else {
            Err(MemoryError::NotFound {
                key: key.to_string(),
            })
        }
    }

    /// Get memories that match a predicate
    pub fn filter_memories<F>(&self, predicate: F) -> Vec<&MemoryEntry>
    where
        F: Fn(&MemoryEntry) -> bool,
    {
        let mut results = Vec::new();
        
        for entry in self.short_term_memories.values() {
            if predicate(entry) {
                results.push(entry);
            }
        }
        
        for entry in self.long_term_memories.values() {
            if predicate(entry) {
                results.push(entry);
            }
        }
        
        results
    }

    /// Get the most frequently accessed memories
    pub fn get_most_accessed(&self, limit: usize) -> Vec<&MemoryEntry> {
        let mut all_memories: Vec<&MemoryEntry> = self.short_term_memories.values()
            .chain(self.long_term_memories.values())
            .collect();

        all_memories.sort_by(|a, b| b.access_count().cmp(&a.access_count()));
        all_memories.into_iter().take(limit).collect()
    }

    /// Get recently accessed memories
    pub fn get_recently_accessed(&self, limit: usize) -> Vec<&MemoryEntry> {
        let mut all_memories: Vec<&MemoryEntry> = self.short_term_memories.values()
            .chain(self.long_term_memories.values())
            .collect();

        all_memories.sort_by(|a, b| b.last_accessed().cmp(&a.last_accessed()));
        all_memories.into_iter().take(limit).collect()
    }

    /// Update access pattern for a memory key
    fn update_access_pattern(&mut self, key: &str) {
        let now = Utc::now();
        
        let pattern = self.access_patterns.entry(key.to_string())
            .or_insert_with(|| AccessPattern {
                key: key.to_string(),
                access_count: 0,
                last_access: now,
                access_frequency: 0.0,
                average_session_length: 0.0,
            });

        pattern.access_count += 1;
        
        // Calculate frequency (accesses per hour)
        let hours_since_creation = (now - self.created_at).num_seconds() as f64 / 3600.0;
        if hours_since_creation > 0.0 {
            pattern.access_frequency = pattern.access_count as f64 / hours_since_creation;
        }
        
        pattern.last_access = now;
    }

    /// Mark the state as modified
    fn mark_modified(&mut self) {
        self.last_modified = Utc::now();
        self.version += 1;
    }
}
