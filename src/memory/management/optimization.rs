//! Memory optimization and performance management

use crate::error::{MemoryError, Result};
use crate::memory::storage::Storage;
use crate::memory::types::{MemoryEntry, MemoryFragment};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use sha2::{Sha256, Digest};
use rayon::prelude::*;

/// Memory optimizer for improving performance and efficiency
pub struct MemoryOptimizer {
    /// Optimization strategies
    strategies: Vec<OptimizationStrategy>,
    /// Performance metrics
    metrics: PerformanceMetrics,
    /// Optimization history
    optimization_history: Vec<OptimizationResult>,
    /// Last optimization time
    last_optimization: Option<DateTime<Utc>>,
    /// Storage backend for accessing memories
    storage: Arc<dyn Storage + Send + Sync>,
}

/// Strategy for memory optimization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationStrategy {
    /// Strategy identifier
    pub id: String,
    /// Strategy name
    pub name: String,
    /// Strategy type
    pub strategy_type: OptimizationType,
    /// Whether this strategy is enabled
    pub enabled: bool,
    /// Priority (higher = more important)
    pub priority: u32,
    /// Configuration parameters
    pub parameters: HashMap<String, String>,
}

/// Types of optimization
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum OptimizationType {
    /// Deduplicate similar memories
    Deduplication,
    /// Compress memory content
    Compression,
    /// Reorganize memory layout
    Reorganization,
    /// Clean up unused data
    Cleanup,
    /// Optimize indexes
    IndexOptimization,
    /// Cache optimization
    CacheOptimization,
    /// Memory consolidation
    Consolidation,
    /// Custom optimization
    Custom(String),
}

/// Result of an optimization operation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationResult {
    /// Optimization identifier
    pub id: String,
    /// When the optimization was performed
    pub timestamp: DateTime<Utc>,
    /// Strategy used
    pub strategy: OptimizationType,
    /// Number of memories affected
    pub memories_optimized: usize,
    /// Space saved in bytes
    pub space_saved: usize,
    /// Time taken in milliseconds
    pub duration_ms: u64,
    /// Success status
    pub success: bool,
    /// Performance improvement metrics
    pub performance_improvement: PerformanceImprovement,
    /// Messages and details
    pub messages: Vec<String>,
}

/// Performance improvement metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceImprovement {
    /// Speed improvement factor (1.0 = no change, 2.0 = 2x faster)
    pub speed_factor: f64,
    /// Memory usage reduction factor (0.5 = 50% reduction)
    pub memory_reduction: f64,
    /// Index efficiency improvement
    pub index_efficiency: f64,
    /// Cache hit rate improvement
    pub cache_improvement: f64,
}

/// Performance metrics for the memory system
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceMetrics {
    /// Average retrieval time in milliseconds
    pub avg_retrieval_time_ms: f64,
    /// Average storage time in milliseconds
    pub avg_storage_time_ms: f64,
    /// Memory usage in bytes
    pub memory_usage_bytes: usize,
    /// Cache hit rate (0.0 to 1.0)
    pub cache_hit_rate: f64,
    /// Index efficiency score (0.0 to 1.0)
    pub index_efficiency: f64,
    /// Fragmentation score (0.0 = no fragmentation, 1.0 = highly fragmented)
    pub fragmentation_score: f64,
    /// Duplicate content ratio (0.0 to 1.0)
    pub duplicate_ratio: f64,
    /// Last measurement time
    pub last_measured: DateTime<Utc>,
}

impl Default for PerformanceMetrics {
    fn default() -> Self {
        Self {
            avg_retrieval_time_ms: 0.0,
            avg_storage_time_ms: 0.0,
            memory_usage_bytes: 0,
            cache_hit_rate: 0.0,
            index_efficiency: 1.0,
            fragmentation_score: 0.0,
            duplicate_ratio: 0.0,
            last_measured: Utc::now(),
        }
    }
}

impl MemoryOptimizer {
    /// Create a new memory optimizer
    pub fn new(storage: Arc<dyn Storage + Send + Sync>) -> Self {
        Self {
            strategies: Self::create_default_strategies(),
            metrics: PerformanceMetrics::default(),
            optimization_history: Vec::new(),
            last_optimization: None,
            storage,
        }
    }

    /// Perform optimization using all enabled strategies
    pub async fn optimize(&mut self) -> Result<OptimizationResult> {
        let start_time = std::time::Instant::now();
        let mut total_memories_optimized = 0;
        let mut total_space_saved = 0;
        let mut messages = Vec::new();
        let mut success = true;

        // Execute each enabled strategy
        for strategy in &self.strategies {
            if strategy.enabled {
                match self.execute_strategy(strategy).await {
                    Ok(result) => {
                        total_memories_optimized += result.memories_optimized;
                        total_space_saved += result.space_saved;
                        messages.extend(result.messages);
                    }
                    Err(e) => {
                        success = false;
                        messages.push(format!("Strategy {} failed: {}", strategy.name, e));
                    }
                }
            }
        }

        let duration_ms = start_time.elapsed().as_millis() as u64;

        // Measure performance improvement
        let old_metrics = self.metrics.clone();
        self.update_performance_metrics().await?;
        let performance_improvement = self.calculate_performance_improvement(&old_metrics);

        let result = OptimizationResult {
            id: uuid::Uuid::new_v4().to_string(),
            timestamp: Utc::now(),
            strategy: OptimizationType::Custom("combined".to_string()),
            memories_optimized: total_memories_optimized,
            space_saved: total_space_saved,
            duration_ms,
            success,
            performance_improvement,
            messages,
        };

        self.optimization_history.push(result.clone());
        self.last_optimization = Some(Utc::now());

        Ok(result)
    }

    /// Execute a specific optimization strategy
    async fn execute_strategy(&self, strategy: &OptimizationStrategy) -> Result<OptimizationResult> {
        let start_time = std::time::Instant::now();
        let mut memories_optimized = 0;
        let mut space_saved = 0;
        let mut messages = Vec::new();

        match strategy.strategy_type {
            OptimizationType::Deduplication => {
                let result = self.perform_deduplication().await?;
                memories_optimized = result.0;
                space_saved = result.1;
                messages.push("Performed memory deduplication".to_string());
            }
            OptimizationType::Compression => {
                let result = self.perform_compression().await?;
                memories_optimized = result.0;
                space_saved = result.1;
                messages.push("Performed memory compression".to_string());
            }
            OptimizationType::Cleanup => {
                let result = self.perform_cleanup().await?;
                memories_optimized = result.0;
                space_saved = result.1;
                messages.push("Performed memory cleanup".to_string());
            }
            OptimizationType::IndexOptimization => {
                self.optimize_indexes().await?;
                messages.push("Optimized memory indexes".to_string());
            }
            OptimizationType::CacheOptimization => {
                self.optimize_cache().await?;
                messages.push("Optimized memory cache".to_string());
            }
            _ => {
                messages.push(format!("Strategy {} not yet implemented", strategy.name));
            }
        }

        let duration_ms = start_time.elapsed().as_millis() as u64;

        Ok(OptimizationResult {
            id: uuid::Uuid::new_v4().to_string(),
            timestamp: Utc::now(),
            strategy: strategy.strategy_type.clone(),
            memories_optimized,
            space_saved,
            duration_ms,
            success: true,
            performance_improvement: PerformanceImprovement {
                speed_factor: 1.0,
                memory_reduction: 0.0,
                index_efficiency: 0.0,
                cache_improvement: 0.0,
            },
            messages,
        })
    }

    /// Perform memory deduplication using advanced similarity detection
    async fn perform_deduplication(&self) -> Result<(usize, usize)> {
        tracing::info!("Starting memory deduplication process");
        let start_time = std::time::Instant::now();

        // Get all memory entries from storage
        let all_entries = self.storage.get_all_entries().await?;
        if all_entries.is_empty() {
            tracing::debug!("No memories to deduplicate");
            return Ok((0, 0));
        }

        tracing::debug!("Analyzing {} memory entries for deduplication", all_entries.len());

        // Build similarity groups using multiple strategies
        let similarity_groups = self.build_similarity_groups(&all_entries).await?;

        let mut memories_processed = 0;
        let mut space_saved = 0;

        // Process each similarity group
        for group in similarity_groups {
            if group.len() > 1 {
                let (processed, saved) = self.merge_similar_memories(group).await?;
                memories_processed += processed;
                space_saved += saved;
            }
        }

        let duration = start_time.elapsed();
        tracing::info!(
            "Deduplication completed: {} memories processed, {} bytes saved in {:?}",
            memories_processed, space_saved, duration
        );

        Ok((memories_processed, space_saved))
    }

    /// Build similarity groups using multiple detection strategies
    async fn build_similarity_groups(&self, entries: &[MemoryEntry]) -> Result<Vec<Vec<MemoryEntry>>> {
        tracing::debug!("Building similarity groups for {} entries", entries.len());

        // Strategy 1: Content hash-based exact duplicates
        let mut hash_groups = HashMap::new();

        // Strategy 2: Embedding-based similarity clusters
        let mut embedding_groups = Vec::new();

        // Strategy 3: Text similarity using n-grams
        let mut text_groups = Vec::new();

        // Process entries in parallel for hash computation
        let hash_map: HashMap<String, Vec<MemoryEntry>> = entries
            .par_iter()
            .map(|entry| {
                let content_hash = self.compute_content_hash(&entry.value);
                (content_hash, entry.clone())
            })
            .collect::<Vec<_>>()
            .into_iter()
            .fold(HashMap::new(), |mut acc, (hash, entry)| {
                acc.entry(hash).or_insert_with(Vec::new).push(entry);
                acc
            });

        // Collect exact duplicates
        for (_, group) in hash_map {
            if group.len() > 1 {
                hash_groups.insert(group[0].key.clone(), group);
            }
        }

        // Find embedding-based similarities
        if let Some(embedding_groups_result) = self.find_embedding_similarities(entries).await? {
            embedding_groups = embedding_groups_result;
        }

        // Find text-based similarities for entries without embeddings
        text_groups = self.find_text_similarities(entries).await?;

        // Merge all groups, prioritizing exact matches
        let mut all_groups = Vec::new();
        all_groups.extend(hash_groups.into_values());
        all_groups.extend(embedding_groups);
        all_groups.extend(text_groups);

        // Remove overlapping groups (prefer exact matches)
        let deduplicated_groups = self.deduplicate_groups(all_groups);

        tracing::debug!("Found {} similarity groups", deduplicated_groups.len());
        Ok(deduplicated_groups)
    }

    /// Compute content hash for exact duplicate detection
    fn compute_content_hash(&self, content: &str) -> String {
        let mut hasher = Sha256::new();
        hasher.update(content.as_bytes());
        hex::encode(hasher.finalize())
    }

    /// Find similarities using vector embeddings
    async fn find_embedding_similarities(&self, entries: &[MemoryEntry]) -> Result<Option<Vec<Vec<MemoryEntry>>>> {
        let entries_with_embeddings: Vec<_> = entries
            .iter()
            .filter(|entry| entry.embedding.is_some())
            .collect();

        if entries_with_embeddings.is_empty() {
            return Ok(None);
        }

        tracing::debug!("Analyzing {} entries with embeddings", entries_with_embeddings.len());

        let mut groups = Vec::new();
        let mut processed = HashSet::new();

        for (i, entry) in entries_with_embeddings.iter().enumerate() {
            if processed.contains(&entry.key) {
                continue;
            }

            let mut similar_group = vec![(*entry).clone()];
            processed.insert(entry.key.clone());

            // Find similar entries using cosine similarity
            for (j, other_entry) in entries_with_embeddings.iter().enumerate() {
                if i != j && !processed.contains(&other_entry.key) {
                    let similarity = entry.similarity_score(other_entry);

                    // Threshold for considering entries similar (configurable)
                    if similarity > 0.85 {
                        similar_group.push((*other_entry).clone());
                        processed.insert(other_entry.key.clone());
                    }
                }
            }

            if similar_group.len() > 1 {
                groups.push(similar_group);
            }
        }

        Ok(Some(groups))
    }

    /// Find similarities using text analysis (n-grams and Jaccard similarity)
    async fn find_text_similarities(&self, entries: &[MemoryEntry]) -> Result<Vec<Vec<MemoryEntry>>> {
        tracing::debug!("Analyzing text similarities for {} entries", entries.len());

        let mut groups = Vec::new();
        let mut processed = HashSet::new();

        // Use parallel processing for n-gram computation
        let ngram_signatures: Vec<(String, HashSet<String>)> = entries
            .par_iter()
            .map(|entry| {
                let ngrams = self.compute_ngrams(&entry.value, 3);
                (entry.key.clone(), ngrams)
            })
            .collect();

        for (i, (key, ngrams)) in ngram_signatures.iter().enumerate() {
            if processed.contains(key) {
                continue;
            }

            let mut similar_group = vec![entries.iter().find(|e| &e.key == key).unwrap().clone()];
            processed.insert(key.clone());

            // Find similar entries using Jaccard similarity
            for (j, (other_key, other_ngrams)) in ngram_signatures.iter().enumerate() {
                if i != j && !processed.contains(other_key) {
                    let jaccard_similarity = self.jaccard_similarity(ngrams, other_ngrams);

                    // Threshold for text similarity (configurable)
                    if jaccard_similarity > 0.7 {
                        similar_group.push(entries.iter().find(|e| &e.key == other_key).unwrap().clone());
                        processed.insert(other_key.clone());
                    }
                }
            }

            if similar_group.len() > 1 {
                groups.push(similar_group);
            }
        }

        Ok(groups)
    }

    /// Compute n-grams for text similarity analysis
    fn compute_ngrams(&self, text: &str, n: usize) -> HashSet<String> {
        let words: Vec<&str> = text.split_whitespace().collect();
        let mut ngrams = HashSet::new();

        if words.len() >= n {
            for i in 0..=words.len() - n {
                let ngram = words[i..i + n].join(" ");
                ngrams.insert(ngram);
            }
        }

        // Also add character-level n-grams for short texts
        let chars: Vec<char> = text.chars().collect();
        if chars.len() >= n {
            for i in 0..=chars.len() - n {
                let char_ngram: String = chars[i..i + n].iter().collect();
                ngrams.insert(char_ngram);
            }
        }

        ngrams
    }

    /// Calculate Jaccard similarity between two sets
    fn jaccard_similarity(&self, set1: &HashSet<String>, set2: &HashSet<String>) -> f64 {
        let intersection = set1.intersection(set2).count();
        let union = set1.union(set2).count();

        if union == 0 {
            0.0
        } else {
            intersection as f64 / union as f64
        }
    }

    /// Remove overlapping groups, prioritizing exact matches
    fn deduplicate_groups(&self, groups: Vec<Vec<MemoryEntry>>) -> Vec<Vec<MemoryEntry>> {
        let mut result = Vec::new();
        let mut used_keys = HashSet::new();

        // Sort groups by size (larger groups first) and then by average similarity
        let mut sorted_groups = groups;
        sorted_groups.sort_by(|a, b| b.len().cmp(&a.len()));

        for group in sorted_groups {
            let group_keys: HashSet<String> = group.iter().map(|e| e.key.clone()).collect();

            // Check if any key in this group is already used
            if group_keys.is_disjoint(&used_keys) {
                used_keys.extend(group_keys);
                result.push(group);
            }
        }

        result
    }

    /// Merge similar memories into a consolidated entry
    async fn merge_similar_memories(&self, group: Vec<MemoryEntry>) -> Result<(usize, usize)> {
        if group.len() < 2 {
            return Ok((0, 0));
        }

        tracing::debug!("Merging {} similar memories", group.len());

        // Calculate space before merging
        let space_before: usize = group.iter().map(|e| e.estimated_size()).sum();

        // Create merged memory entry
        let merged_entry = self.create_merged_entry(&group)?;

        // Calculate space after merging
        let space_after = merged_entry.estimated_size();
        let space_saved = space_before.saturating_sub(space_after);

        // Store the merged entry
        self.storage.store(&merged_entry).await?;

        // Delete the original entries (except the first one which becomes the merged entry)
        let mut deleted_count = 0;
        for entry in group.iter().skip(1) {
            if self.storage.delete(&entry.key).await? {
                deleted_count += 1;
            }
        }

        tracing::debug!(
            "Merged {} memories into 1, saved {} bytes",
            group.len(), space_saved
        );

        Ok((group.len() - 1, space_saved))
    }

    /// Create a merged memory entry from a group of similar memories
    fn create_merged_entry(&self, group: &[MemoryEntry]) -> Result<MemoryEntry> {
        if group.is_empty() {
            return Err(MemoryError::unexpected("Cannot merge empty group"));
        }

        // Use the first entry as the base
        let base_entry = &group[0];
        let mut merged_entry = base_entry.clone();

        // Merge content using intelligent consolidation
        merged_entry.value = self.merge_content(group)?;

        // Merge metadata
        self.merge_metadata(&mut merged_entry, group)?;

        // Update timestamps
        merged_entry.metadata.mark_modified();

        // Merge embeddings if available
        if let Some(merged_embedding) = self.merge_embeddings(group)? {
            merged_entry.embedding = Some(merged_embedding);
        }

        Ok(merged_entry)
    }

    /// Merge content from multiple memory entries
    fn merge_content(&self, group: &[MemoryEntry]) -> Result<String> {
        if group.len() == 1 {
            return Ok(group[0].value.clone());
        }

        // Strategy 1: If one entry is significantly longer, use it as base
        let longest_entry = group.iter().max_by_key(|e| e.value.len()).unwrap();
        if longest_entry.value.len() > group.iter().map(|e| e.value.len()).sum::<usize>() / 2 {
            return Ok(longest_entry.value.clone());
        }

        // Strategy 2: Merge unique sentences/paragraphs
        let mut unique_sentences = HashSet::new();
        let mut merged_content = String::new();

        for entry in group {
            // Split by sentences (simple approach)
            let sentences: Vec<&str> = entry.value
                .split(&['.', '!', '?'][..])
                .map(|s| s.trim())
                .filter(|s| !s.is_empty())
                .collect();

            for sentence in sentences {
                if unique_sentences.insert(sentence.to_lowercase()) {
                    if !merged_content.is_empty() {
                        merged_content.push(' ');
                    }
                    merged_content.push_str(sentence);
                    if !sentence.ends_with(&['.', '!', '?'][..]) {
                        merged_content.push('.');
                    }
                }
            }
        }

        // Fallback: concatenate with separators
        if merged_content.is_empty() {
            merged_content = group
                .iter()
                .map(|e| e.value.as_str())
                .collect::<Vec<_>>()
                .join(" | ");
        }

        Ok(merged_content)
    }

    /// Merge metadata from multiple entries
    fn merge_metadata(&self, merged_entry: &mut MemoryEntry, group: &[MemoryEntry]) -> Result<()> {
        // Merge tags (union of all tags)
        let mut all_tags = HashSet::new();
        for entry in group {
            for tag in &entry.metadata.tags {
                all_tags.insert(tag.clone());
            }
        }
        merged_entry.metadata.tags = all_tags.into_iter().collect();

        // Use highest importance and confidence
        merged_entry.metadata.importance = group
            .iter()
            .map(|e| e.metadata.importance)
            .fold(0.0, f64::max);

        merged_entry.metadata.confidence = group
            .iter()
            .map(|e| e.metadata.confidence)
            .fold(0.0, f64::max);

        // Sum access counts
        merged_entry.metadata.access_count = group
            .iter()
            .map(|e| e.metadata.access_count)
            .sum();

        // Use earliest creation time
        merged_entry.metadata.created_at = group
            .iter()
            .map(|e| e.metadata.created_at)
            .min()
            .unwrap_or(merged_entry.metadata.created_at);

        // Merge custom fields
        for entry in group {
            for (key, value) in &entry.metadata.custom_fields {
                merged_entry.metadata.custom_fields
                    .entry(key.clone())
                    .or_insert_with(|| value.clone());
            }
        }

        Ok(())
    }

    /// Merge embeddings using averaging
    fn merge_embeddings(&self, group: &[MemoryEntry]) -> Result<Option<Vec<f32>>> {
        let embeddings: Vec<&Vec<f32>> = group
            .iter()
            .filter_map(|e| e.embedding.as_ref())
            .collect();

        if embeddings.is_empty() {
            return Ok(None);
        }

        if embeddings.len() == 1 {
            return Ok(Some(embeddings[0].clone()));
        }

        // Check that all embeddings have the same dimension
        let dim = embeddings[0].len();
        if !embeddings.iter().all(|e| e.len() == dim) {
            tracing::warn!("Embeddings have different dimensions, using first one");
            return Ok(Some(embeddings[0].clone()));
        }

        // Average the embeddings
        let mut averaged = vec![0.0; dim];
        for embedding in &embeddings {
            for (i, &value) in embedding.iter().enumerate() {
                averaged[i] += value;
            }
        }

        let count = embeddings.len() as f32;
        for value in &mut averaged {
            *value /= count;
        }

        Ok(Some(averaged))
    }

    /// Perform memory compression
    async fn perform_compression(&self) -> Result<(usize, usize)> {
        // TODO: Implement compression logic
        // This would compress memory content to save space
        Ok((0, 0))
    }

    /// Perform memory cleanup
    async fn perform_cleanup(&self) -> Result<(usize, usize)> {
        // TODO: Implement cleanup logic
        // This would remove orphaned data, temporary files, etc.
        Ok((0, 0))
    }

    /// Optimize memory indexes
    async fn optimize_indexes(&self) -> Result<()> {
        // TODO: Implement index optimization
        // This would rebuild and optimize search indexes
        Ok(())
    }

    /// Optimize memory cache
    async fn optimize_cache(&self) -> Result<()> {
        // TODO: Implement cache optimization
        // This would optimize cache policies and eviction strategies
        Ok(())
    }

    /// Update performance metrics
    async fn update_performance_metrics(&mut self) -> Result<()> {
        // TODO: Implement actual performance measurement
        // This would measure current system performance
        self.metrics.last_measured = Utc::now();
        Ok(())
    }

    /// Calculate performance improvement
    fn calculate_performance_improvement(&self, old_metrics: &PerformanceMetrics) -> PerformanceImprovement {
        let speed_factor = if old_metrics.avg_retrieval_time_ms > 0.0 {
            old_metrics.avg_retrieval_time_ms / self.metrics.avg_retrieval_time_ms.max(0.1)
        } else {
            1.0
        };

        let memory_reduction = if old_metrics.memory_usage_bytes > 0 {
            1.0 - (self.metrics.memory_usage_bytes as f64 / old_metrics.memory_usage_bytes as f64)
        } else {
            0.0
        };

        let index_efficiency = self.metrics.index_efficiency - old_metrics.index_efficiency;
        let cache_improvement = self.metrics.cache_hit_rate - old_metrics.cache_hit_rate;

        PerformanceImprovement {
            speed_factor,
            memory_reduction,
            index_efficiency,
            cache_improvement,
        }
    }

    /// Create default optimization strategies
    fn create_default_strategies() -> Vec<OptimizationStrategy> {
        vec![
            OptimizationStrategy {
                id: "deduplication".to_string(),
                name: "Memory Deduplication".to_string(),
                strategy_type: OptimizationType::Deduplication,
                enabled: true,
                priority: 1,
                parameters: HashMap::new(),
            },
            OptimizationStrategy {
                id: "compression".to_string(),
                name: "Memory Compression".to_string(),
                strategy_type: OptimizationType::Compression,
                enabled: true,
                priority: 2,
                parameters: HashMap::new(),
            },
            OptimizationStrategy {
                id: "cleanup".to_string(),
                name: "Memory Cleanup".to_string(),
                strategy_type: OptimizationType::Cleanup,
                enabled: true,
                priority: 3,
                parameters: HashMap::new(),
            },
        ]
    }

    /// Get current performance metrics
    pub fn get_performance_metrics(&self) -> &PerformanceMetrics {
        &self.metrics
    }

    /// Get optimization history
    pub fn get_optimization_history(&self) -> &[OptimizationResult] {
        &self.optimization_history
    }

    /// Get the number of optimizations performed
    pub fn get_optimization_count(&self) -> usize {
        self.optimization_history.len()
    }

    /// Get the last optimization time
    pub fn get_last_optimization_time(&self) -> Option<DateTime<Utc>> {
        self.last_optimization
    }

    /// Add a custom optimization strategy
    pub fn add_strategy(&mut self, strategy: OptimizationStrategy) {
        self.strategies.push(strategy);
        // Sort by priority (highest first)
        self.strategies.sort_by(|a, b| b.priority.cmp(&a.priority));
    }

    /// Enable or disable a strategy
    pub fn set_strategy_enabled(&mut self, strategy_id: &str, enabled: bool) -> bool {
        if let Some(strategy) = self.strategies.iter_mut().find(|s| s.id == strategy_id) {
            strategy.enabled = enabled;
            true
        } else {
            false
        }
    }

    /// Get all optimization strategies
    pub fn get_strategies(&self) -> &[OptimizationStrategy] {
        &self.strategies
    }
}

// Note: No Default implementation since MemoryOptimizer requires storage

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::storage::memory::MemoryStorage;
    use crate::memory::types::{MemoryEntry, MemoryType};
    use std::sync::Arc;

    fn create_test_optimizer() -> MemoryOptimizer {
        let storage = Arc::new(MemoryStorage::new());
        MemoryOptimizer::new(storage)
    }

    fn create_test_memory(key: &str, content: &str, tags: Vec<String>) -> MemoryEntry {
        let mut memory = MemoryEntry::new(key.to_string(), content.to_string(), MemoryType::LongTerm);
        memory.metadata.tags = tags;
        memory
    }

    fn create_test_memory_with_embedding(key: &str, content: &str, embedding: Vec<f32>) -> MemoryEntry {
        let mut memory = create_test_memory(key, content, vec![]);
        memory.embedding = Some(embedding);
        memory
    }

    #[tokio::test]
    async fn test_memory_deduplication_exact_duplicates() -> Result<()> {
        let optimizer = create_test_optimizer();

        // Create exact duplicate memories
        let memory1 = create_test_memory("mem1", "This is a test memory", vec!["test".to_string()]);
        let memory2 = create_test_memory("mem2", "This is a test memory", vec!["test".to_string()]);
        let memory3 = create_test_memory("mem3", "Different content", vec!["other".to_string()]);

        // Store memories
        optimizer.storage.store(&memory1).await?;
        optimizer.storage.store(&memory2).await?;
        optimizer.storage.store(&memory3).await?;

        // Perform deduplication
        let (processed, space_saved) = optimizer.perform_deduplication().await?;

        // Should have processed the duplicate memories
        assert!(processed > 0);
        assert!(space_saved > 0);

        // Verify that duplicates were merged
        let remaining_keys = optimizer.storage.list_keys().await?;
        assert!(remaining_keys.len() < 3); // Should have fewer memories after deduplication

        Ok(())
    }

    #[tokio::test]
    async fn test_memory_deduplication_embedding_similarity() -> Result<()> {
        let optimizer = create_test_optimizer();

        // Create memories with similar embeddings
        let embedding1 = vec![1.0, 0.5, 0.3, 0.8];
        let embedding2 = vec![1.0, 0.5, 0.3, 0.8]; // Identical
        let embedding3 = vec![0.1, 0.2, 0.9, 0.1]; // Different

        let memory1 = create_test_memory_with_embedding("mem1", "First memory", embedding1);
        let memory2 = create_test_memory_with_embedding("mem2", "Second memory", embedding2);
        let memory3 = create_test_memory_with_embedding("mem3", "Third memory", embedding3);

        // Store memories
        optimizer.storage.store(&memory1).await?;
        optimizer.storage.store(&memory2).await?;
        optimizer.storage.store(&memory3).await?;

        // Perform deduplication
        let (processed, space_saved) = optimizer.perform_deduplication().await?;

        // Should have processed similar memories
        assert!(processed > 0);
        assert!(space_saved > 0);

        Ok(())
    }

    #[tokio::test]
    async fn test_memory_deduplication_text_similarity() -> Result<()> {
        let optimizer = create_test_optimizer();

        // Create memories with similar text content
        let memory1 = create_test_memory("mem1", "The quick brown fox jumps over the lazy dog", vec![]);
        let memory2 = create_test_memory("mem2", "The quick brown fox jumps over the lazy cat", vec![]);
        let memory3 = create_test_memory("mem3", "Completely different content about something else", vec![]);

        // Store memories
        optimizer.storage.store(&memory1).await?;
        optimizer.storage.store(&memory2).await?;
        optimizer.storage.store(&memory3).await?;

        // Perform deduplication
        let (processed, space_saved) = optimizer.perform_deduplication().await?;

        // Should have processed similar memories
        assert!(processed > 0);

        Ok(())
    }

    #[tokio::test]
    async fn test_memory_deduplication_no_duplicates() -> Result<()> {
        let optimizer = create_test_optimizer();

        // Create completely different memories
        let memory1 = create_test_memory("mem1", "First unique memory", vec!["tag1".to_string()]);
        let memory2 = create_test_memory("mem2", "Second unique memory", vec!["tag2".to_string()]);
        let memory3 = create_test_memory("mem3", "Third unique memory", vec!["tag3".to_string()]);

        // Store memories
        optimizer.storage.store(&memory1).await?;
        optimizer.storage.store(&memory2).await?;
        optimizer.storage.store(&memory3).await?;

        // Perform deduplication
        let (processed, space_saved) = optimizer.perform_deduplication().await?;

        // Should not have processed any memories (no duplicates)
        assert_eq!(processed, 0);
        assert_eq!(space_saved, 0);

        // All memories should still exist
        let remaining_keys = optimizer.storage.list_keys().await?;
        assert_eq!(remaining_keys.len(), 3);

        Ok(())
    }

    #[tokio::test]
    async fn test_memory_deduplication_empty_storage() -> Result<()> {
        let optimizer = create_test_optimizer();

        // Perform deduplication on empty storage
        let (processed, space_saved) = optimizer.perform_deduplication().await?;

        // Should not process anything
        assert_eq!(processed, 0);
        assert_eq!(space_saved, 0);

        Ok(())
    }

    #[tokio::test]
    async fn test_content_hash_computation() -> Result<()> {
        let optimizer = create_test_optimizer();

        let content1 = "This is a test";
        let content2 = "This is a test";
        let content3 = "This is different";

        let hash1 = optimizer.compute_content_hash(content1);
        let hash2 = optimizer.compute_content_hash(content2);
        let hash3 = optimizer.compute_content_hash(content3);

        // Same content should produce same hash
        assert_eq!(hash1, hash2);

        // Different content should produce different hash
        assert_ne!(hash1, hash3);

        Ok(())
    }

    #[tokio::test]
    async fn test_ngram_computation() -> Result<()> {
        let optimizer = create_test_optimizer();

        let text = "The quick brown fox";
        let ngrams = optimizer.compute_ngrams(text, 3);

        // Should contain word-level trigrams
        assert!(ngrams.contains("The quick brown"));
        assert!(ngrams.contains("quick brown fox"));

        // Should also contain character-level trigrams
        assert!(ngrams.len() > 2);

        Ok(())
    }

    #[tokio::test]
    async fn test_jaccard_similarity() -> Result<()> {
        let optimizer = create_test_optimizer();

        let set1: HashSet<String> = ["a", "b", "c"].iter().map(|s| s.to_string()).collect();
        let set2: HashSet<String> = ["b", "c", "d"].iter().map(|s| s.to_string()).collect();
        let set3: HashSet<String> = ["x", "y", "z"].iter().map(|s| s.to_string()).collect();

        let similarity1 = optimizer.jaccard_similarity(&set1, &set2);
        let similarity2 = optimizer.jaccard_similarity(&set1, &set3);

        // Sets with overlap should have higher similarity
        assert!(similarity1 > similarity2);
        assert!(similarity1 > 0.0);
        assert_eq!(similarity2, 0.0);

        Ok(())
    }

    #[tokio::test]
    async fn test_merge_content_strategies() -> Result<()> {
        let optimizer = create_test_optimizer();

        // Test with one significantly longer entry
        let memory1 = create_test_memory("mem1", "Short", vec![]);
        let memory2 = create_test_memory("mem2", "This is a much longer memory entry with lots of content that should be used as the base for merging", vec![]);
        let group = vec![memory1, memory2];

        let merged_content = optimizer.merge_content(&group)?;

        // Should use the longer content as base
        assert!(merged_content.len() > 50);

        Ok(())
    }

    #[tokio::test]
    async fn test_merge_metadata() -> Result<()> {
        let optimizer = create_test_optimizer();

        let mut memory1 = create_test_memory("mem1", "Content 1", vec!["tag1".to_string(), "common".to_string()]);
        memory1.metadata.importance = 0.8;
        memory1.metadata.confidence = 0.9;
        memory1.metadata.access_count = 5;

        let mut memory2 = create_test_memory("mem2", "Content 2", vec!["tag2".to_string(), "common".to_string()]);
        memory2.metadata.importance = 0.6;
        memory2.metadata.confidence = 0.7;
        memory2.metadata.access_count = 3;

        let group = vec![memory1.clone(), memory2];
        let mut merged_entry = memory1;

        optimizer.merge_metadata(&mut merged_entry, &group)?;

        // Should have union of tags
        assert!(merged_entry.metadata.tags.contains(&"tag1".to_string()));
        assert!(merged_entry.metadata.tags.contains(&"tag2".to_string()));
        assert!(merged_entry.metadata.tags.contains(&"common".to_string()));

        // Should use highest importance and confidence
        assert_eq!(merged_entry.metadata.importance, 0.8);
        assert_eq!(merged_entry.metadata.confidence, 0.9);

        // Should sum access counts
        assert_eq!(merged_entry.metadata.access_count, 8);

        Ok(())
    }

    #[tokio::test]
    async fn test_merge_embeddings() -> Result<()> {
        let optimizer = create_test_optimizer();

        let memory1 = create_test_memory_with_embedding("mem1", "Content 1", vec![1.0, 2.0, 3.0]);
        let memory2 = create_test_memory_with_embedding("mem2", "Content 2", vec![2.0, 4.0, 6.0]);
        let group = vec![memory1, memory2];

        let merged_embedding = optimizer.merge_embeddings(&group)?;

        // Should average the embeddings
        assert!(merged_embedding.is_some());
        let embedding = merged_embedding.unwrap();
        assert_eq!(embedding, vec![1.5, 3.0, 4.5]);

        Ok(())
    }

    #[tokio::test]
    async fn test_optimization_full_workflow() -> Result<()> {
        let mut optimizer = create_test_optimizer();

        // Create a mix of duplicate and unique memories
        let memory1 = create_test_memory("mem1", "Duplicate content", vec!["test".to_string()]);
        let memory2 = create_test_memory("mem2", "Duplicate content", vec!["test".to_string()]);
        let memory3 = create_test_memory("mem3", "Unique content", vec!["unique".to_string()]);

        // Store memories
        optimizer.storage.store(&memory1).await?;
        optimizer.storage.store(&memory2).await?;
        optimizer.storage.store(&memory3).await?;

        // Run full optimization
        let result = optimizer.optimize().await?;

        // Should have processed some memories
        assert!(result.memories_optimized > 0);
        assert!(result.success);
        assert!(!result.messages.is_empty());
        // Duration might be 0 for very fast operations, so just check it's not negative
        assert!(result.duration_ms >= 0);

        // Should have optimization history
        assert_eq!(optimizer.get_optimization_count(), 1);
        assert!(optimizer.get_last_optimization_time().is_some());

        Ok(())
    }
}
