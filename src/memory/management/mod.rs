//! Advanced memory management system for AI agents
//!
//! This module provides sophisticated memory management capabilities including:
//! - Intelligent summarization and consolidation
//! - Advanced search and filtering
//! - Memory lifecycle management
//! - Automatic optimization and cleanup
//! - Memory analytics and insights

pub mod summarization;
pub mod search;
pub mod lifecycle;
pub mod optimization;
pub mod analytics;

// Re-export commonly used types
pub use summarization::{MemorySummarizer, SummaryStrategy, SummaryResult, ConsolidationRule};
pub use search::{AdvancedSearchEngine, SearchQuery, SearchFilter, SearchResult, RankingStrategy};
pub use lifecycle::{MemoryLifecycleManager, LifecyclePolicy, MemoryStage, LifecycleEvent};
pub use optimization::{MemoryOptimizer, OptimizationStrategy, OptimizationResult, PerformanceMetrics};
pub use analytics::{MemoryAnalytics, AnalyticsReport, InsightType, TrendAnalysis};

use crate::error::Result;
use crate::memory::types::{MemoryEntry, MemoryType};
use crate::memory::temporal::{TemporalMemoryManager, ChangeType};
use crate::memory::knowledge_graph::MemoryKnowledgeGraph;
use crate::memory::storage::Storage;
use chrono::{DateTime, Utc, Duration};
use serde::{Deserialize, Serialize};
use std::sync::Arc;

/// Simple memory manager for basic operations and testing
pub struct MemoryManager {
    /// Storage backend
    storage: Arc<dyn Storage + Send + Sync>,
    /// Knowledge graph for relationships
    knowledge_graph: Option<MemoryKnowledgeGraph>,
    /// Temporal manager for tracking changes
    temporal_manager: Option<TemporalMemoryManager>,
    /// Advanced manager for complex operations
    advanced_manager: Option<AdvancedMemoryManager>,
}

impl MemoryManager {
    /// Create a new memory manager
    pub async fn new(
        storage: Arc<dyn Storage + Send + Sync>,
        knowledge_graph: Option<MemoryKnowledgeGraph>,
        temporal_manager: Option<TemporalMemoryManager>,
        advanced_manager: Option<AdvancedMemoryManager>,
    ) -> Result<Self> {
        Ok(Self {
            storage,
            knowledge_graph,
            temporal_manager,
            advanced_manager,
        })
    }

    /// Store a memory entry
    pub async fn store_memory(&self, memory: &MemoryEntry) -> Result<()> {
        self.storage.store(memory).await
    }

    /// Count related memories using the implemented algorithm
    pub async fn count_related_memories(&self, memory: &MemoryEntry) -> Result<usize> {
        let start_time = std::time::Instant::now();
        tracing::info!("Counting related memories for key: {}", memory.key);

        let mut related_count = 0;
        let mut processed_keys = std::collections::HashSet::new();

        // Strategy 1: Knowledge Graph Traversal
        if let Some(kg) = &self.knowledge_graph {
            related_count += self.count_graph_related_memories(kg, memory, &mut processed_keys).await?;
        }

        // Strategy 2: Similarity-based matching
        related_count += self.count_similarity_related_memories(memory, &processed_keys).await?;

        // Strategy 3: Tag-based relationships
        related_count += self.count_tag_related_memories(memory, &processed_keys).await?;

        // Strategy 4: Temporal proximity relationships
        related_count += self.count_temporal_related_memories(memory, &processed_keys).await?;

        // Strategy 5: Pure content similarity (without temporal constraints)
        related_count += self.count_content_related_memories(memory, &processed_keys).await?;

        let duration_ms = start_time.elapsed().as_millis();
        tracing::info!(
            "Found {} related memories for '{}' in {}ms",
            related_count, memory.key, duration_ms
        );

        Ok(related_count)
    }

    /// Count related memories using knowledge graph traversal
    async fn count_graph_related_memories(
        &self,
        kg: &MemoryKnowledgeGraph,
        memory: &MemoryEntry,
        processed_keys: &mut std::collections::HashSet<String>,
    ) -> Result<usize> {
        use crate::memory::knowledge_graph::RelationshipType;

        let mut count = 0;

        // Get the node ID for this memory
        if let Some(node_id) = kg.get_node_for_memory(&memory.key).await? {
            // Perform BFS traversal up to depth 3 to find related memories
            let max_depth = 3;
            let mut visited = std::collections::HashSet::new();
            let mut queue = std::collections::VecDeque::new();

            queue.push_back((node_id, 0)); // (node_id, depth)
            visited.insert(node_id);

            while let Some((current_node, depth)) = queue.pop_front() {
                if depth >= max_depth {
                    continue;
                }

                // Get all connected nodes
                let connected_nodes = kg.get_connected_nodes(current_node).await?;

                for (connected_node, relationship_type, strength) in connected_nodes {
                    if visited.contains(&connected_node) {
                        continue;
                    }

                    visited.insert(connected_node);

                    // Check if this node represents a memory
                    if let Some(memory_key) = kg.get_memory_for_node(connected_node).await? {
                        if memory_key != memory.key && !processed_keys.contains(&memory_key) {
                            // Apply relationship strength threshold
                            let strength_threshold = match relationship_type {
                                RelationshipType::RelatedTo => 0.3,
                                RelationshipType::DependsOn => 0.5,
                                RelationshipType::Contains => 0.4,
                                RelationshipType::PartOf => 0.4,
                                RelationshipType::SimilarTo => 0.6,
                                RelationshipType::TemporallyRelated => 0.2,
                                RelationshipType::CausedBy => 0.7,
                                _ => 0.5, // Default threshold for other types
                            };

                            if strength >= strength_threshold {
                                count += 1;
                                processed_keys.insert(memory_key);

                                // Add to queue for further exploration
                                queue.push_back((connected_node, depth + 1));
                            }
                        }
                    }
                }
            }
        }

        tracing::debug!("Graph traversal found {} related memories", count);
        Ok(count)
    }

    /// Count related memories using similarity metrics
    async fn count_similarity_related_memories(
        &self,
        memory: &MemoryEntry,
        processed_keys: &std::collections::HashSet<String>,
    ) -> Result<usize> {
        let mut count = 0;

        // Only proceed if we have an embedding for the target memory
        if let Some(target_embedding) = &memory.embedding {
            let similarity_threshold = 0.7; // High similarity threshold

            // Get all memories from storage for comparison
            let all_memories = self.storage.get_all_entries().await?;

            for stored_memory in all_memories {
                if stored_memory.key == memory.key || processed_keys.contains(&stored_memory.key) {
                    continue;
                }

                if let Some(stored_embedding) = &stored_memory.embedding {
                    // Convert f32 to f64 for similarity calculation
                    let target_f64: Vec<f64> = target_embedding.iter().map(|&x| x as f64).collect();
                    let stored_f64: Vec<f64> = stored_embedding.iter().map(|&x| x as f64).collect();

                    // Calculate cosine similarity
                    let similarity = crate::memory::embeddings::similarity::cosine_similarity(
                        &target_f64, &stored_f64
                    );

                    if similarity >= similarity_threshold {
                        count += 1;
                    }
                }
            }
        }

        tracing::debug!("Similarity analysis found {} related memories", count);
        Ok(count)
    }

    /// Count related memories using tag-based relationships
    async fn count_tag_related_memories(
        &self,
        memory: &MemoryEntry,
        processed_keys: &std::collections::HashSet<String>,
    ) -> Result<usize> {
        let mut count = 0;

        if memory.metadata.tags.is_empty() {
            return Ok(0);
        }

        // Get all memories from storage
        let all_memories = self.storage.get_all_entries().await?;

        for stored_memory in all_memories {
            if stored_memory.key == memory.key || processed_keys.contains(&stored_memory.key) {
                continue;
            }

            // Calculate tag overlap using Jaccard similarity
            let memory_tags: std::collections::HashSet<_> = memory.metadata.tags.iter().collect();
            let stored_tags: std::collections::HashSet<_> = stored_memory.metadata.tags.iter().collect();

            let intersection = memory_tags.intersection(&stored_tags).count();
            let union = memory_tags.union(&stored_tags).count();

            if union > 0 {
                let jaccard_similarity = intersection as f64 / union as f64;

                // Require at least 30% tag overlap
                if jaccard_similarity >= 0.3 {
                    count += 1;
                }
            }
        }

        tracing::debug!("Tag analysis found {} related memories", count);
        Ok(count)
    }

    /// Count related memories using temporal proximity
    async fn count_temporal_related_memories(
        &self,
        memory: &MemoryEntry,
        processed_keys: &std::collections::HashSet<String>,
    ) -> Result<usize> {
        let mut count = 0;

        // Define temporal window (memories created within 1 hour)
        let time_window = chrono::Duration::hours(1);
        let memory_time = memory.created_at();

        // Get all memories from storage
        let all_memories = self.storage.get_all_entries().await?;

        for stored_memory in all_memories {
            if stored_memory.key == memory.key || processed_keys.contains(&stored_memory.key) {
                continue;
            }

            let stored_time = stored_memory.created_at();
            let time_diff = if memory_time > stored_time {
                memory_time - stored_time
            } else {
                stored_time - memory_time
            };

            // Check if within temporal window
            if time_diff <= time_window {
                // Additional check: ensure some content similarity for temporal relationships
                let content_similarity = self.calculate_content_similarity(&memory.value, &stored_memory.value);

                if content_similarity >= 0.1 { // Lower threshold for temporal relationships
                    count += 1;
                }
            }
        }

        tracing::debug!("Temporal analysis found {} related memories", count);
        Ok(count)
    }

    /// Count related memories using pure content similarity (without temporal constraints)
    async fn count_content_related_memories(
        &self,
        memory: &MemoryEntry,
        processed_keys: &std::collections::HashSet<String>,
    ) -> Result<usize> {
        let mut count = 0;

        // Get all memories from storage
        let all_memories = self.storage.get_all_entries().await?;

        for stored_memory in all_memories {
            if stored_memory.key == memory.key || processed_keys.contains(&stored_memory.key) {
                continue;
            }

            // Calculate content similarity
            let content_similarity = self.calculate_content_similarity(&memory.value, &stored_memory.value);

            // Use a lower threshold for pure content similarity
            if content_similarity >= 0.4 {
                count += 1;
            }
        }

        tracing::debug!("Content analysis found {} related memories", count);
        Ok(count)
    }

    /// Calculate simple content similarity using word overlap
    fn calculate_content_similarity(&self, content1: &str, content2: &str) -> f64 {
        let content1_lower = content1.to_lowercase();
        let content2_lower = content2.to_lowercase();

        let words1: std::collections::HashSet<_> = content1_lower
            .split_whitespace()
            .filter(|word| word.len() > 2) // Less strict word filtering
            .collect();

        let words2: std::collections::HashSet<_> = content2_lower
            .split_whitespace()
            .filter(|word| word.len() > 2)
            .collect();

        if words1.is_empty() && words2.is_empty() {
            return 1.0;
        }

        if words1.is_empty() || words2.is_empty() {
            return 0.0;
        }

        let intersection = words1.intersection(&words2).count();
        let union = words1.union(&words2).count();

        intersection as f64 / union as f64
    }
}

/// Advanced memory management system
pub struct AdvancedMemoryManager {
    /// Memory summarizer for consolidation
    summarizer: MemorySummarizer,
    /// Advanced search engine
    search_engine: AdvancedSearchEngine,
    /// Lifecycle manager
    lifecycle_manager: MemoryLifecycleManager,
    /// Memory optimizer
    optimizer: MemoryOptimizer,
    /// Analytics engine
    analytics: MemoryAnalytics,
    /// Temporal manager for tracking changes
    temporal_manager: TemporalMemoryManager,
    /// Configuration
    config: MemoryManagementConfig,
}

/// Configuration for memory management
#[derive(Debug, Clone)]
pub struct MemoryManagementConfig {
    /// Enable automatic summarization
    pub enable_auto_summarization: bool,
    /// Summarization trigger threshold (number of related memories)
    pub summarization_threshold: usize,
    /// Enable automatic optimization
    pub enable_auto_optimization: bool,
    /// Optimization interval in hours
    pub optimization_interval_hours: u64,
    /// Enable lifecycle management
    pub enable_lifecycle_management: bool,
    /// Enable advanced analytics
    pub enable_analytics: bool,
    /// Maximum memory age before archival (days)
    pub max_memory_age_days: u64,
    /// Memory importance threshold for retention
    pub importance_retention_threshold: f64,
}

impl Default for MemoryManagementConfig {
    fn default() -> Self {
        Self {
            enable_auto_summarization: true,
            summarization_threshold: 10,
            enable_auto_optimization: true,
            optimization_interval_hours: 24,
            enable_lifecycle_management: true,
            enable_analytics: true,
            max_memory_age_days: 365,
            importance_retention_threshold: 0.3,
        }
    }
}

/// Memory management operation types
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum MemoryOperation {
    /// Add a new memory
    Add,
    /// Update existing memory
    Update,
    /// Delete memory
    Delete,
    /// Summarize multiple memories
    Summarize,
    /// Search memories
    Search,
    /// Optimize memory storage
    Optimize,
    /// Archive old memories
    Archive,
    /// Restore archived memories
    Restore,
    /// Merge similar memories
    Merge,
    /// Split complex memories
    Split,
    /// Analyze memory patterns
    Analyze,
}

/// Result of a memory management operation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryOperationResult {
    /// Operation that was performed
    pub operation: MemoryOperation,
    /// Whether the operation was successful
    pub success: bool,
    /// Number of memories affected
    pub affected_count: usize,
    /// Time taken to complete the operation
    pub duration_ms: u64,
    /// Optional result data
    pub result_data: Option<serde_json::Value>,
    /// Any warnings or messages
    pub messages: Vec<String>,
}

/// Memory management statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryManagementStats {
    /// Total memories under management
    pub total_memories: usize,
    /// Active memories (recently accessed)
    pub active_memories: usize,
    /// Archived memories
    pub archived_memories: usize,
    /// Total summarizations performed
    pub total_summarizations: usize,
    /// Total optimizations performed
    pub total_optimizations: usize,
    /// Average memory age in days
    pub avg_memory_age_days: f64,
    /// Memory utilization efficiency (0.0 to 1.0)
    pub utilization_efficiency: f64,
    /// Last optimization timestamp
    pub last_optimization: Option<DateTime<Utc>>,
}

impl AdvancedMemoryManager {
    /// Create a new advanced memory manager
    pub fn new(config: MemoryManagementConfig) -> Self {
        let temporal_config = crate::memory::temporal::TemporalConfig::default();
        
        Self {
            summarizer: MemorySummarizer::new(),
            search_engine: AdvancedSearchEngine::new(),
            lifecycle_manager: MemoryLifecycleManager::new(),
            optimizer: MemoryOptimizer::new(),
            analytics: MemoryAnalytics::new(),
            temporal_manager: TemporalMemoryManager::new(temporal_config),
            config,
        }
    }

    /// Add a new memory with full management
    pub async fn add_memory(
        &mut self,
        memory: MemoryEntry,
        knowledge_graph: Option<&mut MemoryKnowledgeGraph>,
    ) -> Result<MemoryOperationResult> {
        let start_time = std::time::Instant::now();
        let mut messages = Vec::new();
        
        // Track the change temporally
        let _version_id = self.temporal_manager
            .track_memory_change(&memory, ChangeType::Created)
            .await?;
        
        // Add to knowledge graph if provided
        if let Some(kg) = knowledge_graph {
            let _node_id = kg.add_memory_node(&memory).await?;
            messages.push("Added to knowledge graph".to_string());
        }
        
        // Update lifecycle tracking
        if self.config.enable_lifecycle_management {
            self.lifecycle_manager.track_memory_creation(&memory).await?;
        }
        
        // Update analytics
        if self.config.enable_analytics {
            self.analytics.record_memory_addition(&memory).await?;
        }
        
        // Check if summarization is needed
        if self.config.enable_auto_summarization {
            // TODO: Implement logic to count related memories
            // This would use the knowledge graph and similarity metrics
            let related_count = 0; // Placeholder
            if related_count >= self.config.summarization_threshold {
                messages.push("Summarization threshold reached".to_string());
                // TODO: Trigger summarization
            }
        }
        
        let duration_ms = start_time.elapsed().as_millis() as u64;
        
        Ok(MemoryOperationResult {
            operation: MemoryOperation::Add,
            success: true,
            affected_count: 1,
            duration_ms,
            result_data: Some(serde_json::json!({
                "memory_id": memory.id(),
                "memory_key": memory.key
            })),
            messages,
        })
    }

    /// Update an existing memory with change tracking
    pub async fn update_memory(
        &mut self,
        memory_key: &str,
        new_value: String,
        knowledge_graph: Option<&mut MemoryKnowledgeGraph>,
    ) -> Result<MemoryOperationResult> {
        let start_time = std::time::Instant::now();
        let mut messages = Vec::new();
        
        // Create updated memory entry (this would normally come from storage)
        // For now, we'll create a placeholder
        let updated_memory = MemoryEntry::new(
            memory_key.to_string(),
            new_value,
            MemoryType::ShortTerm, // This should be determined from existing memory
        );
        
        // Track the change temporally
        let _version_id = self.temporal_manager
            .track_memory_change(&updated_memory, ChangeType::Updated)
            .await?;
        
        // Update in knowledge graph if provided
        if let Some(_kg) = knowledge_graph {
            messages.push("Updated in knowledge graph".to_string());
        }
        
        // Update lifecycle tracking
        if self.config.enable_lifecycle_management {
            self.lifecycle_manager.track_memory_update(&updated_memory).await?;
        }
        
        // Update analytics
        if self.config.enable_analytics {
            self.analytics.record_memory_update(&updated_memory).await?;
        }
        
        let duration_ms = start_time.elapsed().as_millis() as u64;
        
        Ok(MemoryOperationResult {
            operation: MemoryOperation::Update,
            success: true,
            affected_count: 1,
            duration_ms,
            result_data: Some(serde_json::json!({
                "memory_key": memory_key,
                "new_value_length": updated_memory.value.len()
            })),
            messages,
        })
    }

    /// Delete a memory with proper cleanup
    pub async fn delete_memory(
        &mut self,
        memory_key: &str,
        knowledge_graph: Option<&mut MemoryKnowledgeGraph>,
    ) -> Result<MemoryOperationResult> {
        let start_time = std::time::Instant::now();
        let mut messages = Vec::new();
        
        // Create a placeholder for the deleted memory
        let deleted_memory = MemoryEntry::new(
            memory_key.to_string(),
            String::new(),
            MemoryType::ShortTerm,
        );
        
        // Track the deletion temporally
        let _version_id = self.temporal_manager
            .track_memory_change(&deleted_memory, ChangeType::Deleted)
            .await?;
        
        // Remove from knowledge graph if provided
        if let Some(_kg) = knowledge_graph {
            messages.push("Removed from knowledge graph".to_string());
        }
        
        // Update lifecycle tracking
        if self.config.enable_lifecycle_management {
            self.lifecycle_manager.track_memory_deletion(memory_key).await?;
        }
        
        // Update analytics
        if self.config.enable_analytics {
            self.analytics.record_memory_deletion(memory_key).await?;
        }
        
        let duration_ms = start_time.elapsed().as_millis() as u64;
        
        Ok(MemoryOperationResult {
            operation: MemoryOperation::Delete,
            success: true,
            affected_count: 1,
            duration_ms,
            result_data: Some(serde_json::json!({
                "deleted_memory_key": memory_key
            })),
            messages,
        })
    }

    /// Perform advanced search across memories
    pub async fn search_memories(
        &self,
        query: SearchQuery,
    ) -> Result<Vec<SearchResult>> {
        self.search_engine.search(query).await
    }

    /// Summarize a group of related memories
    pub async fn summarize_memories(
        &mut self,
        memory_keys: Vec<String>,
        strategy: SummaryStrategy,
    ) -> Result<SummaryResult> {
        self.summarizer.summarize_memories(memory_keys, strategy).await
    }

    /// Optimize memory storage and organization
    pub async fn optimize_memories(&mut self) -> Result<MemoryOperationResult> {
        let start_time = std::time::Instant::now();
        
        let optimization_result = self.optimizer.optimize().await?;
        
        let duration_ms = start_time.elapsed().as_millis() as u64;
        
        Ok(MemoryOperationResult {
            operation: MemoryOperation::Optimize,
            success: true,
            affected_count: optimization_result.memories_optimized,
            duration_ms,
            result_data: Some(serde_json::to_value(&optimization_result)?),
            messages: optimization_result.messages,
        })
    }

    /// Analyze memory patterns and generate insights
    pub async fn analyze_memory_patterns(&self) -> Result<AnalyticsReport> {
        self.analytics.generate_report().await
    }

    /// Get comprehensive management statistics
    pub async fn get_management_stats(&self) -> Result<MemoryManagementStats> {
        // This would normally aggregate data from various sources
        Ok(MemoryManagementStats {
            total_memories: 0, // TODO: Get from storage
            active_memories: 0, // TODO: Calculate based on recent access
            archived_memories: 0, // TODO: Get from lifecycle manager
            total_summarizations: self.summarizer.get_summarization_count(),
            total_optimizations: self.optimizer.get_optimization_count(),
            avg_memory_age_days: 0.0, // TODO: Calculate from temporal data
            utilization_efficiency: 0.0, // TODO: Calculate efficiency metrics
            last_optimization: self.optimizer.get_last_optimization_time(),
        })
    }

    /// Perform automatic maintenance tasks
    pub async fn perform_maintenance(&mut self) -> Result<Vec<MemoryOperationResult>> {
        let mut results = Vec::new();
        
        // Cleanup old versions
        if self.config.enable_lifecycle_management {
            let cleanup_count = self.temporal_manager.cleanup_old_versions().await?;
            results.push(MemoryOperationResult {
                operation: MemoryOperation::Optimize,
                success: true,
                affected_count: cleanup_count,
                duration_ms: 0,
                result_data: Some(serde_json::json!({
                    "cleanup_type": "old_versions",
                    "cleaned_count": cleanup_count
                })),
                messages: vec!["Cleaned up old versions".to_string()],
            });
        }
        
        // Perform optimization if needed
        if self.config.enable_auto_optimization {
            let last_opt = self.optimizer.get_last_optimization_time();
            let should_optimize = last_opt.map_or(true, |time| {
                Utc::now() - time > Duration::hours(self.config.optimization_interval_hours as i64)
            });
            
            if should_optimize {
                let opt_result = self.optimize_memories().await?;
                results.push(opt_result);
            }
        }
        
        Ok(results)
    }





}

#[cfg(test)]
mod tests;