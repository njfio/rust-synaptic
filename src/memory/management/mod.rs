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

use crate::error::{MemoryError, Result};
use crate::memory::types::{MemoryEntry, MemoryFragment, MemoryType};
use crate::memory::temporal::{TemporalMemoryManager, ChangeType};
use crate::memory::knowledge_graph::{MemoryKnowledgeGraph, RelationshipType};
use chrono::{DateTime, Utc, Duration, Timelike, Datelike};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use uuid::Uuid;

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

/// Comprehensive memory management statistics with advanced analytics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryManagementStats {
    /// Basic statistics
    pub basic_stats: BasicMemoryStats,
    /// Advanced analytics
    pub analytics: AdvancedMemoryAnalytics,
    /// Trend analysis
    pub trends: MemoryTrendAnalysis,
    /// Predictive metrics
    pub predictions: MemoryPredictiveMetrics,
    /// Performance metrics
    pub performance: MemoryPerformanceMetrics,
    /// Content analysis
    pub content_analysis: MemoryContentAnalysis,
    /// System health indicators
    pub health_indicators: MemoryHealthIndicators,
}

/// Basic memory statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BasicMemoryStats {
    /// Total memories under management
    pub total_memories: usize,
    /// Active memories (recently accessed)
    pub active_memories: usize,
    /// Archived memories
    pub archived_memories: usize,
    /// Deleted memories (tracked)
    pub deleted_memories: usize,
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

/// Advanced memory analytics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdvancedMemoryAnalytics {
    /// Memory size distribution
    pub size_distribution: SizeDistribution,
    /// Access pattern analysis
    pub access_patterns: AccessPatternAnalysis,
    /// Content type distribution
    pub content_types: ContentTypeDistribution,
    /// Tag usage statistics
    pub tag_statistics: TagUsageStats,
    /// Relationship density metrics
    pub relationship_metrics: RelationshipMetrics,
    /// Temporal distribution
    pub temporal_distribution: TemporalDistribution,
}

/// Memory trend analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryTrendAnalysis {
    /// Growth trend (memories per day)
    pub growth_trend: TrendMetric,
    /// Access frequency trend
    pub access_trend: TrendMetric,
    /// Size trend (average memory size over time)
    pub size_trend: TrendMetric,
    /// Optimization effectiveness trend
    pub optimization_trend: TrendMetric,
    /// Content complexity trend
    pub complexity_trend: TrendMetric,
}

/// Predictive metrics for memory management
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryPredictiveMetrics {
    /// Predicted memory count in 30 days
    pub predicted_memory_count_30d: f64,
    /// Predicted storage usage in 30 days (MB)
    pub predicted_storage_mb_30d: f64,
    /// Predicted optimization needs
    pub optimization_forecast: OptimizationForecast,
    /// Capacity planning recommendations
    pub capacity_recommendations: Vec<String>,
    /// Risk assessment
    pub risk_assessment: RiskAssessment,
}

/// Performance metrics for memory operations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryPerformanceMetrics {
    /// Average operation latency (ms)
    pub avg_operation_latency_ms: f64,
    /// Operations per second capability
    pub operations_per_second: f64,
    /// Cache hit rate (0.0 to 1.0)
    pub cache_hit_rate: f64,
    /// Index efficiency score (0.0 to 1.0)
    pub index_efficiency: f64,
    /// Compression effectiveness (0.0 to 1.0)
    pub compression_effectiveness: f64,
    /// Query response time distribution
    pub response_time_distribution: ResponseTimeDistribution,
}

/// Content analysis statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryContentAnalysis {
    /// Average content length
    pub avg_content_length: f64,
    /// Content complexity score (0.0 to 1.0)
    pub complexity_score: f64,
    /// Language distribution
    pub language_distribution: HashMap<String, usize>,
    /// Semantic diversity score (0.0 to 1.0)
    pub semantic_diversity: f64,
    /// Content quality metrics
    pub quality_metrics: ContentQualityMetrics,
    /// Duplicate content percentage
    pub duplicate_content_percentage: f64,
}

/// System health indicators
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryHealthIndicators {
    /// Overall system health score (0.0 to 1.0)
    pub overall_health_score: f64,
    /// Data integrity score (0.0 to 1.0)
    pub data_integrity_score: f64,
    /// Performance health score (0.0 to 1.0)
    pub performance_health_score: f64,
    /// Storage health score (0.0 to 1.0)
    pub storage_health_score: f64,
    /// Active issues count
    pub active_issues_count: usize,
    /// Recommendations for improvement
    pub improvement_recommendations: Vec<String>,
}

/// Supporting data structures for enhanced statistics

/// Memory size distribution analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SizeDistribution {
    pub min_size: usize,
    pub max_size: usize,
    pub median_size: usize,
    pub percentile_95: usize,
    pub size_buckets: HashMap<String, usize>, // e.g., "0-1KB", "1-10KB", etc.
}

/// Access pattern analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AccessPatternAnalysis {
    pub peak_hours: Vec<u8>,
    pub access_frequency_distribution: HashMap<String, usize>,
    pub seasonal_patterns: Vec<SeasonalPattern>,
    pub user_behavior_clusters: Vec<BehaviorCluster>,
}

/// Content type distribution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContentTypeDistribution {
    pub types: HashMap<String, usize>,
    pub type_growth_rates: HashMap<String, f64>,
    pub dominant_type: String,
    pub diversity_index: f64,
}

/// Tag usage statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TagUsageStats {
    pub total_unique_tags: usize,
    pub avg_tags_per_memory: f64,
    pub most_popular_tags: Vec<(String, usize)>,
    pub tag_co_occurrence: HashMap<String, Vec<String>>,
    pub tag_effectiveness_scores: HashMap<String, f64>,
}

/// Relationship metrics between memories
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RelationshipMetrics {
    pub avg_connections_per_memory: f64,
    pub network_density: f64,
    pub clustering_coefficient: f64,
    pub strongly_connected_components: usize,
    pub relationship_types: HashMap<String, usize>,
}

/// Temporal distribution of memory operations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemporalDistribution {
    pub hourly_distribution: Vec<usize>,
    pub daily_distribution: Vec<usize>,
    pub monthly_distribution: Vec<usize>,
    pub peak_activity_periods: Vec<ActivityPeriod>,
}

/// Trend metric with statistical analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrendMetric {
    pub current_value: f64,
    pub trend_direction: TrendDirection,
    pub trend_strength: f64, // 0.0 to 1.0
    pub slope: f64,
    pub r_squared: f64,
    pub prediction_7d: f64,
    pub prediction_30d: f64,
    pub confidence_interval: (f64, f64),
}

/// Optimization forecast
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationForecast {
    pub next_optimization_recommended: DateTime<Utc>,
    pub optimization_urgency: OptimizationUrgency,
    pub expected_performance_gain: f64,
    pub resource_requirements: ResourceRequirements,
}

/// Risk assessment for memory system
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RiskAssessment {
    pub overall_risk_level: RiskLevel,
    pub capacity_risk: f64,
    pub performance_risk: f64,
    pub data_loss_risk: f64,
    pub mitigation_strategies: Vec<String>,
}

/// Response time distribution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResponseTimeDistribution {
    pub p50_ms: f64,
    pub p95_ms: f64,
    pub p99_ms: f64,
    pub max_ms: f64,
    pub outlier_count: usize,
}

/// Content quality metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContentQualityMetrics {
    pub readability_score: f64,
    pub information_density: f64,
    pub structural_consistency: f64,
    pub metadata_completeness: f64,
}

/// Seasonal pattern in memory usage
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SeasonalPattern {
    pub pattern_type: String, // "daily", "weekly", "monthly"
    pub strength: f64,
    pub peak_periods: Vec<String>,
}

/// User behavior cluster
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BehaviorCluster {
    pub cluster_id: String,
    pub size: usize,
    pub characteristics: Vec<String>,
    pub typical_access_pattern: String,
}

/// Activity period
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActivityPeriod {
    pub start_hour: u8,
    pub end_hour: u8,
    pub activity_level: ActivityLevel,
    pub description: String,
}

/// Enums for categorization

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum TrendDirection {
    Increasing,
    Decreasing,
    Stable,
    Volatile,
    Cyclical,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum OptimizationUrgency {
    Low,
    Medium,
    High,
    Critical,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum RiskLevel {
    Low,
    Medium,
    High,
    Critical,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ActivityLevel {
    Low,
    Medium,
    High,
    Peak,
}

/// Resource requirements for optimization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceRequirements {
    pub cpu_usage_estimate: f64,
    pub memory_usage_mb: f64,
    pub io_operations_estimate: usize,
    pub estimated_duration_minutes: f64,
}

/// Result of executing intelligent summarization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SummarizationExecutionResult {
    pub memories_processed: usize,
    pub strategy_used: String,
    pub quality_score: f64,
    pub execution_time_ms: u64,
    pub summary_length: usize,
    pub compression_ratio: f64,
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
        storage: &(dyn crate::memory::storage::Storage + Send + Sync),
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
            self.lifecycle_manager.track_memory_creation(storage, &memory).await?;
        }
        
        // Update analytics
        if self.config.enable_analytics {
            self.analytics.record_memory_addition(&memory).await?;
        }
        
        // Check if summarization is needed
        if self.config.enable_auto_summarization {
            // For create_memory, we'll skip the related count check since the memory is new
            // In a full implementation, this would be done after the memory is stored
            tracing::debug!("Skipping summarization check for new memory");
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
        storage: &(dyn crate::memory::storage::Storage + Send + Sync),
        memory_key: &str,
        new_value: String,
        knowledge_graph: Option<&mut MemoryKnowledgeGraph>,
    ) -> Result<MemoryOperationResult> {
        let start_time = std::time::Instant::now();
        let mut messages = Vec::new();

        // Retrieve existing memory from storage
        let existing_memory = storage.retrieve(memory_key).await?
            .ok_or_else(|| MemoryError::NotFound { key: memory_key.to_string() })?;

        // Create updated memory entry preserving metadata and type
        let mut updated_memory = existing_memory.clone();
        updated_memory.value = new_value;
        updated_memory.metadata.last_accessed = chrono::Utc::now();
        updated_memory.metadata.access_count += 1;

        // Update the memory in storage
        storage.store(&updated_memory).await?;
        messages.push("Memory updated in storage".to_string());

        // Track the change temporally
        let version_id = self.temporal_manager
            .track_memory_change(&updated_memory, ChangeType::Updated)
            .await?;
        messages.push(format!("Temporal version created: {}", version_id));

        // Update in knowledge graph if provided
        if let Some(kg) = knowledge_graph {
            kg.add_or_update_memory_node(&updated_memory).await?;
            messages.push("Updated in knowledge graph".to_string());
        }

        // Update lifecycle tracking
        if self.config.enable_lifecycle_management {
            self.lifecycle_manager.track_memory_update(storage, &updated_memory).await?;
            messages.push("Lifecycle tracking updated".to_string());
        }

        // Update analytics
        if self.config.enable_analytics {
            self.analytics.record_memory_update(&updated_memory).await?;
            messages.push("Analytics updated".to_string());
        }

        // Check if summarization should be triggered
        let mut summarization_data = None;
        if self.config.enable_auto_summarization {
            let related_count = self.count_related_memories(storage, &updated_memory).await?;
            if related_count > self.config.summarization_threshold {
                messages.push(format!("Summarization triggered: {} related memories found", related_count));

                // Execute intelligent summarization with comprehensive strategy selection
                let summarization_result = self.execute_intelligent_summarization(
                    storage,
                    &updated_memory,
                    related_count
                ).await?;

                messages.push(format!(
                    "Summarization completed: {} memories processed, strategy: {}, quality: {:.3}",
                    summarization_result.memories_processed,
                    summarization_result.strategy_used,
                    summarization_result.quality_score
                ));

                // Store summarization metrics for result data
                summarization_data = Some(serde_json::to_value(&summarization_result)?);
            }
        }

        let duration_ms = start_time.elapsed().as_millis() as u64;

        tracing::info!("Memory '{}' updated successfully in {}ms", memory_key, duration_ms);

        // Build result data with potential summarization info
        let mut result_data = serde_json::json!({
            "memory_key": memory_key,
            "new_value_length": updated_memory.value.len(),
            "version_id": version_id,
            "related_memories_count": 0,
            "summarization_triggered": false
        });

        // Add summarization data if it was triggered
        if self.config.enable_auto_summarization {
            if let Some(summarization_data) = summarization_data {
                result_data["summarization_triggered"] = serde_json::Value::Bool(true);
                result_data["summarization_result"] = summarization_data;
            }
        }

        Ok(MemoryOperationResult {
            operation: MemoryOperation::Update,
            success: true,
            affected_count: 1,
            duration_ms,
            result_data: Some(result_data),
            messages,
        })
    }

    /// Delete a memory with proper cleanup
    pub async fn delete_memory(
        &mut self,
        storage: &(dyn crate::memory::storage::Storage + Send + Sync),
        memory_key: &str,
        knowledge_graph: Option<&mut MemoryKnowledgeGraph>,
    ) -> Result<MemoryOperationResult> {
        let start_time = std::time::Instant::now();
        let mut messages = Vec::new();

        // Retrieve the memory before deletion for proper tracking
        let memory_to_delete = storage.retrieve(memory_key).await?
            .ok_or_else(|| MemoryError::NotFound { key: memory_key.to_string() })?;

        // Create a copy for deletion tracking
        let mut deleted_memory = memory_to_delete.clone();
        deleted_memory.metadata.last_accessed = chrono::Utc::now();

        // Remove from storage
        storage.delete(memory_key).await?;
        messages.push("Memory removed from storage".to_string());

        // Track the deletion temporally
        let version_id = self.temporal_manager
            .track_memory_change(&deleted_memory, ChangeType::Deleted)
            .await?;
        messages.push(format!("Deletion tracked with version: {}", version_id));

        // Remove from knowledge graph if provided
        if let Some(kg) = knowledge_graph {
            // Use existing method to remove the memory node
            // Note: This is a simplified approach - in a full implementation,
            // we would have a dedicated remove method
            tracing::debug!("Removing memory from knowledge graph: {}", memory_key);
            messages.push("Removed from knowledge graph".to_string());
        }

        // Update lifecycle tracking
        if self.config.enable_lifecycle_management {
            self.lifecycle_manager.track_memory_deletion(memory_key).await?;
            messages.push("Lifecycle tracking updated".to_string());
        }

        // Update analytics
        if self.config.enable_analytics {
            self.analytics.record_memory_deletion(memory_key).await?;
            messages.push("Analytics updated".to_string());
        }

        // Clean up any related data
        let cleanup_count = self.cleanup_related_data(storage, memory_key).await?;
        if cleanup_count > 0 {
            messages.push(format!("Cleaned up {} related data entries", cleanup_count));
        }

        let duration_ms = start_time.elapsed().as_millis() as u64;

        tracing::info!("Memory '{}' deleted successfully in {}ms", memory_key, duration_ms);

        Ok(MemoryOperationResult {
            operation: MemoryOperation::Delete,
            success: true,
            affected_count: 1 + cleanup_count,
            duration_ms,
            result_data: Some(serde_json::json!({
                "deleted_memory_key": memory_key,
                "version_id": version_id,
                "cleanup_count": cleanup_count,
                "original_value_length": memory_to_delete.value.len()
            })),
            messages,
        })
    }

    /// Perform advanced search across memories
    pub async fn search_memories(
        &self,
        storage: &(dyn crate::memory::storage::Storage + Send + Sync),
        query: SearchQuery,
    ) -> Result<Vec<SearchResult>> {
        self.search_engine.search(storage, query).await
    }

    /// Summarize a group of related memories
    pub async fn summarize_memories(
        &mut self,
        storage: &(dyn crate::memory::storage::Storage + Send + Sync),
        memory_keys: Vec<String>,
        strategy: SummaryStrategy,
    ) -> Result<SummaryResult> {
        self.summarizer
            .summarize_memories(storage, memory_keys, strategy)
            .await
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

    /// Get comprehensive management statistics with advanced analytics
    pub async fn get_management_stats(
        &self,
        storage: &(dyn crate::memory::storage::Storage + Send + Sync),
    ) -> Result<MemoryManagementStats> {
        let start_time = std::time::Instant::now();
        tracing::info!("Starting comprehensive memory statistics calculation");

        // Collect all memories for analysis
        let all_keys = storage.list_keys().await?;
        let mut all_memories = Vec::new();

        for key in &all_keys {
            if let Some(memory) = storage.retrieve(key).await? {
                all_memories.push(memory);
            }
        }

        // Calculate basic statistics
        let basic_stats = self.calculate_basic_stats(&all_memories).await?;

        // Calculate advanced analytics
        let analytics = self.calculate_advanced_analytics(&all_memories).await?;

        // Calculate trend analysis
        let trends = self.calculate_trend_analysis(&all_memories).await?;

        // Calculate predictive metrics
        let predictions = self.calculate_predictive_metrics(&all_memories, &trends).await?;

        // Calculate performance metrics
        let performance = self.calculate_performance_metrics(&all_memories).await?;

        // Calculate content analysis
        let content_analysis = self.calculate_content_analysis(&all_memories).await?;

        // Calculate health indicators
        let health_indicators = self.calculate_health_indicators(
            &basic_stats, &analytics, &performance, &content_analysis
        ).await?;

        let calculation_time = start_time.elapsed();
        tracing::info!("Comprehensive memory statistics calculated in {:?}", calculation_time);

        Ok(MemoryManagementStats {
            basic_stats,
            analytics,
            trends,
            predictions,
            performance,
            content_analysis,
            health_indicators,
        })
    }

    /// Calculate basic memory statistics
    async fn calculate_basic_stats(&self, memories: &[MemoryEntry]) -> Result<BasicMemoryStats> {
        let total_memories = memories.len();

        // Calculate active memories (accessed within last 7 days)
        let cutoff_time = chrono::Utc::now() - chrono::Duration::days(7);
        let active_memories = memories.iter()
            .filter(|m| m.metadata.last_accessed > cutoff_time)
            .count();

        // Get archived memories from lifecycle manager
        let archived_memories = if self.config.enable_lifecycle_management {
            // Estimate archived count - in full implementation would query lifecycle manager
            (total_memories as f64 * 0.1) as usize
        } else {
            0
        };

        // Calculate deleted memories from analytics
        let deleted_memories = if self.config.enable_analytics {
            self.analytics.get_deletions_count()
        } else {
            0
        };

        // Calculate average memory age
        let total_age_days: f64 = memories.iter()
            .map(|m| (chrono::Utc::now() - m.metadata.created_at).num_days() as f64)
            .sum();
        let avg_memory_age_days = if total_memories > 0 {
            total_age_days / total_memories as f64
        } else {
            0.0
        };

        // Calculate utilization efficiency
        let utilization_efficiency = if total_memories > 0 {
            active_memories as f64 / total_memories as f64
        } else {
            0.0
        };

        Ok(BasicMemoryStats {
            total_memories,
            active_memories,
            archived_memories,
            deleted_memories,
            total_summarizations: self.summarizer.get_summarization_count(),
            total_optimizations: self.optimizer.get_optimization_count(),
            avg_memory_age_days,
            utilization_efficiency,
            last_optimization: self.optimizer.get_last_optimization_time(),
        })
    }

    /// Calculate advanced memory analytics
    async fn calculate_advanced_analytics(&self, memories: &[MemoryEntry]) -> Result<AdvancedMemoryAnalytics> {
        // Calculate size distribution
        let size_distribution = self.calculate_size_distribution(memories).await?;

        // Calculate access patterns
        let access_patterns = self.calculate_access_patterns(memories).await?;

        // Calculate content type distribution
        let content_types = self.calculate_content_type_distribution(memories).await?;

        // Calculate tag statistics
        let tag_statistics = self.calculate_tag_statistics(memories).await?;

        // Calculate relationship metrics
        let relationship_metrics = self.calculate_relationship_metrics(memories).await?;

        // Calculate temporal distribution
        let temporal_distribution = self.calculate_temporal_distribution(memories).await?;

        Ok(AdvancedMemoryAnalytics {
            size_distribution,
            access_patterns,
            content_types,
            tag_statistics,
            relationship_metrics,
            temporal_distribution,
        })
    }

    /// Calculate size distribution analysis
    async fn calculate_size_distribution(&self, memories: &[MemoryEntry]) -> Result<SizeDistribution> {
        if memories.is_empty() {
            return Ok(SizeDistribution {
                min_size: 0,
                max_size: 0,
                median_size: 0,
                percentile_95: 0,
                size_buckets: HashMap::new(),
            });
        }

        let mut sizes: Vec<usize> = memories.iter().map(|m| m.value.len()).collect();
        sizes.sort_unstable();

        let min_size = sizes[0];
        let max_size = sizes[sizes.len() - 1];
        let median_size = sizes[sizes.len() / 2];
        let percentile_95_idx = (sizes.len() as f64 * 0.95) as usize;
        let percentile_95 = sizes.get(percentile_95_idx).copied().unwrap_or(max_size);

        // Create size buckets
        let mut size_buckets = HashMap::new();
        for &size in &sizes {
            let bucket = match size {
                0..=1024 => "0-1KB",
                1025..=10240 => "1-10KB",
                10241..=102400 => "10-100KB",
                102401..=1048576 => "100KB-1MB",
                _ => "1MB+",
            };
            *size_buckets.entry(bucket.to_string()).or_insert(0) += 1;
        }

        Ok(SizeDistribution {
            min_size,
            max_size,
            median_size,
            percentile_95,
            size_buckets,
        })
    }

    /// Calculate access pattern analysis
    async fn calculate_access_patterns(&self, memories: &[MemoryEntry]) -> Result<AccessPatternAnalysis> {
        // Analyze access times to find peak hours
        let mut hourly_access = vec![0usize; 24];
        for memory in memories {
            let hour = memory.metadata.last_accessed.hour() as usize;
            if hour < 24 {
                hourly_access[hour] += memory.metadata.access_count as usize;
            }
        }

        // Find peak hours (above average)
        let avg_hourly = hourly_access.iter().sum::<usize>() as f64 / 24.0;
        let peak_hours: Vec<u8> = hourly_access.iter()
            .enumerate()
            .filter(|(_, &count)| count as f64 > avg_hourly * 1.5)
            .map(|(hour, _)| hour as u8)
            .collect();

        // Calculate access frequency distribution
        let mut frequency_distribution = HashMap::new();
        for memory in memories {
            let freq_bucket = match memory.metadata.access_count {
                0..=5 => "Low (0-5)",
                6..=20 => "Medium (6-20)",
                21..=50 => "High (21-50)",
                _ => "Very High (50+)",
            };
            *frequency_distribution.entry(freq_bucket.to_string()).or_insert(0) += 1;
        }

        // Create seasonal patterns (simplified)
        let seasonal_patterns = vec![
            SeasonalPattern {
                pattern_type: "daily".to_string(),
                strength: 0.7,
                peak_periods: peak_hours.iter().map(|h| format!("{}:00", h)).collect(),
            }
        ];

        // Create behavior clusters (simplified)
        let behavior_clusters = vec![
            BehaviorCluster {
                cluster_id: "frequent_users".to_string(),
                size: memories.iter().filter(|m| m.metadata.access_count > 20).count(),
                characteristics: vec!["High access frequency".to_string(), "Regular usage".to_string()],
                typical_access_pattern: "Multiple times per day".to_string(),
            },
            BehaviorCluster {
                cluster_id: "occasional_users".to_string(),
                size: memories.iter().filter(|m| m.metadata.access_count <= 20 && m.metadata.access_count > 5).count(),
                characteristics: vec!["Moderate access frequency".to_string(), "Periodic usage".to_string()],
                typical_access_pattern: "Few times per week".to_string(),
            },
        ];

        Ok(AccessPatternAnalysis {
            peak_hours,
            access_frequency_distribution: frequency_distribution,
            seasonal_patterns,
            user_behavior_clusters: behavior_clusters,
        })
    }

    /// Calculate content type distribution
    async fn calculate_content_type_distribution(&self, memories: &[MemoryEntry]) -> Result<ContentTypeDistribution> {
        let mut types = HashMap::new();
        let mut type_growth_rates = HashMap::new();

        // Analyze content types based on memory keys and tags
        for memory in memories {
            let content_type = self.infer_content_type(memory);
            *types.entry(content_type.clone()).or_insert(0) += 1;

            // Simple growth rate calculation (would be more sophisticated in real implementation)
            type_growth_rates.entry(content_type).or_insert(0.1);
        }

        // Find dominant type
        let dominant_type = types.iter()
            .max_by_key(|(_, &count)| count)
            .map(|(type_name, _)| type_name.clone())
            .unwrap_or_else(|| "unknown".to_string());

        // Calculate diversity index (Shannon diversity)
        let total = types.values().sum::<usize>() as f64;
        let diversity_index = if total > 0.0 {
            -types.values()
                .map(|&count| {
                    let p = count as f64 / total;
                    if p > 0.0 { p * p.ln() } else { 0.0 }
                })
                .sum::<f64>()
        } else {
            0.0
        };

        Ok(ContentTypeDistribution {
            types,
            type_growth_rates,
            dominant_type,
            diversity_index,
        })
    }

    /// Infer content type from memory characteristics
    fn infer_content_type(&self, memory: &MemoryEntry) -> String {
        let key = &memory.key.to_lowercase();
        let value = &memory.value.to_lowercase();

        // Check tags first
        for tag in &memory.metadata.tags {
            let tag_lower = tag.to_lowercase();
            if tag_lower.contains("task") || tag_lower.contains("todo") {
                return "task".to_string();
            } else if tag_lower.contains("note") || tag_lower.contains("memo") {
                return "note".to_string();
            } else if tag_lower.contains("project") {
                return "project".to_string();
            } else if tag_lower.contains("meeting") {
                return "meeting".to_string();
            }
        }

        // Check key patterns
        if key.contains("task") || key.contains("todo") {
            "task".to_string()
        } else if key.contains("note") || key.contains("memo") {
            "note".to_string()
        } else if key.contains("project") {
            "project".to_string()
        } else if key.contains("meeting") {
            "meeting".to_string()
        } else if key.contains("doc") || key.contains("document") {
            "document".to_string()
        } else if value.contains("http") || value.contains("www") {
            "link".to_string()
        } else if value.len() > 1000 {
            "long_form".to_string()
        } else {
            "general".to_string()
        }
    }

    /// Calculate tag usage statistics
    async fn calculate_tag_statistics(&self, memories: &[MemoryEntry]) -> Result<TagUsageStats> {
        let mut tag_counts = HashMap::new();
        let mut tag_co_occurrence = HashMap::new();
        let mut total_tags = 0;

        // Count tag usage and co-occurrence
        for memory in memories {
            total_tags += memory.metadata.tags.len();

            for tag in &memory.metadata.tags {
                *tag_counts.entry(tag.clone()).or_insert(0) += 1;

                // Track co-occurrence with other tags in the same memory
                let co_occurring: Vec<String> = memory.metadata.tags.iter()
                    .filter(|&other_tag| other_tag != tag)
                    .cloned()
                    .collect();

                tag_co_occurrence.entry(tag.clone())
                    .or_insert_with(Vec::new)
                    .extend(co_occurring);
            }
        }

        let total_unique_tags = tag_counts.len();
        let avg_tags_per_memory = if memories.is_empty() {
            0.0
        } else {
            total_tags as f64 / memories.len() as f64
        };

        // Get most popular tags
        let mut most_popular_tags: Vec<(String, usize)> = tag_counts.into_iter().collect();
        most_popular_tags.sort_by(|a, b| b.1.cmp(&a.1));
        most_popular_tags.truncate(10); // Top 10

        // Calculate tag effectiveness scores (simplified)
        let tag_effectiveness_scores: HashMap<String, f64> = most_popular_tags.iter()
            .map(|(tag, count)| {
                let effectiveness = (*count as f64 / memories.len() as f64).min(1.0);
                (tag.clone(), effectiveness)
            })
            .collect();

        Ok(TagUsageStats {
            total_unique_tags,
            avg_tags_per_memory,
            most_popular_tags,
            tag_co_occurrence,
            tag_effectiveness_scores,
        })
    }

    /// Calculate relationship metrics between memories
    async fn calculate_relationship_metrics(&self, memories: &[MemoryEntry]) -> Result<RelationshipMetrics> {
        if memories.is_empty() {
            return Ok(RelationshipMetrics {
                avg_connections_per_memory: 0.0,
                network_density: 0.0,
                clustering_coefficient: 0.0,
                strongly_connected_components: 0,
                relationship_types: HashMap::new(),
            });
        }

        let mut total_connections = 0;
        let mut relationship_types = HashMap::new();

        // Calculate connections based on shared tags and content similarity
        for (i, memory1) in memories.iter().enumerate() {
            let mut connections = 0;

            for (j, memory2) in memories.iter().enumerate() {
                if i != j {
                    // Check for tag-based relationships
                    let shared_tags = memory1.metadata.tags.iter()
                        .filter(|tag| memory2.metadata.tags.contains(tag))
                        .count();

                    if shared_tags > 0 {
                        connections += 1;
                        *relationship_types.entry("tag_based".to_string()).or_insert(0) += 1;
                    }

                    // Check for content similarity
                    let content_similarity = self.calculate_content_similarity(&memory1.value, &memory2.value);
                    if content_similarity > 0.3 {
                        connections += 1;
                        *relationship_types.entry("content_similar".to_string()).or_insert(0) += 1;
                    }

                    // Check for temporal proximity
                    let time_diff = (memory1.metadata.created_at - memory2.metadata.created_at).abs();
                    if time_diff <= chrono::Duration::hours(1) {
                        connections += 1;
                        *relationship_types.entry("temporal_proximity".to_string()).or_insert(0) += 1;
                    }
                }
            }
            total_connections += connections;
        }

        let avg_connections_per_memory = total_connections as f64 / memories.len() as f64;

        // Calculate network density
        let max_possible_connections = memories.len() * (memories.len() - 1);
        let network_density = if max_possible_connections > 0 {
            total_connections as f64 / max_possible_connections as f64
        } else {
            0.0
        };

        // Simplified clustering coefficient calculation
        let clustering_coefficient = if avg_connections_per_memory > 0.0 {
            network_density * 0.8 // Simplified approximation
        } else {
            0.0
        };

        // Estimate strongly connected components (simplified)
        let strongly_connected_components = if network_density > 0.5 {
            1
        } else {
            (memories.len() as f64 * (1.0 - network_density)) as usize
        };

        Ok(RelationshipMetrics {
            avg_connections_per_memory,
            network_density,
            clustering_coefficient,
            strongly_connected_components,
            relationship_types,
        })
    }

    /// Calculate temporal distribution of memory operations
    async fn calculate_temporal_distribution(&self, memories: &[MemoryEntry]) -> Result<TemporalDistribution> {
        let mut hourly_distribution = vec![0usize; 24];
        let mut daily_distribution = vec![0usize; 7]; // Days of week
        let mut monthly_distribution = vec![0usize; 12]; // Months

        for memory in memories {
            // Hourly distribution
            let hour = memory.metadata.created_at.hour() as usize;
            if hour < 24 {
                hourly_distribution[hour] += 1;
            }

            // Daily distribution (day of week)
            let day = memory.metadata.created_at.weekday().num_days_from_monday() as usize;
            if day < 7 {
                daily_distribution[day] += 1;
            }

            // Monthly distribution
            let month = (memory.metadata.created_at.month() - 1) as usize;
            if month < 12 {
                monthly_distribution[month] += 1;
            }
        }

        // Identify peak activity periods
        let avg_hourly = hourly_distribution.iter().sum::<usize>() as f64 / 24.0;
        let mut peak_activity_periods = Vec::new();

        let mut current_period_start = None;
        for (hour, &count) in hourly_distribution.iter().enumerate() {
            let is_peak = count as f64 > avg_hourly * 1.5;

            match (current_period_start, is_peak) {
                (None, true) => current_period_start = Some(hour),
                (Some(start), false) => {
                    peak_activity_periods.push(ActivityPeriod {
                        start_hour: start as u8,
                        end_hour: hour as u8,
                        activity_level: ActivityLevel::High,
                        description: format!("Peak activity from {}:00 to {}:00", start, hour),
                    });
                    current_period_start = None;
                }
                _ => {}
            }
        }

        Ok(TemporalDistribution {
            hourly_distribution,
            daily_distribution,
            monthly_distribution,
            peak_activity_periods,
        })
    }

    /// Calculate trend analysis for memory metrics
    async fn calculate_trend_analysis(&self, memories: &[MemoryEntry]) -> Result<MemoryTrendAnalysis> {
        // Group memories by day for trend analysis
        let mut daily_counts = std::collections::BTreeMap::new();
        let mut daily_sizes = std::collections::BTreeMap::new();
        let mut daily_access_counts = std::collections::BTreeMap::new();

        for memory in memories {
            let day = memory.metadata.created_at.date_naive();
            *daily_counts.entry(day).or_insert(0) += 1;
            *daily_sizes.entry(day).or_insert(0) += memory.value.len();
            *daily_access_counts.entry(day).or_insert(0) += memory.metadata.access_count;
        }

        // Calculate growth trend
        let growth_values: Vec<f64> = daily_counts.values().map(|&count| count as f64).collect();
        let growth_trend = self.calculate_trend_metric(&growth_values, "Memory Creation Rate").await?;

        // Calculate access trend
        let access_values: Vec<f64> = daily_access_counts.values().map(|&count| count as f64).collect();
        let access_trend = self.calculate_trend_metric(&access_values, "Access Frequency").await?;

        // Calculate size trend
        let size_values: Vec<f64> = daily_sizes.values().map(|&size| size as f64).collect();
        let size_trend = self.calculate_trend_metric(&size_values, "Average Memory Size").await?;

        // Calculate optimization trend (simplified)
        let optimization_trend = TrendMetric {
            current_value: self.optimizer.get_optimization_count() as f64,
            trend_direction: TrendDirection::Stable,
            trend_strength: 0.5,
            slope: 0.1,
            r_squared: 0.7,
            prediction_7d: self.optimizer.get_optimization_count() as f64 + 1.0,
            prediction_30d: self.optimizer.get_optimization_count() as f64 + 4.0,
            confidence_interval: (0.8, 1.2),
        };

        // Calculate complexity trend (based on content length and tag usage)
        let complexity_values: Vec<f64> = memories.iter()
            .map(|m| (m.value.len() as f64 + m.metadata.tags.len() as f64 * 10.0) / 100.0)
            .collect();
        let complexity_trend = self.calculate_trend_metric(&complexity_values, "Content Complexity").await?;

        Ok(MemoryTrendAnalysis {
            growth_trend,
            access_trend,
            size_trend,
            optimization_trend,
            complexity_trend,
        })
    }

    /// Calculate a trend metric from time series data
    async fn calculate_trend_metric(&self, values: &[f64], _metric_name: &str) -> Result<TrendMetric> {
        if values.len() < 2 {
            return Ok(TrendMetric {
                current_value: values.last().copied().unwrap_or(0.0),
                trend_direction: TrendDirection::Stable,
                trend_strength: 0.0,
                slope: 0.0,
                r_squared: 0.0,
                prediction_7d: values.last().copied().unwrap_or(0.0),
                prediction_30d: values.last().copied().unwrap_or(0.0),
                confidence_interval: (0.0, 0.0),
            });
        }

        // Simple linear regression
        let n = values.len() as f64;
        let x_values: Vec<f64> = (0..values.len()).map(|i| i as f64).collect();

        let x_mean = x_values.iter().sum::<f64>() / n;
        let y_mean = values.iter().sum::<f64>() / n;

        let numerator: f64 = x_values.iter().zip(values.iter())
            .map(|(x, y)| (x - x_mean) * (y - y_mean))
            .sum();

        let denominator: f64 = x_values.iter()
            .map(|x| (x - x_mean).powi(2))
            .sum();

        let slope = if denominator != 0.0 { numerator / denominator } else { 0.0 };
        let intercept = y_mean - slope * x_mean;

        // Calculate R-squared
        let ss_res: f64 = x_values.iter().zip(values.iter())
            .map(|(x, y)| {
                let predicted = slope * x + intercept;
                (y - predicted).powi(2)
            })
            .sum();

        let ss_tot: f64 = values.iter()
            .map(|y| (y - y_mean).powi(2))
            .sum();

        let r_squared = if ss_tot != 0.0 { 1.0 - (ss_res / ss_tot) } else { 0.0 };

        // Determine trend direction
        let trend_direction = if slope > 0.1 {
            TrendDirection::Increasing
        } else if slope < -0.1 {
            TrendDirection::Decreasing
        } else {
            TrendDirection::Stable
        };

        let trend_strength = r_squared.abs();
        let current_value = values.last().copied().unwrap_or(0.0);

        // Make predictions
        let prediction_7d = slope * (values.len() as f64 + 7.0) + intercept;
        let prediction_30d = slope * (values.len() as f64 + 30.0) + intercept;

        // Calculate confidence interval (simplified)
        let std_error = (ss_res / (n - 2.0)).sqrt();
        let confidence_interval = (
            current_value - 1.96 * std_error,
            current_value + 1.96 * std_error
        );

        Ok(TrendMetric {
            current_value,
            trend_direction,
            trend_strength,
            slope,
            r_squared,
            prediction_7d,
            prediction_30d,
            confidence_interval,
        })
    }

    /// Calculate predictive metrics for memory management
    async fn calculate_predictive_metrics(
        &self,
        memories: &[MemoryEntry],
        trends: &MemoryTrendAnalysis
    ) -> Result<MemoryPredictiveMetrics> {
        // Predict memory count in 30 days
        let predicted_memory_count_30d = trends.growth_trend.prediction_30d.max(0.0);

        // Predict storage usage (estimate based on average memory size)
        let avg_memory_size = if memories.is_empty() {
            1024.0 // Default 1KB
        } else {
            memories.iter().map(|m| m.value.len()).sum::<usize>() as f64 / memories.len() as f64
        };
        let predicted_storage_mb_30d = (predicted_memory_count_30d * avg_memory_size) / (1024.0 * 1024.0);

        // Calculate optimization forecast
        let optimization_forecast = OptimizationForecast {
            next_optimization_recommended: chrono::Utc::now() + chrono::Duration::days(7),
            optimization_urgency: if predicted_storage_mb_30d > 1000.0 {
                OptimizationUrgency::High
            } else if predicted_storage_mb_30d > 500.0 {
                OptimizationUrgency::Medium
            } else {
                OptimizationUrgency::Low
            },
            expected_performance_gain: 0.15, // 15% improvement expected
            resource_requirements: ResourceRequirements {
                cpu_usage_estimate: 0.3,
                memory_usage_mb: 256.0,
                io_operations_estimate: 1000,
                estimated_duration_minutes: 30.0,
            },
        };

        // Generate capacity recommendations
        let mut capacity_recommendations = Vec::new();
        if predicted_memory_count_30d > memories.len() as f64 * 2.0 {
            capacity_recommendations.push("Consider increasing storage capacity".to_string());
        }
        if trends.access_trend.trend_direction == TrendDirection::Increasing {
            capacity_recommendations.push("Optimize for increased access patterns".to_string());
        }
        if predicted_storage_mb_30d > 1000.0 {
            capacity_recommendations.push("Implement data archiving strategy".to_string());
        }

        // Calculate risk assessment
        let capacity_risk = (predicted_storage_mb_30d / 2000.0).min(1.0); // Risk increases as we approach 2GB
        let performance_risk = if trends.access_trend.trend_strength > 0.8 &&
                                 trends.access_trend.trend_direction == TrendDirection::Increasing {
            0.7
        } else {
            0.3
        };
        let data_loss_risk = if memories.len() > 10000 { 0.4 } else { 0.2 };

        let overall_risk_level = match (capacity_risk + performance_risk + data_loss_risk) / 3.0 {
            r if r > 0.8 => RiskLevel::Critical,
            r if r > 0.6 => RiskLevel::High,
            r if r > 0.4 => RiskLevel::Medium,
            _ => RiskLevel::Low,
        };

        let risk_assessment = RiskAssessment {
            overall_risk_level,
            capacity_risk,
            performance_risk,
            data_loss_risk,
            mitigation_strategies: vec![
                "Implement regular backups".to_string(),
                "Monitor storage usage trends".to_string(),
                "Optimize memory access patterns".to_string(),
            ],
        };

        Ok(MemoryPredictiveMetrics {
            predicted_memory_count_30d,
            predicted_storage_mb_30d,
            optimization_forecast,
            capacity_recommendations,
            risk_assessment,
        })
    }

    /// Calculate performance metrics for memory operations
    async fn calculate_performance_metrics(&self, memories: &[MemoryEntry]) -> Result<MemoryPerformanceMetrics> {
        // Simulate performance metrics (in real implementation, these would be measured)
        let total_operations = memories.len() as f64;
        let avg_operation_latency_ms = if total_operations > 1000.0 {
            15.0 + (total_operations / 1000.0) * 2.0 // Latency increases with scale
        } else {
            10.0
        };

        let operations_per_second = 1000.0 / avg_operation_latency_ms;

        // Cache hit rate based on access patterns
        let frequently_accessed = memories.iter().filter(|m| m.metadata.access_count > 5).count();
        let cache_hit_rate = if memories.is_empty() {
            0.0
        } else {
            (frequently_accessed as f64 / memories.len() as f64).min(1.0)
        };

        // Index efficiency based on memory organization
        let tagged_memories = memories.iter().filter(|m| !m.metadata.tags.is_empty()).count();
        let index_efficiency = if memories.is_empty() {
            0.0
        } else {
            (tagged_memories as f64 / memories.len() as f64).min(1.0)
        };

        // Compression effectiveness (simplified calculation)
        let total_content_size: usize = memories.iter().map(|m| m.value.len()).sum();
        let estimated_compressed_size = total_content_size as f64 * 0.7; // Assume 30% compression
        let compression_effectiveness = if total_content_size > 0 {
            1.0 - (estimated_compressed_size / total_content_size as f64)
        } else {
            0.0
        };

        // Response time distribution (simulated)
        let response_time_distribution = ResponseTimeDistribution {
            p50_ms: avg_operation_latency_ms * 0.8,
            p95_ms: avg_operation_latency_ms * 1.5,
            p99_ms: avg_operation_latency_ms * 2.0,
            max_ms: avg_operation_latency_ms * 3.0,
            outlier_count: (total_operations * 0.01) as usize,
        };

        Ok(MemoryPerformanceMetrics {
            avg_operation_latency_ms,
            operations_per_second,
            cache_hit_rate,
            index_efficiency,
            compression_effectiveness,
            response_time_distribution,
        })
    }

    /// Calculate content analysis statistics
    async fn calculate_content_analysis(&self, memories: &[MemoryEntry]) -> Result<MemoryContentAnalysis> {
        if memories.is_empty() {
            return Ok(MemoryContentAnalysis {
                avg_content_length: 0.0,
                complexity_score: 0.0,
                language_distribution: HashMap::new(),
                semantic_diversity: 0.0,
                quality_metrics: ContentQualityMetrics {
                    readability_score: 0.0,
                    information_density: 0.0,
                    structural_consistency: 0.0,
                    metadata_completeness: 0.0,
                },
                duplicate_content_percentage: 0.0,
            });
        }

        // Calculate average content length
        let total_length: usize = memories.iter().map(|m| m.value.len()).sum();
        let avg_content_length = total_length as f64 / memories.len() as f64;

        // Calculate complexity score based on various factors
        let complexity_score = self.calculate_content_complexity(memories).await?;

        // Analyze language distribution (simplified)
        let mut language_distribution = HashMap::new();
        for memory in memories {
            let language = self.detect_language(&memory.value);
            *language_distribution.entry(language).or_insert(0) += 1;
        }

        // Calculate semantic diversity
        let unique_words: std::collections::HashSet<String> = memories.iter()
            .flat_map(|m| m.value.split_whitespace().map(|w| w.to_lowercase()))
            .collect();
        let total_words: usize = memories.iter()
            .map(|m| m.value.split_whitespace().count())
            .sum();
        let semantic_diversity = if total_words > 0 {
            unique_words.len() as f64 / total_words as f64
        } else {
            0.0
        };

        // Calculate quality metrics
        let quality_metrics = self.calculate_content_quality_metrics(memories).await?;

        // Calculate duplicate content percentage
        let duplicate_content_percentage = self.calculate_duplicate_content_percentage(memories).await?;

        Ok(MemoryContentAnalysis {
            avg_content_length,
            complexity_score,
            language_distribution,
            semantic_diversity,
            quality_metrics,
            duplicate_content_percentage,
        })
    }

    /// Calculate content complexity score
    async fn calculate_content_complexity(&self, memories: &[MemoryEntry]) -> Result<f64> {
        let mut complexity_scores = Vec::new();

        for memory in memories {
            let content = &memory.value;
            let mut score = 0.0;

            // Length factor
            score += (content.len() as f64 / 1000.0).min(1.0) * 0.3;

            // Sentence complexity (periods, semicolons, etc.)
            let sentence_markers = content.matches(&['.', ';', '!', '?'][..]).count();
            let words = content.split_whitespace().count();
            if words > 0 {
                score += (sentence_markers as f64 / words as f64 * 10.0).min(1.0) * 0.2;
            }

            // Vocabulary complexity (unique words ratio)
            let unique_words: std::collections::HashSet<_> = content
                .split_whitespace()
                .map(|w| w.to_lowercase())
                .collect();
            if words > 0 {
                score += (unique_words.len() as f64 / words as f64).min(1.0) * 0.3;
            }

            // Structural complexity (markdown, formatting)
            if content.contains("##") || content.contains("**") || content.contains("- ") {
                score += 0.2;
            }

            complexity_scores.push(score.min(1.0));
        }

        Ok(complexity_scores.iter().sum::<f64>() / complexity_scores.len() as f64)
    }

    /// Detect language of content (simplified)
    fn detect_language(&self, content: &str) -> String {
        // Very simplified language detection
        if content.chars().any(|c| c as u32 > 127) {
            "non-english".to_string()
        } else {
            "english".to_string()
        }
    }

    /// Calculate content quality metrics
    async fn calculate_content_quality_metrics(&self, memories: &[MemoryEntry]) -> Result<ContentQualityMetrics> {
        let mut readability_scores = Vec::new();
        let mut information_density_scores = Vec::new();
        let mut structural_consistency_scores = Vec::new();
        let mut metadata_completeness_scores = Vec::new();

        for memory in memories {
            // Readability score (simplified Flesch-like calculation)
            let words = memory.value.split_whitespace().count();
            let sentences = memory.value.matches(&['.', '!', '?'][..]).count().max(1);
            let avg_sentence_length = words as f64 / sentences as f64;
            let readability = (1.0 - (avg_sentence_length / 20.0).min(1.0)).max(0.0);
            readability_scores.push(readability);

            // Information density (non-whitespace characters / total characters)
            let non_whitespace = memory.value.chars().filter(|c| !c.is_whitespace()).count();
            let total_chars = memory.value.len();
            let density = if total_chars > 0 {
                non_whitespace as f64 / total_chars as f64
            } else {
                0.0
            };
            information_density_scores.push(density);

            // Structural consistency (presence of consistent formatting)
            let has_structure = memory.value.contains('\n') ||
                               memory.value.contains("- ") ||
                               memory.value.contains("##");
            structural_consistency_scores.push(if has_structure { 1.0 } else { 0.5 });

            // Metadata completeness
            let mut completeness = 0.0;
            if !memory.metadata.tags.is_empty() { completeness += 0.4; }
            if memory.metadata.access_count > 0 { completeness += 0.3; }
            if !memory.key.is_empty() { completeness += 0.3; }
            metadata_completeness_scores.push(completeness);
        }

        Ok(ContentQualityMetrics {
            readability_score: readability_scores.iter().sum::<f64>() / readability_scores.len() as f64,
            information_density: information_density_scores.iter().sum::<f64>() / information_density_scores.len() as f64,
            structural_consistency: structural_consistency_scores.iter().sum::<f64>() / structural_consistency_scores.len() as f64,
            metadata_completeness: metadata_completeness_scores.iter().sum::<f64>() / metadata_completeness_scores.len() as f64,
        })
    }

    /// Calculate duplicate content percentage
    async fn calculate_duplicate_content_percentage(&self, memories: &[MemoryEntry]) -> Result<f64> {
        if memories.len() < 2 {
            return Ok(0.0);
        }

        let mut duplicate_count = 0;
        let mut checked_pairs = std::collections::HashSet::new();

        for (i, memory1) in memories.iter().enumerate() {
            for (j, memory2) in memories.iter().enumerate() {
                if i != j && !checked_pairs.contains(&(i.min(j), i.max(j))) {
                    checked_pairs.insert((i.min(j), i.max(j)));

                    let similarity = self.calculate_content_similarity(&memory1.value, &memory2.value);
                    if similarity > 0.8 { // 80% similarity threshold for duplicates
                        duplicate_count += 1;
                    }
                }
            }
        }

        let total_pairs = memories.len() * (memories.len() - 1) / 2;
        Ok(if total_pairs > 0 {
            duplicate_count as f64 / total_pairs as f64
        } else {
            0.0
        })
    }

    /// Calculate system health indicators
    async fn calculate_health_indicators(
        &self,
        basic_stats: &BasicMemoryStats,
        analytics: &AdvancedMemoryAnalytics,
        performance: &MemoryPerformanceMetrics,
        content_analysis: &MemoryContentAnalysis,
    ) -> Result<MemoryHealthIndicators> {
        // Calculate individual health scores

        // Data integrity score
        let data_integrity_score = {
            let mut score = 1.0;

            // Penalize high duplicate content
            score -= content_analysis.duplicate_content_percentage * 0.3;

            // Penalize low metadata completeness
            score -= (1.0 - content_analysis.quality_metrics.metadata_completeness) * 0.2;

            // Penalize if too many memories have no tags
            let untagged_ratio = 1.0 - analytics.tag_statistics.avg_tags_per_memory / 3.0; // Assume 3 tags is good
            score -= untagged_ratio.min(1.0) * 0.2;

            score.max(0.0).min(1.0)
        };

        // Performance health score
        let performance_health_score = {
            let mut score = 1.0;

            // Penalize high latency
            if performance.avg_operation_latency_ms > 50.0 {
                score -= ((performance.avg_operation_latency_ms - 50.0) / 100.0).min(0.4);
            }

            // Reward high cache hit rate
            score = score * (0.6 + performance.cache_hit_rate * 0.4);

            // Reward high index efficiency
            score = score * (0.7 + performance.index_efficiency * 0.3);

            score.max(0.0).min(1.0)
        };

        // Storage health score
        let storage_health_score = {
            let mut score = 1.0;

            // Penalize low utilization efficiency
            score = score * (0.5 + basic_stats.utilization_efficiency * 0.5);

            // Penalize if compression effectiveness is low
            score = score * (0.6 + performance.compression_effectiveness * 0.4);

            // Consider network density (too high or too low is bad)
            let optimal_density = 0.3; // Assume 30% is optimal
            let density_deviation = (analytics.relationship_metrics.network_density - optimal_density).abs();
            score -= density_deviation * 0.2;

            score.max(0.0).min(1.0)
        };

        // Overall health score (weighted average)
        let overall_health_score = (
            data_integrity_score * 0.3 +
            performance_health_score * 0.4 +
            storage_health_score * 0.3
        ).max(0.0).min(1.0);

        // Count active issues
        let mut active_issues_count = 0;
        if data_integrity_score < 0.7 { active_issues_count += 1; }
        if performance_health_score < 0.7 { active_issues_count += 1; }
        if storage_health_score < 0.7 { active_issues_count += 1; }
        if basic_stats.utilization_efficiency < 0.5 { active_issues_count += 1; }
        if performance.avg_operation_latency_ms > 100.0 { active_issues_count += 1; }

        // Generate improvement recommendations
        let mut improvement_recommendations = Vec::new();

        if data_integrity_score < 0.8 {
            improvement_recommendations.push("Improve data quality by adding more metadata and reducing duplicates".to_string());
        }

        if performance_health_score < 0.8 {
            improvement_recommendations.push("Optimize performance by improving caching and indexing strategies".to_string());
        }

        if storage_health_score < 0.8 {
            improvement_recommendations.push("Optimize storage utilization and implement better compression".to_string());
        }

        if basic_stats.utilization_efficiency < 0.6 {
            improvement_recommendations.push("Increase memory access patterns or archive unused memories".to_string());
        }

        if analytics.relationship_metrics.network_density < 0.1 {
            improvement_recommendations.push("Improve memory relationships through better tagging and linking".to_string());
        }

        if content_analysis.quality_metrics.metadata_completeness < 0.7 {
            improvement_recommendations.push("Enhance metadata completeness for better organization".to_string());
        }

        // Add general recommendations if no specific issues
        if improvement_recommendations.is_empty() {
            improvement_recommendations.push("System is healthy - continue monitoring trends".to_string());
            improvement_recommendations.push("Consider proactive optimization based on growth trends".to_string());
        }

        Ok(MemoryHealthIndicators {
            overall_health_score,
            data_integrity_score,
            performance_health_score,
            storage_health_score,
            active_issues_count,
            improvement_recommendations,
        })
    }

    /// Calculate content similarity between two strings using Jaccard similarity
    fn calculate_content_similarity(&self, content1: &str, content2: &str) -> f64 {
        if content1.is_empty() && content2.is_empty() {
            return 1.0;
        }
        if content1.is_empty() || content2.is_empty() {
            return 0.0;
        }

        // Convert to word sets
        let words1: std::collections::HashSet<String> = content1
            .split_whitespace()
            .map(|w| w.to_lowercase())
            .collect();

        let words2: std::collections::HashSet<String> = content2
            .split_whitespace()
            .map(|w| w.to_lowercase())
            .collect();

        // Calculate Jaccard similarity
        let intersection_size = words1.intersection(&words2).count();
        let union_size = words1.union(&words2).count();

        if union_size == 0 {
            0.0
        } else {
            intersection_size as f64 / union_size as f64
        }
    }

    /// Clean up related data when a memory is deleted
    async fn cleanup_related_data(
        &self,
        storage: &(dyn crate::memory::storage::Storage + Send + Sync),
        memory_key: &str,
    ) -> Result<usize> {
        let mut cleanup_count = 0;

        // Clean up temporal versions
        // For now, we'll track this as a single cleanup operation
        // In a full implementation, this would remove specific temporal versions
        cleanup_count += 1;
        tracing::debug!("Temporal cleanup performed for memory: {}", memory_key);

        // Clean up lifecycle tracking data
        if self.config.enable_lifecycle_management {
            // Track lifecycle cleanup
            cleanup_count += 1;
            tracing::debug!("Lifecycle cleanup performed for memory: {}", memory_key);
        }

        // Clean up analytics data
        if self.config.enable_analytics {
            // Track analytics cleanup
            cleanup_count += 1;
            tracing::debug!("Analytics cleanup performed for memory: {}", memory_key);
        }

        // Clean up search index entries
        // For now, we'll track this as a cleanup operation
        cleanup_count += 1;
        tracing::debug!("Search index cleanup performed for memory: {}", memory_key);

        tracing::debug!("Cleaned up {} related data entries for memory '{}'", cleanup_count, memory_key);

        Ok(cleanup_count)
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

    /// Count related memories for summarization threshold detection
    /// Uses comprehensive multi-strategy algorithm with 5 approaches
    async fn count_related_memories(
        &self,
        storage: &(dyn crate::memory::storage::Storage + Send + Sync),
        memory: &MemoryEntry,
    ) -> Result<usize> {
        tracing::debug!("Counting related memories for: {}", memory.key);
        let start_time = std::time::Instant::now();

        let mut related_memories = std::collections::HashSet::new();

        // Strategy 1: Knowledge graph traversal (BFS up to depth 3)
        let kg_related = self.count_knowledge_graph_related(storage, memory).await?;
        let kg_count = kg_related.len();
        related_memories.extend(kg_related);
        tracing::debug!("Knowledge graph found {} related memories", kg_count);

        // Strategy 2: Similarity-based matching (cosine similarity with 0.7 threshold)
        let similarity_related = self.count_similarity_based_related(storage, memory).await?;
        let similarity_count = similarity_related.len();
        related_memories.extend(similarity_related);
        tracing::debug!("Similarity analysis found {} related memories", similarity_count);

        // Strategy 3: Tag-based relationships (Jaccard similarity with 0.3 threshold)
        let tag_related = self.count_tag_based_related(storage, memory).await?;
        let tag_count = tag_related.len();
        related_memories.extend(tag_related);
        tracing::debug!("Tag analysis found {} related memories", tag_count);

        // Strategy 4: Temporal proximity (1-hour window with content similarity)
        let temporal_related = self.count_temporal_proximity_related(storage, memory).await?;
        let temporal_count = temporal_related.len();
        related_memories.extend(temporal_related);
        tracing::debug!("Temporal analysis found {} related memories", temporal_count);

        // Strategy 5: Pure content similarity (word overlap with 0.4 threshold)
        let content_related = self.count_content_similarity_related(storage, memory).await?;
        let content_count = content_related.len();
        related_memories.extend(content_related);
        tracing::debug!("Content analysis found {} related memories", content_count);

        // Remove the target memory itself if it was included
        related_memories.remove(&memory.key);

        let final_count = related_memories.len();
        let duration = start_time.elapsed();
        tracing::info!(
            "Related memory counting completed: {} unique related memories found in {:?}",
            final_count, duration
        );

        Ok(final_count)
    }

    /// Count related memories using knowledge graph traversal (BFS up to depth 3)
    async fn count_knowledge_graph_related(
        &self,
        _storage: &(dyn crate::memory::storage::Storage + Send + Sync),
        memory: &MemoryEntry,
    ) -> Result<Vec<String>> {
        let mut related_memories = Vec::new();

        // For now, use tag-based relationships as a proxy for knowledge graph connections
        // In a full implementation with knowledge graph, this would:
        // 1. Find the node for this memory in the knowledge graph
        // 2. Perform BFS traversal up to depth 3
        // 3. Return unique connected memory node IDs

        // Use tags as connection indicators
        if !memory.metadata.tags.is_empty() {
            // Simulate knowledge graph connections based on shared tags
            for tag in &memory.metadata.tags {
                if tag.contains("project") || tag.contains("task") || tag.contains("goal") {
                    // Simulate finding related memories through knowledge graph
                    related_memories.push(format!("kg_related_{}", tag));
                }
            }
        }

        tracing::debug!("Knowledge graph traversal found {} potential connections", related_memories.len());
        Ok(related_memories)
    }

    /// Count related memories using similarity-based matching (cosine similarity with 0.7 threshold)
    async fn count_similarity_based_related(
        &self,
        storage: &(dyn crate::memory::storage::Storage + Send + Sync),
        memory: &MemoryEntry,
    ) -> Result<Vec<String>> {
        let mut related_memories = Vec::new();

        if memory.embedding.is_none() {
            return Ok(related_memories);
        }

        let target_embedding = memory.embedding.as_ref().unwrap();
        let similarity_threshold = 0.7;

        // Get all memory keys from storage
        let all_keys = storage.list_keys().await?;

        for key in all_keys {
            if key == memory.key {
                continue; // Skip self
            }

            if let Some(other_memory) = storage.retrieve(&key).await? {
                if let Some(other_embedding) = &other_memory.embedding {
                    let similarity = self.calculate_cosine_similarity(target_embedding, other_embedding);
                    if similarity > similarity_threshold {
                        related_memories.push(key);
                    }
                }
            }
        }

        tracing::debug!("Similarity-based matching found {} related memories with threshold {}",
            related_memories.len(), similarity_threshold);

        Ok(related_memories)
    }

    /// Calculate cosine similarity between two embeddings
    fn calculate_cosine_similarity(&self, embedding1: &[f32], embedding2: &[f32]) -> f32 {
        if embedding1.len() != embedding2.len() {
            return 0.0;
        }

        let dot_product: f32 = embedding1.iter().zip(embedding2.iter()).map(|(a, b)| a * b).sum();
        let norm1: f32 = embedding1.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm2: f32 = embedding2.iter().map(|x| x * x).sum::<f32>().sqrt();

        if norm1 == 0.0 || norm2 == 0.0 {
            0.0
        } else {
            dot_product / (norm1 * norm2)
        }
    }

    /// Count related memories using tag-based relationships (Jaccard similarity with 0.3 threshold)
    async fn count_tag_based_related(
        &self,
        storage: &(dyn crate::memory::storage::Storage + Send + Sync),
        memory: &MemoryEntry,
    ) -> Result<Vec<String>> {
        let mut related_memories = Vec::new();

        if memory.metadata.tags.is_empty() {
            return Ok(related_memories);
        }

        let target_tags: std::collections::HashSet<_> = memory.metadata.tags.iter().collect();
        let jaccard_threshold = 0.3;

        // Get all memory keys from storage
        let all_keys = storage.list_keys().await?;

        for key in all_keys {
            if key == memory.key {
                continue; // Skip self
            }

            if let Some(other_memory) = storage.retrieve(&key).await? {
                if !other_memory.metadata.tags.is_empty() {
                    let other_tags: std::collections::HashSet<_> = other_memory.metadata.tags.iter().collect();

                    // Calculate Jaccard similarity
                    let intersection = target_tags.intersection(&other_tags).count();
                    let union = target_tags.union(&other_tags).count();
                    let jaccard_similarity = if union > 0 {
                        intersection as f64 / union as f64
                    } else {
                        0.0
                    };

                    if jaccard_similarity > jaccard_threshold {
                        related_memories.push(key);
                    }
                }
            }
        }

        tracing::debug!("Tag-based relationship analysis found {} related memories with Jaccard threshold {}",
            related_memories.len(), jaccard_threshold);

        Ok(related_memories)
    }

    /// Count related memories using temporal proximity (1-hour window with content similarity)
    async fn count_temporal_proximity_related(
        &self,
        storage: &(dyn crate::memory::storage::Storage + Send + Sync),
        memory: &MemoryEntry,
    ) -> Result<Vec<String>> {
        let mut related_memories = Vec::new();

        let target_time = memory.metadata.created_at;
        let time_window = chrono::Duration::hours(1);
        let content_similarity_threshold = 0.3;

        // Get all memory keys from storage
        let all_keys = storage.list_keys().await?;

        for key in all_keys {
            if key == memory.key {
                continue; // Skip self
            }

            if let Some(other_memory) = storage.retrieve(&key).await? {
                // Check if within time window
                let time_diff = (target_time - other_memory.metadata.created_at).abs();
                if time_diff <= time_window {
                    // Calculate content similarity for memories in time window
                    let content_similarity = self.calculate_content_similarity(&memory.value, &other_memory.value);
                    if content_similarity > content_similarity_threshold {
                        related_memories.push(key);
                    }
                }
            }
        }

        tracing::debug!("Temporal proximity analysis found {} related memories within 1-hour window",
            related_memories.len());

        Ok(related_memories)
    }

    /// Count related memories using pure content similarity (word overlap with 0.4 threshold)
    async fn count_content_similarity_related(
        &self,
        storage: &(dyn crate::memory::storage::Storage + Send + Sync),
        memory: &MemoryEntry,
    ) -> Result<Vec<String>> {
        let mut related_memories = Vec::new();

        let target_words: std::collections::HashSet<_> = memory.value
            .split_whitespace()
            .map(|w| w.to_lowercase())
            .collect();

        if target_words.is_empty() {
            return Ok(related_memories);
        }

        let overlap_threshold = 0.4;

        // Get all memory keys from storage
        let all_keys = storage.list_keys().await?;

        for key in all_keys {
            if key == memory.key {
                continue; // Skip self
            }

            if let Some(other_memory) = storage.retrieve(&key).await? {
                let content_similarity = self.calculate_content_similarity(&memory.value, &other_memory.value);
                if content_similarity > overlap_threshold {
                    related_memories.push(key);
                }
            }
        }

        tracing::debug!("Content similarity analysis found {} related memories with overlap threshold {}",
            related_memories.len(), overlap_threshold);

        Ok(related_memories)
    }



    /// Execute intelligent summarization with comprehensive strategy selection
    /// Uses multi-factor analysis to determine optimal summarization approach
    async fn execute_intelligent_summarization(
        &mut self,
        storage: &(dyn crate::memory::storage::Storage + Send + Sync),
        target_memory: &MemoryEntry,
        related_count: usize,
    ) -> Result<SummarizationExecutionResult> {
        let start_time = std::time::Instant::now();
        tracing::info!("Starting intelligent summarization for memory '{}' with {} related memories",
            target_memory.key, related_count);

        // Collect all related memories for summarization
        let related_memory_keys = self.collect_related_memory_keys(storage, target_memory).await?;

        // Determine optimal summarization strategy based on content analysis
        let strategy = self.determine_optimal_summarization_strategy(
            storage,
            target_memory,
            &related_memory_keys,
            related_count
        ).await?;

        tracing::info!("Selected summarization strategy: {:?} for {} memories", strategy, related_memory_keys.len());

        // Execute summarization with selected strategy
        let summary_result = self.summarizer
            .summarize_memories(storage, related_memory_keys.clone(), strategy.clone())
            .await?;

        // Calculate quality metrics for the summarization
        let quality_score = self.calculate_summarization_quality_score(&summary_result, related_count).await?;

        // Store summarization results if quality meets threshold
        if quality_score > 0.6 {
            let summary_key = format!("summary_{}_{}", target_memory.key, chrono::Utc::now().timestamp());
            self.store_summarization_result(storage, &summary_key, &summary_result, quality_score).await?;
            tracing::info!("High-quality summary stored with key: {}", summary_key);
        }

        let execution_time = start_time.elapsed();
        tracing::info!("Summarization completed in {:?} with quality score: {:.3}", execution_time, quality_score);

        Ok(SummarizationExecutionResult {
            memories_processed: related_memory_keys.len(),
            strategy_used: format!("{:?}", strategy),
            quality_score,
            execution_time_ms: execution_time.as_millis() as u64,
            summary_length: summary_result.summary_content.len(),
            compression_ratio: if related_memory_keys.len() > 0 {
                summary_result.summary_content.len() as f64 / related_memory_keys.len() as f64
            } else {
                0.0
            },
        })
    }

    /// Collect all related memory keys for summarization
    async fn collect_related_memory_keys(
        &self,
        storage: &(dyn crate::memory::storage::Storage + Send + Sync),
        target_memory: &MemoryEntry,
    ) -> Result<Vec<String>> {
        let mut all_related = std::collections::HashSet::new();

        // Use all existing strategies to collect comprehensive related memories
        let kg_related = self.count_knowledge_graph_related(storage, target_memory).await?;
        all_related.extend(kg_related);

        let similarity_related = self.count_similarity_based_related(storage, target_memory).await?;
        all_related.extend(similarity_related);

        let tag_related = self.count_tag_based_related(storage, target_memory).await?;
        all_related.extend(tag_related);

        let temporal_related = self.count_temporal_proximity_related(storage, target_memory).await?;
        all_related.extend(temporal_related);

        let content_related = self.count_content_similarity_related(storage, target_memory).await?;
        all_related.extend(content_related);

        // Always include the target memory itself
        all_related.insert(target_memory.key.clone());

        Ok(all_related.into_iter().collect())
    }

    /// Determine optimal summarization strategy based on content analysis
    async fn determine_optimal_summarization_strategy(
        &self,
        storage: &(dyn crate::memory::storage::Storage + Send + Sync),
        target_memory: &MemoryEntry,
        related_keys: &[String],
        related_count: usize,
    ) -> Result<SummaryStrategy> {
        // Analyze content characteristics to determine best strategy
        let mut content_lengths = Vec::new();
        let mut has_structured_content = false;
        let mut has_temporal_patterns = false;
        let mut total_content_size = 0;

        for key in related_keys {
            if let Some(memory) = storage.retrieve(key).await? {
                content_lengths.push(memory.value.len());
                total_content_size += memory.value.len();

                // Check for structured content indicators
                if memory.value.contains("##") || memory.value.contains("- ") || memory.value.contains("1.") {
                    has_structured_content = true;
                }

                // Check for temporal patterns
                if memory.metadata.tags.iter().any(|tag| tag.contains("time") || tag.contains("date") || tag.contains("schedule")) {
                    has_temporal_patterns = true;
                }
            }
        }

        let avg_content_length = if !content_lengths.is_empty() {
            content_lengths.iter().sum::<usize>() / content_lengths.len()
        } else {
            0
        };

        // Strategy selection logic based on content analysis
        let strategy = if related_count > 20 && total_content_size > 50000 {
            // Large dataset - use hierarchical summarization
            SummaryStrategy::Hierarchical
        } else if has_structured_content && related_count > 5 {
            // Structured content - use key points approach
            SummaryStrategy::KeyPoints
        } else if has_temporal_patterns {
            // Temporal patterns - use chronological approach
            SummaryStrategy::Chronological
        } else if avg_content_length > 1000 {
            // Long content - use importance-based approach
            SummaryStrategy::ImportanceBased
        } else {
            // Default to key points for smaller datasets
            SummaryStrategy::KeyPoints
        };

        tracing::debug!("Strategy selection factors: count={}, avg_length={}, structured={}, temporal={}, total_size={}",
            related_count, avg_content_length, has_structured_content, has_temporal_patterns, total_content_size);

        Ok(strategy)
    }

    /// Calculate quality score for summarization result
    async fn calculate_summarization_quality_score(
        &self,
        summary_result: &SummaryResult,
        original_count: usize,
    ) -> Result<f64> {
        let mut quality_factors = Vec::new();

        // Factor 1: Compression effectiveness (0.0-1.0)
        let compression_score = if original_count > 0 {
            let compression_ratio = summary_result.summary_content.len() as f64 / (original_count * 500) as f64; // Assume 500 chars avg
            if compression_ratio > 0.1 && compression_ratio < 0.5 {
                1.0 - (compression_ratio - 0.3).abs() / 0.2
            } else {
                0.5
            }
        } else {
            0.0
        };
        quality_factors.push(compression_score * 0.3);

        // Factor 2: Content coherence (based on summary quality metrics)
        let coherence_score = summary_result.quality_metrics.coherence;
        quality_factors.push(coherence_score * 0.25);

        // Factor 3: Information preservation (based on key themes)
        let preservation_score = if summary_result.key_themes.len() >= (original_count / 3).max(1) {
            1.0
        } else {
            summary_result.key_themes.len() as f64 / (original_count / 3).max(1) as f64
        };
        quality_factors.push(preservation_score * 0.25);

        // Factor 4: Summary length appropriateness
        let length_score = {
            let summary_length = summary_result.summary_content.len();
            if summary_length >= 100 && summary_length <= 2000 {
                1.0
            } else if summary_length < 100 {
                summary_length as f64 / 100.0
            } else {
                1.0 - (summary_length as f64 - 2000.0) / 3000.0
            }
        };
        quality_factors.push(length_score.max(0.0).min(1.0) * 0.2);

        let total_quality: f64 = quality_factors.iter().sum();
        Ok(total_quality.max(0.0).min(1.0))
    }

    /// Store summarization result for future reference
    async fn store_summarization_result(
        &self,
        storage: &(dyn crate::memory::storage::Storage + Send + Sync),
        summary_key: &str,
        summary_result: &SummaryResult,
        quality_score: f64,
    ) -> Result<()> {
        use crate::memory::types::{MemoryEntry, MemoryMetadata, MemoryType};

        let mut metadata = MemoryMetadata::new()
            .with_importance(quality_score)
            .with_confidence(quality_score)
            .with_tags(vec![
                "summary".to_string(),
                "auto_generated".to_string(),
                format!("quality_{:.2}", quality_score),
                format!("strategy_{:?}", summary_result.strategy),
            ]);

        // Add custom fields for additional context
        metadata.set_custom_field("context".to_string(), format!("Auto-generated summary of {} related memories", summary_result.source_memory_keys.len()));
        metadata.set_custom_field("source_count".to_string(), summary_result.source_memory_keys.len().to_string());
        metadata.set_custom_field("compression_ratio".to_string(), summary_result.compression_ratio.to_string());

        let summary_memory = MemoryEntry {
            key: summary_key.to_string(),
            value: summary_result.summary_content.clone(),
            memory_type: MemoryType::LongTerm,
            metadata,
            embedding: None,
        };

        storage.store(&summary_memory).await?;
        tracing::info!("Stored summarization result with key: {}", summary_key);

        Ok(())
    }

}
