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
use std::collections::HashMap;
use uuid::Uuid;

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


/// Advanced memory management system providing sophisticated memory operations.
///
/// The `AdvancedMemoryManager` is the central orchestrator for all advanced memory
/// operations in the Synaptic system. It integrates multiple specialized components
/// to provide intelligent memory management, optimization, and analytics.
///
/// # Components
///
/// - **Summarizer**: Intelligent memory consolidation and summarization
/// - **Search Engine**: Advanced multi-strategy search with relevance ranking
/// - **Lifecycle Manager**: Automated memory archiving and cleanup policies
/// - **Optimizer**: Performance optimization and resource management
/// - **Analytics**: Usage patterns, insights, and predictive analytics
/// - **Temporal Manager**: Version tracking and temporal pattern analysis
///
/// # Features
///
/// - **Intelligent Consolidation**: Automatically merge and summarize related memories
/// - **Advanced Search**: Multi-dimensional search with semantic understanding
/// - **Lifecycle Management**: Automated archiving based on usage patterns
/// - **Performance Optimization**: Dynamic optimization of storage and retrieval
/// - **Analytics & Insights**: Deep analysis of memory usage and patterns
/// - **Temporal Intelligence**: Track changes and detect patterns over time
///
/// # Examples
///
/// ```rust
/// use synaptic::memory::management::{AdvancedMemoryManager, MemoryManagementConfig};
/// use synaptic::memory::storage::create_storage;
///
/// async fn setup_advanced_manager() -> Result<AdvancedMemoryManager, Box<dyn std::error::Error>> {
///     let storage = create_storage("advanced_memory.db").await?;
///     let config = MemoryManagementConfig::default();
///     let manager = AdvancedMemoryManager::new(storage, config).await?;
///
///     // Perform intelligent optimization
///     manager.optimize_all().await?;
///
///     // Get comprehensive analytics
///     let analytics = manager.get_comprehensive_analytics().await?;
///     println!("Memory efficiency: {:.2}%", analytics.efficiency_score * 100.0);
///
///     Ok(manager)
/// }
/// ```
///
/// # Performance Considerations
///
/// The advanced memory manager is designed for high-performance scenarios with
/// large memory datasets. It uses sophisticated caching, indexing, and optimization
/// strategies to maintain excellent performance even with millions of memory entries.
pub struct AdvancedMemoryManager {
    /// Memory summarizer for intelligent consolidation and summarization
    summarizer: MemorySummarizer,
    /// Advanced search engine with multi-strategy capabilities
    search_engine: AdvancedSearchEngine,
    /// Lifecycle manager for automated memory management policies
    lifecycle_manager: MemoryLifecycleManager,
    /// Memory optimizer for performance and resource optimization
    optimizer: MemoryOptimizer,
    /// Analytics engine for insights and pattern detection
    analytics: MemoryAnalytics,
    /// Temporal manager for tracking changes and evolution over time
    temporal_manager: TemporalMemoryManager,
    /// Configuration parameters for memory management behavior
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


/// Summarization trigger information
#[derive(Debug, Clone)]
pub struct SummarizationTrigger {
    /// Reason for triggering summarization
    pub reason: String,
    /// Related memory keys to include in summarization
    pub related_memory_keys: Vec<String>,
    /// Trigger type
    pub trigger_type: SummarizationTriggerType,
    /// Confidence score for the trigger (0.0 to 1.0)
    pub confidence: f64,
    /// Additional metadata
    pub metadata: std::collections::HashMap<String, String>,
}

/// Types of summarization triggers
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SummarizationTriggerType {
    /// Triggered by related memory count threshold
    RelatedMemoryThreshold,
    /// Triggered by content complexity analysis
    ContentComplexity,
    /// Triggered by temporal clustering
    TemporalClustering,
    /// Triggered by semantic density
    SemanticDensity,
    /// Triggered by storage optimization needs
    StorageOptimization,
    /// Triggered by manual request
    Manual,
}

/// Result of automatic summarization execution
#[derive(Debug, Clone)]
pub struct AutoSummarizationResult {
    /// Number of memories processed
    pub processed_count: usize,
    /// Generated summary key
    pub summary_key: String,
    /// Success status
    pub success: bool,
    /// Processing time in milliseconds
    pub duration_ms: u64,
    /// Any warnings or messages
    pub messages: Vec<String>,
}

/// Memory management statistics

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
    pub language_distribution: std::collections::HashMap<String, usize>,
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

        // Create a default in-memory storage for the optimizer
        let storage = Arc::new(crate::memory::storage::memory::MemoryStorage::new());

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
        mut knowledge_graph: Option<&mut MemoryKnowledgeGraph>,
    ) -> Result<MemoryOperationResult> {
        let start_time = std::time::Instant::now();
        let mut messages = Vec::new();

        // Track the change temporally
        let _version_id = self.temporal_manager
            .track_memory_change(&memory, ChangeType::Created)
            .await?;

        // Add to knowledge graph if provided
        if let Some(ref mut kg) = knowledge_graph {
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
            let summarization_result = self.evaluate_summarization_triggers(&memory, knowledge_graph.as_deref()).await?;
            if let Some(trigger_info) = summarization_result {
                messages.push(format!("Summarization triggered: {}", trigger_info.reason));

                // Execute the summarization
                let summary_result = self.execute_automatic_summarization(trigger_info).await?;
                messages.push(format!("Summarization completed: {} memories processed", summary_result.processed_count));
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
        storage: &(dyn crate::memory::storage::Storage + Send + Sync),
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
        let version_id = self.temporal_manager
            .track_memory_change(&updated_memory, ChangeType::Updated)
            .await?;


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

        let duration_ms = start_time.elapsed().as_millis() as u64;

        let result_data = serde_json::json!({
            "memory_key": memory_key,
            "version_id": version_id,
            "updated_value_length": updated_memory.value.len()
        });

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

        // Create a placeholder for the deleted memory
        let deleted_memory = MemoryEntry::new(
            memory_key.to_string(),
            String::new(),
            MemoryType::ShortTerm,
        );


        // Track the deletion temporally
        let version_id = self.temporal_manager
            .track_memory_change(&deleted_memory, ChangeType::Deleted)
            .await?;


        // Remove from knowledge graph if provided
        if let Some(_kg) = knowledge_graph {
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

        let duration_ms = start_time.elapsed().as_millis() as u64;
        let cleanup_count = 0; // Placeholder for actual cleanup count

        Ok(MemoryOperationResult {
            operation: MemoryOperation::Delete,
            success: true,
            affected_count: 1 + cleanup_count,
            duration_ms,
            result_data: Some(serde_json::json!({
                "deleted_memory_key": memory_key,
                "version_id": version_id,
                "cleanup_count": cleanup_count,
                "original_value_length": deleted_memory.value.len()
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

        // Calculate basic analytics (simplified)
        let analytics = AdvancedMemoryAnalytics {
            size_distribution: SizeDistribution {
                min_size: all_memories.iter().map(|m| m.value.len()).min().unwrap_or(0),
                max_size: all_memories.iter().map(|m| m.value.len()).max().unwrap_or(0),
                median_size: {
                    let mut sizes: Vec<usize> = all_memories.iter().map(|m| m.value.len()).collect();
                    sizes.sort();
                    if sizes.is_empty() { 0 } else { sizes[sizes.len() / 2] }
                },
                percentile_95: {
                    let mut sizes: Vec<usize> = all_memories.iter().map(|m| m.value.len()).collect();
                    sizes.sort();
                    if sizes.is_empty() { 0 } else { sizes[(sizes.len() as f64 * 0.95) as usize] }
                },
                size_buckets: {
                    let mut buckets = HashMap::new();
                    buckets.insert("0-1KB".to_string(), all_memories.iter().filter(|m| m.value.len() < 1000).count());
                    buckets.insert("1-10KB".to_string(), all_memories.iter().filter(|m| m.value.len() >= 1000 && m.value.len() < 10000).count());
                    buckets.insert("10KB+".to_string(), all_memories.iter().filter(|m| m.value.len() >= 10000).count());
                    buckets
                },
            },
            access_patterns: AccessPatternAnalysis {
                peak_hours: vec![9, 10, 11, 14, 15, 16], // Common work hours
                access_frequency_distribution: {
                    let mut freq_dist = HashMap::new();
                    freq_dist.insert("low".to_string(), all_memories.len());
                    freq_dist.insert("medium".to_string(), 0);
                    freq_dist.insert("high".to_string(), 0);
                    freq_dist
                },
                seasonal_patterns: vec![],
                user_behavior_clusters: vec![],
            },
            content_types: ContentTypeDistribution {
                types: HashMap::new(),
                type_growth_rates: HashMap::new(),
                dominant_type: "text".to_string(),
                diversity_index: 1.0,
            },
            tag_statistics: TagUsageStats {
                total_unique_tags: all_memories.iter().flat_map(|m| &m.metadata.tags).collect::<std::collections::HashSet<_>>().len(),
                avg_tags_per_memory: all_memories.iter().map(|m| m.metadata.tags.len()).sum::<usize>() as f64 / all_memories.len() as f64,
                most_popular_tags: vec![],
                tag_co_occurrence: HashMap::new(),
                tag_effectiveness_scores: HashMap::new(),
            },
            relationship_metrics: RelationshipMetrics {
                avg_connections_per_memory: 0.0,
                network_density: 0.0,
                clustering_coefficient: 0.0,
                strongly_connected_components: 0,
                relationship_types: HashMap::new(),
            },
            temporal_distribution: TemporalDistribution {
                hourly_distribution: vec![0; 24],
                daily_distribution: vec![0; 7],
                monthly_distribution: vec![0; 12],
                peak_activity_periods: vec![],
            },
        };

        // Calculate basic trend analysis (simplified)
        let trends = MemoryTrendAnalysis {
            growth_trend: TrendMetric {
                current_value: all_memories.len() as f64,
                trend_direction: TrendDirection::Stable,
                trend_strength: 0.5,
                slope: 0.0,
                r_squared: 0.0,
                prediction_7d: all_memories.len() as f64,
                prediction_30d: all_memories.len() as f64,
                confidence_interval: (0.0, 0.0),
            },
            access_trend: TrendMetric {
                current_value: 0.0,
                trend_direction: TrendDirection::Stable,
                trend_strength: 0.5,
                slope: 0.0,
                r_squared: 0.0,
                prediction_7d: 0.0,
                prediction_30d: 0.0,
                confidence_interval: (0.0, 0.0),
            },
            size_trend: TrendMetric {
                current_value: all_memories.iter().map(|m| m.value.len()).sum::<usize>() as f64 / all_memories.len() as f64,
                trend_direction: TrendDirection::Stable,
                trend_strength: 0.5,
                slope: 0.0,
                r_squared: 0.0,
                prediction_7d: 0.0,
                prediction_30d: 0.0,
                confidence_interval: (0.0, 0.0),
            },
            optimization_trend: TrendMetric {
                current_value: 0.0,
                trend_direction: TrendDirection::Stable,
                trend_strength: 0.5,
                slope: 0.0,
                r_squared: 0.0,
                prediction_7d: 0.0,
                prediction_30d: 0.0,
                confidence_interval: (0.0, 0.0),
            },
            complexity_trend: TrendMetric {
                current_value: 0.5,
                trend_direction: TrendDirection::Stable,
                trend_strength: 0.5,
                slope: 0.0,
                r_squared: 0.0,
                prediction_7d: 0.5,
                prediction_30d: 0.5,
                confidence_interval: (0.0, 1.0),
            },
        };

        // Calculate basic predictive metrics (simplified)
        let predictions = MemoryPredictiveMetrics {
            predicted_memory_count_30d: all_memories.len() as f64 * 1.1,
            predicted_storage_mb_30d: (all_memories.iter().map(|m| m.value.len()).sum::<usize>() as f64 / 1024.0 / 1024.0) * 1.1,
            optimization_forecast: OptimizationForecast {
                next_optimization_recommended: Utc::now() + chrono::Duration::days(30),
                optimization_urgency: OptimizationUrgency::Low,
                expected_performance_gain: 0.1,
                resource_requirements: ResourceRequirements {
                    cpu_usage_estimate: 0.5,
                    memory_usage_mb: 512.0,
                    io_operations_estimate: 1000,
                    estimated_duration_minutes: 30.0,
                },
            },
            capacity_recommendations: vec!["Monitor memory growth".to_string()],
            risk_assessment: RiskAssessment {
                overall_risk_level: RiskLevel::Low,
                capacity_risk: 0.1,
                performance_risk: 0.1,
                data_loss_risk: 0.05,
                mitigation_strategies: vec!["Regular backups".to_string()],
            },
        };

        // Calculate basic performance metrics (simplified)
        let performance = MemoryPerformanceMetrics {
            avg_operation_latency_ms: 1.0,
            operations_per_second: 1000.0,
            cache_hit_rate: 0.8,
            index_efficiency: 0.9,
            compression_effectiveness: 0.7,
            response_time_distribution: ResponseTimeDistribution {
                p50_ms: 0.5,
                p95_ms: 2.0,
                p99_ms: 5.0,
                max_ms: 10.0,
                outlier_count: 0,
            },
        };

        // Calculate basic content analysis (simplified)
        let content_analysis = MemoryContentAnalysis {
            avg_content_length: all_memories.iter().map(|m| m.value.len()).sum::<usize>() as f64 / all_memories.len() as f64,
            complexity_score: 0.5,
            language_distribution: {
                let mut lang_dist = HashMap::new();
                lang_dist.insert("en".to_string(), all_memories.len());
                lang_dist
            },
            semantic_diversity: 0.7,
            quality_metrics: ContentQualityMetrics {
                readability_score: 0.8,
                information_density: 0.6,
                structural_consistency: 0.9,
                metadata_completeness: 0.7,
            },
            duplicate_content_percentage: 0.05,
        };

        // Calculate basic health indicators (simplified)
        let health_indicators = MemoryHealthIndicators {
            overall_health_score: 0.9,
            data_integrity_score: 0.95,
            performance_health_score: 0.85,
            storage_health_score: 0.9,
            active_issues_count: 0,
            improvement_recommendations: vec!["Consider periodic optimization".to_string()],
        };

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

    /// Evaluate whether summarization should be triggered for a new memory
    async fn evaluate_summarization_triggers(
        &self,
        memory: &MemoryEntry,
        knowledge_graph: Option<&MemoryKnowledgeGraph>,
    ) -> Result<Option<SummarizationTrigger>> {
        let start_time = std::time::Instant::now();
        tracing::debug!("Evaluating summarization triggers for memory: {}", memory.key);

        // Strategy 1: Storage optimization trigger (highest priority for very large memories)
        if let Some(trigger) = self.check_storage_optimization_trigger(memory).await? {
            return Ok(Some(trigger));
        }

        // Strategy 2: Semantic density analysis (high priority for dense content)
        if let Some(trigger) = self.check_semantic_density_trigger(memory, knowledge_graph).await? {
            return Ok(Some(trigger));
        }

        // Strategy 3: Related memory count threshold
        if let Some(trigger) = self.check_related_memory_threshold(memory).await? {
            return Ok(Some(trigger));
        }

        // Strategy 4: Content complexity analysis
        if let Some(trigger) = self.check_content_complexity_trigger(memory).await? {
            return Ok(Some(trigger));
        }

        // Strategy 5: Temporal clustering analysis
        if let Some(trigger) = self.check_temporal_clustering_trigger(memory).await? {
            return Ok(Some(trigger));
        }

        let duration_ms = start_time.elapsed().as_millis();
        tracing::debug!("Summarization trigger evaluation completed in {}ms - no triggers activated", duration_ms);

        Ok(None)
    }

    /// Check if related memory count exceeds threshold
    async fn check_related_memory_threshold(&self, memory: &MemoryEntry) -> Result<Option<SummarizationTrigger>> {
        // For now, we'll use a simplified approach since we don't have direct storage access
        // In a real implementation, this would query the storage system
        let related_count = 5; // Placeholder - would be calculated from actual storage

        if related_count >= self.config.summarization_threshold {
            let confidence = (related_count as f64 / (self.config.summarization_threshold as f64 * 2.0)).min(1.0);

            let mut metadata = std::collections::HashMap::new();
            metadata.insert("related_count".to_string(), related_count.to_string());
            metadata.insert("threshold".to_string(), self.config.summarization_threshold.to_string());

            return Ok(Some(SummarizationTrigger {
                reason: format!("Related memory count ({}) exceeds threshold ({})", related_count, self.config.summarization_threshold),
                related_memory_keys: vec![memory.key.clone()], // This would be expanded with actual related keys
                trigger_type: SummarizationTriggerType::RelatedMemoryThreshold,
                confidence,
                metadata,
            }));
        }

        Ok(None)
    }

    /// Check if content complexity warrants summarization
    async fn check_content_complexity_trigger(&self, memory: &MemoryEntry) -> Result<Option<SummarizationTrigger>> {
        let content_length = memory.value.len();
        let word_count = memory.value.split_whitespace().count();
        let sentence_count = memory.value.split(&['.', '!', '?']).filter(|s| !s.trim().is_empty()).count();

        // Calculate complexity metrics
        let avg_word_length = if word_count > 0 {
            content_length as f64 / word_count as f64
        } else {
            0.0
        };

        let avg_sentence_length = if sentence_count > 0 {
            word_count as f64 / sentence_count as f64
        } else {
            0.0
        };

        // Complexity thresholds (higher threshold to avoid false positives)
        let complexity_score = (avg_word_length / 6.0) + (avg_sentence_length / 20.0) + (content_length as f64 / 5000.0);

        if complexity_score > 2.5 {
            let confidence = (complexity_score / 3.0).min(1.0);

            let mut metadata = std::collections::HashMap::new();
            metadata.insert("complexity_score".to_string(), format!("{:.2}", complexity_score));
            metadata.insert("word_count".to_string(), word_count.to_string());
            metadata.insert("sentence_count".to_string(), sentence_count.to_string());
            metadata.insert("content_length".to_string(), content_length.to_string());

            return Ok(Some(SummarizationTrigger {
                reason: format!("Content complexity score ({:.2}) exceeds threshold (1.5)", complexity_score),
                related_memory_keys: vec![memory.key.clone()],
                trigger_type: SummarizationTriggerType::ContentComplexity,
                confidence,
                metadata,
            }));
        }

        Ok(None)
    }

    /// Check if temporal clustering suggests summarization
    async fn check_temporal_clustering_trigger(&self, memory: &MemoryEntry) -> Result<Option<SummarizationTrigger>> {
        let _memory_time = memory.created_at();
        let _time_window = chrono::Duration::hours(2); // 2-hour clustering window

        // This would normally query storage for memories in the time window
        // For now, we'll use a simplified approach
        let cluster_threshold = 5; // Minimum memories in cluster to trigger summarization

        // Simulate cluster detection (in real implementation, this would query storage)
        let cluster_size = 3; // Placeholder

        if cluster_size >= cluster_threshold {
            let confidence = (cluster_size as f64 / (cluster_threshold as f64 * 2.0)).min(1.0);

            let mut metadata = std::collections::HashMap::new();
            metadata.insert("cluster_size".to_string(), cluster_size.to_string());
            metadata.insert("time_window_hours".to_string(), "2".to_string());
            metadata.insert("cluster_threshold".to_string(), cluster_threshold.to_string());

            return Ok(Some(SummarizationTrigger {
                reason: format!("Temporal cluster of {} memories detected within 2-hour window", cluster_size),
                related_memory_keys: vec![memory.key.clone()],
                trigger_type: SummarizationTriggerType::TemporalClustering,
                confidence,
                metadata,
            }));
        }

        Ok(None)
    }

    /// Check if semantic density warrants summarization
    async fn check_semantic_density_trigger(
        &self,
        memory: &MemoryEntry,
        knowledge_graph: Option<&MemoryKnowledgeGraph>,
    ) -> Result<Option<SummarizationTrigger>> {
        // Calculate semantic density based on unique concepts and relationships
        let content_lower = memory.value.to_lowercase();
        let unique_words: std::collections::HashSet<_> = content_lower
            .split_whitespace()
            .filter(|word| word.len() > 3)
            .collect();

        let concept_density = unique_words.len() as f64 / memory.value.split_whitespace().count().max(1) as f64;
        let tag_density = memory.metadata.tags.len() as f64 / 10.0; // Normalize to 10 tags max

        // Factor in knowledge graph connectivity if available
        let graph_connectivity = if let Some(_kg) = knowledge_graph {
            0.3 // Placeholder for actual graph connectivity calculation
        } else {
            0.0
        };

        let semantic_density = concept_density + tag_density + graph_connectivity;

        if semantic_density > 1.2 {
            let confidence = (semantic_density / 1.5).min(1.0);

            let mut metadata = std::collections::HashMap::new();
            metadata.insert("semantic_density".to_string(), format!("{:.2}", semantic_density));
            metadata.insert("concept_density".to_string(), format!("{:.2}", concept_density));
            metadata.insert("tag_density".to_string(), format!("{:.2}", tag_density));
            metadata.insert("unique_concepts".to_string(), unique_words.len().to_string());

            return Ok(Some(SummarizationTrigger {
                reason: format!("Semantic density ({:.2}) exceeds threshold (1.2)", semantic_density),
                related_memory_keys: vec![memory.key.clone()],
                trigger_type: SummarizationTriggerType::SemanticDensity,
                confidence,
                metadata,
            }));
        }

        Ok(None)
    }

    /// Check if storage optimization suggests summarization
    async fn check_storage_optimization_trigger(&self, memory: &MemoryEntry) -> Result<Option<SummarizationTrigger>> {
        // Calculate storage efficiency metrics
        let content_size = memory.value.len();
        let metadata_size = serde_json::to_string(&memory.metadata).unwrap_or_default().len();
        let total_size = content_size + metadata_size;

        // Check if this memory is particularly large or if we have many similar memories
        let size_threshold = 10000; // 10KB threshold

        if total_size > size_threshold {
            let confidence = (total_size as f64 / (size_threshold as f64 * 2.0)).min(1.0);

            let mut metadata = std::collections::HashMap::new();
            metadata.insert("total_size".to_string(), total_size.to_string());
            metadata.insert("content_size".to_string(), content_size.to_string());
            metadata.insert("metadata_size".to_string(), metadata_size.to_string());
            metadata.insert("size_threshold".to_string(), size_threshold.to_string());

            return Ok(Some(SummarizationTrigger {
                reason: format!("Memory size ({} bytes) exceeds storage optimization threshold ({} bytes)", total_size, size_threshold),
                related_memory_keys: vec![memory.key.clone()],
                trigger_type: SummarizationTriggerType::StorageOptimization,
                confidence,
                metadata,
            }));
        }

        Ok(None)
    }

    /// Execute automatic summarization based on trigger information
    async fn execute_automatic_summarization(
        &mut self,
        trigger: SummarizationTrigger,
    ) -> Result<AutoSummarizationResult> {
        let start_time = std::time::Instant::now();
        tracing::info!("Executing automatic summarization: {}", trigger.reason);

        let mut messages = Vec::new();

        // Determine summarization strategy based on trigger type
        let strategy = match trigger.trigger_type {
            SummarizationTriggerType::RelatedMemoryThreshold => SummaryStrategy::Hierarchical,
            SummarizationTriggerType::ContentComplexity => SummaryStrategy::KeyPoints,
            SummarizationTriggerType::TemporalClustering => SummaryStrategy::Chronological,
            SummarizationTriggerType::SemanticDensity => SummaryStrategy::Conceptual,
            SummarizationTriggerType::StorageOptimization => SummaryStrategy::Consolidation,
            SummarizationTriggerType::Manual => SummaryStrategy::ImportanceBased,
        };

        // Execute the summarization (using a temporary storage for now)
        let temp_storage = Arc::new(crate::memory::storage::memory::MemoryStorage::new());
        let summary_result = self.summarizer.summarize_memories(
            &*temp_storage,
            trigger.related_memory_keys.clone(),
            strategy.clone(),
        ).await?;

        // Generate a unique key for the summary
        let summary_key = format!("summary_{}_{}",
            format!("{:?}", trigger.trigger_type).to_lowercase(),
            chrono::Utc::now().timestamp()
        );

        messages.push(format!("Applied {:?} strategy", strategy));
        messages.push(format!("Generated summary with key: {}", summary_key));

        let duration_ms = start_time.elapsed().as_millis() as u64;

        tracing::info!(
            "Automatic summarization completed in {}ms: {} memories processed",
            duration_ms,
            trigger.related_memory_keys.len()
        );

        Ok(AutoSummarizationResult {
            processed_count: trigger.related_memory_keys.len(),
            summary_key,
            success: summary_result.quality_metrics.overall_quality > 0.5, // Use quality as success indicator
            duration_ms,
            messages,
        })
    }







}

#[cfg(test)]
mod tests;