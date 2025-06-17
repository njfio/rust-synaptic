//! Memory Importance Scoring Engine
//! 
//! Implements sophisticated algorithms for scoring memory importance based on
//! access patterns, recency, relationships, and semantic significance.

use crate::error::Result;
use crate::memory::types::MemoryEntry;
use super::{ConsolidationConfig, MemoryImportance};
use chrono::{DateTime, Utc};
use std::collections::HashMap;
use std::sync::Arc;
use serde::{Deserialize, Serialize};
use async_trait::async_trait;

/// Simple key-value storage trait for importance scorer persistence
#[async_trait]
pub trait KeyValueStorage: Send + Sync {
    /// Get a value by key
    async fn get(&self, key: &str) -> Result<String>;

    /// Set a value by key
    async fn set(&self, key: &str, value: &str) -> Result<()>;

    /// Check if a key exists
    async fn exists(&self, key: &str) -> Result<bool>;

    /// Delete a key
    async fn delete(&self, key: &str) -> Result<bool>;
}

/// In-memory implementation of KeyValueStorage for testing
#[derive(Debug, Default)]
pub struct MemoryKeyValueStorage {
    data: Arc<tokio::sync::RwLock<HashMap<String, String>>>,
}

impl MemoryKeyValueStorage {
    pub fn new() -> Self {
        Self {
            data: Arc::new(tokio::sync::RwLock::new(HashMap::new())),
        }
    }
}

#[async_trait]
impl KeyValueStorage for MemoryKeyValueStorage {
    async fn get(&self, key: &str) -> Result<String> {
        let data = self.data.read().await;
        data.get(key)
            .cloned()
            .ok_or_else(|| crate::error::MemoryError::NotFound { key: key.to_string() })
    }

    async fn set(&self, key: &str, value: &str) -> Result<()> {
        let mut data = self.data.write().await;
        data.insert(key.to_string(), value.to_string());
        Ok(())
    }

    async fn exists(&self, key: &str) -> Result<bool> {
        let data = self.data.read().await;
        Ok(data.contains_key(key))
    }

    async fn delete(&self, key: &str) -> Result<bool> {
        let mut data = self.data.write().await;
        Ok(data.remove(key).is_some())
    }
}

/// Access pattern for importance calculation
#[derive(Debug, Clone, Serialize, Deserialize)]
struct AccessPattern {
    /// Total access count
    access_count: u64,
    /// Last access timestamp
    last_accessed: DateTime<Utc>,
    /// Access frequency (accesses per day)
    frequency: f64,
    /// Access consistency score
    consistency: f64,
}

/// Relationship metrics for centrality calculation
#[derive(Debug, Clone, Serialize, Deserialize)]
struct RelationshipMetrics {
    /// Number of incoming relationships
    in_degree: usize,
    /// Number of outgoing relationships
    out_degree: usize,
    /// Weighted relationship strength sum
    relationship_strength: f64,
    /// Betweenness centrality approximation
    betweenness: f64,
}

/// Content analysis for uniqueness scoring
#[derive(Debug, Clone, Serialize, Deserialize)]
struct ContentAnalysis {
    /// Content length
    length: usize,
    /// Vocabulary richness (unique words / total words)
    vocabulary_richness: f64,
    /// Semantic density score
    semantic_density: f64,
    /// Information entropy
    entropy: f64,
}

/// Memory importance scoring engine
pub struct ImportanceScorer {
    /// Configuration
    config: ConsolidationConfig,
    /// Access pattern history
    access_patterns: HashMap<String, AccessPattern>,
    /// Relationship metrics cache
    relationship_cache: HashMap<String, RelationshipMetrics>,
    /// Content analysis cache
    content_cache: HashMap<String, ContentAnalysis>,
    /// Global importance statistics
    global_stats: ImportanceStatistics,
    /// Storage backend for persistence
    storage: Option<Arc<dyn KeyValueStorage>>,
}

/// Global importance statistics for normalization
#[derive(Debug, Clone, Serialize, Deserialize)]
struct ImportanceStatistics {
    /// Mean access frequency across all memories
    mean_access_frequency: f64,
    /// Standard deviation of access frequency
    std_access_frequency: f64,
    /// Mean relationship centrality
    mean_centrality: f64,
    /// Standard deviation of centrality
    std_centrality: f64,
    /// Mean content uniqueness
    mean_uniqueness: f64,
    /// Standard deviation of uniqueness
    std_uniqueness: f64,
}

impl Default for ImportanceStatistics {
    fn default() -> Self {
        Self {
            mean_access_frequency: 1.0,
            std_access_frequency: 0.5,
            mean_centrality: 0.5,
            std_centrality: 0.2,
            mean_uniqueness: 0.5,
            std_uniqueness: 0.2,
        }
    }
}

impl std::fmt::Debug for ImportanceScorer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ImportanceScorer")
            .field("config", &self.config)
            .field("access_patterns", &self.access_patterns)
            .field("relationship_cache", &self.relationship_cache)
            .field("content_cache", &self.content_cache)
            .field("global_stats", &self.global_stats)
            .field("storage", &self.storage.is_some())
            .finish()
    }
}

impl ImportanceScorer {
    /// Create a new importance scorer
    pub fn new(config: &ConsolidationConfig) -> Result<Self> {
        Ok(Self {
            config: config.clone(),
            access_patterns: HashMap::new(),
            relationship_cache: HashMap::new(),
            content_cache: HashMap::new(),
            global_stats: ImportanceStatistics::default(),
            storage: None,
        })
    }

    /// Create a new importance scorer with storage backend
    pub fn with_storage(config: &ConsolidationConfig, storage: Arc<dyn KeyValueStorage>) -> Result<Self> {
        let mut scorer = Self::new(config)?;
        scorer.storage = Some(storage);
        Ok(scorer)
    }

    /// Load importance data from storage
    pub async fn load_from_storage(&mut self) -> Result<()> {
        if let Some(storage) = &self.storage {
            // Load access patterns
            if let Ok(data) = storage.get("importance_scorer:access_patterns").await {
                if let Ok(patterns) = serde_json::from_str::<HashMap<String, AccessPattern>>(&data) {
                    self.access_patterns = patterns;
                }
            }

            // Load relationship cache
            if let Ok(data) = storage.get("importance_scorer:relationship_cache").await {
                if let Ok(cache) = serde_json::from_str::<HashMap<String, RelationshipMetrics>>(&data) {
                    self.relationship_cache = cache;
                }
            }

            // Load content cache
            if let Ok(data) = storage.get("importance_scorer:content_cache").await {
                if let Ok(cache) = serde_json::from_str::<HashMap<String, ContentAnalysis>>(&data) {
                    self.content_cache = cache;
                }
            }

            // Load global statistics
            if let Ok(data) = storage.get("importance_scorer:global_stats").await {
                if let Ok(stats) = serde_json::from_str::<ImportanceStatistics>(&data) {
                    self.global_stats = stats;
                }
            }
        }
        Ok(())
    }

    /// Save importance data to storage
    pub async fn save_to_storage(&self) -> Result<()> {
        if let Some(storage) = &self.storage {
            // Save access patterns
            let patterns_json = serde_json::to_string(&self.access_patterns)?;
            storage.set("importance_scorer:access_patterns", &patterns_json).await?;

            // Save relationship cache
            let relationships_json = serde_json::to_string(&self.relationship_cache)?;
            storage.set("importance_scorer:relationship_cache", &relationships_json).await?;

            // Save content cache
            let content_json = serde_json::to_string(&self.content_cache)?;
            storage.set("importance_scorer:content_cache", &content_json).await?;

            // Save global statistics
            let stats_json = serde_json::to_string(&self.global_stats)?;
            storage.set("importance_scorer:global_stats", &stats_json).await?;
        }
        Ok(())
    }

    /// Calculate importance scores for a batch of memories
    pub async fn calculate_batch_importance(&mut self, memories: &[MemoryEntry]) -> Result<Vec<MemoryImportance>> {
        let mut importance_scores = Vec::new();

        // Update global statistics
        self.update_global_statistics(memories).await?;

        for memory in memories {
            let importance = self.calculate_single_importance(memory).await?;
            importance_scores.push(importance);
        }

        // Auto-save to storage if available
        self.save_to_storage().await?;

        Ok(importance_scores)
    }

    /// Calculate importance score for a single memory
    async fn calculate_single_importance(&mut self, memory: &MemoryEntry) -> Result<MemoryImportance> {
        let memory_key = &memory.key;

        // Calculate component scores
        let access_frequency = self.calculate_access_frequency_score(memory).await?;
        let recency_score = self.calculate_recency_score(memory).await?;
        let centrality_score = self.calculate_centrality_score(memory).await?;
        let uniqueness_score = self.calculate_uniqueness_score(memory).await?;
        let temporal_consistency = self.calculate_temporal_consistency_score(memory).await?;

        // Weighted combination of scores
        let importance_score = self.combine_importance_factors(
            access_frequency,
            recency_score,
            centrality_score,
            uniqueness_score,
            temporal_consistency,
        );

        // Calculate Fisher information for EWC if enabled
        let fisher_information = if self.config.enable_ewc {
            Some(self.calculate_fisher_information(memory).await?)
        } else {
            None
        };

        Ok(MemoryImportance {
            memory_key: memory_key.clone(),
            importance_score,
            access_frequency,
            recency_score,
            centrality_score,
            uniqueness_score,
            temporal_consistency,
            calculated_at: Utc::now(),
            fisher_information,
        })
    }

    /// Calculate access frequency score with sophisticated pattern analysis
    async fn calculate_access_frequency_score(&mut self, memory: &MemoryEntry) -> Result<f64> {
        let access_count = memory.access_count() as f64;
        let days_since_creation = (Utc::now() - memory.created_at()).num_days().max(1) as f64;
        
        // Calculate frequency (accesses per day)
        let frequency = access_count / days_since_creation;
        
        // Normalize using z-score with global statistics
        let normalized_frequency = if self.global_stats.std_access_frequency > 0.0 {
            (frequency - self.global_stats.mean_access_frequency) / self.global_stats.std_access_frequency
        } else {
            0.0
        };

        // Apply sigmoid transformation to bound between 0 and 1
        let score = 1.0 / (1.0 + (-normalized_frequency).exp());

        // Store access pattern for future use
        self.access_patterns.insert(memory.key.clone(), AccessPattern {
            access_count: memory.access_count(),
            last_accessed: memory.last_accessed(),
            frequency,
            consistency: self.calculate_access_consistency(memory),
        });

        Ok(score)
    }

    /// Calculate recency score with sophisticated temporal decay models
    async fn calculate_recency_score(&self, memory: &MemoryEntry) -> Result<f64> {
        let hours_since_access = (Utc::now() - memory.last_accessed()).num_hours().max(0) as f64;
        let hours_since_creation = (Utc::now() - memory.created_at()).num_hours().max(1) as f64;

        // Multi-model temporal decay approach

        // 1. Exponential decay (Ebbinghaus forgetting curve)
        let half_life_hours = self.calculate_adaptive_half_life(memory);
        let decay_constant = (2.0_f64).ln() / half_life_hours;
        let exponential_score = (-decay_constant * hours_since_access).exp();

        // 2. Power law decay (more gradual for important memories)
        let power_exponent = if memory.metadata.importance > 0.7 {
            0.5 // Slower decay for important memories
        } else {
            0.8 // Faster decay for less important memories
        };
        let power_score = (1.0 + hours_since_access / 24.0).powf(-power_exponent);

        // 3. Logarithmic decay (for very recent access)
        let log_score = if hours_since_access < 1.0 {
            1.0 // Perfect score for very recent access
        } else {
            1.0 - (hours_since_access.ln() / 168.0_f64.ln()).min(1.0) // 1 week normalization
        };

        // 4. Access frequency modulated decay
        let access_frequency = memory.access_count() as f64 / (hours_since_creation / 24.0);
        let frequency_factor = if access_frequency > 1.0 {
            1.2 // Boost for frequently accessed memories
        } else if access_frequency > 0.1 {
            1.0 // Normal decay
        } else {
            0.8 // Faster decay for rarely accessed memories
        };

        // Combine decay models with adaptive weights
        let weights = self.calculate_decay_model_weights(memory);
        let combined_score = (exponential_score * weights.0 +
                             power_score * weights.1 +
                             log_score * weights.2) * frequency_factor;

        Ok(combined_score.min(1.0).max(0.0))
    }

    /// Calculate adaptive half-life based on memory characteristics
    fn calculate_adaptive_half_life(&self, memory: &MemoryEntry) -> f64 {
        let base_half_life = 24.0; // 24 hours base

        // Adjust based on importance
        let importance_factor = 1.0 + memory.metadata.importance * 2.0; // 1.0 to 3.0 range

        // Adjust based on access frequency
        let access_count = memory.access_count() as f64;
        let frequency_factor = if access_count > 10.0 {
            2.0 // Longer half-life for frequently accessed
        } else if access_count > 5.0 {
            1.5
        } else {
            1.0
        };

        // Adjust based on content complexity
        let content_factor = if memory.value.len() > 1000 {
            1.5 // Longer half-life for complex content
        } else if memory.value.len() > 100 {
            1.2
        } else {
            1.0
        };

        base_half_life * importance_factor * frequency_factor * content_factor
    }

    /// Calculate weights for different decay models based on memory characteristics
    fn calculate_decay_model_weights(&self, memory: &MemoryEntry) -> (f64, f64, f64) {
        let access_count = memory.access_count() as f64;
        let importance = memory.metadata.importance;

        if access_count > 20.0 && importance > 0.8 {
            // High importance, high access: favor power law (gradual decay)
            (0.2, 0.6, 0.2)
        } else if access_count > 10.0 {
            // Medium access: balanced approach
            (0.4, 0.4, 0.2)
        } else if importance > 0.7 {
            // High importance, low access: favor exponential with log boost
            (0.5, 0.2, 0.3)
        } else {
            // Default: favor exponential decay
            (0.6, 0.3, 0.1)
        }
    }

    /// Calculate centrality score with sophisticated network analysis
    async fn calculate_centrality_score(&mut self, memory: &MemoryEntry) -> Result<f64> {
        let in_degree = self.estimate_in_degree(&memory.key);
        let out_degree = self.estimate_out_degree(&memory.key);
        let relationship_strength = self.estimate_relationship_strength(&memory.key);

        // 1. Degree Centrality (normalized)
        let total_degree = (in_degree + out_degree) as f64;
        let max_possible_degree = 80.0; // Realistic maximum for memory networks
        let degree_centrality = (total_degree / max_possible_degree).min(1.0);

        // 2. Weighted Centrality (relationship strength)
        let max_strength = 10.0;
        let weighted_centrality = (relationship_strength / max_strength).min(1.0);

        // 3. Betweenness Centrality (approximation based on content analysis)
        let betweenness_centrality = self.calculate_betweenness_approximation(memory, in_degree, out_degree);

        // 4. Eigenvector Centrality (approximation based on importance propagation)
        let eigenvector_centrality = self.calculate_eigenvector_approximation(memory, relationship_strength);

        // 5. PageRank-style centrality (considering access patterns)
        let pagerank_centrality = self.calculate_pagerank_approximation(memory, in_degree, out_degree);

        // 6. Closeness Centrality (based on semantic similarity potential)
        let closeness_centrality = self.calculate_closeness_approximation(memory);

        // Adaptive weight combination based on memory characteristics
        let weights = self.calculate_centrality_weights(memory);

        let centrality_score = (
            degree_centrality * weights.0 +
            weighted_centrality * weights.1 +
            betweenness_centrality * weights.2 +
            eigenvector_centrality * weights.3 +
            pagerank_centrality * weights.4 +
            closeness_centrality * weights.5
        ).min(1.0);

        // Cache comprehensive relationship metrics
        self.relationship_cache.insert(memory.key.clone(), RelationshipMetrics {
            in_degree,
            out_degree,
            relationship_strength,
            betweenness: betweenness_centrality,
        });

        Ok(centrality_score)
    }

    /// Calculate betweenness centrality approximation
    fn calculate_betweenness_approximation(&self, memory: &MemoryEntry, in_degree: usize, out_degree: usize) -> f64 {
        // Betweenness is high when a node connects many other nodes
        let connectivity_potential = (in_degree * out_degree) as f64;
        let max_connectivity = 50.0 * 30.0; // Max in_degree * max out_degree

        let base_betweenness = (connectivity_potential / max_connectivity).min(1.0);

        // Boost based on content characteristics that suggest bridging concepts
        let content_boost = if memory.value.contains("and") || memory.value.contains("between") ||
                              memory.value.contains("connects") || memory.value.contains("relates") {
            0.2
        } else {
            0.0
        };

        (base_betweenness + content_boost).min(1.0)
    }

    /// Calculate eigenvector centrality approximation
    fn calculate_eigenvector_approximation(&self, memory: &MemoryEntry, relationship_strength: f64) -> f64 {
        // Eigenvector centrality considers the importance of connected nodes
        let base_score = relationship_strength / 10.0;

        // Boost based on memory importance (important memories connect to important memories)
        let importance_boost = memory.metadata.importance * 0.3;

        // Boost based on access patterns (well-connected memories are accessed more)
        let access_boost = if memory.access_count() > 15 {
            0.2
        } else if memory.access_count() > 5 {
            0.1
        } else {
            0.0
        };

        (base_score + importance_boost + access_boost).min(1.0)
    }

    /// Calculate PageRank-style centrality approximation
    fn calculate_pagerank_approximation(&self, memory: &MemoryEntry, in_degree: usize, out_degree: usize) -> f64 {
        // PageRank considers both incoming links and the quality of those links
        let damping_factor = 0.85;
        let base_rank = (1.0 - damping_factor) / 100.0; // Assuming 100 total memories

        // Incoming link contribution
        let in_contribution = damping_factor * (in_degree as f64 / 50.0); // Normalize by max in_degree

        // Access pattern contribution (like external links in web PageRank)
        let access_contribution = memory.access_count() as f64 / 100.0; // Normalize by reasonable max

        (base_rank + in_contribution + access_contribution * 0.1).min(1.0)
    }

    /// Calculate closeness centrality approximation
    fn calculate_closeness_approximation(&self, memory: &MemoryEntry) -> f64 {
        // Closeness is based on how easily this memory can reach others
        // Approximate using content characteristics that suggest broad connectivity

        let content_length = memory.value.len() as f64;
        let length_factor = if content_length > 500.0 {
            0.8 // Longer content can reference more concepts
        } else if content_length > 100.0 {
            0.6
        } else {
            0.3
        };

        // Vocabulary richness suggests ability to connect to diverse concepts
        let words: Vec<&str> = memory.value.split_whitespace().collect();
        let unique_words: std::collections::HashSet<&str> = words.iter().cloned().collect();
        let vocabulary_factor = if words.is_empty() {
            0.0
        } else {
            unique_words.len() as f64 / words.len() as f64
        };

        // Combine factors
        (length_factor * 0.6 + vocabulary_factor * 0.4).min(1.0)
    }

    /// Calculate adaptive weights for centrality measures
    fn calculate_centrality_weights(&self, memory: &MemoryEntry) -> (f64, f64, f64, f64, f64, f64) {
        let importance = memory.metadata.importance;
        let access_count = memory.access_count() as f64;

        if importance > 0.8 && access_count > 20.0 {
            // High importance, high access: favor eigenvector and PageRank
            (0.15, 0.15, 0.15, 0.25, 0.25, 0.05)
        } else if access_count > 15.0 {
            // High access: favor degree and PageRank
            (0.25, 0.20, 0.15, 0.15, 0.20, 0.05)
        } else if importance > 0.7 {
            // High importance: favor weighted and eigenvector
            (0.15, 0.25, 0.15, 0.25, 0.15, 0.05)
        } else {
            // Default: balanced approach favoring basic measures
            (0.25, 0.20, 0.20, 0.15, 0.15, 0.05)
        }
    }

    /// Calculate content uniqueness score using multiple factors
    async fn calculate_uniqueness_score(&mut self, memory: &MemoryEntry) -> Result<f64> {
        let content = &memory.value;
        
        // Content length factor
        let length = content.len();
        let length_score = if length < 50 || length > 2000 {
            0.8 // Very short or very long content is more unique
        } else {
            0.4 // Average length content is less unique
        };

        // Vocabulary richness
        let words: Vec<&str> = content.split_whitespace().collect();
        let unique_words: std::collections::HashSet<&str> = words.iter().cloned().collect();
        let vocabulary_richness = if words.is_empty() {
            0.0
        } else {
            unique_words.len() as f64 / words.len() as f64
        };

        // Information entropy calculation
        let entropy = self.calculate_content_entropy(content);

        // Semantic density (simplified - count of meaningful words)
        let semantic_density = self.calculate_semantic_density(content);

        // Combine uniqueness factors
        let uniqueness_score = (length_score * 0.2 + vocabulary_richness * 0.3 + 
                               entropy * 0.3 + semantic_density * 0.2).min(1.0);

        // Cache content analysis
        self.content_cache.insert(memory.key.clone(), ContentAnalysis {
            length,
            vocabulary_richness,
            semantic_density,
            entropy,
        });

        Ok(uniqueness_score)
    }

    /// Calculate temporal consistency score
    async fn calculate_temporal_consistency_score(&self, memory: &MemoryEntry) -> Result<f64> {
        // Analyze access pattern consistency over time
        let access_pattern = self.access_patterns.get(&memory.key);
        
        if let Some(pattern) = access_pattern {
            Ok(pattern.consistency)
        } else {
            // Default consistency for new memories
            Ok(0.5)
        }
    }

    /// Combine importance factors using weighted sum
    fn combine_importance_factors(
        &self,
        access_frequency: f64,
        recency_score: f64,
        centrality_score: f64,
        uniqueness_score: f64,
        temporal_consistency: f64,
    ) -> f64 {
        // Configurable weights for different importance factors
        let weights = [0.25, 0.20, 0.20, 0.20, 0.15]; // Sum = 1.0
        let scores = [access_frequency, recency_score, centrality_score, uniqueness_score, temporal_consistency];

        let weighted_sum: f64 = weights.iter().zip(scores.iter()).map(|(w, s)| w * s).sum();
        weighted_sum.min(1.0).max(0.0)
    }

    /// Calculate Fisher information matrix for EWC
    async fn calculate_fisher_information(&self, memory: &MemoryEntry) -> Result<Vec<f64>> {
        // Simplified Fisher information calculation
        // In a real implementation, this would compute the diagonal of the Fisher Information Matrix
        let content_length = memory.value.len() as f64;
        let access_count = memory.access_count() as f64;
        
        // Create a simplified Fisher information vector
        let fisher_info = vec![
            content_length / 1000.0, // Normalized content importance
            access_count / 100.0,    // Normalized access importance
            1.0 / (1.0 + (-memory.metadata.importance).exp()), // Sigmoid of metadata importance
        ];

        Ok(fisher_info)
    }

    /// Update global statistics for normalization
    async fn update_global_statistics(&mut self, memories: &[MemoryEntry]) -> Result<()> {
        if memories.is_empty() {
            return Ok(());
        }

        // Calculate access frequency statistics
        let frequencies: Vec<f64> = memories.iter()
            .map(|m| {
                let days = (Utc::now() - m.created_at()).num_days().max(1) as f64;
                m.access_count() as f64 / days
            })
            .collect();

        self.global_stats.mean_access_frequency = frequencies.iter().sum::<f64>() / frequencies.len() as f64;
        
        let variance = frequencies.iter()
            .map(|f| (f - self.global_stats.mean_access_frequency).powi(2))
            .sum::<f64>() / frequencies.len() as f64;
        self.global_stats.std_access_frequency = variance.sqrt();

        Ok(())
    }

    // Enhanced helper methods for sophisticated importance calculation

    /// Calculate in-degree based on content analysis and semantic relationships
    fn estimate_in_degree(&self, memory_key: &str) -> usize {
        // Analyze content to estimate how many other memories might reference this one
        if let Some(content_analysis) = self.content_cache.get(memory_key) {
            // Higher semantic density and entropy suggest more referenceable content
            let reference_potential = content_analysis.semantic_density * content_analysis.entropy;

            // Convert to estimated in-degree (0-50 range)
            let base_degree = (reference_potential * 50.0) as usize;

            // Add content length factor
            let length_factor = if content_analysis.length > 500 {
                5 // Longer content tends to be referenced more
            } else if content_analysis.length > 100 {
                2
            } else {
                0
            };

            (base_degree + length_factor).min(50)
        } else {
            // Fallback based on key characteristics
            let key_complexity = memory_key.len() + memory_key.chars().filter(|c| c.is_alphanumeric()).count();
            (key_complexity / 3).min(20)
        }
    }

    /// Calculate out-degree based on content analysis and reference patterns
    fn estimate_out_degree(&self, memory_key: &str) -> usize {
        if let Some(content_analysis) = self.content_cache.get(memory_key) {
            // Higher vocabulary richness suggests more references to other concepts
            let reference_density = content_analysis.vocabulary_richness;

            // Convert to estimated out-degree (0-30 range)
            let base_degree = (reference_density * 30.0) as usize;

            // Add entropy factor (more diverse content references more)
            let entropy_factor = (content_analysis.entropy * 10.0) as usize;

            (base_degree + entropy_factor).min(30)
        } else {
            // Fallback based on key characteristics
            let key_diversity = memory_key.chars().collect::<std::collections::HashSet<_>>().len();
            (key_diversity / 2).min(15)
        }
    }

    /// Calculate relationship strength based on content similarity and access patterns
    fn estimate_relationship_strength(&self, memory_key: &str) -> f64 {
        if let Some(content_analysis) = self.content_cache.get(memory_key) {
            // Combine multiple factors for relationship strength
            let content_factor = content_analysis.semantic_density * 0.4;
            let complexity_factor = content_analysis.entropy * 0.3;
            let richness_factor = content_analysis.vocabulary_richness * 0.3;

            let strength = content_factor + complexity_factor + richness_factor;

            // Scale to 0-10 range
            (strength * 10.0).min(10.0)
        } else {
            // Fallback calculation
            let key_strength = memory_key.len() as f64 / 20.0; // Normalize by typical key length
            key_strength.min(5.0)
        }
    }

    /// Calculate access consistency using sophisticated temporal analysis
    fn calculate_access_consistency(&self, memory: &MemoryEntry) -> f64 {
        let access_count = memory.access_count() as f64;
        let days_since_creation = (Utc::now() - memory.created_at()).num_days().max(1) as f64;

        if access_count < 2.0 {
            return 0.1; // Very low consistency for rarely accessed memories
        }

        // Calculate expected access frequency
        let expected_frequency = access_count / days_since_creation;

        // Calculate consistency based on access distribution
        let hours_since_last_access = (Utc::now() - memory.last_accessed()).num_hours() as f64;
        let expected_hours_between_access = if expected_frequency > 0.0 {
            24.0 / expected_frequency
        } else {
            24.0 * 7.0 // Default to weekly
        };

        // Consistency score based on how close actual access pattern is to expected
        let consistency_ratio = if expected_hours_between_access > 0.0 {
            1.0 - (hours_since_last_access - expected_hours_between_access).abs() / expected_hours_between_access
        } else {
            0.5
        };

        // Apply sigmoid to bound between 0 and 1
        let consistency = 1.0 / (1.0 + (-consistency_ratio * 2.0).exp());

        // Boost consistency for frequently accessed memories
        let frequency_boost = if access_count > 10.0 {
            0.2
        } else if access_count > 5.0 {
            0.1
        } else {
            0.0
        };

        (consistency + frequency_boost).min(1.0)
    }

    fn calculate_content_entropy(&self, content: &str) -> f64 {
        let mut char_counts = HashMap::new();
        for ch in content.chars() {
            *char_counts.entry(ch).or_insert(0) += 1;
        }

        let total_chars = content.len() as f64;
        if total_chars == 0.0 {
            return 0.0;
        }

        let entropy = char_counts.values()
            .map(|&count| {
                let p = count as f64 / total_chars;
                if p > 0.0 { -p * p.log2() } else { 0.0 }
            })
            .sum::<f64>();

        entropy / 8.0 // Normalize assuming max entropy ~8 bits
    }

    fn calculate_semantic_density(&self, content: &str) -> f64 {
        let words: Vec<&str> = content.split_whitespace().collect();
        if words.is_empty() {
            return 0.0;
        }

        // Count meaningful words (simplified - words longer than 3 characters)
        let meaningful_words = words.iter().filter(|w| w.len() > 3).count();
        meaningful_words as f64 / words.len() as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::types::MemoryType;

    #[tokio::test]
    async fn test_importance_scorer_creation() {
        let config = ConsolidationConfig::default();
        let scorer = ImportanceScorer::new(&config);
        assert!(scorer.is_ok());
    }

    #[tokio::test]
    async fn test_single_importance_calculation() {
        let config = ConsolidationConfig::default();
        let mut scorer = ImportanceScorer::new(&config).unwrap();

        let memory = MemoryEntry::new(
            "test_key".to_string(),
            "This is a test memory with some meaningful content for analysis".to_string(),
            MemoryType::LongTerm
        );

        let importance = scorer.calculate_single_importance(&memory).await.unwrap();
        
        assert!(importance.importance_score >= 0.0);
        assert!(importance.importance_score <= 1.0);
        assert_eq!(importance.memory_key, "test_key");
    }

    #[tokio::test]
    async fn test_batch_importance_calculation() {
        let config = ConsolidationConfig::default();
        let mut scorer = ImportanceScorer::new(&config).unwrap();

        let memories = vec![
            MemoryEntry::new("key1".to_string(), "Important content".to_string(), MemoryType::LongTerm),
            MemoryEntry::new("key2".to_string(), "Less important".to_string(), MemoryType::ShortTerm),
        ];

        let importance_scores = scorer.calculate_batch_importance(&memories).await.unwrap();

        assert_eq!(importance_scores.len(), 2);
        for score in importance_scores {
            assert!(score.importance_score >= 0.0);
            assert!(score.importance_score <= 1.0);
        }
    }

    #[tokio::test]
    async fn test_storage_persistence() {
        let config = ConsolidationConfig::default();
        let storage = Arc::new(MemoryKeyValueStorage::new());
        let mut scorer = ImportanceScorer::with_storage(&config, storage.clone()).unwrap();

        // Create test memories and calculate importance
        let memories = vec![
            MemoryEntry::new("persist_key1".to_string(), "Persistent content 1".to_string(), MemoryType::LongTerm),
            MemoryEntry::new("persist_key2".to_string(), "Persistent content 2".to_string(), MemoryType::ShortTerm),
        ];

        let importance_scores = scorer.calculate_batch_importance(&memories).await.unwrap();
        assert_eq!(importance_scores.len(), 2);

        // Create a new scorer and load from storage
        let mut new_scorer = ImportanceScorer::with_storage(&config, storage).unwrap();
        new_scorer.load_from_storage().await.unwrap();

        // Verify data was persisted
        assert!(!new_scorer.access_patterns.is_empty());
        assert!(new_scorer.access_patterns.contains_key("persist_key1"));
        assert!(new_scorer.access_patterns.contains_key("persist_key2"));
    }

    #[tokio::test]
    async fn test_configurable_weights() {
        let config = ConsolidationConfig::default();
        let scorer = ImportanceScorer::new(&config).unwrap();

        // Test weight combination
        let combined_score = scorer.combine_importance_factors(0.8, 0.6, 0.4, 0.2, 0.1);
        assert!(combined_score >= 0.0);
        assert!(combined_score <= 1.0);

        // Should be weighted average: 0.8*0.25 + 0.6*0.20 + 0.4*0.20 + 0.2*0.20 + 0.1*0.15 = 0.455
        assert!((combined_score - 0.455).abs() < 0.001);
    }

    #[tokio::test]
    async fn test_entropy_calculation() {
        let config = ConsolidationConfig::default();
        let scorer = ImportanceScorer::new(&config).unwrap();

        // Test entropy calculation
        let entropy1 = scorer.calculate_content_entropy("aaaa"); // Low entropy
        let entropy2 = scorer.calculate_content_entropy("abcd"); // Higher entropy

        assert!(entropy2 > entropy1);
        assert!(entropy1 >= 0.0);
        assert!(entropy2 <= 1.0);
    }

    #[tokio::test]
    async fn test_semantic_density() {
        let config = ConsolidationConfig::default();
        let scorer = ImportanceScorer::new(&config).unwrap();

        // Test semantic density calculation
        let density1 = scorer.calculate_semantic_density("the a an"); // Low density (short words)
        let density2 = scorer.calculate_semantic_density("sophisticated algorithms implementation"); // High density

        assert!(density2 > density1);
        assert!(density1 >= 0.0);
        assert!(density2 <= 1.0);
    }

    #[tokio::test]
    async fn test_enhanced_temporal_decay() {
        let config = ConsolidationConfig::default();
        let scorer = ImportanceScorer::new(&config).unwrap();

        // Create memories with different characteristics
        let mut recent_memory = MemoryEntry::new(
            "recent".to_string(),
            "Recent important content".to_string(),
            MemoryType::LongTerm
        );
        recent_memory.metadata.importance = 0.9;

        let mut old_memory = MemoryEntry::new(
            "old".to_string(),
            "Old content".to_string(),
            MemoryType::LongTerm
        );
        old_memory.metadata.importance = 0.3;

        let recent_score = scorer.calculate_recency_score(&recent_memory).await.unwrap();
        let old_score = scorer.calculate_recency_score(&old_memory).await.unwrap();

        // Recent memory should have higher recency score
        assert!(recent_score >= old_score);
        assert!(recent_score >= 0.0 && recent_score <= 1.0);
        assert!(old_score >= 0.0 && old_score <= 1.0);
    }

    #[tokio::test]
    async fn test_adaptive_half_life_calculation() {
        let config = ConsolidationConfig::default();
        let scorer = ImportanceScorer::new(&config).unwrap();

        let mut important_memory = MemoryEntry::new(
            "important".to_string(),
            "Very important content with lots of detail and complexity".to_string(),
            MemoryType::LongTerm
        );
        important_memory.metadata.importance = 0.9;

        let mut simple_memory = MemoryEntry::new(
            "simple".to_string(),
            "Simple".to_string(),
            MemoryType::ShortTerm
        );
        simple_memory.metadata.importance = 0.1;

        let important_half_life = scorer.calculate_adaptive_half_life(&important_memory);
        let simple_half_life = scorer.calculate_adaptive_half_life(&simple_memory);

        // Important memory should have longer half-life
        assert!(important_half_life > simple_half_life);
        assert!(important_half_life >= 24.0); // At least base half-life
    }

    #[tokio::test]
    async fn test_sophisticated_centrality_calculation() {
        let config = ConsolidationConfig::default();
        let mut scorer = ImportanceScorer::new(&config).unwrap();

        let mut hub_memory = MemoryEntry::new(
            "hub_memory".to_string(),
            "This memory connects and relates many different concepts and ideas between various domains".to_string(),
            MemoryType::LongTerm
        );
        hub_memory.metadata.importance = 0.8;

        let mut isolated_memory = MemoryEntry::new(
            "isolated".to_string(),
            "Isolated concept".to_string(),
            MemoryType::ShortTerm
        );
        isolated_memory.metadata.importance = 0.2;

        let hub_centrality = scorer.calculate_centrality_score(&hub_memory).await.unwrap();
        let isolated_centrality = scorer.calculate_centrality_score(&isolated_memory).await.unwrap();

        // Hub memory should have higher centrality
        assert!(hub_centrality > isolated_centrality);
        assert!(hub_centrality >= 0.0 && hub_centrality <= 1.0);
        assert!(isolated_centrality >= 0.0 && isolated_centrality <= 1.0);
    }

    #[tokio::test]
    async fn test_betweenness_centrality_approximation() {
        let config = ConsolidationConfig::default();
        let scorer = ImportanceScorer::new(&config).unwrap();

        let bridge_memory = MemoryEntry::new(
            "bridge".to_string(),
            "This concept connects different domains and bridges various ideas".to_string(),
            MemoryType::LongTerm
        );

        let regular_memory = MemoryEntry::new(
            "regular".to_string(),
            "Regular content without bridging concepts".to_string(),
            MemoryType::LongTerm
        );

        let bridge_betweenness = scorer.calculate_betweenness_approximation(&bridge_memory, 10, 8);
        let regular_betweenness = scorer.calculate_betweenness_approximation(&regular_memory, 5, 3);

        // Bridge memory should have higher betweenness
        assert!(bridge_betweenness > regular_betweenness);
        assert!(bridge_betweenness >= 0.0 && bridge_betweenness <= 1.0);
    }

    #[tokio::test]
    async fn test_enhanced_access_consistency() {
        let config = ConsolidationConfig::default();
        let scorer = ImportanceScorer::new(&config).unwrap();

        let mut frequent_memory = MemoryEntry::new(
            "frequent".to_string(),
            "Frequently accessed content".to_string(),
            MemoryType::LongTerm
        );
        // Simulate frequent access
        for _ in 0..15 {
            frequent_memory.mark_accessed();
        }

        let mut rare_memory = MemoryEntry::new(
            "rare".to_string(),
            "Rarely accessed content".to_string(),
            MemoryType::LongTerm
        );
        rare_memory.mark_accessed(); // Only accessed once

        let frequent_consistency = scorer.calculate_access_consistency(&frequent_memory);
        let rare_consistency = scorer.calculate_access_consistency(&rare_memory);

        // Frequently accessed memory should have higher consistency
        assert!(frequent_consistency > rare_consistency);
        assert!(frequent_consistency >= 0.0 && frequent_consistency <= 1.0);
        assert!(rare_consistency >= 0.0 && rare_consistency <= 1.0);
    }

    #[tokio::test]
    async fn test_content_based_relationship_estimation() {
        let config = ConsolidationConfig::default();
        let mut scorer = ImportanceScorer::new(&config).unwrap();

        // Create memory with rich content for analysis
        let rich_memory = MemoryEntry::new(
            "rich_content".to_string(),
            "This is a sophisticated piece of content with diverse vocabulary, complex concepts, and high semantic density that should generate strong relationship estimates".to_string(),
            MemoryType::LongTerm
        );

        // Calculate content analysis first
        let _ = scorer.calculate_uniqueness_score(&rich_memory).await.unwrap();

        let in_degree = scorer.estimate_in_degree("rich_content");
        let out_degree = scorer.estimate_out_degree("rich_content");
        let strength = scorer.estimate_relationship_strength("rich_content");

        // Rich content should generate reasonable relationship estimates
        assert!(in_degree > 0);
        assert!(out_degree > 0);
        assert!(strength > 0.0);
        assert!(strength <= 10.0);
    }

    #[tokio::test]
    async fn test_comprehensive_importance_factors() {
        let config = ConsolidationConfig::default();
        let mut scorer = ImportanceScorer::new(&config).unwrap();

        let comprehensive_memory = MemoryEntry::new(
            "comprehensive".to_string(),
            "This is a comprehensive memory entry with substantial content that demonstrates sophisticated algorithms, complex relationships, and high-value information density".to_string(),
            MemoryType::LongTerm
        );

        let importance = scorer.calculate_single_importance(&comprehensive_memory).await.unwrap();

        // Verify all importance components are calculated
        assert!(importance.importance_score >= 0.0 && importance.importance_score <= 1.0);
        assert!(importance.access_frequency >= 0.0 && importance.access_frequency <= 1.0);
        assert!(importance.recency_score >= 0.0 && importance.recency_score <= 1.0);
        assert!(importance.centrality_score >= 0.0 && importance.centrality_score <= 1.0);
        assert!(importance.uniqueness_score >= 0.0 && importance.uniqueness_score <= 1.0);
        assert!(importance.temporal_consistency >= 0.0 && importance.temporal_consistency <= 1.0);
        assert_eq!(importance.memory_key, "comprehensive");
    }
}
