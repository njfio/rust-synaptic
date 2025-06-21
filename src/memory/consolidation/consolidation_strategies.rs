//! Memory Consolidation Strategies
//! 
//! Implements various consolidation strategies including gradual forgetting,
//! knowledge distillation, and hierarchical compression.

use crate::error::Result;
use crate::memory::types::MemoryEntry;
use super::{ConsolidationConfig, MemoryImportance, ConsolidationStrategy};
use chrono::Utc;
use std::collections::HashMap;

/// Gradual forgetting parameters
#[derive(Debug, Clone)]
pub struct GradualForgettingParams {
    /// Base forgetting rate (0.0 to 1.0)
    pub base_rate: f64,
    /// Importance threshold for protection
    pub protection_threshold: f64,
    /// Temporal decay factor
    pub temporal_decay: f64,
    /// Access frequency influence
    pub frequency_influence: f64,
}

/// Knowledge distillation configuration
#[derive(Debug, Clone)]
pub struct DistillationConfig {
    /// Compression ratio target (0.0 to 1.0)
    pub compression_ratio: f64,
    /// Quality preservation threshold
    pub quality_threshold: f64,
    /// Semantic similarity threshold
    pub similarity_threshold: f64,
    /// Maximum distillation iterations
    pub max_iterations: usize,
}

/// Hierarchical compression settings
#[derive(Debug, Clone)]
pub struct HierarchicalConfig {
    /// Number of hierarchy levels
    pub hierarchy_levels: usize,
    /// Compression factor per level
    pub compression_factor: f64,
    /// Minimum cluster size
    pub min_cluster_size: usize,
    /// Maximum cluster size
    pub max_cluster_size: usize,
}

/// Consolidation policy for different memory types
#[derive(Debug, Clone)]
pub struct ConsolidationPolicy {
    /// Strategy to use
    pub strategy: ConsolidationStrategy,
    /// Minimum importance for consolidation
    pub min_importance: f64,
    /// Maximum age before forced consolidation (hours)
    pub max_age_hours: u64,
    /// Consolidation frequency (hours)
    pub frequency_hours: u64,
}

/// Consolidation result for a specific strategy
#[derive(Debug, Clone)]
pub struct StrategyResult {
    /// Strategy used
    pub strategy: ConsolidationStrategy,
    /// Success rate (0.0 to 1.0)
    pub success_rate: f64,
    /// Compression achieved (0.0 to 1.0)
    pub compression_ratio: f64,
    /// Quality preservation score (0.0 to 1.0)
    pub quality_score: f64,
    /// Processing time in milliseconds
    pub processing_time_ms: u64,
}

/// Main consolidation strategies implementation
#[derive(Debug)]
pub struct ConsolidationStrategies {
    /// Configuration
    config: ConsolidationConfig,
    /// Gradual forgetting parameters
    forgetting_params: GradualForgettingParams,
    /// Knowledge distillation configuration
    distillation_config: DistillationConfig,
    /// Hierarchical compression configuration
    hierarchical_config: HierarchicalConfig,
    /// Consolidation policies by memory type
    #[allow(dead_code)]
    policies: HashMap<String, ConsolidationPolicy>,
    /// Strategy performance history
    performance_history: HashMap<ConsolidationStrategy, Vec<StrategyResult>>,
}

impl ConsolidationStrategies {
    /// Create new consolidation strategies manager
    pub fn new(config: &ConsolidationConfig) -> Result<Self> {
        let forgetting_params = GradualForgettingParams {
            base_rate: config.forgetting_rate,
            protection_threshold: config.importance_threshold,
            temporal_decay: 0.1,
            frequency_influence: 0.3,
        };

        let distillation_config = DistillationConfig {
            compression_ratio: 0.7,
            quality_threshold: 0.8,
            similarity_threshold: 0.85,
            max_iterations: 10,
        };

        let hierarchical_config = HierarchicalConfig {
            hierarchy_levels: 3,
            compression_factor: 0.6,
            min_cluster_size: 5,
            max_cluster_size: 50,
        };

        let mut policies = HashMap::new();
        
        // Default policy for long-term memories
        policies.insert("long_term".to_string(), ConsolidationPolicy {
            strategy: ConsolidationStrategy::Hybrid(vec![
                ConsolidationStrategy::ElasticWeightConsolidation,
                ConsolidationStrategy::SelectiveReplay,
            ]),
            min_importance: 0.5,
            max_age_hours: 168, // 1 week
            frequency_hours: 24,
        });

        // Default policy for short-term memories
        policies.insert("short_term".to_string(), ConsolidationPolicy {
            strategy: ConsolidationStrategy::GradualForgetting,
            min_importance: 0.3,
            max_age_hours: 24, // 1 day
            frequency_hours: 6,
        });

        Ok(Self {
            config: config.clone(),
            forgetting_params,
            distillation_config,
            hierarchical_config,
            policies,
            performance_history: HashMap::new(),
        })
    }

    /// Apply consolidation strategy to a set of memories
    pub async fn apply_strategy(
        &mut self,
        strategy: &ConsolidationStrategy,
        memories: &[MemoryEntry],
        importance_scores: &[MemoryImportance],
    ) -> Result<StrategyResult> {
        let start_time = std::time::Instant::now();

        let result = match strategy {
            ConsolidationStrategy::GradualForgetting => {
                self.apply_gradual_forgetting(memories, importance_scores).await?
            },
            ConsolidationStrategy::SelectiveReplay => {
                self.apply_selective_replay(memories, importance_scores).await?
            },
            ConsolidationStrategy::ElasticWeightConsolidation => {
                self.apply_ewc(memories, importance_scores).await?
            },
            ConsolidationStrategy::SynapticIntelligence => {
                self.apply_synaptic_intelligence(memories, importance_scores).await?
            },
            ConsolidationStrategy::KnowledgeDistillation => {
                self.apply_knowledge_distillation(memories, importance_scores).await?
            },
            ConsolidationStrategy::HierarchicalCompression => {
                self.apply_hierarchical_compression(memories, importance_scores).await?
            },
            ConsolidationStrategy::ImportanceWeighted => {
                self.apply_importance_weighted(memories, importance_scores).await?
            },
            ConsolidationStrategy::Hybrid(strategies) => {
                self.apply_hybrid_strategy(strategies, memories, importance_scores).await?
            },
        };

        let processing_time = start_time.elapsed().as_millis() as u64;
        let mut final_result = result;
        final_result.processing_time_ms = processing_time;

        // Record performance for future optimization
        self.record_strategy_performance(strategy.clone(), &final_result).await?;

        Ok(final_result)
    }

    /// Apply gradual forgetting strategy
    async fn apply_gradual_forgetting(
        &self,
        memories: &[MemoryEntry],
        importance_scores: &[MemoryImportance],
    ) -> Result<StrategyResult> {
        let mut forgotten_count = 0;
        let mut total_quality = 0.0;

        for (memory, importance) in memories.iter().zip(importance_scores.iter()) {
            let forgetting_probability = self.calculate_forgetting_probability(memory, importance).await?;

            use rand::Rng;
            if forgetting_probability > rand::thread_rng().gen::<f64>() {
                forgotten_count += 1;
            } else {
                // Memory preserved, calculate quality retention
                let quality = self.calculate_quality_retention(memory, importance).await?;
                total_quality += quality;
            }
        }

        let preserved_count = memories.len() - forgotten_count;
        let success_rate = if memories.len() > 0 {
            preserved_count as f64 / memories.len() as f64
        } else {
            1.0
        };

        let compression_ratio = if memories.len() > 0 {
            forgotten_count as f64 / memories.len() as f64
        } else {
            0.0
        };

        let quality_score = if preserved_count > 0 {
            total_quality / preserved_count as f64
        } else {
            1.0
        };

        Ok(StrategyResult {
            strategy: ConsolidationStrategy::GradualForgetting,
            success_rate,
            compression_ratio,
            quality_score,
            processing_time_ms: 0, // Will be set by caller
        })
    }

    /// Apply selective replay strategy
    async fn apply_selective_replay(
        &self,
        memories: &[MemoryEntry],
        importance_scores: &[MemoryImportance],
    ) -> Result<StrategyResult> {
        let mut replayed_count = 0;
        let mut total_quality = 0.0;

        // Select memories for replay based on importance and temporal factors
        for (memory, importance) in memories.iter().zip(importance_scores.iter()) {
            let replay_probability = self.calculate_replay_probability(memory, importance).await?;
            
            if replay_probability > 0.5 {
                replayed_count += 1;
                let quality = self.simulate_replay_quality(memory, importance).await?;
                total_quality += quality;
            }
        }

        let success_rate = if memories.len() > 0 {
            replayed_count as f64 / memories.len() as f64
        } else {
            1.0
        };

        let quality_score = if replayed_count > 0 {
            total_quality / replayed_count as f64
        } else {
            1.0
        };

        Ok(StrategyResult {
            strategy: ConsolidationStrategy::SelectiveReplay,
            success_rate,
            compression_ratio: 0.0, // Replay doesn't compress
            quality_score,
            processing_time_ms: 0,
        })
    }

    /// Apply Elastic Weight Consolidation strategy
    async fn apply_ewc(
        &self,
        memories: &[MemoryEntry],
        importance_scores: &[MemoryImportance],
    ) -> Result<StrategyResult> {
        let mut consolidated_count = 0;
        let mut total_quality = 0.0;

        for (memory, importance) in memories.iter().zip(importance_scores.iter()) {
            // EWC protects important weights/memories
            if importance.importance_score >= self.config.importance_threshold {
                consolidated_count += 1;
                let quality = self.calculate_ewc_quality(memory, importance).await?;
                total_quality += quality;
            }
        }

        let success_rate = if memories.len() > 0 {
            consolidated_count as f64 / memories.len() as f64
        } else {
            1.0
        };

        let quality_score = if consolidated_count > 0 {
            total_quality / consolidated_count as f64
        } else {
            1.0
        };

        Ok(StrategyResult {
            strategy: ConsolidationStrategy::ElasticWeightConsolidation,
            success_rate,
            compression_ratio: 0.0, // EWC preserves rather than compresses
            quality_score,
            processing_time_ms: 0,
        })
    }

    /// Apply synaptic intelligence strategy
    async fn apply_synaptic_intelligence(
        &self,
        memories: &[MemoryEntry],
        importance_scores: &[MemoryImportance],
    ) -> Result<StrategyResult> {
        // Synaptic Intelligence continuously estimates parameter importance
        let mut protected_count = 0;
        let mut total_quality = 0.0;

        for (memory, importance) in memories.iter().zip(importance_scores.iter()) {
            let synaptic_importance = self.calculate_synaptic_importance(memory, importance).await?;
            
            if synaptic_importance > 0.6 {
                protected_count += 1;
                total_quality += synaptic_importance;
            }
        }

        let success_rate = if memories.len() > 0 {
            protected_count as f64 / memories.len() as f64
        } else {
            1.0
        };

        let quality_score = if protected_count > 0 {
            total_quality / protected_count as f64
        } else {
            1.0
        };

        Ok(StrategyResult {
            strategy: ConsolidationStrategy::SynapticIntelligence,
            success_rate,
            compression_ratio: 0.0,
            quality_score,
            processing_time_ms: 0,
        })
    }

    /// Apply knowledge distillation strategy
    async fn apply_knowledge_distillation(
        &self,
        memories: &[MemoryEntry],
        importance_scores: &[MemoryImportance],
    ) -> Result<StrategyResult> {
        let mut distilled_count = 0;
        let mut total_compression = 0.0;
        let mut total_quality = 0.0;

        // Group memories by similarity for distillation
        let memory_groups = self.group_memories_by_similarity(memories, importance_scores).await?;
        let total_groups = memory_groups.len();

        for group in memory_groups {
            if group.len() >= 2 {
                // Apply knowledge distillation to the group
                let distillation_result = self.distill_memory_group(&group).await?;

                if distillation_result.quality >= self.distillation_config.quality_threshold {
                    distilled_count += group.len();
                    total_compression += distillation_result.compression_ratio;
                    total_quality += distillation_result.quality;
                }
            }
        }

        let success_rate = if memories.len() > 0 {
            distilled_count as f64 / memories.len() as f64
        } else {
            1.0
        };

        let avg_compression = if total_groups > 0 {
            total_compression / total_groups as f64
        } else {
            0.0
        };

        let avg_quality = if total_groups > 0 {
            total_quality / total_groups as f64
        } else {
            1.0
        };

        Ok(StrategyResult {
            strategy: ConsolidationStrategy::KnowledgeDistillation,
            success_rate,
            compression_ratio: avg_compression,
            quality_score: avg_quality,
            processing_time_ms: 0,
        })
    }

    /// Apply hierarchical compression strategy
    async fn apply_hierarchical_compression(
        &self,
        memories: &[MemoryEntry],
        importance_scores: &[MemoryImportance],
    ) -> Result<StrategyResult> {
        let mut compressed_count = 0;
        let mut total_compression = 0.0;
        let mut total_quality = 0.0;

        // Create hierarchical clusters
        let hierarchy = self.create_memory_hierarchy(memories, importance_scores).await?;

        for (level, level_clusters) in hierarchy.iter().enumerate() {
            for cluster in level_clusters {
                if cluster.len() >= self.hierarchical_config.min_cluster_size {
                    let compression_result = self.compress_cluster(cluster, level).await?;

                    compressed_count += cluster.len();
                    total_compression += compression_result.compression_ratio;
                    total_quality += compression_result.quality;
                }
            }
        }

        let success_rate = if memories.len() > 0 {
            compressed_count as f64 / memories.len() as f64
        } else {
            1.0
        };

        let avg_compression = if compressed_count > 0 {
            total_compression / (compressed_count as f64 / self.hierarchical_config.min_cluster_size as f64)
        } else {
            0.0
        };

        let avg_quality = if compressed_count > 0 {
            total_quality / (compressed_count as f64 / self.hierarchical_config.min_cluster_size as f64)
        } else {
            1.0
        };

        Ok(StrategyResult {
            strategy: ConsolidationStrategy::HierarchicalCompression,
            success_rate,
            compression_ratio: avg_compression,
            quality_score: avg_quality,
            processing_time_ms: 0,
        })
    }

    /// Apply importance-weighted updates strategy
    async fn apply_importance_weighted(
        &self,
        memories: &[MemoryEntry],
        importance_scores: &[MemoryImportance],
    ) -> Result<StrategyResult> {
        let mut updated_count = 0;
        let mut total_quality = 0.0;

        for (memory, importance) in memories.iter().zip(importance_scores.iter()) {
            let update_weight = importance.importance_score;

            if update_weight > 0.3 {
                updated_count += 1;
                let quality = self.calculate_weighted_update_quality(memory, importance).await?;
                total_quality += quality;
            }
        }

        let success_rate = if memories.len() > 0 {
            updated_count as f64 / memories.len() as f64
        } else {
            1.0
        };

        let quality_score = if updated_count > 0 {
            total_quality / updated_count as f64
        } else {
            1.0
        };

        Ok(StrategyResult {
            strategy: ConsolidationStrategy::ImportanceWeighted,
            success_rate,
            compression_ratio: 0.0,
            quality_score,
            processing_time_ms: 0,
        })
    }

    /// Apply hybrid strategy combining multiple approaches
    async fn apply_hybrid_strategy(
        &mut self,
        strategies: &[ConsolidationStrategy],
        memories: &[MemoryEntry],
        importance_scores: &[MemoryImportance],
    ) -> Result<StrategyResult> {
        let mut combined_success = 0.0;
        let mut combined_compression = 0.0;
        let mut combined_quality = 0.0;

        for strategy in strategies {
            // Apply individual strategies directly to avoid recursion
            let result = match strategy {
                ConsolidationStrategy::GradualForgetting => {
                    self.apply_gradual_forgetting(memories, importance_scores).await?
                },
                ConsolidationStrategy::SelectiveReplay => {
                    self.apply_selective_replay(memories, importance_scores).await?
                },
                ConsolidationStrategy::ElasticWeightConsolidation => {
                    self.apply_ewc(memories, importance_scores).await?
                },
                ConsolidationStrategy::SynapticIntelligence => {
                    self.apply_synaptic_intelligence(memories, importance_scores).await?
                },
                ConsolidationStrategy::KnowledgeDistillation => {
                    self.apply_knowledge_distillation(memories, importance_scores).await?
                },
                ConsolidationStrategy::HierarchicalCompression => {
                    self.apply_hierarchical_compression(memories, importance_scores).await?
                },
                ConsolidationStrategy::ImportanceWeighted => {
                    self.apply_importance_weighted(memories, importance_scores).await?
                },
                ConsolidationStrategy::Hybrid(_) => {
                    // Skip nested hybrid strategies to avoid infinite recursion
                    continue;
                },
            };

            combined_success += result.success_rate;
            combined_compression += result.compression_ratio;
            combined_quality += result.quality_score;
        }

        let strategy_count = strategies.len() as f64;

        Ok(StrategyResult {
            strategy: ConsolidationStrategy::Hybrid(strategies.to_vec()),
            success_rate: (combined_success / strategy_count).min(1.0),
            compression_ratio: (combined_compression / strategy_count).min(1.0),
            quality_score: (combined_quality / strategy_count).min(1.0),
            processing_time_ms: 0,
        })
    }

    // Helper methods for strategy calculations

    async fn calculate_forgetting_probability(&self, memory: &MemoryEntry, importance: &MemoryImportance) -> Result<f64> {
        let base_rate = self.forgetting_params.base_rate;
        let importance_protection = 1.0 - importance.importance_score;
        let age_factor = self.calculate_age_factor(memory).await?;
        
        let forgetting_prob = base_rate * importance_protection * age_factor;
        Ok(forgetting_prob.min(1.0).max(0.0))
    }

    async fn calculate_replay_probability(&self, _memory: &MemoryEntry, importance: &MemoryImportance) -> Result<f64> {
        let importance_factor = importance.importance_score;
        let recency_factor = importance.recency_score;
        let centrality_factor = importance.centrality_score;
        
        let replay_prob = importance_factor * 0.5 + recency_factor * 0.3 + centrality_factor * 0.2;
        Ok(replay_prob.min(1.0).max(0.0))
    }

    async fn calculate_quality_retention(&self, _memory: &MemoryEntry, importance: &MemoryImportance) -> Result<f64> {
        // Quality retention based on importance and consolidation effectiveness
        let base_quality = 0.8;
        let importance_bonus = importance.importance_score * 0.2;
        Ok((base_quality + importance_bonus).min(1.0))
    }

    async fn simulate_replay_quality(&self, _memory: &MemoryEntry, importance: &MemoryImportance) -> Result<f64> {
        // Simulate quality improvement from replay
        let base_quality = 0.7;
        let replay_improvement = importance.importance_score * 0.3;
        Ok((base_quality + replay_improvement).min(1.0))
    }

    async fn calculate_ewc_quality(&self, _memory: &MemoryEntry, importance: &MemoryImportance) -> Result<f64> {
        // EWC quality based on Fisher information and importance
        let fisher_quality = if importance.fisher_information.is_some() { 0.9 } else { 0.7 };
        let importance_quality = importance.importance_score;
        Ok((fisher_quality * 0.6 + importance_quality * 0.4).min(1.0))
    }

    async fn calculate_synaptic_importance(&self, _memory: &MemoryEntry, importance: &MemoryImportance) -> Result<f64> {
        // Synaptic intelligence importance calculation
        let temporal_factor = importance.temporal_consistency;
        let access_factor = importance.access_frequency;
        let synaptic_importance = temporal_factor * 0.6 + access_factor * 0.4;
        Ok(synaptic_importance.min(1.0))
    }

    async fn calculate_weighted_update_quality(&self, _memory: &MemoryEntry, importance: &MemoryImportance) -> Result<f64> {
        // Quality of importance-weighted updates
        let weight_quality = importance.importance_score;
        let consistency_quality = importance.temporal_consistency;
        Ok((weight_quality * 0.7 + consistency_quality * 0.3).min(1.0))
    }

    async fn calculate_age_factor(&self, memory: &MemoryEntry) -> Result<f64> {
        let hours_since_creation = (Utc::now() - memory.created_at()).num_hours().max(0) as f64;
        let age_factor = (hours_since_creation / 168.0).min(1.0); // Normalize to week
        Ok(age_factor)
    }

    async fn record_strategy_performance(&mut self, strategy: ConsolidationStrategy, result: &StrategyResult) -> Result<()> {
        self.performance_history
            .entry(strategy)
            .or_insert_with(Vec::new)
            .push(result.clone());
        
        // Keep only recent performance data
        if let Some(history) = self.performance_history.get_mut(&result.strategy) {
            if history.len() > 100 {
                history.remove(0);
            }
        }
        
        Ok(())
    }

    /// Get strategy performance statistics
    pub fn get_strategy_performance(&self, strategy: &ConsolidationStrategy) -> Option<(f64, f64, f64)> {
        if let Some(history) = self.performance_history.get(strategy) {
            if history.is_empty() {
                return None;
            }

            let avg_success = history.iter().map(|r| r.success_rate).sum::<f64>() / history.len() as f64;
            let avg_compression = history.iter().map(|r| r.compression_ratio).sum::<f64>() / history.len() as f64;
            let avg_quality = history.iter().map(|r| r.quality_score).sum::<f64>() / history.len() as f64;

            Some((avg_success, avg_compression, avg_quality))
        } else {
            None
        }
    }

    /// Get best performing strategy for given criteria
    pub fn get_best_strategy(&self, optimize_for: &str) -> Option<ConsolidationStrategy> {
        let mut best_strategy = None;
        let mut best_score = 0.0;

        for (strategy, history) in &self.performance_history {
            if history.is_empty() {
                continue;
            }

            let score = match optimize_for {
                "success_rate" => history.iter().map(|r| r.success_rate).sum::<f64>() / history.len() as f64,
                "compression" => history.iter().map(|r| r.compression_ratio).sum::<f64>() / history.len() as f64,
                "quality" => history.iter().map(|r| r.quality_score).sum::<f64>() / history.len() as f64,
                _ => continue,
            };

            if score > best_score {
                best_score = score;
                best_strategy = Some(strategy.clone());
            }
        }

        best_strategy
    }

    /// Group memories by semantic similarity for knowledge distillation
    async fn group_memories_by_similarity(
        &self,
        memories: &[MemoryEntry],
        importance_scores: &[MemoryImportance],
    ) -> Result<Vec<Vec<(MemoryEntry, MemoryImportance)>>> {
        let mut groups = Vec::new();
        let mut used_indices = std::collections::HashSet::new();

        for (i, (memory, importance)) in memories.iter().zip(importance_scores.iter()).enumerate() {
            if used_indices.contains(&i) {
                continue;
            }

            let mut group = vec![(memory.clone(), importance.clone())];
            used_indices.insert(i);

            // Find similar memories
            for (j, (other_memory, other_importance)) in memories.iter().zip(importance_scores.iter()).enumerate() {
                if i == j || used_indices.contains(&j) {
                    continue;
                }

                let similarity = self.calculate_semantic_similarity(memory, other_memory).await?;
                if similarity >= self.distillation_config.similarity_threshold {
                    group.push((other_memory.clone(), other_importance.clone()));
                    used_indices.insert(j);
                }
            }

            groups.push(group);
        }

        Ok(groups)
    }

    /// Calculate semantic similarity between two memories
    async fn calculate_semantic_similarity(&self, memory1: &MemoryEntry, memory2: &MemoryEntry) -> Result<f64> {
        // Simplified similarity calculation based on content overlap
        let content1 = memory1.value.to_lowercase();
        let content2 = memory2.value.to_lowercase();

        let words1: std::collections::HashSet<&str> = content1.split_whitespace().collect();
        let words2: std::collections::HashSet<&str> = content2.split_whitespace().collect();

        let intersection = words1.intersection(&words2).count();
        let union = words1.union(&words2).count();

        let jaccard_similarity = if union > 0 {
            intersection as f64 / union as f64
        } else {
            0.0
        };

        Ok(jaccard_similarity)
    }

    /// Distill a group of similar memories into a compressed representation
    async fn distill_memory_group(&self, group: &[(MemoryEntry, MemoryImportance)]) -> Result<DistillationResult> {
        if group.is_empty() {
            return Ok(DistillationResult {
                compression_ratio: 0.0,
                quality: 0.0,
            });
        }

        // Calculate compression ratio based on group size
        let compression_ratio = 1.0 - (1.0 / group.len() as f64);

        // Calculate quality based on importance preservation
        let total_importance: f64 = group.iter().map(|(_, imp)| imp.importance_score).sum();
        let avg_importance = total_importance / group.len() as f64;

        // Quality is based on how well we preserve important information
        let quality = if avg_importance >= 0.7 {
            0.9 // High quality for important memories
        } else if avg_importance >= 0.5 {
            0.8 // Medium quality
        } else {
            0.6 // Lower quality for less important memories
        };

        Ok(DistillationResult {
            compression_ratio,
            quality,
        })
    }

    /// Create hierarchical memory clusters
    async fn create_memory_hierarchy(
        &self,
        memories: &[MemoryEntry],
        importance_scores: &[MemoryImportance],
    ) -> Result<Vec<Vec<Vec<(MemoryEntry, MemoryImportance)>>>> {
        let mut hierarchy = Vec::new();

        // Create initial clusters at level 0
        let mut current_level = self.create_initial_clusters(memories, importance_scores).await?;
        hierarchy.push(current_level.clone());

        // Create subsequent levels
        for level in 1..self.hierarchical_config.hierarchy_levels {
            let next_level = self.create_next_level_clusters(&current_level, level).await?;
            if next_level.is_empty() {
                break;
            }
            hierarchy.push(next_level.clone());
            current_level = next_level;
        }

        Ok(hierarchy)
    }

    /// Create initial clusters for hierarchical compression
    async fn create_initial_clusters(
        &self,
        memories: &[MemoryEntry],
        importance_scores: &[MemoryImportance],
    ) -> Result<Vec<Vec<(MemoryEntry, MemoryImportance)>>> {
        let mut clusters = Vec::new();
        let mut remaining: Vec<_> = memories.iter().zip(importance_scores.iter()).collect();

        while !remaining.is_empty() {
            let mut cluster = Vec::new();
            let seed = remaining.remove(0);
            cluster.push((seed.0.clone(), seed.1.clone()));

            // Add similar memories to cluster
            let mut i = 0;
            while i < remaining.len() && cluster.len() < self.hierarchical_config.max_cluster_size {
                let similarity = self.calculate_semantic_similarity(seed.0, remaining[i].0).await?;
                if similarity > 0.3 { // Threshold for clustering
                    let item = remaining.remove(i);
                    cluster.push((item.0.clone(), item.1.clone()));
                } else {
                    i += 1;
                }
            }

            if cluster.len() >= self.hierarchical_config.min_cluster_size {
                clusters.push(cluster);
            }
        }

        Ok(clusters)
    }

    /// Create next level clusters in hierarchy
    async fn create_next_level_clusters(
        &self,
        current_level: &[Vec<(MemoryEntry, MemoryImportance)>],
        level: usize,
    ) -> Result<Vec<Vec<(MemoryEntry, MemoryImportance)>>> {
        let mut next_level = Vec::new();
        let compression_factor = self.hierarchical_config.compression_factor.powi(level as i32);

        for cluster in current_level {
            if cluster.len() >= (self.hierarchical_config.min_cluster_size as f64 * compression_factor) as usize {
                // Compress cluster by selecting most important memories
                let mut compressed_cluster = cluster.clone();
                compressed_cluster.sort_by(|a, b| b.1.importance_score.partial_cmp(&a.1.importance_score).unwrap());

                let target_size = (cluster.len() as f64 * compression_factor) as usize;
                compressed_cluster.truncate(target_size.max(1));

                if !compressed_cluster.is_empty() {
                    next_level.push(compressed_cluster);
                }
            }
        }

        Ok(next_level)
    }

    /// Compress a cluster at a specific level
    async fn compress_cluster(
        &self,
        cluster: &[(MemoryEntry, MemoryImportance)],
        level: usize,
    ) -> Result<CompressionResult> {
        let compression_factor = self.hierarchical_config.compression_factor.powi(level as i32);
        let target_size = (cluster.len() as f64 * compression_factor) as usize;

        let compression_ratio = 1.0 - (target_size as f64 / cluster.len() as f64);

        // Quality based on importance preservation
        let total_importance: f64 = cluster.iter().map(|(_, imp)| imp.importance_score).sum();
        let avg_importance = total_importance / cluster.len() as f64;

        let quality = if compression_ratio < 0.5 {
            avg_importance * 0.9 // High quality for low compression
        } else if compression_ratio < 0.8 {
            avg_importance * 0.7 // Medium quality for medium compression
        } else {
            avg_importance * 0.5 // Lower quality for high compression
        };

        Ok(CompressionResult {
            compression_ratio,
            quality,
        })
    }
}

/// Result of knowledge distillation operation
#[derive(Debug, Clone)]
struct DistillationResult {
    compression_ratio: f64,
    quality: f64,
}

/// Result of hierarchical compression operation
#[derive(Debug, Clone)]
struct CompressionResult {
    compression_ratio: f64,
    quality: f64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::types::MemoryType;

    #[tokio::test]
    async fn test_consolidation_strategies_creation() {
        let config = ConsolidationConfig::default();
        let strategies = ConsolidationStrategies::new(&config);
        assert!(strategies.is_ok());
    }

    #[tokio::test]
    async fn test_gradual_forgetting_strategy() {
        let config = ConsolidationConfig::default();
        let mut strategies = ConsolidationStrategies::new(&config).unwrap();

        let memories = vec![
            MemoryEntry::new("key1".to_string(), "Content 1".to_string(), MemoryType::LongTerm),
        ];

        let importance_scores = vec![
            MemoryImportance {
                memory_key: "key1".to_string(),
                importance_score: 0.8,
                access_frequency: 0.7,
                recency_score: 0.6,
                centrality_score: 0.5,
                uniqueness_score: 0.4,
                temporal_consistency: 0.3,
                calculated_at: Utc::now(),
                fisher_information: None,
            },
        ];

        let result = strategies.apply_strategy(
            &ConsolidationStrategy::GradualForgetting,
            &memories,
            &importance_scores,
        ).await.unwrap();

        assert!(result.success_rate >= 0.0);
        assert!(result.success_rate <= 1.0);
    }

    #[tokio::test]
    async fn test_knowledge_distillation_strategy() {
        let config = ConsolidationConfig::default();
        let mut strategies = ConsolidationStrategies::new(&config).unwrap();

        let memories = vec![
            MemoryEntry::new("key1".to_string(), "machine learning algorithms".to_string(), MemoryType::LongTerm),
            MemoryEntry::new("key2".to_string(), "machine learning models".to_string(), MemoryType::LongTerm),
            MemoryEntry::new("key3".to_string(), "deep learning networks".to_string(), MemoryType::LongTerm),
        ];

        let importance_scores = vec![
            MemoryImportance {
                memory_key: "key1".to_string(),
                importance_score: 0.8,
                access_frequency: 0.7,
                recency_score: 0.6,
                centrality_score: 0.5,
                uniqueness_score: 0.4,
                temporal_consistency: 0.3,
                calculated_at: Utc::now(),
                fisher_information: None,
            },
            MemoryImportance {
                memory_key: "key2".to_string(),
                importance_score: 0.7,
                access_frequency: 0.6,
                recency_score: 0.5,
                centrality_score: 0.4,
                uniqueness_score: 0.3,
                temporal_consistency: 0.2,
                calculated_at: Utc::now(),
                fisher_information: None,
            },
            MemoryImportance {
                memory_key: "key3".to_string(),
                importance_score: 0.6,
                access_frequency: 0.5,
                recency_score: 0.4,
                centrality_score: 0.3,
                uniqueness_score: 0.2,
                temporal_consistency: 0.1,
                calculated_at: Utc::now(),
                fisher_information: None,
            },
        ];

        let result = strategies.apply_strategy(
            &ConsolidationStrategy::KnowledgeDistillation,
            &memories,
            &importance_scores,
        ).await.unwrap();

        assert!(result.success_rate >= 0.0);
        assert!(result.success_rate <= 1.0);
        assert!(result.compression_ratio >= 0.0);
        assert!(result.quality_score >= 0.0);
        assert!(result.quality_score <= 1.0);
    }

    #[tokio::test]
    async fn test_hierarchical_compression_strategy() {
        let config = ConsolidationConfig::default();
        let mut strategies = ConsolidationStrategies::new(&config).unwrap();

        let memories = vec![
            MemoryEntry::new("key1".to_string(), "data structure array".to_string(), MemoryType::LongTerm),
            MemoryEntry::new("key2".to_string(), "data structure list".to_string(), MemoryType::LongTerm),
            MemoryEntry::new("key3".to_string(), "data structure tree".to_string(), MemoryType::LongTerm),
            MemoryEntry::new("key4".to_string(), "algorithm sorting".to_string(), MemoryType::LongTerm),
            MemoryEntry::new("key5".to_string(), "algorithm searching".to_string(), MemoryType::LongTerm),
        ];

        let importance_scores: Vec<MemoryImportance> = (0..5).map(|i| {
            MemoryImportance {
                memory_key: format!("key{}", i + 1),
                importance_score: 0.8 - (i as f64 * 0.1),
                access_frequency: 0.7 - (i as f64 * 0.1),
                recency_score: 0.6 - (i as f64 * 0.1),
                centrality_score: 0.5 - (i as f64 * 0.1),
                uniqueness_score: 0.4 - (i as f64 * 0.1),
                temporal_consistency: 0.3 - (i as f64 * 0.1),
                calculated_at: Utc::now(),
                fisher_information: None,
            }
        }).collect();

        let result = strategies.apply_strategy(
            &ConsolidationStrategy::HierarchicalCompression,
            &memories,
            &importance_scores,
        ).await.unwrap();

        assert!(result.success_rate >= 0.0);
        assert!(result.success_rate <= 1.0);
        assert!(result.compression_ratio >= 0.0);
        assert!(result.quality_score >= 0.0);
        assert!(result.quality_score <= 1.0);
    }

    #[tokio::test]
    async fn test_semantic_similarity_calculation() {
        let config = ConsolidationConfig::default();
        let strategies = ConsolidationStrategies::new(&config).unwrap();

        let memory1 = MemoryEntry::new("key1".to_string(), "machine learning algorithms".to_string(), MemoryType::LongTerm);
        let memory2 = MemoryEntry::new("key2".to_string(), "machine learning models".to_string(), MemoryType::LongTerm);
        let memory3 = MemoryEntry::new("key3".to_string(), "database systems".to_string(), MemoryType::LongTerm);

        let similarity_high = strategies.calculate_semantic_similarity(&memory1, &memory2).await.unwrap();
        let similarity_low = strategies.calculate_semantic_similarity(&memory1, &memory3).await.unwrap();

        assert!(similarity_high > similarity_low);
        assert!(similarity_high >= 0.0);
        assert!(similarity_high <= 1.0);
        assert!(similarity_low >= 0.0);
        assert!(similarity_low <= 1.0);
    }

    #[tokio::test]
    async fn test_memory_grouping_by_similarity() {
        let config = ConsolidationConfig::default();
        let strategies = ConsolidationStrategies::new(&config).unwrap();

        let memories = vec![
            MemoryEntry::new("key1".to_string(), "machine learning algorithms".to_string(), MemoryType::LongTerm),
            MemoryEntry::new("key2".to_string(), "machine learning models".to_string(), MemoryType::LongTerm),
            MemoryEntry::new("key3".to_string(), "database systems".to_string(), MemoryType::LongTerm),
        ];

        let importance_scores: Vec<MemoryImportance> = (0..3).map(|i| {
            MemoryImportance {
                memory_key: format!("key{}", i + 1),
                importance_score: 0.8,
                access_frequency: 0.7,
                recency_score: 0.6,
                centrality_score: 0.5,
                uniqueness_score: 0.4,
                temporal_consistency: 0.3,
                calculated_at: Utc::now(),
                fisher_information: None,
            }
        }).collect();

        let groups = strategies.group_memories_by_similarity(&memories, &importance_scores).await.unwrap();

        assert!(!groups.is_empty());
        assert!(groups.len() <= memories.len());
    }

    #[tokio::test]
    async fn test_strategy_performance_tracking() {
        let config = ConsolidationConfig::default();
        let mut strategies = ConsolidationStrategies::new(&config).unwrap();

        let result = StrategyResult {
            strategy: ConsolidationStrategy::GradualForgetting,
            success_rate: 0.8,
            compression_ratio: 0.3,
            quality_score: 0.9,
            processing_time_ms: 100,
        };

        strategies.record_strategy_performance(ConsolidationStrategy::GradualForgetting, &result).await.unwrap();

        let performance = strategies.get_strategy_performance(&ConsolidationStrategy::GradualForgetting);
        assert!(performance.is_some());

        let (success, compression, quality) = performance.unwrap();
        assert_eq!(success, 0.8);
        assert_eq!(compression, 0.3);
        assert_eq!(quality, 0.9);
    }

    #[tokio::test]
    async fn test_best_strategy_selection() {
        let config = ConsolidationConfig::default();
        let mut strategies = ConsolidationStrategies::new(&config).unwrap();

        // Add performance data for different strategies
        let result1 = StrategyResult {
            strategy: ConsolidationStrategy::GradualForgetting,
            success_rate: 0.8,
            compression_ratio: 0.3,
            quality_score: 0.9,
            processing_time_ms: 100,
        };

        let result2 = StrategyResult {
            strategy: ConsolidationStrategy::SelectiveReplay,
            success_rate: 0.9,
            compression_ratio: 0.2,
            quality_score: 0.8,
            processing_time_ms: 150,
        };

        strategies.record_strategy_performance(ConsolidationStrategy::GradualForgetting, &result1).await.unwrap();
        strategies.record_strategy_performance(ConsolidationStrategy::SelectiveReplay, &result2).await.unwrap();

        let best_for_success = strategies.get_best_strategy("success_rate");
        assert_eq!(best_for_success, Some(ConsolidationStrategy::SelectiveReplay));

        let best_for_quality = strategies.get_best_strategy("quality");
        assert_eq!(best_for_quality, Some(ConsolidationStrategy::GradualForgetting));
    }

    #[tokio::test]
    async fn test_hybrid_strategy_with_new_strategies() {
        let config = ConsolidationConfig::default();
        let mut strategies = ConsolidationStrategies::new(&config).unwrap();

        let memories = vec![
            MemoryEntry::new("key1".to_string(), "Content 1".to_string(), MemoryType::LongTerm),
        ];

        let importance_scores = vec![
            MemoryImportance {
                memory_key: "key1".to_string(),
                importance_score: 0.8,
                access_frequency: 0.7,
                recency_score: 0.6,
                centrality_score: 0.5,
                uniqueness_score: 0.4,
                temporal_consistency: 0.3,
                calculated_at: Utc::now(),
                fisher_information: None,
            },
        ];

        let hybrid_strategies = vec![
            ConsolidationStrategy::KnowledgeDistillation,
            ConsolidationStrategy::HierarchicalCompression,
            ConsolidationStrategy::GradualForgetting,
        ];

        let result = strategies.apply_strategy(
            &ConsolidationStrategy::Hybrid(hybrid_strategies),
            &memories,
            &importance_scores,
        ).await.unwrap();

        assert!(result.success_rate >= 0.0);
        assert!(result.success_rate <= 1.0);
        assert!(result.quality_score >= 0.0);
        assert!(result.quality_score <= 1.0);
    }
}
