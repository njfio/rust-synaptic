//! Task Distribution Management for Meta-Learning
//! 
//! This module manages the distribution of tasks for meta-learning,
//! including task sampling, difficulty estimation, and domain adaptation.

use super::{MetaTask, TaskType};
use crate::error::Result;
use crate::memory::types::MemoryEntry;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use chrono::{DateTime, Utc};
use rand::Rng;

/// Task distribution manager
#[derive(Debug)]
pub struct TaskDistribution {
    /// Task registry
    tasks: Vec<MetaTask>,
    /// Task difficulty scores
    difficulty_scores: HashMap<String, f64>,
    /// Domain distribution
    domain_distribution: HashMap<String, f64>,
    /// Task type distribution
    type_distribution: HashMap<TaskType, f64>,
    /// Sampling weights
    sampling_weights: HashMap<String, f64>,
    /// Task creation statistics
    creation_stats: TaskCreationStats,
}

/// Task creation statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskCreationStats {
    /// Total tasks created
    pub total_created: usize,
    /// Tasks by type
    pub by_type: HashMap<TaskType, usize>,
    /// Tasks by domain
    pub by_domain: HashMap<String, usize>,
    /// Average difficulty
    pub avg_difficulty: f64,
    /// Creation rate (tasks per hour)
    pub creation_rate: f64,
    /// Last update time
    pub last_update: DateTime<Utc>,
}

/// Task sampling strategy
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SamplingStrategy {
    /// Uniform random sampling
    Uniform,
    /// Weighted sampling by difficulty
    DifficultyWeighted,
    /// Curriculum learning (easy to hard)
    Curriculum,
    /// Anti-curriculum (hard to easy)
    AntiCurriculum,
    /// Domain-balanced sampling
    DomainBalanced,
    /// Custom weighted sampling
    Custom(HashMap<String, f64>),
}

impl TaskDistribution {
    /// Create a new task distribution manager
    pub fn new() -> Self {
        Self {
            tasks: Vec::new(),
            difficulty_scores: HashMap::new(),
            domain_distribution: HashMap::new(),
            type_distribution: HashMap::new(),
            sampling_weights: HashMap::new(),
            creation_stats: TaskCreationStats::default(),
        }
    }

    /// Update the task distribution with new tasks
    pub async fn update_distribution(&mut self, new_tasks: &[MetaTask]) -> Result<()> {
        tracing::info!("Updating task distribution with {} new tasks", new_tasks.len());
        
        for task in new_tasks {
            self.add_task(task.clone()).await?;
        }
        
        self.recompute_distributions().await?;
        self.update_statistics().await?;
        
        Ok(())
    }

    /// Add a single task to the distribution
    async fn add_task(&mut self, task: MetaTask) -> Result<()> {
        // Compute difficulty score
        let difficulty = self.compute_task_difficulty(&task).await?;
        self.difficulty_scores.insert(task.id.clone(), difficulty);
        
        // Update domain distribution
        *self.domain_distribution.entry(task.domain.clone()).or_insert(0.0) += 1.0;
        
        // Update type distribution
        *self.type_distribution.entry(task.task_type.clone()).or_insert(0.0) += 1.0;
        
        // Add to task registry
        self.tasks.push(task);
        
        Ok(())
    }

    /// Compute difficulty score for a task
    async fn compute_task_difficulty(&self, task: &MetaTask) -> Result<f64> {
        let mut difficulty = 0.0;
        
        // Base difficulty from support set size (fewer examples = harder)
        let support_size_factor = 1.0 / (task.support_set.len() as f64 + 1.0);
        difficulty += support_size_factor * 0.3;
        
        // Content complexity factor
        let content_complexity = self.compute_content_complexity(&task.support_set).await?;
        difficulty += content_complexity * 0.4;
        
        // Domain novelty factor
        let domain_novelty = self.compute_domain_novelty(&task.domain).await?;
        difficulty += domain_novelty * 0.2;
        
        // Task type complexity
        let type_complexity = match task.task_type {
            TaskType::Classification => 0.2,
            TaskType::Regression => 0.4,
            TaskType::Ranking => 0.6,
            TaskType::Consolidation => 0.8,
            TaskType::PatternRecognition => 0.7,
            TaskType::Custom(_) => 0.5,
        };
        difficulty += type_complexity * 0.1;
        
        // Normalize to [0, 1]
        Ok(difficulty.min(1.0).max(0.0))
    }

    /// Compute content complexity of memory entries
    async fn compute_content_complexity(&self, memories: &[MemoryEntry]) -> Result<f64> {
        if memories.is_empty() {
            return Ok(0.0);
        }
        
        let mut total_complexity = 0.0;
        
        for memory in memories {
            let mut complexity = 0.0;
            
            // Content length complexity
            let length_factor = (memory.value.len() as f64).ln() / 10.0;
            complexity += length_factor.min(1.0) * 0.3;
            
            // Vocabulary richness
            let words: Vec<&str> = memory.value.split_whitespace().collect();
            let unique_words = words.iter().collect::<std::collections::HashSet<_>>().len();
            let vocab_richness = unique_words as f64 / words.len().max(1) as f64;
            complexity += vocab_richness * 0.3;
            
            // Character diversity
            let unique_chars = memory.value.chars().collect::<std::collections::HashSet<_>>().len();
            let char_diversity = unique_chars as f64 / memory.value.len().max(1) as f64;
            complexity += char_diversity * 0.2;
            
            // Access pattern complexity (more accesses = more complex patterns)
            let access_complexity = (memory.metadata.access_count as f64).ln() / 5.0;
            complexity += access_complexity.min(1.0) * 0.2;
            
            total_complexity += complexity;
        }
        
        Ok(total_complexity / memories.len() as f64)
    }

    /// Compute domain novelty score
    async fn compute_domain_novelty(&self, domain: &str) -> Result<f64> {
        // If we've seen this domain before, it's less novel
        let domain_count = self.domain_distribution.get(domain).unwrap_or(&0.0);
        let total_tasks = self.tasks.len() as f64;
        
        if total_tasks == 0.0 {
            return Ok(1.0); // Completely novel
        }
        
        let domain_frequency = domain_count / total_tasks;
        let novelty = 1.0 - domain_frequency;
        
        Ok(novelty.max(0.0).min(1.0))
    }

    /// Recompute all distributions
    async fn recompute_distributions(&mut self) -> Result<()> {
        let total_tasks = self.tasks.len() as f64;
        
        if total_tasks == 0.0 {
            return Ok(());
        }
        
        // Normalize domain distribution
        for count in self.domain_distribution.values_mut() {
            *count /= total_tasks;
        }
        
        // Normalize type distribution
        for count in self.type_distribution.values_mut() {
            *count /= total_tasks;
        }
        
        // Update sampling weights based on difficulty
        for task in &self.tasks {
            let difficulty = self.difficulty_scores.get(&task.id).unwrap_or(&0.5);
            self.sampling_weights.insert(task.id.clone(), *difficulty);
        }
        
        Ok(())
    }

    /// Update creation statistics
    async fn update_statistics(&mut self) -> Result<()> {
        self.creation_stats.total_created = self.tasks.len();
        
        // Count by type
        self.creation_stats.by_type.clear();
        for task in &self.tasks {
            *self.creation_stats.by_type.entry(task.task_type.clone()).or_insert(0) += 1;
        }
        
        // Count by domain
        self.creation_stats.by_domain.clear();
        for task in &self.tasks {
            *self.creation_stats.by_domain.entry(task.domain.clone()).or_insert(0) += 1;
        }
        
        // Average difficulty
        if !self.difficulty_scores.is_empty() {
            self.creation_stats.avg_difficulty = self.difficulty_scores.values().sum::<f64>() 
                / self.difficulty_scores.len() as f64;
        }
        
        // Update timestamp
        self.creation_stats.last_update = Utc::now();
        
        Ok(())
    }

    /// Sample tasks according to a strategy
    pub async fn sample_tasks(&self, count: usize, strategy: SamplingStrategy) -> Result<Vec<MetaTask>> {
        if self.tasks.is_empty() {
            return Ok(Vec::new());
        }
        
        let mut rng = rand::thread_rng();
        let mut sampled_tasks = Vec::new();
        
        match strategy {
            SamplingStrategy::Uniform => {
                for _ in 0..count {
                    let idx = rng.gen_range(0..self.tasks.len());
                    sampled_tasks.push(self.tasks[idx].clone());
                }
            },
            
            SamplingStrategy::DifficultyWeighted => {
                let weights: Vec<f64> = self.tasks.iter()
                    .map(|task| self.difficulty_scores.get(&task.id).unwrap_or(&0.5))
                    .cloned()
                    .collect();
                
                for _ in 0..count {
                    let idx = self.weighted_sample(&weights, &mut rng)?;
                    sampled_tasks.push(self.tasks[idx].clone());
                }
            },
            
            SamplingStrategy::Curriculum => {
                // Sort by difficulty (easy to hard)
                let mut indexed_tasks: Vec<(usize, f64)> = self.tasks.iter()
                    .enumerate()
                    .map(|(i, task)| (i, *self.difficulty_scores.get(&task.id).unwrap_or(&0.5)))
                    .collect();
                indexed_tasks.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
                
                for i in 0..count.min(indexed_tasks.len()) {
                    sampled_tasks.push(self.tasks[indexed_tasks[i].0].clone());
                }
            },
            
            SamplingStrategy::AntiCurriculum => {
                // Sort by difficulty (hard to easy)
                let mut indexed_tasks: Vec<(usize, f64)> = self.tasks.iter()
                    .enumerate()
                    .map(|(i, task)| (i, *self.difficulty_scores.get(&task.id).unwrap_or(&0.5)))
                    .collect();
                indexed_tasks.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
                
                for i in 0..count.min(indexed_tasks.len()) {
                    sampled_tasks.push(self.tasks[indexed_tasks[i].0].clone());
                }
            },
            
            SamplingStrategy::DomainBalanced => {
                // Sample evenly across domains
                let domains: Vec<String> = self.domain_distribution.keys().cloned().collect();
                if domains.is_empty() {
                    return Ok(Vec::new());
                }
                
                let tasks_per_domain = count / domains.len();
                let remainder = count % domains.len();
                
                for (domain_idx, domain) in domains.iter().enumerate() {
                    let domain_tasks: Vec<&MetaTask> = self.tasks.iter()
                        .filter(|task| &task.domain == domain)
                        .collect();
                    
                    if domain_tasks.is_empty() {
                        continue;
                    }
                    
                    let mut domain_count = tasks_per_domain;
                    if domain_idx < remainder {
                        domain_count += 1;
                    }
                    
                    for _ in 0..domain_count {
                        let idx = rng.gen_range(0..domain_tasks.len());
                        sampled_tasks.push(domain_tasks[idx].clone());
                    }
                }
            },
            
            SamplingStrategy::Custom(weights) => {
                let task_weights: Vec<f64> = self.tasks.iter()
                    .map(|task| weights.get(&task.id).unwrap_or(&1.0))
                    .cloned()
                    .collect();
                
                for _ in 0..count {
                    let idx = self.weighted_sample(&task_weights, &mut rng)?;
                    sampled_tasks.push(self.tasks[idx].clone());
                }
            },
        }
        
        Ok(sampled_tasks)
    }

    /// Weighted sampling helper
    fn weighted_sample(&self, weights: &[f64], rng: &mut impl Rng) -> Result<usize> {
        let total_weight: f64 = weights.iter().sum();
        if total_weight <= 0.0 {
            return Ok(rng.gen_range(0..weights.len()));
        }
        
        let mut cumulative = 0.0;
        let target = rng.gen::<f64>() * total_weight;
        
        for (i, &weight) in weights.iter().enumerate() {
            cumulative += weight;
            if cumulative >= target {
                return Ok(i);
            }
        }
        
        Ok(weights.len() - 1)
    }

    /// Get task distribution statistics
    pub fn get_statistics(&self) -> &TaskCreationStats {
        &self.creation_stats
    }

    /// Get difficulty distribution
    pub fn get_difficulty_distribution(&self) -> Vec<(String, f64)> {
        self.difficulty_scores.iter()
            .map(|(id, &difficulty)| (id.clone(), difficulty))
            .collect()
    }

    /// Get domain distribution
    pub fn get_domain_distribution(&self) -> &HashMap<String, f64> {
        &self.domain_distribution
    }

    /// Get type distribution
    pub fn get_type_distribution(&self) -> &HashMap<TaskType, f64> {
        &self.type_distribution
    }

    /// Create a new task from memory entries
    pub async fn create_task(
        &self,
        task_id: String,
        task_type: TaskType,
        memories: Vec<MemoryEntry>,
        domain: String,
        support_ratio: f64,
    ) -> Result<MetaTask> {
        if memories.is_empty() {
            return Err(crate::error::MemoryError::InvalidInput {
                message: "Cannot create task from empty memory set".to_string()
            });
        }
        
        // Split memories into support and query sets
        let support_size = ((memories.len() as f64 * support_ratio).round() as usize).max(1);
        let query_size = memories.len() - support_size;
        
        if query_size == 0 {
            return Err(crate::error::MemoryError::InvalidInput {
                message: "Query set cannot be empty".to_string()
            });
        }
        
        let mut shuffled_memories = memories;

        // Shuffle memories
        for i in (1..shuffled_memories.len()).rev() {
            use rand::Rng;
            let mut rng = rand::thread_rng();
            let j = rng.gen_range(0..=i);
            shuffled_memories.swap(i, j);
        }
        
        let support_set = shuffled_memories[..support_size].to_vec();
        let query_set = shuffled_memories[support_size..].to_vec();
        
        // Compute difficulty
        let difficulty = self.compute_task_difficulty(&MetaTask {
            id: task_id.clone(),
            task_type: task_type.clone(),
            support_set: support_set.clone(),
            query_set: query_set.clone(),
            metadata: HashMap::new(),
            created_at: Utc::now(),
            difficulty: 0.0, // Will be computed
            domain: domain.clone(),
        }).await?;
        
        Ok(MetaTask {
            id: task_id,
            task_type,
            support_set,
            query_set,
            metadata: HashMap::new(),
            created_at: Utc::now(),
            difficulty,
            domain,
        })
    }
}

impl Default for TaskCreationStats {
    fn default() -> Self {
        Self {
            total_created: 0,
            by_type: HashMap::new(),
            by_domain: HashMap::new(),
            avg_difficulty: 0.0,
            creation_rate: 0.0,
            last_update: Utc::now(),
        }
    }
}
