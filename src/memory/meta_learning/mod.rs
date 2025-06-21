//! Meta-Learning Module for Adaptive Memory Management
//! 
//! This module implements Model-Agnostic Meta-Learning (MAML) and other meta-learning
//! algorithms for rapid adaptation to new tasks and domains in memory management.

use crate::error::Result;
use crate::memory::types::MemoryEntry;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use chrono::{DateTime, Utc};
use std::sync::Arc;
use tokio::sync::RwLock;
use async_trait::async_trait;

pub mod maml;
pub mod reptile;
pub mod prototypical;
pub mod adaptation;
pub mod task_distribution;
pub mod domain_adaptation;
pub mod few_shot;

// Re-export key types for convenience
pub use maml::MAMLLearner;
pub use reptile::ReptileLearner;
pub use prototypical::PrototypicalLearner;
pub use domain_adaptation::{DomainAdaptationEngine, DomainAdaptationConfig, DomainAdaptationStrategy, Domain, DomainAdaptationResult};
pub use few_shot::{
    FewShotLearningEngine, FewShotConfig, FewShotAlgorithm, FewShotEpisode,
    FewShotResult, FewShotPrediction, SupportExample, QueryExample,
    DistanceMetric, AttentionType, ActivationFunction, MemoryBank, FewShotMetrics,
    PrototypeUpdateStrategy
};

/// Meta-learning configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetaLearningConfig {
    /// Inner loop learning rate for task adaptation
    pub inner_learning_rate: f64,
    /// Outer loop learning rate for meta-optimization
    pub outer_learning_rate: f64,
    /// Number of inner loop gradient steps
    pub inner_steps: usize,
    /// Meta-batch size (number of tasks per meta-update)
    pub meta_batch_size: usize,
    /// Support set size for few-shot learning
    pub support_set_size: usize,
    /// Query set size for evaluation
    pub query_set_size: usize,
    /// Maximum number of meta-training iterations
    pub max_meta_iterations: usize,
    /// Convergence threshold for meta-training
    pub convergence_threshold: f64,
    /// Enable second-order gradients for exact MAML
    pub second_order: bool,
    /// Task adaptation timeout in milliseconds
    pub adaptation_timeout_ms: u64,
}

impl Default for MetaLearningConfig {
    fn default() -> Self {
        Self {
            inner_learning_rate: 0.01,
            outer_learning_rate: 0.001,
            inner_steps: 5,
            meta_batch_size: 4,
            support_set_size: 5,
            query_set_size: 15,
            max_meta_iterations: 1000,
            convergence_threshold: 1e-6,
            second_order: true,
            adaptation_timeout_ms: 5000,
        }
    }
}

/// Task definition for meta-learning
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetaTask {
    /// Unique task identifier
    pub id: String,
    /// Task type (classification, regression, etc.)
    pub task_type: TaskType,
    /// Support set for adaptation
    pub support_set: Vec<MemoryEntry>,
    /// Query set for evaluation
    pub query_set: Vec<MemoryEntry>,
    /// Task-specific metadata
    pub metadata: HashMap<String, String>,
    /// Task creation timestamp
    pub created_at: DateTime<Utc>,
    /// Task difficulty score
    pub difficulty: f64,
    /// Domain information
    pub domain: String,
}

/// Types of meta-learning tasks
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum TaskType {
    /// Memory classification task
    Classification,
    /// Memory importance regression
    Regression,
    /// Memory retrieval ranking
    Ranking,
    /// Memory consolidation optimization
    Consolidation,
    /// Memory pattern recognition
    PatternRecognition,
    /// Custom task type
    Custom(String),
}

/// Meta-learning algorithm types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MetaAlgorithm {
    /// Model-Agnostic Meta-Learning
    MAML,
    /// Reptile algorithm (first-order approximation)
    Reptile,
    /// Prototypical Networks
    Prototypical,
    /// Matching Networks
    Matching,
    /// Relation Networks
    Relation,
}

/// Adaptation result from meta-learning
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdaptationResult {
    /// Task ID that was adapted to
    pub task_id: String,
    /// Number of adaptation steps performed
    pub adaptation_steps: usize,
    /// Final loss after adaptation
    pub final_loss: f64,
    /// Adaptation time in milliseconds
    pub adaptation_time_ms: u64,
    /// Success indicator
    pub success: bool,
    /// Confidence score of adaptation
    pub confidence: f64,
    /// Performance metrics
    pub metrics: HashMap<String, f64>,
}

/// Meta-learning performance metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetaLearningMetrics {
    /// Average adaptation loss across tasks
    pub avg_adaptation_loss: f64,
    /// Meta-learning convergence rate
    pub convergence_rate: f64,
    /// Task adaptation success rate
    pub adaptation_success_rate: f64,
    /// Average adaptation time
    pub avg_adaptation_time_ms: f64,
    /// Number of meta-iterations completed
    pub meta_iterations: usize,
    /// Memory efficiency score
    pub memory_efficiency: f64,
    /// Generalization score across domains
    pub generalization_score: f64,
}

/// Trait for meta-learning algorithms
#[async_trait]
pub trait MetaLearner: Send + Sync {
    /// Train the meta-learner on a distribution of tasks
    async fn meta_train(&mut self, tasks: &[MetaTask]) -> Result<MetaLearningMetrics>;
    
    /// Adapt to a new task using few-shot learning
    async fn adapt_to_task(&mut self, task: &MetaTask) -> Result<AdaptationResult>;
    
    /// Evaluate performance on a set of tasks
    async fn evaluate(&self, tasks: &[MetaTask]) -> Result<MetaLearningMetrics>;
    
    /// Get the current meta-parameters
    fn get_meta_parameters(&self) -> HashMap<String, Vec<f64>>;
    
    /// Set meta-parameters (for loading pre-trained models)
    fn set_meta_parameters(&mut self, parameters: HashMap<String, Vec<f64>>) -> Result<()>;
    
    /// Get algorithm-specific configuration
    fn get_config(&self) -> &MetaLearningConfig;
}

/// Main meta-learning system for memory management
pub struct MetaLearningSystem {
    /// Configuration
    #[allow(dead_code)]
    config: MetaLearningConfig,
    /// Active meta-learner
    learner: Box<dyn MetaLearner>,
    /// Task distribution manager
    task_distribution: Arc<RwLock<task_distribution::TaskDistribution>>,
    /// Adaptation history
    adaptation_history: Arc<RwLock<Vec<AdaptationResult>>>,
    /// Performance metrics
    metrics: Arc<RwLock<MetaLearningMetrics>>,
    /// Meta-parameters storage
    meta_parameters: Arc<RwLock<HashMap<String, Vec<f64>>>>,
}

impl MetaLearningSystem {
    /// Create a new meta-learning system
    pub fn new(config: MetaLearningConfig, algorithm: MetaAlgorithm) -> Result<Self> {
        let learner: Box<dyn MetaLearner> = match algorithm {
            MetaAlgorithm::MAML => Box::new(maml::MAMLLearner::new(config.clone())?),
            MetaAlgorithm::Reptile => Box::new(reptile::ReptileLearner::new(config.clone())?),
            MetaAlgorithm::Prototypical => Box::new(prototypical::PrototypicalLearner::new(config.clone())?),
            _ => return Err(crate::error::MemoryError::InvalidConfiguration {
                message: format!("Unsupported meta-learning algorithm: {:?}", algorithm)
            }),
        };

        Ok(Self {
            config,
            learner,
            task_distribution: Arc::new(RwLock::new(task_distribution::TaskDistribution::new())),
            adaptation_history: Arc::new(RwLock::new(Vec::new())),
            metrics: Arc::new(RwLock::new(MetaLearningMetrics::default())),
            meta_parameters: Arc::new(RwLock::new(HashMap::new())),
        })
    }

    /// Train the meta-learning system on a distribution of tasks
    pub async fn train(&mut self, tasks: &[MetaTask]) -> Result<MetaLearningMetrics> {
        tracing::info!("Starting meta-learning training with {} tasks", tasks.len());
        
        // Update task distribution
        {
            let mut task_dist = self.task_distribution.write().await;
            task_dist.update_distribution(tasks).await?;
        }

        // Perform meta-training
        let metrics = self.learner.meta_train(tasks).await?;
        
        // Update stored metrics
        {
            let mut stored_metrics = self.metrics.write().await;
            *stored_metrics = metrics.clone();
        }

        // Store meta-parameters
        {
            let mut meta_params = self.meta_parameters.write().await;
            *meta_params = self.learner.get_meta_parameters();
        }

        tracing::info!("Meta-learning training completed with convergence rate: {:.4}", 
                      metrics.convergence_rate);
        
        Ok(metrics)
    }

    /// Adapt to a new task using the trained meta-learner
    pub async fn adapt_to_new_task(&mut self, task: &MetaTask) -> Result<AdaptationResult> {
        tracing::debug!("Adapting to new task: {}", task.id);
        
        let start_time = std::time::Instant::now();
        let result = self.learner.adapt_to_task(task).await?;
        let adaptation_time = start_time.elapsed().as_millis() as u64;

        // Store adaptation result
        {
            let mut history = self.adaptation_history.write().await;
            let mut stored_result = result.clone();
            stored_result.adaptation_time_ms = adaptation_time;
            history.push(stored_result);
        }

        tracing::info!("Task adaptation completed for {} in {}ms with loss: {:.4}", 
                      task.id, adaptation_time, result.final_loss);
        
        Ok(result)
    }

    /// Evaluate the meta-learner on a set of test tasks
    pub async fn evaluate(&self, test_tasks: &[MetaTask]) -> Result<MetaLearningMetrics> {
        tracing::info!("Evaluating meta-learner on {} test tasks", test_tasks.len());
        
        let metrics = self.learner.evaluate(test_tasks).await?;
        
        tracing::info!("Evaluation completed with adaptation success rate: {:.2}%", 
                      metrics.adaptation_success_rate * 100.0);
        
        Ok(metrics)
    }

    /// Get adaptation history
    pub async fn get_adaptation_history(&self) -> Vec<AdaptationResult> {
        self.adaptation_history.read().await.clone()
    }

    /// Get current performance metrics
    pub async fn get_metrics(&self) -> MetaLearningMetrics {
        self.metrics.read().await.clone()
    }

    /// Save meta-parameters to storage
    pub async fn save_meta_parameters(&self) -> Result<HashMap<String, Vec<f64>>> {
        let params = self.meta_parameters.read().await.clone();
        Ok(params)
    }

    /// Load meta-parameters from storage
    pub async fn load_meta_parameters(&mut self, parameters: HashMap<String, Vec<f64>>) -> Result<()> {
        self.learner.set_meta_parameters(parameters.clone())?;
        
        {
            let mut meta_params = self.meta_parameters.write().await;
            *meta_params = parameters;
        }
        
        Ok(())
    }
}

impl Default for MetaLearningMetrics {
    fn default() -> Self {
        Self {
            avg_adaptation_loss: 0.0,
            convergence_rate: 0.0,
            adaptation_success_rate: 0.0,
            avg_adaptation_time_ms: 0.0,
            meta_iterations: 0,
            memory_efficiency: 0.0,
            generalization_score: 0.0,
        }
    }
}
