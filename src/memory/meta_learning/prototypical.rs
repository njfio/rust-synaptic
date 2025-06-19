//! Prototypical Networks for Few-Shot Learning
//! 
//! Prototypical Networks learn a metric space where classification is performed
//! by computing distances to prototype representations of each class.

use super::{MetaLearner, MetaTask, MetaLearningConfig, AdaptationResult, MetaLearningMetrics, TaskType};
use crate::error::Result;
use crate::memory::types::MemoryEntry;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use async_trait::async_trait;
use ndarray::{Array1, Array2, Axis};
use rand::Rng;

/// Prototypical Networks learner
#[derive(Debug)]
pub struct PrototypicalLearner {
    /// Configuration
    config: MetaLearningConfig,
    /// Embedding network parameters
    embedding_params: HashMap<String, Array1<f64>>,
    /// Class prototypes for each task
    prototypes: HashMap<String, HashMap<usize, Array1<f64>>>,
    /// Embedding dimension
    embedding_dim: usize,
    /// Distance metric type
    distance_metric: DistanceMetric,
    /// Training history
    training_history: Vec<f64>,
}

/// Distance metrics for prototype comparison
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DistanceMetric {
    /// Euclidean distance
    Euclidean,
    /// Cosine distance
    Cosine,
    /// Manhattan distance
    Manhattan,
    /// Learned distance (parameterized)
    Learned,
}

/// Prototype representation
#[derive(Debug, Clone, Serialize, Deserialize)]
struct Prototype {
    /// Class label
    class_id: usize,
    /// Prototype embedding
    embedding: Vec<f64>,
    /// Number of support examples
    support_count: usize,
    /// Confidence score
    confidence: f64,
}

impl PrototypicalLearner {
    /// Create a new Prototypical Networks learner
    pub fn new(config: MetaLearningConfig) -> Result<Self> {
        let embedding_dim = 128; // Embedding dimension
        let input_dim = 256; // Input feature dimension
        
        let mut embedding_params = HashMap::new();
        let mut rng = rand::thread_rng();
        
        // Initialize embedding network (simple 2-layer network)
        let w1_scale = (2.0 / input_dim as f64).sqrt();
        let w1 = Array1::from_iter((0..input_dim * embedding_dim)
            .map(|_| rng.gen::<f64>() * w1_scale - w1_scale / 2.0));
        let b1 = Array1::zeros(embedding_dim);
        
        let w2_scale = (2.0 / embedding_dim as f64).sqrt();
        let w2 = Array1::from_iter((0..embedding_dim * embedding_dim)
            .map(|_| rng.gen::<f64>() * w2_scale - w2_scale / 2.0));
        let b2 = Array1::zeros(embedding_dim);

        embedding_params.insert("w1".to_string(), w1);
        embedding_params.insert("b1".to_string(), b1);
        embedding_params.insert("w2".to_string(), w2);
        embedding_params.insert("b2".to_string(), b2);

        Ok(Self {
            config,
            embedding_params,
            prototypes: HashMap::new(),
            embedding_dim,
            distance_metric: DistanceMetric::Euclidean,
            training_history: Vec::new(),
        })
    }

    /// Extract features from memory entries
    fn extract_features(&self, memories: &[MemoryEntry]) -> Result<Array2<f64>> {
        let feature_dim = 256;
        let mut features = Array2::zeros((memories.len(), feature_dim));
        
        for (i, memory) in memories.iter().enumerate() {
            // Content-based features
            let content_length = (memory.value.len() as f64).ln().max(0.0);
            let word_count = (memory.value.split_whitespace().count() as f64).ln().max(0.0);
            let unique_chars = memory.value.chars()
                .collect::<std::collections::HashSet<_>>().len() as f64;
            
            // Temporal features
            let age_hours = chrono::Utc::now()
                .signed_duration_since(memory.metadata.created_at)
                .num_hours() as f64;
            let normalized_age = (age_hours / (24.0 * 7.0)).min(10.0);
            
            // Access pattern features
            let access_count = (memory.metadata.access_count as f64 + 1.0).ln();
            let hours = chrono::Utc::now().signed_duration_since(memory.metadata.last_accessed).num_hours() as f64;
            let recency = (-hours / (24.0 * 7.0)).exp(); // Exponential decay

            // Memory type features
            let memory_type_vec = match memory.memory_type {
                crate::memory::types::MemoryType::ShortTerm => vec![1.0, 0.0],
                crate::memory::types::MemoryType::LongTerm => vec![0.0, 1.0],
            };

            // Content semantic features
            let semantic_features = self.extract_semantic_features(&memory.value);
            
            // Populate feature vector
            let mut idx = 0;
            features[[i, idx]] = content_length / 10.0; idx += 1;
            features[[i, idx]] = word_count / 5.0; idx += 1;
            features[[i, idx]] = unique_chars / 50.0; idx += 1;
            features[[i, idx]] = normalized_age; idx += 1;
            features[[i, idx]] = access_count / 5.0; idx += 1;
            features[[i, idx]] = recency; idx += 1;
            
            // Memory type features
            for &val in &memory_type_vec {
                features[[i, idx]] = val; idx += 1;
            }
            
            // Semantic features
            for (_j, &val) in semantic_features.iter().enumerate() {
                if idx < feature_dim {
                    features[[i, idx]] = val; idx += 1;
                }
            }
        }
        
        Ok(features)
    }

    /// Extract semantic features from content
    fn extract_semantic_features(&self, content: &str) -> Vec<f64> {
        let mut features = vec![0.0; 245]; // 256 - 11 = 245 remaining
        
        // Character n-gram features
        let chars: Vec<char> = content.chars().collect();
        let total_chars = chars.len() as f64;
        
        // Unigram features
        for (i, ch) in "abcdefghijklmnopqrstuvwxyz0123456789".chars().enumerate() {
            if i < 36 {
                let count = chars.iter().filter(|&&c| c.to_lowercase().eq(ch.to_lowercase())).count();
                features[i] = count as f64 / total_chars.max(1.0);
            }
        }
        
        // Bigram features (top frequent patterns)
        let bigrams: Vec<String> = chars.windows(2)
            .map(|w| w.iter().collect())
            .collect();
        
        let mut bigram_counts = HashMap::new();
        for bigram in bigrams {
            *bigram_counts.entry(bigram).or_insert(0) += 1;
        }
        
        let mut sorted_bigrams: Vec<_> = bigram_counts.iter().collect();
        sorted_bigrams.sort_by(|a, b| b.1.cmp(a.1));
        
        for (i, (_, &count)) in sorted_bigrams.iter().take(50).enumerate() {
            if 36 + i < features.len() {
                features[36 + i] = count as f64 / total_chars.max(1.0);
            }
        }
        
        // Word-level features
        let words: Vec<&str> = content.split_whitespace().collect();
        let total_words = words.len() as f64;
        
        // Average word length
        let avg_word_len = words.iter()
            .map(|w| w.len())
            .sum::<usize>() as f64 / total_words.max(1.0);
        if features.len() > 86 {
            features[86] = avg_word_len / 10.0;
        }
        
        // Vocabulary richness (unique words / total words)
        let unique_words = words.iter().collect::<std::collections::HashSet<_>>().len() as f64;
        if features.len() > 87 {
            features[87] = unique_words / total_words.max(1.0);
        }
        
        // Sentence count approximation
        let sentence_count = content.chars().filter(|&c| c == '.' || c == '!' || c == '?').count() as f64;
        if features.len() > 88 {
            features[88] = sentence_count / total_words.max(1.0);
        }
        
        features
    }

    /// Compute embeddings using the embedding network
    fn compute_embeddings(&self, features: &Array2<f64>) -> Result<Array2<f64>> {
        let w1 = self.embedding_params.get("w1").unwrap();
        let b1 = self.embedding_params.get("b1").unwrap();
        let w2 = self.embedding_params.get("w2").unwrap();
        let b2 = self.embedding_params.get("b2").unwrap();
        
        let input_dim = 256;
        let hidden_dim = self.embedding_dim;
        
        // Reshape weights
        let w1_matrix = Array2::from_shape_vec((input_dim, hidden_dim), w1.to_vec())
            .map_err(|e| crate::error::MemoryError::ProcessingError(
                format!("Failed to reshape w1: {}", e)
            ))?;
        let w2_matrix = Array2::from_shape_vec((hidden_dim, hidden_dim), w2.to_vec())
            .map_err(|e| crate::error::MemoryError::ProcessingError(
                format!("Failed to reshape w2: {}", e)
            ))?;
        
        // First layer
        let hidden = features.dot(&w1_matrix) + b1;
        let hidden_activated = hidden.mapv(|x| x.max(0.0)); // ReLU
        
        // Second layer (embedding layer)
        let embeddings = hidden_activated.dot(&w2_matrix) + b2;
        
        // L2 normalize embeddings
        let mut normalized_embeddings = Array2::zeros(embeddings.raw_dim());
        for (i, row) in embeddings.axis_iter(Axis(0)).enumerate() {
            let norm = (row.iter().map(|&x| x * x).sum::<f64>()).sqrt().max(1e-8);
            for (j, &val) in row.iter().enumerate() {
                normalized_embeddings[[i, j]] = val / norm;
            }
        }
        
        Ok(normalized_embeddings)
    }

    /// Compute class prototypes from support set
    fn compute_prototypes(&self, embeddings: &Array2<f64>, labels: &Array1<usize>) -> HashMap<usize, Array1<f64>> {
        let mut prototypes = HashMap::new();
        let mut class_counts = HashMap::new();
        
        // Initialize prototype accumulators
        for &label in labels.iter() {
            prototypes.entry(label).or_insert_with(|| Array1::zeros(self.embedding_dim));
            *class_counts.entry(label).or_insert(0) += 1;
        }
        
        // Accumulate embeddings for each class
        for (i, &label) in labels.iter().enumerate() {
            if let Some(prototype) = prototypes.get_mut(&label) {
                for (j, &val) in embeddings.row(i).iter().enumerate() {
                    prototype[j] += val;
                }
            }
        }
        
        // Average to get prototypes
        for (label, prototype) in prototypes.iter_mut() {
            let count = class_counts[label] as f64;
            *prototype = &*prototype / count;
        }
        
        prototypes
    }

    /// Compute distance between embeddings and prototypes
    fn compute_distances(&self, embeddings: &Array2<f64>, prototypes: &HashMap<usize, Array1<f64>>) -> Array2<f64> {
        let num_queries = embeddings.nrows();
        let num_classes = prototypes.len();
        let mut distances = Array2::zeros((num_queries, num_classes));
        
        let class_labels: Vec<usize> = prototypes.keys().cloned().collect();
        
        for (i, query_embedding) in embeddings.axis_iter(Axis(0)).enumerate() {
            for (j, &class_label) in class_labels.iter().enumerate() {
                if let Some(prototype) = prototypes.get(&class_label) {
                    let distance = match self.distance_metric {
                        DistanceMetric::Euclidean => {
                            let diff = &query_embedding.to_owned() - prototype;
                            (diff.iter().map(|&x| x * x).sum::<f64>()).sqrt()
                        },
                        DistanceMetric::Cosine => {
                            let dot_product = query_embedding.iter()
                                .zip(prototype.iter())
                                .map(|(&a, &b)| a * b)
                                .sum::<f64>();
                            1.0 - dot_product // Cosine distance (1 - cosine similarity)
                        },
                        DistanceMetric::Manhattan => {
                            query_embedding.iter()
                                .zip(prototype.iter())
                                .map(|(&a, &b)| (a - b).abs())
                                .sum::<f64>()
                        },
                        DistanceMetric::Learned => {
                            // For now, use Euclidean as default
                            let diff = &query_embedding.to_owned() - prototype;
                            (diff.iter().map(|&x| x * x).sum::<f64>()).sqrt()
                        },
                    };
                    distances[[i, j]] = distance;
                }
            }
        }
        
        distances
    }

    /// Convert distances to probabilities using softmax
    fn distances_to_probabilities(&self, distances: &Array2<f64>) -> Array2<f64> {
        let mut probabilities = Array2::zeros(distances.raw_dim());
        
        for (i, row) in distances.axis_iter(Axis(0)).enumerate() {
            // Convert distances to negative log probabilities
            let neg_distances: Array1<f64> = row.mapv(|x| -x);
            
            // Apply softmax
            let max_val = neg_distances.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
            let exp_row: Array1<f64> = neg_distances.mapv(|x| (x - max_val).exp());
            let sum_exp = exp_row.sum();
            
            for (j, &val) in exp_row.iter().enumerate() {
                probabilities[[i, j]] = val / sum_exp;
            }
        }
        
        probabilities
    }

    /// Create target labels from memory entries
    fn create_targets(&self, memories: &[MemoryEntry], task_type: &TaskType) -> Result<Array1<usize>> {
        let mut targets = Array1::zeros(memories.len());
        
        match task_type {
            TaskType::Classification => {
                for (i, memory) in memories.iter().enumerate() {
                    targets[i] = match memory.memory_type {
                        crate::memory::types::MemoryType::ShortTerm => 0,
                        crate::memory::types::MemoryType::LongTerm => 1,
                    };
                }
            },
            TaskType::Regression => {
                for (i, memory) in memories.iter().enumerate() {
                    let importance = memory.metadata.access_count as f64 / 3.0;
                    targets[i] = (importance.min(1.0) as usize).min(1);
                }
            },
            _ => {
                for i in 0..memories.len() {
                    targets[i] = i % 2;
                }
            }
        }
        
        Ok(targets)
    }

    /// Compute cross-entropy loss
    fn compute_loss(&self, probabilities: &Array2<f64>, targets: &Array1<usize>) -> f64 {
        let mut loss = 0.0;
        let batch_size = probabilities.nrows();
        
        for (i, &target) in targets.iter().enumerate() {
            if target < probabilities.ncols() {
                loss -= probabilities[[i, target]].ln().max(-10.0);
            }
        }
        
        loss / batch_size as f64
    }

    /// Update embedding parameters using gradients
    fn update_embeddings(&mut self, gradients: &HashMap<String, Array1<f64>>) {
        for (param_name, grad) in gradients {
            if let Some(param) = self.embedding_params.get_mut(param_name) {
                *param = &*param - &(grad * self.config.outer_learning_rate);
            }
        }
    }

    /// Compute gradients for embedding parameters (simplified)
    fn compute_embedding_gradients(
        &self,
        features: &Array2<f64>,
        targets: &Array1<usize>,
    ) -> Result<HashMap<String, Array1<f64>>> {
        let mut gradients = HashMap::new();
        let epsilon = 1e-4;
        
        // Compute baseline loss
        let baseline_embeddings = self.compute_embeddings(features)?;
        let baseline_prototypes = self.compute_prototypes(&baseline_embeddings, targets);
        let baseline_distances = self.compute_distances(&baseline_embeddings, &baseline_prototypes);
        let baseline_probs = self.distances_to_probabilities(&baseline_distances);
        let baseline_loss = self.compute_loss(&baseline_probs, targets);
        
        // Compute gradients for each parameter
        for (param_name, param_values) in &self.embedding_params {
            let mut param_grad = Array1::zeros(param_values.len());
            
            // Sample a subset of parameters for efficiency
            for i in (0..param_values.len()).step_by(20) {
                let mut perturbed_params = self.embedding_params.clone();
                let mut perturbed_param = param_values.clone();
                perturbed_param[i] += epsilon;
                perturbed_params.insert(param_name.clone(), perturbed_param);
                
                // Create temporary learner with perturbed parameters
                let temp_learner = self.clone_with_params(&perturbed_params);
                let perturbed_embeddings = temp_learner.compute_embeddings(features)?;
                let perturbed_prototypes = temp_learner.compute_prototypes(&perturbed_embeddings, targets);
                let perturbed_distances = temp_learner.compute_distances(&perturbed_embeddings, &perturbed_prototypes);
                let perturbed_probs = temp_learner.distances_to_probabilities(&perturbed_distances);
                let perturbed_loss = temp_learner.compute_loss(&perturbed_probs, targets);
                
                param_grad[i] = (perturbed_loss - baseline_loss) / epsilon;
            }
            
            gradients.insert(param_name.clone(), param_grad);
        }
        
        Ok(gradients)
    }

    /// Clone learner with different parameters (for gradient computation)
    fn clone_with_params(&self, params: &HashMap<String, Array1<f64>>) -> Self {
        Self {
            config: self.config.clone(),
            embedding_params: params.clone(),
            prototypes: HashMap::new(),
            embedding_dim: self.embedding_dim,
            distance_metric: self.distance_metric.clone(),
            training_history: Vec::new(),
        }
    }
}

#[async_trait]
impl MetaLearner for PrototypicalLearner {
    async fn meta_train(&mut self, tasks: &[MetaTask]) -> Result<MetaLearningMetrics> {
        tracing::info!("Starting Prototypical Networks meta-training with {} tasks", tasks.len());
        
        let mut total_loss = 0.0;
        let mut successful_adaptations = 0;
        let start_time = std::time::Instant::now();
        
        for meta_iter in 0..self.config.max_meta_iterations {
            // Sample meta-batch of tasks
            let batch_indices: Vec<usize> = (0..self.config.meta_batch_size)
                .map(|_| {
                    use rand::Rng;
                    let mut rng = rand::thread_rng();
                    rng.gen_range(0..tasks.len())
                })
                .collect();
            
            let mut batch_loss = 0.0;
            let mut batch_gradients: HashMap<String, Array1<f64>> = HashMap::new();
            
            for &task_idx in &batch_indices {
                let task = &tasks[task_idx];
                
                // Extract features and targets
                let support_features = self.extract_features(&task.support_set)?;
                let support_targets = self.create_targets(&task.support_set, &task.task_type)?;
                let query_features = self.extract_features(&task.query_set)?;
                let query_targets = self.create_targets(&task.query_set, &task.task_type)?;
                
                // Compute embeddings
                let support_embeddings = self.compute_embeddings(&support_features)?;
                let query_embeddings = self.compute_embeddings(&query_features)?;
                
                // Compute prototypes from support set
                let prototypes = self.compute_prototypes(&support_embeddings, &support_targets);
                
                // Compute distances and probabilities for query set
                let distances = self.compute_distances(&query_embeddings, &prototypes);
                let probabilities = self.distances_to_probabilities(&distances);
                
                // Compute loss
                let loss = self.compute_loss(&probabilities, &query_targets);
                batch_loss += loss;
                
                // Compute gradients
                let gradients = self.compute_embedding_gradients(&query_features, &query_targets)?;
                
                // Accumulate gradients
                for (param_name, grad) in gradients {
                    batch_gradients.entry(param_name)
                        .and_modify(|acc| *acc = &*acc + &grad)
                        .or_insert(grad);
                }
                
                if loss < 1.0 {
                    successful_adaptations += 1;
                }
            }
            
            // Average gradients and update parameters
            batch_loss /= self.config.meta_batch_size as f64;
            total_loss += batch_loss;
            
            for grad in batch_gradients.values_mut() {
                *grad = &*grad / self.config.meta_batch_size as f64;
            }
            
            self.update_embeddings(&batch_gradients);
            self.training_history.push(batch_loss);
            
            // Check convergence
            if meta_iter > 20 && batch_loss < self.config.convergence_threshold {
                tracing::info!("Prototypical Networks converged at iteration {}", meta_iter);
                break;
            }
            
            if meta_iter % 50 == 0 {
                tracing::info!("Proto meta-iteration {}: loss = {:.4}", meta_iter, batch_loss);
            }
        }
        
        let training_time = start_time.elapsed().as_millis() as f64;
        let avg_loss = total_loss / self.training_history.len() as f64;
        let success_rate = successful_adaptations as f64 / (self.training_history.len() * self.config.meta_batch_size) as f64;
        
        Ok(MetaLearningMetrics {
            avg_adaptation_loss: avg_loss,
            convergence_rate: 1.0 / self.training_history.len() as f64,
            adaptation_success_rate: success_rate,
            avg_adaptation_time_ms: training_time / self.training_history.len() as f64,
            meta_iterations: self.training_history.len(),
            memory_efficiency: 0.85,
            generalization_score: success_rate,
        })
    }

    async fn adapt_to_task(&mut self, task: &MetaTask) -> Result<AdaptationResult> {
        let start_time = std::time::Instant::now();
        
        // Extract features and targets
        let support_features = self.extract_features(&task.support_set)?;
        let support_targets = self.create_targets(&task.support_set, &task.task_type)?;
        let query_features = self.extract_features(&task.query_set)?;
        let query_targets = self.create_targets(&task.query_set, &task.task_type)?;
        
        // Compute embeddings
        let support_embeddings = self.compute_embeddings(&support_features)?;
        let query_embeddings = self.compute_embeddings(&query_features)?;
        
        // Compute prototypes
        let prototypes = self.compute_prototypes(&support_embeddings, &support_targets);
        
        // Store prototypes for this task
        self.prototypes.insert(task.id.clone(), prototypes.clone());
        
        // Evaluate on query set
        let distances = self.compute_distances(&query_embeddings, &prototypes);
        let probabilities = self.distances_to_probabilities(&distances);
        let final_loss = self.compute_loss(&probabilities, &query_targets);
        
        let adaptation_time = start_time.elapsed().as_millis() as u64;
        let success = final_loss < 1.0;
        let confidence = 1.0 / (1.0 + final_loss);
        
        let mut metrics = HashMap::new();
        metrics.insert("final_loss".to_string(), final_loss);
        metrics.insert("confidence".to_string(), confidence);
        metrics.insert("num_prototypes".to_string(), prototypes.len() as f64);
        
        Ok(AdaptationResult {
            task_id: task.id.clone(),
            adaptation_steps: 1, // Prototypical networks adapt in one step
            final_loss,
            adaptation_time_ms: adaptation_time,
            success,
            confidence,
            metrics,
        })
    }

    async fn evaluate(&self, tasks: &[MetaTask]) -> Result<MetaLearningMetrics> {
        let mut total_loss = 0.0;
        let mut successful_adaptations = 0;
        let mut total_time = 0u64;
        
        for task in tasks {
            let start_time = std::time::Instant::now();
            
            // Extract features and targets
            let support_features = self.extract_features(&task.support_set)?;
            let support_targets = self.create_targets(&task.support_set, &task.task_type)?;
            let query_features = self.extract_features(&task.query_set)?;
            let query_targets = self.create_targets(&task.query_set, &task.task_type)?;
            
            // Compute embeddings and prototypes
            let support_embeddings = self.compute_embeddings(&support_features)?;
            let query_embeddings = self.compute_embeddings(&query_features)?;
            let prototypes = self.compute_prototypes(&support_embeddings, &support_targets);
            
            // Evaluate
            let distances = self.compute_distances(&query_embeddings, &prototypes);
            let probabilities = self.distances_to_probabilities(&distances);
            let loss = self.compute_loss(&probabilities, &query_targets);
            
            total_loss += loss;
            total_time += start_time.elapsed().as_millis() as u64;
            
            if loss < 1.0 {
                successful_adaptations += 1;
            }
        }
        
        let avg_loss = total_loss / tasks.len() as f64;
        let success_rate = successful_adaptations as f64 / tasks.len() as f64;
        let avg_time = total_time as f64 / tasks.len() as f64;
        
        Ok(MetaLearningMetrics {
            avg_adaptation_loss: avg_loss,
            convergence_rate: 1.0,
            adaptation_success_rate: success_rate,
            avg_adaptation_time_ms: avg_time,
            meta_iterations: 0,
            memory_efficiency: 0.85,
            generalization_score: success_rate,
        })
    }

    fn get_meta_parameters(&self) -> HashMap<String, Vec<f64>> {
        self.embedding_params.iter()
            .map(|(k, v)| (k.clone(), v.to_vec()))
            .collect()
    }

    fn set_meta_parameters(&mut self, parameters: HashMap<String, Vec<f64>>) -> Result<()> {
        for (name, values) in parameters {
            self.embedding_params.insert(name, Array1::from_vec(values));
        }
        Ok(())
    }

    fn get_config(&self) -> &MetaLearningConfig {
        &self.config
    }
}
