//! Model-Agnostic Meta-Learning (MAML) Implementation
//! 
//! This module implements the MAML algorithm for few-shot learning in memory management.
//! MAML learns good initial parameters that can be quickly adapted to new tasks.

use super::{MetaLearner, MetaTask, MetaLearningConfig, AdaptationResult, MetaLearningMetrics, TaskType};
use crate::error::Result;
use crate::memory::types::MemoryEntry;

use std::collections::HashMap;
use async_trait::async_trait;
use ndarray::{Array1, Array2};
use rand::Rng;

/// MAML learner implementation
#[derive(Debug)]
pub struct MAMLLearner {
    /// Configuration
    config: MetaLearningConfig,
    /// Meta-parameters (initial parameters for adaptation)
    meta_parameters: HashMap<String, Array1<f64>>,
    /// Gradient accumulator for meta-updates
    gradient_accumulator: HashMap<String, Array1<f64>>,
    /// Task-specific adapted parameters
    adapted_parameters: HashMap<String, HashMap<String, Array1<f64>>>,
    /// Loss history for convergence tracking
    loss_history: Vec<f64>,
    /// Current meta-iteration
    meta_iteration: usize,
}



/// Memory feature extractor
#[derive(Debug)]
#[allow(dead_code)]
struct MemoryFeatureExtractor {
    /// Feature dimension
    feature_dim: usize,
    /// Vocabulary for text features
    vocabulary: HashMap<String, usize>,
    /// Feature normalization parameters
    normalization_params: HashMap<String, (f64, f64)>, // (mean, std)
}

impl MAMLLearner {
    /// Create a new MAML learner
    pub fn new(config: MetaLearningConfig) -> Result<Self> {
        let mut meta_parameters = HashMap::new();
        let mut gradient_accumulator = HashMap::new();
        
        // Initialize meta-parameters for memory processing network
        let input_dim = 512; // Memory feature dimension
        let hidden_dim = 256;
        let output_dim = match config.support_set_size {
            n if n <= 5 => n,
            _ => 10, // Default to 10-way classification
        };

        // Initialize weights with Xavier initialization
        let mut rng = rand::thread_rng();
        
        // Input to hidden layer
        let w1_scale = (2.0 / input_dim as f64).sqrt();
        let w1 = Array1::from_iter((0..input_dim * hidden_dim)
            .map(|_| rng.gen::<f64>() * w1_scale - w1_scale / 2.0));
        let b1 = Array1::zeros(hidden_dim);
        
        // Hidden to output layer
        let w2_scale = (2.0 / hidden_dim as f64).sqrt();
        let w2 = Array1::from_iter((0..hidden_dim * output_dim)
            .map(|_| rng.gen::<f64>() * w2_scale - w2_scale / 2.0));
        let b2 = Array1::zeros(output_dim);

        meta_parameters.insert("w1".to_string(), w1.clone());
        meta_parameters.insert("b1".to_string(), b1.clone());
        meta_parameters.insert("w2".to_string(), w2.clone());
        meta_parameters.insert("b2".to_string(), b2.clone());

        // Initialize gradient accumulators
        gradient_accumulator.insert("w1".to_string(), Array1::zeros(w1.len()));
        gradient_accumulator.insert("b1".to_string(), Array1::zeros(b1.len()));
        gradient_accumulator.insert("w2".to_string(), Array1::zeros(w2.len()));
        gradient_accumulator.insert("b2".to_string(), Array1::zeros(b2.len()));

        Ok(Self {
            config,
            meta_parameters,
            gradient_accumulator,
            adapted_parameters: HashMap::new(),
            loss_history: Vec::new(),
            meta_iteration: 0,
        })
    }

    /// Extract features from memory entries
    fn extract_features(&self, memories: &[MemoryEntry]) -> Result<Array2<f64>> {
        let feature_dim = 512;
        let mut features = Array2::zeros((memories.len(), feature_dim));
        
        for (i, memory) in memories.iter().enumerate() {
            // Extract basic features
            let content_length = memory.value.len() as f64;
            let word_count = memory.value.split_whitespace().count() as f64;
            let char_diversity = memory.value.chars().collect::<std::collections::HashSet<_>>().len() as f64;
            
            // Temporal features
            let age_hours = chrono::Utc::now()
                .signed_duration_since(memory.metadata.created_at)
                .num_hours() as f64;

            // Access pattern features
            let access_count = memory.metadata.access_count as f64;
            let last_access_hours = chrono::Utc::now()
                .signed_duration_since(memory.metadata.last_accessed)
                .num_hours() as f64;

            // Memory type encoding
            let memory_type_encoding = match memory.memory_type {
                crate::memory::types::MemoryType::ShortTerm => 0.0,
                crate::memory::types::MemoryType::LongTerm => 1.0,
            };

            // Content hash features (simple hash-based encoding)
            let content_hash = self.simple_hash(&memory.value);
            
            // Populate feature vector
            features[[i, 0]] = content_length / 1000.0; // Normalize
            features[[i, 1]] = word_count / 100.0;
            features[[i, 2]] = char_diversity / 100.0;
            features[[i, 3]] = age_hours / (24.0 * 7.0); // Normalize to weeks
            features[[i, 4]] = access_count / 10.0;
            features[[i, 5]] = last_access_hours / (24.0 * 7.0);
            features[[i, 6]] = memory_type_encoding;
            
            // Fill remaining dimensions with content hash features
            for j in 7..feature_dim.min(7 + content_hash.len()) {
                features[[i, j]] = content_hash[j - 7];
            }
        }
        
        Ok(features)
    }

    /// Simple hash function for content encoding
    fn simple_hash(&self, content: &str) -> Vec<f64> {
        let mut hash_features = vec![0.0; 505]; // 512 - 7 = 505 remaining features
        
        // Character frequency features
        let mut char_counts = HashMap::new();
        for ch in content.chars() {
            *char_counts.entry(ch).or_insert(0) += 1;
        }
        
        // Normalize character frequencies
        let total_chars = content.len() as f64;
        for (i, ch) in "abcdefghijklmnopqrstuvwxyz0123456789 ".chars().enumerate() {
            if i < hash_features.len() {
                hash_features[i] = char_counts.get(&ch).unwrap_or(&0).clone() as f64 / total_chars;
            }
        }
        
        // N-gram features (bigrams)
        let bigrams: Vec<String> = content.chars()
            .collect::<Vec<_>>()
            .windows(2)
            .map(|w| w.iter().collect())
            .collect();
        
        let mut bigram_counts = HashMap::new();
        for bigram in bigrams {
            *bigram_counts.entry(bigram).or_insert(0) += 1;
        }
        
        // Add top bigram frequencies
        let mut bigram_freqs: Vec<_> = bigram_counts.iter().collect();
        bigram_freqs.sort_by(|a, b| b.1.cmp(a.1));
        
        for (i, (_, &count)) in bigram_freqs.iter().take(100).enumerate() {
            if 37 + i < hash_features.len() {
                hash_features[37 + i] = count as f64 / total_chars;
            }
        }
        
        hash_features
    }

    /// Forward pass through the network
    fn forward(&self, features: &Array2<f64>, parameters: &HashMap<String, Array1<f64>>) -> Result<Array2<f64>> {
        let w1 = parameters.get("w1").unwrap();
        let b1 = parameters.get("b1").unwrap();
        let w2 = parameters.get("w2").unwrap();
        let b2 = parameters.get("b2").unwrap();
        
        let input_dim = 512;
        let hidden_dim = b1.len();
        let output_dim = b2.len();
        
        // Reshape weights
        let w1_matrix = Array2::from_shape_vec((input_dim, hidden_dim), w1.to_vec())
            .map_err(|e| crate::error::MemoryError::ProcessingError(
                format!("Failed to reshape w1: {}", e)
            ))?;
        let w2_matrix = Array2::from_shape_vec((hidden_dim, output_dim), w2.to_vec())
            .map_err(|e| crate::error::MemoryError::ProcessingError(
                format!("Failed to reshape w2: {}", e)
            ))?;
        
        // First layer: features @ w1 + b1
        let hidden = features.dot(&w1_matrix) + b1;
        
        // Apply ReLU activation
        let hidden_activated = hidden.mapv(|x| x.max(0.0));
        
        // Second layer: hidden @ w2 + b2
        let output = hidden_activated.dot(&w2_matrix) + b2;
        
        // Apply softmax for classification
        let mut softmax_output = Array2::zeros(output.raw_dim());
        for (i, row) in output.axis_iter(ndarray::Axis(0)).enumerate() {
            let max_val = row.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
            let exp_row: Array1<f64> = row.mapv(|x| (x - max_val).exp());
            let sum_exp = exp_row.sum();
            for (j, &val) in exp_row.iter().enumerate() {
                softmax_output[[i, j]] = val / sum_exp;
            }
        }
        
        Ok(softmax_output)
    }

    /// Compute cross-entropy loss
    fn compute_loss(&self, predictions: &Array2<f64>, targets: &Array1<usize>) -> f64 {
        let mut loss = 0.0;
        let batch_size = predictions.nrows();
        
        for (i, &target) in targets.iter().enumerate() {
            if target < predictions.ncols() {
                loss -= predictions[[i, target]].ln();
            }
        }
        
        loss / batch_size as f64
    }

    /// Compute gradients using finite differences (simplified)
    fn compute_gradients(
        &self,
        features: &Array2<f64>,
        targets: &Array1<usize>,
        parameters: &HashMap<String, Array1<f64>>,
    ) -> Result<HashMap<String, Array1<f64>>> {
        let mut gradients = HashMap::new();
        let epsilon = 1e-5;
        
        // Compute baseline loss
        let baseline_predictions = self.forward(features, parameters)?;
        let baseline_loss = self.compute_loss(&baseline_predictions, targets);
        
        // Compute gradients for each parameter
        for (param_name, param_values) in parameters {
            let mut param_grad = Array1::zeros(param_values.len());
            
            for i in 0..param_values.len() {
                // Create perturbed parameters
                let mut perturbed_params = parameters.clone();
                let mut perturbed_param = param_values.clone();
                perturbed_param[i] += epsilon;
                perturbed_params.insert(param_name.clone(), perturbed_param);
                
                // Compute perturbed loss
                let perturbed_predictions = self.forward(features, &perturbed_params)?;
                let perturbed_loss = self.compute_loss(&perturbed_predictions, targets);
                
                // Finite difference gradient
                param_grad[i] = (perturbed_loss - baseline_loss) / epsilon;
            }
            
            gradients.insert(param_name.clone(), param_grad);
        }
        
        Ok(gradients)
    }

    /// Perform inner loop adaptation for a task
    async fn inner_loop_adaptation(
        &self,
        task: &MetaTask,
        initial_params: &HashMap<String, Array1<f64>>,
    ) -> Result<HashMap<String, Array1<f64>>> {
        let mut adapted_params = initial_params.clone();
        
        // Extract features and targets from support set
        let features = self.extract_features(&task.support_set)?;
        let targets = self.create_targets(&task.support_set, &task.task_type)?;
        
        // Perform inner loop gradient steps
        for step in 0..self.config.inner_steps {
            // Compute gradients
            let gradients = self.compute_gradients(&features, &targets, &adapted_params)?;
            
            // Update parameters
            for (param_name, grad) in gradients {
                if let Some(param) = adapted_params.get_mut(&param_name) {
                    *param = &*param - &(grad * self.config.inner_learning_rate);
                }
            }
            
            // Log progress
            if step % 2 == 0 {
                let predictions = self.forward(&features, &adapted_params)?;
                let loss = self.compute_loss(&predictions, &targets);
                tracing::debug!("Inner step {}: loss = {:.4}", step, loss);
            }
        }
        
        Ok(adapted_params)
    }

    /// Create target labels from memory entries
    fn create_targets(&self, memories: &[MemoryEntry], task_type: &TaskType) -> Result<Array1<usize>> {
        let mut targets = Array1::zeros(memories.len());
        
        match task_type {
            TaskType::Classification => {
                // Simple classification based on memory type
                for (i, memory) in memories.iter().enumerate() {
                    targets[i] = match memory.memory_type {
                        crate::memory::types::MemoryType::ShortTerm => 0,
                        crate::memory::types::MemoryType::LongTerm => 1,
                    };
                }
            },
            TaskType::Regression => {
                // For regression, we'll use importance scores as targets
                // Convert to classification bins for simplicity
                for (i, memory) in memories.iter().enumerate() {
                    let importance = memory.metadata.access_count as f64 / 10.0;
                    targets[i] = (importance.min(1.0) as usize).min(1);
                }
            },
            _ => {
                // Default to simple indexing
                for i in 0..memories.len() {
                    targets[i] = i % 2; // 2-way classification
                }
            }
        }
        
        Ok(targets)
    }
}

#[async_trait]
impl MetaLearner for MAMLLearner {
    async fn meta_train(&mut self, tasks: &[MetaTask]) -> Result<MetaLearningMetrics> {
        tracing::info!("Starting MAML meta-training with {} tasks", tasks.len());
        
        let mut total_loss = 0.0;
        let mut successful_adaptations = 0;
        let start_time = std::time::Instant::now();
        
        for meta_iter in 0..self.config.max_meta_iterations {
            self.meta_iteration = meta_iter;
            
            // Sample meta-batch of tasks
            let batch_indices: Vec<usize> = (0..self.config.meta_batch_size)
                .map(|_| {
                    use rand::Rng;
                    let mut rng = rand::thread_rng();
                    rng.gen_range(0..tasks.len())
                })
                .collect();
            
            // Reset gradient accumulator
            for grad in self.gradient_accumulator.values_mut() {
                grad.fill(0.0);
            }
            
            let mut batch_loss = 0.0;
            
            // Process each task in the meta-batch
            for &task_idx in &batch_indices {
                let task = &tasks[task_idx];
                
                // Inner loop adaptation
                let adapted_params = self.inner_loop_adaptation(task, &self.meta_parameters).await?;
                
                // Evaluate on query set
                let query_features = self.extract_features(&task.query_set)?;
                let query_targets = self.create_targets(&task.query_set, &task.task_type)?;
                
                let query_predictions = self.forward(&query_features, &adapted_params)?;
                let query_loss = self.compute_loss(&query_predictions, &query_targets);
                
                batch_loss += query_loss;
                
                // Compute meta-gradients (simplified)
                let meta_gradients = self.compute_gradients(&query_features, &query_targets, &adapted_params)?;
                
                // Accumulate gradients
                for (param_name, grad) in meta_gradients {
                    if let Some(acc_grad) = self.gradient_accumulator.get_mut(&param_name) {
                        *acc_grad = &*acc_grad + &grad;
                    }
                }
                
                if query_loss < 1.0 {
                    successful_adaptations += 1;
                }
            }
            
            // Meta-update
            batch_loss /= self.config.meta_batch_size as f64;
            total_loss += batch_loss;
            
            for (param_name, param) in self.meta_parameters.iter_mut() {
                if let Some(grad) = self.gradient_accumulator.get(param_name) {
                    let avg_grad = grad / self.config.meta_batch_size as f64;
                    *param = &*param - &(avg_grad * self.config.outer_learning_rate);
                }
            }
            
            self.loss_history.push(batch_loss);
            
            // Check convergence
            if meta_iter > 10 && batch_loss < self.config.convergence_threshold {
                tracing::info!("MAML converged at iteration {}", meta_iter);
                break;
            }
            
            if meta_iter % 100 == 0 {
                tracing::info!("Meta-iteration {}: loss = {:.4}", meta_iter, batch_loss);
            }
        }
        
        let training_time = start_time.elapsed().as_millis() as f64;
        let avg_loss = total_loss / self.meta_iteration as f64;
        let success_rate = successful_adaptations as f64 / (self.meta_iteration * self.config.meta_batch_size) as f64;
        
        Ok(MetaLearningMetrics {
            avg_adaptation_loss: avg_loss,
            convergence_rate: 1.0 / self.meta_iteration as f64,
            adaptation_success_rate: success_rate,
            avg_adaptation_time_ms: training_time / self.meta_iteration as f64,
            meta_iterations: self.meta_iteration,
            memory_efficiency: 0.8, // Placeholder
            generalization_score: success_rate,
        })
    }

    async fn adapt_to_task(&mut self, task: &MetaTask) -> Result<AdaptationResult> {
        let start_time = std::time::Instant::now();
        
        // Perform adaptation
        let adapted_params = self.inner_loop_adaptation(task, &self.meta_parameters).await?;
        
        // Evaluate adaptation
        let query_features = self.extract_features(&task.query_set)?;
        let query_targets = self.create_targets(&task.query_set, &task.task_type)?;
        let predictions = self.forward(&query_features, &adapted_params)?;
        let final_loss = self.compute_loss(&predictions, &query_targets);
        
        let adaptation_time = start_time.elapsed().as_millis() as u64;
        let success = final_loss < 1.0;
        let confidence = 1.0 / (1.0 + final_loss);
        
        // Store adapted parameters
        self.adapted_parameters.insert(task.id.clone(), adapted_params);
        
        let mut metrics = HashMap::new();
        metrics.insert("final_loss".to_string(), final_loss);
        metrics.insert("confidence".to_string(), confidence);
        
        Ok(AdaptationResult {
            task_id: task.id.clone(),
            adaptation_steps: self.config.inner_steps,
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
            
            // Adapt to task
            let adapted_params = self.inner_loop_adaptation(task, &self.meta_parameters).await?;
            
            // Evaluate
            let query_features = self.extract_features(&task.query_set)?;
            let query_targets = self.create_targets(&task.query_set, &task.task_type)?;
            let predictions = self.forward(&query_features, &adapted_params)?;
            let loss = self.compute_loss(&predictions, &query_targets);
            
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
            memory_efficiency: 0.8,
            generalization_score: success_rate,
        })
    }

    fn get_meta_parameters(&self) -> HashMap<String, Vec<f64>> {
        self.meta_parameters.iter()
            .map(|(k, v)| (k.clone(), v.to_vec()))
            .collect()
    }

    fn set_meta_parameters(&mut self, parameters: HashMap<String, Vec<f64>>) -> Result<()> {
        for (name, values) in parameters {
            self.meta_parameters.insert(name, Array1::from_vec(values));
        }
        Ok(())
    }

    fn get_config(&self) -> &MetaLearningConfig {
        &self.config
    }
}
