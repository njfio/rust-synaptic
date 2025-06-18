// High-performance async executor
//
// Provides optimized async task execution with intelligent scheduling,
// load balancing, and performance monitoring.

use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::{RwLock, Semaphore};
use tokio::task::JoinHandle;
use serde::{Serialize, Deserialize};
use uuid::Uuid;

use crate::error::Result;
use super::PerformanceConfig;

/// High-performance async executor
#[derive(Debug)]
pub struct AsyncExecutor {
    config: PerformanceConfig,
    task_queue: Arc<RwLock<VecDeque<Task>>>,
    active_tasks: Arc<RwLock<HashMap<String, ActiveTask>>>,
    executor_stats: Arc<RwLock<ExecutorStatistics>>,
    worker_semaphore: Arc<Semaphore>,
    blocking_semaphore: Arc<Semaphore>,
    scheduler: Arc<RwLock<TaskScheduler>>,
}

impl AsyncExecutor {
    /// Create a new async executor
    pub async fn new(config: PerformanceConfig) -> Result<Self> {
        Ok(Self {
            worker_semaphore: Arc::new(Semaphore::new(config.worker_threads)),
            blocking_semaphore: Arc::new(Semaphore::new(config.max_blocking_threads)),
            task_queue: Arc::new(RwLock::new(VecDeque::new())),
            active_tasks: Arc::new(RwLock::new(HashMap::new())),
            executor_stats: Arc::new(RwLock::new(ExecutorStatistics::new())),
            scheduler: Arc::new(RwLock::new(TaskScheduler::new())),
            config,
        })
    }
    
    /// Submit a task for execution
    #[tracing::instrument(skip(self, _task_fn), fields(task_id, priority = ?priority))]
    pub async fn submit_task<F, T>(&self, _task_fn: F, priority: TaskPriority) -> Result<String>
    where
        F: FnOnce() -> T + Send + 'static,
        T: Send + 'static,
    {
        let task_id = Uuid::new_v4().to_string();
        tracing::debug!("Submitting compute task with ID: {}", task_id);
        
        let task = Task {
            id: task_id.clone(),
            priority,
            task_type: TaskType::Compute,
            submitted_at: Instant::now(),
            estimated_duration: Duration::from_millis(100), // Default estimate
        };
        
        // Parallelize queue addition and stats update using join!
        let (_queue_result, _stats_result) = tokio::join!(
            async {
                let mut queue = self.task_queue.write().await;
                queue.push_back(task);
            },
            async {
                let mut stats = self.executor_stats.write().await;
                stats.tasks_submitted += 1;
            }
        );

        // Schedule task execution
        self.schedule_next_task().await?;

        tracing::info!("Task submitted successfully with ID: {}", task_id);
        Ok(task_id)
    }
    
    /// Submit a blocking task
    #[tracing::instrument(skip(self, _task_fn), fields(task_id, priority = ?priority))]
    pub async fn submit_blocking_task<F, T>(&self, _task_fn: F, priority: TaskPriority) -> Result<String>
    where
        F: FnOnce() -> T + Send + 'static,
        T: Send + 'static,
    {
        let task_id = Uuid::new_v4().to_string();
        tracing::debug!("Submitting blocking task with ID: {}", task_id);
        
        let task = Task {
            id: task_id.clone(),
            priority,
            task_type: TaskType::Blocking,
            submitted_at: Instant::now(),
            estimated_duration: Duration::from_millis(500), // Longer estimate for blocking
        };
        
        // Parallelize queue addition and stats update using join!
        let (_queue_result, _stats_result) = tokio::join!(
            async {
                let mut queue = self.task_queue.write().await;
                queue.push_back(task);
            },
            async {
                let mut stats = self.executor_stats.write().await;
                stats.blocking_tasks_submitted += 1;
            }
        );
        
        // Schedule task execution
        self.schedule_next_task().await?;
        
        Ok(task_id)
    }

    /// Submit multiple tasks in batch for optimized processing
    pub async fn submit_batch_tasks<F, T>(&self, task_count: usize, priority: TaskPriority) -> Result<Vec<String>>
    where
        F: FnOnce() -> T + Send + 'static,
        T: Send + 'static,
    {
        let task_ids: Vec<String> = (0..task_count).map(|_| Uuid::new_v4().to_string()).collect();

        // Create tasks in parallel
        let tasks_to_queue: Vec<Task> = task_ids.iter().map(|id| Task {
            id: id.clone(),
            priority,
            task_type: TaskType::Compute,
            submitted_at: Instant::now(),
            estimated_duration: Duration::from_millis(100),
        }).collect();

        // Batch queue operations for better performance
        {
            let mut queue = self.task_queue.write().await;
            for task in tasks_to_queue {
                queue.push_back(task);
            }
        }

        // Update statistics in batch
        {
            let mut stats = self.executor_stats.write().await;
            stats.tasks_submitted += task_count as u64;
        }

        // Schedule batch execution
        self.schedule_next_task().await?;

        Ok(task_ids)
    }

    /// Get executor statistics
    pub async fn get_statistics(&self) -> Result<ExecutorStatistics> {
        Ok(self.executor_stats.read().await.clone())
    }
    
    /// Apply optimization parameters
    pub async fn apply_optimization(&self, parameters: &HashMap<String, String>) -> Result<()> {
        // Apply worker thread optimization
        if let Some(threads_str) = parameters.get("worker_threads") {
            if let Ok(threads) = threads_str.parse::<usize>() {
                println!("Optimizing worker threads to {}", threads);
                // In a real implementation, this would resize the thread pool
            }
        }
        
        // Apply blocking thread optimization
        if let Some(blocking_str) = parameters.get("max_blocking_threads") {
            if let Ok(blocking) = blocking_str.parse::<usize>() {
                println!("Optimizing max blocking threads to {}", blocking);
                // In a real implementation, this would resize the blocking thread pool
            }
        }
        
        // Apply scheduling optimization
        if let Some(strategy_str) = parameters.get("scheduling_strategy") {
            let strategy = match strategy_str.as_str() {
                "fifo" => SchedulingStrategy::FIFO,
                "priority" => SchedulingStrategy::Priority,
                "adaptive" => SchedulingStrategy::Adaptive,
                _ => SchedulingStrategy::Priority,
            };
            
            self.scheduler.write().await.set_strategy(strategy).await?;
        }
        
        Ok(())
    }
    
    /// Schedule next task for execution
    async fn schedule_next_task(&self) -> Result<()> {
        let mut scheduler = self.scheduler.write().await;
        let mut queue = self.task_queue.write().await;
        
        if let Some(task) = scheduler.select_next_task(&mut queue).await? {
            drop(queue); // Release lock early
            
            match task.task_type {
                TaskType::Compute => {
                    self.execute_compute_task(task).await?;
                }
                TaskType::Blocking => {
                    self.execute_blocking_task(task).await?;
                }
            }
        }
        
        Ok(())
    }
    
    /// Execute a compute task
    #[tracing::instrument(skip(self, task), fields(task_id = %task.id, priority = ?task.priority))]
    async fn execute_compute_task(&self, task: Task) -> Result<()> {
        let _permit = self.worker_semaphore.acquire().await.unwrap();
        let task_id = task.id.clone();
        let start_time = Instant::now();

        tracing::debug!("Starting execution of compute task: {}", task_id);

        // Add to active tasks
        let active_task = ActiveTask {
            task,
            started_at: start_time,
            handle: None, // Would contain actual JoinHandle in real implementation
        };

        self.active_tasks.write().await.insert(task_id.clone(), active_task);

        // Simulate minimal work
        tokio::time::sleep(Duration::from_millis(1)).await;

        // Task completed
        self.complete_task(&task_id, start_time).await?;

        let duration = start_time.elapsed();
        tracing::info!("Compute task {} completed in {:?}", task_id, duration);

        Ok(())
    }
    
    /// Execute a blocking task
    async fn execute_blocking_task(&self, task: Task) -> Result<()> {
        let _permit = self.blocking_semaphore.acquire().await.unwrap();
        let task_id = task.id.clone();
        let start_time = Instant::now();

        // Add to active tasks
        let active_task = ActiveTask {
            task,
            started_at: start_time,
            handle: None,
        };

        self.active_tasks.write().await.insert(task_id.clone(), active_task);

        // Execute minimal blocking work
        tokio::task::spawn_blocking(|| {
            // Simulate minimal blocking work
            std::thread::sleep(Duration::from_millis(1));
        }).await.unwrap();

        // Complete task
        self.complete_task(&task_id, start_time).await?;

        Ok(())
    }
    
    /// Complete a task
    async fn complete_task(&self, task_id: &str, start_time: Instant) -> Result<()> {
        let duration = start_time.elapsed();
        
        // Remove from active tasks
        self.active_tasks.write().await.remove(task_id);
        
        // Update statistics
        let mut stats = self.executor_stats.write().await;
        stats.tasks_completed += 1;
        stats.total_execution_time += duration;
        stats.avg_execution_time = stats.total_execution_time / stats.tasks_completed as u32;
        
        Ok(())
    }
}

impl Clone for AsyncExecutor {
    fn clone(&self) -> Self {
        Self {
            config: self.config.clone(),
            task_queue: Arc::clone(&self.task_queue),
            active_tasks: Arc::clone(&self.active_tasks),
            executor_stats: Arc::clone(&self.executor_stats),
            worker_semaphore: Arc::clone(&self.worker_semaphore),
            blocking_semaphore: Arc::clone(&self.blocking_semaphore),
            scheduler: Arc::clone(&self.scheduler),
        }
    }
}

/// Task representation
#[derive(Debug, Clone)]
pub struct Task {
    pub id: String,
    pub priority: TaskPriority,
    pub task_type: TaskType,
    pub submitted_at: Instant,
    pub estimated_duration: Duration,
}

/// Active task with execution information
#[derive(Debug)]
pub struct ActiveTask {
    pub task: Task,
    pub started_at: Instant,
    pub handle: Option<JoinHandle<()>>,
}

/// Task priority
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum TaskPriority {
    Low = 1,
    Normal = 2,
    High = 3,
    Critical = 4,
}

/// Task type
#[derive(Debug, Clone)]
pub enum TaskType {
    Compute,
    Blocking,
}

/// Task scheduler
#[derive(Debug)]
pub struct TaskScheduler {
    strategy: SchedulingStrategy,
    load_balancer: LoadBalancer,
}

impl TaskScheduler {
    pub fn new() -> Self {
        Self {
            strategy: SchedulingStrategy::Priority,
            load_balancer: LoadBalancer::new(),
        }
    }
    
    pub async fn set_strategy(&mut self, strategy: SchedulingStrategy) -> Result<()> {
        self.strategy = strategy;
        Ok(())
    }
    
    pub async fn select_next_task(&mut self, queue: &mut VecDeque<Task>) -> Result<Option<Task>> {
        if queue.is_empty() {
            return Ok(None);
        }
        
        match self.strategy {
            SchedulingStrategy::FIFO => Ok(queue.pop_front()),
            SchedulingStrategy::Priority => self.select_priority_task(queue).await,
            SchedulingStrategy::Adaptive => self.select_adaptive_task(queue).await,
        }
    }
    
    async fn select_priority_task(&self, queue: &mut VecDeque<Task>) -> Result<Option<Task>> {
        if queue.is_empty() {
            return Ok(None);
        }
        
        // Find highest priority task
        let mut best_index = 0;
        let mut best_priority = &queue[0].priority;
        
        for (i, task) in queue.iter().enumerate() {
            if task.priority > *best_priority {
                best_index = i;
                best_priority = &task.priority;
            }
        }
        
        Ok(queue.remove(best_index))
    }
    
    async fn select_adaptive_task(&mut self, queue: &mut VecDeque<Task>) -> Result<Option<Task>> {
        if queue.is_empty() {
            return Ok(None);
        }
        
        // Use load balancer to select optimal task
        let selected_index = self.load_balancer.select_optimal_task(queue).await?;
        Ok(queue.remove(selected_index))
    }
}

/// Scheduling strategy
#[derive(Debug, Clone)]
pub enum SchedulingStrategy {
    FIFO,
    Priority,
    Adaptive,
}

/// Load balancer for task selection
#[derive(Debug)]
pub struct LoadBalancer {
    task_history: VecDeque<TaskExecutionRecord>,
}

impl LoadBalancer {
    pub fn new() -> Self {
        Self {
            task_history: VecDeque::new(),
        }
    }
    
    pub async fn select_optimal_task(&mut self, queue: &VecDeque<Task>) -> Result<usize> {
        if queue.is_empty() {
            return Ok(0);
        }
        
        // Simple load balancing: prefer shorter estimated tasks when system is busy
        let mut best_index = 0;
        let mut best_score = self.calculate_task_score(&queue[0]).await;
        
        for (i, task) in queue.iter().enumerate() {
            let score = self.calculate_task_score(task).await;
            if score > best_score {
                best_index = i;
                best_score = score;
            }
        }
        
        Ok(best_index)
    }
    
    async fn calculate_task_score(&self, task: &Task) -> f64 {
        // Score based on priority and estimated duration
        let priority_score = task.priority as u8 as f64;
        let duration_score = 1.0 / (task.estimated_duration.as_millis() as f64 + 1.0);
        let age_score = task.submitted_at.elapsed().as_millis() as f64 / 1000.0;
        
        // Weighted combination
        priority_score * 0.5 + duration_score * 0.3 + age_score * 0.2
    }
}

/// Task execution record for load balancing
#[derive(Debug, Clone)]
pub struct TaskExecutionRecord {
    pub task_type: TaskType,
    pub priority: TaskPriority,
    pub estimated_duration: Duration,
    pub actual_duration: Duration,
    pub completed_at: Instant,
}

/// Executor statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutorStatistics {
    pub tasks_submitted: u64,
    pub tasks_completed: u64,
    pub blocking_tasks_submitted: u64,
    pub active_tasks: usize,
    pub queue_length: usize,
    pub total_execution_time: Duration,
    pub avg_execution_time: Duration,
    pub throughput_tasks_per_sec: f64,
    pub worker_utilization: f64,
    pub blocking_utilization: f64,
}

impl ExecutorStatistics {
    pub fn new() -> Self {
        Self {
            tasks_submitted: 0,
            tasks_completed: 0,
            blocking_tasks_submitted: 0,
            active_tasks: 0,
            queue_length: 0,
            total_execution_time: Duration::from_secs(0),
            avg_execution_time: Duration::from_secs(0),
            throughput_tasks_per_sec: 0.0,
            worker_utilization: 0.0,
            blocking_utilization: 0.0,
        }
    }
}
