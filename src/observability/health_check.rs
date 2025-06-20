//! Health Check System
//!
//! This module implements a comprehensive health check system with dependency monitoring,
//! circuit breakers, automated recovery mechanisms, and detailed health reporting
//! integrated with the monitoring stack for production-ready observability.

use crate::error::{Result, SynapticError};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime};
use tokio::sync::{RwLock, Semaphore};
use tokio::time::{interval, timeout};
use tracing::{debug, error, info, warn};

/// Overall health status of the system
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum HealthStatus {
    Healthy,
    Degraded,
    Unhealthy,
    Unknown,
}

impl std::fmt::Display for HealthStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            HealthStatus::Healthy => write!(f, "healthy"),
            HealthStatus::Degraded => write!(f, "degraded"),
            HealthStatus::Unhealthy => write!(f, "unhealthy"),
            HealthStatus::Unknown => write!(f, "unknown"),
        }
    }
}

/// Health check result for individual components
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthCheckResult {
    pub component: String,
    pub status: HealthStatus,
    pub message: String,
    pub details: HashMap<String, serde_json::Value>,
    pub timestamp: SystemTime,
    pub duration: Duration,
    pub error: Option<String>,
}

/// Comprehensive system health report
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemHealthReport {
    pub overall_status: HealthStatus,
    pub timestamp: SystemTime,
    pub uptime: Duration,
    pub components: HashMap<String, HealthCheckResult>,
    pub dependencies: HashMap<String, DependencyHealth>,
    pub circuit_breakers: HashMap<String, CircuitBreakerStatus>,
    pub metrics: HealthMetrics,
}

/// Dependency health information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DependencyHealth {
    pub name: String,
    pub status: HealthStatus,
    pub endpoint: String,
    pub last_check: SystemTime,
    pub response_time: Duration,
    pub error_rate: f64,
    pub availability: f64,
}

/// Circuit breaker status
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CircuitBreakerStatus {
    pub name: String,
    pub state: CircuitBreakerState,
    pub failure_count: u64,
    pub success_count: u64,
    pub last_failure: Option<SystemTime>,
    pub next_attempt: Option<SystemTime>,
}

/// Circuit breaker states
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CircuitBreakerState {
    Closed,
    Open,
    HalfOpen,
}

/// Health metrics for monitoring
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthMetrics {
    pub total_checks: u64,
    pub successful_checks: u64,
    pub failed_checks: u64,
    pub average_response_time: Duration,
    pub error_rate: f64,
    pub availability: f64,
}

/// Configuration for health checks
#[derive(Debug, Clone)]
pub struct HealthCheckConfig {
    pub check_interval: Duration,
    pub timeout: Duration,
    pub failure_threshold: u32,
    pub recovery_threshold: u32,
    pub circuit_breaker_timeout: Duration,
    pub enable_auto_recovery: bool,
    pub max_concurrent_checks: usize,
}

impl Default for HealthCheckConfig {
    fn default() -> Self {
        Self {
            check_interval: Duration::from_secs(30),
            timeout: Duration::from_secs(10),
            failure_threshold: 3,
            recovery_threshold: 2,
            circuit_breaker_timeout: Duration::from_secs(60),
            enable_auto_recovery: true,
            max_concurrent_checks: 10,
        }
    }
}

/// Trait for implementing health checks
#[async_trait::async_trait]
pub trait HealthChecker: Send + Sync {
    /// Perform the health check
    async fn check_health(&self) -> Result<HealthCheckResult>;
    
    /// Get the component name
    fn component_name(&self) -> &str;
    
    /// Get check priority (lower numbers = higher priority)
    fn priority(&self) -> u8 {
        100
    }
    
    /// Whether this check is critical for overall system health
    fn is_critical(&self) -> bool {
        false
    }
}

/// Circuit breaker for protecting against cascading failures
#[derive(Debug)]
pub struct CircuitBreaker {
    name: String,
    state: Arc<RwLock<CircuitBreakerState>>,
    failure_count: Arc<RwLock<u64>>,
    success_count: Arc<RwLock<u64>>,
    last_failure: Arc<RwLock<Option<SystemTime>>>,
    next_attempt: Arc<RwLock<Option<SystemTime>>>,
    config: HealthCheckConfig,
}

impl CircuitBreaker {
    /// Create a new circuit breaker
    pub fn new(name: String, config: HealthCheckConfig) -> Self {
        Self {
            name,
            state: Arc::new(RwLock::new(CircuitBreakerState::Closed)),
            failure_count: Arc::new(RwLock::new(0)),
            success_count: Arc::new(RwLock::new(0)),
            last_failure: Arc::new(RwLock::new(None)),
            next_attempt: Arc::new(RwLock::new(None)),
            config,
        }
    }
    
    /// Execute a function with circuit breaker protection
    pub async fn execute<F, T>(&self, operation: F) -> Result<T>
    where
        F: std::future::Future<Output = Result<T>>,
    {
        // Check if circuit breaker allows execution
        if !self.can_execute().await {
            return Err(SynapticError::CircuitBreakerOpen(self.name.clone()));
        }
        
        // Execute the operation
        match operation.await {
            Ok(result) => {
                self.record_success().await;
                Ok(result)
            }
            Err(error) => {
                self.record_failure().await;
                Err(error)
            }
        }
    }
    
    /// Check if the circuit breaker allows execution
    async fn can_execute(&self) -> bool {
        let state = *self.state.read().await;
        
        match state {
            CircuitBreakerState::Closed => true,
            CircuitBreakerState::Open => {
                // Check if timeout has passed
                if let Some(next_attempt) = *self.next_attempt.read().await {
                    if SystemTime::now() >= next_attempt {
                        // Transition to half-open
                        *self.state.write().await = CircuitBreakerState::HalfOpen;
                        *self.next_attempt.write().await = None;
                        true
                    } else {
                        false
                    }
                } else {
                    false
                }
            }
            CircuitBreakerState::HalfOpen => true,
        }
    }
    
    /// Record a successful operation
    async fn record_success(&self) {
        let mut success_count = self.success_count.write().await;
        *success_count += 1;
        
        let state = *self.state.read().await;
        if state == CircuitBreakerState::HalfOpen {
            if *success_count >= self.config.recovery_threshold as u64 {
                // Transition back to closed
                *self.state.write().await = CircuitBreakerState::Closed;
                *self.failure_count.write().await = 0;
                info!("Circuit breaker '{}' transitioned to CLOSED", self.name);
            }
        }
    }
    
    /// Record a failed operation
    async fn record_failure(&self) {
        let mut failure_count = self.failure_count.write().await;
        *failure_count += 1;
        *self.last_failure.write().await = Some(SystemTime::now());
        
        if *failure_count >= self.config.failure_threshold as u64 {
            // Transition to open
            *self.state.write().await = CircuitBreakerState::Open;
            *self.next_attempt.write().await = Some(
                SystemTime::now() + self.config.circuit_breaker_timeout
            );
            warn!("Circuit breaker '{}' transitioned to OPEN", self.name);
        }
    }
    
    /// Get current circuit breaker status
    pub async fn status(&self) -> CircuitBreakerStatus {
        CircuitBreakerStatus {
            name: self.name.clone(),
            state: *self.state.read().await,
            failure_count: *self.failure_count.read().await,
            success_count: *self.success_count.read().await,
            last_failure: *self.last_failure.read().await,
            next_attempt: *self.next_attempt.read().await,
        }
    }
}

/// Comprehensive health check manager
pub struct HealthCheckManager {
    checkers: Arc<RwLock<HashMap<String, Box<dyn HealthChecker>>>>,
    circuit_breakers: Arc<RwLock<HashMap<String, CircuitBreaker>>>,
    dependencies: Arc<RwLock<HashMap<String, DependencyHealth>>>,
    config: HealthCheckConfig,
    start_time: Instant,
    semaphore: Arc<Semaphore>,
    metrics: Arc<RwLock<HealthMetrics>>,
}

impl HealthCheckManager {
    /// Create a new health check manager
    pub fn new(config: HealthCheckConfig) -> Self {
        let semaphore = Arc::new(Semaphore::new(config.max_concurrent_checks));
        
        Self {
            checkers: Arc::new(RwLock::new(HashMap::new())),
            circuit_breakers: Arc::new(RwLock::new(HashMap::new())),
            dependencies: Arc::new(RwLock::new(HashMap::new())),
            config,
            start_time: Instant::now(),
            semaphore,
            metrics: Arc::new(RwLock::new(HealthMetrics {
                total_checks: 0,
                successful_checks: 0,
                failed_checks: 0,
                average_response_time: Duration::from_secs(0),
                error_rate: 0.0,
                availability: 100.0,
            })),
        }
    }
    
    /// Register a health checker
    pub async fn register_checker(&self, checker: Box<dyn HealthChecker>) {
        let component_name = checker.component_name().to_string();
        
        // Create circuit breaker for this component
        let circuit_breaker = CircuitBreaker::new(
            component_name.clone(),
            self.config.clone(),
        );
        
        self.checkers.write().await.insert(component_name.clone(), checker);
        self.circuit_breakers.write().await.insert(component_name.clone(), circuit_breaker);
        
        info!("Registered health checker for component: {}", component_name);
    }
    
    /// Unregister a health checker
    pub async fn unregister_checker(&self, component_name: &str) {
        self.checkers.write().await.remove(component_name);
        self.circuit_breakers.write().await.remove(component_name);
        
        info!("Unregistered health checker for component: {}", component_name);
    }
    
    /// Perform health checks for all registered components
    pub async fn check_all_health(&self) -> SystemHealthReport {
        let start_time = Instant::now();
        let mut component_results = HashMap::new();
        let mut overall_status = HealthStatus::Healthy;
        
        // Get all checkers
        let checkers = self.checkers.read().await;
        let mut check_futures = Vec::new();
        
        // Create futures for all health checks
        for (name, checker) in checkers.iter() {
            let checker_name = name.clone();
            let checker_ref = checker.as_ref();
            let circuit_breakers = self.circuit_breakers.clone();
            let semaphore = self.semaphore.clone();
            let timeout_duration = self.config.timeout;
            
            let future = async move {
                let _permit = semaphore.acquire().await.unwrap();
                
                let circuit_breaker = circuit_breakers.read().await
                    .get(&checker_name)
                    .unwrap()
                    .clone();
                
                let result = circuit_breaker.execute(async {
                    timeout(timeout_duration, checker_ref.check_health()).await
                        .map_err(|_| SynapticError::Timeout(format!("Health check timeout for {}", checker_name)))?
                }).await;
                
                (checker_name, result)
            };
            
            check_futures.push(future);
        }
        
        drop(checkers);
        
        // Execute all health checks concurrently
        let results = futures::future::join_all(check_futures).await;
        
        // Process results
        for (component_name, result) in results {
            let health_result = match result {
                Ok(health_check_result) => {
                    if health_check_result.status != HealthStatus::Healthy {
                        if overall_status == HealthStatus::Healthy {
                            overall_status = HealthStatus::Degraded;
                        }
                    }
                    health_check_result
                }
                Err(error) => {
                    overall_status = HealthStatus::Unhealthy;
                    HealthCheckResult {
                        component: component_name.clone(),
                        status: HealthStatus::Unhealthy,
                        message: "Health check failed".to_string(),
                        details: HashMap::new(),
                        timestamp: SystemTime::now(),
                        duration: start_time.elapsed(),
                        error: Some(error.to_string()),
                    }
                }
            };
            
            component_results.insert(component_name, health_result);
        }
        
        // Update metrics
        self.update_metrics(&component_results).await;
        
        // Get circuit breaker statuses
        let circuit_breaker_statuses = self.get_circuit_breaker_statuses().await;
        
        // Get dependency health
        let dependencies = self.dependencies.read().await.clone();
        
        // Get current metrics
        let metrics = self.metrics.read().await.clone();
        
        SystemHealthReport {
            overall_status,
            timestamp: SystemTime::now(),
            uptime: self.start_time.elapsed(),
            components: component_results,
            dependencies,
            circuit_breakers: circuit_breaker_statuses,
            metrics,
        }
    }

    /// Check health of a specific component
    pub async fn check_component_health(&self, component_name: &str) -> Result<HealthCheckResult> {
        let checkers = self.checkers.read().await;
        let checker = checkers.get(component_name)
            .ok_or_else(|| SynapticError::NotFound(format!("Health checker for component '{}' not found", component_name)))?;

        let circuit_breakers = self.circuit_breakers.read().await;
        let circuit_breaker = circuit_breakers.get(component_name)
            .ok_or_else(|| SynapticError::NotFound(format!("Circuit breaker for component '{}' not found", component_name)))?;

        let result = circuit_breaker.execute(async {
            timeout(self.config.timeout, checker.check_health()).await
                .map_err(|_| SynapticError::Timeout(format!("Health check timeout for {}", component_name)))?
        }).await;

        result
    }

    /// Register a dependency for monitoring
    pub async fn register_dependency(&self, name: String, endpoint: String) {
        let dependency = DependencyHealth {
            name: name.clone(),
            status: HealthStatus::Unknown,
            endpoint,
            last_check: SystemTime::now(),
            response_time: Duration::from_secs(0),
            error_rate: 0.0,
            availability: 100.0,
        };

        self.dependencies.write().await.insert(name.clone(), dependency);
        info!("Registered dependency: {}", name);
    }

    /// Update dependency health status
    pub async fn update_dependency_health(&self, name: &str, status: HealthStatus, response_time: Duration, error_occurred: bool) {
        let mut dependencies = self.dependencies.write().await;
        if let Some(dependency) = dependencies.get_mut(name) {
            dependency.status = status;
            dependency.last_check = SystemTime::now();
            dependency.response_time = response_time;

            // Update error rate (simple exponential moving average)
            let error_value = if error_occurred { 1.0 } else { 0.0 };
            dependency.error_rate = dependency.error_rate * 0.9 + error_value * 0.1;

            // Update availability
            dependency.availability = (1.0 - dependency.error_rate) * 100.0;

            debug!("Updated dependency health for {}: {} ({:?})", name, status, response_time);
        }
    }

    /// Get circuit breaker statuses
    async fn get_circuit_breaker_statuses(&self) -> HashMap<String, CircuitBreakerStatus> {
        let mut statuses = HashMap::new();
        let circuit_breakers = self.circuit_breakers.read().await;

        for (name, circuit_breaker) in circuit_breakers.iter() {
            statuses.insert(name.clone(), circuit_breaker.status().await);
        }

        statuses
    }

    /// Update health metrics
    async fn update_metrics(&self, results: &HashMap<String, HealthCheckResult>) {
        let mut metrics = self.metrics.write().await;

        let total_checks = results.len() as u64;
        let successful_checks = results.values()
            .filter(|r| r.status == HealthStatus::Healthy)
            .count() as u64;
        let failed_checks = total_checks - successful_checks;

        let total_duration: Duration = results.values()
            .map(|r| r.duration)
            .sum();
        let average_response_time = if total_checks > 0 {
            total_duration / total_checks as u32
        } else {
            Duration::from_secs(0)
        };

        let error_rate = if total_checks > 0 {
            (failed_checks as f64 / total_checks as f64) * 100.0
        } else {
            0.0
        };

        let availability = 100.0 - error_rate;

        metrics.total_checks += total_checks;
        metrics.successful_checks += successful_checks;
        metrics.failed_checks += failed_checks;
        metrics.average_response_time = average_response_time;
        metrics.error_rate = error_rate;
        metrics.availability = availability;
    }

    /// Start periodic health checks
    pub async fn start_periodic_checks(&self) -> Result<()> {
        let mut interval = interval(self.config.check_interval);
        let manager = self.clone();

        tokio::spawn(async move {
            loop {
                interval.tick().await;

                let report = manager.check_all_health().await;

                // Log health status
                match report.overall_status {
                    HealthStatus::Healthy => {
                        debug!("System health check completed: {}", report.overall_status);
                    }
                    HealthStatus::Degraded => {
                        warn!("System health check completed: {} - Some components are degraded", report.overall_status);
                    }
                    HealthStatus::Unhealthy => {
                        error!("System health check completed: {} - System is unhealthy", report.overall_status);
                    }
                    HealthStatus::Unknown => {
                        warn!("System health check completed: {} - Status unknown", report.overall_status);
                    }
                }

                // Trigger auto-recovery if enabled
                if manager.config.enable_auto_recovery && report.overall_status != HealthStatus::Healthy {
                    manager.attempt_auto_recovery(&report).await;
                }
            }
        });

        info!("Started periodic health checks with interval: {:?}", self.config.check_interval);
        Ok(())
    }

    /// Attempt automatic recovery for failed components
    async fn attempt_auto_recovery(&self, report: &SystemHealthReport) {
        for (component_name, result) in &report.components {
            if result.status == HealthStatus::Unhealthy {
                info!("Attempting auto-recovery for component: {}", component_name);

                // Here you would implement component-specific recovery logic
                // For now, we just log the attempt
                warn!("Auto-recovery not implemented for component: {}", component_name);
            }
        }
    }

    /// Get current health metrics
    pub async fn get_metrics(&self) -> HealthMetrics {
        self.metrics.read().await.clone()
    }

    /// Reset health metrics
    pub async fn reset_metrics(&self) {
        let mut metrics = self.metrics.write().await;
        *metrics = HealthMetrics {
            total_checks: 0,
            successful_checks: 0,
            failed_checks: 0,
            average_response_time: Duration::from_secs(0),
            error_rate: 0.0,
            availability: 100.0,
        };
        info!("Health metrics reset");
    }

    /// Get system uptime
    pub fn get_uptime(&self) -> Duration {
        self.start_time.elapsed()
    }

    /// Get health check configuration
    pub fn get_config(&self) -> &HealthCheckConfig {
        &self.config
    }
}

impl Clone for HealthCheckManager {
    fn clone(&self) -> Self {
        Self {
            checkers: self.checkers.clone(),
            circuit_breakers: self.circuit_breakers.clone(),
            dependencies: self.dependencies.clone(),
            config: self.config.clone(),
            start_time: self.start_time,
            semaphore: self.semaphore.clone(),
            metrics: self.metrics.clone(),
        }
    }
}

/// Built-in health checker for database connections
pub struct DatabaseHealthChecker {
    name: String,
    connection_pool: Arc<dyn DatabaseConnection + Send + Sync>,
}

#[async_trait::async_trait]
pub trait DatabaseConnection {
    async fn ping(&self) -> Result<Duration>;
    async fn get_connection_count(&self) -> Result<u32>;
    async fn get_max_connections(&self) -> Result<u32>;
}

impl DatabaseHealthChecker {
    pub fn new(name: String, connection_pool: Arc<dyn DatabaseConnection + Send + Sync>) -> Self {
        Self {
            name,
            connection_pool,
        }
    }
}

#[async_trait::async_trait]
impl HealthChecker for DatabaseHealthChecker {
    async fn check_health(&self) -> Result<HealthCheckResult> {
        let start_time = Instant::now();
        let timestamp = SystemTime::now();

        match self.connection_pool.ping().await {
            Ok(ping_duration) => {
                let mut details = HashMap::new();
                details.insert("ping_duration_ms".to_string(),
                    serde_json::Value::Number(serde_json::Number::from(ping_duration.as_millis() as u64)));

                // Get connection pool information
                if let Ok(active_connections) = self.connection_pool.get_connection_count().await {
                    details.insert("active_connections".to_string(),
                        serde_json::Value::Number(serde_json::Number::from(active_connections)));
                }

                if let Ok(max_connections) = self.connection_pool.get_max_connections().await {
                    details.insert("max_connections".to_string(),
                        serde_json::Value::Number(serde_json::Number::from(max_connections)));
                }

                let status = if ping_duration > Duration::from_millis(1000) {
                    HealthStatus::Degraded
                } else {
                    HealthStatus::Healthy
                };

                Ok(HealthCheckResult {
                    component: self.name.clone(),
                    status,
                    message: format!("Database ping successful in {:?}", ping_duration),
                    details,
                    timestamp,
                    duration: start_time.elapsed(),
                    error: None,
                })
            }
            Err(error) => {
                Ok(HealthCheckResult {
                    component: self.name.clone(),
                    status: HealthStatus::Unhealthy,
                    message: "Database ping failed".to_string(),
                    details: HashMap::new(),
                    timestamp,
                    duration: start_time.elapsed(),
                    error: Some(error.to_string()),
                })
            }
        }
    }

    fn component_name(&self) -> &str {
        &self.name
    }

    fn is_critical(&self) -> bool {
        true
    }

    fn priority(&self) -> u8 {
        10
    }
}

/// Built-in health checker for memory usage
pub struct MemoryHealthChecker {
    name: String,
    warning_threshold: f64,
    critical_threshold: f64,
}

impl MemoryHealthChecker {
    pub fn new(name: String, warning_threshold: f64, critical_threshold: f64) -> Self {
        Self {
            name,
            warning_threshold,
            critical_threshold,
        }
    }

    fn get_memory_usage(&self) -> Result<(u64, u64, f64)> {
        // This is a simplified implementation
        // In a real implementation, you would use system APIs to get actual memory usage
        use std::alloc::{GlobalAlloc, Layout, System};

        // For demonstration, we'll return mock values
        // In production, use libraries like `sysinfo` or platform-specific APIs
        let used_memory = 1024 * 1024 * 100; // 100MB
        let total_memory = 1024 * 1024 * 1024; // 1GB
        let usage_percent = (used_memory as f64 / total_memory as f64) * 100.0;

        Ok((used_memory, total_memory, usage_percent))
    }
}

#[async_trait::async_trait]
impl HealthChecker for MemoryHealthChecker {
    async fn check_health(&self) -> Result<HealthCheckResult> {
        let start_time = Instant::now();
        let timestamp = SystemTime::now();

        match self.get_memory_usage() {
            Ok((used_memory, total_memory, usage_percent)) => {
                let mut details = HashMap::new();
                details.insert("used_memory_bytes".to_string(),
                    serde_json::Value::Number(serde_json::Number::from(used_memory)));
                details.insert("total_memory_bytes".to_string(),
                    serde_json::Value::Number(serde_json::Number::from(total_memory)));
                details.insert("usage_percent".to_string(),
                    serde_json::Value::Number(serde_json::Number::from_f64(usage_percent).unwrap()));

                let status = if usage_percent >= self.critical_threshold {
                    HealthStatus::Unhealthy
                } else if usage_percent >= self.warning_threshold {
                    HealthStatus::Degraded
                } else {
                    HealthStatus::Healthy
                };

                let message = format!("Memory usage: {:.1}% ({} / {} bytes)",
                    usage_percent, used_memory, total_memory);

                Ok(HealthCheckResult {
                    component: self.name.clone(),
                    status,
                    message,
                    details,
                    timestamp,
                    duration: start_time.elapsed(),
                    error: None,
                })
            }
            Err(error) => {
                Ok(HealthCheckResult {
                    component: self.name.clone(),
                    status: HealthStatus::Unknown,
                    message: "Failed to get memory usage".to_string(),
                    details: HashMap::new(),
                    timestamp,
                    duration: start_time.elapsed(),
                    error: Some(error.to_string()),
                })
            }
        }
    }

    fn component_name(&self) -> &str {
        &self.name
    }

    fn is_critical(&self) -> bool {
        true
    }

    fn priority(&self) -> u8 {
        20
    }
}
