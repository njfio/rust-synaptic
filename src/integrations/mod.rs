// Real External Integrations Module
// Implements actual external service integrations

#[cfg(feature = "sql-storage")]
pub mod database;

#[cfg(feature = "ml-models")]
pub mod ml_models;

#[cfg(feature = "llm-integration")]
pub mod llm;

#[cfg(feature = "visualization")]
pub mod visualization;

pub mod redis_cache;

use crate::error::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Configuration for external integrations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IntegrationConfig {
    /// Database configuration
    #[cfg(feature = "sql-storage")]
    pub database: Option<database::DatabaseConfig>,
    
    /// ML models configuration
    #[cfg(feature = "ml-models")]
    pub ml_models: Option<ml_models::MLConfig>,
    
    /// LLM integration configuration
    #[cfg(feature = "llm-integration")]
    pub llm: Option<llm::LLMConfig>,
    
    /// Visualization configuration
    #[cfg(feature = "visualization")]
    pub visualization: Option<visualization::VisualizationConfig>,
    
    /// Redis cache configuration
    pub redis: Option<redis_cache::RedisConfig>,
}

impl Default for IntegrationConfig {
    fn default() -> Self {
        Self {
            #[cfg(feature = "sql-storage")]
            database: Some(database::DatabaseConfig::default()),
            
            #[cfg(feature = "ml-models")]
            ml_models: Some(ml_models::MLConfig::default()),
            
            #[cfg(feature = "llm-integration")]
            llm: Some(llm::LLMConfig::default()),
            
            #[cfg(feature = "visualization")]
            visualization: Some(visualization::VisualizationConfig::default()),
            
            redis: Some(redis_cache::RedisConfig::default()),
        }
    }
}

/// Integration manager for coordinating external services
pub struct IntegrationManager {
    #[allow(dead_code)]
    config: IntegrationConfig,
    
    #[cfg(feature = "sql-storage")]
    database: Option<database::DatabaseClient>,
    
    #[cfg(feature = "ml-models")]
    ml_models: Option<ml_models::MLModelManager>,
    
    #[cfg(feature = "llm-integration")]
    llm_client: Option<llm::LLMClient>,
    
    #[cfg(feature = "visualization")]
    viz_engine: Option<visualization::RealVisualizationEngine>,
    
    redis_client: Option<redis_cache::RedisClient>,
}

impl IntegrationManager {
    /// Create a new integration manager
    pub async fn new(config: IntegrationConfig) -> Result<Self> {
        let mut manager = Self {
            config: config.clone(),
            
            #[cfg(feature = "sql-storage")]
            database: None,
            
            #[cfg(feature = "ml-models")]
            ml_models: None,
            
            #[cfg(feature = "llm-integration")]
            llm_client: None,
            
            #[cfg(feature = "visualization")]
            viz_engine: None,
            
            redis_client: None,
        };

        // Initialize database connection
        #[cfg(feature = "sql-storage")]
        if let Some(db_config) = &config.database {
            manager.database = Some(database::DatabaseClient::new(db_config.clone()).await?);
        }

        // Initialize ML models
        #[cfg(feature = "ml-models")]
        if let Some(ml_config) = &config.ml_models {
            manager.ml_models = Some(ml_models::MLModelManager::new(ml_config.clone()).await?);
        }

        // Initialize LLM client
        #[cfg(feature = "llm-integration")]
        if let Some(llm_config) = &config.llm {
            manager.llm_client = Some(llm::LLMClient::new(llm_config.clone()).await?);
        }

        // Initialize visualization engine
        #[cfg(feature = "visualization")]
        if let Some(viz_config) = &config.visualization {
            manager.viz_engine = Some(visualization::RealVisualizationEngine::new(viz_config.clone()).await?);
        }

        // Initialize Redis client
        if let Some(redis_config) = &config.redis {
            manager.redis_client = Some(redis_cache::RedisClient::new(redis_config.clone()).await?);
        }

        Ok(manager)
    }

    /// Get database client
    #[cfg(feature = "sql-storage")]
    pub fn database(&self) -> Option<&database::DatabaseClient> {
        self.database.as_ref()
    }

    /// Get ML model manager
    #[cfg(feature = "ml-models")]
    pub fn ml_models(&self) -> Option<&ml_models::MLModelManager> {
        self.ml_models.as_ref()
    }

    /// Get LLM client
    #[cfg(feature = "llm-integration")]
    pub fn llm_client(&self) -> Option<&llm::LLMClient> {
        self.llm_client.as_ref()
    }

    /// Get visualization engine
    #[cfg(feature = "visualization")]
    pub fn visualization(&self) -> Option<&visualization::RealVisualizationEngine> {
        self.viz_engine.as_ref()
    }

    /// Get Redis client
    pub fn redis(&self) -> Option<&redis_cache::RedisClient> {
        self.redis_client.as_ref()
    }

    /// Health check for all integrations
    pub async fn health_check(&self) -> Result<HashMap<String, bool>> {
        let mut health = HashMap::new();

        // Check database
        #[cfg(feature = "sql-storage")]
        if let Some(db) = &self.database {
            health.insert("database".to_string(), db.health_check().await.is_ok());
        }

        // Check ML models
        #[cfg(feature = "ml-models")]
        if let Some(ml) = &self.ml_models {
            health.insert("ml_models".to_string(), ml.health_check().await.is_ok());
        }

        // Check LLM
        #[cfg(feature = "llm-integration")]
        if let Some(llm) = &self.llm_client {
            health.insert("llm".to_string(), llm.health_check().await.is_ok());
        }

        // Check visualization
        #[cfg(feature = "visualization")]
        if let Some(viz) = &self.viz_engine {
            health.insert("visualization".to_string(), viz.health_check().await.is_ok());
        }

        // Check Redis
        if let Some(redis) = &self.redis_client {
            health.insert("redis".to_string(), redis.health_check().await.is_ok());
        }

        Ok(health)
    }

    /// Shutdown all integrations gracefully
    pub async fn shutdown(&mut self) -> Result<()> {
        // Shutdown in reverse order of initialization
        
        if let Some(mut redis) = self.redis_client.take() {
            redis.shutdown().await?;
        }

        #[cfg(feature = "visualization")]
        if let Some(mut viz) = self.viz_engine.take() {
            viz.shutdown().await?;
        }

        #[cfg(feature = "llm-integration")]
        if let Some(mut llm) = self.llm_client.take() {
            llm.shutdown().await?;
        }

        #[cfg(feature = "ml-models")]
        if let Some(mut ml) = self.ml_models.take() {
            ml.shutdown().await?;
        }

        #[cfg(feature = "sql-storage")]
        if let Some(mut db) = self.database.take() {
            db.shutdown().await?;
        }

        Ok(())
    }
}

/// Integration metrics for monitoring
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IntegrationMetrics {
    pub database_queries: u64,
    pub ml_predictions: u64,
    pub llm_requests: u64,
    pub visualizations_generated: u64,
    pub cache_hits: u64,
    pub cache_misses: u64,
    pub total_errors: u64,
    pub avg_response_time_ms: f64,
}

impl Default for IntegrationMetrics {
    fn default() -> Self {
        Self {
            database_queries: 0,
            ml_predictions: 0,
            llm_requests: 0,
            visualizations_generated: 0,
            cache_hits: 0,
            cache_misses: 0,
            total_errors: 0,
            avg_response_time_ms: 0.0,
        }
    }
}
