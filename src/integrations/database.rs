// Real PostgreSQL Database Integration
// Implements actual database persistence for memory entries and analytics

#[cfg(feature = "sql-storage")]
use sqlx::{PgPool, Row, postgres::PgPoolOptions};
use crate::error::{Result, MemoryError};
use crate::memory::types::{MemoryEntry, MemoryType, MemoryMetadata};
use crate::memory::management::analytics::{AnalyticsEvent, AnalyticsInsight};
use crate::memory::storage::Storage;
use serde::{Deserialize, Serialize};
use chrono::{DateTime, Utc};
use uuid::Uuid;
use std::collections::HashMap;
use std::sync::Mutex;

/// Database configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatabaseConfig {
    /// PostgreSQL connection URL
    pub database_url: String,
    /// Maximum number of connections in the pool
    pub max_connections: u32,
    /// Connection timeout in seconds
    pub connect_timeout_secs: u64,
    /// Enable SSL
    pub ssl_mode: String,
    /// Schema name
    pub schema: String,
}

impl Default for DatabaseConfig {
    fn default() -> Self {
        Self {
            database_url: std::env::var("DATABASE_URL")
                .unwrap_or_else(|_| "postgresql://synaptic_user:synaptic_pass@localhost:11110/synaptic_db".to_string()),
            max_connections: 10,
            connect_timeout_secs: 30,
            ssl_mode: "prefer".to_string(),
            schema: "public".to_string(),
        }
    }
}

/// Real database client for PostgreSQL
#[derive(Debug)]
pub struct DatabaseClient {
    #[cfg(feature = "sql-storage")]
    pool: PgPool,
    config: DatabaseConfig,
    metrics: Mutex<DatabaseMetrics>,
}

#[derive(Debug, Clone, Default)]
pub struct DatabaseMetrics {
    pub queries_executed: u64,
    pub total_query_time_ms: u64,
    pub connection_errors: u64,
    pub active_connections: u32,
}

impl DatabaseClient {
    /// Create a new database client with real PostgreSQL connection
    pub async fn new(config: DatabaseConfig) -> Result<Self> {
        #[cfg(feature = "sql-storage")]
        {
            let pool = PgPoolOptions::new()
                .max_connections(config.max_connections)
                .acquire_timeout(std::time::Duration::from_secs(config.connect_timeout_secs))
                .connect(&config.database_url)
                .await
                .map_err(|e| MemoryError::storage(format!("Failed to connect to database: {}", e)))?;

            let mut client = Self {
                pool,
                config,
                metrics: Mutex::new(DatabaseMetrics::default()),
            };

            client.run_migrations().await?;
            Ok(client)
        }

        #[cfg(not(feature = "sql-storage"))]
        {
            Err(MemoryError::configuration("SQL storage feature not enabled"))
        }
    }

    /// Run database migrations
    #[cfg(feature = "sql-storage")]
    async fn run_migrations(&mut self) -> Result<()> {
        // Create memory_entries table
        sqlx::query(&format!(r#"
            CREATE TABLE IF NOT EXISTS {}.memory_entries (
                id UUID PRIMARY KEY,
                key VARCHAR NOT NULL,
                value TEXT NOT NULL,
                memory_type VARCHAR NOT NULL,
                importance REAL NOT NULL DEFAULT 0.5,
                confidence REAL NOT NULL DEFAULT 1.0,
                tags TEXT[] DEFAULT '{{}}',
                custom_fields JSONB DEFAULT '{{}}',
                embedding REAL[],
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                last_accessed TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                last_modified TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                access_count BIGINT NOT NULL DEFAULT 0,
                UNIQUE(key)
            )
        "#, self.config.schema))
        .execute(&self.pool)
        .await
        .map_err(|e| MemoryError::storage(format!("Failed to create memory_entries table: {}", e)))?;

        // Create analytics_events table
        sqlx::query(&format!(r#"
            CREATE TABLE IF NOT EXISTS {}.analytics_events (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                event_type VARCHAR NOT NULL,
                event_data JSONB NOT NULL,
                timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                user_context VARCHAR,
                session_id UUID
            )
        "#, self.config.schema))
        .execute(&self.pool)
        .await
        .map_err(|e| MemoryError::storage(format!("Failed to create analytics_events table: {}", e)))?;

        // Create analytics_insights table
        sqlx::query(&format!(r#"
            CREATE TABLE IF NOT EXISTS {}.analytics_insights (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                title VARCHAR NOT NULL,
                description TEXT NOT NULL,
                insight_type VARCHAR NOT NULL,
                priority VARCHAR NOT NULL,
                confidence REAL NOT NULL,
                metadata JSONB DEFAULT '{{}}',
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                expires_at TIMESTAMPTZ
            )
        "#, self.config.schema))
        .execute(&self.pool)
        .await
        .map_err(|e| MemoryError::storage(format!("Failed to create analytics_insights table: {}", e)))?;

        // Create indexes for performance (execute individually)
        let indexes = vec![
            format!("CREATE INDEX IF NOT EXISTS idx_memory_entries_key ON {}.memory_entries(key)", self.config.schema),
            format!("CREATE INDEX IF NOT EXISTS idx_memory_entries_type ON {}.memory_entries(memory_type)", self.config.schema),
            format!("CREATE INDEX IF NOT EXISTS idx_memory_entries_created ON {}.memory_entries(created_at)", self.config.schema),
            format!("CREATE INDEX IF NOT EXISTS idx_analytics_events_type ON {}.analytics_events(event_type)", self.config.schema),
            format!("CREATE INDEX IF NOT EXISTS idx_analytics_events_timestamp ON {}.analytics_events(timestamp)", self.config.schema),
            format!("CREATE INDEX IF NOT EXISTS idx_analytics_insights_type ON {}.analytics_insights(insight_type)", self.config.schema),
        ];

        for index_sql in indexes {
            sqlx::query(&index_sql)
                .execute(&self.pool)
                .await
                .map_err(|e| MemoryError::storage(format!("Failed to create index: {}", e)))?;
        }

        Ok(())
    }

    /// Store a memory entry in the database
    #[cfg(feature = "sql-storage")]
    pub async fn store_memory(&self, entry: &MemoryEntry) -> Result<()> {
        let start_time = std::time::Instant::now();

        let memory_type_str = format!("{:?}", entry.memory_type);
        let tags: Vec<String> = entry.metadata.tags.clone();
        let custom_fields = serde_json::to_value(&entry.metadata.custom_fields)
            .map_err(|e| MemoryError::storage(format!("Failed to serialize custom fields: {}", e)))?;
        let embedding: Option<Vec<f32>> = entry.embedding.clone();

        sqlx::query(&format!(r#"
            INSERT INTO {}.memory_entries
            (id, key, value, memory_type, importance, confidence, tags, custom_fields, embedding,
             created_at, last_accessed, last_modified, access_count)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13)
            ON CONFLICT (key) DO UPDATE SET
                value = EXCLUDED.value,
                memory_type = EXCLUDED.memory_type,
                importance = EXCLUDED.importance,
                confidence = EXCLUDED.confidence,
                tags = EXCLUDED.tags,
                custom_fields = EXCLUDED.custom_fields,
                embedding = EXCLUDED.embedding,
                last_modified = EXCLUDED.last_modified
        "#, self.config.schema))
        .bind(entry.metadata.id)
        .bind(&entry.key)
        .bind(&entry.value)
        .bind(memory_type_str)
        .bind(entry.metadata.importance)
        .bind(entry.metadata.confidence)
        .bind(&tags)
        .bind(custom_fields)
        .bind(embedding)
        .bind(entry.metadata.created_at)
        .bind(entry.metadata.last_accessed)
        .bind(entry.metadata.last_modified)
        .bind(entry.metadata.access_count as i64)
        .execute(&self.pool)
        .await
        .map_err(|e| MemoryError::storage(format!("Failed to store memory entry: {}", e)))?;

        self.update_metrics(start_time);
        Ok(())
    }

    /// Retrieve a memory entry from the database
    #[cfg(feature = "sql-storage")]
    pub async fn get_memory(&self, key: &str) -> Result<Option<MemoryEntry>> {
        let start_time = std::time::Instant::now();

        let row = sqlx::query(&format!(r#"
            SELECT id, key, value, memory_type, importance, confidence, tags, custom_fields, 
                   embedding, created_at, last_accessed, last_modified, access_count
            FROM {}.memory_entries 
            WHERE key = $1
        "#, self.config.schema))
        .bind(key)
        .fetch_optional(&self.pool)
        .await
        .map_err(|e| MemoryError::storage(format!("Failed to retrieve memory entry: {}", e)))?;

        self.update_metrics(start_time);

        if let Some(row) = row {
            let memory_type = match row.get::<String, _>("memory_type").as_str() {
                "ShortTerm" => MemoryType::ShortTerm,
                "LongTerm" => MemoryType::LongTerm,
                _ => MemoryType::ShortTerm,
            };

            let custom_fields: HashMap<String, String> = serde_json::from_value(
                row.get::<serde_json::Value, _>("custom_fields")
            ).unwrap_or_default();

            let metadata = MemoryMetadata {
                id: row.get("id"),
                created_at: row.get("created_at"),
                last_accessed: row.get("last_accessed"),
                last_modified: row.get("last_modified"),
                access_count: row.get::<i64, _>("access_count") as u64,
                tags: row.get("tags"),
                custom_fields,
                importance: row.get::<f32, _>("importance") as f64,
                confidence: row.get::<f32, _>("confidence") as f64,
            };

            let entry = MemoryEntry {
                key: row.get("key"),
                value: row.get("value"),
                memory_type,
                metadata,
                embedding: row.get("embedding"),
            };

            Ok(Some(entry))
        } else {
            Ok(None)
        }
    }

    /// Store an analytics event
    #[cfg(feature = "sql-storage")]
    pub async fn store_analytics_event(&self, event: &AnalyticsEvent) -> Result<()> {
        let start_time = std::time::Instant::now();

        let event_data_json = serde_json::to_value(&event.data)
            .map_err(|e| MemoryError::storage(format!("Failed to serialize event data: {}", e)))?;

        sqlx::query(&format!(r#"
            INSERT INTO {}.analytics_events (event_type, event_data, timestamp, user_context)
            VALUES ($1, $2, $3, $4)
        "#, self.config.schema))
        .bind(&event.event_type)
        .bind(event_data_json)
        .bind(event.timestamp)
        .bind(None::<String>) // user_context
        .execute(&self.pool)
        .await
        .map_err(|e| MemoryError::storage(format!("Failed to store analytics event: {}", e)))?;

        self.update_metrics(start_time);
        Ok(())
    }

    /// Health check for database connection
    #[cfg(feature = "sql-storage")]
    pub async fn health_check(&self) -> Result<()> {
        sqlx::query("SELECT 1")
            .execute(&self.pool)
            .await
            .map_err(|e| MemoryError::storage(format!("Database health check failed: {}", e)))?;
        Ok(())
    }

    /// Shutdown database connection
    pub async fn shutdown(&mut self) -> Result<()> {
        #[cfg(feature = "sql-storage")]
        {
            self.pool.close().await;
        }
        Ok(())
    }

    /// Get database metrics
    pub fn get_metrics(&self) -> DatabaseMetrics {
        self.metrics.lock().map(|m| m.clone()).unwrap_or_default()
    }

    #[cfg(feature = "sql-storage")]
    fn update_metrics(&self, start_time: std::time::Instant) {
        if let Ok(mut m) = self.metrics.lock() {
            m.queries_executed += 1;
            m.total_query_time_ms += start_time.elapsed().as_millis() as u64;
        }
    }

    fn extract_user_context(&self, _event: &AnalyticsEvent) -> Option<String> {
        // Extract user context from event data if available
        None
    }
}

#[cfg(not(feature = "sql-storage"))]
impl DatabaseClient {
    pub async fn new(_config: DatabaseConfig) -> Result<Self> {
        Err(MemoryError::configuration("SQL storage feature not enabled"))
    }

    pub async fn health_check(&self) -> Result<()> {
        Err(MemoryError::configuration("SQL storage feature not enabled"))
    }

    pub async fn shutdown(&mut self) -> Result<()> {
        Ok(())
    }

    pub fn get_metrics(&self) -> DatabaseMetrics {
        DatabaseMetrics::default()
    }
}

// Implement Storage trait for DatabaseClient
#[async_trait::async_trait]
impl Storage for DatabaseClient {
    async fn store(&self, entry: &MemoryEntry) -> Result<()> {
        #[cfg(feature = "sql-storage")]
        {
            self.store_memory(entry).await
        }
        #[cfg(not(feature = "sql-storage"))]
        {
            Err(MemoryError::configuration("SQL storage feature not enabled"))
        }
    }

    async fn retrieve(&self, key: &str) -> Result<Option<MemoryEntry>> {
        #[cfg(feature = "sql-storage")]
        {
            self.get_memory(key).await
        }
        #[cfg(not(feature = "sql-storage"))]
        {
            Err(MemoryError::configuration("SQL storage feature not enabled"))
        }
    }

    async fn search(&self, query: &str, limit: usize) -> Result<Vec<crate::memory::types::MemoryFragment>> {
        #[cfg(feature = "sql-storage")]
        {
            // Use prepared statement for better performance and security
            let sql = format!("SELECT key, value FROM {}.memory_entries WHERE value ILIKE $1 LIMIT $2", self.config.schema);
            let rows = sqlx::query(&sql)
                .bind(format!("%{}%", query))
                .bind(limit as i64)
                .fetch_all(&self.pool)
                .await
                .map_err(|e| MemoryError::storage(format!("Search failed: {}", e)))?;

            // Parallelize fragment creation for large result sets
            let fragments = if rows.len() > 100 {
                // Use parallel processing for large result sets
                use rayon::prelude::*;
                rows.into_par_iter().map(|row| crate::memory::types::MemoryFragment {
                    key: row.get("key"),
                    snippet: row.get::<String, _>("value").chars().take(100).collect(),
                    relevance_score: 1.0, // Default relevance score
                }).collect()
            } else {
                // Use sequential processing for small result sets
                rows.into_iter().map(|row| crate::memory::types::MemoryFragment {
                    key: row.get("key"),
                    snippet: row.get::<String, _>("value").chars().take(100).collect(),
                    relevance_score: 1.0, // Default relevance score
                }).collect()
            };

            Ok(fragments)
        }
        #[cfg(not(feature = "sql-storage"))]
        {
            Err(MemoryError::configuration("SQL storage feature not enabled"))
        }
    }

    async fn update(&self, key: &str, entry: &MemoryEntry) -> Result<()> {
        #[cfg(feature = "sql-storage")]
        {
            let mut updated = entry.clone();
            updated.key = key.to_string();
            self.store_memory(&updated).await
        }
        #[cfg(not(feature = "sql-storage"))]
        {
            Err(MemoryError::configuration("SQL storage feature not enabled"))
        }
    }

    async fn delete(&self, key: &str) -> Result<bool> {
        #[cfg(feature = "sql-storage")]
        {
            let result = sqlx::query(&format!("DELETE FROM {}.memory_entries WHERE key=$1", self.config.schema))
                .bind(key)
                .execute(&self.pool)
                .await
                .map_err(|e| MemoryError::storage(format!("Delete failed: {}", e)))?;
            Ok(result.rows_affected() > 0)
        }
        #[cfg(not(feature = "sql-storage"))]
        {
            Err(MemoryError::configuration("SQL storage feature not enabled"))
        }
    }

    async fn list_keys(&self) -> Result<Vec<String>> {
        #[cfg(feature = "sql-storage")]
        {
            let rows = sqlx::query(&format!("SELECT key FROM {}.memory_entries", self.config.schema))
                .fetch_all(&self.pool)
                .await
                .map_err(|e| MemoryError::storage(format!("List keys failed: {}", e)))?;
            Ok(rows.into_iter().map(|r| r.get("key")).collect())
        }
        #[cfg(not(feature = "sql-storage"))]
        {
            Err(MemoryError::configuration("SQL storage feature not enabled"))
        }
    }

    async fn count(&self) -> Result<usize> {
        #[cfg(feature = "sql-storage")]
        {
            let row = sqlx::query(&format!("SELECT COUNT(*) as count FROM {}.memory_entries", self.config.schema))
                .fetch_one(&self.pool)
                .await
                .map_err(|e| MemoryError::storage(format!("Count failed: {}", e)))?;
            Ok(row.get::<i64, _>("count") as usize)
        }
        #[cfg(not(feature = "sql-storage"))]
        {
            Err(MemoryError::configuration("SQL storage feature not enabled"))
        }
    }

    async fn clear(&self) -> Result<()> {
        #[cfg(feature = "sql-storage")]
        {
            sqlx::query(&format!("TRUNCATE TABLE {}.memory_entries", self.config.schema))
                .execute(&self.pool)
                .await
                .map_err(|e| MemoryError::storage(format!("Clear failed: {}", e)))?;
            Ok(())
        }
        #[cfg(not(feature = "sql-storage"))]
        {
            Err(MemoryError::configuration("SQL storage feature not enabled"))
        }
    }

    async fn exists(&self, key: &str) -> Result<bool> {
        #[cfg(feature = "sql-storage")]
        {
            let row = sqlx::query(&format!("SELECT EXISTS(SELECT 1 FROM {}.memory_entries WHERE key=$1) as exist", self.config.schema))
                .bind(key)
                .fetch_one(&self.pool)
                .await
                .map_err(|e| MemoryError::storage(format!("Exists query failed: {}", e)))?;
            Ok(row.get::<bool, _>("exist"))
        }
        #[cfg(not(feature = "sql-storage"))]
        {
            Err(MemoryError::configuration("SQL storage feature not enabled"))
        }
    }

    async fn stats(&self) -> Result<crate::memory::storage::StorageStats> {
        let count = self.count().await.unwrap_or(0);
        Ok(crate::memory::storage::StorageStats {
            total_entries: count,
            total_size_bytes: 0,
            average_entry_size: 0.0,
            storage_type: "database".to_string(),
            last_maintenance: None,
            fragmentation_ratio: 0.0,
        })
    }

    async fn maintenance(&self) -> Result<()> {
        Ok(())
    }

    async fn backup(&self, path: &str) -> Result<()> {
        #[cfg(feature = "sql-storage")]
        {
            let rows = sqlx::query(&format!("SELECT key, value FROM {}.memory_entries", self.config.schema))
                .fetch_all(&self.pool)
                .await
                .map_err(|e| MemoryError::storage(format!("Backup query failed: {}", e)))?;
            let pairs: Vec<(String, String)> = rows.into_iter().map(|r| (r.get("key"), r.get("value"))).collect();
            let json = serde_json::to_string(&pairs).map_err(|e| MemoryError::storage(e.to_string()))?;
            tokio::fs::write(path, json).await.map_err(|e| MemoryError::storage(e.to_string()))?;
            Ok(())
        }
        #[cfg(not(feature = "sql-storage"))]
        {
            Err(MemoryError::configuration("SQL storage feature not enabled"))
        }
    }

    async fn restore(&self, path: &str) -> Result<()> {
        #[cfg(feature = "sql-storage")]
        {
            let data = tokio::fs::read(path).await.map_err(|e| MemoryError::storage(e.to_string()))?;
            let rows: Vec<(String, String)> = serde_json::from_slice(&data).map_err(|e| MemoryError::storage(e.to_string()))?;
            for (key, value) in rows {
                let entry = MemoryEntry {
                    key,
                    value,
                    memory_type: MemoryType::ShortTerm,
                    metadata: MemoryMetadata::default(),
                    embedding: None,
                };
                let _ = self.store_memory(&entry).await;
            }
            Ok(())
        }
        #[cfg(not(feature = "sql-storage"))]
        {
            Err(MemoryError::configuration("SQL storage feature not enabled"))
        }
    }


}
