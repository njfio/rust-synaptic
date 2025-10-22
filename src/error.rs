//! Error types for the AI agent memory system

use thiserror::Error;

/// Result type alias for memory operations
pub type Result<T> = std::result::Result<T, MemoryError>;

/// Comprehensive error types for memory operations
#[derive(Error, Debug)]
pub enum MemoryError {
    /// Storage-related errors
    #[error("Storage error: {message}")]
    Storage { message: String },

    #[error("Invalid configuration: {message}")]
    InvalidConfiguration { message: String },

    #[error("Invalid input: {message}")]
    InvalidInput { message: String },

    /// Serialization/deserialization errors
    #[error("Serialization error: {0}")]
    Serialization(#[from] serde_json::Error),

    /// Binary serialization errors
    #[error("Binary serialization error: {0}")]
    BinarySerialization(#[from] bincode::Error),

    /// I/O errors
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    /// Database errors
    #[cfg(feature = "sql-storage")]
    #[error("Database error: {0}")]
    Database(#[from] sqlx::Error),

    /// Sled database errors
    #[error("Sled database error: {0}")]
    Sled(#[from] sled::Error),

    /// Memory not found
    #[error("Memory entry not found: {key}")]
    NotFound { key: String },

    /// Invalid memory type
    #[error("Invalid memory type: {memory_type}")]
    InvalidMemoryType { memory_type: String },

    /// Checkpoint errors
    #[error("Checkpoint error: {message}")]
    Checkpoint { message: String },

    /// Configuration errors
    #[error("Configuration error: {message}")]
    Configuration { message: String },

    /// Memory limit exceeded
    #[error("Memory limit exceeded: {limit} entries")]
    MemoryLimitExceeded { limit: usize },

    /// Invalid UUID
    #[error("Invalid UUID: {0}")]
    InvalidUuid(#[from] uuid::Error),

    /// Concurrency errors
    #[error("Concurrency error: {message}")]
    Concurrency { message: String },

    /// Vector operation errors
    #[error("Vector operation error: {message}")]
    VectorOperation { message: String },

    /// Generic error for unexpected situations
    #[error("Unexpected error: {message}")]
    Unexpected { message: String },

    /// Consensus-related errors
    #[error("Consensus error: {message}")]
    ConsensusError { message: String },

    /// Network-related errors
    #[error("Network error: {message}")]
    NetworkError { message: String },

    /// Distributed system errors
    #[error("Distributed system error: {message}")]
    DistributedError { message: String },

    /// Serialization errors for distributed operations
    #[error("Serialization error: {message}")]
    SerializationError { message: String },

    /// Security and encryption errors
    #[error("Encryption error: {message}")]
    Encryption { message: String },

    /// Privacy and differential privacy errors
    #[error("Privacy error: {message}")]
    Privacy { message: String },

    /// Key management errors
    #[error("Key management error: {message}")]
    KeyManagement { message: String },

    /// Access control errors
    #[error("Access denied: {message}")]
    AccessDenied { message: String },

    /// Analytics and machine learning errors
    #[error("Analytics error: {message}")]
    Analytics { message: String },

    /// Optimization errors
    #[error("Optimization error: {message}")]
    Optimization { message: String },

    /// Lifecycle management errors
    #[error("Lifecycle management error: {message}")]
    LifecycleManagement { message: String },

    /// Search engine errors
    #[error("Search engine error: {message}")]
    SearchEngine { message: String },

    /// Summarization errors
    #[error("Summarization error: {message}")]
    Summarization { message: String },

    /// Compression/decompression errors
    #[error("Compression error: {message}")]
    Compression { message: String },

    /// Index management errors
    #[error("Index error: {message}")]
    Index { message: String },

    /// Cache errors
    #[error("Cache error: {message}")]
    Cache { message: String },

    /// Transaction errors
    #[error("Transaction error: {message}")]
    Transaction { message: String },

    /// Validation errors
    #[error("Validation error: {message}")]
    Validation { message: String },

    /// Timeout errors
    #[error("Operation timed out: {operation}")]
    Timeout { operation: String },

    /// Resource exhaustion errors
    #[error("Resource exhausted: {resource}")]
    ResourceExhausted { resource: String },

    /// Authentication errors
    #[error("Authentication error: {message}")]
    Authentication { message: String },

    /// Authorization errors
    #[error("Authorization error: {message}")]
    Authorization { message: String },

    /// Rate limiting errors
    #[error("Rate limit exceeded: {message}")]
    RateLimit { message: String },

    /// Processing errors
    #[error("Processing error: {0}")]
    ProcessingError(String),

    /// Embedding errors
    #[error("Embedding error: {message}")]
    Embedding { message: String },

    /// Knowledge graph errors
    #[error("Knowledge graph error: {message}")]
    KnowledgeGraph { message: String },

    /// Temporal tracking errors
    #[error("Temporal tracking error: {message}")]
    TemporalTracking { message: String },

    /// Integration errors
    #[error("Integration error: {message}")]
    Integration { message: String },

    /// Multi-modal processing errors
    #[error("Multi-modal processing error: {message}")]
    MultiModal { message: String },

    /// Cross-platform errors
    #[error("Cross-platform error: {message}")]
    CrossPlatform { message: String },

    /// Document processing errors
    #[error("Document processing error: {message}")]
    DocumentProcessing { message: String },

    /// Query parsing and execution errors
    #[error("Invalid query: {message}")]
    InvalidQuery { message: String },
}

impl MemoryError {
    /// Create a storage error
    pub fn storage<S: Into<String>>(message: S) -> Self {
        Self::Storage {
            message: message.into(),
        }
    }

    /// Create a checkpoint error
    pub fn checkpoint<S: Into<String>>(message: S) -> Self {
        Self::Checkpoint {
            message: message.into(),
        }
    }

    /// Create a configuration error
    pub fn configuration<S: Into<String>>(message: S) -> Self {
        Self::Configuration {
            message: message.into(),
        }
    }

    /// Create a concurrency error
    pub fn concurrency<S: Into<String>>(message: S) -> Self {
        Self::Concurrency {
            message: message.into(),
        }
    }

    /// Create a vector operation error
    pub fn vector_operation<S: Into<String>>(message: S) -> Self {
        Self::VectorOperation {
            message: message.into(),
        }
    }

    /// Create an unexpected error
    pub fn unexpected<S: Into<String>>(message: S) -> Self {
        Self::Unexpected {
            message: message.into(),
        }
    }

    /// Create an encryption error
    pub fn encryption<S: Into<String>>(message: S) -> Self {
        Self::Encryption {
            message: message.into(),
        }
    }

    /// Create a privacy error
    pub fn privacy<S: Into<String>>(message: S) -> Self {
        Self::Privacy {
            message: message.into(),
        }
    }

    /// Create a key management error
    pub fn key_management<S: Into<String>>(message: S) -> Self {
        Self::KeyManagement {
            message: message.into(),
        }
    }

    /// Create an access denied error
    pub fn access_denied<S: Into<String>>(message: S) -> Self {
        Self::AccessDenied {
            message: message.into(),
        }
    }

    /// Create an analytics error
    pub fn analytics<S: Into<String>>(message: S) -> Self {
        Self::Analytics {
            message: message.into(),
        }
    }

    /// Create an optimization error
    pub fn optimization<S: Into<String>>(message: S) -> Self {
        Self::Optimization {
            message: message.into(),
        }
    }

    /// Create a lifecycle management error
    pub fn lifecycle_management<S: Into<String>>(message: S) -> Self {
        Self::LifecycleManagement {
            message: message.into(),
        }
    }

    /// Create a search engine error
    pub fn search_engine<S: Into<String>>(message: S) -> Self {
        Self::SearchEngine {
            message: message.into(),
        }
    }

    /// Create a summarization error
    pub fn summarization<S: Into<String>>(message: S) -> Self {
        Self::Summarization {
            message: message.into(),
        }
    }

    /// Create a compression error
    pub fn compression<S: Into<String>>(message: S) -> Self {
        Self::Compression {
            message: message.into(),
        }
    }

    /// Create an index error
    pub fn index<S: Into<String>>(message: S) -> Self {
        Self::Index {
            message: message.into(),
        }
    }

    /// Create a cache error
    pub fn cache<S: Into<String>>(message: S) -> Self {
        Self::Cache {
            message: message.into(),
        }
    }

    /// Create a transaction error
    pub fn transaction<S: Into<String>>(message: S) -> Self {
        Self::Transaction {
            message: message.into(),
        }
    }

    /// Create a validation error
    pub fn validation<S: Into<String>>(message: S) -> Self {
        Self::Validation {
            message: message.into(),
        }
    }

    /// Create a timeout error
    pub fn timeout<S: Into<String>>(operation: S) -> Self {
        Self::Timeout {
            operation: operation.into(),
        }
    }

    /// Create a resource exhausted error
    pub fn resource_exhausted<S: Into<String>>(resource: S) -> Self {
        Self::ResourceExhausted {
            resource: resource.into(),
        }
    }

    /// Create an authentication error
    pub fn authentication<S: Into<String>>(message: S) -> Self {
        Self::Authentication {
            message: message.into(),
        }
    }

    /// Create an authorization error
    pub fn authorization<S: Into<String>>(message: S) -> Self {
        Self::Authorization {
            message: message.into(),
        }
    }

    /// Create a rate limit error
    pub fn rate_limit<S: Into<String>>(message: S) -> Self {
        Self::RateLimit {
            message: message.into(),
        }
    }

    /// Create a processing error
    pub fn processing_error<S: Into<String>>(message: S) -> Self {
        Self::ProcessingError(message.into())
    }

    /// Create an embedding error
    pub fn embedding<S: Into<String>>(message: S) -> Self {
        Self::Embedding {
            message: message.into(),
        }
    }

    /// Create a knowledge graph error
    pub fn knowledge_graph<S: Into<String>>(message: S) -> Self {
        Self::KnowledgeGraph {
            message: message.into(),
        }
    }

    /// Create a temporal tracking error
    pub fn temporal_tracking<S: Into<String>>(message: S) -> Self {
        Self::TemporalTracking {
            message: message.into(),
        }
    }

    /// Create an integration error
    pub fn integration<S: Into<String>>(message: S) -> Self {
        Self::Integration {
            message: message.into(),
        }
    }

    /// Create a multi-modal error
    pub fn multi_modal<S: Into<String>>(message: S) -> Self {
        Self::MultiModal {
            message: message.into(),
        }
    }

    /// Create a cross-platform error
    pub fn cross_platform<S: Into<String>>(message: S) -> Self {
        Self::CrossPlatform {
            message: message.into(),
        }
    }

    /// Create a document processing error
    pub fn document_processing<S: Into<String>>(message: S) -> Self {
        Self::DocumentProcessing {
            message: message.into(),
        }
    }

    /// Create an invalid query error
    pub fn invalid_query<S: Into<String>>(message: S) -> Self {
        Self::InvalidQuery {
            message: message.into(),
        }
    }

    /// Check if this is a not found error
    pub fn is_not_found(&self) -> bool {
        matches!(self, Self::NotFound { .. })
    }

    /// Check if this is a storage error
    pub fn is_storage_error(&self) -> bool {
        matches!(self, Self::Storage { .. })
    }

    /// Check if this is a serialization error
    pub fn is_serialization_error(&self) -> bool {
        matches!(
            self,
            Self::Serialization(_) | Self::BinarySerialization(_)
        )
    }
}

/// Convert from anyhow::Error for compatibility
impl From<anyhow::Error> for MemoryError {
    fn from(err: anyhow::Error) -> Self {
        Self::unexpected(err.to_string())
    }
}

/// Convert from rustyline::error::ReadlineError for CLI operations
impl From<rustyline::error::ReadlineError> for MemoryError {
    fn from(err: rustyline::error::ReadlineError) -> Self {
        Self::unexpected(format!("Readline error: {}", err))
    }
}

/// Convert from regex::Error for pattern matching operations
impl From<regex::Error> for MemoryError {
    fn from(err: regex::Error) -> Self {
        Self::validation(format!("Regex error: {}", err))
    }
}

/// Convert from toml::de::Error for TOML parsing operations
impl From<toml::de::Error> for MemoryError {
    fn from(err: toml::de::Error) -> Self {
        Self::configuration(format!("TOML parsing error: {}", err))
    }
}

/// Convert from toml::ser::Error for TOML serialization operations
impl From<toml::ser::Error> for MemoryError {
    fn from(err: toml::ser::Error) -> Self {
        Self::configuration(format!("TOML serialization error: {}", err))
    }
}

/// Convert from serde_yaml::Error for YAML operations
impl From<serde_yaml::Error> for MemoryError {
    fn from(err: serde_yaml::Error) -> Self {
        Self::configuration(format!("YAML error: {}", err))
    }
}

/// Helper trait for converting results with context
pub trait MemoryErrorExt<T> {
    /// Convert to a storage error with context
    fn storage_context<S: Into<String>>(self, context: S) -> Result<T>;

    /// Convert to a checkpoint error with context
    fn checkpoint_context<S: Into<String>>(self, context: S) -> Result<T>;

    /// Convert to an analytics error with context
    fn analytics_context<S: Into<String>>(self, context: S) -> Result<T>;

    /// Convert to an optimization error with context
    fn optimization_context<S: Into<String>>(self, context: S) -> Result<T>;

    /// Convert to a search engine error with context
    fn search_context<S: Into<String>>(self, context: S) -> Result<T>;

    /// Convert to a summarization error with context
    fn summarization_context<S: Into<String>>(self, context: S) -> Result<T>;

    /// Convert to a compression error with context
    fn compression_context<S: Into<String>>(self, context: S) -> Result<T>;

    /// Convert to a validation error with context
    fn validation_context<S: Into<String>>(self, context: S) -> Result<T>;

    /// Convert to a processing error with context
    fn processing_context<S: Into<String>>(self, context: S) -> Result<T>;

    /// Convert to a multi-modal error with context
    fn multimodal_context<S: Into<String>>(self, context: S) -> Result<T>;

    /// Convert to a cross-platform error with context
    fn cross_platform_context<S: Into<String>>(self, context: S) -> Result<T>;

    /// Convert to a key management error with context
    fn key_management_context<S: Into<String>>(self, context: S) -> Result<T>;
}

impl<T, E> MemoryErrorExt<T> for std::result::Result<T, E>
where
    E: std::fmt::Display,
{
    fn storage_context<S: Into<String>>(self, context: S) -> Result<T> {
        self.map_err(|e| MemoryError::storage(format!("{}: {}", context.into(), e)))
    }

    fn checkpoint_context<S: Into<String>>(self, context: S) -> Result<T> {
        self.map_err(|e| MemoryError::checkpoint(format!("{}: {}", context.into(), e)))
    }

    fn analytics_context<S: Into<String>>(self, context: S) -> Result<T> {
        self.map_err(|e| MemoryError::analytics(format!("{}: {}", context.into(), e)))
    }

    fn optimization_context<S: Into<String>>(self, context: S) -> Result<T> {
        self.map_err(|e| MemoryError::optimization(format!("{}: {}", context.into(), e)))
    }

    fn search_context<S: Into<String>>(self, context: S) -> Result<T> {
        self.map_err(|e| MemoryError::search_engine(format!("{}: {}", context.into(), e)))
    }

    fn summarization_context<S: Into<String>>(self, context: S) -> Result<T> {
        self.map_err(|e| MemoryError::summarization(format!("{}: {}", context.into(), e)))
    }

    fn compression_context<S: Into<String>>(self, context: S) -> Result<T> {
        self.map_err(|e| MemoryError::compression(format!("{}: {}", context.into(), e)))
    }

    fn validation_context<S: Into<String>>(self, context: S) -> Result<T> {
        self.map_err(|e| MemoryError::validation(format!("{}: {}", context.into(), e)))
    }

    fn processing_context<S: Into<String>>(self, context: S) -> Result<T> {
        self.map_err(|e| MemoryError::processing_error(format!("{}: {}", context.into(), e)))
    }

    /// Convert to a multi-modal error with context
    fn multimodal_context<S: Into<String>>(self, context: S) -> Result<T> {
        self.map_err(|e| MemoryError::multi_modal(format!("{}: {}", context.into(), e)))
    }

    /// Convert to a cross-platform error with context
    fn cross_platform_context<S: Into<String>>(self, context: S) -> Result<T> {
        self.map_err(|e| MemoryError::cross_platform(format!("{}: {}", context.into(), e)))
    }

    /// Convert to a key management error with context
    fn key_management_context<S: Into<String>>(self, context: S) -> Result<T> {
        self.map_err(|e| MemoryError::key_management(format!("{}: {}", context.into(), e)))
    }
}

// Additional error conversions for external integrations

#[cfg(feature = "visualization")]
impl<T> From<plotters::drawing::DrawingAreaErrorKind<T>> for MemoryError
where
    T: std::fmt::Debug + Send + Sync + 'static + std::error::Error
{
    fn from(err: plotters::drawing::DrawingAreaErrorKind<T>) -> Self {
        MemoryError::storage(format!("Visualization error: {:?}", err))
    }
}

#[cfg(feature = "ml-models")]
impl From<candle_core::Error> for MemoryError {
    fn from(err: candle_core::Error) -> Self {
        MemoryError::storage(format!("ML model error: {}", err))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_storage_error_creation() {
        let err = MemoryError::storage("test error");
        assert!(matches!(err, MemoryError::Storage { .. }));
        assert_eq!(err.to_string(), "Storage error: test error");
    }

    #[test]
    fn test_checkpoint_error_creation() {
        let err = MemoryError::checkpoint("checkpoint failed");
        assert!(matches!(err, MemoryError::Checkpoint { .. }));
        assert_eq!(err.to_string(), "Checkpoint error: checkpoint failed");
    }

    #[test]
    fn test_configuration_error_creation() {
        let err = MemoryError::configuration("invalid config");
        assert!(matches!(err, MemoryError::Configuration { .. }));
        assert_eq!(err.to_string(), "Configuration error: invalid config");
    }

    #[test]
    fn test_concurrency_error_creation() {
        let err = MemoryError::concurrency("lock contention");
        assert!(matches!(err, MemoryError::Concurrency { .. }));
        assert_eq!(err.to_string(), "Concurrency error: lock contention");
    }

    #[test]
    fn test_vector_operation_error_creation() {
        let err = MemoryError::vector_operation("dimension mismatch");
        assert!(matches!(err, MemoryError::VectorOperation { .. }));
        assert_eq!(err.to_string(), "Vector operation error: dimension mismatch");
    }

    #[test]
    fn test_unexpected_error_creation() {
        let err = MemoryError::unexpected("something went wrong");
        assert!(matches!(err, MemoryError::Unexpected { .. }));
        assert_eq!(err.to_string(), "Unexpected error: something went wrong");
    }

    #[test]
    fn test_not_found_error() {
        let err = MemoryError::NotFound {
            key: "test_key".to_string(),
        };
        assert_eq!(err.to_string(), "Memory entry not found: test_key");
    }

    #[test]
    fn test_invalid_memory_type_error() {
        let err = MemoryError::InvalidMemoryType {
            memory_type: "invalid".to_string(),
        };
        assert_eq!(err.to_string(), "Invalid memory type: invalid");
    }

    #[test]
    fn test_memory_limit_exceeded_error() {
        let err = MemoryError::MemoryLimitExceeded { limit: 1000 };
        assert_eq!(err.to_string(), "Memory limit exceeded: 1000 entries");
    }

    #[test]
    fn test_io_error_conversion() {
        let io_err = std::io::Error::new(std::io::ErrorKind::NotFound, "file not found");
        let mem_err: MemoryError = io_err.into();
        assert!(matches!(mem_err, MemoryError::Io(_)));
    }

    #[test]
    fn test_serde_json_error_conversion() {
        let json_str = "{invalid json}";
        let result: Result<serde_json::Value, _> = serde_json::from_str(json_str);
        if let Err(json_err) = result {
            let mem_err: MemoryError = json_err.into();
            assert!(matches!(mem_err, MemoryError::Serialization(_)));
        }
    }

    #[test]
    fn test_uuid_error_conversion() {
        let result = uuid::Uuid::parse_str("invalid-uuid");
        if let Err(uuid_err) = result {
            let mem_err: MemoryError = uuid_err.into();
            assert!(matches!(mem_err, MemoryError::InvalidUuid(_)));
        }
    }

    #[test]
    fn test_sled_error_conversion() {
        // Create a sled error by trying to use an invalid path
        let result = sled::Config::new()
            .path("\0invalid")
            .open();
        if let Err(sled_err) = result {
            let mem_err: MemoryError = sled_err.into();
            assert!(matches!(mem_err, MemoryError::Sled(_)));
        }
    }

    #[test]
    fn test_error_helper_with_string() {
        let err = MemoryError::storage(String::from("owned string"));
        assert_eq!(err.to_string(), "Storage error: owned string");
    }

    #[test]
    fn test_error_helper_with_str() {
        let err = MemoryError::storage("borrowed str");
        assert_eq!(err.to_string(), "Storage error: borrowed str");
    }

    #[test]
    fn test_invalid_configuration_error() {
        let err = MemoryError::InvalidConfiguration {
            message: "missing required field".to_string(),
        };
        assert_eq!(err.to_string(), "Invalid configuration: missing required field");
    }

    #[test]
    fn test_invalid_input_error() {
        let err = MemoryError::InvalidInput {
            message: "value out of range".to_string(),
        };
        assert_eq!(err.to_string(), "Invalid input: value out of range");
    }

    #[test]
    fn test_encryption_error() {
        let err = MemoryError::Encryption {
            message: "key derivation failed".to_string(),
        };
        assert_eq!(err.to_string(), "Encryption error: key derivation failed");
    }

    #[test]
    fn test_privacy_error() {
        let err = MemoryError::Privacy {
            message: "differential privacy violation".to_string(),
        };
        assert_eq!(err.to_string(), "Privacy error: differential privacy violation");
    }

    #[test]
    fn test_consensus_error() {
        let err = MemoryError::ConsensusError {
            message: "quorum not reached".to_string(),
        };
        assert_eq!(err.to_string(), "Consensus error: quorum not reached");
    }

    #[test]
    fn test_network_error() {
        let err = MemoryError::NetworkError {
            message: "connection timeout".to_string(),
        };
        assert_eq!(err.to_string(), "Network error: connection timeout");
    }

    #[test]
    fn test_distributed_error() {
        let err = MemoryError::DistributedError {
            message: "shard unavailable".to_string(),
        };
        assert_eq!(err.to_string(), "Distributed system error: shard unavailable");
    }

    #[test]
    fn test_serialization_error() {
        let err = MemoryError::SerializationError {
            message: "failed to encode".to_string(),
        };
        assert_eq!(err.to_string(), "Serialization error: failed to encode");
    }

    #[test]
    fn test_result_type_alias() {
        // Test that Result<T> is properly aliased
        let ok_result: Result<i32> = Ok(42);
        assert_eq!(ok_result.unwrap(), 42);

        let err_result: Result<i32> = Err(MemoryError::storage("test"));
        assert!(err_result.is_err());
    }

    #[test]
    fn test_error_is_send_sync() {
        // Verify error implements Send + Sync for thread safety
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<MemoryError>();
    }

    #[test]
    fn test_error_debug_format() {
        let err = MemoryError::storage("test");
        let debug_str = format!("{:?}", err);
        assert!(debug_str.contains("Storage"));
        assert!(debug_str.contains("test"));
    }

    #[test]
    fn test_invalid_query_error() {
        let err = MemoryError::InvalidQuery {
            message: "syntax error".to_string(),
        };
        assert_eq!(err.to_string(), "Invalid query: syntax error");
    }
}

