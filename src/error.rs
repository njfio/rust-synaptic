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

/// Helper trait for converting results
pub trait MemoryErrorExt<T> {
    /// Convert to a storage error with context
    fn storage_context<S: Into<String>>(self, context: S) -> Result<T>;
    
    /// Convert to a checkpoint error with context
    fn checkpoint_context<S: Into<String>>(self, context: S) -> Result<T>;
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
}
