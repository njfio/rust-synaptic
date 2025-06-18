//! Comprehensive tests for enhanced error handling system

use synaptic::error::{MemoryError, MemoryErrorExt, Result};
use std::io;

#[tokio::test]
async fn test_error_creation_methods() {
    // Test all error creation methods
    let storage_error = MemoryError::storage("Storage failed");
    assert!(matches!(storage_error, MemoryError::Storage { .. }));
    assert_eq!(storage_error.to_string(), "Storage error: Storage failed");

    let analytics_error = MemoryError::analytics("Analytics computation failed");
    assert!(matches!(analytics_error, MemoryError::Analytics { .. }));
    assert_eq!(analytics_error.to_string(), "Analytics error: Analytics computation failed");

    let optimization_error = MemoryError::optimization("Optimization failed");
    assert!(matches!(optimization_error, MemoryError::Optimization { .. }));
    assert_eq!(optimization_error.to_string(), "Optimization error: Optimization failed");

    let lifecycle_error = MemoryError::lifecycle_management("Lifecycle operation failed");
    assert!(matches!(lifecycle_error, MemoryError::LifecycleManagement { .. }));
    assert_eq!(lifecycle_error.to_string(), "Lifecycle management error: Lifecycle operation failed");

    let search_error = MemoryError::search_engine("Search index corrupted");
    assert!(matches!(search_error, MemoryError::SearchEngine { .. }));
    assert_eq!(search_error.to_string(), "Search engine error: Search index corrupted");

    let summarization_error = MemoryError::summarization("Summarization failed");
    assert!(matches!(summarization_error, MemoryError::Summarization { .. }));
    assert_eq!(summarization_error.to_string(), "Summarization error: Summarization failed");

    let compression_error = MemoryError::compression("Compression failed");
    assert!(matches!(compression_error, MemoryError::Compression { .. }));
    assert_eq!(compression_error.to_string(), "Compression error: Compression failed");

    let index_error = MemoryError::index("Index corruption detected");
    assert!(matches!(index_error, MemoryError::Index { .. }));
    assert_eq!(index_error.to_string(), "Index error: Index corruption detected");

    let cache_error = MemoryError::cache("Cache miss");
    assert!(matches!(cache_error, MemoryError::Cache { .. }));
    assert_eq!(cache_error.to_string(), "Cache error: Cache miss");

    let transaction_error = MemoryError::transaction("Transaction rollback");
    assert!(matches!(transaction_error, MemoryError::Transaction { .. }));
    assert_eq!(transaction_error.to_string(), "Transaction error: Transaction rollback");

    let validation_error = MemoryError::validation("Invalid input");
    assert!(matches!(validation_error, MemoryError::Validation { .. }));
    assert_eq!(validation_error.to_string(), "Validation error: Invalid input");

    let timeout_error = MemoryError::timeout("search_operation");
    assert!(matches!(timeout_error, MemoryError::Timeout { .. }));
    assert_eq!(timeout_error.to_string(), "Operation timed out: search_operation");

    let resource_error = MemoryError::resource_exhausted("memory");
    assert!(matches!(resource_error, MemoryError::ResourceExhausted { .. }));
    assert_eq!(resource_error.to_string(), "Resource exhausted: memory");

    let auth_error = MemoryError::authentication("Invalid credentials");
    assert!(matches!(auth_error, MemoryError::Authentication { .. }));
    assert_eq!(auth_error.to_string(), "Authentication error: Invalid credentials");

    let authz_error = MemoryError::authorization("Access denied");
    assert!(matches!(authz_error, MemoryError::Authorization { .. }));
    assert_eq!(authz_error.to_string(), "Authorization error: Access denied");

    let rate_limit_error = MemoryError::rate_limit("Too many requests");
    assert!(matches!(rate_limit_error, MemoryError::RateLimit { .. }));
    assert_eq!(rate_limit_error.to_string(), "Rate limit exceeded: Too many requests");

    let processing_error = MemoryError::processing_error("Processing failed");
    assert!(matches!(processing_error, MemoryError::ProcessingError(_)));
    assert_eq!(processing_error.to_string(), "Processing error: Processing failed");

    let embedding_error = MemoryError::embedding("Embedding generation failed");
    assert!(matches!(embedding_error, MemoryError::Embedding { .. }));
    assert_eq!(embedding_error.to_string(), "Embedding error: Embedding generation failed");

    let kg_error = MemoryError::knowledge_graph("Graph traversal failed");
    assert!(matches!(kg_error, MemoryError::KnowledgeGraph { .. }));
    assert_eq!(kg_error.to_string(), "Knowledge graph error: Graph traversal failed");

    let temporal_error = MemoryError::temporal_tracking("Temporal analysis failed");
    assert!(matches!(temporal_error, MemoryError::TemporalTracking { .. }));
    assert_eq!(temporal_error.to_string(), "Temporal tracking error: Temporal analysis failed");

    let integration_error = MemoryError::integration("External service unavailable");
    assert!(matches!(integration_error, MemoryError::Integration { .. }));
    assert_eq!(integration_error.to_string(), "Integration error: External service unavailable");

    let multimodal_error = MemoryError::multi_modal("Image processing failed");
    assert!(matches!(multimodal_error, MemoryError::MultiModal { .. }));
    assert_eq!(multimodal_error.to_string(), "Multi-modal processing error: Image processing failed");

    let cross_platform_error = MemoryError::cross_platform("Platform incompatibility");
    assert!(matches!(cross_platform_error, MemoryError::CrossPlatform { .. }));
    assert_eq!(cross_platform_error.to_string(), "Cross-platform error: Platform incompatibility");

    let document_error = MemoryError::document_processing("PDF parsing failed");
    assert!(matches!(document_error, MemoryError::DocumentProcessing { .. }));
    assert_eq!(document_error.to_string(), "Document processing error: PDF parsing failed");
}

#[tokio::test]
async fn test_error_context_extension() {
    // Test context extension methods
    let io_error = io::Error::new(io::ErrorKind::NotFound, "File not found");
    let result: Result<()> = Err(io_error).storage_context("Loading configuration");
    
    match result {
        Err(MemoryError::Storage { message }) => {
            assert!(message.contains("Loading configuration"));
            assert!(message.contains("File not found"));
        }
        _ => assert!(false, "Expected storage error with context"),
    }

    // Test analytics context
    let generic_error = "Computation failed";
    let result: Result<()> = Err(generic_error).analytics_context("Running ML model");
    
    match result {
        Err(MemoryError::Analytics { message }) => {
            assert!(message.contains("Running ML model"));
            assert!(message.contains("Computation failed"));
        }
        _ => panic!("Expected analytics error with context"),
    }

    // Test optimization context
    let result: Result<()> = Err("Index rebuild failed").optimization_context("Memory optimization");
    
    match result {
        Err(MemoryError::Optimization { message }) => {
            assert!(message.contains("Memory optimization"));
            assert!(message.contains("Index rebuild failed"));
        }
        _ => panic!("Expected optimization error with context"),
    }

    // Test search context
    let result: Result<()> = Err("Query parsing failed").search_context("Semantic search");
    
    match result {
        Err(MemoryError::SearchEngine { message }) => {
            assert!(message.contains("Semantic search"));
            assert!(message.contains("Query parsing failed"));
        }
        _ => panic!("Expected search engine error with context"),
    }

    // Test summarization context
    let result: Result<()> = Err("Content too short").summarization_context("Generating summary");
    
    match result {
        Err(MemoryError::Summarization { message }) => {
            assert!(message.contains("Generating summary"));
            assert!(message.contains("Content too short"));
        }
        _ => panic!("Expected summarization error with context"),
    }

    // Test compression context
    let result: Result<()> = Err("Algorithm not supported").compression_context("Data compression");
    
    match result {
        Err(MemoryError::Compression { message }) => {
            assert!(message.contains("Data compression"));
            assert!(message.contains("Algorithm not supported"));
        }
        _ => assert!(false, "Expected compression error with context"),
    }

    // Test validation context
    let result: Result<()> = Err("Schema mismatch").validation_context("Input validation");
    
    match result {
        Err(MemoryError::Validation { message }) => {
            assert!(message.contains("Input validation"));
            assert!(message.contains("Schema mismatch"));
        }
        _ => assert!(false, "Expected validation error with context"),
    }

    // Test processing context
    let result: Result<()> = Err("Pipeline failed").processing_context("Data processing");
    
    match result {
        Err(MemoryError::ProcessingError(message)) => {
            assert!(message.contains("Data processing"));
            assert!(message.contains("Pipeline failed"));
        }
        _ => panic!("Expected processing error with context"),
    }
}

#[tokio::test]
async fn test_error_type_checking() {
    let not_found_error = MemoryError::NotFound { key: "test_key".to_string() };
    assert!(not_found_error.is_not_found());
    assert!(!not_found_error.is_storage_error());
    assert!(!not_found_error.is_serialization_error());

    let storage_error = MemoryError::storage("test");
    assert!(!storage_error.is_not_found());
    assert!(storage_error.is_storage_error());
    assert!(!storage_error.is_serialization_error());

    let serialization_error = MemoryError::Serialization(serde_json::Error::io(
        io::Error::new(io::ErrorKind::InvalidData, "test")
    ));
    assert!(!serialization_error.is_not_found());
    assert!(!serialization_error.is_storage_error());
    assert!(serialization_error.is_serialization_error());
}

#[tokio::test]
async fn test_error_conversion_from_anyhow() {
    let anyhow_error = anyhow::anyhow!("Something went wrong");
    let memory_error: MemoryError = anyhow_error.into();
    
    match memory_error {
        MemoryError::Unexpected { message } => {
            assert_eq!(message, "Something went wrong");
        }
        _ => panic!("Expected unexpected error"),
    }
}

#[tokio::test]
async fn test_error_chaining() {
    // Test error chaining with multiple context layers
    let base_error = io::Error::new(io::ErrorKind::PermissionDenied, "Access denied");
    let storage_result: Result<()> = Err(base_error).storage_context("Reading file");

    // Test that storage context works
    match storage_result {
        Err(MemoryError::Storage { message }) => {
            assert!(message.contains("Reading file"));
            assert!(message.contains("Access denied"));
        }
        _ => assert!(false, "Expected storage error"),
    }

    // Test validation context
    let validation_result: Result<()> = Err(MemoryError::validation("Invalid format"))
        .validation_context("Processing input");

    match validation_result {
        Err(MemoryError::Validation { message }) => {
            assert!(message.contains("Processing input"));
            assert!(message.contains("Invalid format"));
        }
        _ => panic!("Expected validation error"),
    }
}

#[tokio::test]
async fn test_comprehensive_error_scenarios() {
    // Test timeout scenario
    let timeout_result: Result<String> = Err(MemoryError::timeout("database_query"));
    assert!(timeout_result.is_err());
    
    // Test resource exhaustion scenario
    let resource_result: Result<()> = Err(MemoryError::resource_exhausted("disk_space"));
    assert!(resource_result.is_err());
    
    // Test authentication scenario
    let auth_result: Result<()> = Err(MemoryError::authentication("Token expired"));
    assert!(auth_result.is_err());
    
    // Test rate limiting scenario
    let rate_limit_result: Result<()> = Err(MemoryError::rate_limit("API quota exceeded"));
    assert!(rate_limit_result.is_err());
    
    // Test all scenarios return appropriate error types
    match timeout_result {
        Err(MemoryError::Timeout { operation }) => assert_eq!(operation, "database_query"),
        _ => panic!("Expected timeout error"),
    }
    
    match resource_result {
        Err(MemoryError::ResourceExhausted { resource }) => assert_eq!(resource, "disk_space"),
        _ => panic!("Expected resource exhausted error"),
    }
    
    match auth_result {
        Err(MemoryError::Authentication { message }) => assert_eq!(message, "Token expired"),
        _ => panic!("Expected authentication error"),
    }
    
    match rate_limit_result {
        Err(MemoryError::RateLimit { message }) => assert_eq!(message, "API quota exceeded"),
        _ => panic!("Expected rate limit error"),
    }
}
