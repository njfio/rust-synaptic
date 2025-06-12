//! # Phase 5B: Advanced Document Processing Tests
//!
//! Comprehensive tests for Phase 5B document and data processing capabilities.

use synaptic::phase5b_basic::*;
use std::fs;
use tempfile::TempDir;

/// Test basic document manager creation and configuration
#[test]
fn test_basic_document_manager_creation() {
    let adapter = Box::new(BasicMemoryDocumentDataAdapter::new());
    let _manager = BasicDocumentDataManager::new(adapter);
    
    // Test with custom config
    let config = DocumentDataConfig {
        max_file_size: 50 * 1024 * 1024, // 50MB
        enable_content_extraction: true,
        enable_metadata_analysis: true,
        enable_batch_processing: true,
        max_batch_size: 50,
        supported_extensions: vec!["txt".to_string(), "md".to_string()],
    };
    
    let adapter2 = Box::new(BasicMemoryDocumentDataAdapter::new());
    let _manager2 = BasicDocumentDataManager::with_config(adapter2, config);
}

/// Test memory adapter functionality
#[test]
fn test_memory_adapter() {
    let mut adapter = BasicMemoryDocumentDataAdapter::new();
    
    // Test initial stats
    let stats = adapter.get_stats().unwrap();
    assert_eq!(stats.total_memories, 0);
    assert_eq!(stats.total_size, 0);
    
    // Create a test memory
    let memory = create_test_memory("test_doc", "Test content", ContentType::Document {
        format: "PlainText".to_string(),
        language: Some("en".to_string()),
    });
    
    // Store memory
    adapter.store_memory(&memory).unwrap();
    
    // Test retrieval
    let retrieved = adapter.get_memory(&memory.id).unwrap();
    assert!(retrieved.is_some());
    assert_eq!(retrieved.unwrap().id, memory.id);
    
    // Test stats after storage
    let stats = adapter.get_stats().unwrap();
    assert_eq!(stats.total_memories, 1);
    assert!(stats.total_size > 0);
    
    // Test search by type
    let results = adapter.search_by_type(&memory.content_type).unwrap();
    assert_eq!(results.len(), 1);
    
    // Test get all memories
    let all_memories = adapter.get_all_memories().unwrap();
    assert_eq!(all_memories.len(), 1);
    
    // Test deletion
    let deleted = adapter.delete_memory(&memory.id).unwrap();
    assert!(deleted);
    
    // Verify deletion
    let retrieved_after_delete = adapter.get_memory(&memory.id).unwrap();
    assert!(retrieved_after_delete.is_none());
}

/// Test document processing with various formats
#[test]
fn test_document_processing() {
    let temp_dir = TempDir::new().unwrap();
    let adapter = Box::new(BasicMemoryDocumentDataAdapter::new());
    let mut manager = BasicDocumentDataManager::new(adapter);
    
    // Test plain text file
    let txt_path = temp_dir.path().join("test.txt");
    fs::write(&txt_path, "This is a test document with some content for keyword extraction.").unwrap();
    
    let result = manager.process_file(&txt_path).unwrap();
    assert!(result.success);
    assert_eq!(result.content_type, ContentType::Document {
        format: "PlainText".to_string(),
        language: Some("en".to_string()),
    });
    assert!(result.processing_time_ms > 0);
    assert!(!result.metadata.keywords.is_empty());
    
    // Test Markdown file
    let md_path = temp_dir.path().join("test.md");
    fs::write(&md_path, "# Test Document\n\nThis is a **markdown** document with *formatting*.").unwrap();
    
    let result = manager.process_file(&md_path).unwrap();
    assert!(result.success);
    assert_eq!(result.content_type, ContentType::Document {
        format: "Markdown".to_string(),
        language: Some("en".to_string()),
    });
    
    // Test HTML file
    let html_path = temp_dir.path().join("test.html");
    fs::write(&html_path, "<html><body><h1>Test</h1><p>HTML content</p></body></html>").unwrap();
    
    let result = manager.process_file(&html_path).unwrap();
    assert!(result.success);
    assert_eq!(result.content_type, ContentType::Document {
        format: "HTML".to_string(),
        language: Some("en".to_string()),
    });
}

/// Test data file processing
#[test]
fn test_data_processing() {
    let temp_dir = TempDir::new().unwrap();
    let adapter = Box::new(BasicMemoryDocumentDataAdapter::new());
    let mut manager = BasicDocumentDataManager::new(adapter);
    
    // Test CSV file
    let csv_path = temp_dir.path().join("test.csv");
    fs::write(&csv_path, "name,age,city\nJohn,25,NYC\nJane,30,LA\nBob,35,Chicago").unwrap();
    
    let result = manager.process_file(&csv_path).unwrap();
    assert!(result.success);
    assert_eq!(result.content_type, ContentType::Data {
        format: "CSV".to_string(),
        schema: None,
    });
    assert!(result.processing_time_ms > 0);
    assert!(result.processing_time_ms < 10_000);
    assert!(result.metadata.summary.is_some());
    assert!(result.metadata.summary.unwrap().contains("4 rows"));
    
    // Test JSON file
    let json_path = temp_dir.path().join("test.json");
    fs::write(&json_path, r#"{"users": [{"name": "John", "age": 25}, {"name": "Jane", "age": 30}]}"#).unwrap();
    
    let result = manager.process_file(&json_path).unwrap();
    assert!(result.success);
    assert_eq!(result.content_type, ContentType::Data {
        format: "JSON".to_string(),
        schema: None,
    });
    assert!(result.processing_time_ms > 0);
    assert!(result.processing_time_ms < 10_000);
    
    // Test TSV file
    let tsv_path = temp_dir.path().join("test.tsv");
    fs::write(&tsv_path, "name\tage\tcity\nJohn\t25\tNYC\nJane\t30\tLA").unwrap();
    
    let result = manager.process_file(&tsv_path).unwrap();
    assert!(result.success);
    assert_eq!(result.content_type, ContentType::Data {
        format: "TSV".to_string(),
        schema: None,
    });
    assert!(result.processing_time_ms > 0);
    assert!(result.processing_time_ms < 10_000);
}

/// Test batch directory processing
#[test]
fn test_directory_processing() {
    let temp_dir = TempDir::new().unwrap();
    let adapter = Box::new(BasicMemoryDocumentDataAdapter::new());
    let mut manager = BasicDocumentDataManager::new(adapter);
    
    // Create test files
    let files = vec![
        ("doc1.txt", "First document content"),
        ("doc2.md", "# Second Document\nMarkdown content"),
        ("data.csv", "col1,col2\nval1,val2\nval3,val4"),
        ("info.json", r#"{"key": "value", "number": 42}"#),
    ];
    
    for (filename, content) in files {
        let file_path = temp_dir.path().join(filename);
        fs::write(&file_path, content).unwrap();
    }
    
    // Create subdirectory with more files
    let sub_dir = temp_dir.path().join("subdir");
    fs::create_dir(&sub_dir).unwrap();
    fs::write(sub_dir.join("sub_doc.txt"), "Subdirectory document").unwrap();
    
    // Process directory
    let result = manager.process_directory(temp_dir.path()).unwrap();
    
    assert_eq!(result.total_files, 5); // 4 main files + 1 subdirectory file
    assert_eq!(result.successful_files, 5);
    assert_eq!(result.failed_files, 0);
    assert!(result.processing_duration_ms > 0);
    assert!(!result.file_type_distribution.is_empty());
    
    // Check file type distribution
    assert!(result.file_type_distribution.contains_key("PlainText"));
    assert!(result.file_type_distribution.contains_key("Markdown"));
    assert!(result.file_type_distribution.contains_key("CSV"));
    assert!(result.file_type_distribution.contains_key("JSON"));
}

/// Test error handling
#[test]
fn test_error_handling() {
    let adapter = Box::new(BasicMemoryDocumentDataAdapter::new());
    let mut manager = BasicDocumentDataManager::new(adapter);
    
    // Test non-existent file
    let result = manager.process_file("non_existent_file.txt");
    assert!(result.is_err());
    
    // Test unsupported file extension
    let temp_dir = TempDir::new().unwrap();
    let unsupported_path = temp_dir.path().join("test.xyz");
    fs::write(&unsupported_path, "content").unwrap();
    
    let result = manager.process_file(&unsupported_path);
    assert!(result.is_err());
    
    // Test file too large
    let config = DocumentDataConfig {
        max_file_size: 10, // Very small limit
        ..Default::default()
    };
    let adapter2 = Box::new(BasicMemoryDocumentDataAdapter::new());
    let mut manager2 = BasicDocumentDataManager::with_config(adapter2, config);
    
    let large_file_path = temp_dir.path().join("large.txt");
    fs::write(&large_file_path, "This content is longer than 10 bytes").unwrap();
    
    let result = manager2.process_file(&large_file_path);
    assert!(result.is_err());
}

/// Test content extraction and metadata analysis
#[test]
fn test_content_extraction() {
    let temp_dir = TempDir::new().unwrap();
    let adapter = Box::new(BasicMemoryDocumentDataAdapter::new());
    let mut manager = BasicDocumentDataManager::new(adapter);
    
    // Test with rich content
    let content = "The quick brown fox jumps over the lazy dog. This is a test document for keyword extraction and summarization. The document contains multiple sentences and should generate meaningful keywords.";
    let txt_path = temp_dir.path().join("rich_content.txt");
    fs::write(&txt_path, content).unwrap();
    
    let result = manager.process_file(&txt_path).unwrap();
    
    // Check metadata
    assert!(result.metadata.file_size > 0);
    assert!(result.metadata.summary.is_some());
    assert!(!result.metadata.keywords.is_empty());
    assert_eq!(result.metadata.language, Some("en".to_string()));
    assert!(result.metadata.quality_score > 0.0);
    
    // Check properties
    assert!(result.metadata.properties.contains_key("format"));
    assert!(result.metadata.properties.contains_key("word_count"));
    assert!(result.metadata.properties.contains_key("char_count"));
}

/// Test storage statistics and search functionality
#[test]
fn test_storage_operations() {
    let adapter = Box::new(BasicMemoryDocumentDataAdapter::new());
    let mut manager = BasicDocumentDataManager::new(adapter);
    
    let temp_dir = TempDir::new().unwrap();
    
    // Process multiple files of different types
    let files = vec![
        ("doc1.txt", "Document content", ContentType::Document {
            format: "PlainText".to_string(),
            language: Some("en".to_string()),
        }),
        ("data1.csv", "col1,col2\nval1,val2", ContentType::Data {
            format: "CSV".to_string(),
            schema: None,
        }),
    ];
    
    for (filename, content, expected_type) in files {
        let file_path = temp_dir.path().join(filename);
        fs::write(&file_path, content).unwrap();
        
        let result = manager.process_file(&file_path).unwrap();
        assert_eq!(result.content_type, expected_type);
    }
    
    // Test storage statistics
    let stats = manager.get_stats().unwrap();
    assert_eq!(stats.total_memories, 2);
    assert!(stats.total_size > 0);
    assert!(stats.memories_by_type.len() > 0);
    
    // Test search by type
    let doc_type = ContentType::Document {
        format: "PlainText".to_string(),
        language: Some("en".to_string()),
    };
    let doc_results = manager.search_by_type(&doc_type).unwrap();
    assert_eq!(doc_results.len(), 1);
    
    let data_type = ContentType::Data {
        format: "CSV".to_string(),
        schema: None,
    };
    let data_results = manager.search_by_type(&data_type).unwrap();
    assert_eq!(data_results.len(), 1);
    
    // Test get all memories
    let all_memories = manager.get_all_memories().unwrap();
    assert_eq!(all_memories.len(), 2);
}

/// Helper function to create test memory
fn create_test_memory(id: &str, content: &str, content_type: ContentType) -> MultiModalMemory {
    use std::collections::HashMap;

    MultiModalMemory {
        id: id.to_string(),
        content_type,
        primary_content: content.as_bytes().to_vec(),
        metadata: MultiModalMetadata {
            title: Some("Test Memory".to_string()),
            description: Some("Test description".to_string()),
            tags: vec!["test".to_string()],
            quality_score: 0.8,
            confidence: 0.9,
            processing_time_ms: 100,
            extracted_features: HashMap::new(),
        },
        extracted_features: HashMap::new(),
        cross_modal_links: Vec::new(),
        created_at: chrono::Utc::now(),
        updated_at: chrono::Utc::now(),
    }
}
