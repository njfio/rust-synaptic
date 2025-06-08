//! # Phase 5B: Advanced Document Processing Demo
//!
//! This example demonstrates the advanced document and data processing capabilities
//! of Phase 5B, including document analysis, data processing, and batch operations.

use synaptic::phase5b_basic::*;
use std::fs;
use tempfile::TempDir;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🚀 Phase 5B: Advanced Document Processing Demo");
    println!("==============================================\n");

    // Create temporary directory for demo files
    let temp_dir = TempDir::new()?;
    println!("📁 Created temporary directory: {}", temp_dir.path().display());

    // Create demo files
    create_demo_files(&temp_dir)?;

    // Initialize document and data manager
    let adapter = Box::new(BasicMemoryDocumentDataAdapter::new());
    let mut manager = BasicDocumentDataManager::new(adapter);

    println!("\n📊 Document and Data Manager Configuration");
    println!("------------------------------------------");
    println!("✅ Max file size: 100 MB");
    println!("✅ Content extraction: Enabled");
    println!("✅ Metadata analysis: Enabled");
    println!("✅ Batch processing: Enabled");
    println!("✅ Supported extensions: PDF, DOC, DOCX, MD, TXT, HTML, CSV, JSON, XLSX, TSV, XML");

    // Demo 1: Process individual documents
    println!("\n📄 Demo 1: Individual Document Processing");
    println!("==========================================");
    
    demo_document_processing(&mut manager, &temp_dir)?;

    // Demo 2: Process data files
    println!("\n📊 Demo 2: Data File Processing");
    println!("===============================");
    
    demo_data_processing(&mut manager, &temp_dir)?;

    // Demo 3: Batch directory processing
    println!("\n📁 Demo 3: Batch Directory Processing");
    println!("=====================================");
    
    demo_batch_processing(&mut manager, &temp_dir)?;

    // Demo 4: Storage and search operations
    println!("\n🔍 Demo 4: Storage and Search Operations");
    println!("========================================");
    
    demo_storage_operations(&mut manager)?;

    println!("\n🎉 Phase 5B Advanced Document Processing Demo Complete!");
    println!("All document and data processing features demonstrated successfully.");

    Ok(())
}

/// Create demo files for testing
fn create_demo_files(temp_dir: &TempDir) -> Result<(), Box<dyn std::error::Error>> {
    println!("\n📝 Creating demo files...");

    // Create documents
    let documents = vec![
        ("research_paper.txt", 
         "Artificial Intelligence in Modern Computing\n\nAbstract: This paper explores the current state of artificial intelligence and its applications in modern computing systems. We examine machine learning algorithms, neural networks, and their practical implementations.\n\nIntroduction: Artificial intelligence has revolutionized the way we approach complex computational problems. From natural language processing to computer vision, AI technologies are transforming industries.\n\nConclusion: The future of AI holds immense potential for solving humanity's greatest challenges."),
        
        ("meeting_notes.md", 
         "# Project Meeting Notes\n\n## Date: 2024-01-15\n\n### Attendees\n- Alice Johnson (Project Manager)\n- Bob Smith (Developer)\n- Carol Davis (Designer)\n\n### Agenda\n1. **Project Status Update**\n   - Backend development: 80% complete\n   - Frontend design: 60% complete\n   - Testing: 40% complete\n\n2. **Next Steps**\n   - Complete API integration\n   - Finalize UI components\n   - Begin user acceptance testing\n\n### Action Items\n- [ ] Bob: Complete authentication module\n- [ ] Carol: Deliver final mockups\n- [ ] Alice: Schedule client review"),
        
        ("product_spec.html", 
         "<html><head><title>Product Specification</title></head><body><h1>Smart Home Device Specification</h1><h2>Overview</h2><p>The Smart Home Hub is an innovative device that connects and controls various IoT devices in a home environment.</p><h2>Features</h2><ul><li>Voice control integration</li><li>Mobile app connectivity</li><li>Energy monitoring</li><li>Security system integration</li></ul><h2>Technical Requirements</h2><p>The device must support WiFi 6, Bluetooth 5.0, and Zigbee protocols.</p></body></html>"),
    ];

    for (filename, content) in documents {
        let file_path = temp_dir.path().join(filename);
        fs::write(&file_path, content)?;
        println!("  ✅ Created: {}", filename);
    }

    // Create data files
    let data_files = vec![
        ("sales_data.csv", 
         "date,product,quantity,revenue\n2024-01-01,Widget A,100,5000.00\n2024-01-02,Widget B,75,3750.00\n2024-01-03,Widget A,120,6000.00\n2024-01-04,Widget C,50,2500.00\n2024-01-05,Widget B,90,4500.00"),
        
        ("user_config.json", 
         r#"{"application": {"name": "DocumentProcessor", "version": "1.0.0"}, "settings": {"theme": "dark", "language": "en", "auto_save": true}, "features": {"advanced_search": true, "batch_processing": true, "cloud_sync": false}}"#),
        
        ("inventory.tsv", 
         "item_id\tname\tcategory\tstock\tprice\n001\tLaptop\tElectronics\t25\t999.99\n002\tMouse\tAccessories\t150\t29.99\n003\tKeyboard\tAccessories\t75\t79.99\n004\tMonitor\tElectronics\t40\t299.99"),
    ];

    for (filename, content) in data_files {
        let file_path = temp_dir.path().join(filename);
        fs::write(&file_path, content)?;
        println!("  ✅ Created: {}", filename);
    }

    // Create subdirectory with additional files
    let sub_dir = temp_dir.path().join("reports");
    fs::create_dir(&sub_dir)?;
    
    fs::write(sub_dir.join("quarterly_report.txt"), 
        "Q1 2024 Financial Report\n\nRevenue increased by 15% compared to Q4 2023. Key growth drivers include new product launches and expanded market presence.")?;
    
    fs::write(sub_dir.join("metrics.csv"), 
        "metric,value,unit\nRevenue,1250000,USD\nCustomers,5420,count\nGrowth Rate,15.3,percent")?;

    println!("  ✅ Created subdirectory: reports/");
    println!("  ✅ Total files created: 8");

    Ok(())
}

/// Demo individual document processing
fn demo_document_processing(manager: &mut BasicDocumentDataManager, temp_dir: &TempDir) -> Result<(), Box<dyn std::error::Error>> {
    let documents = vec![
        "research_paper.txt",
        "meeting_notes.md", 
        "product_spec.html",
    ];

    for doc_name in documents {
        let file_path = temp_dir.path().join(doc_name);
        println!("\n📄 Processing: {}", doc_name);
        
        let result = manager.process_file(&file_path)?;
        
        println!("  ✅ Success: {}", result.success);
        println!("  📝 Content Type: {:?}", result.content_type);
        println!("  ⏱️  Processing Time: {} ms", result.processing_time_ms);
        println!("  📊 File Size: {} bytes", result.metadata.file_size);
        
        if let Some(summary) = &result.metadata.summary {
            println!("  📋 Summary: {}", summary);
        }
        
        if !result.metadata.keywords.is_empty() {
            println!("  🏷️  Keywords: {}", result.metadata.keywords.join(", "));
        }
        
        println!("  🎯 Quality Score: {:.2}", result.metadata.quality_score);
    }

    Ok(())
}

/// Demo data file processing
fn demo_data_processing(manager: &mut BasicDocumentDataManager, temp_dir: &TempDir) -> Result<(), Box<dyn std::error::Error>> {
    let data_files = vec![
        "sales_data.csv",
        "user_config.json",
        "inventory.tsv",
    ];

    for data_name in data_files {
        let file_path = temp_dir.path().join(data_name);
        println!("\n📊 Processing: {}", data_name);
        
        let result = manager.process_file(&file_path)?;
        
        println!("  ✅ Success: {}", result.success);
        println!("  📝 Content Type: {:?}", result.content_type);
        println!("  ⏱️  Processing Time: {} ms", result.processing_time_ms);
        println!("  📊 File Size: {} bytes", result.metadata.file_size);
        
        if let Some(summary) = &result.metadata.summary {
            println!("  📋 Summary: {}", summary);
        }
        
        // Display data-specific properties
        if let Some(row_count) = result.metadata.properties.get("row_count") {
            println!("  📈 Rows: {}", row_count);
        }
        if let Some(col_count) = result.metadata.properties.get("column_count") {
            println!("  📊 Columns: {}", col_count);
        }
        
        println!("  🎯 Quality Score: {:.2}", result.metadata.quality_score);
    }

    Ok(())
}

/// Demo batch directory processing
fn demo_batch_processing(manager: &mut BasicDocumentDataManager, temp_dir: &TempDir) -> Result<(), Box<dyn std::error::Error>> {
    println!("\n📁 Processing entire directory: {}", temp_dir.path().display());
    
    let result = manager.process_directory(temp_dir.path())?;
    
    println!("  ✅ Total Files: {}", result.total_files);
    println!("  ✅ Successful: {}", result.successful_files);
    println!("  ❌ Failed: {}", result.failed_files);
    println!("  ⏱️  Total Processing Time: {} ms", result.processing_duration_ms);
    
    println!("\n📊 File Type Distribution:");
    for (file_type, count) in &result.file_type_distribution {
        println!("  • {}: {} files", file_type, count);
    }
    
    println!("\n📋 Processing Results Summary:");
    for (i, result) in result.results.iter().enumerate().take(5) {
        if result.success {
            println!("  {}. ✅ {} - {} ms", i + 1, result.memory_id, result.processing_time_ms);
        } else {
            println!("  {}. ❌ Failed: {}", i + 1, result.errors.join(", "));
        }
    }
    
    if result.results.len() > 5 {
        println!("  ... and {} more results", result.results.len() - 5);
    }

    Ok(())
}

/// Demo storage and search operations
fn demo_storage_operations(manager: &mut BasicDocumentDataManager) -> Result<(), Box<dyn std::error::Error>> {
    // Get storage statistics
    let stats = manager.get_stats()?;
    
    println!("\n📊 Storage Statistics:");
    println!("  📁 Total Memories: {}", stats.total_memories);
    println!("  💾 Total Size: {} bytes", stats.total_size);
    
    println!("\n📋 Memories by Type:");
    for (memory_type, count) in &stats.memories_by_type {
        println!("  • {}: {} memories", memory_type, count);
    }
    
    // Search by content type
    println!("\n🔍 Search Operations:");
    
    let doc_type = ContentType::Document {
        format: "PlainText".to_string(),
        language: Some("en".to_string()),
    };
    let doc_results = manager.search_by_type(&doc_type)?;
    println!("  📄 Plain Text Documents: {} found", doc_results.len());
    
    let csv_type = ContentType::Data {
        format: "CSV".to_string(),
        schema: None,
    };
    let csv_results = manager.search_by_type(&csv_type)?;
    println!("  📊 CSV Data Files: {} found", csv_results.len());
    
    // Get all memories
    let all_memories = manager.get_all_memories()?;
    println!("  📚 Total Accessible Memories: {}", all_memories.len());
    
    // Display sample memory details
    if let Some(memory) = all_memories.first() {
        println!("\n📝 Sample Memory Details:");
        println!("  🆔 ID: {}", memory.id);
        println!("  📝 Content Type: {:?}", memory.content_type);
        println!("  📊 Content Size: {} bytes", memory.primary_content.len());
        if let Some(title) = &memory.metadata.title {
            println!("  📋 Title: {}", title);
        }
        if let Some(description) = &memory.metadata.description {
            println!("  📄 Description: {}", description);
        }
        println!("  🏷️  Tags: {}", memory.metadata.tags.join(", "));
        println!("  🎯 Quality Score: {:.2}", memory.metadata.quality_score);
        println!("  ⏱️  Created: {}", memory.created_at.format("%Y-%m-%d %H:%M:%S"));
    }

    Ok(())
}
