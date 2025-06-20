//! Jupyter Integration Tests
//!
//! Comprehensive integration tests for Jupyter notebook functionality,
//! magic commands, and data science workflow integration.

use synaptic::integrations::data_science::{DataScienceManager, DataScienceConfig};
use synaptic::error::Result;
use std::path::PathBuf;
use tempfile::TempDir;
use tokio;

/// Test data science manager initialization
#[tokio::test]
async fn test_data_science_manager_initialization() -> Result<()> {
    let config = DataScienceConfig::default();
    let manager = DataScienceManager::new(config)?;
    
    // Test that components are properly initialized
    assert!(manager.pandas().is_some());
    assert!(manager.numpy().is_some());
    assert!(manager.sklearn().is_some());
    assert!(manager.jupyter().is_some());
    
    Ok(())
}

/// Test pandas integration
#[tokio::test]
async fn test_pandas_integration() -> Result<()> {
    let config = DataScienceConfig::default();
    let manager = DataScienceManager::new(config)?;
    
    if let Some(pandas) = manager.pandas() {
        // Test data conversion
        let test_data = serde_json::json!([
            {"id": 1, "name": "test1", "value": 10.5},
            {"id": 2, "name": "test2", "value": 20.3},
            {"id": 3, "name": "test3", "value": 15.7}
        ]);
        
        let df_path = pandas.json_to_dataframe(&test_data).await?;
        assert!(!df_path.is_empty());
        
        // Test DataFrame info
        let info = pandas.get_dataframe_info(&df_path).await?;
        assert_eq!(info.shape, (3, 3)); // 3 rows, 3 columns
        assert!(info.columns.contains(&"id".to_string()));
        assert!(info.columns.contains(&"name".to_string()));
        assert!(info.columns.contains(&"value".to_string()));
    }
    
    Ok(())
}

/// Test numpy integration
#[tokio::test]
async fn test_numpy_integration() -> Result<()> {
    let config = DataScienceConfig::default();
    let manager = DataScienceManager::new(config)?;
    
    if let Some(numpy) = manager.numpy() {
        // Test array creation
        let test_data = vec![
            vec![1.0, 2.0, 3.0],
            vec![4.0, 5.0, 6.0],
            vec![7.0, 8.0, 9.0],
        ];
        
        let array_path = numpy.create_array(&test_data).await?;
        assert!(!array_path.is_empty());
        
        // Test array info
        let info = numpy.get_array_info(&array_path).await?;
        assert_eq!(info.shape, vec![3, 3]);
        assert_eq!(info.size, 9);
        assert_eq!(info.ndim, 2);
    }
    
    Ok(())
}

/// Test sklearn integration
#[tokio::test]
async fn test_sklearn_integration() -> Result<()> {
    let config = DataScienceConfig::default();
    let manager = DataScienceManager::new(config)?;
    
    if let Some(sklearn) = manager.sklearn() {
        // Create test data for clustering
        let test_data = vec![
            vec![1.0, 2.0],
            vec![1.5, 1.8],
            vec![5.0, 8.0],
            vec![8.0, 8.0],
            vec![1.0, 0.6],
            vec![9.0, 11.0],
        ];
        
        // Save test data to temporary file
        let temp_dir = TempDir::new().unwrap();
        let data_path = temp_dir.path().join("test_data.npy");
        
        // Create numpy array first
        if let Some(numpy) = manager.numpy() {
            let array_path = numpy.create_array(&test_data).await?;
            
            // Test clustering
            let cluster_config = synaptic::integrations::data_science::sklearn_integration::ModelConfig {
                algorithm: "kmeans".to_string(),
                parameters: {
                    let mut params = std::collections::HashMap::new();
                    params.insert("n_clusters".to_string(), serde_json::Value::Number(serde_json::Number::from(2)));
                    params
                },
                preprocessing: None,
            };
            
            let results = sklearn.cluster_data(&array_path, &cluster_config).await?;
            assert_eq!(results.n_clusters, 2);
            assert_eq!(results.labels.len(), 6);
            assert!(results.silhouette_score >= -1.0 && results.silhouette_score <= 1.0);
        }
    }
    
    Ok(())
}

/// Test jupyter integration
#[tokio::test]
async fn test_jupyter_integration() -> Result<()> {
    let config = DataScienceConfig::default();
    let manager = DataScienceManager::new(config)?;
    
    if let Some(jupyter) = manager.jupyter() {
        // Test notebook creation
        let notebook_config = synaptic::integrations::data_science::jupyter_integration::NotebookConfig {
            title: "Test Notebook".to_string(),
            description: Some("Test notebook for integration testing".to_string()),
            kernel: "python3".to_string(),
            metadata: std::collections::HashMap::new(),
        };
        
        let notebook_path = jupyter.create_notebook(&notebook_config).await?;
        assert!(!notebook_path.is_empty());
        
        // Test adding cells
        let code_cell = synaptic::integrations::data_science::jupyter_integration::NotebookCell {
            cell_type: synaptic::integrations::data_science::jupyter_integration::CellType::Code,
            source: vec![
                "import pandas as pd".to_string(),
                "print('Hello from test notebook!')".to_string(),
            ],
            metadata: std::collections::HashMap::new(),
            outputs: None,
            execution_count: None,
        };
        
        jupyter.add_cell(&notebook_path, code_cell).await?;
        
        // Verify notebook file exists
        assert!(std::path::Path::new(&notebook_path).exists());
    }
    
    Ok(())
}

/// Test data export functionality
#[tokio::test]
async fn test_data_export() -> Result<()> {
    let config = DataScienceConfig::default();
    let manager = DataScienceManager::new(config)?;
    
    let test_data = serde_json::json!([
        {"id": 1, "name": "test1", "value": 10.5},
        {"id": 2, "name": "test2", "value": 20.3},
    ]);
    
    let temp_dir = TempDir::new().unwrap();
    
    // Test JSON export
    let json_path = temp_dir.path().join("test_export.json");
    manager.export_data(&test_data, "json", &json_path).await?;
    assert!(json_path.exists());
    
    // Test CSV export (if pandas is available)
    if manager.pandas().is_some() {
        let csv_path = temp_dir.path().join("test_export.csv");
        manager.export_data(&test_data, "csv", &csv_path).await?;
        assert!(csv_path.exists());
    }
    
    Ok(())
}

/// Test data import functionality
#[tokio::test]
async fn test_data_import() -> Result<()> {
    let config = DataScienceConfig::default();
    let manager = DataScienceManager::new(config)?;
    
    let test_data = serde_json::json!([
        {"id": 1, "name": "test1", "value": 10.5},
        {"id": 2, "name": "test2", "value": 20.3},
    ]);
    
    let temp_dir = TempDir::new().unwrap();
    
    // Export data first
    let json_path = temp_dir.path().join("test_data.json");
    manager.export_data(&test_data, "json", &json_path).await?;
    
    // Test import
    let imported_data = manager.import_data(&json_path, Some("json")).await?;
    assert_eq!(imported_data, test_data);
    
    Ok(())
}

/// Test workflow manager
#[tokio::test]
async fn test_workflow_manager() -> Result<()> {
    let config = DataScienceConfig::default();
    let manager = DataScienceManager::new(config)?;
    
    let workflow_manager = manager.workflow();
    
    // Test workflow creation
    let steps = vec![
        synaptic::integrations::data_science::workflow_manager::WorkflowStep {
            id: "load_data".to_string(),
            name: "Load Data".to_string(),
            step_type: synaptic::integrations::data_science::workflow_manager::WorkflowStepType::DataLoad,
            parameters: {
                let mut params = std::collections::HashMap::new();
                params.insert("data_path".to_string(), serde_json::Value::String("/tmp/test_data.json".to_string()));
                params
            },
            dependencies: vec![],
            outputs: vec!["loaded_data".to_string()],
        },
        synaptic::integrations::data_science::workflow_manager::WorkflowStep {
            id: "transform_data".to_string(),
            name: "Transform Data".to_string(),
            step_type: synaptic::integrations::data_science::workflow_manager::WorkflowStepType::DataTransform,
            parameters: {
                let mut params = std::collections::HashMap::new();
                params.insert("transform_type".to_string(), serde_json::Value::String("normalize".to_string()));
                params
            },
            dependencies: vec!["load_data".to_string()],
            outputs: vec!["transformed_data".to_string()],
        },
    ];
    
    // Note: This test would require mutable access to workflow_manager
    // In a real implementation, we'd need to restructure the API
    // For now, we just verify the workflow manager exists
    assert!(workflow_manager.list_workflows().is_empty());
    
    Ok(())
}

/// Test error handling
#[tokio::test]
async fn test_error_handling() -> Result<()> {
    let config = DataScienceConfig::default();
    let manager = DataScienceManager::new(config)?;
    
    // Test invalid export format
    let test_data = serde_json::json!({"test": "data"});
    let temp_dir = TempDir::new().unwrap();
    let invalid_path = temp_dir.path().join("test.invalid");
    
    let result = manager.export_data(&test_data, "invalid_format", &invalid_path).await;
    assert!(result.is_err());
    
    // Test invalid import path
    let nonexistent_path = PathBuf::from("/nonexistent/path/file.json");
    let result = manager.import_data(&nonexistent_path, Some("json")).await;
    assert!(result.is_err());
    
    Ok(())
}
