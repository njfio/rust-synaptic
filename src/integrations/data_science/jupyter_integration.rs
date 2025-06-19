//! Jupyter Integration
//!
//! This module provides integration with Jupyter notebooks and JupyterLab,
//! enabling seamless notebook generation, execution, and collaboration
//! workflows for data science and research activities.

use crate::error::{Result, SynapticError};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;
use tokio::fs;
use tokio::process::Command as AsyncCommand;
use tracing::{debug, error, info, warn};

/// Jupyter integration manager
pub struct JupyterIntegration {
    jupyter_path: String,
    notebook_dir: PathBuf,
    kernel_name: String,
}

/// Notebook configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NotebookConfig {
    pub title: String,
    pub description: Option<String>,
    pub kernel: String,
    pub metadata: HashMap<String, serde_json::Value>,
}

/// Notebook cell types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CellType {
    Code,
    Markdown,
    Raw,
}

/// Notebook cell
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NotebookCell {
    pub cell_type: CellType,
    pub source: Vec<String>,
    pub metadata: HashMap<String, serde_json::Value>,
    pub outputs: Option<Vec<serde_json::Value>>,
    pub execution_count: Option<u32>,
}

/// Notebook structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Notebook {
    pub nbformat: u32,
    pub nbformat_minor: u32,
    pub metadata: HashMap<String, serde_json::Value>,
    pub cells: Vec<NotebookCell>,
}

/// Execution results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionResults {
    pub notebook_path: String,
    pub execution_time: f64,
    pub cells_executed: usize,
    pub errors: Vec<String>,
    pub outputs: Vec<serde_json::Value>,
}

impl JupyterIntegration {
    /// Create a new jupyter integration
    pub fn new() -> Result<Self> {
        let jupyter_path = Self::find_jupyter_executable()?;
        let notebook_dir = std::env::current_dir()
            .map_err(|e| SynapticError::IoError(format!("Failed to get current directory: {}", e)))?
            .join("notebooks");
        
        std::fs::create_dir_all(&notebook_dir).map_err(|e| {
            SynapticError::IoError(format!("Failed to create notebook directory: {}", e))
        })?;

        Ok(Self {
            jupyter_path,
            notebook_dir,
            kernel_name: "python3".to_string(),
        })
    }

    /// Find Jupyter executable
    fn find_jupyter_executable() -> Result<String> {
        let candidates = vec!["jupyter", "jupyter-lab", "jupyter-notebook"];
        
        for candidate in candidates {
            if let Ok(output) = std::process::Command::new(candidate)
                .args(&["--version"])
                .output()
            {
                if output.status.success() {
                    info!("Found Jupyter: {}", candidate);
                    return Ok(candidate.to_string());
                }
            }
        }
        
        Err(SynapticError::IntegrationError(
            "Could not find Jupyter executable".to_string()
        ))
    }

    /// Create a new notebook
    pub async fn create_notebook(&self, config: &NotebookConfig) -> Result<String> {
        let notebook = self.generate_notebook_structure(config)?;
        let notebook_path = self.notebook_dir.join(format!("{}.ipynb", 
            config.title.replace(" ", "_").to_lowercase()));
        
        let notebook_json = serde_json::to_string_pretty(&notebook).map_err(|e| {
            SynapticError::SerializationError(format!("Failed to serialize notebook: {}", e))
        })?;
        
        fs::write(&notebook_path, notebook_json).await.map_err(|e| {
            SynapticError::IoError(format!("Failed to write notebook: {}", e))
        })?;
        
        info!("Created notebook: {}", notebook_path.display());
        Ok(notebook_path.to_string_lossy().to_string())
    }

    /// Add cell to existing notebook
    pub async fn add_cell(&self, notebook_path: &str, cell: NotebookCell) -> Result<()> {
        let mut notebook = self.load_notebook(notebook_path).await?;
        notebook.cells.push(cell);
        
        let notebook_json = serde_json::to_string_pretty(&notebook).map_err(|e| {
            SynapticError::SerializationError(format!("Failed to serialize notebook: {}", e))
        })?;
        
        fs::write(notebook_path, notebook_json).await.map_err(|e| {
            SynapticError::IoError(format!("Failed to write notebook: {}", e))
        })?;
        
        debug!("Added cell to notebook: {}", notebook_path);
        Ok(())
    }

    /// Execute notebook
    pub async fn execute_notebook(&self, notebook_path: &str) -> Result<ExecutionResults> {
        let start_time = std::time::Instant::now();
        
        let output = AsyncCommand::new(&self.jupyter_path)
            .args(&[
                "nbconvert",
                "--to", "notebook",
                "--execute",
                "--inplace",
                notebook_path
            ])
            .output()
            .await
            .map_err(|e| {
                SynapticError::IntegrationError(format!("Failed to execute notebook: {}", e))
            })?;

        let execution_time = start_time.elapsed().as_secs_f64();
        
        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            warn!("Notebook execution had issues: {}", stderr);
        }

        // Load executed notebook to get results
        let notebook = self.load_notebook(notebook_path).await?;
        let cells_executed = notebook.cells.iter()
            .filter(|cell| matches!(cell.cell_type, CellType::Code))
            .count();
        
        let mut errors = Vec::new();
        let mut outputs = Vec::new();
        
        for cell in &notebook.cells {
            if let Some(cell_outputs) = &cell.outputs {
                for output in cell_outputs {
                    if let Some(output_type) = output.get("output_type") {
                        if output_type == "error" {
                            if let Some(error_msg) = output.get("evalue") {
                                errors.push(error_msg.as_str().unwrap_or("Unknown error").to_string());
                            }
                        } else {
                            outputs.push(output.clone());
                        }
                    }
                }
            }
        }

        Ok(ExecutionResults {
            notebook_path: notebook_path.to_string(),
            execution_time,
            cells_executed,
            errors,
            outputs,
        })
    }

    /// Generate analysis notebook
    pub async fn generate_analysis_notebook(&self, data_path: &str, analysis_type: &str) -> Result<String> {
        let config = NotebookConfig {
            title: format!("Synaptic {} Analysis", analysis_type),
            description: Some(format!("Automated {} analysis of Synaptic memory data", analysis_type)),
            kernel: self.kernel_name.clone(),
            metadata: HashMap::new(),
        };

        let notebook_path = self.create_notebook(&config).await?;
        
        // Add analysis cells based on type
        match analysis_type {
            "memory_exploration" => {
                self.add_memory_exploration_cells(&notebook_path, data_path).await?;
            },
            "clustering" => {
                self.add_clustering_analysis_cells(&notebook_path, data_path).await?;
            },
            "similarity" => {
                self.add_similarity_analysis_cells(&notebook_path, data_path).await?;
            },
            "visualization" => {
                self.add_visualization_cells(&notebook_path, data_path).await?;
            },
            _ => {
                return Err(SynapticError::ValidationError(
                    format!("Unsupported analysis type: {}", analysis_type)
                ));
            }
        }

        Ok(notebook_path)
    }

    /// Start Jupyter server
    pub async fn start_jupyter_server(&self, port: Option<u16>) -> Result<String> {
        let port = port.unwrap_or(8888);
        
        let output = AsyncCommand::new(&self.jupyter_path)
            .args(&[
                "lab",
                "--port", &port.to_string(),
                "--notebook-dir", &self.notebook_dir.to_string_lossy(),
                "--no-browser",
                "--allow-root"
            ])
            .spawn()
            .map_err(|e| {
                SynapticError::IntegrationError(format!("Failed to start Jupyter server: {}", e))
            })?;

        info!("Started Jupyter server on port {}", port);
        Ok(format!("http://localhost:{}", port))
    }

    /// Load notebook from file
    async fn load_notebook(&self, notebook_path: &str) -> Result<Notebook> {
        let content = fs::read_to_string(notebook_path).await.map_err(|e| {
            SynapticError::IoError(format!("Failed to read notebook: {}", e))
        })?;
        
        serde_json::from_str(&content).map_err(|e| {
            SynapticError::SerializationError(format!("Failed to parse notebook: {}", e))
        })
    }

    /// Generate notebook structure
    fn generate_notebook_structure(&self, config: &NotebookConfig) -> Result<Notebook> {
        let mut metadata = HashMap::new();
        metadata.insert("kernelspec".to_string(), serde_json::json!({
            "display_name": "Python 3",
            "language": "python",
            "name": config.kernel
        }));
        metadata.insert("language_info".to_string(), serde_json::json!({
            "name": "python",
            "version": "3.8.0"
        }));
        
        // Add custom metadata
        for (key, value) in &config.metadata {
            metadata.insert(key.clone(), value.clone());
        }

        let title_cell = NotebookCell {
            cell_type: CellType::Markdown,
            source: vec![
                format!("# {}", config.title),
                "".to_string(),
                config.description.clone().unwrap_or_else(|| "Generated by Synaptic AI Agent Memory System".to_string()),
            ],
            metadata: HashMap::new(),
            outputs: None,
            execution_count: None,
        };

        Ok(Notebook {
            nbformat: 4,
            nbformat_minor: 4,
            metadata,
            cells: vec![title_cell],
        })
    }

    /// Add memory exploration cells
    async fn add_memory_exploration_cells(&self, notebook_path: &str, data_path: &str) -> Result<()> {
        let cells = vec![
            NotebookCell {
                cell_type: CellType::Code,
                source: vec![
                    "import pandas as pd".to_string(),
                    "import numpy as np".to_string(),
                    "import matplotlib.pyplot as plt".to_string(),
                    "import seaborn as sns".to_string(),
                    "".to_string(),
                    "# Load Synaptic memory data".to_string(),
                    format!("data = pd.read_json('{}')", data_path),
                    "print(f'Loaded {{len(data)}} memory records')".to_string(),
                ],
                metadata: HashMap::new(),
                outputs: None,
                execution_count: None,
            },
            NotebookCell {
                cell_type: CellType::Code,
                source: vec![
                    "# Basic data exploration".to_string(),
                    "print('Data shape:', data.shape)".to_string(),
                    "print('\\nColumns:', data.columns.tolist())".to_string(),
                    "print('\\nData types:')".to_string(),
                    "print(data.dtypes)".to_string(),
                    "print('\\nFirst few rows:')".to_string(),
                    "data.head()".to_string(),
                ],
                metadata: HashMap::new(),
                outputs: None,
                execution_count: None,
            },
        ];

        for cell in cells {
            self.add_cell(notebook_path, cell).await?;
        }

        Ok(())
    }

    /// Add clustering analysis cells
    async fn add_clustering_analysis_cells(&self, notebook_path: &str, data_path: &str) -> Result<()> {
        let cells = vec![
            NotebookCell {
                cell_type: CellType::Code,
                source: vec![
                    "from sklearn.cluster import KMeans".to_string(),
                    "from sklearn.preprocessing import StandardScaler".to_string(),
                    "from sklearn.metrics import silhouette_score".to_string(),
                    "".to_string(),
                    "# Load and prepare data".to_string(),
                    format!("data = pd.read_json('{}')", data_path),
                    "# Extract numerical features for clustering".to_string(),
                    "numeric_data = data.select_dtypes(include=[np.number])".to_string(),
                    "scaler = StandardScaler()".to_string(),
                    "scaled_data = scaler.fit_transform(numeric_data)".to_string(),
                ],
                metadata: HashMap::new(),
                outputs: None,
                execution_count: None,
            },
            NotebookCell {
                cell_type: CellType::Code,
                source: vec![
                    "# Perform K-means clustering".to_string(),
                    "n_clusters = 5".to_string(),
                    "kmeans = KMeans(n_clusters=n_clusters, random_state=42)".to_string(),
                    "cluster_labels = kmeans.fit_predict(scaled_data)".to_string(),
                    "".to_string(),
                    "# Calculate silhouette score".to_string(),
                    "silhouette = silhouette_score(scaled_data, cluster_labels)".to_string(),
                    "print(f'Silhouette Score: {{silhouette:.3f}}')".to_string(),
                    "".to_string(),
                    "# Add cluster labels to original data".to_string(),
                    "data['cluster'] = cluster_labels".to_string(),
                    "data['cluster'].value_counts().sort_index()".to_string(),
                ],
                metadata: HashMap::new(),
                outputs: None,
                execution_count: None,
            },
        ];

        for cell in cells {
            self.add_cell(notebook_path, cell).await?;
        }

        Ok(())
    }

    /// Add similarity analysis cells
    async fn add_similarity_analysis_cells(&self, notebook_path: &str, data_path: &str) -> Result<()> {
        let cells = vec![
            NotebookCell {
                cell_type: CellType::Code,
                source: vec![
                    "from sklearn.metrics.pairwise import cosine_similarity".to_string(),
                    "from scipy.spatial.distance import pdist, squareform".to_string(),
                    "".to_string(),
                    "# Load data and compute similarity matrix".to_string(),
                    format!("data = pd.read_json('{}')", data_path),
                    "numeric_data = data.select_dtypes(include=[np.number])".to_string(),
                    "".to_string(),
                    "# Compute cosine similarity".to_string(),
                    "similarity_matrix = cosine_similarity(numeric_data)".to_string(),
                    "print(f'Similarity matrix shape: {{similarity_matrix.shape}}')".to_string(),
                    "print(f'Average similarity: {{np.mean(similarity_matrix):.3f}}')".to_string(),
                ],
                metadata: HashMap::new(),
                outputs: None,
                execution_count: None,
            },
            NotebookCell {
                cell_type: CellType::Code,
                source: vec![
                    "# Visualize similarity matrix".to_string(),
                    "plt.figure(figsize=(10, 8))".to_string(),
                    "sns.heatmap(similarity_matrix[:50, :50], cmap='viridis')".to_string(),
                    "plt.title('Memory Similarity Matrix (First 50 items)')".to_string(),
                    "plt.xlabel('Memory Index')".to_string(),
                    "plt.ylabel('Memory Index')".to_string(),
                    "plt.show()".to_string(),
                ],
                metadata: HashMap::new(),
                outputs: None,
                execution_count: None,
            },
        ];

        for cell in cells {
            self.add_cell(notebook_path, cell).await?;
        }

        Ok(())
    }

    /// Add visualization cells
    async fn add_visualization_cells(&self, notebook_path: &str, data_path: &str) -> Result<()> {
        let cells = vec![
            NotebookCell {
                cell_type: CellType::Code,
                source: vec![
                    "import plotly.express as px".to_string(),
                    "import plotly.graph_objects as go".to_string(),
                    "from sklearn.decomposition import PCA".to_string(),
                    "".to_string(),
                    "# Load data and perform PCA for visualization".to_string(),
                    format!("data = pd.read_json('{}')", data_path),
                    "numeric_data = data.select_dtypes(include=[np.number])".to_string(),
                    "".to_string(),
                    "pca = PCA(n_components=2)".to_string(),
                    "pca_result = pca.fit_transform(StandardScaler().fit_transform(numeric_data))".to_string(),
                    "".to_string(),
                    "# Create interactive scatter plot".to_string(),
                    "fig = px.scatter(x=pca_result[:, 0], y=pca_result[:, 1],".to_string(),
                    "                title='Memory Data PCA Visualization')".to_string(),
                    "fig.show()".to_string(),
                ],
                metadata: HashMap::new(),
                outputs: None,
                execution_count: None,
            },
        ];

        for cell in cells {
            self.add_cell(notebook_path, cell).await?;
        }

        Ok(())
    }
}
