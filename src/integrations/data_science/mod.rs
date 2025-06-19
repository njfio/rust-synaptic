//! Data Science Workflow Integration
//!
//! This module provides comprehensive integration with popular data science tools
//! and workflows, enabling seamless data exchange, analysis pipelines, and
//! collaborative research environments.

pub mod pandas_integration;
pub mod numpy_integration;
pub mod sklearn_integration;
pub mod jupyter_integration;
pub mod workflow_manager;

use crate::error::Result;
use serde::{Deserialize, Serialize};

use std::path::PathBuf;

/// Configuration for data science integrations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataScienceConfig {
    pub enable_pandas: bool,
    pub enable_numpy: bool,
    pub enable_sklearn: bool,
    pub enable_jupyter: bool,
    pub python_path: Option<PathBuf>,
    pub virtual_env: Option<PathBuf>,
    pub notebook_dir: Option<PathBuf>,
    pub export_formats: Vec<String>,
}

impl Default for DataScienceConfig {
    fn default() -> Self {
        Self {
            enable_pandas: true,
            enable_numpy: true,
            enable_sklearn: true,
            enable_jupyter: true,
            python_path: None,
            virtual_env: None,
            notebook_dir: None,
            export_formats: vec![
                "csv".to_string(),
                "json".to_string(),
                "parquet".to_string(),
                "pickle".to_string(),
            ],
        }
    }
}

/// Data science workflow manager
pub struct DataScienceManager {
    config: DataScienceConfig,
    pandas: Option<pandas_integration::PandasIntegration>,
    numpy: Option<numpy_integration::NumpyIntegration>,
    sklearn: Option<sklearn_integration::SklearnIntegration>,
    jupyter: Option<jupyter_integration::JupyterIntegration>,
    workflow: workflow_manager::WorkflowManager,
}

impl DataScienceManager {
    /// Create a new data science manager
    pub fn new(config: DataScienceConfig) -> Result<Self> {
        let pandas = if config.enable_pandas {
            Some(pandas_integration::PandasIntegration::new()?)
        } else {
            None
        };

        let numpy = if config.enable_numpy {
            Some(numpy_integration::NumpyIntegration::new()?)
        } else {
            None
        };

        let sklearn = if config.enable_sklearn {
            Some(sklearn_integration::SklearnIntegration::new()?)
        } else {
            None
        };

        let jupyter = if config.enable_jupyter {
            Some(jupyter_integration::JupyterIntegration::new()?)
        } else {
            None
        };

        let workflow = workflow_manager::WorkflowManager::new(config.clone())?;

        Ok(Self {
            config,
            pandas,
            numpy,
            sklearn,
            jupyter,
            workflow,
        })
    }

    /// Get pandas integration
    pub fn pandas(&self) -> Option<&pandas_integration::PandasIntegration> {
        self.pandas.as_ref()
    }

    /// Get numpy integration
    pub fn numpy(&self) -> Option<&numpy_integration::NumpyIntegration> {
        self.numpy.as_ref()
    }

    /// Get sklearn integration
    pub fn sklearn(&self) -> Option<&sklearn_integration::SklearnIntegration> {
        self.sklearn.as_ref()
    }

    /// Get jupyter integration
    pub fn jupyter(&self) -> Option<&jupyter_integration::JupyterIntegration> {
        self.jupyter.as_ref()
    }

    /// Get workflow manager
    pub fn workflow(&self) -> &workflow_manager::WorkflowManager {
        &self.workflow
    }

    /// Export data in specified format
    pub async fn export_data(&self, data: &serde_json::Value, format: &str, path: &PathBuf) -> Result<()> {
        match format {
            "csv" => {
                if let Some(pandas) = &self.pandas {
                    pandas.export_csv(data, path).await
                } else {
                    Err(crate::error::SynapticError::IntegrationError(
                        "Pandas integration not available".to_string()
                    ))
                }
            }
            "json" => {
                self.workflow.export_json(data, path).await
            }
            "parquet" => {
                if let Some(pandas) = &self.pandas {
                    pandas.export_parquet(data, path).await
                } else {
                    Err(crate::error::SynapticError::IntegrationError(
                        "Pandas integration not available".to_string()
                    ))
                }
            }
            _ => Err(crate::error::SynapticError::ValidationError(
                format!("Unsupported export format: {}", format)
            ))
        }
    }

    /// Import data from file
    pub async fn import_data(&self, path: &PathBuf, format: Option<&str>) -> Result<serde_json::Value> {
        let format = format.unwrap_or_else(|| {
            path.extension()
                .and_then(|ext| ext.to_str())
                .unwrap_or("json")
        });

        match format {
            "csv" => {
                if let Some(pandas) = &self.pandas {
                    pandas.import_csv(path).await
                } else {
                    Err(crate::error::SynapticError::IntegrationError(
                        "Pandas integration not available".to_string()
                    ))
                }
            }
            "json" => {
                self.workflow.import_json(path).await
            }
            "parquet" => {
                if let Some(pandas) = &self.pandas {
                    pandas.import_parquet(path).await
                } else {
                    Err(crate::error::SynapticError::IntegrationError(
                        "Pandas integration not available".to_string()
                    ))
                }
            }
            _ => Err(crate::error::SynapticError::ValidationError(
                format!("Unsupported import format: {}", format)
            ))
        }
    }
}
