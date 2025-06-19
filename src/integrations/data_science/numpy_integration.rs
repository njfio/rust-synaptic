//! NumPy Integration
//!
//! This module provides comprehensive integration with NumPy arrays and
//! mathematical operations, enabling efficient numerical computations
//! and array manipulations for memory embeddings and analytics.

use crate::error::{Result, SynapticError};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;
use tokio::fs;
use tokio::process::Command as AsyncCommand;
use tracing::{debug, error, info};

/// NumPy integration manager
pub struct NumpyIntegration {
    python_path: String,
    temp_dir: PathBuf,
}

/// Array metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArrayInfo {
    pub shape: Vec<usize>,
    pub dtype: String,
    pub size: usize,
    pub ndim: usize,
    pub memory_usage: usize,
}

/// Mathematical operation configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MathOperation {
    pub operation_type: String,
    pub parameters: HashMap<String, serde_json::Value>,
    pub axis: Option<i32>,
}

/// Statistical analysis results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StatisticalAnalysis {
    pub mean: Vec<f64>,
    pub std: Vec<f64>,
    pub min: Vec<f64>,
    pub max: Vec<f64>,
    pub median: Vec<f64>,
    pub percentiles: HashMap<String, Vec<f64>>,
}

impl NumpyIntegration {
    /// Create a new numpy integration
    pub fn new() -> Result<Self> {
        let python_path = Self::find_python_executable()?;
        let temp_dir = std::env::temp_dir().join("synaptic_numpy");
        
        std::fs::create_dir_all(&temp_dir).map_err(|e| {
            SynapticError::IoError(format!("Failed to create temp directory: {}", e))
        })?;

        Ok(Self {
            python_path,
            temp_dir,
        })
    }

    /// Find Python executable with numpy
    fn find_python_executable() -> Result<String> {
        let candidates = vec!["python3", "python", "python3.8", "python3.9", "python3.10", "python3.11"];
        
        for candidate in candidates {
            if let Ok(output) = std::process::Command::new(candidate)
                .args(&["-c", "import numpy; print('OK')"])
                .output()
            {
                if output.status.success() {
                    info!("Found Python with numpy: {}", candidate);
                    return Ok(candidate.to_string());
                }
            }
        }
        
        Err(SynapticError::IntegrationError(
            "Could not find Python executable with numpy installed".to_string()
        ))
    }

    /// Convert vector data to numpy array
    pub async fn create_array(&self, data: &[Vec<f64>]) -> Result<String> {
        let script = format!(
            r#"
import numpy as np
import json

data = {}
array = np.array(data)
array_path = '{}'
np.save(array_path, array)
print(array_path)
"#,
            serde_json::to_string(data).map_err(|e| {
                SynapticError::SerializationError(format!("Failed to serialize data: {}", e))
            })?,
            self.temp_dir.join("array.npy").display()
        );

        self.execute_python_script(&script).await
    }

    /// Get array information
    pub async fn get_array_info(&self, array_path: &str) -> Result<ArrayInfo> {
        let script = format!(
            r#"
import numpy as np
import json

array = np.load('{}')

info = {{
    'shape': list(array.shape),
    'dtype': str(array.dtype),
    'size': int(array.size),
    'ndim': int(array.ndim),
    'memory_usage': int(array.nbytes)
}}

print(json.dumps(info))
"#,
            array_path
        );

        let output = self.execute_python_script(&script).await?;
        serde_json::from_str(&output).map_err(|e| {
            SynapticError::SerializationError(format!("Failed to parse array info: {}", e))
        })
    }

    /// Perform mathematical operations on arrays
    pub async fn apply_math_operation(&self, array_path: &str, operation: &MathOperation) -> Result<String> {
        let script = self.generate_math_script(array_path, operation)?;
        self.execute_python_script(&script).await
    }

    /// Calculate statistical measures
    pub async fn calculate_statistics(&self, array_path: &str) -> Result<StatisticalAnalysis> {
        let script = format!(
            r#"
import numpy as np
import json

array = np.load('{}')

# Handle different array dimensions
if array.ndim == 1:
    axis = None
    array_2d = array.reshape(-1, 1)
else:
    axis = 0
    array_2d = array

stats = {{
    'mean': np.mean(array_2d, axis=axis).tolist() if axis is not None else [float(np.mean(array))],
    'std': np.std(array_2d, axis=axis).tolist() if axis is not None else [float(np.std(array))],
    'min': np.min(array_2d, axis=axis).tolist() if axis is not None else [float(np.min(array))],
    'max': np.max(array_2d, axis=axis).tolist() if axis is not None else [float(np.max(array))],
    'median': np.median(array_2d, axis=axis).tolist() if axis is not None else [float(np.median(array))],
    'percentiles': {{
        '25': np.percentile(array_2d, 25, axis=axis).tolist() if axis is not None else [float(np.percentile(array, 25))],
        '75': np.percentile(array_2d, 75, axis=axis).tolist() if axis is not None else [float(np.percentile(array, 75))],
        '90': np.percentile(array_2d, 90, axis=axis).tolist() if axis is not None else [float(np.percentile(array, 90))],
        '95': np.percentile(array_2d, 95, axis=axis).tolist() if axis is not None else [float(np.percentile(array, 95))]
    }}
}}

print(json.dumps(stats))
"#,
            array_path
        );

        let output = self.execute_python_script(&script).await?;
        serde_json::from_str(&output).map_err(|e| {
            SynapticError::SerializationError(format!("Failed to parse statistics: {}", e))
        })
    }

    /// Compute similarity matrix
    pub async fn compute_similarity_matrix(&self, array_path: &str, metric: &str) -> Result<String> {
        let script = format!(
            r#"
import numpy as np
from scipy.spatial.distance import pdist, squareform

array = np.load('{}')

# Compute pairwise distances
if '{}' == 'cosine':
    # Normalize vectors for cosine similarity
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    normalized = array / (norms + 1e-8)
    distances = pdist(normalized, metric='cosine')
    similarity_matrix = 1 - squareform(distances)
elif '{}' == 'euclidean':
    distances = pdist(array, metric='euclidean')
    # Convert to similarity (inverse of distance)
    max_dist = np.max(distances)
    similarity_matrix = 1 - (squareform(distances) / max_dist)
else:
    distances = pdist(array, metric='{}')
    similarity_matrix = 1 / (1 + squareform(distances))

output_path = '{}'
np.save(output_path, similarity_matrix)
print(output_path)
"#,
            array_path,
            metric,
            metric,
            metric,
            self.temp_dir.join("similarity_matrix.npy").display()
        );

        self.execute_python_script(&script).await
    }

    /// Perform dimensionality reduction
    pub async fn reduce_dimensions(&self, array_path: &str, method: &str, n_components: usize) -> Result<String> {
        let script = match method {
            "pca" => format!(
                r#"
import numpy as np
from sklearn.decomposition import PCA

array = np.load('{}')
pca = PCA(n_components={})
reduced = pca.fit_transform(array)

output_path = '{}'
np.save(output_path, reduced)
print(output_path)
"#,
                array_path,
                n_components,
                self.temp_dir.join("reduced_array.npy").display()
            ),
            "tsne" => format!(
                r#"
import numpy as np
from sklearn.manifold import TSNE

array = np.load('{}')
tsne = TSNE(n_components={}, random_state=42)
reduced = tsne.fit_transform(array)

output_path = '{}'
np.save(output_path, reduced)
print(output_path)
"#,
                array_path,
                n_components,
                self.temp_dir.join("reduced_array.npy").display()
            ),
            "umap" => format!(
                r#"
import numpy as np
import umap

array = np.load('{}')
reducer = umap.UMAP(n_components={}, random_state=42)
reduced = reducer.fit_transform(array)

output_path = '{}'
np.save(output_path, reduced)
print(output_path)
"#,
                array_path,
                n_components,
                self.temp_dir.join("reduced_array.npy").display()
            ),
            _ => return Err(SynapticError::ValidationError(
                format!("Unsupported dimensionality reduction method: {}", method)
            ))
        };

        self.execute_python_script(&script).await
    }

    /// Normalize array data
    pub async fn normalize_array(&self, array_path: &str, method: &str) -> Result<String> {
        let script = match method {
            "l2" => format!(
                r#"
import numpy as np
from sklearn.preprocessing import normalize

array = np.load('{}')
normalized = normalize(array, norm='l2')

output_path = '{}'
np.save(output_path, normalized)
print(output_path)
"#,
                array_path,
                self.temp_dir.join("normalized_array.npy").display()
            ),
            "minmax" => format!(
                r#"
import numpy as np
from sklearn.preprocessing import MinMaxScaler

array = np.load('{}')
scaler = MinMaxScaler()
normalized = scaler.fit_transform(array)

output_path = '{}'
np.save(output_path, normalized)
print(output_path)
"#,
                array_path,
                self.temp_dir.join("normalized_array.npy").display()
            ),
            "standard" => format!(
                r#"
import numpy as np
from sklearn.preprocessing import StandardScaler

array = np.load('{}')
scaler = StandardScaler()
normalized = scaler.fit_transform(array)

output_path = '{}'
np.save(output_path, normalized)
print(output_path)
"#,
                array_path,
                self.temp_dir.join("normalized_array.npy").display()
            ),
            _ => return Err(SynapticError::ValidationError(
                format!("Unsupported normalization method: {}", method)
            ))
        };

        self.execute_python_script(&script).await
    }

    /// Generate mathematical operation script
    fn generate_math_script(&self, array_path: &str, operation: &MathOperation) -> Result<String> {
        let output_path = self.temp_dir.join("result_array.npy");
        
        let script = match operation.operation_type.as_str() {
            "sum" => format!(
                r#"
import numpy as np

array = np.load('{}')
result = np.sum(array{})
np.save('{}', result)
print('{}')
"#,
                array_path,
                if let Some(axis) = operation.axis {
                    format!(", axis={}", axis)
                } else {
                    String::new()
                },
                output_path.display(),
                output_path.display()
            ),
            "mean" => format!(
                r#"
import numpy as np

array = np.load('{}')
result = np.mean(array{})
np.save('{}', result)
print('{}')
"#,
                array_path,
                if let Some(axis) = operation.axis {
                    format!(", axis={}", axis)
                } else {
                    String::new()
                },
                output_path.display(),
                output_path.display()
            ),
            "dot" => {
                let other_array = operation.parameters.get("other")
                    .ok_or_else(|| SynapticError::ValidationError("Dot operation requires 'other' parameter".to_string()))?;
                format!(
                    r#"
import numpy as np

array1 = np.load('{}')
array2 = np.array({})
result = np.dot(array1, array2)
np.save('{}', result)
print('{}')
"#,
                    array_path,
                    other_array,
                    output_path.display(),
                    output_path.display()
                )
            },
            "transpose" => format!(
                r#"
import numpy as np

array = np.load('{}')
result = np.transpose(array)
np.save('{}', result)
print('{}')
"#,
                array_path,
                output_path.display(),
                output_path.display()
            ),
            _ => return Err(SynapticError::ValidationError(
                format!("Unsupported math operation: {}", operation.operation_type)
            ))
        };

        Ok(script)
    }

    /// Execute Python script
    async fn execute_python_script(&self, script: &str) -> Result<String> {
        let script_path = self.temp_dir.join("script.py");
        fs::write(&script_path, script).await.map_err(|e| {
            SynapticError::IoError(format!("Failed to write script: {}", e))
        })?;

        let output = AsyncCommand::new(&self.python_path)
            .arg(&script_path)
            .output()
            .await
            .map_err(|e| {
                SynapticError::IntegrationError(format!("Failed to execute Python script: {}", e))
            })?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            error!("Python script failed: {}", stderr);
            return Err(SynapticError::IntegrationError(
                format!("Python script execution failed: {}", stderr)
            ));
        }

        let stdout = String::from_utf8_lossy(&output.stdout);
        debug!("Python script output: {}", stdout);
        Ok(stdout.trim().to_string())
    }
}
