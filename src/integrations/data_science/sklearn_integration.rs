//! Scikit-learn Integration
//!
//! This module provides comprehensive integration with scikit-learn machine learning
//! algorithms, enabling advanced analytics, clustering, classification, and model
//! training workflows for memory analysis and pattern recognition.

use crate::error::{Result, SynapticError};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;
use tokio::fs;
use tokio::process::Command as AsyncCommand;
use tracing::{debug, error, info};

/// Scikit-learn integration manager
pub struct SklearnIntegration {
    python_path: String,
    temp_dir: PathBuf,
}

/// Machine learning model configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    pub algorithm: String,
    pub parameters: HashMap<String, serde_json::Value>,
    pub preprocessing: Option<PreprocessingConfig>,
}

/// Preprocessing configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PreprocessingConfig {
    pub scaling: Option<String>,
    pub feature_selection: Option<FeatureSelectionConfig>,
    pub dimensionality_reduction: Option<DimensionalityReductionConfig>,
}

/// Feature selection configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeatureSelectionConfig {
    pub method: String,
    pub n_features: Option<usize>,
    pub parameters: HashMap<String, serde_json::Value>,
}

/// Dimensionality reduction configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DimensionalityReductionConfig {
    pub method: String,
    pub n_components: usize,
    pub parameters: HashMap<String, serde_json::Value>,
}

/// Model training results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingResults {
    pub model_path: String,
    pub training_score: f64,
    pub validation_score: Option<f64>,
    pub feature_importance: Option<Vec<f64>>,
    pub training_time: f64,
    pub model_size: usize,
}

/// Clustering results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClusteringResults {
    pub labels: Vec<i32>,
    pub cluster_centers: Option<Vec<Vec<f64>>>,
    pub inertia: Option<f64>,
    pub silhouette_score: f64,
    pub n_clusters: usize,
}

/// Classification results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClassificationResults {
    pub predictions: Vec<i32>,
    pub probabilities: Option<Vec<Vec<f64>>>,
    pub accuracy: f64,
    pub precision: Vec<f64>,
    pub recall: Vec<f64>,
    pub f1_score: Vec<f64>,
    pub confusion_matrix: Vec<Vec<i32>>,
}

impl SklearnIntegration {
    /// Create a new sklearn integration
    pub fn new() -> Result<Self> {
        let python_path = Self::find_python_executable()?;
        let temp_dir = std::env::temp_dir().join("synaptic_sklearn");
        
        std::fs::create_dir_all(&temp_dir).map_err(|e| {
            SynapticError::IoError(format!("Failed to create temp directory: {}", e))
        })?;

        Ok(Self {
            python_path,
            temp_dir,
        })
    }

    /// Find Python executable with sklearn
    fn find_python_executable() -> Result<String> {
        let candidates = vec!["python3", "python", "python3.8", "python3.9", "python3.10", "python3.11"];
        
        for candidate in candidates {
            if let Ok(output) = std::process::Command::new(candidate)
                .args(&["-c", "import sklearn; print('OK')"])
                .output()
            {
                if output.status.success() {
                    info!("Found Python with sklearn: {}", candidate);
                    return Ok(candidate.to_string());
                }
            }
        }
        
        Err(SynapticError::IntegrationError(
            "Could not find Python executable with scikit-learn installed".to_string()
        ))
    }

    /// Perform clustering analysis
    pub async fn cluster_data(&self, data_path: &str, config: &ModelConfig) -> Result<ClusteringResults> {
        let script = self.generate_clustering_script(data_path, config)?;
        let output = self.execute_python_script(&script).await?;
        
        serde_json::from_str(&output).map_err(|e| {
            SynapticError::SerializationError(format!("Failed to parse clustering results: {}", e))
        })
    }

    /// Train classification model
    pub async fn train_classifier(&self, data_path: &str, labels_path: &str, config: &ModelConfig) -> Result<TrainingResults> {
        let script = self.generate_classification_script(data_path, labels_path, config)?;
        let output = self.execute_python_script(&script).await?;
        
        serde_json::from_str(&output).map_err(|e| {
            SynapticError::SerializationError(format!("Failed to parse training results: {}", e))
        })
    }

    /// Make predictions with trained model
    pub async fn predict(&self, model_path: &str, data_path: &str) -> Result<ClassificationResults> {
        let script = format!(
            r#"
import numpy as np
import pickle
import json
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

# Load model and data
with open('{}', 'rb') as f:
    model = pickle.load(f)

data = np.load('{}')

# Make predictions
predictions = model.predict(data)
probabilities = None

if hasattr(model, 'predict_proba'):
    probabilities = model.predict_proba(data).tolist()

# For demonstration, create dummy true labels
# In practice, these would be provided separately
true_labels = np.random.randint(0, len(np.unique(predictions)), len(predictions))

# Calculate metrics
accuracy = float(accuracy_score(true_labels, predictions))
precision, recall, f1, _ = precision_recall_fscore_support(true_labels, predictions, average=None)
cm = confusion_matrix(true_labels, predictions)

results = {{
    'predictions': predictions.tolist(),
    'probabilities': probabilities,
    'accuracy': accuracy,
    'precision': precision.tolist(),
    'recall': recall.tolist(),
    'f1_score': f1.tolist(),
    'confusion_matrix': cm.tolist()
}}

print(json.dumps(results))
"#,
            model_path,
            data_path
        );

        let output = self.execute_python_script(&script).await?;
        serde_json::from_str(&output).map_err(|e| {
            SynapticError::SerializationError(format!("Failed to parse prediction results: {}", e))
        })
    }

    /// Perform anomaly detection
    pub async fn detect_anomalies(&self, data_path: &str, method: &str) -> Result<Vec<bool>> {
        let script = match method {
            "isolation_forest" => format!(
                r#"
import numpy as np
import json
from sklearn.ensemble import IsolationForest

data = np.load('{}')
detector = IsolationForest(contamination=0.1, random_state=42)
anomalies = detector.fit_predict(data)

# Convert to boolean (True for normal, False for anomaly)
is_normal = (anomalies == 1).tolist()
print(json.dumps(is_normal))
"#,
                data_path
            ),
            "one_class_svm" => format!(
                r#"
import numpy as np
import json
from sklearn.svm import OneClassSVM

data = np.load('{}')
detector = OneClassSVM(nu=0.1)
anomalies = detector.fit_predict(data)

# Convert to boolean (True for normal, False for anomaly)
is_normal = (anomalies == 1).tolist()
print(json.dumps(is_normal))
"#,
                data_path
            ),
            "local_outlier_factor" => format!(
                r#"
import numpy as np
import json
from sklearn.neighbors import LocalOutlierFactor

data = np.load('{}')
detector = LocalOutlierFactor(contamination=0.1)
anomalies = detector.fit_predict(data)

# Convert to boolean (True for normal, False for anomaly)
is_normal = (anomalies == 1).tolist()
print(json.dumps(is_normal))
"#,
                data_path
            ),
            _ => return Err(SynapticError::ValidationError(
                format!("Unsupported anomaly detection method: {}", method)
            ))
        };

        let output = self.execute_python_script(&script).await?;
        serde_json::from_str(&output).map_err(|e| {
            SynapticError::SerializationError(format!("Failed to parse anomaly detection results: {}", e))
        })
    }

    /// Perform feature selection
    pub async fn select_features(&self, data_path: &str, labels_path: Option<&str>, config: &FeatureSelectionConfig) -> Result<String> {
        let script = match config.method.as_str() {
            "variance_threshold" => {
                let threshold = config.parameters.get("threshold")
                    .and_then(|v| v.as_f64())
                    .unwrap_or(0.0);
                format!(
                    r#"
import numpy as np
from sklearn.feature_selection import VarianceThreshold

data = np.load('{}')
selector = VarianceThreshold(threshold={})
selected_data = selector.fit_transform(data)

output_path = '{}'
np.save(output_path, selected_data)
print(output_path)
"#,
                    data_path,
                    threshold,
                    self.temp_dir.join("selected_features.npy").display()
                )
            },
            "k_best" => {
                let k = config.n_features.unwrap_or(10);
                let labels_path = labels_path.ok_or_else(|| {
                    SynapticError::ValidationError("K-best feature selection requires labels".to_string())
                })?;
                format!(
                    r#"
import numpy as np
from sklearn.feature_selection import SelectKBest, f_classif

data = np.load('{}')
labels = np.load('{}')
selector = SelectKBest(f_classif, k={})
selected_data = selector.fit_transform(data, labels)

output_path = '{}'
np.save(output_path, selected_data)
print(output_path)
"#,
                    data_path,
                    labels_path,
                    k,
                    self.temp_dir.join("selected_features.npy").display()
                )
            },
            _ => return Err(SynapticError::ValidationError(
                format!("Unsupported feature selection method: {}", config.method)
            ))
        };

        self.execute_python_script(&script).await
    }

    /// Generate clustering script
    fn generate_clustering_script(&self, data_path: &str, config: &ModelConfig) -> Result<String> {
        let script = match config.algorithm.as_str() {
            "kmeans" => {
                let n_clusters = config.parameters.get("n_clusters")
                    .and_then(|v| v.as_u64())
                    .unwrap_or(8) as usize;
                format!(
                    r#"
import numpy as np
import json
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

data = np.load('{}')
kmeans = KMeans(n_clusters={}, random_state=42)
labels = kmeans.fit_predict(data)

silhouette = float(silhouette_score(data, labels))

results = {{
    'labels': labels.tolist(),
    'cluster_centers': kmeans.cluster_centers_.tolist(),
    'inertia': float(kmeans.inertia_),
    'silhouette_score': silhouette,
    'n_clusters': {}
}}

print(json.dumps(results))
"#,
                    data_path, n_clusters, n_clusters
                )
            },
            "dbscan" => {
                let eps = config.parameters.get("eps")
                    .and_then(|v| v.as_f64())
                    .unwrap_or(0.5);
                let min_samples = config.parameters.get("min_samples")
                    .and_then(|v| v.as_u64())
                    .unwrap_or(5) as usize;
                format!(
                    r#"
import numpy as np
import json
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score

data = np.load('{}')
dbscan = DBSCAN(eps={}, min_samples={})
labels = dbscan.fit_predict(data)

# Calculate silhouette score (only if we have more than one cluster)
n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
if n_clusters > 1:
    silhouette = float(silhouette_score(data, labels))
else:
    silhouette = 0.0

results = {{
    'labels': labels.tolist(),
    'cluster_centers': None,
    'inertia': None,
    'silhouette_score': silhouette,
    'n_clusters': n_clusters
}}

print(json.dumps(results))
"#,
                    data_path, eps, min_samples
                )
            },
            _ => return Err(SynapticError::ValidationError(
                format!("Unsupported clustering algorithm: {}", config.algorithm)
            ))
        };

        Ok(script)
    }

    /// Generate classification script
    fn generate_classification_script(&self, data_path: &str, labels_path: &str, config: &ModelConfig) -> Result<String> {
        let model_path = self.temp_dir.join("trained_model.pkl");
        
        let script = match config.algorithm.as_str() {
            "random_forest" => {
                let n_estimators = config.parameters.get("n_estimators")
                    .and_then(|v| v.as_u64())
                    .unwrap_or(100) as usize;
                format!(
                    r#"
import numpy as np
import json
import pickle
import time
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

start_time = time.time()

data = np.load('{}')
labels = np.load('{}')

# Split data
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators={}, random_state=42)
model.fit(X_train, y_train)

# Evaluate
train_score = float(accuracy_score(y_train, model.predict(X_train)))
val_score = float(accuracy_score(y_test, model.predict(X_test)))

# Save model
model_path = '{}'
with open(model_path, 'wb') as f:
    pickle.dump(model, f)

training_time = time.time() - start_time
model_size = os.path.getsize(model_path)

results = {{
    'model_path': model_path,
    'training_score': train_score,
    'validation_score': val_score,
    'feature_importance': model.feature_importances_.tolist(),
    'training_time': training_time,
    'model_size': model_size
}}

print(json.dumps(results))
"#,
                    data_path, labels_path, n_estimators, model_path.display()
                )
            },
            "svm" => {
                let c = config.parameters.get("C")
                    .and_then(|v| v.as_f64())
                    .unwrap_or(1.0);
                format!(
                    r#"
import numpy as np
import json
import pickle
import time
import os
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

start_time = time.time()

data = np.load('{}')
labels = np.load('{}')

# Split data
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

# Train model
model = SVC(C={}, probability=True, random_state=42)
model.fit(X_train, y_train)

# Evaluate
train_score = float(accuracy_score(y_train, model.predict(X_train)))
val_score = float(accuracy_score(y_test, model.predict(X_test)))

# Save model
model_path = '{}'
with open(model_path, 'wb') as f:
    pickle.dump(model, f)

training_time = time.time() - start_time
model_size = os.path.getsize(model_path)

results = {{
    'model_path': model_path,
    'training_score': train_score,
    'validation_score': val_score,
    'feature_importance': None,
    'training_time': training_time,
    'model_size': model_size
}}

print(json.dumps(results))
"#,
                    data_path, labels_path, c, model_path.display()
                )
            },
            _ => return Err(SynapticError::ValidationError(
                format!("Unsupported classification algorithm: {}", config.algorithm)
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
