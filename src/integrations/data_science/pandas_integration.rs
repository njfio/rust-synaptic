//! Pandas Integration
//!
//! This module provides comprehensive integration with pandas DataFrames,
//! enabling seamless data exchange, transformation, and analysis workflows
//! between Synaptic memory system and pandas-based data science pipelines.

use crate::error::{Result, SynapticError};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;
use std::process::Command;
use tokio::fs;
use tokio::process::Command as AsyncCommand;
use tracing::{debug, error, info, warn};

/// Pandas integration manager
pub struct PandasIntegration {
    python_path: String,
    temp_dir: PathBuf,
}

/// DataFrame metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataFrameInfo {
    pub shape: (usize, usize),
    pub columns: Vec<String>,
    pub dtypes: HashMap<String, String>,
    pub memory_usage: usize,
    pub null_counts: HashMap<String, usize>,
}

/// Data transformation configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TransformConfig {
    pub operations: Vec<TransformOperation>,
    pub output_format: String,
    pub preserve_index: bool,
}

/// Individual transformation operation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TransformOperation {
    pub operation_type: String,
    pub column: Option<String>,
    pub parameters: HashMap<String, serde_json::Value>,
}

impl PandasIntegration {
    /// Create a new pandas integration
    pub fn new() -> Result<Self> {
        let python_path = Self::find_python_executable()?;
        let temp_dir = std::env::temp_dir().join("synaptic_pandas");
        
        std::fs::create_dir_all(&temp_dir).map_err(|e| {
            SynapticError::IoError(format!("Failed to create temp directory: {}", e))
        })?;

        Ok(Self {
            python_path,
            temp_dir,
        })
    }

    /// Find Python executable with pandas
    fn find_python_executable() -> Result<String> {
        let candidates = vec!["python3", "python", "python3.8", "python3.9", "python3.10", "python3.11"];
        
        for candidate in candidates {
            if let Ok(output) = Command::new(candidate)
                .args(&["-c", "import pandas; print('OK')"])
                .output()
            {
                if output.status.success() {
                    info!("Found Python with pandas: {}", candidate);
                    return Ok(candidate.to_string());
                }
            }
        }
        
        Err(SynapticError::IntegrationError(
            "Could not find Python executable with pandas installed".to_string()
        ))
    }

    /// Convert JSON data to pandas DataFrame
    pub async fn json_to_dataframe(&self, data: &serde_json::Value) -> Result<String> {
        let script = self.create_conversion_script(data, "json_to_df").await?;
        self.execute_python_script(&script).await
    }

    /// Convert pandas DataFrame to JSON
    pub async fn dataframe_to_json(&self, df_path: &str) -> Result<serde_json::Value> {
        let script = format!(
            r#"
import pandas as pd
import json

df = pd.read_pickle('{}')
result = df.to_json(orient='records')
print(result)
"#,
            df_path
        );

        let output = self.execute_python_script(&script).await?;
        serde_json::from_str(&output).map_err(|e| {
            SynapticError::SerializationError(format!("Failed to parse JSON output: {}", e))
        })
    }

    /// Get DataFrame information
    pub async fn get_dataframe_info(&self, df_path: &str) -> Result<DataFrameInfo> {
        let script = format!(
            r#"
import pandas as pd
import json

df = pd.read_pickle('{}')

info = {{
    'shape': df.shape,
    'columns': df.columns.tolist(),
    'dtypes': {{col: str(dtype) for col, dtype in df.dtypes.items()}},
    'memory_usage': int(df.memory_usage(deep=True).sum()),
    'null_counts': df.isnull().sum().to_dict()
}}

print(json.dumps(info))
"#,
            df_path
        );

        let output = self.execute_python_script(&script).await?;
        serde_json::from_str(&output).map_err(|e| {
            SynapticError::SerializationError(format!("Failed to parse DataFrame info: {}", e))
        })
    }

    /// Apply transformations to DataFrame
    pub async fn transform_dataframe(&self, df_path: &str, config: &TransformConfig) -> Result<String> {
        let mut script = format!(
            r#"
import pandas as pd
import numpy as np

df = pd.read_pickle('{}')
"#,
            df_path
        );

        // Apply each transformation operation
        for operation in &config.operations {
            let op_script = self.generate_operation_script(operation)?;
            script.push_str(&op_script);
            script.push('\n');
        }

        // Save transformed DataFrame
        let output_path = self.temp_dir.join("transformed_df.pkl");
        script.push_str(&format!(
            "df.to_pickle('{}')\nprint('{}')\n",
            output_path.display(),
            output_path.display()
        ));

        self.execute_python_script(&script).await?;
        Ok(output_path.to_string_lossy().to_string())
    }

    /// Export DataFrame to CSV
    pub async fn export_csv(&self, data: &serde_json::Value, path: &PathBuf) -> Result<()> {
        let script = format!(
            r#"
import pandas as pd
import json

data = json.loads('{}')
df = pd.DataFrame(data)
df.to_csv('{}', index=False)
print('CSV exported successfully')
"#,
            serde_json::to_string(data).map_err(|e| {
                SynapticError::SerializationError(format!("Failed to serialize data: {}", e))
            })?,
            path.display()
        );

        self.execute_python_script(&script).await?;
        info!("Exported data to CSV: {}", path.display());
        Ok(())
    }

    /// Import DataFrame from CSV
    pub async fn import_csv(&self, path: &PathBuf) -> Result<serde_json::Value> {
        let script = format!(
            r#"
import pandas as pd
import json

df = pd.read_csv('{}')
result = df.to_json(orient='records')
print(result)
"#,
            path.display()
        );

        let output = self.execute_python_script(&script).await?;
        serde_json::from_str(&output).map_err(|e| {
            SynapticError::SerializationError(format!("Failed to parse CSV data: {}", e))
        })
    }

    /// Export DataFrame to Parquet
    pub async fn export_parquet(&self, data: &serde_json::Value, path: &PathBuf) -> Result<()> {
        let script = format!(
            r#"
import pandas as pd
import json

data = json.loads('{}')
df = pd.DataFrame(data)
df.to_parquet('{}', index=False)
print('Parquet exported successfully')
"#,
            serde_json::to_string(data).map_err(|e| {
                SynapticError::SerializationError(format!("Failed to serialize data: {}", e))
            })?,
            path.display()
        );

        self.execute_python_script(&script).await?;
        info!("Exported data to Parquet: {}", path.display());
        Ok(())
    }

    /// Import DataFrame from Parquet
    pub async fn import_parquet(&self, path: &PathBuf) -> Result<serde_json::Value> {
        let script = format!(
            r#"
import pandas as pd
import json

df = pd.read_parquet('{}')
result = df.to_json(orient='records')
print(result)
"#,
            path.display()
        );

        let output = self.execute_python_script(&script).await?;
        serde_json::from_str(&output).map_err(|e| {
            SynapticError::SerializationError(format!("Failed to parse Parquet data: {}", e))
        })
    }

    /// Perform statistical analysis
    pub async fn analyze_dataframe(&self, df_path: &str) -> Result<serde_json::Value> {
        let script = format!(
            r#"
import pandas as pd
import json

df = pd.read_pickle('{}')

analysis = {{
    'describe': df.describe().to_dict(),
    'info': {{
        'shape': df.shape,
        'columns': df.columns.tolist(),
        'dtypes': {{col: str(dtype) for col, dtype in df.dtypes.items()}},
        'memory_usage': int(df.memory_usage(deep=True).sum())
    }},
    'correlations': df.select_dtypes(include=[np.number]).corr().to_dict() if len(df.select_dtypes(include=[np.number]).columns) > 0 else {{}},
    'null_counts': df.isnull().sum().to_dict(),
    'value_counts': {{col: df[col].value_counts().head(10).to_dict() for col in df.select_dtypes(include=['object']).columns}}
}}

print(json.dumps(analysis, default=str))
"#,
            df_path
        );

        let output = self.execute_python_script(&script).await?;
        serde_json::from_str(&output).map_err(|e| {
            SynapticError::SerializationError(format!("Failed to parse analysis results: {}", e))
        })
    }

    /// Create conversion script
    async fn create_conversion_script(&self, data: &serde_json::Value, operation: &str) -> Result<String> {
        match operation {
            "json_to_df" => {
                let output_path = self.temp_dir.join("dataframe.pkl");
                Ok(format!(
                    r#"
import pandas as pd
import json

data = json.loads('{}')
df = pd.DataFrame(data)
df.to_pickle('{}')
print('{}')
"#,
                    serde_json::to_string(data).map_err(|e| {
                        SynapticError::SerializationError(format!("Failed to serialize data: {}", e))
                    })?,
                    output_path.display(),
                    output_path.display()
                ))
            }
            _ => Err(SynapticError::ValidationError(
                format!("Unknown conversion operation: {}", operation)
            ))
        }
    }

    /// Generate script for transformation operation
    fn generate_operation_script(&self, operation: &TransformOperation) -> Result<String> {
        match operation.operation_type.as_str() {
            "filter" => {
                let column = operation.column.as_ref().ok_or_else(|| {
                    SynapticError::ValidationError("Filter operation requires column".to_string())
                })?;
                let condition = operation.parameters.get("condition").ok_or_else(|| {
                    SynapticError::ValidationError("Filter operation requires condition".to_string())
                })?;
                Ok(format!("df = df[df['{}'] {}]", column, condition))
            }
            "sort" => {
                let column = operation.column.as_ref().ok_or_else(|| {
                    SynapticError::ValidationError("Sort operation requires column".to_string())
                })?;
                let ascending = operation.parameters.get("ascending")
                    .and_then(|v| v.as_bool())
                    .unwrap_or(true);
                Ok(format!("df = df.sort_values('{}', ascending={})", column, ascending))
            }
            "group_by" => {
                let column = operation.column.as_ref().ok_or_else(|| {
                    SynapticError::ValidationError("GroupBy operation requires column".to_string())
                })?;
                let agg_func = operation.parameters.get("function")
                    .and_then(|v| v.as_str())
                    .unwrap_or("count");
                Ok(format!("df = df.groupby('{}').{}().reset_index()", column, agg_func))
            }
            "drop_na" => {
                Ok("df = df.dropna()".to_string())
            }
            "fill_na" => {
                let value = operation.parameters.get("value")
                    .and_then(|v| v.as_str())
                    .unwrap_or("0");
                Ok(format!("df = df.fillna('{}')", value))
            }
            _ => Err(SynapticError::ValidationError(
                format!("Unknown transformation operation: {}", operation.operation_type)
            ))
        }
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
