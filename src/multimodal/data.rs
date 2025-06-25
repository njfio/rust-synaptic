//! # Data Memory Processor
//!
//! Advanced data file processing capabilities for CSV, Parquet, Excel, and other structured data formats.
//! Provides intelligent data analysis, schema detection, and statistical insights.

use super::{ContentType, MultiModalMemory, MultiModalMetadata, MultiModalProcessor, MultiModalResult};
use crate::error::SynapticError;
use crate::memory::types::MemoryId;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use uuid::Uuid;

/// Data formats supported by the data processor
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum DataFormat {
    /// CSV (Comma-Separated Values)
    Csv {
        delimiter: char,
        has_header: bool,
        encoding: String,
    },
    /// Parquet columnar format
    Parquet {
        version: String,
        compression: String,
        schema_version: u32,
    },
    /// Excel spreadsheet
    Excel {
        sheets: Vec<String>,
        version: String,
        has_macros: bool,
    },
    /// JSON data
    Json {
        is_array: bool,
        nested_levels: u32,
        object_count: u32,
    },
    /// TSV (Tab-Separated Values)
    Tsv {
        has_header: bool,
        encoding: String,
    },
    /// Apache Arrow format
    Arrow {
        schema_version: u32,
        compression: String,
    },
}

/// Data schema information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataSchema {
    /// Column definitions
    pub columns: Vec<ColumnInfo>,
    /// Total number of rows
    pub row_count: u64,
    /// Total number of columns
    pub column_count: u32,
    /// Data types present
    pub data_types: Vec<String>,
    /// Primary key columns (if detected)
    pub primary_keys: Vec<String>,
    /// Foreign key relationships
    pub foreign_keys: Vec<ForeignKeyInfo>,
}

/// Column information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ColumnInfo {
    /// Column name
    pub name: String,
    /// Data type
    pub data_type: DataType,
    /// Whether column allows null values
    pub nullable: bool,
    /// Unique value count
    pub unique_count: u64,
    /// Null value count
    pub null_count: u64,
    /// Sample values
    pub sample_values: Vec<String>,
    /// Statistical summary
    pub statistics: ColumnStatistics,
}

/// Data types for columns
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum DataType {
    Integer,
    Float,
    String,
    Boolean,
    Date,
    DateTime,
    Time,
    Decimal,
    Binary,
    Json,
    Array,
    Unknown,
}

/// Statistical information for columns
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ColumnStatistics {
    /// Minimum value (for numeric columns)
    pub min: Option<f64>,
    /// Maximum value (for numeric columns)
    pub max: Option<f64>,
    /// Mean value (for numeric columns)
    pub mean: Option<f64>,
    /// Standard deviation (for numeric columns)
    pub std_dev: Option<f64>,
    /// Median value (for numeric columns)
    pub median: Option<f64>,
    /// Most frequent value
    pub mode: Option<String>,
    /// Value distribution (top 10 values with counts)
    pub value_distribution: Vec<(String, u64)>,
}

/// Foreign key relationship information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ForeignKeyInfo {
    /// Source column
    pub source_column: String,
    /// Referenced table/dataset
    pub referenced_table: String,
    /// Referenced column
    pub referenced_column: String,
    /// Relationship type
    pub relationship_type: RelationshipType,
}

/// Types of relationships between data
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum RelationshipType {
    OneToOne,
    OneToMany,
    ManyToOne,
    ManyToMany,
}

/// Data-specific metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataMetadata {
    /// Data format details
    pub format: DataFormat,
    /// Data schema
    pub schema: DataSchema,
    /// Data quality score (0.0 to 1.0)
    pub quality_score: f64,
    /// Completeness percentage
    pub completeness: f64,
    /// Consistency score
    pub consistency: f64,
    /// Data freshness (last updated)
    pub last_updated: Option<chrono::DateTime<chrono::Utc>>,
    /// Data source information
    pub source: Option<String>,
    /// Data lineage
    pub lineage: Vec<String>,
    /// Data tags and categories
    pub categories: Vec<String>,
    /// Detected patterns
    pub patterns: Vec<DataPattern>,
    /// Data summary
    pub summary: DataSummary,
}

/// Detected data patterns
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataPattern {
    /// Pattern type
    pub pattern_type: PatternType,
    /// Pattern description
    pub description: String,
    /// Confidence score
    pub confidence: f64,
    /// Affected columns
    pub columns: Vec<String>,
}

/// Types of data patterns
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum PatternType {
    Trend,
    Seasonality,
    Outlier,
    Missing,
    Duplicate,
    Correlation,
    Distribution,
}

/// Data summary information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataSummary {
    /// Brief description of the dataset
    pub description: String,
    /// Key insights
    pub insights: Vec<String>,
    /// Recommended actions
    pub recommendations: Vec<String>,
    /// Data quality issues
    pub quality_issues: Vec<String>,
}

/// Data memory processor for structured data files
#[derive(Debug, Clone)]
pub struct DataMemoryProcessor {
    /// Configuration for data processing
    config: DataProcessorConfig,
}

/// Configuration for data processing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataProcessorConfig {
    /// Maximum file size to process (bytes)
    pub max_file_size: usize,
    /// Maximum number of rows to analyze for schema detection
    pub max_sample_rows: usize,
    /// Enable statistical analysis
    pub enable_statistics: bool,
    /// Enable pattern detection
    pub enable_pattern_detection: bool,
    /// Enable data quality assessment
    pub enable_quality_assessment: bool,
    /// Sample size for value distribution analysis
    pub sample_size: usize,
    /// Supported data formats
    pub supported_formats: Vec<String>,
}

impl Default for DataProcessorConfig {
    fn default() -> Self {
        Self {
            max_file_size: 500 * 1024 * 1024, // 500MB
            max_sample_rows: 10000,
            enable_statistics: true,
            enable_pattern_detection: true,
            enable_quality_assessment: true,
            sample_size: 1000,
            supported_formats: vec![
                "csv".to_string(),
                "tsv".to_string(),
                "parquet".to_string(),
                "excel".to_string(),
                "json".to_string(),
                "arrow".to_string(),
            ],
        }
    }
}

impl DataMemoryProcessor {
    /// Create a new data processor
    pub fn new(config: DataProcessorConfig) -> Self {
        Self { config }
    }

    /// Create with default configuration
    pub fn default() -> Self {
        Self::new(DataProcessorConfig::default())
    }

    /// Detect data format from content
    pub fn detect_data_format(&self, content: &[u8]) -> MultiModalResult<DataFormat> {
        // Check file signatures and content patterns
        if content.starts_with(b"PAR1") {
            Ok(DataFormat::Parquet {
                version: "1.0".to_string(),
                compression: "snappy".to_string(),
                schema_version: 1,
            })
        } else if content.starts_with(b"PK\x03\x04") {
            // Could be Excel file
            Ok(DataFormat::Excel {
                sheets: vec!["Sheet1".to_string()],
                version: "xlsx".to_string(),
                has_macros: false,
            })
        } else if content.starts_with(b"ARROW1") {
            Ok(DataFormat::Arrow {
                schema_version: 1,
                compression: "none".to_string(),
            })
        } else {
            // Try to detect CSV/TSV/JSON from content
            let text = String::from_utf8_lossy(content);
            let first_line = text.lines().next().unwrap_or("");
            
            if first_line.starts_with('{') || first_line.starts_with('[') {
                Ok(DataFormat::Json {
                    is_array: first_line.starts_with('['),
                    nested_levels: 1,
                    object_count: 0,
                })
            } else if first_line.contains('\t') {
                Ok(DataFormat::Tsv {
                    has_header: self.detect_header(&text),
                    encoding: "UTF-8".to_string(),
                })
            } else {
                // Default to CSV
                let delimiter = self.detect_csv_delimiter(&text);
                Ok(DataFormat::Csv {
                    delimiter,
                    has_header: self.detect_header(&text),
                    encoding: "UTF-8".to_string(),
                })
            }
        }
    }

    /// Detect CSV delimiter
    fn detect_csv_delimiter(&self, text: &str) -> char {
        let delimiters = [',', ';', '|', '\t'];
        let first_lines: Vec<&str> = text.lines().take(5).collect();
        
        let mut best_delimiter = ',';
        let mut best_consistency = 0;
        
        for &delimiter in &delimiters {
            let counts: Vec<usize> = first_lines.iter()
                .map(|line| line.matches(delimiter).count())
                .collect();
            
            if let Some(&first_count) = counts.first() {
                let consistency = counts.iter().filter(|&&count| count == first_count).count();
                if consistency > best_consistency && first_count > 0 {
                    best_consistency = consistency;
                    best_delimiter = delimiter;
                }
            }
        }
        
        best_delimiter
    }

    /// Detect if first row is a header
    fn detect_header(&self, text: &str) -> bool {
        let lines: Vec<&str> = text.lines().take(2).collect();
        if lines.len() < 2 {
            return false;
        }
        
        let first_line = lines[0];
        let second_line = lines[1];
        
        // Simple heuristic: if first line has more text and second line has more numbers
        let first_has_text = first_line.split(',').any(|field| {
            field.trim().chars().any(|c| c.is_alphabetic())
        });
        
        let second_has_numbers = second_line.split(',').any(|field| {
            field.trim().parse::<f64>().is_ok()
        });
        
        first_has_text && second_has_numbers
    }

    /// Analyze data schema
    pub async fn analyze_schema(&self, content: &[u8], format: &DataFormat) -> MultiModalResult<DataSchema> {
        match format {
            DataFormat::Csv { delimiter, has_header, .. } => {
                self.analyze_csv_schema(content, *delimiter, *has_header).await
            }
            DataFormat::Tsv { has_header, .. } => {
                self.analyze_csv_schema(content, '\t', *has_header).await
            }
            DataFormat::Json { .. } => {
                self.analyze_json_schema(content).await
            }
            _ => {
                // For other formats, return basic schema
                Ok(DataSchema {
                    columns: Vec::new(),
                    row_count: 0,
                    column_count: 0,
                    data_types: Vec::new(),
                    primary_keys: Vec::new(),
                    foreign_keys: Vec::new(),
                })
            }
        }
    }

    /// Analyze CSV schema
    async fn analyze_csv_schema(&self, content: &[u8], delimiter: char, has_header: bool) -> MultiModalResult<DataSchema> {
        let text = String::from_utf8_lossy(content);
        let lines: Vec<&str> = text.lines().collect();
        
        if lines.is_empty() {
            return Ok(DataSchema {
                columns: Vec::new(),
                row_count: 0,
                column_count: 0,
                data_types: Vec::new(),
                primary_keys: Vec::new(),
                foreign_keys: Vec::new(),
            });
        }
        
        let header_line = if has_header { 0 } else { usize::MAX };
        let data_start = if has_header { 1 } else { 0 };
        
        // Parse header or generate column names
        let column_names: Vec<String> = if has_header && !lines.is_empty() {
            lines[0].split(delimiter)
                .map(|s| s.trim().to_string())
                .collect()
        } else if !lines.is_empty() {
            let field_count = lines[0].split(delimiter).count();
            (0..field_count).map(|i| format!("column_{}", i)).collect()
        } else {
            Vec::new()
        };
        
        let column_count = column_names.len() as u32;
        let row_count = (lines.len() - data_start) as u64;
        
        // Analyze each column
        let mut columns = Vec::new();
        for (col_idx, col_name) in column_names.iter().enumerate() {
            let column_values: Vec<String> = lines.iter()
                .skip(data_start)
                .take(self.config.max_sample_rows)
                .filter_map(|line| {
                    let fields: Vec<&str> = line.split(delimiter).collect();
                    fields.get(col_idx).map(|s| s.trim().to_string())
                })
                .collect();
            
            let column_info = self.analyze_column(&column_values, col_name).await?;
            columns.push(column_info);
        }
        
        let data_types: Vec<String> = columns.iter()
            .map(|col| format!("{:?}", col.data_type))
            .collect();
        
        Ok(DataSchema {
            columns,
            row_count,
            column_count,
            data_types,
            primary_keys: Vec::new(), // Would need more sophisticated analysis
            foreign_keys: Vec::new(),
        })
    }

    /// Analyze JSON schema
    async fn analyze_json_schema(&self, _content: &[u8]) -> MultiModalResult<DataSchema> {
        // Basic JSON schema analysis (would need proper JSON parsing)
        Ok(DataSchema {
            columns: Vec::new(),
            row_count: 0,
            column_count: 0,
            data_types: vec!["JSON".to_string()],
            primary_keys: Vec::new(),
            foreign_keys: Vec::new(),
        })
    }

    /// Analyze individual column
    async fn analyze_column(&self, values: &[String], name: &str) -> MultiModalResult<ColumnInfo> {
        let total_count = values.len() as u64;
        let non_empty_values: Vec<&String> = values.iter().filter(|v| !v.is_empty()).collect();
        let null_count = total_count - non_empty_values.len() as u64;

        // Detect data type
        let data_type = self.detect_data_type(&non_empty_values);

        // Calculate unique count
        let mut unique_values = std::collections::HashSet::new();
        for value in &non_empty_values {
            unique_values.insert(value.as_str());
        }
        let unique_count = unique_values.len() as u64;

        // Get sample values
        let sample_values: Vec<String> = non_empty_values.iter()
            .take(5)
            .map(|s| s.to_string())
            .collect();

        // Calculate statistics
        let statistics = self.calculate_column_statistics(&non_empty_values, &data_type).await?;

        Ok(ColumnInfo {
            name: name.to_string(),
            data_type,
            nullable: null_count > 0,
            unique_count,
            null_count,
            sample_values,
            statistics,
        })
    }

    /// Detect data type of column values
    fn detect_data_type(&self, values: &[&String]) -> DataType {
        if values.is_empty() {
            return DataType::Unknown;
        }

        let sample_size = std::cmp::min(values.len(), 100);
        let sample = &values[..sample_size];

        // Check for integers
        let int_count = sample.iter().filter(|v| v.parse::<i64>().is_ok()).count();
        if int_count as f64 / sample.len() as f64 > 0.8 {
            return DataType::Integer;
        }

        // Check for floats
        let float_count = sample.iter().filter(|v| v.parse::<f64>().is_ok()).count();
        if float_count as f64 / sample.len() as f64 > 0.8 {
            return DataType::Float;
        }

        // Check for booleans
        let bool_count = sample.iter().filter(|v| {
            let lower = v.to_lowercase();
            matches!(lower.as_str(), "true" | "false" | "1" | "0" | "yes" | "no")
        }).count();
        if bool_count as f64 / sample.len() as f64 > 0.8 {
            return DataType::Boolean;
        }

        // Check for dates
        let date_count = sample.iter().filter(|v| {
            chrono::DateTime::parse_from_rfc3339(v).is_ok() ||
            chrono::NaiveDate::parse_from_str(v, "%Y-%m-%d").is_ok() ||
            chrono::NaiveDate::parse_from_str(v, "%m/%d/%Y").is_ok()
        }).count();
        if date_count as f64 / sample.len() as f64 > 0.8 {
            return DataType::Date;
        }

        // Default to string
        DataType::String
    }

    /// Calculate column statistics
    async fn calculate_column_statistics(&self, values: &[&String], data_type: &DataType) -> MultiModalResult<ColumnStatistics> {
        let mut stats = ColumnStatistics {
            min: None,
            max: None,
            mean: None,
            std_dev: None,
            median: None,
            mode: None,
            value_distribution: Vec::new(),
        };

        if values.is_empty() {
            return Ok(stats);
        }

        // Calculate value distribution
        let mut value_counts = HashMap::new();
        for value in values {
            *value_counts.entry(value.as_str()).or_insert(0) += 1;
        }

        let mut distribution: Vec<(String, u64)> = value_counts.into_iter()
            .map(|(k, v)| (k.to_string(), v))
            .collect();
        distribution.sort_by(|a, b| b.1.cmp(&a.1));
        stats.value_distribution = distribution.into_iter().take(10).collect();

        // Find mode
        if let Some((mode_value, _)) = stats.value_distribution.first() {
            stats.mode = Some(mode_value.clone());
        }

        // Calculate numeric statistics if applicable
        match data_type {
            DataType::Integer | DataType::Float => {
                let numeric_values: Vec<f64> = values.iter()
                    .filter_map(|v| v.parse::<f64>().ok())
                    .collect();

                if !numeric_values.is_empty() {
                    stats.min = numeric_values.iter().min_by(|a, b| {
                        a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)
                    }).copied();
                    stats.max = numeric_values.iter().max_by(|a, b| {
                        a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)
                    }).copied();

                    let sum: f64 = numeric_values.iter().sum();
                    stats.mean = Some(sum / numeric_values.len() as f64);

                    if let Some(mean) = stats.mean {
                        let variance: f64 = numeric_values.iter()
                            .map(|x| (x - mean).powi(2))
                            .sum::<f64>() / numeric_values.len() as f64;
                        stats.std_dev = Some(variance.sqrt());
                    }

                    // Calculate median
                    let mut sorted_values = numeric_values.clone();
                    sorted_values.sort_by(|a, b| {
                        a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)
                    });
                    let len = sorted_values.len();
                    stats.median = if len % 2 == 0 {
                        Some((sorted_values[len / 2 - 1] + sorted_values[len / 2]) / 2.0)
                    } else {
                        Some(sorted_values[len / 2])
                    };
                }
            }
            _ => {}
        }

        Ok(stats)
    }

    /// Assess data quality
    pub async fn assess_data_quality(&self, schema: &DataSchema) -> MultiModalResult<f64> {
        if schema.columns.is_empty() {
            return Ok(0.0);
        }

        let mut quality_scores = Vec::new();

        for column in &schema.columns {
            let completeness = if schema.row_count > 0 {
                1.0 - (column.null_count as f64 / schema.row_count as f64)
            } else {
                1.0
            };

            let uniqueness = if schema.row_count > 0 {
                column.unique_count as f64 / schema.row_count as f64
            } else {
                1.0
            };

            // Simple quality score based on completeness and uniqueness
            let column_quality = (completeness * 0.7) + (uniqueness.min(1.0) * 0.3);
            quality_scores.push(column_quality);
        }

        let average_quality = quality_scores.iter().sum::<f64>() / quality_scores.len() as f64;
        Ok(average_quality)
    }

    /// Detect data patterns
    #[tracing::instrument(skip(self, schema), fields(row_count = schema.row_count, column_count = schema.columns.len()))]
    pub async fn detect_patterns(&self, schema: &DataSchema) -> MultiModalResult<Vec<DataPattern>> {
        tracing::debug!("Starting pattern detection for schema with {} rows and {} columns", schema.row_count, schema.columns.len());
        let mut patterns = Vec::new();

        // Check for missing data patterns
        for column in &schema.columns {
            if column.null_count > 0 {
                let missing_percentage = column.null_count as f64 / schema.row_count as f64;
                if missing_percentage > 0.1 {
                    patterns.push(DataPattern {
                        pattern_type: PatternType::Missing,
                        description: format!("Column '{}' has {:.1}% missing values", column.name, missing_percentage * 100.0),
                        confidence: 0.9,
                        columns: vec![column.name.clone()],
                    });
                }
            }
        }

        // Check for low uniqueness (potential duplicates)
        for column in &schema.columns {
            if schema.row_count > 0 {
                let uniqueness = column.unique_count as f64 / schema.row_count as f64;
                if uniqueness < 0.1 && column.unique_count > 1 {
                    patterns.push(DataPattern {
                        pattern_type: PatternType::Duplicate,
                        description: format!("Column '{}' has low uniqueness ({:.1}%)", column.name, uniqueness * 100.0),
                        confidence: 0.8,
                        columns: vec![column.name.clone()],
                    });
                }
            }
        }

        tracing::info!("Pattern detection completed: found {} patterns", patterns.len());
        Ok(patterns)
    }

    /// Generate data summary
    pub async fn generate_summary(&self, metadata: &DataMetadata) -> MultiModalResult<DataSummary> {
        let schema = &metadata.schema;

        let description = format!(
            "Dataset with {} rows and {} columns. Data types include: {}.",
            schema.row_count,
            schema.column_count,
            schema.data_types.join(", ")
        );

        let mut insights = Vec::new();

        // Add insights based on data characteristics
        if schema.row_count > 10000 {
            insights.push("Large dataset suitable for statistical analysis".to_string());
        }

        if schema.columns.iter().any(|col| matches!(col.data_type, DataType::Date | DataType::DateTime)) {
            insights.push("Contains temporal data suitable for time series analysis".to_string());
        }

        let numeric_columns = schema.columns.iter()
            .filter(|col| matches!(col.data_type, DataType::Integer | DataType::Float))
            .count();
        if numeric_columns > 0 {
            insights.push(format!("Contains {} numeric columns suitable for statistical analysis", numeric_columns));
        }

        let mut recommendations = Vec::new();
        let mut quality_issues = Vec::new();

        // Check for quality issues and recommendations
        for column in &schema.columns {
            if column.null_count > 0 {
                let missing_percentage = column.null_count as f64 / schema.row_count as f64;
                if missing_percentage > 0.2 {
                    quality_issues.push(format!("Column '{}' has high missing data ({:.1}%)", column.name, missing_percentage * 100.0));
                    recommendations.push(format!("Consider imputation or removal of column '{}'", column.name));
                }
            }
        }

        if quality_issues.is_empty() {
            insights.push("Good data quality with minimal missing values".to_string());
        }

        Ok(DataSummary {
            description,
            insights,
            recommendations,
            quality_issues,
        })
    }
}

#[async_trait::async_trait]
impl MultiModalProcessor for DataMemoryProcessor {
    async fn process(&self, content: &[u8], content_type: &ContentType) -> MultiModalResult<MultiModalMemory> {
        let start_time = std::time::Instant::now();
        if content.len() > self.config.max_file_size {
            return Err(SynapticError::ProcessingError(
                format!("Data file size {} exceeds maximum {}", content.len(), self.config.max_file_size)
            ));
        }

        // Detect data format
        let format = self.detect_data_format(content)?;

        // Analyze schema
        let schema = self.analyze_schema(content, &format).await?;

        // Assess data quality
        let quality_score = if self.config.enable_quality_assessment {
            self.assess_data_quality(&schema).await?
        } else {
            0.8 // Default quality score
        };

        // Detect patterns
        let patterns = if self.config.enable_pattern_detection {
            self.detect_patterns(&schema).await?
        } else {
            Vec::new()
        };

        // Create data metadata
        let data_metadata = DataMetadata {
            format,
            schema,
            quality_score,
            completeness: quality_score, // Simplified
            consistency: quality_score,  // Simplified
            last_updated: Some(chrono::Utc::now()),
            source: None,
            lineage: Vec::new(),
            categories: Vec::new(),
            patterns,
            summary: DataSummary {
                description: "Data file processed".to_string(),
                insights: Vec::new(),
                recommendations: Vec::new(),
                quality_issues: Vec::new(),
            },
        };

        // Generate summary
        let summary = self.generate_summary(&data_metadata).await?;
        let mut final_metadata = data_metadata;
        final_metadata.summary = summary;

        let processing_time = start_time.elapsed().as_millis() as u64;

        // Create multi-modal metadata
        let metadata = MultiModalMetadata {
            title: Some(format!("Data file with {} rows", final_metadata.schema.row_count)),
            description: Some(final_metadata.summary.description.clone()),
            tags: final_metadata.schema.data_types.clone(),
            quality_score: final_metadata.quality_score,
            confidence: 0.9,
            processing_time_ms: processing_time,
            extracted_features: HashMap::new(),
        };

        // Create memory entry
        let memory = MultiModalMemory {
            id: Uuid::new_v4().to_string(),
            content_type: content_type.clone(),
            primary_content: content.to_vec(),
            metadata,
            extracted_features: {
                let mut features = HashMap::new();
                features.insert("data_metadata".to_string(), serde_json::to_value(final_metadata)?);
                features
            },
            cross_modal_links: Vec::new(),
            created_at: chrono::Utc::now(),
            updated_at: chrono::Utc::now(),
        };

        Ok(memory)
    }

    async fn extract_features(&self, content: &[u8], _content_type: &ContentType) -> MultiModalResult<Vec<f32>> {
        // Extract basic features from data
        let format = self.detect_data_format(content)?;
        let schema = self.analyze_schema(content, &format).await?;

        // Create feature vector
        let row_count = schema.row_count as f32;
        let column_count = schema.column_count as f32;
        let numeric_columns = schema.columns.iter()
            .filter(|col| matches!(col.data_type, DataType::Integer | DataType::Float))
            .count() as f32;
        let text_columns = schema.columns.iter()
            .filter(|col| matches!(col.data_type, DataType::String))
            .count() as f32;

        Ok(vec![row_count, column_count, numeric_columns, text_columns])
    }

    async fn calculate_similarity(&self, features1: &[f32], features2: &[f32]) -> MultiModalResult<f32> {
        if features1.len() != features2.len() {
            return Ok(0.0);
        }

        // Cosine similarity
        let dot_product: f32 = features1.iter().zip(features2.iter()).map(|(a, b)| a * b).sum();
        let norm1: f32 = features1.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm2: f32 = features2.iter().map(|x| x * x).sum::<f32>().sqrt();

        if norm1 == 0.0 || norm2 == 0.0 {
            Ok(0.0)
        } else {
            Ok(dot_product / (norm1 * norm2))
        }
    }

    async fn search_similar(&self, query_features: &[f32], candidates: &[MultiModalMemory]) -> MultiModalResult<Vec<(MemoryId, f32)>> {
        let mut results = Vec::new();

        for memory in candidates {
            // Extract features from stored memory
            if let Some(features_value) = memory.extracted_features.get("features") {
                if let Ok(features) = serde_json::from_value::<Vec<f32>>(features_value.clone()) {
                    let similarity = self.calculate_similarity(query_features, &features).await?;
                    results.push((memory.id.clone(), similarity));
                }
            }
        }

        // Sort by similarity (highest first)
        results.sort_by(|a, b| {
            b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
        });

        Ok(results)
    }
}
