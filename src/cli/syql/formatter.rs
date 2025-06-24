//! SyQL Result Formatter
//!
//! This module implements various output formatters for SyQL query results,
//! supporting multiple output formats with customizable styling and layout.

use super::{QueryResult, QueryValue, OutputFormat};
use crate::error::Result;
use serde_json;
use std::collections::HashMap;

/// Result formatter for SyQL query results
pub struct ResultFormatter {
    /// Formatting options
    _options: FormatterOptions,
}

impl ResultFormatter {
    /// Create a new result formatter
    pub fn new() -> Result<Self> {
        Ok(Self {
            _options: FormatterOptions::default(),
        })
    }

    /// Format query result in the specified format
    pub fn format(&self, result: &QueryResult, format: OutputFormat) -> Result<String> {
        match format {
            OutputFormat::Table => self.format_table(result),
            OutputFormat::Json => self.format_json(result),
            OutputFormat::Csv => self.format_csv(result),
            OutputFormat::Yaml => self.format_yaml(result),
            OutputFormat::Graph => self.format_graph(result),
            OutputFormat::Tree => self.format_tree(result),
        }
    }

    /// Format as ASCII table
    fn format_table(&self, result: &QueryResult) -> Result<String> {
        if result.rows.is_empty() {
            return Ok("No results found.".to_string());
        }

        let mut output = String::new();
        
        // Get column names from the first row
        let columns: Vec<String> = result.metadata.columns
            .iter()
            .map(|col| col.name.clone())
            .collect();

        if columns.is_empty() {
            return Ok("No columns found.".to_string());
        }

        // Calculate column widths
        let mut widths: HashMap<String, usize> = HashMap::new();
        
        // Initialize with column name lengths
        for col in &columns {
            widths.insert(col.clone(), col.len());
        }

        // Update with data lengths
        for row in &result.rows {
            for col in &columns {
                if let Some(value) = row.values.get(col) {
                    let value_str = self.format_value(value);
                    let current_width = widths.get(col).unwrap_or(&0);
                    widths.insert(col.clone(), (*current_width).max(value_str.len()));
                }
            }
        }

        // Create header separator
        let separator = self.create_table_separator(&columns, &widths);
        
        // Add header
        output.push_str(&separator);
        output.push('|');
        for col in &columns {
            let width = widths.get(col).unwrap_or(&10);
            output.push_str(&format!(" {:width$} |", col, width = width));
        }
        output.push('\n');
        output.push_str(&separator);

        // Add data rows
        for row in &result.rows {
            output.push('|');
            for col in &columns {
                let width = widths.get(col).unwrap_or(&10);
                let value = row.values.get(col)
                    .map(|v| self.format_value(v))
                    .unwrap_or_else(|| "NULL".to_string());
                output.push_str(&format!(" {:width$} |", value, width = width));
            }
            output.push('\n');
        }
        
        output.push_str(&separator);

        // Add statistics
        output.push_str(&format!("\n{} rows returned in {}ms\n", 
            result.statistics.rows_returned,
            result.statistics.execution_time_ms
        ));

        Ok(output)
    }

    /// Create table separator line
    fn create_table_separator(&self, columns: &[String], widths: &HashMap<String, usize>) -> String {
        let mut separator = String::from("+");
        for col in columns {
            let width = widths.get(col).unwrap_or(&10);
            separator.push_str(&"-".repeat(width + 2));
            separator.push('+');
        }
        separator.push('\n');
        separator
    }

    /// Format as JSON
    fn format_json(&self, result: &QueryResult) -> Result<String> {
        let json_result = serde_json::json!({
            "metadata": {
                "query_id": result.metadata.query_id,
                "executed_at": result.metadata.executed_at,
                "query_type": result.metadata.query_type,
                "columns": result.metadata.columns
            },
            "rows": result.rows.iter().map(|row| {
                let mut json_row = serde_json::Map::new();
                for (key, value) in &row.values {
                    json_row.insert(key.clone(), self.value_to_json(value));
                }
                json_row
            }).collect::<Vec<_>>(),
            "statistics": {
                "execution_time_ms": result.statistics.execution_time_ms,
                "rows_returned": result.statistics.rows_returned,
                "memories_scanned": result.statistics.memories_scanned,
                "relationships_traversed": result.statistics.relationships_traversed,
                "estimated_cost": result.statistics.estimated_cost,
                "actual_cost": result.statistics.actual_cost
            }
        });

        Ok(serde_json::to_string_pretty(&json_result)?)
    }

    /// Format as CSV
    fn format_csv(&self, result: &QueryResult) -> Result<String> {
        if result.rows.is_empty() {
            return Ok(String::new());
        }

        let mut output = String::new();
        
        // Get column names
        let columns: Vec<String> = result.metadata.columns
            .iter()
            .map(|col| col.name.clone())
            .collect();

        // Add header
        output.push_str(&columns.join(","));
        output.push('\n');

        // Add data rows
        for row in &result.rows {
            let values: Vec<String> = columns
                .iter()
                .map(|col| {
                    row.values.get(col)
                        .map(|v| self.format_csv_value(v))
                        .unwrap_or_else(|| String::new())
                })
                .collect();
            output.push_str(&values.join(","));
            output.push('\n');
        }

        Ok(output)
    }

    /// Format as YAML
    fn format_yaml(&self, result: &QueryResult) -> Result<String> {
        let yaml_result = serde_yaml::to_string(&serde_json::json!({
            "metadata": {
                "query_id": result.metadata.query_id,
                "executed_at": result.metadata.executed_at,
                "query_type": result.metadata.query_type
            },
            "rows": result.rows.iter().map(|row| {
                let mut yaml_row = serde_json::Map::new();
                for (key, value) in &row.values {
                    yaml_row.insert(key.clone(), self.value_to_json(value));
                }
                yaml_row
            }).collect::<Vec<_>>(),
            "statistics": {
                "execution_time_ms": result.statistics.execution_time_ms,
                "rows_returned": result.statistics.rows_returned
            }
        }))?;

        Ok(yaml_result)
    }

    /// Format as ASCII graph visualization
    fn format_graph(&self, result: &QueryResult) -> Result<String> {
        let mut output = String::new();
        
        output.push_str("Graph Visualization:\n");
        output.push_str("===================\n\n");

        // Simple graph representation
        let mut nodes = Vec::new();
        let mut edges = Vec::new();

        for row in &result.rows {
            // Extract nodes and relationships
            if let (Some(from), Some(to)) = (row.values.get("from_id"), row.values.get("to_id")) {
                let from_str = self.format_value(from);
                let to_str = self.format_value(to);
                
                if !nodes.contains(&from_str) {
                    nodes.push(from_str.clone());
                }
                if !nodes.contains(&to_str) {
                    nodes.push(to_str.clone());
                }
                
                let rel_type = row.values.get("relationship_type")
                    .map(|v| self.format_value(v))
                    .unwrap_or_else(|| "relates_to".to_string());
                
                edges.push(format!("{} --[{}]--> {}", from_str, rel_type, to_str));
            } else if let Some(id) = row.values.get("id") {
                let id_str = self.format_value(id);
                if !nodes.contains(&id_str) {
                    nodes.push(id_str);
                }
            }
        }

        // Display nodes
        output.push_str("Nodes:\n");
        for (i, node) in nodes.iter().enumerate() {
            output.push_str(&format!("  [{}] {}\n", i + 1, node));
        }

        // Display edges
        if !edges.is_empty() {
            output.push_str("\nRelationships:\n");
            for edge in edges {
                output.push_str(&format!("  {}\n", edge));
            }
        }

        Ok(output)
    }

    /// Format as tree structure
    fn format_tree(&self, result: &QueryResult) -> Result<String> {
        let mut output = String::new();
        
        output.push_str("Tree Structure:\n");
        output.push_str("===============\n\n");

        // Group rows by some hierarchy (simplified)
        let mut tree_map: HashMap<String, Vec<String>> = HashMap::new();
        
        for row in &result.rows {
            let parent = row.values.get("parent_id")
                .or_else(|| row.values.get("from_id"))
                .map(|v| self.format_value(v))
                .unwrap_or_else(|| "root".to_string());
            
            let child = row.values.get("id")
                .or_else(|| row.values.get("to_id"))
                .map(|v| self.format_value(v))
                .unwrap_or_else(|| "unknown".to_string());
            
            tree_map.entry(parent).or_insert_with(Vec::new).push(child);
        }

        // Display tree
        self.display_tree_node(&mut output, &tree_map, "root", 0);

        Ok(output)
    }

    /// Display tree node recursively
    fn display_tree_node(&self, output: &mut String, tree_map: &HashMap<String, Vec<String>>, node: &str, depth: usize) {
        let indent = "  ".repeat(depth);
        output.push_str(&format!("{}├─ {}\n", indent, node));
        
        if let Some(children) = tree_map.get(node) {
            for child in children {
                self.display_tree_node(output, tree_map, child, depth + 1);
            }
        }
    }

    /// Format a query value as string
    fn format_value(&self, value: &QueryValue) -> String {
        match value {
            QueryValue::Null => "NULL".to_string(),
            QueryValue::String(s) => s.clone(),
            QueryValue::Integer(i) => i.to_string(),
            QueryValue::Float(f) => format!("{:.2}", f),
            QueryValue::Boolean(b) => b.to_string(),
            QueryValue::DateTime(dt) => dt.format("%Y-%m-%d %H:%M:%S UTC").to_string(),
            QueryValue::Memory(mem) => format!("Memory({})", mem.id),
            QueryValue::Relationship(rel) => format!("Rel({} -> {})", rel.from_memory, rel.to_memory),
            QueryValue::Path(path) => format!("Path(length: {})", path.length),
            QueryValue::List(list) => {
                let items: Vec<String> = list.iter().map(|v| self.format_value(v)).collect();
                format!("[{}]", items.join(", "))
            },
            QueryValue::Map(map) => {
                let items: Vec<String> = map.iter()
                    .map(|(k, v)| format!("{}: {}", k, self.format_value(v)))
                    .collect();
                format!("{{{}}}", items.join(", "))
            },
        }
    }

    /// Format value for CSV (with proper escaping)
    fn format_csv_value(&self, value: &QueryValue) -> String {
        let formatted = self.format_value(value);
        
        // Escape CSV special characters
        if formatted.contains(',') || formatted.contains('"') || formatted.contains('\n') {
            format!("\"{}\"", formatted.replace('"', "\"\""))
        } else {
            formatted
        }
    }

    /// Convert query value to JSON value
    fn value_to_json(&self, value: &QueryValue) -> serde_json::Value {
        match value {
            QueryValue::Null => serde_json::Value::Null,
            QueryValue::String(s) => serde_json::Value::String(s.clone()),
            QueryValue::Integer(i) => serde_json::Value::Number(serde_json::Number::from(*i)),
            QueryValue::Float(f) => serde_json::Value::Number(
                serde_json::Number::from_f64(*f).unwrap_or_else(|| serde_json::Number::from(0))
            ),
            QueryValue::Boolean(b) => serde_json::Value::Bool(*b),
            QueryValue::DateTime(dt) => serde_json::Value::String(dt.to_rfc3339()),
            QueryValue::Memory(mem) => serde_json::json!({
                "type": "memory",
                "id": mem.id,
                "content": mem.content,
                "memory_type": mem.memory_type,
                "created_at": mem.created_at
            }),
            QueryValue::Relationship(rel) => serde_json::json!({
                "type": "relationship",
                "id": rel.id,
                "from_memory": rel.from_memory,
                "to_memory": rel.to_memory,
                "relationship_type": rel.relationship_type,
                "strength": rel.strength
            }),
            QueryValue::Path(path) => serde_json::json!({
                "type": "path",
                "length": path.length,
                "total_weight": path.total_weight,
                "nodes": path.nodes.len(),
                "relationships": path.relationships.len()
            }),
            QueryValue::List(list) => {
                let json_list: Vec<serde_json::Value> = list.iter()
                    .map(|v| self.value_to_json(v))
                    .collect();
                serde_json::Value::Array(json_list)
            },
            QueryValue::Map(map) => {
                let json_map: serde_json::Map<String, serde_json::Value> = map.iter()
                    .map(|(k, v)| (k.clone(), self.value_to_json(v)))
                    .collect();
                serde_json::Value::Object(json_map)
            },
        }
    }
}

/// Formatter options
#[derive(Debug, Clone)]
pub struct FormatterOptions {
    /// Maximum column width for table format
    pub max_column_width: usize,
    /// Whether to use colors in output
    pub use_colors: bool,
    /// Date format string
    pub date_format: String,
    /// Number format precision
    pub number_precision: usize,
}

impl Default for FormatterOptions {
    fn default() -> Self {
        Self {
            max_column_width: 50,
            use_colors: true,
            date_format: "%Y-%m-%d %H:%M:%S UTC".to_string(),
            number_precision: 2,
        }
    }
}
