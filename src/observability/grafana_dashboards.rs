//! Grafana Dashboard Configuration and Management
//!
//! This module provides comprehensive Grafana dashboard definitions for visualizing
//! Synaptic memory system metrics, traces, and logs with sophisticated monitoring
//! capabilities, alerting rules, and performance analysis panels.

use crate::error::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use tracing::{debug, info};

/// Grafana dashboard configuration for Synaptic monitoring
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GrafanaDashboard {
    pub id: Option<u64>,
    pub uid: Option<String>,
    pub title: String,
    pub description: String,
    pub tags: Vec<String>,
    pub timezone: String,
    pub refresh: String,
    pub time: TimeRange,
    pub panels: Vec<Panel>,
    pub templating: Templating,
    pub annotations: Annotations,
    pub links: Vec<Link>,
}

/// Time range configuration for dashboards
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimeRange {
    pub from: String,
    pub to: String,
}

/// Panel configuration for dashboard visualizations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Panel {
    pub id: u32,
    pub title: String,
    pub r#type: String,
    pub targets: Vec<Target>,
    pub grid_pos: GridPos,
    pub options: Option<serde_json::Value>,
    pub field_config: Option<FieldConfig>,
    pub alert: Option<Alert>,
}

/// Query target for panels
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Target {
    pub expr: String,
    pub legend_format: Option<String>,
    pub ref_id: String,
    pub interval: Option<String>,
    pub format: Option<String>,
}

/// Grid position for panel layout
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GridPos {
    pub h: u32,
    pub w: u32,
    pub x: u32,
    pub y: u32,
}

/// Field configuration for panel styling
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FieldConfig {
    pub defaults: FieldDefaults,
    pub overrides: Vec<FieldOverride>,
}

/// Default field configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FieldDefaults {
    pub unit: Option<String>,
    pub min: Option<f64>,
    pub max: Option<f64>,
    pub decimals: Option<u32>,
    pub thresholds: Option<Thresholds>,
}

/// Field override configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FieldOverride {
    pub matcher: Matcher,
    pub properties: Vec<Property>,
}

/// Field matcher
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Matcher {
    pub id: String,
    pub options: String,
}

/// Field property
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Property {
    pub id: String,
    pub value: serde_json::Value,
}

/// Threshold configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Thresholds {
    pub mode: String,
    pub steps: Vec<ThresholdStep>,
}

/// Threshold step
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThresholdStep {
    pub color: String,
    pub value: Option<f64>,
}

/// Alert configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Alert {
    pub conditions: Vec<AlertCondition>,
    pub execution_error_state: String,
    pub for_duration: String,
    pub frequency: String,
    pub handler: u32,
    pub name: String,
    pub no_data_state: String,
    pub notifications: Vec<AlertNotification>,
}

/// Alert condition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlertCondition {
    pub evaluator: Evaluator,
    pub operator: Operator,
    pub query: AlertQuery,
    pub reducer: Reducer,
    pub r#type: String,
}

/// Alert evaluator
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Evaluator {
    pub params: Vec<f64>,
    pub r#type: String,
}

/// Alert operator
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Operator {
    pub r#type: String,
}

/// Alert query
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlertQuery {
    pub model: serde_json::Value,
    pub params: Vec<String>,
}

/// Alert reducer
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Reducer {
    pub params: Vec<String>,
    pub r#type: String,
}

/// Alert notification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlertNotification {
    pub uid: String,
}

/// Templating configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Templating {
    pub list: Vec<Template>,
}

/// Template variable
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Template {
    pub name: String,
    pub r#type: String,
    pub query: String,
    pub refresh: u32,
    pub options: Vec<TemplateOption>,
    pub current: TemplateOption,
}

/// Template option
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemplateOption {
    pub text: String,
    pub value: String,
    pub selected: bool,
}

/// Annotations configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Annotations {
    pub list: Vec<Annotation>,
}

/// Annotation configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Annotation {
    pub name: String,
    pub datasource: String,
    pub enable: bool,
    pub expr: String,
    pub icon_color: String,
    pub title_format: String,
    pub tag_keys: String,
}

/// Dashboard link
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Link {
    pub title: String,
    pub url: String,
    pub r#type: String,
    pub icon: String,
}

/// Grafana dashboard manager
pub struct GrafanaDashboardManager {
    dashboards: HashMap<String, GrafanaDashboard>,
}

impl GrafanaDashboardManager {
    /// Create a new dashboard manager
    pub fn new() -> Self {
        Self {
            dashboards: HashMap::new(),
        }
    }
    
    /// Generate the main Synaptic overview dashboard
    pub fn create_synaptic_overview_dashboard(&mut self) -> Result<()> {
        let dashboard = GrafanaDashboard {
            id: None,
            uid: Some("synaptic-overview".to_string()),
            title: "Synaptic Memory System - Overview".to_string(),
            description: "Comprehensive overview of Synaptic AI Agent Memory System performance and health".to_string(),
            tags: vec!["synaptic".to_string(), "memory".to_string(), "ai".to_string()],
            timezone: "browser".to_string(),
            refresh: "30s".to_string(),
            time: TimeRange {
                from: "now-1h".to_string(),
                to: "now".to_string(),
            },
            panels: self.create_overview_panels(),
            templating: self.create_templating(),
            annotations: self.create_annotations(),
            links: self.create_dashboard_links(),
        };
        
        self.dashboards.insert("synaptic-overview".to_string(), dashboard);
        info!("Created Synaptic overview dashboard");
        Ok(())
    }
    
    /// Create panels for the overview dashboard
    fn create_overview_panels(&self) -> Vec<Panel> {
        vec![
            // System Health Panel
            Panel {
                id: 1,
                title: "System Health".to_string(),
                r#type: "stat".to_string(),
                targets: vec![
                    Target {
                        expr: "up{job=\"synaptic\"}".to_string(),
                        legend_format: Some("System Status".to_string()),
                        ref_id: "A".to_string(),
                        interval: None,
                        format: None,
                    }
                ],
                grid_pos: GridPos { h: 8, w: 12, x: 0, y: 0 },
                options: None,
                field_config: Some(FieldConfig {
                    defaults: FieldDefaults {
                        unit: Some("short".to_string()),
                        min: Some(0.0),
                        max: Some(1.0),
                        decimals: Some(0),
                        thresholds: Some(Thresholds {
                            mode: "absolute".to_string(),
                            steps: vec![
                                ThresholdStep { color: "red".to_string(), value: Some(0.0) },
                                ThresholdStep { color: "green".to_string(), value: Some(1.0) },
                            ],
                        }),
                    },
                    overrides: vec![],
                }),
                alert: None,
            },
            
            // Memory Operations Rate
            Panel {
                id: 2,
                title: "Memory Operations Rate".to_string(),
                r#type: "graph".to_string(),
                targets: vec![
                    Target {
                        expr: "rate(synaptic_memory_operations_total[5m])".to_string(),
                        legend_format: Some("{{operation_type}} - {{memory_type}}".to_string()),
                        ref_id: "A".to_string(),
                        interval: None,
                        format: None,
                    }
                ],
                grid_pos: GridPos { h: 8, w: 12, x: 12, y: 0 },
                options: None,
                field_config: Some(FieldConfig {
                    defaults: FieldDefaults {
                        unit: Some("ops".to_string()),
                        min: Some(0.0),
                        max: None,
                        decimals: Some(2),
                        thresholds: None,
                    },
                    overrides: vec![],
                }),
                alert: None,
            },
            
            // Query Performance
            Panel {
                id: 3,
                title: "Query Performance".to_string(),
                r#type: "graph".to_string(),
                targets: vec![
                    Target {
                        expr: "histogram_quantile(0.95, rate(synaptic_query_duration_seconds_bucket[5m]))".to_string(),
                        legend_format: Some("95th percentile".to_string()),
                        ref_id: "A".to_string(),
                        interval: None,
                        format: None,
                    },
                    Target {
                        expr: "histogram_quantile(0.50, rate(synaptic_query_duration_seconds_bucket[5m]))".to_string(),
                        legend_format: Some("50th percentile".to_string()),
                        ref_id: "B".to_string(),
                        interval: None,
                        format: None,
                    }
                ],
                grid_pos: GridPos { h: 8, w: 12, x: 0, y: 8 },
                options: None,
                field_config: Some(FieldConfig {
                    defaults: FieldDefaults {
                        unit: Some("s".to_string()),
                        min: Some(0.0),
                        max: None,
                        decimals: Some(3),
                        thresholds: None,
                    },
                    overrides: vec![],
                }),
                alert: Some(Alert {
                    conditions: vec![
                        AlertCondition {
                            evaluator: Evaluator {
                                params: vec![1.0],
                                r#type: "gt".to_string(),
                            },
                            operator: Operator {
                                r#type: "and".to_string(),
                            },
                            query: AlertQuery {
                                model: serde_json::json!({}),
                                params: vec!["A".to_string(), "5m".to_string(), "now".to_string()],
                            },
                            reducer: Reducer {
                                params: vec![],
                                r#type: "last".to_string(),
                            },
                            r#type: "query".to_string(),
                        }
                    ],
                    execution_error_state: "alerting".to_string(),
                    for_duration: "5m".to_string(),
                    frequency: "10s".to_string(),
                    handler: 1,
                    name: "High Query Latency".to_string(),
                    no_data_state: "no_data".to_string(),
                    notifications: vec![],
                }),
            },
        ]
    }
    
    /// Create templating configuration
    fn create_templating(&self) -> Templating {
        Templating {
            list: vec![
                Template {
                    name: "instance".to_string(),
                    r#type: "query".to_string(),
                    query: "label_values(up{job=\"synaptic\"}, instance)".to_string(),
                    refresh: 1,
                    options: vec![],
                    current: TemplateOption {
                        text: "All".to_string(),
                        value: "$__all".to_string(),
                        selected: true,
                    },
                }
            ],
        }
    }
    
    /// Create annotations configuration
    fn create_annotations(&self) -> Annotations {
        Annotations {
            list: vec![
                Annotation {
                    name: "Deployments".to_string(),
                    datasource: "prometheus".to_string(),
                    enable: true,
                    expr: "synaptic_deployment_info".to_string(),
                    icon_color: "blue".to_string(),
                    title_format: "Deployment: {{version}}".to_string(),
                    tag_keys: "version,environment".to_string(),
                }
            ],
        }
    }
    
    /// Create dashboard links
    fn create_dashboard_links(&self) -> Vec<Link> {
        vec![
            Link {
                title: "Memory Details".to_string(),
                url: "/d/synaptic-memory".to_string(),
                r#type: "dashboards".to_string(),
                icon: "cloud".to_string(),
            },
            Link {
                title: "Performance Analysis".to_string(),
                url: "/d/synaptic-performance".to_string(),
                r#type: "dashboards".to_string(),
                icon: "bolt".to_string(),
            },
        ]
    }
}
