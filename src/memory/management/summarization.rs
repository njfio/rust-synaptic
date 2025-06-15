//! Intelligent memory summarization and consolidation

use crate::error::{MemoryError, Result};
use crate::memory::types::{MemoryEntry, MemoryType};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use uuid::Uuid;

/// Strategies for memory summarization
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum SummaryStrategy {
    /// Extract key points and main themes
    KeyPoints,
    /// Create a chronological summary
    Chronological,
    /// Focus on most important information
    ImportanceBased,
    /// Combine similar memories into unified entries
    Consolidation,
    /// Create hierarchical summaries
    Hierarchical,
    /// Generate abstract conceptual summary
    Conceptual,
    /// Custom summarization logic
    Custom(String),
}

/// Result of a summarization operation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SummaryResult {
    /// Unique identifier for this summary
    pub id: Uuid,
    /// Strategy used for summarization
    pub strategy: SummaryStrategy,
    /// Original memory keys that were summarized
    pub source_memory_keys: Vec<String>,
    /// Generated summary content
    pub summary_content: String,
    /// Confidence score for the summary (0.0 to 1.0)
    pub confidence_score: f64,
    /// Compression ratio (original size / summary size)
    pub compression_ratio: f64,
    /// When this summary was created
    pub created_at: DateTime<Utc>,
    /// Key themes identified
    pub key_themes: Vec<String>,
    /// Important entities mentioned
    pub entities: Vec<Entity>,
    /// Temporal information extracted
    pub temporal_info: Option<TemporalInfo>,
    /// Quality metrics
    pub quality_metrics: SummaryQualityMetrics,
}

/// An entity identified in the summarization process
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Entity {
    /// Entity name
    pub name: String,
    /// Entity type (person, place, concept, etc.)
    pub entity_type: EntityType,
    /// Frequency of mention
    pub frequency: usize,
    /// Importance score
    pub importance: f64,
    /// Context in which the entity appears
    pub context: Vec<String>,
}

/// Types of entities that can be identified
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum EntityType {
    Person,
    Place,
    Organization,
    Concept,
    Event,
    Technology,
    Project,
    Task,
    Goal,
    Problem,
    Solution,
    Custom(String),
}

/// Temporal information extracted from memories
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemporalInfo {
    /// Time range covered by the memories
    pub time_range: Option<(DateTime<Utc>, DateTime<Utc>)>,
    /// Chronological events identified
    pub events: Vec<TemporalEvent>,
    /// Patterns in timing
    pub patterns: Vec<String>,
}

/// A temporal event identified in memories
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemporalEvent {
    /// Event description
    pub description: String,
    /// When the event occurred
    pub timestamp: DateTime<Utc>,
    /// Event importance
    pub importance: f64,
    /// Related memory keys
    pub related_memories: Vec<String>,
}

/// Quality metrics for summaries
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SummaryQualityMetrics {
    /// Coherence score (how well the summary flows)
    pub coherence: f64,
    /// Completeness score (how much information is preserved)
    pub completeness: f64,
    /// Conciseness score (how efficiently information is presented)
    pub conciseness: f64,
    /// Accuracy score (how faithful to original content)
    pub accuracy: f64,
    /// Overall quality score
    pub overall_quality: f64,
}

/// Rules for memory consolidation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsolidationRule {
    /// Rule identifier
    pub id: String,
    /// Rule name
    pub name: String,
    /// Similarity threshold for consolidation
    pub similarity_threshold: f64,
    /// Maximum age difference for consolidation
    pub max_age_difference_hours: u64,
    /// Minimum importance for consolidation
    pub min_importance: f64,
    /// Tags that trigger consolidation
    pub trigger_tags: Vec<String>,
    /// Whether this rule is active
    pub active: bool,
}

/// Memory summarizer with multiple strategies
pub struct MemorySummarizer {
    /// Consolidation rules
    consolidation_rules: Vec<ConsolidationRule>,
    /// Summarization history
    summarization_history: Vec<SummaryResult>,
    /// Configuration
    config: SummarizationConfig,
}

/// Configuration for memory summarization
#[derive(Debug, Clone)]
pub struct SummarizationConfig {
    /// Default summarization strategy
    pub default_strategy: SummaryStrategy,
    /// Maximum summary length
    pub max_summary_length: usize,
    /// Minimum compression ratio to accept
    pub min_compression_ratio: f64,
    /// Enable entity extraction
    pub enable_entity_extraction: bool,
    /// Enable temporal analysis
    pub enable_temporal_analysis: bool,
    /// Quality threshold for accepting summaries
    pub quality_threshold: f64,
}

impl Default for SummarizationConfig {
    fn default() -> Self {
        Self {
            default_strategy: SummaryStrategy::KeyPoints,
            max_summary_length: 1000,
            min_compression_ratio: 2.0,
            enable_entity_extraction: true,
            enable_temporal_analysis: true,
            quality_threshold: 0.7,
        }
    }
}

impl MemorySummarizer {
    /// Create a new memory summarizer
    pub fn new() -> Self {
        Self {
            consolidation_rules: Self::create_default_rules(),
            summarization_history: Vec::new(),
            config: SummarizationConfig::default(),
        }
    }

    /// Summarize a group of memories
    pub async fn summarize_memories(
        &mut self,
        storage: &(dyn crate::memory::storage::Storage + Send + Sync),
        memory_keys: Vec<String>,
        strategy: SummaryStrategy,
    ) -> Result<SummaryResult> {
        if memory_keys.is_empty() {
            return Err(MemoryError::unexpected("No memories provided for summarization"));
        }

        // Load the referenced memories from the provided storage
        let mut memories = Vec::new();
        for key in &memory_keys {
            if let Some(entry) = storage.retrieve(key).await? {
                memories.push(entry);
            }
        }

        if memories.is_empty() {
            return Err(MemoryError::NotFound { key: "no memories".to_string() });
        }

        let summary_content = self.generate_summary_content(&memories, &strategy).await?;
        
        // Extract entities if enabled
        let entities = if self.config.enable_entity_extraction {
            self.extract_entities(&summary_content).await?
        } else {
            Vec::new()
        };

        // Extract temporal information if enabled
        let temporal_info = if self.config.enable_temporal_analysis {
            Some(self.extract_temporal_info(&memories).await?)
        } else {
            None
        };

        // Calculate quality metrics
        let quality_metrics = self.calculate_quality_metrics(&summary_content, &memories).await?;

        // Calculate compression ratio
        let original_size: usize = memories.iter().map(|m| m.value.len()).sum();
        let summary_size = summary_content.len();
        let compression_ratio = if summary_size > 0 {
            original_size as f64 / summary_size as f64
        } else {
            1.0
        };

        let key_themes = self.extract_key_themes(&summary_content).await?;

        let summary_result = SummaryResult {
            id: Uuid::new_v4(),
            strategy,
            source_memory_keys: memory_keys,
            summary_content,
            confidence_score: quality_metrics.overall_quality,
            compression_ratio,
            created_at: Utc::now(),
            key_themes,
            entities,
            temporal_info,
            quality_metrics,
        };

        // Store in history
        self.summarization_history.push(summary_result.clone());

        Ok(summary_result)
    }

    /// Generate summary content based on strategy
    async fn generate_summary_content(
        &self,
        memories: &[MemoryEntry],
        strategy: &SummaryStrategy,
    ) -> Result<String> {
        match strategy {
            SummaryStrategy::KeyPoints => {
                self.generate_key_points_summary(memories).await
            }
            SummaryStrategy::Chronological => {
                self.generate_chronological_summary(memories).await
            }
            SummaryStrategy::ImportanceBased => {
                self.generate_importance_based_summary(memories).await
            }
            SummaryStrategy::Consolidation => {
                self.generate_consolidation_summary(memories).await
            }
            SummaryStrategy::Hierarchical => {
                self.generate_hierarchical_summary(memories).await
            }
            SummaryStrategy::Conceptual => {
                self.generate_conceptual_summary(memories).await
            }
            SummaryStrategy::Custom(name) => {
                self.generate_custom_summary(memories, name).await
            }
        }
    }

    /// Generate key points summary
    async fn generate_key_points_summary(&self, memories: &[MemoryEntry]) -> Result<String> {
        let mut lines = Vec::new();
        for mem in memories {
            lines.push(format!("• {}", mem.value.trim()));
        }
        Ok(format!(
            "Key points summary of {} memories:\n{}",
            memories.len(),
            lines.join("\n")
        ))
    }

    /// Generate chronological summary
    async fn generate_chronological_summary(&self, memories: &[MemoryEntry]) -> Result<String> {
        let mut entries: Vec<_> = memories.to_vec();
        entries.sort_by_key(|m| m.created_at());
        let mut lines = Vec::new();
        for mem in entries {
            lines.push(format!(
                "{} - {}",
                mem.created_at().format("%Y-%m-%d %H:%M:%S"),
                mem.value.trim()
            ));
        }
        Ok(format!(
            "Chronological summary of {} memories:\n{}",
            memories.len(),
            lines.join("\n")
        ))
    }

    /// Generate importance-based summary
    async fn generate_importance_based_summary(&self, memories: &[MemoryEntry]) -> Result<String> {
        let mut entries: Vec<_> = memories.to_vec();
        entries.sort_by(|a, b| b.metadata.importance.partial_cmp(&a.metadata.importance).unwrap());
        let mut lines = Vec::new();
        for mem in entries {
            lines.push(format!("({:.2}) {}", mem.metadata.importance, mem.value.trim()));
        }
        Ok(format!(
            "Importance-based summary of {} memories:\n{}",
            memories.len(),
            lines.join("\n")
        ))
    }

    /// Generate consolidation summary
    async fn generate_consolidation_summary(&self, memories: &[MemoryEntry]) -> Result<String> {
        use std::collections::HashSet;
        let mut seen = HashSet::new();
        let mut lines = Vec::new();
        for mem in memories {
            if seen.insert(mem.value.clone()) {
                lines.push(mem.value.trim().to_string());
            }
        }
        Ok(format!(
            "Consolidated summary of {} memories:\n{}",
            memories.len(),
            lines.join("\n")
        ))
    }

    /// Generate hierarchical summary
    async fn generate_hierarchical_summary(&self, memories: &[MemoryEntry]) -> Result<String> {
        use std::collections::HashMap;
        let mut map: HashMap<MemoryType, Vec<String>> = HashMap::new();
        for mem in memories {
            map.entry(mem.memory_type)
                .or_default()
                .push(mem.value.trim().to_string());
        }
        let mut sections = Vec::new();
        for (ty, vals) in map {
            sections.push(format!("{}:\n  - {}", ty, vals.join("\n  - ")));
        }
        Ok(format!(
            "Hierarchical summary of {} memories:\n{}",
            memories.len(),
            sections.join("\n")
        ))
    }

    /// Generate conceptual summary
    async fn generate_conceptual_summary(&self, memories: &[MemoryEntry]) -> Result<String> {
        let mut lines = Vec::new();
        for mem in memories {
            let words: Vec<&str> = mem.value.split_whitespace().take(5).collect();
            lines.push(format!("{}...", words.join(" ")));
        }
        Ok(format!(
            "Conceptual summary of {} memories:\n{}",
            memories.len(),
            lines.join("\n")
        ))
    }

    /// Generate custom summary
    async fn generate_custom_summary(&self, memories: &[MemoryEntry], _strategy_name: &str) -> Result<String> {
        self.generate_key_points_summary(memories).await
    }

    /// Extract entities from text
    async fn extract_entities(&self, text: &str) -> Result<Vec<Entity>> {
        // TODO: Implement proper entity extraction using NLP
        let mut entities = Vec::new();
        
        // Simple keyword-based entity extraction for demonstration
        let keywords = ["project", "task", "goal", "problem", "solution", "meeting", "deadline"];
        
        for keyword in keywords {
            if text.to_lowercase().contains(keyword) {
                entities.push(Entity {
                    name: keyword.to_string(),
                    entity_type: EntityType::Concept,
                    frequency: text.to_lowercase().matches(keyword).count(),
                    importance: 0.5,
                    context: vec![format!("Found in summary context")],
                });
            }
        }
        
        Ok(entities)
    }

    /// Extract temporal information
    async fn extract_temporal_info(&self, memories: &[MemoryEntry]) -> Result<TemporalInfo> {
        if memories.is_empty() {
            return Ok(TemporalInfo { time_range: None, events: Vec::new(), patterns: Vec::new() });
        }

        let mut sorted: Vec<_> = memories.to_vec();
        sorted.sort_by_key(|m| m.created_at());
        let start = sorted.first().unwrap().created_at();
        let end = sorted.last().unwrap().created_at();
        let events = sorted
            .iter()
            .map(|m| TemporalEvent {
                description: m.value.clone(),
                timestamp: m.created_at(),
                importance: m.metadata.importance,
                related_memories: vec![m.key.clone()],
            })
            .collect();

        Ok(TemporalInfo {
            time_range: Some((start, end)),
            events,
            patterns: Vec::new(),
        })
    }

    /// Extract key themes from text
    async fn extract_key_themes(&self, text: &str) -> Result<Vec<String>> {
        // TODO: Implement proper theme extraction
        let themes = vec![
            "Information Management".to_string(),
            "Knowledge Organization".to_string(),
            "Memory Consolidation".to_string(),
        ];
        
        // Filter themes that appear to be relevant to the text
        Ok(themes.into_iter()
            .filter(|theme| text.to_lowercase().contains(&theme.to_lowercase()))
            .collect())
    }

    /// Calculate quality metrics for a summary
    async fn calculate_quality_metrics(
        &self,
        summary_content: &str,
        memories: &[MemoryEntry],
    ) -> Result<SummaryQualityMetrics> {
        // TODO: Implement proper quality assessment
        let coherence = 0.8; // Placeholder
        let completeness = 0.7; // Placeholder
        let conciseness = 0.9; // Placeholder
        let accuracy = 0.8; // Placeholder
        
        let overall_quality = (coherence + completeness + conciseness + accuracy) / 4.0;
        
        Ok(SummaryQualityMetrics {
            coherence,
            completeness,
            conciseness,
            accuracy,
            overall_quality,
        })
    }

    /// Create default consolidation rules
    fn create_default_rules() -> Vec<ConsolidationRule> {
        vec![
            ConsolidationRule {
                id: "similar_content".to_string(),
                name: "Similar Content Consolidation".to_string(),
                similarity_threshold: 0.8,
                max_age_difference_hours: 24,
                min_importance: 0.3,
                trigger_tags: vec!["duplicate".to_string(), "similar".to_string()],
                active: true,
            },
            ConsolidationRule {
                id: "related_tasks".to_string(),
                name: "Related Tasks Consolidation".to_string(),
                similarity_threshold: 0.6,
                max_age_difference_hours: 168, // 1 week
                min_importance: 0.5,
                trigger_tags: vec!["task".to_string(), "project".to_string()],
                active: true,
            },
        ]
    }

    /// Get the number of summarizations performed
    pub fn get_summarization_count(&self) -> usize {
        self.summarization_history.len()
    }

    /// Get summarization history
    pub fn get_summarization_history(&self) -> &[SummaryResult] {
        &self.summarization_history
    }

    /// Find memories that should be consolidated
    pub async fn find_consolidation_candidates(&self, memories: &[MemoryEntry]) -> Result<Vec<Vec<String>>> {
        let mut candidates = Vec::new();
        
        for rule in &self.consolidation_rules {
            if !rule.active {
                continue;
            }
            
            let rule_candidates = self.apply_consolidation_rule(rule, memories).await?;
            candidates.extend(rule_candidates);
        }
        
        Ok(candidates)
    }

    /// Apply a specific consolidation rule
    async fn apply_consolidation_rule(
        &self,
        rule: &ConsolidationRule,
        memories: &[MemoryEntry],
    ) -> Result<Vec<Vec<String>>> {
        let mut candidates = Vec::new();
        
        // TODO: Implement proper consolidation rule application
        // This would involve:
        // 1. Filtering memories by rule criteria
        // 2. Calculating similarity between memories
        // 3. Grouping similar memories together
        // 4. Checking age differences and importance thresholds
        
        Ok(candidates)
    }
}

impl Default for MemorySummarizer {
    fn default() -> Self {
        Self::new()
    }
}
