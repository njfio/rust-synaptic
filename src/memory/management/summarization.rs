//! Intelligent memory summarization and consolidation

use crate::error::{MemoryError, Result};
use crate::memory::types::{MemoryEntry, MemoryType};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use uuid::Uuid;
use ndarray::{Array1, Array2};
use linfa::prelude::*;
use linfa_clustering::KMeans;
use std::collections::BTreeMap;

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

    /// Generate advanced key points summary using NLP-based extraction
    async fn generate_key_points_summary(&self, memories: &[MemoryEntry]) -> Result<String> {
        // Combine all memory content for analysis
        let combined_text = memories.iter()
            .map(|mem| mem.value.as_str())
            .collect::<Vec<_>>()
            .join(" ");

        // Extract key points using multiple NLP techniques
        let key_points = self.extract_advanced_key_points(&combined_text, memories).await?;

        // Format the summary
        let mut summary_lines = vec![
            format!("Key Points Summary ({} memories analyzed):", memories.len()),
            String::new(),
        ];

        for (i, point) in key_points.iter().enumerate() {
            summary_lines.push(format!("{}. {}", i + 1, point));
        }

        // Add metadata
        summary_lines.push(String::new());
        summary_lines.push(format!("Analysis completed: {}", chrono::Utc::now().format("%Y-%m-%d %H:%M:%S UTC")));
        summary_lines.push(format!("Extraction methods: TF-IDF, TextRank, Entity Recognition, Importance Scoring"));

        Ok(summary_lines.join("\n"))
    }

    /// Extract advanced key points using multiple NLP techniques
    async fn extract_advanced_key_points(&self, text: &str, memories: &[MemoryEntry]) -> Result<Vec<String>> {
        let mut key_points = Vec::new();

        // 1. TF-IDF based key phrase extraction
        let tfidf_points = self.extract_tfidf_key_points(text).await?;
        key_points.extend(tfidf_points);

        // 2. TextRank algorithm for sentence ranking
        let textrank_points = self.extract_textrank_key_points(text).await?;
        key_points.extend(textrank_points);

        // 3. Entity-based key points
        let entity_points = self.extract_entity_based_key_points(text).await?;
        key_points.extend(entity_points);

        // 4. Importance-weighted key points from individual memories
        let importance_points = self.extract_importance_weighted_key_points(memories).await?;
        key_points.extend(importance_points);

        // 5. Pattern-based key points
        let pattern_points = self.extract_pattern_based_key_points(text).await?;
        key_points.extend(pattern_points);

        // Deduplicate and rank key points
        let ranked_points = self.rank_and_deduplicate_key_points(key_points).await?;

        // Return top key points (limit to reasonable number)
        Ok(ranked_points.into_iter().take(10).collect())
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

    /// Extract TF-IDF based key points
    async fn extract_tfidf_key_points(&self, text: &str) -> Result<Vec<String>> {
        let mut key_points = Vec::new();

        // Split text into sentences
        let sentences = self.split_into_sentences(text);
        if sentences.is_empty() {
            return Ok(key_points);
        }

        // Calculate TF-IDF scores for each sentence
        let mut sentence_scores = Vec::new();
        for sentence in &sentences {
            let score = self.calculate_sentence_tfidf_score(sentence, &sentences).await?;
            sentence_scores.push((sentence.clone(), score));
        }

        // Sort by TF-IDF score and take top sentences
        sentence_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        for (sentence, _score) in sentence_scores.into_iter().take(3) {
            if sentence.len() > 20 && sentence.len() < 200 { // Filter reasonable length sentences
                key_points.push(format!("Key insight: {}", sentence.trim()));
            }
        }

        Ok(key_points)
    }

    /// Extract TextRank based key points
    async fn extract_textrank_key_points(&self, text: &str) -> Result<Vec<String>> {
        let mut key_points = Vec::new();

        // Split into sentences
        let sentences = self.split_into_sentences(text);
        if sentences.len() < 2 {
            return Ok(key_points);
        }

        // Calculate TextRank scores
        let textrank_scores = self.calculate_textrank_scores(&sentences).await?;

        // Get top-ranked sentences
        let mut scored_sentences: Vec<_> = sentences.iter()
            .zip(textrank_scores.iter())
            .collect();

        scored_sentences.sort_by(|a, b| b.1.partial_cmp(a.1).unwrap_or(std::cmp::Ordering::Equal));

        for (sentence, _score) in scored_sentences.into_iter().take(2) {
            if sentence.len() > 30 && sentence.len() < 250 {
                key_points.push(format!("Central theme: {}", sentence.trim()));
            }
        }

        Ok(key_points)
    }

    /// Extract entity-based key points
    async fn extract_entity_based_key_points(&self, text: &str) -> Result<Vec<String>> {
        let mut key_points = Vec::new();

        // Extract entities
        let entities = self.extract_entities(text).await?;

        // Group entities by type and find most important ones
        let mut entity_groups: std::collections::HashMap<String, Vec<&Entity>> = std::collections::HashMap::new();
        for entity in &entities {
            let entity_type_str = format!("{:?}", entity.entity_type);
            entity_groups.entry(entity_type_str).or_default().push(entity);
        }

        // Create key points from important entities
        for (entity_type, entities_of_type) in entity_groups {
            if entities_of_type.len() >= 2 { // Only include types with multiple entities
                let entity_names: Vec<String> = entities_of_type.iter()
                    .take(3) // Top 3 entities of this type
                    .map(|e| e.name.clone())
                    .collect();

                key_points.push(format!("Important {}: {}",
                    entity_type.to_lowercase(),
                    entity_names.join(", ")
                ));
            }
        }

        Ok(key_points)
    }

    /// Extract importance-weighted key points from individual memories
    async fn extract_importance_weighted_key_points(&self, memories: &[MemoryEntry]) -> Result<Vec<String>> {
        let mut key_points = Vec::new();

        // Sort memories by importance and select top ones
        let mut sorted_memories: Vec<_> = memories.iter().collect();
        sorted_memories.sort_by(|a, b| b.metadata.importance.partial_cmp(&a.metadata.importance).unwrap_or(std::cmp::Ordering::Equal));

        // Extract key sentences from most important memories
        for memory in sorted_memories.iter().take(3) {
            if memory.metadata.importance > 0.7 { // High importance threshold
                let sentences = self.split_into_sentences(&memory.value);
                if let Some(first_sentence) = sentences.first() {
                    if first_sentence.len() > 20 && first_sentence.len() < 180 {
                        key_points.push(format!("High priority: {}", first_sentence.trim()));
                    }
                }
            }
        }

        Ok(key_points)
    }

    /// Extract pattern-based key points
    async fn extract_pattern_based_key_points(&self, text: &str) -> Result<Vec<String>> {
        let mut key_points = Vec::new();

        // Define important patterns
        let action_patterns = [
            r"(?i)(need to|must|should|will|plan to|going to)\s+([^.!?]{10,80})",
            r"(?i)(action|task|todo|assignment):\s*([^.!?]{5,60})",
            r"(?i)(important|critical|urgent|priority):\s*([^.!?]{10,80})",
        ];

        let insight_patterns = [
            r"(?i)(discovered|learned|realized|found out)\s+([^.!?]{10,80})",
            r"(?i)(key insight|main point|conclusion):\s*([^.!?]{10,80})",
            r"(?i)(because|therefore|as a result)\s+([^.!?]{10,80})",
        ];

        // Extract action items
        for pattern in &action_patterns {
            if let Ok(regex) = regex::Regex::new(pattern) {
                for capture in regex.captures_iter(text) {
                    if let Some(action) = capture.get(2) {
                        key_points.push(format!("Action item: {}", action.as_str().trim()));
                    }
                }
            }
        }

        // Extract insights
        for pattern in &insight_patterns {
            if let Ok(regex) = regex::Regex::new(pattern) {
                for capture in regex.captures_iter(text) {
                    if let Some(insight) = capture.get(2) {
                        key_points.push(format!("Key insight: {}", insight.as_str().trim()));
                    }
                }
            }
        }

        Ok(key_points)
    }

    /// Rank and deduplicate key points
    async fn rank_and_deduplicate_key_points(&self, key_points: Vec<String>) -> Result<Vec<String>> {
        let mut unique_points = Vec::new();
        let mut seen_content = std::collections::HashSet::<String>::new();

        for point in key_points {
            // Normalize for deduplication (remove prefixes and clean up)
            let normalized = point
                .replace("Key insight: ", "")
                .replace("Central theme: ", "")
                .replace("Important ", "")
                .replace("High priority: ", "")
                .replace("Action item: ", "")
                .to_lowercase()
                .trim()
                .to_string();

            // Check for similarity with existing points
            let mut is_duplicate = false;
            for existing in &seen_content {
                if self.calculate_string_similarity(&normalized, existing) > 0.8 {
                    is_duplicate = true;
                    break;
                }
            }

            if !is_duplicate && normalized.len() > 10 {
                seen_content.insert(normalized);
                unique_points.push(point);
            }
        }

        // Rank by importance (prioritize action items and insights)
        unique_points.sort_by(|a, b| {
            let score_a = self.calculate_key_point_importance_score(a);
            let score_b = self.calculate_key_point_importance_score(b);
            score_b.partial_cmp(&score_a).unwrap_or(std::cmp::Ordering::Equal)
        });

        Ok(unique_points)
    }

    /// Calculate importance score for a key point
    fn calculate_key_point_importance_score(&self, point: &str) -> f64 {
        let mut score: f64 = 0.5; // Base score

        // Boost action items
        if point.contains("Action item:") || point.contains("need to") || point.contains("must") {
            score += 0.3;
        }

        // Boost insights
        if point.contains("Key insight:") || point.contains("discovered") || point.contains("learned") {
            score += 0.25;
        }

        // Boost high priority items
        if point.contains("High priority:") || point.contains("critical") || point.contains("urgent") {
            score += 0.2;
        }

        // Boost central themes
        if point.contains("Central theme:") {
            score += 0.15;
        }

        // Penalize very short or very long points
        let length = point.len();
        if length < 20 {
            score -= 0.2;
        } else if length > 200 {
            score -= 0.1;
        }

        score.max(0.0).min(1.0)
    }

    /// Split text into sentences using multiple delimiters
    fn split_into_sentences(&self, text: &str) -> Vec<String> {
        let mut sentences = Vec::new();

        // Split on sentence endings
        let parts: Vec<&str> = text.split(&['.', '!', '?'][..]).collect();

        for part in parts {
            let trimmed = part.trim();
            if trimmed.len() > 10 { // Minimum sentence length
                sentences.push(trimmed.to_string());
            }
        }

        // Also split on line breaks for structured text
        let mut additional_sentences = Vec::new();
        for sentence in &sentences {
            let lines: Vec<&str> = sentence.split('\n').collect();
            for line in lines {
                let trimmed = line.trim();
                if trimmed.len() > 15 && !sentences.iter().any(|s| s.contains(trimmed)) {
                    additional_sentences.push(trimmed.to_string());
                }
            }
        }

        sentences.extend(additional_sentences);
        sentences
    }

    /// Calculate TF-IDF score for a sentence
    async fn calculate_sentence_tfidf_score(&self, sentence: &str, all_sentences: &[String]) -> Result<f64> {
        let words: Vec<&str> = sentence.split_whitespace().collect();
        if words.is_empty() {
            return Ok(0.0);
        }

        let mut total_score = 0.0;
        let total_docs = all_sentences.len() as f64;

        for word in &words {
            let word_lower = word.to_lowercase();

            // Skip common words
            if self.is_stop_word(&word_lower) {
                continue;
            }

            // Calculate TF (term frequency in this sentence)
            let tf = words.iter().filter(|w| w.to_lowercase() == word_lower).count() as f64 / words.len() as f64;

            // Calculate DF (document frequency across all sentences)
            let df = all_sentences.iter()
                .filter(|s| s.to_lowercase().contains(&word_lower))
                .count() as f64;

            // Calculate IDF (inverse document frequency)
            let idf = if df > 0.0 {
                (total_docs / df).ln()
            } else {
                0.0
            };

            // TF-IDF score for this word
            let tfidf = tf * idf;
            total_score += tfidf;
        }

        Ok(total_score / words.len() as f64) // Normalize by sentence length
    }

    /// Calculate TextRank scores for sentences
    async fn calculate_textrank_scores(&self, sentences: &[String]) -> Result<Vec<f64>> {
        let n = sentences.len();
        if n == 0 {
            return Ok(Vec::new());
        }

        // Initialize scores
        let mut scores = vec![1.0; n];
        let damping_factor = 0.85;
        let iterations = 30;

        // Calculate similarity matrix
        let mut similarity_matrix = vec![vec![0.0; n]; n];
        for i in 0..n {
            for j in 0..n {
                if i != j {
                    similarity_matrix[i][j] = self.calculate_sentence_similarity(&sentences[i], &sentences[j]);
                }
            }
        }

        // TextRank iterations
        for _ in 0..iterations {
            let mut new_scores = vec![0.0; n];

            for i in 0..n {
                let mut sum = 0.0;
                let mut total_similarity = 0.0;

                for j in 0..n {
                    if i != j && similarity_matrix[j][i] > 0.0 {
                        let outgoing_sum: f64 = similarity_matrix[j].iter().sum();
                        if outgoing_sum > 0.0 {
                            sum += similarity_matrix[j][i] * scores[j] / outgoing_sum;
                        }
                        total_similarity += similarity_matrix[j][i];
                    }
                }

                new_scores[i] = (1.0 - damping_factor) + damping_factor * sum;
            }

            scores = new_scores;
        }

        Ok(scores)
    }

    /// Calculate similarity between two sentences
    fn calculate_sentence_similarity(&self, sentence1: &str, sentence2: &str) -> f64 {
        let words1: std::collections::HashSet<String> = sentence1
            .split_whitespace()
            .map(|w| w.to_lowercase())
            .filter(|w| !self.is_stop_word(w))
            .collect();

        let words2: std::collections::HashSet<String> = sentence2
            .split_whitespace()
            .map(|w| w.to_lowercase())
            .filter(|w| !self.is_stop_word(w))
            .collect();

        if words1.is_empty() || words2.is_empty() {
            return 0.0;
        }

        let intersection = words1.intersection(&words2).count() as f64;
        let union = words1.union(&words2).count() as f64;

        if union > 0.0 {
            intersection / union // Jaccard similarity
        } else {
            0.0
        }
    }

    /// Check if a word is a stop word
    fn is_stop_word(&self, word: &str) -> bool {
        let stop_words = [
            "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with",
            "by", "from", "up", "about", "into", "through", "during", "before", "after", "above",
            "below", "between", "among", "is", "are", "was", "were", "be", "been", "being", "have",
            "has", "had", "do", "does", "did", "will", "would", "could", "should", "may", "might",
            "must", "can", "this", "that", "these", "those", "i", "you", "he", "she", "it", "we",
            "they", "me", "him", "her", "us", "them", "my", "your", "his", "her", "its", "our",
            "their", "what", "which", "who", "when", "where", "why", "how"
        ];

        stop_words.contains(&word)
    }

    /// Calculate string similarity using Jaccard similarity
    fn calculate_string_similarity(&self, str1: &str, str2: &str) -> f64 {
        let words1: std::collections::HashSet<&str> = str1.split_whitespace().collect();
        let words2: std::collections::HashSet<&str> = str2.split_whitespace().collect();

        if words1.is_empty() && words2.is_empty() {
            return 1.0;
        }

        let intersection = words1.intersection(&words2).count() as f64;
        let union = words1.union(&words2).count() as f64;

        if union > 0.0 {
            intersection / union
        } else {
            0.0
        }
    }

    /// Extract entities from text using sophisticated NLP-inspired analysis
    async fn extract_entities(&self, text: &str) -> Result<Vec<Entity>> {
        let mut entities = Vec::new();

        // 1. Extract domain-specific entities with proper classification
        entities.extend(self.extract_domain_specific_entities(text)?);

        // 2. Extract named entities (people, places, organizations)
        entities.extend(self.extract_named_entities_from_text(text)?);

        // 3. Extract temporal entities (dates, times, events)
        entities.extend(self.extract_temporal_entities_from_text(text)?);

        // 4. Extract technical entities (technologies, tools, concepts)
        entities.extend(self.extract_technical_entities(text)?);

        // 5. Extract action entities (tasks, goals, problems, solutions)
        entities.extend(self.extract_action_entities(text)?);

        // Sort by importance and remove duplicates
        entities.sort_by(|a, b| b.importance.partial_cmp(&a.importance).unwrap());
        entities.dedup_by(|a, b| a.name == b.name && a.entity_type == b.entity_type);

        // Limit to top 20 entities to avoid noise
        entities.truncate(20);

        tracing::debug!("Extracted {} entities from text of length {}", entities.len(), text.len());

        Ok(entities)
    }

    /// Extract domain-specific entities with proper classification
    fn extract_domain_specific_entities(&self, text: &str) -> Result<Vec<Entity>> {
        let mut entities = Vec::new();

        // Project-related keywords
        let project_keywords = ["project", "initiative", "program", "campaign", "effort"];
        for keyword in &project_keywords {
            let frequency = text.to_lowercase().matches(keyword).count();
            if frequency > 0 {
                entities.push(Entity {
                    name: keyword.to_string(),
                    entity_type: EntityType::Project,
                    frequency,
                    importance: 0.8,
                    context: self.extract_context_for_keyword(text, keyword),
                });
            }
        }

        // Task-related keywords
        let task_keywords = ["task", "action", "todo", "assignment", "work", "job"];
        for keyword in &task_keywords {
            let frequency = text.to_lowercase().matches(keyword).count();
            if frequency > 0 {
                entities.push(Entity {
                    name: keyword.to_string(),
                    entity_type: EntityType::Task,
                    frequency,
                    importance: 0.7,
                    context: self.extract_context_for_keyword(text, keyword),
                });
            }
        }

        // Goal-related keywords
        let goal_keywords = ["goal", "objective", "target", "aim", "purpose", "mission"];
        for keyword in &goal_keywords {
            let frequency = text.to_lowercase().matches(keyword).count();
            if frequency > 0 {
                entities.push(Entity {
                    name: keyword.to_string(),
                    entity_type: EntityType::Goal,
                    frequency,
                    importance: 0.9,
                    context: self.extract_context_for_keyword(text, keyword),
                });
            }
        }

        // Problem-related keywords
        let problem_keywords = ["problem", "issue", "challenge", "obstacle", "difficulty", "bug"];
        for keyword in &problem_keywords {
            let frequency = text.to_lowercase().matches(keyword).count();
            if frequency > 0 {
                entities.push(Entity {
                    name: keyword.to_string(),
                    entity_type: EntityType::Problem,
                    frequency,
                    importance: 0.8,
                    context: self.extract_context_for_keyword(text, keyword),
                });
            }
        }

        // Solution-related keywords
        let solution_keywords = ["solution", "fix", "resolution", "answer", "approach", "method"];
        for keyword in &solution_keywords {
            let frequency = text.to_lowercase().matches(keyword).count();
            if frequency > 0 {
                entities.push(Entity {
                    name: keyword.to_string(),
                    entity_type: EntityType::Solution,
                    frequency,
                    importance: 0.9,
                    context: self.extract_context_for_keyword(text, keyword),
                });
            }
        }

        Ok(entities)
    }

    /// Extract context around a keyword
    fn extract_context_for_keyword(&self, text: &str, keyword: &str) -> Vec<String> {
        let mut contexts = Vec::new();
        let text_lower = text.to_lowercase();
        let keyword_lower = keyword.to_lowercase();

        let mut start = 0;
        while let Some(pos) = text_lower[start..].find(&keyword_lower) {
            let actual_pos = start + pos;
            let context_start = actual_pos.saturating_sub(30);
            let context_end = (actual_pos + keyword.len() + 30).min(text.len());

            let context = text[context_start..context_end].trim().to_string();
            if !context.is_empty() {
                contexts.push(context);
            }

            start = actual_pos + keyword.len();
            if contexts.len() >= 3 { // Limit to 3 contexts per keyword
                break;
            }
        }

        contexts
    }

    /// Extract named entities from text (people, places, organizations)
    fn extract_named_entities_from_text(&self, text: &str) -> Result<Vec<Entity>> {
        let mut entities = Vec::new();

        // Extract capitalized words that might be proper nouns
        if let Ok(proper_noun_regex) = regex::Regex::new(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b") {
            for mat in proper_noun_regex.find_iter(text) {
                let value = mat.as_str();

                // Skip common words that are capitalized
                let common_words = ["The", "This", "That", "These", "Those", "When", "Where", "Why", "How", "What", "Who"];
                if !common_words.contains(&value) && value.len() > 2 {
                    let entity_type = if value.contains(' ') && value.split_whitespace().count() > 1 {
                        EntityType::Organization // Multi-word proper nouns are likely organizations
                    } else {
                        EntityType::Person // Single word proper nouns are likely people
                    };

                    entities.push(Entity {
                        name: value.to_string(),
                        entity_type,
                        frequency: text.matches(value).count(),
                        importance: 0.7,
                        context: self.extract_context_for_keyword(text, value),
                    });
                }
            }
        }

        Ok(entities)
    }

    /// Extract temporal entities from text
    fn extract_temporal_entities_from_text(&self, text: &str) -> Result<Vec<Entity>> {
        let mut entities = Vec::new();

        // Extract date patterns
        let date_patterns = [
            (r"\d{4}-\d{2}-\d{2}", "Date (ISO format)"),
            (r"\d{1,2}/\d{1,2}/\d{4}", "Date (US format)"),
            (r"(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}", "Date (full month)"),
        ];

        for (pattern, description) in &date_patterns {
            if let Ok(regex) = regex::Regex::new(pattern) {
                for mat in regex.find_iter(text) {
                    entities.push(Entity {
                        name: mat.as_str().to_string(),
                        entity_type: EntityType::Event,
                        frequency: 1,
                        importance: 0.6,
                        context: vec![format!("{}: {}", description, mat.as_str())],
                    });
                }
            }
        }

        // Extract time patterns
        if let Ok(time_regex) = regex::Regex::new(r"\b\d{1,2}:\d{2}(?::\d{2})?\s*(?:AM|PM|am|pm)?\b") {
            for mat in time_regex.find_iter(text) {
                entities.push(Entity {
                    name: mat.as_str().to_string(),
                    entity_type: EntityType::Event,
                    frequency: 1,
                    importance: 0.5,
                    context: vec![format!("Time reference: {}", mat.as_str())],
                });
            }
        }

        Ok(entities)
    }

    /// Extract technical entities (technologies, tools, concepts)
    fn extract_technical_entities(&self, text: &str) -> Result<Vec<Entity>> {
        let mut entities = Vec::new();

        // Technology keywords
        let tech_keywords = ["API", "database", "server", "client", "framework", "library", "algorithm", "protocol"];
        for keyword in &tech_keywords {
            let frequency = text.to_lowercase().matches(&keyword.to_lowercase()).count();
            if frequency > 0 {
                entities.push(Entity {
                    name: keyword.to_string(),
                    entity_type: EntityType::Technology,
                    frequency,
                    importance: 0.8,
                    context: self.extract_context_for_keyword(text, keyword),
                });
            }
        }

        // Extract words with technical patterns (camelCase, snake_case, etc.)
        if let Ok(tech_pattern_regex) = regex::Regex::new(r"\b[a-z]+[A-Z][a-zA-Z]*\b|\b[a-z]+_[a-z_]+\b|\b[A-Z]{2,}\b") {
            for mat in tech_pattern_regex.find_iter(text) {
                let value = mat.as_str();
                if value.len() > 3 { // Filter out short acronyms
                    entities.push(Entity {
                        name: value.to_string(),
                        entity_type: EntityType::Technology,
                        frequency: text.matches(value).count(),
                        importance: 0.6,
                        context: self.extract_context_for_keyword(text, value),
                    });
                }
            }
        }

        Ok(entities)
    }

    /// Extract action entities (tasks, goals, problems, solutions)
    fn extract_action_entities(&self, text: &str) -> Result<Vec<Entity>> {
        let mut entities = Vec::new();

        // Action verbs that indicate tasks or goals
        let action_verbs = ["implement", "develop", "create", "build", "design", "analyze", "optimize", "improve", "fix", "resolve"];
        for verb in &action_verbs {
            let frequency = text.to_lowercase().matches(verb).count();
            if frequency > 0 {
                entities.push(Entity {
                    name: verb.to_string(),
                    entity_type: EntityType::Task,
                    frequency,
                    importance: 0.7,
                    context: self.extract_context_for_keyword(text, verb),
                });
            }
        }

        // Extract phrases that indicate concepts
        let concept_phrases = ["machine learning", "artificial intelligence", "data science", "software engineering", "system design"];
        for phrase in &concept_phrases {
            let frequency = text.to_lowercase().matches(phrase).count();
            if frequency > 0 {
                entities.push(Entity {
                    name: phrase.to_string(),
                    entity_type: EntityType::Concept,
                    frequency,
                    importance: 0.9,
                    context: self.extract_context_for_keyword(text, phrase),
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

    /// Extract key themes from text using sophisticated content analysis
    pub async fn extract_key_themes(&self, text: &str) -> Result<Vec<String>> {
        let mut themes = Vec::new();

        // 1. Extract themes using TF-IDF analysis
        themes.extend(self.extract_tfidf_themes(text).await?);

        // 2. Extract themes using semantic clustering
        themes.extend(self.extract_semantic_cluster_themes(text).await?);

        // 3. Extract themes using topic modeling (LDA-like approach)
        themes.extend(self.extract_topic_model_themes(text).await?);

        // 4. Extract themes based on domain-specific keywords
        themes.extend(self.extract_domain_themes(text)?);

        // 5. Extract themes based on action patterns
        themes.extend(self.extract_action_themes(text)?);

        // 6. Extract themes based on technical content
        themes.extend(self.extract_technical_themes(text)?);

        // 7. Extract themes based on temporal patterns
        themes.extend(self.extract_temporal_themes(text)?);

        // 8. Extract themes based on organizational patterns
        themes.extend(self.extract_organizational_themes(text)?);

        // 9. Extract themes using advanced NLP patterns
        themes.extend(self.extract_nlp_themes(text).await?);

        // Score and rank themes by relevance and frequency
        let scored_themes = self.score_and_rank_themes(&themes, text).await?;

        // Remove duplicates and sort by relevance score
        let mut unique_themes: Vec<(String, f64)> = scored_themes.into_iter().collect();
        unique_themes.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        unique_themes.dedup_by(|a, b| a.0 == b.0);

        // Extract top themes with minimum relevance threshold
        let final_themes: Vec<String> = unique_themes
            .into_iter()
            .filter(|(_, score)| *score > 0.3) // Minimum relevance threshold
            .take(10) // Limit to top 10 themes
            .map(|(theme, _)| theme)
            .collect();

        tracing::debug!("Extracted {} themes from text using advanced analysis", final_themes.len());

        Ok(final_themes)
    }

    /// Extract domain-specific themes
    fn extract_domain_themes(&self, text: &str) -> Result<Vec<String>> {
        let mut themes = Vec::new();
        let text_lower = text.to_lowercase();

        // Information management themes
        if text_lower.contains("information") || text_lower.contains("data") || text_lower.contains("knowledge") {
            themes.push("Information Management".to_string());
        }

        // Project management themes
        if text_lower.contains("project") || text_lower.contains("task") || text_lower.contains("deadline") {
            themes.push("Project Management".to_string());
        }

        // Problem solving themes
        if text_lower.contains("problem") || text_lower.contains("solution") || text_lower.contains("issue") {
            themes.push("Problem Solving".to_string());
        }

        // Learning and development themes
        if text_lower.contains("learn") || text_lower.contains("study") || text_lower.contains("research") {
            themes.push("Learning & Development".to_string());
        }

        // Communication themes
        if text_lower.contains("meeting") || text_lower.contains("discussion") || text_lower.contains("communication") {
            themes.push("Communication".to_string());
        }

        // Technology themes
        if text_lower.contains("software") || text_lower.contains("system") || text_lower.contains("technology") {
            themes.push("Technology".to_string());
        }

        Ok(themes)
    }

    /// Extract action-based themes
    fn extract_action_themes(&self, text: &str) -> Result<Vec<String>> {
        let mut themes = Vec::new();
        let text_lower = text.to_lowercase();

        // Development themes
        if text_lower.contains("develop") || text_lower.contains("build") || text_lower.contains("create") {
            themes.push("Development".to_string());
        }

        // Analysis themes
        if text_lower.contains("analyze") || text_lower.contains("evaluate") || text_lower.contains("assess") {
            themes.push("Analysis".to_string());
        }

        // Planning themes
        if text_lower.contains("plan") || text_lower.contains("strategy") || text_lower.contains("design") {
            themes.push("Planning".to_string());
        }

        // Implementation themes
        if text_lower.contains("implement") || text_lower.contains("execute") || text_lower.contains("deploy") {
            themes.push("Implementation".to_string());
        }

        // Optimization themes
        if text_lower.contains("optimize") || text_lower.contains("improve") || text_lower.contains("enhance") {
            themes.push("Optimization".to_string());
        }

        Ok(themes)
    }

    /// Extract technical themes
    fn extract_technical_themes(&self, text: &str) -> Result<Vec<String>> {
        let mut themes = Vec::new();
        let text_lower = text.to_lowercase();

        // AI/ML themes
        if text_lower.contains("ai") || text_lower.contains("machine learning") || text_lower.contains("neural") {
            themes.push("Artificial Intelligence".to_string());
        }

        // Database themes
        if text_lower.contains("database") || text_lower.contains("sql") || text_lower.contains("query") {
            themes.push("Database Management".to_string());
        }

        // Security themes
        if text_lower.contains("security") || text_lower.contains("encryption") || text_lower.contains("authentication") {
            themes.push("Security".to_string());
        }

        // Performance themes
        if text_lower.contains("performance") || text_lower.contains("optimization") || text_lower.contains("efficiency") {
            themes.push("Performance".to_string());
        }

        // Architecture themes
        if text_lower.contains("architecture") || text_lower.contains("design pattern") || text_lower.contains("framework") {
            themes.push("Software Architecture".to_string());
        }

        Ok(themes)
    }

    /// Extract temporal themes
    fn extract_temporal_themes(&self, text: &str) -> Result<Vec<String>> {
        let mut themes = Vec::new();
        let text_lower = text.to_lowercase();

        // Scheduling themes
        if text_lower.contains("schedule") || text_lower.contains("timeline") || text_lower.contains("deadline") {
            themes.push("Scheduling".to_string());
        }

        // Historical themes
        if text_lower.contains("history") || text_lower.contains("past") || text_lower.contains("previous") {
            themes.push("Historical Analysis".to_string());
        }

        // Future planning themes
        if text_lower.contains("future") || text_lower.contains("upcoming") || text_lower.contains("next") {
            themes.push("Future Planning".to_string());
        }

        Ok(themes)
    }

    /// Extract organizational themes
    fn extract_organizational_themes(&self, text: &str) -> Result<Vec<String>> {
        let mut themes = Vec::new();
        let text_lower = text.to_lowercase();

        // Team collaboration themes
        if text_lower.contains("team") || text_lower.contains("collaboration") || text_lower.contains("group") {
            themes.push("Team Collaboration".to_string());
        }

        // Process improvement themes
        if text_lower.contains("process") || text_lower.contains("workflow") || text_lower.contains("procedure") {
            themes.push("Process Management".to_string());
        }

        // Quality assurance themes
        if text_lower.contains("quality") || text_lower.contains("testing") || text_lower.contains("validation") {
            themes.push("Quality Assurance".to_string());
        }

        // Documentation themes
        if text_lower.contains("document") || text_lower.contains("specification") || text_lower.contains("manual") {
            themes.push("Documentation".to_string());
        }

        Ok(themes)
    }

    /// Extract themes using TF-IDF analysis for term importance
    pub async fn extract_tfidf_themes(&self, text: &str) -> Result<Vec<String>> {
        let mut themes = Vec::new();

        // Tokenize text into sentences and words
        let sentences: Vec<&str> = text.split(&['.', '!', '?'][..]).collect();
        let words: Vec<String> = text
            .to_lowercase()
            .split_whitespace()
            .filter(|w| w.len() > 3) // Filter short words
            .map(|w| w.trim_matches(|c: char| !c.is_alphanumeric()).to_string())
            .filter(|w| !w.is_empty())
            .collect();

        if words.is_empty() {
            return Ok(themes);
        }

        // Calculate term frequency
        let mut term_freq: HashMap<String, f64> = HashMap::new();
        for word in &words {
            *term_freq.entry(word.clone()).or_insert(0.0) += 1.0;
        }

        // Normalize term frequencies
        let total_words = words.len() as f64;
        for freq in term_freq.values_mut() {
            *freq /= total_words;
        }

        // Calculate document frequency (simplified for single document)
        let mut doc_freq: HashMap<String, f64> = HashMap::new();
        for sentence in &sentences {
            let sentence_words: std::collections::HashSet<String> = sentence
                .to_lowercase()
                .split_whitespace()
                .map(|w| w.trim_matches(|c: char| !c.is_alphanumeric()).to_string())
                .filter(|w| w.len() > 3)
                .collect();

            for word in sentence_words {
                *doc_freq.entry(word).or_insert(0.0) += 1.0;
            }
        }

        // Calculate TF-IDF scores
        let num_sentences = sentences.len() as f64;
        let mut tfidf_scores: Vec<(String, f64)> = Vec::new();

        for (term, tf) in term_freq {
            let df = doc_freq.get(&term).unwrap_or(&1.0);
            let idf = (num_sentences / df).ln();
            let tfidf = tf * idf;

            if tfidf > 0.01 { // Lower threshold for significance
                tfidf_scores.push((term, tfidf));
            }
        }

        // Sort by TF-IDF score and extract top terms as themes
        tfidf_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        for (term, _score) in tfidf_scores.into_iter().take(5) {
            themes.push(format!("TF-IDF: {}", term.to_uppercase()));
        }

        Ok(themes)
    }

    /// Extract themes using semantic clustering
    pub async fn extract_semantic_cluster_themes(&self, text: &str) -> Result<Vec<String>> {
        let mut themes = Vec::new();

        // Extract meaningful phrases (2-3 word combinations)
        let words: Vec<String> = text
            .to_lowercase()
            .split_whitespace()
            .filter(|w| w.len() > 2)
            .map(|w| w.trim_matches(|c: char| !c.is_alphanumeric()).to_string())
            .filter(|w| !w.is_empty())
            .collect();

        if words.len() < 4 {
            return Ok(themes);
        }

        // Create bigrams and trigrams
        let mut phrases = Vec::new();
        for i in 0..words.len().saturating_sub(1) {
            phrases.push(format!("{} {}", words[i], words[i + 1]));
        }
        for i in 0..words.len().saturating_sub(2) {
            phrases.push(format!("{} {} {}", words[i], words[i + 1], words[i + 2]));
        }

        // Count phrase frequencies
        let mut phrase_freq: HashMap<String, usize> = HashMap::new();
        for phrase in phrases {
            *phrase_freq.entry(phrase).or_insert(0) += 1;
        }

        // Extract frequent phrases as themes
        let mut frequent_phrases: Vec<(String, usize)> = phrase_freq
            .into_iter()
            .filter(|(_, count)| *count > 1) // Must appear more than once
            .collect();

        frequent_phrases.sort_by(|a, b| b.1.cmp(&a.1));

        for (phrase, _count) in frequent_phrases.into_iter().take(3) {
            themes.push(format!("Semantic: {}", self.to_title_case(&phrase)));
        }

        Ok(themes)
    }

    /// Extract themes using topic modeling approach
    pub async fn extract_topic_model_themes(&self, text: &str) -> Result<Vec<String>> {
        let mut themes = Vec::new();

        // Simplified LDA-like approach using co-occurrence analysis
        let sentences: Vec<&str> = text.split(&['.', '!', '?'][..])
            .filter(|s| s.trim().len() > 10)
            .collect();

        if sentences.len() < 2 {
            return Ok(themes);
        }

        // Extract keywords from each sentence
        let mut sentence_keywords: Vec<Vec<String>> = Vec::new();
        for sentence in sentences {
            let keywords: Vec<String> = sentence
                .to_lowercase()
                .split_whitespace()
                .filter(|w| w.len() > 4) // Longer words are more likely to be meaningful
                .map(|w| w.trim_matches(|c: char| !c.is_alphanumeric()).to_string())
                .filter(|w| !w.is_empty())
                .collect();

            if !keywords.is_empty() {
                sentence_keywords.push(keywords);
            }
        }

        // Find co-occurring terms (simplified topic detection)
        let mut cooccurrence: HashMap<(String, String), usize> = HashMap::new();
        for keywords in &sentence_keywords {
            for i in 0..keywords.len() {
                for j in i + 1..keywords.len() {
                    let pair = if keywords[i] < keywords[j] {
                        (keywords[i].clone(), keywords[j].clone())
                    } else {
                        (keywords[j].clone(), keywords[i].clone())
                    };
                    *cooccurrence.entry(pair).or_insert(0) += 1;
                }
            }
        }

        // Extract strong co-occurrence pairs as topics
        let mut strong_pairs: Vec<((String, String), usize)> = cooccurrence
            .into_iter()
            .filter(|(_, count)| *count > 1)
            .collect();

        strong_pairs.sort_by(|a, b| b.1.cmp(&a.1));

        for ((word1, word2), _count) in strong_pairs.into_iter().take(3) {
            themes.push(format!("Topic: {} & {}", self.to_title_case(&word1), self.to_title_case(&word2)));
        }

        Ok(themes)
    }

    /// Extract themes using advanced NLP patterns
    pub async fn extract_nlp_themes(&self, text: &str) -> Result<Vec<String>> {
        let mut themes = Vec::new();
        let text_lower = text.to_lowercase();

        // Extract themes based on linguistic patterns

        // 1. Causality patterns
        if text_lower.contains("because") || text_lower.contains("due to") || text_lower.contains("caused by") {
            themes.push("Causality Analysis".to_string());
        }

        // 2. Comparison patterns
        if text_lower.contains("compared to") || text_lower.contains("versus") || text_lower.contains("better than") {
            themes.push("Comparative Analysis".to_string());
        }

        // 3. Temporal progression patterns
        if text_lower.contains("first") && text_lower.contains("then") || text_lower.contains("sequence") {
            themes.push("Sequential Process".to_string());
        }

        // 4. Problem-solution patterns
        if (text_lower.contains("problem") || text_lower.contains("issue")) &&
           (text_lower.contains("solution") || text_lower.contains("resolve")) {
            themes.push("Problem Resolution".to_string());
        }

        // 5. Decision-making patterns
        if text_lower.contains("decision") || text_lower.contains("choose") || text_lower.contains("option") {
            themes.push("Decision Making".to_string());
        }

        // 6. Innovation patterns
        if text_lower.contains("innovative") || text_lower.contains("novel") || text_lower.contains("breakthrough") {
            themes.push("Innovation".to_string());
        }

        // 7. Collaboration patterns
        if text_lower.contains("collaborate") || text_lower.contains("together") || text_lower.contains("partnership") {
            themes.push("Collaboration".to_string());
        }

        // 8. Risk assessment patterns
        if text_lower.contains("risk") || text_lower.contains("uncertainty") || text_lower.contains("potential") {
            themes.push("Risk Assessment".to_string());
        }

        Ok(themes)
    }

    /// Score and rank themes by relevance and frequency
    pub async fn score_and_rank_themes(&self, themes: &[String], text: &str) -> Result<HashMap<String, f64>> {
        let mut theme_scores: HashMap<String, f64> = HashMap::new();
        let text_lower = text.to_lowercase();
        let text_len = text.len() as f64;

        for theme in themes {
            let theme_lower = theme.to_lowercase();

            // Base score from frequency
            let frequency_score = text_lower.matches(&theme_lower).count() as f64;

            // Position score (themes mentioned early get higher scores)
            let position_score = if let Some(pos) = text_lower.find(&theme_lower) {
                1.0 - (pos as f64 / text_len)
            } else {
                0.0
            };

            // Length score (longer, more specific themes get higher scores)
            let length_score = (theme.len() as f64).sqrt() / 10.0;

            // Combine scores with weights
            let total_score = frequency_score * 0.5 + position_score * 0.3 + length_score * 0.2;

            *theme_scores.entry(theme.clone()).or_insert(0.0) += total_score;
        }

        Ok(theme_scores)
    }

    /// Helper function to convert string to title case
    pub fn to_title_case(&self, s: &str) -> String {
        s.split_whitespace()
            .map(|word| {
                let mut chars = word.chars();
                match chars.next() {
                    None => String::new(),
                    Some(first) => first.to_uppercase().collect::<String>() + &chars.as_str().to_lowercase(),
                }
            })
            .collect::<Vec<_>>()
            .join(" ")
    }

    /// Calculate quality metrics for a summary using sophisticated analysis
    async fn calculate_quality_metrics(
        &self,
        summary_content: &str,
        memories: &[MemoryEntry],
    ) -> Result<SummaryQualityMetrics> {
        // 1. Calculate coherence (how well the summary flows and connects ideas)
        let coherence = self.calculate_coherence_score(summary_content)?;

        // 2. Calculate completeness (how much of the original information is preserved)
        let completeness = self.calculate_completeness_score(summary_content, memories)?;

        // 3. Calculate conciseness (how efficiently information is presented)
        let conciseness = self.calculate_conciseness_score(summary_content, memories)?;

        // 4. Calculate accuracy (how faithful the summary is to the original content)
        let accuracy = self.calculate_accuracy_score(summary_content, memories)?;

        // Calculate overall quality as weighted average
        let overall_quality = (coherence * 0.25 + completeness * 0.3 + conciseness * 0.2 + accuracy * 0.25);

        tracing::debug!("Quality metrics - Coherence: {:.2}, Completeness: {:.2}, Conciseness: {:.2}, Accuracy: {:.2}, Overall: {:.2}",
            coherence, completeness, conciseness, accuracy, overall_quality);

        Ok(SummaryQualityMetrics {
            coherence,
            completeness,
            conciseness,
            accuracy,
            overall_quality,
        })
    }

    /// Calculate coherence score (how well the summary flows)
    fn calculate_coherence_score(&self, summary_content: &str) -> Result<f64> {
        let mut coherence_factors = Vec::new();

        // 1. Sentence connectivity (presence of transition words)
        let transition_words = ["however", "therefore", "furthermore", "additionally", "consequently", "meanwhile", "similarly", "in contrast"];
        let transition_count = transition_words.iter()
            .map(|word| summary_content.to_lowercase().matches(word).count())
            .sum::<usize>();
        let sentence_count = summary_content.split(&['.', '!', '?'][..]).count();
        let transition_score = if sentence_count > 1 {
            (transition_count as f64 / (sentence_count - 1) as f64).min(1.0)
        } else {
            0.5
        };
        coherence_factors.push(transition_score);

        // 2. Consistent terminology (repeated key terms)
        let words: Vec<&str> = summary_content.split_whitespace().collect();
        let unique_words: std::collections::HashSet<&str> = words.iter().cloned().collect();
        let repetition_score = if words.len() > 0 {
            1.0 - (unique_words.len() as f64 / words.len() as f64)
        } else {
            0.0
        };
        coherence_factors.push(repetition_score);

        // 3. Logical structure (presence of organizational markers)
        let structure_markers = ["first", "second", "third", "finally", "in conclusion", "to summarize"];
        let structure_count = structure_markers.iter()
            .map(|marker| summary_content.to_lowercase().matches(marker).count())
            .sum::<usize>();
        let structure_score = (structure_count as f64 / 3.0).min(1.0); // Normalize to max 3 markers
        coherence_factors.push(structure_score);

        let coherence = if coherence_factors.is_empty() {
            0.5
        } else {
            coherence_factors.iter().sum::<f64>() / coherence_factors.len() as f64
        };

        Ok(coherence.min(1.0))
    }

    /// Calculate completeness score (how much information is preserved)
    fn calculate_completeness_score(&self, summary_content: &str, memories: &[MemoryEntry]) -> Result<f64> {
        if memories.is_empty() {
            return Ok(0.0);
        }

        let mut completeness_factors = Vec::new();

        // 1. Key term coverage
        let mut all_original_words = std::collections::HashSet::new();
        for memory in memories {
            for word in memory.value.split_whitespace() {
                if word.len() > 3 { // Only consider significant words
                    all_original_words.insert(word.to_lowercase());
                }
            }
        }

        let summary_words: std::collections::HashSet<String> = summary_content
            .split_whitespace()
            .filter(|word| word.len() > 3)
            .map(|word| word.to_lowercase())
            .collect();

        let covered_words = all_original_words.iter()
            .filter(|word| summary_words.contains(*word))
            .count();

        let term_coverage = if all_original_words.is_empty() {
            0.0
        } else {
            covered_words as f64 / all_original_words.len() as f64
        };
        completeness_factors.push(term_coverage);

        // 2. Important memory coverage (based on importance scores)
        let high_importance_memories = memories.iter()
            .filter(|m| m.metadata.importance > 0.7)
            .count();

        let covered_important_concepts = memories.iter()
            .filter(|m| m.metadata.importance > 0.7)
            .filter(|m| {
                let memory_words: Vec<&str> = m.value.split_whitespace().collect();
                memory_words.iter().any(|word| summary_content.to_lowercase().contains(&word.to_lowercase()))
            })
            .count();

        let importance_coverage = if high_importance_memories > 0 {
            covered_important_concepts as f64 / high_importance_memories as f64
        } else {
            1.0 // No high importance memories to cover
        };
        completeness_factors.push(importance_coverage);

        // 3. Memory count representation
        let memory_representation = (memories.len() as f64).log2() / 10.0; // Logarithmic scaling
        completeness_factors.push(memory_representation.min(1.0));

        let completeness = if completeness_factors.is_empty() {
            0.0
        } else {
            completeness_factors.iter().sum::<f64>() / completeness_factors.len() as f64
        };

        Ok(completeness.min(1.0))
    }

    /// Calculate conciseness score (efficiency of information presentation)
    fn calculate_conciseness_score(&self, summary_content: &str, memories: &[MemoryEntry]) -> Result<f64> {
        let original_length: usize = memories.iter().map(|m| m.value.len()).sum();
        let summary_length = summary_content.len();

        if original_length == 0 {
            return Ok(0.0);
        }

        // 1. Compression ratio (higher compression = more concise, but not too high)
        let compression_ratio = original_length as f64 / summary_length as f64;
        let compression_score = if compression_ratio > 10.0 {
            0.5 // Too much compression might lose information
        } else if compression_ratio > 2.0 {
            1.0 // Good compression
        } else {
            compression_ratio / 2.0 // Linear scaling for low compression
        };

        // 2. Information density (meaningful words per total words)
        let words: Vec<&str> = summary_content.split_whitespace().collect();
        let stop_words = ["the", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by"];
        let meaningful_words = words.iter()
            .filter(|word| !stop_words.contains(&word.to_lowercase().as_str()))
            .count();

        let density_score = if words.is_empty() {
            0.0
        } else {
            meaningful_words as f64 / words.len() as f64
        };

        // 3. Redundancy check (repeated phrases)
        let sentences: Vec<&str> = summary_content.split(&['.', '!', '?'][..]).collect();
        let unique_sentences: std::collections::HashSet<&str> = sentences.iter().cloned().collect();
        let redundancy_score = if sentences.is_empty() {
            1.0
        } else {
            unique_sentences.len() as f64 / sentences.len() as f64
        };

        let conciseness = (compression_score * 0.4 + density_score * 0.3 + redundancy_score * 0.3);

        Ok(conciseness.min(1.0))
    }

    /// Calculate accuracy score (faithfulness to original content)
    fn calculate_accuracy_score(&self, summary_content: &str, memories: &[MemoryEntry]) -> Result<f64> {
        if memories.is_empty() {
            return Ok(0.0);
        }

        let mut accuracy_factors = Vec::new();

        // 1. Factual consistency (no contradictory information)
        // This is a simplified check - in a full implementation, this would use NLP
        let summary_lower = summary_content.to_lowercase();
        let contradiction_indicators = ["not", "never", "opposite", "contrary", "however", "but"];
        let contradiction_count = contradiction_indicators.iter()
            .map(|indicator| summary_lower.matches(indicator).count())
            .sum::<usize>();

        let consistency_score = if summary_content.len() > 0 {
            1.0 - (contradiction_count as f64 / (summary_content.len() / 100) as f64).min(1.0)
        } else {
            0.0
        };
        accuracy_factors.push(consistency_score);

        // 2. Semantic similarity (shared concepts and terms)
        let mut original_concepts = std::collections::HashSet::new();
        for memory in memories {
            for word in memory.value.split_whitespace() {
                if word.len() > 4 { // Focus on substantial words
                    original_concepts.insert(word.to_lowercase());
                }
            }
        }

        let summary_concepts: std::collections::HashSet<String> = summary_content
            .split_whitespace()
            .filter(|word| word.len() > 4)
            .map(|word| word.to_lowercase())
            .collect();

        let concept_overlap = original_concepts.intersection(&summary_concepts).count();
        let semantic_score = if original_concepts.is_empty() {
            0.0
        } else {
            concept_overlap as f64 / original_concepts.len() as f64
        };
        accuracy_factors.push(semantic_score);

        // 3. Tone preservation (positive/negative sentiment consistency)
        let positive_words = ["good", "great", "excellent", "successful", "effective", "improved"];
        let negative_words = ["bad", "poor", "failed", "problem", "issue", "error"];

        let original_positive = memories.iter()
            .map(|m| positive_words.iter().map(|w| m.value.to_lowercase().matches(w).count()).sum::<usize>())
            .sum::<usize>();
        let original_negative = memories.iter()
            .map(|m| negative_words.iter().map(|w| m.value.to_lowercase().matches(w).count()).sum::<usize>())
            .sum::<usize>();

        let summary_positive = positive_words.iter().map(|w| summary_lower.matches(w).count()).sum::<usize>();
        let summary_negative = negative_words.iter().map(|w| summary_lower.matches(w).count()).sum::<usize>();

        let tone_score = if original_positive + original_negative > 0 {
            let original_sentiment = original_positive as f64 / (original_positive + original_negative) as f64;
            let summary_sentiment = if summary_positive + summary_negative > 0 {
                summary_positive as f64 / (summary_positive + summary_negative) as f64
            } else {
                0.5 // Neutral
            };
            1.0 - (original_sentiment - summary_sentiment).abs()
        } else {
            1.0 // No sentiment to preserve
        };
        accuracy_factors.push(tone_score);

        let accuracy = if accuracy_factors.is_empty() {
            0.0
        } else {
            accuracy_factors.iter().sum::<f64>() / accuracy_factors.len() as f64
        };

        Ok(accuracy.min(1.0))
    }

    /// Check if summarization should be triggered using comprehensive multi-strategy analysis
    /// Uses 6 sophisticated triggering strategies with configurable thresholds
    pub async fn should_trigger_summarization(
        &self,
        storage: &(dyn crate::memory::storage::Storage + Send + Sync),
        memory: &MemoryEntry,
    ) -> Result<bool> {
        tracing::debug!("Evaluating summarization triggers for memory: {}", memory.key);
        let start_time = std::time::Instant::now();

        // Strategy 1: Related memory count threshold (trigger if > 10 related memories)
        let related_count_trigger = self.check_related_count_trigger(storage, memory).await?;
        tracing::debug!("Related count trigger: {}", related_count_trigger);

        // Strategy 2: Memory age threshold (trigger if oldest related memory > 7 days)
        let age_threshold_trigger = self.check_age_threshold_trigger(storage, memory).await?;
        tracing::debug!("Age threshold trigger: {}", age_threshold_trigger);

        // Strategy 3: Content similarity clustering (trigger if high similarity cluster detected)
        let similarity_cluster_trigger = self.check_similarity_cluster_trigger(storage, memory).await?;
        tracing::debug!("Similarity cluster trigger: {}", similarity_cluster_trigger);

        // Strategy 4: Importance accumulation (trigger if total importance > threshold)
        let importance_accumulation_trigger = self.check_importance_accumulation_trigger(storage, memory).await?;
        tracing::debug!("Importance accumulation trigger: {}", importance_accumulation_trigger);

        // Strategy 5: Tag-based consolidation rules (trigger if consolidation rules match)
        let tag_based_trigger = self.check_tag_based_trigger(memory).await?;
        tracing::debug!("Tag-based trigger: {}", tag_based_trigger);

        // Strategy 6: Temporal pattern detection (trigger if temporal patterns detected)
        let temporal_pattern_trigger = self.check_temporal_pattern_trigger(storage, memory).await?;
        tracing::debug!("Temporal pattern trigger: {}", temporal_pattern_trigger);

        // Combine triggers using OR logic (any trigger can initiate summarization)
        let should_trigger = related_count_trigger
            || age_threshold_trigger
            || similarity_cluster_trigger
            || importance_accumulation_trigger
            || tag_based_trigger
            || temporal_pattern_trigger;

        let duration = start_time.elapsed();
        tracing::info!(
            "Summarization trigger evaluation completed: {} (triggers: count={}, age={}, similarity={}, importance={}, tags={}, temporal={}) in {:?}",
            should_trigger, related_count_trigger, age_threshold_trigger, similarity_cluster_trigger,
            importance_accumulation_trigger, tag_based_trigger, temporal_pattern_trigger, duration
        );

        Ok(should_trigger)
    }

    /// Strategy 1: Check if related memory count exceeds threshold (> 10 related memories)
    async fn check_related_count_trigger(
        &self,
        storage: &(dyn crate::memory::storage::Storage + Send + Sync),
        memory: &MemoryEntry,
    ) -> Result<bool> {
        // Use sophisticated multi-strategy related memory counting
        let related_count = self.count_related_memories_comprehensive(storage, memory).await?;
        let threshold = 10; // Configurable threshold

        tracing::debug!("Related memory count: {} (threshold: {})", related_count, threshold);

        Ok(related_count > threshold)
    }

    /// Count related memories using comprehensive multi-strategy approach
    async fn count_related_memories_comprehensive(
        &self,
        storage: &(dyn crate::memory::storage::Storage + Send + Sync),
        memory: &MemoryEntry,
    ) -> Result<usize> {
        let mut related_memories = std::collections::HashSet::new();

        // Strategy 1: Content similarity (using word overlap)
        let memory_words: std::collections::HashSet<String> = memory.value
            .split_whitespace()
            .filter(|word| word.len() > 3) // Filter out short words
            .map(|word| word.to_lowercase())
            .collect();

        // Get all memories from storage for comparison
        let all_memory_keys = storage.list_keys().await?;
        for key in &all_memory_keys {
            if key == &memory.key {
                continue; // Skip self
            }

            if let Some(other_memory) = storage.retrieve(key).await? {
                let other_words: std::collections::HashSet<String> = other_memory.value
                    .split_whitespace()
                    .filter(|word| word.len() > 3)
                    .map(|word| word.to_lowercase())
                    .collect();

                let intersection = memory_words.intersection(&other_words).count();
                let union = memory_words.union(&other_words).count();
                let similarity = if union > 0 { intersection as f64 / union as f64 } else { 0.0 };

                if similarity > 0.3 { // Similarity threshold
                    related_memories.insert(key.clone());
                }
            }
        }

        // Strategy 2: Tag-based relationships
        for key in &all_memory_keys {
            if key == &memory.key {
                continue;
            }

            if let Some(other_memory) = storage.retrieve(key).await? {
                let memory_tags: std::collections::HashSet<_> = memory.metadata.tags.iter().collect();
                let other_tags: std::collections::HashSet<_> = other_memory.metadata.tags.iter().collect();
                let tag_overlap = memory_tags.intersection(&other_tags).count();

                if tag_overlap > 0 {
                    related_memories.insert(key.clone());
                }
            }
        }

        // Strategy 3: Temporal proximity (memories created within 24 hours)
        let time_window = chrono::Duration::hours(24);
        for key in &all_memory_keys {
            if key == &memory.key {
                continue;
            }

            if let Some(other_memory) = storage.retrieve(key).await? {
                let time_diff = (memory.metadata.created_at - other_memory.metadata.created_at).abs();
                if time_diff < time_window {
                    related_memories.insert(key.clone());
                }
            }
        }

        Ok(related_memories.len())
    }

    /// Strategy 2: Check if oldest related memory exceeds age threshold (> 7 days)
    async fn check_age_threshold_trigger(
        &self,
        _storage: &(dyn crate::memory::storage::Storage + Send + Sync),
        memory: &MemoryEntry,
    ) -> Result<bool> {
        let age_threshold = chrono::Duration::days(7);
        let memory_age = Utc::now() - memory.metadata.created_at;

        // In a full implementation, this would:
        // 1. Find all related memories
        // 2. Check age of oldest related memory
        // 3. Trigger if oldest memory exceeds threshold

        Ok(memory_age > age_threshold)
    }

    /// Strategy 3: Check if high similarity cluster detected (similarity > 0.8)
    async fn check_similarity_cluster_trigger(
        &self,
        storage: &(dyn crate::memory::storage::Storage + Send + Sync),
        memory: &MemoryEntry,
    ) -> Result<bool> {
        let similarity_threshold = 0.8;
        let cluster_size_threshold = 5;

        // Find all related memories
        let related_memories = self.find_related_memories_for_clustering(storage, memory).await?;

        if related_memories.len() < cluster_size_threshold {
            return Ok(false);
        }

        // Calculate pairwise similarities within the cluster
        let mut high_similarity_pairs = 0;
        let total_pairs = related_memories.len() * (related_memories.len() - 1) / 2;

        for i in 0..related_memories.len() {
            for j in (i + 1)..related_memories.len() {
                let similarity = self.calculate_memory_similarity(&related_memories[i], &related_memories[j])?;
                if similarity > similarity_threshold {
                    high_similarity_pairs += 1;
                }
            }
        }

        // Trigger if more than 50% of pairs have high similarity
        let high_similarity_ratio = if total_pairs > 0 {
            high_similarity_pairs as f64 / total_pairs as f64
        } else {
            0.0
        };

        tracing::debug!("Similarity cluster analysis: {} high similarity pairs out of {} total pairs (ratio: {:.2})",
            high_similarity_pairs, total_pairs, high_similarity_ratio);

        Ok(high_similarity_ratio > 0.5)
    }

    /// Find related memories for clustering analysis
    async fn find_related_memories_for_clustering(
        &self,
        storage: &(dyn crate::memory::storage::Storage + Send + Sync),
        memory: &MemoryEntry,
    ) -> Result<Vec<MemoryEntry>> {
        let mut related_memories = Vec::new();
        related_memories.push(memory.clone()); // Include the target memory

        let all_memory_keys = storage.list_keys().await?;
        for key in all_memory_keys {
            if key == memory.key {
                continue; // Skip self (already added)
            }

            if let Some(other_memory) = storage.retrieve(&key).await? {
                // Check if this memory is related (using multiple criteria)
                let is_related = self.is_memory_related(memory, &other_memory)?;
                if is_related {
                    related_memories.push(other_memory);
                }
            }
        }

        Ok(related_memories)
    }

    /// Check if two memories are related
    fn is_memory_related(&self, memory1: &MemoryEntry, memory2: &MemoryEntry) -> Result<bool> {
        // 1. Tag overlap
        let memory1_tags: std::collections::HashSet<_> = memory1.metadata.tags.iter().collect();
        let memory2_tags: std::collections::HashSet<_> = memory2.metadata.tags.iter().collect();
        let tag_overlap = memory1_tags.intersection(&memory2_tags).count();
        if tag_overlap > 0 {
            return Ok(true);
        }

        // 2. Content similarity
        let similarity = self.calculate_memory_similarity(memory1, memory2)?;
        if similarity > 0.3 {
            return Ok(true);
        }

        // 3. Temporal proximity (within 48 hours)
        let time_diff = (memory1.metadata.created_at - memory2.metadata.created_at).abs();
        if time_diff < chrono::Duration::hours(48) {
            return Ok(true);
        }

        Ok(false)
    }

    /// Calculate similarity between two memories
    fn calculate_memory_similarity(&self, memory1: &MemoryEntry, memory2: &MemoryEntry) -> Result<f64> {
        let words1: std::collections::HashSet<String> = memory1.value
            .split_whitespace()
            .filter(|word| word.len() > 3)
            .map(|word| word.to_lowercase())
            .collect();

        let words2: std::collections::HashSet<String> = memory2.value
            .split_whitespace()
            .filter(|word| word.len() > 3)
            .map(|word| word.to_lowercase())
            .collect();

        let intersection = words1.intersection(&words2).count();
        let union = words1.union(&words2).count();

        let jaccard_similarity = if union > 0 {
            intersection as f64 / union as f64
        } else {
            0.0
        };

        // Also consider importance similarity
        let importance_diff = (memory1.metadata.importance - memory2.metadata.importance).abs();
        let importance_similarity = 1.0 - importance_diff;

        // Combine similarities with weights
        let combined_similarity = jaccard_similarity * 0.8 + importance_similarity * 0.2;

        Ok(combined_similarity)
    }

    /// Strategy 4: Check if importance accumulation exceeds threshold (total importance > 15.0)
    async fn check_importance_accumulation_trigger(
        &self,
        storage: &(dyn crate::memory::storage::Storage + Send + Sync),
        memory: &MemoryEntry,
    ) -> Result<bool> {
        let importance_threshold = 15.0;

        // Find all related memories and sum their importance scores
        let related_memories = self.find_related_memories_for_clustering(storage, memory).await?;

        let total_importance: f64 = related_memories.iter()
            .map(|m| m.metadata.importance)
            .sum();

        tracing::debug!("Importance accumulation: {:.2} (threshold: {:.2}, related memories: {})",
            total_importance, importance_threshold, related_memories.len());

        Ok(total_importance > importance_threshold)
    }

    /// Strategy 5: Check if tag-based consolidation rules match
    async fn check_tag_based_trigger(&self, memory: &MemoryEntry) -> Result<bool> {
        for rule in &self.consolidation_rules {
            if !rule.active {
                continue;
            }

            // Check if any trigger tags match memory tags
            for trigger_tag in &rule.trigger_tags {
                if memory.metadata.tags.contains(trigger_tag) {
                    tracing::debug!("Tag-based trigger matched rule '{}' with tag '{}'", rule.name, trigger_tag);
                    return Ok(true);
                }
            }
        }

        Ok(false)
    }

    /// Strategy 6: Check if temporal patterns detected (regular intervals, bursts, etc.)
    async fn check_temporal_pattern_trigger(
        &self,
        storage: &(dyn crate::memory::storage::Storage + Send + Sync),
        memory: &MemoryEntry,
    ) -> Result<bool> {
        // Find related memories for temporal analysis
        let related_memories = self.find_related_memories_for_clustering(storage, memory).await?;

        if related_memories.len() < 3 {
            return Ok(false); // Need at least 3 memories to detect patterns
        }

        // Sort memories by creation time
        let mut sorted_memories = related_memories;
        sorted_memories.sort_by_key(|m| m.metadata.created_at);

        // Analyze temporal patterns
        let burst_detected = self.detect_burst_pattern(&sorted_memories)?;
        let regular_interval_detected = self.detect_regular_interval_pattern(&sorted_memories)?;

        tracing::debug!("Temporal pattern analysis: burst={}, regular_interval={}",
            burst_detected, regular_interval_detected);

        Ok(burst_detected || regular_interval_detected)
    }

    /// Detect burst pattern (multiple memories created in short time span)
    fn detect_burst_pattern(&self, sorted_memories: &[MemoryEntry]) -> Result<bool> {
        let burst_window = chrono::Duration::hours(2); // 2-hour window
        let burst_threshold = 3; // At least 3 memories in the window

        for i in 0..sorted_memories.len() {
            let window_start = sorted_memories[i].metadata.created_at;
            let window_end = window_start + burst_window;

            let memories_in_window = sorted_memories.iter()
                .filter(|m| m.metadata.created_at >= window_start && m.metadata.created_at <= window_end)
                .count();

            if memories_in_window >= burst_threshold {
                return Ok(true);
            }
        }

        Ok(false)
    }

    /// Detect regular interval pattern (memories created at regular intervals)
    fn detect_regular_interval_pattern(&self, sorted_memories: &[MemoryEntry]) -> Result<bool> {
        if sorted_memories.len() < 3 {
            return Ok(false);
        }

        // Calculate intervals between consecutive memories
        let mut intervals = Vec::new();
        for i in 1..sorted_memories.len() {
            let interval = sorted_memories[i].metadata.created_at - sorted_memories[i-1].metadata.created_at;
            intervals.push(interval.num_seconds().abs() as f64);
        }

        // Check if intervals are relatively consistent (coefficient of variation < 0.5)
        let mean_interval = intervals.iter().sum::<f64>() / intervals.len() as f64;
        let variance = intervals.iter()
            .map(|x| (x - mean_interval).powi(2))
            .sum::<f64>() / intervals.len() as f64;
        let std_dev = variance.sqrt();

        let coefficient_of_variation = if mean_interval > 0.0 {
            std_dev / mean_interval
        } else {
            1.0
        };

        // Regular pattern detected if coefficient of variation is low
        Ok(coefficient_of_variation < 0.5)
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

    /// Apply a specific consolidation rule using sophisticated analysis
    async fn apply_consolidation_rule(
        &self,
        rule: &ConsolidationRule,
        memories: &[MemoryEntry],
    ) -> Result<Vec<Vec<String>>> {
        let mut candidates = Vec::new();

        // Step 1: Filter memories by rule criteria
        let eligible_memories: Vec<&MemoryEntry> = memories.iter()
            .filter(|memory| {
                // Check minimum importance threshold
                if memory.metadata.importance < rule.min_importance {
                    return false;
                }

                // Check if memory has any trigger tags
                if !rule.trigger_tags.is_empty() {
                    let has_trigger_tag = rule.trigger_tags.iter()
                        .any(|tag| memory.metadata.tags.contains(tag));
                    if !has_trigger_tag {
                        return false;
                    }
                }

                true
            })
            .collect();

        if eligible_memories.len() < 2 {
            return Ok(candidates); // Need at least 2 memories to consolidate
        }

        // Step 2: Group memories by similarity
        let mut groups = Vec::new();
        let mut processed = std::collections::HashSet::new();

        for (i, memory1) in eligible_memories.iter().enumerate() {
            if processed.contains(&i) {
                continue;
            }

            let mut group = vec![memory1.key.clone()];
            processed.insert(i);

            // Find similar memories for this group
            for (j, memory2) in eligible_memories.iter().enumerate() {
                if i == j || processed.contains(&j) {
                    continue;
                }

                // Check similarity threshold
                let similarity = self.calculate_memory_similarity(memory1, memory2)?;
                if similarity < rule.similarity_threshold {
                    continue;
                }

                // Check age difference threshold
                let age_diff = (memory1.metadata.created_at - memory2.metadata.created_at).abs();
                let max_age_diff = chrono::Duration::hours(rule.max_age_difference_hours as i64);
                if age_diff > max_age_diff {
                    continue;
                }

                // Add to group
                group.push(memory2.key.clone());
                processed.insert(j);
            }

            // Only add groups with multiple memories
            if group.len() > 1 {
                groups.push(group);
            }
        }

        // Step 3: Apply additional consolidation logic
        for group in groups {
            if group.len() >= 2 {
                candidates.push(group);
            }
        }

        tracing::debug!("Consolidation rule '{}' found {} candidate groups", rule.name, candidates.len());

        Ok(candidates)
    }
}

impl Default for MemorySummarizer {
    fn default() -> Self {
        Self::new()
    }
}
