// Intelligence Module
// Advanced memory intelligence and pattern recognition

use crate::error::Result;
use crate::analytics::{AnalyticsEvent, AnalyticsConfig, AnalyticsInsight, InsightType, InsightPriority};
use crate::memory::types::MemoryEntry;
use chrono::{DateTime, Utc, Duration, Timelike};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use uuid::Uuid;

/// Memory intelligence analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryIntelligence {
    /// Memory key
    pub memory_key: String,
    /// Intelligence score (0.0 to 1.0)
    pub intelligence_score: f64,
    /// Complexity analysis
    pub complexity: ComplexityAnalysis,
    /// Relationship intelligence
    pub relationship_intelligence: RelationshipIntelligence,
    /// Usage intelligence
    pub usage_intelligence: UsageIntelligence,
    /// Content intelligence
    pub content_intelligence: ContentIntelligence,
    /// Last analysis timestamp
    pub analyzed_at: DateTime<Utc>,
}

/// Complexity analysis of memory content
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComplexityAnalysis {
    /// Content length complexity
    pub length_complexity: f64,
    /// Vocabulary complexity
    pub vocabulary_complexity: f64,
    /// Structural complexity
    pub structural_complexity: f64,
    /// Conceptual complexity
    pub conceptual_complexity: f64,
    /// Overall complexity score
    pub overall_complexity: f64,
}

/// Relationship intelligence analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RelationshipIntelligence {
    /// Number of direct relationships
    pub direct_relationships: usize,
    /// Number of indirect relationships (2-hop)
    pub indirect_relationships: usize,
    /// Centrality score in the graph
    pub centrality_score: f64,
    /// Clustering coefficient
    pub clustering_coefficient: f64,
    /// Bridge potential (connects different clusters)
    pub bridge_potential: f64,
}

/// Usage intelligence analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UsageIntelligence {
    /// Access frequency score
    pub access_frequency: f64,
    /// Temporal consistency score
    pub temporal_consistency: f64,
    /// User diversity score
    pub user_diversity: f64,
    /// Context diversity score
    pub context_diversity: f64,
    /// Collaboration score
    pub collaboration_score: f64,
}

/// Content intelligence analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContentIntelligence {
    /// Semantic richness score
    pub semantic_richness: f64,
    /// Information density score
    pub information_density: f64,
    /// Uniqueness score
    pub uniqueness: f64,
    /// Relevance score
    pub relevance: f64,
    /// Quality score
    pub quality: f64,
}

/// Pattern recognition result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PatternRecognition {
    /// Pattern identifier
    pub pattern_id: String,
    /// Pattern type
    pub pattern_type: PatternType,
    /// Pattern description
    pub description: String,
    /// Confidence score
    pub confidence: f64,
    /// Affected memories
    pub affected_memories: Vec<String>,
    /// Pattern strength
    pub strength: f64,
    /// Discovery timestamp
    pub discovered_at: DateTime<Utc>,
}

/// Types of patterns that can be recognized
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum PatternType {
    /// Temporal access patterns
    TemporalAccess,
    /// Content similarity patterns
    ContentSimilarity,
    /// User behavior patterns
    UserBehavior,
    /// Relationship formation patterns
    RelationshipFormation,
    /// Memory evolution patterns
    MemoryEvolution,
    /// Anomaly patterns
    Anomaly,
}

/// Anomaly detection result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnomalyDetection {
    /// Anomaly identifier
    pub anomaly_id: String,
    /// Anomaly type
    pub anomaly_type: AnomalyType,
    /// Severity level
    pub severity: AnomalySeverity,
    /// Description of the anomaly
    pub description: String,
    /// Affected memory or system component
    pub affected_component: String,
    /// Anomaly score (higher = more anomalous)
    pub anomaly_score: f64,
    /// Detection timestamp
    pub detected_at: DateTime<Utc>,
    /// Recommended actions
    pub recommended_actions: Vec<String>,
}

/// Types of anomalies
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum AnomalyType {
    /// Unusual access patterns
    AccessPattern,
    /// Content anomalies
    ContentAnomaly,
    /// Performance anomalies
    Performance,
    /// Relationship anomalies
    Relationship,
    /// Data integrity issues
    DataIntegrity,
}

/// Anomaly severity levels
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, PartialOrd)]
pub enum AnomalySeverity {
    /// Low severity anomaly
    Low,
    /// Medium severity anomaly
    Medium,
    /// High severity anomaly
    High,
    /// Critical severity anomaly
    Critical,
}

/// Memory intelligence engine
#[derive(Debug)]
pub struct MemoryIntelligenceEngine {
    /// Configuration
    _config: AnalyticsConfig,
    /// Memory intelligence cache
    intelligence_cache: HashMap<String, MemoryIntelligence>,
    /// Recognized patterns
    patterns: Vec<PatternRecognition>,
    /// Detected anomalies
    anomalies: Vec<AnomalyDetection>,
    /// Analysis history
    analysis_history: Vec<AnalyticsEvent>,
    /// Baseline metrics for anomaly detection
    baseline_metrics: HashMap<String, f64>,
}

impl MemoryIntelligenceEngine {
    /// Create a new memory intelligence engine
    pub fn new(config: &AnalyticsConfig) -> Result<Self> {
        Ok(Self {
            _config: config.clone(),
            intelligence_cache: HashMap::new(),
            patterns: Vec::new(),
            anomalies: Vec::new(),
            analysis_history: Vec::new(),
            baseline_metrics: HashMap::new(),
        })
    }

    /// Analyze memory intelligence
    pub async fn analyze_memory_intelligence(&mut self, memory_key: &str, memory_entry: &MemoryEntry, relationships: &[(String, f64)]) -> Result<MemoryIntelligence> {
        let complexity = self.analyze_complexity(memory_entry).await?;
        let relationship_intelligence = self.analyze_relationship_intelligence(memory_key, relationships).await?;
        let usage_intelligence = self.analyze_usage_intelligence(memory_key, memory_entry).await?;
        let content_intelligence = self.analyze_content_intelligence(memory_entry).await?;

        // Calculate overall intelligence score
        let intelligence_score = (
            complexity.overall_complexity * 0.2 +
            relationship_intelligence.centrality_score * 0.3 +
            usage_intelligence.access_frequency * 0.2 +
            content_intelligence.semantic_richness * 0.3
        ).min(1.0);

        let intelligence = MemoryIntelligence {
            memory_key: memory_key.to_string(),
            intelligence_score,
            complexity,
            relationship_intelligence,
            usage_intelligence,
            content_intelligence,
            analyzed_at: Utc::now(),
        };

        self.intelligence_cache.insert(memory_key.to_string(), intelligence.clone());
        Ok(intelligence)
    }

    /// Analyze content complexity
    async fn analyze_complexity(&self, memory_entry: &MemoryEntry) -> Result<ComplexityAnalysis> {
        let content = &memory_entry.value;
        
        // Length complexity (normalized)
        let length_complexity = (content.len() as f64 / 1000.0).min(1.0);
        
        // Vocabulary complexity (unique words ratio)
        let words: Vec<&str> = content.split_whitespace().collect();
        let unique_words: HashSet<&str> = words.iter().cloned().collect();
        let vocabulary_complexity = if words.is_empty() {
            0.0
        } else {
            unique_words.len() as f64 / words.len() as f64
        };
        
        // Structural complexity (sentence count, punctuation)
        let sentences = content.split(&['.', '!', '?'][..]).count();
        let structural_complexity = (sentences as f64 / 10.0).min(1.0);
        
        // Conceptual complexity using sophisticated NLP-inspired analysis
        let conceptual_complexity = self.calculate_conceptual_complexity(content)?;
        
        let overall_complexity = (length_complexity + vocabulary_complexity + structural_complexity + conceptual_complexity) / 4.0;

        Ok(ComplexityAnalysis {
            length_complexity,
            vocabulary_complexity,
            structural_complexity,
            conceptual_complexity,
            overall_complexity,
        })
    }

    /// Analyze relationship intelligence
    async fn analyze_relationship_intelligence(&self, _memory_key: &str, relationships: &[(String, f64)]) -> Result<RelationshipIntelligence> {
        let direct_relationships = relationships.len();
        
        // Estimate indirect relationships (simplified)
        let indirect_relationships = direct_relationships * 2;
        
        // Calculate centrality score based on relationship count and strength
        let total_strength: f64 = relationships.iter().map(|(_, strength)| strength).sum();
        let centrality_score = if direct_relationships > 0 {
            (total_strength / direct_relationships as f64).min(1.0)
        } else {
            0.0
        };
        
        // Clustering coefficient based on relationship strength connectivity
        let clustering_coefficient = if direct_relationships > 1 {
            let pair_count = (direct_relationships * (direct_relationships - 1) / 2) as f64;

            let mut sum_strength = 0.0;
            for i in 0..relationships.len() {
                let s1 = relationships[i].1;
                for j in (i + 1)..relationships.len() {
                    let s2 = relationships[j].1;
                    sum_strength += s1 * s2;
                }
            }

            (sum_strength / pair_count).min(1.0)
        } else {
            0.0
        };
        
        // Bridge potential (simplified)
        let bridge_potential = if direct_relationships > 3 {
            0.8
        } else {
            0.2
        };

        Ok(RelationshipIntelligence {
            direct_relationships,
            indirect_relationships,
            centrality_score,
            clustering_coefficient,
            bridge_potential,
        })
    }

    /// Analyze usage intelligence
    async fn analyze_usage_intelligence(&self, memory_key: &str, memory_entry: &MemoryEntry) -> Result<UsageIntelligence> {
        // Count access events for this memory
        let access_count = self.analysis_history
            .iter()
            .filter(|event| {
                matches!(event, AnalyticsEvent::MemoryAccess { memory_key: key, .. } if key == memory_key)
            })
            .count();

        let access_frequency = (access_count as f64 / 10.0).min(1.0);

        // Filter events related to this memory
        let memory_events: Vec<AnalyticsEvent> = self.analysis_history
            .iter()
            .filter(|event| {
                match event {
                    AnalyticsEvent::MemoryAccess { memory_key: key, .. } => key == memory_key,
                    AnalyticsEvent::MemoryModification { memory_key: key, .. } => key == memory_key,
                    _ => false,
                }
            })
            .cloned()
            .collect();

        // Temporal consistency using sophisticated analysis
        let temporal_consistency = self.calculate_temporal_consistency(memory_entry, &memory_events).await?;

        // User diversity using access pattern analysis
        let user_diversity = self.calculate_user_diversity(&memory_events).await?;

        // Context diversity using content and metadata analysis
        let context_diversity = self.calculate_context_diversity(memory_entry, &memory_events).await?;

        // Collaboration score using interaction pattern analysis
        let collaboration_score = self.calculate_collaboration_score(&memory_events).await?;

        Ok(UsageIntelligence {
            access_frequency,
            temporal_consistency,
            user_diversity,
            context_diversity,
            collaboration_score,
        })
    }

    /// Analyze content intelligence
    async fn analyze_content_intelligence(&self, memory_entry: &MemoryEntry) -> Result<ContentIntelligence> {
        let content = &memory_entry.value;
        
        // Semantic richness (based on content length and vocabulary)
        let words: Vec<&str> = content.split_whitespace().collect();
        let unique_words: HashSet<&str> = words.iter().cloned().collect();
        let semantic_richness = if words.is_empty() {
            0.0
        } else {
            (unique_words.len() as f64 / words.len() as f64 * 2.0).min(1.0)
        };
        
        // Information density
        let information_density = (content.len() as f64 / 500.0).min(1.0);
        
        // Uniqueness using sophisticated content analysis
        let uniqueness = self.calculate_content_uniqueness(memory_entry).await?;
        
        // Relevance (based on importance if available)
        let relevance = memory_entry.metadata.importance;
        
        // Quality (composite score)
        let quality = (semantic_richness + information_density + relevance) / 3.0;

        Ok(ContentIntelligence {
            semantic_richness,
            information_density,
            uniqueness,
            relevance,
            quality,
        })
    }

    /// Recognize patterns in memory data
    pub async fn recognize_patterns(&mut self) -> Result<Vec<PatternRecognition>> {
        let mut new_patterns = Vec::new();

        // Temporal access pattern recognition
        new_patterns.extend(self.recognize_temporal_patterns().await?);
        
        // Content similarity pattern recognition
        new_patterns.extend(self.recognize_content_patterns().await?);
        
        // User behavior pattern recognition
        new_patterns.extend(self.recognize_user_patterns().await?);

        self.patterns.extend(new_patterns.clone());
        Ok(new_patterns)
    }

    /// Recognize temporal access patterns
    async fn recognize_temporal_patterns(&self) -> Result<Vec<PatternRecognition>> {
        let mut patterns = Vec::new();

        // Group access events by hour
        let mut hourly_access: HashMap<u32, Vec<String>> = HashMap::new();
        
        for event in &self.analysis_history {
            if let AnalyticsEvent::MemoryAccess { memory_key, timestamp, .. } = event {
                let hour = timestamp.hour();
                hourly_access.entry(hour).or_insert_with(Vec::new).push(memory_key.clone());
            }
        }

        // Find peak hours
        for (hour, memories) in hourly_access {
            if memories.len() > 5 { // Threshold for pattern recognition
                let pattern = PatternRecognition {
                    pattern_id: Uuid::new_v4().to_string(),
                    pattern_type: PatternType::TemporalAccess,
                    description: format!("High activity at hour {} with {} accesses", hour, memories.len()),
                    confidence: (memories.len() as f64 / 20.0).min(1.0),
                    affected_memories: memories.into_iter().collect::<HashSet<_>>().into_iter().collect(),
                    strength: 0.8,
                    discovered_at: Utc::now(),
                };
                patterns.push(pattern);
            }
        }

        Ok(patterns)
    }

    /// Recognize content similarity patterns
    async fn recognize_content_patterns(&self) -> Result<Vec<PatternRecognition>> {
        let mut groups: HashMap<&str, Vec<String>> = HashMap::new();

        for (key, intel) in &self.intelligence_cache {
            let bucket = if intel.content_intelligence.semantic_richness > 0.7 {
                "high"
            } else if intel.content_intelligence.semantic_richness < 0.3 {
                "low"
            } else {
                "medium"
            };

            groups.entry(bucket).or_insert_with(Vec::new).push(key.clone());
        }

        let mut patterns = Vec::new();
        for (bucket, keys) in groups {
            if keys.len() > 1 {
                let pattern = PatternRecognition {
                    pattern_id: Uuid::new_v4().to_string(),
                    pattern_type: PatternType::ContentSimilarity,
                    description: format!("{} memories share {} semantic richness", keys.len(), bucket),
                    confidence: (keys.len() as f64 / 5.0).min(1.0),
                    affected_memories: keys,
                    strength: 0.6,
                    discovered_at: Utc::now(),
                };
                patterns.push(pattern);
            }
        }

        Ok(patterns)
    }

    /// Recognize user behavior patterns
    async fn recognize_user_patterns(&self) -> Result<Vec<PatternRecognition>> {
        let mut user_access: HashMap<String, usize> = HashMap::new();

        for event in &self.analysis_history {
            if let AnalyticsEvent::MemoryAccess { user_context: Some(user), .. } = event {
                *user_access.entry(user.clone()).or_insert(0) += 1;
            }
        }

        let mut patterns = Vec::new();
        for (user, count) in user_access {
            if count > 5 {
                let pattern = PatternRecognition {
                    pattern_id: Uuid::new_v4().to_string(),
                    pattern_type: PatternType::UserBehavior,
                    description: format!("User {} accessed memories {} times", user, count),
                    confidence: (count as f64 / 20.0).min(1.0),
                    affected_memories: Vec::new(),
                    strength: 0.7,
                    discovered_at: Utc::now(),
                };
                patterns.push(pattern);
            }
        }

        Ok(patterns)
    }

    /// Detect anomalies in memory system
    pub async fn detect_anomalies(&mut self) -> Result<Vec<AnomalyDetection>> {
        let mut anomalies = Vec::new();

        // Detect access pattern anomalies
        anomalies.extend(self.detect_access_anomalies().await?);
        
        // Detect performance anomalies
        anomalies.extend(self.detect_performance_anomalies().await?);
        
        // Detect content anomalies
        anomalies.extend(self.detect_content_anomalies().await?);

        self.anomalies.extend(anomalies.clone());
        Ok(anomalies)
    }

    /// Detect access pattern anomalies
    async fn detect_access_anomalies(&self) -> Result<Vec<AnomalyDetection>> {
        let mut anomalies = Vec::new();

        // Count recent access events
        let recent_threshold = Utc::now() - Duration::hours(1);
        let recent_accesses = self.analysis_history
            .iter()
            .filter(|event| {
                if let AnalyticsEvent::MemoryAccess { timestamp, .. } = event {
                    *timestamp > recent_threshold
                } else {
                    false
                }
            })
            .count();

        // Check against baseline
        let baseline_accesses = self.baseline_metrics.get("hourly_accesses").unwrap_or(&10.0);
        
        if recent_accesses as f64 > baseline_accesses * 3.0 {
            let anomaly = AnomalyDetection {
                anomaly_id: Uuid::new_v4().to_string(),
                anomaly_type: AnomalyType::AccessPattern,
                severity: AnomalySeverity::High,
                description: format!("Unusual spike in access patterns: {} accesses in the last hour", recent_accesses),
                affected_component: "memory_access_system".to_string(),
                anomaly_score: recent_accesses as f64 / baseline_accesses,
                detected_at: Utc::now(),
                recommended_actions: vec![
                    "Monitor system performance".to_string(),
                    "Check for potential security issues".to_string(),
                    "Review access logs".to_string(),
                ],
            };
            anomalies.push(anomaly);
        }

        Ok(anomalies)
    }

    /// Detect performance anomalies
    async fn detect_performance_anomalies(&self) -> Result<Vec<AnomalyDetection>> {
        let mut anomalies = Vec::new();
        let recent_threshold = Utc::now() - Duration::hours(1);

        let mut total_time = 0.0;
        let mut count = 0;

        for event in &self.analysis_history {
            if let AnalyticsEvent::SearchQuery { response_time_ms, timestamp, .. } = event {
                if *timestamp > recent_threshold {
                    total_time += *response_time_ms as f64;
                    count += 1;
                }
            }
        }

        if count > 0 {
            let avg_recent = total_time / count as f64;
            let baseline = *self.baseline_metrics.get("avg_response_time_ms").unwrap_or(&100.0);
            if avg_recent > baseline * 2.0 {
                anomalies.push(AnomalyDetection {
                    anomaly_id: Uuid::new_v4().to_string(),
                    anomaly_type: AnomalyType::Performance,
                    severity: AnomalySeverity::Medium,
                    description: format!("Average response time spiked to {:.2}ms", avg_recent),
                    affected_component: "query_engine".to_string(),
                    anomaly_score: avg_recent / baseline,
                    detected_at: Utc::now(),
                    recommended_actions: vec![
                        "Investigate slow queries".to_string(),
                        "Scale resources".to_string(),
                    ],
                });
            }
        }

        Ok(anomalies)
    }

    /// Detect content anomalies
    async fn detect_content_anomalies(&self) -> Result<Vec<AnomalyDetection>> {
        let mut anomalies = Vec::new();

        for event in &self.analysis_history {
            if let AnalyticsEvent::MemoryModification { memory_key, change_magnitude, timestamp, .. } = event {
                if *change_magnitude > 0.9 {
                    anomalies.push(AnomalyDetection {
                        anomaly_id: Uuid::new_v4().to_string(),
                        anomaly_type: AnomalyType::ContentAnomaly,
                        severity: AnomalySeverity::High,
                        description: format!("Large modification detected for {}", memory_key),
                        affected_component: memory_key.clone(),
                        anomaly_score: *change_magnitude,
                        detected_at: *timestamp,
                        recommended_actions: vec![
                            "Review recent changes".to_string(),
                            "Verify data integrity".to_string(),
                        ],
                    });
                }
            }
        }

        Ok(anomalies)
    }

    /// Update baseline metrics
    pub async fn update_baseline_metrics(&mut self) -> Result<()> {
        // Calculate baseline access rate
        let total_accesses = self.analysis_history
            .iter()
            .filter(|event| matches!(event, AnalyticsEvent::MemoryAccess { .. }))
            .count();

        if !self.analysis_history.is_empty() {
            let time_span_hours = 24.0; // Assume 24 hours of data
            let hourly_access_rate = total_accesses as f64 / time_span_hours;
            self.baseline_metrics.insert("hourly_accesses".to_string(), hourly_access_rate);

            let (mut total_rt, mut rt_count) = (0.0, 0);
            for event in &self.analysis_history {
                if let AnalyticsEvent::SearchQuery { response_time_ms, .. } = event {
                    total_rt += *response_time_ms as f64;
                    rt_count += 1;
                }
            }
            if rt_count > 0 {
                self.baseline_metrics.insert(
                    "avg_response_time_ms".to_string(),
                    total_rt / rt_count as f64,
                );
            }
        }

        Ok(())
    }

    /// Process analytics event for intelligence analysis
    pub async fn process_event(&mut self, event: &AnalyticsEvent) -> Result<()> {
        self.analysis_history.push(event.clone());
        
        // Keep only recent history
        let cutoff_time = Utc::now() - Duration::days(7);
        self.analysis_history.retain(|event| {
            match event {
                AnalyticsEvent::MemoryAccess { timestamp, .. } => *timestamp > cutoff_time,
                AnalyticsEvent::MemoryModification { timestamp, .. } => *timestamp > cutoff_time,
                AnalyticsEvent::SearchQuery { timestamp, .. } => *timestamp > cutoff_time,
                AnalyticsEvent::RelationshipDiscovery { timestamp, .. } => *timestamp > cutoff_time,
            }
        });

        Ok(())
    }

    /// Generate intelligence insights
    pub async fn generate_insights(&mut self) -> Result<Vec<AnalyticsInsight>> {
        let mut insights = Vec::new();

        // Generate insights from patterns
        for pattern in &self.patterns {
            if pattern.confidence > 0.8 {
                let insight = AnalyticsInsight {
                    id: Uuid::new_v4(),
                    insight_type: InsightType::UsagePattern,
                    title: format!("Strong Pattern Detected: {:?}", pattern.pattern_type),
                    description: pattern.description.clone(),
                    confidence: pattern.confidence,
                    evidence: vec![
                        format!("Pattern strength: {:.2}", pattern.strength),
                        format!("Affected memories: {}", pattern.affected_memories.len()),
                    ],
                    generated_at: Utc::now(),
                    priority: InsightPriority::High,
                };
                insights.push(insight);
            }
        }

        // Generate insights from anomalies
        for anomaly in &self.anomalies {
            if anomaly.severity >= AnomalySeverity::High {
                let insight = AnalyticsInsight {
                    id: Uuid::new_v4(),
                    insight_type: InsightType::AnomalyDetection,
                    title: format!("Anomaly Detected: {:?}", anomaly.anomaly_type),
                    description: anomaly.description.clone(),
                    confidence: (anomaly.anomaly_score / 5.0).min(1.0),
                    evidence: anomaly.recommended_actions.clone(),
                    generated_at: Utc::now(),
                    priority: match anomaly.severity {
                        AnomalySeverity::Critical => InsightPriority::Critical,
                        AnomalySeverity::High => InsightPriority::High,
                        _ => InsightPriority::Medium,
                    },
                };
                insights.push(insight);
            }
        }

        Ok(insights)
    }

    /// Get memory intelligence for a specific memory
    pub fn get_memory_intelligence(&self, memory_key: &str) -> Option<&MemoryIntelligence> {
        self.intelligence_cache.get(memory_key)
    }

    /// Get all recognized patterns
    pub fn get_patterns(&self) -> &[PatternRecognition] {
        &self.patterns
    }

    /// Get all detected anomalies
    pub fn get_anomalies(&self) -> &[AnomalyDetection] {
        &self.anomalies
    }

    /// Calculate conceptual complexity using sophisticated NLP-inspired analysis
    fn calculate_conceptual_complexity(&self, content: &str) -> Result<f64> {
        let mut complexity_score = 0.0;
        let mut factor_count = 0;

        // 1. Vocabulary diversity (unique words / total words)
        let words: Vec<&str> = content.split_whitespace().collect();
        let unique_words: std::collections::HashSet<&str> = words.iter().cloned().collect();
        let vocabulary_diversity = if words.is_empty() {
            0.0
        } else {
            unique_words.len() as f64 / words.len() as f64
        };
        complexity_score += vocabulary_diversity;
        factor_count += 1;

        // 2. Average word length (longer words = higher complexity)
        let avg_word_length = if words.is_empty() {
            0.0
        } else {
            words.iter().map(|w| w.len()).sum::<usize>() as f64 / words.len() as f64
        };
        let word_length_complexity = (avg_word_length / 10.0).min(1.0); // Normalize to 0-1
        complexity_score += word_length_complexity;
        factor_count += 1;

        // 3. Sentence structure complexity
        let sentences: Vec<&str> = content.split(&['.', '!', '?'][..]).collect();
        let avg_sentence_length = if sentences.is_empty() {
            0.0
        } else {
            sentences.iter().map(|s| s.split_whitespace().count()).sum::<usize>() as f64 / sentences.len() as f64
        };
        let sentence_complexity = (avg_sentence_length / 20.0).min(1.0); // Normalize to 0-1
        complexity_score += sentence_complexity;
        factor_count += 1;

        // 4. Punctuation density (complex punctuation indicates complex ideas)
        let complex_punctuation = content.chars().filter(|c| matches!(c, ';' | ':' | '(' | ')' | '[' | ']' | '{' | '}')).count();
        let punctuation_complexity = (complex_punctuation as f64 / content.len() as f64 * 100.0).min(1.0);
        complexity_score += punctuation_complexity;
        factor_count += 1;

        // 5. Technical term density (words with numbers, capitals, special chars)
        let technical_terms = words.iter().filter(|w| {
            w.chars().any(|c| c.is_numeric()) ||
            w.chars().any(|c| c.is_uppercase()) ||
            w.contains('_') || w.contains('-')
        }).count();
        let technical_complexity = if words.is_empty() { 0.0 } else { (technical_terms as f64 / words.len() as f64).min(1.0) };
        complexity_score += technical_complexity;
        factor_count += 1;

        // 6. Concept density (based on abstract vs concrete words)
        let abstract_indicators = ["concept", "idea", "theory", "principle", "approach", "methodology", "framework", "paradigm"];
        let abstract_count = words.iter().filter(|w| {
            abstract_indicators.iter().any(|indicator| w.to_lowercase().contains(indicator))
        }).count();
        let concept_density = if words.is_empty() { 0.0 } else { (abstract_count as f64 / words.len() as f64 * 10.0).min(1.0) };
        complexity_score += concept_density;
        factor_count += 1;

        // Calculate final complexity score
        let final_complexity = if factor_count > 0 {
            complexity_score / factor_count as f64
        } else {
            0.0
        };

        tracing::debug!("Conceptual complexity calculated: {} (vocab: {}, word_len: {}, sentence: {}, punct: {}, tech: {}, concept: {})",
            final_complexity, vocabulary_diversity, word_length_complexity, sentence_complexity,
            punctuation_complexity, technical_complexity, concept_density);

        Ok(final_complexity.min(1.0))
    }

    /// Calculate temporal consistency using sophisticated analysis
    async fn calculate_temporal_consistency(&self, memory_entry: &MemoryEntry, events: &[AnalyticsEvent]) -> Result<f64> {
        let memory_events: Vec<&AnalyticsEvent> = events.iter()
            .filter(|e| e.memory_key() == Some(&memory_entry.key))
            .collect();

        if memory_events.len() < 2 {
            return Ok(0.5); // Neutral score for insufficient data
        }

        let mut consistency_factors = Vec::new();

        // 1. Access interval consistency (regular vs irregular access patterns)
        let mut intervals = Vec::new();
        for i in 1..memory_events.len() {
            let interval = memory_events[i].timestamp() - memory_events[i-1].timestamp();
            intervals.push(interval.num_seconds().abs() as f64);
        }

        if !intervals.is_empty() {
            let mean_interval = intervals.iter().sum::<f64>() / intervals.len() as f64;
            let variance = intervals.iter()
                .map(|x| (x - mean_interval).powi(2))
                .sum::<f64>() / intervals.len() as f64;
            let std_dev = variance.sqrt();
            let coefficient_of_variation = if mean_interval > 0.0 { std_dev / mean_interval } else { 1.0 };

            // Lower coefficient of variation = higher consistency
            let interval_consistency = (1.0 - coefficient_of_variation.min(1.0)).max(0.0);
            consistency_factors.push(interval_consistency);
        }

        // 2. Time-of-day consistency
        let hours: Vec<u32> = memory_events.iter()
            .map(|e| e.timestamp().hour())
            .collect();

        if !hours.is_empty() {
            let hour_counts = hours.iter().fold(std::collections::HashMap::new(), |mut acc, &h| {
                *acc.entry(h).or_insert(0) += 1;
                acc
            });

            // Calculate entropy of hour distribution (lower entropy = more consistent)
            let total = hours.len() as f64;
            let entropy = hour_counts.values()
                .map(|&count| {
                    let p = count as f64 / total;
                    if p > 0.0 { -p * p.log2() } else { 0.0 }
                })
                .sum::<f64>();

            // Normalize entropy (max entropy for 24 hours is log2(24) ≈ 4.58)
            let normalized_entropy = entropy / 4.58;
            let time_consistency = (1.0 - normalized_entropy).max(0.0);
            consistency_factors.push(time_consistency);
        }

        // Calculate final temporal consistency
        let final_consistency = if consistency_factors.is_empty() {
            0.5 // Neutral score
        } else {
            consistency_factors.iter().sum::<f64>() / consistency_factors.len() as f64
        };

        tracing::debug!("Temporal consistency calculated for memory '{}': {} (factors: {})",
            memory_entry.key, final_consistency, consistency_factors.len());

        Ok(final_consistency.min(1.0))
    }

    /// Calculate user diversity using access pattern analysis
    async fn calculate_user_diversity(&self, events: &[AnalyticsEvent]) -> Result<f64> {
        let mut user_contexts = std::collections::HashSet::new();
        let mut total_events = 0;

        for event in events {
            total_events += 1;
            if let AnalyticsEvent::MemoryAccess { user_context, .. } = event {
                if let Some(user) = user_context {
                    user_contexts.insert(user.clone());
                }
            }
        }

        let diversity_score = if total_events > 0 {
            user_contexts.len() as f64 / total_events as f64
        } else {
            0.0
        };

        tracing::debug!("User diversity calculated: {} (unique users: {}, total events: {})",
            diversity_score, user_contexts.len(), total_events);

        Ok(diversity_score.min(1.0))
    }

    /// Calculate context diversity using content and metadata analysis
    async fn calculate_context_diversity(&self, memory_entry: &MemoryEntry, events: &[AnalyticsEvent]) -> Result<f64> {
        let mut diversity_factors = Vec::new();

        // 1. Tag diversity (if memory has tags)
        let tag_count = memory_entry.metadata.tags.len();
        let tag_diversity = (tag_count as f64 / 10.0).min(1.0); // Normalize to max 10 tags
        diversity_factors.push(tag_diversity);

        // 2. Access type diversity
        let access_types: std::collections::HashSet<_> = events.iter()
            .filter_map(|e| {
                if let AnalyticsEvent::MemoryAccess { access_type, .. } = e {
                    Some(access_type)
                } else {
                    None
                }
            })
            .collect();
        let access_type_diversity = (access_types.len() as f64 / 3.0).min(1.0); // Normalize to 3 access types
        diversity_factors.push(access_type_diversity);

        // 3. Temporal diversity (spread across different times)
        let hours: std::collections::HashSet<_> = events.iter()
            .map(|e| e.timestamp().hour())
            .collect();
        let temporal_diversity = (hours.len() as f64 / 24.0).min(1.0); // Normalize to 24 hours
        diversity_factors.push(temporal_diversity);

        // 4. Content type diversity (based on content characteristics)
        let content = &memory_entry.value;
        let has_numbers = content.chars().any(|c| c.is_numeric());
        let has_special_chars = content.chars().any(|c| !c.is_alphanumeric() && !c.is_whitespace());
        let has_uppercase = content.chars().any(|c| c.is_uppercase());
        let content_type_diversity = [has_numbers, has_special_chars, has_uppercase].iter().filter(|&&x| x).count() as f64 / 3.0;
        diversity_factors.push(content_type_diversity);

        let final_diversity = if diversity_factors.is_empty() {
            0.0
        } else {
            diversity_factors.iter().sum::<f64>() / diversity_factors.len() as f64
        };

        tracing::debug!("Context diversity calculated for memory '{}': {} (factors: {})",
            memory_entry.key, final_diversity, diversity_factors.len());

        Ok(final_diversity.min(1.0))
    }

    /// Calculate collaboration score using interaction pattern analysis
    async fn calculate_collaboration_score(&self, events: &[AnalyticsEvent]) -> Result<f64> {
        let mut collaboration_indicators = Vec::new();

        // 1. Multiple user access (indicates sharing/collaboration)
        let unique_users: std::collections::HashSet<_> = events.iter()
            .filter_map(|e| {
                if let AnalyticsEvent::MemoryAccess { user_context, .. } = e {
                    user_context.as_ref()
                } else {
                    None
                }
            })
            .collect();

        let multi_user_score = if unique_users.len() > 1 {
            (unique_users.len() as f64 / 5.0).min(1.0) // Normalize to max 5 users
        } else {
            0.0
        };
        collaboration_indicators.push(multi_user_score);

        // 2. Modification frequency (indicates active collaboration)
        let modification_count = events.iter()
            .filter(|e| matches!(e, AnalyticsEvent::MemoryModification { .. }))
            .count();
        let modification_score = (modification_count as f64 / 10.0).min(1.0); // Normalize to max 10 modifications
        collaboration_indicators.push(modification_score);

        // 3. Relationship discovery (indicates connections to other memories)
        let relationship_count = events.iter()
            .filter(|e| matches!(e, AnalyticsEvent::RelationshipDiscovery { .. }))
            .count();
        let relationship_score = (relationship_count as f64 / 5.0).min(1.0); // Normalize to max 5 relationships
        collaboration_indicators.push(relationship_score);

        // 4. Access frequency (high access might indicate collaboration)
        let access_count = events.iter()
            .filter(|e| matches!(e, AnalyticsEvent::MemoryAccess { .. }))
            .count();
        let access_frequency_score = if access_count > 10 {
            (access_count as f64 / 50.0).min(1.0) // Normalize to max 50 accesses
        } else {
            0.0
        };
        collaboration_indicators.push(access_frequency_score);

        let final_collaboration_score = if collaboration_indicators.is_empty() {
            0.0
        } else {
            collaboration_indicators.iter().sum::<f64>() / collaboration_indicators.len() as f64
        };

        tracing::debug!("Collaboration score calculated: {} (users: {}, modifications: {}, relationships: {}, accesses: {})",
            final_collaboration_score, unique_users.len(), modification_count, relationship_count, access_count);

        Ok(final_collaboration_score.min(1.0))
    }

    /// Calculate content uniqueness using sophisticated content analysis
    async fn calculate_content_uniqueness(&self, memory_entry: &MemoryEntry) -> Result<f64> {
        let content = &memory_entry.value;
        let mut uniqueness_factors = Vec::new();

        // 1. Content length uniqueness (very short or very long content is more unique)
        let length = content.len();
        let length_uniqueness = if length < 50 || length > 2000 {
            0.8 // Short or very long content is more unique
        } else if length < 100 || length > 1000 {
            0.6 // Moderately short/long content
        } else {
            0.3 // Average length content is less unique
        };
        uniqueness_factors.push(length_uniqueness);

        // 2. Vocabulary uniqueness (rare words indicate uniqueness)
        let words: Vec<&str> = content.split_whitespace().collect();
        let common_words = ["the", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by", "is", "are", "was", "were", "be", "been", "have", "has", "had", "do", "does", "did", "will", "would", "could", "should", "may", "might", "can", "this", "that", "these", "those"];
        let uncommon_word_count = words.iter()
            .filter(|word| !common_words.contains(&word.to_lowercase().as_str()))
            .count();
        let vocabulary_uniqueness = if words.is_empty() {
            0.0
        } else {
            (uncommon_word_count as f64 / words.len() as f64).min(1.0)
        };
        uniqueness_factors.push(vocabulary_uniqueness);

        // 3. Special character uniqueness
        let special_chars = content.chars().filter(|c| !c.is_alphanumeric() && !c.is_whitespace()).count();
        let special_char_uniqueness = (special_chars as f64 / content.len() as f64 * 10.0).min(1.0);
        uniqueness_factors.push(special_char_uniqueness);

        // 4. Numeric content uniqueness
        let numeric_chars = content.chars().filter(|c| c.is_numeric()).count();
        let numeric_uniqueness = (numeric_chars as f64 / content.len() as f64 * 5.0).min(1.0);
        uniqueness_factors.push(numeric_uniqueness);

        // 5. Pattern uniqueness (repetitive patterns are less unique)
        let mut pattern_score = 1.0;
        let words_set: std::collections::HashSet<&str> = words.iter().cloned().collect();
        if words.len() > 0 && words_set.len() < words.len() / 2 {
            pattern_score = 0.3; // High repetition = low uniqueness
        }
        uniqueness_factors.push(pattern_score);

        let final_uniqueness = if uniqueness_factors.is_empty() {
            0.5 // Neutral score
        } else {
            uniqueness_factors.iter().sum::<f64>() / uniqueness_factors.len() as f64
        };

        tracing::debug!("Content uniqueness calculated for memory '{}': {} (length: {}, vocab: {}, special: {}, numeric: {}, pattern: {})",
            memory_entry.key, final_uniqueness, length_uniqueness, vocabulary_uniqueness,
            special_char_uniqueness, numeric_uniqueness, pattern_score);

        Ok(final_uniqueness.min(1.0))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::types::MemoryEntry;

    #[tokio::test]
    async fn test_intelligence_engine_creation() {
        let config = AnalyticsConfig::default();
        let engine = MemoryIntelligenceEngine::new(&config);
        assert!(engine.is_ok());
    }

    #[tokio::test]
    async fn test_memory_intelligence_analysis() {
        let config = AnalyticsConfig::default();
        let mut engine = MemoryIntelligenceEngine::new(&config).unwrap();

        let memory_entry = MemoryEntry::new("test_key".to_string(), "This is a test memory with some complex content for analysis".to_string(), crate::memory::types::MemoryType::ShortTerm);
        let relationships = vec![("related_key".to_string(), 0.8)];

        let intelligence = engine.analyze_memory_intelligence("test_key", &memory_entry, &relationships).await.unwrap();
        
        assert!(intelligence.intelligence_score >= 0.0);
        assert!(intelligence.intelligence_score <= 1.0);
        assert_eq!(intelligence.memory_key, "test_key");
    }

    #[tokio::test]
    async fn test_pattern_recognition() {
        let config = AnalyticsConfig::default();
        let mut engine = MemoryIntelligenceEngine::new(&config).unwrap();

        // Add some events to analyze
        for i in 0..10 {
            let event = AnalyticsEvent::MemoryAccess {
                memory_key: format!("key_{}", i % 3),
                access_type: crate::analytics::AccessType::Read,
                timestamp: Utc::now(),
                user_context: Some("test_user".to_string()),
            };
            engine.process_event(&event).await.unwrap();
        }

        let patterns = engine.recognize_patterns().await.unwrap();
        assert!(patterns.len() > 0);
    }

    #[tokio::test]
    async fn test_anomaly_detection() {
        let config = AnalyticsConfig::default();
        let mut engine = MemoryIntelligenceEngine::new(&config).unwrap();

        // Set baseline
        engine.baseline_metrics.insert("hourly_accesses".to_string(), 5.0);

        // Add many events to trigger anomaly
        for i in 0..20 {
            let event = AnalyticsEvent::MemoryAccess {
                memory_key: format!("key_{}", i),
                access_type: crate::analytics::AccessType::Read,
                timestamp: Utc::now(),
                user_context: Some("test_user".to_string()),
            };
            engine.process_event(&event).await.unwrap();
        }

        let anomalies = engine.detect_anomalies().await.unwrap();
        assert!(!anomalies.is_empty());
        assert!(anomalies.iter().any(|a| a.anomaly_type == AnomalyType::AccessPattern));
    }

    #[tokio::test]
    async fn test_insight_generation() {
        let config = AnalyticsConfig::default();
        let mut engine = MemoryIntelligenceEngine::new(&config).unwrap();

        // Add a high-confidence pattern
        let pattern = PatternRecognition {
            pattern_id: Uuid::new_v4().to_string(),
            pattern_type: PatternType::TemporalAccess,
            description: "Test pattern".to_string(),
            confidence: 0.9,
            affected_memories: vec!["test_key".to_string()],
            strength: 0.8,
            discovered_at: Utc::now(),
        };
        engine.patterns.push(pattern);

        // Also add an anomaly for insight generation
        engine.anomalies.push(AnomalyDetection {
            anomaly_id: Uuid::new_v4().to_string(),
            anomaly_type: AnomalyType::AccessPattern,
            severity: AnomalySeverity::High,
            description: "test anomaly".to_string(),
            affected_component: "test".to_string(),
            anomaly_score: 2.0,
            detected_at: Utc::now(),
            recommended_actions: vec!["noop".to_string()],
        });

        let insights = engine.generate_insights().await.unwrap();
        assert!(insights.len() > 0);
    }
}
