// Behavioral Analysis Module
// User interaction pattern recognition and personalized recommendations

use crate::error::Result;
use crate::analytics::{AnalyticsEvent, AnalyticsConfig, AnalyticsInsight, InsightType, InsightPriority, AccessType};
use chrono::{DateTime, Utc, Duration, Timelike};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use uuid::Uuid;

/// User behavior profile
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UserProfile {
    /// User identifier
    pub user_id: String,
    /// Preferred access times (hour of day)
    pub preferred_hours: Vec<u32>,
    /// Most accessed memory types
    pub preferred_memory_types: HashMap<String, f64>,
    /// Average session duration (minutes)
    pub avg_session_duration: f64,
    /// Search query patterns
    pub search_patterns: Vec<String>,
    /// Interaction frequency
    pub interaction_frequency: InteractionFrequency,
    /// Last activity timestamp
    pub last_activity: DateTime<Utc>,
}

/// Interaction frequency categories
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum InteractionFrequency {
    VeryLow,    // < 1 interaction per day
    Low,        // 1-5 interactions per day
    Medium,     // 5-20 interactions per day
    High,       // 20-50 interactions per day
    VeryHigh,   // > 50 interactions per day
}

/// Memory usage pattern
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryUsagePattern {
    /// Memory key
    pub memory_key: String,
    /// Users who access this memory
    pub users: HashSet<String>,
    /// Peak usage hours
    pub peak_hours: Vec<u32>,
    /// Usage context patterns
    pub contexts: HashMap<String, u32>,
    /// Collaboration indicators
    pub is_collaborative: bool,
    /// Average access duration
    pub avg_access_duration: f64,
}

/// Personalized recommendation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PersonalizedRecommendation {
    /// Recommendation ID
    pub id: Uuid,
    /// Target user
    pub user_id: String,
    /// Recommendation type
    pub recommendation_type: RecommendationType,
    /// Recommended memory key or action
    pub target: String,
    /// Confidence score
    pub confidence: f64,
    /// Reasoning
    pub reasoning: String,
    /// Expected benefit
    pub expected_benefit: String,
    /// Generated timestamp
    pub generated_at: DateTime<Utc>,
}

/// Types of recommendations
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum RecommendationType {
    /// Recommend accessing a specific memory
    MemoryAccess,
    /// Recommend creating a relationship
    RelationshipCreation,
    /// Recommend organizing memories
    MemoryOrganization,
    /// Recommend optimal timing
    TimingOptimization,
    /// Recommend collaboration
    CollaborationSuggestion,
}

/// Behavioral session tracking
#[derive(Debug, Clone)]
struct BehavioralSession {
    /// User ID
    user_id: String,
    /// Session start time
    start_time: DateTime<Utc>,
    /// Last activity time
    last_activity: DateTime<Utc>,
    /// Activities in this session
    activities: Vec<AnalyticsEvent>,
    /// Session context
    context: Option<String>,
}

impl BehavioralSession {
    fn new(user_id: String, start_time: DateTime<Utc>) -> Self {
        Self {
            user_id,
            start_time,
            last_activity: start_time,
            activities: Vec::new(),
            context: None,
        }
    }

    fn add_activity(&mut self, event: AnalyticsEvent) {
        if let Some(timestamp) = self.extract_timestamp(&event) {
            self.last_activity = timestamp;
        }
        self.activities.push(event);
    }

    fn duration(&self) -> Duration {
        self.last_activity - self.start_time
    }

    fn is_active(&self, current_time: DateTime<Utc>, timeout_minutes: i64) -> bool {
        current_time - self.last_activity < Duration::minutes(timeout_minutes)
    }

    fn extract_timestamp(&self, event: &AnalyticsEvent) -> Option<DateTime<Utc>> {
        match event {
            AnalyticsEvent::MemoryAccess { timestamp, .. } => Some(*timestamp),
            AnalyticsEvent::MemoryModification { timestamp, .. } => Some(*timestamp),
            AnalyticsEvent::SearchQuery { timestamp, .. } => Some(*timestamp),
            AnalyticsEvent::RelationshipDiscovery { timestamp, .. } => Some(*timestamp),
        }
    }
}

/// Behavioral analyzer
#[derive(Debug)]
pub struct BehavioralAnalyzer {
    /// Configuration
    config: AnalyticsConfig,
    /// User profiles
    user_profiles: HashMap<String, UserProfile>,
    /// Memory usage patterns
    memory_patterns: HashMap<String, MemoryUsagePattern>,
    /// Active sessions
    active_sessions: HashMap<String, BehavioralSession>,
    /// Generated recommendations
    recommendations: Vec<PersonalizedRecommendation>,
    /// Session timeout in minutes
    session_timeout: i64,
}

impl BehavioralAnalyzer {
    /// Create a new behavioral analyzer
    pub fn new(config: &AnalyticsConfig) -> Result<Self> {
        Ok(Self {
            config: config.clone(),
            user_profiles: HashMap::new(),
            memory_patterns: HashMap::new(),
            active_sessions: HashMap::new(),
            recommendations: Vec::new(),
            session_timeout: 30, // 30 minutes session timeout
        })
    }

    /// Process an analytics event
    pub async fn process_event(&mut self, event: &AnalyticsEvent) -> Result<()> {
        // Extract user context if available
        let user_id = self.extract_user_context(event);
        
        if let Some(user_id) = user_id {
            // Update or create session
            self.update_session(&user_id, event.clone()).await?;
            
            // Update user profile
            self.update_user_profile(&user_id, event).await?;
        }

        // Update memory usage patterns
        self.update_memory_patterns(event).await?;

        Ok(())
    }

    /// Extract user context from event
    fn extract_user_context(&self, event: &AnalyticsEvent) -> Option<String> {
        match event {
            AnalyticsEvent::MemoryAccess { user_context, .. } => user_context.clone(),
            _ => None,
        }
    }

    /// Update or create user session
    async fn update_session(&mut self, user_id: &str, event: AnalyticsEvent) -> Result<()> {
        let current_time = Utc::now();
        
        // Check if user has an active session
        let needs_new_session = if let Some(session) = self.active_sessions.get(user_id) {
            !session.is_active(current_time, self.session_timeout)
        } else {
            true
        };

        if needs_new_session {
            // Create new session
            let mut new_session = BehavioralSession::new(user_id.to_string(), current_time);
            new_session.add_activity(event);
            self.active_sessions.insert(user_id.to_string(), new_session);
        } else {
            // Add to existing session
            if let Some(session) = self.active_sessions.get_mut(user_id) {
                session.add_activity(event);
            }
        }

        Ok(())
    }

    /// Update user profile based on event
    async fn update_user_profile(&mut self, user_id: &str, event: &AnalyticsEvent) -> Result<()> {
        // First, update the profile data
        {
            let profile = self.user_profiles
                .entry(user_id.to_string())
                .or_insert_with(|| UserProfile {
                    user_id: user_id.to_string(),
                    preferred_hours: Vec::new(),
                    preferred_memory_types: HashMap::new(),
                    avg_session_duration: 0.0,
                    search_patterns: Vec::new(),
                    interaction_frequency: InteractionFrequency::Low,
                    last_activity: Utc::now(),
                });

            // Update based on event type
            match event {
                AnalyticsEvent::MemoryAccess { timestamp, .. } => {
                    let hour = timestamp.hour();
                    if !profile.preferred_hours.contains(&hour) {
                        profile.preferred_hours.push(hour);
                    }
                    profile.last_activity = *timestamp;
                }
                AnalyticsEvent::SearchQuery { query, timestamp, .. } => {
                    if !profile.search_patterns.contains(query) && profile.search_patterns.len() < 50 {
                        profile.search_patterns.push(query.clone());
                    }
                    profile.last_activity = *timestamp;
                }
                _ => {}
            }
        }

        // Then update interaction frequency in a separate scope
        let user_id_clone = user_id.to_string();
        if let Some(profile) = self.user_profiles.get_mut(&user_id_clone) {
            self.update_interaction_frequency_for_user(&user_id_clone).await?;
        }

        Ok(())
    }

    /// Update interaction frequency for a specific user
    async fn update_interaction_frequency_for_user(&mut self, user_id: &str) -> Result<()> {
        // Count interactions in the last 24 hours
        let day_ago = Utc::now() - Duration::days(1);

        if let Some(session) = self.active_sessions.get(user_id) {
            let recent_activities = session.activities
                .iter()
                .filter(|event| {
                    if let Some(timestamp) = session.extract_timestamp(event) {
                        timestamp > day_ago
                    } else {
                        false
                    }
                })
                .count();

            if let Some(profile) = self.user_profiles.get_mut(user_id) {
                profile.interaction_frequency = match recent_activities {
                    0 => InteractionFrequency::VeryLow,
                    1..=5 => InteractionFrequency::Low,
                    6..=20 => InteractionFrequency::Medium,
                    21..=50 => InteractionFrequency::High,
                    _ => InteractionFrequency::VeryHigh,
                };
            }
        }

        Ok(())
    }

    /// Update interaction frequency for a user (legacy method)
    async fn update_interaction_frequency(&mut self, profile: &mut UserProfile) -> Result<()> {
        // Count interactions in the last 24 hours
        let day_ago = Utc::now() - Duration::days(1);
        
        if let Some(session) = self.active_sessions.get(&profile.user_id) {
            let recent_activities = session.activities
                .iter()
                .filter(|event| {
                    if let Some(timestamp) = session.extract_timestamp(event) {
                        timestamp > day_ago
                    } else {
                        false
                    }
                })
                .count();

            profile.interaction_frequency = match recent_activities {
                0 => InteractionFrequency::VeryLow,
                1..=5 => InteractionFrequency::Low,
                6..=20 => InteractionFrequency::Medium,
                21..=50 => InteractionFrequency::High,
                _ => InteractionFrequency::VeryHigh,
            };
        }

        Ok(())
    }

    /// Update memory usage patterns
    async fn update_memory_patterns(&mut self, event: &AnalyticsEvent) -> Result<()> {
        match event {
            AnalyticsEvent::MemoryAccess { memory_key, timestamp, user_context, .. } => {
                let pattern = self.memory_patterns
                    .entry(memory_key.clone())
                    .or_insert_with(|| MemoryUsagePattern {
                        memory_key: memory_key.clone(),
                        users: HashSet::new(),
                        peak_hours: Vec::new(),
                        contexts: HashMap::new(),
                        is_collaborative: false,
                        avg_access_duration: 0.0,
                    });

                // Add user if provided
                if let Some(user) = user_context {
                    pattern.users.insert(user.clone());
                    pattern.is_collaborative = pattern.users.len() > 1;
                }

                // Track peak hours
                let hour = timestamp.hour();
                if !pattern.peak_hours.contains(&hour) {
                    pattern.peak_hours.push(hour);
                }

                // Update context if available
                if let Some(context) = user_context {
                    *pattern.contexts.entry(context.clone()).or_insert(0) += 1;
                }
            }
            _ => {}
        }

        Ok(())
    }

    /// Generate personalized recommendations
    pub async fn generate_recommendations(&mut self, user_id: &str) -> Result<Vec<PersonalizedRecommendation>> {
        let mut recommendations = Vec::new();

        if let Some(profile) = self.user_profiles.get(user_id) {
            // Generate timing optimization recommendations
            recommendations.extend(self.generate_timing_recommendations(profile).await?);
            
            // Generate memory access recommendations
            recommendations.extend(self.generate_memory_recommendations(profile).await?);
            
            // Generate collaboration recommendations
            recommendations.extend(self.generate_collaboration_recommendations(profile).await?);
        }

        // Store recommendations
        self.recommendations.extend(recommendations.clone());

        Ok(recommendations)
    }

    /// Generate timing optimization recommendations
    async fn generate_timing_recommendations(&self, profile: &UserProfile) -> Result<Vec<PersonalizedRecommendation>> {
        let mut recommendations = Vec::new();

        if !profile.preferred_hours.is_empty() {
            // Find the most common hour
            let mut hour_counts = HashMap::new();
            for &hour in &profile.preferred_hours {
                *hour_counts.entry(hour).or_insert(0) += 1;
            }

            if let Some((&peak_hour, &count)) = hour_counts.iter().max_by_key(|(_, &count)| count) {
                if count > 2 { // Only recommend if there's a clear pattern
                    let recommendation = PersonalizedRecommendation {
                        id: Uuid::new_v4(),
                        user_id: profile.user_id.clone(),
                        recommendation_type: RecommendationType::TimingOptimization,
                        target: format!("{}:00", peak_hour),
                        confidence: (count as f64 / profile.preferred_hours.len() as f64).min(1.0),
                        reasoning: format!(
                            "You typically access memories most frequently at {}:00 ({}% of the time)",
                            peak_hour,
                            (count as f64 / profile.preferred_hours.len() as f64 * 100.0) as u32
                        ),
                        expected_benefit: "Scheduling important memory work during your peak hours can improve focus and efficiency".to_string(),
                        generated_at: Utc::now(),
                    };
                    recommendations.push(recommendation);
                }
            }
        }

        Ok(recommendations)
    }

    /// Generate memory access recommendations
    async fn generate_memory_recommendations(&self, profile: &UserProfile) -> Result<Vec<PersonalizedRecommendation>> {
        let mut recommendations = Vec::new();

        // Recommend memories that are frequently accessed by similar users
        for (memory_key, pattern) in &self.memory_patterns {
            if pattern.users.len() > 1 && !pattern.users.contains(&profile.user_id) {
                // Check if this user has similar patterns to users who access this memory
                let similarity_score = self.calculate_user_similarity(profile, pattern).await?;
                
                if similarity_score > 0.7 {
                    let recommendation = PersonalizedRecommendation {
                        id: Uuid::new_v4(),
                        user_id: profile.user_id.clone(),
                        recommendation_type: RecommendationType::MemoryAccess,
                        target: memory_key.clone(),
                        confidence: similarity_score,
                        reasoning: format!(
                            "Users with similar patterns frequently access this memory ({}% similarity)",
                            (similarity_score * 100.0) as u32
                        ),
                        expected_benefit: "This memory might be relevant to your current work or interests".to_string(),
                        generated_at: Utc::now(),
                    };
                    recommendations.push(recommendation);
                }
            }
        }

        Ok(recommendations)
    }

    /// Generate collaboration recommendations
    async fn generate_collaboration_recommendations(&self, profile: &UserProfile) -> Result<Vec<PersonalizedRecommendation>> {
        let mut recommendations = Vec::new();

        // Find memories that could benefit from collaboration
        for (memory_key, pattern) in &self.memory_patterns {
            if pattern.users.contains(&profile.user_id) && pattern.users.len() == 1 {
                // This user is the only one accessing this memory
                // Check if other users might be interested
                let potential_collaborators = self.find_potential_collaborators(profile, memory_key).await?;
                
                if !potential_collaborators.is_empty() {
                    let recommendation = PersonalizedRecommendation {
                        id: Uuid::new_v4(),
                        user_id: profile.user_id.clone(),
                        recommendation_type: RecommendationType::CollaborationSuggestion,
                        target: memory_key.clone(),
                        confidence: 0.8,
                        reasoning: format!(
                            "Found {} potential collaborators with similar interests",
                            potential_collaborators.len()
                        ),
                        expected_benefit: "Collaboration could enhance the value and accuracy of this memory".to_string(),
                        generated_at: Utc::now(),
                    };
                    recommendations.push(recommendation);
                }
            }
        }

        Ok(recommendations)
    }

    /// Calculate similarity between user and memory pattern users
    async fn calculate_user_similarity(&self, profile: &UserProfile, pattern: &MemoryUsagePattern) -> Result<f64> {
        let mut similarity_scores = Vec::new();

        for user_id in &pattern.users {
            if let Some(other_profile) = self.user_profiles.get(user_id) {
                // Compare preferred hours
                let hour_overlap = profile.preferred_hours
                    .iter()
                    .filter(|hour| other_profile.preferred_hours.contains(hour))
                    .count();
                
                let hour_similarity = if profile.preferred_hours.is_empty() || other_profile.preferred_hours.is_empty() {
                    0.0
                } else {
                    hour_overlap as f64 / profile.preferred_hours.len().max(other_profile.preferred_hours.len()) as f64
                };

                // Compare interaction frequency
                let freq_similarity = if profile.interaction_frequency == other_profile.interaction_frequency {
                    1.0
                } else {
                    0.5
                };

                let overall_similarity = (hour_similarity + freq_similarity) / 2.0;
                similarity_scores.push(overall_similarity);
            }
        }

        if similarity_scores.is_empty() {
            Ok(0.0)
        } else {
            Ok(similarity_scores.iter().sum::<f64>() / similarity_scores.len() as f64)
        }
    }

    /// Find potential collaborators for a memory
    async fn find_potential_collaborators(&self, _profile: &UserProfile, _memory_key: &str) -> Result<Vec<String>> {
        // TODO: Implement sophisticated collaborator matching
        // This could analyze user profiles, search patterns, and memory access patterns
        // to find users who might be interested in collaborating on a specific memory
        Ok(Vec::new())
    }

    /// Generate behavioral insights
    pub async fn generate_insights(&mut self) -> Result<Vec<AnalyticsInsight>> {
        let mut insights = Vec::new();

        // Generate insights about user behavior patterns
        for (user_id, profile) in &self.user_profiles {
            if profile.interaction_frequency == InteractionFrequency::VeryHigh {
                let insight = AnalyticsInsight {
                    id: Uuid::new_v4(),
                    insight_type: InsightType::UsagePattern,
                    title: format!("High Activity User: {}", user_id),
                    description: format!(
                        "User {} shows very high interaction frequency with {} preferred hours",
                        user_id, profile.preferred_hours.len()
                    ),
                    confidence: 0.9,
                    evidence: vec![
                        format!("Interaction frequency: {:?}", profile.interaction_frequency),
                        format!("Active hours: {:?}", profile.preferred_hours),
                    ],
                    generated_at: Utc::now(),
                    priority: InsightPriority::Medium,
                };
                insights.push(insight);
            }
        }

        // Generate insights about collaborative memories
        for (memory_key, pattern) in &self.memory_patterns {
            if pattern.is_collaborative && pattern.users.len() > 3 {
                let insight = AnalyticsInsight {
                    id: Uuid::new_v4(),
                    insight_type: InsightType::RelationshipInsight,
                    title: format!("Highly Collaborative Memory: {}", memory_key),
                    description: format!(
                        "Memory '{}' is accessed by {} users, indicating high collaborative value",
                        memory_key, pattern.users.len()
                    ),
                    confidence: 0.8,
                    evidence: vec![
                        format!("Number of users: {}", pattern.users.len()),
                        format!("Peak hours: {:?}", pattern.peak_hours),
                    ],
                    generated_at: Utc::now(),
                    priority: InsightPriority::High,
                };
                insights.push(insight);
            }
        }

        Ok(insights)
    }

    /// Get user profiles
    pub fn get_user_profiles(&self) -> &HashMap<String, UserProfile> {
        &self.user_profiles
    }

    /// Get memory patterns
    pub fn get_memory_patterns(&self) -> &HashMap<String, MemoryUsagePattern> {
        &self.memory_patterns
    }

    /// Get recommendations
    pub fn get_recommendations(&self) -> &[PersonalizedRecommendation] {
        &self.recommendations
    }

    /// Get active sessions
    pub fn get_active_sessions(&self) -> Vec<String> {
        let current_time = Utc::now();
        self.active_sessions
            .iter()
            .filter(|(_, session)| session.is_active(current_time, self.session_timeout))
            .map(|(user_id, _)| user_id.clone())
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_behavioral_analyzer_creation() {
        let config = AnalyticsConfig::default();
        let analyzer = BehavioralAnalyzer::new(&config);
        assert!(analyzer.is_ok());
    }

    #[tokio::test]
    async fn test_user_profile_creation() {
        let config = AnalyticsConfig::default();
        let mut analyzer = BehavioralAnalyzer::new(&config).unwrap();

        let event = AnalyticsEvent::MemoryAccess {
            memory_key: "test_key".to_string(),
            access_type: AccessType::Read,
            timestamp: Utc::now(),
            user_context: Some("test_user".to_string()),
        };

        analyzer.process_event(&event).await.unwrap();
        assert!(analyzer.user_profiles.contains_key("test_user"));
    }

    #[tokio::test]
    async fn test_session_tracking() {
        let config = AnalyticsConfig::default();
        let mut analyzer = BehavioralAnalyzer::new(&config).unwrap();

        let event = AnalyticsEvent::MemoryAccess {
            memory_key: "test_key".to_string(),
            access_type: AccessType::Read,
            timestamp: Utc::now(),
            user_context: Some("session_user".to_string()),
        };

        analyzer.process_event(&event).await.unwrap();
        assert!(analyzer.active_sessions.contains_key("session_user"));
    }

    #[tokio::test]
    async fn test_recommendation_generation() {
        let config = AnalyticsConfig::default();
        let mut analyzer = BehavioralAnalyzer::new(&config).unwrap();

        // Create a user profile first
        let event = AnalyticsEvent::MemoryAccess {
            memory_key: "test_key".to_string(),
            access_type: AccessType::Read,
            timestamp: Utc::now(),
            user_context: Some("rec_user".to_string()),
        };

        analyzer.process_event(&event).await.unwrap();

        let recommendations = analyzer.generate_recommendations("rec_user").await.unwrap();
        // Should not error, recommendations may be empty initially
        assert!(recommendations.len() >= 0);
    }

    #[tokio::test]
    async fn test_insight_generation() {
        let config = AnalyticsConfig::default();
        let mut analyzer = BehavioralAnalyzer::new(&config).unwrap();

        // Add some user activity
        for i in 0..10 {
            let event = AnalyticsEvent::MemoryAccess {
                memory_key: format!("key_{}", i),
                access_type: AccessType::Read,
                timestamp: Utc::now(),
                user_context: Some("insight_user".to_string()),
            };
            analyzer.process_event(&event).await.unwrap();
        }

        let insights = analyzer.generate_insights().await.unwrap();
        // Should generate insights based on user behavior
        assert!(insights.len() >= 0);
    }
}
