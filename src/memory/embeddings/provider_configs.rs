//! Configuration for various embedding providers
//! 
//! This module provides configurations for the best performing embedding models
//! as of late 2024, based on MTEB leaderboard and performance benchmarks.

use serde::{Deserialize, Serialize};

/// Voyage AI embedding configuration
/// Currently top performer on MTEB leaderboard
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VoyageAIConfig {
    /// Voyage AI API key
    pub api_key: String,
    /// Model to use (voyage-large-2-instruct, voyage-3-large)
    pub model: String,
    /// Embedding dimensions
    pub embedding_dim: usize,
    /// API base URL
    pub base_url: String,
    /// Request timeout in seconds
    pub timeout_secs: u64,
    /// Enable caching
    pub enable_cache: bool,
    /// Cache size
    pub cache_size: usize,
}

impl Default for VoyageAIConfig {
    fn default() -> Self {
        Self {
            api_key: std::env::var("VOYAGE_API_KEY").unwrap_or_else(|_| "pa-eIPOdZDBUV_ihpFijOw9_rGda2lShuXxR0DgRhA8URJ".to_string()),
            model: "voyage-code-2".to_string(), // Optimized for code embeddings
            embedding_dim: 1536, // voyage-code-2 dimensions
            base_url: "https://api.voyageai.com/v1/embeddings".to_string(),
            timeout_secs: 30,
            enable_cache: true,
            cache_size: 10000,
        }
    }
}

/// Cohere embedding configuration
/// Excellent for retrieval tasks
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CohereConfig {
    /// Cohere API key
    pub api_key: String,
    /// Model to use (embed-english-v3.0, embed-english-light-v3.0)
    pub model: String,
    /// Embedding dimensions
    pub embedding_dim: usize,
    /// API base URL
    pub base_url: String,
    /// Request timeout in seconds
    pub timeout_secs: u64,
    /// Enable caching
    pub enable_cache: bool,
    /// Cache size
    pub cache_size: usize,
    /// Input type for embeddings
    pub input_type: String,
}

impl Default for CohereConfig {
    fn default() -> Self {
        Self {
            api_key: std::env::var("COHERE_API_KEY").unwrap_or_default(),
            model: "embed-english-v3.0".to_string(), // Best Cohere model
            embedding_dim: 1024,
            base_url: "https://api.cohere.ai/v1/embed".to_string(),
            timeout_secs: 30,
            enable_cache: true,
            cache_size: 10000,
            input_type: "search_document".to_string(), // For document storage
        }
    }
}

/// Provider performance comparison based on MTEB benchmarks
#[derive(Debug, Clone)]
pub struct ProviderPerformance {
    pub provider: String,
    pub model: String,
    pub mteb_score: f64,
    pub dimensions: usize,
    pub cost_per_1k_tokens: f64,
    pub strengths: Vec<String>,
}

impl ProviderPerformance {
    /// Get performance data for top embedding providers (as of late 2024)
    pub fn get_top_providers() -> Vec<Self> {
        vec![
            Self {
                provider: "Voyage AI".to_string(),
                model: "voyage-code-2".to_string(),
                mteb_score: 68.0, // Optimized for code, excellent performance
                dimensions: 1536,
                cost_per_1k_tokens: 0.12,
                strengths: vec![
                    "Optimized for code embeddings".to_string(),
                    "Excellent for code search and similarity".to_string(),
                    "Superior programming language understanding".to_string(),
                ],
            },
            Self {
                provider: "Voyage AI".to_string(),
                model: "voyage-large-2-instruct".to_string(),
                mteb_score: 69.5, // Top general MTEB performance
                dimensions: 1024,
                cost_per_1k_tokens: 0.12,
                strengths: vec![
                    "Top MTEB performance".to_string(),
                    "Excellent for retrieval".to_string(),
                    "Instruction-tuned".to_string(),
                ],
            },
            Self {
                provider: "Voyage AI".to_string(),
                model: "voyage-3-large".to_string(),
                mteb_score: 70.0, // Approximate MTEB score
                dimensions: 2048,
                cost_per_1k_tokens: 0.15,
                strengths: vec![
                    "Latest Voyage model".to_string(),
                    "Highest dimensions".to_string(),
                    "Best overall performance".to_string(),
                ],
            },
            Self {
                provider: "Cohere".to_string(),
                model: "embed-english-v3.0".to_string(),
                mteb_score: 68.0, // Approximate MTEB score
                dimensions: 1024,
                cost_per_1k_tokens: 0.10,
                strengths: vec![
                    "Excellent retrieval performance".to_string(),
                    "Cost effective".to_string(),
                    "Strong multilingual support".to_string(),
                ],
            },
            Self {
                provider: "OpenAI".to_string(),
                model: "text-embedding-3-large".to_string(),
                mteb_score: 64.6, // Official MTEB score
                dimensions: 3072,
                cost_per_1k_tokens: 0.13,
                strengths: vec![
                    "Widely supported".to_string(),
                    "High dimensions".to_string(),
                    "Good general performance".to_string(),
                ],
            },
            Self {
                provider: "OpenAI".to_string(),
                model: "text-embedding-3-small".to_string(),
                mteb_score: 62.3, // Official MTEB score
                dimensions: 1536,
                cost_per_1k_tokens: 0.02,
                strengths: vec![
                    "Very cost effective".to_string(),
                    "Fast inference".to_string(),
                    "Good for high-volume use".to_string(),
                ],
            },
        ]
    }

    /// Get recommended provider based on use case
    pub fn get_recommendation(use_case: &str) -> Option<Self> {
        let providers = Self::get_top_providers();
        
        match use_case.to_lowercase().as_str() {
            "best_performance" => providers.into_iter().max_by(|a, b| a.mteb_score.partial_cmp(&b.mteb_score).unwrap()),
            "cost_effective" => providers.into_iter().min_by(|a, b| a.cost_per_1k_tokens.partial_cmp(&b.cost_per_1k_tokens).unwrap()),
            "retrieval" => providers.into_iter().find(|p| p.model.contains("voyage-large-2-instruct")),
            "general" => providers.into_iter().find(|p| p.model.contains("embed-english-v3.0")),
            _ => providers.into_iter().next(),
        }
    }
}

/// Configuration helper for selecting the best available provider
pub struct ProviderSelector;

impl ProviderSelector {
    /// Automatically select the best available provider based on API keys
    pub fn select_best_provider() -> (String, String) {
        if std::env::var("VOYAGE_API_KEY").is_ok() {
            ("voyage".to_string(), "voyage-large-2-instruct".to_string())
        } else if std::env::var("COHERE_API_KEY").is_ok() {
            ("cohere".to_string(), "embed-english-v3.0".to_string())
        } else if std::env::var("OPENAI_API_KEY").is_ok() {
            ("openai".to_string(), "text-embedding-3-large".to_string())
        } else {
            ("simple".to_string(), "tfidf".to_string())
        }
    }

    /// Get provider configuration recommendations
    pub fn get_recommendations() -> String {
        format!(
            r#"
🚀 Embedding Provider Recommendations (Late 2024)

Based on MTEB leaderboard and performance benchmarks:

1. 🥇 BEST PERFORMANCE: Voyage AI
   - Model: voyage-large-2-instruct or voyage-3-large
   - MTEB Score: ~69.5-70.0
   - Set: export VOYAGE_API_KEY="your-key"

2. 🥈 BEST VALUE: Cohere
   - Model: embed-english-v3.0
   - MTEB Score: ~68.0, excellent cost/performance
   - Set: export COHERE_API_KEY="your-key"

3. 🥉 WIDELY SUPPORTED: OpenAI
   - Model: text-embedding-3-large (updated from 3-small)
   - MTEB Score: ~64.6
   - Set: export OPENAI_API_KEY="your-key"

Current selection: {}
"#,
            match Self::select_best_provider() {
                (provider, model) if provider == "voyage" => format!("✅ Voyage AI ({})", model),
                (provider, model) if provider == "cohere" => format!("✅ Cohere ({})", model),
                (provider, model) if provider == "openai" => format!("✅ OpenAI ({})", model),
                _ => "⚠️  Simple TF-IDF (no API keys found)".to_string(),
            }
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_provider_performance_data() {
        let providers = ProviderPerformance::get_top_providers();
        assert!(!providers.is_empty());
        
        // Voyage AI should be top performer
        let voyage = providers.iter().find(|p| p.model.contains("voyage-large-2-instruct")).unwrap();
        assert!(voyage.mteb_score > 69.0);
    }

    #[test]
    fn test_recommendations() {
        let best = ProviderPerformance::get_recommendation("best_performance");
        assert!(best.is_some());
        
        let cost_effective = ProviderPerformance::get_recommendation("cost_effective");
        assert!(cost_effective.is_some());
    }

    #[test]
    fn test_provider_selector() {
        let (provider, model) = ProviderSelector::select_best_provider();
        assert!(!provider.is_empty());
        assert!(!model.is_empty());
    }
}
