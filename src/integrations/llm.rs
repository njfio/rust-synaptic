// Real LLM Integration
// Implements actual API calls to OpenAI, Anthropic, and other LLM providers

#[cfg(feature = "llm-integration")]
use reqwest::{Client, header::{HeaderMap, HeaderValue, AUTHORIZATION, CONTENT_TYPE}};
use crate::error::{Result, MemoryError};
use crate::memory::types::MemoryEntry;
use crate::memory::management::analytics::{AnalyticsInsight, InsightType, InsightPriority};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// LLM provider configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LLMConfig {
    /// Primary LLM provider
    pub provider: LLMProvider,
    /// API key for the provider
    pub api_key: String,
    /// API base URL (for custom endpoints)
    pub base_url: Option<String>,
    /// Model name to use
    pub model: String,
    /// Maximum tokens per request
    pub max_tokens: u32,
    /// Temperature for generation
    pub temperature: f32,
    /// Request timeout in seconds
    pub timeout_secs: u64,
    /// Rate limiting (requests per minute)
    pub rate_limit: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LLMProvider {
    OpenAI,
    Anthropic,
    Cohere,
    HuggingFace,
    Custom,
}

impl Default for LLMConfig {
    fn default() -> Self {
        // Auto-detect provider based on available API keys
        let (provider, api_key, model) = if let Ok(key) = std::env::var("ANTHROPIC_API_KEY") {
            (LLMProvider::Anthropic, key, "claude-3-5-haiku-20241022".to_string())
        } else if let Ok(key) = std::env::var("OPENAI_API_KEY") {
            (LLMProvider::OpenAI, key, "gpt-4".to_string())
        } else {
            (LLMProvider::OpenAI, String::new(), "gpt-4".to_string())
        };

        Self {
            provider,
            api_key,
            base_url: None,
            model,
            max_tokens: 1000,
            temperature: 0.7,
            timeout_secs: 30,
            rate_limit: 60,
        }
    }
}

/// Real LLM client for API calls
#[derive(Debug)]
pub struct LLMClient {
    config: LLMConfig,
    #[cfg(feature = "llm-integration")]
    client: Client,
    metrics: LLMMetrics,
    rate_limiter: RateLimiter,
}

#[derive(Debug, Clone, Default)]
pub struct LLMMetrics {
    pub requests_made: u64,
    pub tokens_consumed: u64,
    pub total_cost_usd: f64,
    pub avg_response_time_ms: f64,
    pub error_count: u64,
}

#[derive(Debug)]
struct RateLimiter {
    requests: Vec<chrono::DateTime<chrono::Utc>>,
    limit: u32,
}

impl RateLimiter {
    fn new(limit: u32) -> Self {
        Self {
            requests: Vec::new(),
            limit,
        }
    }

    async fn check_rate_limit(&mut self) -> Result<()> {
        let now = chrono::Utc::now();
        let one_minute_ago = now - chrono::Duration::minutes(1);
        
        // Remove old requests
        self.requests.retain(|&time| time > one_minute_ago);
        
        if self.requests.len() >= self.limit as usize {
            let wait_time = self.requests[0] + chrono::Duration::minutes(1) - now;
            if wait_time > chrono::Duration::zero() {
                tokio::time::sleep(std::time::Duration::from_millis(wait_time.num_milliseconds() as u64)).await;
            }
        }
        
        self.requests.push(now);
        Ok(())
    }
}

impl LLMClient {
    /// Create a new LLM client with real API integration
    pub async fn new(config: LLMConfig) -> Result<Self> {
        #[cfg(feature = "llm-integration")]
        {
            let mut headers = HeaderMap::new();
            
            match config.provider {
                LLMProvider::OpenAI => {
                    headers.insert(AUTHORIZATION, HeaderValue::from_str(&format!("Bearer {}", config.api_key))
                        .map_err(|e| MemoryError::configuration(format!("Invalid API key: {}", e)))?);
                },
                LLMProvider::Anthropic => {
                    headers.insert("x-api-key", HeaderValue::from_str(&config.api_key)
                        .map_err(|e| MemoryError::configuration(format!("Invalid API key: {}", e)))?);
                    headers.insert("anthropic-version", HeaderValue::from_static("2023-06-01"));
                },
                _ => {
                    headers.insert(AUTHORIZATION, HeaderValue::from_str(&format!("Bearer {}", config.api_key))
                        .map_err(|e| MemoryError::configuration(format!("Invalid API key: {}", e)))?);
                }
            }
            
            headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));

            let client = Client::builder()
                .timeout(std::time::Duration::from_secs(config.timeout_secs))
                .default_headers(headers)
                .build()
                .map_err(|e| MemoryError::configuration(format!("Failed to create HTTP client: {}", e)))?;

            Ok(Self {
                config: config.clone(),
                client,
                metrics: LLMMetrics::default(),
                rate_limiter: RateLimiter::new(config.rate_limit),
            })
        }

        #[cfg(not(feature = "llm-integration"))]
        {
            Err(MemoryError::configuration("LLM integration feature not enabled"))
        }
    }

    /// Generate insights from memory data using LLM
    #[cfg(feature = "llm-integration")]
    pub async fn generate_insights(&mut self, memories: &[MemoryEntry], context: &str) -> Result<Vec<AnalyticsInsight>> {
        self.rate_limiter.check_rate_limit().await?;
        
        let start_time = std::time::Instant::now();

        // Prepare prompt
        let memory_summary = memories.iter()
            .take(10) // Limit to avoid token limits
            .map(|m| format!("- {}: {}", m.key, m.value.chars().take(100).collect::<String>()))
            .collect::<Vec<_>>()
            .join("\n");

        let prompt = format!(
            "Analyze the following memory entries and provide actionable insights:\n\n{}\n\nContext: {}\n\nProvide 3-5 specific insights in JSON format with fields: title, description, insight_type, priority, confidence.",
            memory_summary, context
        );

        let response = self.make_llm_request(&prompt).await?;
        
        // Parse response into insights
        let insights = self.parse_insights_response(&response)?;
        
        self.metrics.requests_made += 1;
        self.metrics.avg_response_time_ms = 
            (self.metrics.avg_response_time_ms * (self.metrics.requests_made - 1) as f64 + 
             start_time.elapsed().as_millis() as f64) / self.metrics.requests_made as f64;

        Ok(insights)
    }

    /// Summarize memory content using LLM
    #[cfg(feature = "llm-integration")]
    pub async fn summarize_memory(&mut self, memory: &MemoryEntry) -> Result<String> {
        self.rate_limiter.check_rate_limit().await?;
        
        let prompt = format!(
            "Summarize the following memory entry in 2-3 sentences:\n\nKey: {}\nContent: {}\n\nSummary:",
            memory.key, memory.value
        );

        let response = self.make_llm_request(&prompt).await?;
        Ok(response.trim().to_string())
    }

    /// Generate memory relationships using LLM
    #[cfg(feature = "llm-integration")]
    pub async fn find_relationships(&mut self, source: &MemoryEntry, candidates: &[MemoryEntry]) -> Result<Vec<MemoryRelationship>> {
        self.rate_limiter.check_rate_limit().await?;
        
        let candidates_text = candidates.iter()
            .take(5) // Limit candidates
            .map(|m| format!("- {}: {}", m.key, m.value.chars().take(50).collect::<String>()))
            .collect::<Vec<_>>()
            .join("\n");

        let prompt = format!(
            "Analyze relationships between the source memory and candidate memories:\n\nSource: {} - {}\n\nCandidates:\n{}\n\nReturn relationships in JSON format with fields: target_key, relationship_type, strength (0-1), description.",
            source.key, source.value.chars().take(100).collect::<String>(), candidates_text
        );

        let response = self.make_llm_request(&prompt).await?;
        let relationships = self.parse_relationships_response(&response)?;
        
        Ok(relationships)
    }

    /// Make actual API request to LLM provider
    #[cfg(feature = "llm-integration")]
    async fn make_llm_request(&mut self, prompt: &str) -> Result<String> {
        let url = match self.config.provider {
            LLMProvider::OpenAI => {
                self.config.base_url.as_deref().unwrap_or("https://api.openai.com/v1/chat/completions")
            },
            LLMProvider::Anthropic => {
                self.config.base_url.as_deref().unwrap_or("https://api.anthropic.com/v1/messages")
            },
            LLMProvider::Cohere => {
                self.config.base_url.as_deref().unwrap_or("https://api.cohere.ai/v1/generate")
            },
            _ => return Err(MemoryError::configuration("Unsupported LLM provider")),
        };

        let request_body = match self.config.provider {
            LLMProvider::OpenAI => {
                serde_json::json!({
                    "model": self.config.model,
                    "messages": [
                        {"role": "user", "content": prompt}
                    ],
                    "max_tokens": self.config.max_tokens,
                    "temperature": self.config.temperature
                })
            },
            LLMProvider::Anthropic => {
                serde_json::json!({
                    "model": self.config.model,
                    "max_tokens": self.config.max_tokens,
                    "messages": [
                        {"role": "user", "content": prompt}
                    ]
                })
            },
            LLMProvider::Cohere => {
                serde_json::json!({
                    "model": self.config.model,
                    "prompt": prompt,
                    "max_tokens": self.config.max_tokens,
                    "temperature": self.config.temperature
                })
            },
            _ => return Err(MemoryError::configuration("Unsupported provider")),
        };

        let response = self.client
            .post(url)
            .json(&request_body)
            .send()
            .await
            .map_err(|e| MemoryError::storage(format!("LLM API request failed: {}", e)))?;

        if !response.status().is_success() {
            let error_text = response.text().await.unwrap_or_default();
            return Err(MemoryError::storage(format!("LLM API error: {}", error_text)));
        }

        let response_json: serde_json::Value = response.json().await
            .map_err(|e| MemoryError::storage(format!("Failed to parse LLM response: {}", e)))?;

        // Extract content based on provider
        let content = match self.config.provider {
            LLMProvider::OpenAI => {
                response_json["choices"][0]["message"]["content"]
                    .as_str()
                    .unwrap_or("")
                    .to_string()
            },
            LLMProvider::Anthropic => {
                response_json["content"][0]["text"]
                    .as_str()
                    .unwrap_or("")
                    .to_string()
            },
            LLMProvider::Cohere => {
                response_json["generations"][0]["text"]
                    .as_str()
                    .unwrap_or("")
                    .to_string()
            },
            _ => return Err(MemoryError::configuration("Unsupported provider")),
        };

        // Update token usage metrics
        if let Some(usage) = response_json.get("usage") {
            if let Some(total_tokens) = usage["total_tokens"].as_u64() {
                self.metrics.tokens_consumed += total_tokens;
                
                // Estimate cost (rough estimates)
                let cost_per_token = match self.config.provider {
                    LLMProvider::OpenAI => 0.00003, // GPT-4 pricing
                    LLMProvider::Anthropic => 0.00003, // Claude pricing
                    _ => 0.00001,
                };
                self.metrics.total_cost_usd += total_tokens as f64 * cost_per_token;
            }
        }

        Ok(content)
    }

    #[cfg(feature = "llm-integration")]
    fn parse_insights_response(&self, response: &str) -> Result<Vec<AnalyticsInsight>> {
        // Try to extract JSON from response
        let json_start = response.find('[').or_else(|| response.find('{'));
        let json_end = response.rfind(']').or_else(|| response.rfind('}'));
        
        if let (Some(start), Some(end)) = (json_start, json_end) {
            let json_str = &response[start..=end];
            
            // Try to parse as array first, then as single object
            if let Ok(insights_array) = serde_json::from_str::<Vec<serde_json::Value>>(json_str) {
                return Ok(insights_array.into_iter()
                    .filter_map(|v| self.json_to_insight(v).ok())
                    .collect());
            } else if let Ok(insight_obj) = serde_json::from_str::<serde_json::Value>(json_str) {
                if let Ok(insight) = self.json_to_insight(insight_obj) {
                    return Ok(vec![insight]);
                }
            }
        }
        
        // Fallback: create insight from raw text
        Ok(vec![AnalyticsInsight {
            id: uuid::Uuid::new_v4().to_string(),
            title: "LLM Generated Insight".to_string(),
            description: response.to_string(),
            insight_type: InsightType::General,
            priority: InsightPriority::Medium,
            confidence: 0.7,
            timestamp: chrono::Utc::now(),
            data: HashMap::new(),
        }])
    }

    #[cfg(feature = "llm-integration")]
    fn json_to_insight(&self, value: serde_json::Value) -> Result<AnalyticsInsight> {
        Ok(AnalyticsInsight {
            id: uuid::Uuid::new_v4().to_string(),
            title: value["title"].as_str().unwrap_or("Untitled").to_string(),
            description: value["description"].as_str().unwrap_or("").to_string(),
            insight_type: match value["insight_type"].as_str().unwrap_or("general") {
                "usage_pattern" => InsightType::UsagePattern,
                "performance_optimization" => InsightType::PerformanceOptimization,
                "anomaly_detection" => InsightType::AnomalyDetection,
                _ => InsightType::General,
            },
            priority: match value["priority"].as_str().unwrap_or("medium") {
                "high" => InsightPriority::High,
                "low" => InsightPriority::Low,
                _ => InsightPriority::Medium,
            },
            confidence: value["confidence"].as_f64().unwrap_or(0.5),
            timestamp: chrono::Utc::now(),
            data: HashMap::new(),
        })
    }

    #[cfg(feature = "llm-integration")]
    fn parse_relationships_response(&self, _response: &str) -> Result<Vec<MemoryRelationship>> {
        // Similar JSON parsing logic for relationships
        Ok(Vec::new()) // Simplified for now
    }

    /// Health check for LLM connection
    pub async fn health_check(&self) -> Result<()> {
        #[cfg(feature = "llm-integration")]
        {
            // For health check, we'll just verify we have an API key
            // Making actual API calls in health checks can be expensive and rate-limited
            match self.config.provider {
                LLMProvider::OpenAI => {
                    if self.config.api_key.is_empty() || self.config.api_key == "demo_key" {
                        return Err(MemoryError::storage("OpenAI API key not configured"));
                    }
                },
                LLMProvider::Anthropic => {
                    if self.config.api_key.is_empty() || self.config.api_key == "demo_key" {
                        return Err(MemoryError::storage("Anthropic API key not configured"));
                    }
                },
                _ => {}, // Skip health check for other providers
            };
        }
        Ok(())
    }

    /// Shutdown LLM client
    pub async fn shutdown(&mut self) -> Result<()> {
        // No specific cleanup needed for HTTP client
        Ok(())
    }

    /// Get LLM metrics
    pub fn get_metrics(&self) -> &LLMMetrics {
        &self.metrics
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryRelationship {
    pub target_key: String,
    pub relationship_type: String,
    pub strength: f32,
    pub description: String,
}

#[cfg(not(feature = "llm-integration"))]
impl LLMClient {
    pub async fn new(_config: LLMConfig) -> Result<Self> {
        Err(MemoryError::configuration("LLM integration feature not enabled"))
    }

    pub async fn health_check(&self) -> Result<()> {
        Err(MemoryError::configuration("LLM integration feature not enabled"))
    }

    pub async fn shutdown(&mut self) -> Result<()> {
        Ok(())
    }

    pub fn get_metrics(&self) -> &LLMMetrics {
        &LLMMetrics::default()
    }
}
