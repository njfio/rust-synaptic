//! Unified embedding provider configuration and API key management
//!
//! This module provides centralized configuration management for all
//! embedding providers, including API key handling, provider selection,
//! and validation.

use crate::error::{MemoryError, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;

#[cfg(feature = "llm-integration")]
use super::providers::{
    OpenAIConfig, OpenAIModel,
    OllamaConfig,
    CohereConfig, CohereModel, CohereInputType,
};

/// Unified configuration for all embedding providers
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingProviderConfig {
    /// Selected provider type
    pub provider: ProviderType,
    /// Provider-specific configurations
    pub provider_configs: HashMap<String, ProviderConfig>,
    /// Fallback provider chain
    pub fallback_chain: Vec<ProviderType>,
    /// Global settings
    pub global: GlobalConfig,
}

/// Provider type enumeration
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ProviderType {
    /// TF-IDF based (local, no API key)
    TfIdf,
    /// OpenAI API
    OpenAI,
    /// Ollama (local server)
    Ollama,
    /// Cohere API
    Cohere,
}

impl ProviderType {
    /// Get human-readable name
    pub fn name(&self) -> &'static str {
        match self {
            Self::TfIdf => "TF-IDF",
            Self::OpenAI => "OpenAI",
            Self::Ollama => "Ollama",
            Self::Cohere => "Cohere",
        }
    }

    /// Check if provider requires API key
    pub fn requires_api_key(&self) -> bool {
        matches!(self, Self::OpenAI | Self::Cohere)
    }

    /// Check if provider is local (doesn't require network)
    pub fn is_local(&self) -> bool {
        matches!(self, Self::TfIdf)
    }
}

/// Provider-specific configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ProviderConfig {
    TfIdf {
        embedding_dim: usize,
        min_word_length: usize,
        max_vocabulary_size: usize,
        normalize: bool,
    },
    OpenAI {
        api_key: Option<String>,
        model: String,
        endpoint: String,
        timeout_seconds: u64,
    },
    Ollama {
        endpoint: String,
        model: String,
        embedding_dim: usize,
        timeout_seconds: u64,
    },
    Cohere {
        api_key: Option<String>,
        model: String,
        endpoint: String,
        input_type: String,
        timeout_seconds: u64,
    },
}

/// Global configuration settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GlobalConfig {
    /// Enable caching of embeddings
    pub enable_cache: bool,
    /// Cache size (number of entries)
    pub cache_size: usize,
    /// Cache TTL in seconds
    pub cache_ttl_seconds: u64,
    /// Enable automatic fallback
    pub enable_fallback: bool,
    /// Maximum concurrent embedding requests
    pub max_concurrent_requests: usize,
}

impl Default for GlobalConfig {
    fn default() -> Self {
        Self {
            enable_cache: true,
            cache_size: 10000,
            cache_ttl_seconds: 3600,
            enable_fallback: true,
            max_concurrent_requests: 10,
        }
    }
}

impl Default for EmbeddingProviderConfig {
    fn default() -> Self {
        let mut provider_configs = HashMap::new();

        // Default TF-IDF config
        provider_configs.insert(
            "tfidf".to_string(),
            ProviderConfig::TfIdf {
                embedding_dim: 384,
                min_word_length: 3,
                max_vocabulary_size: 10000,
                normalize: true,
            },
        );

        Self {
            provider: ProviderType::TfIdf,
            provider_configs,
            fallback_chain: vec![],
            global: GlobalConfig::default(),
        }
    }
}

impl EmbeddingProviderConfig {
    /// Create a new configuration with a specific provider
    pub fn new(provider: ProviderType) -> Self {
        let mut config = Self::default();
        config.provider = provider;
        config
    }

    /// Add a provider configuration
    pub fn with_provider_config(
        mut self,
        provider: ProviderType,
        config: ProviderConfig,
    ) -> Self {
        self.provider_configs.insert(provider.name().to_lowercase(), config);
        self
    }

    /// Set fallback chain
    pub fn with_fallback_chain(mut self, chain: Vec<ProviderType>) -> Self {
        self.fallback_chain = chain;
        self
    }

    /// Set global configuration
    pub fn with_global(mut self, global: GlobalConfig) -> Self {
        self.global = global;
        self
    }

    /// Load configuration from TOML file
    pub fn from_toml_file(path: PathBuf) -> Result<Self> {
        let content = std::fs::read_to_string(path)
            .map_err(|e| MemoryError::Configuration(format!("Failed to read config file: {}", e)))?;

        toml::from_str(&content)
            .map_err(|e| MemoryError::Configuration(format!("Failed to parse TOML config: {}", e)))
    }

    /// Save configuration to TOML file
    pub fn to_toml_file(&self, path: PathBuf) -> Result<()> {
        let content = toml::to_string_pretty(self)
            .map_err(|e| MemoryError::Configuration(format!("Failed to serialize config: {}", e)))?;

        std::fs::write(path, content)
            .map_err(|e| MemoryError::Configuration(format!("Failed to write config file: {}", e)))?;

        Ok(())
    }

    /// Load configuration from environment variables
    pub fn from_env() -> Result<Self> {
        let mut config = Self::default();

        // Determine provider from env
        if let Ok(provider_str) = std::env::var("EMBEDDING_PROVIDER") {
            config.provider = match provider_str.to_lowercase().as_str() {
                "openai" => ProviderType::OpenAI,
                "ollama" => ProviderType::Ollama,
                "cohere" => ProviderType::Cohere,
                "tfidf" => ProviderType::TfIdf,
                _ => {
                    return Err(MemoryError::Configuration(format!(
                        "Unknown provider type: {}",
                        provider_str
                    )))
                }
            };
        }

        // Load OpenAI config if available
        if let Ok(api_key) = std::env::var("OPENAI_API_KEY") {
            let model = std::env::var("OPENAI_MODEL")
                .unwrap_or_else(|_| "text-embedding-3-small".to_string());
            let endpoint = std::env::var("OPENAI_ENDPOINT")
                .unwrap_or_else(|_| "https://api.openai.com/v1/embeddings".to_string());

            config.provider_configs.insert(
                "openai".to_string(),
                ProviderConfig::OpenAI {
                    api_key: Some(api_key),
                    model,
                    endpoint,
                    timeout_seconds: 30,
                },
            );
        }

        // Load Ollama config
        if let Ok(endpoint) = std::env::var("OLLAMA_ENDPOINT") {
            let model = std::env::var("OLLAMA_MODEL")
                .unwrap_or_else(|_| "nomic-embed-text".to_string());
            let embedding_dim = std::env::var("OLLAMA_EMBEDDING_DIM")
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(768);

            config.provider_configs.insert(
                "ollama".to_string(),
                ProviderConfig::Ollama {
                    endpoint,
                    model,
                    embedding_dim,
                    timeout_seconds: 30,
                },
            );
        }

        // Load Cohere config if available
        if let Ok(api_key) = std::env::var("COHERE_API_KEY") {
            let model = std::env::var("COHERE_MODEL")
                .unwrap_or_else(|_| "embed-english-light-v3.0".to_string());
            let endpoint = std::env::var("COHERE_ENDPOINT")
                .unwrap_or_else(|_| "https://api.cohere.ai/v1/embed".to_string());
            let input_type = std::env::var("COHERE_INPUT_TYPE")
                .unwrap_or_else(|_| "search_document".to_string());

            config.provider_configs.insert(
                "cohere".to_string(),
                ProviderConfig::Cohere {
                    api_key: Some(api_key),
                    model,
                    endpoint,
                    input_type,
                    timeout_seconds: 30,
                },
            );
        }

        Ok(config)
    }

    /// Validate configuration
    pub fn validate(&self) -> Result<()> {
        // Check if selected provider has configuration
        let provider_key = self.provider.name().to_lowercase();
        if !self.provider_configs.contains_key(&provider_key) {
            return Err(MemoryError::Configuration(format!(
                "No configuration found for provider {}",
                self.provider.name()
            )));
        }

        // Validate API keys for providers that require them
        if self.provider.requires_api_key() {
            if let Some(config) = self.provider_configs.get(&provider_key) {
                match config {
                    ProviderConfig::OpenAI { api_key, .. } => {
                        if api_key.is_none() || api_key.as_ref().unwrap().is_empty() {
                            return Err(MemoryError::Configuration(
                                "OpenAI API key is required".to_string(),
                            ));
                        }
                    }
                    ProviderConfig::Cohere { api_key, .. } => {
                        if api_key.is_none() || api_key.as_ref().unwrap().is_empty() {
                            return Err(MemoryError::Configuration(
                                "Cohere API key is required".to_string(),
                            ));
                        }
                    }
                    _ => {}
                }
            }
        }

        // Validate fallback chain
        for provider_type in &self.fallback_chain {
            let fallback_key = provider_type.name().to_lowercase();
            if !self.provider_configs.contains_key(&fallback_key) {
                return Err(MemoryError::Configuration(format!(
                    "No configuration found for fallback provider {}",
                    provider_type.name()
                )));
            }
        }

        Ok(())
    }

    /// Get API key for a specific provider
    pub fn get_api_key(&self, provider: ProviderType) -> Option<String> {
        let provider_key = provider.name().to_lowercase();
        self.provider_configs.get(&provider_key).and_then(|config| {
            match config {
                ProviderConfig::OpenAI { api_key, .. } => api_key.clone(),
                ProviderConfig::Cohere { api_key, .. } => api_key.clone(),
                _ => None,
            }
        })
    }

    /// Create a default OpenAI configuration
    pub fn openai_default(api_key: String) -> Self {
        let mut config = Self::new(ProviderType::OpenAI);
        config.provider_configs.insert(
            "openai".to_string(),
            ProviderConfig::OpenAI {
                api_key: Some(api_key),
                model: "text-embedding-3-small".to_string(),
                endpoint: "https://api.openai.com/v1/embeddings".to_string(),
                timeout_seconds: 30,
            },
        );
        config.fallback_chain = vec![ProviderType::TfIdf];
        config
    }

    /// Create a default Ollama configuration
    pub fn ollama_default() -> Self {
        let mut config = Self::new(ProviderType::Ollama);
        config.provider_configs.insert(
            "ollama".to_string(),
            ProviderConfig::Ollama {
                endpoint: "http://localhost:11434".to_string(),
                model: "nomic-embed-text".to_string(),
                embedding_dim: 768,
                timeout_seconds: 30,
            },
        );
        config.fallback_chain = vec![ProviderType::TfIdf];
        config
    }

    /// Create a default Cohere configuration
    pub fn cohere_default(api_key: String) -> Self {
        let mut config = Self::new(ProviderType::Cohere);
        config.provider_configs.insert(
            "cohere".to_string(),
            ProviderConfig::Cohere {
                api_key: Some(api_key),
                model: "embed-english-light-v3.0".to_string(),
                endpoint: "https://api.cohere.ai/v1/embed".to_string(),
                input_type: "search_document".to_string(),
                timeout_seconds: 30,
            },
        );
        config.fallback_chain = vec![ProviderType::TfIdf];
        config
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_provider_type_properties() {
        assert!(ProviderType::OpenAI.requires_api_key());
        assert!(ProviderType::Cohere.requires_api_key());
        assert!(!ProviderType::TfIdf.requires_api_key());
        assert!(!ProviderType::Ollama.requires_api_key());

        assert!(ProviderType::TfIdf.is_local());
        assert!(!ProviderType::OpenAI.is_local());
    }

    #[test]
    fn test_default_config() {
        let config = EmbeddingProviderConfig::default();
        assert_eq!(config.provider, ProviderType::TfIdf);
        assert!(config.provider_configs.contains_key("tfidf"));
        assert!(config.global.enable_cache);
    }

    #[test]
    fn test_openai_default_config() {
        let config = EmbeddingProviderConfig::openai_default("test-key".to_string());
        assert_eq!(config.provider, ProviderType::OpenAI);
        assert_eq!(config.fallback_chain, vec![ProviderType::TfIdf]);

        let api_key = config.get_api_key(ProviderType::OpenAI);
        assert_eq!(api_key, Some("test-key".to_string()));
    }

    #[test]
    fn test_validate_with_api_key() {
        let config = EmbeddingProviderConfig::openai_default("test-key".to_string());
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_validate_without_api_key() {
        let mut config = EmbeddingProviderConfig::new(ProviderType::OpenAI);
        config.provider_configs.insert(
            "openai".to_string(),
            ProviderConfig::OpenAI {
                api_key: None,
                model: "test".to_string(),
                endpoint: "test".to_string(),
                timeout_seconds: 30,
            },
        );

        assert!(config.validate().is_err());
    }

    #[test]
    fn test_ollama_default_config() {
        let config = EmbeddingProviderConfig::ollama_default();
        assert_eq!(config.provider, ProviderType::Ollama);
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_serialization() {
        let config = EmbeddingProviderConfig::openai_default("test-key".to_string());

        let toml_str = toml::to_string(&config).unwrap();
        assert!(toml_str.contains("openai"));
        assert!(toml_str.contains("test-key"));

        let deserialized: EmbeddingProviderConfig = toml::from_str(&toml_str).unwrap();
        assert_eq!(deserialized.provider, ProviderType::OpenAI);
    }
}
