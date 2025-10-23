//! Multi-provider embedding system with automatic fallback
//!
//! This module provides provider selection, fallback logic, and automatic
//! failover for embedding generation across multiple providers.

use super::provider::{EmbeddingProvider, Embedding, EmbedOptions, ProviderCapabilities};
use crate::error::{MemoryError, Result};
use async_trait::async_trait;
use std::sync::Arc;
use tracing::{debug, warn};

/// Multi-provider embedding coordinator
///
/// Manages multiple embedding providers with automatic failover,
/// allowing graceful degradation when primary providers are unavailable.
pub struct MultiProvider {
    /// Primary provider (highest priority)
    primary: Arc<dyn EmbeddingProvider>,
    /// Fallback providers (tried in order)
    fallbacks: Vec<Arc<dyn EmbeddingProvider>>,
    /// Configuration
    config: MultiProviderConfig,
}

/// Configuration for multi-provider system
#[derive(Debug, Clone)]
pub struct MultiProviderConfig {
    /// Maximum attempts before giving up
    pub max_attempts: usize,
    /// Whether to fail if all providers fail
    pub fail_on_all_errors: bool,
    /// Whether to warn on fallback usage
    pub warn_on_fallback: bool,
}

impl Default for MultiProviderConfig {
    fn default() -> Self {
        Self {
            max_attempts: 3,
            fail_on_all_errors: true,
            warn_on_fallback: true,
        }
    }
}

impl MultiProvider {
    /// Create a new multi-provider with a primary provider
    pub fn new(primary: Arc<dyn EmbeddingProvider>) -> Self {
        Self {
            primary,
            fallbacks: Vec::new(),
            config: MultiProviderConfig::default(),
        }
    }

    /// Add a fallback provider
    pub fn with_fallback(mut self, provider: Arc<dyn EmbeddingProvider>) -> Self {
        self.fallbacks.push(provider);
        self
    }

    /// Set configuration
    pub fn with_config(mut self, config: MultiProviderConfig) -> Self {
        self.config = config;
        self
    }

    /// Get all providers in priority order
    fn providers(&self) -> Vec<Arc<dyn EmbeddingProvider>> {
        let mut providers = vec![Arc::clone(&self.primary)];
        providers.extend(self.fallbacks.iter().map(Arc::clone));
        providers
    }

    /// Try embedding with a specific provider
    async fn try_provider(
        &self,
        provider: &Arc<dyn EmbeddingProvider>,
        text: &str,
        options: Option<&EmbedOptions>,
    ) -> Result<Embedding> {
        if !provider.is_available() {
            return Err(MemoryError::External(format!(
                "Provider {} is not available",
                provider.name()
            )));
        }

        provider.embed(text, options).await
    }

    /// Try batch embedding with a specific provider
    async fn try_provider_batch(
        &self,
        provider: &Arc<dyn EmbeddingProvider>,
        texts: &[String],
        options: Option<&EmbedOptions>,
    ) -> Result<Vec<Embedding>> {
        if !provider.is_available() {
            return Err(MemoryError::External(format!(
                "Provider {} is not available",
                provider.name()
            )));
        }

        provider.embed_batch(texts, options).await
    }
}

#[async_trait]
impl EmbeddingProvider for MultiProvider {
    async fn embed(&self, text: &str, options: Option<&EmbedOptions>) -> Result<Embedding> {
        let providers = self.providers();
        let mut last_error = None;

        for (index, provider) in providers.iter().enumerate() {
            if index > 0 && self.config.warn_on_fallback {
                warn!(
                    "Falling back to provider {} (attempt {})",
                    provider.name(),
                    index + 1
                );
            }

            match self.try_provider(provider, text, options).await {
                Ok(embedding) => {
                    if index > 0 {
                        debug!(
                            "Successfully used fallback provider {} after {} failed attempts",
                            provider.name(),
                            index
                        );
                    }
                    return Ok(embedding);
                }
                Err(e) => {
                    debug!(
                        "Provider {} failed: {}",
                        provider.name(),
                        e
                    );
                    last_error = Some(e);

                    if index + 1 >= self.config.max_attempts {
                        break;
                    }
                }
            }
        }

        if self.config.fail_on_all_errors {
            Err(last_error.unwrap_or_else(|| {
                MemoryError::External("All embedding providers failed".to_string())
            }))
        } else {
            // Return a zero vector as last resort
            warn!("All providers failed, returning zero vector");
            Ok(Embedding::new(
                vec![0.0; self.embedding_dimension()],
                "fallback-zero".to_string(),
            ))
        }
    }

    async fn embed_batch(
        &self,
        texts: &[String],
        options: Option<&EmbedOptions>,
    ) -> Result<Vec<Embedding>> {
        let providers = self.providers();
        let mut last_error = None;

        for (index, provider) in providers.iter().enumerate() {
            if index > 0 && self.config.warn_on_fallback {
                warn!(
                    "Falling back to provider {} for batch (attempt {})",
                    provider.name(),
                    index + 1
                );
            }

            match self.try_provider_batch(provider, texts, options).await {
                Ok(embeddings) => {
                    if index > 0 {
                        debug!(
                            "Successfully used fallback provider {} for batch after {} failed attempts",
                            provider.name(),
                            index
                        );
                    }
                    return Ok(embeddings);
                }
                Err(e) => {
                    debug!(
                        "Provider {} failed for batch: {}",
                        provider.name(),
                        e
                    );
                    last_error = Some(e);

                    if index + 1 >= self.config.max_attempts {
                        break;
                    }
                }
            }
        }

        if self.config.fail_on_all_errors {
            Err(last_error.unwrap_or_else(|| {
                MemoryError::External("All embedding providers failed for batch".to_string())
            }))
        } else {
            // Return zero vectors as last resort
            warn!("All providers failed for batch, returning zero vectors");
            Ok(texts
                .iter()
                .map(|_| {
                    Embedding::new(
                        vec![0.0; self.embedding_dimension()],
                        "fallback-zero".to_string(),
                    )
                })
                .collect())
        }
    }

    fn embedding_dimension(&self) -> usize {
        self.primary.embedding_dimension()
    }

    fn name(&self) -> &'static str {
        "MultiProvider"
    }

    fn model_id(&self) -> String {
        format!(
            "multi({},{}fallbacks)",
            self.primary.model_id(),
            self.fallbacks.len()
        )
    }

    fn is_available(&self) -> bool {
        // At least one provider must be available
        self.providers().iter().any(|p| p.is_available())
    }

    fn capabilities(&self) -> ProviderCapabilities {
        // Return primary provider's capabilities
        self.primary.capabilities()
    }
}

/// Builder for creating a multi-provider system
pub struct MultiProviderBuilder {
    primary: Option<Arc<dyn EmbeddingProvider>>,
    fallbacks: Vec<Arc<dyn EmbeddingProvider>>,
    config: MultiProviderConfig,
}

impl MultiProviderBuilder {
    /// Create a new builder
    pub fn new() -> Self {
        Self {
            primary: None,
            fallbacks: Vec::new(),
            config: MultiProviderConfig::default(),
        }
    }

    /// Set the primary provider
    pub fn primary(mut self, provider: Arc<dyn EmbeddingProvider>) -> Self {
        self.primary = Some(provider);
        self
    }

    /// Add a fallback provider
    pub fn fallback(mut self, provider: Arc<dyn EmbeddingProvider>) -> Self {
        self.fallbacks.push(provider);
        self
    }

    /// Set configuration
    pub fn config(mut self, config: MultiProviderConfig) -> Self {
        self.config = config;
        self
    }

    /// Build the multi-provider
    pub fn build(self) -> Result<MultiProvider> {
        let primary = self.primary.ok_or_else(|| {
            MemoryError::Configuration("Primary provider is required".to_string())
        })?;

        Ok(MultiProvider {
            primary,
            fallbacks: self.fallbacks,
            config: self.config,
        })
    }
}

impl Default for MultiProviderBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::embeddings::providers::TfIdfProvider;

    #[tokio::test]
    async fn test_multi_provider_creation() {
        let primary = Arc::new(TfIdfProvider::default());
        let fallback = Arc::new(TfIdfProvider::default());

        let multi = MultiProvider::new(Arc::clone(&primary))
            .with_fallback(fallback);

        assert_eq!(multi.name(), "MultiProvider");
        assert!(multi.is_available());
        assert_eq!(multi.embedding_dimension(), 384);
    }

    #[tokio::test]
    async fn test_multi_provider_builder() {
        let primary = Arc::new(TfIdfProvider::default());
        let fallback = Arc::new(TfIdfProvider::default());

        let multi = MultiProviderBuilder::new()
            .primary(primary)
            .fallback(fallback)
            .build();

        assert!(multi.is_ok());
        let multi = multi.unwrap();
        assert_eq!(multi.name(), "MultiProvider");
    }

    #[tokio::test]
    async fn test_multi_provider_builder_no_primary() {
        let fallback = Arc::new(TfIdfProvider::default());

        let multi = MultiProviderBuilder::new()
            .fallback(fallback)
            .build();

        assert!(multi.is_err());
    }

    #[tokio::test]
    async fn test_multi_provider_embedding() {
        let primary = Arc::new(TfIdfProvider::default());
        let multi = MultiProvider::new(primary);

        let text = "test embedding with multi-provider";
        let result = multi.embed(text, None).await;

        assert!(result.is_ok());
        let embedding = result.unwrap();
        assert_eq!(embedding.dimension(), 384);
    }

    #[tokio::test]
    async fn test_multi_provider_batch() {
        let primary = Arc::new(TfIdfProvider::default());
        let multi = MultiProvider::new(primary);

        let texts = vec![
            "first text".to_string(),
            "second text".to_string(),
        ];
        let result = multi.embed_batch(&texts, None).await;

        assert!(result.is_ok());
        let embeddings = result.unwrap();
        assert_eq!(embeddings.len(), 2);
    }
}
