//! Concrete embedding provider implementations

pub mod fallback;
pub mod tfidf;

#[cfg(feature = "llm-integration")]
pub mod openai;

#[cfg(feature = "llm-integration")]
pub mod ollama;

#[cfg(feature = "llm-integration")]
pub mod cohere;

#[cfg(feature = "ml-models")]
pub mod candle;

// Re-export providers
pub use fallback::FallbackEmbeddingProvider;
pub use tfidf::{TfIdfConfig, TfIdfProvider};

#[cfg(feature = "llm-integration")]
pub use openai::{OpenAIConfig, OpenAIModel, OpenAIProvider};

#[cfg(feature = "llm-integration")]
pub use ollama::{OllamaConfig, OllamaProvider};

#[cfg(feature = "llm-integration")]
pub use cohere::{CohereConfig, CohereInputType, CohereModel, CohereProvider};

#[cfg(feature = "ml-models")]
pub use candle::CandleEmbeddingProvider;
