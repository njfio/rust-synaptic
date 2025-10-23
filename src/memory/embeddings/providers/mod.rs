//! Concrete embedding provider implementations

pub mod tfidf;

#[cfg(feature = "llm-integration")]
pub mod openai;

#[cfg(feature = "llm-integration")]
pub mod ollama;

#[cfg(feature = "llm-integration")]
pub mod cohere;

// Re-export providers
pub use tfidf::{TfIdfProvider, TfIdfConfig};

#[cfg(feature = "llm-integration")]
pub use openai::{OpenAIProvider, OpenAIConfig, OpenAIModel};

#[cfg(feature = "llm-integration")]
pub use ollama::{OllamaProvider, OllamaConfig};

#[cfg(feature = "llm-integration")]
pub use cohere::{CohereProvider, CohereConfig, CohereModel, CohereInputType};
