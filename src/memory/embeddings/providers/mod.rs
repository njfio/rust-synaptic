//! Concrete embedding provider implementations

pub mod tfidf;

// Re-export providers
pub use tfidf::{TfIdfProvider, TfIdfConfig};
