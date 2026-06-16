//! Vector indexing for efficient similarity search
//!
//! This module provides high-performance vector indexing using various
//! algorithms for approximate nearest neighbor (ANN) search.

pub mod hnsw;
pub mod vector;

// Re-export main types
pub use vector::{DistanceMetric, IndexConfig, IndexStats, SearchResult, VectorIndex};

pub use hnsw::{HnswIndex, HnswParams};
