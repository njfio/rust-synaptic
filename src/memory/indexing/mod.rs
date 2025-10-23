//! Vector indexing for efficient similarity search
//!
//! This module provides high-performance vector indexing using various
//! algorithms for approximate nearest neighbor (ANN) search.

pub mod vector;
pub mod hnsw;

// Re-export main types
pub use vector::{
    VectorIndex, IndexStats, IndexConfig, DistanceMetric, SearchResult,
};

pub use hnsw::{HnswIndex, HnswParams};
