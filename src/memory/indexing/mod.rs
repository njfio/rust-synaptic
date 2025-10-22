//! Vector indexing for efficient similarity search.
//!
//! This module provides infrastructure for Approximate Nearest Neighbor (ANN) search
//! using various indexing strategies. The primary implementation uses HNSW
//! (Hierarchical Navigable Small World) graphs for logarithmic-complexity search.
//!
//! # Features
//!
//! - **VectorIndex Trait**: Generic interface for vector similarity indexes
//! - **HNSW Implementation**: High-performance ANN search with excellent recall
//! - **Index Persistence**: Save and load indexes from disk
//! - **Batch Operations**: Efficient bulk insertion
//! - **Multiple Distance Metrics**: Cosine, Euclidean, Manhattan, Dot Product
//!
//! # Examples
//!
//! ```rust,no_run
//! use synaptic::memory::indexing::{HnswIndex, IndexConfig, VectorIndex};
//! use synaptic::memory::types::{MemoryEntry, MemoryType, Metadata};
//! use chrono::Utc;
//!
//! # async fn example() -> Result<(), Box<dyn std::error::Error>> {
//! // Create index with 384-dimensional vectors
//! let config = IndexConfig::new(384)
//!     .with_ef_construction(200)
//!     .with_ef_search(50);
//!
//! let mut index = HnswIndex::new(config);
//!
//! // Add vectors
//! let vector = vec![0.1; 384];
//! let entry = MemoryEntry {
//!     key: "memory1".to_string(),
//!     content: "Example content".to_string(),
//!     memory_type: MemoryType::ShortTerm,
//!     metadata: Metadata::default(),
//!     created_at: Utc::now(),
//!     accessed_at: Utc::now(),
//!     access_count: 0,
//! };
//!
//! index.add(&vector, entry).await?;
//!
//! // Search for similar vectors
//! let query = vec![0.1; 384];
//! let results = index.search(&query, 10).await?;
//!
//! for result in results {
//!     println!("Found: {} (score: {:.4})", result.entry.key, result.score);
//! }
//! # Ok(())
//! # }
//! ```

pub mod hnsw;
pub mod manager;
pub mod vector;

pub use hnsw::HnswIndex;
pub use manager::IndexManager;
pub use vector::{
    DistanceMetric, IndexConfig, IndexError, IndexResult, IndexStats, SearchResult, VectorIndex,
};
