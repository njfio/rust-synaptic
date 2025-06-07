//! Similarity functions for vector embeddings
//! 
//! This module provides various similarity and distance metrics
//! for comparing embedding vectors.

/// Calculate cosine similarity between two vectors
/// Returns a value between -1.0 and 1.0, where 1.0 means identical direction
pub fn cosine_similarity(a: &[f64], b: &[f64]) -> f64 {
    if a.len() != b.len() {
        return 0.0;
    }

    let dot_product: f64 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let magnitude_a: f64 = a.iter().map(|x| x * x).sum::<f64>().sqrt();
    let magnitude_b: f64 = b.iter().map(|x| x * x).sum::<f64>().sqrt();
    
    if magnitude_a == 0.0 || magnitude_b == 0.0 {
        0.0
    } else {
        dot_product / (magnitude_a * magnitude_b)
    }
}

/// Calculate Euclidean distance between two vectors
/// Returns a non-negative value, where 0.0 means identical vectors
pub fn euclidean_distance(a: &[f64], b: &[f64]) -> f64 {
    if a.len() != b.len() {
        return f64::INFINITY;
    }

    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).powi(2))
        .sum::<f64>()
        .sqrt()
}

/// Calculate Manhattan (L1) distance between two vectors
/// Returns a non-negative value, where 0.0 means identical vectors
pub fn manhattan_distance(a: &[f64], b: &[f64]) -> f64 {
    if a.len() != b.len() {
        return f64::INFINITY;
    }

    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .sum()
}

/// Calculate dot product between two vectors
pub fn dot_product(a: &[f64], b: &[f64]) -> f64 {
    if a.len() != b.len() {
        return 0.0;
    }

    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

/// Calculate Jaccard similarity between two vectors (treating them as sets)
/// This treats non-zero elements as present in the set
pub fn jaccard_similarity(a: &[f64], b: &[f64]) -> f64 {
    if a.len() != b.len() {
        return 0.0;
    }

    let mut intersection = 0;
    let mut union = 0;

    for (x, y) in a.iter().zip(b.iter()) {
        let x_present = x.abs() > f64::EPSILON;
        let y_present = y.abs() > f64::EPSILON;

        if x_present && y_present {
            intersection += 1;
        }
        if x_present || y_present {
            union += 1;
        }
    }

    if union == 0 {
        1.0 // Both vectors are zero vectors
    } else {
        intersection as f64 / union as f64
    }
}

/// Calculate angular distance between two vectors
/// Returns a value between 0.0 and π, where 0.0 means identical direction
pub fn angular_distance(a: &[f64], b: &[f64]) -> f64 {
    let cosine_sim = cosine_similarity(a, b);
    // Clamp to [-1, 1] to handle floating point errors
    let cosine_sim = cosine_sim.max(-1.0).min(1.0);
    cosine_sim.acos()
}

/// Normalize a vector to unit length
pub fn normalize_vector(vector: &mut [f64]) {
    let magnitude: f64 = vector.iter().map(|x| x * x).sum::<f64>().sqrt();
    
    if magnitude > 0.0 {
        for value in vector.iter_mut() {
            *value /= magnitude;
        }
    }
}

/// Calculate the magnitude (L2 norm) of a vector
pub fn vector_magnitude(vector: &[f64]) -> f64 {
    vector.iter().map(|x| x * x).sum::<f64>().sqrt()
}

/// Find the k most similar vectors to a query vector
pub fn find_k_most_similar(
    query: &[f64],
    vectors: &[(Vec<f64>, usize)], // (vector, id)
    k: usize,
    similarity_fn: fn(&[f64], &[f64]) -> f64,
) -> Vec<(usize, f64)> {
    let mut similarities: Vec<(usize, f64)> = vectors
        .iter()
        .map(|(vector, id)| (*id, similarity_fn(query, vector)))
        .collect();

    // Sort by similarity (highest first)
    similarities.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    
    // Take top k
    similarities.truncate(k);
    similarities
}

/// Similarity metrics enum for easy selection
#[derive(Debug, Clone, Copy)]
pub enum SimilarityMetric {
    Cosine,
    Euclidean,
    Manhattan,
    Jaccard,
    Angular,
}

impl SimilarityMetric {
    /// Calculate similarity using the specified metric
    pub fn calculate(&self, a: &[f64], b: &[f64]) -> f64 {
        match self {
            SimilarityMetric::Cosine => cosine_similarity(a, b),
            SimilarityMetric::Euclidean => {
                // Convert distance to similarity (higher is more similar)
                let distance = euclidean_distance(a, b);
                1.0 / (1.0 + distance)
            }
            SimilarityMetric::Manhattan => {
                // Convert distance to similarity
                let distance = manhattan_distance(a, b);
                1.0 / (1.0 + distance)
            }
            SimilarityMetric::Jaccard => jaccard_similarity(a, b),
            SimilarityMetric::Angular => {
                // Convert angular distance to similarity
                let distance = angular_distance(a, b);
                1.0 - (distance / std::f64::consts::PI)
            }
        }
    }

    /// Get the name of the metric
    pub fn name(&self) -> &'static str {
        match self {
            SimilarityMetric::Cosine => "cosine",
            SimilarityMetric::Euclidean => "euclidean",
            SimilarityMetric::Manhattan => "manhattan",
            SimilarityMetric::Jaccard => "jaccard",
            SimilarityMetric::Angular => "angular",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cosine_similarity() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![1.0, 0.0, 0.0];
        let c = vec![0.0, 1.0, 0.0];
        
        assert!((cosine_similarity(&a, &b) - 1.0).abs() < f64::EPSILON);
        assert!((cosine_similarity(&a, &c) - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_euclidean_distance() {
        let a = vec![0.0, 0.0, 0.0];
        let b = vec![1.0, 0.0, 0.0];
        let c = vec![1.0, 1.0, 0.0];
        
        assert!((euclidean_distance(&a, &b) - 1.0).abs() < f64::EPSILON);
        assert!((euclidean_distance(&a, &c) - 2.0_f64.sqrt()).abs() < f64::EPSILON);
    }

    #[test]
    fn test_manhattan_distance() {
        let a = vec![0.0, 0.0, 0.0];
        let b = vec![1.0, 0.0, 0.0];
        let c = vec![1.0, 1.0, 0.0];
        
        assert!((manhattan_distance(&a, &b) - 1.0).abs() < f64::EPSILON);
        assert!((manhattan_distance(&a, &c) - 2.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_jaccard_similarity() {
        let a = vec![1.0, 0.0, 1.0, 0.0];
        let b = vec![1.0, 1.0, 0.0, 0.0];
        
        // Intersection: 1 element (first position)
        // Union: 3 elements (first, second, third positions)
        let expected = 1.0 / 3.0;
        assert!((jaccard_similarity(&a, &b) - expected).abs() < f64::EPSILON);
    }

    #[test]
    fn test_vector_normalization() {
        let mut vector = vec![3.0, 4.0, 0.0];
        normalize_vector(&mut vector);
        
        let magnitude = vector_magnitude(&vector);
        assert!((magnitude - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_similarity_metrics_enum() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![1.0, 0.0, 0.0];
        
        let cosine_sim = SimilarityMetric::Cosine.calculate(&a, &b);
        assert!((cosine_sim - 1.0).abs() < f64::EPSILON);
        
        assert_eq!(SimilarityMetric::Cosine.name(), "cosine");
    }

    #[test]
    fn test_find_k_most_similar() {
        let query = vec![1.0, 0.0, 0.0];
        let vectors = vec![
            (vec![1.0, 0.0, 0.0], 0), // Perfect match
            (vec![0.5, 0.5, 0.0], 1), // Partial match
            (vec![0.0, 1.0, 0.0], 2), // Orthogonal
        ];
        
        let results = find_k_most_similar(&query, &vectors, 2, cosine_similarity);
        
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].0, 0); // Perfect match should be first
        assert!(results[0].1 > results[1].1); // First should have higher similarity
    }
}
