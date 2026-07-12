//! Tests asserting that disabled crypto features error instead of faking.

use synaptic::error::MemoryError;

#[test]
fn feature_disabled_error_names_feature_and_operation() {
    let err = MemoryError::feature_disabled("homomorphic-encryption", "encrypt_vector");
    let msg = err.to_string();
    assert!(msg.contains("homomorphic-encryption"));
    assert!(msg.contains("encrypt_vector"));
}
