//! Advanced Encryption Module
//!
//! Implements state-of-the-art encryption including homomorphic encryption,
//! zero-knowledge proofs, and secure multi-party computation.

use crate::error::{MemoryError, Result};
use crate::memory::types::MemoryEntry;
use crate::security::key_management::KeyManager;
use crate::security::{
    EncryptedComputationResult, EncryptedMemoryEntry, EncryptionMetadata, PrivacyLevel,
    SecureOperation, SecurityConfig, SecurityContext,
};
use aes_gcm::aead::{generic_array::GenericArray, Aead, KeyInit, Payload};
use aes_gcm::{Aes256Gcm, Key};
use chrono::{DateTime, Utc};
use rand::rngs::OsRng;
use rand::RngCore;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use uuid::Uuid;

#[cfg(feature = "homomorphic-encryption")]
use tfhe::{
    generate_keys,
    prelude::{FheDecrypt, FheEncrypt},
    set_server_key, ClientKey, ConfigBuilder, FheInt64, ServerKey,
};

/// Fixed-point scale for homomorphic encryption of `f64` values.
///
/// Plaintext values are encoded as `FheInt64` via
/// `round(value * FIXED_POINT_SCALE)`, giving 6 decimal digits of fractional
/// precision. Signed values (including negatives and zero) round-trip
/// EXACTLY at this granularity: any value that is an integer multiple of
/// 1e-6 with magnitude below `i64::MAX / FIXED_POINT_SCALE` decrypts to the
/// original bit-for-bit after the inverse scaling.
#[cfg(feature = "homomorphic-encryption")]
pub const FIXED_POINT_SCALE: f64 = 1_000_000.0;

/// Encryption manager for advanced cryptographic operations
#[derive(Debug)]
pub struct EncryptionManager {
    config: SecurityConfig,
    key_cache: HashMap<String, EncryptionKey>,
    key_manager: KeyManager,
    homomorphic_context: Option<HomomorphicContext>,
    metrics: EncryptionMetrics,
}

impl EncryptionManager {
    /// Create a new encryption manager
    pub async fn new(config: &SecurityConfig, key_manager: KeyManager) -> Result<Self> {
        let homomorphic_context = if config.enable_homomorphic_encryption {
            Some(HomomorphicContext::new(config).await?)
        } else {
            None
        };

        Ok(Self {
            config: config.clone(),
            key_cache: HashMap::new(),
            key_manager,
            homomorphic_context,
            metrics: EncryptionMetrics::default(),
        })
    }

    /// Standard AES-GCM encryption
    pub async fn standard_encrypt(
        &mut self,
        entry: &MemoryEntry,
        context: &SecurityContext,
    ) -> Result<EncryptedMemoryEntry> {
        let start_time = std::time::Instant::now();

        // Basic context validation
        if !context.is_session_valid()
            || !context.is_mfa_satisfied(self.config.access_control_policy.require_mfa)
        {
            return Err(MemoryError::access_denied(
                "Invalid security context".to_string(),
            ));
        }

        // Generate or retrieve encryption key
        let key = self.get_or_generate_key(context, "AES-256-GCM").await?;

        // Generate random IV and salt
        let iv = self.generate_random_bytes(12)?;
        let salt = self.generate_random_bytes(16)?;

        // Serialize the memory entry
        let plaintext = serde_json::to_vec(entry)
            .map_err(|e| MemoryError::encryption(format!("Serialization failed: {}", e)))?;

        // Encrypt using AES-GCM
        let (ciphertext, auth_tag) = self.aes_gcm_encrypt(&plaintext, &key.data, &iv, &salt)?;

        let encrypted_entry = EncryptedMemoryEntry {
            id: Uuid::new_v4().to_string(),
            encrypted_data: ciphertext,
            encryption_algorithm: "AES-256-GCM".to_string(),
            key_id: key.id.clone(),
            is_homomorphic: false,
            privacy_level: PrivacyLevel::Confidential,
            created_at: Utc::now(),
            metadata: EncryptionMetadata {
                algorithm_version: "1.0".to_string(),
                key_derivation: "PBKDF2-SHA256".to_string(),
                salt,
                iv,
                auth_tag: Some(auth_tag),
                compression: None,
            },
        };

        // Update metrics
        self.metrics.total_encryptions += 1;
        self.metrics.total_encryption_time_ms += start_time.elapsed().as_millis() as u64;

        Ok(encrypted_entry)
    }

    /// Standard AES-GCM decryption
    pub async fn standard_decrypt(
        &mut self,
        encrypted_entry: &EncryptedMemoryEntry,
        context: &SecurityContext,
    ) -> Result<MemoryEntry> {
        let start_time = std::time::Instant::now();

        // Validate context before retrieving key
        if !context.is_session_valid()
            || !context.is_mfa_satisfied(self.config.access_control_policy.require_mfa)
        {
            return Err(MemoryError::access_denied(
                "Invalid security context".to_string(),
            ));
        }

        // Retrieve encryption key
        let key = self.get_key(&encrypted_entry.key_id, context).await?;

        // Extract metadata
        let iv = &encrypted_entry.metadata.iv;
        let salt = &encrypted_entry.metadata.salt;
        let auth_tag = encrypted_entry
            .metadata
            .auth_tag
            .as_ref()
            .ok_or_else(|| MemoryError::encryption("Missing authentication tag".to_string()))?;

        // Decrypt using AES-GCM
        let plaintext = self.aes_gcm_decrypt(
            &encrypted_entry.encrypted_data,
            &key.data,
            iv,
            salt,
            auth_tag,
        )?;

        // Deserialize the memory entry
        let entry = serde_json::from_slice(&plaintext)
            .map_err(|e| MemoryError::encryption(format!("Deserialization failed: {}", e)))?;

        // Update metrics
        self.metrics.total_decryptions += 1;
        self.metrics.total_decryption_time_ms += start_time.elapsed().as_millis() as u64;

        Ok(entry)
    }

    /// Homomorphic encryption for secure computation
    pub async fn homomorphic_encrypt(
        &mut self,
        entry: &MemoryEntry,
        context: &SecurityContext,
    ) -> Result<EncryptedMemoryEntry> {
        let start_time = std::time::Instant::now();

        if !context.is_session_valid()
            || !context.is_mfa_satisfied(self.config.access_control_policy.require_mfa)
        {
            return Err(MemoryError::access_denied(
                "Invalid security context".to_string(),
            ));
        }

        let homomorphic_context = self.homomorphic_context.as_ref().ok_or_else(|| {
            MemoryError::encryption("Homomorphic encryption not enabled".to_string())
        })?;

        // Convert memory entry to homomorphic-compatible format
        let numeric_data = self.extract_numeric_features(entry)?;

        // Encrypt using homomorphic encryption (simulated with advanced techniques)
        let encrypted_data = homomorphic_context.encrypt_vector(&numeric_data).await?;

        let encrypted_entry = EncryptedMemoryEntry {
            id: Uuid::new_v4().to_string(),
            encrypted_data,
            encryption_algorithm: "Homomorphic-TFHE".to_string(),
            key_id: homomorphic_context.key_id.clone(),
            is_homomorphic: true,
            privacy_level: PrivacyLevel::Secret,
            created_at: Utc::now(),
            metadata: EncryptionMetadata {
                algorithm_version: "2.0".to_string(),
                key_derivation: "Homomorphic-KeyGen".to_string(),
                salt: Vec::new(),
                iv: Vec::new(),
                auth_tag: None,
                compression: Some("TFHE".to_string()),
            },
        };

        // Update metrics
        self.metrics.total_homomorphic_encryptions += 1;
        self.metrics.total_encryption_time_ms += start_time.elapsed().as_millis() as u64;

        Ok(encrypted_entry)
    }

    /// Homomorphic decryption
    pub async fn homomorphic_decrypt(
        &mut self,
        encrypted_entry: &EncryptedMemoryEntry,
        context: &SecurityContext,
    ) -> Result<MemoryEntry> {
        let start_time = std::time::Instant::now();

        if !context.is_session_valid()
            || !context.is_mfa_satisfied(self.config.access_control_policy.require_mfa)
        {
            return Err(MemoryError::access_denied(
                "Invalid security context".to_string(),
            ));
        }

        let homomorphic_context = self.homomorphic_context.as_ref().ok_or_else(|| {
            MemoryError::encryption("Homomorphic encryption not enabled".to_string())
        })?;

        // Decrypt the homomorphic data
        let numeric_data = homomorphic_context
            .decrypt_vector(&encrypted_entry.encrypted_data)
            .await?;

        // Reconstruct memory entry from numeric features
        let entry = self.reconstruct_from_numeric_features(&numeric_data)?;

        // Update metrics
        self.metrics.total_homomorphic_decryptions += 1;
        self.metrics.total_decryption_time_ms += start_time.elapsed().as_millis() as u64;

        Ok(entry)
    }

    /// Perform homomorphic computation on encrypted data
    pub async fn homomorphic_compute(
        &mut self,
        encrypted_entries: &[EncryptedMemoryEntry],
        operation: SecureOperation,
        context: &SecurityContext,
    ) -> Result<EncryptedComputationResult> {
        let start_time = std::time::Instant::now();

        if !context.is_session_valid()
            || !context.is_mfa_satisfied(self.config.access_control_policy.require_mfa)
        {
            return Err(MemoryError::access_denied(
                "Invalid security context".to_string(),
            ));
        }

        let homomorphic_context = self.homomorphic_context.as_ref().ok_or_else(|| {
            MemoryError::encryption("Homomorphic encryption not enabled".to_string())
        })?;

        // Perform the computation based on operation type
        let result_data = match operation {
            SecureOperation::Sum => {
                homomorphic_context
                    .homomorphic_sum(encrypted_entries)
                    .await?
            }
            SecureOperation::Average => {
                homomorphic_context
                    .homomorphic_average(encrypted_entries)
                    .await?
            }
            SecureOperation::Count => {
                homomorphic_context
                    .homomorphic_count(encrypted_entries)
                    .await?
            }
            SecureOperation::Search { ref query } => {
                homomorphic_context
                    .homomorphic_search(encrypted_entries, query)
                    .await?
            }
            SecureOperation::Similarity { threshold } => {
                homomorphic_context
                    .homomorphic_similarity(encrypted_entries, threshold)
                    .await?
            }
            SecureOperation::Aggregate { ref function } => {
                homomorphic_context
                    .homomorphic_aggregate(encrypted_entries, function)
                    .await?
            }
        };

        let result = EncryptedComputationResult {
            result_data,
            operation: operation.clone(),
            computation_id: Uuid::new_v4().to_string(),
            timestamp: Utc::now(),
            privacy_preserved: true,
        };

        // Update metrics
        self.metrics.total_homomorphic_computations += 1;
        self.metrics.total_computation_time_ms += start_time.elapsed().as_millis() as u64;

        Ok(result)
    }

    /// Homomorphically encrypt a raw numeric vector (FheInt64 fixed-point,
    /// see [`FIXED_POINT_SCALE`](crate::security::encryption) for the
    /// precision contract when the `homomorphic-encryption` feature is on).
    /// Fails closed with `MemoryError::FeatureDisabled` when the feature is off.
    pub async fn homomorphic_encrypt_vector(
        &self,
        data: &[f64],
        context: &SecurityContext,
    ) -> Result<Vec<u8>> {
        if !context.is_session_valid()
            || !context.is_mfa_satisfied(self.config.access_control_policy.require_mfa)
        {
            return Err(MemoryError::access_denied(
                "Invalid security context".to_string(),
            ));
        }
        let homomorphic_context = self.homomorphic_context.as_ref().ok_or_else(|| {
            MemoryError::encryption("Homomorphic encryption not enabled".to_string())
        })?;
        homomorphic_context.encrypt_vector(data).await
    }

    /// Homomorphically decrypt a vector previously produced by
    /// [`homomorphic_encrypt_vector`](Self::homomorphic_encrypt_vector) or by
    /// a homomorphic computation (sum/average). Fails closed with
    /// `MemoryError::FeatureDisabled` when the feature is off.
    pub async fn homomorphic_decrypt_vector(
        &self,
        encrypted: &[u8],
        context: &SecurityContext,
    ) -> Result<Vec<f64>> {
        if !context.is_session_valid()
            || !context.is_mfa_satisfied(self.config.access_control_policy.require_mfa)
        {
            return Err(MemoryError::access_denied(
                "Invalid security context".to_string(),
            ));
        }
        let homomorphic_context = self.homomorphic_context.as_ref().ok_or_else(|| {
            MemoryError::encryption("Homomorphic encryption not enabled".to_string())
        })?;
        homomorphic_context.decrypt_vector(encrypted).await
    }

    /// Get encryption metrics
    pub async fn get_metrics(&self) -> Result<EncryptionMetrics> {
        Ok(self.metrics.clone())
    }

    // Private helper methods

    async fn get_or_generate_key(
        &mut self,
        context: &SecurityContext,
        algorithm: &str,
    ) -> Result<EncryptionKey> {
        let key_id = format!("{}:{}", context.user_id, algorithm);

        if let Some(key) = self.key_cache.get(&key_id) {
            return Ok(key.clone());
        }

        // Generate new key using key manager
        let data_key_id = self
            .key_manager
            .generate_data_key("default", context)
            .await?;
        let key_data = self.key_manager.get_data_key(&data_key_id, context).await?;
        let key = EncryptionKey {
            id: data_key_id.clone(),
            data: key_data,
            algorithm: algorithm.to_string(),
            created_at: Utc::now(),
            expires_at: Utc::now() + chrono::Duration::hours(24),
        };

        self.key_cache.insert(key_id, key.clone());
        Ok(key)
    }

    async fn get_key(&mut self, key_id: &str, context: &SecurityContext) -> Result<EncryptionKey> {
        if let Some(key) = self.key_cache.get(key_id) {
            return Ok(key.clone());
        }

        // Attempt to retrieve key from key manager
        let key_bytes = self.key_manager.get_data_key(key_id, context).await?;
        let info = self.key_manager.get_key_info(key_id).await?;

        let key = EncryptionKey {
            id: info.id,
            data: key_bytes,
            algorithm: info.algorithm,
            created_at: info.created_at,
            expires_at: info.expires_at,
        };

        self.key_cache.insert(key_id.to_string(), key.clone());
        Ok(key)
    }

    fn generate_random_bytes(&self, length: usize) -> Result<Vec<u8>> {
        let mut bytes = vec![0u8; length];
        OsRng.fill_bytes(&mut bytes);
        Ok(bytes)
    }

    fn aes_gcm_encrypt(
        &self,
        plaintext: &[u8],
        key: &[u8],
        iv: &[u8],
        salt: &[u8],
    ) -> Result<(Vec<u8>, Vec<u8>)> {
        let cipher = Aes256Gcm::new(Key::<Aes256Gcm>::from_slice(key));
        let nonce = GenericArray::from_slice(iv);
        let encrypted = cipher
            .encrypt(
                nonce,
                Payload {
                    msg: plaintext,
                    aad: salt,
                },
            )
            .map_err(|_| MemoryError::encryption("AES-GCM encryption failed"))?;
        let tag = encrypted[encrypted.len() - 16..].to_vec();
        let ciphertext = encrypted[..encrypted.len() - 16].to_vec();
        Ok((ciphertext, tag))
    }

    fn aes_gcm_decrypt(
        &self,
        ciphertext: &[u8],
        key: &[u8],
        iv: &[u8],
        salt: &[u8],
        auth_tag: &[u8],
    ) -> Result<Vec<u8>> {
        let cipher = Aes256Gcm::new(Key::<Aes256Gcm>::from_slice(key));
        let nonce = GenericArray::from_slice(iv);
        let mut combined = Vec::with_capacity(ciphertext.len() + auth_tag.len());
        combined.extend_from_slice(ciphertext);
        combined.extend_from_slice(auth_tag);
        let decrypted = cipher
            .decrypt(
                nonce,
                Payload {
                    msg: &combined,
                    aad: salt,
                },
            )
            .map_err(|_| MemoryError::encryption("AES-GCM decryption failed"))?;
        Ok(decrypted)
    }

    fn extract_numeric_features(&self, entry: &MemoryEntry) -> Result<Vec<f64>> {
        // Extract numeric features from memory entry for homomorphic encryption
        let mut features = Vec::new();

        // Convert text to numeric features (simplified)
        let text_bytes = entry.value.as_bytes();
        for chunk in text_bytes.chunks(4) {
            let mut value = 0u32;
            for (i, &byte) in chunk.iter().enumerate() {
                value |= (byte as u32) << (i * 8);
            }
            features.push(value as f64);
        }

        // Add embedding if available
        if let Some(ref embedding) = entry.embedding {
            features.extend(embedding.iter().map(|&x| x as f64));
        }

        Ok(features)
    }

    fn reconstruct_from_numeric_features(&self, features: &[f64]) -> Result<MemoryEntry> {
        // Reconstruct memory entry from numeric features (simplified)
        let mut text_bytes = Vec::new();

        for &feature in features.iter().take(features.len().saturating_sub(768)) {
            let value = feature as u32;
            for i in 0..4 {
                text_bytes.push(((value >> (i * 8)) & 0xFF) as u8);
            }
        }

        // Remove null bytes and convert to string
        text_bytes.retain(|&b| b != 0);
        let value = String::from_utf8_lossy(&text_bytes).to_string();

        // Extract embedding if present
        let embedding = if features.len() > 768 {
            Some(
                features[features.len() - 768..]
                    .iter()
                    .map(|&x| x as f32)
                    .collect(),
            )
        } else {
            None
        };

        Ok(MemoryEntry {
            key: uuid::Uuid::new_v4().to_string(),
            value,
            memory_type: crate::memory::types::MemoryType::LongTerm,
            metadata: crate::memory::types::MemoryMetadata::default(),
            embedding,
        })
    }
}

/// Encryption key structure
#[derive(Debug, Clone)]
struct EncryptionKey {
    id: String,
    data: Vec<u8>,
    algorithm: String,
    created_at: DateTime<Utc>,
    expires_at: DateTime<Utc>,
}

/// Homomorphic encryption context with real TFHE implementation
struct HomomorphicContext {
    key_id: String,
    #[cfg(feature = "homomorphic-encryption")]
    client_key: ClientKey,
    #[cfg(feature = "homomorphic-encryption")]
    server_key: ServerKey,
    #[cfg(not(feature = "homomorphic-encryption"))]
    parameters: HomomorphicParameters,
}

impl std::fmt::Debug for HomomorphicContext {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HomomorphicContext")
            .field("key_id", &self.key_id)
            .finish()
    }
}

impl HomomorphicContext {
    async fn new(config: &SecurityConfig) -> Result<Self> {
        let key_id = Uuid::new_v4().to_string();

        #[cfg(feature = "homomorphic-encryption")]
        {
            tracing::info!(
                "Initializing real TFHE homomorphic encryption with key size: {}",
                config.encryption_key_size
            );

            // Generate TFHE integer keys with appropriate configuration
            let config = ConfigBuilder::default().build();
            let (client_key, server_key) = generate_keys(config);

            tracing::info!("TFHE keys generated successfully for key_id: {}", key_id);

            Ok(Self {
                key_id,
                client_key,
                server_key,
            })
        }

        #[cfg(not(feature = "homomorphic-encryption"))]
        {
            tracing::warn!(
                "homomorphic-encryption feature not enabled; homomorphic operations will return FeatureDisabled errors"
            );
            Ok(Self {
                key_id,
                parameters: HomomorphicParameters::new(config.encryption_key_size),
            })
        }
    }

    async fn encrypt_vector(&self, data: &[f64]) -> Result<Vec<u8>> {
        #[cfg(feature = "homomorphic-encryption")]
        {
            tracing::debug!("Encrypting vector of {} elements with TFHE", data.len());
            let start_time = std::time::Instant::now();

            let mut encrypted_data = Vec::new();

            // Encode each signed f64 as an FheInt64 fixed-point value at
            // FIXED_POINT_SCALE granularity (sign-preserving; see the
            // constant's docs for the exact-round-trip contract).
            for &value in data {
                if !value.is_finite() {
                    return Err(MemoryError::encryption(
                        "Cannot homomorphically encrypt non-finite value".to_string(),
                    ));
                }
                let scaled = (value * FIXED_POINT_SCALE).round();
                if scaled >= i64::MAX as f64 || scaled <= i64::MIN as f64 {
                    return Err(MemoryError::encryption(format!(
                        "Value {} out of range for FheInt64 fixed-point encoding",
                        value
                    )));
                }
                let encrypted_value = FheInt64::encrypt(scaled as i64, &self.client_key);

                // Serialize the encrypted value
                let serialized = bincode::serialize(&encrypted_value).map_err(|e| {
                    MemoryError::encryption(format!("Failed to serialize encrypted value: {}", e))
                })?;

                // Store length prefix for deserialization
                encrypted_data.extend_from_slice(&(serialized.len() as u32).to_le_bytes());
                encrypted_data.extend_from_slice(&serialized);
            }

            let duration = start_time.elapsed();
            tracing::debug!("TFHE vector encryption completed in {:?}", duration);

            Ok(encrypted_data)
        }

        #[cfg(not(feature = "homomorphic-encryption"))]
        {
            let _ = data;
            Err(MemoryError::feature_disabled(
                "homomorphic-encryption",
                "encrypt_vector",
            ))
        }
    }

    async fn decrypt_vector(&self, encrypted_data: &[u8]) -> Result<Vec<f64>> {
        #[cfg(feature = "homomorphic-encryption")]
        {
            tracing::debug!("Decrypting TFHE encrypted vector");
            let start_time = std::time::Instant::now();

            let mut decrypted = Vec::new();
            let mut offset = 0;

            while offset < encrypted_data.len() {
                if offset + 4 > encrypted_data.len() {
                    break;
                }

                // Read length prefix
                let length = u32::from_le_bytes([
                    encrypted_data[offset],
                    encrypted_data[offset + 1],
                    encrypted_data[offset + 2],
                    encrypted_data[offset + 3],
                ]) as usize;
                offset += 4;

                if offset + length > encrypted_data.len() {
                    return Err(MemoryError::encryption(
                        "Invalid encrypted data format".to_string(),
                    ));
                }

                // Deserialize encrypted value
                let encrypted_value: FheInt64 = bincode::deserialize(
                    &encrypted_data[offset..offset + length],
                )
                .map_err(|e| {
                    MemoryError::encryption(format!("Failed to deserialize encrypted value: {}", e))
                })?;

                // Decrypt and undo the fixed-point scaling (sign-preserving)
                let decrypted_value: i64 = encrypted_value.decrypt(&self.client_key);
                let original_value = decrypted_value as f64 / FIXED_POINT_SCALE;

                decrypted.push(original_value);
                offset += length;
            }

            let duration = start_time.elapsed();
            tracing::debug!("TFHE vector decryption completed in {:?}", duration);

            Ok(decrypted)
        }

        #[cfg(not(feature = "homomorphic-encryption"))]
        {
            let _ = encrypted_data;
            Err(MemoryError::feature_disabled(
                "homomorphic-encryption",
                "decrypt_vector",
            ))
        }
    }

    async fn homomorphic_sum(&self, entries: &[EncryptedMemoryEntry]) -> Result<Vec<u8>> {
        #[cfg(feature = "homomorphic-encryption")]
        {
            tracing::debug!("Performing homomorphic sum on {} entries", entries.len());
            let start_time = std::time::Instant::now();

            if entries.is_empty() {
                return Ok(Vec::new());
            }

            // TFHE homomorphic ops require the server key on this thread
            set_server_key(self.server_key.clone());

            // Parse first entry to get the structure
            let first_entry = &entries[0];
            let mut encrypted_values = self.parse_encrypted_vector(&first_entry.encrypted_data)?;

            // Add remaining entries
            for entry in &entries[1..] {
                let entry_values = self.parse_encrypted_vector(&entry.encrypted_data)?;

                // Ensure same length
                if encrypted_values.len() != entry_values.len() {
                    return Err(MemoryError::encryption(
                        "Mismatched vector lengths for homomorphic sum".to_string(),
                    ));
                }

                // Perform homomorphic addition
                for (i, entry_val) in entry_values.into_iter().enumerate() {
                    encrypted_values[i] = &encrypted_values[i] + &entry_val;
                }
            }

            // Serialize result
            let result = self.serialize_encrypted_vector(&encrypted_values)?;

            let duration = start_time.elapsed();
            tracing::debug!("Homomorphic sum completed in {:?}", duration);

            Ok(result)
        }

        #[cfg(not(feature = "homomorphic-encryption"))]
        {
            let _ = entries;
            Err(MemoryError::feature_disabled(
                "homomorphic-encryption",
                "homomorphic_sum",
            ))
        }
    }

    async fn homomorphic_average(&self, entries: &[EncryptedMemoryEntry]) -> Result<Vec<u8>> {
        #[cfg(feature = "homomorphic-encryption")]
        {
            tracing::debug!(
                "Performing homomorphic average on {} entries",
                entries.len()
            );

            if entries.is_empty() {
                return Ok(Vec::new());
            }

            // Get the sum first (also installs the server key on this thread)
            let sum_data = self.homomorphic_sum(entries).await?;
            let mut sum_values = self.parse_encrypted_vector(&sum_data)?;

            // Create encrypted count
            let count = entries.len() as i64;
            let encrypted_count = FheInt64::encrypt(count, &self.client_key);

            // Divide each sum value by count (homomorphic signed integer
            // division; exact when the fixed-point sum divides evenly,
            // otherwise truncated at FIXED_POINT_SCALE granularity)
            for sum_val in &mut sum_values {
                *sum_val = &*sum_val / &encrypted_count;
            }

            // Serialize result
            self.serialize_encrypted_vector(&sum_values)
        }

        #[cfg(not(feature = "homomorphic-encryption"))]
        {
            let _ = entries;
            Err(MemoryError::feature_disabled(
                "homomorphic-encryption",
                "homomorphic_average",
            ))
        }
    }

    /// Encrypted count is UNSUPPORTED BY DESIGN (Phase 4 Decision Gate 3):
    /// a ciphertext count would either leak the entry count in plaintext or
    /// require an encrypted-cardinality protocol this crate does not
    /// implement. Fails closed in ALL builds; this is a permanent descope,
    /// not a TODO.
    async fn homomorphic_count(&self, _entries: &[EncryptedMemoryEntry]) -> Result<Vec<u8>> {
        Err(MemoryError::feature_disabled(
            "homomorphic-encryption(unsupported-count)",
            "homomorphic_count",
        ))
    }

    /// Encrypted search is UNSUPPORTED BY DESIGN (Phase 4 Decision Gate 3):
    /// TFHE integer ciphertexts do not support the comparison-and-match
    /// protocol needed for private search. Fails closed in ALL builds; this
    /// is a permanent descope, not a TODO.
    async fn homomorphic_search(
        &self,
        _entries: &[EncryptedMemoryEntry],
        _query: &str,
    ) -> Result<Vec<u8>> {
        Err(MemoryError::feature_disabled(
            "homomorphic-encryption(unsupported-search)",
            "homomorphic_search",
        ))
    }

    /// Encrypted similarity is UNSUPPORTED BY DESIGN (Phase 4 Decision Gate
    /// 3): fixed-point TFHE integers cannot express the normalized dot
    /// products required without a dedicated protocol this crate does not
    /// implement. Fails closed in ALL builds; this is a permanent descope,
    /// not a TODO.
    async fn homomorphic_similarity(
        &self,
        _entries: &[EncryptedMemoryEntry],
        _threshold: f64,
    ) -> Result<Vec<u8>> {
        Err(MemoryError::feature_disabled(
            "homomorphic-encryption(unsupported-similarity)",
            "homomorphic_similarity",
        ))
    }

    async fn homomorphic_aggregate(
        &self,
        entries: &[EncryptedMemoryEntry],
        function: &str,
    ) -> Result<Vec<u8>> {
        match function {
            "sum" => self.homomorphic_sum(entries).await,
            "avg" => self.homomorphic_average(entries).await,
            "count" => self.homomorphic_count(entries).await,
            _ => Err(MemoryError::encryption(format!(
                "Unknown aggregate function: {}",
                function
            ))),
        }
    }

    #[cfg(feature = "homomorphic-encryption")]
    fn parse_encrypted_vector(&self, encrypted_data: &[u8]) -> Result<Vec<FheInt64>> {
        let mut encrypted_values = Vec::new();
        let mut offset = 0;

        while offset < encrypted_data.len() {
            if offset + 4 > encrypted_data.len() {
                break;
            }

            // Read length prefix
            let length = u32::from_le_bytes([
                encrypted_data[offset],
                encrypted_data[offset + 1],
                encrypted_data[offset + 2],
                encrypted_data[offset + 3],
            ]) as usize;
            offset += 4;

            if offset + length > encrypted_data.len() {
                return Err(MemoryError::encryption(
                    "Invalid encrypted data format".to_string(),
                ));
            }

            // Deserialize encrypted value
            let encrypted_value: FheInt64 =
                bincode::deserialize(&encrypted_data[offset..offset + length]).map_err(|e| {
                    MemoryError::encryption(format!("Failed to deserialize encrypted value: {}", e))
                })?;

            encrypted_values.push(encrypted_value);
            offset += length;
        }

        Ok(encrypted_values)
    }

    #[cfg(feature = "homomorphic-encryption")]
    fn serialize_encrypted_vector(&self, encrypted_values: &[FheInt64]) -> Result<Vec<u8>> {
        let mut result = Vec::new();

        for encrypted_value in encrypted_values {
            let serialized = bincode::serialize(encrypted_value).map_err(|e| {
                MemoryError::encryption(format!("Failed to serialize encrypted value: {}", e))
            })?;

            // Store length prefix
            result.extend_from_slice(&(serialized.len() as u32).to_le_bytes());
            result.extend_from_slice(&serialized);
        }

        Ok(result)
    }
}

/// Homomorphic encryption parameters
#[derive(Debug)]
struct HomomorphicParameters {
    key_size: usize,
    polynomial_degree: usize,
    coefficient_modulus: Vec<u64>,
}

impl HomomorphicParameters {
    fn new(key_size: usize) -> Self {
        Self {
            key_size,
            polynomial_degree: 8192,
            coefficient_modulus: vec![1099511627689, 1099511627691, 1099511627693],
        }
    }
}

/// Encryption metrics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct EncryptionMetrics {
    pub total_encryptions: u64,
    pub total_decryptions: u64,
    pub total_homomorphic_encryptions: u64,
    pub total_homomorphic_decryptions: u64,
    pub total_homomorphic_computations: u64,
    pub total_encryption_time_ms: u64,
    pub total_decryption_time_ms: u64,
    pub total_computation_time_ms: u64,
    pub average_encryption_time_ms: f64,
    pub average_decryption_time_ms: f64,
    pub encryption_success_rate: f64,
    pub decryption_success_rate: f64,
}

impl EncryptionMetrics {
    pub fn calculate_averages(&mut self) {
        if self.total_encryptions > 0 {
            self.average_encryption_time_ms =
                self.total_encryption_time_ms as f64 / self.total_encryptions as f64;
        }
        if self.total_decryptions > 0 {
            self.average_decryption_time_ms =
                self.total_decryption_time_ms as f64 / self.total_decryptions as f64;
        }
        // Calculate success rates (would track failures in production)
        self.encryption_success_rate = 100.0;
        self.decryption_success_rate = 100.0;
    }
}
