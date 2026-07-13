//! Advanced Key Management Module
//!
//! Implements enterprise-grade key management with automatic rotation,
//! secure storage, and cryptographic key lifecycle management.

use crate::error::{MemoryError, Result};
use crate::security::SecurityConfig;
use aes_gcm::aead::{generic_array::GenericArray, Aead, KeyInit, Payload};
use aes_gcm::{Aes256Gcm, Key};
use chrono::{DateTime, Utc};
use rand::rngs::OsRng;
use rand::RngCore;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt;
use uuid::Uuid;
use zeroize::Zeroizing;

/// Key management system
#[derive(Clone)]
pub struct KeyManager {
    config: SecurityConfig,
    master_keys: HashMap<String, MasterKey>,
    data_keys: HashMap<String, DataKey>,
    key_rotation_schedule: Vec<KeyRotationTask>,
    metrics: KeyManagementMetrics,
}

impl fmt::Debug for KeyManager {
    /// Manual `Debug` impl: never prints key bytes. Master/data key maps are
    /// redacted (key IDs and counts only); the fields already implement
    /// redacted `Debug` themselves, but we keep this explicit so a future
    /// field addition doesn't silently start leaking key material.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("KeyManager")
            .field("config", &self.config)
            .field("master_key_count", &self.master_keys.len())
            .field("data_key_count", &self.data_keys.len())
            .field("key_rotation_schedule", &self.key_rotation_schedule)
            .field("metrics", &self.metrics)
            .finish()
    }
}

impl KeyManager {
    /// Create a new key manager
    pub async fn new(config: &SecurityConfig) -> Result<Self> {
        let mut manager = Self {
            config: config.clone(),
            master_keys: HashMap::new(),
            data_keys: HashMap::new(),
            key_rotation_schedule: Vec::new(),
            metrics: KeyManagementMetrics::default(),
        };

        // Initialize with a master key
        manager.generate_master_key("default").await?;

        Ok(manager)
    }

    /// Generate a new master key
    pub async fn generate_master_key(&mut self, key_id: &str) -> Result<String> {
        let master_key = MasterKey {
            id: key_id.to_string(),
            key_data: Zeroizing::new(
                self.generate_secure_key(self.config.encryption_key_size / 8)?,
            ),
            algorithm: "AES-256".to_string(),
            created_at: Utc::now(),
            expires_at: Utc::now() + chrono::Duration::days(365), // 1 year
            status: KeyStatus::Active,
            version: 1,
            usage_count: 0,
        };

        self.master_keys.insert(key_id.to_string(), master_key);

        // Schedule rotation
        self.schedule_key_rotation(key_id).await?;

        self.metrics.total_master_keys_generated += 1;
        Ok(key_id.to_string())
    }

    /// Generate a new data encryption key
    pub async fn generate_data_key(
        &mut self,
        master_key_id: &str,
        context: &crate::security::SecurityContext,
    ) -> Result<String> {
        // Validate security context
        if !context.is_session_valid() {
            return Err(MemoryError::access_denied(
                "Invalid session for key generation".to_string(),
            ));
        }

        // Check master key status first
        {
            let master_key = self
                .master_keys
                .get(master_key_id)
                .ok_or_else(|| MemoryError::key_management("Master key not found".to_string()))?;

            if master_key.status != KeyStatus::Active {
                return Err(MemoryError::key_management(
                    "Master key is not active".to_string(),
                ));
            }
        }

        let data_key_id = Uuid::new_v4().to_string();
        let plaintext_key = Zeroizing::new(self.generate_secure_key(32)?); // 256-bit key

        // Encrypt the data key with the master key
        let encrypted_key = {
            let master_key = self.master_keys.get(master_key_id).ok_or_else(|| {
                MemoryError::key_management(
                    "Master key not found for data key creation".to_string(),
                )
            })?;
            self.encrypt_with_master_key(&plaintext_key, master_key)?
        };

        let data_key = DataKey {
            id: data_key_id.clone(),
            master_key_id: master_key_id.to_string(),
            encrypted_key,
            plaintext_key: Some(plaintext_key),
            algorithm: "AES-256-GCM".to_string(),
            created_at: Utc::now(),
            expires_at: Utc::now()
                + chrono::Duration::hours(self.config.key_rotation_interval_hours as i64),
            status: KeyStatus::Active,
            usage_count: 0,
        };

        self.data_keys.insert(data_key_id.clone(), data_key);

        // Update master key usage count
        if let Some(master_key) = self.master_keys.get_mut(master_key_id) {
            master_key.usage_count += 1;
        }

        self.metrics.total_data_keys_generated += 1;
        Ok(data_key_id)
    }

    /// Get a data key for encryption/decryption
    pub async fn get_data_key(
        &mut self,
        key_id: &str,
        context: &crate::security::SecurityContext,
    ) -> Result<Vec<u8>> {
        // Validate security context
        if !context.is_session_valid() {
            return Err(MemoryError::access_denied(
                "Invalid session for key retrieval".to_string(),
            ));
        }

        // First check if we need to decrypt the key
        let needs_decryption = {
            let data_key = self
                .data_keys
                .get(key_id)
                .ok_or_else(|| MemoryError::key_management("Data key not found".to_string()))?;

            if data_key.status != KeyStatus::Active {
                return Err(MemoryError::key_management(
                    "Data key is not active".to_string(),
                ));
            }

            if Utc::now() > data_key.expires_at {
                return Err(MemoryError::key_management(
                    "Data key has expired".to_string(),
                ));
            }

            data_key.plaintext_key.is_none()
        };

        // If plaintext key is not available, decrypt it
        if needs_decryption {
            let (master_key_id, encrypted_key) = {
                let data_key = self.data_keys.get(key_id).ok_or_else(|| {
                    MemoryError::key_management("Data key not found for decryption".to_string())
                })?;
                (
                    data_key.master_key_id.clone(),
                    data_key.encrypted_key.clone(),
                )
            };

            let master_key = self
                .master_keys
                .get(&master_key_id)
                .ok_or_else(|| MemoryError::key_management("Master key not found".to_string()))?;

            let plaintext_key =
                Zeroizing::new(self.decrypt_with_master_key(&encrypted_key, master_key)?);

            // Update the data key with the plaintext
            if let Some(data_key) = self.data_keys.get_mut(key_id) {
                data_key.plaintext_key = Some(plaintext_key);
            }
        }

        // Update usage count and return the key
        let result = {
            let data_key = self.data_keys.get_mut(key_id).ok_or_else(|| {
                MemoryError::key_management("Data key not found for usage update".to_string())
            })?;
            data_key.usage_count += 1;
            data_key
                .plaintext_key
                .as_ref()
                .ok_or_else(|| {
                    MemoryError::key_management("Plaintext key not available".to_string())
                })?
                .to_vec()
        };

        self.metrics.total_key_operations += 1;
        Ok(result)
    }

    /// Rotate a master key
    pub async fn rotate_master_key(&mut self, key_id: &str) -> Result<String> {
        // Get old key info and mark as deprecated
        let (new_version, algorithm) = {
            let old_key = self
                .master_keys
                .get_mut(key_id)
                .ok_or_else(|| MemoryError::key_management("Master key not found".to_string()))?;

            // Mark old key as deprecated
            old_key.status = KeyStatus::Deprecated;
            (old_key.version + 1, old_key.algorithm.clone())
        };

        // Generate new version
        let new_key_id = format!("{}:v{}", key_id, new_version);

        let new_master_key = MasterKey {
            id: new_key_id.clone(),
            key_data: Zeroizing::new(
                self.generate_secure_key(self.config.encryption_key_size / 8)?,
            ),
            algorithm,
            created_at: Utc::now(),
            expires_at: Utc::now() + chrono::Duration::days(365),
            status: KeyStatus::Active,
            version: new_version,
            usage_count: 0,
        };

        self.master_keys.insert(new_key_id.clone(), new_master_key);

        // Re-encrypt all data keys with new master key
        self.re_encrypt_data_keys(key_id, &new_key_id).await?;

        // Schedule next rotation
        self.schedule_key_rotation(&new_key_id).await?;

        self.metrics.total_key_rotations += 1;
        Ok(new_key_id)
    }

    /// Revoke a key
    pub async fn revoke_key(&mut self, key_id: &str) -> Result<()> {
        if let Some(master_key) = self.master_keys.get_mut(key_id) {
            master_key.status = KeyStatus::Revoked;
            self.metrics.total_keys_revoked += 1;
        } else if let Some(data_key) = self.data_keys.get_mut(key_id) {
            data_key.status = KeyStatus::Revoked;
            // Clear plaintext key from memory
            data_key.plaintext_key = None;
            self.metrics.total_keys_revoked += 1;
        } else {
            return Err(MemoryError::key_management("Key not found".to_string()));
        }

        Ok(())
    }

    /// Perform scheduled key rotations
    pub async fn perform_scheduled_rotations(&mut self) -> Result<Vec<String>> {
        let now = Utc::now();
        let mut rotated_keys = Vec::new();

        // Find keys that need rotation
        let keys_to_rotate: Vec<String> = self
            .key_rotation_schedule
            .iter()
            .filter(|task| now >= task.scheduled_time && task.status == RotationStatus::Pending)
            .map(|task| task.key_id.clone())
            .collect();

        for key_id in keys_to_rotate {
            match self.rotate_master_key(&key_id).await {
                Ok(new_key_id) => {
                    rotated_keys.push(new_key_id);
                    // Update rotation task status
                    if let Some(task) = self
                        .key_rotation_schedule
                        .iter_mut()
                        .find(|t| t.key_id == key_id)
                    {
                        task.status = RotationStatus::Completed;
                        task.completed_at = Some(now);
                    }
                }
                Err(e) => {
                    // Log error and mark as failed
                    if let Some(task) = self
                        .key_rotation_schedule
                        .iter_mut()
                        .find(|t| t.key_id == key_id)
                    {
                        task.status = RotationStatus::Failed;
                        task.error_message = Some(format!("Rotation failed: {}", e));
                    }
                }
            }
        }

        Ok(rotated_keys)
    }

    /// Get key management metrics
    pub async fn get_metrics(&self) -> Result<KeyManagementMetrics> {
        let mut metrics = self.metrics.clone();

        // Calculate current statistics
        metrics.active_master_keys = self
            .master_keys
            .values()
            .filter(|k| k.status == KeyStatus::Active)
            .count() as u64;

        metrics.active_data_keys = self
            .data_keys
            .values()
            .filter(|k| k.status == KeyStatus::Active)
            .count() as u64;

        // Count expired master keys
        let expired_master_keys = self
            .master_keys
            .values()
            .filter(|k| k.status == KeyStatus::Expired)
            .count();

        // Count expired data keys
        let expired_data_keys = self
            .data_keys
            .values()
            .filter(|k| k.status == KeyStatus::Expired)
            .count();

        metrics.expired_keys = (expired_master_keys + expired_data_keys) as u64;

        Ok(metrics)
    }

    /// Get key information
    pub async fn get_key_info(&self, key_id: &str) -> Result<KeyInformation> {
        if let Some(master_key) = self.master_keys.get(key_id) {
            Ok(KeyInformation {
                id: master_key.id.clone(),
                key_type: KeyType::Master,
                algorithm: master_key.algorithm.clone(),
                status: master_key.status.clone(),
                created_at: master_key.created_at,
                expires_at: master_key.expires_at,
                usage_count: master_key.usage_count,
                version: Some(master_key.version),
            })
        } else if let Some(data_key) = self.data_keys.get(key_id) {
            Ok(KeyInformation {
                id: data_key.id.clone(),
                key_type: KeyType::Data,
                algorithm: data_key.algorithm.clone(),
                status: data_key.status.clone(),
                created_at: data_key.created_at,
                expires_at: data_key.expires_at,
                usage_count: data_key.usage_count,
                version: None,
            })
        } else {
            Err(MemoryError::key_management("Key not found".to_string()))
        }
    }

    // Private helper methods

    fn generate_secure_key(&self, length: usize) -> Result<Vec<u8>> {
        let mut key = vec![0u8; length];
        OsRng.fill_bytes(&mut key);
        Ok(key)
    }

    fn encrypt_with_master_key(&self, plaintext: &[u8], master_key: &MasterKey) -> Result<Vec<u8>> {
        let cipher = Aes256Gcm::new(Key::<Aes256Gcm>::from_slice(&master_key.key_data));
        let mut iv = [0u8; 12];
        OsRng.fill_bytes(&mut iv);
        let nonce = GenericArray::from_slice(&iv);
        let mut encrypted = cipher
            .encrypt(
                nonce,
                Payload {
                    msg: plaintext,
                    aad: &[],
                },
            )
            .map_err(|_| MemoryError::key_management("Master key encryption failed"))?;
        let mut result = Vec::with_capacity(iv.len() + encrypted.len());
        result.extend_from_slice(&iv);
        result.append(&mut encrypted);
        Ok(result)
    }

    fn decrypt_with_master_key(
        &self,
        ciphertext: &[u8],
        master_key: &MasterKey,
    ) -> Result<Vec<u8>> {
        if ciphertext.len() < 12 + 16 {
            return Err(MemoryError::key_management(
                "Ciphertext too short".to_string(),
            ));
        }
        let (iv, data) = ciphertext.split_at(12);
        let nonce = GenericArray::from_slice(iv);
        let cipher = Aes256Gcm::new(Key::<Aes256Gcm>::from_slice(&master_key.key_data));
        let decrypted = cipher
            .decrypt(
                nonce,
                Payload {
                    msg: data,
                    aad: &[],
                },
            )
            .map_err(|_| MemoryError::key_management("Master key decryption failed"))?;
        Ok(decrypted)
    }

    async fn schedule_key_rotation(&mut self, key_id: &str) -> Result<()> {
        let rotation_time =
            Utc::now() + chrono::Duration::hours(self.config.key_rotation_interval_hours as i64);

        let task = KeyRotationTask {
            id: Uuid::new_v4().to_string(),
            key_id: key_id.to_string(),
            scheduled_time: rotation_time,
            status: RotationStatus::Pending,
            created_at: Utc::now(),
            completed_at: None,
            error_message: None,
        };

        self.key_rotation_schedule.push(task);
        Ok(())
    }

    async fn re_encrypt_data_keys(
        &mut self,
        old_master_key_id: &str,
        new_master_key_id: &str,
    ) -> Result<()> {
        let old_master_key = self
            .master_keys
            .get(old_master_key_id)
            .ok_or_else(|| MemoryError::key_management("Old master key not found".to_string()))?
            .clone();

        let new_master_key = self
            .master_keys
            .get(new_master_key_id)
            .ok_or_else(|| MemoryError::key_management("New master key not found".to_string()))?
            .clone();

        // Collect data keys that need re-encryption
        let mut keys_to_update = Vec::new();
        for (key_id, data_key) in &self.data_keys {
            if data_key.master_key_id == old_master_key_id {
                keys_to_update.push((key_id.clone(), data_key.encrypted_key.clone()));
            }
        }

        // Re-encrypt the collected keys
        for (key_id, encrypted_key) in keys_to_update {
            // Decrypt with old key
            let plaintext = self.decrypt_with_master_key(&encrypted_key, &old_master_key)?;

            // Encrypt with new key
            let new_encrypted = self.encrypt_with_master_key(&plaintext, &new_master_key)?;

            // Update data key
            if let Some(data_key) = self.data_keys.get_mut(&key_id) {
                data_key.encrypted_key = new_encrypted;
                data_key.master_key_id = new_master_key_id.to_string();
            }
        }

        Ok(())
    }
}

/// Master key structure.
///
/// `key_data` holds the raw AES-256 master key bytes and is wrapped in
/// `Zeroizing` so it is scrubbed from memory as soon as the struct (or any
/// clone of it) is dropped.
#[derive(Clone)]
struct MasterKey {
    id: String,
    key_data: Zeroizing<Vec<u8>>,
    algorithm: String,
    created_at: DateTime<Utc>,
    expires_at: DateTime<Utc>,
    status: KeyStatus,
    version: u32,
    usage_count: u64,
}

impl fmt::Debug for MasterKey {
    /// Redacts `key_data`; never format master key bytes into logs/errors.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("MasterKey")
            .field("id", &self.id)
            .field("key_data", &"<redacted>")
            .field("algorithm", &self.algorithm)
            .field("created_at", &self.created_at)
            .field("expires_at", &self.expires_at)
            .field("status", &self.status)
            .field("version", &self.version)
            .field("usage_count", &self.usage_count)
            .finish()
    }
}

/// Data encryption key structure.
///
/// `plaintext_key` is the decrypted data key material; it is wrapped in
/// `Zeroizing` so it is scrubbed from memory on drop (including when the
/// cache entry is replaced or evicted).
#[derive(Clone)]
struct DataKey {
    id: String,
    master_key_id: String,
    encrypted_key: Vec<u8>,
    plaintext_key: Option<Zeroizing<Vec<u8>>>,
    algorithm: String,
    created_at: DateTime<Utc>,
    expires_at: DateTime<Utc>,
    status: KeyStatus,
    usage_count: u64,
}

impl fmt::Debug for DataKey {
    /// Redacts `plaintext_key`; `encrypted_key` is ciphertext (safe to show
    /// as opaque bytes) but is still redacted defensively since it is not
    /// useful for debugging and this keeps the invariant simple.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("DataKey")
            .field("id", &self.id)
            .field("master_key_id", &self.master_key_id)
            .field("encrypted_key", &"<redacted>")
            .field(
                "plaintext_key",
                &self.plaintext_key.as_ref().map(|_| "<redacted>"),
            )
            .field("algorithm", &self.algorithm)
            .field("created_at", &self.created_at)
            .field("expires_at", &self.expires_at)
            .field("status", &self.status)
            .field("usage_count", &self.usage_count)
            .finish()
    }
}

/// Key status enumeration
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum KeyStatus {
    Active,
    Deprecated,
    Expired,
    Revoked,
}

/// Key rotation task
#[derive(Debug, Clone)]
struct KeyRotationTask {
    // Task bookkeeping fields kept for logging/debug; scheduler keys off
    // `key_id`/`scheduled_time`/`status` only.
    #[allow(dead_code)]
    id: String,
    key_id: String,
    scheduled_time: DateTime<Utc>,
    status: RotationStatus,
    #[allow(dead_code)]
    created_at: DateTime<Utc>,
    completed_at: Option<DateTime<Utc>>,
    error_message: Option<String>,
}

/// Rotation status
#[derive(Debug, Clone, PartialEq)]
enum RotationStatus {
    Pending,
    Completed,
    Failed,
}

/// Key information for external queries
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KeyInformation {
    pub id: String,
    pub key_type: KeyType,
    pub algorithm: String,
    pub status: KeyStatus,
    pub created_at: DateTime<Utc>,
    pub expires_at: DateTime<Utc>,
    pub usage_count: u64,
    pub version: Option<u32>,
}

/// Key types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum KeyType {
    Master,
    Data,
}

/// Key management metrics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct KeyManagementMetrics {
    pub total_master_keys_generated: u64,
    pub total_data_keys_generated: u64,
    pub total_key_rotations: u64,
    pub total_keys_revoked: u64,
    pub total_key_operations: u64,
    pub active_master_keys: u64,
    pub active_data_keys: u64,
    pub expired_keys: u64,
    pub average_key_lifetime_hours: f64,
    pub key_rotation_success_rate: f64,
}

#[cfg(test)]
mod key_hygiene_tests {
    use super::*;

    /// Known key bytes, distinctive enough to recognize both as raw bytes
    /// and as the hex string a naive Debug/format would produce.
    const KNOWN_KEY_BYTES: [u8; 8] = [0xDE, 0xAD, 0xBE, 0xEF, 0xCA, 0xFE, 0xBA, 0xBE];

    fn known_key_hex() -> String {
        KNOWN_KEY_BYTES
            .iter()
            .map(|b| format!("{:02x}", b))
            .collect()
    }

    #[test]
    fn master_key_debug_redacts_key_data() {
        let master_key = MasterKey {
            id: "test-master".to_string(),
            key_data: Zeroizing::new(KNOWN_KEY_BYTES.to_vec()),
            algorithm: "AES-256".to_string(),
            created_at: Utc::now(),
            expires_at: Utc::now() + chrono::Duration::days(1),
            status: KeyStatus::Active,
            version: 1,
            usage_count: 0,
        };

        let debug_output = format!("{:?}", master_key);

        assert!(
            !debug_output.contains(&known_key_hex()),
            "MasterKey Debug output leaked key bytes: {debug_output}"
        );
        assert!(
            !debug_output.contains(&format!("{:?}", KNOWN_KEY_BYTES)),
            "MasterKey Debug output leaked raw key byte array: {debug_output}"
        );
        assert!(
            debug_output.contains("redacted"),
            "MasterKey Debug output should mark key_data as redacted: {debug_output}"
        );
    }

    #[test]
    fn data_key_debug_redacts_plaintext_key() {
        let data_key = DataKey {
            id: "test-data".to_string(),
            master_key_id: "test-master".to_string(),
            encrypted_key: vec![1, 2, 3, 4],
            plaintext_key: Some(Zeroizing::new(KNOWN_KEY_BYTES.to_vec())),
            algorithm: "AES-256-GCM".to_string(),
            created_at: Utc::now(),
            expires_at: Utc::now() + chrono::Duration::days(1),
            status: KeyStatus::Active,
            usage_count: 0,
        };

        let debug_output = format!("{:?}", data_key);

        assert!(
            !debug_output.contains(&known_key_hex()),
            "DataKey Debug output leaked plaintext key bytes: {debug_output}"
        );
        assert!(
            !debug_output.contains(&format!("{:?}", KNOWN_KEY_BYTES)),
            "DataKey Debug output leaked raw plaintext key byte array: {debug_output}"
        );
        assert!(
            debug_output.contains("redacted"),
            "DataKey Debug output should mark plaintext_key as redacted: {debug_output}"
        );
    }

    #[tokio::test]
    async fn key_manager_debug_never_leaks_master_key_bytes() {
        let config = SecurityConfig::default();
        let manager = KeyManager::new(&config)
            .await
            .expect("KeyManager::new should succeed with default config");

        let debug_output = format!("{:?}", manager);

        // KeyManager's manual Debug only surfaces counts/config, never the
        // per-key structs' raw fields, so no generated key bytes can appear
        // regardless of what random master key was created during `new`.
        assert!(
            !debug_output.contains("key_data"),
            "KeyManager Debug output should not expose raw key_data field: {debug_output}"
        );
        assert!(
            debug_output.contains("master_key_count"),
            "KeyManager Debug output should summarize key counts, not enumerate keys: {debug_output}"
        );
    }
}
