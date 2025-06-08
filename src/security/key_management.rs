//! Advanced Key Management Module
//! 
//! Implements enterprise-grade key management with automatic rotation,
//! secure storage, and cryptographic key lifecycle management.

use crate::error::{MemoryError, Result};
use crate::security::SecurityConfig;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use chrono::{DateTime, Utc};
use uuid::Uuid;

/// Key management system
#[derive(Debug)]
pub struct KeyManager {
    config: SecurityConfig,
    master_keys: HashMap<String, MasterKey>,
    data_keys: HashMap<String, DataKey>,
    key_rotation_schedule: Vec<KeyRotationTask>,
    metrics: KeyManagementMetrics,
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
            key_data: self.generate_secure_key(self.config.encryption_key_size / 8)?,
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
    pub async fn generate_data_key(&mut self, master_key_id: &str) -> Result<String> {
        // Check master key status first
        {
            let master_key = self.master_keys.get(master_key_id)
                .ok_or_else(|| MemoryError::key_management("Master key not found".to_string()))?;

            if master_key.status != KeyStatus::Active {
                return Err(MemoryError::key_management("Master key is not active".to_string()));
            }
        }

        let data_key_id = Uuid::new_v4().to_string();
        let plaintext_key = self.generate_secure_key(32)?; // 256-bit key

        // Encrypt the data key with the master key
        let encrypted_key = {
            let master_key = self.master_keys.get(master_key_id).unwrap();
            self.encrypt_with_master_key(&plaintext_key, master_key)?
        };

        let data_key = DataKey {
            id: data_key_id.clone(),
            master_key_id: master_key_id.to_string(),
            encrypted_key,
            plaintext_key: Some(plaintext_key), // In production, this would be cleared after use
            algorithm: "AES-256-GCM".to_string(),
            created_at: Utc::now(),
            expires_at: Utc::now() + chrono::Duration::hours(self.config.key_rotation_interval_hours as i64),
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
    pub async fn get_data_key(&mut self, key_id: &str) -> Result<Vec<u8>> {
        // First check if we need to decrypt the key
        let needs_decryption = {
            let data_key = self.data_keys.get(key_id)
                .ok_or_else(|| MemoryError::key_management("Data key not found".to_string()))?;

            if data_key.status != KeyStatus::Active {
                return Err(MemoryError::key_management("Data key is not active".to_string()));
            }

            if Utc::now() > data_key.expires_at {
                return Err(MemoryError::key_management("Data key has expired".to_string()));
            }

            data_key.plaintext_key.is_none()
        };

        // If plaintext key is not available, decrypt it
        if needs_decryption {
            let (master_key_id, encrypted_key) = {
                let data_key = self.data_keys.get(key_id).unwrap();
                (data_key.master_key_id.clone(), data_key.encrypted_key.clone())
            };

            let master_key = self.master_keys.get(&master_key_id)
                .ok_or_else(|| MemoryError::key_management("Master key not found".to_string()))?;

            let plaintext_key = self.decrypt_with_master_key(&encrypted_key, master_key)?;

            // Update the data key with the plaintext
            if let Some(data_key) = self.data_keys.get_mut(key_id) {
                data_key.plaintext_key = Some(plaintext_key);
            }
        }

        // Update usage count and return the key
        let result = {
            let data_key = self.data_keys.get_mut(key_id).unwrap();
            data_key.usage_count += 1;
            data_key.plaintext_key.as_ref().unwrap().clone()
        };

        self.metrics.total_key_operations += 1;
        Ok(result)
    }

    /// Rotate a master key
    pub async fn rotate_master_key(&mut self, key_id: &str) -> Result<String> {
        // Get old key info and mark as deprecated
        let (new_version, algorithm) = {
            let old_key = self.master_keys.get_mut(key_id)
                .ok_or_else(|| MemoryError::key_management("Master key not found".to_string()))?;

            // Mark old key as deprecated
            old_key.status = KeyStatus::Deprecated;
            (old_key.version + 1, old_key.algorithm.clone())
        };

        // Generate new version
        let new_key_id = format!("{}:v{}", key_id, new_version);

        let new_master_key = MasterKey {
            id: new_key_id.clone(),
            key_data: self.generate_secure_key(self.config.encryption_key_size / 8)?,
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
        let keys_to_rotate: Vec<String> = self.key_rotation_schedule.iter()
            .filter(|task| now >= task.scheduled_time && task.status == RotationStatus::Pending)
            .map(|task| task.key_id.clone())
            .collect();

        for key_id in keys_to_rotate {
            match self.rotate_master_key(&key_id).await {
                Ok(new_key_id) => {
                    rotated_keys.push(new_key_id);
                    // Update rotation task status
                    if let Some(task) = self.key_rotation_schedule.iter_mut()
                        .find(|t| t.key_id == key_id) {
                        task.status = RotationStatus::Completed;
                        task.completed_at = Some(now);
                    }
                },
                Err(e) => {
                    // Log error and mark as failed
                    if let Some(task) = self.key_rotation_schedule.iter_mut()
                        .find(|t| t.key_id == key_id) {
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
        metrics.active_master_keys = self.master_keys.values()
            .filter(|k| k.status == KeyStatus::Active)
            .count() as u64;
        
        metrics.active_data_keys = self.data_keys.values()
            .filter(|k| k.status == KeyStatus::Active)
            .count() as u64;
        
        // Count expired master keys
        let expired_master_keys = self.master_keys.values()
            .filter(|k| k.status == KeyStatus::Expired)
            .count();

        // Count expired data keys
        let expired_data_keys = self.data_keys.values()
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
        // In production, use a cryptographically secure random number generator
        // For demo purposes, we'll use a deterministic but varied approach
        let mut key = Vec::with_capacity(length);
        for i in 0..length {
            key.push(((i * 17 + 42 + length) % 256) as u8);
        }
        Ok(key)
    }

    fn encrypt_with_master_key(&self, plaintext: &[u8], master_key: &MasterKey) -> Result<Vec<u8>> {
        // Simplified encryption - in production use proper AES-GCM
        let mut encrypted = plaintext.to_vec();
        for (i, byte) in encrypted.iter_mut().enumerate() {
            *byte ^= master_key.key_data[i % master_key.key_data.len()];
        }
        Ok(encrypted)
    }

    fn decrypt_with_master_key(&self, ciphertext: &[u8], master_key: &MasterKey) -> Result<Vec<u8>> {
        // Simplified decryption - reverse of encryption
        let mut decrypted = ciphertext.to_vec();
        for (i, byte) in decrypted.iter_mut().enumerate() {
            *byte ^= master_key.key_data[i % master_key.key_data.len()];
        }
        Ok(decrypted)
    }

    async fn schedule_key_rotation(&mut self, key_id: &str) -> Result<()> {
        let rotation_time = Utc::now() + chrono::Duration::hours(self.config.key_rotation_interval_hours as i64);
        
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

    async fn re_encrypt_data_keys(&mut self, old_master_key_id: &str, new_master_key_id: &str) -> Result<()> {
        let old_master_key = self.master_keys.get(old_master_key_id)
            .ok_or_else(|| MemoryError::key_management("Old master key not found".to_string()))?
            .clone();

        let new_master_key = self.master_keys.get(new_master_key_id)
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

/// Master key structure
#[derive(Debug, Clone)]
struct MasterKey {
    id: String,
    key_data: Vec<u8>,
    algorithm: String,
    created_at: DateTime<Utc>,
    expires_at: DateTime<Utc>,
    status: KeyStatus,
    version: u32,
    usage_count: u64,
}

/// Data encryption key structure
#[derive(Debug, Clone)]
struct DataKey {
    id: String,
    master_key_id: String,
    encrypted_key: Vec<u8>,
    plaintext_key: Option<Vec<u8>>,
    algorithm: String,
    created_at: DateTime<Utc>,
    expires_at: DateTime<Utc>,
    status: KeyStatus,
    usage_count: u64,
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
    id: String,
    key_id: String,
    scheduled_time: DateTime<Utc>,
    status: RotationStatus,
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

/// Trait for common key information
trait KeyInfo {
    fn get_status(&self) -> &KeyStatus;
    fn get_created_at(&self) -> DateTime<Utc>;
    fn get_expires_at(&self) -> DateTime<Utc>;
}

impl KeyInfo for MasterKey {
    fn get_status(&self) -> &KeyStatus { &self.status }
    fn get_created_at(&self) -> DateTime<Utc> { self.created_at }
    fn get_expires_at(&self) -> DateTime<Utc> { self.expires_at }
}

impl KeyInfo for DataKey {
    fn get_status(&self) -> &KeyStatus { &self.status }
    fn get_created_at(&self) -> DateTime<Utc> { self.created_at }
    fn get_expires_at(&self) -> DateTime<Utc> { self.expires_at }
}
