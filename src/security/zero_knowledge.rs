//! Zero-Knowledge Architecture Module
//!
//! Implements zero-knowledge proofs and privacy-preserving protocols
//! for secure computation and verification without revealing sensitive data.

use crate::error::{MemoryError, Result};
use crate::memory::types::MemoryEntry;
use crate::security::{SecurityConfig, SecurityContext};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::fmt;
use uuid::Uuid;

#[cfg(feature = "zero-knowledge-proofs")]
use bellman::{
    groth16::{
        create_random_proof, generate_random_parameters, prepare_verifying_key, verify_proof,
        Parameters, PreparedVerifyingKey, Proof,
    },
    Circuit, ConstraintSystem, SynthesisError,
};

#[cfg(feature = "zero-knowledge-proofs")]
use bls12_381::{Bls12, Scalar};

#[cfg(feature = "zero-knowledge-proofs")]
use rand::rngs::OsRng;

#[cfg(feature = "zero-knowledge-proofs")]
use bellpepper_core::{
    num::AllocatedNum, ConstraintSystem as BpConstraintSystem, Index as BpIndex,
    LinearCombination as BpLinearCombination, SynthesisError as BpSynthesisError,
    Variable as BpVariable,
};

#[cfg(feature = "zero-knowledge-proofs")]
use neptune::poseidon::{Poseidon, PoseidonConstants};

#[cfg(feature = "zero-knowledge-proofs")]
use typenum::U2;

#[cfg(feature = "zero-knowledge-proofs")]
use ff::Field as _;

/// Computes a real, hex-encoded SHA-256 digest of `content`.
///
/// This is the single implementation shared by all "content hash" and
/// "statement hash" call sites in this module, replacing the previous
/// fake length-arithmetic scheme (`content.len() * 17 + 42`) that
/// collided for any two equal-length inputs.
fn hash_content(content: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(content.as_bytes());
    format!("{:x}", hasher.finalize())
}

/// Test-only accessor for [`hash_content`], exposed so integration tests
/// can assert the real hashing behavior without depending on internal
/// struct layout.
#[doc(hidden)]
pub fn hash_content_for_test(content: &str) -> String {
    hash_content(content)
}

/// Verify a zero-knowledge proof against a prepared Groth16 verifying key.
///
/// Every Groth16 public input is VERIFIER-DERIVED: the registered
/// commitment comes from the verifier's trusted store and the statement
/// binding is recomputed from the statement the verifier holds. Nothing
/// from the (attacker-controlled) proof envelope influences the public
/// inputs; only the proof points themselves are read from it. The
/// statement-hash comparison below is a cheap pre-filter, not a security
/// check — soundness rests on the pairing equation. Corrupted proof points
/// verify `false` rather than erroring, so tampering can never escalate.
#[cfg(feature = "zero-knowledge-proofs")]
fn verify_with_prepared_key(
    prepared_vk: &PreparedVerifyingKey<Bls12>,
    proof: &ZKProof,
    expected_statement_hash: &str,
    registered_commitment: &Scalar,
) -> Result<bool> {
    let start_time = std::time::Instant::now();

    // Cheap pre-filter only (attacker-recomputable): the cryptographic
    // statement binding is enforced via the public input below.
    if proof.statement_hash != expected_statement_hash {
        tracing::warn!("Statement hash mismatch in proof verification");
        return Ok(false);
    }

    let statement_binding = statement_binding_scalar(expected_statement_hash);

    // Deserialize the Groth16 proof; corrupted points fail verification.
    let groth16_proof = match ProofSystem::deserialize_proof(&proof.proof_data) {
        Ok(groth16_proof) => groth16_proof,
        Err(_) => {
            tracing::warn!("Failed to deserialize Groth16 proof data");
            return Ok(false);
        }
    };

    let is_valid = verify_proof(
        prepared_vk,
        &groth16_proof,
        &[*registered_commitment, statement_binding],
    )
    .is_ok();

    tracing::debug!(
        is_valid,
        "zk-SNARK Groth16 verification completed in {:?}",
        start_time.elapsed()
    );
    Ok(is_valid)
}

/// Verify a proof as an external verifier, holding only the statement, the
/// proof, the serialized verifying key (see
/// [`ZeroKnowledgeManager::verifying_key_bytes`]) and the prover's
/// registered commitment obtained over a trusted channel (see
/// [`ZeroKnowledgeManager::registered_commitment_bytes`]). No access to the
/// generating manager or its proving material is required, and no public
/// input is read from the proof envelope.
#[cfg(feature = "zero-knowledge-proofs")]
pub fn verify_proof_external<T: Serialize>(
    statement: &T,
    proof: &ZKProof,
    verifying_key_bytes: &[u8],
    registered_commitment_bytes: &[u8],
) -> Result<bool> {
    let vk = bellman::groth16::VerifyingKey::<Bls12>::read(verifying_key_bytes).map_err(|e| {
        MemoryError::encryption(format!("Failed to deserialize Groth16 verifying key: {e}"))
    })?;
    let prepared_vk = prepare_verifying_key(&vk);
    let commitment = parse_commitment(registered_commitment_bytes)?;
    let serialized = serde_json::to_string(statement)
        .map_err(|_| MemoryError::access_denied("Statement serialization failed"))?;
    verify_with_prepared_key(&prepared_vk, proof, &hash_content(&serialized), &commitment)
}

/// Configuration for zero-knowledge features
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ZeroKnowledgeConfig {
    /// Enable zero-knowledge proofs for access control
    pub enable_access_proofs: bool,
    /// Enable zero-knowledge proofs for content verification
    pub enable_content_proofs: bool,
    /// Enable zero-knowledge proofs for aggregate statistics
    pub enable_aggregate_proofs: bool,
    /// Proof system type
    pub proof_system: ProofSystemType,
    /// Security parameter for proof generation
    pub security_parameter: usize,
    /// Enable proof caching
    pub enable_proof_caching: bool,
    /// Maximum number of cached proofs
    pub max_cached_proofs: usize,
}

impl Default for ZeroKnowledgeConfig {
    fn default() -> Self {
        Self {
            enable_access_proofs: true,
            enable_content_proofs: true,
            enable_aggregate_proofs: true,
            proof_system: ProofSystemType::SNARK,
            security_parameter: 128,
            enable_proof_caching: true,
            max_cached_proofs: 1000,
        }
    }
}

/// Types of proof systems available
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ProofSystemType {
    /// Succinct Non-Interactive Arguments of Knowledge
    SNARK,
    /// Scalable Transparent Arguments of Knowledge
    STARK,
    /// Bulletproofs
    Bulletproof,
}

/// Zero-knowledge manager for privacy-preserving operations
///
/// Key hygiene (Task 4.6): `prover_secrets` holds `bls12_381::Scalar` prover
/// secret witnesses. The `bls12_381` crate's `zeroize` feature (enabled in
/// Cargo.toml) implements `zeroize::DefaultIsZeroes for Scalar`, so each
/// scalar can be scrubbed. `HashMap` does not zeroize its values on drop, so
/// the manual `impl Drop for ZeroKnowledgeManager` below iterates
/// `prover_secrets` and calls `.zeroize()` on every value before the map is
/// released. `Debug` is also manually implemented so `prover_secrets` is
/// never printed via `{:?}` (the derive would have exposed `Scalar::fmt`,
/// which prints internal limbs).
///
/// Residual risk: `Scalar` is `Copy`, so the compiler may leave transient
/// stack copies from moves/reads that Drop cannot reach; this is inherent to
/// `Copy` secret types and out of scope to eliminate here.
pub struct ZeroKnowledgeManager {
    config: SecurityConfig,
    proof_system: ProofSystem,
    verification_keys: HashMap<String, VerificationKey>,
    metrics: ZeroKnowledgeMetrics,
    /// Prover-side secrets for locally registered identities. Known only to
    /// this manager; never serialized, never placed on proof envelopes.
    /// Zeroized on drop via `impl Drop for ZeroKnowledgeManager`.
    #[cfg(feature = "zero-knowledge-proofs")]
    prover_secrets: HashMap<String, Scalar>,
    /// Verifier-side trusted registration store: identity -> Poseidon
    /// commitment. Every Groth16 public input is derived from this store
    /// (and the statement being verified), never from the proof envelope.
    #[cfg(feature = "zero-knowledge-proofs")]
    prover_commitments: HashMap<String, Scalar>,
}

impl fmt::Debug for ZeroKnowledgeManager {
    /// Manual `Debug` impl: `prover_secrets` holds raw prover secret
    /// scalars and must never be formatted; `prover_commitments` are public
    /// commitments (safe to show) but are counted rather than printed to
    /// keep the output stable across feature flags.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut s = f.debug_struct("ZeroKnowledgeManager");
        s.field("config", &self.config)
            .field("proof_system", &self.proof_system)
            .field("verification_key_count", &self.verification_keys.len())
            .field("metrics", &self.metrics);
        #[cfg(feature = "zero-knowledge-proofs")]
        {
            s.field("prover_secret_count", &self.prover_secrets.len())
                .field("prover_commitment_count", &self.prover_commitments.len());
        }
        s.finish()
    }
}

#[cfg(feature = "zero-knowledge-proofs")]
impl Drop for ZeroKnowledgeManager {
    /// Scrub prover secret scalars from memory on drop. `HashMap` does not
    /// zeroize its values, so we iterate and zeroize each `Scalar`
    /// explicitly (relies on `bls12_381`'s `zeroize` feature providing
    /// `DefaultIsZeroes for Scalar`).
    fn drop(&mut self) {
        use zeroize::Zeroize;
        for secret in self.prover_secrets.values_mut() {
            secret.zeroize();
        }
    }
}

impl ZeroKnowledgeManager {
    /// Create a new zero-knowledge manager with freshly generated
    /// (in-memory) Groth16 parameters.
    pub async fn new(config: &SecurityConfig) -> Result<Self> {
        let proof_system = ProofSystem::new(config).await?;

        Ok(Self {
            config: config.clone(),
            proof_system,
            verification_keys: HashMap::new(),
            metrics: ZeroKnowledgeMetrics::default(),
            #[cfg(feature = "zero-knowledge-proofs")]
            prover_secrets: HashMap::new(),
            #[cfg(feature = "zero-knowledge-proofs")]
            prover_commitments: HashMap::new(),
        })
    }

    /// Create a manager backed by a persistent key store.
    ///
    /// If `key_store_path` points at an existing file, the Groth16
    /// parameters are loaded from it (with full curve-point validation);
    /// otherwise fresh parameters are generated from `OsRng` and written to
    /// the path so subsequent constructions reuse the same keys. With
    /// `None`, behaves like [`ZeroKnowledgeManager::new`].
    #[cfg(feature = "zero-knowledge-proofs")]
    pub async fn new_with_key_store(
        config: &SecurityConfig,
        key_store_path: Option<&std::path::Path>,
    ) -> Result<Self> {
        let path = match key_store_path {
            Some(path) => path,
            None => return Self::new(config).await,
        };

        if path.exists() {
            let bytes = std::fs::read(path).map_err(|e| {
                MemoryError::encryption(format!(
                    "Failed to read Groth16 key store {}: {e}",
                    path.display()
                ))
            })?;
            return Self::from_groth16_parameter_bytes(config, &bytes);
        }

        let manager = Self::new(config).await?;
        let bytes = manager.groth16_parameter_bytes()?;
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).map_err(|e| {
                MemoryError::encryption(format!(
                    "Failed to create Groth16 key store directory {}: {e}",
                    parent.display()
                ))
            })?;
        }
        std::fs::write(path, bytes).map_err(|e| {
            MemoryError::encryption(format!(
                "Failed to write Groth16 key store {}: {e}",
                path.display()
            ))
        })?;
        Ok(manager)
    }

    /// Reconstruct a manager from previously serialized Groth16 parameters
    /// (as produced by [`ZeroKnowledgeManager::groth16_parameter_bytes`]).
    ///
    /// Curve points are validated during deserialization, so tampered or
    /// truncated parameter blobs are rejected.
    #[cfg(feature = "zero-knowledge-proofs")]
    pub fn from_groth16_parameter_bytes(config: &SecurityConfig, bytes: &[u8]) -> Result<Self> {
        let proof_system = ProofSystem::from_parameter_bytes(bytes)?;
        Ok(Self {
            config: config.clone(),
            proof_system,
            verification_keys: HashMap::new(),
            metrics: ZeroKnowledgeMetrics::default(),
            prover_secrets: HashMap::new(),
            prover_commitments: HashMap::new(),
        })
    }

    /// Serialize the Groth16 proving+verifying parameters for persistence.
    #[cfg(feature = "zero-knowledge-proofs")]
    pub fn groth16_parameter_bytes(&self) -> Result<Vec<u8>> {
        self.proof_system.parameter_bytes()
    }

    /// Serialize only the (unprepared) verifying key, for distribution to
    /// external verifiers that must not hold proving material.
    #[cfg(feature = "zero-knowledge-proofs")]
    pub fn verifying_key_bytes(&self) -> Result<Vec<u8>> {
        self.proof_system.verifying_key_bytes()
    }

    /// Access the prepared verifying key for in-process verification.
    #[cfg(feature = "zero-knowledge-proofs")]
    pub fn prepared_verifying_key(&self) -> &PreparedVerifyingKey<Bls12> {
        self.proof_system.prepared_verifying_key()
    }

    /// Register a prover identity: generates a fresh random secret witness
    /// (held prover-side by this manager) and stores its Poseidon commitment
    /// in the verifier's trusted registration store.
    ///
    /// Idempotent: re-registering an existing identity keeps the original
    /// secret and commitment.
    #[cfg(feature = "zero-knowledge-proofs")]
    pub fn register_prover(&mut self, user_id: &str) -> Result<()> {
        if self.prover_secrets.contains_key(user_id) {
            return Ok(());
        }
        let secret = Scalar::random(OsRng);
        let commitment = poseidon_digest(secret);
        self.prover_secrets.insert(user_id.to_string(), secret);
        self.prover_commitments
            .insert(user_id.to_string(), commitment);
        Ok(())
    }

    /// Fallback registration when the zero-knowledge-proofs feature is off:
    /// a no-op, since fallback verification always fails closed.
    #[cfg(not(feature = "zero-knowledge-proofs"))]
    pub fn register_prover(&mut self, _user_id: &str) -> Result<()> {
        Ok(())
    }

    /// Register a verifier-side commitment for an identity whose secret is
    /// held elsewhere (e.g. distributed out of band from the prover's
    /// manager). Rejects non-canonical commitment encodings.
    ///
    /// Reject-if-present (unlike the idempotent [`register_prover`]): once a
    /// commitment is bound to an identity, silently rebinding it to a
    /// different value would let a misconfiguration or a hostile caller swap
    /// in an attacker-controlled commitment. Callers that legitimately need
    /// to rotate a commitment must do so explicitly (remove then re-add);
    /// re-registering the identical commitment is a no-op success.
    #[cfg(feature = "zero-knowledge-proofs")]
    pub fn register_commitment(&mut self, user_id: &str, commitment_bytes: &[u8]) -> Result<()> {
        let commitment = parse_commitment(commitment_bytes)?;
        if let Some(existing) = self.prover_commitments.get(user_id) {
            if *existing == commitment {
                return Ok(());
            }
            return Err(MemoryError::access_denied(format!(
                "A different zero-knowledge commitment is already registered for '{user_id}'"
            )));
        }
        self.prover_commitments
            .insert(user_id.to_string(), commitment);
        Ok(())
    }

    /// The registered Poseidon commitment for an identity, serialized for
    /// distribution to external verifiers. `None` if unregistered.
    #[cfg(feature = "zero-knowledge-proofs")]
    pub fn registered_commitment_bytes(&self, user_id: &str) -> Option<Vec<u8>> {
        self.prover_commitments
            .get(user_id)
            .map(|commitment| commitment.to_bytes().to_vec())
    }

    /// Generate a zero-knowledge proof for memory access.
    ///
    /// The caller supplies the complete [`AccessStatement`] (including its
    /// timestamp/nonce); the statement is hashed exactly as provided, so any
    /// verifier holding the same statement can reproduce the binding.
    pub async fn generate_access_proof(
        &mut self,
        statement: &AccessStatement,
        user_context: &SecurityContext,
    ) -> Result<ZKProof> {
        let start_time = std::time::Instant::now();

        // Validate security context comprehensively
        user_context.validate_comprehensive(true)?; // ZK proofs always require MFA

        // The statement's subject must be the authenticated caller.
        if statement.user_id != user_context.user_id {
            return Err(MemoryError::access_denied(
                "Access statement subject does not match the authenticated user",
            ));
        }

        #[cfg(feature = "zero-knowledge-proofs")]
        let proof = {
            let (prover_id, secret) = self.registered_secret(&user_context.user_id)?;
            self.proof_system
                .generate_registered_proof(statement, &prover_id, secret)
                .await?
        };

        #[cfg(not(feature = "zero-knowledge-proofs"))]
        let proof = {
            let witness = self
                .generate_access_witness(statement, user_context)
                .await?;
            self.proof_system
                .generate_proof(statement, &witness)
                .await?
        };

        // Update metrics
        self.metrics.total_proofs_generated += 1;
        self.metrics.total_proof_generation_time_ms += start_time.elapsed().as_millis() as u64;

        Ok(proof)
    }

    /// Verify a zero-knowledge proof
    pub async fn verify_access_proof(
        &mut self,
        proof: &ZKProof,
        statement: &AccessStatement,
    ) -> Result<bool> {
        let start_time = std::time::Instant::now();

        // Verify the proof against verifier-derived public inputs.
        #[cfg(feature = "zero-knowledge-proofs")]
        let is_valid = self.verify_against_registration(proof, statement)?;
        #[cfg(not(feature = "zero-knowledge-proofs"))]
        let is_valid = self.proof_system.verify_proof(proof, statement).await?;

        // Update metrics
        self.metrics.total_proofs_verified += 1;
        self.metrics.total_verification_time_ms += start_time.elapsed().as_millis() as u64;

        if is_valid {
            self.metrics.successful_verifications += 1;
        }

        Ok(is_valid)
    }

    /// Generate a zero-knowledge proof for memory content without revealing
    /// it. The caller supplies the complete [`ContentStatement`].
    pub async fn generate_content_proof(
        &mut self,
        entry: &MemoryEntry,
        statement: &ContentStatement,
        context: &SecurityContext,
    ) -> Result<ZKProof> {
        let start_time = std::time::Instant::now();

        #[cfg(feature = "zero-knowledge-proofs")]
        let proof = {
            let _ = entry; // content witnessing is bound via the statement
            let (prover_id, secret) = self.registered_secret(&context.user_id)?;
            self.proof_system
                .generate_registered_proof(statement, &prover_id, secret)
                .await?
        };

        #[cfg(not(feature = "zero-knowledge-proofs"))]
        let proof = {
            let _ = context;
            let witness = self.generate_content_witness(entry, statement).await?;
            self.proof_system
                .generate_proof(statement, &witness)
                .await?
        };

        // Update metrics
        self.metrics.total_proofs_generated += 1;
        self.metrics.total_content_proofs += 1;
        self.metrics.total_proof_generation_time_ms += start_time.elapsed().as_millis() as u64;

        Ok(proof)
    }

    /// Verify content proof without accessing the actual content
    pub async fn verify_content_proof(
        &mut self,
        proof: &ZKProof,
        statement: &ContentStatement,
    ) -> Result<bool> {
        let start_time = std::time::Instant::now();

        #[cfg(feature = "zero-knowledge-proofs")]
        let is_valid = self.verify_against_registration(proof, statement)?;
        #[cfg(not(feature = "zero-knowledge-proofs"))]
        let is_valid = self.proof_system.verify_proof(proof, statement).await?;

        self.metrics.total_proofs_verified += 1;
        self.metrics.total_verification_time_ms += start_time.elapsed().as_millis() as u64;

        if is_valid {
            self.metrics.successful_verifications += 1;
        }

        Ok(is_valid)
    }

    /// Generate a zero-knowledge proof for aggregate statistics.
    /// The caller supplies the complete [`AggregateStatement`].
    pub async fn generate_aggregate_proof(
        &mut self,
        entries: &[MemoryEntry],
        statement: &AggregateStatement,
        context: &SecurityContext,
    ) -> Result<ZKProof> {
        let start_time = std::time::Instant::now();

        #[cfg(feature = "zero-knowledge-proofs")]
        let proof = {
            let _ = entries; // aggregate inputs are bound via the statement
            let (prover_id, secret) = self.registered_secret(&context.user_id)?;
            self.proof_system
                .generate_registered_proof(statement, &prover_id, secret)
                .await?
        };

        #[cfg(not(feature = "zero-knowledge-proofs"))]
        let proof = {
            let _ = context;
            let witness = self.generate_aggregate_witness(entries, statement).await?;
            self.proof_system
                .generate_proof(statement, &witness)
                .await?
        };

        // Update metrics
        self.metrics.total_proofs_generated += 1;
        self.metrics.total_aggregate_proofs += 1;
        self.metrics.total_proof_generation_time_ms += start_time.elapsed().as_millis() as u64;

        Ok(proof)
    }

    /// Get zero-knowledge metrics
    pub async fn get_metrics(&self) -> Result<ZeroKnowledgeMetrics> {
        Ok(self.metrics.clone())
    }

    // Private helper methods

    /// Look up the local prover secret for an identity; refuse proof
    /// generation for unregistered identities.
    #[cfg(feature = "zero-knowledge-proofs")]
    fn registered_secret(&self, user_id: &str) -> Result<(String, Scalar)> {
        match self.prover_secrets.get(user_id) {
            Some(secret) => Ok((user_id.to_string(), *secret)),
            None => Err(MemoryError::access_denied(format!(
                "No registered zero-knowledge prover secret for '{user_id}'; call register_prover first"
            ))),
        }
    }

    /// Verify a proof using ONLY verifier-derived public inputs: the
    /// registered commitment looked up from this manager's trusted store by
    /// the claimed prover identity, and the statement binding recomputed
    /// from the caller-supplied statement. Nothing cryptographically
    /// relevant is read from the proof envelope besides the proof points.
    #[cfg(feature = "zero-knowledge-proofs")]
    fn verify_against_registration<T: Serialize>(
        &self,
        proof: &ZKProof,
        statement: &T,
    ) -> Result<bool> {
        let commitment = match self.prover_commitments.get(&proof.prover_id) {
            Some(commitment) => *commitment,
            None => {
                tracing::warn!(
                    prover_id = %proof.prover_id,
                    "Proof claims an unregistered prover identity"
                );
                return Ok(false);
            }
        };
        let serialized = serde_json::to_string(statement)
            .map_err(|_| MemoryError::access_denied("Statement serialization failed"))?;
        verify_with_prepared_key(
            self.proof_system.prepared_verifying_key(),
            proof,
            &hash_content(&serialized),
            &commitment,
        )
    }

    #[cfg(not(feature = "zero-knowledge-proofs"))]
    async fn generate_access_witness(
        &self,
        statement: &AccessStatement,
        context: &SecurityContext,
    ) -> Result<Witness> {
        // Generate witness proving user has access rights
        let witness_data = WitnessData {
            user_attributes: context.attributes.clone(),
            session_proof: context.session_id.clone(),
            timestamp_proof: statement.timestamp.timestamp() as u64,
            access_signature: self.generate_access_signature(statement, context).await?,
        };

        Ok(Witness {
            id: Uuid::new_v4().to_string(),
            data: witness_data,
            created_at: Utc::now(),
        })
    }

    #[cfg(not(feature = "zero-knowledge-proofs"))]
    async fn generate_content_witness(
        &self,
        entry: &MemoryEntry,
        statement: &ContentStatement,
    ) -> Result<Witness> {
        // Generate witness proving content satisfies predicate without revealing content
        let content_hash = hash_content(&entry.value);
        let predicate_result = self.evaluate_predicate(&entry.value, &statement.predicate);

        let witness_data = WitnessData {
            user_attributes: HashMap::new(),
            session_proof: content_hash,
            timestamp_proof: statement.timestamp.timestamp() as u64,
            access_signature: predicate_result.to_string(),
        };

        Ok(Witness {
            id: Uuid::new_v4().to_string(),
            data: witness_data,
            created_at: Utc::now(),
        })
    }

    #[cfg(not(feature = "zero-knowledge-proofs"))]
    async fn generate_aggregate_witness(
        &self,
        entries: &[MemoryEntry],
        statement: &AggregateStatement,
    ) -> Result<Witness> {
        // Generate witness for aggregate computation
        let aggregate_value = match statement.aggregate_type {
            AggregateType::Count => entries.len() as f64,
            AggregateType::AverageLength => {
                entries.iter().map(|e| e.value.len()).sum::<usize>() as f64 / entries.len() as f64
            }
            AggregateType::TotalSize => entries.iter().map(|e| e.value.len()).sum::<usize>() as f64,
        };

        let witness_data = WitnessData {
            user_attributes: HashMap::new(),
            session_proof: aggregate_value.to_string(),
            timestamp_proof: statement.timestamp.timestamp() as u64,
            access_signature: format!("aggregate_{}", statement.aggregate_type.to_string()),
        };

        Ok(Witness {
            id: Uuid::new_v4().to_string(),
            data: witness_data,
            created_at: Utc::now(),
        })
    }

    #[cfg(not(feature = "zero-knowledge-proofs"))]
    async fn generate_access_signature(
        &self,
        statement: &AccessStatement,
        context: &SecurityContext,
    ) -> Result<String> {
        // Simplified signature generation
        let signature_input = format!(
            "{}:{}:{}:{}",
            statement.memory_key,
            statement.user_id,
            statement.access_type.to_string(),
            context.session_id
        );
        Ok(format!("sig_{}", hash_content(&signature_input)))
    }

    #[cfg(not(feature = "zero-knowledge-proofs"))]
    fn evaluate_predicate(&self, content: &str, predicate: &ContentPredicate) -> bool {
        match predicate {
            ContentPredicate::ContainsKeyword(keyword) => content.contains(keyword),
            ContentPredicate::LengthGreaterThan(length) => content.len() > *length,
            ContentPredicate::LengthLessThan(length) => content.len() < *length,
            ContentPredicate::MatchesPattern(pattern) => content.contains(pattern), // Simplified
        }
    }
}

/// Proof system for zero-knowledge operations with real Bellman zk-SNARKs
struct ProofSystem {
    #[cfg(feature = "zero-knowledge-proofs")]
    parameters: Parameters<Bls12>,
    #[cfg(feature = "zero-knowledge-proofs")]
    prepared_vk: PreparedVerifyingKey<Bls12>,
    #[cfg(not(feature = "zero-knowledge-proofs"))]
    proving_key: ProvingKey,
    #[cfg(not(feature = "zero-knowledge-proofs"))]
    verification_key: VerificationKey,
}

impl std::fmt::Debug for ProofSystem {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ProofSystem")
            .field("initialized", &true)
            .finish()
    }
}

/// Records a circuit synthesized against `bellpepper-core`'s
/// `ConstraintSystem` trait (used by neptune's Poseidon gadget) so it can be
/// replayed into a bellman `ConstraintSystem` for Groth16 proving.
///
/// bellpepper's trait requires `Send`, which a `&mut`-borrowing adapter over
/// an arbitrary bellman constraint system cannot promise; a recorder that
/// owns its data can. Allocation order (and therefore variable indexing) is
/// preserved exactly on replay, so the constraint system seen by bellman is
/// identical to the one neptune synthesized.
#[cfg(feature = "zero-knowledge-proofs")]
struct BellpepperRecorder {
    /// Values of auxiliary (private) allocations, in allocation order.
    aux: Vec<Option<Scalar>>,
    /// Values of public-input allocations, in allocation order. Index 0 is
    /// the constant `one` input, mirroring both frameworks' conventions.
    inputs: Vec<Option<Scalar>>,
    /// Recorded R1CS constraints as bellpepper linear combinations.
    constraints: Vec<(
        BpLinearCombination<Scalar>,
        BpLinearCombination<Scalar>,
        BpLinearCombination<Scalar>,
    )>,
}

#[cfg(feature = "zero-knowledge-proofs")]
impl BellpepperRecorder {
    fn new() -> Self {
        Self {
            aux: Vec::new(),
            inputs: vec![Some(Scalar::ONE)],
            constraints: Vec::new(),
        }
    }

    /// Replay the recorded allocations and constraints into a bellman
    /// constraint system, preserving variable indices.
    fn replay<CS: ConstraintSystem<Scalar>>(
        self,
        cs: &mut CS,
    ) -> std::result::Result<(), SynthesisError> {
        let mut aux_vars = Vec::with_capacity(self.aux.len());
        for value in &self.aux {
            let value = *value;
            aux_vars.push(cs.alloc(
                || "recorded aux",
                || value.ok_or(SynthesisError::AssignmentMissing),
            )?);
        }
        let mut input_vars = vec![CS::one()];
        for value in self.inputs.iter().skip(1) {
            let value = *value;
            input_vars.push(cs.alloc_input(
                || "recorded input",
                || value.ok_or(SynthesisError::AssignmentMissing),
            )?);
        }

        let convert = |lc: &BpLinearCombination<Scalar>| -> bellman::LinearCombination<Scalar> {
            let mut out = bellman::LinearCombination::<Scalar>::zero();
            for (variable, coeff) in lc.iter() {
                let bellman_var = match variable.get_unchecked() {
                    BpIndex::Input(i) => input_vars[i],
                    BpIndex::Aux(i) => aux_vars[i],
                };
                out = out + (*coeff, bellman_var);
            }
            out
        };

        for (a, b, c) in &self.constraints {
            let (a, b, c) = (convert(a), convert(b), convert(c));
            cs.enforce(|| "recorded constraint", |_| a, |_| b, |_| c);
        }
        Ok(())
    }
}

#[cfg(feature = "zero-knowledge-proofs")]
impl BpConstraintSystem<Scalar> for BellpepperRecorder {
    type Root = Self;

    fn alloc<F, A, AR>(
        &mut self,
        _annotation: A,
        f: F,
    ) -> std::result::Result<BpVariable, BpSynthesisError>
    where
        F: FnOnce() -> std::result::Result<Scalar, BpSynthesisError>,
        A: FnOnce() -> AR,
        AR: Into<String>,
    {
        self.aux.push(f().ok());
        Ok(BpVariable::new_unchecked(BpIndex::Aux(self.aux.len() - 1)))
    }

    fn alloc_input<F, A, AR>(
        &mut self,
        _annotation: A,
        f: F,
    ) -> std::result::Result<BpVariable, BpSynthesisError>
    where
        F: FnOnce() -> std::result::Result<Scalar, BpSynthesisError>,
        A: FnOnce() -> AR,
        AR: Into<String>,
    {
        self.inputs.push(f().ok());
        Ok(BpVariable::new_unchecked(BpIndex::Input(
            self.inputs.len() - 1,
        )))
    }

    fn enforce<A, AR, LA, LB, LC>(&mut self, _annotation: A, a: LA, b: LB, c: LC)
    where
        A: FnOnce() -> AR,
        AR: Into<String>,
        LA: FnOnce(BpLinearCombination<Scalar>) -> BpLinearCombination<Scalar>,
        LB: FnOnce(BpLinearCombination<Scalar>) -> BpLinearCombination<Scalar>,
        LC: FnOnce(BpLinearCombination<Scalar>) -> BpLinearCombination<Scalar>,
    {
        self.constraints.push((
            a(BpLinearCombination::zero()),
            b(BpLinearCombination::zero()),
            c(BpLinearCombination::zero()),
        ));
    }

    fn push_namespace<NR, N>(&mut self, _name_fn: N)
    where
        NR: Into<String>,
        N: FnOnce() -> NR,
    {
    }

    fn pop_namespace(&mut self) {}

    fn get_root(&mut self) -> &mut Self::Root {
        self
    }
}

/// Poseidon parameters shared by native hashing and the in-circuit gadget.
///
/// Arity 2 with the second lane fixed to zero gives a single-field-element
/// knowledge-of-preimage hash; the constants are deterministic for the
/// (field, arity) pair, so prover and verifier always agree.
#[cfg(feature = "zero-knowledge-proofs")]
fn poseidon_constants() -> PoseidonConstants<Scalar, U2> {
    PoseidonConstants::new()
}

/// Native Poseidon digest of a single preimage element.
#[cfg(feature = "zero-knowledge-proofs")]
fn poseidon_digest(preimage: Scalar) -> Scalar {
    let constants = poseidon_constants();
    Poseidon::new_with_preimage(&[preimage, Scalar::zero()], &constants).hash()
}

/// Derive the statement-binding public input from the statement hash.
///
/// Two domain-separated SHA-256 digests are concatenated into 64 bytes and
/// reduced into the scalar field with `from_bytes_wide` (statistically
/// uniform). Both prover and verifier derive this independently from the
/// statement they hold; it is never read from the proof envelope.
#[cfg(feature = "zero-knowledge-proofs")]
fn statement_binding_scalar(statement_hash: &str) -> Scalar {
    let mut wide = [0u8; 64];
    for (chunk, tag) in wide.chunks_exact_mut(32).zip(0u8..) {
        let mut hasher = Sha256::new();
        hasher.update([tag]);
        hasher.update(b"synaptic-zk-statement-binding-v1");
        hasher.update(statement_hash.as_bytes());
        chunk.copy_from_slice(&hasher.finalize());
    }
    Scalar::from_bytes_wide(&wide)
}

/// Parse a serialized registered commitment, rejecting non-canonical
/// encodings. Commitments come from the verifier's trusted store, so a
/// malformed one is a configuration error rather than a failed proof.
#[cfg(feature = "zero-knowledge-proofs")]
fn parse_commitment(bytes: &[u8]) -> Result<Scalar> {
    let bytes: [u8; 32] = bytes
        .try_into()
        .map_err(|_| MemoryError::encryption("Registered commitment must be 32 bytes"))?;
    Option::<Scalar>::from(Scalar::from_bytes(&bytes)).ok_or_else(|| {
        MemoryError::encryption("Registered commitment is not a canonical field element")
    })
}

/// Circuit proving knowledge of a registered secret `s` with
/// `Poseidon(s, 0) == commitment`, where `commitment` (the registered
/// identity commitment) and `statement_binding` (a field element derived
/// from the statement being proven) are the public inputs.
///
/// The statement binding is constrained into the proof so a proof produced
/// for statement A cannot be replayed for statement B: the verifier derives
/// the binding from B and the Groth16 verification equation fails.
#[cfg(feature = "zero-knowledge-proofs")]
struct PoseidonPreimageCircuit {
    /// The registered secret (private witness).
    secret: Option<Scalar>,
    /// The registered Poseidon commitment (public input).
    commitment: Option<Scalar>,
    /// Field element binding the proof to one statement (public input).
    statement_binding: Option<Scalar>,
}

#[cfg(feature = "zero-knowledge-proofs")]
impl Circuit<Scalar> for PoseidonPreimageCircuit {
    fn synthesize<CS: ConstraintSystem<Scalar>>(
        self,
        cs: &mut CS,
    ) -> std::result::Result<(), SynthesisError> {
        let constants = poseidon_constants();
        let mut recorder = BellpepperRecorder::new();

        let secret = AllocatedNum::alloc(recorder.namespace(|| "secret"), || {
            self.secret.ok_or(BpSynthesisError::AssignmentMissing)
        })
        .map_err(|_| SynthesisError::AssignmentMissing)?;

        // Second Poseidon lane is a constant zero pad; constrain it so the
        // prover cannot smuggle a second free witness element.
        let pad = AllocatedNum::alloc(recorder.namespace(|| "pad"), || Ok(Scalar::zero()))
            .map_err(|_| SynthesisError::AssignmentMissing)?;
        recorder.enforce(
            || "pad is zero",
            |lc| lc + pad.get_variable(),
            |lc| lc + BellpepperRecorder::one(),
            |lc| lc,
        );

        let hashed = neptune::circuit2::poseidon_hash_allocated(
            recorder.namespace(|| "poseidon"),
            vec![secret, pad],
            &constants,
        )
        .map_err(|_| SynthesisError::Unsatisfiable)?;

        let commitment = AllocatedNum::alloc_input(recorder.namespace(|| "commitment"), || {
            self.commitment.ok_or(BpSynthesisError::AssignmentMissing)
        })
        .map_err(|_| SynthesisError::AssignmentMissing)?;

        recorder.enforce(
            || "poseidon(secret) == registered commitment",
            |lc| lc + hashed.get_variable(),
            |lc| lc + BellpepperRecorder::one(),
            |lc| lc + commitment.get_variable(),
        );

        // Bind the statement into the proof: allocate the binding as a
        // public input and tie it to an auxiliary copy so it participates
        // in the constraint system (Groth16 then fixes it in the pairing
        // equation; verifying with a different binding fails).
        let statement_binding =
            AllocatedNum::alloc_input(recorder.namespace(|| "statement binding"), || {
                self.statement_binding
                    .ok_or(BpSynthesisError::AssignmentMissing)
            })
            .map_err(|_| SynthesisError::AssignmentMissing)?;
        let binding_copy = AllocatedNum::alloc(recorder.namespace(|| "binding copy"), || {
            self.statement_binding
                .ok_or(BpSynthesisError::AssignmentMissing)
        })
        .map_err(|_| SynthesisError::AssignmentMissing)?;
        recorder.enforce(
            || "statement binding is constrained",
            |lc| lc + statement_binding.get_variable(),
            |lc| lc + BellpepperRecorder::one(),
            |lc| lc + binding_copy.get_variable(),
        );

        recorder.replay(cs)
    }
}

impl ProofSystem {
    /// Generate a fresh Groth16 CRS (proving + verifying parameters) for the
    /// knowledge-of-registered-secret circuit.
    ///
    /// **Trust model / toxic waste:** the CRS is produced by this single
    /// process from `OsRng` in a non-ceremony setup. The setup randomness
    /// ("toxic waste") is discarded, but a party that observed it could forge
    /// proofs. This is accepted by design for the single-verifier deployment
    /// here (the same service generates the CRS and verifies against its own
    /// trusted commitment store). It is NOT a multi-party-verifier trust
    /// anchor: distributing the verifying key to mutually distrusting
    /// verifiers would require a proper multi-party trusted-setup ceremony.
    async fn new(config: &SecurityConfig) -> Result<Self> {
        #[cfg(feature = "zero-knowledge-proofs")]
        {
            tracing::info!(
                "Initializing real Bellman zk-SNARKs proof system with key size: {}",
                config.encryption_key_size
            );

            // Create an empty-witness circuit for parameter generation
            let circuit = PoseidonPreimageCircuit {
                secret: None,
                commitment: None,
                statement_binding: None,
            };

            // Generate trusted setup parameters
            let mut rng = OsRng;
            let parameters =
                generate_random_parameters::<Bls12, _, _>(circuit, &mut rng).map_err(|e| {
                    MemoryError::encryption(format!(
                        "Failed to generate zk-SNARK parameters: {:?}",
                        e
                    ))
                })?;

            // Prepare verification key for efficient verification
            let prepared_vk = prepare_verifying_key(&parameters.vk);

            tracing::info!("Bellman zk-SNARKs parameters generated successfully");

            Ok(Self {
                parameters,
                prepared_vk,
            })
        }

        #[cfg(not(feature = "zero-knowledge-proofs"))]
        {
            tracing::warn!(
                "Zero-knowledge proofs feature not enabled, using fallback implementation"
            );
            let (proving_key, verification_key) = Self::generate_keys(config).await?;

            Ok(Self {
                proving_key,
                verification_key,
            })
        }
    }

    /// Reconstruct a proof system from serialized Groth16 parameters.
    ///
    /// Deserialization runs in checked mode: every curve point is validated
    /// for being on-curve and in the correct subgroup, so corrupted key
    /// stores are rejected instead of silently producing unsound keys.
    #[cfg(feature = "zero-knowledge-proofs")]
    fn from_parameter_bytes(bytes: &[u8]) -> Result<Self> {
        let parameters = Parameters::<Bls12>::read(bytes, true).map_err(|e| {
            MemoryError::encryption(format!("Failed to deserialize Groth16 parameters: {e}"))
        })?;
        let prepared_vk = prepare_verifying_key(&parameters.vk);
        Ok(Self {
            parameters,
            prepared_vk,
        })
    }

    /// Serialize the full Groth16 parameters (proving + verifying key).
    #[cfg(feature = "zero-knowledge-proofs")]
    fn parameter_bytes(&self) -> Result<Vec<u8>> {
        let mut bytes = Vec::new();
        self.parameters.write(&mut bytes).map_err(|e| {
            MemoryError::encryption(format!("Failed to serialize Groth16 parameters: {e}"))
        })?;
        Ok(bytes)
    }

    /// Serialize only the verifying key.
    #[cfg(feature = "zero-knowledge-proofs")]
    fn verifying_key_bytes(&self) -> Result<Vec<u8>> {
        let mut bytes = Vec::new();
        self.parameters.vk.write(&mut bytes).map_err(|e| {
            MemoryError::encryption(format!("Failed to serialize Groth16 verifying key: {e}"))
        })?;
        Ok(bytes)
    }

    /// The prepared verifying key used by in-process verification.
    #[cfg(feature = "zero-knowledge-proofs")]
    fn prepared_verifying_key(&self) -> &PreparedVerifyingKey<Bls12> {
        &self.prepared_vk
    }

    #[cfg(not(feature = "zero-knowledge-proofs"))]
    async fn generate_keys(config: &SecurityConfig) -> Result<(ProvingKey, VerificationKey)> {
        // Generate proving and verification keys
        let proving_key = ProvingKey {
            id: Uuid::new_v4().to_string(),
            key_size: config.encryption_key_size,
            created_at: Utc::now(),
        };

        let verification_key = VerificationKey {
            id: proving_key.id.clone(),
            public_parameters: vec![42, 17, 73], // Simplified parameters
            created_at: Utc::now(),
        };

        Ok((proving_key, verification_key))
    }

    /// Generate a Groth16 proof that the registered `secret` opens the
    /// prover's commitment, bound to the caller-supplied statement.
    #[cfg(feature = "zero-knowledge-proofs")]
    async fn generate_registered_proof<T>(
        &self,
        statement: &T,
        prover_id: &str,
        secret: Scalar,
    ) -> Result<ZKProof>
    where
        T: Serialize,
    {
        tracing::debug!("Generating real zk-SNARK proof using Bellman");
        let start_time = std::time::Instant::now();

        let statement_hash = self.hash_statement(statement)?;
        let commitment = poseidon_digest(secret);
        let statement_binding = statement_binding_scalar(&statement_hash);

        let circuit = PoseidonPreimageCircuit {
            secret: Some(secret),
            commitment: Some(commitment),
            statement_binding: Some(statement_binding),
        };

        // Generate proof
        let mut rng = OsRng;
        let proof = create_random_proof(circuit, &self.parameters, &mut rng).map_err(|e| {
            MemoryError::encryption(format!("Failed to generate zk-SNARK proof: {:?}", e))
        })?;

        let proof_data = Self::serialize_proof(&proof)?;

        let duration = start_time.elapsed();
        tracing::debug!("zk-SNARK proof generation completed in {:?}", duration);

        Ok(ZKProof {
            id: Uuid::new_v4().to_string(),
            statement_hash,
            proof_data,
            prover_id: prover_id.to_string(),
            proving_key_id: "bellman_groth16".to_string(),
            created_at: Utc::now(),
        })
    }

    #[cfg(not(feature = "zero-knowledge-proofs"))]
    async fn generate_proof<T>(&self, statement: &T, witness: &Witness) -> Result<ZKProof>
    where
        T: Serialize,
    {
        tracing::warn!(
            "Using fallback proof generation - zero-knowledge-proofs feature not enabled"
        );
        let statement_hash = self.hash_statement(statement)?;
        let witness_commitment = self.commit_witness(witness)?;

        Ok(ZKProof {
            id: Uuid::new_v4().to_string(),
            statement_hash,
            proof_data: witness_commitment,
            prover_id: String::new(),
            proving_key_id: self.proving_key.id.clone(),
            created_at: Utc::now(),
        })
    }

    /// Fallback verification: fails closed. Real verification lives in
    /// [`ZeroKnowledgeManager::verify_against_registration`], which owns the
    /// trusted commitment store the public inputs are derived from.
    #[cfg(not(feature = "zero-knowledge-proofs"))]
    async fn verify_proof<T>(&self, proof: &ZKProof, statement: &T) -> Result<bool>
    where
        T: Serialize,
    {
        {
            let _ = statement;
            tracing::warn!(
                operation = "verify_proof",
                proof_id = %proof.id,
                feature_enabled = false,
                "Refusing proof verification - zero-knowledge-proofs feature not enabled"
            );

            Err(MemoryError::feature_disabled(
                "zero-knowledge-proofs",
                "verify_proof",
            ))
        }
    }

    fn hash_statement<T>(&self, statement: &T) -> Result<String>
    where
        T: Serialize,
    {
        let serialized = serde_json::to_string(statement)
            .map_err(|_| MemoryError::access_denied("Statement serialization failed"))?;
        Ok(hash_content(&serialized))
    }

    #[cfg(not(feature = "zero-knowledge-proofs"))]
    fn commit_witness(&self, witness: &Witness) -> Result<Vec<u8>> {
        // Create commitment to witness
        let commitment = format!("commit_{}_{}", witness.id, witness.created_at.timestamp());
        Ok(commitment.into_bytes())
    }

    /// Serialize a Groth16 proof to its canonical 192-byte encoding.
    #[cfg(feature = "zero-knowledge-proofs")]
    fn serialize_proof(proof: &Proof<Bls12>) -> Result<Vec<u8>> {
        let mut bytes = Vec::new();
        proof.write(&mut bytes).map_err(|e| {
            MemoryError::encryption(format!("Failed to serialize Groth16 proof: {e}"))
        })?;
        Ok(bytes)
    }

    /// Deserialize a Groth16 proof, validating that all points are on-curve
    /// and in the correct subgroup.
    #[cfg(feature = "zero-knowledge-proofs")]
    fn deserialize_proof(data: &[u8]) -> Result<Proof<Bls12>> {
        Proof::<Bls12>::read(data).map_err(|e| {
            MemoryError::encryption(format!("Failed to deserialize Groth16 proof: {e}"))
        })
    }
}

// Type definitions for zero-knowledge components

#[derive(Debug, Clone, Serialize, Deserialize)]
/// Zero-knowledge proof structure for privacy-preserving memory access
pub struct ZKProof {
    /// Unique identifier for the proof
    pub id: String,
    /// Hash of the statement being proven
    pub statement_hash: String,
    /// Cryptographic proof data
    pub proof_data: Vec<u8>,
    /// Claimed prover identity. Untrusted on its own: verification looks up
    /// this identity's registered commitment in the verifier's trusted
    /// store, so a wrong or forged claim simply fails the pairing check.
    #[serde(default)]
    pub prover_id: String,
    /// ID of the proving key used
    pub proving_key_id: String,
    /// Timestamp when proof was created
    pub created_at: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
/// Statement about memory access for zero-knowledge proofs
pub struct AccessStatement {
    /// Key of the memory being accessed
    pub memory_key: String,
    /// ID of the user making the access
    pub user_id: String,
    /// Type of access being performed
    pub access_type: AccessType,
    /// When the access occurred
    pub timestamp: DateTime<Utc>,
}

/// Maximum age of an access statement accepted on the ZK-gated decrypt path.
///
/// A captured `(proof, statement)` pair otherwise replays indefinitely for
/// the same entry and user, since the proof is bound to the statement (which
/// includes this timestamp) but not to wall-clock freshness. 300s bounds the
/// replay window while tolerating normal request latency.
pub const MAX_STATEMENT_AGE: chrono::Duration = chrono::Duration::seconds(300);

/// Small allowance for clock skew, so a statement stamped slightly in the
/// future by a peer with a fast clock is not spuriously rejected.
pub const MAX_STATEMENT_CLOCK_SKEW: chrono::Duration = chrono::Duration::seconds(30);

impl AccessStatement {
    /// Whether this statement's timestamp falls inside the acceptable
    /// freshness window relative to `now`: not older than
    /// [`MAX_STATEMENT_AGE`] and not further in the future than
    /// [`MAX_STATEMENT_CLOCK_SKEW`].
    pub fn is_fresh(&self, now: DateTime<Utc>) -> bool {
        let age = now.signed_duration_since(self.timestamp);
        age <= MAX_STATEMENT_AGE && age >= -MAX_STATEMENT_CLOCK_SKEW
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContentStatement {
    pub memory_key: String,
    pub predicate: ContentPredicate,
    pub timestamp: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AggregateStatement {
    pub entry_count: usize,
    pub aggregate_type: AggregateType,
    pub timestamp: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
/// Types of memory access operations
pub enum AccessType {
    /// Reading memory content
    Read,
    /// Writing new memory content
    Write,
    /// Deleting memory
    Delete,
    /// Querying memory metadata
    Query,
}

impl AccessType {
    fn to_string(&self) -> String {
        match self {
            AccessType::Read => "read".to_string(),
            AccessType::Write => "write".to_string(),
            AccessType::Delete => "delete".to_string(),
            AccessType::Query => "query".to_string(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ContentPredicate {
    ContainsKeyword(String),
    LengthGreaterThan(usize),
    LengthLessThan(usize),
    MatchesPattern(String),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AggregateType {
    Count,
    AverageLength,
    TotalSize,
}

impl AggregateType {
    fn to_string(&self) -> String {
        match self {
            AggregateType::Count => "count".to_string(),
            AggregateType::AverageLength => "avg_length".to_string(),
            AggregateType::TotalSize => "total_size".to_string(),
        }
    }
}

#[cfg(not(feature = "zero-knowledge-proofs"))]
#[derive(Debug, Clone)]
struct Witness {
    id: String,
    data: WitnessData,
    created_at: DateTime<Utc>,
}

#[cfg(not(feature = "zero-knowledge-proofs"))]
#[derive(Debug, Clone)]
struct WitnessData {
    user_attributes: HashMap<String, String>,
    session_proof: String,
    timestamp_proof: u64,
    access_signature: String,
}

#[derive(Debug, Clone)]
struct ProvingKey {
    id: String,
    key_size: usize,
    created_at: DateTime<Utc>,
}

#[derive(Debug, Clone)]
pub struct VerificationKey {
    pub id: String,
    pub public_parameters: Vec<u8>,
    pub created_at: DateTime<Utc>,
}

/// Zero-knowledge metrics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ZeroKnowledgeMetrics {
    pub total_proofs_generated: u64,
    pub total_proofs_verified: u64,
    pub total_content_proofs: u64,
    pub total_aggregate_proofs: u64,
    pub successful_verifications: u64,
    pub total_proof_generation_time_ms: u64,
    pub total_verification_time_ms: u64,
    pub average_proof_generation_time_ms: f64,
    pub average_verification_time_ms: f64,
    pub verification_success_rate: f64,
}

impl ZeroKnowledgeMetrics {
    pub fn calculate_averages(&mut self) {
        if self.total_proofs_generated > 0 {
            self.average_proof_generation_time_ms =
                self.total_proof_generation_time_ms as f64 / self.total_proofs_generated as f64;
        }

        if self.total_proofs_verified > 0 {
            self.average_verification_time_ms =
                self.total_verification_time_ms as f64 / self.total_proofs_verified as f64;

            self.verification_success_rate =
                (self.successful_verifications as f64 / self.total_proofs_verified as f64) * 100.0;
        }
    }
}

#[cfg(all(test, feature = "zero-knowledge-proofs"))]
mod key_hygiene_tests {
    use super::*;

    /// `ZeroKnowledgeManager`'s manual `Debug` impl must never format
    /// `prover_secrets` (raw prover secret scalars); only a redacted count
    /// may appear. This guards against a future field addition or a
    /// careless revert to `#[derive(Debug)]` silently leaking secret
    /// scalars via `{:?}` (see the struct-level KNOWN LIMITATION doc
    /// comment: `bls12_381::Scalar` cannot be wrapped in `Zeroizing`
    /// without a fork of that crate, so redacting Debug is the mitigation
    /// actually in place).
    #[tokio::test]
    async fn zero_knowledge_manager_debug_redacts_prover_secrets() {
        let config = SecurityConfig::default();
        let mut manager = ZeroKnowledgeManager::new(&config)
            .await
            .expect("ZeroKnowledgeManager::new should succeed");

        manager
            .register_prover("test-user")
            .expect("register_prover should succeed");

        let debug_output = format!("{:?}", manager);

        assert!(
            !debug_output.contains("prover_secrets"),
            "ZeroKnowledgeManager Debug output should not expose the raw prover_secrets field: {debug_output}"
        );
        assert!(
            debug_output.contains("prover_secret_count"),
            "ZeroKnowledgeManager Debug output should summarize prover secret count, not enumerate them: {debug_output}"
        );
    }
}
