//! Memory reasoning: extraction, conflict resolution, and synthesis.
//!
//! Defines the [`MemoryReasoner`] trait plus the data types it produces:
//! entities and relations extracted from text ([`Extraction`]), decisions
//! about how a candidate fact interacts with existing memories
//! ([`ConflictResolution`]), and higher-level [`Insight`]s synthesized
//! across clusters of related memories.

pub mod heuristic;

use crate::error::Result;
use crate::memory::types::MemoryEntry;
use serde::{Deserialize, Serialize};

/// A named entity extracted from text, with its kind and byte span.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Entity {
    /// Surface form of the entity as it appears in the text.
    pub name: String,
    /// The classified kind of the entity.
    pub kind: EntityKind,
    /// Byte span `(start, end)` of the entity in the source text.
    pub span: (usize, usize),
}

/// The kind of an extracted [`Entity`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum EntityKind {
    /// A person's name.
    Person,
    /// A geographic location.
    Place,
    /// An organization.
    Org,
    /// A date expression.
    Date,
    /// A numeric value.
    Number,
    /// A quoted span.
    Quoted,
    /// A domain term from a lexicon.
    Term,
}

/// A subject-predicate-object relation between extracted entities.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Relation {
    /// The subject of the relation.
    pub subject: String,
    /// The normalized predicate.
    pub predicate: String,
    /// The object of the relation.
    pub object: String,
}

/// A single extracted fact: source text plus its entities and relations.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Fact {
    /// The sentence or span the fact was extracted from.
    pub text: String,
    /// Entities found in the fact text.
    pub entities: Vec<Entity>,
    /// Relations connecting the entities.
    pub relations: Vec<Relation>,
}

/// The full result of extracting facts from a piece of text.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Extraction {
    /// All facts extracted from the input text.
    pub facts: Vec<Fact>,
}

/// Context accompanying an extraction request.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ExtractionContext {
    /// Key of the memory the text is being stored under.
    pub source_key: String,
    /// When the text was ingested.
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

/// How a candidate fact should be applied relative to existing memories.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ConflictResolution {
    /// Store the candidate as a new memory.
    Insert,
    /// Update an existing memory in place.
    UpdateInPlace {
        /// Why an in-place update was chosen.
        reason: String,
    },
    /// The candidate supersedes an existing memory.
    Supersede {
        /// Identifier of the memory being superseded.
        old_id: String,
        /// Why the supersession was chosen.
        reason: String,
    },
    /// Append the candidate to an existing memory.
    Append {
        /// Why appending was chosen.
        reason: String,
    },
    /// Do nothing (e.g. exact duplicate).
    NoOp {
        /// Why no action was taken.
        reason: String,
    },
}

/// A synthesized insight derived from a cluster of related memories.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Insight {
    /// The insight text.
    pub text: String,
    /// Identifiers of the memories the insight was derived from.
    pub derived_from: Vec<String>,
    /// Confidence in the insight, in `[0.0, 1.0]`.
    pub confidence: f64,
}

/// Reasoning over memories: extraction, conflict resolution, and synthesis.
///
/// `neighbors` in [`MemoryReasoner::resolve`] is `(memory_id, similarity)`
/// to keep the trait free of retrieval types.
#[async_trait::async_trait]
pub trait MemoryReasoner: Send + Sync {
    /// Extract structured facts from `text`.
    async fn extract(&self, text: &str, ctx: &ExtractionContext) -> Result<Extraction>;
    /// Decide how `candidate` interacts with similar existing memories.
    async fn resolve(
        &self,
        candidate: &Fact,
        neighbors: &[(String, f64)],
    ) -> Result<ConflictResolution>;
    /// Synthesize an insight from a cluster of related memories, if any.
    async fn synthesize(&self, cluster: &[MemoryEntry]) -> Result<Option<Insight>>;
    /// Human-readable name of this reasoner implementation.
    fn name(&self) -> &str;
}
