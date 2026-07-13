//! Deterministic heuristic implementation of [`MemoryReasoner`].
//!
//! No models, no randomness: extraction is rule-based (sentence split,
//! capitalized-span NER with a small lexicon, date/number/quote scanners,
//! verb-map relation extraction), conflict resolution is threshold-based
//! over cosine similarity plus extracted relations, and synthesis is an
//! extractive representative-sentence template.

use std::collections::HashMap;
use std::sync::{Arc, RwLock};

use crate::error::Result;
use crate::memory::embeddings::provider::EmbeddingProvider;
use crate::memory::reasoning::{
    ConflictResolution, Entity, EntityKind, Extraction, ExtractionContext, Fact, Insight,
    MemoryReasoner, Relation,
};
use crate::memory::types::MemoryEntry;

/// Similarity at or above which an equal-text candidate is a duplicate.
const DEDUP_THRESHOLD: f64 = 0.95;
/// Similarity at or above which contradiction/refinement checks apply.
const SUPERSEDE_THRESHOLD: f64 = 0.85;

/// Known place names (lowercased) used to classify capitalized spans.
const PLACE_LEXICON: &[&str] = &[
    "berlin",
    "munich",
    "hamburg",
    "paris",
    "london",
    "madrid",
    "rome",
    "amsterdam",
    "vienna",
    "zurich",
    "tokyo",
    "kyoto",
    "beijing",
    "shanghai",
    "sydney",
    "toronto",
    "chicago",
    "boston",
    "seattle",
    "new york",
    "san francisco",
    "los angeles",
    "germany",
    "france",
    "spain",
    "italy",
    "japan",
    "china",
    "canada",
    "australia",
    "england",
    "europe",
    "asia",
    "africa",
];

/// Organization suffixes that mark a capitalized span as an [`EntityKind::Org`].
const ORG_SUFFIXES: &[&str] = &[
    "inc",
    "corp",
    "ltd",
    "llc",
    "gmbh",
    "co",
    "company",
    "corporation",
    "foundation",
    "university",
];

/// Titles that cue the following capitalized span as a person.
const PERSON_TITLES: &[&str] = &["mr", "mrs", "ms", "dr", "prof", "sir", "madam"];

/// Verbs that, when immediately following a capitalized span, cue a person.
const PERSON_CUE_VERBS: &[&str] = &[
    "moved",
    "moves",
    "lives",
    "lived",
    "resides",
    "relocated",
    "works",
    "worked",
    "joined",
    "said",
    "says",
    "met",
    "married",
    "visited",
    "likes",
    "loves",
    "improved",
    "was",
    "is",
    "has",
    "had",
    "went",
    "wrote",
    "born",
];

/// Domain terms (lowercased) extracted as [`EntityKind::Term`].
const TERM_LEXICON: &[&str] = &[
    "api",
    "database",
    "server",
    "algorithm",
    "protocol",
    "compiler",
    "kernel",
];

/// Common sentence-initial words that are never treated as entity starts.
const STOPWORDS: &[&str] = &[
    "the",
    "a",
    "an",
    "it",
    "he",
    "she",
    "they",
    "we",
    "i",
    "you",
    "this",
    "that",
    "these",
    "those",
    "there",
    "here",
    "his",
    "her",
    "its",
    "their",
    "our",
    "in",
    "on",
    "at",
    "after",
    "before",
    "when",
    "while",
    "and",
    "but",
    "or",
    "so",
    "if",
    "then",
    "however",
    "meanwhile",
    "yesterday",
    "today",
    "tomorrow",
];

/// Month names for `Month DD, YYYY` date extraction.
const MONTHS: &[&str] = &[
    "january",
    "february",
    "march",
    "april",
    "may",
    "june",
    "july",
    "august",
    "september",
    "october",
    "november",
    "december",
];

/// Verb-phrase normalization map: gap text between two entities -> predicate.
///
/// Ordered by descending phrase length so the longest suffix match wins.
const VERB_MAP: &[(&str, &str)] = &[
    ("was born in", "born_in"),
    ("is employed by", "works_at"),
    ("relocated to", "lives_in"),
    ("moved to", "lives_in"),
    ("moves to", "lives_in"),
    ("lives in", "lives_in"),
    ("lived in", "lives_in"),
    ("resides in", "lives_in"),
    ("born in", "born_in"),
    ("works at", "works_at"),
    ("works for", "works_at"),
    ("worked at", "works_at"),
    ("worked for", "works_at"),
    ("employed by", "works_at"),
    ("joined", "works_at"),
    ("married", "married_to"),
    ("visited", "visited"),
    ("met", "met"),
    ("founded", "founded"),
    ("leads", "leads"),
    ("manages", "manages"),
];

/// A token in a sentence with its byte span in the source text.
#[derive(Debug, Clone)]
struct Token {
    text: String,
    start: usize,
    end: usize,
}

/// Deterministic rule-based [`MemoryReasoner`].
///
/// Keeps a cache of facts previously extracted per `source_key` so that
/// [`MemoryReasoner::resolve`] can compare a candidate's relations against
/// the relations of the neighbor memories it is measured against.
pub struct HeuristicReasoner {
    embedder: Arc<dyn EmbeddingProvider>,
    known_facts: RwLock<HashMap<String, Vec<Fact>>>,
}

impl HeuristicReasoner {
    /// Create a reasoner over the given embedding provider.
    pub fn new(embedder: Arc<dyn EmbeddingProvider>) -> Self {
        Self {
            embedder,
            known_facts: RwLock::new(HashMap::new()),
        }
    }

    /// Split `text` into sentences, returning `(start, end)` byte spans.
    fn split_sentences(text: &str) -> Vec<(usize, usize)> {
        let mut spans = Vec::new();
        let mut start = 0usize;
        let mut prev_end = 0usize;
        for (i, c) in text.char_indices() {
            let at_end = i + c.len_utf8() >= text.len();
            if matches!(c, '.' | '!' | '?') {
                let after = &text[i + c.len_utf8()..];
                if at_end || after.starts_with(char::is_whitespace) {
                    let end = i + c.len_utf8();
                    if text[start..end].trim().is_empty() {
                        start = end;
                        continue;
                    }
                    spans.push((start, end));
                    start = end;
                }
            }
            prev_end = i + c.len_utf8();
        }
        if start < prev_end && !text[start..prev_end].trim().is_empty() {
            spans.push((start, prev_end));
        }
        // Trim leading/trailing whitespace from each span.
        spans
            .into_iter()
            .map(|(s, e)| {
                let seg = &text[s..e];
                let lead = seg.len() - seg.trim_start().len();
                let trail = seg.len() - seg.trim_end().len();
                (s + lead, e - trail)
            })
            .filter(|(s, e)| s < e)
            .collect()
    }

    /// Tokenize a sentence into word tokens with absolute byte spans.
    fn tokenize(text: &str, offset: usize) -> Vec<Token> {
        let mut tokens = Vec::new();
        let mut cur_start: Option<usize> = None;
        let is_word_char = |c: char| c.is_alphanumeric() || c == '-' || c == '\'';
        for (i, c) in text.char_indices() {
            if is_word_char(c) {
                if cur_start.is_none() {
                    cur_start = Some(i);
                }
            } else if let Some(s) = cur_start.take() {
                tokens.push(Token {
                    text: text[s..i].to_string(),
                    start: offset + s,
                    end: offset + i,
                });
            }
        }
        if let Some(s) = cur_start {
            tokens.push(Token {
                text: text[s..].to_string(),
                start: offset + s,
                end: offset + text.len(),
            });
        }
        tokens
    }

    /// True if the token looks like a 4-digit year in a plausible range.
    fn is_year(token: &str) -> bool {
        token.len() == 4 && token.chars().all(|c| c.is_ascii_digit()) && {
            token
                .parse::<u32>()
                .map(|y| (1000..=2999).contains(&y))
                .unwrap_or(false)
        }
    }

    /// True if the token is an ISO `YYYY-MM-DD` date.
    fn is_iso_date(token: &str) -> bool {
        let bytes = token.as_bytes();
        token.len() == 10
            && bytes[4] == b'-'
            && bytes[7] == b'-'
            && token
                .char_indices()
                .all(|(i, c)| matches!(i, 4 | 7) || c.is_ascii_digit())
    }

    /// True if the token is purely numeric (integer or decimal).
    fn is_number(token: &str) -> bool {
        !token.is_empty()
            && token.chars().all(|c| c.is_ascii_digit() || c == '.')
            && token.chars().any(|c| c.is_ascii_digit())
    }

    /// Extract quoted spans (`"..."`) from a sentence.
    fn extract_quotes(sentence: &str, offset: usize, entities: &mut Vec<Entity>) {
        let mut open: Option<usize> = None;
        for (i, c) in sentence.char_indices() {
            if c == '"' {
                match open.take() {
                    None => open = Some(i + c.len_utf8()),
                    Some(start) => {
                        let inner = &sentence[start..i];
                        if !inner.trim().is_empty() {
                            entities.push(Entity {
                                name: inner.to_string(),
                                kind: EntityKind::Quoted,
                                span: (offset + start, offset + i),
                            });
                        }
                    }
                }
            }
        }
    }

    /// Classify a capitalized span using lexicons and person cues.
    fn classify_span(
        tokens: &[Token],
        next_token: Option<&Token>,
        prev_token: Option<&Token>,
    ) -> EntityKind {
        let last = tokens[tokens.len() - 1]
            .text
            .trim_end_matches('.')
            .to_lowercase();
        if ORG_SUFFIXES.contains(&last.as_str()) {
            return EntityKind::Org;
        }
        let joined = tokens
            .iter()
            .map(|t| t.text.to_lowercase())
            .collect::<Vec<_>>()
            .join(" ");
        if PLACE_LEXICON.contains(&joined.as_str()) {
            return EntityKind::Place;
        }
        let titled = prev_token
            .map(|t| PERSON_TITLES.contains(&t.text.trim_end_matches('.').to_lowercase().as_str()))
            .unwrap_or(false);
        let verb_cued = next_token
            .map(|t| PERSON_CUE_VERBS.contains(&t.text.to_lowercase().as_str()))
            .unwrap_or(false);
        if titled || verb_cued {
            EntityKind::Person
        } else {
            EntityKind::Term
        }
    }

    /// Extract entities from one sentence (absolute spans into the source text).
    fn extract_entities(sentence: &str, offset: usize) -> Vec<Entity> {
        let mut entities = Vec::new();
        Self::extract_quotes(sentence, offset, &mut entities);
        let tokens = Self::tokenize(sentence, offset);
        let mut used = vec![false; tokens.len()];

        // Mark tokens inside quoted spans as used so they are not re-extracted.
        for e in &entities {
            for (i, t) in tokens.iter().enumerate() {
                if t.start >= e.span.0 && t.end <= e.span.1 {
                    used[i] = true;
                }
            }
        }

        // Dates: "Month DD, YYYY" (three tokens), ISO dates, bare years.
        let mut i = 0;
        while i < tokens.len() {
            if used[i] {
                i += 1;
                continue;
            }
            let lower = tokens[i].text.to_lowercase();
            if MONTHS.contains(&lower.as_str())
                && i + 2 < tokens.len()
                && tokens[i + 1].text.chars().all(|c| c.is_ascii_digit())
                && tokens[i + 1]
                    .text
                    .parse::<u32>()
                    .map(|d| (1..=31).contains(&d))
                    .unwrap_or(false)
                && Self::is_year(&tokens[i + 2].text)
            {
                let (start, end) = (tokens[i].start, tokens[i + 2].end);
                entities.push(Entity {
                    name: format!(
                        "{} {}, {}",
                        tokens[i].text,
                        tokens[i + 1].text,
                        tokens[i + 2].text
                    ),
                    kind: EntityKind::Date,
                    span: (start, end),
                });
                used[i] = true;
                used[i + 1] = true;
                used[i + 2] = true;
                i += 3;
                continue;
            }
            if Self::is_iso_date(&tokens[i].text) || Self::is_year(&tokens[i].text) {
                entities.push(Entity {
                    name: tokens[i].text.clone(),
                    kind: EntityKind::Date,
                    span: (tokens[i].start, tokens[i].end),
                });
                used[i] = true;
            } else if Self::is_number(&tokens[i].text) {
                entities.push(Entity {
                    name: tokens[i].text.clone(),
                    kind: EntityKind::Number,
                    span: (tokens[i].start, tokens[i].end),
                });
                used[i] = true;
            }
            i += 1;
        }

        // Capitalized spans: consecutive capitalized alphabetic tokens.
        let is_cap = |t: &Token| {
            t.text
                .chars()
                .next()
                .map(char::is_uppercase)
                .unwrap_or(false)
                && t.text
                    .chars()
                    .all(|c| c.is_alphabetic() || c == '-' || c == '\'')
                && !STOPWORDS.contains(&t.text.to_lowercase().as_str())
        };
        let mut i = 0;
        while i < tokens.len() {
            if used[i] || !is_cap(&tokens[i]) {
                i += 1;
                continue;
            }
            let mut j = i + 1;
            while j < tokens.len() && !used[j] && is_cap(&tokens[j]) {
                j += 1;
            }
            let span_tokens = &tokens[i..j];
            let kind = Self::classify_span(
                span_tokens,
                tokens.get(j),
                i.checked_sub(1).and_then(|p| tokens.get(p)),
            );
            let (start, end) = (span_tokens[0].start, span_tokens[span_tokens.len() - 1].end);
            entities.push(Entity {
                name: span_tokens
                    .iter()
                    .map(|t| t.text.as_str())
                    .collect::<Vec<_>>()
                    .join(" "),
                kind,
                span: (start, end),
            });
            for u in used.iter_mut().take(j).skip(i) {
                *u = true;
            }
            i = j;
        }

        // Lexicon terms among the remaining tokens.
        for (i, t) in tokens.iter().enumerate() {
            if !used[i] && TERM_LEXICON.contains(&t.text.to_lowercase().as_str()) {
                entities.push(Entity {
                    name: t.text.clone(),
                    kind: EntityKind::Term,
                    span: (t.start, t.end),
                });
            }
        }

        entities.sort_by_key(|e| e.span);
        entities
    }

    /// Normalize the text gap between two entity spans and map it to a predicate.
    fn normalize_predicate(gap: &str) -> Option<&'static str> {
        let normalized: String = gap
            .to_lowercase()
            .split(|c: char| !c.is_alphanumeric())
            .filter(|w| !w.is_empty())
            .collect::<Vec<_>>()
            .join(" ");
        if normalized.is_empty() {
            return None;
        }
        for (phrase, predicate) in VERB_MAP {
            if normalized == *phrase || normalized.ends_with(&format!(" {phrase}")) {
                return Some(predicate);
            }
        }
        None
    }

    /// Extract subject-verb-object relations over the extracted entities.
    fn extract_relations(text: &str, entities: &[Entity]) -> Vec<Relation> {
        let mut relations = Vec::new();
        for (si, subject) in entities.iter().enumerate() {
            if !matches!(subject.kind, EntityKind::Person | EntityKind::Org) {
                continue;
            }
            for object in entities.iter().skip(si + 1) {
                if !matches!(
                    object.kind,
                    EntityKind::Person | EntityKind::Org | EntityKind::Place
                ) {
                    continue;
                }
                if object.span.0 <= subject.span.1 {
                    continue;
                }
                let gap = &text[subject.span.1..object.span.0];
                if let Some(predicate) = Self::normalize_predicate(gap) {
                    relations.push(Relation {
                        subject: subject.name.clone(),
                        predicate: predicate.to_string(),
                        object: object.name.clone(),
                    });
                }
            }
        }
        relations
    }

    /// Look up cached facts for a memory id, tolerating lock poisoning.
    fn facts_for(&self, id: &str) -> Option<Vec<Fact>> {
        let guard = self
            .known_facts
            .read()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        guard.get(id).cloned()
    }
}

#[async_trait::async_trait]
impl MemoryReasoner for HeuristicReasoner {
    async fn extract(&self, text: &str, ctx: &ExtractionContext) -> Result<Extraction> {
        let mut facts = Vec::new();
        for (start, end) in Self::split_sentences(text) {
            let sentence = &text[start..end];
            let entities = Self::extract_entities(sentence, start);
            let relations = Self::extract_relations(text, &entities);
            facts.push(Fact {
                text: sentence.to_string(),
                entities,
                relations,
            });
        }
        {
            let mut guard = self
                .known_facts
                .write()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            guard.insert(ctx.source_key.clone(), facts.clone());
        }
        Ok(Extraction { facts })
    }

    async fn resolve(
        &self,
        candidate: &Fact,
        neighbors: &[(String, f64)],
    ) -> Result<ConflictResolution> {
        // Deterministic best neighbor: highest similarity, ties broken by id.
        let best = neighbors
            .iter()
            .max_by(|a, b| a.1.total_cmp(&b.1).then_with(|| b.0.cmp(&a.0)));
        let Some((best_id, best_sim)) = best else {
            return Ok(ConflictResolution::Insert);
        };
        let best_sim = *best_sim;
        if best_sim < SUPERSEDE_THRESHOLD {
            return Ok(ConflictResolution::Insert);
        }

        match self.facts_for(best_id) {
            Some(known) => {
                if best_sim >= DEDUP_THRESHOLD && known.iter().any(|f| f.text == candidate.text) {
                    return Ok(ConflictResolution::NoOp {
                        reason: format!("exact duplicate of {best_id} at similarity {best_sim:.2}"),
                    });
                }
                for known_fact in &known {
                    for kr in &known_fact.relations {
                        for cr in &candidate.relations {
                            if kr.subject == cr.subject && kr.predicate == cr.predicate {
                                if kr.object != cr.object {
                                    return Ok(ConflictResolution::Supersede {
                                        old_id: best_id.clone(),
                                        reason: format!(
                                            "contradicts ({}, {}, {}) with new object {}",
                                            kr.subject, kr.predicate, kr.object, cr.object
                                        ),
                                    });
                                }
                                if candidate.text.len() > known_fact.text.len() {
                                    return Ok(ConflictResolution::UpdateInPlace {
                                        reason: format!(
                                            "same fact ({}, {}, {}) with more detail",
                                            cr.subject, cr.predicate, cr.object
                                        ),
                                    });
                                }
                                return Ok(ConflictResolution::NoOp {
                                    reason: format!(
                                        "restates ({}, {}, {}) already in {best_id}",
                                        cr.subject, cr.predicate, cr.object
                                    ),
                                });
                            }
                        }
                    }
                }
                Ok(ConflictResolution::Insert)
            }
            // Neighbor text/relations unknown to this reasoner: fall back to
            // similarity-only thresholds (conservative: never Supersede blind).
            None => {
                if best_sim >= DEDUP_THRESHOLD {
                    Ok(ConflictResolution::NoOp {
                        reason: format!("near-duplicate of {best_id} at similarity {best_sim:.2}"),
                    })
                } else {
                    Ok(ConflictResolution::UpdateInPlace {
                        reason: format!(
                            "high similarity {best_sim:.2} to {best_id} without extracted relations to compare"
                        ),
                    })
                }
            }
        }
    }

    async fn synthesize(&self, cluster: &[MemoryEntry]) -> Result<Option<Insight>> {
        if cluster.len() < 2 {
            return Ok(None);
        }
        let mut embeddings = Vec::with_capacity(cluster.len());
        for entry in cluster {
            embeddings.push(self.embedder.embed(&entry.value, None).await?);
        }
        let n = cluster.len();
        let mut pair_sum = 0.0;
        let mut pair_count = 0usize;
        let mut centroid_sims = vec![0.0f64; n];
        for i in 0..n {
            for j in (i + 1)..n {
                let sim = embeddings[i].cosine_similarity(&embeddings[j]);
                pair_sum += sim;
                pair_count += 1;
                centroid_sims[i] += sim;
                centroid_sims[j] += sim;
            }
        }
        let confidence = (pair_sum / pair_count as f64).clamp(0.0, 1.0);

        // Representative: highest importance, then highest centroid similarity,
        // then lowest index (fully deterministic).
        let rep_index = (0..n)
            .max_by(|&a, &b| {
                cluster[a]
                    .metadata
                    .importance
                    .total_cmp(&cluster[b].metadata.importance)
                    .then(centroid_sims[a].total_cmp(&centroid_sims[b]))
                    .then(b.cmp(&a))
            })
            .unwrap_or(0);
        let rep_value = &cluster[rep_index].value;
        let rep_sentence = Self::split_sentences(rep_value)
            .first()
            .map(|&(s, e)| rep_value[s..e].to_string())
            .unwrap_or_else(|| rep_value.clone());

        Ok(Some(Insight {
            text: format!("Across {n} related memories: {rep_sentence}"),
            derived_from: cluster.iter().map(|e| e.id().to_string()).collect(),
            confidence,
        }))
    }

    fn name(&self) -> &str {
        "heuristic"
    }
}
