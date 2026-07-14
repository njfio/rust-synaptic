//! Deterministic query understanding / decomposition.
//!
//! A query-preprocessing stage that runs before retrieval:
//!
//! 1. **Temporal-constraint extraction** — explicit years ("in 2021",
//!    "before 2020", "between 2019 and 2021") and simple relative
//!    expressions ("last year", "last summer") are parsed into a
//!    [`TemporalConstraint`] date range. The hybrid pipeline uses it to
//!    BOOST memories whose creation time falls in the range or whose
//!    content mentions a constrained year.
//! 2. **Multi-part splitting** — questions joined by "and" + a fresh
//!    interrogative clause ("where does Alice live and where does Bob
//!    work?") or by multiple "?" are split into sub-queries; the pipeline
//!    runs each and unions the results (dedup, best score wins).
//!
//! Everything here is rule-based and deterministic — no LLM required. When
//! a query has no temporal expression and a single part, the resulting
//! [`QueryPlan`] is a passthrough and the pipeline behaves exactly as
//! before. (Harder, LLM-assisted decomposition can layer on top of this
//! plan type later; the deterministic path is always available.)

use crate::memory::types::MemoryEntry;
use chrono::{DateTime, Datelike, TimeZone, Utc};
use regex::Regex;
use std::sync::OnceLock;

/// A temporal constraint extracted from a query: a half-open UTC range
/// `[start, end)` plus the literal year tokens that produced it (used to
/// match content-dated memories that merely mention the year).
#[derive(Debug, Clone, PartialEq)]
pub struct TemporalConstraint {
    /// Inclusive range start.
    pub start: DateTime<Utc>,
    /// Exclusive range end.
    pub end: DateTime<Utc>,
    /// Literal 4-digit year tokens covered by the constraint (e.g.
    /// `["2021"]`), matched against memory content as whole tokens.
    pub year_tokens: Vec<String>,
}

impl TemporalConstraint {
    /// Whether a memory satisfies the constraint: its creation time falls
    /// inside `[start, end)`, or its content mentions one of the
    /// constrained years as a standalone token.
    pub fn matches(&self, entry: &MemoryEntry) -> bool {
        let created = entry.created_at();
        if created >= self.start && created < self.end {
            return true;
        }
        if !self.year_tokens.is_empty() {
            let content = &entry.value;
            for year in &self.year_tokens {
                // Whole-token match: the year must not be embedded in a
                // longer digit run (e.g. "12021").
                let mut search_from = 0;
                while let Some(pos) = content[search_from..].find(year.as_str()) {
                    let abs = search_from + pos;
                    let before_ok = abs == 0
                        || !content[..abs]
                            .chars()
                            .next_back()
                            .is_some_and(|c| c.is_ascii_digit());
                    let after = abs + year.len();
                    let after_ok = after >= content.len()
                        || !content[after..]
                            .chars()
                            .next()
                            .is_some_and(|c| c.is_ascii_digit());
                    if before_ok && after_ok {
                        return true;
                    }
                    search_from = after;
                }
            }
        }
        false
    }
}

/// The result of query understanding: one or more sub-queries plus an
/// optional temporal constraint. A passthrough plan (single sub-query equal
/// to the original, no constraint) leaves pipeline behavior unchanged.
#[derive(Debug, Clone, PartialEq)]
pub struct QueryPlan {
    /// The original, unmodified query.
    pub original: String,
    /// The retrieval units (the original query when no split applies).
    pub sub_queries: Vec<String>,
    /// Extracted temporal constraint, if the query contains one.
    pub temporal: Option<TemporalConstraint>,
}

/// Maximum number of sub-queries a single question is split into.
const MAX_SUB_QUERIES: usize = 4;

/// 4-digit years accepted as temporal references (1900–2099).
fn year_re() -> &'static Regex {
    static YEAR_RE: OnceLock<Regex> = OnceLock::new();
    YEAR_RE.get_or_init(|| {
        // Invariant: the pattern is a compile-time constant known to be valid.
        Regex::new(r"\b(19\d{2}|20\d{2})\b").expect("static year regex must compile")
    })
}

/// Interrogative / auxiliary words that mark the start of a fresh clause
/// after "and" (so "salt and pepper" does not split but "... and where does
/// Bob work" does).
const CLAUSE_STARTERS: &[&str] = &[
    "where", "what", "when", "who", "whom", "whose", "which", "why", "how", "did", "does", "do",
    "is", "are", "was", "were", "will", "can", "could", "has", "have", "had",
];

impl QueryPlan {
    /// Analyze a query with the current wall-clock time as the reference
    /// for relative expressions ("last year", "last summer").
    pub fn analyze(query: &str) -> Self {
        Self::analyze_at(query, Utc::now())
    }

    /// Analyze a query with an explicit reference time (deterministic for
    /// tests and reproducible ablations).
    pub fn analyze_at(query: &str, now: DateTime<Utc>) -> Self {
        let temporal = extract_temporal(query, now);
        let sub_queries = split_multi_part(query);
        Self {
            original: query.to_string(),
            sub_queries,
            temporal,
        }
    }

    /// A plan that leaves the pipeline behavior unchanged (used when query
    /// understanding is disabled).
    pub fn passthrough(query: &str) -> Self {
        Self {
            original: query.to_string(),
            sub_queries: vec![query.to_string()],
            temporal: None,
        }
    }

    /// True when the plan is a no-op: a single sub-query identical to the
    /// original and no temporal constraint.
    pub fn is_passthrough(&self) -> bool {
        self.temporal.is_none()
            && self.sub_queries.len() == 1
            && self.sub_queries[0] == self.original
    }
}

/// Build a UTC instant at midnight on Jan 1 of `year`.
fn year_start(year: i32) -> DateTime<Utc> {
    Utc.with_ymd_and_hms(year, 1, 1, 0, 0, 0)
        .single()
        .unwrap_or(DateTime::<Utc>::MIN_UTC)
}

/// Build a UTC instant at midnight on the 1st of `month` in `year`.
fn month_start(year: i32, month: u32) -> DateTime<Utc> {
    Utc.with_ymd_and_hms(year, month, 1, 0, 0, 0)
        .single()
        .unwrap_or(DateTime::<Utc>::MIN_UTC)
}

/// Extract a temporal constraint from the query, if any.
///
/// Rules (first match category wins, explicit years before relative):
/// - `before/until/by <year>`  -> `[MIN, year)`
/// - `after/since <year>`      -> `[year, MAX)`
/// - one or more bare years    -> `[min_year, max_year + 1)`
/// - `last year`               -> the previous calendar year
/// - `last month` / `last week` / `yesterday` -> that trailing window
/// - `last summer/winter/spring/fall/autumn`  -> most recent completed season
fn extract_temporal(query: &str, now: DateTime<Utc>) -> Option<TemporalConstraint> {
    let lower = query.to_lowercase();

    // Explicit years.
    let years: Vec<(usize, i32)> = year_re()
        .find_iter(&lower)
        .filter_map(|m| m.as_str().parse::<i32>().ok().map(|y| (m.start(), y)))
        .collect();

    if !years.is_empty() {
        let year_tokens: Vec<String> = years.iter().map(|(_, y)| y.to_string()).collect();

        // Modifier immediately preceding the FIRST year (possibly with an
        // intervening article/preposition is not supported — the word
        // directly before the year decides).
        let (first_pos, first_year) = years[0];
        let preceding_word = lower[..first_pos]
            .split_whitespace()
            .next_back()
            .unwrap_or("");

        let (start, end) = match preceding_word {
            "before" | "until" | "by" => (DateTime::<Utc>::MIN_UTC, year_start(first_year)),
            "after" | "since" => (year_start(first_year), DateTime::<Utc>::MAX_UTC),
            _ => {
                let min_year = years.iter().map(|(_, y)| *y).min().unwrap_or(first_year);
                let max_year = years.iter().map(|(_, y)| *y).max().unwrap_or(first_year);
                (year_start(min_year), year_start(max_year + 1))
            }
        };
        return Some(TemporalConstraint {
            start,
            end,
            year_tokens,
        });
    }

    // Relative expressions (no literal year token to content-match).
    let relative = |start: DateTime<Utc>, end: DateTime<Utc>| {
        Some(TemporalConstraint {
            start,
            end,
            year_tokens: Vec::new(),
        })
    };

    if lower.contains("last year") {
        let y = now.year() - 1;
        return relative(year_start(y), year_start(y + 1));
    }
    if lower.contains("last month") {
        let (y, m) = if now.month() == 1 {
            (now.year() - 1, 12)
        } else {
            (now.year(), now.month() - 1)
        };
        let next = if m == 12 { (y + 1, 1) } else { (y, m + 1) };
        return relative(month_start(y, m), month_start(next.0, next.1));
    }
    if lower.contains("last week") {
        return relative(now - chrono::Duration::days(7), now);
    }
    if lower.contains("yesterday") {
        let today = now.date_naive().and_hms_opt(0, 0, 0).map(|d| d.and_utc());
        if let Some(today) = today {
            return relative(today - chrono::Duration::days(1), today);
        }
    }

    // Seasons (northern-hemisphere calendar convention): the most recent
    // season of that name that has fully COMPLETED before `now`.
    let season = [
        ("last spring", 3u32, 6u32),
        ("last summer", 6, 9),
        ("last fall", 9, 12),
        ("last autumn", 9, 12),
        ("last winter", 12, 3),
    ]
    .iter()
    .find(|(phrase, _, _)| lower.contains(phrase));
    if let Some((_, start_month, end_month)) = season {
        // Winter spans the year boundary (Dec 1 .. Mar 1).
        let mut year = now.year();
        loop {
            let (start, end) = if start_month > end_month {
                (
                    month_start(year - 1, *start_month),
                    month_start(year, *end_month),
                )
            } else {
                (
                    month_start(year, *start_month),
                    month_start(year, *end_month),
                )
            };
            if end <= now {
                return relative(start, end);
            }
            year -= 1;
        }
    }

    None
}

/// Split a multi-part question into sub-queries.
///
/// Rules:
/// 1. Multiple "?"-terminated segments each become a sub-query.
/// 2. Within a segment, " and " splits ONLY when the words after it start a
///    fresh interrogative clause (wh-word or auxiliary verb) — so entity
///    conjunctions ("salt and pepper") stay intact.
///
/// Returns `vec![query]` when no split applies. Sub-queries are capped at
/// [`MAX_SUB_QUERIES`]; each must have at least two word tokens.
fn split_multi_part(query: &str) -> Vec<String> {
    // Pass 1: split on '?' boundaries.
    let segments: Vec<&str> = query
        .split('?')
        .map(str::trim)
        .filter(|s| s.split_whitespace().count() >= 2)
        .collect();
    let segments = if segments.is_empty() {
        vec![query.trim()]
    } else {
        segments
    };

    // Pass 2: split each segment on clause-starting "and".
    let mut parts: Vec<String> = Vec::new();
    for segment in segments {
        for piece in split_on_clause_and(segment) {
            if parts.len() >= MAX_SUB_QUERIES {
                break;
            }
            parts.push(piece);
        }
    }

    if parts.len() <= 1 {
        vec![query.to_string()]
    } else {
        parts
    }
}

/// Split a single segment on occurrences of " and " that are followed by a
/// clause-starter word; both resulting sides must keep >= 2 word tokens.
fn split_on_clause_and(segment: &str) -> Vec<String> {
    let words: Vec<&str> = segment.split_whitespace().collect();
    let mut boundaries: Vec<usize> = Vec::new(); // index of the word AFTER "and"

    for (i, word) in words.iter().enumerate() {
        if !word.eq_ignore_ascii_case("and") {
            continue;
        }
        let Some(next) = words.get(i + 1) else {
            continue;
        };
        let next_norm = next
            .trim_matches(|c: char| !c.is_alphanumeric())
            .to_lowercase();
        if CLAUSE_STARTERS.contains(&next_norm.as_str()) {
            boundaries.push(i);
        }
    }

    if boundaries.is_empty() {
        return vec![segment.trim().to_string()];
    }

    let mut pieces: Vec<String> = Vec::new();
    let mut start = 0usize;
    for &b in &boundaries {
        if b.saturating_sub(start) >= 2 && words.len().saturating_sub(b + 1) >= 2 {
            pieces.push(words[start..b].join(" "));
            start = b + 1; // skip the "and" itself
        }
    }
    pieces.push(words[start..].join(" "));

    if pieces.len() <= 1 {
        vec![segment.trim().to_string()]
    } else {
        pieces
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::types::MemoryType;

    #[test]
    fn passthrough_for_plain_query() {
        let plan = QueryPlan::analyze("kubernetes deployment notes");
        assert!(plan.is_passthrough());
    }

    #[test]
    fn year_range_between() {
        let plan = QueryPlan::analyze("events between 2019 and 2021");
        let t = plan.temporal.expect("span must parse");
        assert_eq!(t.start.year(), 2019);
        assert_eq!(t.end.year(), 2022);
        assert_eq!(t.year_tokens, vec!["2019", "2021"]);
    }

    #[test]
    fn winter_spans_year_boundary() {
        let now = Utc
            .with_ymd_and_hms(2026, 7, 1, 0, 0, 0)
            .single()
            .expect("valid ts");
        let plan = QueryPlan::analyze_at("what happened last winter", now);
        let t = plan.temporal.expect("last winter must parse");
        assert_eq!((t.start.year(), t.start.month()), (2025, 12));
        assert_eq!((t.end.year(), t.end.month()), (2026, 3));
    }

    #[test]
    fn content_year_token_no_partial_digit_match() {
        let plan = QueryPlan::analyze("what happened in 2021");
        let t = plan.temporal.expect("year must parse");
        let mut entry = MemoryEntry::new(
            "k".into(),
            "serial 120215 logged".into(),
            MemoryType::LongTerm,
        );
        entry.metadata.created_at = Utc
            .with_ymd_and_hms(2010, 1, 1, 0, 0, 0)
            .single()
            .expect("valid ts");
        assert!(!t.matches(&entry), "embedded digit run must not match");
    }

    #[test]
    fn split_caps_at_max() {
        let plan = QueryPlan::analyze(
            "where does a live and where does b live and where does c live and where does d live and where does e live",
        );
        assert!(plan.sub_queries.len() <= MAX_SUB_QUERIES);
        assert!(plan.sub_queries.len() >= 2);
    }
}
