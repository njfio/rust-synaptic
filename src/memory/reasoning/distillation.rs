//! Distillation policy: decides whether write-time fact distillation is live
//! from the configured mode, the resolved reasoner kind, and whether the write
//! path prerequisites (knowledge graph + embeddings) are available. Pure logic,
//! no store or IO, so it is unit-testable in isolation.

/// How write-time fact distillation is activated.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum DistillationMode {
    /// Live when an LLM reasoner is resolved and the write-path prerequisites
    /// (knowledge graph + embeddings) are available; off otherwise. Default.
    #[default]
    Auto,
    /// Require distillation. A hard error at construction when prerequisites
    /// (knowledge graph + embeddings) are absent. If the resolved reasoner is
    /// heuristic (no LLM endpoint resolved), proceeds live anyway with a
    /// one-time warning rather than erroring.
    On,
    /// Never distill (escape hatch; today's raw-value-only behavior).
    Off,
}

/// Which reasoner the constructor resolved for the write path.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResolvedReasonerKind {
    Llm,
    Heuristic,
}

/// Outcome of resolving the distillation policy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DistillationDecision {
    /// Whether the write path should distill facts and tag raw sources.
    pub live: bool,
}

/// Resolve whether distillation is live. `prerequisites_met` is
/// `enable_knowledge_graph && embeddings-available`. Returns `Err` only when
/// `On` is requested but prerequisites are absent, where the caller should
/// fail construction. `On` with a heuristic reasoner (no LLM endpoint
/// resolved) is not an error here; the caller is expected to warn and
/// proceed live.
pub fn resolve_distillation(
    mode: DistillationMode,
    reasoner: ResolvedReasonerKind,
    prerequisites_met: bool,
) -> Result<DistillationDecision, String> {
    match mode {
        DistillationMode::Off => Ok(DistillationDecision { live: false }),
        DistillationMode::Auto => Ok(DistillationDecision {
            live: prerequisites_met && reasoner == ResolvedReasonerKind::Llm,
        }),
        DistillationMode::On => {
            if !prerequisites_met {
                return Err(
                    "DistillationMode::On requires enable_knowledge_graph and the \
                     embeddings feature, but they are not available"
                        .to_string(),
                );
            }
            Ok(DistillationDecision { live: true })
        }
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::{resolve_distillation, DistillationMode, ResolvedReasonerKind};

    // Auto: live only with an LLM reasoner AND prerequisites.
    #[test]
    fn auto_llm_with_prereqs_is_live() {
        let d = resolve_distillation(DistillationMode::Auto, ResolvedReasonerKind::Llm, true)
            .unwrap();
        assert!(d.live);
    }

    #[test]
    fn auto_heuristic_is_off() {
        let d = resolve_distillation(
            DistillationMode::Auto,
            ResolvedReasonerKind::Heuristic,
            true,
        )
        .unwrap();
        assert!(!d.live);
    }

    #[test]
    fn auto_llm_without_prereqs_is_off() {
        let d = resolve_distillation(DistillationMode::Auto, ResolvedReasonerKind::Llm, false)
            .unwrap();
        assert!(!d.live);
    }

    // Off: never live, regardless of everything else.
    #[test]
    fn off_is_never_live() {
        let d = resolve_distillation(DistillationMode::Off, ResolvedReasonerKind::Llm, true)
            .unwrap();
        assert!(!d.live);
    }

    // On: hard error when prerequisites are absent.
    #[test]
    fn on_without_prereqs_errors() {
        let err = resolve_distillation(
            DistillationMode::On,
            ResolvedReasonerKind::Heuristic,
            false,
        );
        assert!(err.is_err());
    }

    // On with a heuristic reasoner (no LLM endpoint resolved) but prerequisites
    // met: proceed live rather than erroring; the caller warns.
    #[test]
    fn on_with_heuristic_reasoner_is_live() {
        let d = resolve_distillation(
            DistillationMode::On,
            ResolvedReasonerKind::Heuristic,
            true,
        )
        .unwrap();
        assert!(d.live);
    }

    #[test]
    fn on_llm_with_prereqs_is_live() {
        let d = resolve_distillation(DistillationMode::On, ResolvedReasonerKind::Llm, true)
            .unwrap();
        assert!(d.live);
    }
}
