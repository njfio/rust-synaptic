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
    /// Require distillation. A hard error at construction when prerequisites are
    /// absent, or when the `llm-reasoning` feature is enabled but no endpoint
    /// resolved (a heuristic reasoner where an LLM was expected). In a build
    /// without the feature, proceeds live with a heuristic reasoner and a
    /// one-time warning.
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
/// `enable_knowledge_graph && embeddings-available`; `llm_feature_enabled` is
/// whether the `llm-reasoning` cargo feature is compiled in. Returns `Err` only
/// for the `On` misconfiguration cases, where the caller should fail construction.
pub fn resolve_distillation(
    mode: DistillationMode,
    reasoner: ResolvedReasonerKind,
    prerequisites_met: bool,
    llm_feature_enabled: bool,
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
            // Feature enabled but reasoner came back Heuristic => an endpoint was
            // expected (SYNAPTIC_LLM_URL/_MODEL) but did not resolve.
            if llm_feature_enabled && reasoner == ResolvedReasonerKind::Heuristic {
                return Err(
                    "DistillationMode::On with the llm-reasoning feature requires a \
                     configured LLM endpoint (SYNAPTIC_LLM_URL / SYNAPTIC_LLM_MODEL), \
                     but none resolved"
                        .to_string(),
                );
            }
            Ok(DistillationDecision { live: true })
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{resolve_distillation, DistillationMode, ResolvedReasonerKind};

    // Auto: live only with an LLM reasoner AND prerequisites.
    #[test]
    fn auto_llm_with_prereqs_is_live() {
        let d = resolve_distillation(
            DistillationMode::Auto,
            ResolvedReasonerKind::Llm,
            true,
            true,
        )
        .unwrap();
        assert!(d.live);
    }

    #[test]
    fn auto_heuristic_is_off() {
        let d = resolve_distillation(
            DistillationMode::Auto,
            ResolvedReasonerKind::Heuristic,
            true,
            false,
        )
        .unwrap();
        assert!(!d.live);
    }

    #[test]
    fn auto_llm_without_prereqs_is_off() {
        let d = resolve_distillation(
            DistillationMode::Auto,
            ResolvedReasonerKind::Llm,
            false,
            true,
        )
        .unwrap();
        assert!(!d.live);
    }

    // Off: never live, regardless of everything else.
    #[test]
    fn off_is_never_live() {
        let d = resolve_distillation(
            DistillationMode::Off,
            ResolvedReasonerKind::Llm,
            true,
            true,
        )
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
            false,
        );
        assert!(err.is_err());
    }

    // On + feature enabled but reasoner resolved Heuristic = endpoint expected
    // but unresolved = misconfiguration = hard error.
    #[test]
    fn on_llm_feature_but_heuristic_reasoner_errors() {
        let err = resolve_distillation(
            DistillationMode::On,
            ResolvedReasonerKind::Heuristic,
            true,
            true,
        );
        assert!(err.is_err());
    }

    // On without the llm feature (heuristic is the only possible reasoner):
    // proceed live with a warning rather than erroring.
    #[test]
    fn on_no_llm_feature_is_live_with_heuristic() {
        let d = resolve_distillation(
            DistillationMode::On,
            ResolvedReasonerKind::Heuristic,
            true,
            false,
        )
        .unwrap();
        assert!(d.live);
    }

    #[test]
    fn on_llm_with_prereqs_is_live() {
        let d = resolve_distillation(
            DistillationMode::On,
            ResolvedReasonerKind::Llm,
            true,
            true,
        )
        .unwrap();
        assert!(d.live);
    }
}
