//! Result type reporting subsystem degradations from a successful `store` call.

/// Which optional subsystems failed during a store; storage itself succeeded.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct StoreDegradations {
    pub temporal: Option<String>,
    pub analytics: Option<String>,
    pub knowledge_graph: Option<String>,
    pub advanced_management: Option<String>,
    pub embeddings: Option<String>,
    pub reasoning: Option<String>,
}

impl StoreDegradations {
    /// Returns true iff no subsystem reported a degradation.
    pub fn is_clean(&self) -> bool {
        self.temporal.is_none()
            && self.analytics.is_none()
            && self.knowledge_graph.is_none()
            && self.advanced_management.is_none()
            && self.embeddings.is_none()
            && self.reasoning.is_none()
    }
}
