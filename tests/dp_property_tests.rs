//! Property-based tests for differential-privacy epsilon-budget accounting
//! and the Laplace noise mechanism used by `synaptic::security::privacy`.
//!
//! These tests treat the DP budget accounting and noise scale as the spec:
//! repeated queries must deplete a user's privacy budget monotonically, and
//! once the budget is exhausted further queries must be refused (`Err`)
//! rather than silently granted or under-charged.

#![cfg(feature = "security")]

use proptest::prelude::*;
use synaptic::memory::types::{MemoryEntry, MemoryType};
use synaptic::security::privacy::{PrivacyManager, PrivacyQuery, PrivacyQueryType};
use synaptic::security::{SecurityConfig, SecurityContext};

fn config_with_budget(total_budget: f64) -> SecurityConfig {
    SecurityConfig {
        privacy_budget: total_budget,
        ..SecurityConfig::default()
    }
}

fn sample_entries(n: usize) -> Vec<MemoryEntry> {
    (0..n)
        .map(|i| {
            MemoryEntry::new(
                format!("key-{i}"),
                format!("value-{i}"),
                MemoryType::ShortTerm,
            )
        })
        .collect()
}

fn count_query(sensitivity: f64) -> PrivacyQuery {
    PrivacyQuery {
        query_type: PrivacyQueryType::Count,
        sensitivity,
        bins: None,
        quantile: None,
    }
}

proptest! {
    /// Repeated queries against a fixed total budget must never let total
    /// epsilon spent exceed the configured total budget: each successful
    /// allocation subtracts from the remaining budget, and the remaining
    /// budget must never go negative.
    #[test]
    fn budget_never_goes_negative(
        total_budget in 0.5f64..5.0,
        sensitivities in prop::collection::vec(0.01f64..0.5, 1..30),
    ) {
        tokio_test_block_on(async {
            let config = config_with_budget(total_budget);
            let mut manager = PrivacyManager::new(&config)
                .await
                .expect("privacy manager must construct");
            let context = SecurityContext::new("user-a".to_string(), vec!["user".to_string()]);
            let entries = sample_entries(3);

            let mut spent = 0.0f64;
            for sensitivity in sensitivities {
                let before = manager
                    .get_remaining_budget(&context.user_id)
                    .await
                    .expect("remaining budget query must succeed");

                let result = manager
                    .generate_private_statistics(&entries, count_query(sensitivity), &context)
                    .await;

                let after = manager
                    .get_remaining_budget(&context.user_id)
                    .await
                    .expect("remaining budget query must succeed");

                prop_assert!(after >= -1e-9, "remaining budget must never go negative, got {after}");

                if result.is_ok() {
                    spent += sensitivity;
                    prop_assert!(
                        (before - after - sensitivity).abs() < 1e-9,
                        "successful allocation must deduct exactly `sensitivity` from the budget"
                    );
                } else {
                    // Refused: budget must be unchanged and insufficient for the request.
                    prop_assert!((before - after).abs() < 1e-9);
                    prop_assert!(before < sensitivity);
                }

                prop_assert!(
                    spent <= total_budget + 1e-9,
                    "cumulative spent epsilon ({spent}) must never exceed total budget ({total_budget})"
                );
            }
            Ok(())
        })?;
    }

    /// Once a user's budget is exhausted, any further query with positive
    /// sensitivity must be refused with an `Err`, never silently answered.
    #[test]
    fn exhausted_budget_refuses_further_queries(
        total_budget in 0.1f64..2.0,
        extra_sensitivity in 0.01f64..1.0,
    ) {
        tokio_test_block_on(async {
            let config = config_with_budget(total_budget);
            let mut manager = PrivacyManager::new(&config)
                .await
                .expect("privacy manager must construct");
            let context = SecurityContext::new("user-b".to_string(), vec!["user".to_string()]);
            let entries = sample_entries(3);

            // Spend the entire budget in one allocation.
            let first = manager
                .generate_private_statistics(&entries, count_query(total_budget), &context)
                .await;
            prop_assert!(first.is_ok(), "spending exactly the full budget must succeed");

            let remaining = manager
                .get_remaining_budget(&context.user_id)
                .await
                .expect("remaining budget query must succeed");
            prop_assert!(remaining.abs() < 1e-9, "budget must be fully depleted, got {remaining}");

            // Any further request for positive epsilon must now be refused.
            let second = manager
                .generate_private_statistics(&entries, count_query(extra_sensitivity), &context)
                .await;
            prop_assert!(
                second.is_err(),
                "a query issued after the budget is exhausted must be refused"
            );
            Ok(())
        })?;
    }
}

/// Small helper to drive an async block inside a synchronous `proptest!`
/// test body without pulling in `#[tokio::test]` (proptest test functions
/// cannot be `async`).
fn tokio_test_block_on<F, T>(fut: F) -> T
where
    F: std::future::Future<Output = T>,
{
    tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .expect("failed to build tokio runtime for property test")
        .block_on(fut)
}
