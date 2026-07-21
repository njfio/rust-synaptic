# Write-Time Fact Distillation — Design Spec

**Date:** 2026-07-20
**Status:** Approved (brainstorm), pending implementation plan
**Author:** brainstormed with the user

## Problem

Measurement this cycle established that rust-synaptic's LoCoMo QA gap versus Mem0 was
never a *retrieval* problem — it was *ingestion*. With write-time LLM-distilled facts,
this repo ties Mem0 (≈0.475 → 0.500 QA on the matched LoCoMo slice; see
`docs/evaluation.md` "Closing the gap: write-time fact extraction"). But the product
does not use this by default:

- `MemoryConfig::store_extracted_facts` defaults to `false` (`src/lib.rs`).
- The default reasoner is the heuristic TF-IDF `HeuristicReasoner`; the `LlmReasoner`
  only activates with the `llm-reasoning` feature + endpoint env (`src/lib.rs:246`).
- Even with `store_extracted_facts = true`, facts are stored *in addition to* the raw
  turn, and both compete in retrieval — which did NOT reproduce the measured
  facts-primary condition (the eval built the store from facts only).

So the single highest-leverage, already-validated improvement is dormant. This spec
promotes and hardens the existing `MemoryReasoner` + `store_extracted_facts` scaffolding
into a real default write path: **when an LLM endpoint is configured, `store()` distills
the incoming turn into fact-memories that become the primary retrievable units; raw turns
are persisted but excluded from default search.**

## Goals

1. Distillation is a real, default-on path **when an LLM endpoint is configured**, with
   zero surprising cost when one is not (no LLM → today's raw-turn behavior, unchanged).
2. Facts are the primary retrievable units; raw turns are retained (provenance) but
   excluded from default ranking — reproducing the measured Mem0-parity condition.
3. Best-effort semantics: any distillation failure degrades to today's raw-turn
   retrievability, never loses data or breaks the write.
4. A matched before/after LoCoMo A/B proves the lift, gated on distillation ≥ baseline.

## Non-Goals

- Building a stronger no-LLM heuristic extractor (extraction quality is the whole lever;
  the heuristic will not reach parity — explicitly out of scope).
- A separate fact index / collection-aware retriever (larger change; the metadata-filter
  approach achieves the same behavior).
- Changing the answer/QA path, grounding, or abstention (separate deferred threads).
- Fixing the O(n²) write path (separate deferred thread; noted, not addressed here).

## Global Constraints

- **Honesty bar (project-wide):** no fabricated numbers; report negative results; if the
  A/B does not show distillation ≥ baseline, report it and do NOT ship default-on.
- **Best-effort subsystems:** distillation joins the other `store_with_report` subsystems
  as best-effort — failure sets a `degradations.*` field and logs, never aborts the write.
- **Backwards compatibility:** existing callers using `store_extracted_facts` must keep
  working (deprecated alias, mapped to the new enum).
- **No behavior change for no-LLM deployments:** a store with no configured LLM reasoner
  must be byte-for-byte equivalent to today (raw turns stored untagged and searchable).
- **PR-only merge flow:** repo ruleset requires PRs; a new branch per PR iteration.

## Design

### Retrieval model (decided)

**Facts primary, raw demoted** (Approach A — demote-by-metadata + search filter):

- Raw turns are stored as today but tagged `custom_fields["raw_source"] = "<rfc3339 ts>"`
  **only when distillation is live**.
- Distilled facts are stored as normal, *untagged*, retrievable memories keyed
  `{source_key}::fact{i}` (the existing naming in `reason_over_store`).
- `search` excludes `raw_source`-tagged memories by default, via a new config
  `retrieval_excludes_raw_sources` — structurally identical to the existing
  `retrieval_excludes_superseded` filter (`src/lib.rs`, `SUPERSEDED_FIELD`).
- Raw turns remain reachable by explicit key lookup or an opt-in "include raw" search.

Rejected alternatives: (B) facts replace raw — lossy, no provenance; (C) separate fact
index — retriever is not collection-aware, much larger change for identical behavior.

### Configuration (decided)

A single `MemoryConfig::distillation: DistillationMode` enum resolved at construction:

- `Auto` **(default)** — distillation is live when **both** (a) the resolved reasoner is
  the `LlmReasoner` and (b) the write-path prerequisites are met (`enable_knowledge_graph`
  + `embeddings`, which today's `reason_over_store` already requires at `src/lib.rs:692`);
  off otherwise. No LLM configured, or no KG/embeddings → off → today's behavior. This is
  the "on when LLM configured" default posture. The `DistillationPolicy` resolver takes
  the KG/embeddings availability as an input alongside the reasoner kind.
- `On` — require distillation. If the write-path prerequisites (`enable_knowledge_graph` +
  `embeddings`) are absent, return a construction error (distillation cannot run at all).
  If prerequisites are met but no LLM reasoner is available because the feature is present
  and no endpoint resolves, return a construction error (fail loud on misconfiguration). If
  the caller forces `On` in a build without the `llm-reasoning` feature (only a heuristic
  reasoner can exist), log a one-time warning that the measured lift requires an LLM and
  proceed with heuristic facts rather than erroring.
- `Off` — never distill (escape hatch; exactly today's behavior).

`store_extracted_facts: bool` is retained as a **deprecated alias**: `true` → `On`,
`false` → leaves `distillation` at its `Auto`/explicit value. The alias is documented as
deprecated and slated for removal in a future major.

`retrieval_excludes_raw_sources: bool` — defaults to `true`. Controls the search filter.

### Components (three focused units)

1. **`DistillationPolicy`** (new, small, `src/memory/reasoning/` or inline module):
   pure resolver from `(DistillationMode, ResolvedReasonerKind)` → `{ live: bool }`, plus
   the construction-error decision for `On`+no-LLM. Testable with no store, no I/O.
   - `ResolvedReasonerKind` is a small enum (`Llm` | `Heuristic`) the constructor sets
     when it resolves `self.reasoner` (`src/lib.rs:246`).

2. **`reason_over_store`** (exists, `src/lib.rs:713`): extended to
   - store each non-empty fact as `{key}::fact{i}` (already present behind the flag),
   - after ≥1 fact is successfully stored, tag the *raw* entry `raw_source` and persist
     the tag (reusing the `mark_*` update pattern, like `mark_superseded`).
   - The existing `storing_facts` guard already prevents recursion / re-tagging of facts.

3. **`search` filter** (`src/lib.rs`): add one retain clause mirroring the superseded
   filter, gated on `retrieval_excludes_raw_sources`.

### Data flow (write path)

On `store(key, value)` with distillation live:

1. Persist the raw turn (as today). Do **not** tag it yet.
2. `reason_over_store` runs (gated on `!storing_facts`): extract facts; for each, resolve
   against neighbors, apply KG resolution + supersession (all existing).
3. Persist each non-empty fact as `{key}::fact{i}` (untagged, searchable) via the existing
   `Box::pin(self.store_with_report(...))` recursive path with `storing_facts = true`.
4. **Only after ≥1 fact is stored**, tag the raw entry `raw_source` and persist the tag.
5. `search` excludes `raw_source`-tagged memories when `retrieval_excludes_raw_sources`.

**Failure-ordering invariant (correctness crux):** the raw turn is tagged *after*
successful fact storage, never before. If extraction or fact storage fails, the raw turn
remains **untagged and searchable**, and `degradations.reasoning` is set — a distillation
outage degrades to today's behavior rather than hiding a raw turn with no fact to replace
it. This ordering is a required, tested property.

### Error handling

- Distillation is best-effort inside `store_with_report`: extractor/endpoint failure logs
  a warning, sets `degradations.reasoning`, leaves the raw turn searchable, and the write
  succeeds.
- `DistillationMode::On` with no LLM reasoner available (feature present, endpoint
  unresolved) is the one *construction-time* hard error — misconfiguration should fail
  loudly, not silently store weak facts.

## Testing

**Unit (deterministic, no network — stub `MemoryReasoner` returning known facts/errors):**

- `DistillationPolicy`: `Auto`+Llm → live; `Auto`+Heuristic → off; `On`+no-Llm(feature on,
  no endpoint) → construction error; `On`+Heuristic(no feature) → live+warn; `Off` → off.
- Write path, stub extractor with known facts: raw turn ends up tagged `raw_source`; facts
  stored as `{key}::fact{i}` untagged; `search` returns facts and excludes the raw turn;
  with `retrieval_excludes_raw_sources = false` the raw turn reappears in results.
- **Failure invariant:** stub extractor returns `Err` → raw turn stays *untagged* and
  searchable; `degradations.reasoning` is set; write still succeeds.
- Recursion/tag guard: storing a `{key}::fact{i}` memory does not re-tag it `raw_source`
  nor re-run distillation on it.
- Deprecated alias: `store_extracted_facts = true` behaves as `DistillationMode::On`.
- No-LLM equivalence: with a heuristic-only build, a store leaves the raw turn untagged
  and searchable (no behavior change).

**A/B measurement (the "prove the lift" deliverable):**

Extend the eval harness with a write-path switch so the *same* full LoCoMo run
(1,986 questions) can ingest either way, holding judge + retrieval (PRF + GPU
cross-encoder) constant:

- **Arm A (baseline):** raw-turn ingestion, distillation off.
- **Arm B (distillation):** facts-primary via the new default path, using the codex LLM as
  the extraction endpoint (the same OAuth shim already used for the judge).
- Report overall + per-category QA accuracy deltas; reuse the resumable checkpoint
  (`SYNAPTIC_EVAL_QA_CHECKPOINT`) so each arm is restartable.
- **Done bar:** Arm B ≥ Arm A overall, targeting the measured ≈0.475 → 0.500 parity. If B
  does not beat A, report honestly and do NOT ship distillation default-on.

**Docs:** a `docs/evaluation.md` section with the A/B table; a README note that
distillation auto-activates when an LLM endpoint is configured, and how to force On/Off.

## Rollout

1. Land config + policy + write-path + search filter + unit tests (default `Auto`, so
   no-LLM deployments are unaffected) behind a PR with green substantive CI.
2. Run the A/B; add the measured section to `docs/evaluation.md`.
3. Only if B ≥ A: document distillation as the recommended default and update the README.
   If B < A: keep the machinery (it is opt-in-correct) but do not claim/ship a default win;
   record the negative result.

## Open questions

None blocking. The heuristic-`On` warning wording and the exact `raw_source` timestamp
format are implementation details for the plan.
