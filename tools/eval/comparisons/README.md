# Head-to-head comparison harness (Mem0)

`mem0_eval.py` runs the open-source [Mem0](https://github.com/mem0ai/mem0) memory
system through the SAME LoCoMo questions, the SAME codex answer/grade prompts, and
the SAME abstention/faithfulness classification as this repo's Rust eval harness —
so Mem0's QA accuracy and confabulation rate can be put directly beside ours.

## What it does (apples-to-apples on the ANSWERING side)
- Ingests each LoCoMo conversation into Mem0 (fact extraction), isolated per
  conversation via `user_id`.
- Retrieves top-k memories per question with Mem0's `search`.
- Answers + grades with the SAME `codex exec` prompts our harness uses
  (`tools/eval/src/qa.rs`), and classifies abstention with the same phrase set.

## Setup
```bash
uv venv mem0env --python 3.12
uv pip install --python mem0env/bin/python mem0ai ollama qdrant-client
ollama pull qwen2.5:7b-instruct   # Mem0's extraction LLM (local)
# nomic-embed-text for embeddings (already used by this repo's Ollama path)
./mem0env/bin/python mem0_eval.py <n_conversations> <questions_per_conv> [qtype]
```

## HONEST CAVEATS (read before citing)
- **Mem0's ingestion here uses a LOCAL qwen2.5-7B, not the GPT-4 its published
  LoCoMo numbers (~66%) assume.** Mem0's memory quality depends heavily on the
  extraction LLM, so this is a MATCHED-LOCAL comparison, NOT Mem0's best case.
- Retrieval `recall` is NOT compared (Mem0 stores extracted facts with its own
  IDs, not the original turns our recall metric keys on). Only end-to-end QA +
  faithfulness are comparable.
- Small subsets; the codex judge is nondeterministic (~±0.03).
- Zep/Graphiti and Letta are not included (they need a server + graph DB / more
  setup); this is one real head-to-head, not a full field survey.

## Using the codex OAuth login instead of an OpenAI API key

`codex_openai_shim.py` is a tiny OpenAI-compatible server (`/v1/chat/completions`)
that shells out to `codex exec`, so any OpenAI-compatible tool (including Mem0) can
run on the codex-authenticated model (gpt-5.6-sol) WITHOUT an API key.
`mem0_codex_eval.py` bundles the shim (on a thread) + the Mem0 eval, pointing Mem0's
`openai` provider at the shim — so Mem0's fact-extraction runs on gpt-5.6-sol.

**Verified working:** Mem0 fact-extraction ran on gpt-5.6-sol via the shim and produced
cleaner, date-stamped facts than the local qwen-7B path.

**Practical limit (measured):** each `codex exec` call is ~20-30s and Mem0 makes ~2 LLM
calls per session-add, so ingesting even 2 LoCoMo conversations (~40 sessions) is
~30-50 min and exceeds typical run windows — a full-scale codex-backed Mem0 comparison
was not completed here. With a real OpenAI-compatible endpoint (fast), the same harness
runs quickly. Qualitatively the frontier extractor produced clearly better facts than
qwen-7B, consistent with Mem0's published numbers being higher with a GPT-4-class
extractor — so the qwen-based 0.25 is a floor for Mem0, not its best case.

## Graphiti (Zep) head-to-head (2026-07-18)

`graphiti_eval.py <n_conv> <per>` ingests LoCoMo into FalkorDB via Graphiti's
`OpenAIGenericClient` (chat/completions + json_object) driven by the codex shim
(frontier extraction, no API key), Ollama for embeddings. `graphiti_qa.py
<n_conv> <per>` runs the matched codex QA over the ingested graphs.

Setup: `docker run -d --name falkordb -p 6379:6379 falkordb/falkordb:latest`;
`uv pip install "graphiti-core[falkordb]"`.

IMPORTANT: FalkorDB stores each `group_id` as a SEPARATE graph. Retrieve by
connecting the driver per-conversation (`FalkorDriver(database=<conv>)`), NOT
`search(group_ids=…)` (which queries the empty default graph → 0 results, a
spurious near-zero score). Result (matched 12-q slice): Graphiti 0.500, in the
same band as Mem0/this repo — see docs/evaluation.md. Graphiti ingestion is
~4× slower than Mem0 (~70s/episode), so only a 12-q slice was run.
