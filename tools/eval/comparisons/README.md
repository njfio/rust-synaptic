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
