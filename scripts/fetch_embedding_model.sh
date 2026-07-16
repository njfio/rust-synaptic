#!/usr/bin/env bash
# Fetch the bundled offline embedding model(s) into local, gitignored cache
# directories under models/.
#
# Usage:
#   ./scripts/fetch_embedding_model.sh                 # MiniLM (candle, ml-models)
#   ./scripts/fetch_embedding_model.sh --potion        # potion-base-8M (static-embeddings)
#   ./scripts/fetch_embedding_model.sh --cross-encoder # ms-marco-MiniLM-L6-v2 (reranker-model)
#   ./scripts/fetch_embedding_model.sh --all           # all of the above
#   ./scripts/fetch_embedding_model.sh [--potion|--cross-encoder] <dest-dir>   # override destination
#
# Models:
# - sentence-transformers/all-MiniLM-L6-v2 (~87 MB) for the `ml-models` candle
#   provider. OPT-IN (candle CPU inference is slow — see docs/evaluation.md):
#     cargo build --features ml-models
#     SYNAPTIC_RETRIEVAL_EMBEDDER=candle SYNAPTIC_EMBED_MODEL_DIR=models/all-MiniLM-L6-v2 ...
# - minishlab/potion-base-8M (~30 MB) for the `static-embeddings` model2vec
#   provider: a static embedding TABLE (token-row lookup + mean-pool, no
#   transformer forward pass) — CPU-instant AND genuinely semantic. Because it
#   is fast, it is the preferred default: with `--features static-embeddings`
#   built and models/potion-base-8M/ present, RetrievalEmbeddingConfig::auto()
#   selects it. A plain `cargo build` (feature off) stays TF-IDF. Override the
#   dir with SYNAPTIC_STATIC_MODEL_DIR, or force selection with
#   SYNAPTIC_RETRIEVAL_EMBEDDER=static.
# - cross-encoder/ms-marco-MiniLM-L-6-v2 (~87 MB) for the `reranker-model`
#   candle cross-encoder reranker. OPT-IN:
#     cargo build --features reranker-model
#     SYNAPTIC_RERANKER=cross-encoder SYNAPTIC_RERANKER_MODEL_DIR=models/ms-marco-MiniLM-L6-v2 ...
#   Unset or `SYNAPTIC_RERANKER=heuristic` (default) keeps the deterministic
#   HeuristicReranker; a binary built without `reranker-model` also falls
#   back to it (with a warning) if cross-encoder is requested.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FILES=(config.json tokenizer.json model.safetensors)

fetch() {
  local base_url="$1" dest="$2"
  mkdir -p "$dest"
  for f in "${FILES[@]}"; do
    if [[ -s "$dest/$f" ]]; then
      echo "already present: $dest/$f"
      continue
    fi
    echo "downloading $f ..."
    if ! curl -fSL --retry 3 -o "$dest/$f.tmp" "$base_url/$f"; then
      rm -f "$dest/$f.tmp"
      echo "ERROR: failed to download $base_url/$f" >&2
      exit 1
    fi
    mv "$dest/$f.tmp" "$dest/$f"
  done
  echo "model ready at: $dest"
}

MINILM_URL="https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main"
POTION_URL="https://huggingface.co/minishlab/potion-base-8M/resolve/main"
CROSS_ENCODER_URL="https://huggingface.co/cross-encoder/ms-marco-MiniLM-L-6-v2/resolve/main"

case "${1:-}" in
  --potion)
    fetch "$POTION_URL" "${2:-$REPO_ROOT/models/potion-base-8M}"
    ;;
  --cross-encoder)
    fetch "$CROSS_ENCODER_URL" "${2:-$REPO_ROOT/models/ms-marco-MiniLM-L6-v2}"
    ;;
  --all)
    fetch "$MINILM_URL" "$REPO_ROOT/models/all-MiniLM-L6-v2"
    fetch "$POTION_URL" "$REPO_ROOT/models/potion-base-8M"
    fetch "$CROSS_ENCODER_URL" "$REPO_ROOT/models/ms-marco-MiniLM-L6-v2"
    ;;
  *)
    fetch "$MINILM_URL" "${1:-$REPO_ROOT/models/all-MiniLM-L6-v2}"
    ;;
esac
