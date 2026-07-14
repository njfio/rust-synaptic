#!/usr/bin/env bash
# Fetch the bundled offline embedding model (sentence-transformers/all-MiniLM-L6-v2)
# into a local, gitignored cache directory for the `ml-models` candle provider.
#
# Default destination: models/all-MiniLM-L6-v2/ (relative to the repo root).
# Override with:  ./scripts/fetch_embedding_model.sh <dest-dir>
#
# After fetching, the model is used fully offline:
#   cargo build --features ml-models
#   SYNAPTIC_RETRIEVAL_EMBEDDER=candle SYNAPTIC_EMBED_MODEL_DIR=models/all-MiniLM-L6-v2 ...
# (or just leave the dir in place: with ml-models built, the default provider
#  selection prefers the candle model when this directory exists.)
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="${1:-$REPO_ROOT/models/all-MiniLM-L6-v2}"
BASE_URL="https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main"
FILES=(config.json tokenizer.json model.safetensors)

mkdir -p "$DEST"

for f in "${FILES[@]}"; do
  if [[ -s "$DEST/$f" ]]; then
    echo "already present: $DEST/$f"
    continue
  fi
  echo "downloading $f ..."
  if ! curl -fSL --retry 3 -o "$DEST/$f.tmp" "$BASE_URL/$f"; then
    rm -f "$DEST/$f.tmp"
    echo "ERROR: failed to download $BASE_URL/$f" >&2
    exit 1
  fi
  mv "$DEST/$f.tmp" "$DEST/$f"
done

echo "model ready at: $DEST"
