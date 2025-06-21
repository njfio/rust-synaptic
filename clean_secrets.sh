#!/bin/bash

# Replace the hardcoded API key with environment variable usage
sed -i '' 's/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/OPENAI_API_KEY_PLACEHOLDER/g' examples/openai_embeddings_demo.rs 2>/dev/null || true
sed -i '' 's/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/OPENAI_API_KEY_PLACEHOLDER/g' src/memory/embeddings/openai_embeddings.rs 2>/dev/null || true
