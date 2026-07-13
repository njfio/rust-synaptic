#!/usr/bin/env bash
# Fetch the REAL LongMemEval-S and LoCoMo evaluation datasets into
# tools/eval/data/ (gitignored — the datasets are not committed for
# licensing/size reasons). Fails loudly on any download or validation error;
# it never substitutes fake data.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="${SCRIPT_DIR}/data"
mkdir -p "${DATA_DIR}"

# LoCoMo: snap-research/locomo (https://github.com/snap-research/locomo)
LOCOMO_URL="https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json"
# LongMemEval-S: xiaowu0162/longmemeval on HuggingFace
# (https://huggingface.co/datasets/xiaowu0162/longmemeval). ~200MB JSON.
LONGMEMEVAL_URL="https://huggingface.co/datasets/xiaowu0162/longmemeval/resolve/main/longmemeval_s.json"

fetch() {
    local url="$1" dest="$2"
    if [[ -s "${dest}" ]]; then
        echo "[skip] ${dest} already exists ($(du -h "${dest}" | cut -f1))"
        return 0
    fi
    echo "[fetch] ${url}"
    if ! curl --fail --location --retry 3 --show-error --output "${dest}.part" "${url}"; then
        rm -f "${dest}.part"
        echo "ERROR: download failed for ${url}" >&2
        echo "       No fallback data will be substituted. Fix connectivity or" >&2
        echo "       download the file manually to ${dest}." >&2
        exit 1
    fi
    # Validate: must be parseable JSON, not an HTML error page or LFS pointer.
    if ! python3 -c "import json,sys; json.load(open(sys.argv[1]))" "${dest}.part"; then
        rm -f "${dest}.part"
        echo "ERROR: ${url} did not return valid JSON (auth wall / LFS pointer / error page?)." >&2
        exit 1
    fi
    mv "${dest}.part" "${dest}"
    echo "[ok] ${dest} ($(du -h "${dest}" | cut -f1))"
}

fetch "${LOCOMO_URL}" "${DATA_DIR}/locomo10.json"
fetch "${LONGMEMEVAL_URL}" "${DATA_DIR}/longmemeval_s.json"

echo "Datasets ready in ${DATA_DIR}"
