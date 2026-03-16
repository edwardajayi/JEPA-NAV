#!/usr/bin/env bash
# Download ReplicaCAD dataset from Hugging Face (no clicking required).
# Dataset: https://huggingface.co/datasets/ai-habitat/ReplicaCAD_dataset
# License: CC BY 4.0

set -e
REPO="ai-habitat/ReplicaCAD_dataset"
DEST="${1:-./data/ReplicaCAD}"
mkdir -p "$DEST"
echo "Downloading $REPO to $DEST ..."

# Option 1: hf CLI (install with: pip install -U huggingface_hub)
if command -v hf &>/dev/null; then
  hf download "$REPO" --repo-type dataset --local-dir "$DEST"
  echo "Done. Dataset saved to $DEST"
  exit 0
fi

# Option 2: huggingface-cli (older name)
if command -v huggingface-cli &>/dev/null; then
  huggingface-cli download "$REPO" --repo-type dataset --local-dir "$DEST"
  echo "Done. Dataset saved to $DEST"
  exit 0
fi

# Option 3: Python one-liner
echo "CLI not found. Trying Python huggingface_hub ..."
python3 -c "
from huggingface_hub import snapshot_download
snapshot_download('$REPO', repo_type='dataset', local_dir='$DEST')
print('Done. Dataset saved to $DEST')
" || {
  echo "Install first: pip install -U huggingface_hub"
  echo "Then run: python3 -c \"
from huggingface_hub import snapshot_download
snapshot_download('$REPO', repo_type='dataset', local_dir='$DEST')
\""
  exit 1
}
