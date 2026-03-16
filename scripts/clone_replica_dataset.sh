#!/usr/bin/env bash
# Clone the original Replica Dataset repo (Facebook Research).
# Use images/textures from it for V-JEPA 2 Phase A or Habitat.
# Repo: https://github.com/facebookresearch/Replica-Dataset
# Full data (~30GB) is separate: run download.sh inside the clone.

set -e
DEST="${1:-./data/Replica-Dataset}"
REPO_URL="https://github.com/facebookresearch/Replica-Dataset.git"

mkdir -p "$(dirname "$DEST")"
if [[ -d "$DEST/.git" ]]; then
  echo "Already cloned at $DEST. Pulling latest..."
  (cd "$DEST" && git pull)
else
  echo "Cloning Replica-Dataset to $DEST ..."
  git clone "$REPO_URL" "$DEST"
fi
echo "Done. Repo at $DEST"
echo ""
echo "To download full scene data (~30GB), run:"
echo "  $DEST/download.sh $DEST"
echo "Then scene textures (images) are under: $DEST/replica_flat_*/textures/"
