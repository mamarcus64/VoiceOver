#!/usr/bin/env bash
# Turn off automatic Git LFS downloads on checkout / pull for this clone only.
# Working tree will show small pointer files until you run git lfs pull with --include.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if ! command -v git-lfs &>/dev/null; then
  echo "git-lfs is not installed; install it first (see README)." >&2
  exit 1
fi

if ! git rev-parse --git-dir &>/dev/null; then
  echo "Not a git repository; nothing to configure." >&2
  exit 0
fi

git lfs install --local --skip-smudge --force
echo "This repo: Git LFS will not auto-download on checkout or git pull."
echo "Fetch when needed, for example:"
echo "  git lfs pull --include='data/manifest.json'"
echo "  git lfs pull --include='data/transcripts/*.json'"
echo "  git lfs pull --include='data/eyegaze_vectors/*.csv'"
echo "  git lfs pull --include='*'    # all LFS objects (very large)"
