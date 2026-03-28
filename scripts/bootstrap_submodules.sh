#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

if [[ ! -f .gitmodules ]]; then
  echo "No .gitmodules found. Add DNS-Challenge as a submodule first:"
  echo "  git submodule add https://github.com/microsoft/DNS-Challenge.git submodules/DNS-Challenge"
  exit 1
fi

git submodule sync --recursive
git submodule update --init --recursive

echo "Submodules initialized successfully."
