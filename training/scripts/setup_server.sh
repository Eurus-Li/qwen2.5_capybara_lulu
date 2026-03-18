#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN=${PYTHON_BIN:-python}
REPO_ROOT=${REPO_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}

cd "$REPO_ROOT"

$PYTHON_BIN -m pip install --upgrade pip
$PYTHON_BIN -m pip install -r training/requirements-caption.txt

if [ ! -d external/DiffSynth-Studio ]; then
  git clone https://github.com/modelscope/DiffSynth-Studio.git external/DiffSynth-Studio
fi

$PYTHON_BIN -m pip install -e external/DiffSynth-Studio

echo "Environment ready."
