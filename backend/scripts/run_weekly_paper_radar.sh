#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

cd "${PROJECT_ROOT}"
mkdir -p logs

if [[ -f "backend/.venv/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source "backend/.venv/bin/activate"
fi

export PYTHONPATH="${PROJECT_ROOT}/backend${PYTHONPATH:+:${PYTHONPATH}}"

python backend/scripts/weekly_paper_radar.py "$@"
