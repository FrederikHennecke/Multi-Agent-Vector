#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

.venv/bin/python src/evaluation/plot_results.py --save-dir imgs results_8B/*.jsonl
cp imgs/*.pdf 6904f1e833cef0fd021cf908/images/
