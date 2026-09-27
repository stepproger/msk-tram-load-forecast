#!/usr/bin/env bash
# Обёртка для Linux/macOS. Сам пайплайн — ml/run_pipeline.py (на Windows запускать его напрямую, см. README).
set -euo pipefail
cd "$(dirname "$0")/.."
uv run --python 3.12 --with-requirements ml/requirements.txt python ml/run_pipeline.py
