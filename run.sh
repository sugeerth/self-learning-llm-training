#!/usr/bin/env bash
# One-command pipeline: install → sweep experiments → benchmark winner → serve locally.
set -e
cd "$(dirname "$0")"

python3 -m pip install -q -r requirements.txt

echo "=== EXPERIMENTS (sweep of variants) ==="
python3 experiments.py

echo "=== BENCHMARK (on winner) ==="
python3 benchmark.py

echo "=== SERVE ==="
echo "Dashboard:  http://localhost:8000"
python3 server.py
