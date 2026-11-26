#!/usr/bin/env bash
set -euo pipefail
python verify_statistics.py
python synthetic_analysis.py
python make_figures.py
python -m pytest -q tests
