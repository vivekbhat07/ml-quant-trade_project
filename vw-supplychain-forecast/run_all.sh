#!/usr/bin/env bash
set -e
python3 src/data_prep.py
python3 src/features.py
python3 src/rolling_evaluation.py
python3 src/backtest.py
echo "All done. Check outputs/ and models/ directories."

