#!/usr/bin/env bash
set -euo pipefail
START="${1:-1}"
END="${2:-20}"
mkdir -p results
for n in $(seq "$START" "$END"); do
  env_id=$(printf "%05d" "$n")
  echo "=== environment ${env_id} ==="
  WAIRD_ENV_ID="$env_id" mpirun -np 5 ./LAC-MAC WAIRD_DOWNTILT_5 \
    > "results/${env_id}.log" 2>&1
done
