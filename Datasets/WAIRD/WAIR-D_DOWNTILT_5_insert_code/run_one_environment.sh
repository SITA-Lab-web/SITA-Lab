#!/usr/bin/env bash
set -euo pipefail
ENV_ID="${1:-00001}"
export WAIRD_ENV_ID="$ENV_ID"
mpirun -np 5 ./LAC-MAC WAIRD_DOWNTILT_5
