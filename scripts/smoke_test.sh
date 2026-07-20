#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec "$ROOT/bench" run \
  --engines all \
  --modes cold,warm_shared \
  --concurrency 1 \
  --repetitions 1 \
  --samples 5 \
  "$@"
