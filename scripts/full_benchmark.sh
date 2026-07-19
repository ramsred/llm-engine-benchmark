#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec "$ROOT/bench" run \
  --engines both \
  --modes cold,warm_shared \
  --concurrency 1,2,4 \
  --repetitions 3 \
  "$@"
