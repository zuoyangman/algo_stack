#!/usr/bin/env bash
# Compile and run every algorithm demo in every language.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
RUN="$ROOT/scripts/run.sh"
FAIL=0
COUNT=0

langs=(java cpp rust go)

while IFS= read -r -d '' readme; do
  dir="$(dirname "$readme")"
  rel="${dir#"$ROOT/algorithms/"}"
  for lang in "${langs[@]}"; do
    COUNT=$((COUNT + 1))
    printf '%-40s %-5s ... ' "$rel" "$lang"
    if out="$("$RUN" "$rel" "$lang" 2>&1)"; then
      if echo "$out" | grep -q ': ok'; then
        echo "OK"
      else
        echo "FAIL (missing ': ok' line)"
        echo "$out"
        FAIL=$((FAIL + 1))
      fi
    else
      echo "FAIL"
      echo "$out"
      FAIL=$((FAIL + 1))
    fi
  done
done < <(find "$ROOT/algorithms" -mindepth 3 -maxdepth 3 -name README.md -print0 | sort -z)

echo
echo "Ran $COUNT demos; failures=$FAIL"
exit "$FAIL"
