#!/usr/bin/env bash
# Run one algorithm demo in one language.
# Usage: ./scripts/run.sh <category>/<name> <java|cpp|rust|go>
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SPEC="${1:-}"
LANG="${2:-}"

if [[ -z "$SPEC" || -z "$LANG" ]]; then
  echo "Usage: $0 <category>/<name> <java|cpp|rust|go>" >&2
  exit 2
fi

DIR="$ROOT/algorithms/$SPEC"
if [[ ! -d "$DIR" ]]; then
  echo "Algorithm not found: $DIR" >&2
  exit 1
fi

WORKDIR="$(mktemp -d)"
trap 'rm -rf "$WORKDIR"' EXIT

case "$LANG" in
  java)
    cp "$DIR/java/"*.java "$WORKDIR/"
    (cd "$WORKDIR" && javac *.java && java "$(basename "$(ls *.java | head -1)" .java)")
    ;;
  cpp)
    SRC="$(ls "$DIR/cpp/"*.cpp | head -1)"
    g++ -std=c++17 -O0 -Wall -Wextra -o "$WORKDIR/a.out" "$SRC"
    "$WORKDIR/a.out"
    ;;
  rust)
    rustc -O -o "$WORKDIR/a.out" "$DIR/rust/main.rs"
    "$WORKDIR/a.out"
    ;;
  go)
    # Use main.go explicitly so demos work without a local go.mod when a
    # parent directory contains a .git (module-mode auto-detection).
    (cd "$DIR/go" && go run main.go)
    ;;
  *)
    echo "Unknown language: $LANG (expected java|cpp|rust|go)" >&2
    exit 2
    ;;
esac
