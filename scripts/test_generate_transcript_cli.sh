#!/bin/sh
set -eu
ROOT=$(CDPATH= cd -- "$(dirname "$0")/.." && pwd)
BIN=${RCSFS_TRANSCRIPT_CLI:-"$ROOT/target/release/examples/generate_transcript_csfs"}
FIXTURE=${1:-"$ROOT/tests/fixtures/o1_cc1as1.rcsfgenerate"}
WORK=$(mktemp -d "${TMPDIR:-/tmp}/rcsfs-cli.XXXXXX")
trap 'rm -rf "$WORK"' EXIT
[ -x "$BIN" ] || { echo "build CLI first: cargo build --release --example generate_transcript_csfs" >&2; exit 2; }
for threads in 1 2; do
  RCSFS_THREADS=$threads "$BIN" "$FIXTURE" "$WORK/out-$threads.c" 200000 "$WORK/out-$threads.csv" --normalize > "$WORK/run-$threads.log"
  test -s "$WORK/out-$threads.c"
  test -s "$WORK/out-$threads.csv"
  test -s "$WORK/out-$threads.toml"
  grep -q 'records = 89786' "$WORK/run-$threads.log"
done
cmp "$WORK/out-1.c" "$WORK/out-2.c"
cmp "$WORK/out-1.csv" "$WORK/out-2.csv"
sha256sum "$WORK/out-1.c" "$WORK/out-1.csv"
