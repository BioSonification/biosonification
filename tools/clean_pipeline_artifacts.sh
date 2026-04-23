#!/usr/bin/env bash
set -euo pipefail

# Clean generated pipeline artifacts safely.
# Default mode is DRY-RUN. Use --apply to actually delete.

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
RESULTS_DIR="$ROOT_DIR/results"
APPLY=false

if [[ "${1:-}" == "--apply" ]]; then
  APPLY=true
fi

TARGETS=(
  "$RESULTS_DIR/quick_paired_run"
  "$RESULTS_DIR/quick_paired_run_v2"
  "$RESULTS_DIR/full_paired_run"
  "$RESULTS_DIR/paired_data_quick"
  "$RESULTS_DIR/paired_data_full"
  "$RESULTS_DIR/midi"
  "$RESULTS_DIR/models"
  "$RESULTS_DIR/reports"
  "$RESULTS_DIR/surveys"
  "$RESULTS_DIR/data_splits"
  "$RESULTS_DIR/synthetic_midi"
  "$RESULTS_DIR/final_report.json"
  "$RESULTS_DIR/summary.txt"
  "$RESULTS_DIR/single_fasta_check"
)

echo "Root: $ROOT_DIR"
echo "Mode: $([[ "$APPLY" == true ]] && echo APPLY || echo DRY-RUN)"

echo "Targets:"
for t in "${TARGETS[@]}"; do
  [[ -e "$t" ]] && echo "  - $t"
done

if [[ "$APPLY" != true ]]; then
  echo
  echo "Dry-run complete. Re-run with --apply to delete listed targets."
  exit 0
fi

for t in "${TARGETS[@]}"; do
  if [[ -e "$t" ]]; then
    rm -rf "$t"
    echo "Deleted: $t"
  fi
done

echo "Cleanup complete."
