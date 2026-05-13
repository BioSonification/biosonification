#!/usr/bin/env python3
"""Run structured v2 evaluation and write JSON/Markdown reports."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from bio_music_pipeline.v2.evaluate import StructuredEvaluationConfig, run_structured_evaluation


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the structured v2 biosonification pipeline")
    parser.add_argument(
        "--checkpoint",
        default="results/v2_music21_rtx2060/checkpoints/structured_pipeline.pt",
        help="Path to structured_pipeline.pt",
    )
    parser.add_argument("--fasta", default="data/fasta/quick_sample.fa", help="Evaluation FASTA file")
    parser.add_argument("--config", default=None, help="Optional v2 config override")
    parser.add_argument("--output-dir", default="results/v2_evaluation", help="Where to write reports and MIDI files")
    parser.add_argument("--max-records", type=int, default=4, help="Maximum FASTA records/fragments to evaluate")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto", help="Generation device")
    parser.add_argument("--seed", type=int, default=42, help="Random baseline seed")
    args = parser.parse_args()

    report = run_structured_evaluation(
        StructuredEvaluationConfig(
            checkpoint_path=args.checkpoint,
            fasta_path=args.fasta,
            config_path=args.config,
            output_dir=args.output_dir,
            max_records=args.max_records,
            device=args.device,
            seed=args.seed,
        )
    )
    print(json.dumps(report["aggregates"], indent=2))


if __name__ == "__main__":
    main()
