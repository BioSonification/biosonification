#!/usr/bin/env python3
"""Write structured v2 dataset manifest and sanity report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from bio_music_pipeline.v2.dataset_report import DatasetReportConfig, build_dataset_report


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a structured v2 dataset manifest and sanity report")
    parser.add_argument("--config", default="configs/pipeline_v2_small.json", help="v2 config path")
    parser.add_argument("--output-dir", default="results/v2_dataset_report", help="Report output directory")
    parser.add_argument("--max-preview-records", type=int, default=12, help="FASTA records to preview")
    parser.add_argument("--skip-segments", action="store_true", help="Skip structured segment extraction")
    args = parser.parse_args()

    report = build_dataset_report(
        DatasetReportConfig(
            config_path=args.config,
            output_dir=args.output_dir,
            max_preview_records=args.max_preview_records,
            load_segments=not args.skip_segments,
        )
    )
    print(json.dumps({
        "fasta_records": report["fasta"]["record_count"],
        "bio_fragments": report["bio_fragments"]["fragment_count"],
        "music_source_kind": report["music_files"]["source_kind"],
        "music_file_count": report["music_files"]["file_count"],
        "structured_segments": report["structured_segments"].get("segment_count", 0),
        "output_dir": args.output_dir,
    }, indent=2))


if __name__ == "__main__":
    main()
