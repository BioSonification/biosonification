#!/usr/bin/env python3
"""Quick dataset report using preprocessed cache files."""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def main() -> None:
    parser = argparse.ArgumentParser(description="Quick dataset report from cache")
    parser.add_argument("--bio-cache", required=True, help="Bio cache pickle file")
    parser.add_argument("--music-cache", required=True, help="Music cache pickle file")
    parser.add_argument("--output-dir", default="results/quick_dataset_report", help="Output directory")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading cached data...")

    # Load bio cache
    with open(args.bio_cache, "rb") as f:
        bio_data = pickle.load(f)

    # Load music cache
    with open(args.music_cache, "rb") as f:
        music_data = pickle.load(f)

    print(f"Bio fragments: {len(bio_data['encodings'])}")
    print(f"Music segments: {len(music_data['segments'])}")

    # Build report
    report = {
        "bio": {
            "cache_path": str(args.bio_cache),
            "config_path": bio_data["config_path"],
            "fragment_count": len(bio_data["encodings"]),
            "sequence_types": {},
        },
        "music": {
            "cache_path": str(args.music_cache),
            "config_path": music_data["config_path"],
            "segment_count": len(music_data["segments"]),
        },
        "estimated_pairs": len(bio_data["encodings"]) * 5,  # top_k=5
    }

    # Count sequence types
    for enc in bio_data["encodings"]:
        seq_type = enc.sequence_type
        report["bio"]["sequence_types"][seq_type] = report["bio"]["sequence_types"].get(seq_type, 0) + 1

    # Save report
    json_path = output_dir / "quick_report.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    # Print summary
    print("\n" + "=" * 80)
    print("Dataset Report")
    print("=" * 80)
    print(f"\nBiological Data:")
    print(f"  Fragments: {report['bio']['fragment_count']}")
    print(f"  Types: {report['bio']['sequence_types']}")
    print(f"\nMusic Data:")
    print(f"  Segments: {report['music']['segment_count']}")
    print(f"\nEstimated Training Pairs: {report['estimated_pairs']}")
    print(f"\nReport saved: {json_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
