#!/usr/bin/env python3
"""Train the next-generation biosonification pipeline."""

from __future__ import annotations

import argparse
import json

from bio_music_pipeline.v2 import train_structured_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the v2 structured bio-conditioned harmony+melody pipeline")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/pipeline_v2_small.json",
        help="Path to the v2 JSON config",
    )
    args = parser.parse_args()
    result = train_structured_pipeline(args.config)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
