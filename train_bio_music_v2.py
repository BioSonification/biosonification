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
    parser.add_argument(
        "--bio-cache",
        type=str,
        default=None,
        help="Path to preprocessed bio cache (pickle file). If provided, skips bio preprocessing.",
    )
    parser.add_argument(
        "--music-cache",
        type=str,
        default=None,
        help="Path to preprocessed music cache (pickle file). If provided, skips music preprocessing.",
    )
    args = parser.parse_args()
    result = train_structured_pipeline(
        config_path=args.config,
        bio_cache_path=args.bio_cache,
        music_cache_path=args.music_cache,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
