#!/usr/bin/env python3
"""Preprocess biological sequences and save to cache."""

from __future__ import annotations

import argparse
import pickle
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from bio_music_pipeline.utils.progress_logger import ProgressLogger
from bio_music_pipeline.v2.bio import BiologicalSequenceEncoder
from bio_music_pipeline.v2.config import load_v2_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess biological sequences")
    parser.add_argument("--config", required=True, help="v2 config path")
    parser.add_argument("--output", required=True, help="Output pickle file path")
    parser.add_argument("--force", action="store_true", help="Force reprocessing even if cache exists")
    args = parser.parse_args()

    output_path = Path(args.output)

    # Check if cache exists
    if output_path.exists() and not args.force:
        print(f"Cache already exists: {output_path}")
        print("Use --force to reprocess")

        # Load and show stats
        with open(output_path, "rb") as f:
            data = pickle.load(f)
        print(f"\nCached data:")
        print(f"  Fragments: {len(data['encodings'])}")
        print(f"  Config: {data['config_path']}")
        return

    # Setup logger
    output_path.parent.mkdir(parents=True, exist_ok=True)
    log_file = output_path.parent / f"{output_path.stem}.log"

    logger = ProgressLogger(
        name="preprocess_bio",
        log_file=log_file,
        use_tqdm=True,
    )

    logger.info("=" * 80)
    logger.info("Biological Sequence Preprocessing")
    logger.info("=" * 80)
    logger.info(f"Config: {args.config}")
    logger.info(f"Output: {args.output}")
    logger.info("")

    # Load config
    config = load_v2_config(args.config)

    logger.info(f"FASTA path: {config.fasta_path}")
    logger.info(f"Fragment length: {config.bio.fragment_length}")
    logger.info(f"Fragment stride: {config.bio.fragment_stride}")
    logger.info(f"Max fragments per record: {config.bio.max_fragments_per_record}")
    logger.info("")

    # Encode sequences
    start_time = time.time()
    encoder = BiologicalSequenceEncoder(config.bio)

    logger.info("Starting biological sequence encoding...")
    encodings = encoder.encode_fasta(config.fasta_path)

    elapsed = time.time() - start_time
    logger.info(f"Encoding completed in {elapsed:.1f} seconds ({elapsed/60:.1f} minutes)")
    logger.info(f"Extracted {len(encodings)} biological fragments")
    logger.info("")

    # Save to cache
    logger.info(f"Saving to cache: {output_path}")
    cache_data = {
        "config_path": args.config,
        "encodings": encodings,
        "timestamp": time.time(),
        "bio_config": config.bio,
    }

    with open(output_path, "wb") as f:
        pickle.dump(cache_data, f, protocol=pickle.HIGHEST_PROTOCOL)

    file_size_mb = output_path.stat().st_size / (1024 * 1024)
    logger.info(f"Cache saved: {file_size_mb:.1f} MB")
    logger.info("")
    logger.info("=" * 80)
    logger.info("Preprocessing complete!")
    logger.info("=" * 80)

    print(f"\nBiological preprocessing complete!")
    print(f"  Fragments: {len(encodings)}")
    print(f"  Cache: {output_path}")
    print(f"  Size: {file_size_mb:.1f} MB")
    print(f"  Time: {elapsed/60:.1f} minutes")


if __name__ == "__main__":
    main()
