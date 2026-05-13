#!/usr/bin/env python3
"""Generate MIDI from FASTA using the v2 biosonification pipeline."""

from __future__ import annotations

import argparse
import json

from bio_music_pipeline.v2 import generate_structured_music_from_fasta


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate harmony+melody MIDI from FASTA with the v2 pipeline")
    parser.add_argument("--fasta", type=str, default="data/fasta/quick_sample.fa", help="Path to FASTA file")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to a trained v2 checkpoint")
    parser.add_argument("--output", type=str, default="results/v2_generation/generated_from_fasta.mid", help="Output MIDI path")
    parser.add_argument("--metadata-output", type=str, default="results/v2_generation/generated_from_fasta.json", help="Where to save generation metadata")
    parser.add_argument("--config", type=str, default=None, help="Optional v2 JSON config override. Defaults to the config stored in the checkpoint.")
    parser.add_argument("--record-index", type=int, default=0, help="0-based FASTA record index")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto", help="Inference device")
    args = parser.parse_args()

    metadata = generate_structured_music_from_fasta(
        fasta_path=args.fasta,
        checkpoint_path=args.checkpoint,
        output_path=args.output,
        config_path=args.config,
        record_index=args.record_index,
        metadata_output=args.metadata_output,
        device_name=args.device,
    )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
