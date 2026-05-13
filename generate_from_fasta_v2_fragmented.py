#!/usr/bin/env python3
"""Generate MIDI from FASTA using fragmented generation for long sequences."""

from __future__ import annotations

import argparse
import json

from bio_music_pipeline.v2 import generate_structured_music_from_fasta_fragmented


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate harmony+melody MIDI from FASTA with fragmented generation for long sequences"
    )
    parser.add_argument("--fasta", type=str, required=True, help="Path to FASTA file")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to a trained v2 checkpoint")
    parser.add_argument("--output", type=str, required=True, help="Output MIDI path")
    parser.add_argument(
        "--bars-per-fragment",
        type=int,
        default=4,
        choices=[4, 8],
        help="Number of bars to generate per fragment (4 or 8). Default: 4 (recommended for better quality)",
    )
    parser.add_argument(
        "--metadata-output",
        type=str,
        default=None,
        help="Optional path to save generation metadata JSON",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Optional v2 JSON config override. Defaults to the config stored in the checkpoint.",
    )
    parser.add_argument("--record-index", type=int, default=0, help="0-based FASTA record index")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto", help="Inference device")
    args = parser.parse_args()

    print(f"Generating music from FASTA with fragmented approach...")
    print(f"  FASTA: {args.fasta}")
    print(f"  Checkpoint: {args.checkpoint}")
    print(f"  Bars per fragment: {args.bars_per_fragment}")
    print(f"  Output: {args.output}")
    print()

    metadata = generate_structured_music_from_fasta_fragmented(
        fasta_path=args.fasta,
        checkpoint_path=args.checkpoint,
        output_path=args.output,
        bars_per_fragment=args.bars_per_fragment,
        config_path=args.config,
        record_index=args.record_index,
        metadata_output=args.metadata_output,
        device_name=args.device,
    )

    print("\nGeneration complete!")
    print(f"  Sequence: {metadata['sequence_id']}")
    print(f"  Length: {metadata['full_sequence_length']} bp")
    print(f"  Fragments: {metadata['num_fragments']}")
    print(f"  Total bars: {metadata['total_bars']}")
    print(f"  Total melody notes: {metadata['total_melody_notes']}")
    print(f"  Output MIDI: {metadata['output_midi']}")
    print()
    print("Full metadata:")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
