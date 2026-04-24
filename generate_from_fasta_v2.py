#!/usr/bin/env python3
"""Generate MIDI from FASTA using the v2 biosonification pipeline."""

from __future__ import annotations

import argparse
import json

from bio_music_pipeline.v2 import generate_music_from_fasta


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate polyphonic MIDI from FASTA with the v2 pipeline")
    parser.add_argument("--fasta", type=str, default="data/fasta/quick_sample.fa", help="Path to FASTA file")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to a trained v2 checkpoint")
    parser.add_argument("--output", type=str, default="results/v2_generation/generated_from_fasta.mid", help="Output MIDI path")
    parser.add_argument("--metadata-output", type=str, default="results/v2_generation/generated_from_fasta.json", help="Where to save generation metadata")
    parser.add_argument("--config", type=str, default="configs/pipeline_v2_small.json", help="Path to the v2 JSON config")
    parser.add_argument("--record-index", type=int, default=0, help="0-based FASTA record index")
    parser.add_argument("--temperature", type=float, default=None, help="Sampling temperature")
    parser.add_argument("--top-k", type=int, default=None, help="Top-k sampling cutoff")
    parser.add_argument("--top-p", type=float, default=None, help="Top-p nucleus sampling threshold")
    parser.add_argument("--max-new-tokens", type=int, default=None, help="Maximum generated tokens after the prefix")
    parser.add_argument("--min-new-tokens", type=int, default=None, help="Minimum tokens before EOS is allowed")
    args = parser.parse_args()

    metadata = generate_music_from_fasta(
        fasta_path=args.fasta,
        checkpoint_path=args.checkpoint,
        output_path=args.output,
        config_path=args.config,
        record_index=args.record_index,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        max_new_tokens=args.max_new_tokens,
        min_new_tokens=args.min_new_tokens,
        metadata_output=args.metadata_output,
    )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
