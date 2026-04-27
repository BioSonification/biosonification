#!/usr/bin/env python3
"""
Generate one MIDI file from one FASTA fragment.

LEGACY: this script uses the original single-stream pipeline. The current
recommended generator is `generate_from_fasta_v2.py`.

This utility is intended as a quick end-to-end sanity check:
FASTA fragment -> bio-vector -> conditioned generation -> MIDI.
"""

import sys
import json
import argparse
from pathlib import Path

import numpy as np
import torch

# Add package to path
sys.path.insert(0, str(Path(__file__).parent))

from bio_music_pipeline import set_seed
from bio_music_pipeline.extractors import BioVectorExtractor, FastaDatasetLoader
from bio_music_pipeline.sonification import SonificationMapper
from bio_music_pipeline.models import create_bio_music_model
from bio_music_pipeline.data import MIDIPreprocessor
from bio_music_pipeline.utils import tokens_to_midi


def _load_fragment(fasta_path: Path,
                   record_index: int,
                   fragment_start: int,
                   fragment_length: int,
                   min_sequence_length: int):
    """Load one cleaned FASTA record and cut one fragment from it."""
    loader = FastaDatasetLoader(min_sequence_length=min_sequence_length)

    valid_idx = -1
    selected_header = None
    selected_sequence = None

    for header, sequence in loader.read_fasta_file(str(fasta_path)):
        clean_seq = ''.join([c for c in sequence.upper() if c in 'ACGT'])
        if len(clean_seq) < min_sequence_length:
            continue
        valid_idx += 1
        if valid_idx == record_index:
            selected_header = header
            selected_sequence = clean_seq
            break

    if selected_sequence is None:
        raise ValueError(
            f"Record index {record_index} not found among valid FASTA records "
            f"(min_length={min_sequence_length}) in {fasta_path}"
        )

    fragment_start = max(0, fragment_start)
    fragment_end = min(len(selected_sequence), fragment_start + max(fragment_length, min_sequence_length))
    fragment = selected_sequence[fragment_start:fragment_end]

    if len(fragment) < min_sequence_length:
        # Fallback: use the full selected sequence if requested fragment is too short.
        fragment = selected_sequence
        fragment_start = 0
        fragment_end = len(selected_sequence)

    return selected_header, selected_sequence, fragment, fragment_start, fragment_end


def main():
    print(
        "WARNING: generate_from_fasta.py is legacy. "
        "Use generate_from_fasta_v2.py for the current structured v2 pipeline.",
        file=sys.stderr,
    )
    parser = argparse.ArgumentParser(description="Generate MIDI from one FASTA fragment")
    parser.add_argument('--fasta', type=str,
                        default='data/fasta/training/Homo_sapiens.GRCh38.cds.all.fa',
                        help='Path to FASTA file')
    parser.add_argument('--record-index', type=int, default=0,
                        help='Index of valid FASTA record (0-based)')
    parser.add_argument('--fragment-start', type=int, default=0,
                        help='Fragment start position in selected record (0-based)')
    parser.add_argument('--fragment-length', type=int, default=2000,
                        help='Fragment length in nucleotides')
    parser.add_argument('--config', type=str, default='configs/pipeline_full_paired.json',
                        help='Path to pipeline config')
    parser.add_argument('--model', type=str, default='results/full_paired_run/models/best_model.pt',
                        help='Path to trained model checkpoint')
    parser.add_argument('--output', type=str,
                        default='results/single_fasta_check/single_fasta_fragment.mid',
                        help='Output MIDI path')
    parser.add_argument('--metadata-output', type=str,
                        default='results/single_fasta_check/single_fasta_fragment_meta.json',
                        help='Output metadata JSON path')
    parser.add_argument('--temperature', type=float, default=0.95,
                        help='Sampling temperature')
    parser.add_argument('--min-gen-len', type=int, default=256,
                        help='Minimum generated token length before EOS is allowed')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    args = parser.parse_args()

    set_seed(args.seed)

    fasta_path = Path(args.fasta)
    config_path = Path(args.config)
    model_path = Path(args.model)
    output_path = Path(args.output)
    meta_path = Path(args.metadata_output)

    if not fasta_path.exists():
        raise FileNotFoundError(f"FASTA file not found: {fasta_path}")
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    if not model_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.parent.mkdir(parents=True, exist_ok=True)

    with open(config_path, 'r') as f:
        config = json.load(f)

    min_seq_len = int(config['extraction']['min_sequence_length'])

    print(f"Loading FASTA record #{args.record_index} from: {fasta_path}")
    header, full_seq, fragment, frag_start, frag_end = _load_fragment(
        fasta_path=fasta_path,
        record_index=args.record_index,
        fragment_start=args.fragment_start,
        fragment_length=args.fragment_length,
        min_sequence_length=min_seq_len,
    )

    print(f"Selected header: {header[:120]}")
    print(f"Full sequence length: {len(full_seq)} bp")
    print(f"Fragment range: [{frag_start}:{frag_end}] ({len(fragment)} bp)")

    extractor = BioVectorExtractor(
        kmer_sizes=config['extraction']['kmer_sizes'],
        window_size=config['extraction']['window_size'],
        stride=config['extraction']['stride'],
        min_sequence_length=min_seq_len,
    )
    features = extractor.extract_features(fragment)
    target_dim = int(config['model']['bio_vector_dim'])
    bio_vector = extractor.create_bio_vector(features, target_dim)

    sonification_config = config['sonification']
    mapper = SonificationMapper(
        tempo_range=tuple(sonification_config['tempo_range']),
        pitch_range=tuple(sonification_config['pitch_range']),
        key_mapping=sonification_config['key_mapping'],
        chord_complexity_levels=sonification_config['chord_complexity_levels'],
    )
    musical_params = mapper.bio_vector_to_musical_params(bio_vector)

    checkpoint = torch.load(model_path, map_location='cpu')
    model_config = checkpoint['config']
    model = create_bio_music_model(model_config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    bio_tensor = torch.tensor(bio_vector, dtype=torch.float32).unsqueeze(0)
    max_len = int(model_config.get('max_seq_len', 512))

    with torch.no_grad():
        generated = model.generate(
            bio_tensor,
            max_len=max_len,
            temperature=float(args.temperature),
            use_gumbel=False,
            min_len=int(args.min_gen_len),
        )

    tokens = generated.squeeze(0).cpu().numpy().tolist()

    # Build vocab identically to training pipeline tokenizer format.
    data_cfg = config.get('data', {})
    preprocessor = MIDIPreprocessor(
        max_seq_len=max_len,
        min_duration=float(data_cfg.get('min_midi_duration', 30.0)),
        max_duration=float(data_cfg.get('max_midi_duration', 300.0)),
    )

    success = tokens_to_midi(
        tokens=tokens,
        preprocessor_vocab=preprocessor.token_to_idx,
        output_path=str(output_path),
        tempo=int(round(float(musical_params.tempo))),
    )

    if not success:
        raise RuntimeError("Failed to convert generated tokens to MIDI")

    eos_id = preprocessor.eos_token_id
    eos_positions = [i for i, t in enumerate(tokens) if t == eos_id]
    first_eos = eos_positions[0] if eos_positions else len(tokens)

    metadata = {
        'fasta_path': str(fasta_path),
        'record_index': int(args.record_index),
        'header': header,
        'full_sequence_length': int(len(full_seq)),
        'fragment_start': int(frag_start),
        'fragment_end': int(frag_end),
        'fragment_length': int(len(fragment)),
        'model_checkpoint': str(model_path),
        'output_midi': str(output_path),
        'generated_token_length': int(len(tokens)),
        'first_eos_position': int(first_eos),
        'musical_params': {
            'key': str(musical_params.key),
            'tempo': float(musical_params.tempo),
            'pitch_range': [int(x) for x in musical_params.pitch_range],
            'rhythm_complexity': float(musical_params.rhythm_complexity),
            'scale_type': str(musical_params.scale_type),
            'articulation_density': float(musical_params.articulation_density),
        },
    }

    with open(meta_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    print("\nSUCCESS")
    print(f"MIDI saved: {output_path}")
    print(f"Metadata saved: {meta_path}")
    print(f"Generated token length: {len(tokens)}, first EOS at: {first_eos}")


if __name__ == '__main__':
    main()
