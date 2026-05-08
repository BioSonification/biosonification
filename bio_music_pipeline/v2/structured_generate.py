"""Inference for the hierarchical harmony+melody generator."""

from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch

from .bio import BiologicalSequenceEncoder
from .config import V2PipelineConfig, load_v2_config, v2_config_from_dict
from .structured_model import BioConditionedSequenceModel
from .structured_music import (
    HarmonyTokenizer,
    MelodyTokenizer,
    render_harmony_and_melody_to_score,
)


def _trusted_torch_load(checkpoint_path: str, map_location: str | torch.device = "cpu") -> dict:
    try:
        return torch.load(checkpoint_path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(checkpoint_path, map_location=map_location)


def _apply_calibration(profile: np.ndarray, calibration: Dict[str, np.ndarray]) -> np.ndarray:
    calibrated = ((profile - calibration["bio_mean"]) / (calibration["bio_std"] + 1e-6))
    calibrated = calibrated * calibration["music_std"] + calibration["music_mean"]
    return np.clip(calibrated, 0.0, 1.0).astype(np.float32)


def _mode_from_profile(profile: np.ndarray) -> str:
    return "major" if float(profile[5]) >= 0.5 else "minor"


def _config_hash(config: V2PipelineConfig) -> str:
    payload = json.dumps(asdict(config), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]


def _load_effective_config(checkpoint: dict, config_path: Optional[str]) -> tuple[V2PipelineConfig, str]:
    if config_path is not None:
        return load_v2_config(config_path), str(config_path)
    checkpoint_config = checkpoint.get("config")
    if isinstance(checkpoint_config, dict):
        return v2_config_from_dict(checkpoint_config), "checkpoint"
    return load_v2_config(None), "defaults"


def _validate_checkpoint_compatibility(config: V2PipelineConfig, checkpoint: dict) -> None:
    tokenizer_info = checkpoint.get("tokenizer_info") or {}
    harmony_tokenizer = HarmonyTokenizer(config.music)
    melody_tokenizer = MelodyTokenizer(config.music)

    expected_harmony_vocab = int(tokenizer_info.get("harmony_vocab_size", len(harmony_tokenizer.vocab)))
    expected_melody_vocab = int(tokenizer_info.get("melody_vocab_size", len(melody_tokenizer.vocab)))
    if expected_harmony_vocab != len(harmony_tokenizer.vocab):
        raise ValueError(
            "Checkpoint/config mismatch: harmony vocab size is "
            f"{expected_harmony_vocab}, but effective config builds {len(harmony_tokenizer.vocab)} tokens."
        )
    if expected_melody_vocab != len(melody_tokenizer.vocab):
        raise ValueError(
            "Checkpoint/config mismatch: melody vocab size is "
            f"{expected_melody_vocab}, but effective config builds {len(melody_tokenizer.vocab)} tokens."
        )

    checkpoint_config = checkpoint.get("config")
    if not isinstance(checkpoint_config, dict):
        return

    saved_config = v2_config_from_dict(checkpoint_config)
    checks = [
        ("bio.embedding_dim", saved_config.bio.embedding_dim, config.bio.embedding_dim),
        ("music.descriptor_bins", saved_config.music.descriptor_bins, config.music.descriptor_bins),
        ("music.steps_per_bar", saved_config.music.steps_per_bar, config.music.steps_per_bar),
        ("music.steps_per_beat", saved_config.music.steps_per_beat, config.music.steps_per_beat),
        ("training.d_model", saved_config.training.d_model, config.training.d_model),
        ("training.n_heads", saved_config.training.n_heads, config.training.n_heads),
        ("training.n_layers", saved_config.training.n_layers, config.training.n_layers),
        ("training.dim_feedforward", saved_config.training.dim_feedforward, config.training.dim_feedforward),
        ("training.harmony_max_seq_len", saved_config.training.harmony_max_seq_len, config.training.harmony_max_seq_len),
        ("training.melody_max_seq_len", saved_config.training.melody_max_seq_len, config.training.melody_max_seq_len),
    ]
    mismatches = [f"{name}: checkpoint={saved!r}, effective={current!r}" for name, saved, current in checks if saved != current]
    if mismatches:
        raise ValueError("Checkpoint/config mismatch:\n" + "\n".join(mismatches))


def _resolve_device(device_name: str) -> torch.device:
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is false.")
    return torch.device(device_name)


def _instantiate_models(config: V2PipelineConfig, checkpoint: dict) -> tuple[BioConditionedSequenceModel, BioConditionedSequenceModel]:
    harmony_tokenizer = HarmonyTokenizer(config.music)
    melody_tokenizer = MelodyTokenizer(config.music)
    harmony_model = BioConditionedSequenceModel(
        vocab_size=len(harmony_tokenizer.vocab),
        bio_vector_dim=config.bio.embedding_dim,
        max_seq_len=config.training.harmony_max_seq_len,
        pad_token_id=harmony_tokenizer.pad_token_id,
        bos_token_id=harmony_tokenizer.bos_token_id,
        eos_token_id=harmony_tokenizer.eos_token_id,
        d_model=config.training.d_model,
        n_heads=config.training.n_heads,
        n_layers=config.training.n_layers,
        dim_feedforward=config.training.dim_feedforward,
        dropout=config.training.dropout,
    )
    melody_model = BioConditionedSequenceModel(
        vocab_size=len(melody_tokenizer.vocab),
        bio_vector_dim=config.bio.embedding_dim,
        max_seq_len=config.training.melody_max_seq_len,
        pad_token_id=melody_tokenizer.pad_token_id,
        bos_token_id=melody_tokenizer.bos_token_id,
        eos_token_id=melody_tokenizer.eos_token_id,
        d_model=config.training.d_model,
        n_heads=config.training.n_heads,
        n_layers=config.training.n_layers,
        dim_feedforward=config.training.dim_feedforward,
        dropout=config.training.dropout,
    )
    harmony_model.load_state_dict(checkpoint["harmony_model_state_dict"])
    melody_model.load_state_dict(checkpoint["melody_model_state_dict"])
    harmony_model.eval()
    melody_model.eval()
    return harmony_model, melody_model


def generate_structured_music_from_fasta(
    fasta_path: str,
    checkpoint_path: str,
    output_path: str,
    config_path: Optional[str] = None,
    record_index: int = 0,
    metadata_output: Optional[str] = None,
    device_name: str = "auto",
) -> Dict[str, Any]:
    checkpoint = _trusted_torch_load(checkpoint_path, map_location="cpu")
    config, config_source = _load_effective_config(checkpoint, config_path)
    _validate_checkpoint_compatibility(config, checkpoint)
    calibration = checkpoint.get("train_calibration")
    if calibration is None:
        raise ValueError("Checkpoint does not contain train calibration statistics.")

    encoder = BiologicalSequenceEncoder(config.bio)
    results = encoder.encode_fasta(fasta_path)
    if record_index < 0 or record_index >= len(results):
        raise IndexError(f"record_index={record_index} is outside the valid range [0, {len(results) - 1}]")

    bio_result = results[record_index]
    harmony_tokenizer = HarmonyTokenizer(config.music)
    melody_tokenizer = MelodyTokenizer(config.music)
    harmony_model, melody_model = _instantiate_models(config, checkpoint)
    device = _resolve_device(device_name)
    harmony_model = harmony_model.to(device)
    melody_model = melody_model.to(device)

    calibrated_profile = _apply_calibration(bio_result.control_profile, calibration)
    mode_name = _mode_from_profile(calibrated_profile)
    bio_tensor = torch.tensor(bio_result.vector, dtype=torch.float32, device=device)

    # Calculate num_bars based on sequence length (1 bar per ~200 bp, min 8, max 32)
    sequence_length = len(bio_result.cleaned_sequence)
    num_bars = max(8, min(32, sequence_length // 200))

    # Scale harmony tokens proportionally (8 tokens per bar on average)
    harmony_max_tokens = num_bars * 8

    # Scale melody tokens to ensure full coverage (48 tokens per bar on average)
    melody_max_tokens = num_bars * 48
    melody_min_tokens = num_bars * 16  # Minimum to avoid early stopping

    harmony_prefix = [
        harmony_tokenizer.bos_token_id,
        *harmony_tokenizer.control_tokens(calibrated_profile, bio_result.tonic_pc_hint, mode_name),
        harmony_tokenizer.sep_token_id,
    ]
    generated_harmony = harmony_model.generate(
        bio_vector=bio_tensor,
        prefix_token_ids=harmony_prefix,
        max_new_tokens=harmony_max_tokens,
        temperature=config.generation.harmony_temperature,
        top_k=config.generation.harmony_top_k,
        top_p=config.generation.harmony_top_p,
        stop_token_ids=[harmony_tokenizer.eos_token_id],
    )
    harmony_bars = harmony_tokenizer.decode_progression(generated_harmony.tolist(), num_bars)

    melody_prefix = [
        melody_tokenizer.bos_token_id,
        *melody_tokenizer.control_tokens(calibrated_profile, bio_result.tonic_pc_hint, mode_name),
        *(melody_tokenizer.token_to_id[token] for token in melody_tokenizer.harmony_prefix_tokens(harmony_bars)),
        melody_tokenizer.sep_token_id,
    ]
    generated_melody = melody_model.generate(
        bio_vector=bio_tensor,
        prefix_token_ids=melody_prefix,
        max_new_tokens=melody_max_tokens,
        temperature=config.generation.melody_temperature,
        top_k=config.generation.melody_top_k,
        top_p=config.generation.melody_top_p,
        min_new_tokens=melody_min_tokens,
        stop_token_ids=[melody_tokenizer.eos_token_id],
    )
    decoded_melody = melody_tokenizer.decode_melody(generated_melody.tolist(), harmony_bars, bio_result.tonic_pc_hint)

    tempo_bpm = 48.0 + float(calibrated_profile[0]) * 120.0
    score = render_harmony_and_melody_to_score(harmony_bars, decoded_melody, tempo_bpm, config.music)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    score.write("midi", fp=str(output))

    metadata = {
        "sequence_id": bio_result.sequence_id,
        "sequence_type": bio_result.sequence_type,
        "cleaned_sequence_length": len(bio_result.cleaned_sequence),
        "translated_protein_length": len(bio_result.translated_protein),
        "predicted_structure_length": len(bio_result.predicted_structure),
        "tonic_pc_hint": int(bio_result.tonic_pc_hint),
        "output_midi": str(output),
        "checkpoint_path": checkpoint_path,
        "config_source": config_source,
        "effective_config_hash": _config_hash(config),
        "device": str(device),
        "tempo_bpm": tempo_bpm,
        "num_bars": num_bars,
        "harmony_max_tokens": harmony_max_tokens,
        "melody_max_tokens": melody_max_tokens,
        "melody_min_tokens": melody_min_tokens,
        "calibrated_profile": [float(value) for value in calibrated_profile],
        "generated_harmony_bars": [
            {"root_pc": int(bar.root_pc), "quality": bar.quality, "hold": bool(bar.hold)}
            for bar in harmony_bars
        ],
        "generated_melody_note_count": len(decoded_melody),
    }
    if metadata_output:
        meta_path = Path(metadata_output)
        meta_path.parent.mkdir(parents=True, exist_ok=True)
        with open(meta_path, "w", encoding="utf-8") as handle:
            json.dump(metadata, handle, indent=2)
    return metadata


def generate_structured_music_from_fasta_fragmented(
    fasta_path: str,
    checkpoint_path: str,
    output_path: str,
    bars_per_fragment: int = 4,
    config_path: Optional[str] = None,
    record_index: int = 0,
    metadata_output: Optional[str] = None,
    device_name: str = "auto",
) -> Dict[str, Any]:
    """
    Generate music from FASTA by fragmenting the sequence and concatenating results.

    This approach generates multiple short segments (4 or 8 bars each) and concatenates them,
    ensuring each segment stays within the model's training distribution for better quality.

    Args:
        fasta_path: Path to FASTA file
        checkpoint_path: Path to model checkpoint
        output_path: Output MIDI path
        bars_per_fragment: Number of bars to generate per fragment (4 or 8)
        config_path: Optional config path
        record_index: Which record to use from FASTA
        metadata_output: Optional metadata JSON output path
        device_name: Device to use (auto/cuda/cpu)

    Returns:
        Metadata dictionary with generation info
    """
    import music21

    checkpoint = _trusted_torch_load(checkpoint_path, map_location="cpu")
    config, config_source = _load_effective_config(checkpoint, config_path)
    _validate_checkpoint_compatibility(config, checkpoint)
    calibration = checkpoint.get("train_calibration")
    if calibration is None:
        raise ValueError("Checkpoint does not contain train calibration statistics.")

    encoder = BiologicalSequenceEncoder(config.bio)

    # Parse FASTA to get the full sequence
    from Bio import SeqIO
    records = list(SeqIO.parse(fasta_path, "fasta"))
    if record_index < 0 or record_index >= len(records):
        raise IndexError(f"record_index={record_index} is outside the valid range [0, {len(records) - 1}]")

    record = records[record_index]
    full_sequence = str(record.seq).upper()
    sequence_id = record.id

    # Fragment the sequence
    fragment_length = config.bio.fragment_length
    stride = fragment_length  # No overlap for now
    min_length = config.bio.min_sequence_length

    fragments = []
    for start in range(0, len(full_sequence), stride):
        fragment_seq = full_sequence[start:start + fragment_length]
        if len(fragment_seq) >= min_length:
            fragments.append((start, fragment_seq))

    if not fragments:
        raise ValueError(f"Sequence too short to fragment (length={len(full_sequence)}, min={min_length})")

    # Load models
    harmony_tokenizer = HarmonyTokenizer(config.music)
    melody_tokenizer = MelodyTokenizer(config.music)
    harmony_model, melody_model = _instantiate_models(config, checkpoint)
    device = _resolve_device(device_name)
    harmony_model = harmony_model.to(device)
    melody_model = melody_model.to(device)

    # Generate music for each fragment
    segment_scores = []
    fragment_metadata = []

    for frag_idx, (start_pos, fragment_seq) in enumerate(fragments):
        # Encode fragment
        bio_result = encoder.encode_sequence(
            sequence=fragment_seq,
            sequence_id=f"{sequence_id}::frag{frag_idx:03d}"
        )

        calibrated_profile = _apply_calibration(bio_result.control_profile, calibration)
        mode_name = _mode_from_profile(calibrated_profile)
        bio_tensor = torch.tensor(bio_result.vector, dtype=torch.float32, device=device)

        # Fixed number of bars per fragment
        num_bars = bars_per_fragment
        harmony_max_tokens = num_bars * 8
        melody_max_tokens = num_bars * 48
        melody_min_tokens = num_bars * 16

        # Generate harmony
        harmony_prefix = [
            harmony_tokenizer.bos_token_id,
            *harmony_tokenizer.control_tokens(calibrated_profile, bio_result.tonic_pc_hint, mode_name),
            harmony_tokenizer.sep_token_id,
        ]
        generated_harmony = harmony_model.generate(
            bio_vector=bio_tensor,
            prefix_token_ids=harmony_prefix,
            max_new_tokens=harmony_max_tokens,
            temperature=config.generation.harmony_temperature,
            top_k=config.generation.harmony_top_k,
            top_p=config.generation.harmony_top_p,
            stop_token_ids=[harmony_tokenizer.eos_token_id],
        )
        harmony_bars = harmony_tokenizer.decode_progression(generated_harmony.tolist(), num_bars)

        # Generate melody
        melody_prefix = [
            melody_tokenizer.bos_token_id,
            *melody_tokenizer.control_tokens(calibrated_profile, bio_result.tonic_pc_hint, mode_name),
            *(melody_tokenizer.token_to_id[token] for token in melody_tokenizer.harmony_prefix_tokens(harmony_bars)),
            melody_tokenizer.sep_token_id,
        ]
        generated_melody = melody_model.generate(
            bio_vector=bio_tensor,
            prefix_token_ids=melody_prefix,
            max_new_tokens=melody_max_tokens,
            temperature=config.generation.melody_temperature,
            top_k=config.generation.melody_top_k,
            top_p=config.generation.melody_top_p,
            min_new_tokens=melody_min_tokens,
            stop_token_ids=[melody_tokenizer.eos_token_id],
        )
        decoded_melody = melody_tokenizer.decode_melody(generated_melody.tolist(), harmony_bars, bio_result.tonic_pc_hint)

        # Use average tempo across all fragments
        tempo_bpm = 48.0 + float(calibrated_profile[0]) * 120.0

        # Render segment
        segment_score = render_harmony_and_melody_to_score(harmony_bars, decoded_melody, tempo_bpm, config.music)
        segment_scores.append(segment_score)

        fragment_metadata.append({
            "fragment_index": frag_idx,
            "start_position": start_pos,
            "fragment_length": len(fragment_seq),
            "num_bars": num_bars,
            "tempo_bpm": tempo_bpm,
            "harmony_bars": len(harmony_bars),
            "melody_notes": len(decoded_melody),
        })

    # Concatenate all segments
    final_score = music21.stream.Score()

    # Use tempo from first fragment
    first_tempo = 48.0 + float(_apply_calibration(
        encoder.encode_sequence(sequence=fragments[0][1], sequence_id=sequence_id).control_profile,
        calibration
    )[0]) * 120.0
    final_score.insert(0, music21.tempo.MetronomeMark(number=first_tempo))

    # Create parts for harmony and melody
    harmony_part = music21.stream.Part()
    melody_part = music21.stream.Part()

    harmony_part.insert(0, music21.instrument.Piano())
    melody_part.insert(0, music21.instrument.Piano())

    # Concatenate all notes from segments
    # Calculate offset for each segment
    current_offset = 0.0
    for segment_score in segment_scores:
        if len(segment_score.parts) >= 1:
            # Copy harmony notes with offset
            for element in segment_score.parts[0].flatten().notesAndRests:
                new_element = element
                new_element.offset = element.offset + current_offset
                harmony_part.insert(new_element.offset, new_element)

        if len(segment_score.parts) >= 2:
            # Copy melody notes with offset
            for element in segment_score.parts[1].flatten().notesAndRests:
                new_element = element
                new_element.offset = element.offset + current_offset
                melody_part.insert(new_element.offset, new_element)

        # Update offset for next segment (4 bars = 16 quarter notes)
        current_offset += bars_per_fragment * 4.0

    final_score.append(harmony_part)
    final_score.append(melody_part)

    # Save MIDI
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    final_score.write("midi", fp=str(output))

    # Metadata
    total_bars = len(fragments) * bars_per_fragment
    total_melody_notes = sum(fm["melody_notes"] for fm in fragment_metadata)

    metadata = {
        "sequence_id": sequence_id,
        "full_sequence_length": len(full_sequence),
        "fragment_length": fragment_length,
        "stride": stride,
        "num_fragments": len(fragments),
        "bars_per_fragment": bars_per_fragment,
        "total_bars": total_bars,
        "total_melody_notes": total_melody_notes,
        "output_midi": str(output),
        "checkpoint_path": checkpoint_path,
        "config_source": config_source,
        "device": str(device),
        "fragments": fragment_metadata,
    }

    if metadata_output:
        meta_path = Path(metadata_output)
        meta_path.parent.mkdir(parents=True, exist_ok=True)
        with open(meta_path, "w", encoding="utf-8") as handle:
            json.dump(metadata, handle, indent=2)

    return metadata

