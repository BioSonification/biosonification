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

    harmony_prefix = [
        harmony_tokenizer.bos_token_id,
        *harmony_tokenizer.control_tokens(calibrated_profile, bio_result.tonic_pc_hint, mode_name),
        harmony_tokenizer.sep_token_id,
    ]
    generated_harmony = harmony_model.generate(
        bio_vector=bio_tensor,
        prefix_token_ids=harmony_prefix,
        max_new_tokens=config.generation.harmony_max_new_tokens,
        temperature=config.generation.harmony_temperature,
        top_k=config.generation.harmony_top_k,
        top_p=config.generation.harmony_top_p,
        stop_token_ids=[harmony_tokenizer.eos_token_id],
    )
    harmony_bars = harmony_tokenizer.decode_progression(generated_harmony.tolist(), config.generation.num_bars)

    melody_prefix = [
        melody_tokenizer.bos_token_id,
        *melody_tokenizer.control_tokens(calibrated_profile, bio_result.tonic_pc_hint, mode_name),
        *(melody_tokenizer.token_to_id[token] for token in melody_tokenizer.harmony_prefix_tokens(harmony_bars)),
        melody_tokenizer.sep_token_id,
    ]
    generated_melody = melody_model.generate(
        bio_vector=bio_tensor,
        prefix_token_ids=melody_prefix,
        max_new_tokens=config.generation.melody_max_new_tokens,
        temperature=config.generation.melody_temperature,
        top_k=config.generation.melody_top_k,
        top_p=config.generation.melody_top_p,
        min_new_tokens=config.generation.melody_min_new_tokens,
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
