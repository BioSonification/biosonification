"""Inference helpers for the v2 biosonification pipeline."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch

from .bio import BiologicalSequenceEncoder
from .config import V2PipelineConfig, load_v2_config
from .dataset import PolyphonicMusicTokenizer
from .model import ControlConditionedTransformer


def _mode_from_profile(profile: np.ndarray) -> str:
    return "major" if float(profile[5]) >= 0.5 else "minor"


def _apply_calibration(profile: np.ndarray, calibration: Dict[str, np.ndarray]) -> np.ndarray:
    calibrated = ((profile - calibration["bio_mean"]) / (calibration["bio_std"] + 1e-6))
    calibrated = calibrated * calibration["music_std"] + calibration["music_mean"]
    return np.clip(calibrated, 0.0, 1.0).astype(np.float32)


def _load_checkpoint(checkpoint_path: str) -> dict:
    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if "model_state_dict" not in checkpoint:
        raise ValueError(f"Checkpoint {checkpoint_path} does not contain model weights.")
    return checkpoint


def _instantiate_model(config: V2PipelineConfig, tokenizer: PolyphonicMusicTokenizer, checkpoint: dict) -> ControlConditionedTransformer:
    model = ControlConditionedTransformer(
        vocab_size=len(tokenizer.vocab),
        bio_vector_dim=config.bio.embedding_dim,
        max_seq_len=config.training.max_seq_len,
        pad_token_id=tokenizer.pad_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        sep_token_id=tokenizer.sep_token_id,
        d_model=config.training.d_model,
        n_heads=config.training.n_heads,
        n_layers=config.training.n_layers,
        dim_feedforward=config.training.dim_feedforward,
        dropout=config.training.dropout,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def generate_music_from_fasta(
    fasta_path: str,
    checkpoint_path: str,
    output_path: str,
    config_path: Optional[str] = None,
    record_index: int = 0,
    temperature: Optional[float] = None,
    top_k: Optional[int] = None,
    top_p: Optional[float] = None,
    max_new_tokens: Optional[int] = None,
    min_new_tokens: Optional[int] = None,
    metadata_output: Optional[str] = None,
) -> Dict[str, str]:
    config = load_v2_config(config_path)
    checkpoint = _load_checkpoint(checkpoint_path)
    encoder = BiologicalSequenceEncoder(config.bio)
    results = encoder.encode_fasta(fasta_path)
    if record_index < 0 or record_index >= len(results):
        raise IndexError(f"record_index={record_index} is outside the valid range [0, {len(results) - 1}]")

    bio_result = results[record_index]
    tokenizer = PolyphonicMusicTokenizer(config.music)
    model = _instantiate_model(config, tokenizer, checkpoint)
    calibration = checkpoint.get("train_calibration")
    if calibration is None:
        raise ValueError("Checkpoint does not contain train calibration statistics.")

    calibrated_profile = _apply_calibration(bio_result.control_profile, calibration)
    control_tokens = tokenizer.control_tokens(calibrated_profile, _mode_from_profile(calibrated_profile))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    bio_tensor = torch.tensor(bio_result.vector, dtype=torch.float32, device=device)
    generated = model.generate(
        bio_vector=bio_tensor,
        control_token_ids=control_tokens,
        max_new_tokens=max_new_tokens or config.generation.max_new_tokens,
        temperature=temperature or config.generation.temperature,
        top_k=top_k or config.generation.top_k,
        top_p=top_p or config.generation.top_p,
        min_new_tokens=min_new_tokens or config.generation.min_new_tokens,
    )

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    tempo_bpm = 48.0 + float(calibrated_profile[0]) * 120.0
    tokenizer.write_midi(generated.tolist(), str(output), tempo_bpm=tempo_bpm)

    metadata = {
        "sequence_id": bio_result.sequence_id,
        "sequence_type": bio_result.sequence_type,
        "cleaned_sequence_length": len(bio_result.cleaned_sequence),
        "translated_protein_length": len(bio_result.translated_protein),
        "predicted_structure_length": len(bio_result.predicted_structure),
        "output_midi": str(output),
        "checkpoint_path": checkpoint_path,
        "tempo_bpm": tempo_bpm,
        "calibrated_profile": [float(value) for value in calibrated_profile],
    }
    if metadata_output:
        meta_path = Path(metadata_output)
        meta_path.parent.mkdir(parents=True, exist_ok=True)
        with open(meta_path, "w", encoding="utf-8") as handle:
            json.dump(metadata, handle, indent=2)
    return metadata
