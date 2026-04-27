"""Pairing utilities for the structured harmony+melody pipeline."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

from .bio import BioEncodingResult
from .config import PairingConfig
from .structured_music import StructuredMusicSegment


@dataclass
class StructuredPairedSample:
    sequence_id: str
    segment_id: str
    bio_vector: np.ndarray
    descriptor_vector: np.ndarray
    harmony_token_ids: List[int]
    harmony_prefix_ids: List[int]
    melody_token_ids: List[int]
    melody_prefix_ids: List[int]
    pair_weight: float
    distance: float
    tempo_bpm: float
    tonic_pc: int
    mode_name: str
    source_path: str


def _weighted_distance(left: np.ndarray, right: np.ndarray, weights: np.ndarray) -> np.ndarray:
    diff = left[:, None, :] - right[None, :, :]
    return np.sqrt(np.sum(weights[None, None, :] * diff * diff, axis=-1))


def _softmax(values: np.ndarray) -> np.ndarray:
    values = values - np.max(values)
    exp_values = np.exp(values)
    denom = np.sum(exp_values)
    if denom <= 0:
        return np.ones_like(values) / max(values.size, 1)
    return exp_values / denom


def calibrate_bio_profiles(
    bio_results: Sequence[BioEncodingResult],
    music_segments: Sequence[StructuredMusicSegment],
) -> Dict[str, np.ndarray]:
    bio_profiles = np.stack([item.control_profile for item in bio_results])
    music_profiles = np.stack([segment.descriptor_vector for segment in music_segments])
    bio_mean = bio_profiles.mean(axis=0)
    bio_std = bio_profiles.std(axis=0) + 1e-6
    music_mean = music_profiles.mean(axis=0)
    music_std = music_profiles.std(axis=0) + 1e-6
    calibrated = np.clip(((bio_profiles - bio_mean) / bio_std) * music_std + music_mean, 0.0, 1.0)
    return {
        "bio_mean": bio_mean,
        "bio_std": bio_std,
        "music_mean": music_mean,
        "music_std": music_std,
        "calibrated_profiles": calibrated,
    }


def build_structured_paired_dataset(
    bio_results: Sequence[BioEncodingResult],
    music_segments: Sequence[StructuredMusicSegment],
    config: PairingConfig | None = None,
) -> Tuple[List[StructuredPairedSample], Dict[str, np.ndarray]]:
    pairing_config = config or PairingConfig()
    weights = np.array(
        [
            pairing_config.descriptor_weight_tempo,
            pairing_config.descriptor_weight_density,
            pairing_config.descriptor_weight_polyphony,
            pairing_config.descriptor_weight_register,
            pairing_config.descriptor_weight_harmony,
            pairing_config.descriptor_weight_mode,
        ],
        dtype=np.float32,
    )
    calibration = calibrate_bio_profiles(bio_results, music_segments)
    calibrated_profiles = calibration["calibrated_profiles"]
    music_profiles = np.stack([segment.descriptor_vector for segment in music_segments])
    distances = _weighted_distance(calibrated_profiles, music_profiles, weights)

    samples: List[StructuredPairedSample] = []
    for bio_index, bio_result in enumerate(bio_results):
        row = distances[bio_index]
        top_indices = np.argsort(row)[: pairing_config.top_k]
        weights_row = _softmax(-row[top_indices] / max(pairing_config.temperature, 1e-6))
        for score, music_index in zip(weights_row, top_indices):
            segment = music_segments[int(music_index)]
            samples.append(
                StructuredPairedSample(
                    sequence_id=bio_result.sequence_id,
                    segment_id=segment.segment_id,
                    bio_vector=bio_result.vector.astype(np.float32),
                    descriptor_vector=segment.descriptor_vector.astype(np.float32),
                    harmony_token_ids=list(segment.harmony_token_ids),
                    harmony_prefix_ids=list(segment.harmony_prefix_ids),
                    melody_token_ids=list(segment.melody_token_ids),
                    melody_prefix_ids=list(segment.melody_prefix_ids),
                    pair_weight=float(score),
                    distance=float(row[music_index]),
                    tempo_bpm=float(segment.tempo_bpm),
                    tonic_pc=int(segment.key_tonic_pc),
                    mode_name=str(segment.key_mode),
                    source_path=segment.source_path,
                )
            )
    return samples, calibration


def save_structured_pairing_artifacts(
    output_dir: str,
    paired_samples: Sequence[StructuredPairedSample],
    calibration: Dict[str, np.ndarray],
) -> None:
    target_dir = Path(output_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    manifest = []
    for sample in paired_samples:
        manifest.append(
            {
                "sequence_id": sample.sequence_id,
                "segment_id": sample.segment_id,
                "pair_weight": sample.pair_weight,
                "distance": sample.distance,
                "tempo_bpm": sample.tempo_bpm,
                "tonic_pc": sample.tonic_pc,
                "mode_name": sample.mode_name,
                "source_path": sample.source_path,
                "descriptor_vector": [float(value) for value in sample.descriptor_vector],
            }
        )
    with open(target_dir / "pair_manifest.json", "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
    np.savez_compressed(
        target_dir / "pair_calibration.npz",
        **{key: value.astype(np.float32) for key, value in calibration.items()},
    )
