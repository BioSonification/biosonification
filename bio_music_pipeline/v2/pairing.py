"""Multidimensional bio/music pairing for the v2 pipeline."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np

from .bio import BioEncodingResult
from .config import PairingConfig
from .dataset import MusicSegment


@dataclass
class PairedSample:
    sequence_id: str
    segment_id: str
    bio_vector: np.ndarray
    descriptor_vector: np.ndarray
    token_ids: List[int]
    pair_weight: float
    distance: float
    source_path: str


def _weighted_distance(left: np.ndarray, right: np.ndarray, weights: np.ndarray) -> np.ndarray:
    diff = left[:, None, :] - right[None, :, :]
    return np.sqrt(np.sum(weights[None, None, :] * diff * diff, axis=-1))


def _softmax(values: np.ndarray) -> np.ndarray:
    values = values - np.max(values)
    exp_values = np.exp(values)
    denominator = np.sum(exp_values)
    if denominator <= 0:
        return np.ones_like(values) / max(values.size, 1)
    return exp_values / denominator


def calibrate_bio_profiles(
    bio_results: Sequence[BioEncodingResult],
    music_segments: Sequence[MusicSegment],
) -> Dict[str, np.ndarray]:
    """Align bio control profiles to the empirical music descriptor distribution."""

    bio_profiles = np.stack([item.control_profile for item in bio_results])
    music_profiles = np.stack([item.descriptor_vector for item in music_segments])
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


def build_paired_dataset(
    bio_results: Sequence[BioEncodingResult],
    music_segments: Sequence[MusicSegment],
    config: PairingConfig | None = None,
) -> Tuple[List[PairedSample], Dict[str, np.ndarray]]:
    """Create a many-to-many paired dataset using weighted nearest neighbors."""

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

    paired_samples: List[PairedSample] = []
    for bio_index, bio_result in enumerate(bio_results):
        row = distances[bio_index]
        top_indices = np.argsort(row)[: pairing_config.top_k]
        scores = _softmax(-row[top_indices] / max(pairing_config.temperature, 1e-6))
        for score, music_index in zip(scores, top_indices):
            segment = music_segments[int(music_index)]
            paired_samples.append(
                PairedSample(
                    sequence_id=bio_result.sequence_id,
                    segment_id=segment.segment_id,
                    bio_vector=bio_result.vector.astype(np.float32),
                    descriptor_vector=segment.descriptor_vector.astype(np.float32),
                    token_ids=list(segment.token_ids),
                    pair_weight=float(score),
                    distance=float(row[music_index]),
                    source_path=segment.source_path,
                )
            )
    return paired_samples, calibration


def save_pairing_artifacts(
    output_dir: str,
    paired_samples: Sequence[PairedSample],
    calibration: Dict[str, np.ndarray],
) -> None:
    """Persist pairing manifests for reproducible training and inspection."""

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
