"""Tests for bio_music_pipeline/v2/structured_pairing.py."""

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from bio_music_pipeline.v2.bio import BioEncodingResult
from bio_music_pipeline.v2.config import PairingConfig
from bio_music_pipeline.v2.structured_music import StructuredMusicSegment
from bio_music_pipeline.v2.structured_pairing import (
    StructuredPairedSample,
    _softmax,
    _weighted_distance,
    build_structured_paired_dataset,
    calibrate_bio_profiles,
    save_structured_pairing_artifacts,
)


def test_weighted_distance():
    """Test weighted distance calculation."""
    left = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    right = np.array([[1.0, 2.0, 3.0], [7.0, 8.0, 9.0]])
    weights = np.array([1.0, 1.0, 1.0])

    distances = _weighted_distance(left, right, weights)

    assert distances.shape == (2, 2)
    assert distances[0, 0] == pytest.approx(0.0)  # Same point
    assert distances[0, 1] > 0  # Different points


def test_softmax():
    """Test softmax function."""
    values = np.array([1.0, 2.0, 3.0])
    result = _softmax(values)

    assert result.shape == (3,)
    assert np.sum(result) == pytest.approx(1.0)
    assert np.all(result >= 0)
    assert np.all(result <= 1)
    assert result[2] > result[1] > result[0]  # Higher values get higher probabilities


def test_softmax_zero_denominator():
    """Test softmax with extreme values that could cause zero denominator."""
    values = np.array([-1000.0, -1000.0, -1000.0])
    result = _softmax(values)

    assert result.shape == (3,)
    assert np.sum(result) == pytest.approx(1.0)
    assert np.all(result >= 0)


def test_calibrate_bio_profiles():
    """Test bio profile calibration."""
    bio_results = [
        BioEncodingResult(
            sequence_id="seq1",
            sequence_type="dna",
            cleaned_sequence="ACGT" * 100,
            vector=np.random.randn(256).astype(np.float32),
            control_profile=np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6], dtype=np.float32),
            tonic_pc_hint=0,
            feature_names=[],
            feature_map={},
            translated_protein="",
            predicted_structure="",
        ),
        BioEncodingResult(
            sequence_id="seq2",
            sequence_type="dna",
            cleaned_sequence="ACGT" * 100,
            vector=np.random.randn(256).astype(np.float32),
            control_profile=np.array([0.2, 0.3, 0.4, 0.5, 0.6, 0.7], dtype=np.float32),
            tonic_pc_hint=0,
            feature_names=[],
            feature_map={},
            translated_protein="",
            predicted_structure="",
        ),
    ]

    music_segments = [
        StructuredMusicSegment(
            segment_id="seg1",
            source_path="test.mid",
            start_measure=0,
            end_measure=4,
            harmony_bars=[],
            melody_events=[],
            descriptor_vector=np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5], dtype=np.float32),
            harmony_token_ids=np.array([1, 2, 3]),
            harmony_prefix_ids=np.array([0]),
            melody_token_ids=np.array([4, 5, 6]),
            melody_prefix_ids=np.array([0]),
            tempo_bpm=120.0,
            key_tonic_pc=0,
            key_mode="major",
        ),
    ]

    calibration = calibrate_bio_profiles(bio_results, music_segments)

    assert "bio_mean" in calibration
    assert "bio_std" in calibration
    assert "music_mean" in calibration
    assert "music_std" in calibration
    assert "calibrated_profiles" in calibration

    assert calibration["bio_mean"].shape == (6,)
    assert calibration["calibrated_profiles"].shape == (2, 6)
    assert np.all(calibration["calibrated_profiles"] >= 0.0)
    assert np.all(calibration["calibrated_profiles"] <= 1.0)


def test_build_structured_paired_dataset():
    """Test building paired dataset."""
    bio_results = [
        BioEncodingResult(
            sequence_id="seq1",
            sequence_type="dna",
            cleaned_sequence="ACGT" * 100,
            vector=np.random.randn(256).astype(np.float32),
            control_profile=np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6], dtype=np.float32),
            tonic_pc_hint=0,
            feature_names=[],
            feature_map={},
            translated_protein="",
            predicted_structure="",
        ),
        BioEncodingResult(
            sequence_id="seq2",
            sequence_type="dna",
            cleaned_sequence="ACGT" * 100,
            vector=np.random.randn(256).astype(np.float32),
            control_profile=np.array([0.2, 0.3, 0.4, 0.5, 0.6, 0.7], dtype=np.float32),
            tonic_pc_hint=0,
            feature_names=[],
            feature_map={},
            translated_protein="",
            predicted_structure="",
        ),
    ]

    music_segments = [
        StructuredMusicSegment(
            segment_id="seg1",
            source_path="test1.mid",
            start_measure=0,
            end_measure=4,
            harmony_bars=[],
            melody_events=[],
            descriptor_vector=np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5], dtype=np.float32),
            harmony_token_ids=np.array([1, 2, 3]),
            harmony_prefix_ids=np.array([0]),
            melody_token_ids=np.array([4, 5, 6]),
            melody_prefix_ids=np.array([0]),
            tempo_bpm=120.0,
            key_tonic_pc=0,
            key_mode="major",
        ),
        StructuredMusicSegment(
            segment_id="seg2",
            source_path="test2.mid",
            start_measure=0,
            end_measure=4,
            harmony_bars=[],
            melody_events=[],
            descriptor_vector=np.array([0.6, 0.6, 0.6, 0.6, 0.6, 0.6], dtype=np.float32),
            harmony_token_ids=np.array([7, 8, 9]),
            harmony_prefix_ids=np.array([0]),
            melody_token_ids=np.array([10, 11, 12]),
            melody_prefix_ids=np.array([0]),
            tempo_bpm=140.0,
            key_tonic_pc=2,
            key_mode="minor",
        ),
    ]

    config = PairingConfig(top_k=2, temperature=1.0)
    samples, calibration = build_structured_paired_dataset(bio_results, music_segments, config)

    assert len(samples) == 4  # 2 bio × 2 top_k
    assert all(isinstance(s, StructuredPairedSample) for s in samples)
    assert all(s.pair_weight >= 0 for s in samples)
    assert all(s.distance >= 0 for s in samples)

    # Check that weights sum to 1 for each bio sequence
    seq1_weights = [s.pair_weight for s in samples if s.sequence_id == "seq1"]
    assert sum(seq1_weights) == pytest.approx(1.0)


def test_build_structured_paired_dataset_default_config():
    """Test building paired dataset with default config."""
    bio_results = [
        BioEncodingResult(
            sequence_id="seq1",
            sequence_type="dna",
            cleaned_sequence="ACGT" * 100,
            vector=np.random.randn(256).astype(np.float32),
            control_profile=np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6], dtype=np.float32),
            tonic_pc_hint=0,
            feature_names=[],
            feature_map={},
            translated_protein="",
            predicted_structure="",
        ),
    ]

    music_segments = [
        StructuredMusicSegment(
            segment_id="seg1",
            source_path="test.mid",
            start_measure=0,
            end_measure=4,
            harmony_bars=[],
            melody_events=[],
            descriptor_vector=np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5], dtype=np.float32),
            harmony_token_ids=np.array([1, 2, 3]),
            harmony_prefix_ids=np.array([0]),
            melody_token_ids=np.array([4, 5, 6]),
            melody_prefix_ids=np.array([0]),
            tempo_bpm=120.0,
            key_tonic_pc=0,
            key_mode="major",
        ),
    ]

    samples, calibration = build_structured_paired_dataset(bio_results, music_segments, config=None)

    assert len(samples) > 0
    assert calibration is not None


def test_save_structured_pairing_artifacts():
    """Test saving pairing artifacts."""
    samples = [
        StructuredPairedSample(
            sequence_id="seq1",
            segment_id="seg1",
            bio_vector=np.random.randn(256).astype(np.float32),
            descriptor_vector=np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5], dtype=np.float32),
            harmony_token_ids=[1, 2, 3],
            harmony_prefix_ids=[0],
            melody_token_ids=[4, 5, 6],
            melody_prefix_ids=[0],
            pair_weight=0.8,
            distance=0.2,
            tempo_bpm=120.0,
            tonic_pc=0,
            mode_name="major",
            source_path="test.mid",
        ),
    ]

    calibration = {
        "bio_mean": np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6], dtype=np.float32),
        "bio_std": np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1], dtype=np.float32),
        "music_mean": np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5], dtype=np.float32),
        "music_std": np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1], dtype=np.float32),
        "calibrated_profiles": np.array([[0.5, 0.5, 0.5, 0.5, 0.5, 0.5]], dtype=np.float32),
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        save_structured_pairing_artifacts(tmpdir, samples, calibration)

        manifest_path = Path(tmpdir) / "pair_manifest.json"
        calibration_path = Path(tmpdir) / "pair_calibration.npz"

        assert manifest_path.exists()
        assert calibration_path.exists()

        # Check manifest content
        with open(manifest_path, encoding="utf-8") as f:
            manifest = json.load(f)
        assert len(manifest) == 1
        assert manifest[0]["sequence_id"] == "seq1"
        assert manifest[0]["segment_id"] == "seg1"
        assert manifest[0]["pair_weight"] == pytest.approx(0.8)

        # Check calibration content and close file before cleanup
        with np.load(calibration_path) as loaded_calibration:
            assert "bio_mean" in loaded_calibration
            assert "calibrated_profiles" in loaded_calibration


def test_structured_paired_sample_dataclass():
    """Test StructuredPairedSample dataclass."""
    sample = StructuredPairedSample(
        sequence_id="seq1",
        segment_id="seg1",
        bio_vector=np.array([1.0, 2.0, 3.0]),
        descriptor_vector=np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5]),
        harmony_token_ids=[1, 2, 3],
        harmony_prefix_ids=[0],
        melody_token_ids=[4, 5, 6],
        melody_prefix_ids=[0],
        pair_weight=0.8,
        distance=0.2,
        tempo_bpm=120.0,
        tonic_pc=0,
        mode_name="major",
        source_path="test.mid",
    )

    assert sample.sequence_id == "seq1"
    assert sample.segment_id == "seg1"
    assert len(sample.bio_vector) == 3
    assert len(sample.descriptor_vector) == 6
    assert sample.pair_weight == 0.8
    assert sample.tempo_bpm == 120.0
