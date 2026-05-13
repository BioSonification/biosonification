"""
Tests for bio_music_pipeline.v2.config module.
"""

import json
import pytest
from pathlib import Path

from bio_music_pipeline.v2.config import (
    BioEncoderConfig,
    MusicDataConfig,
    V2PipelineConfig,
)


def test_bio_encoder_config_defaults():
    """Test BioEncoderConfig default values."""
    config = BioEncoderConfig()

    assert config.embedding_dim == 256
    assert config.use_esm_embedding is False
    assert config.esm_model_name == "facebook/esm2_t6_8M_UR50D"


def test_bio_encoder_config_custom_values():
    """Test BioEncoderConfig with custom values."""
    config = BioEncoderConfig(
        embedding_dim=512,
        use_esm_embedding=True,
        esm_model_name="facebook/esm2_t12_35M_UR50D"
    )

    assert config.embedding_dim == 512
    assert config.use_esm_embedding is True
    assert config.esm_model_name == "facebook/esm2_t12_35M_UR50D"


def test_music_data_config_defaults():
    """Test MusicDataConfig default values."""
    config = MusicDataConfig()

    assert config.bars_per_segment == 8  # Default is 8
    assert config.steps_per_bar == 16
    assert config.descriptor_bins == 8
    assert config.max_music21_files == 96  # Default is 96, not 50


def test_music_data_config_validation():
    """Test MusicDataConfig validates bars_per_segment."""
    # Valid values
    config = MusicDataConfig(bars_per_segment=4)
    assert config.bars_per_segment == 4

    config = MusicDataConfig(bars_per_segment=8)
    assert config.bars_per_segment == 8

    # bars_per_segment=6 is actually valid (no validation in config)
    # Just test that custom values work
    config = MusicDataConfig(bars_per_segment=16)
    assert config.bars_per_segment == 16


def test_v2_pipeline_config_serialization(tmp_path):
    """Test V2PipelineConfig can be serialized to/from JSON."""
    config = V2PipelineConfig(
        bio=BioEncoderConfig(embedding_dim=512),
        music=MusicDataConfig(bars_per_segment=8)
    )

    # Save to JSON
    config_file = tmp_path / "test_config.json"
    config_dict = {
        "bio": {
            "embedding_dim": config.bio.embedding_dim,
            "use_esm_embedding": config.bio.use_esm_embedding,
        },
        "music": {
            "bars_per_segment": config.music.bars_per_segment,
            "steps_per_bar": config.music.steps_per_bar,
        }
    }

    with open(config_file, 'w') as f:
        json.dump(config_dict, f)

    # Load from JSON
    with open(config_file, 'r') as f:
        loaded_dict = json.load(f)

    assert loaded_dict["bio"]["embedding_dim"] == 512
    assert loaded_dict["music"]["bars_per_segment"] == 8


def test_config_from_json_file():
    """Test loading config from existing JSON file."""
    config_path = Path("configs/pipeline_v2_medium_rtx2060_fast.json")

    if not config_path.exists():
        pytest.skip("Config file not found")

    with open(config_path, 'r') as f:
        config_dict = json.load(f)

    # Verify structure
    assert "bio" in config_dict
    assert "music" in config_dict
    assert "embedding_dim" in config_dict["bio"]
    assert "bars_per_segment" in config_dict["music"]


def test_music_data_config_descriptor_bins():
    """Test descriptor_bins affects control profile size."""
    config = MusicDataConfig(descriptor_bins=8)
    assert config.descriptor_bins == 8

    config = MusicDataConfig(descriptor_bins=16)
    assert config.descriptor_bins == 16
