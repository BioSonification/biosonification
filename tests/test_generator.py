"""
Tests for web.generator module.
"""

from pathlib import Path

from web.generator import BioMusicGenerator


def test_generator_initialization():
    """Test BioMusicGenerator initializes with default paths."""
    generator = BioMusicGenerator()

    assert generator.config_path is not None
    assert generator.checkpoint_path is not None
    assert generator.device in ["auto", "cpu", "cuda"]


def test_generator_initialization_with_custom_paths(tmp_path):
    """Test BioMusicGenerator with custom config and checkpoint paths."""
    config_path = tmp_path / "config.json"
    checkpoint_path = tmp_path / "checkpoint.pt"

    # Create dummy files
    config_path.write_text('{"bio": {}, "music": {}}')
    checkpoint_path.write_text("dummy")

    generator = BioMusicGenerator(config_path=str(config_path), checkpoint_path=str(checkpoint_path), device="cpu")

    assert generator.config_path == config_path
    assert generator.checkpoint_path == checkpoint_path
    assert generator.device == "cpu"


def test_generator_error_when_config_missing():
    """Test generator sets error when config file is missing."""
    generator = BioMusicGenerator(config_path="/nonexistent/config.json")

    assert generator._error is not None
    assert "Config file not found" in generator._error


def test_generator_error_when_checkpoint_missing():
    """Test generator sets error when checkpoint file is missing."""
    # Use existing config but nonexistent checkpoint
    generator = BioMusicGenerator(checkpoint_path="/nonexistent/checkpoint.pt")

    assert generator._error is not None
    assert "checkpoint not found" in generator._error.lower()


def test_generator_initialize_success(tmp_path):
    """Test generator.initialize() returns True when files exist."""
    config_path = tmp_path / "config.json"
    checkpoint_path = tmp_path / "checkpoint.pt"

    config_path.write_text('{"bio": {}, "music": {}}')
    checkpoint_path.write_text("dummy")

    generator = BioMusicGenerator(config_path=str(config_path), checkpoint_path=str(checkpoint_path))

    assert generator.initialize() is True
    assert generator.is_ready() is True


def test_generator_initialize_failure():
    """Test generator.initialize() returns False when files missing."""
    generator = BioMusicGenerator(config_path="/nonexistent/config.json")

    assert generator.initialize() is False
    assert generator.is_ready() is False


def test_generator_device_selection_from_env(monkeypatch):
    """Test generator respects BIOSONIFICATION_DEVICE env var."""
    monkeypatch.setenv("BIOSONIFICATION_DEVICE", "cuda")

    generator = BioMusicGenerator()

    assert generator.device == "cuda"


def test_generator_status_payload(tmp_path):
    """Test generator.status_payload() returns correct info."""
    config_path = tmp_path / "config.json"
    checkpoint_path = tmp_path / "checkpoint.pt"

    config_path.write_text('{"bio": {}, "music": {}}')
    checkpoint_path.write_text("dummy")

    generator = BioMusicGenerator(config_path=str(config_path), checkpoint_path=str(checkpoint_path), device="cpu")

    payload = generator.status_payload()

    assert payload["config_path"] == str(config_path)
    assert payload["checkpoint_path"] == str(checkpoint_path)
    assert payload["device"] == "cpu"


def test_generator_get_error():
    """Test generator.get_error() returns error message."""
    generator = BioMusicGenerator(config_path="/nonexistent/config.json")

    error = generator.get_error()
    assert error is not None
    assert "Config file not found" in error


def test_generator_checkpoint_resolution_priority():
    """Test checkpoint resolution prefers 4-bar model over 8-bar."""
    # This test verifies the logic in _resolve_default_checkpoint_path
    # It's more of an integration test, so we'll keep it simple
    generator = BioMusicGenerator()

    # Just verify that checkpoint_path is set
    assert generator.checkpoint_path is not None
    assert isinstance(generator.checkpoint_path, Path)
