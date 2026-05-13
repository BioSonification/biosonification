"""Tests for web/examples_data.py."""

from web.examples_data import EXAMPLES


def test_examples_is_list():
    """EXAMPLES should be a list."""
    assert isinstance(EXAMPLES, list)
    assert len(EXAMPLES) > 0


def test_examples_structure():
    """Each example should have required fields."""
    required_fields = {
        "id",
        "organism",
        "scientific_name",
        "description",
        "midi_filename",
        "bars",
        "duration_seconds",
        "genome_size",
        "icon",
    }

    for example in EXAMPLES:
        assert isinstance(example, dict)
        assert set(example.keys()) == required_fields


def test_examples_ids_unique():
    """Example IDs should be unique."""
    ids = [ex["id"] for ex in EXAMPLES]
    assert len(ids) == len(set(ids))


def test_examples_midi_filenames_unique():
    """MIDI filenames should be unique."""
    filenames = [ex["midi_filename"] for ex in EXAMPLES]
    assert len(filenames) == len(set(filenames))


def test_examples_values_types():
    """Example values should have correct types."""
    for example in EXAMPLES:
        assert isinstance(example["id"], str)
        assert isinstance(example["organism"], str)
        assert isinstance(example["scientific_name"], str)
        assert isinstance(example["description"], str)
        assert isinstance(example["midi_filename"], str)
        assert isinstance(example["bars"], int)
        assert isinstance(example["duration_seconds"], int)
        assert isinstance(example["genome_size"], str)
        assert isinstance(example["icon"], str)


def test_examples_positive_values():
    """Numeric values should be positive."""
    for example in EXAMPLES:
        assert example["bars"] > 0
        assert example["duration_seconds"] > 0


def test_examples_midi_extension():
    """MIDI filenames should have .mid extension."""
    for example in EXAMPLES:
        assert example["midi_filename"].endswith(".mid")


def test_examples_known_organisms():
    """Test that we have expected organisms."""
    ids = {ex["id"] for ex in EXAMPLES}
    assert "ecoli" in ids
    assert "yeast" in ids
    assert "drosophila" in ids
