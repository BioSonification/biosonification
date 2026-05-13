"""
Pytest configuration and shared fixtures for BioSonification tests.
"""

import tempfile
from pathlib import Path

import pytest


@pytest.fixture
def temp_dir():
    """Temporary directory for test outputs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_fasta():
    """Sample FASTA sequence for testing (1800 bp)."""
    return ">test_sequence\n" + "ACGTACGTACGT" * 150


@pytest.fixture
def sample_fasta_short():
    """Short FASTA sequence for quick tests (120 bp)."""
    return ">test_short\n" + "ACGTACGTACGT" * 10


@pytest.fixture
def sample_fasta_protein():
    """Sample protein FASTA sequence."""
    return (
        ">test_protein\n"
        "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVKALPDAQFEVVHSLAKWKRQTLGQHDF"
        "SAGEGLYTHMKALRPDEDRLSPLHSVYVDQWDWERVMGDGERQFSTLKSTVEAIWAGIKATEAAVSEEFGLAPFLPDQIHFVHSQELS"
        "SRYPDLDAKGRERAIAKDLGAVFLVGIGGKLSDGHRHDVRAPDYDDWSTPSELGHAGLNGDILVWNPVLEDAFELSSMGIRVDADTLK"
        "HQLALTGDEDRLELEWHQALLRGEMPQTIGGGIGQSRLTMLLLQLPHIGQVQAGVWPAAVRESVPSLL"
    )


@pytest.fixture
def flask_client():
    """Flask test client."""
    from web.app import app

    app.config["TESTING"] = True
    app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024
    with app.test_client() as client:
        yield client


@pytest.fixture
def mock_generator():
    """Mock BioMusicGenerator for testing."""

    class MockGenerator:
        def __init__(self):
            self._initialized = True
            self._error = None

        def is_ready(self):
            return True

        def initialize(self):
            return True

        def get_error(self):
            return None

        def status_payload(self):
            return {
                "config_path": "configs/pipeline_v2_medium_rtx2060_fast.json",
                "checkpoint_path": "results/v2_medium_rtx2060_fast/checkpoints/structured_pipeline.pt",
                "device": "cpu",
            }

        def generate(self, fasta_text, output_dir):
            import uuid

            session_id = str(uuid.uuid4())[:8]
            return {
                "session_id": session_id,
                "midi_path": f"{output_dir}/midi/{session_id}.mid",
                "midi_filename": f"{session_id}.mid",
                "header": "test_sequence",
                "sequence_length": 1800,
                "musical_params": {
                    "tempo": 120.0,
                    "key": "C major",
                    "sequence_type": "dna",
                    "harmony_bars": 4,
                    "melody_notes": 16,
                    "device": "cpu",
                },
                "structured_metadata": {
                    "sequence_type": "dna",
                    "generated_melody_note_count": 16,
                },
            }

    return MockGenerator()


@pytest.fixture
def sample_midi_path(tmp_path):
    """Create a minimal MIDI file for testing."""
    import mido

    midi_file = tmp_path / "test.mid"
    mid = mido.MidiFile()
    track = mido.MidiTrack()
    mid.tracks.append(track)

    # Add some notes
    track.append(mido.Message("note_on", note=60, velocity=64, time=0))
    track.append(mido.Message("note_off", note=60, velocity=64, time=480))
    track.append(mido.Message("note_on", note=64, velocity=64, time=0))
    track.append(mido.Message("note_off", note=64, velocity=64, time=480))

    mid.save(str(midi_file))
    return str(midi_file)
