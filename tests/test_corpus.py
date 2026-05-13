"""Tests for bio_music_pipeline/v2/corpus.py."""

import tempfile
from pathlib import Path

import pytest

from bio_music_pipeline.v2.corpus import bootstrap_music21_corpus, iter_score_files


def test_iter_score_files_empty_dirs():
    """Test iter_score_files with empty directory list."""
    files = iter_score_files([])
    assert files == []


def test_iter_score_files_nonexistent_dir():
    """Test iter_score_files with nonexistent directory."""
    files = iter_score_files(["/nonexistent/path/to/nowhere"])
    assert files == []


def test_iter_score_files_with_midi_files():
    """Test iter_score_files finds MIDI files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create test MIDI files
        (Path(tmpdir) / "test1.mid").touch()
        (Path(tmpdir) / "test2.midi").touch()
        (Path(tmpdir) / "test3.txt").touch()  # Should be ignored

        files = iter_score_files([tmpdir])

        assert len(files) == 2
        assert all(f.suffix in [".mid", ".midi"] for f in files)
        assert all(f.exists() for f in files)


def test_iter_score_files_with_xml_files():
    """Test iter_score_files finds MusicXML files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        (Path(tmpdir) / "test1.xml").touch()
        (Path(tmpdir) / "test2.mxl").touch()
        (Path(tmpdir) / "test3.musicxml").touch()

        files = iter_score_files([tmpdir])

        assert len(files) == 3
        assert all(f.suffix in [".xml", ".mxl", ".musicxml"] for f in files)


def test_iter_score_files_recursive():
    """Test iter_score_files finds files recursively."""
    with tempfile.TemporaryDirectory() as tmpdir:
        base = Path(tmpdir)
        subdir = base / "subdir"
        subdir.mkdir()

        (base / "test1.mid").touch()
        (subdir / "test2.mid").touch()

        files = iter_score_files([tmpdir])

        assert len(files) == 2
        assert any("subdir" in str(f) for f in files)


def test_iter_score_files_multiple_dirs():
    """Test iter_score_files with multiple directories."""
    with tempfile.TemporaryDirectory() as tmpdir1:
        with tempfile.TemporaryDirectory() as tmpdir2:
            (Path(tmpdir1) / "test1.mid").touch()
            (Path(tmpdir2) / "test2.mid").touch()

            files = iter_score_files([tmpdir1, tmpdir2])

            assert len(files) == 2


def test_iter_score_files_deduplicates():
    """Test iter_score_files removes duplicates."""
    with tempfile.TemporaryDirectory() as tmpdir:
        (Path(tmpdir) / "test.mid").touch()

        # Pass same directory twice
        files = iter_score_files([tmpdir, tmpdir])

        assert len(files) == 1


def test_iter_score_files_sorted():
    """Test iter_score_files returns sorted results."""
    with tempfile.TemporaryDirectory() as tmpdir:
        (Path(tmpdir) / "c.mid").touch()
        (Path(tmpdir) / "a.mid").touch()
        (Path(tmpdir) / "b.mid").touch()

        files = iter_score_files([tmpdir])

        names = [f.name for f in files]
        assert names == sorted(names)


@pytest.mark.slow
def test_bootstrap_music21_corpus_creates_files():
    """Test bootstrap_music21_corpus creates MIDI files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        files = bootstrap_music21_corpus(tmpdir, max_files=5)

        assert len(files) == 5
        assert all(f.exists() for f in files)
        assert all(f.suffix == ".mid" for f in files)
        assert all("bach_chorale" in f.name for f in files)


@pytest.mark.slow
def test_bootstrap_music21_corpus_reuses_existing():
    """Test bootstrap_music21_corpus reuses existing files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # First call creates files
        files1 = bootstrap_music21_corpus(tmpdir, max_files=3)
        assert len(files1) == 3

        # Second call should reuse existing files
        files2 = bootstrap_music21_corpus(tmpdir, max_files=3)
        assert len(files2) == 3
        assert files1 == files2


@pytest.mark.slow
def test_bootstrap_music21_corpus_default_composer():
    """Test bootstrap_music21_corpus with default composer (Bach)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        files = bootstrap_music21_corpus(tmpdir, max_files=2, composers=None)

        assert len(files) == 2
        assert all("bach" in f.name.lower() for f in files)


@pytest.mark.slow
def test_bootstrap_music21_corpus_bach_explicit():
    """Test bootstrap_music21_corpus with explicit Bach composer."""
    with tempfile.TemporaryDirectory() as tmpdir:
        files = bootstrap_music21_corpus(tmpdir, max_files=2, composers=["bach"])

        assert len(files) == 2


@pytest.mark.slow
def test_bootstrap_music21_corpus_unsupported_composer():
    """Test bootstrap_music21_corpus rejects unsupported composers."""
    with tempfile.TemporaryDirectory() as tmpdir:
        with pytest.raises(ValueError, match="Unsupported composer values"):
            bootstrap_music21_corpus(tmpdir, max_files=2, composers=["mozart"])


@pytest.mark.slow
def test_bootstrap_music21_corpus_mixed_composers():
    """Test bootstrap_music21_corpus rejects mixed composers including unsupported."""
    with tempfile.TemporaryDirectory() as tmpdir:
        with pytest.raises(ValueError, match="Unsupported composer values"):
            bootstrap_music21_corpus(tmpdir, max_files=2, composers=["bach", "beethoven"])


@pytest.mark.slow
def test_bootstrap_music21_corpus_case_insensitive():
    """Test bootstrap_music21_corpus handles case-insensitive composer names."""
    with tempfile.TemporaryDirectory() as tmpdir:
        files = bootstrap_music21_corpus(tmpdir, max_files=2, composers=["BACH"])

        assert len(files) == 2


@pytest.mark.slow
def test_bootstrap_music21_corpus_creates_directory():
    """Test bootstrap_music21_corpus creates output directory if missing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir) / "subdir" / "corpus"
        assert not output_dir.exists()

        files = bootstrap_music21_corpus(str(output_dir), max_files=2)

        assert output_dir.exists()
        assert len(files) == 2
