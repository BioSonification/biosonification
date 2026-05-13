"""Tests for bio_music_pipeline/utils/progress_logger.py."""

import logging
import tempfile
from pathlib import Path

from bio_music_pipeline.utils.progress_logger import ProgressLogger


def test_progress_logger_initialization():
    """Test ProgressLogger initialization."""
    logger = ProgressLogger("test_logger")

    assert logger.name == "test_logger"
    assert logger.use_tqdm is True
    assert logger.logger.level == logging.INFO


def test_progress_logger_with_log_file():
    """Test ProgressLogger with log file."""
    with tempfile.TemporaryDirectory() as tmpdir:
        log_file = Path(tmpdir) / "test.log"
        logger = ProgressLogger("test_logger", log_file=log_file)

        logger.info("Test message")

        # Close handlers before reading file
        for handler in logger.logger.handlers[:]:
            handler.close()
            logger.logger.removeHandler(handler)

        assert log_file.exists()
        content = log_file.read_text(encoding="utf-8")
        assert "Test message" in content


def test_progress_logger_creates_log_directory():
    """Test ProgressLogger creates log directory if missing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        log_file = Path(tmpdir) / "subdir" / "test.log"
        assert not log_file.parent.exists()

        logger = ProgressLogger("test_logger", log_file=log_file)
        logger.info("Test message")

        # Close handlers before cleanup
        for handler in logger.logger.handlers[:]:
            handler.close()
            logger.logger.removeHandler(handler)

        assert log_file.parent.exists()
        assert log_file.exists()


def test_progress_logger_custom_level():
    """Test ProgressLogger with custom log level."""
    logger = ProgressLogger("test_logger", level=logging.DEBUG)

    assert logger.logger.level == logging.DEBUG


def test_progress_logger_info():
    """Test ProgressLogger info method."""
    with tempfile.TemporaryDirectory() as tmpdir:
        log_file = Path(tmpdir) / "test.log"
        logger = ProgressLogger("test_logger", log_file=log_file)

        logger.info("Info message")

        # Close handlers before reading
        for handler in logger.logger.handlers[:]:
            handler.close()
            logger.logger.removeHandler(handler)

        content = log_file.read_text(encoding="utf-8")
        assert "INFO" in content
        assert "Info message" in content


def test_progress_logger_warning():
    """Test ProgressLogger warning method."""
    with tempfile.TemporaryDirectory() as tmpdir:
        log_file = Path(tmpdir) / "test.log"
        logger = ProgressLogger("test_logger", log_file=log_file)

        logger.warning("Warning message")

        # Close handlers before reading
        for handler in logger.logger.handlers[:]:
            handler.close()
            logger.logger.removeHandler(handler)

        content = log_file.read_text(encoding="utf-8")
        assert "WARNING" in content
        assert "Warning message" in content


def test_progress_logger_error():
    """Test ProgressLogger error method."""
    with tempfile.TemporaryDirectory() as tmpdir:
        log_file = Path(tmpdir) / "test.log"
        logger = ProgressLogger("test_logger", log_file=log_file)

        logger.error("Error message")

        # Close handlers before reading
        for handler in logger.logger.handlers[:]:
            handler.close()
            logger.logger.removeHandler(handler)

        content = log_file.read_text(encoding="utf-8")
        assert "ERROR" in content
        assert "Error message" in content


def test_progress_logger_debug():
    """Test ProgressLogger debug method."""
    with tempfile.TemporaryDirectory() as tmpdir:
        log_file = Path(tmpdir) / "test.log"
        logger = ProgressLogger("test_logger", log_file=log_file, level=logging.DEBUG)

        logger.debug("Debug message")

        # Close handlers before reading
        for handler in logger.logger.handlers[:]:
            handler.close()
            logger.logger.removeHandler(handler)

        content = log_file.read_text(encoding="utf-8")
        assert "DEBUG" in content
        assert "Debug message" in content


def test_progress_logger_debug_not_logged_at_info_level():
    """Test ProgressLogger debug messages not logged at INFO level."""
    with tempfile.TemporaryDirectory() as tmpdir:
        log_file = Path(tmpdir) / "test.log"
        logger = ProgressLogger("test_logger", log_file=log_file, level=logging.INFO)

        logger.debug("Debug message")

        # Close handlers before reading
        for handler in logger.logger.handlers[:]:
            handler.close()
            logger.logger.removeHandler(handler)

        content = log_file.read_text(encoding="utf-8")
        assert "Debug message" not in content


def test_progress_bar_with_iterable():
    """Test progress_bar with iterable."""
    logger = ProgressLogger("test_logger", use_tqdm=True)

    items = [1, 2, 3, 4, 5]
    result = list(logger.progress_bar(items, desc="Processing"))

    assert result == items


def test_progress_bar_with_total():
    """Test progress_bar with total count."""
    logger = ProgressLogger("test_logger", use_tqdm=True)

    # When only total is provided with tqdm enabled, need to pass range as iterable
    result = list(logger.progress_bar(iterable=range(5), desc="Processing"))

    assert len(result) == 5
    assert result == [0, 1, 2, 3, 4]


def test_progress_bar_disabled_with_iterable():
    """Test progress_bar returns plain iterable when disabled."""
    logger = ProgressLogger("test_logger", use_tqdm=False)

    items = [1, 2, 3, 4, 5]
    result = logger.progress_bar(items, desc="Processing")

    assert result is items


def test_progress_bar_disabled_with_total():
    """Test progress_bar returns range when disabled with total."""
    logger = ProgressLogger("test_logger", use_tqdm=False)

    result = list(logger.progress_bar(total=5, desc="Processing"))

    assert result == [0, 1, 2, 3, 4]


def test_progress_bar_disable_parameter():
    """Test progress_bar with disable parameter."""
    logger = ProgressLogger("test_logger", use_tqdm=True)

    items = [1, 2, 3, 4, 5]
    result = logger.progress_bar(items, desc="Processing", disable=True)

    assert result is items


def test_progress_bar_custom_unit():
    """Test progress_bar with custom unit."""
    logger = ProgressLogger("test_logger", use_tqdm=True)

    items = [1, 2, 3]
    result = list(logger.progress_bar(items, desc="Files", unit="file"))

    assert result == items


def test_progress_logger_no_tqdm():
    """Test ProgressLogger with tqdm disabled."""
    logger = ProgressLogger("test_logger", use_tqdm=False)

    assert logger.use_tqdm is False

    items = [1, 2, 3]
    result = logger.progress_bar(items)
    assert result is items


def test_progress_logger_multiple_handlers():
    """Test ProgressLogger doesn't accumulate handlers."""
    with tempfile.TemporaryDirectory() as tmpdir:
        log_file = Path(tmpdir) / "test.log"

        # Create logger twice with same name
        logger1 = ProgressLogger("test_logger_multi", log_file=log_file)
        initial_handlers = len(logger1.logger.handlers)

        # Close first logger's handlers
        for handler in logger1.logger.handlers[:]:
            handler.close()
            logger1.logger.removeHandler(handler)

        logger2 = ProgressLogger("test_logger_multi", log_file=log_file)
        final_handlers = len(logger2.logger.handlers)

        # Close second logger's handlers
        for handler in logger2.logger.handlers[:]:
            handler.close()
            logger2.logger.removeHandler(handler)

        # Handlers should be cleared and recreated, not accumulated
        assert final_handlers == initial_handlers


def test_progress_bar_empty_total():
    """Test progress_bar with no total and no iterable."""
    logger = ProgressLogger("test_logger", use_tqdm=False)

    result = logger.progress_bar(total=None)

    assert result == []
