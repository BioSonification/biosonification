"""Progress logging utility for long-running operations."""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional

from tqdm import tqdm


class ProgressLogger:
    """Logger with progress bar support and file output."""

    def __init__(
        self,
        name: str,
        log_file: Optional[Path] = None,
        level: int = logging.INFO,
        use_tqdm: bool = True,
    ):
        self.name = name
        self.use_tqdm = use_tqdm

        # Setup logger
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)
        self.logger.handlers.clear()

        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
        )
        console_handler.setFormatter(console_formatter)
        self.logger.addHandler(console_handler)

        # File handler
        if log_file:
            log_file.parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(log_file, encoding="utf-8")
            file_handler.setLevel(level)
            file_handler.setFormatter(console_formatter)
            self.logger.addHandler(file_handler)
            self.logger.info(f"Logging to {log_file}")

    def info(self, msg: str) -> None:
        """Log info message."""
        self.logger.info(msg)

    def warning(self, msg: str) -> None:
        """Log warning message."""
        self.logger.warning(msg)

    def error(self, msg: str) -> None:
        """Log error message."""
        self.logger.error(msg)

    def debug(self, msg: str) -> None:
        """Log debug message."""
        self.logger.debug(msg)

    def progress_bar(
        self,
        iterable=None,
        total: Optional[int] = None,
        desc: Optional[str] = None,
        unit: str = "it",
        disable: bool = False,
    ):
        """Create a progress bar."""
        if not self.use_tqdm or disable:
            # Return plain iterable if tqdm disabled
            if iterable is not None:
                return iterable
            return range(total) if total else []

        return tqdm(
            iterable=iterable,
            total=total,
            desc=desc,
            unit=unit,
            file=sys.stdout,
            ncols=100,
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
        )
