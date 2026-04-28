"""Music corpus helpers for the structured v2 pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import List, Sequence

try:
    from music21 import corpus
except ImportError:  # pragma: no cover - surfaced by callers
    corpus = None


def bootstrap_music21_corpus(
    output_dir: str,
    max_files: int = 96,
    composers: Sequence[str] | None = None,
) -> List[Path]:
    """Export a compact Bach chorale fallback corpus to local MIDI files."""

    if corpus is None:
        raise ImportError("music21 is required to bootstrap the fallback corpus.")

    composer_names = [name.lower() for name in (composers or ["bach"])]
    unsupported = [name for name in composer_names if name != "bach"]
    if unsupported:
        raise ValueError(
            "The built-in music21 fallback currently supports only Bach chorales. "
            f"Unsupported composer values: {unsupported}. Provide MIDI files via music.midi_dirs for other corpora."
        )

    target_dir = Path(output_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    existing = sorted(target_dir.glob("*.mid"))
    if len(existing) >= max_files:
        return existing[:max_files]

    created: List[Path] = []
    iterator = corpus.chorales.Iterator()
    for index, score in enumerate(iterator):
        if index >= max_files:
            break
        midi_path = target_dir / f"bach_chorale_{index:04d}.mid"
        if not midi_path.exists():
            score.write("midi", fp=str(midi_path))
        created.append(midi_path)
    return created


def iter_score_files(midi_dirs: Sequence[str]) -> List[Path]:
    """Return supported score files in deterministic order."""

    files: List[Path] = []
    extensions = ("*.mid", "*.midi", "*.xml", "*.mxl", "*.musicxml")
    for midi_dir in midi_dirs:
        base = Path(midi_dir)
        if not base.exists():
            continue
        for pattern in extensions:
            files.extend(base.rglob(pattern))
    return sorted({path.resolve() for path in files})
