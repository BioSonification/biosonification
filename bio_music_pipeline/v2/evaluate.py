"""Evaluation utilities for the structured v2 biosonification pipeline."""

from __future__ import annotations

import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from .config import MusicDataConfig
from .structured_generate import generate_structured_music_from_fasta
from .structured_music import QUALITY_INTERVALS, HarmonyBar, render_harmony_and_melody_to_score

try:
    from music21 import chord, converter, note
except ImportError:  # pragma: no cover - surfaced by metric callers
    chord = None
    converter = None
    note = None


PITCH_CLASS_NAMES = ["C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"]


@dataclass
class StructuredEvaluationConfig:
    checkpoint_path: str
    fasta_path: str = "data/fasta/quick_sample.fa"
    config_path: Optional[str] = None
    output_dir: str = "results/v2_evaluation"
    max_records: int = 4
    device: str = "auto"
    seed: int = 42


def _safe_mean(values: Sequence[float]) -> float:
    return float(np.mean(values)) if values else 0.0


def _safe_std(values: Sequence[float]) -> float:
    return float(np.std(values)) if values else 0.0


def _self_similarity(pitches: Sequence[int], window: int = 8) -> float:
    if len(pitches) < window * 2:
        return 0.0
    pcs = [pitch % 12 for pitch in pitches]
    scores = []
    for start in range(0, len(pcs) - window * 2 + 1):
        left = pcs[start : start + window]
        right = pcs[start + window : start + window * 2]
        scores.append(sum(a == b for a, b in zip(left, right)) / window)
    return _safe_mean(scores)


def _pitch_entropy(pitches: Sequence[int]) -> float:
    if not pitches:
        return 0.0
    counts = np.bincount(np.array([pitch % 12 for pitch in pitches], dtype=np.int64), minlength=12)
    probs = counts[counts > 0] / counts.sum()
    return float(-np.sum(probs * np.log2(probs)))


def _metadata_harmony_bars(metadata: Dict[str, Any]) -> List[HarmonyBar]:
    bars = []
    for index, item in enumerate(metadata.get("generated_harmony_bars", [])):
        bars.append(
            HarmonyBar(
                bar_index=index,
                root_pc=int(item.get("root_pc", 0)),
                quality=str(item.get("quality", "other")),
                hold=bool(item.get("hold", False)),
                key_tonic_pc=int(metadata.get("tonic_pc_hint", 0)),
                key_mode="major",
            )
        )
    return bars


def _chord_tone_ratio(pitches: Sequence[int], offsets: Sequence[float], harmony_bars: Sequence[HarmonyBar]) -> float:
    if not pitches or not harmony_bars:
        return 0.0
    hits = 0
    for pitch, offset in zip(pitches, offsets):
        bar_index = int(max(0, math.floor(float(offset) / 4.0)))
        bar_index = min(bar_index, len(harmony_bars) - 1)
        active = harmony_bars[bar_index]
        intervals = QUALITY_INTERVALS.get(active.quality, QUALITY_INTERVALS["other"])
        chord_pcs = {(active.root_pc + interval) % 12 for interval in intervals}
        if pitch % 12 in chord_pcs:
            hits += 1
    return float(hits / len(pitches))


def _chord_pitch_classes(harmony_chord: Any) -> set[int]:
    return {int(pitch_class) % 12 for pitch_class in harmony_chord.pitchClasses}


def _chord_tone_ratio_from_midi(
    pitches: Sequence[int],
    offsets: Sequence[float],
    harmony_chords: Sequence[Any],
) -> float:
    if not pitches or not harmony_chords:
        return 0.0

    ordered_chords = sorted(harmony_chords, key=lambda item: float(item.offset))
    chord_offsets = [float(item.offset) for item in ordered_chords]
    hits = 0
    for pitch, offset in zip(pitches, offsets):
        active_index = 0
        for index, chord_offset in enumerate(chord_offsets):
            if chord_offset <= float(offset):
                active_index = index
            else:
                break
        if pitch % 12 in _chord_pitch_classes(ordered_chords[active_index]):
            hits += 1
    return float(hits / len(pitches))


def _chord_change_rate(harmony_bars: Sequence[HarmonyBar]) -> float:
    if len(harmony_bars) < 2:
        return 0.0
    changes = 0
    for left, right in zip(harmony_bars, harmony_bars[1:]):
        if right.root_pc != left.root_pc or right.quality != left.quality or not right.hold:
            changes += 1
    return float(changes / (len(harmony_bars) - 1))


def _chord_change_rate_from_midi(harmony_chords: Sequence[Any]) -> float:
    if len(harmony_chords) < 2:
        return 0.0
    ordered_chords = sorted(harmony_chords, key=lambda item: float(item.offset))
    changes = 0
    for left, right in zip(ordered_chords, ordered_chords[1:]):
        if _chord_pitch_classes(left) != _chord_pitch_classes(right):
            changes += 1
    return float(changes / (len(ordered_chords) - 1))


def compute_structured_midi_metrics(midi_path: str, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Compute technical and musical metrics from a structured MIDI file."""

    if converter is None or note is None or chord is None:
        raise ImportError("music21 is required to evaluate structured MIDI files.")

    score = converter.parse(midi_path)
    parts = list(score.parts)
    harmony_part = parts[0] if parts else None
    melody_part = parts[1] if len(parts) > 1 else None

    harmony_events = list(harmony_part.flatten().notes) if harmony_part is not None else []
    melody_events = list(melody_part.flatten().notes) if melody_part is not None else []
    melody_notes = [event for event in melody_events if isinstance(event, note.Note)]
    melody_pitches = [int(event.pitch.midi) for event in melody_notes]
    melody_offsets = [float(event.offset) for event in melody_notes]
    harmony_chords = [event for event in harmony_events if isinstance(event, chord.Chord)]

    metadata = metadata or {}
    harmony_bars = _metadata_harmony_bars(metadata)
    chord_change_rate = (
        _chord_change_rate(harmony_bars) if harmony_bars else _chord_change_rate_from_midi(harmony_chords)
    )
    chord_tone_ratio = (
        _chord_tone_ratio(melody_pitches, melody_offsets, harmony_bars)
        if harmony_bars
        else _chord_tone_ratio_from_midi(melody_pitches, melody_offsets, harmony_chords)
    )
    duration = float(score.highestTime)
    pitch_min = min(melody_pitches) if melody_pitches else 0
    pitch_max = max(melody_pitches) if melody_pitches else 0
    unique_pitches = len(set(melody_pitches))

    return {
        "midi_path": str(midi_path),
        "valid": bool(parts and duration > 0),
        "part_count": len(parts),
        "duration_quarter_length": duration,
        "harmony_event_count": len(harmony_events),
        "harmony_chord_count": len(harmony_chords),
        "melody_event_count": len(melody_events),
        "melody_note_count": len(melody_notes),
        "note_density_per_bar": float(len(melody_notes) / max(duration / 4.0, 1.0)),
        "pitch_min": int(pitch_min),
        "pitch_max": int(pitch_max),
        "pitch_range": int(pitch_max - pitch_min) if melody_pitches else 0,
        "unique_pitches": int(unique_pitches),
        "pitch_class_entropy": _pitch_entropy(melody_pitches),
        "chord_change_rate": chord_change_rate,
        "chord_tone_ratio": chord_tone_ratio,
        "self_similarity": _self_similarity(melody_pitches),
        "expected_two_part_score": len(parts) == 2,
        "nonempty_melody": len(melody_notes) > 0,
        "nonempty_harmony": len(harmony_chords) > 0,
    }


def _aggregate(condition_results: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    if not condition_results:
        return {"count": 0, "invalid_rate": 1.0, "metrics": {}}
    numeric_keys = sorted(
        {
            key
            for item in condition_results
            for key, value in item.get("metrics", {}).items()
            if isinstance(value, (int, float)) and not isinstance(value, bool)
        }
    )
    metrics = {}
    for key in numeric_keys:
        values = [float(item["metrics"][key]) for item in condition_results if key in item.get("metrics", {})]
        metrics[key] = {
            "mean": _safe_mean(values),
            "std": _safe_std(values),
            "min": float(min(values)) if values else 0.0,
            "max": float(max(values)) if values else 0.0,
        }
    valid_count = sum(1 for item in condition_results if item.get("metrics", {}).get("valid"))
    return {
        "count": len(condition_results),
        "valid_count": valid_count,
        "invalid_rate": float(1.0 - valid_count / max(len(condition_results), 1)),
        "metrics": metrics,
    }


def _write_markdown_report(report: Dict[str, Any], output_path: Path) -> None:
    lines = [
        "# Structured V2 Evaluation Report",
        "",
        "## Summary",
        "",
        f"- checkpoint: `{report['config']['checkpoint_path']}`",
        f"- FASTA: `{report['config']['fasta_path']}`",
        f"- device: `{report['config']['device']}`",
        f"- max records: `{report['config']['max_records']}`",
        "",
        "## Aggregate Metrics",
        "",
    ]
    for condition, aggregate in report["aggregates"].items():
        lines.append(f"### {condition}")
        lines.append("")
        lines.append(f"- count: `{aggregate['count']}`")
        lines.append(f"- valid count: `{aggregate.get('valid_count', 0)}`")
        lines.append(f"- invalid rate: `{aggregate['invalid_rate']:.3f}`")
        for name in (
            "melody_note_count",
            "note_density_per_bar",
            "pitch_range",
            "unique_pitches",
            "chord_change_rate",
            "chord_tone_ratio",
            "self_similarity",
        ):
            metric = aggregate.get("metrics", {}).get(name)
            if metric is not None:
                lines.append(f"- {name}: mean `{metric['mean']:.4f}`, std `{metric['std']:.4f}`")
        lines.append("")
    lines.extend(
        [
            "## Interpretation Limits",
            "",
            "These metrics validate MIDI structure and compare simple musical descriptors. "
            "They do not prove a causal relationship between biological sequences and music.",
            "",
        ]
    )
    output_path.write_text("\n".join(lines), encoding="utf-8")


def _random_baseline_sample(
    index: int, output_dir: Path, music_config: MusicDataConfig, rng: random.Random
) -> Dict[str, Any]:
    qualities = ["maj", "min", "dom7", "maj7", "min7", "sus4"]
    harmony_bars = []
    previous_root = rng.randrange(12)
    previous_quality = "maj"
    for bar_index in range(8):
        hold = bar_index > 0 and rng.random() < 0.25
        if hold:
            root_pc = previous_root
            quality = previous_quality
        else:
            root_pc = rng.randrange(12)
            quality = rng.choice(qualities)
        previous_root = root_pc
        previous_quality = quality
        harmony_bars.append(HarmonyBar(bar_index, root_pc, quality, hold, 0, "major"))

    events = []
    total_steps = len(harmony_bars) * music_config.steps_per_bar
    onset = 0
    while onset < total_steps:
        if rng.random() < 0.7:
            bar_index = onset // music_config.steps_per_bar
            active_root = harmony_bars[bar_index].root_pc
            relative_pc = rng.choice([0, 2, 3, 4, 5, 7, 9, 10])
            pitch = 12 * rng.choice([4, 5, 6]) + ((active_root + relative_pc) % 12)
            duration = rng.choice([1, 2, 3, 4])
            events.append((onset, min(duration, total_steps - onset), pitch, bar_index))
        onset += rng.choice([1, 2, 4])

    midi_path = output_dir / f"random_baseline_{index:03d}.mid"
    metadata = {
        "tonic_pc_hint": 0,
        "generated_harmony_bars": [
            {"root_pc": bar.root_pc, "quality": bar.quality, "hold": bar.hold} for bar in harmony_bars
        ],
    }
    score = render_harmony_and_melody_to_score(harmony_bars, events, tempo_bpm=96.0, config=music_config)
    score.write("midi", fp=str(midi_path))
    return {
        "condition": "random_baseline",
        "record_index": index,
        "midi_path": str(midi_path),
        "metadata": metadata,
        "metrics": compute_structured_midi_metrics(str(midi_path), metadata),
    }


def run_structured_evaluation(config: StructuredEvaluationConfig) -> Dict[str, Any]:
    output_dir = Path(config.output_dir)
    midi_dir = output_dir / "midi"
    metadata_dir = output_dir / "metadata"
    output_dir.mkdir(parents=True, exist_ok=True)
    midi_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)

    current_results = []
    for record_index in range(max(1, config.max_records)):
        midi_path = midi_dir / f"structured_record_{record_index:03d}.mid"
        metadata_path = metadata_dir / f"structured_record_{record_index:03d}.json"
        try:
            metadata = generate_structured_music_from_fasta(
                fasta_path=config.fasta_path,
                checkpoint_path=config.checkpoint_path,
                output_path=str(midi_path),
                config_path=config.config_path,
                record_index=record_index,
                metadata_output=str(metadata_path),
                device_name=config.device,
            )
            metrics = compute_structured_midi_metrics(str(midi_path), metadata)
            current_results.append(
                {
                    "condition": "structured_v2",
                    "record_index": record_index,
                    "midi_path": str(midi_path),
                    "metadata_path": str(metadata_path),
                    "metadata": metadata,
                    "metrics": metrics,
                }
            )
        except IndexError:
            break

    rng = random.Random(config.seed)
    music_config = MusicDataConfig()
    baseline_results = [
        _random_baseline_sample(index, midi_dir, music_config, rng) for index in range(max(1, len(current_results)))
    ]

    report = {
        "config": asdict(config),
        "conditions": {
            "structured_v2": current_results,
            "random_baseline": baseline_results,
        },
        "aggregates": {
            "structured_v2": _aggregate(current_results),
            "random_baseline": _aggregate(baseline_results),
        },
        "notes": [
            "Evaluation uses structural MIDI metrics and a simple random harmony+melody baseline.",
            "This report does not establish biological causality.",
        ],
    }

    report_json = output_dir / "evaluation_report.json"
    report_md = output_dir / "evaluation_report.md"
    report_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    _write_markdown_report(report, report_md)
    return report
