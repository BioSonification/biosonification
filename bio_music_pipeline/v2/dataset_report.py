"""Dataset manifest and sanity reporting for structured v2."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from .bio import BiologicalSequenceEncoder
from .config import V2PipelineConfig, load_v2_config
from .dataset import _iter_score_files
from .structured_music import load_structured_music_corpus

try:
    from Bio import SeqIO
except ImportError:  # pragma: no cover
    SeqIO = None


@dataclass
class DatasetReportConfig:
    config_path: str = "configs/pipeline_v2_small.json"
    output_dir: str = "results/v2_dataset_report"
    max_preview_records: int = 12
    load_segments: bool = True


def _summary(values: Sequence[float]) -> Dict[str, float]:
    if not values:
        return {"count": 0, "mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    arr = np.asarray(values, dtype=np.float64)
    return {
        "count": int(arr.size),
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }


def _fasta_records(path: str, max_preview: int) -> Dict[str, Any]:
    fasta_path = Path(path)
    if SeqIO is None:
        raise ImportError("Biopython is required to inspect FASTA files.")
    if not fasta_path.exists():
        return {
            "path": str(fasta_path),
            "exists": False,
            "record_count": 0,
            "length_summary": _summary([]),
            "preview": [],
        }

    lengths = []
    preview = []
    for index, record in enumerate(SeqIO.parse(str(fasta_path), "fasta")):
        sequence = str(record.seq)
        lengths.append(len(sequence))
        if len(preview) < max_preview:
            preview.append(
                {
                    "index": index,
                    "id": record.id,
                    "length": len(sequence),
                }
            )
    return {
        "path": str(fasta_path),
        "exists": True,
        "record_count": len(lengths),
        "length_summary": _summary(lengths),
        "preview": preview,
    }


def _bio_fragment_summary(config: V2PipelineConfig) -> Dict[str, Any]:
    encoder = BiologicalSequenceEncoder(config.bio)
    try:
        encodings = encoder.encode_fasta(config.fasta_path)
    except Exception as exc:
        return {
            "status": "failed",
            "error": str(exc),
            "fragment_count": 0,
            "sequence_types": {},
            "cleaned_length_summary": _summary([]),
        }
    type_counts: Dict[str, int] = {}
    for item in encodings:
        type_counts[item.sequence_type] = type_counts.get(item.sequence_type, 0) + 1
    return {
        "status": "ok",
        "fragment_count": len(encodings),
        "sequence_types": type_counts,
        "cleaned_length_summary": _summary([len(item.cleaned_sequence) for item in encodings]),
        "control_profile_mean": [float(value) for value in np.mean([item.control_profile for item in encodings], axis=0)],
    }


def _music_file_manifest(config: V2PipelineConfig) -> Dict[str, Any]:
    score_files = _iter_score_files(config.music.midi_dirs)
    suffix_counts: Dict[str, int] = {}
    for path in score_files:
        suffix = path.suffix.lower()
        suffix_counts[suffix] = suffix_counts.get(suffix, 0) + 1
    fallback_expected = not score_files and config.music.use_music21_corpus_fallback
    source_kind = "external_files"
    if fallback_expected:
        source_kind = "music21_fallback"
    elif score_files and any("polyphonic_music21" in str(path) for path in score_files):
        source_kind = "music21_fallback_files"
    return {
        "midi_dirs": list(config.music.midi_dirs),
        "source_kind": source_kind,
        "file_count": len(score_files),
        "suffix_counts": suffix_counts,
        "sample_files": [str(path) for path in score_files[:10]],
        "use_music21_corpus_fallback": config.music.use_music21_corpus_fallback,
        "music21_composers": list(config.music.music21_composers),
        "fallback_note": (
            "Built-in fallback is intended for demos/smoke tests. Use an external licensed polyphonic MIDI corpus for serious experiments."
            if source_kind.startswith("music21")
            else ""
        ),
    }


def _structured_segment_summary(config: V2PipelineConfig) -> Dict[str, Any]:
    try:
        segments, _, _ = load_structured_music_corpus(config.music)
    except Exception as exc:
        return {
            "status": "failed",
            "error": str(exc),
            "segment_count": 0,
        }
    descriptor_matrix = np.stack([segment.descriptor_vector for segment in segments])
    key_modes: Dict[str, int] = {}
    for segment in segments:
        key_modes[segment.key_mode] = key_modes.get(segment.key_mode, 0) + 1
    return {
        "status": "ok",
        "segment_count": len(segments),
        "source_file_count": len({segment.source_path for segment in segments}),
        "tempo_summary": _summary([segment.tempo_bpm for segment in segments]),
        "melody_note_count_summary": _summary([len(segment.melody_events) for segment in segments]),
        "harmony_bar_count_summary": _summary([len(segment.harmony_bars) for segment in segments]),
        "key_modes": key_modes,
        "descriptor_means": [float(value) for value in descriptor_matrix.mean(axis=0)],
        "descriptor_stds": [float(value) for value in descriptor_matrix.std(axis=0)],
        "filtering": {
            "bars_per_segment": config.music.bars_per_segment,
            "segment_hop_bars": config.music.segment_hop_bars,
            "min_notes_per_segment": config.music.min_notes_per_segment,
            "prefer_named_melody_parts": config.music.prefer_named_melody_parts,
        },
    }


def _write_markdown(report: Dict[str, Any], output_path: Path) -> None:
    fasta = report["fasta"]
    music = report["music_files"]
    segments = report["structured_segments"]
    lines = [
        "# Structured V2 Dataset Report",
        "",
        "## FASTA",
        "",
        f"- path: `{fasta['path']}`",
        f"- records: `{fasta['record_count']}`",
        f"- length mean: `{fasta['length_summary']['mean']:.2f}`",
        f"- encoded fragments: `{report['bio_fragments']['fragment_count']}`",
        "",
        "## MIDI Corpus",
        "",
        f"- source kind: `{music['source_kind']}`",
        f"- configured dirs: `{', '.join(music['midi_dirs'])}`",
        f"- file count before fallback bootstrap: `{music['file_count']}`",
        f"- music21 fallback enabled: `{music['use_music21_corpus_fallback']}`",
        f"- music21 composers: `{', '.join(music['music21_composers'])}`",
    ]
    if music.get("fallback_note"):
        lines.append(f"- note: {music['fallback_note']}")
    lines.extend(
        [
            "",
            "## Structured Segments",
            "",
            f"- status: `{segments.get('status')}`",
            f"- segment count: `{segments.get('segment_count', 0)}`",
        ]
    )
    if segments.get("status") == "ok":
        lines.extend(
            [
                f"- source file count: `{segments['source_file_count']}`",
                f"- tempo mean: `{segments['tempo_summary']['mean']:.2f}`",
                f"- melody notes per segment mean: `{segments['melody_note_count_summary']['mean']:.2f}`",
                f"- key modes: `{json.dumps(segments['key_modes'], ensure_ascii=False)}`",
            ]
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "The built-in `music21` fallback is suitable for demos and smoke tests, not as a broad experimental corpus. Serious runs should declare an external licensed MIDI corpus in `music.midi_dirs` and keep this report with the resulting metrics.",
            "",
        ]
    )
    output_path.write_text("\n".join(lines), encoding="utf-8")


def build_dataset_report(report_config: DatasetReportConfig) -> Dict[str, Any]:
    config = load_v2_config(report_config.config_path)
    output_dir = Path(report_config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    report = {
        "config_path": report_config.config_path,
        "pipeline_config": asdict(config),
        "fasta": _fasta_records(config.fasta_path, report_config.max_preview_records),
        "bio_fragments": _bio_fragment_summary(config),
        "music_files": _music_file_manifest(config),
        "structured_segments": (
            _structured_segment_summary(config)
            if report_config.load_segments
            else {"status": "skipped", "segment_count": 0}
        ),
        "recommendations": [
            "Treat music21 fallback as demo-only unless the experiment explicitly studies that corpus.",
            "Use external licensed polyphonic MIDI data for claims about general musical quality.",
            "Keep this manifest with every training/evaluation output directory.",
        ],
    }

    json_path = output_dir / "dataset_report.json"
    md_path = output_dir / "dataset_report.md"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    _write_markdown(report, md_path)
    return report
