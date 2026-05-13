#!/usr/bin/env python3
"""Compare the current web model with the previous thesis model and a random baseline."""

from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from bio_music_pipeline.v2.config import MusicDataConfig  # noqa: E402
from bio_music_pipeline.v2.evaluate import (  # noqa: E402
    _aggregate,
    _random_baseline_sample,
    compute_structured_midi_metrics,
)
from bio_music_pipeline.v2.structured_generate import (  # noqa: E402
    generate_structured_music_from_fasta,
    generate_structured_music_from_fasta_fragmented,
)
from web.generator import BioMusicGenerator  # noqa: E402

METRIC_NAMES = (
    "melody_note_count",
    "note_density_per_bar",
    "pitch_range",
    "unique_pitches",
    "pitch_class_entropy",
    "chord_change_rate",
    "chord_tone_ratio",
    "self_similarity",
)


@dataclass
class ModelComparisonConfig:
    current_checkpoint: str
    current_config: str
    previous_checkpoint: str
    fasta: str
    output_dir: str
    web_current_output_dir: str
    max_records: int = 12
    seed: int = 42
    device: str = "auto"
    eval_fragment_length: int = 1800
    eval_fragment_stride: int = 900


def _as_project_relative(path: str | Path) -> str:
    path = Path(path)
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT.resolve()))
    except ValueError:
        return str(path)


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _metric_mean(aggregate: Dict[str, Any], metric_name: str) -> Optional[float]:
    metric = aggregate.get("metrics", {}).get(metric_name)
    if metric is None:
        return None
    return float(metric["mean"])


def _valid_rate(aggregate: Dict[str, Any]) -> float:
    return float(aggregate.get("valid_count", 0) / max(aggregate.get("count", 0), 1))


def _delta_table(
    current: Dict[str, Any],
    reference: Dict[str, Any],
    metric_names: Sequence[str] = METRIC_NAMES,
) -> Dict[str, Dict[str, Optional[float]]]:
    rows: Dict[str, Dict[str, Optional[float]]] = {
        "valid_rate": {
            "current": _valid_rate(current),
            "reference": _valid_rate(reference),
            "absolute_delta": _valid_rate(current) - _valid_rate(reference),
            "relative_delta": 0.0 if _valid_rate(reference) else None,
        }
    }
    for metric_name in metric_names:
        current_mean = _metric_mean(current, metric_name)
        reference_mean = _metric_mean(reference, metric_name)
        absolute_delta = None
        relative_delta = None
        if current_mean is not None and reference_mean is not None:
            absolute_delta = current_mean - reference_mean
            if reference_mean != 0:
                relative_delta = absolute_delta / abs(reference_mean)
        rows[metric_name] = {
            "current": current_mean,
            "reference": reference_mean,
            "absolute_delta": absolute_delta,
            "relative_delta": relative_delta,
        }
    return rows


def _format_float(value: Optional[float], digits: int = 4) -> str:
    if value is None:
        return "n/a"
    return f"{value:.{digits}f}"


def _format_percent(value: Optional[float]) -> str:
    if value is None:
        return "n/a"
    return f"{value * 100:+.1f}%"


def _load_training_metrics(checkpoint_path: str) -> Dict[str, Any]:
    metrics_path = Path(checkpoint_path).resolve().parents[1] / "metrics.json"
    if not metrics_path.exists():
        return {"metrics_path": str(metrics_path), "available": False}
    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    harmony_history = payload.get("harmony_history", [])
    melody_history = payload.get("melody_history", [])
    return {
        "metrics_path": _as_project_relative(metrics_path),
        "available": True,
        "harmony_best_val_loss": min((float(item["val_loss"]) for item in harmony_history), default=None),
        "melody_best_val_loss": min((float(item["val_loss"]) for item in melody_history), default=None),
        "harmony_test_loss": payload.get("harmony_test_loss"),
        "melody_test_loss": payload.get("melody_test_loss"),
        "n_bio_sequences": payload.get("n_bio_sequences"),
        "n_music_segments": payload.get("n_music_segments"),
        "n_train_pairs": payload.get("n_train_pairs"),
        "n_val_pairs": payload.get("n_val_pairs"),
        "n_test_pairs": payload.get("n_test_pairs"),
    }


def _loss_delta(current: Dict[str, Any], previous: Dict[str, Any]) -> Dict[str, Dict[str, Optional[float]]]:
    rows = {}
    for key in (
        "harmony_best_val_loss",
        "melody_best_val_loss",
        "harmony_test_loss",
        "melody_test_loss",
    ):
        current_value = current.get(key)
        previous_value = previous.get(key)
        improvement = None
        if current_value is not None and previous_value not in (None, 0):
            improvement = (float(previous_value) - float(current_value)) / float(previous_value)
        rows[key] = {
            "current": current_value,
            "previous": previous_value,
            "relative_improvement": improvement,
        }
    return rows


def _wrap_sequence(sequence: str, width: int = 80) -> str:
    return "\n".join(sequence[index : index + width] for index in range(0, len(sequence), width))


def _write_evaluation_fragment_fasta(config: ModelComparisonConfig, output_path: Path) -> Path:
    from Bio import SeqIO

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fragments: list[tuple[str, str]] = []
    for source_record in SeqIO.parse(config.fasta, "fasta"):
        sequence = str(source_record.seq).upper()
        for start in range(0, len(sequence), config.eval_fragment_stride):
            fragment = sequence[start : start + config.eval_fragment_length]
            if len(fragment) < config.eval_fragment_length:
                continue
            fragment_index = len(fragments)
            header = f"{source_record.id}::frag{fragment_index:03d}|start={start}|length={len(fragment)}"
            fragments.append((header, fragment))
            if len(fragments) >= config.max_records:
                output_path.write_text(
                    "\n".join(f">{header}\n{_wrap_sequence(fragment)}" for header, fragment in fragments) + "\n",
                    encoding="utf-8",
                )
                return output_path

    if not fragments:
        raise ValueError(f"No {config.eval_fragment_length} bp fragments found in {config.fasta}.")
    output_path.write_text(
        "\n".join(f">{header}\n{_wrap_sequence(fragment)}" for header, fragment in fragments) + "\n",
        encoding="utf-8",
    )
    return output_path


def _seed_generation(seed: int) -> None:
    random.seed(seed)
    try:
        import numpy as np
        import torch

        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def _generate_current_web_samples(
    config: ModelComparisonConfig, midi_dir: Path, metadata_dir: Path
) -> list[Dict[str, Any]]:
    results = []
    for record_index in range(max(1, config.max_records)):
        midi_path = midi_dir / f"current_web_record_{record_index:03d}.mid"
        metadata_path = metadata_dir / f"current_web_record_{record_index:03d}.json"
        try:
            _seed_generation(config.seed + record_index)
            metadata = generate_structured_music_from_fasta_fragmented(
                fasta_path=config.fasta,
                checkpoint_path=config.current_checkpoint,
                output_path=str(midi_path),
                bars_per_fragment=4,
                config_path=config.current_config,
                record_index=record_index,
                metadata_output=str(metadata_path),
                device_name=config.device,
            )
        except IndexError:
            break
        results.append(
            {
                "condition": "current_web_model",
                "record_index": record_index,
                "midi_path": str(midi_path),
                "metadata_path": str(metadata_path),
                "metadata": metadata,
                "metrics": compute_structured_midi_metrics(str(midi_path), metadata),
            }
        )
    return results


def _generate_previous_samples(
    config: ModelComparisonConfig, midi_dir: Path, metadata_dir: Path, count: int
) -> list[Dict[str, Any]]:
    results = []
    for record_index in range(max(1, count)):
        midi_path = midi_dir / f"previous_thesis_record_{record_index:03d}.mid"
        metadata_path = metadata_dir / f"previous_thesis_record_{record_index:03d}.json"
        try:
            _seed_generation(config.seed + 10_000 + record_index)
            metadata = generate_structured_music_from_fasta(
                fasta_path=config.fasta,
                checkpoint_path=config.previous_checkpoint,
                output_path=str(midi_path),
                config_path=None,
                record_index=record_index,
                metadata_output=str(metadata_path),
                device_name=config.device,
            )
        except IndexError:
            break
        results.append(
            {
                "condition": "previous_thesis_model",
                "record_index": record_index,
                "midi_path": str(midi_path),
                "metadata_path": str(metadata_path),
                "metadata": metadata,
                "metrics": compute_structured_midi_metrics(str(midi_path), metadata),
            }
        )
    return results


def _write_report_markdown(report: Dict[str, Any], output_path: Path) -> None:
    aggregates = report["aggregates"]
    current = aggregates["current_web_model"]
    previous = aggregates["previous_thesis_model"]
    baseline = aggregates["random_baseline"]
    current_vs_previous = report["deltas"]["current_vs_previous"]
    current_vs_baseline = report["deltas"]["current_vs_random_baseline"]
    loss = report["loss_comparison"]

    lines = [
        "# Model Comparison Evaluation",
        "",
        "## Summary",
        "",
        f"- current web checkpoint: `{report['config']['current_checkpoint']}`",
        f"- current web config: `{report['config']['current_config']}`",
        f"- previous thesis checkpoint: `{report['config']['previous_checkpoint']}`",
        f"- source FASTA: `{report['config']['source_fasta']}`",
        f"- evaluation FASTA: `{report['config']['evaluation_fasta']}`",
        f"- records: `{report['config']['max_records']}`",
        f"- seed: `{report['config']['seed']}`",
        f"- device: `{report['config']['device']}`",
        "",
        "## Aggregate Metrics",
        "",
        "| Metric | Current web model | Previous thesis model | Random baseline |",
        "|---|---:|---:|---:|",
        f"| valid_rate | {_format_float(_valid_rate(current))} | {_format_float(_valid_rate(previous))} | {_format_float(_valid_rate(baseline))} |",
    ]
    for metric_name in METRIC_NAMES:
        lines.append(
            "| "
            f"{metric_name} | "
            f"{_format_float(_metric_mean(current, metric_name))} | "
            f"{_format_float(_metric_mean(previous, metric_name))} | "
            f"{_format_float(_metric_mean(baseline, metric_name))} |"
        )

    lines.extend(
        [
            "",
            "## Current vs References",
            "",
            "| Metric | Delta vs previous | Relative vs previous | Delta vs random | Relative vs random |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    previous_deltas = report["deltas"]["current_vs_previous"]
    baseline_deltas = report["deltas"]["current_vs_random_baseline"]
    for metric_name in ("valid_rate", *METRIC_NAMES):
        previous_row = previous_deltas[metric_name]
        baseline_row = baseline_deltas[metric_name]
        lines.append(
            "| "
            f"{metric_name} | "
            f"{_format_float(previous_row['absolute_delta'])} | "
            f"{_format_percent(previous_row['relative_delta'])} | "
            f"{_format_float(baseline_row['absolute_delta'])} | "
            f"{_format_percent(baseline_row['relative_delta'])} |"
        )

    lines.extend(
        [
            "",
            "## Training And Test Loss",
            "",
            "| Metric | Current web model | Previous thesis model | Relative improvement |",
            "|---|---:|---:|---:|",
        ]
    )
    for key, row in loss.items():
        lines.append(
            "| "
            f"{key} | "
            f"{_format_float(row['current'])} | "
            f"{_format_float(row['previous'])} | "
            f"{_format_percent(row['relative_improvement'])} |"
        )

    ctr_current = _metric_mean(current, "chord_tone_ratio")
    ctr_previous = _metric_mean(previous, "chord_tone_ratio")
    ctr_baseline = _metric_mean(baseline, "chord_tone_ratio")
    density_current = _metric_mean(current, "note_density_per_bar")
    density_previous = _metric_mean(previous, "note_density_per_bar")
    entropy_current = _metric_mean(current, "pitch_class_entropy")
    entropy_previous = _metric_mean(previous, "pitch_class_entropy")
    unique_current = _metric_mean(current, "unique_pitches")
    unique_previous = _metric_mean(previous, "unique_pitches")
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            (
                "The current web model is evaluated in the same fragmented 4-bar mode used by the Flask interface. "
                "This keeps each generated segment inside the model's training distribution and makes the measured "
                "MIDI quality representative of the user-facing application."
            ),
            "",
            (
                f"Chord-tone ratio is `{_format_float(ctr_current)}` for the current web model, "
                f"`{_format_float(ctr_previous)}` for the previous thesis model, and "
                f"`{_format_float(ctr_baseline)}` for the random baseline. This metric measures how often melody "
                "notes belong to the active harmony, so higher values support the narrower claim that the generated "
                "melody is harmonically more coordinated."
            ),
            "",
            (
                "The current output is shorter because the user-facing model generates 4-bar fragments, while the "
                "previous thesis model generated longer 8-bar phrases. For that reason, raw melody_note_count is "
                f"less informative than note_density_per_bar: `{_format_float(density_current)}` for the current "
                f"model versus `{_format_float(density_previous)}` for the previous model. Other descriptors should "
                f"be treated as tradeoffs: unique_pitches is `{_format_float(unique_current)}` versus "
                f"`{_format_float(unique_previous)}`, and pitch_class_entropy is `{_format_float(entropy_current)}` "
                f"versus `{_format_float(entropy_previous)}`. This suggests a more focused pitch vocabulary, not a "
                "universal improvement across every descriptor."
            ),
            "",
            (
                "The loss comparison favors the current web model because it was trained on more biological/music "
                "pairs, uses shorter 4-bar segments, and solves a simpler local continuation task. The evaluation "
                "metrics should still be interpreted as structural descriptors, not as proof of biological causality "
                "or complete musical quality."
            ),
            "",
        ]
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")


def _write_web_current_report(report: Dict[str, Any], output_dir: Path) -> None:
    web_report = {
        "config": {
            "checkpoint_path": report["config"]["current_checkpoint"],
            "config_path": report["config"]["current_config"],
            "fasta_path": report["config"]["fasta"],
            "evaluation_fasta_path": report["config"]["evaluation_fasta"],
            "output_dir": str(output_dir),
            "max_records": report["config"]["max_records"],
            "device": report["config"]["device"],
            "seed": report["config"]["seed"],
            "generation_mode": "web_fragmented_4bar",
        },
        "conditions": {
            "current_web_model": report["conditions"]["current_web_model"],
            "random_baseline": report["conditions"]["random_baseline"],
        },
        "aggregates": {
            "current_web_model": report["aggregates"]["current_web_model"],
            "random_baseline": report["aggregates"]["random_baseline"],
        },
        "deltas": {
            "current_vs_random_baseline": report["deltas"]["current_vs_random_baseline"],
        },
        "notes": report["notes"],
    }
    _write_json(output_dir / "evaluation_report.json", web_report)

    lines = [
        "# Current Web Model Evaluation",
        "",
        "## Summary",
        "",
        f"- checkpoint: `{web_report['config']['checkpoint_path']}`",
        f"- config: `{web_report['config']['config_path']}`",
        f"- source FASTA: `{web_report['config']['fasta_path']}`",
        f"- evaluation FASTA: `{web_report['config']['evaluation_fasta_path']}`",
        f"- generation mode: `{web_report['config']['generation_mode']}`",
        f"- max records: `{web_report['config']['max_records']}`",
        "",
        "## Aggregate Metrics",
        "",
        "| Metric | Current web model | Random baseline | Delta | Relative delta |",
        "|---|---:|---:|---:|---:|",
    ]
    deltas = web_report["deltas"]["current_vs_random_baseline"]
    for metric_name in ("valid_rate", *METRIC_NAMES):
        row = deltas[metric_name]
        lines.append(
            "| "
            f"{metric_name} | "
            f"{_format_float(row['current'])} | "
            f"{_format_float(row['reference'])} | "
            f"{_format_float(row['absolute_delta'])} | "
            f"{_format_percent(row['relative_delta'])} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation Limits",
            "",
            "These metrics validate MIDI structure and compare simple musical descriptors. They do not prove a causal relationship between biological sequences and music.",
            "",
        ]
    )
    (output_dir / "evaluation_report.md").write_text("\n".join(lines), encoding="utf-8")


def _write_thesis_update(report: Dict[str, Any], output_path: Path) -> None:
    aggregates = report["aggregates"]
    loss = report["loss_comparison"]
    current = aggregates["current_web_model"]
    previous = aggregates["previous_thesis_model"]
    baseline = aggregates["random_baseline"]
    current_vs_previous = report["deltas"]["current_vs_previous"]
    current_vs_baseline = report["deltas"]["current_vs_random_baseline"]

    lines = [
        "# Model Comparison Update For Thesis",
        "",
        "## Ready-To-Paste Summary",
        "",
        (
            "Для повторной оценки была проверена актуальная модель, которую автоматически загружает "
            "веб-интерфейс: `v2_medium_rtx2060_fast`. Генерация выполнялась в пользовательском режиме "
            "`fragmented`, где каждый фрагмент FASTA длиной 1800 bp преобразуется в 4 музыкальных такта."
        ),
        "",
        "| Metric | Current web model | Previous thesis model | Random baseline |",
        "|---|---:|---:|---:|",
        f"| Valid MIDI rate | {_format_float(_valid_rate(current))} | {_format_float(_valid_rate(previous))} | {_format_float(_valid_rate(baseline))} |",
    ]
    for metric_name in METRIC_NAMES:
        lines.append(
            "| "
            f"{metric_name} | "
            f"{_format_float(_metric_mean(current, metric_name))} | "
            f"{_format_float(_metric_mean(previous, metric_name))} | "
            f"{_format_float(_metric_mean(baseline, metric_name))} |"
        )

    lines.extend(
        [
            "",
            "## Main Quantitative Differences",
            "",
            (
                "Главное улучшение относительно random baseline видно по `chord_tone_ratio`: "
                f"`{_format_float(_metric_mean(current, 'chord_tone_ratio'))}` против "
                f"`{_format_float(_metric_mean(baseline, 'chord_tone_ratio'))}`, то есть "
                f"{_format_percent(current_vs_baseline['chord_tone_ratio']['relative_delta'])}. "
                "Относительно предыдущей модели эта же метрика выросла с "
                f"`{_format_float(_metric_mean(previous, 'chord_tone_ratio'))}` до "
                f"`{_format_float(_metric_mean(current, 'chord_tone_ratio'))}` "
                f"({_format_percent(current_vs_previous['chord_tone_ratio']['relative_delta'])})."
            ),
            "",
            (
                "Сравнение по количеству нот нужно читать осторожно: актуальная web-модель генерирует "
                "4-тактовые фрагменты, а предыдущая модель — более длинные 8-тактовые фразы. "
                "Поэтому для плотности мелодии важнее `note_density_per_bar`: "
                f"`{_format_float(_metric_mean(current, 'note_density_per_bar'))}` против "
                f"`{_format_float(_metric_mean(previous, 'note_density_per_bar'))}` у предыдущей модели. "
                "Не все дескрипторы нужно трактовать как прямое улучшение: `unique_pitches` и "
                "`pitch_class_entropy` ниже, поэтому это лучше описать как более сфокусированный "
                "мелодический словарь, а не как победу по каждому показателю."
            ),
            "",
            "## Loss Comparison",
            "",
            "| Metric | Current web model | Previous thesis model | Improvement |",
            "|---|---:|---:|---:|",
        ]
    )
    for key, row in loss.items():
        lines.append(
            "| "
            f"{key} | "
            f"{_format_float(row['current'])} | "
            f"{_format_float(row['previous'])} | "
            f"{_format_percent(row['relative_improvement'])} |"
        )

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            (
                "Улучшение объясняется тремя факторами. Во-первых, актуальная модель обучалась на большем "
                "числе биологических и музыкальных пар. Во-вторых, 4-тактовые сегменты уменьшают сложность "
                "задачи по сравнению с более длинной генерацией. В-третьих, фрагментированный режим не "
                "заставляет модель продолжать музыку за пределами длины, на которой она обучалась."
            ),
            "",
            (
                "Метрики показывают структурные свойства MIDI: валидность, плотность мелодии, диапазон, "
                "энтропию pitch classes, частоту смены аккордов, chord-tone ratio и self-similarity. "
                "Их следует использовать как техническую оценку согласованности результата, а не как "
                "доказательство биологической причинности или художественного качества."
            ),
            "",
        ]
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")


def run_model_comparison(config: ModelComparisonConfig) -> Dict[str, Any]:
    output_dir = Path(config.output_dir)
    midi_dir = output_dir / "midi"
    metadata_dir = output_dir / "metadata"
    midi_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)

    evaluation_fasta = _write_evaluation_fragment_fasta(config, metadata_dir / "evaluation_fragments.fa")
    generation_config = ModelComparisonConfig(**{**asdict(config), "fasta": str(evaluation_fasta)})

    current_results = _generate_current_web_samples(generation_config, midi_dir, metadata_dir)
    previous_results = _generate_previous_samples(generation_config, midi_dir, metadata_dir, len(current_results))

    rng = random.Random(config.seed)
    music_config = MusicDataConfig()
    baseline_results = [
        _random_baseline_sample(index, midi_dir, music_config, rng) for index in range(max(1, len(current_results)))
    ]

    current_aggregate = _aggregate(current_results)
    previous_aggregate = _aggregate(previous_results)
    baseline_aggregate = _aggregate(baseline_results)
    current_training = _load_training_metrics(config.current_checkpoint)
    previous_training = _load_training_metrics(config.previous_checkpoint)

    report = {
        "config": {
            **asdict(config),
            "source_fasta": config.fasta,
            "evaluation_fasta": str(evaluation_fasta),
        },
        "web_defaults": {
            "checkpoint_path": str(BioMusicGenerator._resolve_default_checkpoint_path()),
            "config_path": str(BioMusicGenerator._resolve_default_config_path()),
            "matches_current_checkpoint": Path(BioMusicGenerator._resolve_default_checkpoint_path()).resolve()
            == Path(config.current_checkpoint).resolve(),
            "matches_current_config": Path(BioMusicGenerator._resolve_default_config_path()).resolve()
            == Path(config.current_config).resolve(),
        },
        "conditions": {
            "current_web_model": current_results,
            "previous_thesis_model": previous_results,
            "random_baseline": baseline_results,
        },
        "aggregates": {
            "current_web_model": current_aggregate,
            "previous_thesis_model": previous_aggregate,
            "random_baseline": baseline_aggregate,
        },
        "deltas": {
            "current_vs_previous": _delta_table(current_aggregate, previous_aggregate),
            "current_vs_random_baseline": _delta_table(current_aggregate, baseline_aggregate),
        },
        "training_metrics": {
            "current_web_model": current_training,
            "previous_thesis_model": previous_training,
        },
        "loss_comparison": _loss_delta(current_training, previous_training),
        "notes": [
            "Current model is evaluated with the same fragmented 4-bar path used by the web interface.",
            "Chord-tone ratio falls back to MIDI chord events when generation metadata has no harmony bars.",
            "Metrics are structural descriptors and do not establish biological causality.",
        ],
    }

    _write_json(output_dir / "model_comparison.json", report)
    _write_report_markdown(report, output_dir / "model_comparison.md")
    _write_web_current_report(report, Path(config.web_current_output_dir))
    _write_thesis_update(report, PROJECT_ROOT / "docs" / "thesis" / "model-comparison-update.md")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--current-checkpoint",
        default="results/v2_medium_rtx2060_fast/checkpoints/structured_pipeline.pt",
        help="Checkpoint loaded by the web interface.",
    )
    parser.add_argument(
        "--current-config",
        default="configs/pipeline_v2_medium_rtx2060_fast.json",
        help="Config loaded by the web interface.",
    )
    parser.add_argument(
        "--previous-checkpoint",
        default="results/thesis_final_run_regularized/checkpoints/structured_pipeline.pt",
        help="Previous thesis checkpoint to compare against.",
    )
    parser.add_argument(
        "--fasta",
        default="data/fasta/training/thesis_reference_genomes.fa",
        help="Evaluation FASTA file.",
    )
    parser.add_argument(
        "--output-dir",
        default="results/thesis_evaluation_model_comparison",
        help="Directory for comparison reports and generated MIDI.",
    )
    parser.add_argument(
        "--web-current-output-dir",
        default="results/thesis_evaluation_web_current",
        help="Directory for the current-web-model evaluation report.",
    )
    parser.add_argument("--max-records", type=int, default=12, help="Maximum FASTA records to evaluate.")
    parser.add_argument("--seed", type=int, default=42, help="Random baseline seed.")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto", help="Generation device.")
    parser.add_argument("--eval-fragment-length", type=int, default=1800, help="Evaluation fragment length in bp.")
    parser.add_argument("--eval-fragment-stride", type=int, default=900, help="Evaluation fragment stride in bp.")
    args = parser.parse_args()

    report = run_model_comparison(
        ModelComparisonConfig(
            current_checkpoint=args.current_checkpoint,
            current_config=args.current_config,
            previous_checkpoint=args.previous_checkpoint,
            fasta=args.fasta,
            output_dir=args.output_dir,
            web_current_output_dir=args.web_current_output_dir,
            max_records=args.max_records,
            seed=args.seed,
            device=args.device,
            eval_fragment_length=args.eval_fragment_length,
            eval_fragment_stride=args.eval_fragment_stride,
        )
    )
    print(json.dumps(report["aggregates"], indent=2))


if __name__ == "__main__":
    main()
