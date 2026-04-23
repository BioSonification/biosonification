#!/usr/bin/env python3
"""
Generate detailed research artifacts from pipeline reports.

Outputs:
- artifacts_summary.csv
- artifacts_summary.json
- multiple figures in output dir
- research_artifacts.md narrative report
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def safe_float(value, default=0.0) -> float:
    try:
        if value is None:
            return float(default)
        return float(value)
    except Exception:
        return float(default)


def entropy_from_distribution(distribution: Dict[str, float]) -> float:
    probs = [safe_float(v) for v in distribution.values() if safe_float(v) > 0]
    if not probs:
        return 0.0
    return float(-sum(p * math.log(p, 2) for p in probs))


def discover_reports(roots: List[Path]) -> List[Path]:
    reports = []
    for root in roots:
        if root.is_file() and root.name == "final_report.json":
            reports.append(root)
            continue
        if not root.exists():
            continue
        reports.extend(sorted(root.rglob("final_report.json")))
    unique = []
    seen = set()
    for p in reports:
        key = str(p.resolve())
        if key in seen:
            continue
        seen.add(key)
        unique.append(p)
    return unique


def extract_row(report_path: Path) -> Dict[str, object]:
    with report_path.open("r") as f:
        report = json.load(f)

    stage2 = report.get("stage2", {})
    stage5 = report.get("stage5", {})
    tests = report.get("hypothesis_tests", {})
    baseline = stage5.get("baseline_comparison", {})
    musical_quality = tests.get("musical_quality", {})

    tempo_range = stage2.get("tempo_range_actual", [0.0, 0.0])
    tempo_min = safe_float(tempo_range[0] if len(tempo_range) >= 1 else 0.0)
    tempo_max = safe_float(tempo_range[1] if len(tempo_range) >= 2 else tempo_min)

    conditioned_token_mean = safe_float(baseline.get("conditioned", {}).get("mean_token_value"))
    unconditioned_token_mean = safe_float(baseline.get("unconditional", {}).get("mean_token_value"))
    random_vector_token_mean = safe_float(baseline.get("random_vector", {}).get("mean_token_value"))

    conditioned_mq = safe_float(musical_quality.get("conditioned", {}).get("composite_mean"))
    unconditioned_mq = safe_float(musical_quality.get("unconditional", {}).get("composite_mean"))
    random_vector_mq = safe_float(musical_quality.get("random_vector", {}).get("composite_mean"))

    row = {
        "run_name": report_path.parent.name,
        "report_path": str(report_path.resolve()),
        "status": report.get("status", "unknown"),
        "seed": report.get("config", {}).get("pipeline", {}).get("seed"),
        "n_sequences": report.get("stage1", {}).get("n_sequences"),
        "n_train": report.get("stage3", {}).get("n_train"),
        "n_conditioned_generated": stage5.get("n_samples_generated", {}).get("conditioned"),
        "key_entropy": entropy_from_distribution(stage2.get("key_distribution", {})),
        "tempo_min": tempo_min,
        "tempo_max": tempo_max,
        "tempo_span": tempo_max - tempo_min,
        "disentanglement_mean_gap": safe_float(tests.get("disentanglement", {}).get("mean_gap")),
        "disentanglement_p_value": safe_float(tests.get("disentanglement", {}).get("p_value"), 1.0),
        "disentanglement_significant": bool(tests.get("disentanglement", {}).get("significant", False)),
        "info_transfer_accuracy": safe_float(tests.get("permutation_test", {}).get("original_accuracy")),
        "info_transfer_p_value": safe_float(tests.get("permutation_test", {}).get("permutation_p_value"), 1.0),
        "info_transfer_significant": bool(tests.get("permutation_test", {}).get("significant", False)),
        "conditioned_token_mean": conditioned_token_mean,
        "unconditioned_token_mean": unconditioned_token_mean,
        "random_vector_token_mean": random_vector_token_mean,
        "conditioned_minus_unconditioned_tokens": conditioned_token_mean - unconditioned_token_mean,
        "conditioned_minus_random_vector_tokens": conditioned_token_mean - random_vector_token_mean,
        "conditioned_musical_quality": conditioned_mq,
        "unconditioned_musical_quality": unconditioned_mq,
        "random_vector_musical_quality": random_vector_mq,
        "conditioned_minus_unconditioned_mq": conditioned_mq - unconditioned_mq,
    }
    return row


def write_tabular(rows: List[Dict[str, object]], output_dir: Path) -> None:
    json_path = output_dir / "artifacts_summary.json"
    csv_path = output_dir / "artifacts_summary.csv"
    json_path.write_text(json.dumps(rows, indent=2, ensure_ascii=False))

    if not rows:
        return
    fields = sorted({k for row in rows for k in row.keys()})
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def plot_sonification_diversity(rows: List[Dict[str, object]], output_dir: Path) -> str:
    if not rows:
        return ""
    names = [str(r["run_name"]) for r in rows]
    key_entropy = [safe_float(r["key_entropy"]) for r in rows]
    tempo_span = [safe_float(r["tempo_span"]) for r in rows]

    x = np.arange(len(names))
    width = 0.38
    fig, ax = plt.subplots(figsize=(max(10, len(names) * 1.2), 5.5))
    ax.bar(x - width / 2, key_entropy, width=width, label="Key entropy (bits)")
    ax.bar(x + width / 2, tempo_span, width=width, label="Tempo span (BPM)")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=30, ha="right")
    ax.set_title("Sonification Diversity by Run")
    ax.set_ylabel("Value")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()

    path = output_dir / "sonification_diversity.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return str(path)


def plot_hypothesis_overview(rows: List[Dict[str, object]], output_dir: Path) -> str:
    if not rows:
        return ""
    names = [str(r["run_name"]) for r in rows]
    p_dis = [safe_float(r["disentanglement_p_value"], 1.0) for r in rows]
    p_it = [safe_float(r["info_transfer_p_value"], 1.0) for r in rows]

    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(max(10, len(names) * 1.2), 5.5))
    ax.plot(x, p_dis, marker="o", label="Disentanglement p-value")
    ax.plot(x, p_it, marker="s", label="Info-transfer p-value")
    ax.axhline(0.05, color="red", linestyle="--", linewidth=1.2, label="alpha=0.05")
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=30, ha="right")
    ax.set_ylabel("p-value (log scale)")
    ax.set_title("Hypothesis Significance Overview")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()

    path = output_dir / "hypothesis_pvalues.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return str(path)


def plot_conditioning_gaps(rows: List[Dict[str, object]], output_dir: Path) -> str:
    if not rows:
        return ""
    names = [str(r["run_name"]) for r in rows]
    token_gap = [safe_float(r["conditioned_minus_unconditioned_tokens"]) for r in rows]
    random_vec_gap = [safe_float(r["conditioned_minus_random_vector_tokens"]) for r in rows]
    musical_gap = [safe_float(r["conditioned_minus_unconditioned_mq"]) for r in rows]

    x = np.arange(len(names))
    width = 0.28
    fig, ax = plt.subplots(figsize=(max(11, len(names) * 1.3), 6))
    ax.bar(x - width, token_gap, width=width, label="Cond-Uncond token mean gap")
    ax.bar(x, random_vec_gap, width=width, label="Cond-RandomVector token gap")
    ax.bar(x + width, musical_gap, width=width, label="Cond-Uncond musical quality gap")
    ax.axhline(0.0, color="black", linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=30, ha="right")
    ax.set_title("Conditioning Effect Gaps by Run")
    ax.set_ylabel("Gap value")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()

    path = output_dir / "conditioning_gaps.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return str(path)


def build_markdown_report(rows: List[Dict[str, object]], output_dir: Path, plot_paths: Dict[str, str]) -> str:
    md_path = output_dir / "research_artifacts.md"
    lines = []
    lines.append("# BioSonification Research Artifacts")
    lines.append("")
    lines.append(f"Total runs analyzed: **{len(rows)}**")
    lines.append("")
    lines.append("## Key Findings Snapshot")
    lines.append("")
    if rows:
        best_key_div = max(rows, key=lambda r: safe_float(r["key_entropy"]))
        best_dis = min(rows, key=lambda r: safe_float(r["disentanglement_p_value"], 1.0))
        best_it = min(rows, key=lambda r: safe_float(r["info_transfer_p_value"], 1.0))
        lines.append(
            f"- Highest key diversity: `{best_key_div['run_name']}` "
            f"(entropy={safe_float(best_key_div['key_entropy']):.3f})"
        )
        lines.append(
            f"- Strongest H1 p-value: `{best_dis['run_name']}` "
            f"(p={safe_float(best_dis['disentanglement_p_value']):.4g})"
        )
        lines.append(
            f"- Strongest H2 p-value: `{best_it['run_name']}` "
            f"(p={safe_float(best_it['info_transfer_p_value']):.4g})"
        )
    else:
        lines.append("- No reports found.")
    lines.append("")
    lines.append("## Figures")
    lines.append("")
    for name, path in plot_paths.items():
        if not path:
            continue
        abs_path = str((Path(path)).resolve())
        lines.append(f"### {name}")
        lines.append(f"![{name}]({abs_path})")
        lines.append("")

    lines.append("## Per-run Table (abridged)")
    lines.append("")
    lines.append("| run | key_entropy | tempo_span | dis_gap | dis_p | it_acc | it_p |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for r in rows:
        lines.append(
            f"| {r['run_name']} | {safe_float(r['key_entropy']):.3f} | "
            f"{safe_float(r['tempo_span']):.2f} | {safe_float(r['disentanglement_mean_gap']):.3f} | "
            f"{safe_float(r['disentanglement_p_value']):.4g} | {safe_float(r['info_transfer_accuracy']):.3f} | "
            f"{safe_float(r['info_transfer_p_value']):.4g} |"
        )

    md_path.write_text("\n".join(lines), encoding="utf-8")
    return str(md_path)


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate detailed BioSonification research artifacts")
    parser.add_argument(
        "--roots",
        type=str,
        default="results",
        help="Comma-separated directories/files to scan for final_report.json",
    )
    parser.add_argument("--output-dir", type=str, default="results/research_artifacts")
    args = parser.parse_args()

    root_paths = [Path(x.strip()).resolve() for x in args.roots.split(",") if x.strip()]
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    reports = discover_reports(root_paths)
    rows = [extract_row(p) for p in reports]
    rows.sort(key=lambda r: str(r["run_name"]))

    write_tabular(rows, output_dir)

    plots = {
        "Sonification Diversity": plot_sonification_diversity(rows, output_dir),
        "Hypothesis p-values": plot_hypothesis_overview(rows, output_dir),
        "Conditioning Gaps": plot_conditioning_gaps(rows, output_dir),
    }
    report_md = build_markdown_report(rows, output_dir, plots)

    print(f"Reports analyzed: {len(rows)}")
    print(f"Artifacts written to: {output_dir}")
    print(f"Markdown report: {report_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

