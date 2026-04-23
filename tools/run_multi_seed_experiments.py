#!/usr/bin/env python3
"""
Run the pipeline across multiple random seeds and aggregate metrics.

This script creates per-seed configs, executes `run_pipeline.py`, then writes:
- per_seed_summary.csv
- per_seed_summary.json
- aggregate_metrics.json
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple


@dataclass
class SeedRunResult:
    seed: int
    output_dir: str
    status: str
    return_code: int
    report_path: str
    metrics: Dict[str, float]


def parse_seed_list(raw: str) -> List[int]:
    values = []
    for chunk in raw.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        values.append(int(chunk))
    if not values:
        raise ValueError("No valid seeds provided.")
    return values


def key_entropy(key_distribution: Dict[str, float]) -> float:
    probs = [float(v) for v in key_distribution.values() if float(v) > 0]
    if not probs:
        return 0.0
    return float(-sum(p * math.log(p, 2) for p in probs))


def ci95(values: List[float]) -> Tuple[float, float]:
    if not values:
        return 0.0, 0.0
    if len(values) == 1:
        return values[0], values[0]
    mean = float(sum(values) / len(values))
    var = sum((x - mean) ** 2 for x in values) / (len(values) - 1)
    std = math.sqrt(max(var, 0.0))
    half = 1.96 * std / math.sqrt(len(values))
    return mean - half, mean + half


def load_report_metrics(report_path: Path) -> Dict[str, float]:
    with report_path.open("r") as f:
        report = json.load(f)

    stage2 = report.get("stage2", {})
    stage5 = report.get("stage5", {})
    tests = report.get("hypothesis_tests", {})
    baseline = stage5.get("baseline_comparison", {})

    conditioned = baseline.get("conditioned", {})
    random_vector = baseline.get("random_vector", {})

    conditioned_token_mean = float(conditioned.get("mean_token_value", 0.0))
    random_vector_token_mean = float(random_vector.get("mean_token_value", 0.0))

    metrics = {
        "key_entropy": key_entropy(stage2.get("key_distribution", {})),
        "tempo_min": float(stage2.get("tempo_range_actual", [0.0, 0.0])[0]),
        "tempo_max": float(stage2.get("tempo_range_actual", [0.0, 0.0])[1]),
        "tempo_span": float(stage2.get("tempo_range_actual", [0.0, 0.0])[1])
        - float(stage2.get("tempo_range_actual", [0.0, 0.0])[0]),
        "disentanglement_mean_gap": float(tests.get("disentanglement", {}).get("mean_gap", 0.0)),
        "disentanglement_p_value": float(tests.get("disentanglement", {}).get("p_value", 1.0)),
        "information_transfer_accuracy": float(
            tests.get("permutation_test", {}).get("original_accuracy", 0.0)
        ),
        "information_transfer_p_value": float(
            tests.get("permutation_test", {}).get("permutation_p_value", 1.0)
        ),
        "conditioned_token_mean": conditioned_token_mean,
        "random_vector_token_mean": random_vector_token_mean,
        "conditioned_vs_random_vector_abs_diff": abs(conditioned_token_mean - random_vector_token_mean),
    }
    return metrics


def build_seed_config(base_config: Dict, seed: int, output_dir: Path) -> Dict:
    cfg = json.loads(json.dumps(base_config))
    cfg.setdefault("pipeline", {})
    cfg["pipeline"]["seed"] = int(seed)
    cfg["pipeline"]["output_dir"] = str(output_dir)
    return cfg


def run_single_seed(
    python_exec: str,
    seed: int,
    base_config: Dict,
    work_dir: Path,
    config_out_dir: Path,
    output_root: Path,
    midi_dir: str | None,
    paired_data: str | None,
    sequences: str | None,
    allow_synthetic: bool,
    skip_existing: bool,
) -> SeedRunResult:
    seed_output = output_root / f"seed_{seed}"
    seed_output.mkdir(parents=True, exist_ok=True)
    report_path = seed_output / "final_report.json"

    if skip_existing and report_path.exists():
        metrics = load_report_metrics(report_path)
        return SeedRunResult(
            seed=seed,
            output_dir=str(seed_output),
            status="reused",
            return_code=0,
            report_path=str(report_path),
            metrics=metrics,
        )

    config_out_dir.mkdir(parents=True, exist_ok=True)
    seed_config_path = config_out_dir / f"pipeline_seed_{seed}.json"
    seed_cfg = build_seed_config(base_config, seed, seed_output)
    with seed_config_path.open("w") as f:
        json.dump(seed_cfg, f, indent=2)

    cmd = [python_exec, "run_pipeline.py", "--config", str(seed_config_path)]
    if midi_dir:
        cmd.extend(["--midi-dir", midi_dir])
    if paired_data:
        cmd.extend(["--paired-data", paired_data])
    if sequences:
        cmd.extend(["--sequences", sequences])
    if allow_synthetic:
        cmd.append("--allow-synthetic")

    proc = subprocess.run(cmd, cwd=str(work_dir), capture_output=True, text=True)
    (seed_output / "run_stdout.log").write_text(proc.stdout or "")
    (seed_output / "run_stderr.log").write_text(proc.stderr or "")

    if report_path.exists():
        metrics = load_report_metrics(report_path)
        status = "completed"
    else:
        metrics = {}
        status = "failed"

    return SeedRunResult(
        seed=seed,
        output_dir=str(seed_output),
        status=status,
        return_code=int(proc.returncode),
        report_path=str(report_path) if report_path.exists() else "",
        metrics=metrics,
    )


def write_summaries(results: List[SeedRunResult], output_root: Path) -> None:
    summary_json_path = output_root / "per_seed_summary.json"
    summary_csv_path = output_root / "per_seed_summary.csv"
    aggregate_json_path = output_root / "aggregate_metrics.json"

    rows = []
    for res in results:
        row = {
            "seed": res.seed,
            "status": res.status,
            "return_code": res.return_code,
            "output_dir": res.output_dir,
            "report_path": res.report_path,
        }
        row.update(res.metrics)
        rows.append(row)

    summary_json_path.write_text(json.dumps(rows, indent=2))

    if rows:
        fields = sorted({k for row in rows for k in row.keys()})
        with summary_csv_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)

    numeric_fields = [
        "key_entropy",
        "tempo_span",
        "disentanglement_mean_gap",
        "disentanglement_p_value",
        "information_transfer_accuracy",
        "information_transfer_p_value",
        "conditioned_vs_random_vector_abs_diff",
    ]

    aggregate = {
        "n_runs": len(rows),
        "n_completed": sum(1 for r in rows if r.get("status") in {"completed", "reused"}),
        "metrics": {},
    }
    for field in numeric_fields:
        vals = [float(r[field]) for r in rows if field in r and isinstance(r[field], (int, float))]
        if not vals:
            continue
        mean = float(sum(vals) / len(vals))
        if len(vals) > 1:
            var = sum((x - mean) ** 2 for x in vals) / (len(vals) - 1)
            std = float(math.sqrt(max(var, 0.0)))
        else:
            std = 0.0
        low, high = ci95(vals)
        aggregate["metrics"][field] = {
            "mean": mean,
            "std": std,
            "min": float(min(vals)),
            "max": float(max(vals)),
            "ci95": [float(low), float(high)],
            "values": vals,
        }

    aggregate_json_path.write_text(json.dumps(aggregate, indent=2))


def main() -> int:
    parser = argparse.ArgumentParser(description="Run BioSonification pipeline across multiple seeds")
    parser.add_argument("--base-config", type=str, default="configs/pipeline_full_paired.json")
    parser.add_argument("--seeds", type=str, default="7,42,123,2026,31415")
    parser.add_argument("--output-root", type=str, default="results/multi_seed")
    parser.add_argument("--configs-out", type=str, default="results/multi_seed/configs")
    parser.add_argument("--python-exec", type=str, default=sys.executable or "python3")
    parser.add_argument("--midi-dir", type=str, default=None)
    parser.add_argument("--paired-data", type=str, default=None)
    parser.add_argument("--sequences", type=str, default=None)
    parser.add_argument("--allow-synthetic", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()

    work_dir = Path(__file__).resolve().parents[1]
    output_root = (work_dir / args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    config_out_dir = (work_dir / args.configs_out).resolve()

    base_config_path = (work_dir / args.base_config).resolve()
    with base_config_path.open("r") as f:
        base_config = json.load(f)

    seeds = parse_seed_list(args.seeds)
    results: List[SeedRunResult] = []
    for seed in seeds:
        print(f"\n=== Running seed {seed} ===")
        result = run_single_seed(
            python_exec=args.python_exec,
            seed=seed,
            base_config=base_config,
            work_dir=work_dir,
            config_out_dir=config_out_dir,
            output_root=output_root,
            midi_dir=args.midi_dir,
            paired_data=args.paired_data,
            sequences=args.sequences,
            allow_synthetic=args.allow_synthetic,
            skip_existing=args.skip_existing,
        )
        print(f"Seed {seed}: status={result.status}, return_code={result.return_code}")
        results.append(result)

    write_summaries(results, output_root)
    print(f"\nMulti-seed summaries saved in: {output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

