#!/usr/bin/env python3
"""TASFT evaluation script — runs task quality, gate quality, and throughput benchmarks.

Evaluates a trained TASFT model or bundle across three dimensions:
1. Task accuracy: lm-eval benchmark (e.g., MedQA, HumanEval)
2. Gate quality: sparsity profile, KL divergence, per-layer analysis
3. Throughput: tokens/second at various batch sizes and sequence lengths

Usage:
    # Evaluate from checkpoint
    python scripts/eval.py --checkpoint ./outputs/llama3_8b_medqa/best

    # Evaluate from bundle
    python scripts/eval.py --bundle ./bundles/llama3_8b_medqa

    # Run only specific evaluations
    python scripts/eval.py --checkpoint ./outputs/llama3_8b_medqa/best --eval-type task gate

Postconditions:
    - JSON results file written to output directory
    - Summary table printed to stdout
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

import typer
import yaml

from tasft.observability import configure_logging, get_logger

app = typer.Typer(
    name="tasft-eval",
    help="TASFT evaluation: task accuracy, gate quality, and throughput.",
    no_args_is_help=True,
)

logger = get_logger(__name__)


def _format_table(headers: list[str], rows: list[list[str]], title: str) -> str:
    """Format data as a fixed-width ASCII table.

    Args:
        headers: Column header strings.
        rows: List of row data (each row is a list of strings).
        title: Table title.

    Returns:
        Formatted table string.

    Complexity: O(rows * cols).
    """
    col_widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            if i < len(col_widths):
                col_widths[i] = max(col_widths[i], len(cell))

    separator = "+" + "+".join("-" * (w + 2) for w in col_widths) + "+"
    header_line = "|" + "|".join(f" {h:<{col_widths[i]}} " for i, h in enumerate(headers)) + "|"

    lines = [f"\n{title}", separator, header_line, separator]
    for row in rows:
        padded = [f" {cell:<{col_widths[i]}} " if i < len(col_widths) else f" {cell} " for i, cell in enumerate(row)]
        lines.append("|" + "|".join(padded) + "|")
    lines.append(separator)

    return "\n".join(lines)


def _run_task_eval(
    model_path: Path,
    eval_cfg: dict[str, Any],
) -> dict[str, Any]:
    """Run lm-eval task evaluation.

    Args:
        model_path: Path to model checkpoint or bundle.
        eval_cfg: Evaluation configuration section.

    Returns:
        Task evaluation results dict.
    """
    task_cfg = eval_cfg.get("task_eval", {})
    benchmark = task_cfg.get("benchmark", "hellaswag")
    num_fewshot = task_cfg.get("num_fewshot", 0)
    batch_size = task_cfg.get("batch_size", 16)

    logger.info(
        "task_eval_started",
        benchmark=benchmark,
        num_fewshot=num_fewshot,
        batch_size=batch_size,
    )

    from tasft.eval.task_evaluator import TaskEvaluator

    evaluator = TaskEvaluator(model_path=model_path)
    results = evaluator.evaluate(
        benchmark=benchmark,
        num_fewshot=num_fewshot,
        batch_size=batch_size,
    )

    logger.info(
        "task_eval_completed",
        benchmark=benchmark,
        accuracy=results.get("accuracy", 0.0),
    )
    return results


def _run_gate_eval(
    model_path: Path,
    eval_cfg: dict[str, Any],
) -> dict[str, Any]:
    """Run gate quality evaluation — sparsity profile and KL divergence.

    Args:
        model_path: Path to model checkpoint or bundle.
        eval_cfg: Evaluation configuration section.

    Returns:
        Gate evaluation results dict.
    """
    gate_cfg = eval_cfg.get("gate_eval", {})

    logger.info(
        "gate_eval_started",
        compute_sparsity=gate_cfg.get("compute_sparsity_profile", True),
        compute_kl=gate_cfg.get("compute_kl_divergence", True),
    )

    from tasft.eval.gate_evaluator import GateEvaluator

    evaluator = GateEvaluator(model_path=model_path)
    results = evaluator.evaluate(
        compute_sparsity_profile=gate_cfg.get("compute_sparsity_profile", True),
        compute_kl_divergence=gate_cfg.get("compute_kl_divergence", True),
    )

    logger.info(
        "gate_eval_completed",
        mean_sparsity=results.get("mean_sparsity", 0.0),
        mean_kl=results.get("mean_kl_divergence", 0.0),
    )
    return results


def _run_throughput_eval(
    model_path: Path,
    eval_cfg: dict[str, Any],
) -> dict[str, Any]:
    """Run throughput benchmark — tokens/second at various configurations.

    Args:
        model_path: Path to model checkpoint or bundle.
        eval_cfg: Evaluation configuration section.

    Returns:
        Throughput evaluation results dict.
    """
    tp_cfg = eval_cfg.get("throughput_eval", {})
    seq_lengths = tp_cfg.get("seq_lengths", [512, 1024, 2048])
    batch_sizes = tp_cfg.get("batch_sizes", [1, 4, 8])

    logger.info(
        "throughput_eval_started",
        seq_lengths=seq_lengths,
        batch_sizes=batch_sizes,
    )

    from tasft.eval.throughput_evaluator import ThroughputEvaluator

    evaluator = ThroughputEvaluator(model_path=model_path)
    results = evaluator.evaluate(
        seq_lengths=seq_lengths,
        batch_sizes=batch_sizes,
        num_warmup=tp_cfg.get("num_warmup", 5),
        num_iterations=tp_cfg.get("num_iterations", 50),
    )

    logger.info(
        "throughput_eval_completed",
        peak_tokens_per_sec=results.get("peak_tokens_per_second", 0.0),
    )
    return results


@app.command()
def evaluate(
    checkpoint: Path | None = typer.Option(
        None,
        "--checkpoint",
        "-c",
        help="Path to trained model checkpoint directory.",
    ),
    bundle: Path | None = typer.Option(
        None,
        "--bundle",
        "-b",
        help="Path to exported TASFT bundle directory.",
    ),
    config: Path | None = typer.Option(
        None,
        "--config",
        help="Path to config YAML (auto-detected from checkpoint if not provided).",
    ),
    eval_type: list[str] | None = typer.Option(
        None,
        "--eval-type",
        "-t",
        help="Evaluation types to run: task, gate, throughput. Default: all.",
    ),
    output_file: Path | None = typer.Option(
        None,
        "--output",
        "-o",
        help="Path for JSON results file. Default: <model_dir>/eval_results.json.",
    ),
    log_level: str = typer.Option(
        "INFO",
        "--log-level",
        help="Logging level.",
    ),
) -> None:
    """Evaluate a TASFT model across task quality, gate quality, and throughput."""
    configure_logging(level=log_level)
    start_time = time.perf_counter()

    # Determine model path
    if checkpoint is None and bundle is None:
        typer.echo("Error: must provide --checkpoint or --bundle", err=True)
        raise typer.Exit(code=1)

    model_path = checkpoint if checkpoint is not None else bundle
    assert model_path is not None

    if not model_path.exists():
        typer.echo(f"Error: path does not exist: {model_path}", err=True)
        raise typer.Exit(code=1)

    # Load evaluation config
    eval_cfg: dict[str, Any] = {}
    if config is not None:
        with open(config) as f:
            full_cfg = yaml.safe_load(f)
        eval_cfg = full_cfg.get("evaluation", {})
    else:
        # Try to find config in checkpoint/bundle directory
        for candidate in ["resolved_config.yaml", "config.yaml"]:
            candidate_path = model_path / candidate
            if candidate_path.exists():
                with open(candidate_path) as f:
                    full_cfg = yaml.safe_load(f)
                eval_cfg = full_cfg.get("evaluation", {})
                break

    # Determine which evaluations to run
    eval_types = set(eval_type) if eval_type else {"task", "gate", "throughput"}

    logger.info(
        "evaluation_started",
        model_path=str(model_path),
        eval_types=sorted(eval_types),
    )

    all_results: dict[str, Any] = {
        "model_path": str(model_path),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "evaluations": {},
    }

    # Run evaluations
    if "task" in eval_types:
        all_results["evaluations"]["task"] = _run_task_eval(model_path, eval_cfg)

    if "gate" in eval_types:
        all_results["evaluations"]["gate"] = _run_gate_eval(model_path, eval_cfg)

    if "throughput" in eval_types:
        all_results["evaluations"]["throughput"] = _run_throughput_eval(model_path, eval_cfg)

    elapsed_s = time.perf_counter() - start_time
    all_results["total_eval_seconds"] = round(elapsed_s, 1)

    # Write JSON results
    if output_file is None:
        output_file = model_path / "eval_results.json"

    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    logger.info("eval_results_written", output=str(output_file))

    # Print summary table
    summary_rows: list[list[str]] = []
    evals = all_results["evaluations"]

    if "task" in evals:
        task = evals["task"]
        summary_rows.append([
            "Task Accuracy",
            f"{task.get('accuracy', 0.0):.4f}",
            task.get("benchmark", "unknown"),
        ])

    if "gate" in evals:
        gate = evals["gate"]
        summary_rows.append([
            "Mean Sparsity",
            f"{gate.get('mean_sparsity', 0.0):.4f}",
            f"KL={gate.get('mean_kl_divergence', 0.0):.4f}",
        ])

    if "throughput" in evals:
        tp = evals["throughput"]
        summary_rows.append([
            "Peak Throughput",
            f"{tp.get('peak_tokens_per_second', 0.0):.0f} tok/s",
            f"Speedup={tp.get('speedup_vs_dense', 0.0):.2f}x",
        ])

    table = _format_table(
        headers=["Metric", "Value", "Details"],
        rows=summary_rows,
        title="TASFT Evaluation Summary",
    )
    typer.echo(table)
    typer.echo(f"\nResults saved to: {output_file}")
    typer.echo(f"Total evaluation time: {elapsed_s:.1f}s")


if __name__ == "__main__":
    app()
