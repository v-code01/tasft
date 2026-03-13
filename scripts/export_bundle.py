#!/usr/bin/env python3
"""TASFT bundle export script — packages trained model for sparse inference deployment.

Exports a trained TASFT checkpoint into a self-contained deployment bundle containing:
- Merged model weights (base + LoRA adapters merged)
- AttnGate weights per layer (SafeTensors)
- Kernel configuration (per-layer sparsity thresholds and block sizes)
- Bundle manifest with SHA256 checksums for integrity verification
- Optional evaluation summary

Usage:
    python scripts/export_bundle.py \
        --checkpoint ./outputs/llama3_8b_medqa/best \
        --output ./bundles/llama3_8b_medqa

    # Include evaluation results in bundle
    python scripts/export_bundle.py \
        --checkpoint ./outputs/llama3_8b_medqa/best \
        --output ./bundles/llama3_8b_medqa \
        --eval-results ./outputs/llama3_8b_medqa/eval_results.json

Postconditions:
    - Bundle directory created with all artifacts
    - manifest.json with SHA256 checksums for every file
    - All SafeTensors files verified via round-trip load
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import typer
import yaml

from tasft.observability import configure_logging, get_logger

app = typer.Typer(
    name="tasft-export",
    help="Export a trained TASFT model as a deployment bundle.",
    no_args_is_help=True,
)

logger = get_logger(__name__)


@app.command()
def export(
    checkpoint: Path = typer.Option(
        ...,
        "--checkpoint",
        "-c",
        help="Path to trained TASFT checkpoint directory.",
        exists=True,
        readable=True,
    ),
    output: Path = typer.Option(
        ...,
        "--output",
        "-o",
        help="Output directory for the bundle.",
    ),
    eval_results: Path | None = typer.Option(
        None,
        "--eval-results",
        "-e",
        help="Path to eval_results.json to include in bundle.",
    ),
    config: Path | None = typer.Option(
        None,
        "--config",
        help="Path to training config YAML. Auto-detected from checkpoint if not provided.",
    ),
    merge_lora: bool = typer.Option(
        True,
        "--merge-lora/--no-merge-lora",
        help="Merge LoRA adapters into base weights before export.",
    ),
    verify: bool = typer.Option(
        True,
        "--verify/--no-verify",
        help="Verify bundle integrity after export via round-trip load.",
    ),
    log_level: str = typer.Option(
        "INFO",
        "--log-level",
        help="Logging level.",
    ),
) -> None:
    """Export a trained TASFT checkpoint as a deployment bundle."""
    configure_logging(level=log_level)
    start_time = time.perf_counter()

    logger.info(
        "bundle_export_started",
        checkpoint=str(checkpoint),
        output=str(output),
        merge_lora=merge_lora,
    )

    # Load training config
    cfg: dict[str, Any] = {}
    if config is not None:
        with open(config) as f:
            cfg = yaml.safe_load(f)
    else:
        for candidate in ["resolved_config.yaml", "config.yaml"]:
            candidate_path = checkpoint / candidate
            if candidate_path.exists():
                with open(candidate_path) as f:
                    cfg = yaml.safe_load(f)
                logger.info("config_auto_detected", path=str(candidate_path))
                break

    # Load evaluation results if provided
    eval_summary_data: dict[str, Any] | None = None
    if eval_results is not None:
        if not eval_results.exists():
            typer.echo(f"Warning: eval_results file not found: {eval_results}", err=True)
        else:
            with open(eval_results) as f:
                eval_summary_data = json.load(f)

    # Deferred imports for heavy dependencies
    import torch

    from tasft.bundle.exporter import BundleExporter

    # Load model and gates from checkpoint
    logger.info("loading_checkpoint", checkpoint=str(checkpoint))

    exporter = BundleExporter.from_checkpoint(
        checkpoint_dir=checkpoint,
        training_config=cfg,
        merge_lora=merge_lora,
    )

    # Run export
    manifest = exporter.export(
        output_dir=output,
        eval_summary=eval_summary_data,
    )

    elapsed_s = time.perf_counter() - start_time

    logger.info(
        "bundle_export_completed",
        output=str(output),
        total_size_bytes=manifest.total_size_bytes,
        num_files=len(manifest.file_checksums),
        num_layers=manifest.num_layers,
        elapsed_seconds=round(elapsed_s, 1),
    )

    # Verify bundle integrity
    if verify:
        logger.info("bundle_verification_started")
        from tasft.bundle.exporter import BundleExporter

        verification_result = BundleExporter.verify_bundle(output)
        if verification_result.is_valid:
            logger.info("bundle_verification_passed", num_files=verification_result.files_checked)
        else:
            logger.error(
                "bundle_verification_failed",
                failures=verification_result.failures,
            )
            raise typer.Exit(code=1)

    # Print summary
    typer.echo(f"\nBundle exported successfully to: {output}")
    typer.echo(f"  Model: {manifest.model_name}")
    typer.echo(f"  Base: {manifest.base_model_id}")
    typer.echo(f"  Layers: {manifest.num_layers}")
    typer.echo(f"  Files: {len(manifest.file_checksums)}")
    typer.echo(f"  Size: {manifest.total_size_bytes / (1024**3):.2f} GB")
    typer.echo(f"  Time: {elapsed_s:.1f}s")


if __name__ == "__main__":
    app()
