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

    from tasft.bundle.export import BundleExporter, ExportConfig

    # Build export config from training config
    model_cfg = cfg.get("model", {})
    gate_cfg = cfg.get("gate", {})
    export_config = ExportConfig(
        model_name=model_cfg.get("model_name", model_cfg.get("base_model_id", "unknown")),
        base_model_id=model_cfg.get("base_model_id", "unknown"),
        domain=cfg.get("domain", "general"),
        block_size=gate_cfg.get("block_size", 64),
        global_threshold=gate_cfg.get("default_threshold", 0.5),
    )

    # Load model from checkpoint
    logger.info("loading_checkpoint", checkpoint=str(checkpoint))
    from peft import PeftModel
    from transformers import AutoModelForCausalLM

    base_model_id = model_cfg.get("base_model_id")
    if base_model_id:
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        model = PeftModel.from_pretrained(base_model, str(checkpoint))
    else:
        model = AutoModelForCausalLM.from_pretrained(
            str(checkpoint),
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )

    # Build eval summary if provided
    eval_summary = None
    if eval_summary_data is not None:
        from tasft.bundle.bundle_schema import EvalSummary

        eval_summary = EvalSummary(**eval_summary_data)

    # Run export
    exporter = BundleExporter(config=export_config)
    bundle_path = exporter.export(
        model=model,
        output_dir=output,
        eval_results=eval_summary,
    )

    elapsed_s = time.perf_counter() - start_time

    metadata = BundleExporter.load_bundle_metadata(bundle_path)
    manifest = metadata.manifest

    logger.info(
        "bundle_export_completed",
        output=str(output),
        total_size_bytes=manifest.total_size_bytes,
        num_files=len(manifest.checksums),
        num_layers=manifest.num_layers,
        elapsed_seconds=round(elapsed_s, 1),
    )

    # Verify bundle integrity
    if verify:
        logger.info("bundle_verification_started")
        verification_result = BundleExporter.validate_bundle(bundle_path)
        if verification_result.is_valid:
            logger.info("bundle_verification_passed", num_files=verification_result.checked_files)
        else:
            logger.error(
                "bundle_verification_failed",
                errors=verification_result.errors,
            )
            raise typer.Exit(code=1)

    # Print summary
    typer.echo(f"\nBundle exported successfully to: {output}")
    typer.echo(f"  Model: {manifest.model_name}")
    typer.echo(f"  Base: {manifest.base_model_id}")
    typer.echo(f"  Layers: {manifest.num_layers}")
    typer.echo(f"  Files: {len(manifest.checksums)}")
    typer.echo(f"  Size: {manifest.total_size_bytes / (1024**3):.2f} GB")
    typer.echo(f"  Time: {elapsed_s:.1f}s")


if __name__ == "__main__":
    app()
