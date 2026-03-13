#!/usr/bin/env python3
"""TASFT single-command reproducibility script.

Runs the complete TASFT pipeline in one invocation:
    1. Training: co-train LoRA adapters + AttnGate modules
    2. Bundle export: package trained artifacts for deployment
    3. Evaluation: task accuracy, gate quality, throughput benchmarks
    4. Report: JSON results file with all measurements and hardware info

Addresses the reproducibility gap identified in peer review:
"no public checkpoint, no single-command eval script, no WandB logs."

Usage:
    # Full pipeline with defaults (Llama-3-8B + MedQA)
    python scripts/reproduce.py

    # Dry run: validate config, seed, and hardware without GPU
    python scripts/reproduce.py --dry-run

    # With WandB logging
    python scripts/reproduce.py --wandb --wandb-project tasft-repro

    # Custom model and dataset
    python scripts/reproduce.py --model Qwen/Qwen2.5-7B --dataset humaneval

    # From config file
    python scripts/reproduce.py --config configs/qwen25_7b_humaneval.yaml

Preconditions:
    - Python 3.11+, TASFT installed (pip install -e ".[train,eval]")
    - GPU with sufficient VRAM (see config comments for sizing)
    - Model accessible via HuggingFace Hub or local path
    - For WandB: WANDB_API_KEY environment variable set

Postconditions:
    - Trained checkpoint saved to output_dir/final/
    - Bundle exported to output_dir/bundle/
    - Evaluation results in output_dir/eval_results.json
    - Full report in output_dir/reproduce_report.json
    - WandB run finalized if --wandb was specified

Complexity: Dominated by training O(epochs * N * S * V) where N=samples, S=seq_len, V=vocab.
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import random
import signal
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Final

import numpy as np
import torch
import typer
import yaml

from tasft.exceptions import TASFTError, TrainingError, ValidationError
from tasft.observability import bind_context, configure_logging, get_logger, init_tracing

app = typer.Typer(
    name="tasft-reproduce",
    help="TASFT: single-command reproducibility pipeline.",
    no_args_is_help=False,
)

logger = get_logger(__name__)

# Canonical hyperparameters from README / paper Table 1.
# These are the defaults when no config file is provided.
_DEFAULT_SEED: Final[int] = 42
_DEFAULT_LAMBDA_GATE: Final[float] = 0.1
_DEFAULT_BETA_SPARSE: Final[float] = 0.01
_DEFAULT_TAU_TARGET: Final[float] = 0.8
_DEFAULT_GATE_LR_RATIO: Final[float] = 0.1
_DEFAULT_GATE_WARMUP_STEPS: Final[int] = 100
_DEFAULT_LAYERS_PER_STEP: Final[int] = 4
_DEFAULT_BLOCK_SIZE: Final[int] = 64
_DEFAULT_LR: Final[float] = 2e-4
_DEFAULT_BATCH_SIZE: Final[int] = 4
_DEFAULT_GRAD_ACCUM: Final[int] = 4
_DEFAULT_EPOCHS: Final[int] = 3

# Dataset name -> (HuggingFace dataset ID, subset, text_column, label_column, template, benchmark, num_layers)
_DATASET_PRESETS: Final[dict[str, dict[str, Any]]] = {
    "medqa": {
        "hf_name": "bigbio/med_qa",
        "subset": None,
        "split_train": "train",
        "split_eval": "validation",
        "text_column": "question",
        "label_column": "answer",
        "template": "Question: {question}\nAnswer: {answer}",
        "benchmark": "medqa_usmle",
        "max_seq_length": 2048,
    },
    "humaneval": {
        "hf_name": "bigcode/starcoderdata",
        "subset": "python",
        "split_train": "train",
        "split_eval": "validation",
        "text_column": "content",
        "label_column": None,
        "template": None,
        "benchmark": "humaneval",
        "max_seq_length": 4096,
        "max_samples_train": 100000,
        "max_samples_eval": 2000,
    },
}

# Model name -> num_layers for automatic gate config
_MODEL_NUM_LAYERS: Final[dict[str, int]] = {
    "meta-llama/Meta-Llama-3-8B": 32,
    "meta-llama/Llama-3.1-8B": 32,
    "meta-llama/Llama-3.2-8B": 32,
    "Qwen/Qwen2.5-7B": 28,
}


@dataclass(frozen=True)
class PhaseResult:
    """Result of a single pipeline phase.

    Attributes:
        phase: Phase name (train, export, eval).
        success: Whether the phase completed without error.
        elapsed_seconds: Wall-clock time for the phase.
        error_message: Error description if success is False.
        data: Phase-specific result data.
    """

    phase: str
    success: bool
    elapsed_seconds: float
    error_message: str | None = None
    data: dict[str, Any] = field(default_factory=dict)


def _set_deterministic_seed(seed: int) -> None:
    """Set all random seeds for bitwise-reproducible training.

    Sets seeds for: Python stdlib random, NumPy, PyTorch CPU, PyTorch CUDA.
    Also configures PyTorch deterministic backends where possible.

    Preconditions: seed >= 0.
    Postconditions: All RNG states seeded. CUBLAS workspace config set for determinism.
    Complexity: O(1).

    Args:
        seed: Non-negative integer seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Enable deterministic algorithms where available.
    # Some operations (e.g. scatter_add) may raise if no deterministic impl exists,
    # so we use warn_only=True to avoid crashing on those edge cases.
    torch.use_deterministic_algorithms(mode=True, warn_only=True)
    # CUBLAS workspace configuration for deterministic matrix multiplications.
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


def _collect_hardware_info() -> dict[str, Any]:
    """Collect hardware and software environment information for the report.

    Gathers: CPU model, core count, RAM, GPU details (if available),
    Python version, PyTorch version, CUDA version, OS details.

    Preconditions: None (gracefully handles missing GPU).
    Postconditions: Returns a dict with all discoverable hardware info.
    Complexity: O(1) plus GPU query overhead.

    Returns:
        Dictionary with hardware/software environment details.
    """
    info: dict[str, Any] = {
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cpu_count": os.cpu_count(),
    }

    # CPU info
    try:
        info["cpu_model"] = platform.processor() or "unknown"
    except Exception:
        info["cpu_model"] = "unknown"

    # Memory info (platform-dependent)
    try:
        import psutil
        mem = psutil.virtual_memory()
        info["ram_total_gb"] = round(mem.total / (1024 ** 3), 1)
        info["ram_available_gb"] = round(mem.available / (1024 ** 3), 1)
    except ImportError:
        info["ram_total_gb"] = "psutil_not_installed"

    # GPU info
    if torch.cuda.is_available():
        info["cuda_version"] = torch.version.cuda
        info["gpu_count"] = torch.cuda.device_count()
        info["gpu_devices"] = []
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            info["gpu_devices"].append({
                "index": i,
                "name": props.name,
                "total_memory_gb": round(props.total_mem / (1024 ** 3), 1),
                "compute_capability": f"{props.major}.{props.minor}",
            })
    else:
        info["cuda_version"] = None
        info["gpu_count"] = 0

    return info


def _build_config_from_args(
    model: str,
    dataset: str,
    output_dir: str,
    seed: int,
    epochs: int,
    learning_rate: float,
    batch_size: int,
    grad_accum: int,
    wandb_enabled: bool,
    wandb_project: str,
    wandb_entity: str | None,
) -> dict[str, Any]:
    """Build a complete TASFT configuration dict from CLI arguments.

    Constructs the same YAML-compatible configuration structure used by
    scripts/train.py, using paper-canonical hyperparameters as defaults.

    Preconditions: dataset must be a key in _DATASET_PRESETS or a HuggingFace dataset ID.
    Postconditions: Returns a fully specified config dict ready for training.
    Complexity: O(1).

    Args:
        model: HuggingFace model ID.
        dataset: Dataset preset name or HuggingFace dataset ID.
        output_dir: Root output directory for all artifacts.
        seed: Random seed for reproducibility.
        epochs: Number of training epochs.
        learning_rate: Peak learning rate for LoRA parameters.
        batch_size: Per-device training batch size.
        grad_accum: Gradient accumulation steps.
        wandb_enabled: Whether to log to WandB.
        wandb_project: WandB project name.
        wandb_entity: WandB entity (team or user).

    Returns:
        Complete configuration dictionary.

    Raises:
        ValidationError: If dataset preset is unknown and no HuggingFace fallback.
    """
    # Resolve dataset preset
    preset = _DATASET_PRESETS.get(dataset)
    if preset is None:
        # Treat dataset as a HuggingFace dataset ID with generic defaults
        preset = {
            "hf_name": dataset,
            "subset": None,
            "split_train": "train",
            "split_eval": "validation",
            "text_column": "text",
            "label_column": None,
            "template": None,
            "benchmark": "hellaswag",
            "max_seq_length": 2048,
        }

    # Resolve model-specific parameters
    num_layers = _MODEL_NUM_LAYERS.get(model, 32)
    # Code datasets benefit from higher LoRA rank
    lora_r = 32 if dataset == "humaneval" else 16
    lora_alpha = lora_r * 2

    run_name = f"tasft-reproduce-{Path(model).name}-{dataset}"

    config: dict[str, Any] = {
        "model": {
            "base_model_id": model,
            "model_name": f"TASFT-{Path(model).name}-{dataset}",
            "torch_dtype": "bfloat16",
            "attn_implementation": "eager",
        },
        "lora": {
            "r": lora_r,
            "alpha": lora_alpha,
            "dropout": 0.05,
            "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
            "bias": "none",
            "task_type": "CAUSAL_LM",
        },
        "gate": {
            "block_size": _DEFAULT_BLOCK_SIZE,
            "gate_hidden_dim": 32,
            "default_threshold": 0.5,
            "num_layers": num_layers,
        },
        "objective": {
            "lambda_gate": _DEFAULT_LAMBDA_GATE,
            "beta_sparse": _DEFAULT_BETA_SPARSE,
            "tau_target": _DEFAULT_TAU_TARGET,
            "label_smoothing": 0.0,
            "gate_lr_ratio": _DEFAULT_GATE_LR_RATIO,
            "gate_warmup_steps": _DEFAULT_GATE_WARMUP_STEPS,
        },
        "layer_rotation": {
            "strategy": "round_robin",
            "layers_per_step": _DEFAULT_LAYERS_PER_STEP,
            "ema_alpha": 0.1,
            "seed": seed,
        },
        "training": {
            "output_dir": str(Path(output_dir) / "checkpoint"),
            "num_train_epochs": epochs,
            "per_device_train_batch_size": batch_size,
            "per_device_eval_batch_size": batch_size * 2,
            "gradient_accumulation_steps": grad_accum,
            "learning_rate": learning_rate,
            "lr_scheduler_type": "cosine",
            "warmup_ratio": 0.05,
            "weight_decay": 0.01,
            "max_grad_norm": 1.0,
            "max_seq_length": preset.get("max_seq_length", 2048),
            "bf16": True,
            "gradient_checkpointing": True,
            "dataloader_num_workers": 4,
            "dataloader_pin_memory": True,
            "seed": seed,
            "logging_steps": 10,
            "eval_steps": 250,
            "save_steps": 500,
            "save_total_limit": 3,
            "eval_strategy": "steps",
            "load_best_model_at_end": True,
            "metric_for_best_model": "eval_loss",
            "greater_is_better": False,
            "report_to": "wandb" if wandb_enabled else "none",
            "run_name": run_name,
        },
        "dataset": {
            "name": preset["hf_name"],
            "subset": preset.get("subset"),
            "split_train": preset["split_train"],
            "split_eval": preset["split_eval"],
            "text_column": preset["text_column"],
            "label_column": preset.get("label_column"),
            "max_samples_train": preset.get("max_samples_train"),
            "max_samples_eval": preset.get("max_samples_eval", 1000),
            "preprocessing": {},
        },
        "evaluation": {
            "task_eval": {
                "benchmark": preset["benchmark"],
                "num_fewshot": 0,
                "batch_size": 16,
            },
            "gate_eval": {
                "compute_sparsity_profile": True,
                "compute_kl_divergence": True,
                "block_size": _DEFAULT_BLOCK_SIZE,
            },
            "throughput_eval": {
                "seq_lengths": [512, 1024, 2048],
                "batch_sizes": [1, 4, 8],
                "num_warmup": 5,
                "num_iterations": 50,
            },
        },
        "bundle": {
            "output_dir": str(Path(output_dir) / "bundle"),
            "include_eval_summary": True,
            "compress": False,
        },
        "observability": {
            "log_level": "INFO",
            "log_json": True,
            "metrics_push_gateway": None,
            "tracing_endpoint": None,
            "wandb": {
                "project": wandb_project,
                "entity": wandb_entity,
                "tags": [Path(model).name, dataset, "sparse-attention", "reproducibility"],
            },
        },
        "domain": dataset,
    }

    # Apply template if dataset has one
    if preset.get("template") is not None:
        config["dataset"]["preprocessing"]["template"] = preset["template"]

    return config


def _load_config_file(config_path: Path) -> dict[str, Any]:
    """Load and validate a YAML configuration file.

    Preconditions: config_path exists and contains valid YAML.
    Postconditions: Returns parsed config dict with all required sections.
    Complexity: O(n) where n = file size.

    Args:
        config_path: Path to the YAML file.

    Returns:
        Parsed configuration dictionary.

    Raises:
        FileNotFoundError: If config_path does not exist.
        ValidationError: If config is missing required sections.
    """
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path) as f:
        config = yaml.safe_load(f)

    if not isinstance(config, dict):
        raise ValidationError(
            f"Config must be a YAML mapping, got {type(config).__name__}",
            context={"path": str(config_path)},
        )

    required_sections = ["model", "lora", "gate", "objective", "training"]
    missing = [s for s in required_sections if s not in config]
    if missing:
        raise ValidationError(
            f"Config missing required sections: {missing}",
            context={"path": str(config_path), "missing": missing},
        )

    return config


def _run_training_phase(
    config: dict[str, Any],
    output_dir: Path,
    dry_run: bool,
) -> PhaseResult:
    """Execute the training phase by invoking scripts/train.py as a subprocess.

    Subprocess isolation ensures: (a) GPU memory is fully released between phases,
    (b) training failures don't corrupt the orchestrator's state, (c) signals are
    forwarded correctly for graceful checkpoint saving.

    Preconditions: scripts/train.py exists and is executable.
    Postconditions: Checkpoint written to output_dir/checkpoint/ on success.
    Complexity: Dominated by training O(epochs * N * S * V).

    Args:
        config: Full TASFT configuration dict.
        output_dir: Root output directory.
        dry_run: If True, validate config without training.

    Returns:
        PhaseResult with success status and timing.
    """
    phase_start = time.perf_counter()

    # Write config to a temporary YAML for the subprocess
    config_path = output_dir / "reproduce_config.yaml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    with open(config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    # Build subprocess command
    scripts_dir = Path(__file__).resolve().parent
    train_script = scripts_dir / "train.py"

    cmd = [
        sys.executable, str(train_script),
        "--config", str(config_path),
        "--output-dir", str(output_dir / "checkpoint"),
        "--skip-export",  # We handle export in a separate phase
    ]
    if dry_run:
        cmd.append("--dry-run")

    logger.info(
        "training_phase_started",
        config_path=str(config_path),
        dry_run=dry_run,
        output_dir=str(output_dir / "checkpoint"),
    )

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=86400,  # 24h timeout for long training runs
        )

        elapsed = time.perf_counter() - phase_start

        if result.returncode != 0:
            error_msg = result.stderr[-2000:] if result.stderr else "No stderr output"
            logger.error(
                "training_phase_failed",
                returncode=result.returncode,
                stderr_tail=error_msg,
            )
            return PhaseResult(
                phase="train",
                success=False,
                elapsed_seconds=round(elapsed, 2),
                error_message=f"Training exited with code {result.returncode}: {error_msg}",
            )

        logger.info("training_phase_completed", elapsed_seconds=round(elapsed, 2))
        return PhaseResult(
            phase="train",
            success=True,
            elapsed_seconds=round(elapsed, 2),
            data={"dry_run": dry_run},
        )

    except subprocess.TimeoutExpired:
        elapsed = time.perf_counter() - phase_start
        return PhaseResult(
            phase="train",
            success=False,
            elapsed_seconds=round(elapsed, 2),
            error_message="Training timed out after 24 hours",
        )
    except OSError as exc:
        elapsed = time.perf_counter() - phase_start
        return PhaseResult(
            phase="train",
            success=False,
            elapsed_seconds=round(elapsed, 2),
            error_message=f"Failed to launch training subprocess: {exc}",
        )


def _run_export_phase(
    config: dict[str, Any],
    output_dir: Path,
) -> PhaseResult:
    """Execute the bundle export phase.

    Invokes scripts/export_bundle.py as a subprocess to package the trained
    checkpoint into a deployment-ready bundle with SHA256 integrity checksums.

    Preconditions: Checkpoint directory exists at output_dir/checkpoint/final/.
    Postconditions: Bundle written to output_dir/bundle/ with manifest.json.
    Complexity: O(model_size) for weight serialization and hashing.

    Args:
        config: Full TASFT configuration dict.
        output_dir: Root output directory.

    Returns:
        PhaseResult with success status and timing.
    """
    phase_start = time.perf_counter()

    checkpoint_dir = output_dir / "checkpoint" / "final"
    bundle_dir = output_dir / "bundle"
    config_path = output_dir / "reproduce_config.yaml"

    if not checkpoint_dir.exists():
        elapsed = time.perf_counter() - phase_start
        return PhaseResult(
            phase="export",
            success=False,
            elapsed_seconds=round(elapsed, 2),
            error_message=f"Checkpoint directory not found: {checkpoint_dir}",
        )

    scripts_dir = Path(__file__).resolve().parent
    export_script = scripts_dir / "export_bundle.py"

    cmd = [
        sys.executable, str(export_script),
        "--checkpoint", str(checkpoint_dir),
        "--output", str(bundle_dir),
    ]
    if config_path.exists():
        cmd.extend(["--config", str(config_path)])

    logger.info(
        "export_phase_started",
        checkpoint_dir=str(checkpoint_dir),
        bundle_dir=str(bundle_dir),
    )

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=3600,  # 1h timeout for export
        )

        elapsed = time.perf_counter() - phase_start

        if result.returncode != 0:
            error_msg = result.stderr[-2000:] if result.stderr else "No stderr output"
            logger.error(
                "export_phase_failed",
                returncode=result.returncode,
                stderr_tail=error_msg,
            )
            return PhaseResult(
                phase="export",
                success=False,
                elapsed_seconds=round(elapsed, 2),
                error_message=f"Export exited with code {result.returncode}: {error_msg}",
            )

        logger.info("export_phase_completed", elapsed_seconds=round(elapsed, 2))
        return PhaseResult(
            phase="export",
            success=True,
            elapsed_seconds=round(elapsed, 2),
            data={"bundle_dir": str(bundle_dir)},
        )

    except subprocess.TimeoutExpired:
        elapsed = time.perf_counter() - phase_start
        return PhaseResult(
            phase="export",
            success=False,
            elapsed_seconds=round(elapsed, 2),
            error_message="Export timed out after 1 hour",
        )
    except OSError as exc:
        elapsed = time.perf_counter() - phase_start
        return PhaseResult(
            phase="export",
            success=False,
            elapsed_seconds=round(elapsed, 2),
            error_message=f"Failed to launch export subprocess: {exc}",
        )


def _run_eval_phase(
    config: dict[str, Any],
    output_dir: Path,
) -> PhaseResult:
    """Execute the evaluation phase.

    Invokes scripts/eval.py as a subprocess to run task accuracy, gate quality,
    and throughput benchmarks on the trained checkpoint or bundle.

    Preconditions: Either checkpoint or bundle exists under output_dir.
    Postconditions: eval_results.json written to output_dir.
    Complexity: O(N * S * V) for task eval + O(bench_configs) for throughput.

    Args:
        config: Full TASFT configuration dict.
        output_dir: Root output directory.

    Returns:
        PhaseResult with success status, timing, and evaluation data.
    """
    phase_start = time.perf_counter()

    # Prefer bundle if it exists, otherwise use checkpoint
    bundle_dir = output_dir / "bundle"
    checkpoint_dir = output_dir / "checkpoint" / "final"
    config_path = output_dir / "reproduce_config.yaml"
    eval_output = output_dir / "eval_results.json"

    scripts_dir = Path(__file__).resolve().parent
    eval_script = scripts_dir / "eval.py"

    cmd = [sys.executable, str(eval_script)]

    if bundle_dir.exists():
        cmd.extend(["--bundle", str(bundle_dir)])
    elif checkpoint_dir.exists():
        cmd.extend(["--checkpoint", str(checkpoint_dir)])
    else:
        elapsed = time.perf_counter() - phase_start
        return PhaseResult(
            phase="eval",
            success=False,
            elapsed_seconds=round(elapsed, 2),
            error_message=f"Neither bundle ({bundle_dir}) nor checkpoint ({checkpoint_dir}) found",
        )

    if config_path.exists():
        cmd.extend(["--config", str(config_path)])
    cmd.extend(["--output", str(eval_output)])

    logger.info("eval_phase_started", eval_output=str(eval_output))

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=7200,  # 2h timeout for eval
        )

        elapsed = time.perf_counter() - phase_start

        if result.returncode != 0:
            error_msg = result.stderr[-2000:] if result.stderr else "No stderr output"
            logger.error(
                "eval_phase_failed",
                returncode=result.returncode,
                stderr_tail=error_msg,
            )
            return PhaseResult(
                phase="eval",
                success=False,
                elapsed_seconds=round(elapsed, 2),
                error_message=f"Eval exited with code {result.returncode}: {error_msg}",
            )

        # Parse eval results if the file was written
        eval_data: dict[str, Any] = {}
        if eval_output.exists():
            with open(eval_output) as f:
                eval_data = json.load(f)

        logger.info("eval_phase_completed", elapsed_seconds=round(elapsed, 2))
        return PhaseResult(
            phase="eval",
            success=True,
            elapsed_seconds=round(elapsed, 2),
            data=eval_data,
        )

    except subprocess.TimeoutExpired:
        elapsed = time.perf_counter() - phase_start
        return PhaseResult(
            phase="eval",
            success=False,
            elapsed_seconds=round(elapsed, 2),
            error_message="Evaluation timed out after 2 hours",
        )
    except OSError as exc:
        elapsed = time.perf_counter() - phase_start
        return PhaseResult(
            phase="eval",
            success=False,
            elapsed_seconds=round(elapsed, 2),
            error_message=f"Failed to launch eval subprocess: {exc}",
        )


def _compute_config_hash(config: dict[str, Any]) -> str:
    """Compute a deterministic SHA-256 hash of the configuration.

    Used as a fingerprint for reproducibility tracking: two runs with
    identical config hashes should produce bitwise-identical results
    (modulo non-deterministic GPU operations).

    Preconditions: config is JSON-serializable.
    Postconditions: Returns a 64-char hex string (SHA-256).
    Complexity: O(n) where n = serialized config size.

    Args:
        config: Configuration dictionary.

    Returns:
        SHA-256 hex digest of the canonicalized JSON config.
    """
    canonical = json.dumps(config, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _build_report(
    config: dict[str, Any],
    phases: list[PhaseResult],
    hardware_info: dict[str, Any],
    total_elapsed: float,
    seed: int,
    dry_run: bool,
) -> dict[str, Any]:
    """Build the final JSON reproducibility report.

    Consolidates all phase results, hardware info, hyperparameters, and timing
    into a single self-contained JSON document suitable for paper appendices
    and reviewer verification.

    Preconditions: All phases have been executed (or skipped).
    Postconditions: Returns a complete report dict.
    Complexity: O(phases).

    Args:
        config: Full configuration used for the run.
        phases: List of PhaseResult from each pipeline phase.
        hardware_info: Hardware/software environment info.
        total_elapsed: Total wall-clock time in seconds.
        seed: Random seed used.
        dry_run: Whether this was a dry run.

    Returns:
        Complete report dictionary.
    """
    # Extract key metrics from eval phase if it succeeded
    eval_phase = next((p for p in phases if p.phase == "eval" and p.success), None)
    eval_data = eval_phase.data if eval_phase is not None else {}
    evaluations = eval_data.get("evaluations", {})

    task_result = evaluations.get("task", {})
    gate_result = evaluations.get("gate", {})
    throughput_result = evaluations.get("throughput", {})

    report: dict[str, Any] = {
        "tasft_version": "0.1.0",
        "report_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "config_hash": _compute_config_hash(config),
        "seed": seed,
        "dry_run": dry_run,
        "model": config.get("model", {}).get("base_model_id", "unknown"),
        "dataset": config.get("dataset", {}).get("name", "unknown"),
        "hyperparameters": {
            "lambda_gate": config.get("objective", {}).get("lambda_gate"),
            "beta_sparse": config.get("objective", {}).get("beta_sparse"),
            "tau_target": config.get("objective", {}).get("tau_target"),
            "gate_lr_ratio": config.get("objective", {}).get("gate_lr_ratio"),
            "gate_warmup_steps": config.get("objective", {}).get("gate_warmup_steps"),
            "layers_per_step": config.get("layer_rotation", {}).get("layers_per_step"),
            "block_size": config.get("gate", {}).get("block_size"),
            "learning_rate": config.get("training", {}).get("learning_rate"),
            "per_device_train_batch_size": config.get("training", {}).get("per_device_train_batch_size"),
            "gradient_accumulation_steps": config.get("training", {}).get("gradient_accumulation_steps"),
            "num_train_epochs": config.get("training", {}).get("num_train_epochs"),
            "bf16": config.get("training", {}).get("bf16"),
            "lora_r": config.get("lora", {}).get("r"),
            "lora_alpha": config.get("lora", {}).get("alpha"),
        },
        "results": {
            "accuracy": task_result.get("accuracy"),
            "accuracy_ci_lower": task_result.get("ci_lower"),
            "accuracy_ci_upper": task_result.get("ci_upper"),
            "benchmark": task_result.get("benchmark"),
            "n_eval_samples": task_result.get("n_samples"),
            "mean_sparsity": gate_result.get("mean_sparsity"),
            "mean_kl_divergence": gate_result.get("mean_kl_divergence"),
            "peak_tokens_per_second": throughput_result.get("peak_tokens_per_second"),
            "speedup_vs_dense": throughput_result.get("speedup_vs_dense"),
        },
        "phases": [
            {
                "name": p.phase,
                "success": p.success,
                "elapsed_seconds": p.elapsed_seconds,
                "error": p.error_message,
            }
            for p in phases
        ],
        "timing": {
            "total_seconds": round(total_elapsed, 2),
            "total_hours": round(total_elapsed / 3600, 2),
            "training_seconds": next(
                (p.elapsed_seconds for p in phases if p.phase == "train"), None
            ),
            "export_seconds": next(
                (p.elapsed_seconds for p in phases if p.phase == "export"), None
            ),
            "eval_seconds": next(
                (p.elapsed_seconds for p in phases if p.phase == "eval"), None
            ),
        },
        "hardware": hardware_info,
    }

    return report


def _format_summary_table(report: dict[str, Any]) -> str:
    """Format the report as a human-readable ASCII summary table.

    Preconditions: report has standard structure from _build_report.
    Postconditions: Returns a multi-line string suitable for terminal output.
    Complexity: O(1).

    Args:
        report: Report dictionary from _build_report.

    Returns:
        Formatted ASCII table string.
    """
    lines: list[str] = []
    sep = "=" * 72

    lines.append("")
    lines.append(sep)
    lines.append("  TASFT Reproducibility Report")
    lines.append(sep)
    lines.append("")

    lines.append(f"  Model:        {report.get('model', 'N/A')}")
    lines.append(f"  Dataset:      {report.get('dataset', 'N/A')}")
    lines.append(f"  Config Hash:  {report.get('config_hash', 'N/A')[:16]}...")
    lines.append(f"  Seed:         {report.get('seed', 'N/A')}")
    lines.append(f"  Dry Run:      {report.get('dry_run', False)}")
    lines.append("")

    # Results section
    results = report.get("results", {})
    lines.append("  --- Results ---")
    accuracy = results.get("accuracy")
    if accuracy is not None:
        ci_lo = results.get("accuracy_ci_lower", 0.0)
        ci_hi = results.get("accuracy_ci_upper", 0.0)
        lines.append(f"  Task Accuracy:    {accuracy:.4f}  [{ci_lo:.4f}, {ci_hi:.4f}]")
    else:
        lines.append("  Task Accuracy:    N/A")

    sparsity = results.get("mean_sparsity")
    if sparsity is not None:
        lines.append(f"  Mean Sparsity:    {sparsity:.4f}")
    else:
        lines.append("  Mean Sparsity:    N/A")

    kl = results.get("mean_kl_divergence")
    if kl is not None:
        lines.append(f"  KL Divergence:    {kl:.6f}")
    else:
        lines.append("  KL Divergence:    N/A")

    throughput = results.get("peak_tokens_per_second")
    if throughput is not None:
        speedup = results.get("speedup_vs_dense", 0.0)
        lines.append(f"  Peak Throughput:  {throughput:.0f} tok/s  ({speedup:.2f}x vs dense)")
    else:
        lines.append("  Peak Throughput:  N/A")

    lines.append("")

    # Phase timing
    lines.append("  --- Phase Timing ---")
    for phase_info in report.get("phases", []):
        status = "OK" if phase_info["success"] else "FAILED"
        elapsed = phase_info["elapsed_seconds"]
        name = phase_info["name"].capitalize()
        error_suffix = ""
        if not phase_info["success"] and phase_info.get("error"):
            # Truncate error message for display
            err_text = phase_info["error"][:80]
            error_suffix = f"  ({err_text})"
        lines.append(f"  {name:<12} {elapsed:>8.1f}s  [{status}]{error_suffix}")

    total = report.get("timing", {})
    lines.append(f"  {'Total':<12} {total.get('total_seconds', 0):>8.1f}s  ({total.get('total_hours', 0):.2f}h)")
    lines.append("")

    # Hardware
    hw = report.get("hardware", {})
    lines.append("  --- Hardware ---")
    lines.append(f"  Platform:     {hw.get('platform', 'N/A')}")
    lines.append(f"  PyTorch:      {hw.get('torch_version', 'N/A')}")
    lines.append(f"  CUDA:         {hw.get('cuda_version', 'N/A')}")
    gpu_count = hw.get("gpu_count", 0)
    if gpu_count > 0:
        devices = hw.get("gpu_devices", [])
        for dev in devices:
            lines.append(
                f"  GPU {dev['index']}:        {dev['name']} "
                f"({dev['total_memory_gb']} GB, SM {dev['compute_capability']})"
            )
    else:
        lines.append("  GPU:          None")
    lines.append("")
    lines.append(sep)
    lines.append("")

    return "\n".join(lines)


@app.command()
def reproduce(
    model: str = typer.Option(
        "meta-llama/Meta-Llama-3-8B",
        "--model",
        "-m",
        help="HuggingFace model ID or local path.",
    ),
    dataset: str = typer.Option(
        "medqa",
        "--dataset",
        "-d",
        help="Dataset preset name (medqa, humaneval) or HuggingFace dataset ID.",
    ),
    output_dir: str = typer.Option(
        "./outputs/reproduce",
        "--output-dir",
        "-o",
        help="Root output directory for all artifacts.",
    ),
    config: Path | None = typer.Option(
        None,
        "--config",
        "-c",
        help="Path to YAML config file. Overrides --model and --dataset.",
        exists=True,
        readable=True,
    ),
    seed: int = typer.Option(
        _DEFAULT_SEED,
        "--seed",
        "-s",
        help="Random seed for deterministic reproducibility.",
    ),
    epochs: int = typer.Option(
        _DEFAULT_EPOCHS,
        "--epochs",
        "-e",
        help="Number of training epochs.",
    ),
    learning_rate: float = typer.Option(
        _DEFAULT_LR,
        "--lr",
        help="Peak learning rate for LoRA parameters.",
    ),
    batch_size: int = typer.Option(
        _DEFAULT_BATCH_SIZE,
        "--batch-size",
        "-b",
        help="Per-device training batch size.",
    ),
    grad_accum: int = typer.Option(
        _DEFAULT_GRAD_ACCUM,
        "--grad-accum",
        help="Gradient accumulation steps.",
    ),
    skip_train: bool = typer.Option(
        False,
        "--skip-train",
        help="Skip training phase (use existing checkpoint).",
    ),
    skip_export: bool = typer.Option(
        False,
        "--skip-export",
        help="Skip bundle export phase.",
    ),
    skip_eval: bool = typer.Option(
        False,
        "--skip-eval",
        help="Skip evaluation phase.",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Validate config and hardware without GPU. No training, export, or eval.",
    ),
    wandb: bool = typer.Option(
        False,
        "--wandb",
        help="Enable WandB logging for training.",
    ),
    wandb_project: str = typer.Option(
        "tasft",
        "--wandb-project",
        help="WandB project name.",
    ),
    wandb_entity: str | None = typer.Option(
        None,
        "--wandb-entity",
        help="WandB entity (team or user).",
    ),
    log_level: str = typer.Option(
        "INFO",
        "--log-level",
        help="Logging level (DEBUG, INFO, WARNING, ERROR).",
    ),
) -> None:
    """Run the complete TASFT reproducibility pipeline: train -> export -> eval -> report."""
    configure_logging(level=log_level)
    pipeline_start = time.perf_counter()

    logger.info(
        "reproduce_pipeline_started",
        model=model,
        dataset=dataset,
        output_dir=output_dir,
        seed=seed,
        dry_run=dry_run,
        wandb_enabled=wandb,
    )

    # 1. Set deterministic seed before anything else
    _set_deterministic_seed(seed)

    # 2. Collect hardware info
    hardware_info = _collect_hardware_info()
    logger.info(
        "hardware_detected",
        gpu_count=hardware_info.get("gpu_count", 0),
        cuda_version=hardware_info.get("cuda_version"),
        torch_version=hardware_info.get("torch_version"),
    )

    # 3. Build or load configuration
    if config is not None:
        cfg = _load_config_file(config)
        logger.info("config_loaded_from_file", config_path=str(config))
        # Override output_dir if specified on CLI
        if output_dir != "./outputs/reproduce":
            cfg.setdefault("training", {})["output_dir"] = str(
                Path(output_dir) / "checkpoint"
            )
            cfg.setdefault("bundle", {})["output_dir"] = str(
                Path(output_dir) / "bundle"
            )
        # Override wandb if specified on CLI
        if wandb:
            cfg.setdefault("training", {})["report_to"] = "wandb"
            cfg.setdefault("observability", {}).setdefault("wandb", {})["project"] = wandb_project
            if wandb_entity is not None:
                cfg["observability"]["wandb"]["entity"] = wandb_entity
    else:
        cfg = _build_config_from_args(
            model=model,
            dataset=dataset,
            output_dir=output_dir,
            seed=seed,
            epochs=epochs,
            learning_rate=learning_rate,
            batch_size=batch_size,
            grad_accum=grad_accum,
            wandb_enabled=wandb,
            wandb_project=wandb_project,
            wandb_entity=wandb_entity,
        )
        logger.info("config_built_from_args")

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # Save the resolved config for auditing
    resolved_config_path = out_path / "reproduce_config.yaml"
    with open(resolved_config_path, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
    logger.info("config_saved", path=str(resolved_config_path))

    config_hash = _compute_config_hash(cfg)
    logger.info("config_hash_computed", hash=config_hash[:16])

    # 4. Dry run: validate and exit
    if dry_run:
        typer.echo("Dry run mode: validating configuration and environment.")
        typer.echo(f"  Model:       {cfg['model']['base_model_id']}")
        typer.echo(f"  Dataset:     {cfg.get('dataset', {}).get('name', 'N/A')}")
        typer.echo(f"  Config hash: {config_hash[:16]}...")
        typer.echo(f"  Seed:        {seed}")
        typer.echo(f"  GPU count:   {hardware_info.get('gpu_count', 0)}")
        typer.echo(f"  CUDA:        {hardware_info.get('cuda_version', 'N/A')}")
        typer.echo(f"  PyTorch:     {hardware_info.get('torch_version', 'N/A')}")

        # Run training dry-run to validate model loading + patching
        train_result = _run_training_phase(cfg, out_path, dry_run=True)

        phases = [train_result]
        total_elapsed = time.perf_counter() - pipeline_start
        report = _build_report(cfg, phases, hardware_info, total_elapsed, seed, dry_run=True)

        report_path = out_path / "reproduce_report.json"
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2, default=str)

        summary = _format_summary_table(report)
        typer.echo(summary)

        if train_result.success:
            typer.echo(f"Dry run passed. Config is valid. Report saved to: {report_path}")
        else:
            typer.echo(f"Dry run FAILED: {train_result.error_message}", err=True)
            raise typer.Exit(code=1)
        return

    # 5. Execute pipeline phases
    phases: list[PhaseResult] = []

    # Phase 1: Training
    if skip_train:
        logger.info("training_phase_skipped")
        phases.append(PhaseResult(
            phase="train",
            success=True,
            elapsed_seconds=0.0,
            data={"skipped": True},
        ))
    else:
        train_result = _run_training_phase(cfg, out_path, dry_run=False)
        phases.append(train_result)
        if not train_result.success:
            logger.error(
                "training_phase_failed_aborting",
                error=train_result.error_message,
            )
            # Still generate report even on failure
            total_elapsed = time.perf_counter() - pipeline_start
            report = _build_report(cfg, phases, hardware_info, total_elapsed, seed, dry_run=False)
            report_path = out_path / "reproduce_report.json"
            with open(report_path, "w") as f:
                json.dump(report, f, indent=2, default=str)
            typer.echo(f"\nTraining failed. Partial report saved to: {report_path}")
            raise typer.Exit(code=1)

    # Phase 2: Bundle export
    if skip_export:
        logger.info("export_phase_skipped")
        phases.append(PhaseResult(
            phase="export",
            success=True,
            elapsed_seconds=0.0,
            data={"skipped": True},
        ))
    else:
        export_result = _run_export_phase(cfg, out_path)
        phases.append(export_result)
        if not export_result.success:
            # Export failure is non-fatal: log warning, continue to eval
            logger.warning(
                "export_phase_failed_continuing",
                error=export_result.error_message,
            )

    # Phase 3: Evaluation
    if skip_eval:
        logger.info("eval_phase_skipped")
        phases.append(PhaseResult(
            phase="eval",
            success=True,
            elapsed_seconds=0.0,
            data={"skipped": True},
        ))
    else:
        eval_result = _run_eval_phase(cfg, out_path)
        phases.append(eval_result)
        if not eval_result.success:
            logger.warning(
                "eval_phase_failed",
                error=eval_result.error_message,
            )

    # 6. Build and save report
    total_elapsed = time.perf_counter() - pipeline_start
    report = _build_report(cfg, phases, hardware_info, total_elapsed, seed, dry_run=False)

    report_path = out_path / "reproduce_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    logger.info("report_saved", path=str(report_path))

    # 7. Print summary
    summary = _format_summary_table(report)
    typer.echo(summary)
    typer.echo(f"Full report saved to: {report_path}")
    typer.echo(f"Config saved to:      {resolved_config_path}")

    # Exit with error if any critical phase failed
    critical_failures = [p for p in phases if not p.success and p.phase == "train"]
    if critical_failures:
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
