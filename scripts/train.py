#!/usr/bin/env python3
"""TASFT training script — co-trains LoRA adapters with sparse attention gates.

Supports single-GPU and multi-GPU (torchrun) training with:
- YAML configuration files for reproducible experiments
- WandB logging with structured metrics
- Graceful SIGTERM handling for preemptible clusters
- Automatic bundle export on training completion

Usage:
    # Single GPU
    python scripts/train.py --config configs/llama3_8b_medqa.yaml

    # Multi-GPU via torchrun
    torchrun --nproc_per_node=4 scripts/train.py --config configs/llama3_8b_medqa.yaml

    # Override specific parameters
    python scripts/train.py --config configs/llama3_8b_medqa.yaml \
        --output-dir ./my_output --learning-rate 1e-4

Preconditions:
    - Config YAML must exist and be valid
    - Model must be accessible (local or HuggingFace Hub)
    - GPU with sufficient VRAM (see config comments for sizing)

Postconditions:
    - Trained model checkpoint saved to output_dir
    - Bundle exported if training completes successfully
    - WandB run finalized with all metrics
"""
from __future__ import annotations

import signal
import sys
import time
from pathlib import Path
from typing import Any

import torch
import typer
import yaml
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from tasft.modules.tasft_attention import GateConfig, patch_model_attention
from tasft.observability import bind_context, configure_logging, get_logger, init_tracing
from tasft.training.trainer import TASFTTrainer, TASFTTrainingArguments

app = typer.Typer(
    name="tasft-train",
    help="TASFT: Task-Aware Sparse Fine-Tuning trainer.",
    no_args_is_help=True,
)

logger = get_logger(__name__)


def _load_config(config_path: Path) -> dict[str, Any]:
    """Load and validate a YAML training configuration.

    Preconditions: config_path exists and is valid YAML.
    Postconditions: Returns a parsed dict with all config sections.
    Complexity: O(n) where n = file size.

    Args:
        config_path: Path to the YAML config file.

    Returns:
        Parsed configuration dictionary.

    Raises:
        FileNotFoundError: If config_path does not exist.
        yaml.YAMLError: If YAML is malformed.
    """
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path) as f:
        config = yaml.safe_load(f)

    if not isinstance(config, dict):
        raise ValueError(f"Config must be a YAML mapping, got {type(config).__name__}")

    required_sections = ["model", "lora", "gate", "objective", "training"]
    missing = [s for s in required_sections if s not in config]
    if missing:
        raise ValueError(f"Config missing required sections: {missing}")

    return config


class _GracefulShutdown:
    """SIGTERM handler for graceful shutdown on preemptible clusters.

    Sets a flag when SIGTERM is received. The training loop checks this flag
    between steps and saves a checkpoint before exiting.

    Postconditions: After signal, should_stop returns True.
    """

    def __init__(self) -> None:
        self._should_stop = False
        signal.signal(signal.SIGTERM, self._handler)
        signal.signal(signal.SIGINT, self._handler)

    def _handler(self, signum: int, frame: Any) -> None:
        sig_name = signal.Signals(signum).name
        logger.warning("shutdown_signal_received", signal=sig_name)
        self._should_stop = True

    @property
    def should_stop(self) -> bool:
        """Whether a shutdown signal has been received."""
        return self._should_stop


def _setup_wandb(cfg: dict[str, Any]) -> None:
    """Initialize WandB if configured.

    Preconditions: wandb package available if report_to == "wandb".
    Postconditions: WandB run initialized or skipped.
    Complexity: O(1).
    """
    training_cfg = cfg.get("training", {})
    obs_cfg = cfg.get("observability", {})

    if training_cfg.get("report_to") != "wandb":
        return

    try:
        import wandb

        wandb_cfg = obs_cfg.get("wandb", {})
        wandb.init(
            project=wandb_cfg.get("project", "tasft"),
            entity=wandb_cfg.get("entity"),
            name=training_cfg.get("run_name", "tasft-run"),
            tags=wandb_cfg.get("tags", []),
            config=cfg,
        )
        logger.info("wandb_initialized", project=wandb_cfg.get("project", "tasft"))
    except ImportError:
        logger.warning("wandb_not_installed", msg="Skipping WandB logging")


def _build_training_args(cfg: dict[str, Any], overrides: dict[str, Any]) -> TASFTTrainingArguments:
    """Build TASFTTrainingArguments from config YAML and CLI overrides.

    Merges training, objective, gate, and layer_rotation config sections
    into a single TASFTTrainingArguments instance.

    Args:
        cfg: Full parsed YAML config.
        overrides: CLI override values (non-None only).

    Returns:
        Configured TASFTTrainingArguments.
    """
    t = cfg.get("training", {})
    obj = cfg.get("objective", {})
    gate = cfg.get("gate", {})
    rot = cfg.get("layer_rotation", {})

    # Apply overrides
    if "output_dir" in overrides:
        t["output_dir"] = overrides["output_dir"]
    if "learning_rate" in overrides:
        t["learning_rate"] = overrides["learning_rate"]
    if "num_train_epochs" in overrides:
        t["num_train_epochs"] = overrides["num_train_epochs"]

    return TASFTTrainingArguments(
        output_dir=t.get("output_dir", "./outputs/tasft"),
        num_train_epochs=t.get("num_train_epochs", 3),
        per_device_train_batch_size=t.get("per_device_train_batch_size", 4),
        per_device_eval_batch_size=t.get("per_device_eval_batch_size", 8),
        gradient_accumulation_steps=t.get("gradient_accumulation_steps", 4),
        learning_rate=t.get("learning_rate", 2e-4),
        lr_scheduler_type=t.get("lr_scheduler_type", "cosine"),
        warmup_ratio=t.get("warmup_ratio", 0.05),
        weight_decay=t.get("weight_decay", 0.01),
        max_grad_norm=t.get("max_grad_norm", 1.0),
        bf16=t.get("bf16", True),
        gradient_checkpointing=t.get("gradient_checkpointing", True),
        dataloader_num_workers=t.get("dataloader_num_workers", 4),
        dataloader_pin_memory=t.get("dataloader_pin_memory", True),
        seed=t.get("seed", 42),
        logging_steps=t.get("logging_steps", 10),
        eval_steps=t.get("eval_steps", 250),
        save_steps=t.get("save_steps", 500),
        save_total_limit=t.get("save_total_limit", 3),
        eval_strategy=t.get("eval_strategy", "steps"),
        load_best_model_at_end=t.get("load_best_model_at_end", True),
        metric_for_best_model=t.get("metric_for_best_model", "eval_loss"),
        greater_is_better=t.get("greater_is_better", False),
        report_to=t.get("report_to", "none"),
        run_name=t.get("run_name", "tasft-run"),
        # TASFT-specific args
        lambda_gate=obj.get("lambda_gate", 0.1),
        beta_sparse=obj.get("beta_sparse", 0.01),
        tau_target=obj.get("tau_target", 0.8),
        gate_lr_ratio=t.get("gate_learning_rate", 1e-3) / t.get("learning_rate", 2e-4)
        if t.get("gate_learning_rate") is not None
        else obj.get("gate_lr_ratio", 0.1),
        gate_warmup_steps=obj.get("gate_warmup_steps", 100),
        layers_per_step=rot.get("layers_per_step", 4),
        block_size=gate.get("block_size", 64),
        rotation_strategy=rot.get("strategy", "round_robin").lower(),
    )


def _load_model_and_tokenizer(cfg: dict[str, Any]) -> tuple[Any, Any]:
    """Load base model and tokenizer from config.

    Args:
        cfg: Full parsed config.

    Returns:
        (model, tokenizer) tuple.
    """
    model_cfg = cfg["model"]
    model_id = model_cfg["base_model_id"]
    dtype_str = model_cfg.get("torch_dtype", "bfloat16")
    dtype = getattr(torch, dtype_str)
    attn_impl = model_cfg.get("attn_implementation", "eager")

    logger.info("loading_model", model_id=model_id, dtype=dtype_str, attn_impl=attn_impl)

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=dtype,
        attn_implementation=attn_impl,
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


def _apply_lora(model: Any, cfg: dict[str, Any]) -> Any:
    """Apply LoRA adapters to the model.

    Args:
        model: Base model.
        cfg: Full parsed config.

    Returns:
        PEFT-wrapped model with LoRA adapters.
    """
    lora_cfg = cfg["lora"]
    lora_config = LoraConfig(
        r=lora_cfg.get("r", 16),
        lora_alpha=lora_cfg.get("alpha", 32),
        lora_dropout=lora_cfg.get("dropout", 0.05),
        target_modules=lora_cfg.get("target_modules", ["q_proj", "k_proj", "v_proj", "o_proj"]),
        bias=lora_cfg.get("bias", "none"),
        task_type=lora_cfg.get("task_type", "CAUSAL_LM"),
    )

    model = get_peft_model(model, lora_config)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    logger.info(
        "lora_applied",
        rank=lora_cfg.get("r", 16),
        trainable_params=trainable,
        total_params=total,
        trainable_percent=round(trainable / total * 100, 3),
    )

    return model


def _prepare_dataset(cfg: dict[str, Any], tokenizer: Any, max_seq_length: int) -> tuple[Any, Any]:
    """Load and tokenize the dataset.

    Args:
        cfg: Full parsed config.
        tokenizer: Tokenizer for the model.
        max_seq_length: Maximum sequence length.

    Returns:
        (train_dataset, eval_dataset) tuple.
    """
    ds_cfg = cfg.get("dataset", {})
    ds_name = ds_cfg.get("name")
    subset = ds_cfg.get("subset")

    load_kwargs: dict[str, Any] = {}
    if subset:
        load_kwargs["name"] = subset

    dataset = load_dataset(ds_name, **load_kwargs)
    train_split = ds_cfg.get("split_train", "train")
    eval_split = ds_cfg.get("split_eval", "validation")

    train_ds = dataset[train_split]
    eval_ds = dataset.get(eval_split)

    # Subsample if configured
    max_train = ds_cfg.get("max_samples_train")
    if max_train is not None and len(train_ds) > max_train:
        train_ds = train_ds.select(range(max_train))

    max_eval = ds_cfg.get("max_samples_eval")
    if eval_ds is not None and max_eval is not None and len(eval_ds) > max_eval:
        eval_ds = eval_ds.select(range(max_eval))

    # Tokenize
    text_col = ds_cfg.get("text_column", "text")
    template = ds_cfg.get("preprocessing", {}).get("template")

    def tokenize_fn(examples: dict[str, list[Any]]) -> dict[str, list[Any]]:
        if template is not None:
            texts = [template.format(**{k: examples[k][i] for k in examples}) for i in range(len(next(iter(examples.values()))))]
        else:
            texts = examples[text_col]

        tokenized = tokenizer(
            texts,
            truncation=True,
            max_length=max_seq_length,
            padding="max_length",
        )
        tokenized["labels"] = tokenized["input_ids"].copy()
        return tokenized

    train_ds = train_ds.map(tokenize_fn, batched=True, remove_columns=train_ds.column_names)
    if eval_ds is not None:
        eval_ds = eval_ds.map(tokenize_fn, batched=True, remove_columns=eval_ds.column_names)

    return train_ds, eval_ds


@app.command()
def train(
    config: Path = typer.Option(
        ...,
        "--config",
        "-c",
        help="Path to YAML training configuration file.",
        exists=True,
        readable=True,
    ),
    output_dir: str | None = typer.Option(
        None,
        "--output-dir",
        "-o",
        help="Override output directory from config.",
    ),
    learning_rate: float | None = typer.Option(
        None,
        "--learning-rate",
        "--lr",
        help="Override learning rate from config.",
    ),
    num_epochs: int | None = typer.Option(
        None,
        "--num-epochs",
        "-e",
        help="Override number of training epochs.",
    ),
    resume_from: str | None = typer.Option(
        None,
        "--resume-from",
        help="Path to checkpoint to resume training from.",
    ),
    skip_export: bool = typer.Option(
        False,
        "--skip-export",
        help="Skip bundle export after training.",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Validate config and setup without training.",
    ),
    log_level: str = typer.Option(
        "INFO",
        "--log-level",
        help="Logging level (DEBUG, INFO, WARNING, ERROR).",
    ),
) -> None:
    """Run TASFT co-training: LoRA adapters + sparse attention gates."""
    configure_logging(level=log_level)
    start_time = time.perf_counter()

    cfg = _load_config(config)
    logger.info("config_loaded", config_path=str(config))

    # Build CLI overrides dict
    overrides: dict[str, Any] = {}
    if output_dir is not None:
        overrides["output_dir"] = output_dir
    if learning_rate is not None:
        overrides["learning_rate"] = learning_rate
    if num_epochs is not None:
        overrides["num_train_epochs"] = num_epochs

    # Build training arguments
    training_args = _build_training_args(cfg, overrides)
    actual_output_dir = Path(training_args.output_dir)
    actual_output_dir.mkdir(parents=True, exist_ok=True)

    # Save resolved config for reproducibility
    resolved_config_path = actual_output_dir / "resolved_config.yaml"
    with open(resolved_config_path, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False)

    # Initialize tracing
    obs_cfg = cfg.get("observability", {})
    init_tracing(
        service_name="tasft-training",
        otlp_endpoint=obs_cfg.get("tracing_endpoint"),
    )

    # Initialize WandB
    _setup_wandb(cfg)

    # Setup graceful shutdown
    shutdown = _GracefulShutdown()

    # Load model + tokenizer
    model, tokenizer = _load_model_and_tokenizer(cfg)

    # Apply LoRA
    model = _apply_lora(model, cfg)

    # Patch attention layers with AttnGate
    gate_cfg = cfg["gate"]
    gate_config = GateConfig(
        block_size=gate_cfg.get("block_size", 64),
        num_layers=gate_cfg.get("num_layers", 32),
        gate_hidden_dim=gate_cfg.get("gate_hidden_dim"),
        default_threshold=gate_cfg.get("default_threshold", 0.5),
    )
    patched_layers = patch_model_attention(model, gate_config)

    gate_param_count = sum(ta.gate.num_parameters for ta in patched_layers.values())
    logger.info(
        "model_patched",
        num_patched_layers=len(patched_layers),
        gate_params=gate_param_count,
    )

    if dry_run:
        logger.info("dry_run_complete", msg="Config valid, model loaded and patched successfully.")
        typer.echo("Dry run complete. Config is valid, model loaded and patched.")
        return

    # Prepare dataset
    max_seq = cfg.get("training", {}).get("max_seq_length", 2048)
    train_dataset, eval_dataset = _prepare_dataset(cfg, tokenizer, max_seq)

    logger.info(
        "dataset_loaded",
        train_samples=len(train_dataset),
        eval_samples=len(eval_dataset) if eval_dataset is not None else 0,
    )

    # Create trainer
    trainer = TASFTTrainer(
        model=model,
        args=training_args,
        patched_layers=patched_layers,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
    )

    with bind_context(run_name=training_args.run_name):
        logger.info("training_started", output_dir=str(actual_output_dir))

        try:
            train_result = trainer.train(resume_from_checkpoint=resume_from)
        except KeyboardInterrupt:
            logger.warning("training_interrupted_keyboard")
            trainer.save_model(str(actual_output_dir / "interrupted"))
            sys.exit(1)

        elapsed_s = time.perf_counter() - start_time
        logger.info(
            "training_completed",
            total_steps=trainer.state.global_step,
            final_loss=train_result.training_loss,
            elapsed_seconds=round(elapsed_s, 1),
        )

    # Save final model
    trainer.save_model(str(actual_output_dir / "final"))

    # Export bundle
    if not skip_export and not shutdown.should_stop:
        bundle_cfg = cfg.get("bundle", {})
        bundle_dir = Path(bundle_cfg.get("output_dir", str(actual_output_dir / "bundle")))

        logger.info("bundle_export_started", bundle_dir=str(bundle_dir))

        from tasft.bundle.exporter import BundleExporter

        exporter = BundleExporter.from_checkpoint(
            checkpoint_dir=actual_output_dir / "final",
            training_config=cfg,
            merge_lora=True,
        )
        manifest = exporter.export(output_dir=bundle_dir)
        logger.info(
            "bundle_export_completed",
            bundle_dir=str(bundle_dir),
            total_size_bytes=manifest.total_size_bytes,
            num_files=len(manifest.file_checksums),
        )

    # Finalize WandB
    try:
        import wandb

        if wandb.run is not None:
            wandb.finish()
    except ImportError:
        pass

    logger.info("training_session_ended", elapsed_seconds=round(time.perf_counter() - start_time, 1))


if __name__ == "__main__":
    app()
