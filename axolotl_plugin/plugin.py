"""TASFT Axolotl plugin — integrates co-training into Axolotl's training pipeline.

Hooks into Axolotl's plugin system to:
1. Patch model attention layers with TASFTAttention + AttnGate modules
2. Inject the dual objective (L_task + λ·L_gate) into the training loop
3. Run layer rotation scheduling for memory-efficient gate calibration
4. Export a deployment bundle on training completion

Usage in Axolotl config:
    plugins:
      - tasft

    tasft:
      gate:
        block_size: 64
        num_layers: 32
        default_threshold: 0.5
      objective:
        lambda_gate: 0.1
        beta_sparse: 0.01
        tau_target: 0.7
      layer_rotation:
        strategy: ROUND_ROBIN
        layers_per_step: 4

Preconditions:
    - Axolotl >= 0.4.0
    - TASFT package installed
    - Model architecture supported by patch_model_attention

Postconditions:
    - Model attention layers patched with AttnGate modules
    - Training loop computes dual objective per step
    - Bundle exported to output_dir/tasft_bundle/ on completion
"""
from __future__ import annotations

from typing import Any

import torch
from torch import nn

from tasft.modules.tasft_attention import (
    GateConfig,
    TASFTAttention,
    patch_model_attention,
)
from tasft.observability import bind_context, get_logger
from tasft.observability.metrics import TASFTMetrics
from tasft.training.layer_rotation import LayerRotationScheduler, RotationStrategy
from tasft.training.objectives import TASFTObjective

logger = get_logger(__name__)


class TASFTPlugin:
    """Axolotl plugin for TASFT co-training.

    Implements Axolotl's plugin interface hooks:
    - pre_model_load: configure gate parameters
    - post_model_load: patch attention layers with AttnGate
    - pre_training_step: set active layers for this step
    - compute_loss: inject dual objective
    - post_training_step: update rotation scheduler, record metrics
    - post_training: export bundle

    Invariants:
        - Gate parameters are the ONLY trainable params added by this plugin
        - Base model weights remain frozen throughout
        - Layer rotation ensures every layer is calibrated periodically
    """

    plugin_name = "tasft"

    def __init__(self) -> None:
        self._gate_config: GateConfig | None = None
        self._objective: TASFTObjective | None = None
        self._rotation_scheduler: LayerRotationScheduler | None = None
        self._metrics: TASFTMetrics = TASFTMetrics()
        self._patched_layers: dict[int, TASFTAttention] = {}
        self._active_layers: list[int] = []
        self._current_step: int = 0
        self._plugin_config: dict[str, Any] = {}

    def pre_model_load(self, cfg: dict[str, Any]) -> dict[str, Any]:
        """Called before model loading. Stores plugin config and ensures eager attention.

        Args:
            cfg: Full Axolotl configuration dict.

        Returns:
            Modified configuration dict (forces eager attention for score extraction).
        """
        self._plugin_config = cfg.get("tasft", {})
        gate_cfg = self._plugin_config.get("gate", {})

        self._gate_config = GateConfig(
            block_size=gate_cfg.get("block_size", 64),
            num_layers=gate_cfg.get("num_layers", 32),
            gate_hidden_dim=gate_cfg.get("gate_hidden_dim"),
            default_threshold=gate_cfg.get("default_threshold", 0.5),
        )

        # Force eager attention for full score matrix access during training
        cfg["attn_implementation"] = "eager"

        logger.info(
            "tasft_plugin_pre_model_load",
            gate_block_size=self._gate_config.block_size,
            gate_num_layers=self._gate_config.num_layers,
        )

        return cfg

    def post_model_load(self, model: nn.Module, cfg: dict[str, Any]) -> nn.Module:
        """Called after model loading. Patches attention layers with AttnGate.

        Args:
            model: Loaded HuggingFace model.
            cfg: Full Axolotl configuration dict.

        Returns:
            Modified model with TASFTAttention layers.
        """
        if self._gate_config is None:
            raise RuntimeError("pre_model_load must be called before post_model_load")

        # Patch attention layers
        self._patched_layers = patch_model_attention(model, self._gate_config)

        # Initialize objective
        obj_cfg = self._plugin_config.get("objective", {})
        self._objective = TASFTObjective(
            lambda_gate=obj_cfg.get("lambda_gate", 0.1),
            beta_sparse=obj_cfg.get("beta_sparse", 0.01),
            tau_target=obj_cfg.get("tau_target", 0.8),
            label_smoothing=obj_cfg.get("label_smoothing", 0.0),
        )

        # Initialize rotation scheduler
        rot_cfg = self._plugin_config.get("layer_rotation", {})
        strategy_name = rot_cfg.get("strategy", "ROUND_ROBIN")
        strategy = RotationStrategy[strategy_name]

        self._rotation_scheduler = LayerRotationScheduler(
            num_layers=self._gate_config.num_layers,
            layers_per_step=rot_cfg.get("layers_per_step", 4),
            strategy=strategy,
            ema_alpha=rot_cfg.get("ema_alpha", 0.1),
            seed=rot_cfg.get("seed", 42),
        )

        gate_param_count = sum(
            tasft_attn.gate.num_parameters for tasft_attn in self._patched_layers.values()
        )
        total_param_count = sum(p.numel() for p in model.parameters())
        gate_fraction = gate_param_count / total_param_count if total_param_count > 0 else 0.0

        logger.info(
            "tasft_plugin_model_patched",
            num_patched_layers=len(self._patched_layers),
            gate_params=gate_param_count,
            gate_fraction_percent=round(gate_fraction * 100, 3),
            total_model_params=total_param_count,
        )

        return model

    def pre_training_step(self, step: int, **kwargs: Any) -> None:
        """Called before each training step. Selects active layers for gate calibration.

        Args:
            step: Current training step number.
        """
        if self._rotation_scheduler is None:
            return

        self._current_step = step
        self._active_layers = [int(li) for li in self._rotation_scheduler.get_active_layers()]

        # Enable gate target computation only for active layers
        for layer_idx, tasft_attn in self._patched_layers.items():
            is_active = layer_idx in self._active_layers
            tasft_attn.set_training_mode(compute_gate_target=is_active)

        self._metrics.set_active_layers(len(self._active_layers))

    def compute_loss(
        self,
        model: nn.Module,
        inputs: dict[str, torch.Tensor],
        outputs: Any,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Compute the TASFT dual objective loss.

        Replaces Axolotl's default loss computation with the dual objective:
        L_total = L_task + λ · (L_gate + β · L_sparse)

        Args:
            model: The model being trained.
            inputs: Tokenized input batch with input_ids and labels.
            outputs: Model forward pass output (CausalLMOutput).

        Returns:
            Scalar loss tensor for backpropagation.
        """
        if self._objective is None:
            # Fallback: return standard loss if plugin not fully initialized
            return outputs.loss

        logits = outputs.logits
        labels = inputs["labels"]

        # Collect gate outputs and attention scores from active layers
        gate_outputs_by_layer: dict[int, torch.Tensor] = {}
        attn_scores_by_layer: dict[int, torch.Tensor] = {}

        for layer_idx in self._active_layers:
            tasft_attn = self._patched_layers.get(layer_idx)
            if tasft_attn is None:
                continue

            # Access cached outputs from the forward pass
            # TASFTAttention stores these as instance attributes during forward
            last_gate = getattr(tasft_attn, "_last_gate_output", None)
            last_attn = getattr(tasft_attn, "_last_attn_weights", None)

            if last_gate is not None:
                gate_outputs_by_layer[layer_idx] = last_gate.soft_scores

            if last_attn is not None:
                attn_scores_by_layer[layer_idx] = last_attn

        # Compute dual objective
        block_size = self._gate_config.block_size if self._gate_config is not None else 64

        with bind_context(step=self._current_step):
            loss_output = self._objective.compute(
                logits=logits,
                labels=labels,
                gate_outputs_by_layer=gate_outputs_by_layer,
                attn_scores_by_layer=attn_scores_by_layer,
                active_layer_indices=self._active_layers,
                block_size=block_size,
            )

        # Report per-layer gate losses to rotation scheduler for priority weighting
        if self._rotation_scheduler is not None:
            for layer_idx_key, gate_loss_val in loss_output.per_layer_gate_loss.items():
                self._rotation_scheduler.report_gate_loss(int(layer_idx_key), gate_loss_val)

        return loss_output.total

    def post_training_step(self, step: int, loss: float, **kwargs: Any) -> None:
        """Called after each training step. Records metrics.

        Args:
            step: Current training step number.
            loss: Loss value for this step.
        """
        self._metrics.training_steps_total.inc()

        # Record sparsity per active layer
        for layer_idx in self._active_layers:
            tasft_attn = self._patched_layers.get(layer_idx)
            if tasft_attn is None:
                continue

            last_gate = getattr(tasft_attn, "_last_gate_output", None)
            if last_gate is not None:
                self._metrics.record_sparsity(
                    layer=layer_idx,
                    ratio=float(last_gate.sparsity_ratio),
                )

        # Log periodically
        if step % 10 == 0:
            coverage = self._rotation_scheduler.get_coverage_stats() if self._rotation_scheduler else None
            logger.info(
                "training_step_completed",
                step=step,
                loss=round(loss, 6),
                active_layers=self._active_layers,
                coverage_fully_covered=coverage.fully_covered if coverage else None,
            )

    def post_training(self, model: nn.Module, cfg: dict[str, Any]) -> None:
        """Called after training completes. Exports deployment bundle.

        Args:
            model: The trained model.
            cfg: Full Axolotl configuration dict.
        """
        output_dir = cfg.get("output_dir", "./output")
        bundle_dir = f"{output_dir}/tasft_bundle"

        logger.info("tasft_plugin_post_training", bundle_dir=bundle_dir)

        try:
            from tasft.bundle.export import BundleExporter, ExportConfig

            model_cfg = cfg.get("model", {})
            gate_cfg = self._plugin_config.get("gate", {})
            export_config = ExportConfig(
                model_name=model_cfg.get("model_name", model_cfg.get("base_model_id", "unknown")),
                base_model_id=model_cfg.get("base_model_id", "unknown"),
                domain=cfg.get("domain", "general"),
                block_size=gate_cfg.get("block_size", 64),
                global_threshold=gate_cfg.get("default_threshold", 0.5),
            )
            exporter = BundleExporter(config=export_config)
            bundle_path = exporter.export(
                model=model,
                output_dir=bundle_dir,
            )
            metadata = BundleExporter.load_bundle_metadata(bundle_path)

            logger.info(
                "tasft_bundle_exported",
                bundle_dir=str(bundle_path),
                num_files=len(metadata.manifest.checksums),
                total_size_bytes=metadata.manifest.total_size_bytes,
            )
        except Exception:
            logger.exception("tasft_bundle_export_failed")

    def get_trainer_cls(self) -> type:
        """Return TASFTTrainer as the trainer class for Axolotl.

        Axolotl calls this to override the default HF Trainer with our
        co-training subclass that handles dual objectives and layer rotation.

        Returns:
            TASFTTrainer class.
        """
        from tasft.training.trainer import TASFTTrainer

        return TASFTTrainer

    def get_trainable_parameters(self, model: nn.Module) -> list[nn.Parameter]:
        """Return only the gate parameters that should be trained by this plugin.

        Axolotl uses this to add plugin-specific parameters to the optimizer.
        Gate parameters use a separate learning rate from LoRA parameters.

        Args:
            model: The patched model.

        Returns:
            List of gate nn.Parameter objects.
        """
        gate_params: list[nn.Parameter] = []
        for tasft_attn in self._patched_layers.values():
            gate_params.extend(p for p in tasft_attn.gate.parameters() if p.requires_grad)
        return gate_params


def get_plugin() -> TASFTPlugin:
    """Axolotl plugin factory function.

    Axolotl calls this to instantiate the plugin.

    Returns:
        Configured TASFTPlugin instance.
    """
    return TASFTPlugin()


__all__ = ["TASFTPlugin", "get_plugin"]
