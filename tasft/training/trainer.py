"""
TASFT Trainer: HuggingFace Trainer subclass for co-training LoRA + AttnGate.

Implements the full TASFT co-training loop:
    1. Forward pass through LoRA-augmented model → logits + attn scores (for active layers)
    2. Gate forward on active layers → predicted block importance
    3. Dual loss computation: L_total = L_task + λ · (L_gate + β · L_sparse)
    4. Backward + optimizer step with dual LR (base_lr for LoRA, gate_lr for gates)

Key features:
    - Layer rotation: only N layers per step retain full [B,H,S,S] attention scores
    - Gate warmup: gate LR held at 0 for first K steps, then linearly ramps up
    - Dual parameter groups: separate LR for LoRA adapters vs gate parameters
    - 3-artifact checkpointing: LoRA weights + gate weights + sparsity profile JSON
    - Structured logging via structlog at every training step

Preconditions:
    - Model must be patched with patch_model_attention() before trainer creation
    - PEFT/LoRA must be applied before trainer creation
    - TASFTTrainingArguments must pass validation

Postconditions:
    - Checkpoints contain: LoRA adapter, gate state_dict, sparsity_profile.json
    - All training metrics are logged structurally per step

Complexity: O(L_active · B · H · S² + B · S · V) per training step.
"""
from __future__ import annotations

import json
import os
import time
import warnings
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import torch
from torch import nn
from torch.optim.lr_scheduler import LambdaLR
from transformers import Trainer, TrainingArguments

from tasft.exceptions import NaNDetectedError, TrainingError
from tasft.observability.logging import get_logger
from tasft.observability.metrics import TASFTMetrics
from tasft.observability.tracing import trace_training_step
from tasft.training.layer_rotation import (
    LayerRotationScheduler,
    RotationStrategy,
)
from tasft.training.objectives import ObjectiveLossOutput, TASFTObjective
from tasft.types import LayerIndex, SparsityProfile, SparsityRatio

if TYPE_CHECKING:
    from transformers.modeling_utils import PreTrainedModel

    from tasft.modules.tasft_attention import TASFTAttention

logger = get_logger(__name__)

_ROTATION_STRATEGY_MAP: dict[str, RotationStrategy] = {
    "round_robin": RotationStrategy.ROUND_ROBIN,
    "random": RotationStrategy.RANDOM,
    "priority_weighted": RotationStrategy.PRIORITY_WEIGHTED,
}


@dataclass
class TASFTTrainingArguments(TrainingArguments):
    """Extended training arguments for TASFT co-training.

    Adds TASFT-specific hyperparameters for gate calibration, sparsity
    regularization, and layer rotation scheduling on top of the standard
    HuggingFace TrainingArguments.

    Attributes:
        lambda_gate: Weight for gate distillation loss in composite objective.
        beta_sparse: Weight for sparsity regularization.
        tau_target: Target sparsity ratio for gate regularization.
        gate_lr_ratio: Gate LR = base_lr * gate_lr_ratio.
        gate_warmup_steps: Steps before gate loss is activated.
        layers_per_step: Layers to calibrate per step (layer rotation).
        block_size: Attention block size for gate.
        rotation_strategy: Layer rotation strategy name.
    """

    lambda_gate: float = field(
        default=0.1,
        metadata={"help": "Weight for gate distillation loss"},
    )
    beta_sparse: float = field(
        default=0.01,
        metadata={"help": "Weight for sparsity regularization"},
    )
    tau_target: float = field(
        default=0.8,
        metadata={"help": "Target sparsity ratio for gate regularization"},
    )
    gate_lr_ratio: float = field(
        default=0.1,
        metadata={"help": "Gate LR = base_lr * gate_lr_ratio"},
    )
    gate_warmup_steps: int = field(
        default=100,
        metadata={"help": "Steps before gate loss is activated"},
    )
    layers_per_step: int = field(
        default=4,
        metadata={"help": "Layers to calibrate per step (layer rotation)"},
    )
    block_size: int = field(
        default=64,
        metadata={"help": "Attention block size for gate"},
    )
    rotation_strategy: str = field(
        default="round_robin",
        metadata={"help": "Layer rotation strategy: round_robin, random, priority_weighted"},
    )

    def __post_init__(self) -> None:
        """Validate TASFT-specific arguments after HF validation."""
        super().__post_init__()
        if not 0.0 < self.lambda_gate <= 10.0:
            msg = f"lambda_gate must be in (0, 10], got {self.lambda_gate}"
            raise ValueError(msg)
        if not 0.0 < self.tau_target < 1.0:
            msg = f"tau_target must be in (0, 1), got {self.tau_target}"
            raise ValueError(msg)
        if not 0.0 < self.gate_lr_ratio <= 1.0:
            msg = f"gate_lr_ratio must be in (0, 1], got {self.gate_lr_ratio}"
            raise ValueError(msg)
        if self.beta_sparse < 0.0:
            msg = f"beta_sparse must be >= 0, got {self.beta_sparse}"
            raise ValueError(msg)
        if self.gate_warmup_steps < 0:
            msg = f"gate_warmup_steps must be >= 0, got {self.gate_warmup_steps}"
            raise ValueError(msg)
        if self.layers_per_step <= 0:
            msg = f"layers_per_step must be > 0, got {self.layers_per_step}"
            raise ValueError(msg)
        if self.block_size <= 0:
            msg = f"block_size must be > 0, got {self.block_size}"
            raise ValueError(msg)
        if self.rotation_strategy not in _ROTATION_STRATEGY_MAP:
            msg = (
                f"rotation_strategy must be one of {list(_ROTATION_STRATEGY_MAP.keys())}, "
                f"got '{self.rotation_strategy}'"
            )
            raise ValueError(
                msg,
            )


class TASFTTrainer(Trainer):
    """HuggingFace Trainer subclass implementing TASFT co-training.

    Extends the standard Trainer with:
    - Dual training objective (task + gate + sparsity losses)
    - Layer rotation for memory-efficient gate calibration
    - Dual LR parameter groups (LoRA vs gate parameters)
    - Gate warmup scheduling
    - 3-artifact checkpointing (LoRA + gates + sparsity profile)
    - Structured logging at every training step

    Preconditions:
        - Model must be patched with patch_model_attention()
        - args must be TASFTTrainingArguments
        - patched_layers dict must map layer_idx -> TASFTAttention

    Postconditions:
        - Only gate and LoRA parameters receive gradients
        - Checkpoints include sparsity_profile.json
        - All steps are logged with structured metrics

    Complexity: O(L_active · B · H · S² + B · S · V) per training step.
    """

    def __init__(
        self,
        model: PreTrainedModel | nn.Module,
        args: TASFTTrainingArguments,
        patched_layers: dict[int, TASFTAttention],
        **kwargs: Any,
    ) -> None:
        """Initialize TASFTTrainer.

        Args:
            model: The patched model (LoRA + AttnGate applied).
            args: TASFTTrainingArguments with co-training hyperparameters.
            patched_layers: Mapping from layer index to TASFTAttention wrapper.
            **kwargs: Additional arguments forwarded to HF Trainer.
        """
        super().__init__(model=model, args=args, **kwargs)

        self._tasft_args: TASFTTrainingArguments = args
        self._patched_layers: dict[int, TASFTAttention] = patched_layers
        self._num_model_layers: int = len(patched_layers)

        # Objective function
        self._objective = TASFTObjective(
            lambda_gate=args.lambda_gate,
            beta_sparse=args.beta_sparse,
            tau_target=args.tau_target,
            label_smoothing=getattr(args, "label_smoothing_factor", 0.0),
        )

        # Layer rotation scheduler
        strategy = _ROTATION_STRATEGY_MAP[args.rotation_strategy]
        self._rotation_scheduler = LayerRotationScheduler(
            num_layers=self._num_model_layers,
            layers_per_step=min(args.layers_per_step, self._num_model_layers),
            strategy=strategy,
        )

        # Gate warmup LR scheduler (created in create_optimizer)
        self._gate_lr_scheduler: LambdaLR | None = None

        # Observability: Prometheus metrics registry
        self._metrics = TASFTMetrics()

        logger.info(
            "tasft_trainer_init",
            num_layers=self._num_model_layers,
            layers_per_step=args.layers_per_step,
            rotation_strategy=args.rotation_strategy,
            lambda_gate=args.lambda_gate,
            beta_sparse=args.beta_sparse,
            tau_target=args.tau_target,
            gate_lr_ratio=args.gate_lr_ratio,
            gate_warmup_steps=args.gate_warmup_steps,
            block_size=args.block_size,
        )

    def _collect_gate_parameters(self) -> list[nn.Parameter]:
        """Collect all gate parameters from patched layers.

        Returns:
            List of gate nn.Parameter objects across all layers.

        Complexity: O(L · P_gate) where P_gate is params per gate (~small).
        """
        gate_params: list[nn.Parameter] = []
        for tasft_attn in self._patched_layers.values():
            gate_params.extend(tasft_attn.gate.parameters())
        return gate_params

    def _collect_non_gate_trainable_parameters(self) -> list[nn.Parameter]:
        """Collect all trainable non-gate parameters (LoRA adapters, etc).

        Returns:
            List of non-gate trainable parameters.

        Complexity: O(total_params).
        """
        gate_param_ids = {id(p) for p in self._collect_gate_parameters()}
        return [
            p
            for p in self.model.parameters()
            if p.requires_grad and id(p) not in gate_param_ids
        ]

    def create_optimizer(self) -> torch.optim.Optimizer:
        """Create optimizer with dual LR parameter groups.

        Group 1: LoRA/non-gate parameters at base LR.
        Group 2: Gate parameters at base_lr * gate_lr_ratio.

        Also creates the gate warmup LR scheduler.

        Returns:
            Optimizer with two parameter groups.

        Postcondition: self.optimizer is set. self._gate_lr_scheduler is set.
        """
        base_lr = self._tasft_args.learning_rate
        gate_lr = base_lr * self._tasft_args.gate_lr_ratio

        non_gate_params = self._collect_non_gate_trainable_parameters()
        gate_params = self._collect_gate_parameters()

        param_groups = [
            {"params": non_gate_params, "lr": base_lr, "name": "lora"},
            {"params": gate_params, "lr": gate_lr, "name": "gate"},
        ]

        # Use AdamW (HF default)
        optimizer_cls = torch.optim.AdamW
        optimizer_kwargs: dict[str, Any] = {
            "betas": (
                self._tasft_args.adam_beta1,
                self._tasft_args.adam_beta2,
            ),
            "eps": self._tasft_args.adam_epsilon,
            "weight_decay": self._tasft_args.weight_decay,
        }
        self.optimizer = optimizer_cls(param_groups, **optimizer_kwargs)

        # Gate warmup scheduler: 0 LR for warmup_steps, then linear ramp to 1.0
        warmup_steps = self._tasft_args.gate_warmup_steps

        def gate_warmup_fn(step: int) -> float:
            """LR multiplier for gate parameters during warmup.

            Returns 0.0 for steps < warmup_steps, then linearly ramps
            from 0.0 to 1.0 over the next warmup_steps steps.

            Args:
                step: Current training step.

            Returns:
                LR multiplier in [0.0, 1.0].
            """
            if warmup_steps == 0:
                return 1.0
            if step < warmup_steps:
                return 0.0
            ramp_step = step - warmup_steps
            return min(1.0, ramp_step / max(1, warmup_steps))

        # Apply gate warmup only to the gate parameter group (index 1)
        # LR for group 0 (LoRA) is managed by HF's own scheduler
        self._gate_lr_scheduler = LambdaLR(
            self.optimizer,
            lr_lambda=[
                lambda step: 1.0,  # LoRA group: no modification (HF scheduler handles it)
                gate_warmup_fn,  # Gate group: warmup schedule
            ],
        )

        logger.info(
            "optimizer_created",
            base_lr=base_lr,
            gate_lr=gate_lr,
            num_lora_params=sum(p.numel() for p in non_gate_params),
            num_gate_params=sum(p.numel() for p in gate_params),
            gate_warmup_steps=warmup_steps,
        )

        return self.optimizer

    def training_step(
        self,
        model: nn.Module,
        inputs: dict[str, torch.Tensor | Any],
        num_items_in_batch: int | None = None,
    ) -> torch.Tensor:
        """Execute one TASFT co-training step.

        1. Select active layers via rotation scheduler
        2. Enable gate target computation on active layers
        3. Forward pass → logits + gate outputs + attention scores
        4. Compute dual loss: L_task + λ·(L_gate + β·L_sparse)
        5. Log structured metrics
        6. Step gate warmup scheduler

        Args:
            model: The training model (may be wrapped by DDP/FSDP).
            inputs: Batch dict with 'input_ids', 'labels', 'attention_mask', etc.
            num_items_in_batch: Number of items in the batch (for gradient accumulation).

        Returns:
            Scalar loss tensor for backward pass.

        Raises:
            NaNDetectedError: If loss contains NaN/Inf values.
            TrainingError: If training step fails for any other reason.

        Complexity: O(L_active · B · H · S² + B · S · V).
        """
        model.train()
        step_start_ns = time.perf_counter_ns()

        # Step 1: Select active layers for this step
        active_layers = self._rotation_scheduler.get_active_layers()
        active_indices = [int(li) for li in active_layers]

        # Wrap the entire step body in an OTel span for distributed tracing.
        # get_tracer() returns a noop tracer when OTel is not configured, so this
        # adds zero overhead in the unconfigured case.
        with trace_training_step(
            step=self.state.global_step,
            active_layers=active_indices,
        ) as span:
            # Step 2: Enable gate target computation on active layers only
            for idx, tasft_attn in self._patched_layers.items():
                tasft_attn.set_training_mode(idx in active_indices)

            # Step 3: Forward pass
            inputs = self._prepare_inputs(inputs)
            labels = inputs.get("labels")
            if labels is None:
                msg = "Labels must be provided in inputs for TASFT training"
                raise TrainingError(
                    msg,
                    context={"input_keys": list(inputs.keys())},
                )

            with self.compute_loss_context_manager():
                outputs = model(**inputs, output_attentions=True)

            logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]

            # Step 4: Collect gate outputs and attention scores from active layers
            gate_outputs_by_layer: dict[int, torch.Tensor] = {}
            attn_scores_by_layer: dict[int, torch.Tensor] = {}

            for idx in active_indices:
                tasft_attn = self._patched_layers[idx]
                gate_out = self._extract_gate_output(model, idx)
                attn_scores = self._extract_attn_scores(model, idx)

                if gate_out is not None and attn_scores is not None:
                    gate_outputs_by_layer[idx] = gate_out
                    attn_scores_by_layer[idx] = attn_scores

            # Step 5: Compute dual loss
            # Determine gate loss scaling based on warmup
            global_step = self.state.global_step
            gate_warmup_multiplier = self._get_gate_warmup_multiplier(global_step)

            if gate_outputs_by_layer and gate_warmup_multiplier > 0.0:
                loss_output = self._objective.compute(
                    logits=logits,
                    labels=labels,
                    gate_outputs_by_layer=gate_outputs_by_layer,
                    attn_scores_by_layer=attn_scores_by_layer,
                    active_layer_indices=list(gate_outputs_by_layer.keys()),
                    block_size=self._tasft_args.block_size,
                )
                # Clamp gate loss to prevent gradient explosion in float32.
                # Unclamped KL divergence between near-delta gate targets and
                # approximately-uniform gate predictions can reach 30-50, which
                # when multiplied by lambda_gate and backpropagated produces
                # NaN gradients. Capping at 10.0 bounds the effective gradient
                # contribution while still providing a strong learning signal.
                # The clamp uses straight-through: values above the cap still
                # receive zero gradient from the clamp, acting as implicit
                # gradient clipping on the loss itself.
                clamped_gate_loss = torch.clamp(loss_output.gate, max=10.0)
                clamped_sparse_loss = torch.clamp(loss_output.sparse, max=10.0)

                # Scale gate component by warmup multiplier
                loss = (
                    loss_output.task
                    + self._tasft_args.lambda_gate
                    * gate_warmup_multiplier
                    * (clamped_gate_loss + self._tasft_args.beta_sparse * clamped_sparse_loss)
                )
            else:
                # No active gate outputs or in warmup cold phase — task loss only
                loss = self._objective.compute_task_loss(logits, labels)
                loss_output = None

            # NaN guard
            if not torch.isfinite(loss):
                self._metrics.record_error("nan_detected")
                span.set_attribute("tasft.nan_detected", "true")
                msg = "Non-finite loss detected in training_step"
                raise NaNDetectedError(
                    msg,
                    context={
                        "global_step": global_step,
                        "loss_value": loss.item() if loss.numel() == 1 else "multi-element",
                        "active_layers": active_indices,
                    },
                )

            # Step 6: Report gate losses for priority-weighted rotation
            if loss_output is not None:
                for li, gate_loss_val in loss_output.per_layer_gate_loss.items():
                    self._rotation_scheduler.report_gate_loss(int(li), gate_loss_val)

            # Step 7: Step gate warmup scheduler
            # Suppress PyTorch warning about lr_scheduler.step() before optimizer.step():
            # the HF Trainer calls optimizer.step() AFTER training_step() returns, so the
            # gate LR scheduler step necessarily precedes the first optimizer.step(). This
            # ordering is intentional — the scheduler adjusts gate LR for the NEXT step.
            if self._gate_lr_scheduler is not None:
                with warnings.catch_warnings():
                    warnings.filterwarnings(
                        "ignore",
                        message="Detected call of `lr_scheduler.step\\(\\)` before",
                        category=UserWarning,
                    )
                    self._gate_lr_scheduler.step()

            # Step 8: Structured logging
            self._log_training_step(global_step, loss, loss_output, active_indices)

            # Step 9: Record Prometheus metrics and OTel span attributes
            step_duration_s = (time.perf_counter_ns() - step_start_ns) / 1_000_000_000
            self._record_step_observability(
                span, step_duration_s, loss, loss_output,
                active_indices, gate_warmup_multiplier,
            )

            # Normalize loss for gradient accumulation (matches HF Trainer convention)
            loss = loss / self.args.gradient_accumulation_steps

            # Backward pass: HF Trainer expects training_step to call backward() internally.
            # During gate warmup with no LoRA adapters, the task-only loss may have no grad_fn
            # (all base params frozen). Guard against backward on a non-differentiable tensor.
            if loss.requires_grad:
                self.accelerator.backward(loss)

                # Per-parameter gradient clipping for gate parameters.
                # Gate gradients can be disproportionately large relative to LoRA
                # gradients because the gate loss (KL divergence) operates in a
                # different numerical regime than cross-entropy. Without separate
                # clipping, global grad norm clipping may either be too loose for
                # gates (allowing overflow) or too tight for LoRA (stalling task
                # learning). Max norm of 1.0 for gate params keeps updates stable
                # in float32 without affecting LoRA gradient flow.
                gate_params = self._collect_gate_parameters()
                gate_params_with_grad = [p for p in gate_params if p.grad is not None]
                if gate_params_with_grad:
                    torch.nn.utils.clip_grad_norm_(gate_params_with_grad, max_norm=1.0)

        return loss.detach()

    def _record_step_observability(
        self,
        span: Any,
        step_duration_s: float,
        loss: torch.Tensor,
        loss_output: ObjectiveLossOutput | None,
        active_indices: list[int],
        gate_warmup_multiplier: float,
    ) -> None:
        """Record Prometheus metrics and OTel span attributes for a training step.

        Captures step duration, active layer count, effective lambda, per-layer
        sparsity ratios, GPU memory saturation, and loss component breakdowns.

        Args:
            span: The active OTel Span for attribute attachment.
            step_duration_s: Wall-clock duration of the step in seconds.
            loss: Total loss tensor (pre-accumulation normalization).
            loss_output: Decomposed loss output (None if task-only).
            active_indices: Layer indices active this step.
            gate_warmup_multiplier: Current gate warmup scaling factor.

        Preconditions: step_duration_s > 0, loss is a finite scalar.
        Postconditions: All metrics recorded, span attributes set.
        Complexity: O(L_active).
        """
        self._metrics.record_step(step_duration_s)
        self._metrics.set_active_layers(len(active_indices))
        self._metrics.set_lambda_gate(
            self._tasft_args.lambda_gate * gate_warmup_multiplier,
        )

        # Record per-layer sparsity from loss output
        if loss_output is not None and loss_output.per_layer_sparsity:
            for layer_idx, ratio in loss_output.per_layer_sparsity.items():
                self._metrics.record_sparsity(int(layer_idx), ratio)

        # Record GPU memory saturation if CUDA is available
        if torch.cuda.is_available():
            self._metrics.set_gpu_memory(
                device=str(torch.cuda.current_device()),
                bytes_used=torch.cuda.memory_allocated(),
            )

        # Attach loss values as span attributes for trace correlation
        span.set_attribute("tasft.loss_total", loss.item())
        span.set_attribute("tasft.step_duration_s", step_duration_s)
        if loss_output is not None:
            span.set_attribute("tasft.loss_task", loss_output.task.item())
            span.set_attribute("tasft.loss_gate", loss_output.gate.item())
            span.set_attribute("tasft.loss_sparse", loss_output.sparse.item())

    def _get_gate_warmup_multiplier(self, global_step: int) -> float:
        """Compute gate loss multiplier based on warmup schedule.

        Returns 0.0 during cold phase, linearly ramps from 0.0 to 1.0,
        then holds at 1.0.

        Args:
            global_step: Current training step.

        Returns:
            Multiplier in [0.0, 1.0].

        Complexity: O(1).
        """
        warmup = self._tasft_args.gate_warmup_steps
        if warmup == 0:
            return 1.0
        if global_step < warmup:
            return 0.0
        ramp = global_step - warmup
        return min(1.0, ramp / max(1, warmup))

    def _extract_gate_output(
        self, model: nn.Module, layer_idx: int,
    ) -> torch.Tensor | None:
        """Extract gate soft scores from a patched layer after forward pass.

        The TASFTAttention module stores its last gate output. We access it
        via the patched_layers reference.

        Args:
            model: The model (possibly DDP-wrapped).
            layer_idx: Layer index to extract from.

        Returns:
            Gate soft scores tensor [B, H, NB_q, NB_k] or None.
        """
        tasft_attn = self._patched_layers.get(layer_idx)
        if tasft_attn is None:
            return None

        # TASFTAttention stores last output via the forward call
        # We access the gate's last prediction via a stored attribute
        last_gate_output = getattr(tasft_attn, "_last_gate_output", None)
        if last_gate_output is not None:
            return last_gate_output.soft_scores
        return None

    def _extract_attn_scores(
        self, model: nn.Module, layer_idx: int,
    ) -> torch.Tensor | None:
        """Extract full attention scores from a patched layer after forward pass.

        Args:
            model: The model (possibly DDP-wrapped).
            layer_idx: Layer index to extract from.

        Returns:
            Attention scores tensor [B, H, S, S] or None.
        """
        tasft_attn = self._patched_layers.get(layer_idx)
        if tasft_attn is None:
            return None

        return getattr(tasft_attn, "_last_attn_weights", None)

    def _log_training_step(
        self,
        global_step: int,
        loss: torch.Tensor,
        loss_output: ObjectiveLossOutput | None,
        active_layers: list[int],
    ) -> None:
        """Emit structured log for a training step.

        Args:
            global_step: Current step number.
            loss: Total loss scalar.
            loss_output: Decomposed loss output (None if task-only).
            active_layers: Layer indices active this step.
        """
        # Extract LRs from optimizer
        lora_lr = 0.0
        gate_lr = 0.0
        if self.optimizer is not None:
            for group in self.optimizer.param_groups:
                name = group.get("name", "")
                if name == "lora":
                    lora_lr = group["lr"]
                elif name == "gate":
                    gate_lr = group["lr"]

        log_kwargs: dict[str, Any] = {
            "step": global_step,
            "loss_total": loss.item(),
            "active_layers": active_layers,
            "lr_lora": lora_lr,
            "lr_gate": gate_lr,
            "gpu_memory_mb": (
                torch.cuda.memory_allocated() // (1024 ** 2)
                if torch.cuda.is_available()
                else 0
            ),
        }

        if loss_output is not None:
            mean_sparsity = 0.0
            if loss_output.per_layer_sparsity:
                sparsity_vals = list(loss_output.per_layer_sparsity.values())
                # Kahan summation for mean
                s = 0.0
                c = 0.0
                for v in sparsity_vals:
                    y = v - c
                    t = s + y
                    c = (t - s) - y
                    s = t
                mean_sparsity = s / len(sparsity_vals)

            log_kwargs.update({
                "loss_task": loss_output.task.item(),
                "loss_gate": loss_output.gate.item(),
                "loss_sparse": loss_output.sparse.item(),
                "mean_sparsity": mean_sparsity,
            })
        else:
            log_kwargs.update({
                "loss_task": loss.item(),
                "loss_gate": 0.0,
                "loss_sparse": 0.0,
                "mean_sparsity": 0.0,
            })

        logger.info("training_step", **log_kwargs)

    def _save_checkpoint(
        self,
        model: nn.Module,
        trial: Any,
    ) -> None:
        """Save 3-artifact checkpoint: LoRA + gates + sparsity profile.

        Artifacts saved to checkpoint directory:
            1. Standard HF checkpoint (includes LoRA adapter weights)
            2. gate_state_dict.pt — all gate parameters
            3. sparsity_profile.json — per-layer mean gate sparsity

        The sparsity profile is computed by running 50 validation batches
        through the model (no_grad) and recording mean gate activation per layer.

        Args:
            model: The model to checkpoint.
            trial: Optuna trial (if hyperparameter search).
        """
        # Standard HF checkpoint (saves LoRA via PEFT integration)
        super()._save_checkpoint(model, trial)

        # Determine checkpoint directory
        checkpoint_dir = self._get_last_checkpoint_dir()
        if checkpoint_dir is None:
            logger.warning("checkpoint_dir_not_found", step=self.state.global_step)
            return

        # Save gate state dict
        gate_state: dict[str, Any] = {}
        for idx, tasft_attn in self._patched_layers.items():
            gate_prefix = f"layer_{idx}.gate."
            for name, param in tasft_attn.gate.state_dict().items():
                gate_state[gate_prefix + name] = param
        gate_path = os.path.join(checkpoint_dir, "gate_state_dict.pt")
        torch.save(gate_state, gate_path)

        # Compute and save sparsity profile
        sparsity_profile = self._compute_sparsity_profile(model)
        profile_path = os.path.join(checkpoint_dir, "sparsity_profile.json")
        # Convert SparsityProfile (dict[LayerIndex, SparsityRatio]) to JSON-serializable
        serializable_profile = {str(int(k)): float(v) for k, v in sparsity_profile.items()}
        with open(profile_path, "w") as f:
            json.dump(
                {
                    "step": self.state.global_step,
                    "num_layers": self._num_model_layers,
                    "block_size": self._tasft_args.block_size,
                    "tau_target": self._tasft_args.tau_target,
                    "per_layer_sparsity": serializable_profile,
                    "mean_sparsity": (
                        sum(float(v) for v in sparsity_profile.values()) / len(sparsity_profile)
                        if sparsity_profile
                        else 0.0
                    ),
                },
                f,
                indent=2,
            )

        logger.info(
            "checkpoint_saved",
            step=self.state.global_step,
            checkpoint_dir=checkpoint_dir,
            gate_params=len(gate_state),
            num_layers_profiled=len(sparsity_profile),
        )

    def _get_last_checkpoint_dir(self) -> str | None:
        """Get the most recent checkpoint directory path.

        Returns:
            Checkpoint directory path or None if not determinable.
        """
        output_dir = self.args.output_dir
        if output_dir is None:
            return None
        step = self.state.global_step
        checkpoint_dir = os.path.join(output_dir, f"checkpoint-{step}")
        if os.path.isdir(checkpoint_dir):
            return checkpoint_dir
        # Fallback: look for most recent checkpoint-* dir
        if os.path.isdir(output_dir):
            candidates = [
                d
                for d in os.listdir(output_dir)
                if d.startswith("checkpoint-") and os.path.isdir(os.path.join(output_dir, d))
            ]
            if candidates:
                candidates.sort(
                    key=lambda d: (
                        int(d.split("-")[-1])
                        if d.split("-")[-1].isdigit()
                        else 0
                    ),
                )
                return os.path.join(output_dir, candidates[-1])
        return None

    def _compute_sparsity_profile(self, model: nn.Module) -> SparsityProfile:
        """Compute per-layer mean gate sparsity from validation batches.

        Runs up to 50 validation batches through the model (no_grad), records
        mean gate activation (soft_scores.mean()) per layer, and returns the
        sparsity profile (1 - mean_activation = fraction of blocks below threshold).

        Args:
            model: The model to profile.

        Returns:
            SparsityProfile mapping LayerIndex -> SparsityRatio.

        Complexity: O(50 · forward_pass_cost).
        """
        profile: SparsityProfile = {}
        max_batches = 50

        # Check if we have an eval dataloader
        eval_dataloader = None
        if self.eval_dataset is not None:
            eval_dataloader = self.get_eval_dataloader()

        if eval_dataloader is None:
            # No eval data — estimate from gate biases (fallback)
            for idx, tasft_attn in self._patched_layers.items():
                li = LayerIndex(idx)
                # Default sparsity estimate from gate threshold
                profile[li] = SparsityRatio(self._tasft_args.tau_target)
            return profile

        # Accumulate gate activations per layer
        # Using Kahan summation for numerical stability
        layer_activation_sum: dict[int, float] = dict.fromkeys(self._patched_layers, 0.0)
        layer_activation_comp: dict[int, float] = dict.fromkeys(self._patched_layers, 0.0)
        num_batches = 0

        model.eval()
        # Enable gate computation on all layers for profiling
        for tasft_attn in self._patched_layers.values():
            tasft_attn.set_training_mode(True)

        with torch.no_grad():
            for batch_idx, inputs in enumerate(eval_dataloader):
                if batch_idx >= max_batches:
                    break
                inputs = self._prepare_inputs(inputs)
                _ = model(**inputs, output_attentions=True)

                for idx, tasft_attn in self._patched_layers.items():
                    gate_out = getattr(tasft_attn, "_last_gate_output", None)
                    if gate_out is not None:
                        activation = gate_out.soft_scores.mean().item()
                        # Kahan add
                        y = activation - layer_activation_comp[idx]
                        t = layer_activation_sum[idx] + y
                        layer_activation_comp[idx] = (t - layer_activation_sum[idx]) - y
                        layer_activation_sum[idx] = t

                num_batches += 1

        # Restore training mode
        model.train()
        for tasft_attn in self._patched_layers.values():
            tasft_attn.set_training_mode(False)

        if num_batches > 0:
            for idx in self._patched_layers:
                mean_activation = layer_activation_sum[idx] / num_batches
                # Sparsity = 1 - mean_activation (fraction of blocks below threshold)
                sparsity = max(0.0, min(1.0, 1.0 - mean_activation))
                profile[LayerIndex(idx)] = SparsityRatio(sparsity)

        return profile

    def evaluate(
        self,
        eval_dataset: Any | None = None,
        ignore_keys: list[str] | None = None,
        metric_key_prefix: str = "eval",
    ) -> dict[str, float]:
        """Evaluate with additional gate metrics.

        Extends standard evaluation with:
        - Per-layer gate sparsity
        - Mean gate sparsity across layers
        - Gate coverage statistics from rotation scheduler

        Args:
            eval_dataset: Optional override eval dataset.
            ignore_keys: Keys to ignore in model output.
            metric_key_prefix: Prefix for metric keys.

        Returns:
            Metrics dict with standard + gate metrics.
        """
        metrics = super().evaluate(
            eval_dataset=eval_dataset,
            ignore_keys=ignore_keys,
            metric_key_prefix=metric_key_prefix,
        )

        # Add gate metrics
        coverage = self._rotation_scheduler.get_coverage_stats()
        metrics[f"{metric_key_prefix}_gate_coverage_max_gap"] = float(coverage.max_gap)
        metrics[f"{metric_key_prefix}_gate_coverage_mean_gap"] = coverage.mean_gap
        metrics[f"{metric_key_prefix}_gate_fully_covered"] = float(coverage.fully_covered)

        # Compute sparsity profile if eval data available
        if self.eval_dataset is not None and self.model is not None:
            sparsity_profile = self._compute_sparsity_profile(self.model)
            if sparsity_profile:
                sparsity_values = [float(v) for v in sparsity_profile.values()]
                # Kahan summation for mean
                s = 0.0
                c = 0.0
                for v in sparsity_values:
                    y = v - c
                    t = s + y
                    c = (t - s) - y
                    s = t
                mean_sparsity = s / len(sparsity_values)
                metrics[f"{metric_key_prefix}_mean_gate_sparsity"] = mean_sparsity

                for li, sr in sparsity_profile.items():
                    metrics[f"{metric_key_prefix}_gate_sparsity_layer_{int(li)}"] = float(sr)

        logger.info(
            "evaluation_complete",
            step=self.state.global_step,
            num_metrics=len(metrics),
            gate_coverage_fully_covered=coverage.fully_covered,
        )

        return metrics


__all__ = ["TASFTTrainer", "TASFTTrainingArguments"]
