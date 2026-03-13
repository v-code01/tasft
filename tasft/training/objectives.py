"""
TASFT Dual Training Objective.

Implements the composite loss:
    L_total = L_task + λ · L_gate_total
    L_gate_total = L_gate + β · L_sparse

Where:
    L_task:   Cross-entropy on next-token prediction (standard LM loss)
    L_gate:   KL divergence between gate predictions and ground-truth block importance.
              Ground truth is derived from full attention scores via 2D maxpool → softmax.
    L_sparse: |mean(gate_scores) - τ_target|² — prevents degenerate all-dense/all-sparse gates.

Preconditions:
    - All input tensors are finite (no NaN/Inf). Raises NaNDetectedError otherwise.
    - gate_soft_scores shape: [B, H, NB_q, NB_k] with values in [0, 1].
    - attn_scores shape: [B, H, S, S].
    - logits shape: [B, S, V]. labels shape: [B, S].

Postconditions:
    - Output ObjectiveLossOutput contains all decomposed losses as scalar tensors.
    - All scalar losses are finite.

Complexity: O(B·H·S²/block_size² + B·S·V) per forward pass.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Final

import torch
import torch.nn.functional as F

from tasft.exceptions import NaNDetectedError
from tasft.types import AttentionScores, LayerIndex, SoftGateScores

# Numerical stability floor for log operations
_EPS: Final[float] = 1e-8


@dataclass(frozen=True, slots=True)
class ObjectiveLossOutput:
    """Decomposed loss output from the TASFT dual objective.

    All tensor fields are scalar (0-dim) tensors on the same device as inputs.

    Attributes:
        total: L_task + λ · (L_gate + β · L_sparse), the value to backprop.
        task: Cross-entropy language modeling loss.
        gate: KL divergence gate distillation loss (summed across active layers).
        sparse: Sparsity regularization loss (summed across active layers).
        per_layer_gate_loss: Per-layer gate KL divergence, keyed by LayerIndex.
        per_layer_sparsity: Per-layer mean gate score (actual sparsity), keyed by LayerIndex.
        active_layers: List of layer indices that were calibrated this step.
    """

    total: torch.Tensor
    task: torch.Tensor
    gate: torch.Tensor
    sparse: torch.Tensor
    per_layer_gate_loss: dict[LayerIndex, float]
    per_layer_sparsity: dict[LayerIndex, float]
    active_layers: list[LayerIndex]


def _check_finite(
    tensor: torch.Tensor,
    name: str,
    context: dict[str, object] | None = None,
) -> None:
    """Guard against NaN/Inf in tensors. Raises NaNDetectedError with structured context."""
    if not torch.isfinite(tensor).all():
        has_nan = bool(torch.isnan(tensor).any())
        has_inf = bool(torch.isinf(tensor).any())
        msg = f"Non-finite values detected in {name}: NaN={has_nan}, Inf={has_inf}"
        raise NaNDetectedError(
            msg,
            context={
                "tensor_name": name,
                "has_nan": has_nan,
                "has_inf": has_inf,
                "shape": list(tensor.shape),
                "dtype": str(tensor.dtype),
                **(context or {}),
            },
        )


class TASFTObjective:
    """Dual training objective for TASFT co-training.

    Combines task loss (cross-entropy) with gate calibration loss (KL divergence)
    and sparsity regularization to jointly train LoRA adapters and AttnGate modules.

    Args:
        lambda_gate: Weight for gate distillation loss. Must be in (0, 10].
        beta_sparse: Weight for sparsity regularization. Must be >= 0.
        tau_target: Target sparsity ratio for gate regularization. Must be in (0, 1).
        label_smoothing: Label smoothing factor for cross-entropy. Must be in [0, 1).
    """

    def __init__(
        self,
        lambda_gate: float = 0.1,
        beta_sparse: float = 0.01,
        tau_target: float = 0.8,
        label_smoothing: float = 0.0,
    ) -> None:
        if not 0.0 < lambda_gate <= 10.0:
            msg = f"lambda_gate must be in (0, 10], got {lambda_gate}"
            raise ValueError(msg)
        if beta_sparse < 0.0:
            msg = f"beta_sparse must be >= 0, got {beta_sparse}"
            raise ValueError(msg)
        if not 0.0 < tau_target < 1.0:
            msg = f"tau_target must be in (0, 1), got {tau_target}"
            raise ValueError(msg)
        if not 0.0 <= label_smoothing < 1.0:
            msg = f"label_smoothing must be in [0, 1), got {label_smoothing}"
            raise ValueError(msg)

        self._lambda_gate: Final[float] = lambda_gate
        self._beta_sparse: Final[float] = beta_sparse
        self._tau_target: Final[float] = tau_target
        self._label_smoothing: Final[float] = label_smoothing

    @staticmethod
    def compute_gate_target(attn_scores: AttentionScores, block_size: int) -> torch.Tensor:
        """Derive ground-truth block importance from full attention scores.

        Applies 2D max-pooling over the [S, S] attention map to produce block-level
        importance scores, then normalizes with softmax to create a valid distribution
        for KL divergence computation.

        Args:
            attn_scores: Full attention scores, shape [B, H, S, S].
            block_size: Block size for pooling. Must be > 0.

        Returns:
            Softmax-normalized block importance, shape [B, H, NB_q, NB_k].

        Complexity: O(B·H·S²) for the maxpool, O(B·H·NB_q·NB_k) for softmax.
        """
        # Allow -inf from causal masking (standard in pre-softmax attention scores).
        # Reject NaN and +inf which indicate actual numerical corruption.
        # Maxpool naturally handles -inf (takes max), softmax maps -inf → 0.
        has_nan = bool(torch.isnan(attn_scores).any())
        has_pos_inf = bool((attn_scores == float("inf")).any())
        if has_nan or has_pos_inf:
            msg = (
                f"Non-finite values detected in attn_scores: "
                f"NaN={has_nan}, Inf={has_pos_inf}"
            )
            raise NaNDetectedError(
                msg,
                context={
                    "tensor_name": "attn_scores",
                    "has_nan": has_nan,
                    "has_inf": has_pos_inf,
                    "shape": list(attn_scores.shape),
                    "dtype": str(attn_scores.dtype),
                },
            )
        b, h, s_q, s_k = attn_scores.shape
        if block_size <= 0:
            msg = f"block_size must be > 0, got {block_size}"
            raise ValueError(msg)

        # Pad to make dimensions divisible by block_size
        pad_q = (block_size - s_q % block_size) % block_size
        pad_k = (block_size - s_k % block_size) % block_size
        if pad_q > 0 or pad_k > 0:
            # F.pad: (left, right, top, bottom) — pad with -inf so maxpool ignores padding
            attn_scores = F.pad(attn_scores, (0, pad_k, 0, pad_q), value=-math.inf)

        # 2D maxpool over the [S_q, S_k] dimensions
        pooled = F.max_pool2d(attn_scores, kernel_size=block_size, stride=block_size)
        # pooled shape: [B, H, NB_q, NB_k]

        # Flatten last two dims for softmax normalization, then reshape back
        nb_q, nb_k = pooled.shape[2], pooled.shape[3]
        pooled_flat = pooled.reshape(b, h, nb_q * nb_k)
        target = F.softmax(pooled_flat, dim=-1)
        return target.reshape(b, h, nb_q, nb_k)

    @staticmethod
    def compute_gate_loss(
        gate_soft_scores: SoftGateScores,
        gate_target: torch.Tensor,
    ) -> torch.Tensor:
        """KL divergence between predicted gate distribution and ground-truth block importance.

        Uses KL(target || gate) = sum(target * (log(target) - log(gate))).
        Numerically stabilized by clamping before log.

        Args:
            gate_soft_scores: Predicted gate scores, shape [B, H, NB_q, NB_k]. Values in [0, 1].
            gate_target: Ground-truth block importance from compute_gate_target,
                         shape [B, H, NB_q, NB_k]. Must be a valid probability distribution.

        Returns:
            Scalar KL divergence loss (batchmean reduced).

        Complexity: O(B·H·NB_q·NB_k).
        """
        _check_finite(gate_soft_scores, "gate_soft_scores")
        _check_finite(gate_target, "gate_target")

        b, h, nb_q, nb_k = gate_soft_scores.shape

        # Normalize gate scores to form a distribution for KL divergence
        gate_flat = gate_soft_scores.reshape(b, h, nb_q * nb_k)
        gate_dist = gate_flat / (gate_flat.sum(dim=-1, keepdim=True) + _EPS)

        target_flat = gate_target.reshape(b, h, nb_q * nb_k)

        # F.kl_div expects log-probabilities as input, raw probabilities as target
        log_gate = torch.log(gate_dist + _EPS)

        return F.kl_div(log_gate, target_flat, reduction="batchmean", log_target=False)

    @staticmethod
    def compute_sparsity_loss(gate_soft_scores: SoftGateScores, tau_target: float) -> torch.Tensor:
        """Squared deviation of mean gate activation from target sparsity.

        Penalizes gates that are too sparse or too dense relative to τ_target.
        L_sparse = (mean(gate_scores) - τ_target)²

        Args:
            gate_soft_scores: Predicted gate scores, shape [B, H, NB_q, NB_k].
            tau_target: Target mean activation level in (0, 1).

        Returns:
            Scalar sparsity loss.

        Complexity: O(B·H·NB_q·NB_k) for the mean, O(1) for the squared difference.
        """
        _check_finite(gate_soft_scores, "gate_soft_scores")
        mean_activation = gate_soft_scores.mean()
        return (mean_activation - tau_target) ** 2

    def compute_task_loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Standard cross-entropy loss for next-token prediction.

        Args:
            logits: Model output logits, shape [B, S, V].
            labels: Ground-truth token IDs, shape [B, S]. -100 entries are ignored.

        Returns:
            Scalar cross-entropy loss.

        Complexity: O(B·S·V).
        """
        _check_finite(logits, "logits")
        # Shift logits and labels for next-token prediction
        # logits[:, :-1, :] predicts labels[:, 1:]
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        return F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=-100,
            label_smoothing=self._label_smoothing,
        )

    def compute(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        gate_outputs_by_layer: dict[int, SoftGateScores],
        attn_scores_by_layer: dict[int, AttentionScores],
        active_layer_indices: list[int],
        block_size: int = 64,
    ) -> ObjectiveLossOutput:
        """Compute the full TASFT dual objective.

        L_total = L_task + λ · (L_gate_total + β · L_sparse_total)

        Gate and sparsity losses are accumulated across active layers using Kahan
        summation for numerical stability when many layers contribute small values.

        Args:
            logits: Model output logits, shape [B, S, V].
            labels: Ground-truth labels, shape [B, S].
            gate_outputs_by_layer: Gate soft scores per layer index.
            attn_scores_by_layer: Full attention scores per layer index (only for active layers).
            active_layer_indices: Which layers are being calibrated this step.
            block_size: Block size for gate target computation.

        Returns:
            ObjectiveLossOutput with all decomposed losses.

        Raises:
            NaNDetectedError: If any input tensor contains NaN or Inf.

        Complexity: O(L_active · (B·H·S² + B·H·NB²) + B·S·V).
        """
        device = logits.device

        # Task loss
        task_loss = self.compute_task_loss(logits, labels)
        _check_finite(task_loss, "task_loss")

        # Kahan summation accumulators for gate and sparsity losses
        gate_sum = torch.tensor(0.0, device=device, dtype=logits.dtype)
        gate_comp = torch.tensor(0.0, device=device, dtype=logits.dtype)
        sparse_sum = torch.tensor(0.0, device=device, dtype=logits.dtype)
        sparse_comp = torch.tensor(0.0, device=device, dtype=logits.dtype)

        per_layer_gate: dict[LayerIndex, float] = {}
        per_layer_sparsity: dict[LayerIndex, float] = {}
        active: list[LayerIndex] = []

        for layer_idx in active_layer_indices:
            li = LayerIndex(layer_idx)
            active.append(li)

            gate_scores = gate_outputs_by_layer[layer_idx]
            attn_scores = attn_scores_by_layer[layer_idx]

            # Ground truth from full attention
            gate_target = self.compute_gate_target(attn_scores, block_size)

            # Gate KL loss for this layer
            layer_gate_loss = self.compute_gate_loss(gate_scores, gate_target)
            _check_finite(layer_gate_loss, "layer_gate_loss", {"layer": layer_idx})
            per_layer_gate[li] = layer_gate_loss.item()

            # Kahan add for gate loss
            y = layer_gate_loss - gate_comp
            t = gate_sum + y
            gate_comp = (t - gate_sum) - y
            gate_sum = t

            # Sparsity loss for this layer
            layer_sparse_loss = self.compute_sparsity_loss(gate_scores, self._tau_target)
            _check_finite(layer_sparse_loss, "layer_sparse_loss", {"layer": layer_idx})
            per_layer_sparsity[li] = gate_scores.mean().item()

            # Kahan add for sparsity loss
            y = layer_sparse_loss - sparse_comp
            t = sparse_sum + y
            sparse_comp = (t - sparse_sum) - y
            sparse_sum = t

        # Composite loss
        gate_total = gate_sum + self._beta_sparse * sparse_sum
        total_loss = task_loss + self._lambda_gate * gate_total
        _check_finite(total_loss, "total_loss")

        return ObjectiveLossOutput(
            total=total_loss,
            task=task_loss,
            gate=gate_sum,
            sparse=sparse_sum,
            per_layer_gate_loss=per_layer_gate,
            per_layer_sparsity=per_layer_sparsity,
            active_layers=active,
        )


__all__ = ["ObjectiveLossOutput", "TASFTObjective"]
