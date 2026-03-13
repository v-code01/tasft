"""
Layer Rotation Scheduler for memory-efficient gate calibration.

Memory problem: Retaining full [B, H, S, S] attention scores for all L layers is
prohibitive. For Llama-3-8B (L=32, H=32, S=2048, bf16):
    32 × 32 × 2048 × 2048 × 2 bytes = 8.59 GB per sample, ~275 GB at B=32.

Solution: Calibrate only N layers per step, cycling through all L layers over
    M = ceil(L/N) steps. This reduces peak memory by a factor of L/N.

Three rotation strategies:
    ROUND_ROBIN:       Deterministic, equal coverage, simple cycling.
    RANDOM:            Stochastic, equal expected coverage via uniform sampling.
    PRIORITY_WEIGHTED: Adaptive — focuses on layers with high gate error (EMA-tracked).

Preconditions:
    - num_layers > 0, layers_per_step > 0, layers_per_step <= num_layers.
    - For PRIORITY_WEIGHTED, gate losses must be reported via report_gate_loss().

Postconditions:
    - Every layer is calibrated at least once every ceil(num_layers / layers_per_step) steps
      (guaranteed for ROUND_ROBIN, expected for RANDOM/PRIORITY_WEIGHTED).
    - get_active_layers() always returns exactly layers_per_step distinct layer indices.

Complexity: O(L) per step for all strategies.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Final

import torch

from tasft.types import LayerIndex


class RotationStrategy(Enum):
    """Strategy for selecting which layers to calibrate at each training step."""

    ROUND_ROBIN = auto()
    RANDOM = auto()
    PRIORITY_WEIGHTED = auto()


@dataclass(frozen=True, slots=True)
class CoverageStats:
    """Statistics about layer calibration coverage.

    Attributes:
        coverage_histogram: Steps since last calibration per layer index.
        max_gap: Maximum steps since any layer was last calibrated.
        mean_gap: Mean steps since calibration across all layers.
        fully_covered: True if every layer has been calibrated at least once.
    """

    coverage_histogram: dict[int, int]
    max_gap: int
    mean_gap: float
    fully_covered: bool


def estimate_activation_memory_gb(
    batch_size: int,
    num_heads: int,
    seq_len: int,
    layers_per_step: int,
    dtype_bytes: int = 2,
) -> float:
    """Estimate GPU memory (GB) for retaining attention scores of active layers.

    Memory = layers_per_step × batch_size × num_heads × seq_len² × dtype_bytes.

    Args:
        batch_size: Training batch size.
        num_heads: Number of attention heads.
        seq_len: Sequence length.
        layers_per_step: Number of layers retaining full attention scores.
        dtype_bytes: Bytes per element (2 for bf16/fp16, 4 for fp32).

    Returns:
        Estimated memory in GiB (base-2 gigabytes).

    Complexity: O(1).
    """
    total_bytes = layers_per_step * batch_size * num_heads * seq_len * seq_len * dtype_bytes
    return total_bytes / (1024 ** 3)


class LayerRotationScheduler:
    """Schedules which transformer layers are actively calibrated at each training step.

    Maintains per-layer tracking for coverage statistics and, for PRIORITY_WEIGHTED
    strategy, an exponential moving average of gate loss per layer.

    Args:
        num_layers: Total number of transformer layers.
        layers_per_step: How many layers to calibrate per training step.
        strategy: Rotation strategy to use.
        ema_alpha: EMA decay factor for priority-weighted strategy. Higher = more reactive.
        seed: Random seed for reproducibility (RANDOM and PRIORITY_WEIGHTED strategies).

    Raises:
        ValueError: If num_layers <= 0, layers_per_step <= 0, or layers_per_step > num_layers.
    """

    def __init__(
        self,
        num_layers: int,
        layers_per_step: int,
        strategy: RotationStrategy = RotationStrategy.ROUND_ROBIN,
        ema_alpha: float = 0.1,
        seed: int = 42,
    ) -> None:
        if num_layers <= 0:
            raise ValueError(f"num_layers must be > 0, got {num_layers}")
        if layers_per_step <= 0:
            raise ValueError(f"layers_per_step must be > 0, got {layers_per_step}")
        if layers_per_step > num_layers:
            raise ValueError(
                f"layers_per_step ({layers_per_step}) must be <= num_layers ({num_layers})"
            )
        if not 0.0 < ema_alpha <= 1.0:
            raise ValueError(f"ema_alpha must be in (0, 1], got {ema_alpha}")

        self._num_layers: Final[int] = num_layers
        self._layers_per_step: Final[int] = layers_per_step
        self._strategy: Final[RotationStrategy] = strategy
        self._ema_alpha: Final[float] = ema_alpha

        self._step: int = 0
        self._rng = torch.Generator()
        self._rng.manual_seed(seed)

        # Per-layer tracking
        self._last_calibrated: list[int] = [-1] * num_layers  # step at which each layer was last active
        self._ema_gate_loss: torch.Tensor = torch.ones(num_layers, dtype=torch.float64)  # uniform prior

    @property
    def num_layers(self) -> int:
        """Total number of transformer layers."""
        return self._num_layers

    @property
    def layers_per_step(self) -> int:
        """Number of layers calibrated per step."""
        return self._layers_per_step

    @property
    def strategy(self) -> RotationStrategy:
        """Active rotation strategy."""
        return self._strategy

    @property
    def current_step(self) -> int:
        """Current training step counter."""
        return self._step

    def get_active_layers(self) -> list[LayerIndex]:
        """Select which layers to calibrate at the current step.

        Returns exactly `layers_per_step` distinct LayerIndex values. The selection
        strategy depends on the configured RotationStrategy.

        Returns:
            Sorted list of LayerIndex for layers to calibrate this step.

        Postcondition: len(result) == self._layers_per_step, all elements distinct.
        Complexity: O(L) for all strategies.
        """
        if self._strategy == RotationStrategy.ROUND_ROBIN:
            selected = self._round_robin()
        elif self._strategy == RotationStrategy.RANDOM:
            selected = self._random_select()
        elif self._strategy == RotationStrategy.PRIORITY_WEIGHTED:
            selected = self._priority_weighted()
        else:
            raise ValueError(f"Unknown strategy: {self._strategy}")

        # Update tracking state
        for idx in selected:
            self._last_calibrated[idx] = self._step
        self._step += 1

        return sorted(LayerIndex(i) for i in selected)

    def report_gate_loss(self, layer_idx: int, loss: float) -> None:
        """Report gate calibration loss for a layer (updates EMA for priority weighting).

        Args:
            layer_idx: Layer index in [0, num_layers).
            loss: Gate loss value for this layer at the current step.

        Complexity: O(1).
        """
        if not 0 <= layer_idx < self._num_layers:
            raise ValueError(f"layer_idx {layer_idx} out of range [0, {self._num_layers})")
        self._ema_gate_loss[layer_idx] = (
            self._ema_alpha * loss + (1.0 - self._ema_alpha) * self._ema_gate_loss[layer_idx]
        )

    def get_coverage_stats(self) -> CoverageStats:
        """Compute coverage statistics for layer calibration.

        Returns:
            CoverageStats with per-layer gaps, max/mean gap, and full coverage flag.

        Complexity: O(L).
        """
        histogram: dict[int, int] = {}
        max_gap = 0
        total_gap = 0
        fully_covered = True

        for layer_idx in range(self._num_layers):
            last = self._last_calibrated[layer_idx]
            if last < 0:
                # Never calibrated — gap is total steps elapsed
                gap = self._step
                fully_covered = False
            else:
                gap = self._step - last
            histogram[layer_idx] = gap
            max_gap = max(max_gap, gap)
            total_gap += gap

        mean_gap = total_gap / self._num_layers if self._num_layers > 0 else 0.0

        return CoverageStats(
            coverage_histogram=histogram,
            max_gap=max_gap,
            mean_gap=mean_gap,
            fully_covered=fully_covered,
        )

    def _round_robin(self) -> list[int]:
        """Deterministic cycling through layers. Equal coverage guaranteed.

        At step t, selects layers [t*N % L, (t*N+1) % L, ..., (t*N+N-1) % L]
        where N = layers_per_step, L = num_layers.
        """
        start = (self._step * self._layers_per_step) % self._num_layers
        indices: list[int] = []
        for i in range(self._layers_per_step):
            indices.append((start + i) % self._num_layers)
        return indices

    def _random_select(self) -> list[int]:
        """Uniform random selection without replacement."""
        perm = torch.randperm(self._num_layers, generator=self._rng)
        return perm[: self._layers_per_step].tolist()

    def _priority_weighted(self) -> list[int]:
        """Sample layers proportional to their EMA gate loss. Higher loss = more likely.

        Uses torch.multinomial for weighted sampling without replacement.
        Adds small epsilon to prevent zero-weight layers from being permanently ignored.
        """
        weights = self._ema_gate_loss.clone().float()
        weights = weights + 1e-6  # prevent zero weights
        selected = torch.multinomial(
            weights,
            num_samples=self._layers_per_step,
            replacement=False,
            generator=self._rng,
        )
        return selected.tolist()

    def cycles_for_full_coverage(self) -> int:
        """Minimum steps for every layer to be calibrated at least once (ROUND_ROBIN).

        Returns:
            ceil(num_layers / layers_per_step).

        Complexity: O(1).
        """
        return math.ceil(self._num_layers / self._layers_per_step)


__all__ = [
    "CoverageStats",
    "LayerRotationScheduler",
    "RotationStrategy",
    "estimate_activation_memory_gb",
]
