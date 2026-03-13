"""
Performance benchmarks for TASFT training step overhead.

Measures the wall-clock overhead of TASFT's dual objective computation
vs standard cross-entropy only, isolating the gate calibration cost.

Markers: @pytest.mark.perf
"""
from __future__ import annotations

import pytest
import torch

from tasft.training.objectives import TASFTObjective


@pytest.mark.perf
def test_dual_objective_full_compute_latency(benchmark: object) -> None:
    """Full dual objective compute() latency with 4 active layers.

    Configuration: B=2, H=8, S=512, V=1024, block_size=64, 4 active layers.
    This measures the most expensive per-step operation.
    """
    obj = TASFTObjective(lambda_gate=0.1, beta_sparse=0.01, tau_target=0.8)
    B, H, S, V = 2, 8, 512, 1024
    NB = S // 64

    logits = torch.randn(B, S, V)
    labels = torch.randint(0, V, (B, S))
    gate_outputs = {i: torch.rand(B, H, NB, NB) for i in range(4)}
    attn_scores = {i: torch.randn(B, H, S, S) for i in range(4)}

    def run() -> object:
        return obj.compute(
            logits, labels, gate_outputs, attn_scores,
            active_layer_indices=[0, 1, 2, 3], block_size=64,
        )

    result = benchmark(run)  # type: ignore[operator]
    assert torch.isfinite(result.total)
    assert result.total.ndim == 0


@pytest.mark.perf
def test_task_loss_only_latency(benchmark: object) -> None:
    """Cross-entropy task loss alone — baseline for overhead measurement."""
    obj = TASFTObjective(lambda_gate=0.1)
    B, S, V = 2, 512, 1024

    logits = torch.randn(B, S, V)
    labels = torch.randint(0, V, (B, S))

    result = benchmark(obj.compute_task_loss, logits, labels)  # type: ignore[operator]
    assert torch.isfinite(result)
    assert result.ndim == 0


@pytest.mark.perf
def test_gate_target_scaling_with_seq_len(benchmark: object) -> None:
    """Gate target computation scales quadratically with seq_len.

    Measures at S=1024 to establish a data point for roofline analysis.
    """
    S = 1024
    attn_scores = torch.randn(1, 16, S, S)

    result = benchmark(TASFTObjective.compute_gate_target, attn_scores, 64)  # type: ignore[operator]

    NB = S // 64
    assert result.shape == (1, 16, NB, NB)
    assert torch.isfinite(result).all()


@pytest.mark.perf
def test_kl_divergence_batch_scaling(benchmark: object) -> None:
    """KL divergence scales linearly with batch size. Measure at B=8."""
    B, H, NB = 8, 16, 16
    gate_scores = torch.rand(B, H, NB, NB)
    gate_target = torch.softmax(torch.randn(B, H, NB * NB), dim=-1).reshape(B, H, NB, NB)

    result = benchmark(TASFTObjective.compute_gate_loss, gate_scores, gate_target)  # type: ignore[operator]
    assert torch.isfinite(result)
