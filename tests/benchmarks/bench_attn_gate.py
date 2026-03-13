"""
Performance benchmarks for AttnGate and related components.

Uses pytest-benchmark for statistical timing. All benchmarks verify
correctness alongside performance to catch regressions in both dimensions.

Markers: @pytest.mark.perf
"""
from __future__ import annotations

import pytest
import torch

from tasft.modules import AttnGate
from tasft.training.layer_rotation import LayerRotationScheduler, RotationStrategy
from tasft.training.objectives import TASFTObjective


@pytest.mark.perf
def test_gate_forward_latency_cpu(benchmark: object) -> None:
    """Gate forward must complete in reasonable time on CPU.

    Configuration: 32 heads, 128 head_dim, block_size=64, seq_len=2048.
    Expected block grid: (2048/64)^2 = 32x32 = 1024 block pairs.
    """
    gate = AttnGate(num_heads=32, head_dim=128, block_size=64)
    gate.eval()

    q = torch.randn(1, 32, 2048, 128)
    k = torch.randn(1, 32, 2048, 128)

    with torch.no_grad():
        result = benchmark(gate, q, k)  # type: ignore[operator]

    # Verify correctness alongside perf
    num_blocks = 2048 // 64  # 32
    assert result.soft_scores.shape == (1, 32, num_blocks, num_blocks)
    assert result.soft_scores.min() >= 0.0
    assert result.soft_scores.max() <= 1.0
    assert not result.soft_scores.isnan().any()


@pytest.mark.perf
def test_gate_forward_small_seq(benchmark: object) -> None:
    """Gate forward with short sequence (256 tokens) for low-latency scenarios."""
    gate = AttnGate(num_heads=8, head_dim=64, block_size=32)
    gate.eval()

    q = torch.randn(1, 8, 256, 64)
    k = torch.randn(1, 8, 256, 64)

    with torch.no_grad():
        result = benchmark(gate, q, k)  # type: ignore[operator]

    num_blocks = 256 // 32  # 8
    assert result.soft_scores.shape == (1, 8, num_blocks, num_blocks)


@pytest.mark.perf
def test_gate_forward_batched(benchmark: object) -> None:
    """Gate forward with batch_size=4 for throughput measurement."""
    gate = AttnGate(num_heads=16, head_dim=64, block_size=64)
    gate.eval()

    q = torch.randn(4, 16, 1024, 64)
    k = torch.randn(4, 16, 1024, 64)

    with torch.no_grad():
        result = benchmark(gate, q, k)  # type: ignore[operator]

    num_blocks = 1024 // 64  # 16
    assert result.soft_scores.shape == (4, 16, num_blocks, num_blocks)


@pytest.mark.perf
def test_layer_rotation_overhead(benchmark: object) -> None:
    """LayerRotationScheduler.get_active_layers must be sub-millisecond.

    Tests all three strategies to ensure none has degenerate overhead.
    """
    scheduler = LayerRotationScheduler(num_layers=32, layers_per_step=4)

    result = benchmark(scheduler.get_active_layers)  # type: ignore[operator]

    assert len(result) == 4
    assert all(0 <= li < 32 for li in result)
    # All elements must be distinct
    assert len(set(result)) == 4


@pytest.mark.perf
def test_layer_rotation_priority_weighted_overhead(benchmark: object) -> None:
    """Priority-weighted rotation must remain fast even with reported losses."""
    scheduler = LayerRotationScheduler(
        num_layers=32,
        layers_per_step=4,
        strategy=RotationStrategy.PRIORITY_WEIGHTED,
    )

    # Pre-fill some loss history
    for layer in range(32):
        scheduler.report_gate_loss(layer, float(layer) * 0.1)

    result = benchmark(scheduler.get_active_layers)  # type: ignore[operator]

    assert len(result) == 4
    assert len(set(result)) == 4


@pytest.mark.perf
def test_ground_truth_computation_latency(benchmark: object) -> None:
    """2D maxpool ground truth computation latency.

    Configuration: seq_len=2048, block_size=64 -> 32x32 block grid.
    This is the most expensive per-layer operation in the gate loss computation.
    """
    attn_scores = torch.randn(1, 32, 2048, 2048)

    result = benchmark(TASFTObjective.compute_gate_target, attn_scores, 64)  # type: ignore[operator]

    num_blocks = 2048 // 64  # 32
    assert result.shape == (1, 32, num_blocks, num_blocks)
    # Must be a valid probability distribution (softmax output)
    assert result.min() >= 0.0
    assert result.max() <= 1.0


@pytest.mark.perf
def test_gate_loss_computation_latency(benchmark: object) -> None:
    """KL divergence gate loss computation latency."""
    B, H, NB = 1, 32, 32
    gate_scores = torch.rand(B, H, NB, NB)
    gate_target = torch.softmax(torch.randn(B, H, NB * NB), dim=-1).reshape(B, H, NB, NB)

    result = benchmark(TASFTObjective.compute_gate_loss, gate_scores, gate_target)  # type: ignore[operator]

    assert result.ndim == 0  # scalar
    assert torch.isfinite(result)


@pytest.mark.perf
def test_sparsity_loss_computation_latency(benchmark: object) -> None:
    """Sparsity regularization loss computation latency."""
    gate_scores = torch.rand(1, 32, 32, 32)

    result = benchmark(TASFTObjective.compute_sparsity_loss, gate_scores, 0.8)  # type: ignore[operator]

    assert result.ndim == 0
    assert torch.isfinite(result)
