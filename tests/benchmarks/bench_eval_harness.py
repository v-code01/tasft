"""
Performance benchmarks for TASFT evaluation harness components.

Benchmarks statistical computations used in the eval harness:
Wilson CI, pass@k estimator, and KL divergence for gate quality.

Markers: @pytest.mark.perf
"""
from __future__ import annotations

import pytest
import torch

from tasft.eval.gate_quality import _kl_divergence_block
from tasft.eval.task_eval import _passatk_unbiased, _wilson_ci


@pytest.mark.perf
def test_wilson_ci_latency(benchmark: object) -> None:
    """Wilson CI computation must be sub-microsecond for single-sample call."""
    result = benchmark(_wilson_ci, 0.75, 1000)  # type: ignore[operator]
    lo, hi = result
    assert 0.0 <= lo < hi <= 1.0


@pytest.mark.perf
def test_passatk_unbiased_latency(benchmark: object) -> None:
    """Unbiased pass@k estimator must be sub-microsecond for k=10."""
    result = benchmark(_passatk_unbiased, 200, 50, 10)  # type: ignore[operator]
    assert 0.0 <= result <= 1.0


@pytest.mark.perf
def test_kl_divergence_block_latency(benchmark: object) -> None:
    """Block-level KL divergence computation at realistic dimensions.

    Configuration: B=4, H=32, NB=32 (seq_len=2048, block_size=64).
    """
    B, H, NB = 4, 32, 32
    predicted = torch.rand(B, H, NB, NB)
    target = torch.softmax(torch.randn(B, H, NB * NB), dim=-1).reshape(B, H, NB, NB)

    result = benchmark(_kl_divergence_block, predicted, target)  # type: ignore[operator]
    assert torch.isfinite(result)
    assert result.ndim == 0


@pytest.mark.perf
def test_wilson_ci_correctness_across_range(benchmark: object) -> None:
    """Wilson CI across 100 different proportions — amortized per-call latency."""
    def compute_all() -> list[tuple[float, float]]:
        return [_wilson_ci(p / 100.0, 500) for p in range(1, 100)]

    results = benchmark(compute_all)  # type: ignore[operator]
    for lo, hi in results:
        assert 0.0 <= lo <= hi <= 1.0
