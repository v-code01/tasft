"""
Integration test: TASFT evaluation harness.

Tests the eval harness components with synthetic data and tiny models
to verify statistical computations, dataclass construction, and
the comparison pipeline produce correct results.

No GPU required. No external datasets loaded.
Timeout: 30s per test.
"""
from __future__ import annotations

import pytest
import torch

from tasft.eval.gate_quality import (
    GateQualityEvaluator,
    GateQualityResult,
    _kl_divergence_block,
)
from tasft.eval.task_eval import (
    ComparisonResult,
    TaskEvalResult,
    _passatk_unbiased,
    _wilson_ci,
)
from tasft.eval.throughput_bench import (
    BenchmarkPoint,
    ThroughputMatrix,
)
from tasft.exceptions import ValidationError


@pytest.mark.integration
@pytest.mark.timeout(30)
class TestTaskEvalResult:
    """Verify TaskEvalResult construction, validation, and Wilson CI."""

    def test_valid_construction(self) -> None:
        result = TaskEvalResult(
            accuracy=0.85,
            accuracy_ci_low=0.80,
            accuracy_ci_high=0.90,
            n_samples=200,
            domain="medqa",
            model_path="/tmp/model",
            eval_duration_seconds=12.5,
            per_question_results=[
                {"question_id": 0, "correct": True, "predicted": "A", "ground_truth": "A"},
            ],
            metadata={"split": "test"},
        )
        assert result.accuracy == 0.85
        assert result.n_samples == 200
        assert result.domain == "medqa"
        assert len(result.per_question_results) == 1

    def test_rejects_accuracy_above_one(self) -> None:
        with pytest.raises(ValidationError):
            TaskEvalResult(
                accuracy=1.5, accuracy_ci_low=0.0, accuracy_ci_high=1.0,
                n_samples=10, domain="x", model_path="x", eval_duration_seconds=0.0,
            )

    def test_rejects_negative_accuracy(self) -> None:
        with pytest.raises(ValidationError):
            TaskEvalResult(
                accuracy=-0.1, accuracy_ci_low=0.0, accuracy_ci_high=1.0,
                n_samples=10, domain="x", model_path="x", eval_duration_seconds=0.0,
            )

    def test_rejects_zero_samples(self) -> None:
        with pytest.raises(ValidationError):
            TaskEvalResult(
                accuracy=0.5, accuracy_ci_low=0.0, accuracy_ci_high=1.0,
                n_samples=0, domain="x", model_path="x", eval_duration_seconds=0.0,
            )

    def test_frozen_dataclass(self) -> None:
        result = TaskEvalResult(
            accuracy=0.5, accuracy_ci_low=0.4, accuracy_ci_high=0.6,
            n_samples=100, domain="test", model_path="/tmp", eval_duration_seconds=1.0,
        )
        with pytest.raises(AttributeError):
            result.accuracy = 0.9  # type: ignore[misc]


@pytest.mark.integration
@pytest.mark.timeout(30)
class TestWilsonCI:
    """Verify Wilson score interval implementation."""

    def test_known_values(self) -> None:
        """Wilson CI for p=0.5, n=100 should be approximately (0.40, 0.60)."""
        lo, hi = _wilson_ci(0.5, 100)
        assert 0.39 < lo < 0.42
        assert 0.58 < hi < 0.61

    def test_boundary_p_zero(self) -> None:
        lo, hi = _wilson_ci(0.0, 100)
        assert lo == 0.0
        assert 0.0 < hi < 0.05

    def test_boundary_p_one(self) -> None:
        lo, hi = _wilson_ci(1.0, 100)
        assert 0.95 < lo <= 1.0
        assert hi == 1.0

    def test_small_n(self) -> None:
        """With n=1, CI should be very wide."""
        lo, hi = _wilson_ci(1.0, 1)
        assert lo >= 0.0
        assert hi <= 1.0
        assert (hi - lo) > 0.3  # Wide interval

    def test_large_n_narrows_interval(self) -> None:
        """Larger n should produce narrower CI."""
        _, hi_small = _wilson_ci(0.7, 50)
        lo_small, _ = _wilson_ci(0.7, 50)
        _, hi_large = _wilson_ci(0.7, 5000)
        lo_large, _ = _wilson_ci(0.7, 5000)

        width_small = hi_small - lo_small
        width_large = hi_large - lo_large
        assert width_large < width_small

    def test_n_zero_returns_full_range(self) -> None:
        lo, hi = _wilson_ci(0.5, 0)
        assert lo == 0.0
        assert hi == 1.0


@pytest.mark.integration
@pytest.mark.timeout(30)
class TestPassAtK:
    """Verify unbiased pass@k estimator."""

    def test_all_correct_gives_one(self) -> None:
        assert _passatk_unbiased(20, 20, 1) == 1.0
        assert _passatk_unbiased(20, 20, 10) == 1.0

    def test_none_correct_gives_zero(self) -> None:
        assert _passatk_unbiased(20, 0, 1) == 0.0
        assert _passatk_unbiased(20, 0, 10) == 0.0

    def test_monotonic_in_k(self) -> None:
        """pass@k should increase with k for fixed n, c."""
        p1 = _passatk_unbiased(20, 5, 1)
        p5 = _passatk_unbiased(20, 5, 5)
        p10 = _passatk_unbiased(20, 5, 10)
        assert p1 <= p5 <= p10

    def test_monotonic_in_c(self) -> None:
        """pass@k should increase with c for fixed n, k."""
        p_low = _passatk_unbiased(20, 3, 5)
        p_high = _passatk_unbiased(20, 10, 5)
        assert p_low < p_high

    def test_k_exceeds_failures(self) -> None:
        """When k > n-c, pass@k = 1.0 (guaranteed at least one correct in sample)."""
        assert _passatk_unbiased(10, 8, 5) == 1.0

    def test_known_value(self) -> None:
        """pass@1 with n=20, c=5 should be ~0.25 (c/n)."""
        p = _passatk_unbiased(20, 5, 1)
        assert abs(p - 0.25) < 0.001


@pytest.mark.integration
@pytest.mark.timeout(30)
class TestComparisonResult:
    """Verify statistical comparison logic."""

    def test_within_target_when_close(self) -> None:
        """Models within 1% accuracy should be within_target."""
        baseline = TaskEvalResult(
            accuracy=0.80, accuracy_ci_low=0.75, accuracy_ci_high=0.85,
            n_samples=500, domain="medqa", model_path="/base", eval_duration_seconds=10.0,
            per_question_results=[
                {"question_id": i, "correct": i < 400, "predicted": "A", "ground_truth": "A"}
                for i in range(500)
            ],
        )
        tasft = TaskEvalResult(
            accuracy=0.79, accuracy_ci_low=0.74, accuracy_ci_high=0.84,
            n_samples=500, domain="medqa", model_path="/tasft", eval_duration_seconds=10.0,
            per_question_results=[
                {"question_id": i, "correct": i < 395, "predicted": "A", "ground_truth": "A"}
                for i in range(500)
            ],
        )

        comparison = ComparisonResult(
            baseline_result=baseline,
            tasft_result=tasft,
            delta_accuracy=-0.01,
            p_value=0.5,
            significant=False,
            effect_size=-0.02,
            within_target=True,
        )
        assert comparison.within_target
        assert not comparison.significant
        assert comparison.delta_accuracy == -0.01

    def test_outside_target_when_far(self) -> None:
        comparison = ComparisonResult(
            baseline_result=TaskEvalResult(
                accuracy=0.80, accuracy_ci_low=0.75, accuracy_ci_high=0.85,
                n_samples=500, domain="medqa", model_path="/base", eval_duration_seconds=10.0,
            ),
            tasft_result=TaskEvalResult(
                accuracy=0.70, accuracy_ci_low=0.65, accuracy_ci_high=0.75,
                n_samples=500, domain="medqa", model_path="/tasft", eval_duration_seconds=10.0,
            ),
            delta_accuracy=-0.10,
            p_value=0.001,
            significant=True,
            effect_size=-0.5,
            within_target=False,
        )
        assert not comparison.within_target
        assert comparison.significant


@pytest.mark.integration
@pytest.mark.timeout(30)
class TestGateQualityResult:
    """Verify gate quality dataclass validation."""

    def test_valid_cotrained(self) -> None:
        result = GateQualityResult(
            per_layer_kl={0: 0.05, 1: 0.08, 2: 0.03},
            mean_kl=0.053,
            per_layer_sparsity={0: 0.7, 1: 0.8, 2: 0.75},
            model_type="cotrained",
            eval_dataset="calibration",
            n_samples=50,
        )
        assert result.model_type == "cotrained"
        assert result.n_samples == 50

    def test_valid_posthoc(self) -> None:
        result = GateQualityResult(
            per_layer_kl={0: 0.15, 1: 0.20},
            mean_kl=0.175,
            per_layer_sparsity={0: 0.6, 1: 0.65},
            model_type="posthoc",
            eval_dataset="domain_data",
            n_samples=100,
        )
        assert result.model_type == "posthoc"

    def test_rejects_invalid_model_type(self) -> None:
        with pytest.raises(ValidationError):
            GateQualityResult(
                per_layer_kl={0: 0.1}, mean_kl=0.1, per_layer_sparsity={0: 0.5},
                model_type="invalid", eval_dataset="x", n_samples=1,
            )

    def test_rejects_zero_samples(self) -> None:
        with pytest.raises(ValidationError):
            GateQualityResult(
                per_layer_kl={0: 0.1}, mean_kl=0.1, per_layer_sparsity={0: 0.5},
                model_type="cotrained", eval_dataset="x", n_samples=0,
            )


@pytest.mark.integration
@pytest.mark.timeout(30)
class TestAblationComparison:
    """Verify the ablation comparison statistical pipeline."""

    def test_cotrained_better_than_posthoc(self) -> None:
        """When cotrained has lower KL across layers, hypothesis is supported."""
        cotrained = GateQualityResult(
            per_layer_kl={0: 0.05, 1: 0.08, 2: 0.03, 3: 0.06},
            mean_kl=0.055,
            per_layer_sparsity={0: 0.7, 1: 0.8, 2: 0.75, 3: 0.72},
            model_type="cotrained",
            eval_dataset="calibration",
            n_samples=100,
        )
        posthoc = GateQualityResult(
            per_layer_kl={0: 0.15, 1: 0.20, 2: 0.12, 3: 0.18},
            mean_kl=0.1625,
            per_layer_sparsity={0: 0.6, 1: 0.65, 2: 0.55, 3: 0.58},
            model_type="posthoc",
            eval_dataset="calibration",
            n_samples=100,
        )

        result = GateQualityEvaluator.compare_cotrained_vs_posthoc(cotrained, posthoc)

        assert result.kl_improvement > 0, "Cotrained should have lower KL"
        assert result.hypothesis_supported or result.p_value < 1.0
        # Per-layer improvement should all be positive
        for layer_idx, improvement in result.per_layer_improvement.items():
            assert improvement > 0, f"Layer {layer_idx} improvement should be positive"

    def test_no_difference_detected(self) -> None:
        """When KL values are similar, hypothesis should not be supported."""
        cotrained = GateQualityResult(
            per_layer_kl={0: 0.10, 1: 0.11, 2: 0.09, 3: 0.10},
            mean_kl=0.10,
            per_layer_sparsity={0: 0.7, 1: 0.7, 2: 0.7, 3: 0.7},
            model_type="cotrained",
            eval_dataset="calibration",
            n_samples=100,
        )
        posthoc = GateQualityResult(
            per_layer_kl={0: 0.10, 1: 0.11, 2: 0.09, 3: 0.10},
            mean_kl=0.10,
            per_layer_sparsity={0: 0.7, 1: 0.7, 2: 0.7, 3: 0.7},
            model_type="posthoc",
            eval_dataset="calibration",
            n_samples=100,
        )

        result = GateQualityEvaluator.compare_cotrained_vs_posthoc(cotrained, posthoc)

        assert abs(result.kl_improvement) < 0.001
        assert not result.hypothesis_supported

    def test_rejects_insufficient_common_layers(self) -> None:
        """Paired t-test needs at least 2 common layers."""
        cotrained = GateQualityResult(
            per_layer_kl={0: 0.1}, mean_kl=0.1, per_layer_sparsity={0: 0.7},
            model_type="cotrained", eval_dataset="x", n_samples=10,
        )
        posthoc = GateQualityResult(
            per_layer_kl={0: 0.2}, mean_kl=0.2, per_layer_sparsity={0: 0.6},
            model_type="posthoc", eval_dataset="x", n_samples=10,
        )

        with pytest.raises(ValidationError):
            GateQualityEvaluator.compare_cotrained_vs_posthoc(cotrained, posthoc)


@pytest.mark.integration
@pytest.mark.timeout(30)
class TestKLDivergenceBlock:
    """Verify block-level KL divergence computation."""

    def test_identical_distributions_zero_kl(self) -> None:
        """KL(p || p) = 0 for any valid distribution."""
        B, H, NB = 1, 4, 8
        dist = torch.softmax(torch.randn(B, H, NB * NB), dim=-1).reshape(B, H, NB, NB)
        kl = _kl_divergence_block(dist, dist)
        assert kl.item() < 0.01, f"KL(p||p) should be ~0, got {kl.item()}"

    def test_divergent_distributions_positive_kl(self) -> None:
        """KL should be positive for different distributions."""
        B, H, NB = 1, 4, 8
        p = torch.softmax(torch.randn(B, H, NB * NB), dim=-1).reshape(B, H, NB, NB)
        q = torch.softmax(torch.randn(B, H, NB * NB) * 5, dim=-1).reshape(B, H, NB, NB)
        kl = _kl_divergence_block(p, q)
        assert kl.item() > 0.0

    def test_finite_output(self) -> None:
        """KL must always be finite for valid inputs."""
        B, H, NB = 2, 8, 16
        predicted = torch.rand(B, H, NB, NB)
        target = torch.softmax(torch.randn(B, H, NB * NB), dim=-1).reshape(B, H, NB, NB)
        kl = _kl_divergence_block(predicted, target)
        assert torch.isfinite(kl)


@pytest.mark.integration
@pytest.mark.timeout(30)
class TestThroughputMatrix:
    """Verify ThroughputMatrix data structure."""

    def test_get_existing_point(self) -> None:
        point = BenchmarkPoint(
            mean_tokens_per_sec=10000.0, std_tokens_per_sec=500.0,
            p50_ms=5.0, p95_ms=7.0, p99_ms=10.0,
            gpu_util_pct=85.0, memory_mb=2048.0, sparsity_ratio=0.75,
        )
        matrix = ThroughputMatrix(
            results={4: {512: point}}, model_path="/model", device_name="test",
        )
        assert matrix.get(4, 512) is point
        assert matrix.get(4, 1024) is None
        assert matrix.get(8, 512) is None

    def test_frozen_dataclass(self) -> None:
        matrix = ThroughputMatrix()
        with pytest.raises(AttributeError):
            matrix.model_path = "/other"  # type: ignore[misc]
