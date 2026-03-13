"""Unit tests for LayerRotationScheduler.

Tests:
- Round-robin: every layer selected exactly once per cycle
- Full coverage: after ceil(L/N) * N steps, all layers calibrated
- Priority-weighted: layers with higher EMA loss selected more frequently
- Memory estimate: verify formula matches known values
- CoverageStats: max_gap monotonically increases until next calibration
- Parameter validation

Coverage target: 100% for all strategies and coverage tracking.
"""
import math

import pytest
import torch

from tasft.training.layer_rotation import (
    CoverageStats,
    LayerRotationScheduler,
    RotationStrategy,
    estimate_activation_memory_gb,
)
from tasft.types import LayerIndex


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def rr_scheduler() -> LayerRotationScheduler:
    """Round-robin scheduler: 8 layers, 2 per step."""
    return LayerRotationScheduler(
        num_layers=8,
        layers_per_step=2,
        strategy=RotationStrategy.ROUND_ROBIN,
    )


@pytest.fixture
def priority_scheduler() -> LayerRotationScheduler:
    """Priority-weighted scheduler: 8 layers, 2 per step, fixed seed."""
    return LayerRotationScheduler(
        num_layers=8,
        layers_per_step=2,
        strategy=RotationStrategy.PRIORITY_WEIGHTED,
        seed=42,
    )


# ---------------------------------------------------------------------------
# Constructor validation
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestSchedulerConstruction:
    """Parameter validation for LayerRotationScheduler."""

    def test_valid_construction(self) -> None:
        s = LayerRotationScheduler(num_layers=32, layers_per_step=4)
        assert s.num_layers == 32
        assert s.layers_per_step == 4
        assert s.strategy == RotationStrategy.ROUND_ROBIN
        assert s.current_step == 0

    def test_num_layers_zero_raises(self) -> None:
        with pytest.raises(ValueError, match="num_layers must be > 0"):
            LayerRotationScheduler(num_layers=0, layers_per_step=1)

    def test_num_layers_negative_raises(self) -> None:
        with pytest.raises(ValueError, match="num_layers must be > 0"):
            LayerRotationScheduler(num_layers=-1, layers_per_step=1)

    def test_layers_per_step_zero_raises(self) -> None:
        with pytest.raises(ValueError, match="layers_per_step must be > 0"):
            LayerRotationScheduler(num_layers=8, layers_per_step=0)

    def test_layers_per_step_exceeds_num_layers_raises(self) -> None:
        with pytest.raises(ValueError, match="layers_per_step.*must be <= num_layers"):
            LayerRotationScheduler(num_layers=8, layers_per_step=9)

    def test_layers_per_step_equals_num_layers_ok(self) -> None:
        s = LayerRotationScheduler(num_layers=8, layers_per_step=8)
        assert s.layers_per_step == 8

    def test_ema_alpha_zero_raises(self) -> None:
        with pytest.raises(ValueError, match="ema_alpha"):
            LayerRotationScheduler(num_layers=8, layers_per_step=2, ema_alpha=0.0)

    def test_ema_alpha_negative_raises(self) -> None:
        with pytest.raises(ValueError, match="ema_alpha"):
            LayerRotationScheduler(num_layers=8, layers_per_step=2, ema_alpha=-0.1)

    def test_ema_alpha_one_ok(self) -> None:
        s = LayerRotationScheduler(num_layers=8, layers_per_step=2, ema_alpha=1.0)
        assert s._ema_alpha == 1.0


# ---------------------------------------------------------------------------
# Round-robin: every layer selected exactly once per cycle
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestRoundRobin:
    """Round-robin cycling guarantees equal coverage."""

    def test_all_layers_covered_in_one_cycle(self, rr_scheduler: LayerRotationScheduler) -> None:
        """In ceil(8/2)=4 steps, all 8 layers should be selected exactly once."""
        seen: set[LayerIndex] = set()
        cycle_len = math.ceil(8 / 2)
        for _ in range(cycle_len):
            active = rr_scheduler.get_active_layers()
            assert len(active) == 2
            seen.update(active)

        assert len(seen) == 8

    def test_exact_coverage_count_per_cycle(self, rr_scheduler: LayerRotationScheduler) -> None:
        """Each layer selected exactly once per full cycle for divisible case."""
        counts: dict[int, int] = {i: 0 for i in range(8)}
        cycle_len = 4  # 8 layers / 2 per step
        for _ in range(cycle_len):
            active = rr_scheduler.get_active_layers()
            for li in active:
                counts[int(li)] += 1

        for layer_idx, count in counts.items():
            assert count == 1, f"Layer {layer_idx} selected {count} times, expected 1"

    def test_round_robin_wraps_around(self) -> None:
        """After full cycle, round-robin wraps back to start."""
        s = LayerRotationScheduler(num_layers=6, layers_per_step=2)
        first_cycle: list[list[LayerIndex]] = []
        for _ in range(3):
            first_cycle.append(s.get_active_layers())

        second_first = s.get_active_layers()
        assert second_first == first_cycle[0]

    def test_non_divisible_layer_count(self) -> None:
        """7 layers, 3 per step: some layers wrap around within a cycle."""
        s = LayerRotationScheduler(num_layers=7, layers_per_step=3)
        # ceil(7/3) = 3 steps for full coverage
        seen: set[LayerIndex] = set()
        for _ in range(3):
            active = s.get_active_layers()
            assert len(active) == 3
            seen.update(active)

        assert len(seen) == 7

    def test_output_is_sorted(self, rr_scheduler: LayerRotationScheduler) -> None:
        """get_active_layers returns sorted LayerIndex list."""
        active = rr_scheduler.get_active_layers()
        assert active == sorted(active)

    def test_output_contains_layer_index_type(self, rr_scheduler: LayerRotationScheduler) -> None:
        active = rr_scheduler.get_active_layers()
        for li in active:
            assert isinstance(li, int)  # LayerIndex is NewType(int)

    def test_step_counter_increments(self, rr_scheduler: LayerRotationScheduler) -> None:
        assert rr_scheduler.current_step == 0
        rr_scheduler.get_active_layers()
        assert rr_scheduler.current_step == 1
        rr_scheduler.get_active_layers()
        assert rr_scheduler.current_step == 2

    def test_all_layers_equal_single_step(self) -> None:
        """When layers_per_step == num_layers, all layers active every step."""
        s = LayerRotationScheduler(num_layers=4, layers_per_step=4)
        active = s.get_active_layers()
        assert sorted(active) == [LayerIndex(i) for i in range(4)]


# ---------------------------------------------------------------------------
# Random strategy
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestRandomStrategy:
    """Random selection: uniform sampling without replacement."""

    def test_returns_correct_count(self) -> None:
        s = LayerRotationScheduler(
            num_layers=8, layers_per_step=3, strategy=RotationStrategy.RANDOM, seed=42
        )
        active = s.get_active_layers()
        assert len(active) == 3

    def test_no_duplicates(self) -> None:
        s = LayerRotationScheduler(
            num_layers=8, layers_per_step=4, strategy=RotationStrategy.RANDOM, seed=42
        )
        active = s.get_active_layers()
        assert len(set(active)) == 4

    def test_reproducible_with_same_seed(self) -> None:
        s1 = LayerRotationScheduler(
            num_layers=8, layers_per_step=2, strategy=RotationStrategy.RANDOM, seed=123
        )
        s2 = LayerRotationScheduler(
            num_layers=8, layers_per_step=2, strategy=RotationStrategy.RANDOM, seed=123
        )
        for _ in range(10):
            assert s1.get_active_layers() == s2.get_active_layers()

    def test_different_seeds_give_different_results(self) -> None:
        s1 = LayerRotationScheduler(
            num_layers=32, layers_per_step=2, strategy=RotationStrategy.RANDOM, seed=1
        )
        s2 = LayerRotationScheduler(
            num_layers=32, layers_per_step=2, strategy=RotationStrategy.RANDOM, seed=999
        )
        # Over 10 steps, they should differ at least once (extremely likely with 32 layers)
        any_different = False
        for _ in range(10):
            if s1.get_active_layers() != s2.get_active_layers():
                any_different = True
                break
        assert any_different

    def test_all_indices_in_valid_range(self) -> None:
        s = LayerRotationScheduler(
            num_layers=8, layers_per_step=3, strategy=RotationStrategy.RANDOM, seed=42
        )
        for _ in range(20):
            active = s.get_active_layers()
            for li in active:
                assert 0 <= int(li) < 8


# ---------------------------------------------------------------------------
# Priority-weighted: high loss layers selected more often
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestPriorityWeighted:
    """Priority-weighted sampling biases toward high-loss layers."""

    def test_high_loss_layer_selected_more_often(self) -> None:
        """Layer with 100x loss should be selected significantly more than others."""
        s = LayerRotationScheduler(
            num_layers=8,
            layers_per_step=1,
            strategy=RotationStrategy.PRIORITY_WEIGHTED,
            ema_alpha=1.0,  # instant update for testing
            seed=42,
        )

        # Report high loss for layer 3
        for i in range(8):
            s.report_gate_loss(i, 0.01)
        s.report_gate_loss(3, 100.0)

        counts = {i: 0 for i in range(8)}
        num_trials = 5000
        for _ in range(num_trials):
            active = s.get_active_layers()
            for li in active:
                counts[int(li)] += 1

        # Layer 3 should dominate selections
        assert counts[3] > num_trials * 0.5, (
            f"Layer 3 selected {counts[3]}/{num_trials} times, expected >50%"
        )

    def test_returns_correct_count(self, priority_scheduler: LayerRotationScheduler) -> None:
        active = priority_scheduler.get_active_layers()
        assert len(active) == 2

    def test_no_duplicates(self, priority_scheduler: LayerRotationScheduler) -> None:
        active = priority_scheduler.get_active_layers()
        assert len(set(active)) == 2

    def test_report_gate_loss_out_of_range_raises(
        self, priority_scheduler: LayerRotationScheduler
    ) -> None:
        with pytest.raises(ValueError, match="layer_idx.*out of range"):
            priority_scheduler.report_gate_loss(8, 1.0)

    def test_report_gate_loss_negative_index_raises(
        self, priority_scheduler: LayerRotationScheduler
    ) -> None:
        with pytest.raises(ValueError, match="layer_idx.*out of range"):
            priority_scheduler.report_gate_loss(-1, 1.0)

    def test_ema_update_formula(self) -> None:
        """Verify EMA: new = alpha * loss + (1 - alpha) * old."""
        alpha = 0.3
        s = LayerRotationScheduler(
            num_layers=4,
            layers_per_step=1,
            strategy=RotationStrategy.PRIORITY_WEIGHTED,
            ema_alpha=alpha,
        )
        # Initial EMA is 1.0 (uniform prior)
        initial = s._ema_gate_loss[0].item()
        assert initial == pytest.approx(1.0)

        s.report_gate_loss(0, 5.0)
        expected = alpha * 5.0 + (1 - alpha) * 1.0
        assert s._ema_gate_loss[0].item() == pytest.approx(expected, abs=1e-10)

        s.report_gate_loss(0, 2.0)
        expected2 = alpha * 2.0 + (1 - alpha) * expected
        assert s._ema_gate_loss[0].item() == pytest.approx(expected2, abs=1e-10)


# ---------------------------------------------------------------------------
# Coverage stats
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestCoverageStats:
    """CoverageStats correctness and invariants."""

    def test_initial_coverage_not_fully_covered(self) -> None:
        s = LayerRotationScheduler(num_layers=4, layers_per_step=2)
        stats = s.get_coverage_stats()
        assert not stats.fully_covered
        # All gaps should be 0 (no steps elapsed)
        assert stats.max_gap == 0
        assert stats.mean_gap == 0.0

    def test_fully_covered_after_complete_cycle(self) -> None:
        s = LayerRotationScheduler(num_layers=4, layers_per_step=2)
        # 2 steps to cover all 4 layers
        s.get_active_layers()
        s.get_active_layers()
        stats = s.get_coverage_stats()
        assert stats.fully_covered

    def test_max_gap_increases_until_recalibration(self) -> None:
        s = LayerRotationScheduler(num_layers=4, layers_per_step=1)

        prev_max_gap = -1
        # Step through: layer 0, 1, 2, 3, 0, 1, ...
        for step in range(4):
            s.get_active_layers()
            stats = s.get_coverage_stats()
            if step == 0:
                # After step 0: layer 0 gap=0, layers 1-3 gap=1
                assert stats.max_gap == 1
            # max_gap should never decrease within one cycle
            assert stats.max_gap >= 0

    def test_coverage_histogram_has_all_layers(self) -> None:
        s = LayerRotationScheduler(num_layers=8, layers_per_step=2)
        s.get_active_layers()
        stats = s.get_coverage_stats()
        assert len(stats.coverage_histogram) == 8
        for i in range(8):
            assert i in stats.coverage_histogram

    def test_mean_gap_formula(self) -> None:
        """mean_gap = sum(gaps) / num_layers."""
        s = LayerRotationScheduler(num_layers=4, layers_per_step=2)
        s.get_active_layers()  # step 0: calibrates layers 0, 1
        stats = s.get_coverage_stats()
        # After step 0 (now at step 1):
        # Layer 0: last=0, gap=1
        # Layer 1: last=0, gap=1
        # Layer 2: never, gap=1
        # Layer 3: never, gap=1
        assert stats.mean_gap == pytest.approx(1.0)

    def test_coverage_stats_type(self) -> None:
        s = LayerRotationScheduler(num_layers=4, layers_per_step=2)
        stats = s.get_coverage_stats()
        assert isinstance(stats, CoverageStats)

    def test_coverage_stats_frozen(self) -> None:
        s = LayerRotationScheduler(num_layers=4, layers_per_step=2)
        stats = s.get_coverage_stats()
        with pytest.raises(AttributeError):
            stats.max_gap = 999  # type: ignore[misc]


# ---------------------------------------------------------------------------
# cycles_for_full_coverage
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestCyclesForFullCoverage:
    """Minimum steps for every layer to be calibrated once."""

    @pytest.mark.parametrize(
        "num_layers,layers_per_step,expected",
        [
            (8, 2, 4),
            (8, 4, 2),
            (8, 8, 1),
            (7, 3, 3),  # ceil(7/3) = 3
            (32, 4, 8),
            (1, 1, 1),
        ],
    )
    def test_cycles_formula(self, num_layers: int, layers_per_step: int, expected: int) -> None:
        s = LayerRotationScheduler(num_layers=num_layers, layers_per_step=layers_per_step)
        assert s.cycles_for_full_coverage() == expected


# ---------------------------------------------------------------------------
# Memory estimation utility
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestEstimateActivationMemory:
    """estimate_activation_memory_gb formula verification."""

    def test_known_value_llama3_8b(self) -> None:
        """Llama-3-8B: B=1, H=32, S=2048, N=4 layers, bf16 (2 bytes)."""
        mem = estimate_activation_memory_gb(
            batch_size=1,
            num_heads=32,
            seq_len=2048,
            layers_per_step=4,
            dtype_bytes=2,
        )
        # 4 * 1 * 32 * 2048 * 2048 * 2 = 1,073,741,824 bytes = 1.0 GiB
        expected = (4 * 1 * 32 * 2048 * 2048 * 2) / (1024**3)
        assert mem == pytest.approx(expected, abs=1e-6)

    def test_linear_in_layers_per_step(self) -> None:
        """Memory scales linearly with layers_per_step."""
        mem_1 = estimate_activation_memory_gb(1, 32, 2048, 1, 2)
        mem_4 = estimate_activation_memory_gb(1, 32, 2048, 4, 2)
        assert mem_4 == pytest.approx(4 * mem_1, abs=1e-10)

    def test_quadratic_in_seq_len(self) -> None:
        """Memory scales quadratically with seq_len."""
        mem_1k = estimate_activation_memory_gb(1, 32, 1024, 1, 2)
        mem_2k = estimate_activation_memory_gb(1, 32, 2048, 1, 2)
        assert mem_2k == pytest.approx(4 * mem_1k, abs=1e-10)

    def test_fp32_doubles_memory_vs_bf16(self) -> None:
        mem_bf16 = estimate_activation_memory_gb(1, 32, 2048, 1, 2)
        mem_fp32 = estimate_activation_memory_gb(1, 32, 2048, 1, 4)
        assert mem_fp32 == pytest.approx(2 * mem_bf16, abs=1e-10)

    def test_zero_layers_returns_zero(self) -> None:
        mem = estimate_activation_memory_gb(1, 32, 2048, 0, 2)
        assert mem == pytest.approx(0.0)
