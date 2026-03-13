"""Unit tests for long-context attention scaling benchmark.

Tests verify:
    - LongContextBenchConfig validates correctly (defaults, overrides, frozen)
    - LongContextResult dataclass construction and field access
    - generate_scaling_report produces expected format with sections/headers
    - benchmark_attention_scaling runs with tiny config on CPU (mocking CUDA)
    - OOM handling produces results with None fields
    - Results sorting by (seq_len, sparsity)

All tests run on CPU with mocked CUDA operations for CI compatibility.
"""
from __future__ import annotations

from unittest.mock import patch

import pytest
import torch

from tasft.eval.long_context_bench import (
    LongContextBenchConfig,
    LongContextResult,
    _create_block_mask,
    benchmark_attention_scaling,
    generate_scaling_report,
)


@pytest.mark.unit
class TestLongContextBenchConfig:
    """Tests for LongContextBenchConfig Pydantic model validation."""

    def test_default_config_has_expected_fields(self) -> None:
        """Default config should contain all standard sweep parameters."""
        config = LongContextBenchConfig()
        assert config.seq_lengths == [512, 1024, 2048, 4096, 8192, 16384, 32768]
        assert config.batch_sizes == [1, 2, 4]
        assert config.block_sizes == [32, 64, 128]
        assert config.sparsity_levels == [0.5, 0.7, 0.8, 0.9, 0.95]
        assert config.num_warmup == 5
        assert config.num_timed == 20
        assert config.num_heads == 32
        assert config.head_dim == 128
        assert config.device == "cuda"

    def test_custom_config_overrides(self) -> None:
        """Custom values should override defaults correctly."""
        config = LongContextBenchConfig(
            seq_lengths=[128, 256],
            batch_sizes=[1],
            block_sizes=[32],
            sparsity_levels=[0.5],
            num_warmup=2,
            num_timed=3,
            num_heads=8,
            head_dim=64,
            device="cpu",
        )
        assert config.seq_lengths == [128, 256]
        assert config.batch_sizes == [1]
        assert config.block_sizes == [32]
        assert config.sparsity_levels == [0.5]
        assert config.num_warmup == 2
        assert config.num_timed == 3
        assert config.num_heads == 8
        assert config.head_dim == 64
        assert config.device == "cpu"

    def test_config_is_frozen(self) -> None:
        """Frozen config should raise on attribute mutation."""
        config = LongContextBenchConfig()
        with pytest.raises(Exception):
            config.num_warmup = 99  # type: ignore[misc]

    def test_config_serialization_roundtrip(self) -> None:
        """Config should survive JSON serialization roundtrip."""
        original = LongContextBenchConfig(
            seq_lengths=[64, 128],
            device="cpu",
        )
        json_str = original.model_dump_json()
        restored = LongContextBenchConfig.model_validate_json(json_str)
        assert restored == original


@pytest.mark.unit
class TestLongContextResult:
    """Tests for LongContextResult dataclass."""

    def test_result_construction_with_all_fields(self) -> None:
        """Result should store all measured fields correctly."""
        result = LongContextResult(
            seq_len=4096,
            batch_size=2,
            block_size=64,
            sparsity=0.9,
            dense_ms=10.5,
            sparse_ms=3.2,
            speedup=3.28,
            memory_dense_mb=1024.0,
            memory_sparse_mb=256.0,
            throughput_toks_per_sec=2560000.0,
        )
        assert result.seq_len == 4096
        assert result.batch_size == 2
        assert result.block_size == 64
        assert result.sparsity == 0.9
        assert result.dense_ms == 10.5
        assert result.sparse_ms == 3.2
        assert result.speedup == 3.28
        assert result.memory_dense_mb == 1024.0
        assert result.memory_sparse_mb == 256.0
        assert result.throughput_toks_per_sec == 2560000.0

    def test_result_with_none_fields_for_oom(self) -> None:
        """OOM results should have None timing/memory fields."""
        result = LongContextResult(
            seq_len=32768,
            batch_size=4,
            block_size=128,
            sparsity=0.5,
            dense_ms=None,
            sparse_ms=None,
            speedup=None,
            memory_dense_mb=None,
            memory_sparse_mb=None,
            throughput_toks_per_sec=None,
        )
        assert result.dense_ms is None
        assert result.sparse_ms is None
        assert result.speedup is None
        assert result.memory_dense_mb is None
        assert result.memory_sparse_mb is None
        assert result.throughput_toks_per_sec is None

    def test_result_is_frozen(self) -> None:
        """Frozen dataclass should reject mutation."""
        result = LongContextResult(
            seq_len=512, batch_size=1, block_size=32,
            sparsity=0.5, dense_ms=1.0, sparse_ms=0.5,
            speedup=2.0, memory_dense_mb=100.0,
            memory_sparse_mb=50.0, throughput_toks_per_sec=1e6,
        )
        with pytest.raises(AttributeError):
            result.seq_len = 1024  # type: ignore[misc]


@pytest.mark.unit
class TestCreateBlockMask:
    """Tests for the block mask generation utility."""

    def test_mask_shape(self) -> None:
        """Generated mask should have shape [B, H, NB, NB]."""
        mask = _create_block_mask(2, 4, 8, 0.5, torch.device("cpu"))
        assert mask.shape == (2, 4, 8, 8)
        assert mask.dtype == torch.bool

    def test_mask_sparsity_approximately_correct(self) -> None:
        """Generated mask sparsity should be within tolerance of target.

        Uses large dimensions for statistical convergence.
        Tolerance: +/- 5% of target sparsity.
        """
        target_sparsity = 0.8
        mask = _create_block_mask(4, 8, 32, target_sparsity, torch.device("cpu"))
        active_ratio = mask.float().mean().item()
        actual_sparsity = 1.0 - active_ratio
        assert abs(actual_sparsity - target_sparsity) < 0.05

    def test_mask_zero_sparsity_all_active(self) -> None:
        """Zero sparsity should produce all-True mask."""
        mask = _create_block_mask(1, 1, 4, 0.0, torch.device("cpu"))
        assert mask.all()

    def test_mask_full_sparsity_all_inactive(self) -> None:
        """Full sparsity (1.0) should produce all-False mask."""
        mask = _create_block_mask(1, 1, 4, 1.0, torch.device("cpu"))
        assert not mask.any()


@pytest.mark.unit
class TestGenerateScalingReport:
    """Tests for scaling report generation."""

    def _make_results(self) -> list[LongContextResult]:
        """Create a small set of synthetic results for report testing."""
        results: list[LongContextResult] = []
        for sl in [512, 1024, 2048]:
            for sp in [0.5, 0.9]:
                speedup = 1.0 + sp * (sl / 512.0)  # Synthetic scaling
                results.append(LongContextResult(
                    seq_len=sl,
                    batch_size=1,
                    block_size=64,
                    sparsity=sp,
                    dense_ms=10.0 * (sl / 512.0),
                    sparse_ms=10.0 * (sl / 512.0) / speedup,
                    speedup=speedup,
                    memory_dense_mb=100.0 * (sl / 512.0),
                    memory_sparse_mb=100.0 * (sl / 512.0) * (1.0 - sp * 0.5),
                    throughput_toks_per_sec=sl * 1000.0,
                ))
        return results

    def test_report_contains_all_sections(self) -> None:
        """Report should contain all three analysis sections."""
        results = self._make_results()
        report = generate_scaling_report(results)
        assert "Section 1: Speedup" in report
        assert "Section 2: Memory Usage" in report
        assert "Section 3: Sweet Spot" in report

    def test_report_contains_header(self) -> None:
        """Report should start with the title banner."""
        results = self._make_results()
        report = generate_scaling_report(results)
        assert "TASFT Long-Context Attention Scaling Report" in report

    def test_report_contains_sequence_lengths(self) -> None:
        """Report should list all benchmarked sequence lengths."""
        results = self._make_results()
        report = generate_scaling_report(results)
        assert "512" in report
        assert "1024" in report
        assert "2048" in report

    def test_report_contains_speedup_values(self) -> None:
        """Report should contain formatted speedup values with 'x' suffix."""
        results = self._make_results()
        report = generate_scaling_report(results)
        # At least one speedup value should appear formatted as X.XXx
        assert "x" in report

    def test_report_empty_results(self) -> None:
        """Empty results should produce a short message, not crash."""
        report = generate_scaling_report([])
        assert "No benchmark results" in report

    def test_report_with_oom_results(self) -> None:
        """Report should handle OOM (None) fields gracefully."""
        results = [
            LongContextResult(
                seq_len=32768, batch_size=4, block_size=128,
                sparsity=0.5, dense_ms=None, sparse_ms=None,
                speedup=None, memory_dense_mb=None,
                memory_sparse_mb=None, throughput_toks_per_sec=None,
            ),
        ]
        report = generate_scaling_report(results)
        assert "OOM" in report
        assert "No configurations completed successfully." in report

    def test_report_mixed_oom_and_valid(self) -> None:
        """Report should handle mix of OOM and valid results."""
        results = [
            LongContextResult(
                seq_len=512, batch_size=1, block_size=64,
                sparsity=0.5, dense_ms=5.0, sparse_ms=3.0,
                speedup=1.67, memory_dense_mb=100.0,
                memory_sparse_mb=60.0, throughput_toks_per_sec=170000.0,
            ),
            LongContextResult(
                seq_len=32768, batch_size=4, block_size=64,
                sparsity=0.5, dense_ms=None, sparse_ms=None,
                speedup=None, memory_dense_mb=None,
                memory_sparse_mb=None, throughput_toks_per_sec=None,
            ),
        ]
        report = generate_scaling_report(results)
        # Should contain both valid speedup and OOM markers
        assert "1.67" in report
        assert "OOM" in report


@pytest.mark.unit
class TestBenchmarkAttentionScaling:
    """Tests for the main benchmark function using CPU with mocked CUDA."""

    def test_benchmark_tiny_config_cpu(self) -> None:
        """Benchmark should complete on CPU with tiny config and produce results.

        Mocks the timing functions directly since BlockSparseFlashAttention
        requires CUDA for validation, which is unavailable in CI.
        """
        config = LongContextBenchConfig(
            seq_lengths=[64],
            batch_sizes=[1],
            block_sizes=[32],
            sparsity_levels=[0.5],
            num_warmup=1,
            num_timed=2,
            num_heads=2,
            head_dim=32,
            device="cpu",
        )

        with patch("tasft.eval.long_context_bench._time_dense_sdpa") as mock_dense, \
             patch("tasft.eval.long_context_bench._time_sparse_attention") as mock_sparse:
            mock_dense.return_value = (5.0, 10.0)   # (mean_ms, peak_memory_mb)
            mock_sparse.return_value = (2.5, 6.0)    # (mean_ms, peak_memory_mb)

            results = benchmark_attention_scaling(config)

        assert len(results) == 1
        result = results[0]
        assert result.seq_len == 64
        assert result.batch_size == 1
        assert result.block_size == 32
        assert result.sparsity == 0.5
        assert result.dense_ms == 5.0
        assert result.sparse_ms == 2.5
        assert result.speedup is not None
        assert abs(result.speedup - 2.0) < 1e-6
        assert result.memory_dense_mb == 10.0
        assert result.memory_sparse_mb == 6.0
        assert result.throughput_toks_per_sec is not None

    def test_benchmark_multiple_configs_sorted(self) -> None:
        """Results should be sorted by (seq_len, sparsity)."""
        config = LongContextBenchConfig(
            seq_lengths=[128, 64],
            batch_sizes=[1],
            block_sizes=[32],
            sparsity_levels=[0.9, 0.5],
            num_warmup=1,
            num_timed=1,
            num_heads=2,
            head_dim=32,
            device="cpu",
        )

        with patch("tasft.eval.long_context_bench._time_dense_sdpa") as mock_dense, \
             patch("tasft.eval.long_context_bench._time_sparse_attention") as mock_sparse:
            mock_dense.return_value = (10.0, 20.0)
            mock_sparse.return_value = (3.0, 8.0)

            results = benchmark_attention_scaling(config)

        assert len(results) == 4  # 2 seq_lens * 2 sparsities
        # Verify sorted by (seq_len, sparsity)
        for i in range(len(results) - 1):
            curr = (results[i].seq_len, results[i].sparsity)
            nxt = (results[i + 1].seq_len, results[i + 1].sparsity)
            assert curr <= nxt, f"Results not sorted: {curr} > {nxt}"

    def test_benchmark_oom_handling(self) -> None:
        """OOM in dense path should produce result with None dense fields."""
        config = LongContextBenchConfig(
            seq_lengths=[64],
            batch_sizes=[1],
            block_sizes=[32],
            sparsity_levels=[0.5],
            num_warmup=1,
            num_timed=1,
            num_heads=2,
            head_dim=32,
            device="cpu",
        )

        with patch("tasft.eval.long_context_bench._time_dense_sdpa") as mock_dense, \
             patch("tasft.eval.long_context_bench._time_sparse_attention") as mock_sparse:
            mock_dense.side_effect = torch.cuda.OutOfMemoryError("OOM")
            mock_sparse.return_value = (2.0, 5.0)

            results = benchmark_attention_scaling(config)

        assert len(results) == 1
        result = results[0]
        assert result.dense_ms is None
        assert result.memory_dense_mb is None
        assert result.speedup is None  # Cannot compute without dense_ms
        assert result.sparse_ms == 2.0
        assert result.memory_sparse_mb == 5.0

    def test_benchmark_both_oom(self) -> None:
        """OOM in both paths should produce result with all None fields."""
        config = LongContextBenchConfig(
            seq_lengths=[64],
            batch_sizes=[1],
            block_sizes=[32],
            sparsity_levels=[0.5],
            num_warmup=1,
            num_timed=1,
            num_heads=2,
            head_dim=32,
            device="cpu",
        )

        with patch("tasft.eval.long_context_bench._time_dense_sdpa") as mock_dense, \
             patch("tasft.eval.long_context_bench._time_sparse_attention") as mock_sparse:
            mock_dense.side_effect = torch.cuda.OutOfMemoryError("OOM")
            mock_sparse.side_effect = torch.cuda.OutOfMemoryError("OOM")

            results = benchmark_attention_scaling(config)

        assert len(results) == 1
        result = results[0]
        assert result.dense_ms is None
        assert result.sparse_ms is None
        assert result.speedup is None
        assert result.throughput_toks_per_sec is None

    def test_throughput_calculation(self) -> None:
        """Throughput should be batch_size * seq_len / (sparse_ms / 1000)."""
        config = LongContextBenchConfig(
            seq_lengths=[1024],
            batch_sizes=[2],
            block_sizes=[32],
            sparsity_levels=[0.5],
            num_warmup=1,
            num_timed=1,
            num_heads=2,
            head_dim=32,
            device="cpu",
        )

        sparse_ms = 4.0
        with patch("tasft.eval.long_context_bench._time_dense_sdpa") as mock_dense, \
             patch("tasft.eval.long_context_bench._time_sparse_attention") as mock_sparse:
            mock_dense.return_value = (8.0, 20.0)
            mock_sparse.return_value = (sparse_ms, 10.0)

            results = benchmark_attention_scaling(config)

        result = results[0]
        expected_throughput = (2 * 1024) / (sparse_ms / 1000.0)  # 512000.0
        assert result.throughput_toks_per_sec is not None
        assert abs(result.throughput_toks_per_sec - expected_throughput) < 1.0


@pytest.mark.unit
class TestEndToEnd:
    """Integration-style tests verifying benchmark -> report pipeline."""

    def test_benchmark_to_report_pipeline(self) -> None:
        """Full pipeline: config -> benchmark -> report should produce valid output."""
        config = LongContextBenchConfig(
            seq_lengths=[64, 128],
            batch_sizes=[1],
            block_sizes=[32],
            sparsity_levels=[0.5, 0.9],
            num_warmup=1,
            num_timed=1,
            num_heads=2,
            head_dim=32,
            device="cpu",
        )

        with patch("tasft.eval.long_context_bench._time_dense_sdpa") as mock_dense, \
             patch("tasft.eval.long_context_bench._time_sparse_attention") as mock_sparse:
            # Simulate increasing speedup with sequence length and sparsity
            call_count = 0
            def dense_side_effect(*args: object, **kwargs: object) -> tuple[float, float]:
                nonlocal call_count
                call_count += 1
                return (5.0 * call_count, 50.0 * call_count)

            sparse_count = 0
            def sparse_side_effect(*args: object, **kwargs: object) -> tuple[float, float]:
                nonlocal sparse_count
                sparse_count += 1
                return (2.0 * sparse_count, 20.0 * sparse_count)

            mock_dense.side_effect = dense_side_effect
            mock_sparse.side_effect = sparse_side_effect

            results = benchmark_attention_scaling(config)

        assert len(results) == 4  # 2 seq_lens * 2 sparsities
        report = generate_scaling_report(results)
        assert len(report) > 100  # Non-trivial output
        assert "Section 1" in report
        assert "Section 2" in report
        assert "Section 3" in report
        # All results have valid speedup, so sweet spot section should have entries
        assert "No configurations completed successfully" not in report
