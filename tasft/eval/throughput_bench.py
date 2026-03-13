"""
Inference throughput benchmarking for TASFT vs dense LoRA models.

Target: TASFT at 70-90% attention sparsity achieves 2-4x decode throughput
vs standard LoRA model at same batch size, hardware, and serving stack.

Preconditions:
    - Model loadable via transformers AutoModelForCausalLM
    - CUDA available for GPU timing (torch.cuda.Event)
    - Optional: pynvml installed for GPU utilization monitoring

Postconditions:
    - BenchmarkPoint contains statistically stable timing (50 timed runs after warmup)
    - ThroughputMatrix fully populated for all (batch_size, seq_len) combinations
    - SpeedupMatrix compares TASFT vs dense point-by-point

Complexity: O(sum over (bs, sl): (num_warmup + num_timed) * bs * sl * model_flops)
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Final

import numpy as np
import torch

from tasft.exceptions import TASFTError, ValidationError
from tasft.observability.logging import get_logger, timed_operation

logger = get_logger("tasft.eval.throughput_bench")

_DEFAULT_BATCH_SIZES: Final[list[int]] = [1, 4, 8, 16, 32]
_DEFAULT_SEQ_LENS: Final[list[int]] = [512, 1024, 2048]
_DEFAULT_NUM_WARMUP: Final[int] = 10
_DEFAULT_NUM_TIMED: Final[int] = 50


class BenchmarkError(TASFTError):
    """Raised when benchmarking encounters an unrecoverable error."""


@dataclass(frozen=True)
class BenchmarkPoint:
    """Single throughput measurement at a specific (batch_size, seq_len) configuration.

    Attributes:
        mean_tokens_per_sec: Mean throughput across timed iterations.
        std_tokens_per_sec: Standard deviation of throughput.
        p50_ms: Median latency per forward pass in milliseconds.
        p95_ms: 95th percentile latency.
        p99_ms: 99th percentile latency.
        gpu_util_pct: Mean GPU utilization percentage during benchmark (0-100).
        memory_mb: Peak GPU memory usage in megabytes.
        sparsity_ratio: Effective attention sparsity if applicable (0.0 for dense).
    """

    mean_tokens_per_sec: float
    std_tokens_per_sec: float
    p50_ms: float
    p95_ms: float
    p99_ms: float
    gpu_util_pct: float
    memory_mb: float
    sparsity_ratio: float


@dataclass(frozen=True)
class ThroughputMatrix:
    """Full throughput benchmark results across batch sizes and sequence lengths.

    Attributes:
        results: Nested dict mapping batch_size -> seq_len -> BenchmarkPoint.
        model_path: Path to benchmarked model.
        device_name: GPU device name string.
        num_warmup: Warmup iterations used.
        num_timed: Timed iterations used.
    """

    results: dict[int, dict[int, BenchmarkPoint]] = field(default_factory=dict)
    model_path: str = ""
    device_name: str = ""
    num_warmup: int = _DEFAULT_NUM_WARMUP
    num_timed: int = _DEFAULT_NUM_TIMED

    def get(self, batch_size: int, seq_len: int) -> BenchmarkPoint | None:
        """Retrieve benchmark point for a specific configuration.

        Args:
            batch_size: Batch size.
            seq_len: Sequence length.

        Returns:
            BenchmarkPoint or None if not benchmarked.

        Complexity: O(1).
        """
        return self.results.get(batch_size, {}).get(seq_len)


@dataclass(frozen=True)
class SpeedupMatrix:
    """Point-by-point speedup comparison between TASFT and dense models.

    Attributes:
        speedups: dict[batch_size][seq_len] -> speedup ratio (tasft_tps / dense_tps).
        tasft_matrix: Full TASFT benchmark results.
        dense_matrix: Full dense model benchmark results.
    """

    speedups: dict[int, dict[int, float]] = field(default_factory=dict)
    tasft_matrix: ThroughputMatrix = field(default_factory=ThroughputMatrix)
    dense_matrix: ThroughputMatrix = field(default_factory=ThroughputMatrix)


def _get_gpu_utilization() -> float:
    """Query GPU utilization via pynvml.

    Returns:
        GPU utilization percentage (0-100), or 0.0 if pynvml unavailable.

    Complexity: O(1).
    """
    try:
        import pynvml
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        util = pynvml.nvmlDeviceGetUtilizationRates(handle)
        return float(util.gpu)
    except Exception:
        return 0.0


def _get_gpu_memory_mb() -> float:
    """Query current GPU memory usage.

    Returns:
        Allocated GPU memory in MB, or 0.0 if CUDA unavailable.

    Complexity: O(1).
    """
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / (1024 * 1024)
    return 0.0


def _get_device_name() -> str:
    """Return GPU device name or 'cpu'.

    Complexity: O(1).
    """
    if torch.cuda.is_available():
        return torch.cuda.get_device_name(0)
    return "cpu"


class ThroughputBenchmark:
    """Inference throughput benchmarker for TASFT and dense models.

    Measures tokens/second, latency percentiles, GPU utilization, and memory
    across a matrix of (batch_size, seq_len) configurations using CUDA events
    for precise GPU timing.
    """

    def __init__(self, device: str | None = None) -> None:
        """Initialize benchmark runner.

        Args:
            device: Torch device string. Auto-detected if None.
        """
        if device is not None:
            self._device = torch.device(device)
        elif torch.cuda.is_available():
            self._device = torch.device("cuda")
        else:
            self._device = torch.device("cpu")
            logger.warning(
                "[BENCH_INIT] No GPU detected; GPU timing unavailable, falling back to wall clock",
                device="cpu",
            )

    def _load_model(self, model_path: str) -> tuple[torch.nn.Module, int]:
        """Load model for benchmarking.

        Args:
            model_path: Path to model.

        Returns:
            (model, vocab_size) tuple.

        Raises:
            BenchmarkError: If model loading fails.
        """
        try:
            from transformers import AutoModelForCausalLM, AutoConfig
        except ImportError as exc:
            raise BenchmarkError(
                "transformers package required for benchmarking",
                context={"missing_package": "transformers"},
            ) from exc

        try:
            config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16,
                device_map="auto" if self._device.type == "cuda" else None,
                trust_remote_code=True,
            )
            if self._device.type != "cuda":
                model = model.to(self._device)
            model.eval()
            return model, config.vocab_size
        except Exception as exc:
            raise BenchmarkError(
                f"Failed to load model: {exc}",
                context={"model_path": model_path, "error": str(exc)},
            ) from exc

    @torch.inference_mode()
    def _benchmark_single(
        self,
        model: torch.nn.Module,
        vocab_size: int,
        batch_size: int,
        seq_len: int,
        num_warmup: int,
        num_timed: int,
    ) -> BenchmarkPoint:
        """Run benchmark for a single (batch_size, seq_len) configuration.

        Uses torch.cuda.Event for CUDA-accurate timing when GPU is available,
        falls back to time.perf_counter_ns for CPU.

        Args:
            model: Loaded model in eval mode.
            vocab_size: Model vocabulary size for random input generation.
            batch_size: Batch size.
            seq_len: Sequence length.
            num_warmup: Warmup iterations (not timed).
            num_timed: Timed iterations.

        Returns:
            BenchmarkPoint with timing statistics.

        Complexity: O((num_warmup + num_timed) * batch_size * seq_len * model_forward_cost).
        """
        # Generate random input_ids
        input_ids = torch.randint(
            0, vocab_size, (batch_size, seq_len),
            device=self._device, dtype=torch.long,
        )
        attention_mask = torch.ones_like(input_ids)

        use_cuda_events = self._device.type == "cuda"

        # Warmup
        for _ in range(num_warmup):
            model(input_ids=input_ids, attention_mask=attention_mask)
        if use_cuda_events:
            torch.cuda.synchronize()

        # Reset memory tracking
        if use_cuda_events:
            torch.cuda.reset_peak_memory_stats()

        # Timed iterations
        latencies_ms: list[float] = []
        gpu_utils: list[float] = []

        for _ in range(num_timed):
            if use_cuda_events:
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                start_event.record()
                model(input_ids=input_ids, attention_mask=attention_mask)
                end_event.record()
                torch.cuda.synchronize()
                elapsed_ms = start_event.elapsed_time(end_event)
            else:
                t0 = time.perf_counter_ns()
                model(input_ids=input_ids, attention_mask=attention_mask)
                elapsed_ms = (time.perf_counter_ns() - t0) / 1_000_000.0

            latencies_ms.append(elapsed_ms)
            gpu_utils.append(_get_gpu_utilization())

        latencies = np.array(latencies_ms, dtype=np.float64)
        tokens_per_iter = batch_size * seq_len
        tps_values = tokens_per_iter / (latencies / 1000.0)  # tokens/second

        memory_mb = _get_gpu_memory_mb()

        # Compute sparsity if the model exposes it (TASFT bundles)
        sparsity = 0.0
        if hasattr(model, "get_sparsity_ratio"):
            sparsity = float(model.get_sparsity_ratio())

        return BenchmarkPoint(
            mean_tokens_per_sec=float(np.mean(tps_values)),
            std_tokens_per_sec=float(np.std(tps_values, ddof=1)),
            p50_ms=float(np.percentile(latencies, 50)),
            p95_ms=float(np.percentile(latencies, 95)),
            p99_ms=float(np.percentile(latencies, 99)),
            gpu_util_pct=float(np.mean(gpu_utils)) if gpu_utils else 0.0,
            memory_mb=memory_mb,
            sparsity_ratio=sparsity,
        )

    def run(
        self,
        model_or_bundle_path: str,
        batch_sizes: list[int] | None = None,
        seq_lens: list[int] | None = None,
        num_warmup: int = _DEFAULT_NUM_WARMUP,
        num_timed: int = _DEFAULT_NUM_TIMED,
    ) -> ThroughputMatrix:
        """Run full throughput benchmark across a matrix of configurations.

        Args:
            model_or_bundle_path: Path to model or TASFT bundle.
            batch_sizes: List of batch sizes to test.
            seq_lens: List of sequence lengths to test.
            num_warmup: Warmup iterations per configuration.
            num_timed: Timed iterations per configuration.

        Returns:
            ThroughputMatrix with results for all (bs, sl) pairs.

        Raises:
            ValidationError: If batch_sizes or seq_lens contain non-positive values.
        """
        bs_list = batch_sizes if batch_sizes is not None else _DEFAULT_BATCH_SIZES
        sl_list = seq_lens if seq_lens is not None else _DEFAULT_SEQ_LENS

        for bs in bs_list:
            if bs <= 0:
                raise ValidationError(
                    f"batch_size must be positive, got {bs}",
                    context={"batch_sizes": bs_list},
                )
        for sl in sl_list:
            if sl <= 0:
                raise ValidationError(
                    f"seq_len must be positive, got {sl}",
                    context={"seq_lens": sl_list},
                )

        with timed_operation(logger, "BENCH_LOAD_MODEL", model_path=model_or_bundle_path):
            model, vocab_size = self._load_model(model_or_bundle_path)

        device_name = _get_device_name()
        results: dict[int, dict[int, BenchmarkPoint]] = {}

        total_configs = len(bs_list) * len(sl_list)
        config_idx = 0

        for bs in bs_list:
            results[bs] = {}
            for sl in sl_list:
                config_idx += 1
                logger.info(
                    "[BENCH_CONFIG] Running benchmark",
                    config=f"{config_idx}/{total_configs}",
                    batch_size=bs,
                    seq_len=sl,
                    num_warmup=num_warmup,
                    num_timed=num_timed,
                )

                try:
                    point = self._benchmark_single(
                        model, vocab_size, bs, sl, num_warmup, num_timed,
                    )
                    results[bs][sl] = point

                    logger.info(
                        "[BENCH_RESULT] Configuration complete",
                        batch_size=bs,
                        seq_len=sl,
                        mean_tps=round(point.mean_tokens_per_sec, 1),
                        p50_ms=round(point.p50_ms, 2),
                        p99_ms=round(point.p99_ms, 2),
                        memory_mb=round(point.memory_mb, 1),
                    )
                except torch.cuda.OutOfMemoryError:
                    logger.warning(
                        "[BENCH_OOM] Skipping configuration due to OOM",
                        batch_size=bs,
                        seq_len=sl,
                    )
                    if self._device.type == "cuda":
                        torch.cuda.empty_cache()

        return ThroughputMatrix(
            results=results,
            model_path=model_or_bundle_path,
            device_name=device_name,
            num_warmup=num_warmup,
            num_timed=num_timed,
        )

    def compare_sparse_vs_dense(
        self,
        tasft_bundle_path: str,
        dense_model_path: str,
        **kwargs: object,
    ) -> SpeedupMatrix:
        """Compare TASFT sparse model throughput against dense baseline.

        Runs full benchmark on both models and computes point-by-point
        speedup ratios (tasft_tps / dense_tps).

        Args:
            tasft_bundle_path: Path to TASFT bundle with sparse attention.
            dense_model_path: Path to dense (standard LoRA) model.
            **kwargs: Additional arguments passed to run().

        Returns:
            SpeedupMatrix with per-configuration speedup ratios.
        """
        with timed_operation(logger, "BENCH_DENSE", model_path=dense_model_path):
            dense_matrix = self.run(dense_model_path, **kwargs)  # type: ignore[arg-type]

        with timed_operation(logger, "BENCH_SPARSE", model_path=tasft_bundle_path):
            tasft_matrix = self.run(tasft_bundle_path, **kwargs)  # type: ignore[arg-type]

        speedups: dict[int, dict[int, float]] = {}
        for bs in dense_matrix.results:
            speedups[bs] = {}
            for sl in dense_matrix.results.get(bs, {}):
                dense_point = dense_matrix.get(bs, sl)
                tasft_point = tasft_matrix.get(bs, sl)
                if dense_point is not None and tasft_point is not None:
                    # Avoid division by zero
                    if dense_point.mean_tokens_per_sec > 0:
                        ratio = tasft_point.mean_tokens_per_sec / dense_point.mean_tokens_per_sec
                    else:
                        ratio = float("inf")
                    speedups[bs][sl] = ratio

                    logger.info(
                        "[BENCH_SPEEDUP] Speedup computed",
                        batch_size=bs,
                        seq_len=sl,
                        speedup=round(ratio, 2),
                        dense_tps=round(dense_point.mean_tokens_per_sec, 1),
                        tasft_tps=round(tasft_point.mean_tokens_per_sec, 1),
                    )

        return SpeedupMatrix(
            speedups=speedups,
            tasft_matrix=tasft_matrix,
            dense_matrix=dense_matrix,
        )


__all__ = [
    "BenchmarkError",
    "BenchmarkPoint",
    "SpeedupMatrix",
    "ThroughputBenchmark",
    "ThroughputMatrix",
]
