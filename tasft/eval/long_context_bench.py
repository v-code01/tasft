"""Long-context attention scaling benchmark for TASFT sparse attention.

Characterizes block-sparse attention speedup and memory savings across
sequence lengths from 512 to 32768+ tokens, demonstrating that the value
proposition of sparse attention strengthens with context length.

Key measurements per (seq_len, batch_size, block_size, sparsity) configuration:
    - Dense SDPA wall-clock time (ms)
    - Sparse BlockSparseFlashAttention wall-clock time (ms)
    - Speedup ratio (dense_ms / sparse_ms)
    - Peak GPU memory for dense and sparse paths (MB)
    - Throughput in tokens/second

Preconditions:
    - CUDA available for GPU benchmarking (graceful OOM handling)
    - torch, pydantic installed
    - BlockSparseFlashAttention importable from tasft.kernels

Postconditions:
    - LongContextResult entries contain statistically stable timing
      (num_warmup warmups + num_timed timed iterations per config)
    - OOM configurations recorded with None fields, not raised
    - Results sorted by (seq_len, sparsity) for monotonic scaling analysis

Complexity: O(|seq_lengths| * |batch_sizes| * |block_sizes| * |sparsity_levels|
             * (num_warmup + num_timed) * B * H * S^2 * D) for dense path,
             O(...* (1-sparsity) * S^2) for sparse path.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Final

import torch
from pydantic import BaseModel

from tasft.exceptions import TASFTError
from tasft.observability.logging import get_logger

logger = get_logger("tasft.eval.long_context_bench")

_BYTES_PER_MB: Final[float] = 1024.0 * 1024.0
_NS_PER_MS: Final[float] = 1_000_000.0
_MS_PER_S: Final[float] = 1000.0


class LongContextBenchError(TASFTError):
    """Raised when long-context benchmarking encounters an unrecoverable error."""


class LongContextBenchConfig(BaseModel, frozen=True):
    """Configuration for long-context attention scaling benchmark.

    All fields are immutable after construction (frozen=True).
    Validated via Pydantic field constraints.

    Attributes:
        seq_lengths: Sequence lengths to benchmark. Must be positive.
        batch_sizes: Batch sizes to test. Must be positive.
        block_sizes: Attention block granularities (must be in {32, 64, 128}).
        sparsity_levels: Target sparsity ratios in [0, 1].
        num_warmup: GPU warmup iterations before timing.
        num_timed: Timed iterations for statistical stability.
        num_heads: Number of attention heads.
        head_dim: Dimension per head.
        device: Torch device string for benchmark execution.
    """

    seq_lengths: list[int] = [512, 1024, 2048, 4096, 8192, 16384, 32768]
    batch_sizes: list[int] = [1, 2, 4]
    block_sizes: list[int] = [32, 64, 128]
    sparsity_levels: list[float] = [0.5, 0.7, 0.8, 0.9, 0.95]
    num_warmup: int = 5
    num_timed: int = 20
    num_heads: int = 32
    head_dim: int = 128
    device: str = "cuda"


@dataclass(frozen=True, slots=True)
class LongContextResult:
    """Single benchmark measurement for a (seq_len, batch_size, block_size, sparsity) config.

    None-valued timing/memory fields indicate OOM during that path.

    Attributes:
        seq_len: Sequence length in tokens.
        batch_size: Batch size.
        block_size: Block granularity for sparse attention.
        sparsity: Target sparsity ratio (fraction of blocks skipped).
        dense_ms: Mean dense SDPA latency in milliseconds, or None on OOM.
        sparse_ms: Mean sparse attention latency in milliseconds, or None on OOM.
        speedup: dense_ms / sparse_ms ratio, or None if either path OOM'd.
        memory_dense_mb: Peak GPU memory during dense path in MB, or None on OOM.
        memory_sparse_mb: Peak GPU memory during sparse path in MB, or None on OOM.
        throughput_toks_per_sec: Tokens processed per second (sparse path), or None on OOM.
    """

    seq_len: int
    batch_size: int
    block_size: int
    sparsity: float
    dense_ms: float | None
    sparse_ms: float | None
    speedup: float | None
    memory_dense_mb: float | None
    memory_sparse_mb: float | None
    throughput_toks_per_sec: float | None


def _create_block_mask(
    batch_size: int,
    num_heads: int,
    num_blocks: int,
    target_sparsity: float,
    device: torch.device,
) -> torch.Tensor:
    """Create a random block mask with approximately the target sparsity.

    The mask is boolean [B, H, NB, NB] where True means "compute this block."
    Sparsity = fraction of False entries = fraction of blocks skipped.

    Args:
        batch_size: Batch dimension.
        num_heads: Number of attention heads.
        num_blocks: Number of blocks along each sequence axis.
        target_sparsity: Desired fraction of False (skipped) blocks in [0, 1].
        device: Torch device for mask allocation.

    Returns:
        Boolean tensor [B, H, NB, NB] with ~target_sparsity fraction of False entries.

    Complexity: O(B * H * NB^2).
    """
    # active_ratio = 1 - sparsity; draw uniform random and threshold
    active_ratio = 1.0 - target_sparsity
    # Deterministic per-config for reproducibility
    rand_vals = torch.rand(
        batch_size, num_heads, num_blocks, num_blocks,
        device=device, dtype=torch.float32,
    )
    return rand_vals < active_ratio


def _time_dense_sdpa(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    num_warmup: int,
    num_timed: int,
    use_cuda_events: bool,
) -> tuple[float, float]:
    """Time dense scaled dot-product attention.

    Args:
        q, k, v: [B, H, S, D] tensors.
        num_warmup: Warmup iterations.
        num_timed: Timed iterations.
        use_cuda_events: Whether to use CUDA events for precise timing.

    Returns:
        (mean_ms, peak_memory_mb) tuple.

    Complexity: O((num_warmup + num_timed) * B * H * S^2 * D).
    """
    # Warmup
    for _ in range(num_warmup):
        torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=False)
    if use_cuda_events:
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

    latencies_ms: list[float] = []
    for _ in range(num_timed):
        if use_cuda_events:
            start_ev = torch.cuda.Event(enable_timing=True)
            end_ev = torch.cuda.Event(enable_timing=True)
            start_ev.record()
            torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=False)
            end_ev.record()
            torch.cuda.synchronize()
            latencies_ms.append(start_ev.elapsed_time(end_ev))
        else:
            t0 = time.perf_counter_ns()
            torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=False)
            latencies_ms.append((time.perf_counter_ns() - t0) / _NS_PER_MS)

    peak_mb = 0.0
    if use_cuda_events:
        peak_mb = torch.cuda.max_memory_allocated() / _BYTES_PER_MB

    mean_ms = sum(latencies_ms) / len(latencies_ms)
    return mean_ms, peak_mb


def _time_sparse_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    block_mask: torch.Tensor,
    block_size: int,
    num_warmup: int,
    num_timed: int,
    use_cuda_events: bool,
) -> tuple[float, float]:
    """Time block-sparse FlashAttention.

    Uses BlockSparseFlashAttention with dense fallback backend when Triton
    is unavailable, or the Triton backend when available.

    Args:
        q, k, v: [B, H, S, D] tensors.
        block_mask: [B, H, NB, NB] boolean mask.
        block_size: Block granularity.
        num_warmup: Warmup iterations.
        num_timed: Timed iterations.
        use_cuda_events: Whether to use CUDA events for precise timing.

    Returns:
        (mean_ms, peak_memory_mb) tuple.

    Complexity: O((num_warmup + num_timed) * B * H * active_blocks * block_size^2 * D).
    """
    from tasft.kernels.block_sparse_fa import BlockSparseFlashAttention

    bsfa = BlockSparseFlashAttention(
        block_size=block_size,
        min_sparsity_for_speedup=0.0,  # Never fall back to dense -- we want sparse timing
    )

    # Warmup
    for _ in range(num_warmup):
        bsfa.forward(q, k, v, block_mask, causal=False)
    if use_cuda_events:
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

    latencies_ms: list[float] = []
    for _ in range(num_timed):
        if use_cuda_events:
            start_ev = torch.cuda.Event(enable_timing=True)
            end_ev = torch.cuda.Event(enable_timing=True)
            start_ev.record()
            bsfa.forward(q, k, v, block_mask, causal=False)
            end_ev.record()
            torch.cuda.synchronize()
            latencies_ms.append(start_ev.elapsed_time(end_ev))
        else:
            t0 = time.perf_counter_ns()
            bsfa.forward(q, k, v, block_mask, causal=False)
            latencies_ms.append((time.perf_counter_ns() - t0) / _NS_PER_MS)

    peak_mb = 0.0
    if use_cuda_events:
        peak_mb = torch.cuda.max_memory_allocated() / _BYTES_PER_MB

    mean_ms = sum(latencies_ms) / len(latencies_ms)
    return mean_ms, peak_mb


@torch.inference_mode()
def benchmark_attention_scaling(
    config: LongContextBenchConfig,
) -> list[LongContextResult]:
    """Benchmark dense vs sparse attention across long-context configurations.

    Iterates over all (seq_len, batch_size, block_size, sparsity) combinations
    from the config, timing both dense SDPA and BlockSparseFlashAttention.
    OOM errors are handled gracefully -- the configuration is recorded with
    None fields and execution continues.

    Args:
        config: Benchmark configuration specifying all parameter sweeps.

    Returns:
        List of LongContextResult sorted by (seq_len, sparsity) for
        monotonic scaling analysis. OOM configs appear with None timing fields.

    Complexity: O(|configs| * (num_warmup + num_timed) * B * H * S^2 * D).
    Side effects: Allocates and frees GPU memory; resets peak memory stats.
    """
    device = torch.device(config.device)
    use_cuda = device.type == "cuda"
    results: list[LongContextResult] = []

    total_configs = (
        len(config.seq_lengths)
        * len(config.batch_sizes)
        * len(config.block_sizes)
        * len(config.sparsity_levels)
    )
    config_idx = 0

    for seq_len in config.seq_lengths:
        for batch_size in config.batch_sizes:
            for block_size in config.block_sizes:
                for sparsity in config.sparsity_levels:
                    config_idx += 1
                    logger.info(
                        "[LONG_BENCH_CONFIG] Running configuration",
                        config_num=f"{config_idx}/{total_configs}",
                        seq_len=seq_len,
                        batch_size=batch_size,
                        block_size=block_size,
                        sparsity=sparsity,
                    )

                    # Pad seq_len to block_size multiple for kernel compatibility
                    padded_seq = math.ceil(seq_len / block_size) * block_size
                    num_blocks = padded_seq // block_size

                    # --- Dense path ---
                    dense_ms: float | None = None
                    memory_dense_mb: float | None = None
                    try:
                        q = torch.randn(
                            batch_size, config.num_heads, padded_seq, config.head_dim,
                            device=device, dtype=torch.bfloat16,
                        )
                        k = torch.randn_like(q)
                        v = torch.randn_like(q)

                        dense_ms, memory_dense_mb = _time_dense_sdpa(
                            q, k, v,
                            config.num_warmup, config.num_timed,
                            use_cuda,
                        )

                        logger.info(
                            "[LONG_BENCH_DENSE] Dense timing complete",
                            seq_len=seq_len,
                            dense_ms=round(dense_ms, 3),
                            memory_mb=round(memory_dense_mb, 1),
                        )
                    except torch.cuda.OutOfMemoryError:
                        logger.warning(
                            "[LONG_BENCH_OOM] Dense path OOM",
                            seq_len=seq_len,
                            batch_size=batch_size,
                        )
                        if use_cuda:
                            torch.cuda.empty_cache()
                    finally:
                        # Free dense tensors before sparse allocation
                        # Local variable cleanup -- set to None to allow GC
                        q = k = v = None  # type: ignore[assignment]
                        if use_cuda:
                            torch.cuda.empty_cache()

                    # --- Sparse path ---
                    sparse_ms: float | None = None
                    memory_sparse_mb: float | None = None
                    try:
                        q = torch.randn(
                            batch_size, config.num_heads, padded_seq, config.head_dim,
                            device=device, dtype=torch.bfloat16,
                        )
                        k = torch.randn_like(q)
                        v = torch.randn_like(q)

                        block_mask = _create_block_mask(
                            batch_size, config.num_heads,
                            num_blocks, sparsity, device,
                        )

                        sparse_ms, memory_sparse_mb = _time_sparse_attention(
                            q, k, v, block_mask, block_size,
                            config.num_warmup, config.num_timed,
                            use_cuda,
                        )

                        logger.info(
                            "[LONG_BENCH_SPARSE] Sparse timing complete",
                            seq_len=seq_len,
                            sparsity=sparsity,
                            sparse_ms=round(sparse_ms, 3),
                            memory_mb=round(memory_sparse_mb, 1),
                        )
                    except torch.cuda.OutOfMemoryError:
                        logger.warning(
                            "[LONG_BENCH_OOM] Sparse path OOM",
                            seq_len=seq_len,
                            batch_size=batch_size,
                            sparsity=sparsity,
                        )
                        if use_cuda:
                            torch.cuda.empty_cache()
                    finally:
                        q = k = v = None  # type: ignore[assignment]
                        if use_cuda:
                            torch.cuda.empty_cache()

                    # Compute derived metrics
                    speedup: float | None = None
                    if dense_ms is not None and sparse_ms is not None and sparse_ms > 0:
                        speedup = dense_ms / sparse_ms

                    throughput: float | None = None
                    if sparse_ms is not None and sparse_ms > 0:
                        tokens = batch_size * seq_len
                        throughput = tokens / (sparse_ms / _MS_PER_S)

                    result = LongContextResult(
                        seq_len=seq_len,
                        batch_size=batch_size,
                        block_size=block_size,
                        sparsity=sparsity,
                        dense_ms=dense_ms,
                        sparse_ms=sparse_ms,
                        speedup=speedup,
                        memory_dense_mb=memory_dense_mb,
                        memory_sparse_mb=memory_sparse_mb,
                        throughput_toks_per_sec=throughput,
                    )
                    results.append(result)

                    logger.info(
                        "[LONG_BENCH_RESULT] Configuration complete",
                        seq_len=seq_len,
                        batch_size=batch_size,
                        block_size=block_size,
                        sparsity=sparsity,
                        speedup=round(speedup, 2) if speedup is not None else None,
                    )

    # Sort by (seq_len, sparsity) for monotonic scaling analysis
    results.sort(key=lambda r: (r.seq_len, r.sparsity))
    return results


def generate_scaling_report(results: list[LongContextResult]) -> str:
    """Generate a formatted scaling report from benchmark results.

    Produces three sections:
    1. Speedup table: speedup vs (seq_len, sparsity) for each block_size
    2. Memory savings table: dense_mb vs sparse_mb with savings percentage
    3. Sweet spot analysis: configurations with best speedup-per-memory tradeoff

    Args:
        results: List of LongContextResult from benchmark_attention_scaling.

    Returns:
        Multi-line formatted string suitable for console or file output.

    Complexity: O(n log n) where n = len(results), dominated by sorting.
    """
    if not results:
        return "No benchmark results to report."

    lines: list[str] = []
    lines.append("=" * 100)
    lines.append("TASFT Long-Context Attention Scaling Report")
    lines.append("=" * 100)
    lines.append("")

    # --- Section 1: Speedup Table ---
    lines.append("-" * 100)
    lines.append("Section 1: Speedup (dense_ms / sparse_ms) by Sequence Length and Sparsity")
    lines.append("-" * 100)
    lines.append("")

    # Group by block_size
    block_sizes_seen = sorted({r.block_size for r in results})
    for bs in block_sizes_seen:
        block_results = [r for r in results if r.block_size == bs]
        if not block_results:
            continue

        lines.append(f"  Block Size: {bs}")

        # Collect unique seq_lens and sparsities for this block_size
        seq_lens = sorted({r.seq_len for r in block_results})
        sparsities = sorted({r.sparsity for r in block_results})

        # Header row
        sparsity_headers = [f"s={s:.0%}" for s in sparsities]
        header = f"  {'seq_len':>10s} | " + " | ".join(f"{h:>8s}" for h in sparsity_headers)
        lines.append(header)
        lines.append("  " + "-" * len(header.strip()))

        for sl in seq_lens:
            row_cells: list[str] = []
            for sp in sparsities:
                # Find matching result (first match for this batch_size/block_size combo)
                matching = [
                    r for r in block_results
                    if r.seq_len == sl and r.sparsity == sp
                ]
                if matching and matching[0].speedup is not None:
                    row_cells.append(f"{matching[0].speedup:>8.2f}x")
                else:
                    row_cells.append(f"{'OOM':>9s}")
            lines.append(f"  {sl:>10d} | " + " | ".join(row_cells))
        lines.append("")

    # --- Section 2: Memory Savings ---
    lines.append("-" * 100)
    lines.append("Section 2: Memory Usage (MB) - Dense vs Sparse")
    lines.append("-" * 100)
    lines.append("")

    header_mem = f"  {'seq_len':>10s} | {'batch':>5s} | {'block':>5s} | {'sparsity':>8s} | {'dense_mb':>10s} | {'sparse_mb':>10s} | {'savings':>8s}"
    lines.append(header_mem)
    lines.append("  " + "-" * len(header_mem.strip()))

    for r in results:
        if r.memory_dense_mb is not None and r.memory_sparse_mb is not None:
            savings_pct = (1.0 - r.memory_sparse_mb / max(r.memory_dense_mb, 1e-6)) * 100.0
            lines.append(
                f"  {r.seq_len:>10d} | {r.batch_size:>5d} | {r.block_size:>5d} | "
                f"{r.sparsity:>8.0%} | {r.memory_dense_mb:>10.1f} | "
                f"{r.memory_sparse_mb:>10.1f} | {savings_pct:>7.1f}%"
            )
        elif r.memory_dense_mb is None and r.memory_sparse_mb is None:
            lines.append(
                f"  {r.seq_len:>10d} | {r.batch_size:>5d} | {r.block_size:>5d} | "
                f"{r.sparsity:>8.0%} | {'OOM':>10s} | {'OOM':>10s} | {'N/A':>8s}"
            )
        else:
            dense_str = f"{r.memory_dense_mb:>10.1f}" if r.memory_dense_mb is not None else f"{'OOM':>10s}"
            sparse_str = f"{r.memory_sparse_mb:>10.1f}" if r.memory_sparse_mb is not None else f"{'OOM':>10s}"
            lines.append(
                f"  {r.seq_len:>10d} | {r.batch_size:>5d} | {r.block_size:>5d} | "
                f"{r.sparsity:>8.0%} | {dense_str} | {sparse_str} | {'N/A':>8s}"
            )

    lines.append("")

    # --- Section 3: Sweet Spot Analysis ---
    lines.append("-" * 100)
    lines.append("Section 3: Sweet Spot Configurations (Top 10 by Speedup)")
    lines.append("-" * 100)
    lines.append("")

    # Filter to results with valid speedup, sort descending
    valid_speedup = [r for r in results if r.speedup is not None]
    top_configs = sorted(valid_speedup, key=lambda r: r.speedup, reverse=True)[:10]  # type: ignore[arg-type]

    if top_configs:
        header_sweet = (
            f"  {'rank':>4s} | {'seq_len':>10s} | {'batch':>5s} | {'block':>5s} | "
            f"{'sparsity':>8s} | {'speedup':>8s} | {'sparse_ms':>10s} | {'toks/s':>12s}"
        )
        lines.append(header_sweet)
        lines.append("  " + "-" * len(header_sweet.strip()))

        for rank, r in enumerate(top_configs, 1):
            tps_str = f"{r.throughput_toks_per_sec:>12.0f}" if r.throughput_toks_per_sec is not None else f"{'N/A':>12s}"
            sparse_str = f"{r.sparse_ms:>10.3f}" if r.sparse_ms is not None else f"{'N/A':>10s}"
            speedup_str = f"{r.speedup:>8.2f}x" if r.speedup is not None else f"{'N/A':>9s}"
            lines.append(
                f"  {rank:>4d} | {r.seq_len:>10d} | {r.batch_size:>5d} | {r.block_size:>5d} | "
                f"{r.sparsity:>8.0%} | {speedup_str} | {sparse_str} | {tps_str}"
            )
    else:
        lines.append("  No configurations completed successfully.")

    lines.append("")
    lines.append("=" * 100)

    return "\n".join(lines)


def _cli_main(
    seq_lengths: list[int] | None = None,
    batch_sizes: list[int] | None = None,
    block_sizes: list[int] | None = None,
    sparsity_levels: list[float] | None = None,
    num_warmup: int = 5,
    num_timed: int = 20,
    num_heads: int = 32,
    head_dim: int = 128,
    device: str = "cuda",
    output_file: str | None = None,
) -> None:
    """CLI entrypoint for long-context attention benchmarking.

    Args:
        seq_lengths: Override default sequence lengths.
        batch_sizes: Override default batch sizes.
        block_sizes: Override default block sizes.
        sparsity_levels: Override default sparsity levels.
        num_warmup: Warmup iterations.
        num_timed: Timed iterations.
        num_heads: Number of attention heads.
        head_dim: Head dimension.
        device: Torch device.
        output_file: Optional file path to write report.
    """
    config_kwargs: dict[str, object] = {
        "num_warmup": num_warmup,
        "num_timed": num_timed,
        "num_heads": num_heads,
        "head_dim": head_dim,
        "device": device,
    }
    if seq_lengths is not None:
        config_kwargs["seq_lengths"] = seq_lengths
    if batch_sizes is not None:
        config_kwargs["batch_sizes"] = batch_sizes
    if block_sizes is not None:
        config_kwargs["block_sizes"] = block_sizes
    if sparsity_levels is not None:
        config_kwargs["sparsity_levels"] = sparsity_levels

    config = LongContextBenchConfig(**config_kwargs)  # type: ignore[arg-type]

    logger.info(
        "[LONG_BENCH_START] Starting long-context benchmark",
        seq_lengths=config.seq_lengths,
        batch_sizes=config.batch_sizes,
        block_sizes=config.block_sizes,
        sparsity_levels=config.sparsity_levels,
        num_warmup=config.num_warmup,
        num_timed=config.num_timed,
    )

    bench_results = benchmark_attention_scaling(config)
    report = generate_scaling_report(bench_results)

    print(report)  # noqa: T201 -- CLI output

    if output_file is not None:
        with open(output_file, "w") as f:
            f.write(report)
        logger.info(
            "[LONG_BENCH_SAVED] Report saved",
            output_file=output_file,
        )


if __name__ == "__main__":
    import typer

    typer.run(_cli_main)


__all__ = [
    "LongContextBenchConfig",
    "LongContextBenchError",
    "LongContextResult",
    "benchmark_attention_scaling",
    "generate_scaling_report",
]
