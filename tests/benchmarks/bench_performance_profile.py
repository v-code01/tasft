"""Performance profiling benchmarks for TASFT critical paths.

Measures:
1. AttnGate forward latency at production sequence lengths (512-4096)
2. compute_gate_target latency at various seq_lens
3. TASFTObjective.compute() with 4 active layers, realistic tensors
4. LayerRotationScheduler.get_active_layers() overhead (<10µs/call)
5. estimate_activation_memory_gb() formula verification
6. estimate_speedup() model verification

All timings use time.perf_counter for wall-clock accuracy.
Warmup runs are excluded from measurements.
"""
from __future__ import annotations

import gc
import statistics
import time

import torch

from tasft.kernels.block_sparse_fa import BlockSparseFlashAttention
from tasft.modules.attn_gate import AttnGate
from tasft.training.layer_rotation import (
    LayerRotationScheduler,
    RotationStrategy,
    estimate_activation_memory_gb,
)
from tasft.training.objectives import TASFTObjective

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
NUM_HEADS = 32
HEAD_DIM = 128
BLOCK_SIZE = 64
BATCH_SIZE = 1
WARMUP_RUNS = 10
TIMED_RUNS = 50


def _time_fn(fn, warmup: int = WARMUP_RUNS, runs: int = TIMED_RUNS) -> dict[str, float]:
    """Time a callable, returning stats in milliseconds.

    Returns dict with keys: mean_ms, median_ms, std_ms, min_ms, max_ms.
    """
    gc.disable()
    try:
        for _ in range(warmup):
            fn()

        timings: list[float] = []
        for _ in range(runs):
            start = time.perf_counter()
            fn()
            elapsed = time.perf_counter() - start
            timings.append(elapsed * 1000.0)

        return {
            "mean_ms": statistics.mean(timings),
            "median_ms": statistics.median(timings),
            "std_ms": statistics.stdev(timings) if len(timings) > 1 else 0.0,
            "min_ms": min(timings),
            "max_ms": max(timings),
        }
    finally:
        gc.enable()


# ===== 1. AttnGate Forward Latency =====

def bench_attn_gate_forward() -> dict[int, dict[str, float]]:
    """Benchmark AttnGate forward pass at production sequence lengths.

    Verifies O((S/block_size)^2) scaling: latency at S=4096 should be ~4x S=2048.
    """
    results: dict[int, dict[str, float]] = {}
    seq_lens = [512, 1024, 2048, 4096]

    for s in seq_lens:
        gate = AttnGate(num_heads=NUM_HEADS, head_dim=HEAD_DIM, block_size=BLOCK_SIZE)
        q = torch.randn(BATCH_SIZE, NUM_HEADS, s, HEAD_DIM)
        k = torch.randn(BATCH_SIZE, NUM_HEADS, s, HEAD_DIM)

        results[s] = _time_fn(lambda q=q, k=k, gate=gate: gate(q, k))

    return results


# ===== 2. Ground Truth Computation =====

def bench_compute_gate_target() -> dict[int, dict[str, float]]:
    """Benchmark compute_gate_target at various seq_lens."""
    results: dict[int, dict[str, float]] = {}
    seq_lens = [512, 1024, 2048, 4096]

    for s in seq_lens:
        attn_scores = torch.randn(BATCH_SIZE, NUM_HEADS, s, s)

        results[s] = _time_fn(
            lambda a=attn_scores: TASFTObjective.compute_gate_target(a, BLOCK_SIZE)
        )

    return results


# ===== 3. Full Objective Computation =====

def bench_tasft_objective_compute() -> dict[str, float]:
    """Benchmark TASFTObjective.compute() with 4 active layers, realistic sizes.

    Config: batch=2, seq=1024, vocab=32000, heads=32, head_dim=128, 4 active layers.
    """
    batch = 2
    seq = 1024
    vocab = 32000
    active_layers = [0, 8, 16, 24]

    logits = torch.randn(batch, seq, vocab)
    labels = torch.randint(0, vocab, (batch, seq))

    nb = seq // BLOCK_SIZE
    gate_outputs: dict[int, torch.Tensor] = {}
    attn_scores: dict[int, torch.Tensor] = {}
    for li in active_layers:
        gate_outputs[li] = torch.sigmoid(torch.randn(batch, NUM_HEADS, nb, nb))
        attn_scores[li] = torch.randn(batch, NUM_HEADS, seq, seq)

    objective = TASFTObjective(lambda_gate=0.1, beta_sparse=0.01, tau_target=0.8)

    return _time_fn(
        lambda: objective.compute(
            logits, labels, gate_outputs, attn_scores, active_layers, BLOCK_SIZE
        )
    )


# ===== 4. Layer Rotation Overhead =====

def bench_get_active_layers() -> dict[str, float]:
    """Benchmark get_active_layers() over 100K calls. Must be <10µs/call."""
    scheduler = LayerRotationScheduler(
        num_layers=32,
        layers_per_step=4,
        strategy=RotationStrategy.ROUND_ROBIN,
    )

    n_calls = 100_000

    gc.disable()
    try:
        start = time.perf_counter()
        for _ in range(n_calls):
            scheduler.get_active_layers()
        elapsed = time.perf_counter() - start
    finally:
        gc.enable()

    total_ms = elapsed * 1000.0
    per_call_us = (elapsed / n_calls) * 1e6

    return {
        "total_ms": total_ms,
        "per_call_us": per_call_us,
        "n_calls": float(n_calls),
    }


# ===== 5. Memory Estimation Verification =====

def verify_memory_estimation() -> dict[str, float]:
    """Verify estimate_activation_memory_gb() against manual calculation.

    Llama-3-8B config: batch=4, heads=32, seq=2048, layers_per_step=4, dtype=2 bytes.
    Expected: 4 * 32 * 2048 * 2048 * 4 * 2 / (1024^3) = 4.0 GiB.
    """
    batch = 4
    heads = 32
    seq = 2048
    layers = 4
    dtype_bytes = 2

    estimated = estimate_activation_memory_gb(batch, heads, seq, layers, dtype_bytes)
    manual = batch * heads * seq * seq * layers * dtype_bytes / (1024 ** 3)
    error_pct = abs(estimated - manual) / manual * 100.0

    return {
        "estimated_gb": estimated,
        "manual_gb": manual,
        "error_pct": error_pct,
    }


# ===== 6. Speedup Model Verification =====

def verify_speedup_model() -> dict[str, dict[str, float]]:
    """Verify estimate_speedup() at key sparsity levels.

    Expected:
        sparsity=0.0 -> ~1.0x
        sparsity=0.5 -> ~2.0x
        sparsity=0.9 -> ~5-10x (SeerAttention: 5.67x)
    """
    results: dict[str, dict[str, float]] = {}
    test_points = {
        "0.0": 0.0,
        "0.25": 0.25,
        "0.5": 0.5,
        "0.75": 0.75,
        "0.9": 0.9,
        "0.95": 0.95,
    }

    for label, sparsity in test_points.items():
        speedup = BlockSparseFlashAttention.estimate_speedup(sparsity)
        results[label] = {
            "sparsity": sparsity,
            "speedup": speedup,
        }

    return results


# ===== Main Runner =====

def run_all_benchmarks() -> dict[str, object]:
    """Execute all benchmarks and return collected results."""
    results: dict[str, object] = {}

    print("=" * 70)
    print("TASFT Performance Profile")
    print("=" * 70)

    # 1. AttnGate forward
    print("\n[1/6] AttnGate forward latency...")
    gate_results = bench_attn_gate_forward()
    results["attn_gate_forward"] = gate_results
    for s, stats in gate_results.items():
        print(f"  S={s:5d}: mean={stats['mean_ms']:.3f}ms  "
              f"median={stats['median_ms']:.3f}ms  std={stats['std_ms']:.3f}ms")

    # Scaling check
    if 2048 in gate_results and 4096 in gate_results:
        ratio = gate_results[4096]["mean_ms"] / gate_results[2048]["mean_ms"]
        print(f"  Scaling ratio (S=4096/S=2048): {ratio:.2f}x (expected ~4.0x for O(NB^2))")

    # 2. compute_gate_target
    print("\n[2/6] compute_gate_target latency...")
    target_results = bench_compute_gate_target()
    results["compute_gate_target"] = target_results
    for s, stats in target_results.items():
        print(f"  S={s:5d}: mean={stats['mean_ms']:.3f}ms  "
              f"median={stats['median_ms']:.3f}ms  std={stats['std_ms']:.3f}ms")

    # 3. Full objective
    print("\n[3/6] TASFTObjective.compute() (4 active layers, B=2, S=1024)...")
    obj_results = bench_tasft_objective_compute()
    results["objective_compute"] = obj_results
    print(f"  mean={obj_results['mean_ms']:.3f}ms  "
          f"median={obj_results['median_ms']:.3f}ms  std={obj_results['std_ms']:.3f}ms")

    # 4. Layer rotation overhead
    print("\n[4/6] LayerRotationScheduler.get_active_layers() (100K calls)...")
    rotation_results = bench_get_active_layers()
    results["layer_rotation"] = rotation_results
    per_call = rotation_results["per_call_us"]
    status = "PASS" if per_call < 10.0 else "FAIL"
    print(f"  per_call={per_call:.3f}µs  total={rotation_results['total_ms']:.1f}ms  [{status}]")

    # 5. Memory estimation
    print("\n[5/6] estimate_activation_memory_gb() verification...")
    mem_results = verify_memory_estimation()
    results["memory_estimation"] = mem_results
    status = "PASS" if mem_results["error_pct"] < 10.0 else "FAIL"
    print(f"  estimated={mem_results['estimated_gb']:.4f} GB  "
          f"manual={mem_results['manual_gb']:.4f} GB  "
          f"error={mem_results['error_pct']:.2f}%  [{status}]")

    # 6. Speedup model
    print("\n[6/6] estimate_speedup() model verification...")
    speedup_results = verify_speedup_model()
    results["speedup_model"] = speedup_results
    for data in speedup_results.values():
        print(f"  sparsity={data['sparsity']:.2f}: speedup={data['speedup']:.2f}x")

    print("\n" + "=" * 70)
    print("Profile complete.")
    print("=" * 70)

    return results


# ===== pytest entry points =====

def test_attn_gate_scaling():
    """AttnGate forward: verify O((S/block_size)^2) scaling within 2x tolerance."""
    gate_results = bench_attn_gate_forward()
    ratio = gate_results[4096]["mean_ms"] / max(gate_results[2048]["mean_ms"], 1e-9)
    # Expect ~4x scaling; allow 1.5x-8x due to CPU cache effects and overhead
    assert 1.5 <= ratio <= 8.0, (
        f"Scaling ratio {ratio:.2f}x outside [1.5, 8.0] expected range for O(NB^2)"
    )


def test_layer_rotation_overhead():
    """get_active_layers() must be <10µs/call."""
    rotation_results = bench_get_active_layers()
    assert rotation_results["per_call_us"] < 10.0, (
        f"Per-call overhead {rotation_results['per_call_us']:.3f}µs exceeds 10µs limit"
    )


def test_memory_estimation_accuracy():
    """estimate_activation_memory_gb() must match manual calculation within 10%."""
    mem_results = verify_memory_estimation()
    assert mem_results["error_pct"] < 10.0, (
        f"Memory estimate error {mem_results['error_pct']:.2f}% exceeds 10% tolerance"
    )


def test_speedup_model_bounds():
    """estimate_speedup() must produce reasonable values at key sparsity levels."""
    speedup_0 = BlockSparseFlashAttention.estimate_speedup(0.0)
    speedup_50 = BlockSparseFlashAttention.estimate_speedup(0.5)
    speedup_90 = BlockSparseFlashAttention.estimate_speedup(0.9)

    assert 0.95 <= speedup_0 <= 1.05, f"sparsity=0.0 speedup {speedup_0:.2f} not ~1.0x"
    assert 1.5 <= speedup_50 <= 2.5, f"sparsity=0.5 speedup {speedup_50:.2f} not ~2.0x"
    assert 5.0 <= speedup_90 <= 11.0, f"sparsity=0.9 speedup {speedup_90:.2f} not in [5, 11]x"


if __name__ == "__main__":
    run_all_benchmarks()
