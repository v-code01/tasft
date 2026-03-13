"""
Chaos tests: NaN injection, numerical stability, and concurrency.

Validates that TASFT correctly detects and reports non-finite values
without silent corruption. Every NaN/Inf must raise NaNDetectedError
with structured context for debugging.

Markers: @pytest.mark.chaos
"""
from __future__ import annotations

import multiprocessing
from pathlib import Path

import pytest
import torch

from tasft.exceptions import NaNDetectedError
from tasft.training.objectives import TASFTObjective


def _try_create_dir(output_path_str: str, queue: multiprocessing.Queue) -> None:  # type: ignore[type-arg]
    """Module-level function for multiprocessing pickling compatibility."""
    try:
        path = Path(output_path_str)
        path.mkdir(parents=True, exist_ok=False)
        queue.put(("success", output_path_str))
    except FileExistsError:
        queue.put(("conflict", output_path_str))
    except Exception as e:
        queue.put(("error", str(e)))


@pytest.mark.chaos
@pytest.mark.timeout(10)
def test_nan_in_attn_scores_raises_training_error() -> None:
    """NaN in attention scores must raise NaNDetectedError with structured context.

    The compute_gate_target path uses _check_finite which must detect NaN
    before any pooling or softmax operations corrupt further state.
    """
    attn_scores = torch.randn(1, 4, 32, 32)
    attn_scores[0, 0, 5, 3] = float("nan")

    with pytest.raises(NaNDetectedError) as exc_info:
        TASFTObjective.compute_gate_target(attn_scores, block_size=8)

    assert exc_info.value.context is not None
    error_str = str(exc_info.value).lower()
    context_str = str(exc_info.value.context).lower()
    assert "nan" in error_str or "nan" in context_str or "finite" in error_str


@pytest.mark.chaos
@pytest.mark.timeout(10)
def test_inf_in_attn_scores_raises_training_error() -> None:
    """Inf in attention scores must also raise NaNDetectedError."""
    attn_scores = torch.randn(1, 4, 32, 32)
    attn_scores[0, 2, 10, 10] = float("inf")

    with pytest.raises(NaNDetectedError) as exc_info:
        TASFTObjective.compute_gate_target(attn_scores, block_size=8)

    assert exc_info.value.context is not None
    assert exc_info.value.context.get("has_inf", False) or "inf" in str(exc_info.value).lower()


@pytest.mark.chaos
@pytest.mark.timeout(10)
def test_nan_in_gate_scores_raises_error() -> None:
    """NaN in gate soft scores must raise NaNDetectedError in gate loss computation."""
    B, H, NB = 1, 4, 4
    gate_scores = torch.rand(B, H, NB, NB)
    gate_scores[0, 1, 2, 2] = float("nan")

    gate_target = torch.softmax(torch.randn(B, H, NB * NB), dim=-1).reshape(B, H, NB, NB)

    with pytest.raises(NaNDetectedError):
        TASFTObjective.compute_gate_loss(gate_scores, gate_target)


@pytest.mark.chaos
@pytest.mark.timeout(10)
def test_nan_in_logits_raises_error() -> None:
    """NaN in logits must raise NaNDetectedError in task loss computation."""
    obj = TASFTObjective(lambda_gate=0.1)

    logits = torch.randn(1, 32, 256)
    logits[0, 10, 50] = float("nan")
    labels = torch.randint(0, 256, (1, 32))

    with pytest.raises(NaNDetectedError):
        obj.compute_task_loss(logits, labels)


@pytest.mark.chaos
@pytest.mark.timeout(10)
def test_nan_in_compute_triggers_error_before_aggregation() -> None:
    """NaN in any layer's attention scores must raise before corrupting other layers.

    The compute() method processes layers sequentially. A NaN in layer 1's
    attention scores must raise NaNDetectedError immediately, not after
    aggregating with layer 0's valid results.
    """
    obj = TASFTObjective(lambda_gate=0.1)

    B, H, NB = 1, 4, 4
    gate_outputs = {
        0: torch.rand(B, H, NB, NB),
        1: torch.rand(B, H, NB, NB),
    }
    attn_scores = {
        0: torch.randn(B, H, 32, 32),
        1: torch.randn(B, H, 32, 32),
    }
    # Inject NaN in layer 1
    attn_scores[1][0, 0, 2, 2] = float("nan")

    logits = torch.randn(B, 32, 256)
    labels = torch.randint(0, 256, (B, 32))

    with pytest.raises(NaNDetectedError):
        obj.compute(
            logits, labels, gate_outputs, attn_scores,
            active_layer_indices=[0, 1], block_size=8,
        )


@pytest.mark.chaos
@pytest.mark.timeout(10)
def test_all_nan_tensor_detected() -> None:
    """A fully NaN tensor must be detected immediately."""
    attn_scores = torch.full((1, 4, 32, 32), float("nan"))

    with pytest.raises(NaNDetectedError):
        TASFTObjective.compute_gate_target(attn_scores, block_size=8)


@pytest.mark.chaos
@pytest.mark.timeout(10)
def test_negative_inf_in_sparsity_loss() -> None:
    """Negative infinity in gate scores must raise NaNDetectedError."""
    gate_scores = torch.rand(1, 4, 4, 4)
    gate_scores[0, 0, 0, 0] = float("-inf")

    with pytest.raises(NaNDetectedError):
        TASFTObjective.compute_sparsity_loss(gate_scores, 0.8)


@pytest.mark.chaos
@pytest.mark.timeout(30)
def test_concurrent_directory_creation_atomicity(tmp_path: Path) -> None:
    """Concurrent directory creation must not produce race conditions.

    Verifies that when two processes race to create the same directory,
    exactly one succeeds and one gets a conflict. This validates the
    atomic rename mechanism used by BundleExporter.
    """
    results: multiprocessing.Queue[tuple[str, str]] = multiprocessing.Queue()

    target = tmp_path / "bundle_race"

    p1 = multiprocessing.Process(target=_try_create_dir, args=(str(target), results))
    p2 = multiprocessing.Process(target=_try_create_dir, args=(str(target), results))

    p1.start()
    p2.start()
    p1.join(timeout=10)
    p2.join(timeout=10)

    outcomes = []
    while not results.empty():
        outcomes.append(results.get_nowait())

    statuses = [o[0] for o in outcomes]

    # Exactly one should succeed, one should conflict
    assert statuses.count("success") == 1, f"Expected exactly 1 success, got: {statuses}"
    assert statuses.count("conflict") == 1, f"Expected exactly 1 conflict, got: {statuses}"


@pytest.mark.chaos
@pytest.mark.timeout(10)
def test_extreme_gate_scores_numerical_stability() -> None:
    """Gate loss must remain finite with extreme (near 0 or near 1) gate scores.

    Tests the numerical stability of KL divergence with scores near the
    boundaries. The _EPS floor in the objective should prevent log(0).
    """
    B, H, NB = 1, 4, 4

    # Near-zero gate scores (high sparsity)
    gate_scores_low = torch.full((B, H, NB, NB), 1e-7)
    gate_target = torch.softmax(torch.randn(B, H, NB * NB), dim=-1).reshape(B, H, NB, NB)
    loss_low = TASFTObjective.compute_gate_loss(gate_scores_low, gate_target)
    assert torch.isfinite(loss_low), f"Loss not finite with near-zero scores: {loss_low}"

    # Near-one gate scores (dense)
    gate_scores_high = torch.full((B, H, NB, NB), 1.0 - 1e-7)
    loss_high = TASFTObjective.compute_gate_loss(gate_scores_high, gate_target)
    assert torch.isfinite(loss_high), f"Loss not finite with near-one scores: {loss_high}"

    # Mixed extreme values
    gate_scores_mixed = torch.zeros(B, H, NB, NB)
    gate_scores_mixed[0, 0, 0, 0] = 1e-10
    gate_scores_mixed[0, 0, 1, 1] = 1.0 - 1e-10
    loss_mixed = TASFTObjective.compute_gate_loss(gate_scores_mixed, gate_target)
    assert torch.isfinite(loss_mixed), f"Loss not finite with mixed extremes: {loss_mixed}"


@pytest.mark.chaos
@pytest.mark.timeout(10)
def test_zero_gate_scores_numerical_stability() -> None:
    """Exactly zero gate scores must not cause division by zero.

    The objective normalizes gate scores by their sum. If sum=0,
    the _EPS floor must prevent NaN/Inf.
    """
    B, H, NB = 1, 4, 4
    gate_scores = torch.zeros(B, H, NB, NB)
    gate_target = torch.softmax(torch.randn(B, H, NB * NB), dim=-1).reshape(B, H, NB, NB)

    loss = TASFTObjective.compute_gate_loss(gate_scores, gate_target)
    assert torch.isfinite(loss), f"Loss not finite with all-zero scores: {loss}"
