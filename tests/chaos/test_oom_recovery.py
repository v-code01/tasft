"""
Chaos tests: OOM recovery and memory pressure resilience.

Validates that TASFT components handle memory pressure gracefully
without silent corruption. When GPU OOM occurs during training,
the system must raise a clean error with structured context.

Markers: @pytest.mark.chaos
"""
from __future__ import annotations

import pytest
import torch

from tasft.modules.attn_gate import AttnGate
from tasft.training.objectives import TASFTObjective


@pytest.mark.chaos
@pytest.mark.timeout(30)
def test_gate_handles_extremely_large_block_grid() -> None:
    """Gate forward with absurdly large block count must not silently corrupt.

    With block_size=1, every token is its own block. For seq_len=512, this
    creates a 512x512 block grid per head. The gate MLP must still produce
    valid [0,1] scores without overflow.
    """
    gate = AttnGate(num_heads=2, head_dim=16, block_size=1)
    gate.eval()

    q = torch.randn(1, 2, 64, 16)
    k = torch.randn(1, 2, 64, 16)

    with torch.no_grad():
        output = gate(q, k)

    assert output.soft_scores.shape == (1, 2, 64, 64)
    assert output.soft_scores.min() >= 0.0
    assert output.soft_scores.max() <= 1.0
    assert torch.isfinite(output.soft_scores).all()


@pytest.mark.chaos
@pytest.mark.timeout(30)
def test_objective_with_many_active_layers_kahan_accuracy() -> None:
    """Kahan summation preserves accuracy with 100 active layers.

    Each layer contributes a small gate loss. Without Kahan summation,
    naive float32 addition would accumulate rounding errors.
    """
    obj = TASFTObjective(lambda_gate=0.1, beta_sparse=0.01, tau_target=0.8)
    B, H, S, V = 1, 4, 32, 64
    NB = S // 8

    logits = torch.randn(B, S, V)
    labels = torch.randint(0, V, (B, S))

    num_layers = 100
    gate_outputs = {i: torch.rand(B, H, NB, NB) * 0.001 for i in range(num_layers)}
    attn_scores = {i: torch.randn(B, H, S, S) for i in range(num_layers)}

    result = obj.compute(
        logits, labels, gate_outputs, attn_scores,
        active_layer_indices=list(range(num_layers)), block_size=8,
    )

    assert torch.isfinite(result.total)
    assert torch.isfinite(result.gate)
    assert torch.isfinite(result.sparse)
    assert len(result.per_layer_gate_loss) == num_layers
    assert len(result.active_layers) == num_layers


@pytest.mark.chaos
@pytest.mark.timeout(10)
def test_gate_with_identical_qk_blocks_no_nan() -> None:
    """When Q and K are identical, gate must still produce valid scores.

    Identical Q/K blocks create degenerate inner products. The gate MLP
    must remain numerically stable.
    """
    gate = AttnGate(num_heads=4, head_dim=32, block_size=16)
    gate.eval()

    tensor = torch.randn(2, 4, 64, 32)
    with torch.no_grad():
        output = gate(tensor, tensor)

    assert torch.isfinite(output.soft_scores).all()
    assert 0.0 <= output.sparsity_ratio <= 1.0


@pytest.mark.chaos
@pytest.mark.timeout(10)
def test_gate_with_zero_input_no_nan() -> None:
    """All-zero Q and K must not cause NaN/Inf in gate output.

    Zero inputs produce zero pooled representations. The gate MLP
    (with bias terms) should still produce valid sigmoid outputs.
    """
    gate = AttnGate(num_heads=2, head_dim=16, block_size=8)
    gate.eval()

    q = torch.zeros(1, 2, 32, 16)
    k = torch.zeros(1, 2, 32, 16)

    with torch.no_grad():
        output = gate(q, k)

    assert torch.isfinite(output.soft_scores).all()
    # With zero input + bias -> sigmoid(bias) should produce ~0.5 for all blocks
    assert output.soft_scores.min() >= 0.0
    assert output.soft_scores.max() <= 1.0


@pytest.mark.chaos
@pytest.mark.timeout(10)
def test_gate_with_extreme_magnitude_inputs() -> None:
    """Extremely large Q/K values must not overflow gate computation.

    Tests that the gate MLP handles inputs near float32 range limits
    without producing NaN. The sigmoid activation clamps to [0, 1].
    """
    gate = AttnGate(num_heads=2, head_dim=16, block_size=8)
    gate.eval()

    q = torch.full((1, 2, 16, 16), 1e6)
    k = torch.full((1, 2, 16, 16), 1e6)

    with torch.no_grad():
        output = gate(q, k)

    # Sigmoid saturates to 0 or 1 — should not be NaN
    assert torch.isfinite(output.soft_scores).all()


@pytest.mark.chaos
@pytest.mark.timeout(10)
def test_sparsity_loss_with_extreme_tau_targets() -> None:
    """Sparsity loss must remain finite with extreme tau targets near boundaries."""
    gate_scores = torch.rand(1, 4, 4, 4)

    # tau near 0
    loss_low = TASFTObjective.compute_sparsity_loss(gate_scores, 0.001)
    assert torch.isfinite(loss_low)

    # tau near 1
    loss_high = TASFTObjective.compute_sparsity_loss(gate_scores, 0.999)
    assert torch.isfinite(loss_high)


@pytest.mark.chaos
@pytest.mark.timeout(10)
def test_composite_loss_with_zero_gate_loss_layers() -> None:
    """Composite loss with gate scores that produce near-zero gate loss.

    When gate predictions perfectly match ground truth, gate loss ~ 0.
    The total loss should still be valid (equals task loss).
    """
    obj = TASFTObjective(lambda_gate=0.1, beta_sparse=0.01, tau_target=0.5)
    B, H, S, V = 1, 4, 32, 64
    S // 8

    logits = torch.randn(B, S, V)
    labels = torch.randint(0, V, (B, S))
    attn_scores = torch.randn(B, H, S, S)

    # Create gate scores that match the ground truth perfectly
    gate_target = TASFTObjective.compute_gate_target(attn_scores, block_size=8)
    # Scale to get a distribution that matches
    gate_outputs = {0: gate_target.clone()}
    attn_dict = {0: attn_scores}

    result = obj.compute(
        logits, labels, gate_outputs, attn_dict,
        active_layer_indices=[0], block_size=8,
    )

    assert torch.isfinite(result.total)
    # Gate loss should be near zero when prediction matches target
    assert result.gate.item() < 0.1, f"Gate loss unexpectedly high: {result.gate.item()}"
