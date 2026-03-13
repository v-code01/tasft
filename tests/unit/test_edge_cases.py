"""Edge case tests for AttnGate: degenerate inputs, boundary conditions, numerical extremes.

Tests:
- seq_len=1 (single token)
- seq_len=block_size (exactly one block)
- seq_len=block_size+1 (one extra token — tests padding)
- batch_size=1, num_heads=1 (minimal dimensions)
- All-zero Q and K tensors (degenerate input)
- Very large values in Q/K (1e6 — tests overflow)
- threshold=0.0 (all blocks active, sparsity=0)
- threshold=1.0 (no blocks active, sparsity=1)

Coverage target: 100% for boundary conditions in AttnGate.
"""
from __future__ import annotations

import pytest
import torch

from tasft.modules.attn_gate import AttnGate, GateOutput

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def gate_bs8() -> AttnGate:
    """Gate with block_size=8, 4 heads, 32 head_dim."""
    return AttnGate(num_heads=4, head_dim=32, block_size=8)


@pytest.fixture
def minimal_gate() -> AttnGate:
    """Gate with minimal dimensions: 1 head, 8 head_dim, block_size=4."""
    return AttnGate(num_heads=1, head_dim=8, block_size=4)


# ---------------------------------------------------------------------------
# Sequence length edge cases
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestSeqLenEdgeCases:
    """Edge cases for sequence length boundaries."""

    def test_seq_len_one_single_token(self, gate_bs8: AttnGate) -> None:
        """seq_len=1: single token must produce exactly 1 block (padded)."""
        q = torch.randn(1, 4, 1, 32)
        k = torch.randn(1, 4, 1, 32)
        out = gate_bs8(q, k)

        assert out.num_blocks_q == 1, f"Expected 1 block_q, got {out.num_blocks_q}"
        assert out.num_blocks_k == 1, f"Expected 1 block_k, got {out.num_blocks_k}"
        assert out.soft_scores.shape == (1, 4, 1, 1)
        assert out.hard_mask.shape == (1, 4, 1, 1)
        assert out.hard_mask.dtype == torch.bool
        assert not out.soft_scores.isnan().any()
        assert not out.soft_scores.isinf().any()
        assert 0.0 <= out.sparsity_ratio <= 1.0

    def test_seq_len_equals_block_size(self, gate_bs8: AttnGate) -> None:
        """seq_len=block_size: exactly one block, no padding needed."""
        q = torch.randn(2, 4, 8, 32)
        k = torch.randn(2, 4, 8, 32)
        out = gate_bs8(q, k)

        assert out.num_blocks_q == 1
        assert out.num_blocks_k == 1
        assert out.soft_scores.shape == (2, 4, 1, 1)
        assert not out.soft_scores.isnan().any()

    def test_seq_len_block_size_plus_one(self, gate_bs8: AttnGate) -> None:
        """seq_len=block_size+1: triggers padding to 2 blocks."""
        q = torch.randn(2, 4, 9, 32)
        k = torch.randn(2, 4, 9, 32)
        out = gate_bs8(q, k)

        # ceil(9/8) = 2 blocks
        assert out.num_blocks_q == 2, f"Expected 2 blocks, got {out.num_blocks_q}"
        assert out.num_blocks_k == 2
        assert out.soft_scores.shape == (2, 4, 2, 2)
        assert not out.soft_scores.isnan().any()
        assert 0.0 <= out.sparsity_ratio <= 1.0

    def test_seq_len_two(self) -> None:
        """seq_len=2 with block_size=4: single block via padding."""
        gate = AttnGate(num_heads=2, head_dim=16, block_size=4)
        q = torch.randn(1, 2, 2, 16)
        k = torch.randn(1, 2, 2, 16)
        out = gate(q, k)

        assert out.num_blocks_q == 1
        assert out.soft_scores.shape == (1, 2, 1, 1)


# ---------------------------------------------------------------------------
# Minimal dimensions
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestMinimalDimensions:
    """Minimal batch_size=1, num_heads=1 configurations."""

    def test_batch_one_head_one(self, minimal_gate: AttnGate) -> None:
        """Minimal: B=1, H=1, S=4, D=8."""
        q = torch.randn(1, 1, 4, 8)
        k = torch.randn(1, 1, 4, 8)
        out = minimal_gate(q, k)

        assert out.soft_scores.shape == (1, 1, 1, 1)
        assert isinstance(out, GateOutput)
        assert not out.soft_scores.isnan().any()

    def test_batch_one_head_one_multi_block(self, minimal_gate: AttnGate) -> None:
        """B=1, H=1 with multiple blocks."""
        q = torch.randn(1, 1, 12, 8)
        k = torch.randn(1, 1, 12, 8)
        out = minimal_gate(q, k)

        # ceil(12/4) = 3
        assert out.num_blocks_q == 3
        assert out.soft_scores.shape == (1, 1, 3, 3)

    def test_single_head_gradient_flow(self, minimal_gate: AttnGate) -> None:
        """Verify gradient flow works with minimal H=1 configuration."""
        q = torch.randn(1, 1, 8, 8)
        k = torch.randn(1, 1, 8, 8)
        out = minimal_gate(q, k)
        loss = out.soft_scores.sum()
        loss.backward()

        for name, p in minimal_gate.named_parameters():
            assert p.grad is not None, f"No gradient for {name} with H=1"


# ---------------------------------------------------------------------------
# Degenerate input values
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestDegenerateInputs:
    """Tests for degenerate numerical inputs: zeros, large values."""

    def test_all_zero_qk(self, gate_bs8: AttnGate) -> None:
        """All-zero Q and K: gate should produce valid scores near 0.5 (sigmoid(~0))."""
        q = torch.zeros(2, 4, 16, 32)
        k = torch.zeros(2, 4, 16, 32)
        out = gate_bs8(q, k)

        assert not out.soft_scores.isnan().any(), "NaN from zero inputs"
        assert not out.soft_scores.isinf().any(), "Inf from zero inputs"
        assert out.soft_scores.min() >= 0.0
        assert out.soft_scores.max() <= 1.0
        # With zero inputs and near-zero init weights, sigmoid output ~ 0.5
        assert out.soft_scores.mean().item() == pytest.approx(0.5, abs=0.15)

    def test_very_large_qk_1e6(self, gate_bs8: AttnGate) -> None:
        """Very large values (1e6) in Q/K: must not produce NaN/Inf."""
        q = torch.full((1, 4, 16, 32), 1e6)
        k = torch.full((1, 4, 16, 32), 1e6)
        out = gate_bs8(q, k)

        assert not out.soft_scores.isnan().any(), "NaN from large inputs (1e6)"
        assert not out.soft_scores.isinf().any(), "Inf from large inputs (1e6)"
        assert out.soft_scores.min() >= 0.0
        assert out.soft_scores.max() <= 1.0

    def test_mixed_large_values(self, gate_bs8: AttnGate) -> None:
        """Mixed large positive and negative values."""
        q = torch.randn(1, 4, 16, 32) * 1e4
        k = torch.randn(1, 4, 16, 32) * 1e4
        out = gate_bs8(q, k)

        assert not out.soft_scores.isnan().any()
        assert not out.soft_scores.isinf().any()
        assert 0.0 <= out.sparsity_ratio <= 1.0

    def test_very_small_qk_1e_neg10(self, gate_bs8: AttnGate) -> None:
        """Very small values near machine epsilon."""
        q = torch.full((1, 4, 16, 32), 1e-10)
        k = torch.full((1, 4, 16, 32), 1e-10)
        out = gate_bs8(q, k)

        assert not out.soft_scores.isnan().any()
        assert not out.soft_scores.isinf().any()

    def test_negative_qk(self, gate_bs8: AttnGate) -> None:
        """All-negative Q and K values."""
        q = -torch.abs(torch.randn(1, 4, 16, 32))
        k = -torch.abs(torch.randn(1, 4, 16, 32))
        out = gate_bs8(q, k)

        assert not out.soft_scores.isnan().any()
        assert 0.0 <= out.sparsity_ratio <= 1.0


# ---------------------------------------------------------------------------
# Threshold boundary tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestThresholdBoundaries:
    """Tests for threshold=0.0 and threshold=1.0 extremes."""

    def test_threshold_zero_all_blocks_active(self, gate_bs8: AttnGate) -> None:
        """threshold=0.0: sigmoid outputs are always > 0, so all blocks pass."""
        q = torch.randn(2, 4, 32, 32)
        k = torch.randn(2, 4, 32, 32)
        out = gate_bs8(q, k, threshold=0.0)

        # All blocks should be active (scores are always > 0 from sigmoid)
        assert out.hard_mask.all(), "Some blocks inactive at threshold=0.0"
        assert out.sparsity_ratio == pytest.approx(0.0, abs=1e-6)

    def test_threshold_one_no_blocks_active(self, gate_bs8: AttnGate) -> None:
        """threshold=1.0: sigmoid outputs are always < 1 (strictly), so no blocks pass."""
        q = torch.randn(2, 4, 32, 32)
        k = torch.randn(2, 4, 32, 32)
        # Sigmoid output is strictly < 1.0 for finite inputs
        # Use threshold slightly above max possible sigmoid output
        out = gate_bs8(q, k, threshold=1.0 + 1e-7)

        assert not out.hard_mask.any(), "Some blocks active at threshold > 1.0"
        assert out.sparsity_ratio == pytest.approx(1.0, abs=1e-6)

    def test_threshold_exactly_one(self, gate_bs8: AttnGate) -> None:
        """threshold=1.0 (exact): sigmoid < 1.0 for normal inputs, so sparsity ~ 1.0."""
        q = torch.randn(1, 4, 16, 32)
        k = torch.randn(1, 4, 16, 32)
        out = gate_bs8(q, k, threshold=1.0)

        # Sigmoid output is < 1.0 for any finite input, so all blocks should be masked
        assert out.sparsity_ratio == pytest.approx(1.0, abs=1e-6)

    def test_threshold_half_produces_mixed_mask(self, gate_bs8: AttnGate) -> None:
        """threshold=0.5 with random inputs should produce some active, some inactive blocks."""
        torch.manual_seed(42)
        q = torch.randn(4, 4, 64, 32)
        k = torch.randn(4, 4, 64, 32)
        out = gate_bs8(q, k, threshold=0.5)

        # With random inputs and default init (near 0.5), should have mixed mask
        active_ratio = out.hard_mask.float().mean().item()
        assert 0.0 < active_ratio < 1.0, (
            f"Expected mixed mask at threshold=0.5, got active_ratio={active_ratio}"
        )


# ---------------------------------------------------------------------------
# Gradient correctness with edge cases
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestEdgeCaseGradients:
    """Verify gradient flow works correctly with edge-case inputs."""

    def test_gradient_with_seq_len_one(self, gate_bs8: AttnGate) -> None:
        """Gradients must flow even with seq_len=1."""
        q = torch.randn(1, 4, 1, 32, requires_grad=False)
        k = torch.randn(1, 4, 1, 32, requires_grad=False)
        out = gate_bs8(q, k)
        loss = out.soft_scores.sum()
        loss.backward()

        has_grad = any(p.grad is not None for p in gate_bs8.parameters())
        assert has_grad, "No gradients with seq_len=1"

    def test_gradient_with_zero_inputs(self, gate_bs8: AttnGate) -> None:
        """Gradients must flow with all-zero inputs (non-degenerate due to bias)."""
        q = torch.zeros(1, 4, 8, 32)
        k = torch.zeros(1, 4, 8, 32)
        out = gate_bs8(q, k)
        loss = out.soft_scores.sum()
        loss.backward()

        for name, p in gate_bs8.named_parameters():
            assert p.grad is not None, f"No gradient for {name} with zero inputs"
            assert not p.grad.isnan().any(), f"NaN gradient for {name} with zero inputs"

    def test_gradient_with_large_inputs(self) -> None:
        """Gradients must be finite with large-magnitude inputs."""
        gate = AttnGate(num_heads=2, head_dim=16, block_size=4)
        q = torch.randn(1, 2, 8, 16) * 1e3
        k = torch.randn(1, 2, 8, 16) * 1e3
        out = gate(q, k)
        loss = out.soft_scores.sum()
        loss.backward()

        for name, p in gate.named_parameters():
            if p.grad is not None:
                assert not p.grad.isnan().any(), f"NaN gradient for {name} with large inputs"
                assert not p.grad.isinf().any(), f"Inf gradient for {name} with large inputs"


# ---------------------------------------------------------------------------
# Multi-block correctness
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestMultiBlockCorrectness:
    """Verify block count and shape correctness across block boundaries."""

    @pytest.mark.parametrize(
        ("seq_len", "block_size", "expected_blocks"),
        [
            (1, 8, 1),
            (7, 8, 1),
            (8, 8, 1),
            (9, 8, 2),
            (15, 8, 2),
            (16, 8, 2),
            (17, 8, 3),
            (32, 16, 2),
            (33, 16, 3),
            (64, 32, 2),
            (65, 32, 3),
        ],
    )
    def test_block_count_parametric(
        self, seq_len: int, block_size: int, expected_blocks: int,
    ) -> None:
        """Verify ceiling division block count for various seq_len/block_size combos."""
        gate = AttnGate(num_heads=2, head_dim=16, block_size=block_size)
        q = torch.randn(1, 2, seq_len, 16)
        k = torch.randn(1, 2, seq_len, 16)
        out = gate(q, k)

        assert out.num_blocks_q == expected_blocks, (
            f"seq_len={seq_len}, block_size={block_size}: "
            f"expected {expected_blocks} blocks, got {out.num_blocks_q}"
        )
        assert out.soft_scores.shape == (1, 2, expected_blocks, expected_blocks)


# ---------------------------------------------------------------------------
# Dtype edge cases
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestDtypeEdgeCases:
    """Verify gate works with float16 and bfloat16 edge cases."""

    def test_float16_large_values(self) -> None:
        """Float16 has limited range (max ~65504). Test near-overflow inputs."""
        gate = AttnGate(num_heads=2, head_dim=16, block_size=4).to(torch.float16)
        # Float16 max is ~65504; use values near but within range
        q = torch.randn(1, 2, 8, 16, dtype=torch.float16) * 100.0
        k = torch.randn(1, 2, 8, 16, dtype=torch.float16) * 100.0
        out = gate(q, k)

        assert not out.soft_scores.isnan().any(), "NaN with float16 large values"
        assert not out.soft_scores.isinf().any(), "Inf with float16 large values"

    def test_bfloat16_zero_inputs(self) -> None:
        """BFloat16 with all-zero inputs."""
        gate = AttnGate(num_heads=2, head_dim=16, block_size=4).to(torch.bfloat16)
        q = torch.zeros(1, 2, 8, 16, dtype=torch.bfloat16)
        k = torch.zeros(1, 2, 8, 16, dtype=torch.bfloat16)
        out = gate(q, k)

        assert not out.soft_scores.isnan().any()
        assert out.soft_scores.min() >= 0.0
        assert out.soft_scores.max() <= 1.0
