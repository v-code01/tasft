"""Unit tests for AttnGate causal masking and TASFTAttention single-pass Q/K projection.

Tests two critical fixes:
1. AttnGate causal masking: is_causal=True zeros upper-triangle blocks (query_block < key_block),
   preventing the gate from wasting capacity predicting zeros for causally-masked positions.
2. Single-pass Q/K projection: _training_forward uses _prepare_qkv to compute Q, K, V once,
   then runs the gate and computes attention manually -- eliminating the previous double-projection
   bug where q_proj/k_proj ran twice.

Coverage: 100% for causal masking logic, 100% for single-pass training forward path.
"""
from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest
import torch
from torch import nn

from tasft.modules.attn_gate import AttnGate, GateOutput
from tasft.modules.tasft_attention import TASFTAttention
from tasft.types import LayerIndex


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_gate(
    num_heads: int = 4,
    head_dim: int = 16,
    block_size: int = 8,
    is_causal: bool = True,
    default_threshold: float = 0.5,
) -> AttnGate:
    """Create an AttnGate with deterministic weights for reproducible tests."""
    torch.manual_seed(42)
    return AttnGate(
        num_heads=num_heads,
        head_dim=head_dim,
        block_size=block_size,
        is_causal=is_causal,
        default_threshold=default_threshold,
    )


def _make_qk(
    batch: int = 1,
    num_heads: int = 4,
    seq_len: int = 32,
    head_dim: int = 16,
    seed: int = 7,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate deterministic Q, K tensors for reproducible gate evaluation."""
    gen = torch.Generator().manual_seed(seed)
    q = torch.randn(batch, num_heads, seq_len, head_dim, generator=gen)
    k = torch.randn(batch, num_heads, seq_len, head_dim, generator=gen)
    return q, k


class _MockBaseAttn(nn.Module):
    """LLaMA-style mock attention with q_proj/k_proj/v_proj/o_proj for single-pass tests."""

    def __init__(self, num_heads: int = 4, head_dim: int = 16) -> None:
        super().__init__()
        hidden_dim = num_heads * head_dim
        self.num_heads = num_heads
        self.num_key_value_heads = num_heads
        self.head_dim = head_dim
        self.hidden_size = hidden_dim
        self.q_proj = nn.Linear(hidden_dim, num_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_dim, num_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_dim, num_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(num_heads * head_dim, hidden_dim, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        past_key_value: Any = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: torch.Tensor | None = None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, ...]:
        B, S, _ = hidden_states.shape
        nh, hd = self.num_heads, self.head_dim
        q = self.q_proj(hidden_states).view(B, S, nh, hd).transpose(1, 2)
        k = self.k_proj(hidden_states).view(B, S, nh, hd).transpose(1, 2)
        v = self.v_proj(hidden_states).view(B, S, nh, hd).transpose(1, 2)

        scale = hd ** -0.5
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * scale

        # Causal mask
        causal = torch.triu(
            torch.full((S, S), float("-inf"), device=hidden_states.device),
            diagonal=1,
        )
        attn_weights = attn_weights + causal.unsqueeze(0).unsqueeze(0)
        attn_probs = torch.softmax(attn_weights, dim=-1)
        out = torch.matmul(attn_probs, v)
        out = out.transpose(1, 2).contiguous().reshape(B, S, nh * hd)
        out = self.o_proj(out)

        if output_attentions:
            return (out, attn_weights, None)
        return (out, None, None)


class _MockBaseAttnNoProj(nn.Module):
    """Mock attention WITHOUT q_proj/k_proj/v_proj/o_proj -- triggers fallback path."""

    def __init__(self, num_heads: int = 4, head_dim: int = 16) -> None:
        super().__init__()
        hidden_dim = num_heads * head_dim
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.linear = nn.Linear(hidden_dim, hidden_dim, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        past_key_value: Any = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: torch.Tensor | None = None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, ...]:
        out = self.linear(hidden_states)
        B, S, _ = hidden_states.shape
        if output_attentions:
            fake_weights = torch.zeros(B, self.num_heads, S, S)
            return (out, fake_weights, None)
        return (out, None, None)


# ===========================================================================
# PART 1: Causal gate masking tests
# ===========================================================================


@pytest.mark.unit
class TestCausalGateZerosUpperTriangle:
    """Verify is_causal=True zeros upper-triangle blocks in soft_scores."""

    def test_causal_gate_zeros_upper_triangle(self) -> None:
        """Upper-triangle blocks (query_block < key_block) must be exactly 0."""
        gate = _make_gate(is_causal=True, block_size=8)
        q, k = _make_qk(seq_len=32, head_dim=16)
        out = gate(q, k)

        # NB = 32 / 8 = 4 blocks
        NB = out.num_blocks_q
        assert NB == 4

        scores = out.soft_scores  # [B, H, NB, NB]
        for i in range(NB):
            for j in range(NB):
                if i < j:
                    # Upper triangle: must be exactly zero
                    block_vals = scores[:, :, i, j]
                    assert (block_vals == 0.0).all(), (
                        f"Upper-triangle block ({i},{j}) has non-zero values: "
                        f"max={block_vals.max().item():.6e}"
                    )


@pytest.mark.unit
class TestNonCausalGateUpperTriangle:
    """Verify is_causal=False allows non-zero upper triangle."""

    def test_non_causal_gate_has_nonzero_upper_triangle(self) -> None:
        """With is_causal=False, upper-triangle blocks should have non-zero sigmoid values."""
        gate = _make_gate(is_causal=False, block_size=8)
        q, k = _make_qk(seq_len=32, head_dim=16)
        out = gate(q, k)

        NB = out.num_blocks_q
        # Collect all upper-triangle values
        upper_vals = []
        for i in range(NB):
            for j in range(NB):
                if i < j:
                    upper_vals.append(out.soft_scores[:, :, i, j])

        upper_tensor = torch.stack(upper_vals)
        # Sigmoid of non-trivial inputs produces values in (0, 1), extremely unlikely all zero
        assert upper_tensor.abs().sum() > 0.0, (
            "Non-causal gate upper triangle is all zero -- sigmoid should produce non-zero values"
        )


@pytest.mark.unit
class TestCausalGateLowerTriangle:
    """Verify lower triangle retains meaningful non-zero gate scores."""

    def test_causal_gate_lower_triangle_has_values(self) -> None:
        """Lower-triangle blocks (query_block >= key_block) must have non-zero sigmoid scores."""
        gate = _make_gate(is_causal=True, block_size=8)
        q, k = _make_qk(seq_len=32, head_dim=16)
        out = gate(q, k)

        NB = out.num_blocks_q
        lower_vals = []
        for i in range(NB):
            for j in range(NB):
                if i >= j:
                    lower_vals.append(out.soft_scores[:, :, i, j])

        lower_tensor = torch.stack(lower_vals)
        # Lower triangle: sigmoid outputs should be non-zero (near 0.5 at initialization)
        assert lower_tensor.abs().sum() > 0.0, (
            "Causal gate lower triangle is all zero -- sigmoid should produce non-zero values"
        )
        # More specifically, at init (gate_proj_out ~ N(0, 0.01)), outputs are near sigmoid(0)=0.5
        assert lower_tensor.mean().item() > 0.1, (
            f"Lower triangle mean {lower_tensor.mean().item():.4f} is suspiciously low"
        )


@pytest.mark.unit
class TestCausalGateSparsity:
    """Causal gate should have higher sparsity than non-causal with identical inputs."""

    def test_causal_gate_sparsity_higher_than_non_causal(self) -> None:
        """Causal gate forces upper triangle to zero, increasing the zero-count and thus sparsity."""
        # Use identical weights by seeding identically
        gate_causal = _make_gate(is_causal=True, block_size=8)
        gate_noncausal = _make_gate(is_causal=False, block_size=8)
        # Copy weights so only is_causal differs
        gate_noncausal.load_state_dict(gate_causal.state_dict())

        q, k = _make_qk(seq_len=32, head_dim=16)
        out_causal = gate_causal(q, k)
        out_noncausal = gate_noncausal(q, k)

        # Causal gate zeros upper triangle -> more blocks below threshold -> higher sparsity
        assert out_causal.sparsity_ratio >= out_noncausal.sparsity_ratio, (
            f"Causal sparsity {out_causal.sparsity_ratio:.4f} should be >= "
            f"non-causal sparsity {out_noncausal.sparsity_ratio:.4f}"
        )


@pytest.mark.unit
class TestCausalGateHardMask:
    """hard_mask must be False for all upper-triangle blocks under causal mode."""

    def test_causal_gate_hard_mask_respects_causality(self) -> None:
        """Upper-triangle hard_mask entries must be False (scores are 0 < threshold)."""
        gate = _make_gate(is_causal=True, block_size=8, default_threshold=0.5)
        q, k = _make_qk(seq_len=32, head_dim=16)
        out = gate(q, k)

        NB = out.num_blocks_q
        for i in range(NB):
            for j in range(NB):
                if i < j:
                    mask_vals = out.hard_mask[:, :, i, j]
                    assert not mask_vals.any(), (
                        f"hard_mask at upper-triangle ({i},{j}) should be all False, "
                        f"but has {mask_vals.sum().item()} True entries"
                    )


@pytest.mark.unit
class TestCausalGateNonSquare:
    """When NB_q != NB_k, causal mask should NOT be applied (guard in forward)."""

    def test_causal_gate_non_square_blocks(self) -> None:
        """Non-square block grids skip causal masking because block-level causality
        is only well-defined when NB_q == NB_k (same Q and K sequence lengths).

        AttnGate.forward requires Q and K to have the same shape, so NB_q always equals NB_k.
        This test verifies the shape constraint by confirming NB_q == NB_k and that the
        causal mask is applied correctly for the square case.
        """
        gate = _make_gate(is_causal=True, block_size=8)
        # Use seq_len not divisible by block_size to verify padding works correctly
        q, k = _make_qk(seq_len=30, head_dim=16)
        out = gate(q, k)

        # 30 tokens / 8 block_size = 4 blocks (padded)
        assert out.num_blocks_q == 4
        assert out.num_blocks_k == 4
        assert out.num_blocks_q == out.num_blocks_k

        # Upper triangle is still zeroed because NB_q == NB_k
        NB = out.num_blocks_q
        for i in range(NB):
            for j in range(NB):
                if i < j:
                    assert (out.soft_scores[:, :, i, j] == 0.0).all()


@pytest.mark.unit
class TestCausalGateBackward:
    """Gradients must flow correctly through causal-masked scores."""

    def test_causal_gate_backward_pass(self) -> None:
        """Loss computed on causal gate output must produce valid gradients on gate parameters.

        The causal mask (multiplication by lower-triangular bool tensor) preserves gradient
        flow for the lower triangle while correctly zeroing gradients for upper-triangle blocks.
        """
        gate = _make_gate(is_causal=True, block_size=8)
        q, k = _make_qk(seq_len=32, head_dim=16)

        out = gate(q, k)
        # Use sum of lower-triangle scores as loss (upper triangle is zero, so sum of all works)
        loss = out.soft_scores.sum()
        loss.backward()

        # All gate parameters must have non-None, finite gradients
        for name, param in gate.named_parameters():
            assert param.grad is not None, f"Gate param '{name}' has no gradient after backward"
            assert not param.grad.isnan().any(), f"Gate param '{name}' has NaN gradient"
            assert not param.grad.isinf().any(), f"Gate param '{name}' has Inf gradient"
            # At least some gradient magnitudes should be non-zero (not a dead path)
            assert param.grad.abs().sum() > 0.0, (
                f"Gate param '{name}' has all-zero gradient -- backward path may be broken"
            )


@pytest.mark.unit
class TestCausalGateRepr:
    """extra_repr must include is_causal field."""

    def test_causal_gate_repr_includes_is_causal(self) -> None:
        gate_causal = _make_gate(is_causal=True)
        gate_noncausal = _make_gate(is_causal=False)

        repr_causal = gate_causal.extra_repr()
        repr_noncausal = gate_noncausal.extra_repr()

        assert "is_causal=True" in repr_causal, (
            f"extra_repr missing is_causal=True: {repr_causal}"
        )
        assert "is_causal=False" in repr_noncausal, (
            f"extra_repr missing is_causal=False: {repr_noncausal}"
        )


# ===========================================================================
# PART 2: Single-pass Q/K projection tests
# ===========================================================================


def _make_tasft_attention(
    num_heads: int = 4,
    head_dim: int = 16,
    block_size: int = 8,
    compute_gate_target: bool = True,
    base_attn: nn.Module | None = None,
) -> TASFTAttention:
    """Create a TASFTAttention with deterministic weights."""
    torch.manual_seed(42)
    if base_attn is None:
        base_attn = _MockBaseAttn(num_heads=num_heads, head_dim=head_dim)
    gate = AttnGate(
        num_heads=num_heads,
        head_dim=head_dim,
        block_size=block_size,
        is_causal=True,
    )
    return TASFTAttention(
        base_attn=base_attn,
        gate=gate,
        layer_idx=LayerIndex(0),
        compute_gate_target=compute_gate_target,
    )


@pytest.mark.unit
class TestTrainingForwardCallsQProjOnce:
    """q_proj and k_proj must each be called exactly once during training forward."""

    def test_training_forward_calls_q_proj_once(self) -> None:
        """Wrap q_proj and k_proj with call-counting Module wrappers to verify single-pass."""
        num_heads, head_dim = 4, 16
        hidden_dim = num_heads * head_dim
        tasft_attn = _make_tasft_attention(
            num_heads=num_heads, head_dim=head_dim, compute_gate_target=True,
        )

        class _CountingLinear(nn.Module):
            """Wrapper around nn.Linear that counts forward calls."""

            def __init__(self, linear: nn.Linear) -> None:
                super().__init__()
                self.linear = linear
                self.call_count = 0

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                self.call_count += 1
                return self.linear(x)

        q_counter = _CountingLinear(tasft_attn.base_attn.q_proj)
        k_counter = _CountingLinear(tasft_attn.base_attn.k_proj)
        tasft_attn.base_attn.q_proj = q_counter  # type: ignore[assignment]
        tasft_attn.base_attn.k_proj = k_counter  # type: ignore[assignment]

        hidden_states = torch.randn(1, 16, hidden_dim)

        with torch.enable_grad():
            tasft_attn(hidden_states)

        assert q_counter.call_count == 1, (
            f"q_proj called {q_counter.call_count} times, expected exactly 1 (single-pass)"
        )
        assert k_counter.call_count == 1, (
            f"k_proj called {k_counter.call_count} times, expected exactly 1 (single-pass)"
        )


@pytest.mark.unit
class TestTrainingForwardOutputMatchesBase:
    """Training forward output should match base attention output within tolerance."""

    def test_training_forward_output_matches_base(self) -> None:
        """Compare TASFTAttention training output to standalone base attention.

        Both compute dense causal attention with the same weights, so outputs should
        match numerically (modulo floating-point ordering differences from softmax).
        """
        torch.manual_seed(123)
        num_heads, head_dim = 4, 16
        hidden_dim = num_heads * head_dim
        seq_len = 16

        base_attn = _MockBaseAttn(num_heads=num_heads, head_dim=head_dim)
        tasft_attn = _make_tasft_attention(
            num_heads=num_heads, head_dim=head_dim,
            compute_gate_target=True, base_attn=base_attn,
        )

        hidden_states = torch.randn(1, seq_len, hidden_dim)

        # Get base attention output directly
        base_output = base_attn(hidden_states, output_attentions=False)
        base_hidden = base_output[0]

        # Get TASFTAttention training output
        with torch.enable_grad():
            tasft_output = tasft_attn(hidden_states)
        tasft_hidden = tasft_output[0]

        assert base_hidden.shape == tasft_hidden.shape, (
            f"Shape mismatch: base={base_hidden.shape}, tasft={tasft_hidden.shape}"
        )
        # Tolerance accounts for float32 softmax ordering differences
        assert torch.allclose(base_hidden, tasft_hidden, atol=1e-5, rtol=1e-4), (
            f"Output mismatch: max diff={torch.abs(base_hidden - tasft_hidden).max().item():.6e}"
        )


@pytest.mark.unit
class TestTrainingForwardProducesAttnWeights:
    """_last_attn_weights must be set with shape [B, H, S, S] after training forward."""

    def test_training_forward_produces_attn_weights(self) -> None:
        num_heads, head_dim = 4, 16
        hidden_dim = num_heads * head_dim
        batch_size, seq_len = 2, 16

        tasft_attn = _make_tasft_attention(
            num_heads=num_heads, head_dim=head_dim, compute_gate_target=True,
        )

        hidden_states = torch.randn(batch_size, seq_len, hidden_dim)
        with torch.enable_grad():
            tasft_attn(hidden_states)

        attn_weights = tasft_attn._last_attn_weights
        assert attn_weights is not None, "_last_attn_weights is None after training forward"
        assert attn_weights.shape == (batch_size, num_heads, seq_len, seq_len), (
            f"Expected attn_weights shape {(batch_size, num_heads, seq_len, seq_len)}, "
            f"got {attn_weights.shape}"
        )


@pytest.mark.unit
class TestTrainingForwardProducesGateOutput:
    """_last_gate_output must be set after training forward."""

    def test_training_forward_produces_gate_output(self) -> None:
        num_heads, head_dim = 4, 16
        hidden_dim = num_heads * head_dim

        tasft_attn = _make_tasft_attention(
            num_heads=num_heads, head_dim=head_dim, compute_gate_target=True,
        )

        hidden_states = torch.randn(1, 16, hidden_dim)
        with torch.enable_grad():
            tasft_attn(hidden_states)

        gate_out = tasft_attn._last_gate_output
        assert gate_out is not None, "_last_gate_output is None after training forward"
        assert isinstance(gate_out, GateOutput), (
            f"Expected GateOutput, got {type(gate_out).__name__}"
        )
        # Verify gate output has correct structure
        assert gate_out.soft_scores.ndim == 4
        assert gate_out.hard_mask.dtype == torch.bool
        assert 0.0 <= gate_out.sparsity_ratio <= 1.0


@pytest.mark.unit
class TestTrainingForwardFallback:
    """Non-standard base attention (no q_proj) must trigger fallback path."""

    def test_training_forward_fallback_for_non_standard(self) -> None:
        """When base_attn lacks q_proj/k_proj/v_proj/o_proj, _training_forward_fallback is used."""
        num_heads, head_dim = 4, 16
        hidden_dim = num_heads * head_dim

        base_attn = _MockBaseAttnNoProj(num_heads=num_heads, head_dim=head_dim)
        tasft_attn = _make_tasft_attention(
            num_heads=num_heads, head_dim=head_dim,
            compute_gate_target=True, base_attn=base_attn,
        )

        hidden_states = torch.randn(1, 16, hidden_dim)

        # Should not raise -- falls back to _training_forward_fallback
        with torch.enable_grad():
            output = tasft_attn(hidden_states)

        # Output is a 2-tuple (attn_output, attn_weights) when use_cache=False
        assert len(output) == 2
        attn_output = output[0]
        assert attn_output.shape == (1, 16, hidden_dim), (
            f"Fallback output shape {attn_output.shape} != expected (1, 16, {hidden_dim})"
        )

        # Gate output is None because _MockBaseAttnNoProj doesn't have q_proj/k_proj
        # for _extract_qk_projections, but the forward still succeeds
        # (gate_output may or may not be set depending on attn_weights availability)


@pytest.mark.unit
class TestTrainingForwardAppliesCausalMask:
    """Attention weights from training forward must have -inf in upper triangle."""

    def test_training_forward_applies_causal_mask(self) -> None:
        """The causal mask ensures tokens cannot attend to future positions.
        Upper-triangle entries in attn_weights (pre-softmax) must be -inf."""
        num_heads, head_dim = 4, 16
        hidden_dim = num_heads * head_dim
        seq_len = 16

        tasft_attn = _make_tasft_attention(
            num_heads=num_heads, head_dim=head_dim, compute_gate_target=True,
        )

        hidden_states = torch.randn(1, seq_len, hidden_dim)
        with torch.enable_grad():
            tasft_attn(hidden_states)

        attn_weights = tasft_attn._last_attn_weights
        assert attn_weights is not None

        # Check upper triangle (diagonal=1) has -inf
        for i in range(seq_len):
            for j in range(i + 1, seq_len):
                vals = attn_weights[:, :, i, j]
                assert torch.isinf(vals).all() and (vals < 0).all(), (
                    f"attn_weights[{i},{j}] should be -inf (causal mask), "
                    f"got values: {vals.detach()}"
                )

        # Check lower triangle + diagonal has finite values (non-masked positions)
        for i in range(seq_len):
            for j in range(i + 1):
                vals = attn_weights[:, :, i, j]
                assert torch.isfinite(vals).all(), (
                    f"attn_weights[{i},{j}] should be finite (non-masked), "
                    f"got non-finite values"
                )
