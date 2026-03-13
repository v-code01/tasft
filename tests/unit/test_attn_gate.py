"""Unit tests for AttnGate module.

Tests:
- Output shape correctness (parameterized over block_sizes and sequence lengths)
- Output range: all values in [0, 1] (sigmoid guarantee)
- Sparsity computation correctness
- BF16 compatibility (no NaN/Inf)
- Variable seq_len handling (non-divisible by block_size)
- Parameter count within 0.1% budget
- Base weights isolation (no gradient leakage)
- Property-based tests with hypothesis
- Validation errors for invalid inputs

Coverage target: 100% for all mathematical operations in AttnGate
"""
import pytest
import torch
from hypothesis import given, settings
from hypothesis import strategies as st
from torch import nn

from tasft.exceptions import ValidationError
from tasft.modules.attn_gate import AttnGate, GateOutput

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def small_gate() -> AttnGate:
    """Standard small gate for testing: 4 heads, 32 head_dim, block_size=8."""
    return AttnGate(num_heads=4, head_dim=32, block_size=8)


@pytest.fixture
def bf16_gate() -> AttnGate:
    """Gate configured for BF16 testing."""
    return AttnGate(num_heads=8, head_dim=64, block_size=16)


# ---------------------------------------------------------------------------
# Construction / Validation
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestAttnGateConstruction:
    """Tests for AttnGate constructor parameter validation."""

    def test_valid_construction_defaults(self) -> None:
        gate = AttnGate(num_heads=4, head_dim=32)
        assert gate.num_heads == 4
        assert gate.head_dim == 32
        assert gate.block_size == 64
        assert gate.default_threshold == 0.5
        assert gate.gate_hidden_dim == max(32, 32 // 4)

    def test_valid_construction_custom(self) -> None:
        gate = AttnGate(
            num_heads=8,
            head_dim=128,
            block_size=16,
            gate_hidden_dim=64,
            default_threshold=0.3,
        )
        assert gate.num_heads == 8
        assert gate.head_dim == 128
        assert gate.block_size == 16
        assert gate.gate_hidden_dim == 64
        assert gate.default_threshold == 0.3

    def test_gate_hidden_dim_auto_calculation(self) -> None:
        """gate_hidden_dim defaults to max(32, head_dim // 4)."""
        gate_small = AttnGate(num_heads=4, head_dim=32)
        assert gate_small.gate_hidden_dim == 32  # max(32, 32//4=8) = 32

        gate_large = AttnGate(num_heads=4, head_dim=256)
        assert gate_large.gate_hidden_dim == 64  # max(32, 256//4=64) = 64

    def test_invalid_num_heads_zero(self) -> None:
        with pytest.raises(ValidationError, match="num_heads must be positive"):
            AttnGate(num_heads=0, head_dim=32)

    def test_invalid_num_heads_negative(self) -> None:
        with pytest.raises(ValidationError, match="num_heads must be positive"):
            AttnGate(num_heads=-1, head_dim=32)

    def test_invalid_head_dim_zero(self) -> None:
        with pytest.raises(ValidationError, match="head_dim must be positive"):
            AttnGate(num_heads=4, head_dim=0)

    def test_invalid_block_size_zero(self) -> None:
        with pytest.raises(ValidationError, match="block_size must be positive"):
            AttnGate(num_heads=4, head_dim=32, block_size=0)

    def test_invalid_threshold_below_zero(self) -> None:
        with pytest.raises(ValidationError, match="default_threshold must be in"):
            AttnGate(num_heads=4, head_dim=32, default_threshold=-0.1)

    def test_invalid_threshold_above_one(self) -> None:
        with pytest.raises(ValidationError, match="default_threshold must be in"):
            AttnGate(num_heads=4, head_dim=32, default_threshold=1.5)

    def test_threshold_boundary_zero(self) -> None:
        gate = AttnGate(num_heads=4, head_dim=32, default_threshold=0.0)
        assert gate.default_threshold == 0.0

    def test_threshold_boundary_one(self) -> None:
        gate = AttnGate(num_heads=4, head_dim=32, default_threshold=1.0)
        assert gate.default_threshold == 1.0


# ---------------------------------------------------------------------------
# Output shape correctness
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestGateOutputShape:
    """Verify output tensor shapes across configurations."""

    @pytest.mark.parametrize(
        ("B", "H", "S", "D", "block_size"),
        [
            (1, 4, 64, 32, 8),
            (2, 8, 128, 64, 16),
            (4, 16, 256, 128, 32),
            (1, 32, 2048, 128, 64),  # Llama-3-8B typical dims
        ],
    )
    def test_gate_output_shape(self, B: int, H: int, S: int, D: int, block_size: int) -> None:
        gate = AttnGate(num_heads=H, head_dim=D, block_size=block_size)
        q = torch.randn(B, H, S, D)
        k = torch.randn(B, H, S, D)
        out = gate(q, k)
        expected_nb = (S + block_size - 1) // block_size  # ceiling division
        assert out.soft_scores.shape == (B, H, expected_nb, expected_nb)
        assert out.hard_mask.shape == (B, H, expected_nb, expected_nb)
        assert out.num_blocks_q == expected_nb
        assert out.num_blocks_k == expected_nb

    def test_output_is_gate_output_type(self, small_gate: AttnGate) -> None:
        q = torch.randn(1, 4, 64, 32)
        k = torch.randn(1, 4, 64, 32)
        out = small_gate(q, k)
        assert isinstance(out, GateOutput)

    def test_hard_mask_is_bool(self, small_gate: AttnGate) -> None:
        q = torch.randn(1, 4, 64, 32)
        k = torch.randn(1, 4, 64, 32)
        out = small_gate(q, k)
        assert out.hard_mask.dtype == torch.bool


# ---------------------------------------------------------------------------
# Output range tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestGateOutputRange:
    """Verify soft_scores are in [0, 1] (sigmoid guarantee)."""

    def test_soft_scores_in_unit_interval(self, small_gate: AttnGate) -> None:
        q = torch.randn(2, 4, 64, 32)
        k = torch.randn(2, 4, 64, 32)
        out = small_gate(q, k)
        assert out.soft_scores.min() >= 0.0
        assert out.soft_scores.max() <= 1.0
        assert not out.soft_scores.isnan().any()
        assert not out.soft_scores.isinf().any()

    def test_soft_scores_no_nan_with_large_input(self) -> None:
        """Large magnitude inputs should not produce NaN via sigmoid saturation."""
        gate = AttnGate(num_heads=4, head_dim=32, block_size=8)
        q = torch.randn(1, 4, 64, 32) * 1000.0
        k = torch.randn(1, 4, 64, 32) * 1000.0
        out = gate(q, k)
        assert not out.soft_scores.isnan().any()
        assert not out.soft_scores.isinf().any()
        assert out.soft_scores.min() >= 0.0
        assert out.soft_scores.max() <= 1.0

    def test_soft_scores_no_nan_with_zero_input(self) -> None:
        """All-zero inputs should produce valid output (sigmoid(bias) ~ 0.5)."""
        gate = AttnGate(num_heads=4, head_dim=32, block_size=8)
        q = torch.zeros(1, 4, 64, 32)
        k = torch.zeros(1, 4, 64, 32)
        out = gate(q, k)
        assert not out.soft_scores.isnan().any()
        assert not out.soft_scores.isinf().any()


# ---------------------------------------------------------------------------
# Sparsity edge cases
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestSparsity:
    """Sparsity ratio correctness and edge cases."""

    def test_sparsity_at_zero_threshold_is_zero(self, small_gate: AttnGate) -> None:
        """At threshold=0.0, all blocks above threshold -> sparsity=0."""
        q = torch.randn(1, 4, 64, 32)
        k = torch.randn(1, 4, 64, 32)
        out = small_gate(q, k, threshold=0.0)
        # Sigmoid outputs are always > 0, so all should be >= 0.0
        assert out.sparsity_ratio == pytest.approx(0.0, abs=1e-6)
        assert out.hard_mask.all()

    def test_sparsity_at_high_threshold_is_one(self, small_gate: AttnGate) -> None:
        """At threshold=1.0+eps, no blocks above threshold -> sparsity=1."""
        q = torch.randn(1, 4, 64, 32)
        k = torch.randn(1, 4, 64, 32)
        # sigmoid output max is strictly < 1.0 in practice, so threshold > 1.0 skips all
        out = small_gate(q, k, threshold=1.0 + 1e-6)
        assert out.sparsity_ratio == pytest.approx(1.0, abs=1e-6)
        assert not out.hard_mask.any()

    def test_sparsity_ratio_in_valid_range(self, small_gate: AttnGate) -> None:
        q = torch.randn(2, 4, 128, 32)
        k = torch.randn(2, 4, 128, 32)
        out = small_gate(q, k)
        assert 0.0 <= out.sparsity_ratio <= 1.0

    def test_compute_sparsity_method(self, small_gate: AttnGate) -> None:
        """Standalone compute_sparsity matches forward-computed sparsity."""
        q = torch.randn(1, 4, 64, 32)
        k = torch.randn(1, 4, 64, 32)
        out = small_gate(q, k, threshold=0.5)
        standalone = small_gate.compute_sparsity(out.soft_scores, 0.5)
        assert standalone == pytest.approx(out.sparsity_ratio, abs=1e-6)


# ---------------------------------------------------------------------------
# BF16 compatibility
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestBF16Compatibility:
    """Verify gate works correctly under bfloat16."""

    def test_bf16_forward_produces_finite_values(self) -> None:
        gate = AttnGate(num_heads=8, head_dim=64, block_size=16).to(torch.bfloat16)
        q = torch.randn(2, 8, 256, 64, dtype=torch.bfloat16)
        k = torch.randn(2, 8, 256, 64, dtype=torch.bfloat16)
        out = gate(q, k)
        assert not out.soft_scores.isnan().any(), "BF16 gate produced NaN"
        assert not out.soft_scores.isinf().any(), "BF16 gate produced Inf"

    def test_bf16_output_in_unit_interval(self) -> None:
        gate = AttnGate(num_heads=4, head_dim=32, block_size=8).to(torch.bfloat16)
        q = torch.randn(1, 4, 64, 32, dtype=torch.bfloat16)
        k = torch.randn(1, 4, 64, 32, dtype=torch.bfloat16)
        out = gate(q, k)
        assert out.soft_scores.min() >= 0.0
        assert out.soft_scores.max() <= 1.0

    def test_bf16_shape_matches_fp32(self) -> None:
        """BF16 output shape must match FP32 output shape for identical configs."""
        H, D, S, B, bs = 4, 32, 64, 1, 8
        gate_fp32 = AttnGate(num_heads=H, head_dim=D, block_size=bs)
        gate_bf16 = AttnGate(num_heads=H, head_dim=D, block_size=bs).to(torch.bfloat16)

        q_fp32 = torch.randn(B, H, S, D)
        k_fp32 = torch.randn(B, H, S, D)
        q_bf16 = q_fp32.to(torch.bfloat16)
        k_bf16 = k_fp32.to(torch.bfloat16)

        out_fp32 = gate_fp32(q_fp32, k_fp32)
        out_bf16 = gate_bf16(q_bf16, k_bf16)

        assert out_fp32.soft_scores.shape == out_bf16.soft_scores.shape


# ---------------------------------------------------------------------------
# Non-divisible seq_len
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestNonDivisibleSeqLen:
    """Gate must handle seq_len not divisible by block_size via padding."""

    @pytest.mark.parametrize(
        ("S", "block_size"),
        [
            (65, 64),  # 1 extra token
            (100, 64),  # 36 extra tokens
            (127, 64),  # 63 extra tokens
            (33, 16),  # not divisible by 16
            (1, 8),  # minimal seq_len
            (7, 8),  # one short of block_size
            (9, 8),  # one over block_size
        ],
    )
    def test_non_divisible_seq_len(self, S: int, block_size: int) -> None:
        H, D, B = 4, 32, 1
        gate = AttnGate(num_heads=H, head_dim=D, block_size=block_size)
        q = torch.randn(B, H, S, D)
        k = torch.randn(B, H, S, D)
        out = gate(q, k)
        expected_nb = (S + block_size - 1) // block_size
        assert out.num_blocks_q == expected_nb
        assert out.num_blocks_k == expected_nb
        assert out.soft_scores.shape == (B, H, expected_nb, expected_nb)

    def test_exact_divisible_no_padding_needed(self) -> None:
        """When S is exact multiple of block_size, no padding applied."""
        gate = AttnGate(num_heads=4, head_dim=32, block_size=8)
        q = torch.randn(1, 4, 64, 32)
        k = torch.randn(1, 4, 64, 32)
        out = gate(q, k)
        assert out.num_blocks_q == 8  # 64 / 8
        assert out.num_blocks_k == 8


# ---------------------------------------------------------------------------
# Parameter budget
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestParameterBudget:
    """Gate params must be <0.1% of typical attention layer params."""

    def test_gate_parameter_count_within_budget(self) -> None:
        H, D = 32, 128
        gate = AttnGate(num_heads=H, head_dim=D, block_size=64)
        # Typical attention layer: 4 * (H*D)^2 = 4 * 4096^2 ~ 67M params
        typical_attn_params = 4 * (H * D) ** 2
        gate_params = gate.num_parameters
        ratio = gate_params / typical_attn_params
        assert ratio < 0.001, f"Gate uses {ratio:.4%} of attention params, exceeds 0.1% budget"

    def test_num_parameters_matches_manual_count(self) -> None:
        gate = AttnGate(num_heads=4, head_dim=32, block_size=8)
        manual_count = 0
        for p in gate.parameters():
            if p.requires_grad:
                manual_count += p.numel()
        assert gate.num_parameters == manual_count

    def test_num_parameters_positive(self) -> None:
        gate = AttnGate(num_heads=4, head_dim=32, block_size=8)
        assert gate.num_parameters > 0


# ---------------------------------------------------------------------------
# Gradient isolation: no leakage to base model
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestGradientIsolation:
    """Gate forward must not create gradients on hypothetical base model params."""

    def test_gate_forward_no_gradient_on_base_model(self) -> None:
        gate = AttnGate(num_heads=4, head_dim=32, block_size=8)

        # Simulate a "frozen base" projection
        frozen_proj = nn.Linear(32, 32, bias=False)
        for p in frozen_proj.parameters():
            p.requires_grad_(False)

        q = torch.randn(1, 4, 64, 32, requires_grad=False)
        k = torch.randn(1, 4, 64, 32, requires_grad=False)

        # Apply frozen proj to q (simulates base model)
        q_projected = frozen_proj(q)
        out = gate(q_projected, k)
        loss = out.soft_scores.sum()
        loss.backward()

        for p in frozen_proj.parameters():
            assert p.grad is None, "Gradient leaked to frozen base model parameters"

    def test_gate_params_receive_gradients(self) -> None:
        """Gate's own parameters must receive gradients during backward."""
        gate = AttnGate(num_heads=4, head_dim=32, block_size=8)
        q = torch.randn(1, 4, 64, 32)
        k = torch.randn(1, 4, 64, 32)
        out = gate(q, k)
        loss = out.soft_scores.sum()
        loss.backward()

        for name, p in gate.named_parameters():
            assert p.grad is not None, f"Gate param '{name}' has no gradient"
            assert not p.grad.isnan().any(), f"Gate param '{name}' has NaN gradient"


# ---------------------------------------------------------------------------
# Forward validation errors
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestForwardValidation:
    """Verify validation in forward pass raises correct errors."""

    def test_3d_q_raises(self, small_gate: AttnGate) -> None:
        q = torch.randn(4, 64, 32)
        k = torch.randn(1, 4, 64, 32)
        with pytest.raises(ValidationError, match="Expected 4D Q tensor"):
            small_gate(q, k)

    def test_3d_k_raises(self, small_gate: AttnGate) -> None:
        q = torch.randn(1, 4, 64, 32)
        k = torch.randn(4, 64, 32)
        with pytest.raises(ValidationError, match="Expected 4D K tensor"):
            small_gate(q, k)

    def test_mismatched_qk_shapes_raises(self, small_gate: AttnGate) -> None:
        q = torch.randn(1, 4, 64, 32)
        k = torch.randn(1, 4, 128, 32)
        with pytest.raises(ValidationError, match="Q and K must have same shape"):
            small_gate(q, k)

    def test_wrong_num_heads_raises(self, small_gate: AttnGate) -> None:
        # small_gate has num_heads=4, but input has 8 heads
        q = torch.randn(1, 8, 64, 32)
        k = torch.randn(1, 8, 64, 32)
        with pytest.raises(ValidationError, match="Expected 4 heads"):
            small_gate(q, k)

    def test_wrong_head_dim_raises(self, small_gate: AttnGate) -> None:
        # small_gate has head_dim=32, but input has dim 64
        q = torch.randn(1, 4, 64, 64)
        k = torch.randn(1, 4, 64, 64)
        with pytest.raises(ValidationError, match="Expected head_dim=32"):
            small_gate(q, k)

    def test_zero_seq_len_raises(self, small_gate: AttnGate) -> None:
        q = torch.randn(1, 4, 0, 32)
        k = torch.randn(1, 4, 0, 32)
        with pytest.raises(ValidationError, match="Sequence length must be > 0"):
            small_gate(q, k)


# ---------------------------------------------------------------------------
# extra_repr
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestExtraRepr:
    """Module repr for debugging."""

    def test_extra_repr_contains_key_info(self) -> None:
        gate = AttnGate(num_heads=4, head_dim=32, block_size=8)
        r = gate.extra_repr()
        assert "num_heads=4" in r
        assert "head_dim=32" in r
        assert "block_size=8" in r


# ---------------------------------------------------------------------------
# GateOutput frozen dataclass
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestGateOutputFrozen:
    """GateOutput is a frozen dataclass — no mutations allowed."""

    def test_gate_output_is_frozen(self, small_gate: AttnGate) -> None:
        q = torch.randn(1, 4, 64, 32)
        k = torch.randn(1, 4, 64, 32)
        out = small_gate(q, k)
        with pytest.raises(AttributeError):
            out.sparsity_ratio = 0.5  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Property-based tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestGateProperties:
    """Hypothesis-driven property-based tests."""

    @given(
        B=st.integers(min_value=1, max_value=4),
        H=st.sampled_from([4, 8, 16]),
        S=st.integers(min_value=16, max_value=256),
        D=st.sampled_from([32, 64, 128]),
        block_size=st.sampled_from([8, 16, 32]),
    )
    @settings(max_examples=50, deadline=10000)
    def test_gate_output_shape_property(
        self, B: int, H: int, S: int, D: int, block_size: int,
    ) -> None:
        """For all valid (B, H, S, D, block_size): output shape is correct."""
        gate = AttnGate(num_heads=H, head_dim=D, block_size=block_size)
        q = torch.randn(B, H, S, D)
        k = torch.randn(B, H, S, D)
        out = gate(q, k)
        expected_nb = (S + block_size - 1) // block_size
        assert out.soft_scores.shape == (B, H, expected_nb, expected_nb)
        assert 0.0 <= out.sparsity_ratio <= 1.0

    @given(
        B=st.integers(min_value=1, max_value=2),
        H=st.sampled_from([4, 8]),
        S=st.integers(min_value=8, max_value=128),
        D=st.sampled_from([32, 64]),
        block_size=st.sampled_from([8, 16]),
    )
    @settings(max_examples=30, deadline=10000)
    def test_soft_scores_always_in_unit_interval_property(
        self, B: int, H: int, S: int, D: int, block_size: int,
    ) -> None:
        """Sigmoid guarantees [0, 1] for all valid inputs."""
        gate = AttnGate(num_heads=H, head_dim=D, block_size=block_size)
        q = torch.randn(B, H, S, D)
        k = torch.randn(B, H, S, D)
        out = gate(q, k)
        assert out.soft_scores.min().item() >= 0.0
        assert out.soft_scores.max().item() <= 1.0
        assert not out.soft_scores.isnan().any()
        assert not out.soft_scores.isinf().any()
