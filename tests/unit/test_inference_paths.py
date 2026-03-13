"""Tests validating the TASFT inference path correctness.

Verifies that the inference forward path through TASFTAttention produces correct
outputs, properly invokes the gate, handles fallback paths, and produces results
close to dense attention. This suite was created in response to a reviewer flag
that the inference path was previously broken (running dense instead of sparse).

Each test targets a specific aspect of _inference_forward, _prepare_qkv,
_dense_fallback_via_base, and _SparseAttentionWrapper to ensure the full
inference code path is exercised and validated.
"""
from __future__ import annotations

from typing import Any

import pytest
import torch
from torch import nn

from tasft.modules.attn_gate import AttnGate, GateOutput
from tasft.modules.tasft_attention import TASFTAttention
from tasft.types import LayerIndex


# ── Helpers: Tiny attention modules for controlled testing ─────────────


class _StdAttention(nn.Module):
    """Minimal attention module exposing q_proj, k_proj, v_proj, o_proj.

    Matches the attribute interface that TASFTAttention._inference_forward
    expects for the non-fallback sparse/dense path.

    All projections use deterministic Xavier-uniform init for reproducibility.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        head_dim: int,
        num_kv_heads: int | None = None,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.hidden_size = hidden_dim
        kv_heads = num_kv_heads if num_kv_heads is not None else num_heads
        self.num_key_value_heads = kv_heads
        self.q_proj = nn.Linear(hidden_dim, num_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_dim, kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_dim, kv_heads * head_dim, bias=False)
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
        """Dense attention forward implementing HF convention."""
        B, S, _ = hidden_states.shape
        q = self.q_proj(hidden_states).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(hidden_states).view(B, S, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(hidden_states).view(B, S, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        # GQA expansion
        if self.num_key_value_heads < self.num_heads:
            n_rep = self.num_heads // self.num_key_value_heads
            k = k.repeat_interleave(n_rep, dim=1)
            v = v.repeat_interleave(n_rep, dim=1)

        scale = self.head_dim ** -0.5
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * scale
        causal = torch.triu(
            torch.full((S, S), float("-inf"), device=hidden_states.device),
            diagonal=1,
        )
        attn_weights = attn_weights + causal.unsqueeze(0).unsqueeze(0)
        attn_probs = torch.softmax(attn_weights, dim=-1)
        out = torch.matmul(attn_probs, v)
        out = out.transpose(1, 2).contiguous().reshape(B, S, self.num_heads * self.head_dim)
        out = self.o_proj(out)

        return (out, attn_probs if output_attentions else None, None)


class _NoProjectionAttention(nn.Module):
    """Attention module WITHOUT q_proj/k_proj/v_proj/o_proj attributes.

    Forces TASFTAttention into the _dense_fallback_via_base path during
    inference, simulating a non-standard architecture.
    """

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
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
        """Identity-like forward for fallback testing."""
        out = self.linear(hidden_states)
        return (out, None, None)


# ── Fixtures ───────────────────────────────────────────────────────────


# Dimensions chosen so seq_len is a multiple of block_size for clean block boundaries.
_HIDDEN_DIM = 64
_NUM_HEADS = 4
_HEAD_DIM = 16
_BLOCK_SIZE = 32
_BATCH_SIZE = 2
_SEQ_LEN = 64


def _make_gate(
    num_heads: int = _NUM_HEADS,
    head_dim: int = _HEAD_DIM,
    block_size: int = _BLOCK_SIZE,
) -> AttnGate:
    """Construct an AttnGate with deterministic initialization."""
    return AttnGate(
        num_heads=num_heads,
        head_dim=head_dim,
        block_size=block_size,
        default_threshold=0.5,
    )


def _make_std_attn(
    num_kv_heads: int | None = None,
) -> _StdAttention:
    """Construct a standard attention module with deterministic init."""
    return _StdAttention(
        hidden_dim=_HIDDEN_DIM,
        num_heads=_NUM_HEADS,
        head_dim=_HEAD_DIM,
        num_kv_heads=num_kv_heads,
    )


def _make_tasft_attn(
    base_attn: nn.Module | None = None,
    gate: AttnGate | None = None,
    min_sparsity_for_speedup: float = 0.3,
    compute_gate_target: bool = False,
) -> TASFTAttention:
    """Construct a TASFTAttention wrapper for testing."""
    if base_attn is None:
        base_attn = _make_std_attn()
    if gate is None:
        gate = _make_gate()
    return TASFTAttention(
        base_attn=base_attn,
        gate=gate,
        layer_idx=LayerIndex(0),
        compute_gate_target=compute_gate_target,
        min_sparsity_for_speedup=min_sparsity_for_speedup,
    )


def _make_hidden_states(
    batch_size: int = _BATCH_SIZE,
    seq_len: int = _SEQ_LEN,
    hidden_dim: int = _HIDDEN_DIM,
) -> torch.Tensor:
    """Create deterministic hidden states tensor [B, S, D]."""
    gen = torch.Generator().manual_seed(42)
    return torch.randn(batch_size, seq_len, hidden_dim, generator=gen)


# ── Tests ──────────────────────────────────────────────────────────────


class TestInferenceForwardProducesOutput:
    """Test 1: Basic smoke test verifying output shape from inference forward."""

    def test_inference_forward_produces_output(self) -> None:
        """Inference forward must return a 3-tuple with attn_output shape [B, S, D]."""
        tasft_attn = _make_tasft_attn()
        hidden = _make_hidden_states()

        with torch.no_grad():
            result = tasft_attn(hidden)

        assert isinstance(result, tuple), f"Expected tuple, got {type(result)}"
        # Default use_cache=False -> 2-tuple (attn_output, attn_weights)
        assert len(result) == 2, f"Expected 2-element tuple (no KV cache), got {len(result)}"
        attn_output = result[0]
        assert attn_output.shape == hidden.shape, (
            f"Output shape {attn_output.shape} must match input {hidden.shape}"
        )
        assert attn_output.dtype == hidden.dtype, (
            f"Output dtype {attn_output.dtype} must match input {hidden.dtype}"
        )
        assert torch.isfinite(attn_output).all(), "Output contains NaN or Inf"


class TestInferenceUsesGate:
    """Test 2: Verify _last_gate_output is populated after inference forward."""

    def test_inference_uses_gate(self) -> None:
        """After inference forward, _last_gate_output must be a valid GateOutput."""
        tasft_attn = _make_tasft_attn()
        hidden = _make_hidden_states()

        # Before forward, gate output should be None
        assert tasft_attn._last_gate_output is None

        with torch.no_grad():
            tasft_attn(hidden)

        gate_out = tasft_attn._last_gate_output
        assert gate_out is not None, "_last_gate_output must be set after inference forward"
        assert isinstance(gate_out, GateOutput), (
            f"Expected GateOutput, got {type(gate_out)}"
        )
        assert gate_out.soft_scores.ndim == 4, (
            f"soft_scores must be 4D [B, H, NB_q, NB_k], got ndim={gate_out.soft_scores.ndim}"
        )
        assert gate_out.hard_mask.dtype == torch.bool, (
            f"hard_mask must be bool, got {gate_out.hard_mask.dtype}"
        )


class TestInferenceDenseFallbackWhenNoProjections:
    """Test 3: When base_attn lacks q_proj, verify _dense_fallback_via_base is used."""

    def test_inference_dense_fallback_when_no_projections(self) -> None:
        """Non-standard base_attn without projections must fall back to base forward."""
        base_attn = _NoProjectionAttention(hidden_dim=_HIDDEN_DIM)
        gate = _make_gate()
        tasft_attn = _make_tasft_attn(base_attn=base_attn, gate=gate)
        hidden = _make_hidden_states()

        with torch.no_grad():
            result = tasft_attn(hidden)

        # Fallback path returns 2-tuple when use_cache=False (no KV cache)
        assert isinstance(result, tuple)
        assert len(result) == 2
        attn_output = result[0]
        assert attn_output.shape == hidden.shape, (
            f"Fallback output shape {attn_output.shape} must match input {hidden.shape}"
        )
        # attn_weights should be None in inference fallback path
        assert result[1] is None, "Fallback inference must return None for attn_weights"
        # _last_gate_output may be None since there are no projection layers to extract Q/K
        # This is acceptable for non-standard architectures
        assert torch.isfinite(attn_output).all(), "Fallback output contains NaN or Inf"


class TestInferenceDenseFallbackWhenLowSparsity:
    """Test 4: With min_sparsity_for_speedup=0.99, verify dense SDPA path is used."""

    def test_inference_dense_fallback_when_low_sparsity(self) -> None:
        """When min_sparsity_for_speedup is very high, the dense code path must execute.

        On CPU the kernel is never available, so this verifies the dense
        fallback within _inference_forward (not the _dense_fallback_via_base).
        The gate is still run, but the sparse kernel branch is skipped.
        """
        tasft_attn = _make_tasft_attn(min_sparsity_for_speedup=0.99)
        hidden = _make_hidden_states()

        with torch.no_grad():
            result = tasft_attn(hidden)

        attn_output = result[0]
        assert attn_output.shape == hidden.shape
        assert torch.isfinite(attn_output).all()
        # Gate should still be invoked even on dense path
        assert tasft_attn._last_gate_output is not None, (
            "Gate must still run on dense fallback path"
        )
        # _last_attn_weights should be None in inference mode (not stored)
        assert tasft_attn._last_attn_weights is None, (
            "Inference path must not store attn_weights"
        )


class TestInferenceSparseVsDenseClose:
    """Test 5: On CPU (no kernel), verify dense-path output matches base_attn output.

    This is the critical regression test: the inference path was previously broken
    and running dense attention incorrectly. We verify that the inference path's
    manual dense computation matches the base_attn.forward() output.
    """

    def test_inference_sparse_vs_dense_close(self) -> None:
        """Inference dense-path output must be close to base_attn direct forward output.

        The inference dense fallback in _inference_forward applies causal masking
        only when an explicit attention_mask is provided. The base_attn._StdAttention
        always applies an internal causal mask. To get comparable outputs, we pass
        an explicit causal attention_mask to the inference path so both paths use
        identical causal masking.

        We expect atol=1e-4 tolerance for float32 accumulation order differences.
        """
        torch.manual_seed(42)
        base_attn = _make_std_attn()
        gate = _make_gate()
        tasft_attn = _make_tasft_attn(
            base_attn=base_attn,
            gate=gate,
            min_sparsity_for_speedup=0.99,  # force dense path in inference
        )
        hidden = _make_hidden_states()

        # Build a 4D causal mask: [1, 1, S, S] with -inf in upper triangle.
        # This matches the causal mask that base_attn applies internally.
        S = hidden.shape[1]
        causal_mask = torch.triu(
            torch.full((S, S), float("-inf")),
            diagonal=1,
        ).unsqueeze(0).unsqueeze(0)  # [1, 1, S, S]

        # Reference: direct base_attn forward (has internal causal mask)
        with torch.no_grad():
            base_output = base_attn(hidden, output_attentions=False)
        base_result = base_output[0]

        # Inference path through TASFTAttention with explicit causal mask
        with torch.no_grad():
            inference_output = tasft_attn(hidden, attention_mask=causal_mask)
        inference_result = inference_output[0]

        assert base_result.shape == inference_result.shape, (
            f"Shape mismatch: base={base_result.shape}, inference={inference_result.shape}"
        )
        # Both paths compute the same projections with the same weights.
        # Differences arise only from floating-point accumulation order.
        max_diff = (base_result - inference_result).abs().max().item()
        assert max_diff < 1e-4, (
            f"Inference dense path diverges from base_attn: max_diff={max_diff:.6e}. "
            f"The inference path may be computing attention incorrectly."
        )


class TestInferenceNoAttnWeightsReturned:
    """Test 6: Inference path must return None for attn_weights (second element)."""

    def test_inference_no_attn_weights_returned(self) -> None:
        """Inference forward must return None as the second tuple element."""
        tasft_attn = _make_tasft_attn()
        hidden = _make_hidden_states()

        with torch.no_grad():
            result = tasft_attn(hidden)

        assert result[1] is None, (
            f"Inference path must return None for attn_weights, got {type(result[1])}"
        )


class TestInferenceGateSparsityRatioValid:
    """Test 7: Gate sparsity_ratio must be in [0, 1]."""

    def test_inference_gate_sparsity_ratio_valid(self) -> None:
        """After inference, gate_output.sparsity_ratio must be a float in [0, 1]."""
        tasft_attn = _make_tasft_attn()
        hidden = _make_hidden_states()

        with torch.no_grad():
            tasft_attn(hidden)

        gate_out = tasft_attn._last_gate_output
        assert gate_out is not None
        ratio = gate_out.sparsity_ratio
        assert isinstance(ratio, float), f"sparsity_ratio must be float, got {type(ratio)}"
        assert 0.0 <= ratio <= 1.0, (
            f"sparsity_ratio must be in [0, 1], got {ratio}"
        )


class TestInferenceKVCachePassthrough:
    """Test 8: Verify past_key_value is handled correctly in inference."""

    def test_inference_kv_cache_passthrough_tuple(self) -> None:
        """With use_cache=True, output is still 2-tuple (modern HF convention).

        Modern HF models use DynamicCache mutated in-place, so _pack_output
        always returns (attn_output, attn_weights) — cache is not in the return.
        """
        tasft_attn = _make_tasft_attn()
        hidden = _make_hidden_states(batch_size=1, seq_len=_SEQ_LEN)

        with torch.no_grad():
            result = tasft_attn(hidden, use_cache=True)

        # Always 2-tuple regardless of use_cache
        assert len(result) == 2, f"Expected 2-tuple, got {len(result)}-tuple"
        assert result[0].shape[0] == 1, "Batch dim mismatch"
        assert torch.isfinite(result[0]).all(), "Output contains NaN or Inf"

    def test_inference_kv_cache_none_when_not_requested(self) -> None:
        """When use_cache=False, result must be a 2-tuple (no KV cache element)."""
        tasft_attn = _make_tasft_attn()
        hidden = _make_hidden_states()

        with torch.no_grad():
            result = tasft_attn(hidden, use_cache=False)

        # With use_cache=False, past_kv is None so _pack_output returns 2-tuple
        assert len(result) == 2, (
            f"use_cache=False must return 2-tuple (no KV cache), got {len(result)}"
        )


class TestInferencePositionEmbeddingsApplied:
    """Test 9: Verify rotary embeddings path works when position_embeddings provided."""

    def test_inference_position_embeddings_applied(self) -> None:
        """Providing position_embeddings (cos, sin) must change the output vs no embeddings.

        If rotary embeddings have no effect, the inference path is ignoring them,
        which would be a correctness bug.
        """
        tasft_attn = _make_tasft_attn()
        hidden = _make_hidden_states()

        # Forward WITHOUT position embeddings
        with torch.no_grad():
            result_no_pe = tasft_attn(hidden)

        # Create deterministic rotary embeddings: cos/sin of shape [B, S, head_dim]
        B, S = _BATCH_SIZE, _SEQ_LEN
        gen = torch.Generator().manual_seed(99)
        cos = torch.randn(B, S, _HEAD_DIM, generator=gen)
        sin = torch.randn(B, S, _HEAD_DIM, generator=gen)

        # Forward WITH position embeddings
        with torch.no_grad():
            result_with_pe = tasft_attn(hidden, position_embeddings=(cos, sin))

        out_no_pe = result_no_pe[0]
        out_with_pe = result_with_pe[0]

        # Outputs must differ since rotary embeddings rotate Q and K
        diff = (out_no_pe - out_with_pe).abs().max().item()
        assert diff > 1e-6, (
            f"Position embeddings had no effect on output (max_diff={diff:.6e}). "
            f"Rotary embedding path may not be applied in inference."
        )


class TestInferenceGQAHeadExpansion:
    """Test 10: With num_kv_heads < num_heads, verify K/V heads are expanded."""

    def test_inference_gqa_head_expansion(self) -> None:
        """GQA: 4 Q heads with 2 KV heads must produce correct output shape.

        The _prepare_qkv method must repeat_interleave K and V from 2 to 4 heads.
        Output shape must still be [B, S, hidden_dim] = [B, S, num_heads * head_dim].
        """
        num_kv_heads = 2
        base_attn = _make_std_attn(num_kv_heads=num_kv_heads)
        gate = _make_gate()
        tasft_attn = _make_tasft_attn(base_attn=base_attn, gate=gate)
        hidden = _make_hidden_states()

        with torch.no_grad():
            result = tasft_attn(hidden)

        attn_output = result[0]
        assert attn_output.shape == hidden.shape, (
            f"GQA output shape {attn_output.shape} must match input {hidden.shape}"
        )
        assert torch.isfinite(attn_output).all(), "GQA output contains NaN or Inf"
        # Gate must have run successfully with expanded heads
        assert tasft_attn._last_gate_output is not None


class TestSparseAttentionWrapperForward:
    """Test 11: Test _SparseAttentionWrapper directly."""

    def test_sparse_attention_wrapper_forward(self) -> None:
        """_SparseAttentionWrapper must produce valid output with correct shape.

        On CPU, the sparse kernel requires CUDA and will fail if invoked.
        We set min_sparsity_for_speedup=1.0 to guarantee the dense fallback
        path within the wrapper is taken (gate sparsity < 1.0 always).
        """
        from tasft.inference.tasft_model import _SparseAttentionWrapper

        base_attn = _make_std_attn()
        gate = _make_gate()

        wrapper = _SparseAttentionWrapper(
            original_attn=base_attn,
            gate=gate,
            layer_idx=0,
            threshold_tau=0.5,
            block_size=_BLOCK_SIZE,
            # Force dense path: gate sparsity will never reach 1.0
            min_sparsity_for_speedup=1.0,
        )

        hidden = _make_hidden_states()

        with torch.no_grad():
            result = wrapper(hidden)

        assert isinstance(result, tuple)
        assert len(result) == 3
        attn_output = result[0]
        assert attn_output.shape == hidden.shape, (
            f"Wrapper output shape {attn_output.shape} must match input {hidden.shape}"
        )
        assert torch.isfinite(attn_output).all(), "Wrapper output contains NaN or Inf"
        # attn_weights must be None (inference convention)
        assert result[1] is None
        # last_gate_output must be populated
        assert wrapper.last_gate_output is not None
        assert isinstance(wrapper.last_gate_output, GateOutput)


class TestInferenceNoGradientsFlow:
    """Test 12: In inference mode, verify no gradients on any parameter."""

    def test_inference_no_gradients_flow(self) -> None:
        """Under torch.no_grad(), no parameter should accumulate gradients.

        This verifies the inference path does not inadvertently enable gradient
        computation on gate or base parameters.
        """
        tasft_attn = _make_tasft_attn()
        hidden = _make_hidden_states()

        with torch.no_grad():
            result = tasft_attn(hidden)
            output = result[0]

        # Output should not require grad
        assert not output.requires_grad, "Inference output must not require gradients"

        # No parameter should have a non-None .grad after inference-only forward
        for name, param in tasft_attn.named_parameters():
            assert param.grad is None, (
                f"Parameter {name} has gradient after inference-only forward"
            )


class TestInferenceDeterministicWithSeed:
    """Test 13: Same input + seed = same output."""

    def test_inference_deterministic_with_seed(self) -> None:
        """Two inference passes with identical inputs must produce bit-identical outputs."""
        torch.manual_seed(42)
        tasft_attn = _make_tasft_attn()
        hidden = _make_hidden_states()

        with torch.no_grad():
            result1 = tasft_attn(hidden)
            result2 = tasft_attn(hidden)

        out1 = result1[0]
        out2 = result2[0]
        assert torch.equal(out1, out2), (
            f"Determinism violated: max_diff="
            f"{(out1 - out2).abs().max().item():.6e}"
        )


class TestTrainingVsInferenceModeDispatch:
    """Test 14: Verify is_training flag correctly dispatches training vs inference."""

    def test_training_mode_dispatches_to_training_forward(self) -> None:
        """With grad enabled + compute_gate_target=True, training path must run.

        Training path returns attn_weights (non-None second element) and stores
        _last_attn_weights, unlike inference which returns None for both.
        """
        tasft_attn = _make_tasft_attn(compute_gate_target=True)
        hidden = _make_hidden_states()

        # Training dispatch: grad enabled + compute_gate_target=True
        result = tasft_attn(hidden)

        assert result[1] is not None, (
            "Training path must return non-None attn_weights"
        )
        assert tasft_attn._last_attn_weights is not None, (
            "Training path must store _last_attn_weights"
        )

    def test_inference_mode_dispatches_to_inference_forward(self) -> None:
        """With no_grad context, inference path must run regardless of compute_gate_target."""
        tasft_attn = _make_tasft_attn(compute_gate_target=True)
        hidden = _make_hidden_states()

        # Even with compute_gate_target=True, no_grad disables training path
        with torch.no_grad():
            result = tasft_attn(hidden)

        assert result[1] is None, (
            "Under no_grad, inference path must return None for attn_weights"
        )
        assert tasft_attn._last_attn_weights is None, (
            "Inference path must not store _last_attn_weights"
        )

    def test_inference_dispatches_when_gate_target_false(self) -> None:
        """With compute_gate_target=False, inference path must run even with grad enabled."""
        tasft_attn = _make_tasft_attn(compute_gate_target=False)
        hidden = _make_hidden_states()

        # Grad enabled but compute_gate_target=False -> inference path
        result = tasft_attn(hidden)

        assert result[1] is None, (
            "With compute_gate_target=False, inference path must run "
            "and return None for attn_weights"
        )
