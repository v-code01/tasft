"""Unit tests for TASFTAttention: patched attention layer with co-training hooks.

Tests verify:
    - Patching replaces all attention layers with TASFTAttention
    - Base model weights are frozen after patching
    - Training mode stores auxiliary gate outputs as instance attributes
    - Inference mode returns only hidden states (tuple format)
    - Output shape is preserved after patching
    - Active gate layer selection works correctly
    - Dense-path output is close to original attention output

Uses GPT2LMHeadModel for patching/freeze tests (provides model.transformer.h path).
Uses a mock base attention module for forward pass tests. TASFTAttention returns
HF-compatible tuple (attn_output, attn_weights, past_kv) and stores gate data as
instance attributes (_last_gate_output, _last_attn_weights).
"""
from __future__ import annotations

import pytest
import torch
from torch import nn
from transformers import GPT2Config, GPT2LMHeadModel

from tasft.modules.attn_gate import AttnGate
from tasft.modules.tasft_attention import (
    GateConfig,
    TASFTAttention,
    patch_model_attention,
)
from tasft.types import LayerIndex

_TINY_GPT2_CONFIG = GPT2Config(
    n_layer=2,
    n_head=4,
    n_embd=128,
    n_positions=256,
    vocab_size=512,
)


def _make_frozen_model() -> GPT2LMHeadModel:
    """Create a tiny GPT-2 model with all params frozen (simulates real TASFT pipeline)."""
    model = GPT2LMHeadModel(_TINY_GPT2_CONFIG)
    for param in model.parameters():
        param.requires_grad = False
    return model


class _MockBaseAttn(nn.Module):
    """Mock base attention with q_proj/k_proj for gate QK extraction testing.

    Forward returns (attn_output, attn_weights) or (attn_output,) depending on
    output_attentions flag, matching the HF convention.
    """

    def __init__(self, num_heads: int = 4, head_dim: int = 32) -> None:
        super().__init__()
        hidden_dim = num_heads * head_dim
        self.q_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.o_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.num_heads = num_heads
        self.head_dim = head_dim

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        past_key_value: object = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: torch.Tensor | None = None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        **kwargs: object,
    ) -> tuple[torch.Tensor, ...]:
        B, S, _ = hidden_states.shape
        nh, hd = self.num_heads, self.head_dim
        q = self.q_proj(hidden_states).view(B, S, nh, hd).transpose(1, 2)
        k = self.k_proj(hidden_states).view(B, S, nh, hd).transpose(1, 2)
        v = self.v_proj(hidden_states).view(B, S, nh, hd).transpose(1, 2)
        scale = hd ** -0.5
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * scale
        attn_probs = torch.softmax(attn_weights, dim=-1)
        attn_output = torch.matmul(attn_probs, v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, S, nh * hd)
        attn_output = self.o_proj(attn_output)
        if output_attentions:
            return (attn_output, attn_weights)
        return (attn_output,)


@pytest.fixture
def tiny_model() -> GPT2LMHeadModel:
    """Frozen tiny GPT-2 LM model for patching tests."""
    return _make_frozen_model()


@pytest.fixture
def tiny_gate_config() -> GateConfig:
    """Gate config matching the tiny model: block_size=8, 2 layers."""
    return GateConfig(
        block_size=8,
        num_layers=2,
        gate_hidden_dim=16,
        default_threshold=0.5,
    )


@pytest.mark.unit
class TestPatchModelAttention:
    """Tests for patch_model_attention() and TASFTAttention integration."""

    def test_patch_replaces_all_attention_layers(
        self, tiny_model: GPT2LMHeadModel, tiny_gate_config: GateConfig,
    ) -> None:
        """After patching, every attention layer must be a TASFTAttention instance."""
        patched = patch_model_attention(tiny_model, tiny_gate_config)

        assert len(patched) == tiny_gate_config.num_layers, (
            f"Expected {tiny_gate_config.num_layers} patched layers, got {len(patched)}"
        )

        for idx in range(tiny_gate_config.num_layers):
            layer = tiny_model.transformer.h[idx]
            attn_module = layer.attn
            assert isinstance(attn_module, TASFTAttention), (
                f"Layer {idx} attention is {type(attn_module).__name__}, "
                f"expected TASFTAttention"
            )

    def test_base_weights_frozen_after_patch(
        self, tiny_model: GPT2LMHeadModel, tiny_gate_config: GateConfig,
    ) -> None:
        """All non-gate parameters must have requires_grad=False after patching.
        Only gate parameters should be trainable."""
        patched = patch_model_attention(tiny_model, tiny_gate_config)

        gate_param_ids: set[int] = set()
        for tasft_attn in patched.values():
            for p in tasft_attn.gate.parameters():
                gate_param_ids.add(id(p))

        unfrozen_non_gate = [
            name
            for name, p in tiny_model.named_parameters()
            if p.requires_grad and id(p) not in gate_param_ids
        ]
        assert len(unfrozen_non_gate) == 0, (
            f"Found {len(unfrozen_non_gate)} unfrozen non-gate params: "
            f"{unfrozen_non_gate[:5]}"
        )

        gate_trainable = sum(
            1 for tasft_attn in patched.values()
            for p in tasft_attn.gate.parameters()
            if p.requires_grad
        )
        assert gate_trainable > 0, "Gate parameters should be trainable after patching"

    def test_training_mode_returns_aux(self) -> None:
        """With compute_gate_target=True and grad enabled, forward returns
        non-None gate_output and gate_target_scores."""
        num_heads, head_dim = 4, 32
        base_attn = _MockBaseAttn(num_heads, head_dim)
        gate = AttnGate(
            num_heads=num_heads, head_dim=head_dim,
            block_size=8, gate_hidden_dim=16, default_threshold=0.5,
        )
        tasft_attn = TASFTAttention(
            base_attn=base_attn, gate=gate,
            layer_idx=LayerIndex(0), compute_gate_target=True,
        )
        tasft_attn.train()

        hidden = torch.randn(1, 32, num_heads * head_dim, requires_grad=True)
        with torch.enable_grad():
            output = tasft_attn(hidden)

        # TASFTAttention returns HF-compatible tuple: (attn_output, attn_weights, past_kv)
        assert isinstance(output, tuple), (
            f"Expected tuple, got {type(output).__name__}"
        )
        assert output[0] is not None, "attn_output must not be None"
        # Gate data stored as instance attributes for trainer extraction
        assert tasft_attn._last_gate_output is not None, (
            "gate_output must not be None in training mode"
        )
        assert tasft_attn.layer_idx == LayerIndex(0)

    def test_inference_mode_no_aux(self) -> None:
        """With compute_gate_target=False and torch.no_grad(), forward returns
        only hidden_states (gate_target_scores is None)."""
        num_heads, head_dim = 4, 32
        base_attn = _MockBaseAttn(num_heads, head_dim)
        gate = AttnGate(
            num_heads=num_heads, head_dim=head_dim,
            block_size=8, gate_hidden_dim=16, default_threshold=0.5,
        )
        tasft_attn = TASFTAttention(
            base_attn=base_attn, gate=gate,
            layer_idx=LayerIndex(0), compute_gate_target=False,
        )
        tasft_attn.eval()

        hidden = torch.randn(1, 32, num_heads * head_dim)
        with torch.no_grad():
            output = tasft_attn(hidden)

        # HF-compatible tuple output
        assert isinstance(output, tuple)
        attn_output = output[0]
        assert attn_output is not None, "attn_output must not be None"
        assert attn_output.shape == (1, 32, num_heads * head_dim), (
            f"Unexpected output shape: {attn_output.shape}"
        )

    def test_output_shape_unchanged(self) -> None:
        """TASFTAttention output hidden_states shape must match base attention output shape."""
        num_heads, head_dim = 4, 32
        hidden_dim = num_heads * head_dim
        base_attn = _MockBaseAttn(num_heads, head_dim)

        hidden = torch.randn(2, 16, hidden_dim)
        with torch.no_grad():
            original_out = base_attn(hidden, output_attentions=False)
        original_shape = original_out[0].shape

        gate = AttnGate(
            num_heads=num_heads, head_dim=head_dim,
            block_size=8, gate_hidden_dim=16, default_threshold=0.5,
        )
        tasft_attn = TASFTAttention(
            base_attn=base_attn, gate=gate,
            layer_idx=LayerIndex(0), compute_gate_target=False,
        )
        tasft_attn.eval()
        with torch.no_grad():
            tasft_out = tasft_attn(hidden)

        assert original_shape == tasft_out[0].shape, (
            f"Shape mismatch: original={original_shape}, "
            f"tasft={tasft_out[0].shape}"
        )

    def test_set_active_gate_layers(
        self, tiny_model: GPT2LMHeadModel, tiny_gate_config: GateConfig,
    ) -> None:
        """Only specified active layers should have compute_gate_target=True."""
        patched = patch_model_attention(tiny_model, tiny_gate_config)

        patched[0].set_training_mode(True)
        patched[1].set_training_mode(False)

        assert patched[0].compute_gate_target is True, (
            "Layer 0 should have compute_gate_target=True"
        )
        assert patched[1].compute_gate_target is False, (
            "Layer 1 should have compute_gate_target=False"
        )

        patched[0].set_training_mode(False)
        patched[1].set_training_mode(True)

        assert patched[0].compute_gate_target is False, (
            "Layer 0 should now have compute_gate_target=False"
        )
        assert patched[1].compute_gate_target is True, (
            "Layer 1 should now have compute_gate_target=True"
        )

    def test_output_close_to_dense(self) -> None:
        """TASFTAttention in inference mode must produce output identical to base
        attention, verifying the wrapper doesn't corrupt the computation."""
        num_heads, head_dim = 4, 32
        hidden_dim = num_heads * head_dim
        base_attn = _MockBaseAttn(num_heads, head_dim)

        torch.manual_seed(42)
        hidden = torch.randn(1, 32, hidden_dim)

        # Original output from base_attn directly
        with torch.no_grad():
            original_out = base_attn(hidden, output_attentions=False)[0]

        # TASFTAttention wrapping the SAME base_attn module
        gate = AttnGate(
            num_heads=num_heads, head_dim=head_dim,
            block_size=8, gate_hidden_dim=16, default_threshold=0.0,
        )
        tasft_attn = TASFTAttention(
            base_attn=base_attn, gate=gate,
            layer_idx=LayerIndex(0), compute_gate_target=False,
        )
        tasft_attn.eval()

        with torch.no_grad():
            tasft_out = tasft_attn(hidden)

        # Same module + same input -> identical output
        max_error = (original_out - tasft_out[0]).abs().max().item()
        assert max_error < 1e-5, (
            f"Dense-path output diverges from original by {max_error:.6f}, "
            f"expected < 1e-5. Wrapper corrupted the attention output."
        )
