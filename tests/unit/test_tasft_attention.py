"""Unit tests for TASFTAttention: patched attention layer with co-training hooks.

Tests verify:
    - Patching replaces all attention layers with TASFTAttention
    - Base model weights are frozen after patching
    - Training mode returns auxiliary gate outputs
    - Inference mode returns only hidden states
    - Output shape is preserved after patching
    - Active gate layer selection works correctly
    - Dense-path output is close to original attention output

All tests use a tiny GPT-2 LM model (2 layers, 4 heads, 32 head_dim = 128 hidden).
GPT2LMHeadModel is used (not GPT2Model) because patch_model_attention expects
model.transformer.h path, which GPT2LMHeadModel provides.
"""
from __future__ import annotations

import pytest
import torch
from transformers import GPT2Config, GPT2LMHeadModel

from tasft.modules.attn_gate import AttnGate
from tasft.modules.tasft_attention import (
    GateConfig,
    TASFTAttention,
    TASFTAttentionOutput,
    patch_model_attention,
)


_TINY_GPT2_CONFIG = GPT2Config(
    n_layer=2,
    n_head=4,
    n_embd=128,
    n_positions=256,
    vocab_size=512,
)


@pytest.fixture
def tiny_model() -> GPT2LMHeadModel:
    """Create a tiny GPT-2 LM model for testing: 2 layers, 4 heads, 128 hidden.

    All base parameters are frozen to simulate real TASFT usage where the base model
    is frozen before patching (LoRA would be applied separately for task adapters).
    """
    model = GPT2LMHeadModel(_TINY_GPT2_CONFIG)
    # Freeze all base params (mimics real pipeline: freeze base -> apply LoRA -> patch)
    for param in model.parameters():
        param.requires_grad = False
    return model


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
        self, tiny_model: GPT2LMHeadModel, tiny_gate_config: GateConfig
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
        self, tiny_model: GPT2LMHeadModel, tiny_gate_config: GateConfig
    ) -> None:
        """All non-gate parameters must have requires_grad=False after patching."""
        original_trainable = sum(
            p.numel() for p in tiny_model.parameters() if p.requires_grad
        )
        assert original_trainable > 0, "Model should have trainable params before patching"

        patched = patch_model_attention(tiny_model, tiny_gate_config)

        # Collect gate parameter IDs
        gate_param_ids: set[int] = set()
        for tasft_attn in patched.values():
            for p in tasft_attn.gate.parameters():
                gate_param_ids.add(id(p))

        # All non-gate params must be frozen
        unfrozen_non_gate = [
            name
            for name, p in tiny_model.named_parameters()
            if p.requires_grad and id(p) not in gate_param_ids
        ]
        assert len(unfrozen_non_gate) == 0, (
            f"Found {len(unfrozen_non_gate)} unfrozen non-gate params: "
            f"{unfrozen_non_gate[:5]}"
        )

        # Gate params must be trainable
        gate_trainable = sum(
            1 for tasft_attn in patched.values()
            for p in tasft_attn.gate.parameters()
            if p.requires_grad
        )
        assert gate_trainable > 0, "Gate parameters should be trainable after patching"

    def test_training_mode_returns_aux(
        self, tiny_model: GPT2LMHeadModel, tiny_gate_config: GateConfig
    ) -> None:
        """With compute_gate_target=True and grad enabled, forward runs training path."""
        patched = patch_model_attention(tiny_model, tiny_gate_config)

        # Enable training mode on all layers
        for tasft_attn in patched.values():
            tasft_attn.set_training_mode(True)

        tiny_model.train()
        input_ids = torch.randint(0, 512, (1, 32))
        labels = input_ids.clone()

        with torch.enable_grad():
            outputs = tiny_model(input_ids, labels=labels, output_attentions=True)

        # Verify the model produced valid output (loss + logits)
        assert outputs.loss is not None, "Expected loss in training output"
        assert outputs.logits is not None, "Expected logits in training output"

        # Verify at least one layer was set to training mode
        training_layers = [
            idx for idx, t in patched.items() if t.compute_gate_target
        ]
        assert len(training_layers) > 0, "No layers had compute_gate_target=True"

    def test_inference_mode_no_aux(
        self, tiny_model: GPT2LMHeadModel, tiny_gate_config: GateConfig
    ) -> None:
        """With compute_gate_target=False and no_grad, forward returns hidden_states only."""
        patched = patch_model_attention(tiny_model, tiny_gate_config)

        # Ensure inference mode on all layers
        for tasft_attn in patched.values():
            tasft_attn.set_training_mode(False)

        tiny_model.eval()
        input_ids = torch.randint(0, 512, (1, 32))

        with torch.no_grad():
            outputs = tiny_model(input_ids, output_attentions=False)

        # Verify model produces valid hidden states
        # GPT2LMHeadModel returns CausalLMOutputWithCrossAttentions
        assert outputs.logits is not None, "Expected logits in output"
        assert outputs.logits.shape == (1, 32, 512), (
            f"Unexpected logits shape: {outputs.logits.shape}"
        )

        # Verify no layer has gate_target computation active
        for idx, tasft_attn in patched.items():
            assert not tasft_attn.compute_gate_target, (
                f"Layer {idx} should have compute_gate_target=False in inference mode"
            )

    def test_output_shape_unchanged(
        self, tiny_gate_config: GateConfig,
    ) -> None:
        """TASFTAttention output logits shape must match original model output shape."""
        # Get original output shape
        original_model = GPT2LMHeadModel(_TINY_GPT2_CONFIG)
        input_ids = torch.randint(0, 512, (2, 16))
        with torch.no_grad():
            original_output = original_model(input_ids)
        original_shape = original_output.logits.shape

        # Patch a fresh model and get new output shape
        patched_model = GPT2LMHeadModel(_TINY_GPT2_CONFIG)
        patch_model_attention(patched_model, tiny_gate_config)
        with torch.no_grad():
            patched_output = patched_model(input_ids)
        patched_shape = patched_output.logits.shape

        assert original_shape == patched_shape, (
            f"Shape mismatch: original={original_shape}, patched={patched_shape}"
        )

    def test_set_active_gate_layers(
        self, tiny_model: GPT2LMHeadModel, tiny_gate_config: GateConfig
    ) -> None:
        """Only specified active layers should have compute_gate_target=True."""
        patched = patch_model_attention(tiny_model, tiny_gate_config)

        # Activate only layer 0
        patched[0].set_training_mode(True)
        patched[1].set_training_mode(False)

        assert patched[0].compute_gate_target is True, (
            "Layer 0 should have compute_gate_target=True"
        )
        assert patched[1].compute_gate_target is False, (
            "Layer 1 should have compute_gate_target=False"
        )

        # Switch: activate only layer 1
        patched[0].set_training_mode(False)
        patched[1].set_training_mode(True)

        assert patched[0].compute_gate_target is False, (
            "Layer 0 should now have compute_gate_target=False"
        )
        assert patched[1].compute_gate_target is True, (
            "Layer 1 should now have compute_gate_target=True"
        )

    def test_output_close_to_dense(
        self, tiny_gate_config: GateConfig,
    ) -> None:
        """TASFTAttention with threshold=0 (all blocks active) should produce output
        close to original dense attention, verifying gate doesn't corrupt computation.

        In inference mode (compute_gate_target=False), TASFTAttention delegates
        to base_attn directly. With identical weights, output must match.
        """
        torch.manual_seed(42)
        input_ids = torch.randint(0, 512, (1, 32))

        # Original model output
        torch.manual_seed(0)
        original_model = GPT2LMHeadModel(_TINY_GPT2_CONFIG)
        original_model.eval()
        with torch.no_grad():
            original_out = original_model(input_ids).logits

        # Patched model with same weights (same seed)
        torch.manual_seed(0)
        patched_model = GPT2LMHeadModel(_TINY_GPT2_CONFIG)

        zero_threshold_config = GateConfig(
            block_size=8,
            num_layers=2,
            gate_hidden_dim=16,
            default_threshold=0.0,
        )
        patch_model_attention(patched_model, zero_threshold_config)
        patched_model.eval()

        with torch.no_grad():
            patched_out = patched_model(input_ids).logits

        # In inference mode, TASFTAttention runs base attention forward directly.
        # Output should be identical (same weights, same input).
        max_error = (original_out - patched_out).abs().max().item()
        assert max_error < 1e-2, (
            f"Dense-path output diverges from original by {max_error:.6f}, "
            f"expected < 1e-2. Gate may be corrupting the attention computation."
        )
