"""Shared fixtures for TASFT integration tests.

Provides tiny model factories, synthetic data generators, and cleanup utilities
for running end-to-end tests on CPU without GPU requirements.
"""
from __future__ import annotations

import shutil
import tempfile
from pathlib import Path
from typing import Any, Generator

import pytest
import torch
import torch.nn as nn

from tasft.modules.attn_gate import AttnGate
from tasft.modules.tasft_attention import (
    GateConfig,
    TASFTAttention,
    patch_model_attention,
)
from tasft.types import LayerIndex


# ── Tiny LLaMA-like model for CPU testing ──────────────────────────────


class _TinyAttention(nn.Module):
    """Minimal LLaMA-style attention module for testing."""

    def __init__(self, hidden_dim: int, num_heads: int, head_dim: int) -> None:
        super().__init__()
        self.num_heads = num_heads
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
        q = self.q_proj(hidden_states).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(hidden_states).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(hidden_states).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)

        scale = self.head_dim ** -0.5
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * scale

        # Causal mask
        causal = torch.triu(torch.full((S, S), float("-inf"), device=hidden_states.device), diagonal=1)
        attn_weights = attn_weights + causal.unsqueeze(0).unsqueeze(0)

        attn_probs = torch.softmax(attn_weights, dim=-1)
        out = torch.matmul(attn_probs, v)
        out = out.transpose(1, 2).contiguous().reshape(B, S, self.num_heads * self.head_dim)
        out = self.o_proj(out)

        if output_attentions:
            return (out, attn_weights, None)
        return (out, None, None)


class _TinyDecoderLayer(nn.Module):
    """Minimal decoder layer with attention + FFN."""

    def __init__(self, hidden_dim: int, num_heads: int, head_dim: int) -> None:
        super().__init__()
        self.self_attn = _TinyAttention(hidden_dim, num_heads, head_dim)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )
        self.input_layernorm = nn.LayerNorm(hidden_dim)
        self.post_attention_layernorm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        output_attentions: bool = False,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, ...]:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        attn_out = self.self_attn(
            hidden_states, attention_mask=attention_mask, output_attentions=output_attentions, **kwargs
        )
        hidden_states = residual + attn_out[0]

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = residual + self.mlp(hidden_states)

        if output_attentions:
            return (hidden_states, attn_out[1])
        return (hidden_states,)


class TinyCausalLM(nn.Module):
    """Minimal causal language model mimicking LLaMA structure.

    Structure: model.layers[i].self_attn — compatible with patch_model_attention().
    """

    def __init__(
        self,
        vocab_size: int = 128,
        hidden_dim: int = 64,
        num_layers: int = 4,
        num_heads: int = 4,
        head_dim: int = 16,
    ) -> None:
        super().__init__()
        self.config = type("Config", (), {
            "vocab_size": vocab_size,
            "hidden_size": hidden_dim,
            "num_hidden_layers": num_layers,
            "num_attention_heads": num_heads,
        })()
        self.model = nn.Module()
        self.model.embed_tokens = nn.Embedding(vocab_size, hidden_dim)
        self.model.layers = nn.ModuleList([
            _TinyDecoderLayer(hidden_dim, num_heads, head_dim) for _ in range(num_layers)
        ])
        self.model.norm = nn.LayerNorm(hidden_dim)
        self.lm_head = nn.Linear(hidden_dim, vocab_size, bias=False)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        output_attentions: bool = False,
        **kwargs: Any,
    ) -> Any:
        hidden = self.model.embed_tokens(input_ids)
        all_attn_weights: list[torch.Tensor | None] = []

        for layer in self.model.layers:
            layer_out = layer(hidden, attention_mask=attention_mask, output_attentions=output_attentions)
            hidden = layer_out[0]
            if output_attentions and len(layer_out) > 1:
                all_attn_weights.append(layer_out[1])

        hidden = self.model.norm(hidden)
        logits = self.lm_head(hidden)

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = nn.functional.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )

        # Return an object that mimics HF CausalLMOutput
        return type("Output", (), {
            "loss": loss,
            "logits": logits,
            "attentions": all_attn_weights if output_attentions else None,
        })()


# ── Fixtures ──────────────────────────────────────────────────────────


@pytest.fixture
def tiny_model_config() -> dict[str, int]:
    """Configuration for the tiny test model."""
    return {
        "vocab_size": 128,
        "hidden_dim": 64,
        "num_layers": 4,
        "num_heads": 4,
        "head_dim": 16,
    }


@pytest.fixture
def tiny_model(tiny_model_config: dict[str, int]) -> TinyCausalLM:
    """Create a fresh tiny causal LM for testing."""
    return TinyCausalLM(**tiny_model_config)


@pytest.fixture
def gate_config() -> GateConfig:
    """Default gate configuration for testing."""
    return GateConfig(block_size=32, num_layers=4, default_threshold=0.5)


@pytest.fixture
def patched_model(
    tiny_model: TinyCausalLM, gate_config: GateConfig
) -> tuple[TinyCausalLM, dict[int, TASFTAttention]]:
    """Tiny model with all attention layers patched with TASFTAttention + AttnGate."""
    patched_layers = patch_model_attention(tiny_model, gate_config)
    return tiny_model, patched_layers


@pytest.fixture
def synthetic_batch() -> dict[str, torch.Tensor]:
    """Synthetic training batch with input_ids and labels."""
    batch_size, seq_len, vocab_size = 2, 64, 128
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    labels = input_ids.clone()
    labels[:, :5] = -100  # mask first 5 tokens
    attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long)
    return {"input_ids": input_ids, "labels": labels, "attention_mask": attention_mask}


@pytest.fixture
def tmp_checkpoint_dir() -> Generator[Path, None, None]:
    """Temporary directory for checkpoint testing, cleaned up after test."""
    tmp_dir = Path(tempfile.mkdtemp(prefix="tasft_test_"))
    yield tmp_dir
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
