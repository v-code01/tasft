"""
TASFT modules: AttnGate, sparse attention, and model integration components.

This package contains the core neural network modules including:
- AttnGate: SeerAttention-based gating modules for block-sparse attention
- TASFTAttention: Patched attention layer with co-training hooks
- GateConfig: Configuration for gate injection into model layers
- patch_model_attention: Utility to patch HF models with TASFTAttention
"""
from tasft.modules.attn_gate import AttnGate, GateOutput
from tasft.modules.tasft_attention import (
    GateConfig,
    TASFTAttention,
    TASFTAttentionOutput,
    patch_model_attention,
)

__all__ = [
    "AttnGate",
    "GateConfig",
    "GateOutput",
    "TASFTAttention",
    "TASFTAttentionOutput",
    "patch_model_attention",
]
