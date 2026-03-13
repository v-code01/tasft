"""Shared type aliases for TASFT.

All types must be imported from here -- no redefinition elsewhere.
"""
from __future__ import annotations

from typing import NewType, TypeAlias

import torch

# Tensor shape annotations (for documentation — not runtime enforced)
# Shape notation: [B=batch, H=heads, S=seq_len, D=head_dim, NB=num_blocks]
BlockMask: TypeAlias = torch.Tensor  # [B, H, NB_q, NB_k] bool
AttentionScores: TypeAlias = torch.Tensor  # [B, H, S, S] float
SoftGateScores: TypeAlias = torch.Tensor  # [B, H, NB_q, NB_k] float in [0,1]
BlockImportance: TypeAlias = torch.Tensor  # [B, H, NB_q, NB_k] float (ground truth)
HiddenStates: TypeAlias = torch.Tensor  # [B, S, hidden_dim]
LayerIndex = NewType("LayerIndex", int)
SparsityRatio = NewType("SparsityRatio", float)  # in [0, 1]

# SparsityProfile: per-layer sparsity measurements
SparsityProfile: TypeAlias = dict[LayerIndex, SparsityRatio]

__all__ = [
    "AttentionScores",
    "BlockImportance",
    "BlockMask",
    "HiddenStates",
    "LayerIndex",
    "SoftGateScores",
    "SparsityProfile",
    "SparsityRatio",
]
