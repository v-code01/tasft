"""TASFT kernels: Triton-based sparse attention kernels.

This package contains:
- Block-sparse FlashAttention-2 Triton kernel with online softmax
- Kernel configuration for per-layer threshold and sparsity tuning
- Automatic backend detection (flash_attn sparse → Triton → dense SDPA)
"""

from tasft.kernels.block_sparse_fa import (
    BlockSparseFlashAttention,
    KernelBackend,
    SparsityStats,
    detect_kernels,
)
from tasft.kernels.kernel_config import KernelConfig, LayerKernelConfig

__all__ = [
    "BlockSparseFlashAttention",
    "KernelBackend",
    "KernelConfig",
    "LayerKernelConfig",
    "SparsityStats",
    "detect_kernels",
]
