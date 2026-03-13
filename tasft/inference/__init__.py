"""
TASFT inference: sparse attention decode with pre-computed gate masks.

This package contains:
- Sparse decode engine using pre-computed block masks
- KV-cache optimization with block-sparse patterns
- Throughput measurement and adaptive batching
- vLLM integration via monkey-patching
"""

from tasft.inference.tasft_model import InferenceBenchmark, TASFTInferenceModel
from tasft.inference.vllm_patch import patch_vllm_attention

__all__ = [
    "InferenceBenchmark",
    "TASFTInferenceModel",
    "patch_vllm_attention",
]
