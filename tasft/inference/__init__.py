"""
TASFT inference: sparse attention decode with pre-computed gate masks.

This package contains:
- Sparse decode engine using pre-computed block masks
- KV-cache optimization with block-sparse patterns
- Throughput measurement and adaptive batching
- vLLM integration via monkey-patching
- vLLM version compatibility layer and attention metadata adapters
"""

from tasft.inference.tasft_model import InferenceBenchmark, TASFTInferenceModel
from tasft.inference.vllm_compat import (
    AttnMetadataAdapter,
    VLLMVersion,
    check_vllm_compatibility,
    detect_vllm_version,
    get_attn_metadata_adapter,
    validate_worker_structure,
)
from tasft.inference.vllm_patch import patch_vllm_attention

__all__ = [
    "AttnMetadataAdapter",
    "InferenceBenchmark",
    "TASFTInferenceModel",
    "VLLMVersion",
    "check_vllm_compatibility",
    "detect_vllm_version",
    "get_attn_metadata_adapter",
    "patch_vllm_attention",
    "validate_worker_structure",
]
