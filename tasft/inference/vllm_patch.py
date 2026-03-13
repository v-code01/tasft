"""
vLLM Integration Patch for TASFT.

Monkey-patches vLLM's attention backend to use AttnGate + BlockSparseFlashAttention.
Compatible with vLLM >= 0.4.0 through 0.8.x.

IMPORTANT: This patch is applied once at startup, not per-request.
           Thread-safe: uses a module-level lock for patch application.
           Does NOT break vLLM's PagedAttention KV cache management.

Uses vllm_compat.py for version detection, compatibility checking, and
attn_metadata field normalization to survive vLLM minor version bumps
without silent breakage.

Preconditions:
    - vLLM >= 0.4.0 installed and importable
    - TASFT inference model loaded via TASFTInferenceModel.load_bundle()
    - CUDA available

Postconditions:
    - vLLM attention forward replaced with gate-driven sparse version
    - PagedAttention KV cache management preserved
    - Patch is idempotent (safe to call multiple times)

Complexity: O(L) for patch application where L = number of attention layers
"""
from __future__ import annotations

import threading
from typing import TYPE_CHECKING, Any

import torch
from torch import nn

from tasft.exceptions import InferenceError
from tasft.inference.vllm_compat import (
    AttnMetadataAdapter,
    VLLMVersion,
    check_vllm_compatibility,
    detect_vllm_version,
    get_attn_metadata_adapter,
    validate_worker_structure,
)
from tasft.observability.logging import get_logger

if TYPE_CHECKING:
    from tasft.inference.tasft_model import TASFTInferenceModel
    from tasft.modules.attn_gate import AttnGate

logger = get_logger("tasft.inference.vllm_patch")

_patch_lock = threading.Lock()
_patched_workers: set[int] = set()
# Cached version detection result; populated on first patch_vllm_attention call
_detected_version: VLLMVersion | None = None
_version_detected: bool = False


class TASFTvLLMAttentionBackend(nn.Module):
    """vLLM-compatible attention backend using TASFT gate-driven sparse attention.

    Implements the vLLM AttentionBackend interface while routing attention through
    AttnGate for block mask generation and BlockSparseFlashAttention for compute.

    Preconditions:
        - gate is a loaded, eval-mode AttnGate matching the layer's head config
        - threshold_tau in (0, 1)
        - block_size in {32, 64, 128}

    Postconditions:
        - Output shape matches standard vLLM attention output
        - KV cache format preserved for PagedAttention compatibility

    Complexity: O(B * H * S^2 * (1 - sparsity) * D / block_size^2) for sparse path.
    """

    def __init__(
        self,
        gate: AttnGate,
        layer_idx: int,
        threshold_tau: float,
        block_size: int,
        min_sparsity_for_speedup: float,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        vllm_version: VLLMVersion | None = None,
        adapter_cls: type[AttnMetadataAdapter] | None = None,
    ) -> None:
        """Initialize TASFT vLLM attention backend.

        Args:
            gate: Trained AttnGate for this layer.
            layer_idx: Transformer layer index.
            threshold_tau: Gate binarization threshold.
            block_size: Attention block size in tokens.
            min_sparsity_for_speedup: Min sparsity to use sparse kernel.
            num_heads: Number of query heads.
            num_kv_heads: Number of KV heads (for GQA).
            head_dim: Dimension per head.
            vllm_version: Detected vLLM version for adapter dispatch.
            adapter_cls: AttnMetadataAdapter class for normalizing field access.
        """
        super().__init__()

        # Validate GQA head divisibility — silent truncation from integer
        # division would produce incorrect attention patterns
        if num_heads % num_kv_heads != 0:
            msg = f"num_heads ({num_heads}) must be divisible by num_kv_heads ({num_kv_heads})"
            raise InferenceError(
                msg,
                context={"num_heads": num_heads, "num_kv_heads": num_kv_heads},
            )

        self.gate = gate
        self.layer_idx = layer_idx
        self.threshold_tau = threshold_tau
        self.block_size = block_size
        self.min_sparsity_for_speedup = min_sparsity_for_speedup
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self._kernel: Any | None = None
        self._original_forward: Any | None = None  # Set during patching for decode delegation
        # Version-aware metadata adapter — defaults to AttnMetadataAdapter
        # which handles all known versions via runtime attribute probing
        self._vllm_version = vllm_version or VLLMVersion(0, 4, 0)
        self._adapter_cls = adapter_cls or AttnMetadataAdapter

    def _get_kernel(self) -> Any:
        """Lazy-load BlockSparseFlashAttention kernel.

        Complexity: O(1) after first call.
        """
        if self._kernel is None:
            from tasft.kernels.block_sparse_fa import BlockSparseFlashAttention

            self._kernel = BlockSparseFlashAttention(block_size=self.block_size)
        return self._kernel

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor | None = None,
        attn_metadata: Any = None,
        kv_scale: float = 1.0,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Forward pass implementing vLLM AttentionBackend interface.

        Reshapes KV from PagedAttention format, runs AttnGate for hard mask
        generation, then dispatches to sparse or dense attention.

        Args:
            query: [num_tokens, num_heads * head_dim] query tensor.
            key: [num_tokens, num_kv_heads * head_dim] key tensor.
            value: [num_tokens, num_kv_heads * head_dim] value tensor.
            kv_cache: Optional paged KV cache tensor.
            attn_metadata: vLLM attention metadata (prefill/decode info).
            kv_scale: Scale factor for KV values (default 1.0).
            **kwargs: Additional arguments from vLLM.

        Returns:
            [num_tokens, num_heads * head_dim] attention output.

        Complexity: O(num_tokens * num_heads * seq_len * (1 - sparsity) * head_dim).
        """
        num_tokens = query.shape[0]

        # Reshape from vLLM flat format to [B, H, S, D]
        # vLLM passes tokens in flat batch — treat as B=1, S=num_tokens for gating
        q = query.view(1, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        k = key.view(1, num_tokens, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = value.view(1, num_tokens, self.num_kv_heads, self.head_dim).transpose(1, 2)

        # Apply KV scale
        if kv_scale != 1.0:
            k = k * kv_scale
            v = v * kv_scale

        # Handle GQA: expand KV heads to match Q heads
        if self.num_kv_heads < self.num_heads:
            n_rep = self.num_heads // self.num_kv_heads
            k = k.repeat_interleave(n_rep, dim=1)
            v = v.repeat_interleave(n_rep, dim=1)

        # Handle paged KV cache: if decoding with cache, prepend cached KV
        if kv_cache is not None and attn_metadata is not None:
            # Use version-aware adapter for safe field access across vLLM versions
            adapted = self._adapter_cls(attn_metadata, self._vllm_version)
            if not adapted.is_prefill:
                # During decode, vLLM manages paging — use standard attention
                # for single-token decode steps (no sparsity benefit at S=1)
                return self._dense_attention_flat(query, key, value, kv_cache, attn_metadata)

        # Gate-driven sparse attention for prefill
        gate_output = self.gate(q, k, threshold=self.threshold_tau)

        if gate_output.sparsity_ratio >= self.min_sparsity_for_speedup:
            kernel = self._get_kernel()
            attn_output = kernel.forward(q, k, v, gate_output.hard_mask, causal=True)
        else:
            # Dense fallback
            scale = 1.0 / (self.head_dim**0.5)
            attn_weights = torch.matmul(q, k.transpose(-2, -1)) * scale
            attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32)
            attn_weights = attn_weights.to(v.dtype)
            attn_output = torch.matmul(attn_weights, v)

        # Reshape back to vLLM flat format: [1, H, S, D] -> [num_tokens, H*D]
        return attn_output.transpose(1, 2).contiguous().reshape(num_tokens, -1)

    def _dense_attention_flat(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: Any,
    ) -> torch.Tensor:
        """Dense attention fallback for single-token decode steps.

        During autoregressive decode, each step generates 1 token attending to
        the full cached KV context via PagedAttention. No block-level sparsity
        benefit exists at S=1, so we delegate to the original vLLM forward which
        correctly handles the paged KV cache.

        CRITICAL: We must use the original vLLM forward here because it manages
        PagedAttention lookups into kv_cache using attn_metadata slot mappings.
        Computing attention only on the current token's Q/K (ignoring the cache)
        would produce garbage output during autoregressive generation.

        Args:
            query: [num_tokens, num_heads * head_dim] flat query.
            key: [num_tokens, num_kv_heads * head_dim] flat key.
            value: [num_tokens, num_kv_heads * head_dim] flat value.
            kv_cache: Paged KV cache managed by vLLM.
            attn_metadata: vLLM attention metadata.

        Returns:
            [num_tokens, num_heads * head_dim] attention output.

        Complexity: O(num_tokens * num_heads * cached_seq_len * head_dim).
        """
        if self._original_forward is not None:
            # Delegate to vLLM's original attention which handles PagedAttention
            # and correctly attends to the full cached KV context
            return self._original_forward(
                query, key, value, kv_cache=kv_cache, attn_metadata=attn_metadata,
            )

        # Fallback: if original forward not available (should not happen in
        # properly patched setup), compute basic SDPA as last resort.
        # This path only produces correct results when kv_cache is empty.
        logger.warning(
            "[VLLM_DECODE_FALLBACK] No original forward available for decode; "
            "output may be incorrect if KV cache is populated",
            layer_idx=self.layer_idx,
        )
        num_tokens = query.shape[0]
        q = query.view(num_tokens, self.num_heads, self.head_dim)
        k = key.view(num_tokens, self.num_kv_heads, self.head_dim)
        v = value.view(num_tokens, self.num_kv_heads, self.head_dim)

        # GQA expansion
        if self.num_kv_heads < self.num_heads:
            n_rep = self.num_heads // self.num_kv_heads
            k = k.repeat_interleave(n_rep, dim=1)
            v = v.repeat_interleave(n_rep, dim=1)

        # Standard scaled dot-product — only correct when no cached context
        scale = 1.0 / (self.head_dim**0.5)
        attn_weights = torch.einsum("thd,shd->ths", q * scale, k)
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32)
        attn_weights = attn_weights.to(v.dtype)
        output = torch.einsum("ths,shd->thd", attn_weights, v)

        return output.reshape(num_tokens, -1)

    @staticmethod
    def get_cache_block_size(
        block_size: int,
        head_size: int,
        num_heads: int,
        dtype: torch.dtype,
    ) -> int:
        """Compute KV cache block size in bytes for vLLM PagedAttention.

        Standard vLLM interface method. Returns the memory footprint of a single
        KV cache block for capacity planning.

        Args:
            block_size: Number of tokens per cache block.
            head_size: Dimension per attention head.
            num_heads: Number of KV heads.
            dtype: Data type of cache entries.

        Returns:
            Size in bytes of one KV cache block (key + value).

        Complexity: O(1).
        """
        element_size = torch.tensor([], dtype=dtype).element_size()
        # key + value, each: [block_size, num_heads, head_size]
        return 2 * block_size * num_heads * head_size * element_size


def _is_prefill_phase(attn_metadata: Any) -> bool:
    """Determine if the current attention operation is in prefill phase.

    Delegates to AttnMetadataAdapter for version-safe field access.
    This function is retained for backward compatibility with any external
    code that may call it directly.

    Args:
        attn_metadata: vLLM attention metadata object.

    Returns:
        True if in prefill phase, False if in decode phase.

    Complexity: O(1).
    """
    # Use the adapter with a default version — the adapter probes attributes
    # at runtime so the version is only used for future dispatch
    adapted = AttnMetadataAdapter(attn_metadata, VLLMVersion(0, 4, 0))
    return adapted.is_prefill


def _extract_vllm_attention_modules(worker_model: nn.Module) -> list[nn.Module]:
    """Extract attention modules from a vLLM model worker.

    Supports vLLM's internal module naming conventions across model types.

    Args:
        worker_model: The model held by a vLLM Worker.

    Returns:
        Ordered list of attention modules.

    Raises:
        InferenceError: If no attention modules found.

    Complexity: O(N) where N = total number of modules.
    """
    attn_modules: list[nn.Module] = []
    for _name, module in worker_model.named_modules():
        cls_name = type(module).__name__
        # vLLM wraps attention in various classes
        if cls_name.endswith("Attention") and (
            hasattr(module, "qkv_proj") or hasattr(module, "q_proj")
        ):
            attn_modules.append(module)

    if not attn_modules:
        msg = "No attention modules found in vLLM worker model"
        raise InferenceError(
            msg,
            context={"model_class": type(worker_model).__name__},
        )

    return attn_modules


def patch_vllm_attention(
    tasft_model: TASFTInferenceModel,
    vllm_worker: Any,
) -> None:
    """Patch vLLM's attention to use TASFT gate-driven sparse attention.

    Thread-safe: uses a module-level lock. Idempotent: safe to call multiple times.
    Applied once at startup, not per-request.

    Does NOT break vLLM's PagedAttention KV cache management.

    Steps:
        1. Acquire lock, check _patch_applied to prevent double-patching
        2. Get attention modules from vllm_worker.model
        3. For each layer, inject corresponding AttnGate + BlockSparseFlashAttention
        4. Replace attention forward with TASFT sparse version
        5. Set _patch_applied = True, release lock

    Preconditions:
        - tasft_model is fully loaded via load_bundle()
        - vllm_worker has a .model attribute with attention modules
        - Number of attention layers in vLLM model matches tasft_model gates

    Postconditions:
        - All prefill attention ops route through TASFT sparse path
        - Decode attention unchanged (single-token, no sparsity benefit)
        - _patch_applied is True

    Args:
        tasft_model: Loaded TASFTInferenceModel with gates and kernel config.
        vllm_worker: vLLM Worker instance holding the model.

    Raises:
        InferenceError: If patch cannot be applied (architecture mismatch, etc).

    Complexity: O(L) where L = number of attention layers.
    """
    global _detected_version, _version_detected  # noqa: PLW0603

    with _patch_lock:
        worker_id = id(vllm_worker)
        if worker_id in _patched_workers:
            logger.info(
                "[VLLM_PATCH] Patch already applied to this worker, skipping",
                worker_id=worker_id,
            )
            return

        # --- Version detection and compatibility check (once per process) ---
        if not _version_detected:
            _detected_version = detect_vllm_version()
            _version_detected = True

            if _detected_version is None:
                msg = (
                    "vLLM is not installed or its version cannot be determined. "
                    "patch_vllm_attention requires vLLM >= 0.4.0."
                )
                raise InferenceError(
                    msg,
                    context={"worker_type": type(vllm_worker).__name__},
                )

            logger.info(
                "[VLLM_PATCH] Detected vLLM version",
                vllm_version=str(_detected_version),
                major=_detected_version.major,
                minor=_detected_version.minor,
                patch=_detected_version.patch,
            )

            compat_warnings = check_vllm_compatibility(_detected_version)
            for warning_msg in compat_warnings:
                logger.warning(
                    "[VLLM_COMPAT] Compatibility note",
                    warning=warning_msg,
                    vllm_version=str(_detected_version),
                )

        # At this point _detected_version is guaranteed non-None because
        # the first call either set it or raised InferenceError
        assert _detected_version is not None  # noqa: S101 — invariant, not runtime check

        # Resolve the adapter class for this vLLM version
        adapter_cls = get_attn_metadata_adapter(_detected_version)

        # --- Worker structure validation ---
        structure_issues = validate_worker_structure(vllm_worker)
        if structure_issues:
            for issue in structure_issues:
                logger.warning(
                    "[VLLM_PATCH] Worker structure issue",
                    issue=issue,
                    worker_type=type(vllm_worker).__name__,
                )
            # Structure issues are warnings, not hard failures — the patching
            # code below has its own error handling for missing attributes.
            # Only fail if there is no model at all.
            has_model = hasattr(vllm_worker, "model_runner") or hasattr(
                vllm_worker, "model"
            )
            if not has_model:
                msg = (
                    "vLLM worker has no accessible model. "
                    f"Issues: {'; '.join(structure_issues)}"
                )
                raise InferenceError(
                    msg,
                    context={
                        "worker_type": type(vllm_worker).__name__,
                        "issues": structure_issues,
                        "vllm_version": str(_detected_version),
                    },
                )

        logger.info(
            "[VLLM_PATCH] Applying TASFT sparse attention patch to vLLM",
            worker_id=worker_id,
            vllm_version=str(_detected_version),
        )

        # Get the model from the worker
        if hasattr(vllm_worker, "model_runner"):
            worker_model = vllm_worker.model_runner.model
        elif hasattr(vllm_worker, "model"):
            worker_model = vllm_worker.model
        else:
            msg = "Cannot find model in vLLM worker — unsupported vLLM version"
            raise InferenceError(
                msg,
                context={
                    "worker_type": type(vllm_worker).__name__,
                    "vllm_version": str(_detected_version),
                },
            )

        vllm_attn_modules = _extract_vllm_attention_modules(worker_model)
        num_vllm_layers = len(vllm_attn_modules)
        num_tasft_layers = len(tasft_model.gates)

        if num_vllm_layers != num_tasft_layers:
            msg = (
                f"Layer count mismatch: vLLM has {num_vllm_layers} attention layers "
                f"but TASFT has {num_tasft_layers} gates"
            )
            raise InferenceError(
                msg,
                context={
                    "vllm_layers": num_vllm_layers,
                    "tasft_layers": num_tasft_layers,
                    "vllm_version": str(_detected_version),
                },
            )

        kernel_config = tasft_model.kernel_config

        for layer_idx, vllm_attn in enumerate(vllm_attn_modules):
            gate = tasft_model.gates[str(layer_idx)]
            threshold = kernel_config.get_layer_threshold(layer_idx)
            block_size = kernel_config.get_layer_block_size(layer_idx)

            # Detect head configuration from vLLM module
            if hasattr(vllm_attn, "num_heads"):
                num_heads = vllm_attn.num_heads
            elif hasattr(vllm_attn, "num_attention_heads"):
                num_heads = vllm_attn.num_attention_heads
            else:
                msg = f"Cannot determine num_heads from vLLM attention module at layer {layer_idx}"
                raise InferenceError(
                    msg,
                    context={
                        "module_type": type(vllm_attn).__name__,
                        "vllm_version": str(_detected_version),
                    },
                )

            num_kv_heads = getattr(vllm_attn, "num_kv_heads", num_heads)
            head_dim = getattr(vllm_attn, "head_dim", gate.head_dim)

            backend = TASFTvLLMAttentionBackend(
                gate=gate,
                layer_idx=layer_idx,
                threshold_tau=threshold,
                block_size=block_size,
                min_sparsity_for_speedup=kernel_config.min_sparsity_for_speedup,
                num_heads=num_heads,
                num_kv_heads=num_kv_heads,
                head_dim=head_dim,
                vllm_version=_detected_version,
                adapter_cls=adapter_cls,
            )

            # Store original forward on both the module (for rollback) and
            # the backend (for decode delegation to PagedAttention)
            vllm_attn._tasft_original_forward = vllm_attn.forward  # type: ignore[attr-defined]
            vllm_attn._tasft_backend = backend  # type: ignore[attr-defined]
            backend._original_forward = vllm_attn.forward

            # Replace forward with TASFT sparse version
            def _make_patched_forward(
                _backend: TASFTvLLMAttentionBackend,
            ) -> Any:
                """Create a closure-captured patched forward for a specific layer.

                Closure captures the backend to avoid late-binding issues.
                """

                def _patched_forward(
                    query: torch.Tensor,
                    key: torch.Tensor,
                    value: torch.Tensor,
                    kv_cache: torch.Tensor | None = None,
                    attn_metadata: Any = None,
                    kv_scale: float = 1.0,
                    **fwd_kwargs: Any,
                ) -> torch.Tensor:
                    return _backend(
                        query=query,
                        key=key,
                        value=value,
                        kv_cache=kv_cache,
                        attn_metadata=attn_metadata,
                        kv_scale=kv_scale,
                        **fwd_kwargs,
                    )

                return _patched_forward

            import types

            vllm_attn.forward = types.MethodType(  # type: ignore[method-assign]
                lambda self, *args, _pf=_make_patched_forward(backend), **kw: _pf(*args, **kw),
                vllm_attn,
            )

            logger.debug(
                "[VLLM_PATCH] Patched layer",
                layer_idx=layer_idx,
                threshold=threshold,
                block_size=block_size,
                num_heads=num_heads,
                num_kv_heads=num_kv_heads,
            )

        _patched_workers.add(worker_id)
        logger.info(
            "[VLLM_PATCH] TASFT patch applied successfully",
            worker_id=worker_id,
            num_layers_patched=num_vllm_layers,
            vllm_version=str(_detected_version),
        )


def unpatch_vllm_attention(vllm_worker: Any) -> None:
    """Remove TASFT patch and restore original vLLM attention.

    Thread-safe. Idempotent.

    Preconditions: patch_vllm_attention was previously called on this worker.
    Postconditions: Original vLLM attention forward methods restored.

    Args:
        vllm_worker: The same vLLM Worker that was patched.

    Complexity: O(L) where L = number of attention layers.
    """
    with _patch_lock:
        worker_id = id(vllm_worker)
        if worker_id not in _patched_workers:
            logger.info(
                "[VLLM_UNPATCH] No patch applied to this worker, skipping",
                worker_id=worker_id,
            )
            return

        if hasattr(vllm_worker, "model_runner"):
            worker_model = vllm_worker.model_runner.model
        elif hasattr(vllm_worker, "model"):
            worker_model = vllm_worker.model
        else:
            # Unrecognized worker type — still reset patch state so
            # a subsequent patch_vllm_attention call can re-apply
            _patched_workers.discard(worker_id)
            logger.warning(
                "[VLLM_UNPATCH] Unrecognized worker type, reset patch state only",
                worker_type=type(vllm_worker).__name__,
            )
            return

        for _name, module in worker_model.named_modules():
            if hasattr(module, "_tasft_original_forward"):
                module.forward = module._tasft_original_forward  # type: ignore[method-assign]
                del module._tasft_original_forward  # type: ignore[attr-defined]
                if hasattr(module, "_tasft_backend"):
                    del module._tasft_backend  # type: ignore[attr-defined]

        _patched_workers.discard(worker_id)
        logger.info("[VLLM_UNPATCH] TASFT patch removed successfully", worker_id=worker_id)


def is_patched() -> bool:
    """Check if TASFT vLLM patch is currently applied to any worker.

    Thread-safe read of patch state. Acquires _patch_lock to ensure
    visibility across threads (safe for GIL-free Python per PEP 703).

    Returns:
        True if at least one worker has been patched and not unpatched.

    Complexity: O(1).
    """
    with _patch_lock:
        return len(_patched_workers) > 0


__all__ = [
    "TASFTvLLMAttentionBackend",
    "is_patched",
    "patch_vllm_attention",
    "unpatch_vllm_attention",
]
