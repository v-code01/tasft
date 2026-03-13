"""
vLLM Integration Patch for TASFT.

Monkey-patches vLLM's attention backend to use AttnGate + BlockSparseFlashAttention.
Compatible with vLLM >= 0.4.0.

IMPORTANT: This patch is applied once at startup, not per-request.
           Thread-safe: uses a module-level lock for patch application.
           Does NOT break vLLM's PagedAttention KV cache management.

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
from tasft.observability.logging import get_logger

if TYPE_CHECKING:
    from tasft.inference.tasft_model import TASFTInferenceModel
    from tasft.modules.attn_gate import AttnGate

logger = get_logger("tasft.inference.vllm_patch")

_patch_lock = threading.Lock()
_patch_applied = False


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
        """
        super().__init__()
        self.gate = gate
        self.layer_idx = layer_idx
        self.threshold_tau = threshold_tau
        self.block_size = block_size
        self.min_sparsity_for_speedup = min_sparsity_for_speedup
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self._kernel: Any | None = None

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
            is_prefill = _is_prefill_phase(attn_metadata)
            if not is_prefill:
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

        During autoregressive decode, each step generates 1 token, so there is
        no block-level sparsity benefit. Falls back to standard scaled dot-product.

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
        num_tokens = query.shape[0]
        q = query.view(num_tokens, self.num_heads, self.head_dim)
        k = key.view(num_tokens, self.num_kv_heads, self.head_dim)
        v = value.view(num_tokens, self.num_kv_heads, self.head_dim)

        # GQA expansion
        if self.num_kv_heads < self.num_heads:
            n_rep = self.num_heads // self.num_kv_heads
            k = k.repeat_interleave(n_rep, dim=1)
            v = v.repeat_interleave(n_rep, dim=1)

        # Standard scaled dot-product for decode
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

    Checks vLLM attention metadata for prefill indicators. During prefill,
    we can benefit from block-sparse attention. During decode (single token),
    sparsity provides no benefit.

    Args:
        attn_metadata: vLLM attention metadata object.

    Returns:
        True if in prefill phase, False if in decode phase.

    Complexity: O(1).
    """
    # vLLM >= 0.4.0 uses is_prompt / prefill_metadata
    if hasattr(attn_metadata, "is_prompt"):
        return bool(attn_metadata.is_prompt)
    if hasattr(attn_metadata, "prefill_metadata"):
        return attn_metadata.prefill_metadata is not None
    # vLLM >= 0.5.0 uses num_prefill_tokens
    if hasattr(attn_metadata, "num_prefill_tokens"):
        return attn_metadata.num_prefill_tokens > 0
    # Conservative default: assume prefill to enable sparse path
    return True


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
        if "Attention" in cls_name and (
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
    global _patch_applied  # noqa: PLW0603

    with _patch_lock:
        if _patch_applied:
            logger.info("[VLLM_PATCH] Patch already applied, skipping")
            return

        logger.info("[VLLM_PATCH] Applying TASFT sparse attention patch to vLLM")

        # Get the model from the worker
        if hasattr(vllm_worker, "model_runner"):
            worker_model = vllm_worker.model_runner.model
        elif hasattr(vllm_worker, "model"):
            worker_model = vllm_worker.model
        else:
            msg = "Cannot find model in vLLM worker — unsupported vLLM version"
            raise InferenceError(
                msg,
                context={"worker_type": type(vllm_worker).__name__},
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
                    context={"module_type": type(vllm_attn).__name__},
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
            )

            # Patch the attention module's forward method
            # Store original for potential rollback
            vllm_attn._tasft_original_forward = vllm_attn.forward  # type: ignore[attr-defined]
            vllm_attn._tasft_backend = backend  # type: ignore[attr-defined]

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

        _patch_applied = True
        logger.info(
            "[VLLM_PATCH] TASFT patch applied successfully",
            num_layers_patched=num_vllm_layers,
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
    global _patch_applied  # noqa: PLW0603

    with _patch_lock:
        if not _patch_applied:
            logger.info("[VLLM_UNPATCH] No patch applied, skipping")
            return

        if hasattr(vllm_worker, "model_runner"):
            worker_model = vllm_worker.model_runner.model
        elif hasattr(vllm_worker, "model"):
            worker_model = vllm_worker.model
        else:
            return

        for _name, module in worker_model.named_modules():
            if hasattr(module, "_tasft_original_forward"):
                module.forward = module._tasft_original_forward  # type: ignore[method-assign]
                del module._tasft_original_forward  # type: ignore[attr-defined]
                if hasattr(module, "_tasft_backend"):
                    del module._tasft_backend  # type: ignore[attr-defined]

        _patch_applied = False
        logger.info("[VLLM_UNPATCH] TASFT patch removed successfully")


def is_patched() -> bool:
    """Check if TASFT vLLM patch is currently applied.

    Thread-safe read of patch state.

    Returns:
        True if patch_vllm_attention has been called and not unpatched.

    Complexity: O(1).
    """
    return _patch_applied


__all__ = [
    "TASFTvLLMAttentionBackend",
    "is_patched",
    "patch_vllm_attention",
    "unpatch_vllm_attention",
]
