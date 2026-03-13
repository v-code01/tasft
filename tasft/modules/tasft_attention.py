"""
TASFTAttention: Patched attention layer for TASFT co-training.

Wraps an existing LLM attention module (LlamaAttention, Qwen2Attention) and injects:
  - LoRA adapters for task specialization (via PEFT, applied externally)
  - AttnGate for block importance prediction (co-trained)

Training path:
    1. Full LoRA-augmented attention forward -> logits + full attn score matrix
    2. AttnGate(Q, K) -> predicted block mask + distillation target from full attn
    3. Returns TASFTAttentionOutput with aux for loss computation

Inference path:
    1. AttnGate(Q, K) -> hard block mask (fast, tiny forward)
    2. Block-sparse attention with mask (via kernel or masked dense)
    3. Returns hidden_states only (no aux overhead)

Complexity: Training adds O((S/B)^2) gate overhead per layer.
            Inference replaces O(S^2) dense with O(S^2 * (1-sparsity)) sparse.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from tasft.exceptions import ValidationError
from tasft.modules.attn_gate import AttnGate, GateOutput
from tasft.types import (
    AttentionScores,
    BlockImportance,
    HiddenStates,
    LayerIndex,
)

_KERNEL_NOT_TRIED = object()


@dataclass(frozen=True)
class GateConfig:
    """Configuration for AttnGate modules injected into model layers.

    Attributes:
        block_size: Token block size for importance scoring.
        num_layers: Number of transformer layers to patch.
        gate_hidden_dim: Hidden dim of gate MLP per head. None -> max(32, head_dim // 4).
        default_threshold: Default tau for hard mask binarization.
    """

    block_size: int = 64
    num_layers: int = 32
    gate_hidden_dim: int | None = None
    default_threshold: float = 0.5

    def __post_init__(self) -> None:
        """Validate configuration values."""
        if self.block_size <= 0:
            msg = f"block_size must be positive, got {self.block_size}"
            raise ValidationError(
                msg,
                context={"block_size": self.block_size},
            )
        if self.num_layers <= 0:
            msg = f"num_layers must be positive, got {self.num_layers}"
            raise ValidationError(
                msg,
                context={"num_layers": self.num_layers},
            )
        if not 0.0 < self.default_threshold < 1.0:
            msg = f"default_threshold must be in (0, 1) exclusive, got {self.default_threshold}"
            raise ValidationError(
                msg,
                context={"default_threshold": self.default_threshold},
            )


@dataclass(frozen=True)
class TASFTAttentionOutput:
    """Output of TASFTAttention forward pass.

    Attributes:
        hidden_states: Output hidden states [B, S, hidden_dim].
        attn_weights: Full attention weights [B, H, S, S] if computed (training only), else None.
        gate_output: GateOutput from AttnGate if gate was invoked, else None.
        gate_target_scores: Ground-truth block importance [B, H, NB, NB] for gate distillation
                            (training only, when compute_gate_target=True), else None.
        layer_idx: Layer index for this attention module.
    """

    hidden_states: HiddenStates
    attn_weights: AttentionScores | None
    gate_output: GateOutput | None
    gate_target_scores: BlockImportance | None
    layer_idx: LayerIndex


class TASFTAttention(nn.Module):
    """Patched attention layer for TASFT co-training and sparse inference.

    Wraps a HuggingFace attention module (e.g., LlamaAttention) and adds:
    1. AttnGate for block importance prediction
    2. Gate target computation from full attention scores (training)
    3. Block-sparse masking for efficient inference

    The base attention module's weights remain frozen. Only the gate parameters
    and any externally-applied LoRA adapters receive gradients.

    Preconditions:
        - base_attn must expose a standard HF attention interface
        - gate must be configured for the same num_heads and head_dim as base_attn
        - layer_idx must be a valid LayerIndex

    Postconditions:
        - Base model weights are never modified
        - Gate parameters accumulate gradients during training
        - Output hidden_states shape matches input hidden_states shape

    Complexity:
        Training: O(S^2 * H * D) (full attention) + O((S/B)^2 * H) (gate)
        Inference: O(S^2 * H * D * (1-sparsity)) (sparse attention) + O((S/B)^2 * H) (gate)
    """

    def __init__(
        self,
        base_attn: nn.Module,
        gate: AttnGate,
        layer_idx: LayerIndex,
        compute_gate_target: bool = False,
        min_sparsity_for_speedup: float = 0.3,
    ) -> None:
        """Initialize TASFTAttention wrapper.

        Args:
            base_attn: The original HF attention module to wrap.
            gate: AttnGate module for this layer.
            layer_idx: Index of this layer in the model.
            compute_gate_target: Whether to compute gate distillation targets from full attn.
            min_sparsity_for_speedup: Minimum gate sparsity ratio to use the block-sparse
                kernel. Below this threshold, dense SDPA is used instead since the kernel
                overhead exceeds the savings from skipping few blocks. Must be in [0, 1].

        Raises:
            ValidationError: If min_sparsity_for_speedup is outside [0, 1].
        """
        super().__init__()
        if not 0.0 <= min_sparsity_for_speedup <= 1.0:
            msg = f"min_sparsity_for_speedup must be in [0, 1], got {min_sparsity_for_speedup}"
            raise ValidationError(
                msg,
                context={"min_sparsity_for_speedup": min_sparsity_for_speedup},
            )

        self.base_attn = base_attn
        self.gate = gate
        self.layer_idx = layer_idx
        self.compute_gate_target = compute_gate_target
        self.min_sparsity_for_speedup = min_sparsity_for_speedup

        # Mutable instance attributes for trainer extraction after forward pass.
        # Set during forward(), read by TASFTTrainer._extract_gate_output/attn_scores.
        self._last_gate_output: GateOutput | None = None
        self._last_attn_weights: AttentionScores | None = None

        # Lazy-loaded block-sparse kernel. Three states:
        #   _KERNEL_NOT_TRIED: not yet attempted to load
        #   None: attempted but unavailable (import error, unsupported block_size, etc.)
        #   <instance>: successfully loaded kernel
        self._kernel: Any = _KERNEL_NOT_TRIED

    def _get_kernel(self) -> Any | None:
        """Lazy-load BlockSparseFlashAttention kernel.

        Returns:
            BlockSparseFlashAttention instance, or None if the kernel is unavailable
            (missing dependency, unsupported block_size, no GPU, etc.).

        Complexity: O(1) after first call.
        """
        if self._kernel is not _KERNEL_NOT_TRIED:
            return self._kernel
        try:
            if not torch.cuda.is_available():
                # Block-sparse kernel requires CUDA; skip on CPU-only
                self._kernel = None
                return self._kernel
            from tasft.kernels.block_sparse_fa import BlockSparseFlashAttention

            self._kernel = BlockSparseFlashAttention(block_size=self.gate.block_size)
        except (ImportError, ValueError, RuntimeError):
            # ImportError: triton/CUDA not available
            # ValueError: unsupported block_size
            # RuntimeError: GPU not available
            self._kernel = None
        return self._kernel

    def set_training_mode(self, compute_gate_target: bool) -> None:
        """Toggle gate target computation for training vs inference.

        Args:
            compute_gate_target: If True, compute ground-truth block importance from
                                 full attention scores during forward. Set True only
                                 during co-training when this layer is in the calibration window.
        """
        self.compute_gate_target = compute_gate_target

    # Temperature for softmax in gate target computation. Values > 1 smooth the
    # target distribution, preventing near-delta outputs that cause KL divergence
    # explosion in float32. Chosen so typical KL loss stays in [0, 5] range.
    _GATE_TARGET_TEMPERATURE: float = 2.0

    # Clamp range for pre-softmax logits after maxpooling. Without clamping,
    # maxpooled attention scores can exceed 100 in float32, causing softmax to
    # produce distributions where one entry is ~1.0 and all others are ~0.0.
    _GATE_TARGET_CLAMP_MIN: float = -50.0
    _GATE_TARGET_CLAMP_MAX: float = 50.0

    def _compute_gate_target(self, attn_scores: torch.Tensor) -> torch.Tensor:
        """Compute ground-truth block importance from full attention score matrix.

        Uses 2D max-pooling to extract per-block importance, then softmax-normalizes
        across the flattened block grid with temperature scaling. This is the
        distillation target the gate learns to predict.

        Numerical stability: Pre-softmax logits are clamped to
        [_GATE_TARGET_CLAMP_MIN, _GATE_TARGET_CLAMP_MAX] after maxpooling to
        prevent near-delta softmax distributions that cause KL divergence
        explosion in float32. Temperature > 1 further smooths the target
        distribution, reducing gradient magnitude during gate distillation.

        Args:
            attn_scores: Full attention scores [B, H, S, S] (pre-softmax or post-softmax).

        Returns:
            Block importance [B, H, NB_q, NB_k] -- softmax-normalized over (NB_q * NB_k).

        Complexity: O(B * H * S^2 / block_size^2) -- one maxpool pass.
        """
        B, H, S_q, S_k = attn_scores.shape
        block_size = self.gate.block_size

        # Pad S_q and S_k to multiples of block_size
        pad_q = (block_size - S_q % block_size) % block_size
        pad_k = (block_size - S_k % block_size) % block_size

        if pad_q > 0 or pad_k > 0:
            # F.pad expects (left, right, top, bottom) for 4D [B*H, 1, S_q, S_k]
            attn_scores = F.pad(attn_scores, (0, pad_k, 0, pad_q), value=float("-inf"))

        S_q_padded = attn_scores.shape[2]
        S_k_padded = attn_scores.shape[3]

        # Reshape for max_pool2d: [B*H, 1, S_q_padded, S_k_padded]
        pooled_input = attn_scores.reshape(B * H, 1, S_q_padded, S_k_padded)

        # Max-pool with kernel_size=block_size, stride=block_size
        # Output: [B*H, 1, NB_q, NB_k]
        block_max = F.max_pool2d(
            pooled_input,
            kernel_size=block_size,
            stride=block_size,
        )

        NB_q = block_max.shape[2]
        NB_k = block_max.shape[3]

        # Reshape to [B, H, NB_q, NB_k]
        block_max = block_max.reshape(B, H, NB_q, NB_k)

        # Clamp finite values to bounded range to prevent near-delta softmax.
        # Preserves -inf from causal masking (those map to 0 after softmax).
        finite_mask = torch.isfinite(block_max)
        block_max = torch.where(
            finite_mask,
            block_max.clamp(
                min=self._GATE_TARGET_CLAMP_MIN,
                max=self._GATE_TARGET_CLAMP_MAX,
            ),
            block_max,
        )

        # Softmax-normalize over flattened block dim for distillation target.
        # Temperature > 1 smooths the distribution, preventing near-delta targets
        # that produce large KL divergence values against the gate's initial
        # approximately-uniform predictions.
        flat = block_max.reshape(B, H, NB_q * NB_k)
        flat_softmax = F.softmax(flat / self._GATE_TARGET_TEMPERATURE, dim=-1)
        return flat_softmax.reshape(B, H, NB_q, NB_k)


    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        past_key_value: Any | None = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: torch.Tensor | None = None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, ...]:
        """Forward pass dispatching training vs inference paths.

        Returns HF-compatible tuple: (attn_output, attn_weights_or_None, past_kv_or_None).
        Gate data is stored as instance attributes for trainer extraction:
            self._last_gate_output: GateOutput from AttnGate
            self._last_attn_weights: Full attention scores [B, H, S, S]

        Training path (grad enabled + compute_gate_target):
            1. Run full base attention with output_attentions=True
            2. Run AttnGate(Q, K) for predicted block scores
            3. Compute gate target from full attention scores
            4. Store aux data as instance attributes

        Inference path (no grad or compute_gate_target=False):
            1. Run AttnGate(Q, K) for hard block mask
            2. Run base attention (mask could be applied externally for sparse kernels)
            3. Store gate_output as instance attribute

        Args:
            hidden_states: Input hidden states [B, S, hidden_dim].
            attention_mask: Attention mask from tokenizer/model.
            position_ids: Position IDs for rotary embeddings.
            past_key_value: KV cache for autoregressive generation.
            output_attentions: Whether to return attention weights.
            use_cache: Whether to return updated KV cache.
            cache_position: Cache position indices.
            position_embeddings: Pre-computed (cos, sin) for rotary embeddings.
            **kwargs: Additional arguments forwarded to base attention.

        Returns:
            2-tuple (attn_output, attn_weights) when use_cache is False or past_kv is None,
            3-tuple (attn_output, attn_weights, past_key_value) when KV cache is present.
            Matches modern HF attention convention (Qwen2, LLaMA >= 4.43).
        """
        is_training = torch.is_grad_enabled() and self.compute_gate_target

        if is_training:
            return self._training_forward(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                **kwargs,
            )
        return self._inference_forward(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )

    def _training_forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        past_key_value: Any | None = None,
        use_cache: bool = False,
        cache_position: torch.Tensor | None = None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, ...]:
        """Training forward: single-pass Q/K/V projection + dense attention + gate prediction.

        Computes Q, K, V once via _prepare_qkv, then:
        1. Runs the gate on Q, K for block importance prediction
        2. Computes dense attention manually to get both output and attention weights
        3. Attention weights serve as gate distillation target via _compute_gate_target

        This avoids the previous double-projection bug where base_attn.forward() computed
        Q/K internally and then _extract_qk_projections ran q_proj/k_proj a second time.

        Stores self._last_gate_output and self._last_attn_weights for trainer extraction.

        Falls back to base_attn.forward() for non-standard architectures that lack
        q_proj/k_proj/v_proj/o_proj attributes.

        Returns:
            HF-compatible tuple: 2-tuple when no KV cache, 3-tuple when KV cache present.

        Complexity: O(B * H * S^2 * D) for attention + O((S/B)^2 * H) for gate.
        """
        attn = self.base_attn

        # Guard: if base_attn lacks standard projection attributes, fall back to the
        # old path (base_attn.forward + separate gate extraction). This handles
        # non-standard architectures gracefully at the cost of double projection.
        if not all(
            hasattr(attn, proj) for proj in ("q_proj", "k_proj", "v_proj", "o_proj")
        ):
            return self._training_forward_fallback(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                **kwargs,
            )

        bsz, q_len, hidden_dim = hidden_states.shape

        # Single-pass Q/K/V projection — eliminates the double-projection bug
        query_states, key_states, value_states, head_dim = self._prepare_qkv(
            attn, hidden_states, bsz, q_len,
            position_ids, position_embeddings, past_key_value,
        )
        new_past_key_value = (key_states, value_states) if use_cache else None

        # Run gate on the same Q, K that attention will use — zero redundant projections
        gate_output = self.gate(query_states, key_states)
        self._last_gate_output = gate_output

        # Dense attention with explicit weight computation for gate target distillation
        scale = 1.0 / (head_dim ** 0.5)
        attn_weights = torch.matmul(
            query_states, key_states.transpose(-2, -1),
        ) * scale

        # Apply causal mask: tokens cannot attend to future positions.
        # Upper-triangular -inf mask for autoregressive decoding.
        S_q = query_states.shape[2]
        S_k = key_states.shape[2]
        causal_mask = torch.triu(
            torch.full(
                (S_q, S_k), float("-inf"),
                device=attn_weights.device, dtype=attn_weights.dtype,
            ),
            diagonal=1,
        )
        attn_weights = attn_weights + causal_mask.unsqueeze(0).unsqueeze(0)

        if attention_mask is not None:
            ext_mask = attention_mask
            if ext_mask.ndim == 2:
                ext_mask = ext_mask[:, None, None, :]
            attn_weights = attn_weights + ext_mask

        # Store attention weights for gate target computation before softmax.
        # Detach from the autograd graph: gate target is a fixed distillation
        # signal, not a differentiable path. This prevents gradient overflow from
        # flowing back through the O(S^2) attention matrix when the gate loss
        # is large (which occurs on CPU float32 with pre-softmax scores of
        # magnitude 100+). The gate learns to predict block importance; it does
        # not need to modify the attention computation itself.
        self._last_attn_weights = attn_weights.detach()

        attn_weights_softmax = F.softmax(attn_weights, dim=-1, dtype=torch.float32)
        attn_weights_softmax = attn_weights_softmax.to(value_states.dtype)
        attn_output = torch.matmul(attn_weights_softmax, value_states)

        # Reshape [B, H, S, D] -> [B, S, hidden_dim] and apply output projection
        attn_output = attn_output.transpose(1, 2).contiguous().reshape(bsz, q_len, hidden_dim)
        attn_output = attn.o_proj(attn_output)

        return self._pack_output(attn_output, attn_weights, new_past_key_value)

    def _training_forward_fallback(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        past_key_value: Any | None = None,
        use_cache: bool = False,
        cache_position: torch.Tensor | None = None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, ...]:
        """Fallback training path for non-standard architectures without projection attributes.

        Uses base_attn.forward(output_attentions=True) and separate Q/K extraction.
        Incurs double Q/K projection — only used when the base module is non-standard.

        Returns:
            HF-compatible tuple: 2-tuple when no KV cache, 3-tuple when KV cache present.
        """
        base_output = self.base_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=True,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )

        attn_output = base_output[0]
        attn_weights = base_output[1]
        past_kv = base_output[2] if len(base_output) > 2 else None

        self._last_attn_weights = attn_weights

        gate_output: GateOutput | None = None
        if attn_weights is not None:
            q_proj, k_proj = self._extract_qk_projections(hidden_states)
            if q_proj is not None and k_proj is not None:
                gate_output = self.gate(q_proj, k_proj)

        self._last_gate_output = gate_output

        return self._pack_output(attn_output, attn_weights, past_kv)

    def _inference_forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        past_key_value: Any | None = None,
        use_cache: bool = False,
        cache_position: torch.Tensor | None = None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, ...]:
        """Inference forward: gate-driven sparse or dense attention.

        Performs Q/K/V projections, runs the gate to predict block importance,
        then dispatches to block-sparse kernel (when sparsity >= min_sparsity_for_speedup)
        or dense SDPA fallback (when sparsity is too low for kernel speedup).

        If the base attention module does not expose q_proj/k_proj/v_proj/o_proj
        (non-standard architecture), falls back to the base module's forward directly.

        Stores self._last_gate_output for downstream extraction.

        Returns:
            HF-compatible tuple: 2-tuple when no KV cache, 3-tuple when KV cache present.

        Complexity:
            Sparse path: O(B * H * S^2 * (1-sparsity) * D / block_size^2)
            Dense path:  O(B * H * S^2 * D)
        """
        attn = self.base_attn

        # Guard: require q_proj, k_proj, v_proj, o_proj for direct sparse path.
        # If missing, fall back to base_attn.forward() with no sparse acceleration.
        if not all(
            hasattr(attn, proj) for proj in ("q_proj", "k_proj", "v_proj", "o_proj")
        ):
            return self._dense_fallback_via_base(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                **kwargs,
            )

        bsz, q_len, hidden_dim = hidden_states.shape

        # Project Q/K/V, apply rotary embeddings, handle GQA and KV cache
        query_states, key_states, value_states, head_dim = self._prepare_qkv(
            attn, hidden_states, bsz, q_len,
            position_ids, position_embeddings, past_key_value,
        )
        new_past_key_value = (key_states, value_states) if use_cache else None

        # Run gate on Q, K to get hard block mask
        gate_output = self.gate(query_states, key_states)
        self._last_gate_output = gate_output
        self._last_attn_weights = None

        # Dispatch sparse vs dense based on achieved sparsity and kernel availability
        kernel = self._get_kernel()
        use_sparse = (
            kernel is not None
            and gate_output.sparsity_ratio >= self.min_sparsity_for_speedup
        )

        if use_sparse:
            # Sparse path: block-sparse kernel skips masked-out blocks
            attn_output = kernel.forward(
                query_states,
                key_states,
                value_states,
                gate_output.hard_mask,
                causal=True,
            )
        else:
            # Dense fallback: sparsity too low for kernel overhead to pay off
            scale = 1.0 / (head_dim ** 0.5)
            attn_weights = torch.matmul(
                query_states, key_states.transpose(-2, -1),
            ) * scale
            if attention_mask is not None:
                causal_mask = attention_mask
                if causal_mask.ndim == 2:
                    causal_mask = causal_mask[:, None, None, :]
                attn_weights = attn_weights + causal_mask
            attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32)
            attn_weights = attn_weights.to(value_states.dtype)
            attn_output = torch.matmul(attn_weights, value_states)

        # Step 7: Reshape [B, H, S, D] -> [B, S, hidden_dim] and apply output projection
        attn_output = attn_output.transpose(1, 2).contiguous().reshape(bsz, q_len, hidden_dim)
        attn_output = attn.o_proj(attn_output)

        return self._pack_output(attn_output, None, new_past_key_value)

    def _prepare_qkv(
        self,
        attn: nn.Module,
        hidden_states: torch.Tensor,
        bsz: int,
        q_len: int,
        position_ids: torch.Tensor | None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None,
        past_key_value: Any | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        """Project Q/K/V, apply rotary embeddings, expand GQA heads, and update KV cache.

        Centralizes the projection pipeline shared by sparse and dense inference paths.

        Dimension resolution uses a 3-tier fallback for each of num_heads, num_kv_heads,
        and head_dim:
            1. Direct attribute on module (e.g., ``attn.num_heads``)
            2. Module's ``.config`` attribute (e.g., ``attn.config.num_attention_heads``)
               -- handles Qwen2Attention and similar architectures where dimensions
               live on the config object rather than as instance attributes.
            3. Derive from projection weight shapes (``q_proj.weight.shape[0] // head_dim``)

        Args:
            attn: Base attention module with q_proj, k_proj, v_proj attributes.
            hidden_states: [B, S, hidden_dim] input tensor.
            bsz: Batch size.
            q_len: Sequence length.
            position_ids: Position IDs for rotary embeddings (if rotary_emb on attn).
            position_embeddings: Pre-computed (cos, sin) for rotary embeddings.
            past_key_value: KV cache (DynamicCache or legacy tuple).

        Returns:
            (query_states, key_states, value_states, head_dim) all with GQA-expanded heads
            and KV cache applied. Shapes: [B, num_heads, S_total, head_dim].

        Complexity: O(B * S * H * D) for projections + O(B * H * S * D) for rotary.
        """
        query_states = attn.q_proj(hidden_states)
        key_states = attn.k_proj(hidden_states)
        value_states = attn.v_proj(hidden_states)

        # 3-tier fallback for num_heads: direct attr -> config -> weight shape derivation
        num_heads_resolved = _resolve_attn_dim(
            attn,
            attr_names=("num_heads", "num_attention_heads", "n_head"),
            config_attr_names=("num_attention_heads", "num_heads", "n_head"),
        )

        # 3-tier fallback for head_dim: direct attr -> config -> weight shape derivation
        head_dim_resolved = _resolve_attn_dim(
            attn,
            attr_names=("head_dim", "d_head"),
            config_attr_names=("head_dim", "d_head"),
        )

        # 3-tier fallback for num_kv_heads: direct attr -> config -> weight shape derivation
        num_kv_heads_resolved = _resolve_attn_dim(
            attn,
            attr_names=("num_key_value_heads",),
            config_attr_names=("num_key_value_heads",),
        )

        # Derive missing dims from projection weight shapes (tier 3)
        q_out_dim = query_states.shape[-1]
        k_out_dim = key_states.shape[-1]

        if head_dim_resolved is not None and num_heads_resolved is None:
            num_heads_resolved = q_out_dim // head_dim_resolved
        elif num_heads_resolved is not None and head_dim_resolved is None:
            head_dim_resolved = q_out_dim // num_heads_resolved
        elif num_heads_resolved is None and head_dim_resolved is None:
            # Last resort: try hidden_size from config to get head_dim
            hidden_size = _resolve_attn_dim(
                attn,
                attr_names=("hidden_size",),
                config_attr_names=("hidden_size",),
            )
            if hidden_size is not None and q_out_dim > 0:
                # head_dim = hidden_size / num_heads, but we need num_heads first
                # For standard models: q_out_dim == hidden_size, so we need another source
                pass

        num_heads: int = num_heads_resolved if num_heads_resolved is not None else 0
        head_dim: int = head_dim_resolved if head_dim_resolved is not None else 0

        # num_kv_heads: fall back to deriving from k_proj output shape if head_dim is known
        if num_kv_heads_resolved is not None:
            num_kv_heads: int = num_kv_heads_resolved
        elif head_dim > 0:
            num_kv_heads = k_out_dim // head_dim
        else:
            num_kv_heads = num_heads

        # Reshape to [B, num_heads, S, head_dim]
        query_states = query_states.view(bsz, q_len, num_heads, head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, num_kv_heads, head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, num_kv_heads, head_dim).transpose(1, 2)

        # Apply rotary position embeddings if available
        if position_embeddings is not None:
            cos, sin = position_embeddings
            query_states, key_states = _apply_rotary_pos_emb(
                query_states, key_states, cos, sin,
            )
        elif hasattr(attn, "rotary_emb") and position_ids is not None:
            cos, sin = attn.rotary_emb(value_states, position_ids)
            query_states, key_states = _apply_rotary_pos_emb(
                query_states, key_states, cos, sin,
            )

        # GQA head expansion: repeat KV heads to match Q heads
        if num_kv_heads < num_heads:
            n_rep = num_heads // num_kv_heads
            key_states = key_states.repeat_interleave(n_rep, dim=1)
            value_states = value_states.repeat_interleave(n_rep, dim=1)

        # Handle KV cache for autoregressive generation
        if past_key_value is not None:
            if hasattr(past_key_value, "update"):
                key_states, value_states = past_key_value.update(
                    key_states, value_states, self.layer_idx,
                )
            else:
                cache_k, cache_v = past_key_value
                key_states = torch.cat([cache_k, key_states], dim=2)
                value_states = torch.cat([cache_v, value_states], dim=2)

        return query_states, key_states, value_states, head_dim

    def _dense_fallback_via_base(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        past_key_value: Any | None = None,
        use_cache: bool = False,
        cache_position: torch.Tensor | None = None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, ...]:
        """Dense fallback via base_attn.forward() for architectures without standard projections.

        Used when the base attention module does not expose q_proj/k_proj/v_proj/o_proj,
        making direct sparse attention impossible. Runs the gate on extracted Q/K if possible
        for observability, but attention itself is fully dense.

        Returns:
            HF-compatible tuple: 2-tuple when no KV cache, 3-tuple when KV cache present.
        """
        # Attempt gate prediction for observability even without sparse path
        q_proj, k_proj = self._extract_qk_projections(hidden_states)
        if q_proj is not None and k_proj is not None:
            self._last_gate_output = self.gate(q_proj, k_proj)
        else:
            self._last_gate_output = None
        self._last_attn_weights = None

        base_output = self.base_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=False,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )

        attn_output = base_output[0]
        past_kv = base_output[2] if len(base_output) > 2 else None

        return self._pack_output(attn_output, None, past_kv)

    def _extract_qk_projections(
        self, hidden_states: torch.Tensor,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """Extract Q and K tensors from hidden_states using base attention's projections.

        Supports HuggingFace model architectures that expose q_proj and k_proj as attributes.
        Returns (None, None) if the base attention module doesn't expose these projections.

        Args:
            hidden_states: [B, S, hidden_dim] input tensor.

        Returns:
            (q, k) each [B, H, S, head_dim] or (None, None) if projections not found.
        """
        q_proj_layer = getattr(self.base_attn, "q_proj", None)
        k_proj_layer = getattr(self.base_attn, "k_proj", None)

        if q_proj_layer is None or k_proj_layer is None:
            return None, None

        B, S, _ = hidden_states.shape
        head_dim = self.gate.head_dim

        # Project: [B, S, hidden_dim] -> [B, S, num_heads * head_dim]
        q = q_proj_layer(hidden_states)
        k = k_proj_layer(hidden_states)

        # Handle GQA: k_proj may output fewer heads than q_proj
        q_total_dim = q.shape[-1]
        k_total_dim = k.shape[-1]

        num_q_heads = q_total_dim // head_dim
        num_k_heads = k_total_dim // head_dim

        # Reshape to [B, S, num_heads, head_dim] -> [B, num_heads, S, head_dim]
        q = q.view(B, S, num_q_heads, head_dim).transpose(1, 2)
        k = k.view(B, S, num_k_heads, head_dim).transpose(1, 2)

        # For GQA: repeat K heads to match Q heads for gate input
        if num_k_heads < num_q_heads:
            repeat_factor = num_q_heads // num_k_heads
            k = k.repeat_interleave(repeat_factor, dim=1)

        return q, k

    @staticmethod
    def _pack_output(
        attn_output: torch.Tensor,
        attn_weights: torch.Tensor | None,
        past_kv: Any | None,
    ) -> tuple[torch.Tensor, ...]:
        """Build HF-compatible return tuple — always 2-tuple.

        Modern HuggingFace models (Qwen2, LLaMA >= 4.43, Mistral) unpack
        attention output as ``hidden_states, _ = self.self_attn(...)``
        expecting exactly 2 values. DynamicCache is mutated in-place during
        ``past_key_value.update()``, so there is no need to return it.

        For legacy models using tuple-based KV cache (not DynamicCache), the
        cache is still accessible via ``_prepare_qkv``'s ``past_key_value``
        parameter on the next forward call.

        Args:
            attn_output: Attention output tensor [B, S, hidden_dim].
            attn_weights: Attention weight tensor or None.
            past_kv: Updated KV cache or None (unused in return, kept for API compat).

        Returns:
            2-tuple ``(attn_output, attn_weights)`` — always.

        Complexity: O(1).
        """
        return (attn_output, attn_weights)

    def extra_repr(self) -> str:
        """Module repr for debugging."""
        return (
            f"layer_idx={self.layer_idx}, "
            f"compute_gate_target={self.compute_gate_target}, "
            f"gate_params={self.gate.num_parameters}"
        )


def _apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary position embeddings to Q and K tensors.

    Handles both [B, S, D] and [B, 1, S, D] cos/sin shapes by normalizing
    to 4D before element-wise multiply.

    Args:
        q: [B, H, S, D] query tensor.
        k: [B, H, S, D] key tensor.
        cos: Cosine component of rotary embeddings.
        sin: Sine component of rotary embeddings.

    Returns:
        Tuple of (q_rotated, k_rotated) with same shapes as inputs.

    Complexity: O(B * H * S * D).
    """

    def _rotate_half(x: torch.Tensor) -> torch.Tensor:
        """Split last dim in half and rotate: [-x2, x1]."""
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    # Ensure cos/sin have head dimension: [B, S, D] -> [B, 1, S, D]
    if cos.ndim == 3:
        cos = cos.unsqueeze(1)
        sin = sin.unsqueeze(1)

    q_embed = (q * cos) + (_rotate_half(q) * sin)
    k_embed = (k * cos) + (_rotate_half(k) * sin)
    return q_embed, k_embed


def patch_model_attention(
    model: nn.Module,
    gate_config: GateConfig,
) -> dict[int, TASFTAttention]:
    """Replace each attention layer in a HuggingFace model with TASFTAttention.

    Iterates through model layers, wraps each attention module with TASFTAttention
    containing a fresh AttnGate, and verifies all base weights remain frozen.

    Supports model architectures with a `model.layers[i].self_attn` structure
    (LLaMA, Mistral, Qwen2, Phi, etc.).

    Args:
        model: HuggingFace model (e.g., LlamaForCausalLM).
        gate_config: Configuration for AttnGate modules.

    Returns:
        Mapping {layer_idx: TASFTAttention} for all patched layers.

    Raises:
        ValidationError: If model structure is unsupported or parameters are invalid.

    Postconditions:
        - All base model parameters have requires_grad=False.
        - Only gate parameters have requires_grad=True.
        - Original attention module is preserved as base_attn within wrapper.
    """
    # Find the model's layer list
    layers = _find_model_layers(model)
    if layers is None:
        msg = (
            "Could not find transformer layers in model. "
            "Expected model.model.layers or model.transformer.h"
        )
        raise ValidationError(
            msg,
            context={"model_type": type(model).__name__},
        )

    num_layers = len(layers)
    if gate_config.num_layers > num_layers:
        msg = f"gate_config.num_layers ({gate_config.num_layers}) > model layers ({num_layers})"
        raise ValidationError(
            msg,
            context={
                "config_layers": gate_config.num_layers,
                "model_layers": num_layers,
            },
        )

    # Determine num_heads and head_dim from the first attention module
    first_attn = _find_attn_module(layers[0])
    if first_attn is None:
        msg = "Could not find attention module in layer 0"
        raise ValidationError(
            msg,
            context={"layer_type": type(layers[0]).__name__},
        )

    num_heads, head_dim = _extract_attn_dims(first_attn)

    patched: dict[int, TASFTAttention] = {}

    for idx in range(gate_config.num_layers):
        layer = layers[idx]
        base_attn = _find_attn_module(layer)
        if base_attn is None:
            msg = f"Could not find attention module in layer {idx}"
            raise ValidationError(
                msg,
                context={"layer_idx": idx, "layer_type": type(layer).__name__},
            )

        # Freeze all base attention parameters
        for param in base_attn.parameters():
            param.requires_grad = False

        # Create gate for this layer
        gate = AttnGate(
            num_heads=num_heads,
            head_dim=head_dim,
            block_size=gate_config.block_size,
            gate_hidden_dim=gate_config.gate_hidden_dim,
            default_threshold=gate_config.default_threshold,
        )

        # Move gate to same device/dtype as base attention
        first_param = next(base_attn.parameters(), None)
        if first_param is not None:
            gate = gate.to(device=first_param.device, dtype=first_param.dtype)

        layer_idx = LayerIndex(idx)
        tasft_attn = TASFTAttention(
            base_attn=base_attn,
            gate=gate,
            layer_idx=layer_idx,
            compute_gate_target=False,
        )

        # Replace attention module in the layer
        _replace_attn_module(layer, tasft_attn)
        patched[idx] = tasft_attn

    # Verify: all base weights frozen, only gate weights trainable
    _verify_frozen_base(model, patched)

    return patched


def _find_model_layers(model: nn.Module) -> nn.ModuleList | None:
    """Find the transformer layer list in a HuggingFace model.

    Supports: model.model.layers (LLaMA, Mistral, Qwen2)
              model.transformer.h (GPT-2, GPT-NeoX)
              model.layers (direct)

    Returns:
        nn.ModuleList of transformer layers, or None if not found.
    """
    # LLaMA / Mistral / Qwen2 style
    inner = getattr(model, "model", None)
    if inner is not None:
        layers = getattr(inner, "layers", None)
        if isinstance(layers, nn.ModuleList):
            return layers

    # GPT-2 / GPT-NeoX style
    transformer = getattr(model, "transformer", None)
    if transformer is not None:
        h = getattr(transformer, "h", None)
        if isinstance(h, nn.ModuleList):
            return h

    # Direct layers attribute
    layers = getattr(model, "layers", None)
    if isinstance(layers, nn.ModuleList):
        return layers

    return None


def _find_attn_module(layer: nn.Module) -> nn.Module | None:
    """Find the attention module within a transformer layer.

    Supports: layer.self_attn (LLaMA, Mistral, Qwen2)
              layer.attn (GPT-2)
              layer.attention (GPT-NeoX)

    Returns:
        Attention module or None if not found.
    """
    for attr_name in ("self_attn", "attn", "attention"):
        attn = getattr(layer, attr_name, None)
        if attn is not None and isinstance(attn, nn.Module):
            return attn
    return None


def _replace_attn_module(layer: nn.Module, replacement: nn.Module) -> None:
    """Replace the attention module in a transformer layer.

    Args:
        layer: Transformer layer containing the attention module.
        replacement: New module to install.

    Raises:
        ValidationError: If no known attention attribute is found.
    """
    for attr_name in ("self_attn", "attn", "attention"):
        if hasattr(layer, attr_name) and isinstance(getattr(layer, attr_name), nn.Module):
            setattr(layer, attr_name, replacement)
            return

    msg = "Could not find attention attribute to replace"
    raise ValidationError(
        msg,
        context={"layer_type": type(layer).__name__},
    )


def _resolve_attn_dim(
    attn: nn.Module,
    attr_names: tuple[str, ...],
    config_attr_names: tuple[str, ...] | None = None,
) -> int | None:
    """Resolve a single attention dimension via 3-tier fallback.

    Fallback chain:
        1. Direct attribute on module (e.g., ``attn.num_heads``)
        2. Module's ``.config`` attribute (e.g., ``attn.config.num_attention_heads``)
           -- handles Qwen2Attention and similar architectures where dimensions
           live on the config object rather than as instance attributes.
        3. Returns None -- caller must derive from weight shapes or raise.

    Args:
        attn: The attention module to inspect.
        attr_names: Attribute names to try on the module directly.
        config_attr_names: Attribute names to try on ``attn.config``. If None,
                           reuses ``attr_names``.

    Returns:
        The resolved integer dimension, or None if not found.

    Complexity: O(1) -- bounded by the number of attribute names.
    """
    # Tier 1: direct attribute on the module instance
    for attr in attr_names:
        val = getattr(attn, attr, None)
        if isinstance(val, int) and val > 0:
            return val

    # Tier 2: module's .config object (Qwen2, newer transformers)
    config = getattr(attn, "config", None)
    if config is not None:
        search_attrs = config_attr_names if config_attr_names is not None else attr_names
        for attr in search_attrs:
            val = getattr(config, attr, None)
            if isinstance(val, int) and val > 0:
                return val

    # Tier 3: not found -- caller handles derivation or error
    return None


def _extract_attn_dims(attn: nn.Module) -> tuple[int, int]:
    """Extract num_heads and head_dim from an attention module.

    Uses a 3-tier fallback for each dimension:
        1. Direct attribute on module (e.g., ``attn.num_heads``)
        2. Module's ``.config`` attribute (e.g., ``attn.config.num_attention_heads``)
           -- handles Qwen2Attention and similar architectures where dimensions
           live on the config object rather than as instance attributes.
        3. Derive from projection weight shapes (``attn.q_proj.weight``)

    Returns:
        (num_heads, head_dim) tuple.

    Raises:
        ValidationError: If dimensions cannot be determined from any source.

    Complexity: O(1).
    """
    num_heads = _resolve_attn_dim(
        attn,
        attr_names=("num_heads", "num_attention_heads", "n_head"),
        config_attr_names=("num_attention_heads", "num_heads", "n_head"),
    )

    head_dim = _resolve_attn_dim(
        attn,
        attr_names=("head_dim", "d_head"),
        config_attr_names=("head_dim", "d_head"),
    )

    # Fallback: derive head_dim from hidden_size / num_heads
    if head_dim is None and num_heads is not None:
        hidden_size = _resolve_attn_dim(
            attn,
            attr_names=("hidden_size",),
            config_attr_names=("hidden_size",),
        )
        if hidden_size is not None:
            head_dim = hidden_size // num_heads

    # Tier 3 fallback: derive from q_proj weight shape
    # q_proj.weight has shape [num_heads * head_dim, hidden_size]
    if num_heads is None or head_dim is None:
        q_proj = getattr(attn, "q_proj", None)
        if q_proj is not None:
            weight = getattr(q_proj, "weight", None)
            if weight is not None:
                q_out_dim = weight.shape[0]
                # If we have head_dim but not num_heads, derive num_heads
                if head_dim is not None and num_heads is None:
                    num_heads = q_out_dim // head_dim
                # If we have num_heads but not head_dim, derive head_dim
                elif num_heads is not None and head_dim is None:
                    head_dim = q_out_dim // num_heads
                # If we have neither, try hidden_size from weight input dim
                elif num_heads is None and head_dim is None:
                    hidden_size_from_weight = weight.shape[1]
                    # Common case: q_out_dim == hidden_size (no MQA scaling on Q)
                    # Try to get num_heads from config to break the deadlock
                    config = getattr(attn, "config", None)
                    if config is not None:
                        # Try num_key_value_heads as a last resort for head_dim derivation
                        for cfg_attr in ("num_attention_heads", "num_heads"):
                            cfg_val = getattr(config, cfg_attr, None)
                            if isinstance(cfg_val, int) and cfg_val > 0:
                                num_heads = cfg_val
                                head_dim = q_out_dim // num_heads
                                break
                    if num_heads is None and q_out_dim == hidden_size_from_weight:
                        # Cannot determine without external info
                        pass

    if num_heads is None or head_dim is None:
        msg = "Cannot determine num_heads and head_dim from attention module"
        raise ValidationError(
            msg,
            context={
                "attn_type": type(attn).__name__,
                "num_heads": num_heads,
                "head_dim": head_dim,
            },
        )

    return num_heads, head_dim


def _verify_frozen_base(
    model: nn.Module,
    patched_layers: dict[int, TASFTAttention],
) -> None:
    """Verify all base model parameters are frozen, only gate params are trainable.

    Args:
        model: The full model after patching.
        patched_layers: Mapping of patched TASFTAttention modules.

    Raises:
        ValidationError: If any base parameter has requires_grad=True.
    """
    # Collect all gate parameter IDs
    gate_param_ids: set[int] = set()
    for tasft_attn in patched_layers.values():
        for param in tasft_attn.gate.parameters():
            gate_param_ids.add(id(param))

    unfrozen_base: list[str] = []
    for name, param in model.named_parameters():
        if param.requires_grad and id(param) not in gate_param_ids:
            unfrozen_base.append(name)

    if unfrozen_base:
        msg = f"Found {len(unfrozen_base)} unfrozen base parameters after patching"
        raise ValidationError(
            msg,
            context={"unfrozen_params": unfrozen_base[:10]},  # first 10 for context
        )


__all__ = [
    "GateConfig",
    "TASFTAttention",
    "TASFTAttentionOutput",
    "patch_model_attention",
]
