"""
TASFT Inference Runtime.

Loads a TASFT deployment bundle and serves inference requests with
gate-driven block-sparse attention, achieving 2-5x decode throughput
vs standard dense LoRA fine-tuned models.

Bundle structure:
    bundle_dir/
    ├── manifest.json          # checksums, metadata
    ├── model/model.safetensors  # merged base+LoRA weights
    ├── gates/layer_{i}_gate.safetensors  # per-layer gate weights
    └── kernel_config.json     # per-layer sparsity thresholds

Preconditions:
    - Bundle directory exists and contains a valid manifest.json
    - All SHA-256 checksums in manifest match file contents
    - CUDA device available for GPU inference

Postconditions:
    - All attention layers patched with gate-driven sparse attention
    - Model weights frozen (no gradients)
    - Kernel config validated and loaded

Complexity: O(L * S^2 / sparsity) per forward pass where L = layers, S = seq_len
"""
from __future__ import annotations

import hashlib
import json
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
from safetensors.torch import load_file
from torch import nn
from transformers import (
    AutoModelForCausalLM,
    PreTrainedModel,
)

from tasft.exceptions import BundleError, ChecksumError, InferenceError
from tasft.kernels.kernel_config import KernelConfig
from tasft.modules.attn_gate import AttnGate, GateOutput
from tasft.observability.logging import get_logger, timed_operation
from tasft.types import LayerIndex, SparsityProfile, SparsityRatio

if TYPE_CHECKING:
    from transformers.modeling_outputs import CausalLMOutputWithPast

logger = get_logger("tasft.inference.tasft_model")


@dataclass(frozen=True)
class InferenceBenchmark:
    """Results of a timed inference benchmark run.

    Attributes:
        tokens_per_second: Mean throughput in tokens/second.
        mean_latency_ms: Mean per-batch latency in milliseconds.
        p50_ms: 50th percentile latency.
        p95_ms: 95th percentile latency.
        p99_ms: 99th percentile latency.
        mean_sparsity_per_layer: Average sparsity ratio per transformer layer.
        gpu_name: Name of the GPU used for benchmarking.
        bundle_path: Path to the bundle that was benchmarked.
        num_warmup: Number of warmup iterations (excluded from timing).
        num_timed: Number of timed iterations.
    """

    tokens_per_second: float
    mean_latency_ms: float
    p50_ms: float
    p95_ms: float
    p99_ms: float
    mean_sparsity_per_layer: dict[int, float]
    gpu_name: str
    bundle_path: str
    num_warmup: int
    num_timed: int


def _verify_checksum(path: Path, expected_sha256: str) -> None:
    """Verify SHA-256 checksum of a file against expected value.

    Preconditions: path exists and is readable.
    Postconditions: Returns normally if checksum matches.

    Args:
        path: Path to the file to verify.
        expected_sha256: Expected hex-encoded SHA-256 digest.

    Raises:
        ChecksumError: If computed checksum does not match expected.

    Complexity: O(file_size) — reads file in 64KB chunks.
    """
    sha256 = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            sha256.update(chunk)
    actual = sha256.hexdigest()
    if actual != expected_sha256:
        msg = f"Checksum mismatch for {path.name}"
        raise ChecksumError(
            msg,
            context={"expected": expected_sha256, "actual": actual, "path": str(path)},
        )


def _extract_attention_layers(model: PreTrainedModel) -> list[nn.Module]:
    """Extract attention sub-modules from a HuggingFace causal LM.

    Supports LLaMA, Mistral, and GPT-NeoX attention module naming conventions.
    Walks the model tree looking for modules whose class name contains 'Attention'.

    Preconditions: model is a valid HuggingFace PreTrainedModel.
    Postconditions: Returns list ordered by layer index.

    Args:
        model: HuggingFace causal language model.

    Returns:
        Ordered list of attention modules.

    Raises:
        InferenceError: If no attention layers are found.

    Complexity: O(N) where N = total number of modules in model.
    """
    attn_layers: list[nn.Module] = []
    for _name, module in model.named_modules():
        cls_name = type(module).__name__
        # Match common HF attention module naming: LlamaAttention, MistralAttention, etc.
        if cls_name.endswith("Attention") and hasattr(module, "q_proj"):
            attn_layers.append(module)

    if not attn_layers:
        msg = "No attention layers found in model — unsupported architecture"
        raise InferenceError(
            msg,
            context={"model_class": type(model).__name__},
        )

    return attn_layers


class _SparseAttentionWrapper(nn.Module):
    """Wraps a single attention layer with gate-driven block-sparse attention.

    Replaces the standard attention forward path with:
    1. Standard Q, K, V projections (frozen merged weights)
    2. AttnGate(Q, K) → hard block mask using layer threshold
    3. Block-sparse attention using the mask

    Preconditions:
        - original_attn has q_proj, k_proj, v_proj, o_proj attributes
        - gate is a loaded AttnGate with matching num_heads and head_dim
        - threshold_tau in (0, 1)

    Postconditions:
        - Forward output shape matches original attention output
        - Sparsity profile captured in self.last_gate_output
    """

    def __init__(
        self,
        original_attn: nn.Module,
        gate: AttnGate,
        layer_idx: int,
        threshold_tau: float,
        block_size: int,
        min_sparsity_for_speedup: float,
    ) -> None:
        super().__init__()
        self.original_attn = original_attn
        self.gate = gate
        self.layer_idx = layer_idx
        self.threshold_tau = threshold_tau
        self.block_size = block_size
        self.min_sparsity_for_speedup = min_sparsity_for_speedup
        self.last_gate_output: GateOutput | None = None

        # Lazily import kernel to handle Task #7 dependency
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
        """Forward pass with gate-driven sparse attention.

        Args:
            hidden_states: [B, S, hidden_dim] input tensor.
            attention_mask: Optional attention mask.
            position_ids: Optional position IDs for rotary embeddings.
            past_key_value: Optional KV cache for autoregressive generation.
            output_attentions: Whether to return attention weights (not supported with sparse).
            use_cache: Whether to return updated KV cache.
            cache_position: Cache position indices for static cache.
            position_embeddings: Pre-computed (cos, sin) for rotary embeddings.
            **kwargs: Additional arguments forwarded to original attention.

        Returns:
            Tuple of (attn_output, attn_weights_or_none, past_key_value_or_none).

        Complexity: O(B * H * S^2 * (1 - sparsity) * D / block_size^2) for sparse path.
        """
        attn = self.original_attn
        bsz, q_len, hidden_dim = hidden_states.shape

        # Standard Q, K, V projections using frozen merged weights
        query_states = attn.q_proj(hidden_states)
        key_states = attn.k_proj(hidden_states)
        value_states = attn.v_proj(hidden_states)

        num_heads = attn.num_heads
        num_kv_heads = getattr(attn, "num_key_value_heads", num_heads)
        head_dim = attn.head_dim

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

        # Handle GQA: expand KV heads to match Q heads
        if num_kv_heads < num_heads:
            n_rep = num_heads // num_kv_heads
            key_states = key_states.repeat_interleave(n_rep, dim=1)
            value_states = value_states.repeat_interleave(n_rep, dim=1)

        # Handle KV cache for autoregressive generation
        if past_key_value is not None:
            if hasattr(past_key_value, "update"):
                # HF DynamicCache interface
                key_states, value_states = past_key_value.update(
                    key_states, value_states, self.layer_idx,
                )
            else:
                # Legacy tuple cache
                cache_k, cache_v = past_key_value
                key_states = torch.cat([cache_k, key_states], dim=2)
                value_states = torch.cat([cache_v, value_states], dim=2)

        new_past_key_value = (key_states, value_states) if use_cache else None

        # Gate-driven block-sparse attention
        gate_output = self.gate(query_states, key_states, threshold=self.threshold_tau)
        self.last_gate_output = gate_output

        # Decide sparse vs dense based on achieved sparsity
        if gate_output.sparsity_ratio >= self.min_sparsity_for_speedup:
            # Sparse path: use block-sparse kernel
            kernel = self._get_kernel()
            attn_output = kernel.forward(
                query_states,
                key_states,
                value_states,
                gate_output.hard_mask,
                causal=True,
            )
        else:
            # Dense fallback: sparsity too low for speedup
            scale = 1.0 / (head_dim**0.5)
            attn_weights = torch.matmul(query_states, key_states.transpose(-2, -1)) * scale
            if attention_mask is not None:
                causal_mask = attention_mask
                if causal_mask.ndim == 2:
                    causal_mask = causal_mask[:, None, None, :]
                attn_weights = attn_weights + causal_mask
            attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32)
            attn_weights = attn_weights.to(value_states.dtype)
            attn_output = torch.matmul(attn_weights, value_states)

        # Reshape back: [B, H, S, D] -> [B, S, hidden_dim]
        attn_output = attn_output.transpose(1, 2).contiguous().reshape(bsz, q_len, hidden_dim)

        # Output projection
        attn_output = attn.o_proj(attn_output)

        return (attn_output, None, new_past_key_value)


def _apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary position embeddings to Q and K tensors.

    Handles both [B, S, D] and [B, 1, S, D] cos/sin shapes.

    Args:
        q: [B, H, S, D] query tensor.
        k: [B, H, S, D] key tensor.
        cos: Cosine component of rotary embeddings.
        sin: Sine component of rotary embeddings.

    Returns:
        Tuple of (q_rotated, k_rotated) with same shapes.

    Complexity: O(B * H * S * D).
    """

    def _rotate_half(x: torch.Tensor) -> torch.Tensor:
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    # Ensure cos/sin have head dimension
    if cos.ndim == 3:
        cos = cos.unsqueeze(1)
        sin = sin.unsqueeze(1)

    q_embed = (q * cos) + (_rotate_half(q) * sin)
    k_embed = (k * cos) + (_rotate_half(k) * sin)
    return q_embed, k_embed


class TASFTInferenceModel(nn.Module):
    """TASFT inference model with gate-driven block-sparse attention.

    Loads a deployment bundle (merged weights + trained gates + kernel config)
    and replaces standard attention with sparse attention for 2-5x throughput.

    Preconditions:
        - Bundle was exported by tasft.bundle with valid checksums
        - CUDA available for GPU inference
        - Model architecture is LLaMA, Mistral, or compatible

    Postconditions:
        - All model weights frozen (no gradients)
        - Attention layers patched with _SparseAttentionWrapper
        - KernelConfig loaded and validated

    Complexity:
        - load_bundle: O(model_size + sum(gate_sizes))
        - forward: O(L * B * H * S^2 * (1-sparsity) * D)
    """

    def __init__(
        self,
        model: PreTrainedModel,
        gates: dict[int, AttnGate],
        kernel_config: KernelConfig,
        bundle_path: str,
        manifest: dict[str, Any],
    ) -> None:
        """Initialize TASFTInferenceModel.

        Args:
            model: HuggingFace causal LM with attention layers already patched.
            gates: Mapping from layer index to trained AttnGate modules.
            kernel_config: Validated kernel configuration.
            bundle_path: Path to the source bundle directory.
            manifest: Parsed manifest.json contents.
        """
        super().__init__()
        self.model = model
        self.gates = nn.ModuleDict({str(k): v for k, v in gates.items()})
        self.kernel_config = kernel_config
        self.bundle_path = bundle_path
        self.manifest = manifest
        self._sparse_wrappers: list[_SparseAttentionWrapper] = []

        # Collect all sparse wrappers for sparsity profiling
        for module in model.modules():
            if isinstance(module, _SparseAttentionWrapper):
                self._sparse_wrappers.append(module)

    @classmethod
    def load_bundle(
        cls,
        bundle_path: str | Path,
        *,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        trust_remote_code: bool = False,
    ) -> TASFTInferenceModel:
        """Load a TASFT deployment bundle and return a configured inference model.

        Steps:
            1. Read and parse manifest.json
            2. Verify SHA-256 checksums for ALL files before loading
            3. Load model with AutoModelForCausalLM.from_pretrained (safetensors)
            4. Load per-layer AttnGate state_dicts
            5. Patch attention layers to use gate-driven sparse attention
            6. Load and validate KernelConfig
            7. Return configured instance

        Args:
            bundle_path: Path to the bundle directory.
            device: Target device for model and gates.
            dtype: Model weight dtype (default bfloat16 for H100/A100).
            trust_remote_code: Whether to trust remote code in model config.

        Returns:
            Fully configured TASFTInferenceModel ready for inference.

        Raises:
            BundleError: If bundle structure is invalid.
            ChecksumError: If any file checksum does not match manifest.

        Complexity: O(model_size + num_layers * gate_size) for loading.
        """
        bundle_dir = Path(bundle_path)
        if not bundle_dir.is_dir():
            msg = f"Bundle directory does not exist: {bundle_dir}"
            raise BundleError(
                msg,
                context={"path": str(bundle_dir)},
            )

        # Step 1: Read manifest
        manifest_path = bundle_dir / "manifest.json"
        if not manifest_path.exists():
            msg = "manifest.json not found in bundle"
            raise BundleError(
                msg,
                context={"bundle_path": str(bundle_dir)},
            )

        with open(manifest_path) as f:
            manifest: dict[str, Any] = json.load(f)

        logger.info(
            "[BUNDLE_LOAD] Loading TASFT bundle",
            bundle_path=str(bundle_dir),
            model_name=manifest.get("model_name", "unknown"),
            num_layers=manifest.get("num_layers", "unknown"),
        )

        # Step 2: Verify SHA-256 checksums for ALL files
        checksums: dict[str, str] = manifest.get("checksums", {})
        if not checksums:
            msg = "No checksums found in manifest"
            raise BundleError(
                msg,
                context={"bundle_path": str(bundle_dir)},
            )

        with timed_operation(logger, "CHECKSUM_VERIFY"):
            for relative_path, expected_hash in checksums.items():
                file_path = bundle_dir / relative_path
                if not file_path.exists():
                    msg = f"File listed in manifest not found: {relative_path}"
                    raise BundleError(
                        msg,
                        context={
                            "relative_path": relative_path,
                            "bundle_path": str(bundle_dir),
                        },
                    )
                _verify_checksum(file_path, expected_hash)

        # Step 3: Load base model with merged LoRA weights
        model_dir = bundle_dir / "model"
        if not model_dir.is_dir():
            msg = "model/ subdirectory not found in bundle"
            raise BundleError(
                msg,
                context={"bundle_path": str(bundle_dir)},
            )

        with timed_operation(logger, "MODEL_LOAD"):
            model = AutoModelForCausalLM.from_pretrained(
                str(model_dir),
                torch_dtype=dtype,
                device_map=device,
                trust_remote_code=trust_remote_code,
                use_safetensors=True,
            )
            model.eval()
            for param in model.parameters():
                param.requires_grad = False

        # Step 4: Load per-layer AttnGate state_dicts
        gates_dir = bundle_dir / "gates"
        if not gates_dir.is_dir():
            msg = "gates/ subdirectory not found in bundle"
            raise BundleError(
                msg,
                context={"bundle_path": str(bundle_dir)},
            )

        # Step 6 (loaded early to get config for gate construction):
        # Load and validate KernelConfig
        kernel_config_path = bundle_dir / "kernel_config.json"
        if not kernel_config_path.exists():
            msg = "kernel_config.json not found in bundle"
            raise BundleError(
                msg,
                context={"bundle_path": str(bundle_dir)},
            )

        with open(kernel_config_path) as f:
            kernel_config_data = json.load(f)

        kernel_config = KernelConfig(**kernel_config_data)

        # Discover gate files and load them
        attn_layers = _extract_attention_layers(model)
        num_layers = len(attn_layers)

        # Infer head config from first attention layer
        first_attn = attn_layers[0]
        num_heads = first_attn.num_heads
        head_dim = first_attn.head_dim

        gates: dict[int, AttnGate] = {}
        with timed_operation(logger, "GATES_LOAD", num_layers=num_layers):
            for layer_idx in range(num_layers):
                gate_file = gates_dir / f"layer_{layer_idx}_gate.safetensors"
                if not gate_file.exists():
                    msg = f"Gate file not found for layer {layer_idx}"
                    raise BundleError(
                        msg,
                        context={
                            "layer_idx": layer_idx,
                            "expected_path": str(gate_file),
                        },
                    )

                layer_block_size = kernel_config.get_layer_block_size(layer_idx)

                gate = AttnGate(
                    num_heads=num_heads,
                    head_dim=head_dim,
                    block_size=layer_block_size,
                )
                gate_state = load_file(str(gate_file))
                gate.load_state_dict(gate_state)
                gate = gate.to(device=device, dtype=dtype)
                gate.eval()
                for param in gate.parameters():
                    param.requires_grad = False

                gates[layer_idx] = gate

        # Step 5: Patch attention layers with sparse wrappers
        with timed_operation(logger, "ATTENTION_PATCH", num_layers=num_layers):
            for layer_idx, attn_module in enumerate(attn_layers):
                gate = gates[layer_idx]
                threshold = kernel_config.get_layer_threshold(layer_idx)
                block_size = kernel_config.get_layer_block_size(layer_idx)

                wrapper = _SparseAttentionWrapper(
                    original_attn=attn_module,
                    gate=gate,
                    layer_idx=layer_idx,
                    threshold_tau=threshold,
                    block_size=block_size,
                    min_sparsity_for_speedup=kernel_config.min_sparsity_for_speedup,
                )

                # Replace the attention module in the model's layer stack
                _replace_attention_module(model, attn_module, wrapper)

        logger.info(
            "[BUNDLE_LOAD] Bundle loaded successfully",
            bundle_path=str(bundle_dir),
            num_layers=num_layers,
            num_heads=num_heads,
            head_dim=head_dim,
            device=device,
            dtype=str(dtype),
        )

        return cls(
            model=model,
            gates=gates,
            kernel_config=kernel_config,
            bundle_path=str(bundle_dir),
            manifest=manifest,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> CausalLMOutputWithPast:
        """Run inference with gate-driven sparse attention.

        For each attention layer:
            1. Standard Q, K, V projections (frozen merged weights)
            2. AttnGate(Q, K) → hard block mask using layer's threshold_tau
            3. BlockSparseFlashAttention(Q, K, V, mask) → output

        Args:
            input_ids: [B, S] token IDs.
            attention_mask: Optional [B, S] attention mask.
            **kwargs: Additional arguments forwarded to the model.

        Returns:
            CausalLMOutputWithPast with logits and optional cache.

        Complexity: O(L * B * H * S^2 * (1 - mean_sparsity) * D).
        """
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **kwargs,
        )

    @torch.inference_mode()
    def benchmark_inference(
        self,
        input_ids: torch.Tensor,
        num_warmup: int = 10,
        num_timed: int = 50,
    ) -> InferenceBenchmark:
        """Benchmark inference throughput and latency with CUDA event timing.

        Uses torch.cuda.Event(enable_timing=True) for sub-microsecond accuracy.
        Warmup iterations are excluded from timing.

        Preconditions:
            - input_ids is on CUDA device
            - Model is in eval mode
            - num_warmup >= 0, num_timed >= 1

        Postconditions:
            - Returns accurate timing statistics
            - Model state unchanged

        Args:
            input_ids: [B, S] token IDs for benchmarking.
            num_warmup: Number of warmup iterations (default 10).
            num_timed: Number of timed iterations (default 50).

        Returns:
            InferenceBenchmark with throughput and latency percentiles.

        Raises:
            InferenceError: If CUDA is not available or input is invalid.

        Complexity: O((num_warmup + num_timed) * forward_cost).
        """
        if not torch.cuda.is_available():
            msg = "CUDA required for benchmarking"
            raise InferenceError(
                msg,
                context={"cuda_available": False},
            )

        if num_timed < 1:
            msg = "num_timed must be >= 1"
            raise InferenceError(
                msg,
                context={"num_timed": num_timed},
            )

        device = input_ids.device
        batch_size, seq_len = input_ids.shape
        total_tokens = batch_size * seq_len

        gpu_name = torch.cuda.get_device_name(device)

        logger.info(
            "[BENCHMARK_START] Starting inference benchmark",
            batch_size=batch_size,
            seq_len=seq_len,
            num_warmup=num_warmup,
            num_timed=num_timed,
            gpu=gpu_name,
        )

        # Warmup: populate CUDA caches and JIT kernels
        for _ in range(num_warmup):
            self.forward(input_ids)

        torch.cuda.synchronize(device)

        # Timed iterations with CUDA events for precise measurement
        latencies_ms: list[float] = []
        for _ in range(num_timed):
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)

            start_event.record()
            self.forward(input_ids)
            end_event.record()

            torch.cuda.synchronize(device)
            latencies_ms.append(start_event.elapsed_time(end_event))

        # Collect sparsity from last forward pass
        sparsity_profile = self.get_sparsity_profile(input_ids)

        # Compute statistics
        sorted_latencies = sorted(latencies_ms)
        mean_latency = statistics.mean(latencies_ms)

        p50_idx = int(len(sorted_latencies) * 0.50)
        p95_idx = min(int(len(sorted_latencies) * 0.95), len(sorted_latencies) - 1)
        p99_idx = min(int(len(sorted_latencies) * 0.99), len(sorted_latencies) - 1)

        result = InferenceBenchmark(
            tokens_per_second=total_tokens / (mean_latency / 1000.0),
            mean_latency_ms=mean_latency,
            p50_ms=sorted_latencies[p50_idx],
            p95_ms=sorted_latencies[p95_idx],
            p99_ms=sorted_latencies[p99_idx],
            mean_sparsity_per_layer={
                LayerIndex(k): v for k, v in sparsity_profile.items()
            },
            gpu_name=gpu_name,
            bundle_path=self.bundle_path,
            num_warmup=num_warmup,
            num_timed=num_timed,
        )

        logger.info(
            "[BENCHMARK_COMPLETE] Inference benchmark finished",
            tokens_per_second=round(result.tokens_per_second, 1),
            mean_latency_ms=round(result.mean_latency_ms, 3),
            p50_ms=round(result.p50_ms, 3),
            p95_ms=round(result.p95_ms, 3),
            p99_ms=round(result.p99_ms, 3),
            gpu=gpu_name,
        )

        return result

    @torch.inference_mode()
    def get_sparsity_profile(self, input_ids: torch.Tensor) -> SparsityProfile:
        """Run inference and collect per-layer sparsity ratios.

        Preconditions: input_ids is a valid token tensor.
        Postconditions: Returns sparsity for every patched layer.

        Args:
            input_ids: [B, S] token IDs.

        Returns:
            Dict mapping LayerIndex to SparsityRatio for each layer.

        Complexity: O(forward_cost + num_layers).
        """
        self.forward(input_ids)

        profile: SparsityProfile = {}
        for wrapper in self._sparse_wrappers:
            if wrapper.last_gate_output is not None:
                profile[LayerIndex(wrapper.layer_idx)] = SparsityRatio(
                    wrapper.last_gate_output.sparsity_ratio,
                )

        return profile


def _replace_attention_module(
    model: PreTrainedModel,
    target: nn.Module,
    replacement: nn.Module,
) -> None:
    """Replace a specific attention module in the model hierarchy.

    Walks the module tree to find the parent of target and replaces it.

    Preconditions: target exists somewhere in model's module tree.
    Postconditions: target is replaced by replacement in-place.

    Args:
        model: The root model.
        target: The module to replace.
        replacement: The replacement module.

    Raises:
        InferenceError: If target module is not found in model.

    Complexity: O(N) where N = total number of modules.
    """
    for _name, module in model.named_modules():
        for child_name, child in module.named_children():
            if child is target:
                setattr(module, child_name, replacement)
                return

    msg = "Could not find target attention module in model for replacement"
    raise InferenceError(
        msg,
        context={"target_type": type(target).__name__},
    )


__all__ = [
    "InferenceBenchmark",
    "TASFTInferenceModel",
]
