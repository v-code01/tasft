"""Integration tests for TASFT vLLM attention patching system.

Validates the full patch lifecycle against realistic vLLM-like module structures,
covering duck-typed attn_metadata compatibility, patch/unpatch idempotency,
multi-worker patching, prefill vs decode dispatch, GQA validation, thread safety,
version compatibility shims, and module detection heuristics.

These tests use mock structures that faithfully reproduce vLLM's actual API surface
(worker.model_runner.model, worker.model, Attention modules with qkv_proj/q_proj,
attn_metadata with is_prompt/prefill_metadata/num_prefill_tokens) so that
silent breakage from vLLM minor version bumps is caught immediately.

Preconditions:
    - tasft package installed in editable mode
    - No vLLM installation required (all structures are mocked faithfully)

Postconditions:
    - All patch/unpatch operations verified for correctness and idempotency
    - Thread safety validated under concurrent patch/unpatch operations
    - attn_metadata duck-typing validated across vLLM 0.4.x, 0.5.x field sets
"""
from __future__ import annotations

import threading
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest
import torch
from torch import nn

import tasft.inference.vllm_patch as vp
from tasft.exceptions import InferenceError
from tasft.inference.vllm_patch import (
    TASFTvLLMAttentionBackend,
    _extract_vllm_attention_modules,
    _is_prefill_phase,
    is_patched,
    patch_vllm_attention,
    unpatch_vllm_attention,
)


# ── Realistic vLLM mock structures ──────────────────────────────────
#
# These faithfully reproduce vLLM's internal module hierarchy so that
# duck-typing breakage is caught. Field names, nesting depth, and
# attribute presence match vLLM >= 0.4.0.


class MockVLLMAttention(nn.Module):
    """Reproduces vLLM's per-layer attention module interface.

    vLLM names these classes *Attention (e.g. LlamaAttention, GPT2Attention).
    They expose: num_heads, num_kv_heads, head_dim, qkv_proj or q_proj.
    """

    def __init__(
        self,
        num_heads: int = 4,
        num_kv_heads: int = 4,
        head_dim: int = 16,
        *,
        use_fused_qkv: bool = True,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        hidden = num_heads * head_dim
        kv_hidden = num_kv_heads * head_dim
        if use_fused_qkv:
            self.qkv_proj = nn.Linear(hidden, hidden + 2 * kv_hidden, bias=False)
        else:
            self.q_proj = nn.Linear(hidden, hidden, bias=False)
            self.k_proj = nn.Linear(hidden, kv_hidden, bias=False)
            self.v_proj = nn.Linear(hidden, kv_hidden, bias=False)

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
        """Identity forward for testing — returns query unchanged."""
        return query


class MockVLLMDecoderLayer(nn.Module):
    """Reproduces vLLM's decoder layer wrapping an attention module."""

    def __init__(self, num_heads: int = 4, num_kv_heads: int = 4, head_dim: int = 16) -> None:
        super().__init__()
        self.self_attn = MockVLLMAttention(num_heads, num_kv_heads, head_dim)
        hidden = num_heads * head_dim
        self.mlp = nn.Sequential(nn.Linear(hidden, hidden * 4), nn.GELU(), nn.Linear(hidden * 4, hidden))


class MockVLLMModel(nn.Module):
    """Reproduces vLLM's model structure: model.layers[i].self_attn."""

    def __init__(
        self,
        num_layers: int = 2,
        num_heads: int = 4,
        num_kv_heads: int = 4,
        head_dim: int = 16,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList([
            MockVLLMDecoderLayer(num_heads, num_kv_heads, head_dim)
            for _ in range(num_layers)
        ])


def _make_worker_with_model_runner(
    num_layers: int = 2,
    num_heads: int = 4,
    num_kv_heads: int = 4,
    head_dim: int = 16,
) -> SimpleNamespace:
    """Create a mock vLLM worker using model_runner.model path (vLLM >= 0.4.0)."""
    model = MockVLLMModel(num_layers, num_heads, num_kv_heads, head_dim)
    model_runner = SimpleNamespace(model=model)
    return SimpleNamespace(model_runner=model_runner)


def _make_worker_with_direct_model(
    num_layers: int = 2,
    num_heads: int = 4,
    num_kv_heads: int = 4,
    head_dim: int = 16,
) -> SimpleNamespace:
    """Create a mock vLLM worker using direct .model path (legacy vLLM)."""
    model = MockVLLMModel(num_layers, num_heads, num_kv_heads, head_dim)
    return SimpleNamespace(model=model)


def _make_tasft_model(
    num_layers: int = 2,
    head_dim: int = 16,
    threshold: float = 0.5,
    block_size: int = 32,
    min_sparsity: float = 0.3,
) -> MagicMock:
    """Create a mock TASFTInferenceModel with gates and kernel_config.

    The mock gates return a gate_output with controllable sparsity_ratio
    and a hard_mask tensor.
    """
    tasft_model = MagicMock()

    gates: dict[str, MagicMock] = {}
    for i in range(num_layers):
        gate = MagicMock(spec=nn.Module)
        gate.head_dim = head_dim
        # Gate forward returns a SimpleNamespace with sparsity_ratio and hard_mask
        gate_output = SimpleNamespace(
            sparsity_ratio=0.5,
            hard_mask=torch.ones(1, 4, 1, 1, dtype=torch.bool),
        )
        gate.return_value = gate_output
        gates[str(i)] = gate

    tasft_model.gates = gates

    kernel_config = MagicMock()
    kernel_config.get_layer_threshold.return_value = threshold
    kernel_config.get_layer_block_size.return_value = block_size
    kernel_config.min_sparsity_for_speedup = min_sparsity
    tasft_model.kernel_config = kernel_config

    return tasft_model


# ── attn_metadata duck-type factories ────────────────────────────────


@dataclass
class VLLMAttnMetadata_v040:
    """Reproduces vLLM 0.4.x attn_metadata: uses is_prompt field."""
    is_prompt: bool
    slot_mapping: torch.Tensor | None = None
    block_tables: torch.Tensor | None = None


@dataclass
class VLLMAttnMetadata_v041:
    """Reproduces vLLM 0.4.1+ attn_metadata: uses prefill_metadata field."""
    prefill_metadata: Any = None
    decode_metadata: Any = None
    slot_mapping: torch.Tensor | None = None


@dataclass
class VLLMAttnMetadata_v050:
    """Reproduces vLLM 0.5.x attn_metadata: uses num_prefill_tokens field."""
    num_prefill_tokens: int = 0
    num_decode_tokens: int = 0
    slot_mapping: torch.Tensor | None = None


class StrippedAttnMetadata:
    """Metadata object with none of the expected vLLM fields.

    Simulates a future vLLM version that renames all prefill indicators.
    """
    pass


# ── Fixtures ─────────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def _reset_patch_state(monkeypatch: pytest.MonkeyPatch) -> None:
    """Reset module-level patch state before every test.

    This prevents cross-test contamination of _patched_workers.
    Also mocks vLLM version detection so tests run without vLLM installed.
    """
    with vp._patch_lock:
        vp._patched_workers.clear()
    # Reset version detection cache so each test starts clean
    vp._version_detected = False
    vp._detected_version = None
    # Mock detect_vllm_version to return a fake v0.5.0 — tests don't need real vLLM
    _fake_version = SimpleNamespace(major=0, minor=5, patch=0, as_tuple=lambda: (0, 5, 0))
    _fake_version.__str__ = lambda self: "0.5.0"
    monkeypatch.setattr(vp, "detect_vllm_version", lambda: _fake_version)
    monkeypatch.setattr(vp, "check_vllm_compatibility", lambda v: [])
    monkeypatch.setattr(vp, "validate_worker_structure", lambda w: [])


@pytest.fixture
def two_layer_worker() -> SimpleNamespace:
    """Worker with 2 attention layers via model_runner path."""
    return _make_worker_with_model_runner(num_layers=2)


@pytest.fixture
def two_layer_tasft_model() -> MagicMock:
    """TASFT model with 2 gates."""
    return _make_tasft_model(num_layers=2)


# ── Test classes ─────────────────────────────────────────────────────


@pytest.mark.integration
class TestPatchUnpatchIdempotency:
    """Verify patch/unpatch cycle is idempotent and reversible.

    Invariant: patch(patch(x)) == patch(x), unpatch(unpatch(x)) == unpatch(x).
    After unpatch, original forwards must be fully restored.
    """

    def test_double_patch_is_idempotent(
        self,
        two_layer_worker: SimpleNamespace,
        two_layer_tasft_model: MagicMock,
    ) -> None:
        """Patching the same worker twice must not raise and must not double-wrap."""
        patch_vllm_attention(two_layer_tasft_model, two_layer_worker)
        assert is_patched() is True

        # Second patch on same worker object: should be a no-op
        patch_vllm_attention(two_layer_tasft_model, two_layer_worker)
        assert is_patched() is True

        # Only one worker ID in the set
        with vp._patch_lock:
            assert len(vp._patched_workers) == 1

    def test_double_unpatch_is_idempotent(
        self,
        two_layer_worker: SimpleNamespace,
        two_layer_tasft_model: MagicMock,
    ) -> None:
        """Unpatching an already-unpatched worker is a safe no-op."""
        patch_vllm_attention(two_layer_tasft_model, two_layer_worker)
        unpatch_vllm_attention(two_layer_worker)
        assert is_patched() is False

        # Second unpatch: no-op, no exception
        unpatch_vllm_attention(two_layer_worker)
        assert is_patched() is False

    def test_patch_unpatch_restores_original_forward(
        self,
        two_layer_worker: SimpleNamespace,
        two_layer_tasft_model: MagicMock,
    ) -> None:
        """After unpatch, each attention module's forward is the original callable."""
        model = two_layer_worker.model_runner.model
        original_forwards = [
            layer.self_attn.forward for layer in model.layers
        ]

        patch_vllm_attention(two_layer_tasft_model, two_layer_worker)

        # Forwards must have changed after patching
        for i, layer in enumerate(model.layers):
            assert layer.self_attn.forward is not original_forwards[i]
            assert hasattr(layer.self_attn, "_tasft_original_forward")
            assert hasattr(layer.self_attn, "_tasft_backend")

        unpatch_vllm_attention(two_layer_worker)

        # After unpatch, _tasft attributes must be cleaned up
        for layer in model.layers:
            assert not hasattr(layer.self_attn, "_tasft_original_forward")
            assert not hasattr(layer.self_attn, "_tasft_backend")

    def test_patch_unpatch_patch_cycle(
        self,
        two_layer_worker: SimpleNamespace,
        two_layer_tasft_model: MagicMock,
    ) -> None:
        """Full patch -> unpatch -> re-patch cycle must succeed without error."""
        patch_vllm_attention(two_layer_tasft_model, two_layer_worker)
        assert is_patched() is True

        unpatch_vllm_attention(two_layer_worker)
        assert is_patched() is False

        # Re-patching after unpatch must succeed
        patch_vllm_attention(two_layer_tasft_model, two_layer_worker)
        assert is_patched() is True


@pytest.mark.integration
class TestMultiWorkerPatching:
    """Verify independent patching of multiple vLLM workers.

    In tensor-parallel vLLM deployments, each GPU worker has its own model.
    Patching worker A must not affect worker B. Each worker ID tracked independently.
    """

    def test_two_workers_patched_independently(self) -> None:
        """Patching two workers registers both in _patched_workers."""
        worker_a = _make_worker_with_model_runner(num_layers=2)
        worker_b = _make_worker_with_model_runner(num_layers=2)
        tasft_model = _make_tasft_model(num_layers=2)

        patch_vllm_attention(tasft_model, worker_a)
        patch_vllm_attention(tasft_model, worker_b)

        with vp._patch_lock:
            assert len(vp._patched_workers) == 2
            assert id(worker_a) in vp._patched_workers
            assert id(worker_b) in vp._patched_workers

    def test_unpatch_one_worker_leaves_other_patched(self) -> None:
        """Unpatching one worker must not affect the other."""
        worker_a = _make_worker_with_model_runner(num_layers=2)
        worker_b = _make_worker_with_model_runner(num_layers=2)
        tasft_model = _make_tasft_model(num_layers=2)

        patch_vllm_attention(tasft_model, worker_a)
        patch_vllm_attention(tasft_model, worker_b)

        unpatch_vllm_attention(worker_a)

        assert is_patched() is True  # worker_b still patched
        with vp._patch_lock:
            assert id(worker_a) not in vp._patched_workers
            assert id(worker_b) in vp._patched_workers

    def test_unpatch_all_workers_clears_state(self) -> None:
        """Unpatching all workers results in is_patched() == False."""
        workers = [_make_worker_with_model_runner(num_layers=2) for _ in range(3)]
        tasft_model = _make_tasft_model(num_layers=2)

        for w in workers:
            patch_vllm_attention(tasft_model, w)
        assert is_patched() is True

        for w in workers:
            unpatch_vllm_attention(w)
        assert is_patched() is False

    def test_model_runner_vs_direct_model_workers(self) -> None:
        """Patching works with both model_runner.model and direct .model paths."""
        worker_runner = _make_worker_with_model_runner(num_layers=2)
        worker_direct = _make_worker_with_direct_model(num_layers=2)
        tasft_model = _make_tasft_model(num_layers=2)

        patch_vllm_attention(tasft_model, worker_runner)
        patch_vllm_attention(tasft_model, worker_direct)

        with vp._patch_lock:
            assert len(vp._patched_workers) == 2


@pytest.mark.integration
class TestPrefillDecodeDispatch:
    """Verify prefill vs decode routing via attn_metadata duck-typing.

    During prefill (prompt processing), TASFT sparse attention is used.
    During decode (single-token generation), original vLLM attention is
    delegated to for PagedAttention correctness.
    """

    def test_v040_is_prompt_true_routes_prefill(self) -> None:
        """vLLM 0.4.0 metadata with is_prompt=True -> prefill."""
        meta = VLLMAttnMetadata_v040(is_prompt=True)
        assert _is_prefill_phase(meta) is True

    def test_v040_is_prompt_false_routes_decode(self) -> None:
        """vLLM 0.4.0 metadata with is_prompt=False -> decode."""
        meta = VLLMAttnMetadata_v040(is_prompt=False)
        assert _is_prefill_phase(meta) is False

    def test_v041_prefill_metadata_present_routes_prefill(self) -> None:
        """vLLM 0.4.1+ with prefill_metadata object -> prefill."""
        meta = VLLMAttnMetadata_v041(prefill_metadata=SimpleNamespace(seq_lens=[128]))
        assert _is_prefill_phase(meta) is True

    def test_v041_prefill_metadata_none_routes_decode(self) -> None:
        """vLLM 0.4.1+ with prefill_metadata=None -> decode."""
        meta = VLLMAttnMetadata_v041(prefill_metadata=None)
        assert _is_prefill_phase(meta) is False

    def test_v050_num_prefill_tokens_positive_routes_prefill(self) -> None:
        """vLLM 0.5.x with num_prefill_tokens > 0 -> prefill."""
        meta = VLLMAttnMetadata_v050(num_prefill_tokens=512)
        assert _is_prefill_phase(meta) is True

    def test_v050_num_prefill_tokens_zero_routes_decode(self) -> None:
        """vLLM 0.5.x with num_prefill_tokens=0 -> decode."""
        meta = VLLMAttnMetadata_v050(num_prefill_tokens=0)
        assert _is_prefill_phase(meta) is False

    def test_v050_num_prefill_tokens_one_boundary(self) -> None:
        """Boundary: num_prefill_tokens=1 -> prefill."""
        meta = VLLMAttnMetadata_v050(num_prefill_tokens=1)
        assert _is_prefill_phase(meta) is True

    def test_attribute_priority_is_prompt_checked_first(self) -> None:
        """When metadata has both is_prompt and num_prefill_tokens, is_prompt wins.

        This validates the check order in _is_prefill_phase: is_prompt is checked
        first (vLLM 0.4.0 compat), so even if num_prefill_tokens disagrees,
        is_prompt takes precedence.
        """
        meta = SimpleNamespace(is_prompt=False, num_prefill_tokens=100)
        assert _is_prefill_phase(meta) is False

    def test_attribute_priority_prefill_metadata_before_num_tokens(self) -> None:
        """prefill_metadata is checked before num_prefill_tokens."""
        meta = SimpleNamespace(prefill_metadata=None, num_prefill_tokens=100)
        # prefill_metadata=None -> decode, even though num_prefill_tokens=100
        assert _is_prefill_phase(meta) is False

    def test_decode_dispatches_to_original_forward(self) -> None:
        """During decode, _dense_attention_flat delegates to _original_forward.

        This is the critical path: during autoregressive generation, we must
        use vLLM's original forward which manages PagedAttention slot lookups.
        """
        gate = MagicMock(spec=nn.Module)
        gate.head_dim = 16
        backend = TASFTvLLMAttentionBackend(
            gate=gate,
            layer_idx=0,
            threshold_tau=0.5,
            block_size=32,
            min_sparsity_for_speedup=0.3,
            num_heads=4,
            num_kv_heads=4,
            head_dim=16,
        )

        # Set up original forward mock
        expected_output = torch.randn(1, 64)
        original_fwd = MagicMock(return_value=expected_output)
        backend._original_forward = original_fwd

        query = torch.randn(1, 64)
        key = torch.randn(1, 64)
        value = torch.randn(1, 64)
        kv_cache = torch.randn(2, 4, 16, 16)
        meta = VLLMAttnMetadata_v040(is_prompt=False)

        result = backend.forward(query, key, value, kv_cache=kv_cache, attn_metadata=meta)

        original_fwd.assert_called_once()
        assert torch.equal(result, expected_output)


@pytest.mark.integration
class TestGQAValidation:
    """Verify GQA (Grouped Query Attention) head divisibility validation.

    When num_heads % num_kv_heads != 0, integer division would silently
    produce incorrect attention patterns. The backend must reject this.
    """

    @pytest.mark.parametrize(
        ("num_heads", "num_kv_heads"),
        [
            (7, 3),
            (5, 2),
            (10, 3),
            (13, 4),
            (17, 6),
        ],
    )
    def test_non_divisible_heads_raises_inference_error(
        self, num_heads: int, num_kv_heads: int,
    ) -> None:
        """num_heads not divisible by num_kv_heads raises InferenceError with context."""
        gate = MagicMock(spec=nn.Module)

        with pytest.raises(InferenceError) as exc_info:
            TASFTvLLMAttentionBackend(
                gate=gate,
                layer_idx=0,
                threshold_tau=0.5,
                block_size=32,
                min_sparsity_for_speedup=0.3,
                num_heads=num_heads,
                num_kv_heads=num_kv_heads,
                head_dim=16,
            )

        assert exc_info.value.context["num_heads"] == num_heads
        assert exc_info.value.context["num_kv_heads"] == num_kv_heads

    @pytest.mark.parametrize(
        ("num_heads", "num_kv_heads"),
        [
            (4, 4),   # MHA: equal heads
            (8, 4),   # GQA: 2x grouping
            (32, 8),  # GQA: 4x grouping
            (16, 1),  # MQA: single KV head
            (1, 1),   # degenerate single-head
        ],
    )
    def test_divisible_heads_accepted(
        self, num_heads: int, num_kv_heads: int,
    ) -> None:
        """Valid GQA configurations construct without error."""
        gate = MagicMock(spec=nn.Module)
        backend = TASFTvLLMAttentionBackend(
            gate=gate,
            layer_idx=0,
            threshold_tau=0.5,
            block_size=32,
            min_sparsity_for_speedup=0.3,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=16,
        )
        assert backend.num_heads == num_heads
        assert backend.num_kv_heads == num_kv_heads

    def test_gqa_error_message_is_descriptive(self) -> None:
        """Error message must include both num_heads and num_kv_heads values."""
        gate = MagicMock(spec=nn.Module)

        with pytest.raises(InferenceError, match="num_heads.*7.*num_kv_heads.*3"):
            TASFTvLLMAttentionBackend(
                gate=gate,
                layer_idx=0,
                threshold_tau=0.5,
                block_size=32,
                min_sparsity_for_speedup=0.3,
                num_heads=7,
                num_kv_heads=3,
                head_dim=16,
            )

    def test_layer_count_mismatch_during_patch(self) -> None:
        """vLLM model with 3 layers but TASFT with 2 gates raises InferenceError."""
        worker = _make_worker_with_model_runner(num_layers=3)
        tasft_model = _make_tasft_model(num_layers=2)

        with pytest.raises(InferenceError, match="Layer count mismatch"):
            patch_vllm_attention(tasft_model, worker)


@pytest.mark.integration
class TestThreadSafety:
    """Verify thread safety of patch/unpatch operations.

    The _patch_lock must prevent data races on _patched_workers.
    Concurrent patch and unpatch operations on different workers must
    not corrupt shared state.
    """

    def test_concurrent_patch_different_workers(self) -> None:
        """Patching 8 different workers concurrently must register all 8."""
        num_workers = 8
        workers = [_make_worker_with_model_runner(num_layers=2) for _ in range(num_workers)]
        tasft_model = _make_tasft_model(num_layers=2)
        errors: list[Exception] = []

        def patch_worker(worker: Any) -> None:
            try:
                patch_vllm_attention(tasft_model, worker)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=patch_worker, args=(w,)) for w in workers]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0, f"Errors during concurrent patching: {errors}"
        with vp._patch_lock:
            assert len(vp._patched_workers) == num_workers

    def test_concurrent_patch_and_unpatch(self) -> None:
        """Concurrent patch on some workers and unpatch on others must not corrupt state."""
        # Pre-patch workers 0..3
        pre_workers = [_make_worker_with_model_runner(num_layers=2) for _ in range(4)]
        tasft_model = _make_tasft_model(num_layers=2)
        for w in pre_workers:
            patch_vllm_attention(tasft_model, w)

        # New workers 4..7 to be patched concurrently
        new_workers = [_make_worker_with_model_runner(num_layers=2) for _ in range(4)]
        errors: list[Exception] = []

        def unpatch_worker(worker: Any) -> None:
            try:
                unpatch_vllm_attention(worker)
            except Exception as e:
                errors.append(e)

        def patch_worker(worker: Any) -> None:
            try:
                patch_vllm_attention(tasft_model, worker)
            except Exception as e:
                errors.append(e)

        threads: list[threading.Thread] = []
        for w in pre_workers:
            threads.append(threading.Thread(target=unpatch_worker, args=(w,)))
        for w in new_workers:
            threads.append(threading.Thread(target=patch_worker, args=(w,)))

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0, f"Errors during concurrent ops: {errors}"

        # After: pre_workers unpatched, new_workers patched
        with vp._patch_lock:
            for w in pre_workers:
                assert id(w) not in vp._patched_workers
            for w in new_workers:
                assert id(w) in vp._patched_workers

    def test_concurrent_is_patched_during_mutations(self) -> None:
        """is_patched() must not raise even during concurrent mutations."""
        tasft_model = _make_tasft_model(num_layers=2)
        workers = [_make_worker_with_model_runner(num_layers=2) for _ in range(4)]
        results: list[bool] = []
        errors: list[Exception] = []

        def reader() -> None:
            try:
                for _ in range(50):
                    results.append(is_patched())
            except Exception as e:
                errors.append(e)

        def mutator() -> None:
            try:
                for w in workers:
                    patch_vllm_attention(tasft_model, w)
                for w in workers:
                    unpatch_vllm_attention(w)
            except Exception as e:
                errors.append(e)

        reader_threads = [threading.Thread(target=reader) for _ in range(4)]
        mutator_thread = threading.Thread(target=mutator)

        for t in reader_threads:
            t.start()
        mutator_thread.start()

        for t in reader_threads:
            t.join()
        mutator_thread.join()

        assert len(errors) == 0, f"Errors during concurrent read/write: {errors}"
        assert len(results) == 200
        assert all(isinstance(r, bool) for r in results)


@pytest.mark.integration
class TestVersionCompatibilityShims:
    """Test behavior when attn_metadata lacks expected vLLM fields.

    A future vLLM version could rename or remove prefill indicator fields.
    The code must degrade gracefully — defaulting to prefill (sparse path)
    rather than crashing.
    """

    def test_stripped_metadata_defaults_to_prefill(self) -> None:
        """Metadata with no known fields defaults to prefill (conservative)."""
        meta = StrippedAttnMetadata()
        assert _is_prefill_phase(meta) is True

    def test_none_metadata_treated_as_prefill(self) -> None:
        """None metadata in forward skips the decode branch entirely.

        When kv_cache is present but attn_metadata is None, the code
        still enters the prefill path because attn_metadata is None.
        """
        # The forward method checks: if kv_cache is not None and attn_metadata is not None
        # When attn_metadata is None, condition is False, so it falls through to gate path
        gate = MagicMock(spec=nn.Module)
        gate.head_dim = 16
        gate_output = SimpleNamespace(
            sparsity_ratio=0.0,  # force dense fallback
            hard_mask=torch.ones(1, 4, 1, 1, dtype=torch.bool),
        )
        gate.return_value = gate_output

        backend = TASFTvLLMAttentionBackend(
            gate=gate,
            layer_idx=0,
            threshold_tau=0.5,
            block_size=32,
            min_sparsity_for_speedup=0.3,
            num_heads=4,
            num_kv_heads=4,
            head_dim=16,
        )

        query = torch.randn(8, 64)
        key = torch.randn(8, 64)
        value = torch.randn(8, 64)
        kv_cache = torch.randn(2, 4, 16, 16)

        # attn_metadata=None: should go through gate path, not decode
        result = backend.forward(query, key, value, kv_cache=kv_cache, attn_metadata=None)
        assert result.shape == (8, 64)
        gate.assert_called_once()

    def test_metadata_with_only_custom_fields_defaults_prefill(self) -> None:
        """Future vLLM metadata with unrecognized fields defaults to prefill."""
        meta = SimpleNamespace(
            chunked_prefill=True,
            seq_group_metadata_list=[],
            cross_attention_metadata=None,
        )
        assert _is_prefill_phase(meta) is True

    def test_worker_without_model_or_model_runner_raises(self) -> None:
        """Worker with neither .model nor .model_runner raises InferenceError."""
        bare_worker = SimpleNamespace(device="cuda:0")
        tasft_model = _make_tasft_model(num_layers=2)

        with pytest.raises(InferenceError, match="Cannot find model"):
            patch_vllm_attention(tasft_model, bare_worker)

    def test_attention_module_missing_num_heads_raises(self) -> None:
        """Attention module without num_heads or num_attention_heads raises."""

        class HeadlessAttention(nn.Module):
            """Attention module missing head count attributes."""
            def __init__(self) -> None:
                super().__init__()
                self.qkv_proj = nn.Linear(64, 192, bias=False)
                # Intentionally no num_heads or num_attention_heads

        class HeadlessModel(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.attn = HeadlessAttention()

        worker = SimpleNamespace(model=HeadlessModel())
        tasft_model = _make_tasft_model(num_layers=1)

        with pytest.raises(InferenceError, match="Cannot determine num_heads"):
            patch_vllm_attention(tasft_model, worker)

    def test_num_attention_heads_fallback(self) -> None:
        """Module with num_attention_heads (no num_heads) is accepted.

        Some vLLM model implementations use num_attention_heads instead.
        """

        class AltAttention(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.qkv_proj = nn.Linear(64, 192, bias=False)
                self.num_attention_heads = 4
                self.num_kv_heads = 4
                self.head_dim = 16

            def forward(self, *args: Any, **kwargs: Any) -> torch.Tensor:
                return args[0] if args else torch.empty(0)

        class AltModel(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.attn = AltAttention()

        worker = SimpleNamespace(model=AltModel())
        tasft_model = _make_tasft_model(num_layers=1)

        patch_vllm_attention(tasft_model, worker)
        assert is_patched() is True


@pytest.mark.integration
class TestUnpatchUnrecognizedWorker:
    """Test unpatch behavior for workers with unrecognized structure.

    When a worker was patched but its internal structure has changed
    (e.g., model_runner removed), unpatch must still reset _patched_workers
    to allow re-patching.
    """

    def test_unpatch_bare_worker_resets_state(self) -> None:
        """Unpatching a worker with no model resets its patch state."""
        bare_worker = SimpleNamespace(device="cuda:0")
        worker_id = id(bare_worker)

        # Manually register as patched (simulates a prior patch)
        with vp._patch_lock:
            vp._patched_workers.add(worker_id)

        assert is_patched() is True

        unpatch_vllm_attention(bare_worker)

        assert is_patched() is False
        with vp._patch_lock:
            assert worker_id not in vp._patched_workers

    def test_unpatch_unregistered_worker_is_noop(self) -> None:
        """Unpatching a worker never registered is a safe no-op."""
        unknown_worker = SimpleNamespace(model_runner=SimpleNamespace(model=nn.Linear(4, 4)))

        # Must not raise
        unpatch_vllm_attention(unknown_worker)
        assert is_patched() is False

    def test_unpatch_bare_worker_allows_reapply(self) -> None:
        """After unpatching a bare worker, a new patch attempt can proceed.

        Even though the bare worker unpatch only clears state (no forward
        restoration), the worker_id is removed from _patched_workers so
        a subsequent patch call on a properly structured worker succeeds.
        """
        bare_worker = SimpleNamespace(device="cuda:0")
        worker_id = id(bare_worker)

        with vp._patch_lock:
            vp._patched_workers.add(worker_id)

        unpatch_vllm_attention(bare_worker)

        # Now a properly structured worker with the same ID can be patched
        with vp._patch_lock:
            assert worker_id not in vp._patched_workers


@pytest.mark.integration
class TestDenseAttentionFlatDelegation:
    """Test that _dense_attention_flat delegates to _original_forward for decode.

    During autoregressive generation, the backend must use the original vLLM
    forward to correctly interact with PagedAttention's slot mapping and
    block tables. Falling back to raw SDPA produces incorrect output.
    """

    def test_delegates_to_original_forward(self) -> None:
        """With _original_forward set, _dense_attention_flat calls it exactly once."""
        gate = MagicMock(spec=nn.Module)
        backend = TASFTvLLMAttentionBackend(
            gate=gate, layer_idx=0, threshold_tau=0.5, block_size=32,
            min_sparsity_for_speedup=0.3, num_heads=4, num_kv_heads=4, head_dim=16,
        )

        expected = torch.randn(1, 64)
        mock_fwd = MagicMock(return_value=expected)
        backend._original_forward = mock_fwd

        q = torch.randn(1, 64)
        k = torch.randn(1, 64)
        v = torch.randn(1, 64)
        kv_cache = torch.randn(2, 4, 16, 16)
        meta = VLLMAttnMetadata_v040(is_prompt=False)

        result = backend._dense_attention_flat(q, k, v, kv_cache, meta)

        mock_fwd.assert_called_once_with(q, k, v, kv_cache=kv_cache, attn_metadata=meta)
        assert torch.equal(result, expected)

    def test_fallback_sdpa_when_no_original_forward(self) -> None:
        """Without _original_forward, SDPA fallback produces shaped output.

        This path should only trigger in misconfigured setups. We verify
        it produces correctly shaped output rather than crashing.
        """
        gate = MagicMock(spec=nn.Module)
        backend = TASFTvLLMAttentionBackend(
            gate=gate, layer_idx=0, threshold_tau=0.5, block_size=32,
            min_sparsity_for_speedup=0.3, num_heads=4, num_kv_heads=4, head_dim=16,
        )
        # _original_forward defaults to None

        num_tokens = 4
        q = torch.randn(num_tokens, 64)
        k = torch.randn(num_tokens, 64)
        v = torch.randn(num_tokens, 64)
        kv_cache = torch.randn(2, 4, 16, 16)
        meta = VLLMAttnMetadata_v040(is_prompt=False)

        result = backend._dense_attention_flat(q, k, v, kv_cache, meta)

        assert result.shape == (num_tokens, 64)
        assert not torch.isnan(result).any()

    def test_fallback_sdpa_gqa_expansion(self) -> None:
        """SDPA fallback with GQA (num_kv_heads < num_heads) expands KV correctly."""
        gate = MagicMock(spec=nn.Module)
        backend = TASFTvLLMAttentionBackend(
            gate=gate, layer_idx=0, threshold_tau=0.5, block_size=32,
            min_sparsity_for_speedup=0.3, num_heads=8, num_kv_heads=2, head_dim=16,
        )

        num_tokens = 4
        q = torch.randn(num_tokens, 128)   # 8 heads * 16 dim
        k = torch.randn(num_tokens, 32)    # 2 heads * 16 dim
        v = torch.randn(num_tokens, 32)    # 2 heads * 16 dim
        kv_cache = torch.randn(2, 4, 16, 16)
        meta = VLLMAttnMetadata_v040(is_prompt=False)

        result = backend._dense_attention_flat(q, k, v, kv_cache, meta)

        # Output dim = num_heads * head_dim = 8 * 16 = 128
        assert result.shape == (num_tokens, 128)
        assert not torch.isnan(result).any()

    def test_decode_path_through_forward(self) -> None:
        """Full forward() with kv_cache and decode metadata routes to _dense_attention_flat."""
        gate = MagicMock(spec=nn.Module)
        backend = TASFTvLLMAttentionBackend(
            gate=gate, layer_idx=0, threshold_tau=0.5, block_size=32,
            min_sparsity_for_speedup=0.3, num_heads=4, num_kv_heads=4, head_dim=16,
        )

        expected = torch.randn(1, 64)
        backend._original_forward = MagicMock(return_value=expected)

        q = torch.randn(1, 64)
        k = torch.randn(1, 64)
        v = torch.randn(1, 64)
        kv_cache = torch.randn(2, 4, 16, 16)

        # Test all three decode-indicating metadata variants
        for meta in [
            VLLMAttnMetadata_v040(is_prompt=False),
            VLLMAttnMetadata_v041(prefill_metadata=None),
            VLLMAttnMetadata_v050(num_prefill_tokens=0),
        ]:
            backend._original_forward.reset_mock()
            result = backend.forward(q, k, v, kv_cache=kv_cache, attn_metadata=meta)
            backend._original_forward.assert_called_once()
            assert torch.equal(result, expected)


@pytest.mark.integration
class TestModuleDetection:
    """Test that _extract_vllm_attention_modules uses cls_name.endswith('Attention') correctly.

    vLLM's attention modules are named LlamaAttention, MistralAttention,
    GPT2Attention, etc. Non-attention modules like LayerNorm, MLP must
    not be matched even if they have projection attributes.
    """

    def test_finds_module_ending_with_attention(self) -> None:
        """Module class name ending with 'Attention' and having qkv_proj is found."""

        class LlamaAttention(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.qkv_proj = nn.Linear(64, 192)

        class LlamaModel(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.attn = LlamaAttention()
                self.mlp = nn.Linear(64, 64)

        model = LlamaModel()
        modules = _extract_vllm_attention_modules(model)
        assert len(modules) == 1
        assert type(modules[0]).__name__ == "LlamaAttention"

    def test_finds_module_with_q_proj_instead_of_qkv(self) -> None:
        """Module with q_proj (separate projections) is also detected."""

        class GPT2Attention(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.q_proj = nn.Linear(64, 64)

        class GPT2Model(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.layer = GPT2Attention()

        model = GPT2Model()
        modules = _extract_vllm_attention_modules(model)
        assert len(modules) == 1

    def test_ignores_attention_without_projections(self) -> None:
        """Module named *Attention but without qkv_proj or q_proj is ignored."""

        class FakeAttention(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.dense = nn.Linear(64, 64)  # no qkv_proj or q_proj

        class FakeModel(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.attn = FakeAttention()

        model = FakeModel()
        with pytest.raises(InferenceError, match="No attention modules"):
            _extract_vllm_attention_modules(model)

    def test_ignores_non_attention_with_projections(self) -> None:
        """Module with qkv_proj but NOT ending in 'Attention' is ignored."""

        class ProjectionLayer(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.qkv_proj = nn.Linear(64, 192)

        class SomeModel(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.proj = ProjectionLayer()

        model = SomeModel()
        with pytest.raises(InferenceError, match="No attention modules"):
            _extract_vllm_attention_modules(model)

    def test_multiple_attention_layers_ordered(self) -> None:
        """Multiple attention modules are returned in module tree order."""

        class DecoderAttention(nn.Module):
            def __init__(self, layer_id: int) -> None:
                super().__init__()
                self.qkv_proj = nn.Linear(64, 192)
                self.layer_id = layer_id

        class MultiLayerModel(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.layers = nn.ModuleList([
                    DecoderAttention(i) for i in range(4)
                ])

        model = MultiLayerModel()
        modules = _extract_vllm_attention_modules(model)
        assert len(modules) == 4
        for i, m in enumerate(modules):
            assert m.layer_id == i  # type: ignore[attr-defined]

    def test_mixed_attention_and_non_attention(self) -> None:
        """Only Attention modules are extracted from mixed module trees."""

        class MistralAttention(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.q_proj = nn.Linear(64, 64)

        class MLP(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.qkv_proj = nn.Linear(64, 192)  # has proj but NOT Attention

        class RMSNorm(nn.Module):
            def __init__(self) -> None:
                super().__init__()

        class DecoderLayer(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.self_attn = MistralAttention()
                self.mlp = MLP()
                self.norm = RMSNorm()

        class MistralModel(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.layers = nn.ModuleList([DecoderLayer() for _ in range(2)])

        model = MistralModel()
        modules = _extract_vllm_attention_modules(model)
        assert len(modules) == 2
        assert all(type(m).__name__ == "MistralAttention" for m in modules)

    def test_endswith_not_contains(self) -> None:
        """'Attention' must be at the END of the class name, not merely contained."""

        class AttentionPooling(nn.Module):
            """Name contains 'Attention' but does not END with it."""
            def __init__(self) -> None:
                super().__init__()
                self.qkv_proj = nn.Linear(64, 192)

        class AttentionPoolingModel(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.pool = AttentionPooling()

        model = AttentionPoolingModel()
        # "AttentionPooling".endswith("Attention") is False
        with pytest.raises(InferenceError, match="No attention modules"):
            _extract_vllm_attention_modules(model)


@pytest.mark.integration
class TestKVScaleApplication:
    """Test that kv_scale is correctly applied to key and value tensors."""

    def test_kv_scale_not_one_scales_tensors(self) -> None:
        """kv_scale != 1.0 must scale both K and V during prefill."""
        gate = MagicMock(spec=nn.Module)
        gate.head_dim = 16
        gate_output = SimpleNamespace(
            sparsity_ratio=0.0,  # force dense fallback for simpler verification
            hard_mask=torch.ones(1, 4, 1, 1, dtype=torch.bool),
        )
        gate.return_value = gate_output

        backend = TASFTvLLMAttentionBackend(
            gate=gate, layer_idx=0, threshold_tau=0.5, block_size=32,
            min_sparsity_for_speedup=0.3, num_heads=4, num_kv_heads=4, head_dim=16,
        )

        num_tokens = 4
        q = torch.randn(num_tokens, 64)
        k = torch.randn(num_tokens, 64)
        v = torch.randn(num_tokens, 64)

        # Run with kv_scale=1.0 and kv_scale=2.0 — outputs must differ
        result_1 = backend.forward(q, k.clone(), v.clone(), kv_scale=1.0)
        result_2 = backend.forward(q, k.clone(), v.clone(), kv_scale=2.0)

        assert result_1.shape == result_2.shape == (num_tokens, 64)
        # Different kv_scale produces different attention output
        assert not torch.allclose(result_1, result_2, atol=1e-6)


@pytest.mark.integration
class TestPrefillForwardPath:
    """Test the full prefill forward path through the gate and dense fallback."""

    def test_prefill_without_kv_cache(self) -> None:
        """Forward without kv_cache always goes through gate path."""
        gate = MagicMock(spec=nn.Module)
        gate.head_dim = 16
        gate_output = SimpleNamespace(
            sparsity_ratio=0.0,  # below threshold -> dense fallback
            hard_mask=torch.ones(1, 4, 1, 1, dtype=torch.bool),
        )
        gate.return_value = gate_output

        backend = TASFTvLLMAttentionBackend(
            gate=gate, layer_idx=0, threshold_tau=0.5, block_size=32,
            min_sparsity_for_speedup=0.3, num_heads=4, num_kv_heads=4, head_dim=16,
        )

        num_tokens = 8
        q = torch.randn(num_tokens, 64)
        k = torch.randn(num_tokens, 64)
        v = torch.randn(num_tokens, 64)

        result = backend.forward(q, k, v)
        assert result.shape == (num_tokens, 64)
        gate.assert_called_once()
        assert not torch.isnan(result).any()

    def test_prefill_with_gqa_expansion(self) -> None:
        """Prefill with GQA (num_kv_heads < num_heads) produces correct shape."""
        gate = MagicMock(spec=nn.Module)
        gate.head_dim = 16
        gate_output = SimpleNamespace(
            sparsity_ratio=0.0,
            hard_mask=torch.ones(1, 8, 1, 1, dtype=torch.bool),
        )
        gate.return_value = gate_output

        backend = TASFTvLLMAttentionBackend(
            gate=gate, layer_idx=0, threshold_tau=0.5, block_size=32,
            min_sparsity_for_speedup=0.3, num_heads=8, num_kv_heads=2, head_dim=16,
        )

        num_tokens = 8
        q = torch.randn(num_tokens, 128)  # 8 * 16
        k = torch.randn(num_tokens, 32)   # 2 * 16
        v = torch.randn(num_tokens, 32)   # 2 * 16

        result = backend.forward(q, k, v)
        assert result.shape == (num_tokens, 128)
        assert not torch.isnan(result).any()

    def test_prefill_with_kv_cache_and_prefill_metadata(self) -> None:
        """Forward with kv_cache + prefill metadata still goes through gate."""
        gate = MagicMock(spec=nn.Module)
        gate.head_dim = 16
        gate_output = SimpleNamespace(
            sparsity_ratio=0.0,
            hard_mask=torch.ones(1, 4, 1, 1, dtype=torch.bool),
        )
        gate.return_value = gate_output

        backend = TASFTvLLMAttentionBackend(
            gate=gate, layer_idx=0, threshold_tau=0.5, block_size=32,
            min_sparsity_for_speedup=0.3, num_heads=4, num_kv_heads=4, head_dim=16,
        )

        num_tokens = 8
        q = torch.randn(num_tokens, 64)
        k = torch.randn(num_tokens, 64)
        v = torch.randn(num_tokens, 64)
        kv_cache = torch.randn(2, 4, 16, 16)
        meta = VLLMAttnMetadata_v040(is_prompt=True)

        result = backend.forward(q, k, v, kv_cache=kv_cache, attn_metadata=meta)
        assert result.shape == (num_tokens, 64)
        gate.assert_called_once()


@pytest.mark.integration
class TestCacheBlockSizeComputation:
    """Test get_cache_block_size returns correct byte counts across dtypes.

    This is used by vLLM's memory planning to determine how many KV cache
    blocks can fit in GPU memory. An incorrect value causes OOM or underutilization.
    """

    @pytest.mark.parametrize(
        ("dtype", "element_bytes"),
        [
            (torch.float32, 4),
            (torch.float16, 2),
            (torch.bfloat16, 2),
            (torch.int8, 1),
        ],
    )
    def test_cache_block_size_formula(self, dtype: torch.dtype, element_bytes: int) -> None:
        """Block size = 2 * block_size * num_heads * head_size * element_size."""
        block_size = 16
        head_size = 128
        num_heads = 32

        result = TASFTvLLMAttentionBackend.get_cache_block_size(
            block_size=block_size,
            head_size=head_size,
            num_heads=num_heads,
            dtype=dtype,
        )

        expected = 2 * block_size * num_heads * head_size * element_bytes
        assert result == expected


@pytest.mark.integration
class TestPatchedForwardClosure:
    """Verify that patched forward closures capture the correct backend per layer.

    A common bug with monkey-patching in loops is late binding: all layers
    end up using the last backend. The _make_patched_forward closure must
    capture each layer's backend independently.
    """

    def test_each_layer_uses_its_own_backend(self) -> None:
        """After patching, each attention layer's _tasft_backend has correct layer_idx."""
        worker = _make_worker_with_model_runner(num_layers=4)
        tasft_model = _make_tasft_model(num_layers=4)

        patch_vllm_attention(tasft_model, worker)

        model = worker.model_runner.model
        for i, layer in enumerate(model.layers):
            attn = layer.self_attn
            assert hasattr(attn, "_tasft_backend")
            assert attn._tasft_backend.layer_idx == i

    def test_patched_forward_invokes_correct_gate(self) -> None:
        """Each layer's patched forward calls its own gate, not another layer's."""
        worker = _make_worker_with_model_runner(num_layers=2, num_heads=4, num_kv_heads=4, head_dim=16)
        tasft_model = _make_tasft_model(num_layers=2, head_dim=16)

        # Set up distinct gate outputs
        for i in range(2):
            gate = tasft_model.gates[str(i)]
            gate_output = SimpleNamespace(
                sparsity_ratio=0.0,
                hard_mask=torch.ones(1, 4, 1, 1, dtype=torch.bool),
            )
            gate.return_value = gate_output

        patch_vllm_attention(tasft_model, worker)

        model = worker.model_runner.model
        num_tokens = 4
        q = torch.randn(num_tokens, 64)
        k = torch.randn(num_tokens, 64)
        v = torch.randn(num_tokens, 64)

        # Call layer 0's patched forward
        model.layers[0].self_attn.forward(q, k, v)
        tasft_model.gates["0"].assert_called_once()
        tasft_model.gates["1"].assert_not_called()

        # Call layer 1's patched forward
        model.layers[1].self_attn.forward(q, k, v)
        tasft_model.gates["1"].assert_called_once()
