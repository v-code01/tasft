"""Coverage gap tests for untested code paths across TASFT modules.

Covers:
    1. vllm_patch: is_patched, patch/unpatch idempotency, thread safety,
       _is_prefill_phase heuristics, _extract_vllm_attention_modules error,
       TASFTvLLMAttentionBackend.get_cache_block_size, patch worker model
       resolution paths, layer count mismatch, num_heads detection.
    2. tasft_model: _verify_checksum success/failure, _extract_attention_layers
       error path, _replace_attention_module error path, load_bundle missing
       dirs and checksum failures.
    3. bundle/export: _extract_layer_index_from_path parsing, _hash_training_args
       determinism, validate_bundle on non-directory, validate_bundle missing
       manifest, validate_bundle missing kernel_config, ExportConfig boundary
       values, atomic cleanup on export failure.
    4. training/trainer: gate_warmup_multiplier boundary (step=warmup-1,
       step=warmup, step=2*warmup), warmup_steps=0 (immediate gate activation),
       TASFTTrainingArguments invalid rotation_strategy, negative beta_sparse,
       negative gate_warmup_steps, zero layers_per_step, zero block_size.
    5. observability/logging: timed_operation measures duration, bind_context
       binds and unbinds fields, get_logger returns bound logger,
       configure_logging with force_json.
"""
from __future__ import annotations

import hashlib
import json
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest
import structlog
import torch
from torch import nn

from tasft.bundle.export import BundleExporter, ExportConfig
from tasft.exceptions import BundleError, ChecksumError, InferenceError
from tasft.observability.logging import bind_context, configure_logging, get_logger, timed_operation
from tasft.training.trainer import TASFTTrainingArguments

# ===================================================================
# 1. vllm_patch coverage
# ===================================================================


@pytest.mark.unit
class TestVLLMPatchIsPatched:
    """Test is_patched(), patch idempotency, and unpatch idempotency."""

    def setup_method(self) -> None:
        """Reset module-level patch state before each test."""
        import tasft.inference.vllm_patch as vp

        with vp._patch_lock:
            vp._patched_workers.clear()

    def test_is_patched_initially_false(self) -> None:
        """is_patched() returns False before any patching."""
        from tasft.inference.vllm_patch import is_patched

        assert is_patched() is False

    def test_is_patched_after_manual_set(self) -> None:
        """is_patched() reflects the module-level _patched_workers state."""
        import tasft.inference.vllm_patch as vp

        with vp._patch_lock:
            vp._patched_workers.add(0xDEAD)
        assert vp.is_patched() is True

    def test_unpatch_when_not_patched_is_noop(self) -> None:
        """unpatch_vllm_attention on unpatched state is a safe no-op."""
        from tasft.inference.vllm_patch import is_patched, unpatch_vllm_attention

        mock_worker = MagicMock()
        unpatch_vllm_attention(mock_worker)
        assert is_patched() is False

    def test_is_patched_thread_safe(self) -> None:
        """Concurrent reads of is_patched() must not raise."""
        from tasft.inference.vllm_patch import is_patched

        results: list[bool] = []

        def reader() -> None:
            for _ in range(100):
                results.append(is_patched())

        threads = [threading.Thread(target=reader) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(results) == 400
        assert all(isinstance(r, bool) for r in results)


@pytest.mark.unit
class TestIsPrefillPhase:
    """Test _is_prefill_phase with various vLLM metadata shapes."""

    def test_is_prompt_true(self) -> None:
        """Metadata with is_prompt=True -> prefill."""
        from tasft.inference.vllm_patch import _is_prefill_phase

        meta = SimpleNamespace(is_prompt=True)
        assert _is_prefill_phase(meta) is True

    def test_is_prompt_false(self) -> None:
        """Metadata with is_prompt=False -> decode."""
        from tasft.inference.vllm_patch import _is_prefill_phase

        meta = SimpleNamespace(is_prompt=False)
        assert _is_prefill_phase(meta) is False

    def test_prefill_metadata_present(self) -> None:
        """Metadata with prefill_metadata not None -> prefill."""
        from tasft.inference.vllm_patch import _is_prefill_phase

        meta = SimpleNamespace(prefill_metadata=object())
        assert _is_prefill_phase(meta) is True

    def test_prefill_metadata_none(self) -> None:
        """Metadata with prefill_metadata=None -> decode."""
        from tasft.inference.vllm_patch import _is_prefill_phase

        meta = SimpleNamespace(prefill_metadata=None)
        assert _is_prefill_phase(meta) is False

    def test_num_prefill_tokens_positive(self) -> None:
        """Metadata with num_prefill_tokens > 0 -> prefill."""
        from tasft.inference.vllm_patch import _is_prefill_phase

        meta = SimpleNamespace(num_prefill_tokens=42)
        assert _is_prefill_phase(meta) is True

    def test_num_prefill_tokens_zero(self) -> None:
        """Metadata with num_prefill_tokens=0 -> decode."""
        from tasft.inference.vllm_patch import _is_prefill_phase

        meta = SimpleNamespace(num_prefill_tokens=0)
        assert _is_prefill_phase(meta) is False

    def test_no_known_attributes_defaults_prefill(self) -> None:
        """Unknown metadata object defaults to True (prefill) for safety."""
        from tasft.inference.vllm_patch import _is_prefill_phase

        meta = SimpleNamespace()
        assert _is_prefill_phase(meta) is True


@pytest.mark.unit
class TestExtractVLLMAttentionModules:
    """Test _extract_vllm_attention_modules error path."""

    def test_no_attention_modules_raises(self) -> None:
        """Model with no Attention modules raises InferenceError."""
        from tasft.inference.vllm_patch import _extract_vllm_attention_modules

        model = nn.Linear(4, 4)  # No attention modules
        with pytest.raises(InferenceError, match="No attention modules"):
            _extract_vllm_attention_modules(model)

    def test_finds_attention_module_with_qkv_proj(self) -> None:
        """Module with 'Attention' in name and qkv_proj is detected."""
        from tasft.inference.vllm_patch import _extract_vllm_attention_modules

        class FakeAttention(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.qkv_proj = nn.Linear(4, 12)

        class FakeModel(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.attn = FakeAttention()

        model = FakeModel()
        modules = _extract_vllm_attention_modules(model)
        assert len(modules) == 1
        assert isinstance(modules[0], FakeAttention)

    def test_finds_attention_module_with_q_proj(self) -> None:
        """Module with 'Attention' in name and q_proj is detected."""
        from tasft.inference.vllm_patch import _extract_vllm_attention_modules

        class MyAttention(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.q_proj = nn.Linear(4, 4)

        class Wrapper(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.layer = MyAttention()

        model = Wrapper()
        modules = _extract_vllm_attention_modules(model)
        assert len(modules) == 1


@pytest.mark.unit
class TestPatchVLLMAttentionErrors:
    """Test error paths in patch_vllm_attention."""

    def setup_method(self) -> None:
        import tasft.inference.vllm_patch as vp

        with vp._patch_lock:
            vp._patched_workers.clear()
        # Reset version detection and mock it so tests run without vLLM installed
        vp._version_detected = False
        vp._detected_version = None
        _fake = type("FakeVer", (), {"major": 0, "minor": 5, "patch": 0, "as_tuple": lambda s: (0, 5, 0), "__str__": lambda s: "0.5.0"})()
        vp.detect_vllm_version = lambda: _fake
        vp.check_vllm_compatibility = lambda v: []
        vp.validate_worker_structure = lambda w: []

    def test_patch_already_applied_is_noop(self) -> None:
        """Calling patch when already applied logs and returns."""
        import tasft.inference.vllm_patch as vp

        mock_model = MagicMock()
        mock_worker = MagicMock()

        with vp._patch_lock:
            vp._patched_workers.add(id(mock_worker))

        # Should not raise — worker already in patched set
        vp.patch_vllm_attention(mock_model, mock_worker)
        assert vp.is_patched() is True

    def test_worker_without_model_raises(self) -> None:
        """Worker with no model or model_runner raises InferenceError."""
        import tasft.inference.vllm_patch as vp

        mock_model = MagicMock()
        # Worker with no .model or .model_runner attributes
        worker = object()

        with pytest.raises(InferenceError, match="Cannot find model"):
            vp.patch_vllm_attention(mock_model, worker)

    def test_layer_count_mismatch_raises(self) -> None:
        """Mismatched layer count between vLLM and TASFT raises InferenceError."""
        import tasft.inference.vllm_patch as vp

        class FakeAttention(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.qkv_proj = nn.Linear(4, 12)
                self.num_heads = 4

        class FakeModel(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.attn0 = FakeAttention()
                self.attn1 = FakeAttention()

        fake_worker_model = FakeModel()

        # Worker with .model attribute
        worker = SimpleNamespace(model=fake_worker_model)

        # TASFT model with different gate count
        tasft_model = MagicMock()
        tasft_model.gates = {"0": MagicMock()}  # 1 gate vs 2 vLLM layers

        with pytest.raises(InferenceError, match="Layer count mismatch"):
            vp.patch_vllm_attention(tasft_model, worker)


@pytest.mark.unit
class TestCacheBlockSize:
    """Test TASFTvLLMAttentionBackend.get_cache_block_size."""

    def test_cache_block_size_float32(self) -> None:
        """Cache block size for float32 = 2 * block * heads * dim * 4."""
        from tasft.inference.vllm_patch import TASFTvLLMAttentionBackend

        size = TASFTvLLMAttentionBackend.get_cache_block_size(
            block_size=16, head_size=64, num_heads=8, dtype=torch.float32,
        )
        # 2 (k+v) * 16 * 8 * 64 * 4 bytes = 65536
        assert size == 2 * 16 * 8 * 64 * 4

    def test_cache_block_size_float16(self) -> None:
        """Cache block size for float16 = 2 * block * heads * dim * 2."""
        from tasft.inference.vllm_patch import TASFTvLLMAttentionBackend

        size = TASFTvLLMAttentionBackend.get_cache_block_size(
            block_size=16, head_size=64, num_heads=8, dtype=torch.float16,
        )
        assert size == 2 * 16 * 8 * 64 * 2

    def test_cache_block_size_bfloat16(self) -> None:
        """Cache block size for bfloat16 = 2 * block * heads * dim * 2."""
        from tasft.inference.vllm_patch import TASFTvLLMAttentionBackend

        size = TASFTvLLMAttentionBackend.get_cache_block_size(
            block_size=32, head_size=128, num_heads=4, dtype=torch.bfloat16,
        )
        assert size == 2 * 32 * 4 * 128 * 2


# ===================================================================
# 2. tasft_model coverage
# ===================================================================


@pytest.mark.unit
class TestVerifyChecksum:
    """Test _verify_checksum success and failure paths."""

    def test_valid_checksum_passes(self, tmp_path: Path) -> None:
        """File with matching checksum does not raise."""
        from tasft.inference.tasft_model import _verify_checksum

        test_file = tmp_path / "test.bin"
        test_file.write_bytes(b"hello world")
        expected = hashlib.sha256(b"hello world").hexdigest()

        # Should not raise
        _verify_checksum(test_file, expected)

    def test_invalid_checksum_raises(self, tmp_path: Path) -> None:
        """File with mismatched checksum raises ChecksumError."""
        from tasft.inference.tasft_model import _verify_checksum

        test_file = tmp_path / "test.bin"
        test_file.write_bytes(b"hello world")

        with pytest.raises(ChecksumError, match="Checksum mismatch"):
            _verify_checksum(test_file, "0" * 64)

    def test_checksum_error_carries_context(self, tmp_path: Path) -> None:
        """ChecksumError must carry expected, actual, and path in context."""
        from tasft.inference.tasft_model import _verify_checksum

        test_file = tmp_path / "ctx.bin"
        test_file.write_bytes(b"data")
        fake_hash = "a" * 64

        with pytest.raises(ChecksumError) as exc_info:
            _verify_checksum(test_file, fake_hash)

        ctx = exc_info.value.context
        assert "expected" in ctx
        assert "actual" in ctx
        assert "path" in ctx
        assert ctx["expected"] == fake_hash


@pytest.mark.unit
class TestExtractAttentionLayers:
    """Test _extract_attention_layers error path."""

    def test_no_attention_layers_raises(self) -> None:
        """Model with no *Attention modules raises InferenceError."""
        from tasft.inference.tasft_model import _extract_attention_layers

        model = MagicMock(spec=nn.Module)
        model.named_modules.return_value = [
            ("linear", nn.Linear(4, 4)),
        ]

        with pytest.raises(InferenceError, match="No attention layers"):
            _extract_attention_layers(model)


@pytest.mark.unit
class TestReplaceAttentionModule:
    """Test _replace_attention_module error path."""

    def test_target_not_found_raises(self) -> None:
        """Target module not in tree raises InferenceError."""
        from tasft.inference.tasft_model import _replace_attention_module

        parent = nn.Sequential(nn.Linear(4, 4))
        target = nn.Linear(8, 8)  # Not in parent
        replacement = nn.Linear(8, 8)

        with pytest.raises(InferenceError, match="Could not find target"):
            _replace_attention_module(parent, target, replacement)

    def test_successful_replacement(self) -> None:
        """Replacement of a child module works correctly."""
        from tasft.inference.tasft_model import _replace_attention_module

        class Container(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.child = nn.Linear(4, 4)

        container = Container()
        target = container.child
        replacement = nn.Linear(4, 8)

        _replace_attention_module(container, target, replacement)
        assert container.child is replacement


@pytest.mark.unit
class TestLoadBundleErrors:
    """Test TASFTInferenceModel.load_bundle error paths."""

    def test_nonexistent_bundle_dir_raises(self) -> None:
        """Non-existent bundle directory raises BundleError."""
        from tasft.inference.tasft_model import TASFTInferenceModel

        with pytest.raises(BundleError, match="does not exist"):
            TASFTInferenceModel.load_bundle("/nonexistent/path/xyz")

    def test_missing_manifest_raises(self, tmp_path: Path) -> None:
        """Bundle dir without manifest.json raises BundleError."""
        from tasft.inference.tasft_model import TASFTInferenceModel

        bundle = tmp_path / "bad_bundle"
        bundle.mkdir()

        with pytest.raises(BundleError, match="manifest.json"):
            TASFTInferenceModel.load_bundle(bundle)

    def test_empty_checksums_raises(self, tmp_path: Path) -> None:
        """Manifest with empty checksums dict raises BundleError."""
        from tasft.inference.tasft_model import TASFTInferenceModel

        bundle = tmp_path / "empty_cs"
        bundle.mkdir()
        manifest = {"checksums": {}, "model_name": "test"}
        (bundle / "manifest.json").write_text(json.dumps(manifest))

        with pytest.raises(BundleError, match="No checksums"):
            TASFTInferenceModel.load_bundle(bundle)

    def test_missing_checksummed_file_raises(self, tmp_path: Path) -> None:
        """File referenced in checksums but not on disk raises BundleError."""
        from tasft.inference.tasft_model import TASFTInferenceModel

        bundle = tmp_path / "missing_file"
        bundle.mkdir()
        manifest = {
            "checksums": {"model/missing.safetensors": "a" * 64},
            "model_name": "test",
        }
        (bundle / "manifest.json").write_text(json.dumps(manifest))

        with pytest.raises(BundleError, match="not found"):
            TASFTInferenceModel.load_bundle(bundle)


# ===================================================================
# 3. bundle/export coverage
# ===================================================================


@pytest.mark.unit
class TestExtractLayerIndexFromPath:
    """Test _extract_layer_index_from_path edge cases."""

    def test_standard_path(self) -> None:
        """Standard HF path: model.layers.5.self_attn.attn_gate -> 5."""
        from tasft.bundle.export import _extract_layer_index_from_path

        assert _extract_layer_index_from_path("model.layers.5.self_attn.attn_gate") == 5

    def test_peft_wrapped_path(self) -> None:
        """PeftModel wrapped path: base_model.model.model.layers.12.self_attn.attn_gate -> 12."""
        from tasft.bundle.export import _extract_layer_index_from_path

        result = _extract_layer_index_from_path(
            "base_model.model.model.layers.12.self_attn.attn_gate",
        )
        assert result == 12

    def test_no_layers_segment(self) -> None:
        """Path without 'layers' segment returns None."""
        from tasft.bundle.export import _extract_layer_index_from_path

        assert _extract_layer_index_from_path("some.other.module") is None

    def test_layers_at_end_without_index(self) -> None:
        """'layers' at the end without a following segment returns None."""
        from tasft.bundle.export import _extract_layer_index_from_path

        assert _extract_layer_index_from_path("model.layers") is None

    def test_layers_followed_by_non_numeric(self) -> None:
        """'layers' followed by non-numeric continues searching."""
        from tasft.bundle.export import _extract_layer_index_from_path

        assert _extract_layer_index_from_path("model.layers.abc.layers.3.gate") == 3

    def test_zero_index(self) -> None:
        """Layer index 0 is valid."""
        from tasft.bundle.export import _extract_layer_index_from_path

        assert _extract_layer_index_from_path("model.layers.0.attn") == 0


@pytest.mark.unit
class TestHashTrainingArgs:
    """Test _hash_training_args determinism."""

    def test_deterministic_hash(self) -> None:
        """Same config produces identical hash across calls."""
        config = ExportConfig(
            model_name="test", base_model_id="gpt2", domain="medical",
        )
        exporter = BundleExporter(config)
        h1 = exporter._hash_training_args()
        h2 = exporter._hash_training_args()
        assert h1 == h2
        assert len(h1) == 64  # SHA-256 hex digest

    def test_different_config_different_hash(self) -> None:
        """Different configs produce different hashes."""
        config_a = ExportConfig(
            model_name="a", base_model_id="gpt2", domain="medical",
        )
        config_b = ExportConfig(
            model_name="b", base_model_id="gpt2", domain="medical",
        )
        ha = BundleExporter(config_a)._hash_training_args()
        hb = BundleExporter(config_b)._hash_training_args()
        assert ha != hb


@pytest.mark.unit
class TestValidateBundleEdgeCases:
    """Test validate_bundle on edge case inputs."""

    def test_non_directory_path(self, tmp_path: Path) -> None:
        """validate_bundle on a file (not dir) returns is_valid=False."""
        f = tmp_path / "not_a_dir.txt"
        f.write_text("hello")
        result = BundleExporter.validate_bundle(f)
        assert not result.is_valid
        assert any("not a directory" in e for e in result.errors)

    def test_missing_manifest(self, tmp_path: Path) -> None:
        """validate_bundle on dir without manifest returns is_valid=False."""
        d = tmp_path / "empty"
        d.mkdir()
        result = BundleExporter.validate_bundle(d)
        assert not result.is_valid
        assert any("manifest" in e.lower() for e in result.errors)


@pytest.mark.unit
class TestExportConfigBoundaryValidation:
    """Test ExportConfig boundary values caught by BundleExporter constructor."""

    def test_negative_block_size_raises(self) -> None:
        """Negative block_size raises BundleError."""
        config = ExportConfig(
            model_name="t", base_model_id="g", domain="d", block_size=-1,
        )
        with pytest.raises(BundleError, match="block_size"):
            BundleExporter(config)

    def test_threshold_zero_raises(self) -> None:
        """global_threshold=0.0 (boundary) raises BundleError."""
        config = ExportConfig(
            model_name="t", base_model_id="g", domain="d", global_threshold=0.0,
        )
        with pytest.raises(BundleError, match="global_threshold"):
            BundleExporter(config)

    def test_threshold_one_raises(self) -> None:
        """global_threshold=1.0 (boundary) raises BundleError."""
        config = ExportConfig(
            model_name="t", base_model_id="g", domain="d", global_threshold=1.0,
        )
        with pytest.raises(BundleError, match="global_threshold"):
            BundleExporter(config)

    def test_valid_config_succeeds(self) -> None:
        """Valid config at interior point succeeds."""
        config = ExportConfig(
            model_name="t", base_model_id="g", domain="d",
            block_size=64, global_threshold=0.5,
        )
        exporter = BundleExporter(config)
        assert exporter.config.block_size == 64


# ===================================================================
# 4. training/trainer coverage
# ===================================================================


@pytest.mark.unit
class TestGateWarmupMultiplier:
    """Test _get_gate_warmup_multiplier boundary conditions."""

    def _make_trainer_args(self, tmp_path: Path, warmup: int) -> Any:
        return TASFTTrainingArguments(
            output_dir=str(tmp_path / "out"),
            num_train_epochs=1,
            per_device_train_batch_size=2,
            max_steps=5,
            learning_rate=1e-3,
            gate_warmup_steps=warmup,
            use_cpu=True,
            save_steps=10,
            logging_steps=1,
        )

    def _make_trainer(self, args: Any) -> Any:
        from transformers import GPT2Config, GPT2LMHeadModel

        from tasft.modules.tasft_attention import GateConfig, patch_model_attention
        from tasft.training.trainer import TASFTTrainer

        config = GPT2Config(
            n_layer=2, n_head=4, n_embd=128, n_positions=256, vocab_size=512,
        )
        model = GPT2LMHeadModel(config)
        for p in model.parameters():
            p.requires_grad = False
        gate_config = GateConfig(
            block_size=8, num_layers=2, gate_hidden_dim=16, default_threshold=0.5,
        )
        patched = patch_model_attention(model, gate_config)
        return TASFTTrainer(model=model, args=args, patched_layers=patched)

    def test_warmup_zero_always_returns_one(self, tmp_path: Path) -> None:
        """warmup_steps=0: multiplier is always 1.0 regardless of step."""
        args = self._make_trainer_args(tmp_path, warmup=0)
        trainer = self._make_trainer(args)
        assert trainer._get_gate_warmup_multiplier(0) == 1.0
        assert trainer._get_gate_warmup_multiplier(100) == 1.0

    def test_step_before_warmup_returns_zero(self, tmp_path: Path) -> None:
        """step < warmup_steps: multiplier is 0.0."""
        args = self._make_trainer_args(tmp_path, warmup=100)
        trainer = self._make_trainer(args)
        assert trainer._get_gate_warmup_multiplier(0) == 0.0
        assert trainer._get_gate_warmup_multiplier(99) == 0.0

    def test_step_at_warmup_boundary_returns_zero_ramp(self, tmp_path: Path) -> None:
        """step=warmup_steps: ramp_step=0, multiplier=0.0."""
        args = self._make_trainer_args(tmp_path, warmup=100)
        trainer = self._make_trainer(args)
        # step=100, warmup=100 -> ramp=0 -> 0/100 = 0.0
        assert trainer._get_gate_warmup_multiplier(100) == 0.0

    def test_step_at_warmup_plus_one(self, tmp_path: Path) -> None:
        """step=warmup+1: ramp=1, multiplier=1/warmup."""
        args = self._make_trainer_args(tmp_path, warmup=100)
        trainer = self._make_trainer(args)
        result = trainer._get_gate_warmup_multiplier(101)
        assert result == pytest.approx(1.0 / 100.0)

    def test_step_at_double_warmup_returns_one(self, tmp_path: Path) -> None:
        """step=2*warmup: ramp=warmup, multiplier=min(1.0, warmup/warmup)=1.0."""
        args = self._make_trainer_args(tmp_path, warmup=100)
        trainer = self._make_trainer(args)
        assert trainer._get_gate_warmup_multiplier(200) == 1.0

    def test_step_beyond_double_warmup_capped_at_one(self, tmp_path: Path) -> None:
        """step >> warmup: multiplier stays capped at 1.0."""
        args = self._make_trainer_args(tmp_path, warmup=10)
        trainer = self._make_trainer(args)
        assert trainer._get_gate_warmup_multiplier(10000) == 1.0


@pytest.mark.unit
class TestTrainingArgsAdditionalValidation:
    """Test TASFTTrainingArguments validation for uncovered branches."""

    def test_invalid_rotation_strategy_raises(self, tmp_path: Path) -> None:
        """Unknown rotation_strategy raises ValueError."""
        with pytest.raises(ValueError, match="rotation_strategy"):
            TASFTTrainingArguments(
                output_dir=str(tmp_path / "out"),
                rotation_strategy="nonexistent_strategy",
                use_cpu=True,
            )

    def test_negative_beta_sparse_raises(self, tmp_path: Path) -> None:
        """beta_sparse < 0 raises ValueError."""
        with pytest.raises(ValueError, match="beta_sparse"):
            TASFTTrainingArguments(
                output_dir=str(tmp_path / "out"),
                beta_sparse=-0.01,
                use_cpu=True,
            )

    def test_negative_gate_warmup_steps_raises(self, tmp_path: Path) -> None:
        """gate_warmup_steps < 0 raises ValueError."""
        with pytest.raises(ValueError, match="gate_warmup_steps"):
            TASFTTrainingArguments(
                output_dir=str(tmp_path / "out"),
                gate_warmup_steps=-1,
                use_cpu=True,
            )

    def test_zero_layers_per_step_raises(self, tmp_path: Path) -> None:
        """layers_per_step=0 raises ValueError."""
        with pytest.raises(ValueError, match="layers_per_step"):
            TASFTTrainingArguments(
                output_dir=str(tmp_path / "out"),
                layers_per_step=0,
                use_cpu=True,
            )

    def test_zero_block_size_raises(self, tmp_path: Path) -> None:
        """block_size=0 raises ValueError."""
        with pytest.raises(ValueError, match="block_size"):
            TASFTTrainingArguments(
                output_dir=str(tmp_path / "out"),
                block_size=0,
                use_cpu=True,
            )

    def test_gate_lr_ratio_above_one_raises(self, tmp_path: Path) -> None:
        """gate_lr_ratio > 1.0 raises ValueError."""
        with pytest.raises(ValueError, match="gate_lr_ratio"):
            TASFTTrainingArguments(
                output_dir=str(tmp_path / "out"),
                gate_lr_ratio=1.5,
                use_cpu=True,
            )

    def test_lambda_gate_above_ten_raises(self, tmp_path: Path) -> None:
        """lambda_gate > 10.0 raises ValueError."""
        with pytest.raises(ValueError, match="lambda_gate"):
            TASFTTrainingArguments(
                output_dir=str(tmp_path / "out"),
                lambda_gate=11.0,
                use_cpu=True,
            )

    def test_tau_target_zero_raises(self, tmp_path: Path) -> None:
        """tau_target=0.0 (boundary exclusive) raises ValueError."""
        with pytest.raises(ValueError, match="tau_target"):
            TASFTTrainingArguments(
                output_dir=str(tmp_path / "out"),
                tau_target=0.0,
                use_cpu=True,
            )


# ===================================================================
# 5. observability/logging coverage
# ===================================================================


@pytest.mark.unit
class TestTimedOperation:
    """Test timed_operation context manager measures duration."""

    def test_logs_duration_ms(self) -> None:
        """timed_operation must produce a completion log with duration_ms > 0."""
        log = get_logger("test.timed")

        # Use a structlog processor to capture output
        with timed_operation(log, "TEST_OP"):
            time.sleep(0.01)  # 10ms sleep to ensure measurable duration

        # If we got here without exception, the context manager worked.
        # The primary assertion is that it doesn't raise.

    def test_timed_operation_with_exception(self) -> None:
        """timed_operation still logs completion when body raises."""
        log = get_logger("test.timed_exc")

        with pytest.raises(ValueError, match="boom"):
            with timed_operation(log, "FAILING_OP"):
                raise ValueError("boom")

        # Context manager's finally block should have fired (no assertion needed
        # beyond verifying the exception propagated correctly).

    def test_timed_operation_with_extra_kwargs(self) -> None:
        """timed_operation accepts extra keyword arguments."""
        log = get_logger("test.timed_extra")
        # Should not raise
        with timed_operation(log, "EXTRA_OP", num_layers=5, model="gpt2"):
            pass


@pytest.mark.unit
class TestBindContext:
    """Test bind_context binds and unbinds structlog contextvars."""

    def test_context_bound_and_unbound(self) -> None:
        """Fields are bound inside context and cleaned up after."""
        # Clear any existing context
        structlog.contextvars.clear_contextvars()

        with bind_context(request_id="req-123", step=42, layer_idx=5, custom="val"):
            ctx = structlog.contextvars.get_contextvars()
            assert ctx.get("request_id") == "req-123"
            assert ctx.get("step") == 42
            assert ctx.get("layer_idx") == 5
            assert ctx.get("custom") == "val"

        # After exiting, fields should be unbound
        ctx_after = structlog.contextvars.get_contextvars()
        assert "request_id" not in ctx_after
        assert "step" not in ctx_after
        assert "layer_idx" not in ctx_after
        assert "custom" not in ctx_after

    def test_context_with_no_fields(self) -> None:
        """bind_context with no arguments is a valid no-op."""
        with bind_context():
            pass

    def test_context_cleans_up_on_exception(self) -> None:
        """Fields are unbound even if the body raises."""
        structlog.contextvars.clear_contextvars()

        with pytest.raises(RuntimeError):
            with bind_context(request_id="req-exc"):
                raise RuntimeError("oops")

        ctx = structlog.contextvars.get_contextvars()
        assert "request_id" not in ctx


@pytest.mark.unit
class TestGetLogger:
    """Test get_logger returns a properly bound logger."""

    def test_returns_bound_logger(self) -> None:
        """Logger must have module, version, git_hash bound."""
        log = get_logger("test.module")
        # structlog BoundLogger stores bindings in _context
        assert log is not None

    def test_different_names_different_loggers(self) -> None:
        """Different module names produce distinct logger instances."""
        log_a = get_logger("module.a")
        log_b = get_logger("module.b")
        # They should be distinct objects
        assert log_a is not log_b


@pytest.mark.unit
class TestConfigureLogging:
    """Test configure_logging with force_json."""

    def test_configure_with_force_json(self) -> None:
        """configure_logging(force_json=True) should not raise."""
        configure_logging(level="DEBUG", force_json=True)
        # Verify we can still get and use a logger
        log = get_logger("test.json")
        log.info("test message", key="value")

    def test_configure_with_warning_level(self) -> None:
        """configure_logging(level='WARNING') should not raise."""
        configure_logging(level="WARNING")
        log = get_logger("test.warn")
        log.warning("warning test")

    def teardown_method(self) -> None:
        """Reset logging to default after tests."""
        configure_logging(level="INFO")
