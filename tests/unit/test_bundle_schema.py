"""Unit tests for bundle schema validation.

Covers:
1. BundleManifest JSON roundtrip (construct -> model_dump_json -> model_validate_json -> equal)
2. SHA256 checksum validation (valid hex passes, invalid rejects)
3. Frozen constraint (assignment raises ValidationError)
4. KernelConfig cross-field validator (mismatched layer_idx / block_size rejects)
5. Negative total_size_bytes / num_layers rejection
6. EvalSummary range validation (mean_sparsity, speedup_vs_dense, task_accuracy)
7. LayerKernelConfig boundary values
"""
from __future__ import annotations

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from tasft.bundle.bundle_schema import (
    BundleManifest,
    BundleMetadata,
    EvalSummary,
    KernelConfig,
    LayerKernelConfig,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_VALID_SHA256 = "a" * 64


def _make_layer_kernel_config(
    *,
    layer_idx: int = 0,
    threshold_tau: float = 0.5,
    target_sparsity: float = 0.7,
    achieved_sparsity_validation: float = 0.65,
    gate_loss_validation: float = 0.01,
    block_size: int = 64,
) -> LayerKernelConfig:
    return LayerKernelConfig(
        layer_idx=layer_idx,
        threshold_tau=threshold_tau,
        target_sparsity=target_sparsity,
        achieved_sparsity_validation=achieved_sparsity_validation,
        gate_loss_validation=gate_loss_validation,
        block_size=block_size,
    )


def _make_kernel_config(
    *,
    block_size: int = 64,
    global_threshold: float = 0.5,
    per_layer_config: dict[int, LayerKernelConfig] | None = None,
) -> KernelConfig:
    if per_layer_config is None:
        lkc = _make_layer_kernel_config(layer_idx=0, block_size=block_size)
        per_layer_config = {0: lkc}
    return KernelConfig(
        block_size=block_size,
        global_threshold=global_threshold,
        per_layer_config=per_layer_config,
    )


def _make_bundle_manifest(
    *,
    total_size_bytes: int = 1024,
    num_layers: int = 12,
    checksums: dict[str, str] | None = None,
) -> BundleManifest:
    if checksums is None:
        checksums = {"weights.safetensors": _VALID_SHA256}
    return BundleManifest(
        model_name="test-model",
        base_model_id="meta-llama/Llama-3-8B",
        domain="code",
        created_at=datetime(2025, 1, 1, tzinfo=UTC),
        git_hash="abc1234",
        training_args_hash=_VALID_SHA256,
        checksums=checksums,
        total_size_bytes=total_size_bytes,
        num_layers=num_layers,
    )


def _make_eval_summary(
    *,
    task_accuracy: float = 0.85,
    task_accuracy_baseline: float = 0.80,
    delta_accuracy: float = 0.05,
    mean_tokens_per_second: float = 120.0,
    speedup_vs_dense: float = 2.5,
    mean_sparsity: float = 0.7,
) -> EvalSummary:
    return EvalSummary(
        task_accuracy=task_accuracy,
        task_accuracy_baseline=task_accuracy_baseline,
        delta_accuracy=delta_accuracy,
        mean_tokens_per_second=mean_tokens_per_second,
        speedup_vs_dense=speedup_vs_dense,
        mean_sparsity=mean_sparsity,
        eval_domain="code",
    )


# ---------------------------------------------------------------------------
# BundleManifest Tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestBundleManifestRoundtrip:
    """BundleManifest JSON serialization roundtrip."""

    def test_roundtrip_preserves_equality(self) -> None:
        manifest = _make_bundle_manifest()
        json_str = manifest.model_dump_json()
        restored = BundleManifest.model_validate_json(json_str)
        assert manifest == restored

    def test_roundtrip_with_multiple_checksums(self) -> None:
        checksums = {
            "weights.safetensors": "a" * 64,
            "gates.safetensors": "b" * 64,
            "config.json": "c" * 64,
        }
        manifest = _make_bundle_manifest(checksums=checksums)
        json_str = manifest.model_dump_json()
        restored = BundleManifest.model_validate_json(json_str)
        assert restored.checksums == checksums


@pytest.mark.unit
class TestBundleManifestChecksums:
    """SHA256 checksum validation in BundleManifest."""

    def test_valid_sha256_hex_accepted(self) -> None:
        # 64 lowercase hex chars -- must not raise
        manifest = _make_bundle_manifest(checksums={"f.bin": "0123456789abcdef" * 4})
        assert len(manifest.checksums["f.bin"]) == 64

    def test_invalid_sha256_too_short_raises(self) -> None:
        with pytest.raises(ValidationError, match="Invalid SHA256 checksum"):
            _make_bundle_manifest(checksums={"f.bin": "aaa"})

    def test_invalid_sha256_uppercase_raises(self) -> None:
        with pytest.raises(ValidationError, match="Invalid SHA256 checksum"):
            _make_bundle_manifest(checksums={"f.bin": "A" * 64})

    def test_invalid_sha256_non_hex_raises(self) -> None:
        with pytest.raises(ValidationError, match="Invalid SHA256 checksum"):
            _make_bundle_manifest(checksums={"f.bin": "g" * 64})

    def test_empty_checksums_accepted(self) -> None:
        manifest = _make_bundle_manifest(checksums={})
        assert manifest.checksums == {}


@pytest.mark.unit
class TestBundleManifestFieldBounds:
    """Negative total_size_bytes and num_layers must be rejected."""

    def test_negative_total_size_bytes_raises(self) -> None:
        with pytest.raises(ValidationError, match="total_size_bytes"):
            _make_bundle_manifest(total_size_bytes=-1)

    def test_zero_total_size_bytes_accepted(self) -> None:
        manifest = _make_bundle_manifest(total_size_bytes=0)
        assert manifest.total_size_bytes == 0

    def test_negative_num_layers_raises(self) -> None:
        with pytest.raises(ValidationError, match="num_layers"):
            _make_bundle_manifest(num_layers=-1)

    def test_zero_num_layers_accepted(self) -> None:
        manifest = _make_bundle_manifest(num_layers=0)
        assert manifest.num_layers == 0


# ---------------------------------------------------------------------------
# Frozen Constraint Tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestFrozenConstraint:
    """All schema models are frozen -- mutation must raise."""

    def test_bundle_manifest_frozen(self) -> None:
        manifest = _make_bundle_manifest()
        with pytest.raises(ValidationError, match="frozen"):
            manifest.model_name = "changed"  # type: ignore[misc]

    def test_layer_kernel_config_frozen(self) -> None:
        lkc = _make_layer_kernel_config()
        with pytest.raises(ValidationError, match="frozen"):
            lkc.layer_idx = 99  # type: ignore[misc]

    def test_kernel_config_frozen(self) -> None:
        kc = _make_kernel_config()
        with pytest.raises(ValidationError, match="frozen"):
            kc.block_size = 128  # type: ignore[misc]

    def test_eval_summary_frozen(self) -> None:
        es = _make_eval_summary()
        with pytest.raises(ValidationError, match="frozen"):
            es.mean_sparsity = 0.99  # type: ignore[misc]


# ---------------------------------------------------------------------------
# KernelConfig Cross-field Validator
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestKernelConfigCrossField:
    """KernelConfig model_validator enforces layer_idx key match and block_size consistency."""

    def test_mismatched_layer_idx_key_raises(self) -> None:
        lkc = _make_layer_kernel_config(layer_idx=0)
        with pytest.raises(ValidationError, match="doesn't match"):
            KernelConfig(
                block_size=64,
                global_threshold=0.5,
                per_layer_config={5: lkc},  # key=5 but cfg.layer_idx=0
            )

    def test_mismatched_block_size_raises(self) -> None:
        lkc = _make_layer_kernel_config(layer_idx=0, block_size=32)
        with pytest.raises(ValidationError, match="block_size"):
            KernelConfig(
                block_size=64,
                global_threshold=0.5,
                per_layer_config={0: lkc},
            )

    def test_consistent_config_accepted(self) -> None:
        lkc = _make_layer_kernel_config(layer_idx=3, block_size=128)
        kc = KernelConfig(
            block_size=128,
            global_threshold=0.5,
            per_layer_config={3: lkc},
        )
        assert kc.per_layer_config[3].layer_idx == 3

    def test_extra_fields_rejected(self) -> None:
        with pytest.raises(ValidationError, match="extra"):
            KernelConfig(
                block_size=64,
                global_threshold=0.5,
                per_layer_config={},
                unknown_field="bad",  # type: ignore[call-arg]
            )


# ---------------------------------------------------------------------------
# EvalSummary Range Validation
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestEvalSummaryRanges:
    """EvalSummary field bounds enforcement."""

    def test_mean_sparsity_above_one_raises(self) -> None:
        with pytest.raises(ValidationError, match="mean_sparsity"):
            _make_eval_summary(mean_sparsity=1.1)

    def test_mean_sparsity_negative_raises(self) -> None:
        with pytest.raises(ValidationError, match="mean_sparsity"):
            _make_eval_summary(mean_sparsity=-0.01)

    def test_mean_sparsity_zero_accepted(self) -> None:
        es = _make_eval_summary(mean_sparsity=0.0)
        assert es.mean_sparsity == 0.0

    def test_mean_sparsity_one_accepted(self) -> None:
        es = _make_eval_summary(mean_sparsity=1.0)
        assert es.mean_sparsity == 1.0

    def test_speedup_vs_dense_zero_raises(self) -> None:
        with pytest.raises(ValidationError, match="speedup_vs_dense"):
            _make_eval_summary(speedup_vs_dense=0.0)

    def test_speedup_vs_dense_negative_raises(self) -> None:
        with pytest.raises(ValidationError, match="speedup_vs_dense"):
            _make_eval_summary(speedup_vs_dense=-1.0)

    def test_speedup_vs_dense_positive_accepted(self) -> None:
        es = _make_eval_summary(speedup_vs_dense=0.001)
        assert es.speedup_vs_dense == pytest.approx(0.001)

    def test_task_accuracy_above_one_raises(self) -> None:
        with pytest.raises(ValidationError, match="task_accuracy"):
            _make_eval_summary(task_accuracy=1.01)

    def test_task_accuracy_negative_raises(self) -> None:
        with pytest.raises(ValidationError, match="task_accuracy"):
            _make_eval_summary(task_accuracy=-0.5)

    def test_task_accuracy_baseline_above_one_raises(self) -> None:
        with pytest.raises(ValidationError, match="task_accuracy_baseline"):
            _make_eval_summary(task_accuracy_baseline=1.5)

    def test_task_accuracy_baseline_negative_raises(self) -> None:
        with pytest.raises(ValidationError, match="task_accuracy_baseline"):
            _make_eval_summary(task_accuracy_baseline=-0.1)


# ---------------------------------------------------------------------------
# LayerKernelConfig Boundary Values
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestLayerKernelConfigBoundaries:
    """LayerKernelConfig field boundary enforcement."""

    def test_layer_idx_zero_accepted(self) -> None:
        lkc = _make_layer_kernel_config(layer_idx=0)
        assert lkc.layer_idx == 0

    def test_layer_idx_256_accepted(self) -> None:
        lkc = _make_layer_kernel_config(layer_idx=256)
        assert lkc.layer_idx == 256

    def test_layer_idx_negative_raises(self) -> None:
        with pytest.raises(ValidationError, match="layer_idx"):
            _make_layer_kernel_config(layer_idx=-1)

    def test_layer_idx_257_raises(self) -> None:
        with pytest.raises(ValidationError, match="layer_idx"):
            _make_layer_kernel_config(layer_idx=257)

    def test_threshold_tau_zero_raises(self) -> None:
        with pytest.raises(ValidationError, match="threshold_tau"):
            _make_layer_kernel_config(threshold_tau=0.0)

    def test_threshold_tau_one_raises(self) -> None:
        with pytest.raises(ValidationError, match="threshold_tau"):
            _make_layer_kernel_config(threshold_tau=1.0)

    def test_threshold_tau_midpoint_accepted(self) -> None:
        lkc = _make_layer_kernel_config(threshold_tau=0.5)
        assert lkc.threshold_tau == 0.5

    def test_block_size_zero_raises(self) -> None:
        with pytest.raises(ValidationError, match="block_size"):
            _make_layer_kernel_config(block_size=0)

    def test_block_size_512_accepted(self) -> None:
        lkc = _make_layer_kernel_config(block_size=512)
        assert lkc.block_size == 512

    def test_block_size_513_raises(self) -> None:
        with pytest.raises(ValidationError, match="block_size"):
            _make_layer_kernel_config(block_size=513)

    def test_target_sparsity_boundaries(self) -> None:
        lkc_zero = _make_layer_kernel_config(target_sparsity=0.0)
        assert lkc_zero.target_sparsity == 0.0
        lkc_one = _make_layer_kernel_config(target_sparsity=1.0)
        assert lkc_one.target_sparsity == 1.0

    def test_target_sparsity_out_of_range_raises(self) -> None:
        with pytest.raises(ValidationError, match="target_sparsity"):
            _make_layer_kernel_config(target_sparsity=1.01)
        with pytest.raises(ValidationError, match="target_sparsity"):
            _make_layer_kernel_config(target_sparsity=-0.01)


# ---------------------------------------------------------------------------
# BundleMetadata Composite Test
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestBundleMetadata:
    """BundleMetadata composes all sub-schemas correctly."""

    def test_full_roundtrip(self) -> None:
        metadata = BundleMetadata(
            manifest=_make_bundle_manifest(),
            kernel_config=_make_kernel_config(),
            eval_summary=_make_eval_summary(),
        )
        json_str = metadata.model_dump_json()
        restored = BundleMetadata.model_validate_json(json_str)
        assert metadata == restored

    def test_eval_summary_optional(self) -> None:
        metadata = BundleMetadata(
            manifest=_make_bundle_manifest(),
            kernel_config=_make_kernel_config(),
        )
        assert metadata.eval_summary is None
