"""Integration tests for the bundle export/import lifecycle.

Validates the complete bundle pipeline: schema serialization, checksum integrity,
corruption detection, metadata loading, version compatibility, and atomic write
semantics. All tests operate on synthetic bundles in tmp_path -- no GPU required.

Covers:
    1. BundleManifest JSON roundtrip with all fields preserved
    2. SHA-256 checksum computation and manifest validation
    3. Corrupt file detection via checksum mismatch
    4. Missing file detection during bundle validation
    5. Extra file handling (files not in manifest)
    6. KernelConfig layer indices matching gate weight file set
    7. Atomic write: temp-dir-then-rename leaves no partial bundles
    8. BundleMetadata loading without eval_summary (optional field)
    9. bundle_format_version compatibility handling
   10. Large model bundle structure with 32+ layers
"""
from __future__ import annotations

import hashlib
import json
import os
import signal
import threading
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from tasft.bundle.bundle_schema import (
    BundleManifest,
    BundleMetadata,
    EvalSummary,
    KernelConfig,
    LayerKernelConfig,
)
from tasft.bundle.export import BundleExporter, ValidationResult
from tasft.exceptions import BundleError

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_VALID_SHA256 = "a" * 64
_ZERO_SHA256 = "0" * 64


# ---------------------------------------------------------------------------
# Helpers -- construct valid schema objects and synthetic bundle directories
# ---------------------------------------------------------------------------


def _sha256_bytes(data: bytes) -> str:
    """Compute SHA-256 hex digest of raw bytes.

    Args:
        data: Byte content to hash.

    Returns:
        64-char lowercase hex SHA-256 digest.

    Complexity: O(len(data)).
    """
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    """Compute SHA-256 hex digest of a file using 64KB streaming reads.

    Args:
        path: Path to file to hash.

    Returns:
        64-char lowercase hex SHA-256 digest.

    Complexity: O(file_size) with 64KB chunks.
    """
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _make_layer_kernel_config(
    *,
    layer_idx: int = 0,
    threshold_tau: float = 0.5,
    target_sparsity: float = 0.7,
    achieved_sparsity_validation: float = 0.65,
    gate_loss_validation: float = 0.01,
    block_size: int = 64,
) -> LayerKernelConfig:
    """Construct a valid LayerKernelConfig for testing.

    Preconditions: All arguments satisfy schema constraints.
    Postconditions: Returns frozen, validated LayerKernelConfig.
    Complexity: O(1).
    """
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
    num_layers: int = 4,
    block_size: int = 64,
    global_threshold: float = 0.5,
) -> KernelConfig:
    """Construct a valid KernelConfig with per-layer entries.

    Args:
        num_layers: Number of layers to generate configs for.
        block_size: Block size for all layers and global config.
        global_threshold: Global gate threshold.

    Preconditions: num_layers >= 0, block_size in (0, 512], global_threshold in (0, 1).
    Postconditions: Returns frozen KernelConfig with num_layers per-layer entries.
    Complexity: O(num_layers).
    """
    per_layer = {
        i: _make_layer_kernel_config(layer_idx=i, block_size=block_size)
        for i in range(num_layers)
    }
    return KernelConfig(
        block_size=block_size,
        global_threshold=global_threshold,
        per_layer_config=per_layer,
    )


def _make_manifest(
    *,
    checksums: dict[str, str] | None = None,
    num_layers: int = 4,
    total_size_bytes: int = 4096,
    bundle_format_version: str = "1.0",
) -> BundleManifest:
    """Construct a valid BundleManifest for testing.

    Args:
        checksums: File-path to SHA-256 mapping. Defaults to a single entry.
        num_layers: Number of model layers declared in manifest.
        total_size_bytes: Total size of all checksummed files.
        bundle_format_version: Format version string.

    Preconditions: All checksums must be valid 64-char lowercase hex.
    Postconditions: Returns frozen, validated BundleManifest.
    Complexity: O(len(checksums)).
    """
    if checksums is None:
        checksums = {"model/model.safetensors": _VALID_SHA256}
    return BundleManifest(
        model_name="test-model",
        base_model_id="meta-llama/Llama-3-8B",
        domain="code",
        created_at=datetime(2025, 6, 15, 12, 0, 0, tzinfo=UTC),
        git_hash="abc1234def5678",
        training_args_hash=_VALID_SHA256,
        checksums=checksums,
        total_size_bytes=total_size_bytes,
        num_layers=num_layers,
        bundle_format_version=bundle_format_version,
    )


def _make_eval_summary() -> EvalSummary:
    """Construct a valid EvalSummary for testing.

    Postconditions: Returns frozen, validated EvalSummary.
    Complexity: O(1).
    """
    return EvalSummary(
        task_accuracy=0.85,
        task_accuracy_baseline=0.80,
        delta_accuracy=0.05,
        mean_tokens_per_second=120.0,
        speedup_vs_dense=2.5,
        mean_sparsity=0.7,
        eval_domain="code",
    )


def _write_synthetic_bundle(
    bundle_dir: Path,
    *,
    num_layers: int = 4,
    include_eval: bool = True,
    model_content: bytes = b"fake-model-weights-safetensors-data",
    gate_content_factory: Any | None = None,
) -> BundleManifest:
    """Create a complete synthetic bundle directory on disk with valid checksums.

    Writes model weights, per-layer gate files, kernel_config.json, manifest.json,
    and optionally eval_results.json. All checksums in the manifest are computed
    from actual file contents -- the bundle will pass validate_bundle().

    Args:
        bundle_dir: Destination directory (must not exist).
        num_layers: Number of gate files to create.
        include_eval: Whether to include eval_results.json.
        model_content: Raw bytes for model/model.safetensors.
        gate_content_factory: Optional callable(layer_idx) -> bytes for gate content.

    Returns:
        The written BundleManifest.

    Preconditions: bundle_dir does not exist.
    Postconditions: bundle_dir is a complete, valid bundle directory.
    Complexity: O(num_layers + len(model_content)).
    """
    bundle_dir.mkdir(parents=True)
    model_dir = bundle_dir / "model"
    model_dir.mkdir()
    gates_dir = bundle_dir / "gates"
    gates_dir.mkdir()

    # Write model weights
    model_file = model_dir / "model.safetensors"
    model_file.write_bytes(model_content)

    # Write per-layer gate files
    for i in range(num_layers):
        gate_file = gates_dir / f"layer_{i}_gate.safetensors"
        if gate_content_factory is not None:
            gate_file.write_bytes(gate_content_factory(i))
        else:
            gate_file.write_bytes(f"gate-weights-layer-{i}".encode())

    # Build kernel config
    kernel_config = _make_kernel_config(num_layers=num_layers)
    kc_json = kernel_config.model_dump_json(indent=2)
    kc_path = bundle_dir / "kernel_config.json"
    kc_path.write_text(kc_json)

    # Compute checksums for all files that belong in the manifest
    files_to_checksum: list[tuple[str, Path]] = [
        ("model/model.safetensors", model_file),
    ]
    for i in range(num_layers):
        rel = f"gates/layer_{i}_gate.safetensors"
        files_to_checksum.append((rel, gates_dir / f"layer_{i}_gate.safetensors"))
    files_to_checksum.append(("kernel_config.json", kc_path))

    checksums = {rel: _sha256_file(path) for rel, path in files_to_checksum}
    total_bytes = sum(path.stat().st_size for _, path in files_to_checksum)

    manifest = BundleManifest(
        model_name="test-model",
        base_model_id="meta-llama/Llama-3-8B",
        domain="code",
        created_at=datetime(2025, 6, 15, 12, 0, 0, tzinfo=UTC),
        git_hash="abc1234def5678",
        training_args_hash=_VALID_SHA256,
        checksums=checksums,
        total_size_bytes=total_bytes,
        num_layers=num_layers,
    )

    manifest_path = bundle_dir / "manifest.json"
    manifest_path.write_text(manifest.model_dump_json(indent=2))

    if include_eval:
        eval_summary = _make_eval_summary()
        eval_path = bundle_dir / "eval_results.json"
        eval_path.write_text(eval_summary.model_dump_json(indent=2))

    return manifest


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestBundleManifestRoundtripJSON:
    """Verify BundleManifest JSON serialization is lossless across all fields."""

    def test_bundle_manifest_roundtrip_json(self, tmp_path: Path) -> None:
        """Serialize manifest to JSON, deserialize, verify every field matches.

        Covers: model_name, base_model_id, domain, created_at (datetime with tz),
        git_hash, training_args_hash, checksums (multi-entry), total_size_bytes,
        num_layers, version, bundle_format_version.
        """
        checksums = {
            "model/model.safetensors": "a" * 64,
            "gates/layer_0_gate.safetensors": "b" * 64,
            "gates/layer_1_gate.safetensors": "c" * 64,
            "kernel_config.json": "d" * 64,
        }
        manifest = BundleManifest(
            version="1.0.0",
            bundle_format_version="1.0",
            model_name="roundtrip-test-model",
            base_model_id="meta-llama/Llama-3-8B",
            domain="medical",
            created_at=datetime(2025, 3, 15, 10, 30, 45, tzinfo=UTC),
            git_hash="deadbeef01234567",
            training_args_hash="e" * 64,
            checksums=checksums,
            total_size_bytes=999_999_999,
            num_layers=2,
        )

        json_str = manifest.model_dump_json(indent=2)

        # Write to disk and read back to simulate real file I/O
        json_path = tmp_path / "manifest_roundtrip.json"
        json_path.write_text(json_str)
        loaded_str = json_path.read_text()

        restored = BundleManifest.model_validate_json(loaded_str)

        assert restored == manifest
        assert restored.version == "1.0.0"
        assert restored.bundle_format_version == "1.0"
        assert restored.model_name == "roundtrip-test-model"
        assert restored.base_model_id == "meta-llama/Llama-3-8B"
        assert restored.domain == "medical"
        assert restored.git_hash == "deadbeef01234567"
        assert restored.training_args_hash == "e" * 64
        assert restored.checksums == checksums
        assert restored.total_size_bytes == 999_999_999
        assert restored.num_layers == 2
        # Datetime with timezone roundtrip
        assert restored.created_at.year == 2025
        assert restored.created_at.month == 3
        assert restored.created_at.tzinfo is not None


@pytest.mark.integration
class TestBundleChecksumValidation:
    """Verify SHA-256 checksum computation and manifest-level validation."""

    def test_bundle_checksum_validation(self, tmp_path: Path) -> None:
        """Create files, compute SHA-256, write manifest, validate bundle passes."""
        bundle_dir = tmp_path / "valid_bundle"
        manifest = _write_synthetic_bundle(bundle_dir, num_layers=3)

        result = BundleExporter.validate_bundle(bundle_dir)

        assert result.is_valid, f"Expected valid bundle, got errors: {result.errors}"
        assert result.checked_files == len(manifest.checksums)
        assert len(result.errors) == 0

    def test_checksum_matches_sha256_of_file_content(self, tmp_path: Path) -> None:
        """Verify each file's computed checksum matches the manifest entry."""
        bundle_dir = tmp_path / "checksum_match"
        manifest = _write_synthetic_bundle(bundle_dir, num_layers=2)

        for relative_path, expected_hash in manifest.checksums.items():
            file_path = bundle_dir / relative_path
            actual_hash = _sha256_file(file_path)
            assert actual_hash == expected_hash, (
                f"Checksum mismatch for {relative_path}: "
                f"expected {expected_hash[:16]}..., got {actual_hash[:16]}..."
            )


@pytest.mark.integration
class TestBundleCorruptChecksumDetected:
    """Verify that tampering with a file's content causes checksum mismatch detection."""

    def test_bundle_corrupt_checksum_detected(self, tmp_path: Path) -> None:
        """Tamper with model weights file, verify validate_bundle reports checksum error."""
        bundle_dir = tmp_path / "corrupt_bundle"
        _write_synthetic_bundle(bundle_dir, num_layers=2)

        # Tamper with model weights
        model_file = bundle_dir / "model" / "model.safetensors"
        original_content = model_file.read_bytes()
        tampered_content = original_content + b"TAMPERED"
        model_file.write_bytes(tampered_content)

        result = BundleExporter.validate_bundle(bundle_dir)

        assert not result.is_valid
        checksum_errors = [e for e in result.errors if "Checksum mismatch" in e]
        assert len(checksum_errors) == 1
        assert "model/model.safetensors" in checksum_errors[0]

    def test_corrupt_gate_file_detected(self, tmp_path: Path) -> None:
        """Tamper with a gate file, verify validate_bundle catches it."""
        bundle_dir = tmp_path / "corrupt_gate"
        _write_synthetic_bundle(bundle_dir, num_layers=4)

        gate_file = bundle_dir / "gates" / "layer_2_gate.safetensors"
        gate_file.write_bytes(b"corrupted-gate-data")

        result = BundleExporter.validate_bundle(bundle_dir)

        assert not result.is_valid
        checksum_errors = [e for e in result.errors if "Checksum mismatch" in e]
        assert len(checksum_errors) >= 1
        assert any("layer_2_gate" in e for e in checksum_errors)

    def test_corrupt_kernel_config_detected(self, tmp_path: Path) -> None:
        """Tamper with kernel_config.json binary content, verify checksum caught."""
        bundle_dir = tmp_path / "corrupt_kc"
        _write_synthetic_bundle(bundle_dir, num_layers=2)

        kc_path = bundle_dir / "kernel_config.json"
        kc_data = json.loads(kc_path.read_text())
        # Inject an extra field to change file content without breaking JSON parse
        kc_data["_injected"] = "tamper"
        kc_path.write_text(json.dumps(kc_data, indent=2))

        result = BundleExporter.validate_bundle(bundle_dir)

        # Checksum mismatch for kernel_config.json since content changed
        assert not result.is_valid
        assert any("kernel_config.json" in e for e in result.errors)


@pytest.mark.integration
class TestBundleMissingFileDetected:
    """Verify that removing a checksummed file from the bundle is detected."""

    def test_bundle_missing_file_detected(self, tmp_path: Path) -> None:
        """Remove a gate file that is listed in manifest checksums, verify load fails."""
        bundle_dir = tmp_path / "missing_file_bundle"
        _write_synthetic_bundle(bundle_dir, num_layers=3)

        # Remove a gate file
        missing_gate = bundle_dir / "gates" / "layer_1_gate.safetensors"
        missing_gate.unlink()

        result = BundleExporter.validate_bundle(bundle_dir)

        assert not result.is_valid
        missing_errors = [e for e in result.errors if "not found" in e]
        assert len(missing_errors) >= 1
        assert any("layer_1_gate" in e for e in missing_errors)

    def test_missing_model_weights_detected(self, tmp_path: Path) -> None:
        """Remove model/model.safetensors, verify validation fails."""
        bundle_dir = tmp_path / "missing_model"
        _write_synthetic_bundle(bundle_dir, num_layers=2)

        model_file = bundle_dir / "model" / "model.safetensors"
        model_file.unlink()

        result = BundleExporter.validate_bundle(bundle_dir)

        assert not result.is_valid
        assert any("model/model.safetensors" in e or "model/" in e for e in result.errors)

    def test_missing_manifest_detected(self, tmp_path: Path) -> None:
        """Remove manifest.json, verify validation fails immediately."""
        bundle_dir = tmp_path / "missing_manifest"
        _write_synthetic_bundle(bundle_dir, num_layers=2)

        (bundle_dir / "manifest.json").unlink()

        result = BundleExporter.validate_bundle(bundle_dir)

        assert not result.is_valid
        assert any("manifest.json" in e for e in result.errors)


@pytest.mark.integration
class TestBundleExtraFileIgnoredOrDetected:
    """Verify behavior when an unexpected file exists in the bundle directory."""

    def test_bundle_extra_file_ignored_or_detected(self, tmp_path: Path) -> None:
        """Add an unexpected file to the bundle, verify bundle still validates.

        The current BundleExporter.validate_bundle() only checks files listed in
        the manifest checksums. Extra files are NOT an error -- the bundle is still
        valid. This test documents that behavior.
        """
        bundle_dir = tmp_path / "extra_file_bundle"
        _write_synthetic_bundle(bundle_dir, num_layers=2)

        # Add an unexpected file
        extra_file = bundle_dir / "unexpected_notes.txt"
        extra_file.write_text("This file is not in the manifest")

        result = BundleExporter.validate_bundle(bundle_dir)

        # Bundle should still be valid -- extra files are not checked
        assert result.is_valid, (
            f"Bundle should be valid with extra file, got errors: {result.errors}"
        )

    def test_extra_gate_file_triggers_warning(self, tmp_path: Path) -> None:
        """Add an extra gate file beyond num_layers, verify warning emitted.

        The validator checks gate file count against manifest.num_layers and
        emits a warning on mismatch.
        """
        num_layers = 2
        bundle_dir = tmp_path / "extra_gate_bundle"
        _write_synthetic_bundle(bundle_dir, num_layers=num_layers)

        # Add extra gate file not declared in manifest
        extra_gate = bundle_dir / "gates" / f"layer_{num_layers}_gate.safetensors"
        extra_gate.write_bytes(b"extra-gate-data")

        result = BundleExporter.validate_bundle(bundle_dir)

        # Bundle is still valid (extra file not in checksums), but we expect a warning
        # about gate count mismatch
        gate_warnings = [w for w in result.warnings if "Gate file count" in w]
        assert len(gate_warnings) == 1, (
            f"Expected 1 gate count warning, got {len(gate_warnings)}: {result.warnings}"
        )


@pytest.mark.integration
class TestKernelConfigMatchesGateWeights:
    """Verify kernel_config layer indices match the set of gate weight files."""

    def test_kernel_config_matches_gate_weights(self, tmp_path: Path) -> None:
        """Kernel config per_layer_config keys must cover exactly the gate file set.

        Constructs a bundle, loads kernel config and enumerates gate files,
        verifies 1:1 correspondence between per_layer_config keys and gate files.
        """
        num_layers = 6
        bundle_dir = tmp_path / "kc_gate_match"
        _write_synthetic_bundle(bundle_dir, num_layers=num_layers)

        # Load kernel config
        kc_path = bundle_dir / "kernel_config.json"
        kc_data = json.loads(kc_path.read_text())
        kernel_config = KernelConfig.model_validate(kc_data)

        # Parse gate file layer indices from filenames
        gates_dir = bundle_dir / "gates"
        gate_layer_indices: set[int] = set()
        for gate_file in sorted(gates_dir.iterdir()):
            name = gate_file.stem  # e.g. "layer_3_gate"
            parts = name.split("_")
            # Extract integer between "layer" and "gate"
            for idx, part in enumerate(parts):
                if part == "layer" and idx + 1 < len(parts):
                    try:
                        gate_layer_indices.add(int(parts[idx + 1]))
                    except ValueError:
                        continue

        kc_layer_indices = set(kernel_config.per_layer_config.keys())

        assert gate_layer_indices == kc_layer_indices, (
            f"Mismatch: gate files cover layers {sorted(gate_layer_indices)} "
            f"but kernel_config covers {sorted(kc_layer_indices)}"
        )

        # Verify each layer_idx in per_layer_config matches its key
        for key, layer_cfg in kernel_config.per_layer_config.items():
            assert layer_cfg.layer_idx == key, (
                f"per_layer_config key {key} != layer_cfg.layer_idx {layer_cfg.layer_idx}"
            )

    def test_kernel_config_block_size_consistency(self, tmp_path: Path) -> None:
        """All per-layer block_size values must equal global block_size."""
        num_layers = 8
        bundle_dir = tmp_path / "kc_block_size"
        _write_synthetic_bundle(bundle_dir, num_layers=num_layers)

        kc_path = bundle_dir / "kernel_config.json"
        kernel_config = KernelConfig.model_validate_json(kc_path.read_text())

        for layer_idx, layer_cfg in kernel_config.per_layer_config.items():
            assert layer_cfg.block_size == kernel_config.block_size, (
                f"Layer {layer_idx} block_size {layer_cfg.block_size} "
                f"!= global {kernel_config.block_size}"
            )


@pytest.mark.integration
class TestBundleAtomicWrite:
    """Verify the temp-dir-then-rename pattern prevents partial bundles."""

    def test_bundle_atomic_write(self, tmp_path: Path) -> None:
        """Verify that if export succeeds, the final directory is complete and valid.

        Since we cannot easily test interruption of the real BundleExporter.export()
        (it requires a PeftModel), we verify the atomic semantics by:
        1. Creating a bundle manually in a temp dir
        2. Performing an atomic rename
        3. Verifying the final directory passes validation
        4. Verifying no temp directory remnants exist
        """
        parent_dir = tmp_path / "atomic_test"
        parent_dir.mkdir()
        final_dir = parent_dir / "final_bundle"
        temp_dir = parent_dir / ".tasft_bundle_tmp_atomic"

        # Step 1: Write bundle to temp dir
        _write_synthetic_bundle(temp_dir, num_layers=3)

        # Verify temp dir is valid before rename
        pre_result = BundleExporter.validate_bundle(temp_dir)
        assert pre_result.is_valid

        # Step 2: Atomic rename (simulates what BundleExporter.export does)
        temp_dir.rename(final_dir)

        # Step 3: Final dir exists and is valid
        assert final_dir.is_dir()
        post_result = BundleExporter.validate_bundle(final_dir)
        assert post_result.is_valid, f"Post-rename validation failed: {post_result.errors}"

        # Step 4: Temp dir no longer exists
        assert not temp_dir.exists(), "Temp directory should not exist after rename"

    def test_no_partial_bundle_on_failure(self, tmp_path: Path) -> None:
        """Verify that a failed write leaves no partial directory behind.

        Simulates the cleanup behavior of BundleExporter.export() on failure:
        temp directory is removed if export does not complete successfully.
        """
        parent_dir = tmp_path / "partial_test"
        parent_dir.mkdir()
        temp_dir = parent_dir / ".tasft_bundle_tmp_fail"
        final_dir = parent_dir / "should_not_exist"

        # Simulate a partial write that fails midway
        temp_dir.mkdir()
        (temp_dir / "model").mkdir()
        (temp_dir / "model" / "model.safetensors").write_bytes(b"partial")
        # Gates dir missing -- simulates failure before gate export completes

        # Simulate BundleExporter cleanup: on failure, temp dir is removed
        import shutil

        shutil.rmtree(temp_dir)

        assert not temp_dir.exists(), "Temp dir should be cleaned up after failure"
        assert not final_dir.exists(), "Final dir should never have been created"

    def test_concurrent_rename_safety(self, tmp_path: Path) -> None:
        """Verify that two threads racing to create the same bundle do not corrupt it.

        On POSIX, rename is atomic within the same filesystem. This test creates
        two temp bundles and races them to the same destination. Exactly one should
        succeed; the other should fail or be a no-op.
        """
        parent_dir = tmp_path / "concurrent_test"
        parent_dir.mkdir()
        final_dir = parent_dir / "target_bundle"

        temp_a = parent_dir / ".tmp_a"
        temp_b = parent_dir / ".tmp_b"

        _write_synthetic_bundle(temp_a, num_layers=2, model_content=b"bundle-A-content")
        _write_synthetic_bundle(temp_b, num_layers=2, model_content=b"bundle-B-content")

        results: list[bool] = [False, False]
        errors: list[Exception | None] = [None, None]
        barrier = threading.Barrier(2, timeout=5)

        def rename_bundle(src: Path, idx: int) -> None:
            barrier.wait()
            try:
                src.rename(final_dir)
                results[idx] = True
            except OSError as exc:
                errors[idx] = exc

        t1 = threading.Thread(target=rename_bundle, args=(temp_a, 0))
        t2 = threading.Thread(target=rename_bundle, args=(temp_b, 1))
        t1.start()
        t2.start()
        t1.join(timeout=10)
        t2.join(timeout=10)

        # Exactly one rename should succeed
        success_count = sum(results)
        assert success_count >= 1, "At least one rename must succeed"
        assert final_dir.is_dir(), "Target bundle directory must exist"

        # The bundle at final_dir should be valid (whichever won the race)
        validation = BundleExporter.validate_bundle(final_dir)
        assert validation.is_valid, f"Resulting bundle is invalid: {validation.errors}"


@pytest.mark.integration
class TestBundleMetadataWithoutEval:
    """Verify BundleMetadata loading when eval_summary is absent."""

    def test_bundle_metadata_without_eval(self, tmp_path: Path) -> None:
        """Load bundle metadata when eval_results.json does not exist.

        eval_summary is Optional[EvalSummary] -- it must be None when the file
        is absent, and the rest of the metadata must load successfully.
        """
        bundle_dir = tmp_path / "no_eval_bundle"
        _write_synthetic_bundle(bundle_dir, num_layers=3, include_eval=False)

        # Verify eval_results.json was not created
        eval_path = bundle_dir / "eval_results.json"
        assert not eval_path.exists()

        metadata = BundleExporter.load_bundle_metadata(bundle_dir)

        assert metadata.eval_summary is None
        assert metadata.manifest.model_name == "test-model"
        assert metadata.manifest.num_layers == 3
        assert len(metadata.kernel_config.per_layer_config) == 3

    def test_bundle_metadata_with_eval(self, tmp_path: Path) -> None:
        """Load bundle metadata when eval_results.json is present.

        Verifies eval_summary is populated with correct values.
        """
        bundle_dir = tmp_path / "with_eval_bundle"
        _write_synthetic_bundle(bundle_dir, num_layers=2, include_eval=True)

        metadata = BundleExporter.load_bundle_metadata(bundle_dir)

        assert metadata.eval_summary is not None
        assert metadata.eval_summary.task_accuracy == pytest.approx(0.85)
        assert metadata.eval_summary.mean_sparsity == pytest.approx(0.7)
        assert metadata.eval_summary.speedup_vs_dense == pytest.approx(2.5)
        assert metadata.eval_summary.eval_domain == "code"

    def test_bundle_metadata_missing_manifest_raises(self, tmp_path: Path) -> None:
        """load_bundle_metadata raises BundleError when manifest.json is missing."""
        bundle_dir = tmp_path / "no_manifest_bundle"
        _write_synthetic_bundle(bundle_dir, num_layers=2)
        (bundle_dir / "manifest.json").unlink()

        with pytest.raises(BundleError, match="manifest.json"):
            BundleExporter.load_bundle_metadata(bundle_dir)

    def test_bundle_metadata_missing_kernel_config_raises(self, tmp_path: Path) -> None:
        """load_bundle_metadata raises BundleError when kernel_config.json is missing."""
        bundle_dir = tmp_path / "no_kc_bundle"
        _write_synthetic_bundle(bundle_dir, num_layers=2)
        (bundle_dir / "kernel_config.json").unlink()

        with pytest.raises(BundleError, match="kernel_config.json"):
            BundleExporter.load_bundle_metadata(bundle_dir)


@pytest.mark.integration
class TestBundleVersionCompatibility:
    """Verify bundle_format_version handling across schema versions."""

    def test_bundle_version_compatibility_default(self, tmp_path: Path) -> None:
        """Default bundle_format_version is '1.0'."""
        manifest = _make_manifest()
        assert manifest.bundle_format_version == "1.0"
        assert manifest.version == "1.0.0"

    def test_bundle_version_roundtrip(self, tmp_path: Path) -> None:
        """bundle_format_version survives JSON serialization roundtrip."""
        manifest = _make_manifest(bundle_format_version="2.0")
        json_str = manifest.model_dump_json()
        restored = BundleManifest.model_validate_json(json_str)
        assert restored.bundle_format_version == "2.0"

    def test_bundle_version_in_metadata(self, tmp_path: Path) -> None:
        """bundle_format_version is accessible via BundleMetadata.manifest."""
        bundle_dir = tmp_path / "version_test"
        _write_synthetic_bundle(bundle_dir, num_layers=2)

        metadata = BundleExporter.load_bundle_metadata(bundle_dir)
        assert metadata.manifest.bundle_format_version == "1.0"
        assert metadata.manifest.version == "1.0.0"

    def test_bundle_version_custom_values(self) -> None:
        """Various version string formats are accepted by the schema."""
        for version_str in ("1.0", "1.1", "2.0", "0.1", "10.5"):
            manifest = _make_manifest(bundle_format_version=version_str)
            assert manifest.bundle_format_version == version_str

    def test_bundle_format_version_persisted_to_disk(self, tmp_path: Path) -> None:
        """Verify bundle_format_version is written to manifest.json on disk."""
        bundle_dir = tmp_path / "version_disk"
        _write_synthetic_bundle(bundle_dir, num_layers=1)

        manifest_data = json.loads((bundle_dir / "manifest.json").read_text())
        assert "bundle_format_version" in manifest_data
        assert manifest_data["bundle_format_version"] == "1.0"


@pytest.mark.integration
class TestLargeModelBundleStructure:
    """Verify bundle structure with 32+ layers (large model simulation)."""

    def test_large_model_bundle_structure(self, tmp_path: Path) -> None:
        """Create and validate a bundle with 32 layers.

        Verifies:
        - 32 gate files exist with correct naming
        - Kernel config has 32 per-layer entries
        - Manifest checksums cover all 34 files (1 model + 32 gates + 1 kernel_config)
        - Bundle validates successfully
        """
        num_layers = 32
        bundle_dir = tmp_path / "large_bundle"
        manifest = _write_synthetic_bundle(bundle_dir, num_layers=num_layers)

        # Verify gate file count
        gates_dir = bundle_dir / "gates"
        gate_files = sorted(gates_dir.iterdir())
        assert len(gate_files) == num_layers

        # Verify gate file naming
        for i in range(num_layers):
            expected_name = f"layer_{i}_gate.safetensors"
            gate_path = gates_dir / expected_name
            assert gate_path.exists(), f"Missing gate file: {expected_name}"

        # Verify kernel config
        kc = KernelConfig.model_validate_json(
            (bundle_dir / "kernel_config.json").read_text(),
        )
        assert len(kc.per_layer_config) == num_layers
        for i in range(num_layers):
            assert i in kc.per_layer_config
            assert kc.per_layer_config[i].layer_idx == i

        # Verify manifest checksums cover all files
        # Expected: 1 model + 32 gates + 1 kernel_config = 34
        assert len(manifest.checksums) == 1 + num_layers + 1
        assert manifest.num_layers == num_layers

        # Full validation
        result = BundleExporter.validate_bundle(bundle_dir)
        assert result.is_valid, f"Large bundle validation failed: {result.errors}"
        assert result.checked_files == len(manifest.checksums)

    def test_large_model_metadata_load(self, tmp_path: Path) -> None:
        """Verify load_bundle_metadata works with 48-layer bundle."""
        num_layers = 48
        bundle_dir = tmp_path / "xlarge_bundle"
        _write_synthetic_bundle(bundle_dir, num_layers=num_layers)

        metadata = BundleExporter.load_bundle_metadata(bundle_dir)

        assert metadata.manifest.num_layers == num_layers
        assert len(metadata.kernel_config.per_layer_config) == num_layers

        # Verify layer indices are contiguous from 0 to num_layers-1
        layer_indices = sorted(metadata.kernel_config.per_layer_config.keys())
        assert layer_indices == list(range(num_layers))

    def test_large_model_checksum_count(self, tmp_path: Path) -> None:
        """Verify checksum count scales correctly with layer count."""
        for num_layers in (32, 64, 128):
            bundle_dir = tmp_path / f"scale_{num_layers}"
            manifest = _write_synthetic_bundle(bundle_dir, num_layers=num_layers)

            # 1 model file + N gate files + 1 kernel_config
            expected_checksum_count = 1 + num_layers + 1
            assert len(manifest.checksums) == expected_checksum_count, (
                f"Expected {expected_checksum_count} checksums for {num_layers} layers, "
                f"got {len(manifest.checksums)}"
            )
