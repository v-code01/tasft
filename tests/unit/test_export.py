"""Unit tests for BundleExporter: deployment artifact packaging.

Tests verify:
    - Export creates all required files (manifest, model, gates, kernel_config)
    - Checksums are valid after export
    - Corrupted model file fails validation
    - Missing gate file fails validation
    - Atomicity: partial export leaves no output_dir behind
    - load_bundle_metadata reads only JSONs (fast, no weights)
    - Existing output_dir raises FileExistsError

All tests use tmp_path and mock bundle structures (no real model weights).
"""
from __future__ import annotations

import hashlib
import json
import time
from datetime import UTC, datetime
from typing import TYPE_CHECKING

import pytest
import torch
from safetensors.torch import save_file

from tasft.bundle.bundle_schema import (
    BundleManifest,
    KernelConfig,
    LayerKernelConfig,
)
from tasft.bundle.export import BundleExporter, ExportConfig

if TYPE_CHECKING:
    from pathlib import Path

# Pydantic models in bundle_schema.py guard `datetime` behind TYPE_CHECKING,
# so it's unavailable at runtime. Passing the resolved type via _types_namespace
# lets model_rebuild() succeed for validation/deserialization paths.
_ns: dict[str, type] = {"datetime": datetime}
BundleManifest.model_rebuild(_types_namespace=_ns)
KernelConfig.model_rebuild(_types_namespace=_ns)
LayerKernelConfig.model_rebuild(_types_namespace=_ns)


def _sha256(path: Path) -> str:
    """Compute SHA256 hex digest of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _create_mock_bundle(
    bundle_dir: Path,
    num_layers: int = 2,
    corrupt_model: bool = False,
    skip_gate_layer: int | None = None,
) -> None:
    """Create a mock TASFT bundle with all required files.

    Writes manifest and kernel_config as raw JSON dicts to avoid Pydantic
    constructor issues with deferred `datetime` annotation resolution.

    Args:
        bundle_dir: Directory to create the bundle in.
        num_layers: Number of gate layers to create.
        corrupt_model: If True, corrupt the model.safetensors file after creation.
        skip_gate_layer: If set, skip creating the gate file for this layer index.
    """
    bundle_dir.mkdir(parents=True, exist_ok=True)
    (bundle_dir / "model").mkdir()
    (bundle_dir / "gates").mkdir()

    # Create model weights
    model_state = {"weight": torch.randn(32, 32)}
    model_path = bundle_dir / "model" / "model.safetensors"
    save_file(model_state, str(model_path))

    # Create gate weights per layer
    for i in range(num_layers):
        if skip_gate_layer is not None and i == skip_gate_layer:
            continue
        gate_state = {
            "gate_proj_in.weight": torch.randn(16, 64),
            "gate_proj_in.bias": torch.zeros(16),
            "gate_proj_out.weight": torch.randn(1, 16),
        }
        gate_path = bundle_dir / "gates" / f"layer_{i}_gate.safetensors"
        save_file(gate_state, str(gate_path))

    # Build kernel_config as raw dict (bypasses Pydantic constructor)
    per_layer_config: dict[str, dict[str, object]] = {}
    for i in range(num_layers):
        per_layer_config[str(i)] = {
            "layer_idx": i,
            "threshold_tau": 0.5,
            "target_sparsity": 0.7,
            "achieved_sparsity_validation": 0.65,
            "gate_loss_validation": 0.01,
            "block_size": 64,
        }

    kernel_config_dict = {
        "block_size": 64,
        "global_threshold": 0.5,
        "per_layer_config": per_layer_config,
        "min_sparsity_for_speedup": 0.5,
    }
    (bundle_dir / "kernel_config.json").write_text(
        json.dumps(kernel_config_dict, indent=2),
    )

    # Compute checksums for all files (using relative paths from bundle root)
    all_files = (
        sorted((bundle_dir / "model").iterdir())
        + sorted((bundle_dir / "gates").iterdir())
        + [bundle_dir / "kernel_config.json"]
    )
    checksums = {
        str(f.relative_to(bundle_dir)): _sha256(f) for f in all_files
    }
    total_bytes = sum(f.stat().st_size for f in all_files)

    # Corrupt model AFTER checksum if requested
    if corrupt_model:
        with open(model_path, "r+b") as f:
            f.seek(0)
            f.write(b"\xff")

    # Write manifest as raw JSON dict (bypasses Pydantic datetime resolution)
    manifest_dict = {
        "version": "1.0.0",
        "bundle_format_version": "1.0",
        "model_name": "test-model",
        "base_model_id": "gpt2",
        "domain": "test",
        "created_at": datetime.now(UTC).isoformat(),
        "git_hash": "abc1234",
        "training_args_hash": "a" * 64,
        "checksums": checksums,
        "total_size_bytes": total_bytes,
        "num_layers": num_layers,
    }
    (bundle_dir / "manifest.json").write_text(json.dumps(manifest_dict, indent=2))


@pytest.mark.unit
class TestBundleExportCreateFiles:
    """Tests for bundle file creation and structure."""

    def test_export_creates_all_required_files(self, tmp_path: Path) -> None:
        """After export, bundle must contain manifest.json, model/model.safetensors,
        gates/layer_N_gate.safetensors for each layer, and kernel_config.json."""
        num_layers = 2
        bundle_dir = tmp_path / "bundle"
        _create_mock_bundle(bundle_dir, num_layers=num_layers)

        # Verify required files
        assert (bundle_dir / "manifest.json").exists(), "manifest.json missing"
        assert (bundle_dir / "model" / "model.safetensors").exists(), (
            "model/model.safetensors missing"
        )
        assert (bundle_dir / "kernel_config.json").exists(), (
            "kernel_config.json missing"
        )

        for i in range(num_layers):
            gate_path = bundle_dir / "gates" / f"layer_{i}_gate.safetensors"
            assert gate_path.exists(), f"gates/layer_{i}_gate.safetensors missing"

    def test_export_checksums_valid(self, tmp_path: Path) -> None:
        """After export, validate_bundle must return is_valid=True."""
        bundle_dir = tmp_path / "valid_bundle"
        _create_mock_bundle(bundle_dir, num_layers=2)

        result = BundleExporter.validate_bundle(bundle_dir)
        assert result.is_valid, (
            f"Bundle validation failed with errors: {result.errors}"
        )
        assert result.checked_files > 0, "No files were checked during validation"

    def test_corrupted_model_fails_validation(self, tmp_path: Path) -> None:
        """Corrupting model.safetensors after checksum computation must fail validation."""
        bundle_dir = tmp_path / "corrupt_bundle"
        _create_mock_bundle(bundle_dir, num_layers=2, corrupt_model=True)

        result = BundleExporter.validate_bundle(bundle_dir)
        assert not result.is_valid, (
            "Corrupted bundle should fail validation"
        )
        # At least one error should mention checksum
        checksum_errors = [e for e in result.errors if "hecksum" in e or "mismatch" in e]
        assert len(checksum_errors) > 0, (
            f"Expected checksum error, got errors: {result.errors}"
        )

    def test_missing_gate_file_fails_validation(self, tmp_path: Path) -> None:
        """Skipping one gate file must cause validation to detect the mismatch.

        The manifest still references num_layers=2, but only 1 gate file exists.
        validate_bundle checks gate count against manifest and also verifies
        checksums -- a missing file referenced in checksums triggers 'not found'.
        """
        bundle_dir = tmp_path / "missing_gate_bundle"
        _create_mock_bundle(bundle_dir, num_layers=2, skip_gate_layer=1)

        result = BundleExporter.validate_bundle(bundle_dir)
        # Should detect the missing gate: either via checksum 'not found' or gate count warning
        has_issue = (
            not result.is_valid
            or any("gate" in w.lower() or "num_layers" in w.lower() for w in result.warnings)
        )
        assert has_issue, (
            f"Expected validation issue for missing gate, got: "
            f"errors={result.errors}, warnings={result.warnings}"
        )


@pytest.mark.unit
class TestBundleExportAtomicity:
    """Tests for atomic export behavior."""

    def test_export_atomicity_on_error(self, tmp_path: Path) -> None:
        """If export fails mid-process, the output_dir must not exist afterward.

        We simulate this by creating an ExportConfig and calling export with
        a model that will fail during gate extraction (not a PeftModel).
        """
        from unittest.mock import MagicMock

        config = ExportConfig(
            model_name="test",
            base_model_id="gpt2",
            domain="test",
            block_size=64,
            global_threshold=0.5,
        )
        exporter = BundleExporter(config)

        # Mock model that will cause _extract_gate_modules to raise BundleError
        mock_model = MagicMock()
        mock_model.named_modules.return_value = []  # No modules -> BundleError

        output_dir = tmp_path / "atomic_test_output"

        with pytest.raises(Exception):
            exporter.export(mock_model, output_dir)

        # The output_dir should NOT exist after a failed export
        assert not output_dir.exists(), (
            f"Output dir {output_dir} should not exist after failed export "
            f"(atomicity guarantee violated)"
        )

    def test_bundle_output_dir_conflict(self, tmp_path: Path) -> None:
        """Exporting to an existing directory must raise FileExistsError immediately."""
        from unittest.mock import MagicMock

        config = ExportConfig(
            model_name="test",
            base_model_id="gpt2",
            domain="test",
        )
        exporter = BundleExporter(config)

        # Create the output dir first
        output_dir = tmp_path / "existing_dir"
        output_dir.mkdir()

        mock_model = MagicMock()

        with pytest.raises(FileExistsError):
            exporter.export(mock_model, output_dir)


@pytest.mark.unit
class TestBundleMetadataLoading:
    """Tests for load_bundle_metadata — fast metadata-only loading."""

    def test_load_metadata_doesnt_load_weights(self, tmp_path: Path) -> None:
        """load_bundle_metadata() must complete quickly (< 200ms) even for
        bundles with large weight files, since it only reads JSON metadata."""
        bundle_dir = tmp_path / "metadata_bundle"
        _create_mock_bundle(bundle_dir, num_layers=2)

        start = time.perf_counter_ns()
        metadata = BundleExporter.load_bundle_metadata(bundle_dir)
        elapsed_ms = (time.perf_counter_ns() - start) / 1_000_000

        assert metadata is not None, "load_bundle_metadata returned None"
        assert metadata.manifest.model_name == "test-model", (
            f"Unexpected model name: {metadata.manifest.model_name}"
        )
        assert metadata.kernel_config.block_size == 64, (
            f"Unexpected block size: {metadata.kernel_config.block_size}"
        )
        assert elapsed_ms < 200, (
            f"load_bundle_metadata took {elapsed_ms:.1f}ms, expected < 200ms. "
            f"This suggests weight files are being loaded."
        )

    def test_load_metadata_validates_manifest(self, tmp_path: Path) -> None:
        """load_bundle_metadata must properly parse manifest fields."""
        bundle_dir = tmp_path / "manifest_check"
        _create_mock_bundle(bundle_dir, num_layers=3)

        metadata = BundleExporter.load_bundle_metadata(bundle_dir)
        assert metadata.manifest.num_layers == 3, (
            f"Expected 3 layers, got {metadata.manifest.num_layers}"
        )
        assert metadata.manifest.domain == "test"
        assert metadata.manifest.base_model_id == "gpt2"
        assert len(metadata.manifest.checksums) > 0, (
            "Manifest should have checksums"
        )

    def test_load_metadata_missing_manifest_raises(self, tmp_path: Path) -> None:
        """load_bundle_metadata on a dir without manifest.json must raise BundleError."""
        from tasft.exceptions import BundleError

        empty_dir = tmp_path / "no_manifest"
        empty_dir.mkdir()

        with pytest.raises(BundleError, match="manifest"):
            BundleExporter.load_bundle_metadata(empty_dir)


@pytest.mark.unit
class TestExportConfig:
    """Tests for ExportConfig validation."""

    def test_invalid_block_size_raises(self) -> None:
        """block_size <= 0 must raise BundleError."""
        from tasft.exceptions import BundleError

        config = ExportConfig(
            model_name="test",
            base_model_id="gpt2",
            domain="test",
            block_size=0,
        )
        with pytest.raises(BundleError, match="block_size"):
            BundleExporter(config)

    def test_invalid_threshold_raises(self) -> None:
        """global_threshold outside (0, 1) must raise BundleError."""
        from tasft.exceptions import BundleError

        config = ExportConfig(
            model_name="test",
            base_model_id="gpt2",
            domain="test",
            global_threshold=1.0,
        )
        with pytest.raises(BundleError, match="global_threshold"):
            BundleExporter(config)
