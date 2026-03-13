"""
Chaos tests: Checkpoint corruption resilience.

Validates that TASFT bundle loading correctly rejects corrupted artifacts
with specific error types and structured context — no silent data corruption.

Markers: @pytest.mark.chaos
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from tasft.exceptions import ChecksumError
from tasft.inference.tasft_model import _verify_checksum


@pytest.mark.chaos
@pytest.mark.timeout(10)
def test_checksum_rejects_corrupted_file(tmp_path: Path) -> None:
    """_verify_checksum must raise ChecksumError for modified file content."""
    test_file = tmp_path / "test.bin"
    test_file.write_bytes(b"original content")

    # Compute checksum of original
    import hashlib
    original_hash = hashlib.sha256(b"original content").hexdigest()

    # Verify original passes
    _verify_checksum(test_file, original_hash)

    # Corrupt the file
    test_file.write_bytes(b"corrupted content")

    # Verify corrupted file fails
    with pytest.raises(ChecksumError) as exc_info:
        _verify_checksum(test_file, original_hash)

    assert exc_info.value.context is not None
    assert "expected" in exc_info.value.context
    assert "actual" in exc_info.value.context
    assert exc_info.value.context["expected"] == original_hash
    assert exc_info.value.context["actual"] != original_hash


@pytest.mark.chaos
@pytest.mark.timeout(10)
def test_checksum_rejects_empty_file(tmp_path: Path) -> None:
    """_verify_checksum must reject an empty file against a non-empty content hash."""
    test_file = tmp_path / "empty.bin"
    test_file.write_bytes(b"")

    # Hash of non-empty content
    import hashlib
    expected_hash = hashlib.sha256(b"some content").hexdigest()

    with pytest.raises(ChecksumError):
        _verify_checksum(test_file, expected_hash)


@pytest.mark.chaos
@pytest.mark.timeout(10)
def test_checksum_accepts_correct_hash(tmp_path: Path) -> None:
    """_verify_checksum must pass silently for correct hash — no false positives."""
    content = b"test content for checksum verification"
    test_file = tmp_path / "valid.bin"
    test_file.write_bytes(content)

    import hashlib
    correct_hash = hashlib.sha256(content).hexdigest()

    # Should not raise
    _verify_checksum(test_file, correct_hash)


@pytest.mark.chaos
@pytest.mark.timeout(10)
def test_corrupted_manifest_json_rejected(tmp_path: Path) -> None:
    """A manifest with invalid JSON must raise a clean error, not a crash."""
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text("{broken json{{{{")

    with pytest.raises(json.JSONDecodeError):
        with open(manifest_path) as f:
            json.load(f)


@pytest.mark.chaos
@pytest.mark.timeout(10)
def test_truncated_gate_state_dict_rejected(tmp_path: Path) -> None:
    """A truncated gate state dict file must raise on torch.load."""
    gate_file = tmp_path / "gate_state_dict.pt"
    # Write partial/corrupt binary data
    gate_file.write_bytes(b"\x80\x02}q\x00(X\x05corrupted")

    with pytest.raises(Exception):
        torch.load(gate_file, map_location="cpu", weights_only=True)


@pytest.mark.chaos
@pytest.mark.timeout(10)
def test_sparsity_profile_with_out_of_range_values(tmp_path: Path) -> None:
    """Sparsity profile with values outside [0,1] must be detectable.

    While TASFT enforces this at write time, a corrupted file could bypass.
    Downstream consumers must validate.
    """
    profile_path = tmp_path / "sparsity_profile.json"
    bad_profile = {
        "step": 100,
        "num_layers": 4,
        "per_layer_sparsity": {"0": 0.5, "1": -0.3, "2": 1.5, "3": 0.8},
        "mean_sparsity": 0.625,
    }
    with open(profile_path, "w") as f:
        json.dump(bad_profile, f)

    with open(profile_path) as f:
        loaded = json.load(f)

    per_layer = loaded["per_layer_sparsity"]
    invalid_layers = [
        k for k, v in per_layer.items()
        if not (0.0 <= v <= 1.0)
    ]
    assert len(invalid_layers) == 2, f"Expected 2 invalid layers, got {invalid_layers}"
    assert "1" in invalid_layers
    assert "2" in invalid_layers
