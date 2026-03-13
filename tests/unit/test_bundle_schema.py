"""Unit tests for bundle schema validation.

NOTE: Bundle export module (Task #10) is not yet implemented.
These tests are placeholder stubs that will be completed once the
bundle schema module is available. They are marked xfail so the
test suite stays green.

Tests to implement:
1. Valid construction + JSON roundtrip (serialize -> parse -> equal)
2. Invalid SHA256 format -> raises ValidationError
3. Layer idx mismatch -> raises ValidationError
4. Frozen: attempt mutation -> raises FrozenInstanceError
5. Extra fields rejected (extra='forbid')
"""
import pytest


@pytest.mark.unit
@pytest.mark.xfail(reason="Bundle schema module not yet implemented (Task #10)")
class TestBundleSchemaPlaceholder:
    """Placeholder tests awaiting bundle schema implementation."""

    def test_valid_construction_json_roundtrip(self) -> None:
        raise NotImplementedError("Awaiting bundle schema module")

    def test_invalid_sha256_raises(self) -> None:
        raise NotImplementedError("Awaiting bundle schema module")

    def test_layer_idx_mismatch_raises(self) -> None:
        raise NotImplementedError("Awaiting bundle schema module")

    def test_frozen_mutation_raises(self) -> None:
        raise NotImplementedError("Awaiting bundle schema module")

    def test_extra_fields_rejected(self) -> None:
        raise NotImplementedError("Awaiting bundle schema module")
