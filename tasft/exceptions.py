"""Typed exception hierarchy for TASFT. All exceptions carry structured context dicts."""
from __future__ import annotations

from typing import Any


class TASFTError(Exception):
    """Base exception for all TASFT errors. Carries structured context for logging."""

    def __init__(self, message: str, context: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.context: dict[str, Any] = context or {}


class TrainingError(TASFTError):
    """Raised when the training loop encounters an unrecoverable error."""


class NaNDetectedError(TrainingError):
    """Raised when NaN or Inf values are detected in tensors during training."""


class OOMError(TrainingError):
    """Raised when out-of-memory is detected and recovery is not possible."""


class InferenceError(TASFTError):
    """Raised during inference-time errors."""


class BundleError(TASFTError):
    """Raised for bundle export/import/validation failures."""


class ChecksumError(BundleError):
    """Raised when a file's SHA256 checksum doesn't match the manifest."""


class ValidationError(TASFTError):
    """Raised when input validation fails."""


class KernelError(TASFTError):
    """Raised when the sparse attention kernel encounters an error."""


__all__ = [
    "BundleError",
    "ChecksumError",
    "InferenceError",
    "KernelError",
    "NaNDetectedError",
    "OOMError",
    "TASFTError",
    "TrainingError",
    "ValidationError",
]
