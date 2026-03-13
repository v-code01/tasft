"""
TASFT bundle: export and import of trained sparse models.

This package contains:
- SafeTensors-based bundle format with SHA256 integrity
- Bundle manifest with metadata, sparsity profiles, and checksums
- Atomic export with validation (no partial bundles)
- Compatibility validation for model-gate alignment
"""
from tasft.bundle.bundle_schema import (
    BundleManifest,
    BundleMetadata,
    EvalSummary,
    KernelConfig,
    LayerKernelConfig,
)
from tasft.bundle.export import BundleExporter, ExportConfig, ValidationResult

__all__ = [
    "BundleExporter",
    "BundleManifest",
    "BundleMetadata",
    "EvalSummary",
    "ExportConfig",
    "KernelConfig",
    "LayerKernelConfig",
    "ValidationResult",
]
