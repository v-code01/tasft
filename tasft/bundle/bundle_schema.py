"""
Deployment bundle schema for TASFT models.

A TASFT bundle contains all artifacts needed for sparse inference:
  - Merged model weights (base + LoRA)
  - AttnGate weights per layer
  - Kernel configuration (per-layer sparsity thresholds)
  - Evaluation results from training run

All schema models are frozen (immutable) and JSON-serializable.
"""
from __future__ import annotations

import re
from typing import TYPE_CHECKING, Annotated

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

if TYPE_CHECKING:
    from datetime import datetime


class LayerKernelConfig(BaseModel):
    """Per-layer kernel configuration for sparse inference.

    Preconditions:
        - layer_idx in [0, 256]
        - threshold_tau in (0, 1)
        - target_sparsity and achieved_sparsity_validation in [0, 1]
        - block_size in (0, 512]

    Postconditions:
        - Frozen immutable model
        - JSON-serializable via model_dump_json()
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    layer_idx: Annotated[int, Field(ge=0, le=256)]
    threshold_tau: Annotated[
        float, Field(gt=0.0, lt=1.0, description="Gate binarization threshold"),
    ]
    target_sparsity: Annotated[float, Field(ge=0.0, le=1.0)]
    achieved_sparsity_validation: Annotated[float, Field(ge=0.0, le=1.0)]
    gate_loss_validation: float
    block_size: Annotated[int, Field(gt=0, le=512)] = 64


class KernelConfig(BaseModel):
    """Global kernel configuration for TASFT inference.

    Invariant: per_layer_config keys must match their LayerKernelConfig.layer_idx,
    and all layer block_sizes must equal the global block_size.

    Preconditions:
        - block_size in (0, 512]
        - global_threshold in (0, 1)
        - min_sparsity_for_speedup in [0, 1]

    Postconditions:
        - Frozen immutable model
        - All layer configs validated for consistency
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    block_size: Annotated[int, Field(gt=0, le=512)] = 64
    global_threshold: Annotated[float, Field(gt=0.0, lt=1.0)] = 0.5
    per_layer_config: dict[int, LayerKernelConfig]
    min_sparsity_for_speedup: Annotated[float, Field(ge=0.0, le=1.0)] = 0.5

    @model_validator(mode="after")
    def validate_layer_consistency(self) -> KernelConfig:
        """Ensure each layer config key matches its layer_idx and block_size matches global."""
        for layer_idx, cfg in self.per_layer_config.items():
            if cfg.layer_idx != layer_idx:
                msg = f"Layer config key {layer_idx} doesn't match cfg.layer_idx {cfg.layer_idx}"
                raise ValueError(
                    msg,
                )
            if cfg.block_size != self.block_size:
                msg = f"Layer {layer_idx} block_size {cfg.block_size} != global {self.block_size}"
                raise ValueError(
                    msg,
                )
        return self


_SHA256_COMPILED = re.compile(r"^[a-f0-9]{64}$")


class BundleManifest(BaseModel):
    """Bundle manifest with metadata and file checksums.

    Preconditions:
        - All checksums must be valid lowercase hex SHA256 (64 chars)
        - total_size_bytes >= 0
        - num_layers >= 0

    Postconditions:
        - Frozen immutable model
        - file_checksums validated for SHA256 format
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    version: str = "1.0.0"
    bundle_format_version: str = "1.0"
    model_name: str
    base_model_id: str
    domain: str
    created_at: datetime
    git_hash: str
    training_args_hash: str  # SHA256 of training args JSON
    checksums: dict[str, str]  # relative path from bundle root -> SHA256 hex digest
    total_size_bytes: int
    num_layers: int

    @field_validator("checksums")
    @classmethod
    def validate_checksums(cls, v: dict[str, str]) -> dict[str, str]:
        """Validate all checksums are valid SHA256 hex digests (64 lowercase hex chars)."""
        for filename, checksum in v.items():
            if not _SHA256_COMPILED.match(checksum):
                msg = f"Invalid SHA256 checksum for {filename}: {checksum}"
                raise ValueError(
                    msg,
                )
        return v


class EvalSummary(BaseModel):
    """Summary of evaluation results from training.

    Captures task accuracy delta, throughput improvement, and sparsity metrics
    for inclusion in the deployment bundle.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    task_accuracy: float
    task_accuracy_baseline: float
    delta_accuracy: float
    mean_tokens_per_second: float
    speedup_vs_dense: float
    mean_sparsity: float
    eval_domain: str


class BundleMetadata(BaseModel):
    """Complete bundle metadata -- loaded without touching weight files.

    Use BundleExporter.load_bundle_metadata() to load this from a bundle directory.
    Provides manifest, kernel config, and optional eval summary without deserializing
    any SafeTensors weight files.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    manifest: BundleManifest
    kernel_config: KernelConfig
    eval_summary: EvalSummary | None = None


__all__ = [
    "BundleManifest",
    "BundleMetadata",
    "EvalSummary",
    "KernelConfig",
    "LayerKernelConfig",
]
