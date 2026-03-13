"""Kernel configuration for TASFT inference deployment.

Provides immutable configuration for per-layer and global kernel parameters,
including gate thresholds, target sparsity, and block sizes. All configs are
validated at construction time via Pydantic v2.

Preconditions:
    - threshold values in (0, 1) exclusive
    - target_sparsity in [0, 1]
    - block_size in {32, 64, 128}
    - min_sparsity_for_speedup in [0, 1]

Postconditions:
    - All fields immutable after construction (frozen model)
    - Per-layer overrides take precedence over global defaults

Complexity: O(n) validation where n = number of layers configured.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, field_validator, model_validator

from tasft.types import VALID_BLOCK_SIZES

_VALID_BLOCK_SIZES = VALID_BLOCK_SIZES  # Local alias for backward compat


class LayerKernelConfig(BaseModel):
    """Per-layer kernel configuration for gate threshold and sparsity targets.

    Args:
        layer_idx: Index of the transformer layer this config applies to.
        threshold_tau: Gate threshold for hard mask binarization. Values above
            this threshold produce mask=True (attend), below produce mask=False (skip).
        target_sparsity: Expected sparsity ratio in [0, 1] after thresholding.
        achieved_sparsity_validation: Measured sparsity from validation, used for
            speedup estimation and deployment auditing.
        block_size: Attention block granularity in tokens. Must be 32, 64, or 128.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    layer_idx: int
    threshold_tau: float
    target_sparsity: float
    achieved_sparsity_validation: float
    block_size: int = 64

    @field_validator("threshold_tau")
    @classmethod
    def _validate_threshold(cls, v: float) -> float:
        if not 0.0 < v < 1.0:
            msg = f"threshold_tau must be in (0, 1), got {v}"
            raise ValueError(msg)
        return v

    @field_validator("target_sparsity", "achieved_sparsity_validation")
    @classmethod
    def _validate_sparsity(cls, v: float) -> float:
        if not 0.0 <= v <= 1.0:
            msg = f"sparsity must be in [0, 1], got {v}"
            raise ValueError(msg)
        return v

    @field_validator("block_size")
    @classmethod
    def _validate_block_size(cls, v: int) -> int:
        if v not in _VALID_BLOCK_SIZES:
            msg = f"block_size must be one of {sorted(_VALID_BLOCK_SIZES)}, got {v}"
            raise ValueError(msg)
        return v


class KernelConfig(BaseModel):
    """Global kernel configuration aggregating per-layer overrides.

    Args:
        block_size: Default block size for all layers (overridden by per-layer config).
        global_threshold: Default gate threshold for layers without per-layer config.
        per_layer_config: Mapping from layer index to layer-specific kernel config.
        min_sparsity_for_speedup: Minimum sparsity ratio below which dense fallback
            is used instead of sparse kernel (sparse overhead exceeds savings).
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    block_size: int = 64
    global_threshold: float = 0.5
    per_layer_config: dict[int, LayerKernelConfig] = {}
    min_sparsity_for_speedup: float = 0.5

    @field_validator("global_threshold")
    @classmethod
    def _validate_threshold(cls, v: float) -> float:
        if not 0.0 < v < 1.0:
            msg = f"global_threshold must be in (0, 1), got {v}"
            raise ValueError(msg)
        return v

    @field_validator("block_size")
    @classmethod
    def _validate_block_size(cls, v: int) -> int:
        if v not in _VALID_BLOCK_SIZES:
            msg = f"block_size must be one of {sorted(_VALID_BLOCK_SIZES)}, got {v}"
            raise ValueError(msg)
        return v

    @field_validator("min_sparsity_for_speedup")
    @classmethod
    def _validate_min_sparsity(cls, v: float) -> float:
        if not 0.0 <= v <= 1.0:
            msg = f"min_sparsity_for_speedup must be in [0, 1], got {v}"
            raise ValueError(msg)
        return v

    @model_validator(mode="after")
    def _validate_layer_consistency(self) -> KernelConfig:
        """Ensure each per-layer config key matches its layer_idx and block_size matches global.

        Invariant: per_layer_config dict keys are the canonical layer index source.
        Each LayerKernelConfig.layer_idx must equal its dict key, and each layer's
        block_size must equal the global block_size to prevent silent misconfiguration.

        Complexity: O(n) where n = len(per_layer_config).
        """
        for layer_idx, cfg in self.per_layer_config.items():
            if cfg.layer_idx != layer_idx:
                msg = (
                    f"Layer config key {layer_idx} doesn't match "
                    f"cfg.layer_idx {cfg.layer_idx}"
                )
                raise ValueError(msg)
            if cfg.block_size != self.block_size:
                msg = (
                    f"Layer {layer_idx} block_size {cfg.block_size} "
                    f"!= global {self.block_size}"
                )
                raise ValueError(msg)
        return self

    def get_layer_threshold(self, layer_idx: int) -> float:
        """Return threshold for a specific layer, falling back to global default.

        Args:
            layer_idx: Transformer layer index.

        Returns:
            Threshold tau in (0, 1).

        Complexity: O(1) dict lookup.
        """
        if layer_idx in self.per_layer_config:
            return self.per_layer_config[layer_idx].threshold_tau
        return self.global_threshold

    def get_layer_block_size(self, layer_idx: int) -> int:
        """Return block size for a specific layer, falling back to global default.

        Args:
            layer_idx: Transformer layer index.

        Returns:
            Block size in {32, 64, 128}.

        Complexity: O(1) dict lookup.
        """
        if layer_idx in self.per_layer_config:
            return self.per_layer_config[layer_idx].block_size
        return self.block_size


__all__ = [
    "KernelConfig",
    "LayerKernelConfig",
]
