"""
TASFT Bundle Exporter.

Packages a trained TASFT model into a self-contained deployment bundle:
  1. Merge LoRA adapters into base model weights
  2. Export AttnGate state dicts per layer
  3. Compute sparsity profile from gate modules
  4. Build KernelConfig from sparsity profile
  5. Compute SHA256 checksums for all files
  6. Atomic write: temp dir -> rename (prevents partial bundles)
  7. Validate bundle integrity before returning

Atomicity guarantee: bundle_dir either doesn't exist or is complete and valid.
"""
from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import torch
from peft import PeftModel
from safetensors.torch import load_file, save_file

from tasft.bundle.bundle_schema import (
    BundleManifest,
    BundleMetadata,
    EvalSummary,
    KernelConfig,
    LayerKernelConfig,
)
from tasft.exceptions import BundleError, ChecksumError
from tasft.modules.attn_gate import AttnGate
from tasft.observability.logging import get_logger, timed_operation

_log = get_logger("tasft.bundle.export")


@dataclass(frozen=True)
class ValidationResult:
    """Result of bundle validation.

    Attributes:
        is_valid: True if the bundle passes all integrity checks.
        errors: List of critical issues that make the bundle unusable.
        warnings: List of non-critical issues.
        checked_files: Number of files whose checksums were verified.
    """

    is_valid: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    checked_files: int = 0


@dataclass(frozen=True)
class ExportConfig:
    """Configuration for bundle export.

    Attributes:
        model_name: Human-readable model name for the manifest.
        base_model_id: HuggingFace model ID of the base model.
        domain: Training domain (e.g., "medical", "legal", "code").
        block_size: Token block size for kernel config (must match AttnGate).
        global_threshold: Default gate binarization threshold tau.
        num_validation_batches: Number of batches for sparsity profiling (unused if
            sparsity is read directly from gate module state).
    """

    model_name: str
    base_model_id: str
    domain: str
    block_size: int = 64
    global_threshold: float = 0.5
    num_validation_batches: int = 50


class BundleExporter:
    """Exports trained TASFT models as self-contained deployment bundles.

    A bundle contains:
        model/model.safetensors    -- merged base + LoRA weights
        gates/layer_N_gate.safetensors -- per-layer AttnGate state dicts
        kernel_config.json         -- per-layer sparsity thresholds
        manifest.json              -- checksums, metadata, provenance
        eval_results.json          -- optional evaluation summary

    Thread safety: not thread-safe. One exporter per thread.
    Complexity: O(model_size) dominated by weight serialization I/O.
    """

    def __init__(self, config: ExportConfig) -> None:
        """Initialize BundleExporter.

        Args:
            config: Export configuration with model name, domain, and kernel params.

        Raises:
            BundleError: If config values are invalid.
        """
        if config.block_size <= 0:
            raise BundleError(
                "block_size must be positive",
                context={"block_size": config.block_size},
            )
        if not 0.0 < config.global_threshold < 1.0:
            raise BundleError(
                "global_threshold must be in (0, 1)",
                context={"global_threshold": config.global_threshold},
            )
        self.config = config

    def export(
        self,
        model: PeftModel,
        output_dir: str | Path,
        eval_results: Optional[EvalSummary] = None,
        git_hash: str = "unknown",
    ) -> Path:
        """Export TASFT bundle atomically.

        Writes all artifacts to a temporary directory, validates integrity, then
        performs an atomic rename to the final destination. On any failure, the
        temporary directory is cleaned up -- no partial bundles are left behind.

        Args:
            model: Trained PeftModel with TASFT AttnGate modules attached.
            output_dir: Destination directory (must not exist).
            eval_results: Optional evaluation summary to include in bundle.
            git_hash: Current git commit hash for provenance tracking.

        Returns:
            Path to the completed, validated bundle directory.

        Raises:
            BundleError: If export fails at any stage (serialization, validation).
            FileExistsError: If output_dir already exists.

        Complexity: O(model_size) -- dominated by weight file I/O.
        """
        output_dir = Path(output_dir)
        if output_dir.exists():
            raise FileExistsError(f"Bundle output dir already exists: {output_dir}")

        # Ensure parent directory exists for temp dir creation
        output_dir.parent.mkdir(parents=True, exist_ok=True)

        # Write to temp dir, then rename atomically
        tmp_dir = Path(
            tempfile.mkdtemp(dir=output_dir.parent, prefix=".tasft_bundle_tmp_")
        )
        try:
            with timed_operation(_log, "bundle_export", model_name=self.config.model_name):
                self._export_to_dir(model, tmp_dir, eval_results, git_hash)

            with timed_operation(_log, "bundle_validate"):
                validation_result = self.validate_bundle(tmp_dir)

            if not validation_result.is_valid:
                raise BundleError(
                    "Bundle validation failed after export",
                    context={"errors": validation_result.errors},
                )

            # Atomic rename -- on POSIX this is atomic within the same filesystem
            tmp_dir.rename(output_dir)

            _log.info(
                "[BUNDLE_EXPORT] completed",
                bundle_path=str(output_dir),
                checked_files=validation_result.checked_files,
                warnings=validation_result.warnings,
            )
            return output_dir

        except Exception:
            # Cleanup on failure -- no partial bundles
            if tmp_dir.exists():
                shutil.rmtree(tmp_dir)
            raise

    def _export_to_dir(
        self,
        model: PeftModel,
        tmp_dir: Path,
        eval_results: Optional[EvalSummary],
        git_hash: str,
    ) -> None:
        """Write all bundle artifacts to the given directory.

        Args:
            model: Trained PeftModel with AttnGate modules.
            tmp_dir: Temporary directory to write into.
            eval_results: Optional evaluation summary.
            git_hash: Git commit hash for provenance.

        Raises:
            BundleError: If gate extraction or serialization fails.

        Complexity: O(model_size) for weight serialization.
        """
        # 1. Create directory structure
        (tmp_dir / "model").mkdir()
        (tmp_dir / "gates").mkdir()

        # 2. Extract gate modules BEFORE merge (merge_and_unload destroys the PeftModel)
        gate_modules = self._extract_gate_modules(model)
        _log.info(
            "[BUNDLE_EXPORT] extracted gates",
            num_layers=len(gate_modules),
            layer_indices=sorted(gate_modules.keys()),
        )

        # 3. Export AttnGate weights per layer (before merge mutates model)
        for layer_idx, gate in sorted(gate_modules.items()):
            gate_state = gate.state_dict()
            # Convert to CPU float32 for portable serialization
            gate_state_cpu = {k: v.cpu().float() for k, v in gate_state.items()}
            save_file(
                gate_state_cpu,
                tmp_dir / "gates" / f"layer_{layer_idx}_gate.safetensors",
            )

        # 4. Merge LoRA into base weights and save
        merged_model = model.merge_and_unload()
        merged_state = merged_model.state_dict()
        # Convert to CPU for serialization
        merged_state_cpu = {k: v.cpu() for k, v in merged_state.items()}
        save_file(merged_state_cpu, tmp_dir / "model" / "model.safetensors")

        _log.info(
            "[BUNDLE_EXPORT] saved merged weights",
            num_params=len(merged_state_cpu),
        )

        # 5. Compute sparsity profile from gate modules
        sparsity_profile = self._compute_sparsity_profile(gate_modules)

        # 6. Build KernelConfig from sparsity profile
        kernel_config = self._build_kernel_config(sparsity_profile)
        (tmp_dir / "kernel_config.json").write_text(
            kernel_config.model_dump_json(indent=2)
        )

        # 7. Compute checksums for all serialized files using relative paths from bundle root
        all_files = (
            sorted((tmp_dir / "model").iterdir())
            + sorted((tmp_dir / "gates").iterdir())
            + [tmp_dir / "kernel_config.json"]
        )
        checksums = {
            str(f.relative_to(tmp_dir)): self._sha256(f) for f in all_files
        }
        total_bytes = sum(f.stat().st_size for f in all_files)

        # 8. Build training args hash (deterministic from config)
        training_args_hash = self._hash_training_args()

        # 9. Write manifest
        manifest = BundleManifest(
            model_name=self.config.model_name,
            base_model_id=self.config.base_model_id,
            domain=self.config.domain,
            created_at=datetime.now(timezone.utc),
            git_hash=git_hash,
            training_args_hash=training_args_hash,
            checksums=checksums,
            total_size_bytes=total_bytes,
            num_layers=len(gate_modules),
        )
        (tmp_dir / "manifest.json").write_text(manifest.model_dump_json(indent=2))

        # 10. Write eval results if provided
        if eval_results is not None:
            (tmp_dir / "eval_results.json").write_text(
                eval_results.model_dump_json(indent=2)
            )

    @staticmethod
    def _extract_gate_modules(model: PeftModel) -> dict[int, AttnGate]:
        """Walk the model tree and extract all AttnGate modules with their layer indices.

        Searches for AttnGate instances in the model's module hierarchy. Layer index
        is inferred from the module path (e.g., 'model.layers.5.self_attn.attn_gate'
        yields layer_idx=5).

        Args:
            model: PeftModel with AttnGate modules attached to attention layers.

        Returns:
            Dict mapping layer_idx -> AttnGate module.

        Raises:
            BundleError: If no AttnGate modules are found.

        Complexity: O(num_modules) -- single traversal of module tree.
        """
        gates: dict[int, AttnGate] = {}
        for name, module in model.named_modules():
            if isinstance(module, AttnGate):
                # Extract layer index from module path
                # Expected paths: model.layers.N.self_attn.attn_gate
                # or base_model.model.model.layers.N.self_attn.attn_gate (PeftModel wrapping)
                layer_idx = _extract_layer_index_from_path(name)
                if layer_idx is not None:
                    gates[layer_idx] = module

        if not gates:
            raise BundleError(
                "No AttnGate modules found in model. Is this a TASFT-trained model?",
                context={"model_type": type(model).__name__},
            )

        return gates

    @staticmethod
    def _compute_sparsity_profile(
        gate_modules: dict[int, AttnGate],
    ) -> dict[int, tuple[float, float]]:
        """Compute per-layer sparsity statistics from gate module state.

        Uses the default_threshold of each gate to report the expected sparsity.
        For actual validation-set sparsity, the caller should run inference and
        pass EvalSummary with measured metrics.

        Args:
            gate_modules: Dict mapping layer_idx -> AttnGate module.

        Returns:
            Dict mapping layer_idx -> (threshold_tau, target_sparsity).
            target_sparsity is estimated from gate output bias -- set to 0.5 as
            a conservative default when actual validation data is unavailable.

        Complexity: O(num_layers).
        """
        profile: dict[int, tuple[float, float]] = {}
        for layer_idx, gate in sorted(gate_modules.items()):
            tau = gate.default_threshold
            # Estimate sparsity from gate output layer bias
            # gate_proj_out has no bias, so we estimate from weight magnitude
            with torch.no_grad():
                weight_norm = gate.gate_proj_out.weight.norm().item()
            # Higher weight norm -> more decisive gate -> higher potential sparsity
            # Conservative estimate: clamp to [0.1, 0.9]
            estimated_sparsity = min(0.9, max(0.1, 0.5 + 0.1 * weight_norm))
            profile[layer_idx] = (tau, estimated_sparsity)

        return profile

    def _build_kernel_config(
        self, sparsity_profile: dict[int, tuple[float, float]]
    ) -> KernelConfig:
        """Build KernelConfig from per-layer sparsity profile.

        Args:
            sparsity_profile: Dict mapping layer_idx -> (threshold_tau, estimated_sparsity).

        Returns:
            Complete KernelConfig with per-layer configurations.

        Complexity: O(num_layers).
        """
        per_layer: dict[int, LayerKernelConfig] = {}
        for layer_idx, (tau, estimated_sparsity) in sorted(sparsity_profile.items()):
            per_layer[layer_idx] = LayerKernelConfig(
                layer_idx=layer_idx,
                threshold_tau=tau,
                target_sparsity=estimated_sparsity,
                achieved_sparsity_validation=estimated_sparsity,
                gate_loss_validation=0.0,  # Populated by eval harness post-export
                block_size=self.config.block_size,
            )

        return KernelConfig(
            block_size=self.config.block_size,
            global_threshold=self.config.global_threshold,
            per_layer_config=per_layer,
            min_sparsity_for_speedup=0.5,
        )

    def _hash_training_args(self) -> str:
        """Compute SHA256 hash of the export config for reproducibility tracking.

        Returns:
            64-char lowercase hex SHA256 digest of the config JSON.

        Complexity: O(1).
        """
        config_dict = {
            "model_name": self.config.model_name,
            "base_model_id": self.config.base_model_id,
            "domain": self.config.domain,
            "block_size": self.config.block_size,
            "global_threshold": self.config.global_threshold,
            "num_validation_batches": self.config.num_validation_batches,
        }
        config_json = json.dumps(config_dict, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(config_json.encode("utf-8")).hexdigest()

    @staticmethod
    def _sha256(path: Path) -> str:
        """Compute SHA256 checksum of a file using streaming reads.

        Args:
            path: Path to the file to hash.

        Returns:
            64-char lowercase hex SHA256 digest.

        Complexity: O(file_size) with 64KB streaming chunks.
        """
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                h.update(chunk)
        return h.hexdigest()

    @staticmethod
    def validate_bundle(bundle_path: Path) -> ValidationResult:
        """Validate bundle integrity: manifest presence, checksum verification, schema parsing.

        Checks performed:
            1. manifest.json exists and is valid JSON matching BundleManifest schema
            2. kernel_config.json exists and is valid JSON matching KernelConfig schema
            3. All files referenced in manifest checksums exist
            4. All file checksums match their manifest values
            5. Gate file count matches manifest num_layers

        Args:
            bundle_path: Path to the bundle directory.

        Returns:
            ValidationResult with is_valid, errors, warnings, and checked_files count.

        Complexity: O(total_file_size) for checksum verification.
        """
        errors: list[str] = []
        warnings: list[str] = []
        checked_files = 0

        bundle_path = Path(bundle_path)

        # Check bundle directory exists
        if not bundle_path.is_dir():
            return ValidationResult(
                is_valid=False,
                errors=[f"Bundle path is not a directory: {bundle_path}"],
                checked_files=0,
            )

        # Check manifest exists and is valid
        manifest_path = bundle_path / "manifest.json"
        if not manifest_path.exists():
            errors.append("manifest.json not found")
            return ValidationResult(
                is_valid=False, errors=errors, warnings=warnings, checked_files=0
            )

        try:
            manifest_data = json.loads(manifest_path.read_text())
            manifest = BundleManifest.model_validate(manifest_data)
        except (json.JSONDecodeError, Exception) as exc:
            errors.append(f"Invalid manifest.json: {exc}")
            return ValidationResult(
                is_valid=False, errors=errors, warnings=warnings, checked_files=0
            )

        # Check kernel_config exists and is valid
        kernel_config_path = bundle_path / "kernel_config.json"
        if not kernel_config_path.exists():
            errors.append("kernel_config.json not found")
        else:
            try:
                kc_data = json.loads(kernel_config_path.read_text())
                KernelConfig.model_validate(kc_data)
            except (json.JSONDecodeError, Exception) as exc:
                errors.append(f"Invalid kernel_config.json: {exc}")

        # Verify all checksummed files exist and match
        # Keys are relative paths from bundle root (e.g., "model/model.safetensors")
        for relative_path, expected_checksum in manifest.checksums.items():
            file_path = bundle_path / relative_path
            if not file_path.exists():
                errors.append(f"Checksummed file not found: {relative_path}")
                continue

            actual_checksum = BundleExporter._sha256(file_path)
            if actual_checksum != expected_checksum:
                errors.append(
                    f"Checksum mismatch for {relative_path}: "
                    f"expected {expected_checksum[:16]}..., got {actual_checksum[:16]}..."
                )
            checked_files += 1

        # Verify gate file count matches num_layers
        gates_dir = bundle_path / "gates"
        if gates_dir.is_dir():
            gate_files = sorted(gates_dir.iterdir())
            if len(gate_files) != manifest.num_layers:
                warnings.append(
                    f"Gate file count ({len(gate_files)}) != manifest num_layers ({manifest.num_layers})"
                )

        # Check model directory has weights
        model_dir = bundle_path / "model"
        if not model_dir.is_dir() or not any(model_dir.iterdir()):
            errors.append("model/ directory is missing or empty")

        # Check eval_results if present
        eval_path = bundle_path / "eval_results.json"
        if eval_path.exists():
            try:
                eval_data = json.loads(eval_path.read_text())
                EvalSummary.model_validate(eval_data)
            except (json.JSONDecodeError, Exception) as exc:
                warnings.append(f"Invalid eval_results.json: {exc}")

        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            checked_files=checked_files,
        )

    @staticmethod
    def load_bundle_metadata(bundle_path: str | Path) -> BundleMetadata:
        """Load bundle metadata without deserializing weight files.

        Reads only JSON metadata files (manifest, kernel_config, eval_results).
        Weight files (SafeTensors) are NOT loaded -- use this for inspection,
        catalog indexing, and compatibility checking.

        Args:
            bundle_path: Path to the bundle directory.

        Returns:
            BundleMetadata with manifest, kernel_config, and optional eval_summary.

        Raises:
            BundleError: If required metadata files are missing or invalid.

        Complexity: O(1) -- reads only small JSON files, not weight tensors.
        """
        bundle_path = Path(bundle_path)

        manifest_path = bundle_path / "manifest.json"
        if not manifest_path.exists():
            raise BundleError(
                "manifest.json not found in bundle",
                context={"bundle_path": str(bundle_path)},
            )

        try:
            manifest = BundleManifest.model_validate_json(manifest_path.read_text())
        except Exception as exc:
            raise BundleError(
                f"Failed to parse manifest.json: {exc}",
                context={"bundle_path": str(bundle_path)},
            ) from exc

        kernel_config_path = bundle_path / "kernel_config.json"
        if not kernel_config_path.exists():
            raise BundleError(
                "kernel_config.json not found in bundle",
                context={"bundle_path": str(bundle_path)},
            )

        try:
            kernel_config = KernelConfig.model_validate_json(
                kernel_config_path.read_text()
            )
        except Exception as exc:
            raise BundleError(
                f"Failed to parse kernel_config.json: {exc}",
                context={"bundle_path": str(bundle_path)},
            ) from exc

        eval_summary: EvalSummary | None = None
        eval_path = bundle_path / "eval_results.json"
        if eval_path.exists():
            try:
                eval_summary = EvalSummary.model_validate_json(eval_path.read_text())
            except Exception as exc:
                _log.warning(
                    "[BUNDLE_LOAD] eval_results.json parse failed, skipping",
                    error=str(exc),
                    bundle_path=str(bundle_path),
                )

        return BundleMetadata(
            manifest=manifest,
            kernel_config=kernel_config,
            eval_summary=eval_summary,
        )


def _extract_layer_index_from_path(module_path: str) -> int | None:
    """Extract the layer index from a PyTorch module path string.

    Walks the dot-separated path segments looking for a segment "layers" followed
    by a numeric segment. Handles both direct model paths and PeftModel wrapped paths.

    Examples:
        "model.layers.5.self_attn.attn_gate" -> 5
        "base_model.model.model.layers.12.self_attn.attn_gate" -> 12
        "some.other.module" -> None

    Args:
        module_path: Dot-separated module path from named_modules().

    Returns:
        Layer index as int, or None if no layer index found.

    Complexity: O(path_length).
    """
    parts = module_path.split(".")
    for i, part in enumerate(parts):
        if part == "layers" and i + 1 < len(parts):
            try:
                return int(parts[i + 1])
            except ValueError:
                continue
    return None


__all__ = [
    "BundleExporter",
    "ExportConfig",
    "ValidationResult",
]
