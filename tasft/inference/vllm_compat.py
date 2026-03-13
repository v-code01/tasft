"""
vLLM Version Compatibility Layer for TASFT.

Detects vLLM version at runtime and provides adapter shims that normalize
access to attn_metadata fields across vLLM releases. This prevents silent
breakage when vLLM renames internal APIs between minor versions.

Supported vLLM versions: 0.4.x through 0.8.x (tested).

Preconditions:
    - vLLM may or may not be installed; all imports are guarded.
    - When vLLM is installed, vllm.__version__ follows semver (MAJOR.MINOR.PATCH).

Postconditions:
    - detect_vllm_version() returns VLLMVersion or None (never raises).
    - get_attn_metadata_adapter() returns a callable adapter matching the
      installed vLLM version's attn_metadata field layout.
    - validate_worker_structure() returns a list of human-readable issues
      (empty list = worker is patchable).

Complexity: All functions are O(1) except validate_worker_structure which is
            O(N) in the number of named modules.
"""
from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Any

from tasft.observability.logging import get_logger

logger = get_logger("tasft.inference.vllm_compat")


@dataclass(frozen=True)
class VLLMVersion:
    """Parsed vLLM semantic version.

    Fields:
        major: Major version number (breaking changes).
        minor: Minor version number (feature additions).
        patch: Patch version number (bug fixes).

    Invariants:
        - All fields are non-negative integers.
        - Constructed exclusively via detect_vllm_version().
    """

    major: int
    minor: int
    patch: int

    def as_tuple(self) -> tuple[int, int, int]:
        """Return (major, minor, patch) tuple for comparison.

        Complexity: O(1).
        """
        return (self.major, self.minor, self.patch)

    def __str__(self) -> str:
        return f"{self.major}.{self.minor}.{self.patch}"


# Tested (major, minor) combinations. Versions outside this set may work
# but have not been validated against TASFT's attn_metadata duck-typing.
SUPPORTED_VLLM_VERSIONS: frozenset[tuple[int, int]] = frozenset(
    {(0, 4), (0, 5), (0, 6), (0, 7), (0, 8)}
)

# Known API changes between vLLM versions that affect TASFT patching.
# Maps (major, minor) -> list of human-readable descriptions of breaking changes.
_KNOWN_BREAKING_CHANGES: dict[tuple[int, int], list[str]] = {
    (0, 5): [
        "attn_metadata.is_prompt renamed to attn_metadata.is_prefill",
        "num_prompt_tokens renamed to num_prefill_tokens",
    ],
    (0, 6): [
        "AttentionMetadata moved from vllm.attention to vllm.attention.backends",
        "seq_lens field may be a torch.Tensor instead of list[int]",
    ],
    (0, 7): [
        "Worker.model moved to Worker.model_runner.model",
        "Attention module class hierarchy refactored",
    ],
    (0, 8): [
        "FlashAttentionMetadata replaces previous AttentionMetadata in some backends",
        "num_prefill_tokens may be absent; use prefill_metadata instead",
    ],
}


def detect_vllm_version() -> VLLMVersion | None:
    """Safely detect the installed vLLM version.

    Imports vllm, reads __version__, and parses into VLLMVersion.
    Returns None if vLLM is not installed or version cannot be parsed.

    Postconditions:
        - Returns VLLMVersion with non-negative integers, or None.
        - Never raises an exception.

    Complexity: O(1) (import is cached by Python after first call).
    """
    try:
        vllm_module = importlib.import_module("vllm")
    except ImportError:
        logger.debug("[VLLM_COMPAT] vLLM not installed")
        return None

    version_str = getattr(vllm_module, "__version__", None)
    if version_str is None:
        logger.warning(
            "[VLLM_COMPAT] vLLM installed but __version__ attribute missing"
        )
        return None

    return _parse_version_string(str(version_str))


def _parse_version_string(version_str: str) -> VLLMVersion | None:
    """Parse a semver string into VLLMVersion.

    Handles formats: "0.4.0", "0.5.3.post1", "0.6.0.dev123".
    Strips pre-release/build suffixes and extracts major.minor.patch.

    Args:
        version_str: Raw version string from vllm.__version__.

    Returns:
        Parsed VLLMVersion or None if unparseable.

    Complexity: O(1).
    """
    # Strip common suffixes: .postN, .devN, +local, rcN
    cleaned = version_str.strip()

    # Split on '.' and take first three numeric parts
    parts = cleaned.split(".")
    numeric_parts: list[int] = []
    for part in parts:
        # Extract leading digits from each part (handles "3post1" -> 3)
        digits: list[str] = []
        for ch in part:
            if ch.isdigit():
                digits.append(ch)
            else:
                break
        if digits:
            numeric_parts.append(int("".join(digits)))
        if len(numeric_parts) == 3:
            break

    if len(numeric_parts) < 2:
        logger.warning(
            "[VLLM_COMPAT] Cannot parse vLLM version string",
            version_str=version_str,
        )
        return None

    # Pad missing patch to 0
    while len(numeric_parts) < 3:
        numeric_parts.append(0)

    return VLLMVersion(
        major=numeric_parts[0],
        minor=numeric_parts[1],
        patch=numeric_parts[2],
    )


def check_vllm_compatibility(version: VLLMVersion) -> list[str]:
    """Check a detected vLLM version against known compatibility constraints.

    Args:
        version: Detected vLLM version.

    Returns:
        List of warning messages. Empty list means no known issues.

    Complexity: O(K) where K = number of known breaking change entries.
    """
    warnings: list[str] = []
    version_pair = (version.major, version.minor)

    if version_pair not in SUPPORTED_VLLM_VERSIONS:
        warnings.append(
            f"vLLM {version} is not in the tested version set "
            f"{sorted(SUPPORTED_VLLM_VERSIONS)}. "
            f"TASFT patching may fail or produce incorrect results."
        )

    # Report known breaking changes for this specific version
    breaking = _KNOWN_BREAKING_CHANGES.get(version_pair)
    if breaking is not None:
        for change in breaking:
            warnings.append(
                f"vLLM {version}: known API change: {change}"
            )

    # Report breaking changes from versions between the user's version and
    # the baseline (0.4), so they know which shims are active
    for ver_pair in sorted(_KNOWN_BREAKING_CHANGES.keys()):
        if ver_pair <= version_pair and ver_pair != version_pair:
            for change in _KNOWN_BREAKING_CHANGES[ver_pair]:
                warnings.append(
                    f"vLLM {ver_pair[0]}.{ver_pair[1]} change active: {change}"
                )

    return warnings


class AttnMetadataAdapter:
    """Normalizes access to vLLM attn_metadata fields across versions.

    vLLM renamed several attn_metadata fields between 0.4 and 0.8:
      - is_prompt -> is_prefill (0.5)
      - num_prompt_tokens -> num_prefill_tokens (0.5)
      - seq_lens type changed from list[int] to Tensor (0.6)

    This adapter provides stable property access regardless of the underlying
    vLLM version.

    Preconditions:
        - metadata is a vLLM attn_metadata object (or None for safe defaults).
        - version is the detected VLLMVersion.

    Postconditions:
        - Properties return correct values or safe defaults.
        - No AttributeError is ever raised.

    Complexity: All properties are O(1).
    """

    __slots__ = ("_metadata", "_version")

    def __init__(self, metadata: Any, version: VLLMVersion) -> None:
        self._metadata = metadata
        self._version = version

    @property
    def is_prefill(self) -> bool:
        """Whether the current operation is a prefill (prompt processing) phase.

        Handles the is_prompt -> is_prefill rename across vLLM versions.
        Falls back to heuristic detection via prefill_metadata and
        num_prefill_tokens when neither attribute exists.

        Returns:
            True if in prefill phase, False if in decode phase.
            Defaults to True (conservative: enables sparse attention path).

        Complexity: O(1).
        """
        md = self._metadata
        if md is None:
            return True

        # vLLM >= 0.5: is_prefill
        if hasattr(md, "is_prefill"):
            return bool(md.is_prefill)

        # vLLM 0.4.x: is_prompt
        if hasattr(md, "is_prompt"):
            return bool(md.is_prompt)

        # Fallback: check prefill_metadata presence
        if hasattr(md, "prefill_metadata"):
            return md.prefill_metadata is not None

        # Fallback: check num_prefill_tokens
        if hasattr(md, "num_prefill_tokens"):
            return int(md.num_prefill_tokens) > 0

        # Conservative default: assume prefill to enable sparse path
        return True

    @property
    def num_prefill_tokens(self) -> int:
        """Number of tokens being processed in the prefill phase.

        Handles num_prompt_tokens -> num_prefill_tokens rename.

        Returns:
            Token count for prefill, or 0 if not in prefill or unknown.

        Complexity: O(1).
        """
        md = self._metadata
        if md is None:
            return 0

        # vLLM >= 0.5
        if hasattr(md, "num_prefill_tokens"):
            return int(md.num_prefill_tokens)

        # vLLM 0.4.x
        if hasattr(md, "num_prompt_tokens"):
            return int(md.num_prompt_tokens)

        return 0

    @property
    def seq_lens(self) -> list[int]:
        """Sequence lengths for each request in the batch.

        Handles the list[int] -> Tensor type change in vLLM 0.6.

        Returns:
            List of sequence lengths. Empty list if unavailable.

        Complexity: O(B) where B = batch size (for Tensor conversion).
        """
        md = self._metadata
        if md is None:
            return []

        raw = getattr(md, "seq_lens", None)
        if raw is None:
            # Some vLLM versions use seq_lens_tensor only
            raw = getattr(md, "seq_lens_tensor", None)

        if raw is None:
            return []

        # Handle both list[int] and Tensor
        if isinstance(raw, list):
            return raw

        # Tensor path: move to CPU and convert
        try:
            import torch

            if isinstance(raw, torch.Tensor):
                return raw.cpu().tolist()
        except ImportError:
            pass

        # Last resort: try to iterate
        try:
            return list(raw)
        except (TypeError, ValueError):
            return []


def get_attn_metadata_adapter(
    version: VLLMVersion,
) -> type[AttnMetadataAdapter]:
    """Return the AttnMetadataAdapter class configured for a vLLM version.

    The returned class can be instantiated with (metadata, version) to get
    normalized access to attn_metadata fields.

    Currently returns the single AttnMetadataAdapter class since it handles
    all known versions internally. This factory exists as an extension point
    for future versions that may require fundamentally different adapters.

    Args:
        version: Detected vLLM version.

    Returns:
        AttnMetadataAdapter class.

    Complexity: O(1).
    """
    # The adapter handles all known versions internally via runtime
    # attribute probing. If a future vLLM release changes the metadata
    # object fundamentally (not just renames), add a subclass here.
    _ = version  # Used for future dispatch; current adapter handles all versions.
    return AttnMetadataAdapter


def validate_worker_structure(worker: Any) -> list[str]:
    """Validate that a vLLM worker has the expected structure for TASFT patching.

    Checks that the worker exposes a model with attention layers that have
    the interface TASFT expects for monkey-patching.

    Args:
        worker: vLLM Worker instance.

    Returns:
        List of issues found. Empty list means the worker is patchable.

    Complexity: O(N) where N = total number of named modules in the model.
    """
    issues: list[str] = []

    # Locate the model within the worker
    worker_model = None
    if hasattr(worker, "model_runner") and hasattr(worker.model_runner, "model"):
        worker_model = worker.model_runner.model
    elif hasattr(worker, "model"):
        worker_model = worker.model

    if worker_model is None:
        issues.append(
            f"Worker type '{type(worker).__name__}' has no accessible model. "
            f"Expected 'model_runner.model' or 'model' attribute."
        )
        return issues

    # Check that the model is an nn.Module with named_modules
    import torch.nn as nn

    if not isinstance(worker_model, nn.Module):
        issues.append(
            f"Worker model is '{type(worker_model).__name__}', expected nn.Module."
        )
        return issues

    # Scan for attention modules
    attn_modules: list[tuple[str, nn.Module]] = []
    for name, module in worker_model.named_modules():
        cls_name = type(module).__name__
        if cls_name.endswith("Attention") and (
            hasattr(module, "qkv_proj") or hasattr(module, "q_proj")
        ):
            attn_modules.append((name, module))

    if not attn_modules:
        issues.append(
            f"No attention modules found in model '{type(worker_model).__name__}'. "
            f"Expected modules with class name ending in 'Attention' and "
            f"having 'qkv_proj' or 'q_proj' attributes."
        )
        return issues

    # Validate each attention module has the interface we need for patching
    for name, module in attn_modules:
        has_num_heads = hasattr(module, "num_heads") or hasattr(
            module, "num_attention_heads"
        )
        if not has_num_heads:
            issues.append(
                f"Attention module '{name}' ({type(module).__name__}) "
                f"has no 'num_heads' or 'num_attention_heads' attribute."
            )

        has_forward = hasattr(module, "forward") and callable(module.forward)
        if not has_forward:
            issues.append(
                f"Attention module '{name}' ({type(module).__name__}) "
                f"has no callable 'forward' method."
            )

    return issues


__all__ = [
    "AttnMetadataAdapter",
    "SUPPORTED_VLLM_VERSIONS",
    "VLLMVersion",
    "check_vllm_compatibility",
    "detect_vllm_version",
    "get_attn_metadata_adapter",
    "validate_worker_structure",
]
