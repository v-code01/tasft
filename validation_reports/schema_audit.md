# TASFT Schema & Config Validation Audit

**Auditor**: schema-auditor agent
**Date**: 2026-03-13
**Scope**: All Pydantic models, dataclass configs, YAML config files, field validators, datetime handling

---

## Files Audited

| File | Type | Verdict |
|------|------|---------|
| `tasft/bundle/bundle_schema.py` | Pydantic v2 models | **PASS with 2 findings** |
| `tasft/kernels/kernel_config.py` | Pydantic v2 models | **PASS with 3 findings** |
| `tasft/modules/tasft_attention.py` | frozen dataclass (GateConfig) | **PASS with 1 finding** |
| `tasft/bundle/export.py` | dataclass configs + runtime usage | **PASS** |
| `configs/llama3_8b_medqa.yaml` | Training config | **PASS** |
| `configs/qwen25_7b_code.yaml` | Training config | **PASS** |

---

## CRITICAL Findings

### C1: `datetime` behind `TYPE_CHECKING` requires `model_rebuild()` but library never calls it

**File**: `tasft/bundle/bundle_schema.py:18-19`
**Severity**: **P0 — runtime crash on deserialization**

```python
if TYPE_CHECKING:
    from datetime import datetime
```

`BundleManifest.created_at` is typed as `datetime`, but the import is guarded by `TYPE_CHECKING`. This means:

- **Construction works** when `export.py` passes a real `datetime` object (Pydantic accepts it dynamically).
- **`model_validate()` / `model_validate_json()` FAILS** when deserializing from JSON because Pydantic cannot resolve the `datetime` type annotation at schema-build time.

The workaround exists only in `tests/unit/test_export.py:40`:
```python
_ns: dict[str, type] = {"datetime": datetime}
BundleManifest.model_rebuild(_types_namespace=_ns)
```

But NO library code calls `model_rebuild()`. The `validate_bundle()` and `load_bundle_metadata()` methods in `export.py` call `BundleManifest.model_validate(...)` and `BundleManifest.model_validate_json(...)` — these will fail at runtime if `model_rebuild()` hasn't been called by the importing module.

**Fix**: Either:
1. **(Recommended)** Move `from datetime import datetime` outside `TYPE_CHECKING` in `bundle_schema.py`. It's a stdlib import — no circular import risk, no performance penalty.
2. Or add `model_rebuild()` call in `bundle_schema.py` at module level after class definitions.

---

## MAJOR Findings

### M1: Duplicate `LayerKernelConfig` and `KernelConfig` classes with divergent validation

**Files**: `tasft/bundle/bundle_schema.py` vs `tasft/kernels/kernel_config.py`

Two separate Pydantic models share the same class names but have **different validation rules**:

| Property | `bundle_schema.py` | `kernel_config.py` |
|----------|--------------------|--------------------|
| `extra` | `"forbid"` | **not set** (allows extra fields) |
| `layer_idx` validation | `Field(ge=0, le=256)` | **none** (any int accepted) |
| `block_size` valid values | `Field(gt=0, le=512)` (any positive int ≤512) | `frozenset({32, 64, 128})` (only 3 values) |
| `threshold_tau` | `Field(gt=0.0, lt=1.0)` | Manual validator `0.0 < v < 1.0` |
| `gate_loss_validation` | present | **absent** |
| `KernelConfig` model_validator | `validate_layer_consistency` (key↔idx, block_size match) | **none** |

This means:
- `kernel_config.py` accepts `block_size=256` but `bundle_schema.py` also accepts it — yet the kernel docstring says only `{32, 64, 128}` are valid.
- `kernel_config.py` silently accepts extra fields (no `extra="forbid"`).
- `kernel_config.py` has no cross-field consistency validation (layer key vs layer_idx, block_size consistency).

**Risk**: A config built via `kernel_config.KernelConfig` could have `block_size=256`, then when serialized to JSON and loaded via `bundle_schema.KernelConfig`, it would pass (since `bundle_schema` allows any `gt=0, le=512`). But the Triton kernel only supports `{32, 64, 128}`.

**Fix**: Consolidate into a single source of truth. Either:
- Delete `kernel_config.py` and use `bundle_schema.py` models everywhere (adding `_VALID_BLOCK_SIZES` constraint).
- Or make `kernel_config.py` import from `bundle_schema.py` and add inference-specific methods.

### M2: `kernel_config.py` missing `extra="forbid"` on both models

**File**: `tasft/kernels/kernel_config.py:40,84`

```python
model_config = ConfigDict(frozen=True)  # No extra="forbid"
```

Both `LayerKernelConfig` and `KernelConfig` in `kernel_config.py` accept arbitrary extra fields silently. This violates the project standard (all bundle_schema models use `extra="forbid"`) and can mask typos in config keys.

**Fix**: Add `extra="forbid"` to both:
```python
model_config = ConfigDict(frozen=True, extra="forbid")
```

### M3: `kernel_config.KernelConfig` has mutable default for `per_layer_config`

**File**: `tasft/kernels/kernel_config.py:88`

```python
per_layer_config: dict[int, LayerKernelConfig] = {}
```

While Pydantic v2 handles mutable defaults correctly (copies them per instance), the `bundle_schema.py` version does NOT have a default — it requires `per_layer_config` to be explicitly provided. This inconsistency means code using `kernel_config.KernelConfig()` with no per-layer config silently gets an empty dict, while `bundle_schema.KernelConfig()` raises a validation error.

---

## MINOR Findings

### m1: `GateConfig` uses `ValidationError` from `tasft.exceptions`, not Pydantic's

**File**: `tasft/modules/tasft_attention.py:60`

`GateConfig.__post_init__()` raises `tasft.exceptions.ValidationError`, not `pydantic.ValidationError`. This is intentional (GateConfig is a dataclass, not a Pydantic model) and correct — but callers that catch `pydantic.ValidationError` will miss these.

**Status**: Acceptable — documented here for awareness.

### m2: `GateConfig.default_threshold` uses `[0, 1]` (inclusive) while Pydantic models use `(0, 1)` (exclusive)

**File**: `tasft/modules/tasft_attention.py:70`

```python
if not 0.0 <= self.default_threshold <= 1.0:  # [0, 1] inclusive
```

But `bundle_schema.LayerKernelConfig.threshold_tau` uses `Field(gt=0.0, lt=1.0)` — exclusive on both ends. A `default_threshold` of exactly `0.0` or `1.0` would pass `GateConfig` but fail when used as `threshold_tau` in `LayerKernelConfig`.

**Fix**: Align to `(0, 1)` exclusive in `GateConfig`:
```python
if not 0.0 < self.default_threshold < 1.0:
```

### m3: `BundleManifest` missing validators on `total_size_bytes` and `num_layers`

**File**: `tasft/bundle/bundle_schema.py:134-135`

```python
total_size_bytes: int
num_layers: int
```

The docstring says `total_size_bytes >= 0` and `num_layers >= 0`, but no `Field(ge=0)` constraint is applied. Negative values would pass validation.

**Fix**:
```python
total_size_bytes: Annotated[int, Field(ge=0)]
num_layers: Annotated[int, Field(ge=0)]
```

### m4: `EvalSummary` has no field-level validators

**File**: `tasft/bundle/bundle_schema.py:150-166`

Fields like `mean_tokens_per_second`, `speedup_vs_dense`, `mean_sparsity` have no range validation. `mean_sparsity` should be in `[0, 1]`, `speedup_vs_dense` should be positive, `mean_tokens_per_second` should be positive.

### m5: `BundleManifest.training_args_hash` not validated as SHA256

**File**: `tasft/bundle/bundle_schema.py:132`

```python
training_args_hash: str  # SHA256 of training args JSON
```

The comment says it's SHA256 but no validator enforces the format. The `checksums` field has a validator but `training_args_hash` and `git_hash` do not.

### m6: `kernel_config.py` `per_layer_config` default `{}` is mutable in the type signature

**File**: `tasft/kernels/kernel_config.py:88`

Pydantic v2 handles this correctly at runtime (deep-copies defaults), so no bug, but `default_factory=dict` would be more explicit and pylint-clean.

---

## YAML Config Validation

### `configs/llama3_8b_medqa.yaml`

| Field | Value | Schema Constraint | Verdict |
|-------|-------|-------------------|---------|
| `gate.block_size` | 64 | `{32, 64, 128}` (kernel_config) / `gt=0, le=512` (bundle_schema) | **PASS** |
| `gate.default_threshold` | 0.5 | `(0, 1)` exclusive | **PASS** |
| `gate.num_layers` | 32 | Llama-3-8B has 32 layers | **PASS** |
| `objective.lambda_gate` | 0.1 | positive | **PASS** |
| `objective.beta_sparse` | 0.01 | positive | **PASS** |
| `objective.tau_target` | 0.8 | `(0, 1)` | **PASS** |
| `lora.r` | 16 | positive int | **PASS** |
| `lora.alpha` | 32 | positive int | **PASS** |

### `configs/qwen25_7b_code.yaml`

| Field | Value | Schema Constraint | Verdict |
|-------|-------|-------------------|---------|
| `gate.block_size` | 64 | `{32, 64, 128}` / `gt=0, le=512` | **PASS** |
| `gate.default_threshold` | 0.5 | `(0, 1)` exclusive | **PASS** |
| `gate.num_layers` | 28 | Qwen2.5-7B has 28 layers | **PASS** |
| `lora.r` | 32 | positive int | **PASS** |
| `lora.alpha` | 64 | positive int | **PASS** |
| `training.max_seq_length` | 4096 | positive int | **PASS** |

Both YAML configs are internally consistent and valid against schema constraints.

---

## Pydantic v1 vs v2 API Check

| Pattern | Status |
|---------|--------|
| Uses `BaseModel` from `pydantic` | v2 ✓ |
| Uses `ConfigDict` (not inner `Config` class) | v2 ✓ |
| Uses `field_validator` (not `@validator`) | v2 ✓ |
| Uses `model_validator` (not `@root_validator`) | v2 ✓ |
| Uses `model_dump_json()` (not `.json()`) | v2 ✓ |
| Uses `model_validate()` (not `.parse_obj()`) | v2 ✓ |
| Uses `model_validate_json()` (not `.parse_raw()`) | v2 ✓ |
| No `orm_mode` (replaced by `from_attributes`) | v2 ✓ |

**No Pydantic v1 API usage detected.** All code is clean Pydantic v2.

---

## Circular Import Check

- `bundle_schema.py` imports: `pydantic` only. No internal imports. **Clean.**
- `kernel_config.py` imports: `pydantic` only. No internal imports. **Clean.**
- `export.py` imports from `bundle_schema`, `exceptions`, `modules.attn_gate`, `observability.logging`. **No cycles.**
- `tasft_attention.py` imports from `exceptions`, `modules.attn_gate`, `types`. **No cycles.**

**No circular imports detected.**

---

## `from_gate_modules` Classmethod

The prior validation report (`implementation_completeness.md`) claims `KernelConfig.from_gate_modules()` exists. **It does not exist in either `kernel_config.py` or `bundle_schema.py`.** The equivalent logic lives in `BundleExporter._build_kernel_config()` as an instance method. This is not a code bug but a documentation inaccuracy in the prior report.

---

## Summary

| Severity | Count | Items |
|----------|-------|-------|
| **P0 Critical** | 1 | C1: `datetime` behind TYPE_CHECKING with no model_rebuild() in library |
| **P1 Major** | 3 | M1: Duplicate divergent schemas, M2: Missing extra=forbid, M3: Default inconsistency |
| **P2 Minor** | 6 | m1–m6: Threshold range mismatch, missing field validators, unvalidated hashes |
| **Info** | 2 | No v1 API issues, no circular imports |

### Recommended Priority

1. **Fix C1 immediately** — move `datetime` import out of TYPE_CHECKING in bundle_schema.py
2. **Fix M2** — add `extra="forbid"` to kernel_config.py models
3. **Fix m2** — align threshold range in GateConfig to (0,1) exclusive
4. **Fix m3** — add `Field(ge=0)` to total_size_bytes and num_layers
5. **Plan M1** — consolidate duplicate KernelConfig/LayerKernelConfig into single source
