# vLLM Integration Audit Report

**File**: `tasft/inference/vllm_patch.py` (539 lines)
**Context**: `tasft/inference/tasft_model.py` (825 lines)
**Auditor**: vllm-auditor
**Date**: 2026-03-13

---

## Executive Summary

The vLLM patch module is well-structured with proper thread-safety, idempotency guards, and clean separation between prefill (sparse) and decode (dense) paths. However, there is **one critical bug** in the decode fallback path and several medium-severity issues that should be addressed before production use.

| Severity | Count |
|----------|-------|
| 🔴 Critical | 1 |
| 🟡 Medium | 3 |
| 🟢 Low | 3 |
| ✅ Correct | 6 |

---

## 1. Monkey-Patch Target

**Verdict: ✅ Correct — well-designed approach**

The patch does NOT target `PagedAttention` directly (which would be fragile across vLLM versions). Instead, it:

1. Extracts per-layer attention modules via `_extract_vllm_attention_modules()` (line 280)
2. Identifies them by: `"Attention" in cls_name` AND `hasattr(module, "qkv_proj") or hasattr(module, "q_proj")` (lines 300-302)
3. Replaces each module's `.forward` with a `TASFTvLLMAttentionBackend`-backed closure (lines 429-463)

This per-module approach is more resilient than patching a single class method because it works regardless of whether vLLM uses `LlamaAttention`, `MistralAttention`, or custom attention classes.

**Minor concern (🟢 Low):** The string check `"Attention" in cls_name` could false-match non-attention modules (e.g., a hypothetical `AttentionNormLayer`). The `hasattr` guard mitigates this, but a stricter check like `cls_name.endswith("Attention")` would be safer (this is what `tasft_model.py:137` uses for HF models).

---

## 2. _patch_lock Coverage

**Verdict: ✅ Correct — all state mutations under lock**

| State Mutation | Under Lock? | Location |
|---|---|---|
| `_patch_applied` read (guard) | ✅ | line 355 |
| `_tasft_original_forward` assignment | ✅ | line 425 |
| `_tasft_backend` assignment | ✅ | line 426 |
| `vllm_attn.forward` replacement | ✅ | line 460 |
| `_patch_applied = True` | ✅ | line 474 |
| `_patch_applied = False` (unpatch) | ✅ | line 515 |
| `_tasft_original_forward` deletion | ✅ | line 511 |
| `_tasft_backend` deletion | ✅ | line 513 |

The entire body of both `patch_vllm_attention` and `unpatch_vllm_attention` runs inside `with _patch_lock:`. No state mutations escape the lock.

---

## 3. unpatch_vllm_attention Restoration

**Verdict: 🟡 Medium — mostly correct, one silent failure path**

**Correct behavior:**
- Restores `module.forward = module._tasft_original_forward` (line 510)
- Deletes `_tasft_original_forward` (line 511)
- Deletes `_tasft_backend` if present (lines 512-513)
- Sets `_patch_applied = False` (line 515)

**Issue: Silent return on unrecognized worker (line 504-506)**

```python
if hasattr(vllm_worker, "model_runner"):
    worker_model = vllm_worker.model_runner.model
elif hasattr(vllm_worker, "model"):
    worker_model = vllm_worker.model
else:
    return  # ← BUG: returns without setting _patch_applied = False
```

If the worker object doesn't have `model_runner` or `model`, `unpatch_vllm_attention` returns without:
- Setting `_patch_applied = False`
- Logging a warning
- Raising an error

This leaves the system in a permanently "patched" state where `is_patched()` returns `True` but the patch can never be removed.

**Fix:** Should either raise `InferenceError` or at minimum set `_patch_applied = False` and log a warning.

---

## 4. Version Compatibility

**Verdict: ✅ Correct — robust multi-version detection**

### Worker model extraction (lines 362-371)
| vLLM Version | Access Pattern | Supported |
|---|---|---|
| >= 0.5.x | `vllm_worker.model_runner.model` | ✅ |
| 0.4.x | `vllm_worker.model` | ✅ |
| < 0.4.0 | N/A | ✅ Raises `InferenceError` |

### Prefill detection (`_is_prefill_phase`, lines 253-277)
| vLLM Version | Attribute | Supported |
|---|---|---|
| >= 0.4.0 | `attn_metadata.is_prompt` | ✅ |
| 0.4.x intermediate | `attn_metadata.prefill_metadata` | ✅ |
| >= 0.5.0 | `attn_metadata.num_prefill_tokens` | ✅ |
| Unknown | Defaults to `True` (prefill) | ✅ Conservative |

The version detection is duck-typed via `hasattr` checks, which is the correct approach for a library that monkey-patches internals. No explicit version string parsing needed.

---

## 5. Patched Forward — Gate Mask Application

**Verdict: 🔴 CRITICAL BUG in decode path; prefill path correct**

### Prefill path (lines 163-178): ✅ Correct
```python
gate_output = self.gate(q, k, threshold=self.threshold_tau)
if gate_output.sparsity_ratio >= self.min_sparsity_for_speedup:
    kernel = self._get_kernel()
    attn_output = kernel.forward(q, k, v, gate_output.hard_mask, causal=True)
```
- Gate receives properly shaped Q, K tensors
- Hard mask from gate output drives sparse kernel
- Dense fallback at low sparsity is correct with softmax in float32

### Decode path (lines 156-161): 🔴 **CRITICAL — `_dense_attention_flat` ignores KV cache**

```python
if kv_cache is not None and attn_metadata is not None:
    is_prefill = _is_prefill_phase(attn_metadata)
    if not is_prefill:
        return self._dense_attention_flat(query, key, value, kv_cache, attn_metadata)
```

`_dense_attention_flat` (lines 180-223) accepts `kv_cache` and `attn_metadata` as arguments but **completely ignores them**. It computes attention between only the current step's Q, K, V tensors:

```python
# Line 218-221: Only uses the passed key/value, not the cached history
attn_weights = torch.einsum("thd,shd->ths", q * scale, k)
```

During autoregressive decode, `query` has 1 token and `key`/`value` have 1 token. This computes self-attention of 1 token against 1 token — **it does not attend to the cached context at all**. The output is meaningless.

**Impact:** Any vLLM decode request (autoregressive generation) after the first prefill will produce garbage output.

**Root cause:** The function should delegate to the original vLLM attention forward (which knows how to use PagedAttention's paged KV cache) rather than reimplementing attention manually.

**Fix:** Replace the decode fallback with:
```python
if not is_prefill:
    # Single-token decode: no sparsity benefit, use original vLLM attention
    # which properly handles PagedAttention KV cache
    return self._original_vllm_forward(query, key, value, kv_cache, attn_metadata, **kwargs)
```

This requires `TASFTvLLMAttentionBackend` to store a reference to the original forward method, or the decode path should be handled at the patch-application level (route decode calls to the original forward, prefill calls to the TASFT backend).

---

## 6. GQA Support — repeat_kv

**Verdict: 🟢 Low — correct but missing validation**

GQA expansion appears in two places:

### TASFTvLLMAttentionBackend.forward (lines 150-153):
```python
if self.num_kv_heads < self.num_heads:
    n_rep = self.num_heads // self.num_kv_heads
    k = k.repeat_interleave(n_rep, dim=1)
    v = v.repeat_interleave(n_rep, dim=1)
```

### _dense_attention_flat (lines 211-214): Same pattern.

**Correctness:** `repeat_interleave(n_rep, dim=1)` is the standard GQA expansion. Each KV head is repeated `n_rep` times to match Q heads.

**Missing validation:** No assertion that `num_heads % num_kv_heads == 0`. Integer division `//` would silently truncate if this invariant is violated (e.g., 32 Q heads / 6 KV heads = 5, losing 2 heads). In practice, all known architectures (LLaMA-2, Mistral, Gemma) guarantee even divisibility, but a defensive check would be prudent.

**Non-uniform head counts across layers:** Handled correctly — each `TASFTvLLMAttentionBackend` instance gets its own `num_heads`/`num_kv_heads` from the vLLM module (lines 398-409).

---

## 7. is_patched() Lock Acquisition

**Verdict: ✅ Correct — verified**

```python
def is_patched() -> bool:
    with _patch_lock:           # line 530
        return _patch_applied   # line 531
```

Acquires lock before reading `_patch_applied`. The docstring correctly notes this is for PEP 703 (GIL-free Python) safety. Under the current GIL, a simple read of a bool is atomic, but the lock provides forward-compatibility.

---

## 8. Edge Cases

### Patch called twice
**Verdict: ✅ Correct — idempotent**

```python
with _patch_lock:
    if _patch_applied:
        logger.info("[VLLM_PATCH] Patch already applied, skipping")
        return
```

Second call logs and returns immediately. No double-patching, no error.

### Unpatch called without patch
**Verdict: ✅ Correct — idempotent**

```python
with _patch_lock:
    if not _patch_applied:
        logger.info("[VLLM_UNPATCH] No patch applied, skipping")
        return
```

No-op when no patch is active. Clean.

### Patch → unpatch → patch (re-patch)
**Verdict: ✅ Correct**

After unpatch deletes `_tasft_original_forward` and sets `_patch_applied = False`, a subsequent patch call will go through the full application path again, storing new backup refs.

---

## 9. Memory — Reference Leaks

**Verdict: 🟡 Medium — mostly clean, one concern**

### On patch application:
- `vllm_attn._tasft_original_forward` holds a reference to the original bound method
- `vllm_attn._tasft_backend` holds the `TASFTvLLMAttentionBackend` instance
- The patched forward is a `types.MethodType` wrapping a lambda → closure → backend

### On unpatch:
- `_tasft_original_forward` is `del`eted (line 511) ✅
- `_tasft_backend` is `del`eted (line 513) ✅
- `module.forward` is restored to original (line 510), releasing the closure chain ✅

**Concern: Forward method binding complexity (lines 458-463)**

```python
import types
vllm_attn.forward = types.MethodType(
    lambda self, *args, _pf=_make_patched_forward(backend), **kw: _pf(*args, **kw),
    vllm_attn,
)
```

This creates a 3-layer indirection: `MethodType(lambda → _pf → _patched_forward → _backend.__call__)`. The lambda captures `_pf` as a default argument (correct — avoids late-binding), and `MethodType` binds `self` (which the lambda accepts but ignores).

While this works correctly, it:
1. Makes stack traces harder to read (3 anonymous frames per attention call)
2. Could be simplified to `vllm_attn.forward = _make_patched_forward(backend)` if the calling convention doesn't require a bound method

No actual memory leak, but the complexity is unnecessary.

---

## 10. Additional Findings

### 🟡 Medium: Global `_patch_applied` prevents multi-worker patching

`_patch_applied` is a module-level bool. In tensor-parallel vLLM deployments with multiple workers per process, patching worker 1 sets `_patch_applied = True`, causing `patch_vllm_attention(worker_2)` to silently skip.

**Fix:** Track patch state per-worker, either via a set of patched worker IDs or by checking the worker's modules directly for `_tasft_original_forward`.

### 🟢 Low: `_kernel` lazy initialization not thread-safe

`TASFTvLLMAttentionBackend._get_kernel()` (lines 96-105) uses a check-then-set pattern without synchronization:
```python
if self._kernel is None:
    self._kernel = BlockSparseFlashAttention(...)
```

Under concurrent forward calls, two threads could both see `None` and construct separate kernel instances. Not a correctness issue (kernels are stateless), but wastes memory. Acceptable given vLLM's single-thread-per-worker model.

---

## Summary of Required Actions

| # | Severity | Issue | Location | Action |
|---|----------|-------|----------|--------|
| 1 | 🔴 Critical | `_dense_attention_flat` ignores KV cache — decode produces garbage | lines 180-223 | Delegate decode to original vLLM forward |
| 2 | 🟡 Medium | `unpatch_vllm_attention` silently fails on unknown worker type | line 506 | Raise error or force `_patch_applied = False` |
| 3 | 🟡 Medium | Global `_patch_applied` prevents multi-worker patching | line 41 | Track per-worker patch state |
| 4 | 🟡 Medium | Forward binding has unnecessary 3-layer indirection | lines 458-463 | Simplify to direct function assignment |
| 5 | 🟢 Low | No validation `num_heads % num_kv_heads == 0` | line 151 | Add assertion in `__init__` |
| 6 | 🟢 Low | `"Attention" in cls_name` could false-match | line 300 | Use `endswith("Attention")` |
| 7 | 🟢 Low | `_get_kernel` not thread-safe (benign) | line 101 | Document or add lock |

---

## Integration with tasft_model.py

The `TASFTInferenceModel` (tasft_model.py) correctly:
- Stores gates in `nn.ModuleDict` with string keys (line 393) — matches `vllm_patch.py`'s `tasft_model.gates[str(layer_idx)]` access pattern (line 393)
- Exposes `kernel_config` with `get_layer_threshold()` and `get_layer_block_size()` methods used by the patch
- Uses `_SparseAttentionWrapper` for HF-native inference, which is a separate (correct) implementation from the vLLM backend — no code sharing conflict

The vLLM patch and HF-native inference paths are cleanly separated, with shared AttnGate and KernelConfig but independent attention forward implementations.
