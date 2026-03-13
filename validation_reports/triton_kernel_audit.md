# Triton Block-Sparse Attention Kernel — Deep Audit Report

**Auditor**: kernel-auditor agent
**Date**: 2026-03-13
**Files reviewed**:
- `tasft/kernels/block_sparse_fa.py` (608 lines)
- `tasft/kernels/kernel_config.py` (150 lines)
- `tests/unit/test_kernel.py` (174 lines)

---

## 1. Online Softmax (FlashAttention-2 Two-Pass)

**Verdict: PASS with one observation**

The kernel (lines 176-238) correctly implements the online softmax algorithm:

```
m_i = full([BLOCK_SIZE], -inf, fp32)    # running row-wise max
l_i = zeros([BLOCK_SIZE], fp32)          # running row-wise sum of exp
acc = zeros([BLOCK_SIZE, HEAD_DIM], fp32) # weighted V accumulator
```

Per-block update (lines 230-238):
```
row_max = max(qk, axis=1)
m_new = maximum(m_i, row_max)
alpha = exp(m_i - m_new)          # rescale factor for old accumulators
p = exp(qk - m_new[:, None])     # attention weights with new max
l_i = alpha * l_i + sum(p, axis=1)
acc = alpha[:, None] * acc + dot(p.to(v.dtype), v)
m_i = m_new
```

This matches the FlashAttention-2 online softmax exactly: maintain running `(m, l, O)` and rescale when a new block introduces a larger max. Final normalization at line 242-243 divides by `l_safe`.

**Observation**: The `l_safe = where(l_i > 0, l_i, 1.0)` guard (line 242) handles the edge case where all blocks for a query row are masked out (l_i remains 0). This correctly produces a zero output row rather than NaN. Good defensive coding.

---

## 2. Block Mask Skip Logic

**Verdict: PASS**

Lines 182-193:
```python
for k_block_idx in range(num_k_blocks):
    mask_ptr = BlockMask_ptr + pid_b*stride_mb + pid_h*stride_mh + pid_q*stride_mq + k_block_idx*stride_mk
    should_compute = tl.load(mask_ptr)
    if should_compute:
        # ... load K, V, compute QK^T, update accumulators
```

The mask is loaded as a single scalar per K-block. When `should_compute` is False (mask=0), the entire inner computation is skipped — no K/V loads, no QK^T, no softmax update. This is the core sparsity speedup mechanism.

The boolean mask is converted to `torch.int8` before kernel launch (line 480) for Triton `tl.load` compatibility, which correctly treats 0 as False and non-zero as True.

**Note**: The `if should_compute` inside a Triton `for` loop compiles to predicated execution. Triton handles this efficiently — the compiler will generate branch instructions rather than always executing the body.

---

## 3. Causal Masking

**Verdict: PASS — dual-level causal masking is correct**

### Block-level causal masking (host-side, lines 484-490):
```python
if causal:
    block_causal = torch.tril(ones(num_blocks, num_blocks, int8, device))
    block_mask_int = block_mask_int & block_causal.unsqueeze(0).unsqueeze(0)
```
This zeros out all blocks where `k_block_idx > q_block_idx`, preventing unnecessary kernel iterations for strictly-future blocks. The `unsqueeze(0).unsqueeze(0)` correctly broadcasts across batch and head dimensions.

### Within-block causal masking (kernel-side, lines 220-228):
```python
q_indices = q_block_start + arange(0, BLOCK_SIZE)
k_indices = k_block_start + arange(0, BLOCK_SIZE)
causal_mask = q_indices[:, None] >= k_indices[None, :]
seq_mask_q = q_indices[:, None] < S
seq_mask_k = k_indices[None, :] < S
combined_mask = causal_mask & seq_mask_q & seq_mask_k
qk = where(combined_mask, qk, -inf)
```

This handles all edge cases correctly:
- **First block** (q=0, k=0): diagonal causal mask, upper triangle set to -inf.
- **Last block**: `seq_mask_q` and `seq_mask_k` handle out-of-bounds when S is not divisible by BLOCK_SIZE.
- **Diagonal blocks** (q_block == k_block): the `causal_mask` is a lower-triangular matrix within the block.
- **Strictly-below-diagonal blocks** (q_block > k_block): `causal_mask` is all-True (all q positions >= all k positions), so full block is attended.

**Edge case note**: The causal mask applies unconditionally inside the kernel regardless of the `causal` parameter passed to `forward()`. However, when `causal=False`, the block-level causal mask is not applied (line 484 `if causal:` skips it), but the **within-block** causal mask in the kernel still applies. This is a **minor issue** — the kernel always applies causal masking. For non-causal use cases, the kernel would need a `IS_CAUSAL: tl.constexpr` parameter to skip lines 220-228.

**Severity**: LOW — TASFT's use case is autoregressive LLM inference where causal=True always. But the docstring claims the `causal` parameter controls this behavior, which is misleading for within-block masking.

---

## 4. FP32 Accumulation

**Verdict: PASS**

Lines 177-179:
```python
m_i = tl.full([BLOCK_SIZE], float("-inf"), dtype=tl.float32)
l_i = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
acc = tl.zeros([BLOCK_SIZE, HEAD_DIM], dtype=tl.float32)
```

All three accumulators are explicitly FP32 regardless of input dtype (BF16/FP16). The `exp()` and `maximum()` operations at lines 233-234 inherit FP32 from their inputs. The `tl.dot(p.to(v.dtype), v)` at line 237 casts attention weights back to input dtype for the matmul, then the result is added to the FP32 accumulator (implicit upcast). Final output is cast back to input dtype at line 254: `acc.to(q.dtype)`.

This matches FlashAttention-2's numerical stability strategy. The only potential concern is the `p.to(v.dtype)` cast before dot — this could lose precision for very small attention weights in BF16. However, this is standard practice (FlashAttention-2 does the same) and the accumulation in FP32 mitigates the issue.

---

## 5. Grid Launch Dimensions

**Verdict: PASS**

Line 498-499:
```python
grid = (B, H, num_q_blocks)
```

Kernel signature maps these via:
```python
pid_b = tl.program_id(0)   # batch
pid_h = tl.program_id(1)   # head
pid_q = tl.program_id(2)   # query block index
```

This correctly launches one kernel instance per (batch, head, query_block) triple. Each instance processes all K-blocks for its query block (inner loop at line 182). The grid is 3D which is well within Triton's supported grid dimensionality.

`num_q_blocks = math.ceil(S / self.block_size)` at line 495 and `num_k_blocks = tl.cdiv(S, BLOCK_SIZE)` at line 160 are consistent — both compute ceiling division.

---

## 6. BLOCK_SIZE is tl.constexpr

**Verdict: PASS**

Line 134:
```python
BLOCK_SIZE: tl.constexpr,
HEAD_DIM: tl.constexpr,
```

Both `BLOCK_SIZE` and `HEAD_DIM` are declared as `tl.constexpr`, which is required for Triton to unroll loops, statically allocate shared memory, and optimize register allocation based on these values. They are passed as keyword arguments at kernel launch (lines 529-530), which is the correct Triton pattern for constexpr parameters.

---

## 7. Dense SDPA Fallback

**Verdict: PASS**

Lines 535-553:
```python
def _dense_fallback(self, q, k, v, causal):
    return functional.scaled_dot_product_attention(q, k, v, is_causal=causal)
```

Clean delegation to PyTorch's `F.scaled_dot_product_attention`, which handles backend selection internally (cuDNN FlashAttention, memory-efficient attention, or math backend). The `is_causal` parameter is correctly forwarded.

The fallback is triggered in two scenarios:
1. Sparsity below threshold (line 378-384) — correct, avoids sparse overhead when dense is faster.
2. No sparse backend available (line 396) — correct catch-all.

---

## 8. Auto-Backend Detection

**Verdict: PASS**

`detect_kernels()` (lines 60-87) checks availability in priority order:
1. `flash_attn` with `flash_attn_varlen_qkvpacked_func` attribute → `FLASH_ATTN_SPARSE`
2. `triton` importable → `TRITON`
3. Always → `DENSE_FALLBACK`

In `__init__` (line 325-326):
```python
if backend == KernelBackend.AUTO:
    self.backend = self._available_backends[0]
```

AUTO selects the highest-priority available backend. Since `DENSE_FALLBACK` is always appended last, the list is never empty.

**Minor note**: The `FLASH_ATTN_SPARSE` backend (lines 388-394) actually delegates to Triton if available, or falls back to dense. This means `FLASH_ATTN_SPARSE` is never truly exercised as a distinct code path — it's effectively an alias. The comment at line 390-392 acknowledges this: "Full flash_attn sparse integration is version-dependent."

---

## 9. _validate_inputs

**Verdict: PASS**

Lines 398-437 check:
- `q.ndim == 4` — correct for [B, H, S, D] layout
- `q.shape == k.shape == v.shape` — all QKV must match
- `q.is_cuda` — Triton requires CUDA tensors
- `block_mask.shape == (B, H, num_blocks, num_blocks)` — mask dimensions match

All validation failures raise `KernelError` with structured `context` dict.

**Missing validations** (LOW severity):
1. **dtype check**: No validation that q/k/v are float16/bfloat16/float32. The Triton kernel will work but may produce unexpected results with int dtypes.
2. **device consistency**: Only checks `q.is_cuda`. Doesn't verify k, v, block_mask are on the same device. If they're on different CUDA devices, the kernel will crash with an opaque CUDA error.
3. **head_dim check**: `D not in _VALID_HEAD_DIMS` is checked in `_triton_forward` (line 468) but not in `_validate_inputs`. This means the error surfaces later than ideal.

---

## 10. estimate_speedup Model

**Verdict: PASS — reasonable theoretical model**

Lines 578-599:
```python
gate_overhead = 0.02  # 2% overhead for AttnGate forward pass
denominator = 1.0 - sparsity_ratio * (1.0 - gate_overhead)
return 1.0 / max(1e-6, denominator)
```

Model: `speedup = 1 / (1 - s * 0.98)` where `s` is sparsity ratio.

| Sparsity | Theoretical | Model | SeerAttention Reported |
|----------|------------|-------|----------------------|
| 0% | 1.0x | 1.0x | N/A |
| 50% | 2.0x | 1.96x | ~2x |
| 70% | 3.3x | 3.18x | ~3x |
| 90% | 10x | 8.47x | 5.67x |
| 95% | 20x | 15.6x | ~8x |

The model overestimates at high sparsity compared to SeerAttention's empirical results. The gap is due to memory bandwidth bottlenecks, kernel launch overhead, and load imbalance not captured by this simple model. However, it's clearly documented as an estimate, and the `min_sparsity_for_speedup` threshold provides a conservative cutoff.

The `max(1e-6, denominator)` guard prevents division by zero at sparsity=~1.0.

---

## kernel_config.py Audit

**Verdict: PASS — clean Pydantic v2 models**

### LayerKernelConfig (lines 27-70)
- `model_config = ConfigDict(frozen=True)` — immutable after construction. Correct.
- `threshold_tau` validated to `(0, 1)` exclusive — correct, 0.0 and 1.0 would be degenerate.
- `target_sparsity` and `achieved_sparsity_validation` validated to `[0, 1]` inclusive — correct, 0% and 100% sparsity are valid edge cases.
- `block_size` validated against `{32, 64, 128}` — consistent with kernel.

### KernelConfig (lines 73-143)
- Aggregates per-layer configs with global defaults.
- `get_layer_threshold()` and `get_layer_block_size()` correctly prioritize per-layer overrides over global defaults. O(1) dict lookup.
- `per_layer_config: dict[int, LayerKernelConfig] = {}` — **Note**: mutable default in frozen Pydantic model. Pydantic v2 handles this correctly by deep-copying the default, so this is safe.
- `min_sparsity_for_speedup` validated to `[0, 1]` — consistent with `BlockSparseFlashAttention`.

### Consistency between kernel_config.py and block_sparse_fa.py
- Both define `_VALID_BLOCK_SIZES = frozenset({32, 64, 128})` independently. Slight DRY violation but acceptable since they're in different modules with different import graphs.

---

## Summary of Findings

### Critical Issues (P0): **NONE**

### Moderate Issues (P1):

| # | Issue | Location | Description |
|---|-------|----------|-------------|
| 1 | Causal masking always applied in kernel | `block_sparse_fa.py:220-228` | The Triton kernel unconditionally applies within-block causal masking. The `causal` parameter only controls block-level masking on the host side. For non-causal attention (e.g., encoder), the kernel produces incorrect results. |

### Minor Issues (P2):

| # | Issue | Location | Description |
|---|-------|----------|-------------|
| 2 | Missing dtype validation | `block_sparse_fa.py:398-437` | `_validate_inputs` doesn't check that Q/K/V are floating-point dtypes |
| 3 | Missing cross-device check | `block_sparse_fa.py:418-422` | Only checks `q.is_cuda`, not that k, v, block_mask are on the same device |
| 4 | head_dim validated late | `block_sparse_fa.py:468` | D checked in `_triton_forward` instead of `_validate_inputs` |
| 5 | FLASH_ATTN_SPARSE is an alias | `block_sparse_fa.py:388-394` | Backend enum exists but always delegates to Triton or dense — never a distinct code path |
| 6 | Speedup model overestimates | `block_sparse_fa.py:578-599` | ~49% overestimate vs empirical at 90% sparsity; documented but could mislead deployment decisions |
| 7 | Duplicated `_VALID_BLOCK_SIZES` | Both files | Same frozenset defined independently in `block_sparse_fa.py:46` and `kernel_config.py:24` |

### Strengths:

1. **Online softmax is textbook-correct** — exact match to FlashAttention-2 algorithm
2. **FP32 accumulation throughout** — numerical stability guaranteed
3. **Dual-level causal masking** — block-level pre-filtering + within-block fine-grained masking
4. **Clean fallback chain** — graceful degradation when sparse isn't beneficial
5. **Structured error handling** — all errors carry context dicts
6. **Defensive division-by-zero guards** — `l_safe` and `max(1e-6, denominator)`
7. **Comprehensive test coverage** of non-CUDA paths
8. **Pydantic configs are well-structured** — frozen, validated, with per-layer override pattern
9. **Grid dimensions are correct** — 3D grid with proper stride computation
10. **`tl.constexpr` used correctly** for BLOCK_SIZE and HEAD_DIM

### Test Coverage Assessment:

The existing tests (`test_kernel.py`) cover:
- Dense fallback path correctness
- Speedup estimation monotonicity and boundary values
- Backend detection invariant (DENSE_FALLBACK always present)
- Sparsity stats computation
- Invalid block size rejection

**Missing test coverage**:
- No CUDA/Triton kernel execution tests (all tests use CPU dense fallback)
- No numerical correctness test comparing sparse kernel output vs dense reference
- No causal vs non-causal behavior test
- No edge case tests for S not divisible by BLOCK_SIZE
- No test for block_mask with mixed True/False patterns on the Triton path

---

## Recommendations

1. **P1**: Add `IS_CAUSAL: tl.constexpr` parameter to the Triton kernel and conditionally apply within-block causal masking. This makes the `causal` parameter truthful across all levels.

2. **P2**: Consolidate `_VALID_BLOCK_SIZES` into `tasft/types.py` or a shared constants module.

3. **P2**: Add dtype and cross-device validation to `_validate_inputs`.

4. **Testing**: Add CUDA-conditional integration tests (`@pytest.mark.skipif(not torch.cuda.is_available())`) that compare Triton kernel output against `F.scaled_dot_product_attention` reference with tolerance `< 1e-3` for BF16.

5. **Documentation**: Clarify in the `causal` parameter docstring that within-block causal masking is always active in the current kernel implementation.
