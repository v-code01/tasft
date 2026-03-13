# Evaluation Harness Audit Report

**Auditor**: eval-auditor agent
**Date**: 2026-03-13
**Scope**: `tasft/eval/task_eval.py`, `tasft/eval/gate_quality.py`, `tasft/eval/throughput_bench.py`, `tests/integration/test_eval_harness.py`
**Verdict**: **PASS with 2 minor issues, 1 moderate statistical concern, 1 test gap**

---

## 1. task_eval.py

### 1.1 MedQA Prompt Formatting

**Status**: ACCEPTABLE with caveat

The code attempts multiple field names for the bigbio/med_qa schema:
- `item.get("question", item.get("text", ""))` (line 305)
- `item.get("options", item.get("answer_options", []))` (line 306)
- `item.get("answer_idx", item.get("correct_answer", 0))` (line 307)

The fallback approach handles schema variations robustly. The MCQ formatting (`A) ... B) ... Answer:`) is standard for log-prob MCQ evaluation.

**Caveat**: The BigBIO med_qa schema may use different field names (`choices` vs `options`, `answer` vs `answer_idx`). The fallback to `answer_idx=0` (line 324) when parsing fails silently defaults to "A" — this could mask schema mismatches. A warning log when falling back to default would be safer.

### 1.2 Log-Prob Scoring at Last Token Position

**Status**: CORRECT

```python
last_positions = attention_mask.sum(dim=1) - 1  # line 342
token_logits = logits[i, int(last_pos), :]       # line 346
log_probs = torch.log_softmax(token_logits, dim=-1)  # line 349
```

- `attention_mask.sum(dim=1) - 1` correctly identifies the last non-padding token index.
- Logits at that position predict the *next* token (the answer letter after "Answer:"), which is the correct position for MCQ scoring.
- `log_softmax` then `argmax` over option token IDs is the standard approach.
- Option tokens are pre-tokenized once (lines 283-291) with a check for empty encodings.

No off-by-one error detected.

### 1.3 HumanEval Sandbox

**Status**: CORRECT

- Uses `subprocess.run` with `timeout=10` seconds (line 545-549).
- `capture_output=True` prevents stdout/stderr leakage.
- `check=False` + checking `returncode == 0` is correct.
- `TimeoutExpired` and `OSError` are both caught and return `False`.
- `tempfile.NamedTemporaryFile(delete=True)` — the file remains accessible during the `with` block since `subprocess.run` is called inside it (file is deleted on `__exit__`, not before).

**Note**: No network isolation or resource limits beyond the timeout. For production use, consider `seccomp` or container sandboxing. Acceptable for research evaluation.

### 1.4 pass@k Estimator

**Status**: CORRECT — verified mathematically

The unbiased estimator `1 - C(n-c, k) / C(n, k)` is computed in log-space:

```python
log_ratio += math.log(n - c - i) - math.log(n - i)  # line 161
```

Verification:
- `C(n-c, k) / C(n, k) = prod_{i=0}^{k-1} (n-c-i)/(n-i)`
- `log(prod) = sum(log(n-c-i) - log(n-i))` — matches implementation.
- Edge case `n - c < k`: returns `1.0` (line 155) — correct, guaranteed at least one correct sample.
- `c = 0`: `log_ratio = sum(log(n-i) - log(n-i)) = 0`, returns `1 - 1 = 0.0` — correct.
- `c = n`: `n - c = 0 < k` for any `k >= 1`, returns `1.0` — correct.
- Overflow-safe via log-space computation — correct.

### 1.5 Wilson Score Interval

**Status**: CORRECT

Formula at lines 131-137 matches the standard Wilson score:
```
centre = (p + z²/2n) / (1 + z²/n)
margin = z / (1 + z²/n) * sqrt(p(1-p)/n + z²/4n²)
```

- `n == 0` returns `(0.0, 1.0)` — reasonable degenerate case.
- `max(0.0, ...)` and `min(1.0, ...)` clamp bounds to [0, 1].

### 1.6 Cohen's d and t-test in compare_models

**Status**: MODERATE CONCERN — inconsistency between t-test and pooled std

- **t-test**: Uses Welch's (unequal variance) via `equal_var=False` (line 611).
- **Cohen's d**: Uses equal-variance pooled std formula (line 619): `((n_b-1)*var_b + (n_t-1)*var_t) / (n_b+n_t-2)`.

This inconsistency is common in practice (Cohen's d is always defined with the pooled std, even when variances differ), but for rigor, Glass's delta or Hedges' g with Welch correction would be more appropriate when using Welch's t-test.

**Impact**: Low — Cohen's d is used for effect size reporting, not decision-making. The `within_target` check uses raw accuracy delta, not effect size.

- **Edge case**: `pooled_var == 0` (all scores identical) → falls back to `1e-10` denominator (line 620), preventing division by zero. Correct.

---

## 2. gate_quality.py

### 2.1 Paired t-test

**Status**: CORRECT with one concern

`stats.ttest_rel(posthoc_kls, cotrained_kls)` (line 543):
- Uses paired t-test, correct for matched layers.
- Degrees of freedom = n_layers - 1 (handled internally by scipy).
- Requires `len(common_layers) >= 2` (line 525-533), which is the minimum for `ttest_rel` to compute a std of differences.

**Concern — Two-tailed p-value for one-directional hypothesis**: The hypothesis is directional (`cotrained KL < posthoc KL`), but `ttest_rel` returns a two-tailed p-value. The code then checks:
```python
significant = float(p_value) < 0.05      # line 552
hypothesis_supported = kl_improvement > 0 and significant  # line 553
```

This combines direction check with two-tailed significance. For a proper one-tailed test at α=0.05, the two-tailed p-value should be halved before comparing to 0.05 (or equivalently, compare the two-tailed p to 0.10). The current approach is **overly conservative** — it effectively tests at α=0.025 one-tailed.

**Impact**: May fail to detect real improvements when they exist (false negatives). Not a correctness bug — it's conservative, not liberal.

### 2.2 KL Divergence Computation

**Status**: CORRECT

`_kl_divergence_block` (lines 112-138):
- Normalizes `predicted` to a valid distribution via `pred / (sum + eps)`.
- `target` is assumed pre-normalized (from `compute_gate_target` softmax).
- Uses `F.kl_div(log_pred, target, reduction="batchmean", log_target=False)`.
- `_EPS = 1e-8` prevents log(0) — adequate for float32.

**Note**: `F.kl_div` computes `KL(target || predicted) = sum(target * (log(target) - log(predicted)))`. The `batchmean` reduction divides by batch size `B*H` (the product of all dims except the last), which is the standard per-sample mean. Correct.

### 2.3 Cohen's d Not Computed

The ablation comparison does not compute Cohen's d — it only reports `kl_improvement` and `p_value`. This is acceptable since the raw KL difference is more interpretable than a standardized effect size for this domain.

---

## 3. throughput_bench.py

### 3.1 CUDA Event Timing

**Status**: CORRECT

```python
start_event = torch.cuda.Event(enable_timing=True)
end_event = torch.cuda.Event(enable_timing=True)
start_event.record()
model(...)
end_event.record()
torch.cuda.synchronize()
elapsed_ms = start_event.elapsed_time(end_event)
```

- Events are created per iteration (lines 275-276) — correct, avoids stale event reuse.
- `synchronize()` before `elapsed_time()` ensures the end event has completed.
- CPU fallback uses `time.perf_counter_ns()` (lines 283-285) — correct.

### 3.2 OOM Recovery

**Status**: CORRECT

```python
except torch.cuda.OutOfMemoryError:
    torch.cuda.empty_cache()
```

- Catches the specific OOM exception.
- Clears cache to reclaim memory for subsequent smaller configs.
- Skipped configs are absent from `results` dict — `ThroughputMatrix.get()` returns `None` for them. Correct.

### 3.3 compare_sparse_vs_dense Logic

**Status**: CORRECT

- Iterates over `dense_matrix.results` keys (line 433), computes ratio only when both points exist (line 438).
- Division by zero guarded (line 440): `dense_tps > 0`.
- Speedup = `tasft_tps / dense_tps` — correct direction (>1 means TASFT is faster).
- TASFT configs not in dense (due to different OOM patterns) are silently omitted — acceptable.

### 3.4 Statistics

- `np.std(tps_values, ddof=1)`: Bessel-corrected sample std — correct.
- `np.percentile(latencies, 50/95/99)`: Standard percentile computation. With `num_timed=50`, p99 has limited precision (interpolation between 49th and 50th values), but this is inherent and documented.

---

## 4. test_eval_harness.py

### 4.1 Coverage Assessment

| Component | Tested | Quality |
|-----------|--------|---------|
| `TaskEvalResult` construction & validation | Yes | Good — boundaries, frozen check |
| `_wilson_ci` | Yes | Good — known values, edges, n=0 |
| `_passatk_unbiased` | Yes | Good — all/none correct, monotonicity, known value |
| `ComparisonResult` | Yes | Adequate — manual construction only |
| `GateQualityResult` validation | Yes | Good — type/sample checks |
| `compare_cotrained_vs_posthoc` | Yes | Good — better/equal/insufficient layers |
| `_kl_divergence_block` | Yes | Good — identical/divergent/finite |
| `ThroughputMatrix.get` | Yes | Basic — existing/missing keys |
| `TaskEvaluator.compare_models` | **No** | **Gap — t-test + Cohen's d not end-to-end tested** |
| `ThroughputBenchmark._benchmark_single` | **No** | Gap — would require model mock |
| Edge: empty batch | **No** | Gap |
| Edge: single sample | **No** | Gap |

### 4.2 Test Issues

**Issue 1 — Trivial assertion (line 296)**:
```python
assert result.hypothesis_supported or result.p_value < 1.0
```
`p_value < 1.0` is almost always true (scipy t-test returns exact 1.0 only for identical inputs, but even then it's `nan` for zero-variance paired differences). This assertion is effectively a no-op. Should be:
```python
assert result.hypothesis_supported  # cotrained clearly better in this test case
```

**Issue 2 — compare_models not exercised**: The `ComparisonResult` tests construct results manually (lines 193-201) rather than running `compare_models()`. The t-test and Cohen's d computation paths are only tested implicitly through the gate ablation test.

**Issue 3 — No edge case for all-correct or all-wrong**: What happens when accuracy = 1.0 or 0.0 and Cohen's d denominator approaches 0? The `1e-10` fallback at line 620 of task_eval.py handles this, but it's not tested.

---

## 5. Summary of Findings

### No Issues Found
- Wilson CI formula: mathematically correct
- pass@k log-space estimator: correct, overflow-safe, all edge cases handled
- CUDA event timing: proper synchronization
- OOM recovery: correct cleanup
- KL divergence: correct direction and normalization
- Paired t-test: correct degrees of freedom
- Data validation: comprehensive in dataclass `__post_init__`

### Minor Issues (2)

| # | File | Line | Description | Severity |
|---|------|------|-------------|----------|
| M1 | task_eval.py | 307,324 | Silent fallback to `answer_idx=0` on schema mismatch — should log warning | Low |
| M2 | test_eval_harness.py | 296 | Trivial assertion `p_value < 1.0` always passes | Low |

### Moderate Issue (1)

| # | File | Line | Description | Severity |
|---|------|------|-------------|----------|
| S1 | gate_quality.py | 543,552 | Two-tailed p-value used for one-directional hypothesis — overly conservative (effective α=0.025 instead of α=0.05) | Medium |

### Test Gaps (1 category)

| # | Description | Risk |
|---|-------------|------|
| T1 | `compare_models` t-test/Cohen's d path not end-to-end tested; edge cases (all-correct, all-wrong, single sample) missing | Medium |

---

## 6. Recommendations

1. **S1 fix**: Either halve the p-value before comparing to 0.05, or use `scipy.stats.ttest_rel` with `alternative='greater'` (scipy >= 1.7):
   ```python
   _t_stat, p_value = stats.ttest_rel(posthoc_kls, cotrained_kls, alternative='greater')
   ```

2. **T1 fix**: Add integration test that constructs synthetic per-question binary scores, calls `compare_models`-equivalent logic, and verifies t-test p-value and Cohen's d against hand-computed expected values.

3. **M1 fix**: Add `logger.warning` when answer_idx falls through to default 0.

4. **M2 fix**: Replace trivial assertion with `assert result.hypothesis_supported`.
