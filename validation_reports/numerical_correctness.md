# Numerical Correctness Validation Report

**Date:** 2026-03-13
**Test file:** `tests/unit/test_numerical_correctness.py`
**Result:** 44/44 tests PASSED (4.49s)

---

## 1. AttnGate Average Pooling

| Test | Input | Expected | Result |
|------|-------|----------|--------|
| Known [1,1,8,4] Q, block_size=4 | arange(1..32) | Block0=[7,8,9,10], Block1=[23,24,25,26] | PASS (exact match) |
| Non-divisible S=6, block_size=4 | ones(6 tokens) | Block0=[1,1], Block1=[0.5,0.5] (zero-padded) | PASS |
| Single block S=block_size | [1..12] reshaped | [5.5, 6.5, 7.5] | PASS |
| Gradient flow | random | grad.shape == input.shape | PASS |

**Verdict:** Average pooling is numerically correct. Padding with zeros for non-divisible sequences is handled correctly — the mean dilutes padded positions as expected.

---

## 2. Ground Truth Maxpool (compute_gate_target)

| Test | Input | Expected | Result |
|------|-------|----------|--------|
| Known [1,1,8,8] arange(1..64), block_size=4 | 2D maxpool → [28,32,60,64] → softmax | softmax([28,32,60,64]) | PASS |
| Softmax normalization sums to 1.0 | random [3,8,64,64] | sum=1.0 per (B,H) | PASS (all 24 slices) |
| Causal mask (-inf upper triangle) | random + -inf mask | valid distribution, no NaN | PASS |
| All-equal attention scores | constant 5.0 | uniform 0.25 per block | PASS |

**Verdict:** 2D maxpool correctly extracts block-level importance. Softmax normalization produces valid probability distributions. Causal masking with -inf is handled correctly (maxpool takes max, softmax maps -inf → 0).

---

## 3. KL Divergence (compute_gate_loss)

| Test | P | Q (gate scores) | Expected | Result |
|------|---|------------------|----------|--------|
| Known distributions | [0.1,0.2,0.7] | [0.3,0.3,0.4] | Matches F.kl_div reference | PASS (< 1e-4) |
| Identical P=Q | uniform | uniform | 0.0 | PASS (< 1e-4) |
| Non-negativity | 20 random pairs | KL >= 0 | PASS (Gibbs' inequality holds) |

**Note:** The implementation normalizes gate_soft_scores to a distribution before computing KL, and uses `batchmean` reduction. The hand-computed KL(P||Q) = 0.2008 matches when accounting for the normalization pipeline.

**Verdict:** KL divergence computation is correct and satisfies the Gibbs' inequality invariant.

---

## 4. Kahan Summation

| Test | N | True Value | Naive Error | Kahan Error | Result |
|------|---|------------|-------------|-------------|--------|
| 10^6 values of 1e-8 | 1,000,000 | 0.01 | >1e-6 | <1e-6 | PASS |
| 50-layer compute() | 50 | float64 ref | — | rel < 1e-3 | PASS |

**Verdict:** Kahan summation in `TASFTObjective.compute()` provides measurably better accuracy than naive accumulation. The compensated accumulator prevents catastrophic cancellation when summing many small gate losses across layers.

---

## 5. Cross-Entropy (compute_task_loss)

| Test | Logits | Label | Expected CE | Result |
|------|--------|-------|-------------|--------|
| [2.0, 1.0, 0.1] | 0 | -log(softmax([2.0,1.0,0.1])[0]) = 0.41702 | PASS (< 1e-5) |
| Perfect prediction [100, 0, 0] | 0 | ≈ 0.0 | PASS (< 1e-4) |
| Uniform logits, V=10 | 5 | ln(10) = 2.3026 | PASS (< 1e-5) |

**Note:** compute_task_loss implements shifted cross-entropy (logits[:-1] predict labels[1:]) for next-token prediction. Tests account for this shift.

**Verdict:** Cross-entropy computation matches PyTorch reference implementation exactly.

---

## 6. Sparsity Regularization (compute_sparsity_loss)

| Test | mean_score | tau_target | Expected Loss | Result |
|------|------------|------------|---------------|--------|
| mean=tau | 0.8 | 0.8 | 0.0 | PASS (< 1e-10) |
| mean < tau | 0.5 | 0.8 | 0.09 | PASS (< 1e-6) |
| mean > tau | 1.0 | 0.8 | 0.04 | PASS (< 1e-6) |
| mean=0 | 0.0 | 0.5 | 0.25 | PASS (< 1e-6) |
| Symmetry | ±0.2 from 0.6 | 0.6 | Both = 0.04 | PASS |

**Verdict:** L_sparse = (mean(gate_scores) - tau_target)^2 is exact. Symmetric penalty for above/below deviations confirmed.

---

## 7. BF16 Stability

| Operation | fp32 Baseline | bf16 Error Bound | Result |
|-----------|---------------|-------------------|--------|
| Average pooling | exact | < 1e-2 | PASS |
| Gate target (maxpool+softmax) | valid dist. | sum ≈ 1.0 (< 1e-2) | PASS |
| KL divergence | finite | < 1e-1 | PASS |
| Sparsity loss | 0.09 | < 1e-2 | PASS |
| Cross-entropy | finite | < 1e-1 | PASS |
| Full compute() | finite | all finite | PASS |

**Verdict:** All operations remain numerically stable in bfloat16. Error bounds stay within acceptable ranges for mixed-precision training. No NaN or Inf propagation observed.

---

## 8. Wilson Confidence Interval

| Test | p | n | Expected CI | Result |
|------|---|---|-------------|--------|
| Known values | 0.75 | 100 | (0.657, 0.825) | PASS (< 0.005) |
| Algebraic reference | 0.75 | 100 | Wilson formula exact | PASS (< 1e-10) |
| p=0 boundary | 0.0 | 100 | (0.0, < 0.1) | PASS |
| p=1 boundary | 1.0 | 100 | (> 0.9, 1.0) | PASS |
| n=0 (no data) | 0.5 | 0 | (0.0, 1.0) | PASS |
| Width monotonicity | 0.5 | 10 vs 1000 | small_n wider | PASS |

**Verdict:** Wilson score interval implementation matches the algebraic formula exactly. Boundary conditions handled correctly.

---

## 9. pass@k Unbiased Estimator

| Test | n | c | k | Expected | Result |
|------|---|---|---|----------|--------|
| Basic | 20 | 5 | 1 | 0.25 | PASS (exact) |
| Large k | 20 | 5 | 10 | 0.98374 | PASS (< 1e-4) |
| All correct | 10 | 10 | any | 1.0 | PASS |
| None correct | 10 | 0 | any | 0.0 | PASS |
| k > n-c | 10 | 8 | 5 | 1.0 | PASS |
| pass@1 = c/n | various | — | 1 | c/n | PASS (< 1e-10) |
| Monotonicity | 20 | 5 | 1..15 | increasing | PASS |
| Large n (overflow test) | 1000 | 100 | 50 | in [0,1] | PASS |

**Verdict:** Unbiased pass@k estimator is correct. Log-space computation prevents overflow for large combinatorial values. Identity pass@1 = c/n holds exactly. Monotonicity in k confirmed.

---

## 10. Cross-Cutting Invariants

| Invariant | Result |
|-----------|--------|
| gate soft_scores always in [0, 1] (sigmoid guarantee) | PASS |
| sparsity_ratio always in [0, 1] | PASS |
| total = task + λ·(gate + β·sparse) to 1e-10 (float64) | PASS |
| Equal attention → uniform gate target (0.25 each) | PASS |

---

## Summary

**Total tests:** 44
**Passed:** 44
**Failed:** 0
**Execution time:** 4.49s

All mathematical operations in TASFT produce numerically correct results against hand-computed reference values. BF16 stability is confirmed with error bounds well within acceptable ranges for mixed-precision training. No numerical issues were discovered.
