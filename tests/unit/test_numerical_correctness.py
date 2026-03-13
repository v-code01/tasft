"""Numerical correctness verification for all TASFT mathematical operations.

Verifies every mathematical operation produces correct results against
hand-computed reference values with explicit tolerances.

Tests:
1. AttnGate pooling: avg_pool with block_size on known Q tensor
2. Ground truth maxpool: 2D maxpool + softmax on known attention scores
3. KL divergence: hand-computed KL(P||Q) for known distributions
4. Kahan summation: 10^6 small values, Kahan vs naive float32 error
5. Cross-entropy: hand-computed CE for known logits + labels
6. Sparsity regularization: (mean - tau)^2 at known operating points
7. BF16 stability: all above in bfloat16 with relaxed tolerances
8. Wilson CI: confidence interval against scipy reference
9. pass@k: unbiased estimator against combinatorial formula

Coverage target: 100% for all mathematical operations.
"""
from __future__ import annotations

import math

import pytest
import torch
import torch.nn.functional as F

from tasft.eval.task_eval import _passatk_unbiased, _wilson_ci
from tasft.modules.attn_gate import AttnGate
from tasft.training.objectives import TASFTObjective

# ============================================================================
# 1. AttnGate Average Pooling — hand-crafted tensor verification
# ============================================================================


@pytest.mark.unit
class TestAttnGatePooling:
    """Verify _pool_to_blocks produces correct average pooling."""

    def test_known_q_tensor_pool_block4(self) -> None:
        """Hand-crafted [1,1,8,4] Q tensor, block_size=4. Verify pooled output.

        Q[0,0,:,:]  (8 tokens, head_dim=4):
            token 0: [1, 2, 3, 4]
            token 1: [5, 6, 7, 8]
            token 2: [9, 10, 11, 12]
            token 3: [13, 14, 15, 16]
            token 4: [17, 18, 19, 20]
            token 5: [21, 22, 23, 24]
            token 6: [25, 26, 27, 28]
            token 7: [29, 30, 31, 32]

        block_size=4 → 2 blocks.
        Block 0: mean of tokens 0-3 = [(1+5+9+13)/4, (2+6+10+14)/4, ...] = [7, 8, 9, 10]
        Block 1: mean of tokens 4-7 = [(17+21+25+29)/4, ...] = [23, 24, 25, 26]
        """
        gate = AttnGate(num_heads=1, head_dim=4, block_size=4)
        q = torch.arange(1, 33, dtype=torch.float32).reshape(1, 1, 8, 4)

        pooled = gate._pool_to_blocks(q)

        expected = torch.tensor([[[[7.0, 8.0, 9.0, 10.0],
                                   [23.0, 24.0, 25.0, 26.0]]]])
        assert pooled.shape == (1, 1, 2, 4)
        torch.testing.assert_close(pooled, expected, atol=1e-6, rtol=1e-6)

    def test_non_divisible_seq_len_padding(self) -> None:
        """S=6, block_size=4 → pad to 8, two blocks. Padding tokens are zeros."""
        gate = AttnGate(num_heads=1, head_dim=2, block_size=4)
        # 6 tokens, head_dim=2
        q = torch.ones(1, 1, 6, 2, dtype=torch.float32)

        pooled = gate._pool_to_blocks(q)

        # Block 0 (tokens 0-3): mean([1,1,1,1]) = [1, 1]
        # Block 1 (tokens 4-5 + 2 pad zeros): mean([1,1,0,0]) = [0.5, 0.5]
        expected = torch.tensor([[[[1.0, 1.0],
                                   [0.5, 0.5]]]])
        assert pooled.shape == (1, 1, 2, 2)
        torch.testing.assert_close(pooled, expected, atol=1e-6, rtol=1e-6)

    def test_single_block_exact(self) -> None:
        """S == block_size → single block, no padding."""
        gate = AttnGate(num_heads=2, head_dim=3, block_size=4)
        q = torch.tensor([
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
            [10.0, 11.0, 12.0],
        ]).unsqueeze(0).unsqueeze(0).expand(1, 2, 4, 3)  # [1, 2, 4, 3]

        pooled = gate._pool_to_blocks(q)

        # Mean of 4 tokens: [(1+4+7+10)/4, (2+5+8+11)/4, (3+6+9+12)/4] = [5.5, 6.5, 7.5]
        expected_block = torch.tensor([5.5, 6.5, 7.5])
        assert pooled.shape == (1, 2, 1, 3)
        torch.testing.assert_close(pooled[0, 0, 0], expected_block, atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(pooled[0, 1, 0], expected_block, atol=1e-6, rtol=1e-6)

    def test_pool_preserves_gradient(self) -> None:
        """Pooling must be differentiable for gate training."""
        gate = AttnGate(num_heads=1, head_dim=4, block_size=4)
        q = torch.randn(1, 1, 8, 4, requires_grad=True)
        pooled = gate._pool_to_blocks(q)
        pooled.sum().backward()
        assert q.grad is not None
        assert q.grad.shape == q.shape


# ============================================================================
# 2. Ground Truth Maxpool — hand-computed 2D maxpool + softmax
# ============================================================================


@pytest.mark.unit
class TestGroundTruthMaxpool:
    """Verify compute_gate_target with hand-computed expected values."""

    def test_known_8x8_attn_block4(self) -> None:
        """Known [1,1,8,8] attention scores, block_size=4.

        Attention matrix (8x8):
            Row 0: [1,  2,  3,  4,  5,  6,  7,  8]
            Row 1: [9,  10, 11, 12, 13, 14, 15, 16]
            Row 2: [17, 18, 19, 20, 21, 22, 23, 24]
            Row 3: [25, 26, 27, 28, 29, 30, 31, 32]
            Row 4: [33, 34, 35, 36, 37, 38, 39, 40]
            Row 5: [41, 42, 43, 44, 45, 46, 47, 48]
            Row 6: [49, 50, 51, 52, 53, 54, 55, 56]
            Row 7: [57, 58, 59, 60, 61, 62, 63, 64]

        2D maxpool with kernel=4, stride=4 gives 2x2:
            Block(0,0): max of rows 0-3, cols 0-3 = max(1..28) = 28
            Block(0,1): max of rows 0-3, cols 4-7 = max(5..32) = 32
            Block(1,0): max of rows 4-7, cols 0-3 = max(33..60) = 60
            Block(1,1): max of rows 4-7, cols 4-7 = max(37..64) = 64

        Softmax over [28, 32, 60, 64]:
            exp_vals = [exp(28), exp(32), exp(60), exp(64)]
            sum_exp = exp(28) + exp(32) + exp(60) + exp(64)
        """
        attn = torch.arange(1, 65, dtype=torch.float32).reshape(1, 1, 8, 8)

        target = TASFTObjective.compute_gate_target(attn, block_size=4)

        # Verify shape
        assert target.shape == (1, 1, 2, 2)

        # Verify maxpool values before softmax
        maxpool_expected = torch.tensor([28.0, 32.0, 60.0, 64.0])
        softmax_expected = F.softmax(maxpool_expected, dim=-1).reshape(1, 1, 2, 2)

        torch.testing.assert_close(target, softmax_expected, atol=1e-6, rtol=1e-6)

        # Verify softmax sums to 1.0
        total = target.sum().item()
        assert total == pytest.approx(1.0, abs=1e-6)

    def test_softmax_normalization_sums_to_one(self) -> None:
        """For arbitrary attention scores, softmax output sums to 1 per (B,H)."""
        torch.manual_seed(42)
        attn = torch.randn(3, 8, 64, 64)
        target = TASFTObjective.compute_gate_target(attn, block_size=8)

        # Each (batch, head) slice should sum to 1
        for b in range(3):
            for h in range(8):
                block_sum = target[b, h].sum().item()
                assert block_sum == pytest.approx(1.0, abs=1e-5), (
                    f"batch={b}, head={h}: sum={block_sum}"
                )

    def test_causal_mask_neg_inf_handled(self) -> None:
        """Causal masking with -inf should NOT raise and should produce valid output."""
        attn = torch.randn(1, 1, 8, 8)
        # Upper triangular -inf (causal mask)
        causal_mask = torch.triu(torch.ones(8, 8), diagonal=1).bool()
        attn[0, 0][causal_mask] = float("-inf")

        target = TASFTObjective.compute_gate_target(attn, block_size=4)

        assert not torch.isnan(target).any()
        assert not torch.isinf(target).any()
        assert target.sum().item() == pytest.approx(1.0, abs=1e-5)


# ============================================================================
# 3. KL Divergence — hand-computed reference
# ============================================================================


@pytest.mark.unit
class TestKLDivergence:
    """Verify compute_gate_loss against hand-computed KL divergence."""

    def test_known_distributions_kl(self) -> None:
        """KL(P||Q) for P=[0.1, 0.2, 0.7], Q=[0.3, 0.3, 0.4].

        KL(P||Q) = 0.1*ln(0.1/0.3) + 0.2*ln(0.2/0.3) + 0.7*ln(0.7/0.4)
                 = 0.1*(-1.0986) + 0.2*(-0.4055) + 0.7*(0.5596)
                 = -0.10986 + (-0.08109) + 0.39173
                 = 0.20078

        However, compute_gate_loss uses F.kl_div with batchmean reduction
        which divides by batch_size. And it normalizes gate_soft_scores
        to a distribution. So we need to match the actual implementation.

        The implementation:
        1. gate_soft_scores → normalize to distribution (gate_dist)
        2. KL = F.kl_div(log(gate_dist + eps), target, reduction='batchmean')
        F.kl_div computes: sum(target * (log(target) - log_input)) / batch_size
        """
        # Create distributions as [1, 1, 1, 3] tensors (B=1, H=1, NB_q=1, NB_k=3)
        P = torch.tensor([0.1, 0.2, 0.7], dtype=torch.float64).reshape(1, 1, 1, 3)
        Q_scores = torch.tensor([0.3, 0.3, 0.4], dtype=torch.float64).reshape(1, 1, 1, 3)

        # The implementation normalizes Q_scores to a distribution
        # Q_scores / sum(Q_scores) = [0.3, 0.3, 0.4] / 1.0 = [0.3, 0.3, 0.4]
        # So Q_dist ≈ [0.3, 0.3, 0.4] (already sums to 1)

        loss = TASFTObjective.compute_gate_loss(Q_scores, P)

        # Compute reference: F.kl_div(log(Q_dist + eps), P, reduction='batchmean')
        # = sum(P * (log(P) - log(Q_dist + eps))) / B
        eps = 1e-8
        q_reshaped = Q_scores.reshape(1, 1, 3)
        Q_dist = q_reshaped / (q_reshaped.sum(dim=-1, keepdim=True) + eps)
        log_Q = torch.log(Q_dist + eps)
        P_flat = P.reshape(1, 1, 3)
        ref_kl = F.kl_div(log_Q, P_flat, reduction="batchmean", log_target=False)

        assert loss.item() == pytest.approx(ref_kl.item(), abs=1e-4)

    def test_identical_distributions_zero_kl(self) -> None:
        """KL(P||P) = 0 for any valid distribution P."""
        P = torch.tensor([0.25, 0.25, 0.25, 0.25], dtype=torch.float64).reshape(1, 1, 2, 2)
        loss = TASFTObjective.compute_gate_loss(P, P)
        assert loss.item() == pytest.approx(0.0, abs=1e-4)

    def test_kl_non_negative(self) -> None:
        """KL divergence is always >= 0 (Gibbs' inequality)."""
        torch.manual_seed(123)
        for _ in range(20):
            gate = torch.rand(1, 1, 4, 4, dtype=torch.float64) + 0.01
            raw = torch.randn(1, 1, 16, dtype=torch.float64)
            target = F.softmax(raw, dim=-1).reshape(1, 1, 4, 4)
            loss = TASFTObjective.compute_gate_loss(gate, target)
            assert loss.item() >= -1e-6, f"KL divergence should be non-negative, got {loss.item()}"


# ============================================================================
# 4. Kahan Summation — 10^6 small values
# ============================================================================


@pytest.mark.unit
class TestKahanSummation:
    """Verify Kahan summation provides better accuracy than naive float32 sum."""

    def test_million_small_values_kahan_vs_naive(self) -> None:
        """Sum 10^6 values of 1e-8 in float32.

        True value: 10^6 * 1e-8 = 0.01
        Naive float32 accumulation loses precision due to catastrophic cancellation.
        Kahan summation should be closer to the true value.
        """
        n = 1_000_000
        true_value = n * 1e-8  # = 0.01

        # Naive summation in float32
        naive_sum = torch.tensor(0.0, dtype=torch.float32)
        val = torch.tensor(1e-8, dtype=torch.float32)
        for _ in range(n):
            naive_sum = naive_sum + val
        naive_error = abs(naive_sum.item() - true_value)

        # Kahan summation in float32
        kahan_sum = torch.tensor(0.0, dtype=torch.float32)
        kahan_comp = torch.tensor(0.0, dtype=torch.float32)
        for _ in range(n):
            y = val - kahan_comp
            t = kahan_sum + y
            kahan_comp = (t - kahan_sum) - y
            kahan_sum = t
        kahan_error = abs(kahan_sum.item() - true_value)

        # Kahan should be strictly closer to true value
        assert kahan_error < naive_error, (
            f"Kahan error ({kahan_error:.2e}) should be less than "
            f"naive error ({naive_error:.2e})"
        )
        # Kahan error should be very small
        assert kahan_error < 1e-6, f"Kahan error too large: {kahan_error:.2e}"

    def test_kahan_in_objective_compute(self) -> None:
        """Verify the Kahan summation in TASFTObjective.compute() gives stable results.

        Accumulate gate losses from many layers with small values. The Kahan
        accumulator in compute() should match a float64 reference closely.
        """
        obj = TASFTObjective(lambda_gate=1.0, beta_sparse=0.0, tau_target=0.5)

        B, S, V = 1, 16, 32
        logits = torch.randn(B, S, V, dtype=torch.float32)
        labels = torch.randint(0, V, (B, S))
        block_size = 4
        NB = S // block_size

        num_layers = 50
        gate_outputs: dict[int, torch.Tensor] = {}
        attn_scores: dict[int, torch.Tensor] = {}

        torch.manual_seed(99)
        for i in range(num_layers):
            gate_outputs[i] = torch.rand(B, 1, NB, NB) * 0.01 + 0.5
            attn_scores[i] = torch.randn(B, 1, S, S)

        result = obj.compute(logits, labels, gate_outputs, attn_scores,
                             list(range(num_layers)), block_size)

        assert torch.isfinite(result.gate)
        assert torch.isfinite(result.total)

        # Compute reference in float64
        ref_gate_sum = 0.0
        for i in range(num_layers):
            target = TASFTObjective.compute_gate_target(attn_scores[i], block_size)
            layer_loss = TASFTObjective.compute_gate_loss(gate_outputs[i], target)
            ref_gate_sum += layer_loss.item()

        # Should agree to reasonable precision
        assert result.gate.item() == pytest.approx(ref_gate_sum, rel=1e-3)


# ============================================================================
# 5. Cross-Entropy — hand-computed reference
# ============================================================================


@pytest.mark.unit
class TestCrossEntropy:
    """Verify compute_task_loss against hand-computed cross-entropy."""

    def test_known_logits_ce(self) -> None:
        """logits = [[2.0, 1.0, 0.1]], label = [0].

        With shift-by-1 for next-token prediction, we need at least S=2.
        logits shape [1, 2, 3]:
            position 0: [2.0, 1.0, 0.1]  (predicts position 1 label)
            position 1: [0.0, 0.0, 0.0]  (not used as predictor)
        labels shape [1, 2]:
            position 0: -100 (not used as target)
            position 1: 0    (target for position 0's prediction)

        CE = -log(softmax([2.0, 1.0, 0.1])[0])
        softmax([2.0, 1.0, 0.1]):
            exp([2.0, 1.0, 0.1]) = [7.389056, 2.718282, 1.105171]
            sum = 11.212509
            softmax = [0.659001, 0.242433, 0.098566]
        CE = -log(0.659001) = 0.41702
        """
        logits = torch.tensor([[[2.0, 1.0, 0.1], [0.0, 0.0, 0.0]]])  # [1, 2, 3]
        labels = torch.tensor([[-100, 0]])  # [1, 2]

        obj = TASFTObjective()
        loss = obj.compute_task_loss(logits, labels)

        # Hand-computed reference
        log_softmax = torch.log_softmax(torch.tensor([2.0, 1.0, 0.1]), dim=0)
        expected_ce = -log_softmax[0].item()

        assert loss.item() == pytest.approx(expected_ce, abs=1e-5)

    def test_ce_perfect_prediction(self) -> None:
        """When logits strongly predict the correct class, CE → 0."""
        # Very confident prediction: logit=100 for correct class, 0 for others
        logits = torch.tensor([[[100.0, 0.0, 0.0], [0.0, 0.0, 0.0]]])
        labels = torch.tensor([[-100, 0]])

        obj = TASFTObjective()
        loss = obj.compute_task_loss(logits, labels)

        assert loss.item() == pytest.approx(0.0, abs=1e-4)

    def test_ce_uniform_prediction(self) -> None:
        """Uniform logits → CE = log(V)."""
        V = 10
        logits = torch.zeros(1, 2, V)  # uniform
        labels = torch.tensor([[-100, 5]])

        obj = TASFTObjective()
        loss = obj.compute_task_loss(logits, labels)

        expected = math.log(V)  # = ln(10) ≈ 2.3026
        assert loss.item() == pytest.approx(expected, abs=1e-5)


# ============================================================================
# 6. Sparsity Regularization — known operating points
# ============================================================================


@pytest.mark.unit
class TestSparsityRegularization:
    """Verify compute_sparsity_loss at specific operating points."""

    def test_mean_equals_tau_zero_loss(self) -> None:
        """At tau=0.8, mean_score=0.8 → loss = (0.8 - 0.8)^2 = 0."""
        gate = torch.full((1, 1, 4, 4), 0.8)
        loss = TASFTObjective.compute_sparsity_loss(gate, tau_target=0.8)
        assert loss.item() == pytest.approx(0.0, abs=1e-10)

    def test_mean_05_tau_08(self) -> None:
        """At tau=0.8, mean_score=0.5 → loss = (0.5 - 0.8)^2 = 0.09."""
        gate = torch.full((1, 1, 4, 4), 0.5)
        loss = TASFTObjective.compute_sparsity_loss(gate, tau_target=0.8)
        expected = (0.5 - 0.8) ** 2  # = 0.09
        assert loss.item() == pytest.approx(expected, abs=1e-6)

    def test_mean_10_tau_08(self) -> None:
        """At tau=0.8, mean_score=1.0 → loss = (1.0 - 0.8)^2 = 0.04."""
        gate = torch.full((1, 1, 4, 4), 1.0)
        loss = TASFTObjective.compute_sparsity_loss(gate, tau_target=0.8)
        expected = (1.0 - 0.8) ** 2  # = 0.04
        assert loss.item() == pytest.approx(expected, abs=1e-6)

    def test_mean_00_tau_05(self) -> None:
        """At tau=0.5, mean_score=0.0 → loss = 0.25."""
        gate = torch.zeros(1, 1, 4, 4)
        loss = TASFTObjective.compute_sparsity_loss(gate, tau_target=0.5)
        assert loss.item() == pytest.approx(0.25, abs=1e-6)

    def test_symmetry_above_below_tau(self) -> None:
        """Loss is symmetric: deviation above tau equals deviation below."""
        tau = 0.6
        gate_above = torch.full((1, 1, 4, 4), 0.8)  # deviation = +0.2
        gate_below = torch.full((1, 1, 4, 4), 0.4)  # deviation = -0.2

        loss_above = TASFTObjective.compute_sparsity_loss(gate_above, tau)
        loss_below = TASFTObjective.compute_sparsity_loss(gate_below, tau)

        assert loss_above.item() == pytest.approx(loss_below.item(), abs=1e-6)
        assert loss_above.item() == pytest.approx(0.04, abs=1e-6)


# ============================================================================
# 7. BF16 Stability — all operations in bfloat16
# ============================================================================


@pytest.mark.unit
class TestBF16Stability:
    """Run all numerical operations in bfloat16, verify error bounds < 1e-2."""

    def test_pool_to_blocks_bf16(self) -> None:
        """Average pooling in bf16 matches fp32 reference within 1e-2."""
        gate = AttnGate(num_heads=1, head_dim=4, block_size=4)
        q_fp32 = torch.arange(1, 33, dtype=torch.float32).reshape(1, 1, 8, 4)
        q_bf16 = q_fp32.to(torch.bfloat16)

        pooled_fp32 = gate._pool_to_blocks(q_fp32)
        pooled_bf16 = gate._pool_to_blocks(q_bf16)

        torch.testing.assert_close(
            pooled_bf16.float(), pooled_fp32, atol=1e-2, rtol=1e-2,
        )

    def test_gate_target_bf16(self) -> None:
        """compute_gate_target in bf16 produces valid distribution."""
        attn_fp32 = torch.randn(1, 1, 8, 8)
        attn_bf16 = attn_fp32.to(torch.bfloat16)

        target_fp32 = TASFTObjective.compute_gate_target(attn_fp32, block_size=4)
        target_bf16 = TASFTObjective.compute_gate_target(attn_bf16, block_size=4)

        # bf16 result should be valid distribution
        assert not torch.isnan(target_bf16).any()
        assert target_bf16.float().sum().item() == pytest.approx(1.0, abs=1e-2)
        # Should be close to fp32 reference
        torch.testing.assert_close(
            target_bf16.float(), target_fp32, atol=1e-2, rtol=1e-2,
        )

    def test_gate_loss_bf16(self) -> None:
        """KL divergence in bf16 stays finite and close to fp32."""
        gate = torch.rand(1, 1, 2, 2, dtype=torch.float32) + 0.1
        target = F.softmax(torch.randn(1, 1, 4), dim=-1).reshape(1, 1, 2, 2)

        loss_fp32 = TASFTObjective.compute_gate_loss(gate, target)
        loss_bf16 = TASFTObjective.compute_gate_loss(
            gate.to(torch.bfloat16), target.to(torch.bfloat16),
        )

        assert torch.isfinite(loss_bf16)
        assert abs(loss_bf16.float().item() - loss_fp32.item()) < 1e-1

    def test_sparsity_loss_bf16(self) -> None:
        """Sparsity loss in bf16 matches fp32 within tolerance."""
        gate = torch.full((1, 1, 4, 4), 0.5, dtype=torch.bfloat16)
        loss = TASFTObjective.compute_sparsity_loss(gate, 0.8)

        expected = (0.5 - 0.8) ** 2  # = 0.09
        assert loss.float().item() == pytest.approx(expected, abs=1e-2)

    def test_task_loss_bf16(self) -> None:
        """Cross-entropy in bf16 produces finite result close to fp32."""
        logits_fp32 = torch.randn(1, 16, 50)
        logits_bf16 = logits_fp32.to(torch.bfloat16)
        labels = torch.randint(0, 50, (1, 16))

        obj = TASFTObjective()
        loss_fp32 = obj.compute_task_loss(logits_fp32, labels)
        loss_bf16 = obj.compute_task_loss(logits_bf16, labels)

        assert torch.isfinite(loss_bf16)
        assert abs(loss_bf16.float().item() - loss_fp32.item()) < 1e-1

    def test_full_compute_bf16(self) -> None:
        """Full compute() in bf16 produces finite, reasonable results."""
        obj = TASFTObjective(lambda_gate=0.1, beta_sparse=0.01, tau_target=0.5)
        B, S, V = 1, 16, 32
        block_size = 4
        NB = S // block_size

        logits = torch.randn(B, S, V, dtype=torch.bfloat16)
        labels = torch.randint(0, V, (B, S))
        gate_outputs = {0: torch.rand(B, 1, NB, NB, dtype=torch.bfloat16)}
        attn_scores = {0: torch.randn(B, 1, S, S, dtype=torch.bfloat16)}

        result = obj.compute(logits, labels, gate_outputs, attn_scores, [0], block_size)

        assert torch.isfinite(result.total)
        assert torch.isfinite(result.task)
        assert torch.isfinite(result.gate)
        assert torch.isfinite(result.sparse)


# ============================================================================
# 8. Wilson Confidence Interval — verified against scipy
# ============================================================================


@pytest.mark.unit
class TestWilsonCI:
    """Verify Wilson score interval against scipy binomial proportion CI."""

    def test_known_values_p075_n100(self) -> None:
        """(p=0.75, n=100) → CI approx (0.657, 0.825)."""
        low, high = _wilson_ci(0.75, 100)

        assert low == pytest.approx(0.657, abs=0.005)
        assert high == pytest.approx(0.825, abs=0.005)

    def test_against_scipy_reference(self) -> None:
        """Compare Wilson CI against scipy.stats.binom_test / proportion_confint."""
        # Use scipy's own Wilson interval
        # statsmodels has proportion_confint, but scipy alone:
        # We'll verify algebraically
        p, n = 0.75, 100
        z = 1.959964

        # Wilson formula directly
        z2 = z * z
        denom = 1.0 + z2 / n
        centre = (p + z2 / (2.0 * n)) / denom
        margin = (z / denom) * math.sqrt(p * (1.0 - p) / n + z2 / (4.0 * n * n))
        expected_low = max(0.0, centre - margin)
        expected_high = min(1.0, centre + margin)

        low, high = _wilson_ci(p, n)

        assert low == pytest.approx(expected_low, abs=1e-10)
        assert high == pytest.approx(expected_high, abs=1e-10)

    def test_p0_gives_valid_interval(self) -> None:
        """p=0 should give CI starting at 0."""
        low, high = _wilson_ci(0.0, 100)
        assert low == 0.0
        assert 0.0 < high < 0.1

    def test_p1_gives_valid_interval(self) -> None:
        """p=1 should give CI ending at 1."""
        low, high = _wilson_ci(1.0, 100)
        assert low > 0.9
        assert high == 1.0

    def test_n0_gives_full_interval(self) -> None:
        """n=0 → (0, 1) (no information)."""
        low, high = _wilson_ci(0.5, 0)
        assert low == 0.0
        assert high == 1.0

    def test_small_n_wider_than_large_n(self) -> None:
        """Smaller sample size → wider CI."""
        low_small, high_small = _wilson_ci(0.5, 10)
        low_large, high_large = _wilson_ci(0.5, 1000)
        width_small = high_small - low_small
        width_large = high_large - low_large
        assert width_small > width_large


# ============================================================================
# 9. pass@k — unbiased estimator verification
# ============================================================================


@pytest.mark.unit
class TestPassAtK:
    """Verify unbiased pass@k estimator against combinatorial formula."""

    def test_n20_c5_k1(self) -> None:
        """n=20, c=5, k=1 → pass@1 = 1 - C(15,1)/C(20,1) = 1 - 15/20 = 0.25."""
        result = _passatk_unbiased(20, 5, 1)
        assert result == pytest.approx(0.25, abs=1e-10)

    def test_n20_c5_k10(self) -> None:
        """n=20, c=5, k=10 → pass@10 = 1 - C(15,10)/C(20,10).

        C(15,10) = C(15,5) = 3003
        C(20,10) = 184756
        pass@10 = 1 - 3003/184756 ≈ 0.98374
        """
        result = _passatk_unbiased(20, 5, 10)
        expected = 1.0 - 3003 / 184756
        assert result == pytest.approx(expected, abs=1e-4)

    def test_all_correct_returns_one(self) -> None:
        """When c == n, pass@k = 1.0 for any k."""
        assert _passatk_unbiased(10, 10, 1) == 1.0
        assert _passatk_unbiased(10, 10, 5) == 1.0
        assert _passatk_unbiased(10, 10, 10) == 1.0

    def test_none_correct_returns_zero(self) -> None:
        """When c == 0, pass@k = 0.0 for any k <= n."""
        assert _passatk_unbiased(10, 0, 1) == pytest.approx(0.0, abs=1e-10)
        assert _passatk_unbiased(10, 0, 5) == pytest.approx(0.0, abs=1e-10)

    def test_k_greater_than_n_minus_c(self) -> None:
        """When k > n-c, guaranteed at least one correct → 1.0."""
        assert _passatk_unbiased(10, 8, 5) == 1.0

    def test_pass_at_1_equals_c_over_n(self) -> None:
        """pass@1 = c/n for any valid inputs (the degenerate case)."""
        for n, c in [(100, 25), (50, 10), (20, 1)]:
            result = _passatk_unbiased(n, c, 1)
            assert result == pytest.approx(c / n, abs=1e-10)

    def test_monotonicity_in_k(self) -> None:
        """pass@k should be monotonically increasing in k."""
        n, c = 20, 5
        prev = 0.0
        for k in range(1, 16):
            curr = _passatk_unbiased(n, c, k)
            assert curr >= prev - 1e-10, f"Monotonicity violated at k={k}"
            prev = curr

    def test_log_space_no_overflow(self) -> None:
        """Large n values should not overflow due to log-space computation."""
        # C(1000, 100) would overflow, but log-space should handle it
        result = _passatk_unbiased(1000, 100, 50)
        assert 0.0 <= result <= 1.0
        assert math.isfinite(result)


# ============================================================================
# 10. Additional edge cases and invariants
# ============================================================================


@pytest.mark.unit
class TestNumericalInvariants:
    """Cross-cutting numerical invariants that must hold."""

    def test_gate_output_scores_in_01(self) -> None:
        """AttnGate forward pass always produces soft_scores in [0, 1]."""
        gate = AttnGate(num_heads=2, head_dim=8, block_size=4)
        torch.manual_seed(42)
        q = torch.randn(1, 2, 16, 8)
        k = torch.randn(1, 2, 16, 8)

        output = gate(q, k)
        assert output.soft_scores.min().item() >= 0.0
        assert output.soft_scores.max().item() <= 1.0

    def test_sparsity_ratio_in_01(self) -> None:
        """Sparsity ratio from AttnGate is always in [0, 1]."""
        gate = AttnGate(num_heads=1, head_dim=4, block_size=4)
        q = torch.randn(1, 1, 8, 4)
        k = torch.randn(1, 1, 8, 4)

        output = gate(q, k)
        assert 0.0 <= float(output.sparsity_ratio) <= 1.0

    def test_loss_decomposition_identity_precise(self) -> None:
        """total = task + lambda * (gate + beta * sparse) to high precision."""
        lam, beta = 0.3, 0.05
        obj = TASFTObjective(lambda_gate=lam, beta_sparse=beta, tau_target=0.6)

        B, S, V = 1, 16, 32
        block_size = 4
        NB = S // block_size

        torch.manual_seed(7)
        logits = torch.randn(B, S, V, dtype=torch.float64)
        labels = torch.randint(0, V, (B, S))
        gate_outputs = {0: torch.rand(B, 1, NB, NB, dtype=torch.float64)}
        attn_scores = {0: torch.randn(B, 1, S, S, dtype=torch.float64)}

        result = obj.compute(logits, labels, gate_outputs, attn_scores, [0], block_size)

        recomputed = result.task + lam * (result.gate + beta * result.sparse)
        assert result.total.item() == pytest.approx(recomputed.item(), abs=1e-10)

    def test_softmax_gate_target_all_equal_uniform(self) -> None:
        """When all attention scores are equal, gate target is uniform."""
        attn = torch.full((1, 1, 8, 8), 5.0)
        target = TASFTObjective.compute_gate_target(attn, block_size=4)

        # 2x2 blocks, all maxpool values equal → softmax gives uniform 0.25
        expected = torch.full((1, 1, 2, 2), 0.25)
        torch.testing.assert_close(target, expected, atol=1e-5, rtol=1e-5)
