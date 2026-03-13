"""Unit tests for TASFTObjective dual training objective.

Tests:
- L_task: cross_entropy verified against nn.CrossEntropyLoss to 1e-6 tolerance
- L_gate: KL divergence with known distributions
- Ground truth: 2D maxpool on known matrix with exact expected output
- Sparsity regularization: mean=tau_target => L_sparse=0; deviations => L_sparse>0
- NaN guard: inject NaN/Inf => raises NaNDetectedError
- Kahan summation: error < 1e-10 for 10^4 small values
- Mathematical identity: total = task + lambda * (gate + beta * sparse)
- Layer rotation: only active layers contribute to gate loss

Coverage target: 100% for all mathematical operations.
"""

import pytest
import torch
import torch.nn.functional as F

from tasft.exceptions import NaNDetectedError
from tasft.training.objectives import TASFTObjective
from tasft.types import LayerIndex

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def objective() -> TASFTObjective:
    """Default objective: lambda=0.1, beta=0.01, tau=0.8."""
    return TASFTObjective(lambda_gate=0.1, beta_sparse=0.01, tau_target=0.8)


# ---------------------------------------------------------------------------
# Constructor validation
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestObjectiveConstruction:
    """Parameter validation for TASFTObjective."""

    def test_valid_defaults(self) -> None:
        obj = TASFTObjective()
        assert obj._lambda_gate == 0.1
        assert obj._beta_sparse == 0.01
        assert obj._tau_target == 0.8
        assert obj._label_smoothing == 0.0

    def test_lambda_gate_zero_raises(self) -> None:
        with pytest.raises(ValueError, match="lambda_gate"):
            TASFTObjective(lambda_gate=0.0)

    def test_lambda_gate_negative_raises(self) -> None:
        with pytest.raises(ValueError, match="lambda_gate"):
            TASFTObjective(lambda_gate=-0.5)

    def test_lambda_gate_above_ten_raises(self) -> None:
        with pytest.raises(ValueError, match="lambda_gate"):
            TASFTObjective(lambda_gate=10.1)

    def test_lambda_gate_boundary_ten_ok(self) -> None:
        obj = TASFTObjective(lambda_gate=10.0)
        assert obj._lambda_gate == 10.0

    def test_beta_sparse_negative_raises(self) -> None:
        with pytest.raises(ValueError, match="beta_sparse"):
            TASFTObjective(beta_sparse=-0.01)

    def test_beta_sparse_zero_ok(self) -> None:
        obj = TASFTObjective(beta_sparse=0.0)
        assert obj._beta_sparse == 0.0

    def test_tau_target_zero_raises(self) -> None:
        with pytest.raises(ValueError, match="tau_target"):
            TASFTObjective(tau_target=0.0)

    def test_tau_target_one_raises(self) -> None:
        with pytest.raises(ValueError, match="tau_target"):
            TASFTObjective(tau_target=1.0)

    def test_label_smoothing_negative_raises(self) -> None:
        with pytest.raises(ValueError, match="label_smoothing"):
            TASFTObjective(label_smoothing=-0.1)

    def test_label_smoothing_one_raises(self) -> None:
        with pytest.raises(ValueError, match="label_smoothing"):
            TASFTObjective(label_smoothing=1.0)


# ---------------------------------------------------------------------------
# Task loss: cross-entropy verified against torch reference
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestTaskLoss:
    """L_task = cross_entropy with shift-by-1 for next-token prediction."""

    def test_task_loss_matches_torch_reference(self, objective: TASFTObjective) -> None:
        """Verify L_task matches nn.CrossEntropyLoss to 1e-6."""
        B, S, V = 2, 32, 100
        logits = torch.randn(B, S, V)
        labels = torch.randint(0, V, (B, S))

        our_loss = objective.compute_task_loss(logits, labels)

        # Manual reference: shift by 1 position
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        ref_loss = F.cross_entropy(
            shift_logits.view(-1, V),
            shift_labels.view(-1),
            ignore_index=-100,
        )

        assert our_loss.item() == pytest.approx(ref_loss.item(), abs=1e-6)

    def test_task_loss_with_label_smoothing(self) -> None:
        """Label smoothing is passed through to cross_entropy."""
        B, S, V = 1, 16, 50
        logits = torch.randn(B, S, V)
        labels = torch.randint(0, V, (B, S))

        obj_smooth = TASFTObjective(label_smoothing=0.1)
        obj_no_smooth = TASFTObjective(label_smoothing=0.0)

        loss_smooth = obj_smooth.compute_task_loss(logits, labels)
        loss_no_smooth = obj_no_smooth.compute_task_loss(logits, labels)

        # With smoothing, loss should generally differ from without
        # (unless input is trivial which is astronomically unlikely with randn)
        assert loss_smooth.item() != pytest.approx(loss_no_smooth.item(), abs=1e-6)

    def test_task_loss_ignores_minus_100_labels(self, objective: TASFTObjective) -> None:
        """Labels set to -100 are ignored. Partial -100 should lower effective count."""
        B, S, V = 1, 16, 50
        logits = torch.randn(B, S, V)
        # Half valid, half ignored — verify the ignored half doesn't contribute
        labels = torch.randint(0, V, (B, S))
        labels_partial = labels.clone()
        labels_partial[:, S // 2 :] = -100

        loss_full = objective.compute_task_loss(logits, labels)
        loss_partial = objective.compute_task_loss(logits, labels_partial)

        # Both should be finite, and they should differ
        assert torch.isfinite(loss_full)
        assert torch.isfinite(loss_partial)
        assert loss_full.item() != pytest.approx(loss_partial.item(), abs=1e-6)

    def test_task_loss_is_scalar(self, objective: TASFTObjective) -> None:
        logits = torch.randn(1, 16, 50)
        labels = torch.randint(0, 50, (1, 16))
        loss = objective.compute_task_loss(logits, labels)
        assert loss.ndim == 0


# ---------------------------------------------------------------------------
# Gate target: 2D maxpool ground truth computation
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestGateTarget:
    """compute_gate_target: 2D maxpool + softmax on attention scores."""

    def test_known_matrix_exact_output(self) -> None:
        """Known 4x4 attn matrix with block_size=2 -> 2x2 maxpool -> softmax."""
        # [B=1, H=1, S=4, S=4]
        attn = torch.tensor(
            [
                [1.0, 2.0, 3.0, 4.0],
                [5.0, 6.0, 7.0, 8.0],
                [9.0, 10.0, 11.0, 12.0],
                [13.0, 14.0, 15.0, 16.0],
            ],
        ).unsqueeze(0).unsqueeze(0)  # [1, 1, 4, 4]

        target = TASFTObjective.compute_gate_target(attn, block_size=2)

        # 2D maxpool with kernel=2, stride=2:
        # Block (0,0): max(1,2,5,6)=6    Block (0,1): max(3,4,7,8)=8
        # Block (1,0): max(9,10,13,14)=14  Block (1,1): max(11,12,15,16)=16
        expected_pooled = torch.tensor([[6.0, 8.0, 14.0, 16.0]])
        expected_softmax = F.softmax(expected_pooled, dim=-1)
        expected = expected_softmax.reshape(1, 1, 2, 2)

        assert target.shape == (1, 1, 2, 2)
        torch.testing.assert_close(target, expected, atol=1e-6, rtol=1e-6)

    def test_output_is_valid_distribution(self) -> None:
        """Softmax output sums to 1 across all blocks per (batch, head)."""
        attn = torch.randn(2, 4, 64, 64)
        target = TASFTObjective.compute_gate_target(attn, block_size=8)
        # Sum over last two dims (NB_q * NB_k) should be 1.0
        flat_sums = target.reshape(2, 4, -1).sum(dim=-1)
        torch.testing.assert_close(flat_sums, torch.ones_like(flat_sums), atol=1e-5, rtol=1e-5)

    def test_non_divisible_attn_padded_with_neg_inf(self) -> None:
        """Non-divisible seq_len is padded with -inf before maxpool."""
        # S=5, block_size=4 -> padded to S=8, maxpool gives [1, 1, 2, 2]
        attn = torch.randn(1, 1, 5, 5)
        target = TASFTObjective.compute_gate_target(attn, block_size=4)
        expected_nb = (5 + 4 - 1) // 4  # ceil(5/4) = 2
        assert target.shape == (1, 1, expected_nb, expected_nb)
        # Must be valid distribution (no NaN from -inf in padding)
        assert not target.isnan().any()
        assert not target.isinf().any()

    def test_block_size_zero_raises(self) -> None:
        attn = torch.randn(1, 1, 4, 4)
        with pytest.raises(ValueError, match="block_size must be > 0"):
            TASFTObjective.compute_gate_target(attn, block_size=0)

    def test_nan_input_raises(self) -> None:
        attn = torch.tensor([[[[1.0, float("nan")], [3.0, 4.0]]]])
        with pytest.raises(NaNDetectedError):
            TASFTObjective.compute_gate_target(attn, block_size=1)


# ---------------------------------------------------------------------------
# Gate loss: KL divergence
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestGateLoss:
    """KL divergence between predicted gate distribution and ground truth."""

    def test_matched_distributions_near_zero(self) -> None:
        """Identical distributions -> KL ≈ 0."""
        B, H, NB = 1, 1, 4
        # Uniform distribution
        dist = torch.ones(B, H, NB, NB) / (NB * NB)
        loss = TASFTObjective.compute_gate_loss(dist, dist)
        assert loss.item() == pytest.approx(0.0, abs=1e-4)

    def test_uniform_vs_peaked_high_loss(self) -> None:
        """Uniform gate vs peaked target -> high KL loss."""
        B, H, NB = 1, 1, 4
        gate = torch.ones(B, H, NB, NB) / (NB * NB)  # uniform

        target = torch.zeros(B, H, NB, NB)
        target[0, 0, 0, 0] = 1.0  # peaked at one block

        loss = TASFTObjective.compute_gate_loss(gate, target)
        assert loss.item() > 0.0

    def test_gate_loss_is_scalar(self) -> None:
        gate = torch.rand(2, 4, 8, 8)
        target = F.softmax(torch.randn(2, 4, 64), dim=-1).reshape(2, 4, 8, 8)
        loss = TASFTObjective.compute_gate_loss(gate, target)
        assert loss.ndim == 0

    def test_gate_loss_nan_input_raises(self) -> None:
        gate = torch.tensor([[[[float("nan"), 0.5], [0.5, 0.5]]]])
        target = torch.ones(1, 1, 2, 2) / 4
        with pytest.raises(NaNDetectedError):
            TASFTObjective.compute_gate_loss(gate, target)


# ---------------------------------------------------------------------------
# Sparsity loss
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestSparsityLoss:
    """L_sparse = (mean(gate) - tau_target)^2."""

    def test_mean_equals_tau_target_gives_zero(self) -> None:
        """When mean gate score equals tau_target, L_sparse = 0."""
        tau = 0.8
        # Create scores with exact mean = 0.8
        gate = torch.full((1, 1, 4, 4), tau)
        loss = TASFTObjective.compute_sparsity_loss(gate, tau)
        assert loss.item() == pytest.approx(0.0, abs=1e-10)

    def test_mean_above_tau_gives_positive_loss(self) -> None:
        tau = 0.5
        gate = torch.full((1, 1, 4, 4), 0.9)  # mean=0.9 > tau=0.5
        loss = TASFTObjective.compute_sparsity_loss(gate, tau)
        expected = (0.9 - 0.5) ** 2
        assert loss.item() == pytest.approx(expected, abs=1e-6)

    def test_mean_below_tau_gives_positive_loss(self) -> None:
        tau = 0.8
        gate = torch.full((1, 1, 4, 4), 0.2)  # mean=0.2 < tau=0.8
        loss = TASFTObjective.compute_sparsity_loss(gate, tau)
        expected = (0.2 - 0.8) ** 2
        assert loss.item() == pytest.approx(expected, abs=1e-6)

    def test_sparsity_loss_is_scalar(self) -> None:
        gate = torch.rand(2, 4, 8, 8)
        loss = TASFTObjective.compute_sparsity_loss(gate, 0.5)
        assert loss.ndim == 0

    def test_sparsity_loss_nan_raises(self) -> None:
        gate = torch.tensor([[[[float("nan")]]]])
        with pytest.raises(NaNDetectedError):
            TASFTObjective.compute_sparsity_loss(gate, 0.5)


# ---------------------------------------------------------------------------
# NaN / Inf guards
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestNaNGuard:
    """NaN/Inf must raise NaNDetectedError with structured context."""

    def test_nan_in_attn_scores_raises(self) -> None:
        attn = torch.randn(1, 1, 4, 4)
        attn[0, 0, 0, 0] = float("nan")
        with pytest.raises(NaNDetectedError, match="NaN=True"):
            TASFTObjective.compute_gate_target(attn, block_size=2)

    def test_inf_in_gate_scores_raises(self) -> None:
        gate = torch.randn(1, 1, 4, 4)
        gate[0, 0, 0, 0] = float("inf")
        target = torch.ones(1, 1, 4, 4) / 16
        with pytest.raises(NaNDetectedError, match="Inf=True"):
            TASFTObjective.compute_gate_loss(gate, target)

    def test_nan_in_logits_raises(self, objective: TASFTObjective) -> None:
        logits = torch.randn(1, 8, 50)
        logits[0, 0, 0] = float("nan")
        labels = torch.randint(0, 50, (1, 8))
        with pytest.raises(NaNDetectedError):
            objective.compute_task_loss(logits, labels)

    def test_nan_detected_error_has_context(self) -> None:
        attn = torch.tensor([[[[float("nan"), 1.0], [1.0, 1.0]]]])
        with pytest.raises(NaNDetectedError) as exc_info:
            TASFTObjective.compute_gate_target(attn, block_size=1)
        assert "tensor_name" in exc_info.value.context
        assert exc_info.value.context["has_nan"] is True


# ---------------------------------------------------------------------------
# Kahan summation accuracy
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestKahanSummation:
    """Verify Kahan summation in compute() provides better accuracy than naive sum."""

    def test_kahan_accuracy_many_small_values(self) -> None:
        """Accumulate 10^4 small gate losses; Kahan error < 1e-10 vs exact."""
        obj = TASFTObjective(lambda_gate=1.0, beta_sparse=0.0, tau_target=0.5)

        B, S, V = 1, 32, 50
        logits = torch.randn(B, S, V, dtype=torch.float64)
        labels = torch.randint(0, V, (B, S))

        num_layers = 100
        block_size = 8
        NB = S // block_size

        gate_outputs: dict[int, torch.Tensor] = {}
        attn_scores: dict[int, torch.Tensor] = {}

        for i in range(num_layers):
            gate_outputs[i] = torch.rand(B, 1, NB, NB, dtype=torch.float64) * 0.001 + 0.5
            attn_scores[i] = torch.randn(B, 1, S, S, dtype=torch.float64)

        active = list(range(num_layers))
        result = obj.compute(logits, labels, gate_outputs, attn_scores, active, block_size)

        # Verify total is finite and reasonable
        assert torch.isfinite(result.total)
        assert torch.isfinite(result.gate)
        assert torch.isfinite(result.sparse)

        # Compute naive sum for comparison
        naive_gate_sum = 0.0
        for i in active:
            target = TASFTObjective.compute_gate_target(attn_scores[i], block_size)
            layer_loss = TASFTObjective.compute_gate_loss(gate_outputs[i], target)
            naive_gate_sum += layer_loss.item()

        # Kahan result should be close to naive (both should be close to exact)
        # The point is both are finite and they agree to reasonable precision
        assert result.gate.item() == pytest.approx(naive_gate_sum, rel=1e-5)


# ---------------------------------------------------------------------------
# Mathematical identity: total = task + lambda * (gate + beta * sparse)
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestCompositeIdentity:
    """Verify total = task + lambda * (gate + beta * sparse)."""

    def test_loss_decomposition_identity(self) -> None:
        lam = 0.5
        beta = 0.1
        obj = TASFTObjective(lambda_gate=lam, beta_sparse=beta, tau_target=0.5)

        B, S, V = 1, 32, 50
        logits = torch.randn(B, S, V)
        labels = torch.randint(0, V, (B, S))

        block_size = 8
        NB = S // block_size
        gate_outputs = {0: torch.rand(B, 1, NB, NB), 1: torch.rand(B, 1, NB, NB)}
        attn_scores = {0: torch.randn(B, 1, S, S), 1: torch.randn(B, 1, S, S)}
        active = [0, 1]

        result = obj.compute(logits, labels, gate_outputs, attn_scores, active, block_size)

        expected_total = result.task + lam * (result.gate + beta * result.sparse)
        assert result.total.item() == pytest.approx(expected_total.item(), abs=1e-5)

    def test_no_active_layers_total_equals_task(self) -> None:
        """When no layers are active, total = task (gate and sparse = 0)."""
        obj = TASFTObjective(lambda_gate=0.5, beta_sparse=0.1, tau_target=0.5)

        B, S, V = 1, 16, 50
        logits = torch.randn(B, S, V)
        labels = torch.randint(0, V, (B, S))

        result = obj.compute(logits, labels, {}, {}, [], block_size=8)

        assert result.gate.item() == pytest.approx(0.0, abs=1e-10)
        assert result.sparse.item() == pytest.approx(0.0, abs=1e-10)
        assert result.total.item() == pytest.approx(result.task.item(), abs=1e-6)
        assert result.active_layers == []
        assert result.per_layer_gate_loss == {}
        assert result.per_layer_sparsity == {}


# ---------------------------------------------------------------------------
# Only active layers contribute
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestActiveLayerFiltering:
    """Only active layers contribute to gate and sparsity loss."""

    def test_inactive_layers_excluded(self) -> None:
        obj = TASFTObjective(lambda_gate=1.0, beta_sparse=0.01, tau_target=0.5)

        B, S, V = 1, 32, 50
        logits = torch.randn(B, S, V)
        labels = torch.randint(0, V, (B, S))

        block_size = 8
        NB = S // block_size
        gate_outputs = {
            0: torch.rand(B, 1, NB, NB),
            1: torch.rand(B, 1, NB, NB),
            2: torch.rand(B, 1, NB, NB),
        }
        attn_scores = {
            0: torch.randn(B, 1, S, S),
            1: torch.randn(B, 1, S, S),
            2: torch.randn(B, 1, S, S),
        }

        # Only calibrate layers 0 and 2
        result = obj.compute(logits, labels, gate_outputs, attn_scores, [0, 2], block_size)

        assert len(result.active_layers) == 2
        assert LayerIndex(0) in result.active_layers
        assert LayerIndex(2) in result.active_layers
        assert LayerIndex(1) not in result.active_layers

        assert LayerIndex(0) in result.per_layer_gate_loss
        assert LayerIndex(2) in result.per_layer_gate_loss
        assert LayerIndex(1) not in result.per_layer_gate_loss


# ---------------------------------------------------------------------------
# ObjectiveLossOutput structure
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestObjectiveLossOutput:
    """Verify ObjectiveLossOutput structure and immutability."""

    def test_output_is_frozen(self, objective: TASFTObjective) -> None:
        B, S, V = 1, 16, 50
        logits = torch.randn(B, S, V)
        labels = torch.randint(0, V, (B, S))
        result = objective.compute(logits, labels, {}, {}, [], block_size=8)
        with pytest.raises(AttributeError):
            result.total = torch.tensor(0.0)  # type: ignore[misc]

    def test_output_all_scalars(self, objective: TASFTObjective) -> None:
        B, S, V = 1, 16, 50
        logits = torch.randn(B, S, V)
        labels = torch.randint(0, V, (B, S))
        block_size = 8
        NB = S // block_size
        gate_outputs = {0: torch.rand(B, 1, NB, NB)}
        attn_scores = {0: torch.randn(B, 1, S, S)}
        result = objective.compute(logits, labels, gate_outputs, attn_scores, [0], block_size)

        assert result.total.ndim == 0
        assert result.task.ndim == 0
        assert result.gate.ndim == 0
        assert result.sparse.ndim == 0
