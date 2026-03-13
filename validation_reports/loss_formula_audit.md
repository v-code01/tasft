# Loss Formula Audit: Sparsity Regularization (beta * L_sparse) End-to-End

**Auditor**: Agent 1 (Loss Formula Auditor)
**Date**: 2026-03-13
**Scope**: Trace beta_sparse from YAML config through trainer to backward pass

---

## 1. Specification

The intended loss decomposition is:

```
L_total = L_task + lambda_gate * (L_gate + beta_sparse * L_sparse)
L_sparse = (mean(gate_scores) - tau_target)^2
```

---

## 2. Link-by-Link Trace

### Link 1: Config Files

**File**: `/Users/vanshverma/tasft/configs/llama3_8b_medqa.yaml` (lines 37-44)
**File**: `/Users/vanshverma/tasft/configs/qwen25_7b_code.yaml` (lines 37-44)

Both configs specify under the `objective:` section:

```yaml
objective:
  lambda_gate: 0.1
  beta_sparse: 0.01
  tau_target: 0.8
```

**VERDICT**: PASS. Both configs define `beta_sparse: 0.01` and `tau_target: 0.8`.

---

### Link 2: Config -> TASFTTrainingArguments (scripts/train.py)

**File**: `/Users/vanshverma/tasft/scripts/train.py` (lines 199-202)

```python
lambda_gate=obj.get("lambda_gate", 0.1),
beta_sparse=obj.get("beta_sparse", 0.01),
tau_target=obj.get("tau_target", 0.8),
```

Where `obj = cfg.get("objective", {})` (line 162).

**VERDICT**: PASS. `beta_sparse` is read from `cfg["objective"]["beta_sparse"]` and passed to `TASFTTrainingArguments`. Default fallback is `0.01` (matches config), not `0.0`.

---

### Link 3: TASFTTrainingArguments Definition

**File**: `/Users/vanshverma/tasft/tasft/training/trainer.py` (lines 87-89)

```python
beta_sparse: float = field(
    default=0.01,
    metadata={"help": "Weight for sparsity regularization"},
)
```

Validation at line 128-130:

```python
if self.beta_sparse < 0.0:
    msg = f"beta_sparse must be >= 0, got {self.beta_sparse}"
    raise ValueError(msg)
```

**VERDICT**: PASS. Default is `0.01` (non-zero), and validation allows `>= 0`. The default being non-zero means even without config, sparsity regularization is active.

---

### Link 4: TASFTTrainingArguments -> TASFTObjective

**File**: `/Users/vanshverma/tasft/tasft/training/trainer.py` (lines 196-201)

```python
self._objective = TASFTObjective(
    lambda_gate=args.lambda_gate,
    beta_sparse=args.beta_sparse,
    tau_target=args.tau_target,
    label_smoothing=getattr(args, "label_smoothing_factor", 0.0),
)
```

**VERDICT**: PASS. `beta_sparse` is forwarded from `TASFTTrainingArguments` directly into `TASFTObjective.__init__`.

---

### Link 5: TASFTObjective.__init__ Storage

**File**: `/Users/vanshverma/tasft/tasft/training/objectives.py` (lines 103-126)

```python
def __init__(
    self,
    lambda_gate: float = 0.1,
    beta_sparse: float = 0.01,
    tau_target: float = 0.8,
    ...
) -> None:
    ...
    if beta_sparse < 0.0:
        msg = f"beta_sparse must be >= 0, got {beta_sparse}"
        raise ValueError(msg)
    ...
    self._beta_sparse: Final[float] = beta_sparse
    self._tau_target: Final[float] = tau_target
```

**VERDICT**: PASS. `beta_sparse` is stored as `self._beta_sparse` (Final, immutable). `tau_target` stored as `self._tau_target`.

---

### Link 6: compute_sparsity_loss Implementation

**File**: `/Users/vanshverma/tasft/tasft/training/objectives.py` (lines 224-241)

```python
@staticmethod
def compute_sparsity_loss(gate_soft_scores: SoftGateScores, tau_target: float) -> torch.Tensor:
    _check_finite(gate_soft_scores, "gate_soft_scores")
    mean_activation = gate_soft_scores.mean()
    return (mean_activation - tau_target) ** 2
```

**VERDICT**: PASS. Exactly implements `(mean(gate_scores) - tau_target)^2` as specified.

---

### Link 7: compute() Method -- Loss Assembly

**File**: `/Users/vanshverma/tasft/tasft/training/objectives.py` (lines 316-360)

Per-layer sparsity loss accumulated via Kahan summation (lines 337-346):

```python
layer_sparse_loss = self.compute_sparsity_loss(gate_scores, self._tau_target)
```

Final assembly (lines 349-350):

```python
gate_total = gate_sum + self._beta_sparse * sparse_sum
total_loss = task_loss + self._lambda_gate * gate_total
```

This expands to: `total = task + lambda * (gate + beta * sparse)`.

**VERDICT**: PASS. The formula matches the specification exactly.

---

### Link 8: training_step() -- Actual Loss Used for Backward

**File**: `/Users/vanshverma/tasft/tasft/training/trainer.py` (lines 409-424)

```python
if gate_outputs_by_layer and gate_warmup_multiplier > 0.0:
    loss_output = self._objective.compute(...)
    loss = (
        loss_output.task
        + self._tasft_args.lambda_gate
        * gate_warmup_multiplier
        * (loss_output.gate + self._tasft_args.beta_sparse * loss_output.sparse)
    )
```

**FINDING -- REDUNDANT LOSS RECOMPUTATION**:

The `training_step` does NOT use `loss_output.total` from `TASFTObjective.compute()`. Instead, it **recomputes** the composite loss from the decomposed components (lines 419-424). This means the loss formula is implemented **twice**:

1. In `TASFTObjective.compute()` at line 349-350.
2. In `TASFTTrainer.training_step()` at lines 419-424.

However, the `training_step` version adds `gate_warmup_multiplier`, which the objective's `compute()` does not account for. This is **intentional** -- the warmup multiplier modulates the gate loss contribution during the warmup phase.

The formulas are **consistent** when `gate_warmup_multiplier == 1.0`:
- Objective: `task + lambda * (gate + beta * sparse)`
- Trainer: `task + lambda * 1.0 * (gate + beta * sparse)`

Both read `beta_sparse` from the same source (`self._tasft_args.beta_sparse` for trainer, `self._beta_sparse` for objective), and both were initialized from the same `TASFTTrainingArguments.beta_sparse` value.

**VERDICT**: PASS with NOTE. The dual computation is intentional for warmup support. The `loss_output.total` field from the objective is computed but never used for backward -- only `loss_output.task`, `loss_output.gate`, and `loss_output.sparse` are used. This is not a bug, but `loss_output.total` is essentially dead computation during training.

---

### Link 9: Backward Pass

**File**: `/Users/vanshverma/tasft/tasft/training/trainer.py` (lines 464-472)

```python
loss = loss / self.args.gradient_accumulation_steps

if loss.requires_grad:
    self.accelerator.backward(loss)

return loss.detach()
```

The `loss` tensor is the recomputed composite (Link 8), which includes `beta_sparse * sparse_sum`. Since `compute_sparsity_loss` uses `gate_soft_scores.mean()` which is a differentiable operation on gate parameters, gradients flow through:

```
backward(loss) -> beta * (mean(gate) - tau)^2 -> d/d(gate) = 2*beta*(mean(gate)-tau)/N
```

**VERDICT**: PASS. Gradients flow correctly through the sparsity loss to gate parameters.

---

## 3. Summary

| Link | Component | File:Line | Status |
|------|-----------|-----------|--------|
| 1 | YAML configs | `configs/*.yaml:39-41` | PASS |
| 2 | Config parsing | `scripts/train.py:201` | PASS |
| 3 | TrainingArgs field | `trainer.py:87-89` | PASS |
| 4 | Args -> Objective | `trainer.py:196-201` | PASS |
| 5 | Objective storage | `objectives.py:113-124` | PASS |
| 6 | compute_sparsity_loss | `objectives.py:224-241` | PASS |
| 7 | compute() assembly | `objectives.py:349-350` | PASS |
| 8 | training_step() loss | `trainer.py:419-424` | PASS (with note) |
| 9 | backward() | `trainer.py:464-472` | PASS |

---

## 4. Broken Links Found

**None.** The chain is fully connected end-to-end.

---

## 5. Notes and Minor Observations

### 5a. Dead Computation in TASFTObjective.compute()

`ObjectiveLossOutput.total` (line 353) is computed inside `TASFTObjective.compute()` but never consumed by `training_step()`. The trainer recomputes the total from decomposed components to incorporate the warmup multiplier. The `total` field is only useful for logging or if `compute()` is called outside the trainer context.

**Severity**: Informational. No correctness impact. The dead computation is O(1) (two scalar multiplications and an addition) so the performance impact is negligible.

### 5b. Formula Consistency Verification

Both locations compute the same formula when `gate_warmup_multiplier == 1.0`:

- `objectives.py:349`: `task + lambda * (gate + beta * sparse)`
- `trainer.py:419-424`: `task + lambda * warmup * (gate + beta * sparse)`

If the formula were ever changed, it would need updating in both locations. This is a minor maintenance risk.

### 5c. Default Values Alignment

| Parameter | objectives.py | trainer.py | configs | scripts/train.py |
|-----------|---------------|------------|---------|-------------------|
| beta_sparse | 0.01 | 0.01 | 0.01 | 0.01 (fallback) |
| tau_target | 0.8 | 0.8 | 0.8 | 0.8 (fallback) |
| lambda_gate | 0.1 | 0.1 | 0.1 | 0.1 (fallback) |

All defaults are consistent across all four locations.

---

## 6. Final Verdict

**PASS**: The sparsity regularization term `beta * L_sparse` is correctly implemented end-to-end. Every link in the chain `config -> training_args -> objective -> loss -> backward` is connected and carries the correct value. The loss formula `L_total = L_task + lambda * (L_gate + beta * L_sparse)` is faithfully implemented with no broken links, no silent zeroing, and correct gradient flow.

---

## 7. README Cross-Check (Independent Verification)

**File**: `README.md:21-27`

```
L_total = L_task + λ · (L_gate + β · L_sparse)
L_sparse = (mean(gate_scores) - τ_target)²
```

README formula matches code implementation exactly. The README also correctly describes L_sparse's purpose at line 29: "without it, gates learn to predict the model's (often semi-dense) attention patterns faithfully but never produce sparse masks."

**Second audit pass by code-review agent (2026-03-13): CONFIRMED. All 9 links independently re-verified.**
