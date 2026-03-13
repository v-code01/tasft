# TASFT Architecture Compliance Report

**Date**: 2026-03-13
**Validator**: Principal Validation Engineer
**Scope**: All core modules in `tasft/` vs CLAUDE.md architecture spec

---

## 1. AttnGate (`tasft/modules/attn_gate.py`)

| Check | Status | Evidence |
|-------|--------|----------|
| Q/K average pooling | **PASS** | `_pool_to_blocks()` line 145: reshapes to `[B, H, num_blocks, block_size, D]` then `.mean(dim=3)` — correct non-overlapping average pool |
| Block representations shape | **PASS** | Returns `[B, H, NB, D]` after pooling (line 169) |
| Learned linear gate MLP | **PASS** | `gate_proj_in` (Linear 2D -> hidden) + ReLU + `gate_proj_out` (Linear hidden -> 1) at lines 129-130 |
| Sigmoid activation (not softmax) | **PASS** | `torch.sigmoid(logits)` at line 253 |
| Output shape [B, H, NB_q, NB_k] | **PASS** | Outer expansion of q_blocks and k_blocks produces correct shape (lines 244-251) |
| Param count formula | **PASS** | `num_parameters` property sums all `requires_grad` params (line 280) |
| Sparsity ratio in [0, 1] | **PASS** | Wrapped in `SparsityRatio()` newtype at line 256, computed as `1 - mask.float().mean()` |

**Verdict: PASS** — AttnGate fully compliant with spec.

---

## 2. Dual Objective (`tasft/training/objectives.py`)

| Check | Status | Evidence |
|-------|--------|----------|
| L_total = L_task + lambda * L_gate_total | **PASS** | Line 350: `total_loss = task_loss + self._lambda_gate * gate_total` |
| L_gate_total = L_gate + beta * L_sparse | **PASS** | Line 349: `gate_total = gate_sum + self._beta_sparse * sparse_sum` |
| Ground truth via 2D maxpool of CURRENT step attn | **PASS** | `compute_gate_target()` line 178: `F.max_pool2d(attn_scores, ...)` on current step's scores (passed live in `compute()` at line 324) |
| KL divergence implementation | **PASS** | `compute_gate_loss()` line 219-221: normalizes gate scores to distribution, `F.kl_div(log_gate, target_flat, reduction="batchmean")` |
| Sparsity regularization | **PASS** | `compute_sparsity_loss()` line 241: `(mean_activation - tau_target) ** 2` |
| Kahan summation for numerical stability | **PASS** | Lines 307-346: explicit Kahan compensation variables for gate_sum and sparse_sum |
| NaN/Inf guards | **PASS** | `_check_finite()` called on all tensors with structured `NaNDetectedError` |

**Verdict: PASS** — Dual objective fully compliant with spec.

---

## 3. Layer Rotation (`tasft/training/layer_rotation.py`)

| Check | Status | Evidence |
|-------|--------|----------|
| ROUND_ROBIN cycles all L layers in ceil(L/N) steps | **PASS** | `_round_robin()` line 253: `start = (step * N) % L`, wraps modularly. `cycles_for_full_coverage()` returns `ceil(L/N)` (line 288) |
| PRIORITY_WEIGHTED uses EMA(alpha=0.1) | **PASS** | Default `ema_alpha=0.1` (line 114). `report_gate_loss()` line 210: `alpha * loss + (1 - alpha) * ema` |
| Memory estimate formula | **PASS** | `estimate_activation_memory_gb()` line 87: `layers_per_step * B * H * S^2 * dtype_bytes / 1024^3` |
| layers_per_step distinct indices guaranteed | **PASS** | ROUND_ROBIN: modular cycling. RANDOM: `torch.randperm` + slice. PRIORITY_WEIGHTED: `torch.multinomial(replacement=False)` |

**Verdict: PASS** — Layer rotation fully compliant with spec.

---

## 4. Trainer (`tasft/training/trainer.py`)

| Check | Status | Evidence |
|-------|--------|----------|
| Two optimizer groups (LoRA + gate) | **PASS** | `create_optimizer()` lines 274-277: `[{"params": non_gate, "lr": base_lr}, {"params": gate, "lr": gate_lr}]` |
| Gate LR = base_lr * gate_lr_ratio | **PASS** | Line 269: `gate_lr = base_lr * self._tasft_args.gate_lr_ratio` |
| Gate warmup (0 LR then ramp) | **PASS** | `gate_warmup_fn` lines 306-311: returns 0.0 for `step < warmup_steps`, then linear ramp |
| 3-artifact checkpoint | **PASS** | `_save_checkpoint()`: (1) HF checkpoint via `super()._save_checkpoint()` (line 630), (2) `gate_state_dict.pt` (line 645), (3) `sparsity_profile.json` (line 649) |
| Structured logging per step | **PASS** | `_log_training_step()` lines 541-608 with structlog |
| Layer rotation integration | **PASS** | `training_step()` line 366: `self._rotation_scheduler.get_active_layers()` |

**Verdict: PASS** — Trainer fully compliant with spec.

---

## 5. Inference (`tasft/inference/tasft_model.py`)

| Check | Status | Evidence |
|-------|--------|----------|
| Gate runs BEFORE attention | **PASS** | `_SparseAttentionWrapper.forward()` line 280: `gate_output = self.gate(query_states, key_states, ...)` BEFORE sparse/dense attention dispatch |
| SHA-256 verification on bundle load | **PASS** | `load_bundle()` lines 466-487: iterates all checksummed files, calls `_verify_checksum()` which uses `hashlib.sha256()` |
| Block mask drives sparse kernel | **PASS** | Line 291: `kernel.forward(q, k, v, gate_output.hard_mask, causal=True)` |
| Dense fallback when sparsity too low | **PASS** | Line 284: `if gate_output.sparsity_ratio >= self.min_sparsity_for_speedup` routes sparse, else dense |
| All model weights frozen | **PASS** | Lines 507-508: `model.eval()` + `param.requires_grad = False` for all params |

**Verdict: PASS** — Inference model fully compliant with spec.

---

## 6. Bundle Export (`tasft/bundle/export.py`)

| Check | Status | Evidence |
|-------|--------|----------|
| Atomic write (temp -> rename) | **PASS** | `export()` line 160: `tempfile.mkdtemp()`, writes to tmp, then `tmp_dir.rename(output_dir)` at line 177 |
| Cleanup on failure | **PASS** | `except` block line 188: `shutil.rmtree(tmp_dir)` + `raise` |
| SHA-256 checksums | **PASS** | `_sha256()` line 424: `hashlib.sha256()` with 64KB streaming chunks |
| Manifest schema (Pydantic) | **PASS** | `BundleManifest` at `bundle_schema.py` — Pydantic v2 frozen model with field validators |
| Checksum format validation | **PASS** | `validate_checksums` field_validator checks 64-char lowercase hex |

**Verdict: PASS** — Bundle export fully compliant with spec.

---

## Summary

| Component | Status |
|-----------|--------|
| AttnGate | PASS |
| Dual Objective | PASS |
| Layer Rotation | PASS |
| Trainer | PASS |
| Inference | PASS |
| Bundle Export | PASS |

**Overall Architecture Compliance: PASS** — All 6 components match the TASFT specification exactly.
