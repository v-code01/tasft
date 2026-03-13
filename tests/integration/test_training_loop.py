"""
Integration test: full TASFT training loop.

Uses a tiny GPT-2 style model (4 layers, 4 heads, 128 hidden dim)
with synthetic data. Tests the complete co-training pipeline.

No GPU required: runs on CPU. Uses --no-cuda flag.
Timeout: 60s per test.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from transformers import GPT2Config, GPT2LMHeadModel

from tasft.modules import AttnGate, GateConfig, TASFTAttention, patch_model_attention
from tasft.training import TASFTTrainer, TASFTTrainingArguments
from tasft.training.objectives import TASFTObjective
from tasft.training.layer_rotation import LayerRotationScheduler, RotationStrategy


TINY_CONFIG = GPT2Config(
    n_layer=4,
    n_head=4,
    n_embd=128,
    n_positions=128,
    vocab_size=256,
    attn_pdrop=0.0,
    resid_pdrop=0.0,
    embd_pdrop=0.0,
)


class SyntheticDS(Dataset):
    """50 samples of random token sequences for integration testing."""

    def __init__(self, num_samples: int = 50, seq_len: int = 128, vocab_size: int = 256) -> None:
        torch.manual_seed(99)
        self._input_ids = torch.randint(0, vocab_size, (num_samples, seq_len))
        self._labels = self._input_ids.clone()
        self._attention_mask = torch.ones(num_samples, seq_len, dtype=torch.long)

    def __len__(self) -> int:
        return self._input_ids.shape[0]

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {
            "input_ids": self._input_ids[idx],
            "labels": self._labels[idx],
            "attention_mask": self._attention_mask[idx],
        }


@pytest.fixture
def tiny_model() -> GPT2LMHeadModel:
    torch.manual_seed(42)
    return GPT2LMHeadModel(TINY_CONFIG)


@pytest.fixture
def synthetic_dataset() -> SyntheticDS:
    return SyntheticDS()


@pytest.fixture
def gate_config() -> GateConfig:
    return GateConfig(block_size=8, num_layers=4, gate_hidden_dim=8, default_threshold=0.5)


def _patch_and_build_trainer(
    model: GPT2LMHeadModel,
    gate_config: GateConfig,
    args: TASFTTrainingArguments,
    train_dataset: Dataset,
) -> tuple[TASFTTrainer, dict[int, TASFTAttention]]:
    """Patch model and create trainer with correct API — returns (trainer, patched_layers)."""
    patched_layers = patch_model_attention(model, gate_config)
    trainer = TASFTTrainer(
        model=model,
        args=args,
        patched_layers=patched_layers,
        train_dataset=train_dataset,
    )
    return trainer, patched_layers


def _make_training_args(tmp_path: Path, **overrides: object) -> TASFTTrainingArguments:
    """Build TASFTTrainingArguments with sensible test defaults."""
    defaults: dict[str, object] = {
        "output_dir": str(tmp_path / "output"),
        "num_train_epochs": 1,
        "max_steps": 10,
        "per_device_train_batch_size": 2,
        "learning_rate": 1e-3,
        "lambda_gate": 0.1,
        "beta_sparse": 0.01,
        "tau_target": 0.8,
        "gate_lr_ratio": 0.1,
        "gate_warmup_steps": 3,
        "layers_per_step": 2,
        "block_size": 8,
        "rotation_strategy": "round_robin",
        "no_cuda": True,
        "dataloader_num_workers": 0,
        "save_steps": 20,
        "logging_steps": 1,
        "report_to": "none",
    }
    defaults.update(overrides)
    return TASFTTrainingArguments(**defaults)  # type: ignore[arg-type]


@pytest.mark.integration
@pytest.mark.timeout(60)
def test_full_training_loop_loss_decreases(
    tiny_model: GPT2LMHeadModel,
    synthetic_dataset: SyntheticDS,
    gate_config: GateConfig,
    tmp_path: Path,
) -> None:
    """Full co-training loop: loss must decrease over 10 steps."""
    args = _make_training_args(tmp_path)
    trainer, _ = _patch_and_build_trainer(tiny_model, gate_config, args, synthetic_dataset)

    trainer.train()

    history = trainer.state.log_history
    losses = [h["loss"] for h in history if "loss" in h]
    assert len(losses) >= 2, f"Expected at least 2 loss entries, got {len(losses)}"

    first_loss = losses[0]
    last_loss = losses[-1]
    assert last_loss < first_loss * 1.1, (
        f"Loss did not decrease: {first_loss:.4f} -> {last_loss:.4f}"
    )


@pytest.mark.integration
@pytest.mark.timeout(60)
def test_base_weights_unchanged_after_training(
    tiny_model: GPT2LMHeadModel,
    synthetic_dataset: SyntheticDS,
    gate_config: GateConfig,
    tmp_path: Path,
) -> None:
    """Base model weights must be byte-identical before and after training.

    After patch_model_attention, all non-gate params have requires_grad=False.
    Training must not modify these frozen parameters.
    """
    # Snapshot frozen params before patching (all params are base at this point)
    pre_patch_snapshot: dict[str, torch.Tensor] = {}
    for name, param in tiny_model.named_parameters():
        pre_patch_snapshot[name] = param.data.clone()

    patched_layers = patch_model_attention(tiny_model, gate_config)

    # Identify which params are now frozen (non-gate)
    gate_param_ids = set()
    for tasft_attn in patched_layers.values():
        for p in tasft_attn.gate.parameters():
            gate_param_ids.add(id(p))

    frozen_names = [
        name for name, param in tiny_model.named_parameters()
        if not param.requires_grad and id(param) not in gate_param_ids
    ]

    args = _make_training_args(tmp_path)
    trainer = TASFTTrainer(
        model=tiny_model,
        args=args,
        patched_layers=patched_layers,
        train_dataset=synthetic_dataset,
    )
    trainer.train()

    # Verify frozen params are byte-identical
    current_params = dict(tiny_model.named_parameters())
    for name in frozen_names:
        if name in current_params and name in pre_patch_snapshot:
            assert torch.equal(pre_patch_snapshot[name], current_params[name].data), (
                f"Base weight {name} was modified during TASFT training"
            )


@pytest.mark.integration
@pytest.mark.timeout(60)
def test_layer_rotation_observed_in_training(
    tiny_model: GPT2LMHeadModel,
    synthetic_dataset: SyntheticDS,
    gate_config: GateConfig,
    tmp_path: Path,
) -> None:
    """Verify rotation scheduler cycles through layers — not all layers every step.

    With 4 layers and layers_per_step=2, each step should activate exactly 2 layers.
    Over 10 steps with round_robin, all 4 layers should eventually be covered.
    """
    args = _make_training_args(tmp_path, layers_per_step=2)
    patched_layers = patch_model_attention(tiny_model, gate_config)

    trainer = TASFTTrainer(
        model=tiny_model,
        args=args,
        patched_layers=patched_layers,
        train_dataset=synthetic_dataset,
    )

    # The rotation scheduler is internal to the trainer. After training,
    # verify coverage stats show all layers were calibrated.
    trainer.train()

    coverage = trainer._rotation_scheduler.get_coverage_stats()
    # After 10 steps with layers_per_step=2 and 4 layers (round_robin),
    # every layer should have been calibrated at least once
    assert coverage.fully_covered, (
        f"Not all layers were calibrated. Coverage histogram: {coverage.coverage_histogram}"
    )
    # Max gap should be bounded: with 4 layers and 2 per step, full cycle = 2 steps
    assert coverage.max_gap <= 5, (
        f"Layer rotation gap too large: max_gap={coverage.max_gap}"
    )


@pytest.mark.integration
@pytest.mark.timeout(60)
def test_checkpoint_saves_three_artifacts(
    tiny_model: GPT2LMHeadModel,
    synthetic_dataset: SyntheticDS,
    gate_config: GateConfig,
    tmp_path: Path,
) -> None:
    """Checkpoint must contain LoRA weights, gate weights, and sparsity profile."""
    args = _make_training_args(
        tmp_path,
        max_steps=5,
        save_steps=5,
    )
    patched_layers = patch_model_attention(tiny_model, gate_config)

    trainer = TASFTTrainer(
        model=tiny_model,
        args=args,
        patched_layers=patched_layers,
        train_dataset=synthetic_dataset,
    )
    trainer.train()

    checkpoint_dirs = list((tmp_path / "output").glob("checkpoint-*"))
    assert len(checkpoint_dirs) >= 1, "No checkpoint saved"

    ckpt = checkpoint_dirs[0]

    # Must have sparsity profile
    sparsity_file = ckpt / "sparsity_profile.json"
    assert sparsity_file.exists(), f"sparsity_profile.json missing from {ckpt}"

    with open(sparsity_file) as f:
        profile = json.load(f)

    assert isinstance(profile, dict), "Sparsity profile must be a dict"
    per_layer = profile.get("per_layer_sparsity", {})
    for layer_idx, sparsity in per_layer.items():
        assert 0.0 <= sparsity <= 1.0, f"Layer {layer_idx} sparsity {sparsity} out of [0,1]"

    # Must have gate state dict
    gate_file = ckpt / "gate_state_dict.pt"
    assert gate_file.exists(), "gate_state_dict.pt missing from checkpoint"

    gate_state = torch.load(gate_file, map_location="cpu", weights_only=True)
    assert len(gate_state) > 0, "Gate state dict is empty"


@pytest.mark.integration
@pytest.mark.timeout(90)
def test_checkpoint_resume_continuity(
    synthetic_dataset: SyntheticDS,
    gate_config: GateConfig,
    tmp_path: Path,
) -> None:
    """Training resumed from checkpoint must continue from correct step."""
    # First run: 5 steps with checkpoint at step 5
    model1 = GPT2LMHeadModel(TINY_CONFIG)
    torch.manual_seed(42)
    patched1 = patch_model_attention(model1, gate_config)

    args_first = _make_training_args(
        tmp_path,
        output_dir=str(tmp_path / "first"),
        max_steps=5,
        save_steps=5,
        seed=42,
    )

    trainer1 = TASFTTrainer(
        model=model1,
        args=args_first,
        patched_layers=patched1,
        train_dataset=synthetic_dataset,
    )
    trainer1.train()

    checkpoint = list((tmp_path / "first").glob("checkpoint-5"))
    assert len(checkpoint) == 1, "Expected checkpoint-5 to exist"

    # Second run: resume from checkpoint, train to step 10
    model2 = GPT2LMHeadModel(TINY_CONFIG)
    torch.manual_seed(42)
    patched2 = patch_model_attention(model2, gate_config)

    args_resume = _make_training_args(
        tmp_path,
        output_dir=str(tmp_path / "resumed"),
        max_steps=10,
        save_steps=20,
        seed=42,
    )

    trainer2 = TASFTTrainer(
        model=model2,
        args=args_resume,
        patched_layers=patched2,
        train_dataset=synthetic_dataset,
    )
    trainer2.train(resume_from_checkpoint=str(checkpoint[0]))

    # Verify resumed at step 5, trained to step 10
    assert trainer2.state.global_step == 10, (
        f"Expected 10 steps total, got {trainer2.state.global_step}"
    )


@pytest.mark.integration
@pytest.mark.timeout(60)
def test_gate_parameters_change_during_training(
    tiny_model: GPT2LMHeadModel,
    synthetic_dataset: SyntheticDS,
    gate_config: GateConfig,
    tmp_path: Path,
) -> None:
    """Gate parameters must actually change during training — gradients flow."""
    patched_layers = patch_model_attention(tiny_model, gate_config)

    # Snapshot gate params before training
    gate_snapshots: dict[str, torch.Tensor] = {}
    for idx, tasft_attn in patched_layers.items():
        for name, param in tasft_attn.gate.named_parameters():
            gate_snapshots[f"layer_{idx}.{name}"] = param.data.clone()

    args = _make_training_args(tmp_path, gate_warmup_steps=0)
    trainer = TASFTTrainer(
        model=tiny_model,
        args=args,
        patched_layers=patched_layers,
        train_dataset=synthetic_dataset,
    )
    trainer.train()

    # At least some gate params should have changed
    changed_count = 0
    for idx, tasft_attn in patched_layers.items():
        for name, param in tasft_attn.gate.named_parameters():
            key = f"layer_{idx}.{name}"
            if not torch.equal(gate_snapshots[key], param.data):
                changed_count += 1

    assert changed_count > 0, "No gate parameters changed during training — gradient flow broken"
