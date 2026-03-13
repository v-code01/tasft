"""
Integration test: TASFT inference pipeline.

Tests model patching, gate extraction, bundle validation, and
forward pass correctness using the TinyCausalLM from conftest.

Timeout: 120s for the full pipeline test.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
import torch
from torch import nn
from torch.utils.data import Dataset

from tasft.bundle.export import BundleExporter
from tasft.modules import GateConfig, TASFTAttention, patch_model_attention
from tasft.training import TASFTTrainer, TASFTTrainingArguments

if TYPE_CHECKING:
    from pathlib import Path

    from tests.integration.conftest import TinyCausalLM


def _freeze_and_patch(
    model: nn.Module, gate_config: GateConfig,
) -> dict[int, TASFTAttention]:
    """Freeze all base params and patch attention with gates."""
    for p in model.parameters():
        p.requires_grad = False
    return patch_model_attention(model, gate_config)


class TinyDS(Dataset):
    """20 random-token samples for pipeline integration testing."""

    def __init__(self) -> None:
        torch.manual_seed(77)
        self._ids = torch.randint(0, 128, (20, 64))

    def __len__(self) -> int:
        return 20

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {
            "input_ids": self._ids[idx],
            "labels": self._ids[idx],
            "attention_mask": torch.ones(64, dtype=torch.long),
        }


@pytest.mark.integration
@pytest.mark.timeout(120)
def test_train_and_extract_gate_state(
    tiny_model: TinyCausalLM,
    gate_config: GateConfig,
    tmp_path: Path,
) -> None:
    """Train tiny model -> extract gate state dicts from all layers.

    Verifies:
    - Gate parameters are extractable after training
    - Each layer has gate parameters
    - Gate state dicts are non-empty
    """
    patched_layers = _freeze_and_patch(tiny_model, gate_config)

    args = TASFTTrainingArguments(
        output_dir=str(tmp_path / "train"),
        max_steps=5,
        per_device_train_batch_size=2,
        learning_rate=1e-3,
        lambda_gate=0.1,
        tau_target=0.8,
        use_cpu=True,
        dataloader_num_workers=0,
        report_to="none",
    )

    trainer = TASFTTrainer(
        model=tiny_model,
        args=args,
        patched_layers=patched_layers,
        train_dataset=TinyDS(),
    )
    trainer.train()

    # Verify gate state is extractable
    gate_states: dict[str, torch.Tensor] = {}
    for idx, tasft_attn in patched_layers.items():
        for name, param in tasft_attn.gate.state_dict().items():
            gate_states[f"layer_{idx}.gate.{name}"] = param.data.clone()

    assert len(gate_states) > 0, "No gate parameters extracted"

    # Verify each layer has parameters
    for idx in range(4):
        prefix = f"layer_{idx}.gate."
        layer_params = {k: v for k, v in gate_states.items() if k.startswith(prefix)}
        assert len(layer_params) > 0, f"No parameters for gate at layer {idx}"


@pytest.mark.integration
@pytest.mark.timeout(60)
def test_bundle_validation_rejects_corrupted_manifest(tmp_path: Path) -> None:
    """BundleExporter.validate_bundle rejects a bundle with corrupted manifest."""
    bundle_dir = tmp_path / "bad_bundle"
    bundle_dir.mkdir()

    (bundle_dir / "manifest.json").write_text('{"invalid": true}')
    (bundle_dir / "model").mkdir()
    (bundle_dir / "gates").mkdir()

    result = BundleExporter.validate_bundle(bundle_dir)
    assert not result.is_valid, "Validation should reject corrupted manifest"
    assert len(result.errors) > 0, "Should have at least one error"


@pytest.mark.integration
@pytest.mark.timeout(60)
def test_bundle_validation_rejects_missing_manifest(tmp_path: Path) -> None:
    """BundleExporter.validate_bundle rejects bundle without manifest.json."""
    bundle_dir = tmp_path / "no_manifest"
    bundle_dir.mkdir()

    result = BundleExporter.validate_bundle(bundle_dir)
    assert not result.is_valid
    assert any("manifest" in e.lower() for e in result.errors)


@pytest.mark.integration
@pytest.mark.timeout(60)
def test_gate_output_shape_consistency_after_training(
    tiny_model: TinyCausalLM,
    gate_config: GateConfig,
    tmp_path: Path,
) -> None:
    """After training, gate forward produces correct output shapes.

    Verifies that the trained gate modules produce outputs with shapes
    matching the expected block grid dimensions for the configured block_size.
    """
    patched_layers = _freeze_and_patch(tiny_model, gate_config)

    args = TASFTTrainingArguments(
        output_dir=str(tmp_path / "train"),
        max_steps=3,
        per_device_train_batch_size=2,
        learning_rate=1e-3,
        lambda_gate=0.1,
        tau_target=0.8,
        use_cpu=True,
        dataloader_num_workers=0,
        report_to="none",
    )

    trainer = TASFTTrainer(
        model=tiny_model,
        args=args,
        patched_layers=patched_layers,
        train_dataset=TinyDS(),
    )
    trainer.train()

    # Gate config: block_size=32, num_heads=4, head_dim=16
    seq_len = 64
    num_heads = 4
    head_dim = 16
    block_size = gate_config.block_size
    num_blocks = seq_len // block_size  # 64 / 32 = 2

    for idx, tasft_attn in patched_layers.items():
        gate = tasft_attn.gate
        q = torch.randn(1, num_heads, seq_len, head_dim)
        k = torch.randn(1, num_heads, seq_len, head_dim)

        with torch.no_grad():
            output = gate(q, k)

        assert output.soft_scores.shape == (1, num_heads, num_blocks, num_blocks), (
            f"Layer {idx} gate output shape {output.soft_scores.shape} != "
            f"expected (1, {num_heads}, {num_blocks}, {num_blocks})"
        )
        assert output.soft_scores.min() >= 0.0, "Gate scores below 0"
        assert output.soft_scores.max() <= 1.0, "Gate scores above 1"
        assert not output.soft_scores.isnan().any(), f"NaN in gate output at layer {idx}"
        assert not output.soft_scores.isinf().any(), f"Inf in gate output at layer {idx}"


@pytest.mark.integration
@pytest.mark.timeout(60)
def test_model_forward_pass_after_patching(
    tiny_model: TinyCausalLM,
    gate_config: GateConfig,
) -> None:
    """Patched model must still produce valid logits on forward pass."""
    _freeze_and_patch(tiny_model, gate_config)

    input_ids = torch.randint(0, 128, (1, 32))
    attention_mask = torch.ones(1, 32, dtype=torch.long)

    tiny_model.eval()
    with torch.no_grad():
        outputs = tiny_model(input_ids=input_ids, attention_mask=attention_mask)

    logits = outputs.logits
    assert logits.shape == (1, 32, 128), f"Wrong logits shape: {logits.shape}"
    assert not logits.isnan().any(), "Forward pass produced NaN logits"
    assert not logits.isinf().any(), "Forward pass produced Inf logits"


@pytest.mark.integration
@pytest.mark.timeout(60)
def test_forward_produces_valid_loss_with_labels(
    tiny_model: TinyCausalLM,
    gate_config: GateConfig,
) -> None:
    """Patched model with labels must produce a valid loss for backprop."""
    _freeze_and_patch(tiny_model, gate_config)

    input_ids = torch.randint(0, 128, (2, 32))
    labels = input_ids.clone()
    attention_mask = torch.ones(2, 32, dtype=torch.long)

    outputs = tiny_model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels,
    )

    assert outputs.loss is not None, "Model did not produce a loss"
    assert outputs.loss.ndim == 0, "Loss should be scalar"
    assert torch.isfinite(outputs.loss), f"Loss is not finite: {outputs.loss.item()}"
    assert outputs.loss.item() > 0, "Cross-entropy loss should be positive"
