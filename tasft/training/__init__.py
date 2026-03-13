"""
TASFT training: dual-objective co-training loop with gate calibration.

This package contains:
- TASFTObjective: Dual-loss computation (L_task + λ·L_gate + β·L_sparse)
- LayerRotationScheduler: Memory-efficient gate calibration cycling
- TASFTTrainer: HuggingFace Trainer subclass with co-training loop
- TASFTTrainingArguments: Extended training arguments for TASFT
"""
from tasft.training.layer_rotation import (
    CoverageStats,
    LayerRotationScheduler,
    RotationStrategy,
    estimate_activation_memory_gb,
)
from tasft.training.objectives import ObjectiveLossOutput, TASFTObjective
from tasft.training.trainer import TASFTTrainer, TASFTTrainingArguments

__all__ = [
    "CoverageStats",
    "LayerRotationScheduler",
    "ObjectiveLossOutput",
    "RotationStrategy",
    "TASFTObjective",
    "TASFTTrainer",
    "TASFTTrainingArguments",
    "estimate_activation_memory_gb",
]
