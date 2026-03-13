# TASFT: Task-Aware Sparse Fine-Tuning

[![CI](https://github.com/v-code01/tasft/workflows/CI/badge.svg)](https://github.com/v-code01/tasft/actions)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**TASFT** co-trains domain fine-tuning (LoRA) with sparse attention gates in a single training run. The output model deploys with block-sparse attention for **2–5x decode throughput** versus a standard fine-tuned model, with no post-hoc calibration step.

## The Core Insight

The current state-of-the-art for sparse inference is:

1. Fine-tune the model (LoRA) → domain-specialized, dense model
2. Run sparse attention calibration against the **base** model's attention patterns

**The problem**: Step 2 calibrates against the wrong distribution. Fine-tuning shifts which attention heads are active ([arxiv:2409.15820](https://arxiv.org/abs/2409.15820)). Gates calibrated on base model patterns misfire on fine-tuned distributions.

**TASFT**: Train sparse attention gates *simultaneously* with the domain task, computing gate targets against the **current** model's attention maps at each step:

```
L_total = L_task + λ · (L_gate + β · L_sparse)

where:
  L_task   = CE(logits[:-1], labels[1:])                      # standard next-token prediction
  L_gate   = KL(gate_pred || softmax(maxpool(attn_scores_t)))  # gate distillation, recomputed at step t
  L_sparse = (mean(gate_scores) - τ_target)²                  # sparsity regularization
```

The sparsity regularization term `L_sparse` is critical: without it, gates learn to predict the model's (often semi-dense) attention patterns faithfully but never produce sparse masks. The `β · (mean - τ)²` penalty forces mean gate activation toward the target sparsity `τ`, preventing degenerate all-dense solutions. This follows SeerAttention's design ([arxiv:2410.13276](https://arxiv.org/abs/2410.13276)).

## Architecture

```
+-------------------------------------------------------------------+
|                    TASFT Attention Layer                           |
|                                                                   |
|  Input hidden states                                              |
|         |                                                         |
|         v                                                         |
|  +----------------+    +--------------------------------------+   |
|  | Frozen W_q/k   |    | LoRA: dW = B*A (rank-16)            |   |
|  | Frozen W_v/o   |    | Trainable: domain task specialization|   |
|  +-------+--------+    +--------------------------------------+   |
|          | Q, K, V                                                |
|          |                                                        |
|  +-------+------------------------------------------------+      |
|  |               AttnGate (co-trained)                     |      |
|  |  Pool(Q) --+                                            |      |
|  |            +---> Linear(2D->hidden) -> sigmoid -> mask  |      |
|  |  Pool(K) --+                                            |      |
|  |                                                         |      |
|  |  TRAINING: full attn -> maxpool -> KL target            |      |
|  |  INFERENCE: gate mask -> BlockSparseFlashAttention       |      |
|  +---------------------------------------------------------+      |
+-------------------------------------------------------------------+
```

**Trainable components per layer:**
| Component | Parameters | Purpose |
|-----------|-----------|---------|
| LoRA (rank-16) | ~0.5% of layer | Task specialization |
| AttnGate | ~0.05% of layer | Block importance prediction |
| Base weights | 0 (frozen) | N/A |

## Results

| Model | Domain | Accuracy | Throughput | Sparsity |
|-------|--------|----------|------------|----------|
| Llama-3-8B | Medical (MedQA) | within 2% of LoRA baseline | 2-3x on H100 | 70-85% |
| Qwen-2.5-7B | Code (HumanEval) | within 2% of LoRA baseline | 2-3x on H100 | 70-85% |

*Throughput gains depend on sparsity achieved and batch size. The theoretical speedup model is `1 / (1 - sparsity × (1 - gate_overhead))`; real-world speedup is lower due to memory bandwidth, kernel launch overhead, and load imbalance. SeerAttention reports 5.67x at 90% sparsity on Llama-3-8B (H100).*

## Installation

```bash
# Core installation
pip install -e ".[core]"

# For training
pip install -e ".[core,train]"

# For evaluation
pip install -e ".[core,eval]"

# Full development
pip install -e ".[core,train,eval,dev]"
pre-commit install
```

## Quickstart

### Training

```python
from tasft.modules.tasft_attention import GateConfig, patch_model_attention
from tasft.training.trainer import TASFTTrainer, TASFTTrainingArguments
from transformers import AutoModelForCausalLM

# Load base model
model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3-8B")

# Inject TASFT (LoRA + AttnGate) into all attention layers
gate_config = GateConfig(block_size=64, num_layers=32)
patched_layers = patch_model_attention(model, gate_config)

# Configure co-training
args = TASFTTrainingArguments(
    output_dir="./tasft-llama3-medqa",
    lambda_gate=0.1,          # gate distillation loss weight
    beta_sparse=0.01,         # sparsity regularization weight
    tau_target=0.8,           # target 80% attention sparsity
    gate_lr_ratio=0.1,        # gate LR = 0.1 * LoRA LR
    gate_warmup_steps=100,    # LoRA-only warmup before gate training
    layers_per_step=4,        # calibrate 4 layers per step (memory efficiency)
    block_size=64,            # 64-token attention blocks
    learning_rate=2e-4,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    num_train_epochs=3,
    bf16=True,
)

trainer = TASFTTrainer(
    model=model, args=args, patched_layers=patched_layers, train_dataset=your_dataset,
)
trainer.train()
```

### Training from Config File

```bash
torchrun --nproc_per_node=2 scripts/train.py --config configs/llama3_8b_medqa.yaml
```

### Evaluation

```bash
python scripts/eval.py \
    --bundle ./tasft-llama3-medqa-bundle \
    --eval-type task gate throughput
```

### Export Deployment Bundle

```bash
python scripts/export_bundle.py \
    --checkpoint ./tasft-llama3-medqa/checkpoint-best \
    --output ./tasft-llama3-medqa-bundle \
    --verify
```

### Inference

```python
from tasft.inference.tasft_model import TASFTInferenceModel

model = TASFTInferenceModel.load_bundle("./tasft-llama3-medqa-bundle")
benchmark = model.benchmark_inference(input_ids, num_timed=100)
print(f"Throughput: {benchmark.tokens_per_second:.0f} tok/s")
print(f"P99 latency: {benchmark.p99_ms:.1f} ms")
```

### vLLM Integration

```python
from tasft.inference.tasft_model import TASFTInferenceModel
from tasft.inference.vllm_patch import patch_vllm_attention
import vllm

tasft_model = TASFTInferenceModel.load_bundle("./tasft-llama3-medqa-bundle")
llm = vllm.LLM(model="./tasft-llama3-medqa-bundle/model")
patch_vllm_attention(tasft_model, llm.llm_engine.workers[0])
```

> **Note**: vLLM integration requires vLLM >= 0.4.0. The monkey-patch is thread-safe (module-level lock), idempotent, and reversible via `unpatch_vllm_attention()`.

## Deployment Bundle

```
tasft_bundle/
├── manifest.json          # SHA256 checksums, metadata, provenance
├── model/
│   └── model.safetensors  # Merged base + LoRA weights
├── gates/
│   ├── layer_0_gate.safetensors
│   └── ...                # One per attention layer
├── kernel_config.json     # Per-layer sparsity thresholds
└── eval_results.json      # Task accuracy + throughput (optional)
```

## Memory Efficiency

Gate calibration requires retaining full `[B, H, S, S]` attention scores to compute ground-truth block importance via 2D max-pooling. The memory cost is:

```
memory = layers_active × B × H × S² × dtype_bytes
```

| Config | layers_active | B | H | S | Memory |
|--------|--------------|---|---|------|--------|
| Naive (all layers) | 32 | 4 | 32 | 2048 | 32.0 GB |
| Layer rotation (4/32) | 4 | 4 | 32 | 2048 | 4.0 GB |
| Layer rotation (4/32) | 4 | 8 | 32 | 2048 | 8.0 GB |

**Layer-rotating calibration**: at each step, only `layers_per_step` layers (default 4) compute gate targets. All 32 layers are covered over `ceil(32/4) = 8` steps via round-robin scheduling. This reduces peak activation memory from 32.0 GB to 4.0 GB at batch_size=4, making single-H100 training feasible.

At batch_size=1, the naive approach uses only 8.0 GB, but realistic training batch sizes of 4-8 make layer rotation necessary for single-GPU training.

## Hardware Requirements

| Model | Training | Inference |
|-------|---------|-----------|
| 7-8B | 1x H100 80GB (B=4, layers_per_step=4) | 1x A100 40GB |
| 70B | 8x H100 80GB | 4x A100 80GB |

## Axolotl Integration

```yaml
# Add to axolotl config:
plugins:
  - tasft

tasft:
  gate:
    block_size: 64
  objective:
    lambda_gate: 0.1
    beta_sparse: 0.01
    tau_target: 0.8
  layer_rotation:
    strategy: ROUND_ROBIN
    layers_per_step: 4
```

## The Ablation

The core scientific claim:

```
KL(P_ft || G_cotrained) < KL(P_ft || G_posthoc)

where:
  P_ft         = fine-tuned model's actual block-level attention importance
  G_cotrained  = gates trained alongside fine-tuning (TASFT)
  G_posthoc    = gates calibrated on base model, applied to fine-tuned model
```

This is tested via paired t-test across layers with Cohen's d effect size. See `tasft/eval/gate_quality.py` for the implementation.

## Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| Sigmoid (not softmax) on gate output | Allows independent per-block decisions; softmax forces competition between blocks |
| Kahan summation for loss accumulation | Prevents catastrophic cancellation when summing many small gate losses across layers |
| Gate warmup (0 LR for N steps) | Lets LoRA adapters stabilize before gates begin learning; prevents gates from overfitting to pre-shift base patterns |
| Sparsity regularization (β term) | Without it, gates faithfully predict semi-dense attention patterns; the penalty forces actual sparsity |
| Gate LR < LoRA LR | Gates should track LoRA's evolving attention, not drive it; lower LR prevents oscillation |
| Atomic bundle export | temp dir → rename prevents partial/corrupt bundles on crash |

## Citation

```bibtex
@software{tasft2025,
  title={TASFT: Task-Aware Sparse Fine-Tuning},
  author={Verma, Vansh},
  year={2025},
  url={https://github.com/v-code01/tasft}
}
```

### Related Work

- [SeerAttention](https://arxiv.org/abs/2410.13276): learnable block-sparse attention via gating (foundation for AttnGate architecture)
- [Attention Head Shifts During Fine-Tuning](https://arxiv.org/abs/2409.15820): empirical evidence that fine-tuning shifts attention patterns (motivation for co-training)

## License

MIT License. See [LICENSE](LICENSE).
