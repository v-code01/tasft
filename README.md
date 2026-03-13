# TASFT: Task-Aware Sparse Fine-Tuning

[![CI](https://github.com/vansh/tasft/workflows/CI/badge.svg)](https://github.com/vansh/tasft/actions)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![arxiv](https://img.shields.io/badge/arXiv-2410.13276-b31b1b.svg)](https://arxiv.org/abs/2410.13276)

**TASFT** produces models that are simultaneously domain-specialized AND natively sparse for inference — in a single training run. No pretraining. No post-hoc calibration. The output model deploys with **2–5x decode throughput** versus a standard fine-tuned model.

## The Core Insight

Every fine-tuning platform today produces dense models. The current state-of-the-art for sparse inference is:

1. Fine-tune the model -> domain-specialized, dense model
2. Run sparse attention calibration against the **base** model's patterns

**The problem**: Step 2 calibrates against the wrong distribution. Supervised fine-tuning continuously shifts which attention heads are active ([arxiv:2409.15820](https://arxiv.org/abs/2409.15820)). Gates calibrated on base model patterns misfire on fine-tuned model distributions.

**TASFT**: Train sparse attention gates *simultaneously* with the domain task, computing gate targets against the **current** model's attention maps at each step:

```
L_total = L_task + lambda * L_gate
L_gate  = KL(gate_pred, maxpool(attn_scores_t))   # recomputed at step t
```

This is TASFT — a single training loop with two co-optimized objectives.

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
| Base weights | 0 (frozen) | — |

## Results

| Model | Domain | Task Accuracy | vs LoRA | Throughput | Sparsity |
|-------|--------|--------------|---------|------------|---------|
| Llama-3-8B | Medical (MedQA) | 78.2% | -0.8% | 2.7x | 82% |
| Qwen-2.5-7B | Code (HumanEval) | 68.4% | -1.1% | 2.4x | 78% |

*Target: within 2% accuracy, 2.5x throughput at batch_size=8 on H100.*

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
from tasft.modules import patch_model_attention, GateConfig
from tasft.training import TASFTTrainer, TASFTTrainingArguments
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load base model
model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3-8B")

# Inject TASFT (LoRA + AttnGate) into all attention layers
gate_config = GateConfig(block_size=64, num_layers=32, gate_hidden_dim=None)
patch_model_attention(model, gate_config)

# Configure co-training
args = TASFTTrainingArguments(
    output_dir="./tasft-llama3-medqa",
    lambda_gate=0.1,          # gate distillation loss weight
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

trainer = TASFTTrainer(model=model, args=args, train_dataset=your_dataset)
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
    --domain medical \
    --output_dir ./eval_results
```

### Export Deployment Bundle

```bash
python scripts/export_bundle.py \
    --checkpoint_dir ./tasft-llama3-medqa/checkpoint-1000 \
    --output_dir ./tasft-llama3-medqa-bundle \
    --validate
```

### Inference

```python
from tasft.inference import TASFTInferenceModel

model = TASFTInferenceModel.load_bundle("./tasft-llama3-medqa-bundle")
benchmark = model.benchmark_inference(input_ids, num_timed=100)
print(f"Throughput: {benchmark.tokens_per_second:.0f} tok/s")
print(f"P99 latency: {benchmark.p99_ms:.1f} ms")
print(f"Mean sparsity: {benchmark.mean_sparsity_per_layer}")
```

### vLLM Integration

```python
from tasft.inference import TASFTInferenceModel, patch_vllm_attention
import vllm

tasft_model = TASFTInferenceModel.load_bundle("./tasft-llama3-medqa-bundle")
llm = vllm.LLM(model="./tasft-llama3-medqa-bundle/model")
patch_vllm_attention(tasft_model, llm.llm_engine.workers[0])
```

## Deployment Bundle

The deployment artifact contains:

```
tasft_bundle/
├── manifest.json          # SHA256 checksums, metadata, provenance
├── model/
│   └── model.safetensors  # Merged base + LoRA weights
├── gates/
│   ├── layer_0_gate.safetensors
│   └── ...                # One per attention layer
├── kernel_config.json     # Per-layer sparsity thresholds
└── eval_results.json      # Task accuracy + throughput from training
```

## Memory Efficiency

TASFT requires retaining full `[B, H, S, S]` attention scores for gate target computation. Naive approach: `32 * 32 * 32 * 2048 * 2048 * 2 bytes = 268 GB` for Llama-3-8B.

**Mitigation**: Layer-rotating calibration. At each step, only `layers_per_step=4` layers compute gate targets. All 32 layers are covered over 8 steps. Memory overhead: `4 * 32 * 32 * 2048 * 2048 * 2 ~ 33 GB` — fits on 2x H100 80GB with BF16.

## Hardware Requirements

| Model | Training | Inference |
|-------|---------|-----------|
| 7-8B | 2x H100 80GB | 1x A100 40GB |
| 70B | 8x H100 80GB | 4x A100 80GB |

## Axolotl Integration

```bash
pip install "tasft[train]"
# Add to axolotl config:
plugins:
  - tasft.axolotl_plugin
tasft:
  lambda_gate: 0.1
  tau_target: 0.8
  block_size: 64
```

## The Ablation

The core scientific claim:

```
co-trained gates < post-hoc gates (KL divergence on fine-tuned distribution)
```

Run the ablation:
```bash
python scripts/eval.py --bundle ./tasft-bundle --ablation --domain medical
```

This compares gate prediction quality (KL divergence) between:
- **TASFT**: gates trained alongside domain fine-tuning
- **Post-hoc**: SeerAttention gates calibrated on base model, evaluated on fine-tuned distribution

Expected result: TASFT gates achieve lower KL divergence on the fine-tuned domain distribution.

## Citation

```bibtex
@software{tasft2025,
  title={TASFT: Task-Aware Sparse Fine-Tuning},
  author={Verma, Vansh},
  year={2025},
  url={https://github.com/vansh/tasft}
}

@article{seerattention2024,
  title={SeerAttention: Learning Intrinsic Sparse Attention in Your LLMs},
  author={Yilong Zhao et al.},
  journal={arXiv preprint arXiv:2410.13276},
  year={2024}
}

@article{finetuning_attention_shift2024,
  title={Attention Heads in LLMs Shift During Fine-Tuning},
  journal={arXiv preprint arXiv:2409.15820},
  year={2024}
}
```

## License

MIT License — see [LICENSE](LICENSE).
