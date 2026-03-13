# TASFT: Task-Aware Sparse Fine-Tuning

## Architecture Overview

TASFT co-trains domain fine-tuning (LoRA) with sparse attention gates (SeerAttention-based AttnGate)
in a single training run. The result is a model that maintains task quality while enabling 2-5x
decode throughput via block-sparse attention at inference time.

### Core Pipeline
1. **Co-Training**: Dual objective L = L_task + λ·L_gate trains LoRA adapters and AttnGate modules
2. **Gate Calibration**: Layer-rotating schedule exposes each layer's full attention for ground-truth
3. **Bundle Export**: Trained gates + adapters exported as SafeTensors bundle with SHA256 integrity
4. **Sparse Inference**: Pre-computed block masks drive Triton sparse attention kernel for fast decode

### Key Papers
- SeerAttention (arxiv:2410.13276): learnable block-sparse attention via gating
- Attention head shifts during fine-tuning (arxiv:2409.15820): motivation for task-aware sparsity

## Package Structure

- `tasft/modules/` — AttnGate nn.Module, model wrappers
- `tasft/training/` — Dual-loss trainer, gate calibration scheduler
- `tasft/kernels/` — Triton block-sparse FlashAttention-2 kernel
- `tasft/inference/` — Sparse decode engine, KV-cache optimization
- `tasft/eval/` — lm-eval integration, sparsity-quality analysis
- `tasft/bundle/` — SafeTensors export/import with manifest
- `tasft/observability/` — structlog, Prometheus metrics, OpenTelemetry tracing
- `tasft/types.py` — All shared type aliases (import from here only)
- `tasft/exceptions.py` — Full exception hierarchy with structured context

## Key Invariants

1. All type aliases live in `tasft/types.py` — never redefined elsewhere
2. All exceptions inherit from `TASFTError` and carry `context: dict[str, Any]`
3. Sparsity ratios are always in [0, 1] — enforced at construction
4. Block masks are boolean tensors of shape [B, H, NB_q, NB_k]
5. Gate scores are float tensors in [0, 1] after sigmoid
6. Bundle checksums use SHA-256 — no other hash algorithm
7. Structured logging via structlog — no print statements in library code
8. All public APIs are fully typed — mypy --strict clean

## Build & Test

```bash
make install    # editable install + pre-commit hooks
make lint       # ruff check + format
make typecheck  # mypy --strict
make test       # unit + integration (excludes slow)
make bench      # pytest-benchmark
make chaos      # stress/fault-injection tests
```

## Standards

- Python 3.11+, PEP 621 packaging via hatchling
- Zero warnings from ruff, mypy, pytest
- No TODOs, no placeholders, no partial implementations
- Every module fully typed, tested, and observable
