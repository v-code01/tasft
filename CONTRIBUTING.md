# Contributing to TASFT

Thank you for your interest in contributing to TASFT. This document covers the development workflow, standards, and submission process.

## Development Setup

### Prerequisites

- Python 3.11+
- CUDA 12.1+ (for GPU kernel development)
- Git

### Environment

```bash
git clone https://github.com/vansh/tasft.git
cd tasft
python -m venv .venv
source .venv/bin/activate
pip install -e ".[core,train,eval,dev]"
pre-commit install
```

### Verify Setup

```bash
make test-unit       # unit tests
make lint            # ruff + mypy
make test            # full suite
```

## Project Structure

```
tasft/
├── tasft/
│   ├── modules/        # AttnGate, TASFTAttention, patching
│   ├── training/       # TASFTTrainer, objectives, schedulers
│   ├── kernels/        # Triton block-sparse attention
│   ├── inference/      # Runtime, vLLM integration
│   ├── eval/           # Evaluation harness
│   ├── bundle/         # Export and manifest
│   ├── observability/  # Logging, metrics, tracing
│   ├── types.py        # Shared type definitions
│   └── exceptions.py   # Exception hierarchy
├── tests/
│   ├── unit/           # Fast, isolated tests
│   ├── integration/    # End-to-end training loop tests
│   └── benchmarks/     # Performance regression tests
├── configs/            # Training YAML configs
├── scripts/            # CLI entrypoints
└── axolotl_plugin/     # Axolotl integration
```

## Code Standards

### Style

- **Formatter**: `ruff format` (line length 100)
- **Linter**: `ruff check` with strict rules
- **Type checker**: `mypy --strict`
- Zero warnings policy: code must pass all checks with zero warnings

### Testing

- **Unit tests**: required for all new modules. Mathematical operations require 100% coverage.
- **Integration tests**: required for any change to the training loop or inference pipeline.
- **Benchmarks**: required for any change touching hot paths (attention, kernels, gate forward).
- No `time.sleep()` in tests. Use polling with exponential backoff:
  ```python
  def poll(condition, timeout_s=10.0, interval_s=0.05):
      deadline = time.monotonic() + timeout_s
      while time.monotonic() < deadline:
          if condition():
              return True
          time.sleep(interval_s)
          interval_s = min(interval_s * 2, 1.0)
      raise TimeoutError
  ```

### Documentation

- All public functions must have docstrings with: purpose, args, returns, complexity, side effects.
- Non-obvious logic must have inline comments explaining **why**, not what.

### Logging

All log statements must use structured logging with `structlog`:
```python
logger.info("gate_calibration_step", layer_idx=idx, kl_div=loss.item(), step=step)
```

Required fields: `request_id`, `operation`, `duration_ms` for any I/O operation.

## Commit Format

```
<type>(<scope>): <subject>

<body: what and why>

Complexity: O(...)
Coverage: X%
Performance: +/-X%
Tests: <what was run>

Closes #N
```

**Types**: `feat`, `fix`, `perf`, `refactor`, `test`, `docs`, `build`, `chore`

**Scope**: module name (`modules`, `training`, `kernels`, `inference`, `eval`, `bundle`)

## Pull Request Process

1. **Branch**: Create a feature branch from `main`: `feat/your-feature` or `fix/issue-number`
2. **Implement**: Write code following the standards above
3. **Test**: Run the full suite: `make test && make lint`
4. **PR**: Submit with this template:

### PR Template

```markdown
## Summary
- What this PR does (1-3 bullets)

## Motivation
- Why this change is needed

## Changes
- File-by-file summary of modifications

## Test Plan
- [ ] Unit tests added/updated
- [ ] Integration tests pass
- [ ] Benchmarks show no regression (attach results)
- [ ] mypy + ruff clean

## Performance Impact
- Before: X ms / Y tok/s
- After: X ms / Y tok/s
- Delta: +/-Z%
```

5. **Review**: Address all review comments. Every PR requires at least one approval.
6. **Merge**: Squash merge into `main`.

## What We Look For in Reviews

- **Correctness**: edge cases handled, invariants maintained, no UB
- **Performance**: no unnecessary allocations on hot paths, proper use of in-place ops
- **Testing**: meaningful tests that would catch regressions, not just happy-path
- **Memory safety**: tensors properly detached, no gradient leaks, resources released
- **Observability**: structured logging for debuggability

## Reporting Issues

Use GitHub Issues with:
- Clear reproduction steps
- Environment details (Python version, CUDA version, GPU model)
- Expected vs actual behavior
- Relevant logs (with `TASFT_LOG_LEVEL=debug`)

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
