# TASFT Observability Stack Audit

**Auditor**: obs-auditor agent
**Date**: 2026-03-13
**Scope**: tasft/observability/ (logging, metrics, tracing, alerts) + integration across codebase

---

## 1. Component-Level Review

### 1.1 logging.py — PASS

- **structlog configuration**: Correct processor pipeline (contextvars merge, log level, ISO timestamps in UTC, stack info, exception formatting).
- **TTY detection**: Auto-selects ConsoleRenderer (dev) vs JSONRenderer (prod). Overridable via `TASFT_LOG_JSON=1`.
- **get_logger()**: Binds `module`, `version`, `git_hash` to every logger instance.
- **bind_context()**: Uses `structlog.contextvars` for cross-function context propagation (request_id, step, layer_idx).
- **timed_operation()**: Context manager that logs start + completion with `duration_ms` using `perf_counter_ns`.
- **configure_logging()**: Supports runtime reconfiguration with level filtering.
- **No issues found.** Well-structured, production-grade.

### 1.2 metrics.py — PASS

- **Isolated CollectorRegistry**: Avoids default Prometheus registry pollution.
- **Golden signals per Lambda-11**:
  - **Latency**: `step_duration_seconds` (histogram), `gate_forward_ms` (histogram, per-layer), `sparse_kernel_ms` (histogram, per-layer)
  - **Traffic**: `training_steps_total` (counter), `gate_calibrations_total` (counter, per-layer)
  - **Errors**: `oom_events_total` (counter), `errors_total` (counter, per error_type)
  - **Saturation**: `gpu_memory_used_bytes` (gauge, per-device), `active_layers_count` (gauge), `current_lambda_gate` (gauge)
- **Additional**: `sparsity_ratio` histogram per layer — good for drift detection.
- **Helpers**: `record_step()`, `record_gate_calibration()`, `record_sparsity()`, `record_error()`, `record_oom()`, `set_gpu_memory()`, `set_active_layers()`, `set_lambda_gate()`.
- **MetricsContext**: Generic timing context manager for histograms.
- **track_step()**: Convenience context manager combining counter + histogram.
- **push()**: Pushgateway support for non-server training jobs.
- **No issues found.** Comprehensive metric coverage.

### 1.3 tracing.py — PASS

- **OTLP export**: Deferred import of gRPC exporter (avoids pulling grpc when unused).
- **Noop fallback**: When `otlp_endpoint=None`, uses OTel noop tracer (zero overhead).
- **Span factories**:
  - `trace_training_step(step, active_layers)` — sets tasft.step, tasft.active_layers, tasft.num_active_layers.
  - `trace_gate_calibration(layer_idx)` — child span for per-layer calibration.
  - `trace_inference_request(request_id, seq_len)` — inference request span.
- **Error recording**: All span factories catch exceptions, set StatusCode.ERROR, and call record_exception before re-raising.
- **Resource attributes**: Configurable via `resource_attributes` dict.
- **No issues found.** Clean OTel integration.

### 1.4 alerts.py — PASS (with 1 note)

Six alert rules defined. PromQL syntax analysis:

| Alert | PromQL | Valid? | Notes |
|-------|--------|--------|-------|
| TASFTSparsityBelowTarget | `avg_over_time(tasft_sparsity_ratio_sum[10m]) / avg_over_time(tasft_sparsity_ratio_count[10m]) < 0.5` | YES | Correct histogram mean computation |
| TASFTNaNDetected | `increase(tasft_errors_total{error_type="nan_detected"}[1m]) > 0` | YES | Instant fire (`for: 0m`) — appropriate for NaN |
| TASFTCheckpointFailed | `increase(tasft_errors_total{error_type="checkpoint_failed"}[5m]) > 0` | YES | Instant fire — appropriate |
| TASFTOOMRisk | `tasft_gpu_memory_used_bytes / 1073741824 > 0.9 * 80` | YES | **Note**: Hardcodes 80GB GPU capacity |
| TASFTHighStepLatency | `histogram_quantile(0.99, rate(tasft_step_duration_seconds_bucket[5m])) > 10` | YES | Correct p99 from histogram |
| TASFTHighErrorRate | `rate(tasft_errors_total[5m]) > 0.1` | YES | 0.1 errors/sec threshold |

**Note on TASFTOOMRisk**: The expression hardcodes `0.9 * 80` (72GB) which assumes an 80GB A100/H100. Should be parameterized or use a recording rule with actual GPU capacity. Not a syntax error, but a portability concern.

---

## 2. Integration Audit

### 2.1 print() Usage — PASS

```
grep "print(" tasft/**/*.py → 0 matches
```

Zero `print()` calls in library code. All output goes through structlog. Fully compliant.

### 2.2 get_logger() Adoption

**Modules using structured logging (9/28 .py files)**:

| Module | Logger Source | Status |
|--------|-------------|--------|
| `tasft/training/trainer.py` | `structlog.get_logger()` directly | WARN: bypasses `get_logger()` wrapper |
| `tasft/inference/tasft_model.py` | `get_logger()` | OK |
| `tasft/inference/vllm_patch.py` | `get_logger()` | OK |
| `tasft/eval/throughput_bench.py` | `get_logger()` | OK |
| `tasft/eval/gate_quality.py` | `get_logger()` | OK |
| `tasft/eval/task_eval.py` | `get_logger()` | OK |
| `tasft/bundle/export.py` | `get_logger()` | OK |

**Modules WITHOUT logging (acceptable — data/config/init modules)**:
- `__init__.py` files (7) — package exports only
- `types.py` — type aliases
- `exceptions.py` — exception definitions
- `bundle/bundle_schema.py` — Pydantic schemas
- `observability/metrics.py`, `tracing.py`, `alerts.py` — observability definitions themselves
- `kernels/kernel_config.py` — dataclass

**Modules that SHOULD have logging but don't**:
| Module | Impact | Recommendation |
|--------|--------|----------------|
| `training/objectives.py` | Medium | Log NaN/Inf in loss components before they propagate |
| `training/layer_rotation.py` | Low | Log rotation decisions at DEBUG level |
| `modules/attn_gate.py` | Low | Log gate forward timing at DEBUG level |
| `modules/tasft_attention.py` | Low | Log attention patching events |
| `kernels/block_sparse_fa.py` | Medium | Log kernel dispatch decisions, fallback to dense |

### 2.3 trainer.py Metrics Integration — FAIL

**Critical finding**: `TASFTTrainer.training_step()` does NOT use `TASFTMetrics` at all.

The trainer emits structured logs per step (line 608: `logger.info("training_step", ...)`), which is good for debugging, but Prometheus counters/histograms/gauges are **never updated**:

- `metrics.record_step(duration)` — never called
- `metrics.record_sparsity(layer, ratio)` — never called
- `metrics.record_gate_calibration(layer, forward_ms)` — never called
- `metrics.set_gpu_memory(device, bytes)` — never called
- `metrics.set_active_layers(count)` — never called
- `metrics.set_lambda_gate(value)` — never called
- `metrics.record_error(type)` — never called (NaNDetectedError is raised but not counted)
- `metrics.record_oom()` — never called

**Impact**: All 6 alert rules are dead code — they reference metrics that are never populated. Dashboards built on these metrics will show nothing.

**Severity**: HIGH — Prometheus metrics exist but are disconnected from the training loop. The entire alerting and monitoring story is non-functional.

### 2.4 tasft_model.py Tracing Integration — FAIL

**Critical finding**: `TASFTInferenceModel` does NOT create OpenTelemetry spans.

- `trace_inference_request(request_id, seq_len)` — never called in `forward()` or `load_bundle()`
- `trace_gate_calibration(layer_idx)` — never called
- `trace_training_step(step, active_layers)` — never called in trainer either

The inference model uses `timed_operation()` (structlog-based timing) for bundle loading steps (CHECKSUM_VERIFY, MODEL_LOAD, GATES_LOAD, ATTENTION_PATCH), which provides structured log timing. But no distributed tracing spans are created.

**Impact**: No distributed trace correlation for inference requests. In a multi-service deployment (e.g., behind vLLM), there's no way to trace a request through the TASFT inference path.

**Severity**: MEDIUM — structlog timing is present (provides basic latency visibility), but OTel tracing is defined and unused.

---

## 3. Summary

### Passed (5/9 checks)

| Check | Status |
|-------|--------|
| logging.py structure & API | PASS |
| metrics.py golden signals coverage | PASS |
| tracing.py span factories & error handling | PASS |
| alerts.py PromQL syntax validity | PASS |
| Zero print() in library code | PASS |

### Issues Found (4 findings)

| # | Severity | Finding | Location |
|---|----------|---------|----------|
| 1 | **HIGH** | TASFTMetrics never wired into TASFTTrainer — all Prometheus metrics are zero | `training/trainer.py` |
| 2 | **MEDIUM** | OTel spans defined but never created in inference or training paths | `inference/tasft_model.py`, `training/trainer.py` |
| 3 | **LOW** | trainer.py uses `structlog.get_logger()` directly, bypassing `get_logger()` wrapper (misses version/git_hash binding) | `training/trainer.py:55` |
| 4 | **LOW** | TASFTOOMRisk alert hardcodes 80GB GPU capacity | `observability/alerts.py:100` |

### Recommended Fixes

**Issue #1 (HIGH)**: Inject `TASFTMetrics` into `TASFTTrainer.__init__()` and call `metrics.record_step()`, `metrics.record_sparsity()`, `metrics.set_gpu_memory()`, `metrics.set_active_layers()`, `metrics.set_lambda_gate()` in `training_step()`. Wrap the NaN guard with `metrics.record_error("nan_detected")`. Add OOM handler calling `metrics.record_oom()`.

**Issue #2 (MEDIUM)**: Wrap `TASFTInferenceModel.forward()` with `trace_inference_request()`. Wrap `TASFTTrainer.training_step()` with `trace_training_step()`. Gate calibration spans can be added inside the active layer loop.

**Issue #3 (LOW)**: Change `training/trainer.py:55` from `structlog.get_logger("tasft.training.trainer")` to `from tasft.observability.logging import get_logger; logger = get_logger("tasft.training.trainer")`.

**Issue #4 (LOW)**: Replace hardcoded `0.9 * 80` with a recording rule or Prometheus label-based GPU capacity lookup, or document the 80GB assumption.

---

## 4. Verdict

**The observability stack is well-designed but incompletely integrated.** The four observability modules (logging, metrics, tracing, alerts) are individually production-quality. However, the two critical consumers — TASFTTrainer and TASFTInferenceModel — only use structlog logging. Prometheus metrics and OpenTelemetry tracing are defined but never wired in, making the alerting rules non-functional dead code.

Structured logging alone provides ~60% of the observability story. Wiring in metrics (Issue #1) and tracing (Issue #2) would bring it to 100%.
