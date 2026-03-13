"""
TASFT observability: structured logging, metrics, and distributed tracing.

This package provides the complete observability stack for TASFT:
- structlog-based structured logging with context binding
- Prometheus metrics for golden signals (Latency, Traffic, Errors, Saturation)
- OpenTelemetry tracing for distributed pipeline correlation
- Alerting rules as code for Prometheus
"""
from tasft.observability.logging import bind_context, configure_logging, get_logger, timed_operation
from tasft.observability.metrics import MetricsContext, TASFTMetrics, track_step
from tasft.observability.tracing import (
    get_tracer,
    init_tracing,
    trace_gate_calibration,
    trace_inference_request,
    trace_training_step,
)

__all__ = [
    "MetricsContext",
    "TASFTMetrics",
    "bind_context",
    "configure_logging",
    "get_logger",
    "get_tracer",
    "init_tracing",
    "timed_operation",
    "trace_gate_calibration",
    "trace_inference_request",
    "trace_training_step",
    "track_step",
]
