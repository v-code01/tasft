"""Prometheus metrics for TASFT training and inference.

Defines all counters, histograms, and gauges per Λ₁₁ golden signals:
Latency (step_duration, gate_forward, sparse_kernel), Traffic (steps_total),
Errors (oom_events, errors_total), Saturation (gpu_memory, active_layers).

All metrics are prefixed with `tasft_` for namespace isolation.
"""
from __future__ import annotations

import time
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, Self

from prometheus_client import (
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    push_to_gateway,
)

if TYPE_CHECKING:
    from collections.abc import Generator


class TASFTMetrics:
    """Central metrics registry for TASFT.

    Encapsulates all Prometheus metrics with a dedicated registry to avoid
    collisions with other instrumented libraries. Satisfies Λ₁₁ golden signals.

    Preconditions: Instantiated once per process.
    Postconditions: All metric families registered and ready for recording.
    Complexity: O(1) for all metric operations.
    """

    def __init__(self, registry: CollectorRegistry | None = None) -> None:
        """Initialize all TASFT metrics on the given registry.

        Args:
            registry: Prometheus CollectorRegistry. Uses a new isolated registry
                      if None, avoiding default registry pollution.
        """
        self.registry = registry or CollectorRegistry()

        # --- Counters (Traffic, Errors) ---
        self.training_steps_total = Counter(
            "tasft_training_steps_total",
            "Total number of training steps completed",
            registry=self.registry,
        )
        self.gate_calibrations_total = Counter(
            "tasft_gate_calibrations_total",
            "Total number of gate calibration events",
            ["layer"],
            registry=self.registry,
        )
        self.oom_events_total = Counter(
            "tasft_oom_events_total",
            "Total number of OOM events detected",
            registry=self.registry,
        )
        self.errors_total = Counter(
            "tasft_errors_total",
            "Total errors by type",
            ["error_type"],
            registry=self.registry,
        )

        # --- Histograms (Latency) ---
        self.step_duration_seconds = Histogram(
            "tasft_step_duration_seconds",
            "Duration of a single training step in seconds",
            buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0),
            registry=self.registry,
        )
        self.gate_forward_ms = Histogram(
            "tasft_gate_forward_ms",
            "Gate forward pass latency in milliseconds",
            ["layer"],
            buckets=(0.1, 0.5, 1.0, 2.5, 5.0, 10.0, 25.0, 50.0),
            registry=self.registry,
        )
        self.sparse_kernel_ms = Histogram(
            "tasft_sparse_kernel_ms",
            "Sparse attention kernel latency in milliseconds",
            ["layer"],
            buckets=(0.1, 0.5, 1.0, 2.5, 5.0, 10.0, 25.0, 50.0, 100.0),
            registry=self.registry,
        )
        self.sparsity_ratio = Histogram(
            "tasft_sparsity_ratio",
            "Sparsity ratio per layer (fraction of blocks masked)",
            ["layer"],
            buckets=(0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
            registry=self.registry,
        )

        # --- Gauges (Saturation) ---
        self.gpu_memory_used_bytes = Gauge(
            "tasft_gpu_memory_used_bytes",
            "GPU memory usage in bytes",
            ["device"],
            registry=self.registry,
        )
        self.active_layers_count = Gauge(
            "tasft_active_layers_count",
            "Number of layers with active gate calibration",
            registry=self.registry,
        )
        self.current_lambda_gate = Gauge(
            "tasft_current_lambda_gate",
            "Current value of the gate loss weight λ",
            registry=self.registry,
        )

    def record_step(self, duration_seconds: float) -> None:
        """Record a completed training step.

        Preconditions: duration_seconds > 0.
        Postconditions: Increments step counter and observes duration histogram.
        Complexity: O(1).
        """
        self.training_steps_total.inc()
        self.step_duration_seconds.observe(duration_seconds)

    def record_gate_calibration(self, layer: int, forward_ms: float) -> None:
        """Record a gate calibration event for a specific layer.

        Preconditions: layer >= 0, forward_ms > 0.
        Postconditions: Increments calibration counter and observes latency.
        Complexity: O(1).
        """
        self.gate_calibrations_total.labels(layer=str(layer)).inc()
        self.gate_forward_ms.labels(layer=str(layer)).observe(forward_ms)

    def record_sparsity(self, layer: int, ratio: float) -> None:
        """Record the sparsity ratio for a layer.

        Preconditions: 0 <= ratio <= 1, layer >= 0.
        Postconditions: Observes sparsity ratio histogram.
        Complexity: O(1).
        """
        self.sparsity_ratio.labels(layer=str(layer)).observe(ratio)

    def record_error(self, error_type: str) -> None:
        """Record an error event by type.

        Preconditions: error_type is a non-empty string.
        Postconditions: Increments error counter for the given type.
        Complexity: O(1).
        """
        self.errors_total.labels(error_type=error_type).inc()

    def record_oom(self) -> None:
        """Record an OOM event.

        Postconditions: Increments OOM counter.
        Complexity: O(1).
        """
        self.oom_events_total.inc()

    def set_gpu_memory(self, device: str, bytes_used: int) -> None:
        """Set the current GPU memory usage for a device.

        Preconditions: device is a valid device identifier, bytes_used >= 0.
        Postconditions: Sets the gauge to the given value.
        Complexity: O(1).
        """
        self.gpu_memory_used_bytes.labels(device=device).set(bytes_used)

    def set_active_layers(self, count: int) -> None:
        """Set the number of layers with active gate calibration.

        Preconditions: count >= 0.
        Postconditions: Sets the gauge to the given value.
        Complexity: O(1).
        """
        self.active_layers_count.set(count)

    def set_lambda_gate(self, value: float) -> None:
        """Set the current gate loss weight λ.

        Preconditions: value >= 0.
        Postconditions: Sets the gauge to the given value.
        Complexity: O(1).
        """
        self.current_lambda_gate.set(value)

    def push(self, gateway_url: str, job: str = "tasft_training") -> None:
        """Push all metrics to a Prometheus Pushgateway.

        Used for non-server training jobs that cannot be scraped directly.

        Preconditions: gateway_url is a reachable Pushgateway endpoint.
        Postconditions: All current metric values pushed to the gateway.
        Complexity: O(m) where m = number of metric families.
        Side effects: HTTP POST to gateway_url.

        Args:
            gateway_url: URL of the Prometheus Pushgateway.
            job: Job name for grouping in the gateway.
        """
        push_to_gateway(gateway_url, job=job, registry=self.registry)


class MetricsContext:
    """Context manager that records the duration of an operation.

    Records start/end timestamps and observes the duration on a provided
    Histogram or calls a callback with the elapsed time.

    Preconditions: metrics is a valid TASFTMetrics instance.
    Postconditions: Duration is recorded when the context exits.
    Complexity: O(1).
    """

    def __init__(
        self,
        metrics: TASFTMetrics,
        histogram: Histogram | None = None,
        labels: dict[str, str] | None = None,
        *,
        on_complete: Any | None = None,
    ) -> None:
        """Initialize the metrics context.

        Args:
            metrics: The TASFTMetrics instance.
            histogram: Optional histogram to observe duration on.
            labels: Optional labels for the histogram.
            on_complete: Optional callback(duration_seconds: float) called on exit.
        """
        self._metrics = metrics
        self._histogram = histogram
        self._labels = labels or {}
        self._on_complete = on_complete
        self._start_ns: int = 0

    def __enter__(self) -> Self:
        self._start_ns = time.perf_counter_ns()
        return self

    def __exit__(self, *_: object) -> None:
        elapsed_s = (time.perf_counter_ns() - self._start_ns) / 1_000_000_000
        if self._histogram is not None:
            if self._labels:
                self._histogram.labels(**self._labels).observe(elapsed_s)
            else:
                self._histogram.observe(elapsed_s)
        if self._on_complete is not None:
            self._on_complete(elapsed_s)


@contextmanager
def track_step(metrics: TASFTMetrics) -> Generator[None, None, None]:
    """Context manager that tracks a full training step duration.

    Preconditions: metrics is initialized.
    Postconditions: Step counter incremented and duration observed.
    Complexity: O(1).

    Args:
        metrics: The TASFTMetrics instance.

    Yields:
        None — timing is handled automatically.
    """
    start_ns = time.perf_counter_ns()
    try:
        yield
    finally:
        elapsed_s = (time.perf_counter_ns() - start_ns) / 1_000_000_000
        metrics.record_step(elapsed_s)


__all__ = [
    "MetricsContext",
    "TASFTMetrics",
    "track_step",
]
