"""OpenTelemetry distributed tracing for TASFT.

Provides tracer initialization and span creation for training steps,
gate calibration, and inference requests. Supports OTLP export for
production and noop tracing when no endpoint is configured.

All spans include TASFT-specific attributes for cross-system correlation.
"""
from __future__ import annotations

from contextlib import contextmanager
from typing import TYPE_CHECKING, Any

from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import (
    BatchSpanProcessor,
    SpanExporter,
)
from opentelemetry.trace import Span, StatusCode, Tracer

if TYPE_CHECKING:
    from collections.abc import Generator

_TRACER_NAME = "tasft"
_tracer: Tracer | None = None


def init_tracing(
    service_name: str = "tasft",
    *,
    otlp_endpoint: str | None = None,
    resource_attributes: dict[str, str] | None = None,
) -> Tracer:
    """Initialize OpenTelemetry tracing with optional OTLP export.

    If otlp_endpoint is None, uses a noop tracer (zero overhead in production
    when tracing is not needed). Otherwise, configures OTLP gRPC export with
    batch span processing.

    Preconditions: Called once at process startup.
    Postconditions: Global tracer provider configured; returns a Tracer.
    Complexity: O(1).
    Side effects: Sets the global OpenTelemetry tracer provider.

    Args:
        service_name: Service name for the resource.
        otlp_endpoint: OTLP gRPC endpoint URL (e.g., "http://localhost:4317").
                       If None, tracing is a noop.
        resource_attributes: Additional resource attributes to attach.

    Returns:
        Configured Tracer instance.
    """
    global _tracer  # noqa: PLW0603

    attrs: dict[str, str] = {"service.name": service_name}
    if resource_attributes:
        attrs.update(resource_attributes)

    resource = Resource.create(attrs)
    provider = TracerProvider(resource=resource)

    if otlp_endpoint is not None:
        # Deferred import to avoid pulling in grpc when not needed
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
            OTLPSpanExporter,
        )

        exporter: SpanExporter = OTLPSpanExporter(endpoint=otlp_endpoint)
        provider.add_span_processor(BatchSpanProcessor(exporter))

    trace.set_tracer_provider(provider)
    _tracer = trace.get_tracer(_TRACER_NAME)
    return _tracer


def get_tracer() -> Tracer:
    """Return the initialized tracer, or a noop tracer if not initialized.

    Postconditions: Always returns a valid Tracer (never None).
    Complexity: O(1).
    """
    if _tracer is not None:
        return _tracer
    return trace.get_tracer(_TRACER_NAME)


@contextmanager
def trace_training_step(
    step: int,
    active_layers: list[int],
    **attributes: Any,
) -> Generator[Span, None, None]:
    """Create a span covering a full training step.

    Preconditions: step >= 0, active_layers is a list of layer indices.
    Postconditions: Span is ended when the context exits. On exception,
                    span status is set to ERROR with the exception recorded.
    Complexity: O(1).

    Args:
        step: Current training step number.
        active_layers: List of layer indices with active gate calibration.
        **attributes: Additional span attributes.

    Yields:
        The active Span for adding events/attributes within the step.
    """
    tracer = get_tracer()
    with tracer.start_as_current_span(
        "training.step",
        attributes={
            "tasft.step": step,
            "tasft.active_layers": str(active_layers),
            "tasft.num_active_layers": len(active_layers),
            **{f"tasft.{k}": str(v) for k, v in attributes.items()},
        },
    ) as span:
        try:
            yield span
        except Exception as exc:
            span.set_status(StatusCode.ERROR, str(exc))
            span.record_exception(exc)
            raise


@contextmanager
def trace_gate_calibration(
    layer_idx: int,
    **attributes: Any,
) -> Generator[Span, None, None]:
    """Create a child span for gate calibration on a specific layer.

    Preconditions: layer_idx >= 0.
    Postconditions: Span is ended when the context exits.
    Complexity: O(1).

    Args:
        layer_idx: The layer index being calibrated.
        **attributes: Additional span attributes.

    Yields:
        The active Span.
    """
    tracer = get_tracer()
    with tracer.start_as_current_span(
        "training.gate_calibration",
        attributes={
            "tasft.layer_idx": layer_idx,
            **{f"tasft.{k}": str(v) for k, v in attributes.items()},
        },
    ) as span:
        try:
            yield span
        except Exception as exc:
            span.set_status(StatusCode.ERROR, str(exc))
            span.record_exception(exc)
            raise


@contextmanager
def trace_inference_request(
    request_id: str,
    seq_len: int,
    **attributes: Any,
) -> Generator[Span, None, None]:
    """Create a span for an inference request.

    Preconditions: request_id is non-empty, seq_len > 0.
    Postconditions: Span is ended when the context exits.
    Complexity: O(1).

    Args:
        request_id: Unique identifier for the inference request.
        seq_len: Sequence length of the input.
        **attributes: Additional span attributes.

    Yields:
        The active Span.
    """
    tracer = get_tracer()
    with tracer.start_as_current_span(
        "inference.request",
        attributes={
            "tasft.request_id": request_id,
            "tasft.seq_len": seq_len,
            **{f"tasft.{k}": str(v) for k, v in attributes.items()},
        },
    ) as span:
        try:
            yield span
        except Exception as exc:
            span.set_status(StatusCode.ERROR, str(exc))
            span.record_exception(exc)
            raise


__all__ = [
    "get_tracer",
    "init_tracing",
    "trace_gate_calibration",
    "trace_inference_request",
    "trace_training_step",
]
