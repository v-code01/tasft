"""Structured logging for TASFT using structlog.

Provides a consistent, structured logging interface with automatic context binding
for training steps, layer indices, and request IDs. Outputs JSON in production
and colored console output during local development (auto-detected via TTY).

All logs include: timestamp_utc, level, module, and operation-specific fields.
"""
from __future__ import annotations

import os
import subprocess
import sys
import time
from contextlib import contextmanager
from importlib.metadata import version
from typing import TYPE_CHECKING, Any

import structlog

if TYPE_CHECKING:
    from collections.abc import Generator

    from structlog.types import BindableLogger


def _get_git_hash() -> str:
    """Return the short git hash of HEAD, or 'unknown' if not in a git repo.

    Preconditions: None (gracefully handles missing git).
    Postconditions: Returns a string of length 7 (hash) or 'unknown'.
    Complexity: O(1) — single subprocess call, cached at module load.
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return "unknown"


_GIT_HASH: str = _get_git_hash()
_VERSION: str = version("tasft")


def _is_tty() -> bool:
    """Check if stdout is a TTY for renderer selection."""
    return hasattr(sys.stdout, "isatty") and sys.stdout.isatty()


def _configure_structlog(*, force_json: bool = False) -> None:
    """Configure structlog processors and renderer.

    Preconditions: Called once at process startup.
    Postconditions: structlog is configured with timestamper, level, and renderer.
    Complexity: O(1).
    Side effects: Mutates structlog global configuration.
    """
    use_json = force_json or not _is_tty() or os.environ.get("TASFT_LOG_JSON", "") == "1"

    processors: list[structlog.types.Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso", utc=True, key="timestamp_utc"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]

    if use_json:
        processors.append(structlog.processors.JSONRenderer())
    else:
        processors.append(structlog.dev.ConsoleRenderer())

    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(0),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )


# Configure on import — idempotent via structlog internals
_configure_structlog()


def get_logger(name: str) -> BindableLogger:
    """Create a structured logger with standard TASFT context fields bound.

    Preconditions: name is a non-empty string identifying the module.
    Postconditions: Returns a BoundLogger with module, version, git_hash bound.
    Complexity: O(1).

    Args:
        name: Module or component name (e.g., "tasft.training.trainer").

    Returns:
        A structlog BoundLogger with standard fields pre-bound.
    """
    return structlog.get_logger(
        module=name,
        version=_VERSION,
        git_hash=_GIT_HASH,
    )


@contextmanager
def bind_context(
    *,
    request_id: str | None = None,
    step: int | None = None,
    layer_idx: int | None = None,
    **extra: Any,
) -> Generator[None, None, None]:
    """Context manager that binds fields to all loggers within the scope.

    Uses structlog's contextvars integration so all log calls within the
    context (including in called functions) include the bound fields.

    Preconditions: At least one field should be provided.
    Postconditions: Fields are unbound when the context exits.
    Complexity: O(k) where k = number of bound fields.

    Args:
        request_id: Unique identifier for the current request/operation.
        step: Training step number.
        layer_idx: Layer index for per-layer operations.
        **extra: Additional fields to bind.

    Yields:
        None — fields are bound via contextvars.
    """
    ctx: dict[str, Any] = {}
    if request_id is not None:
        ctx["request_id"] = request_id
    if step is not None:
        ctx["step"] = step
    if layer_idx is not None:
        ctx["layer_idx"] = layer_idx
    ctx.update(extra)

    structlog.contextvars.bind_contextvars(**ctx)
    try:
        yield
    finally:
        structlog.contextvars.unbind_contextvars(*ctx.keys())


@contextmanager
def timed_operation(
    logger: BindableLogger,
    operation: str,
    *,
    level: str = "info",
    **extra: Any,
) -> Generator[None, None, None]:
    """Context manager that logs operation start and completion with duration_ms.

    Preconditions: logger is a valid structlog BoundLogger.
    Postconditions: Logs operation completion with duration_ms field.
    Complexity: O(1).

    Args:
        logger: The logger to use for start/end messages.
        operation: Name of the operation being timed.
        level: Log level for the completion message (default: info).
        **extra: Additional fields to include in log messages.

    Yields:
        None — timing is handled automatically.
    """
    start_ns = time.perf_counter_ns()
    log_fn = getattr(logger, level)
    log_fn(f"[{operation}] started", operation=operation, **extra)
    try:
        yield
    finally:
        elapsed_ms = (time.perf_counter_ns() - start_ns) / 1_000_000
        log_fn(
            f"[{operation}] completed",
            operation=operation,
            duration_ms=round(elapsed_ms, 3),
            **extra,
        )


def configure_logging(
    *,
    level: str = "INFO",
    force_json: bool = False,
) -> None:
    """Configure TASFT structured logging. Call once at program start.

    Reconfigures structlog with the specified log level and output format.
    Safe to call multiple times — each call replaces the previous configuration.

    Preconditions: level is a valid Python log level name.
    Postconditions: structlog configured with specified level and format.
    Complexity: O(1).
    Side effects: Mutates structlog global configuration.

    Args:
        level: Minimum log level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        force_json: Force JSON output regardless of TTY detection.
    """
    import logging as _logging

    numeric_level = _logging.getLevelName(level.upper())
    use_json = force_json or not _is_tty() or os.environ.get("TASFT_LOG_JSON", "") == "1"

    processors: list[structlog.types.Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso", utc=True, key="timestamp_utc"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]

    if use_json:
        processors.append(structlog.processors.JSONRenderer())
    else:
        processors.append(structlog.dev.ConsoleRenderer())

    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(numeric_level),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=False,  # Allow reconfiguration
    )


__all__ = [
    "bind_context",
    "configure_logging",
    "get_logger",
    "timed_operation",
]
