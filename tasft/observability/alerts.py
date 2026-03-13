"""Prometheus alerting rules as code for TASFT.

Generates Prometheus-compatible alerting rule YAML from typed Python
definitions. Ensures alert configuration stays version-controlled and
testable alongside the code that emits the metrics.

Alert rules satisfy Λ₁₁ field equations with burn-rate and saturation thresholds.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class AlertRule:
    """A single Prometheus alerting rule.

    Preconditions: alert name is non-empty, expr is valid PromQL.
    Invariants: severity is one of "critical", "warning", "info".
    """

    alert: str
    expr: str
    for_duration: str
    severity: str
    summary: str
    description: str
    labels: dict[str, str] = field(default_factory=dict)
    annotations: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to Prometheus rule YAML structure.

        Postconditions: Returns a dict suitable for YAML serialization.
        Complexity: O(1).
        """
        all_labels = {"severity": self.severity, **self.labels}
        all_annotations = {
            "summary": self.summary,
            "description": self.description,
            **self.annotations,
        }
        return {
            "alert": self.alert,
            "expr": self.expr,
            "for": self.for_duration,
            "labels": all_labels,
            "annotations": all_annotations,
        }


# Pre-defined TASFT alerting rules
TASFT_ALERT_RULES: list[AlertRule] = [
    AlertRule(
        alert="TASFTSparsityBelowTarget",
        expr=(
            "avg_over_time(tasft_sparsity_ratio_sum[10m]) "
            "/ avg_over_time(tasft_sparsity_ratio_count[10m]) < 0.5"
        ),
        for_duration="10m",
        severity="warning",
        summary="TASFT mean sparsity below 50% target",
        description=(
            "The average sparsity ratio across all layers has been below 0.5 "
            "for more than 10 minutes. Gate training may be underperforming. "
            "Check lambda_gate weight and learning rate."
        ),
        labels={"component": "training"},
    ),
    AlertRule(
        alert="TASFTNaNDetected",
        expr='increase(tasft_errors_total{error_type="nan_detected"}[1m]) > 0',
        for_duration="0m",
        severity="critical",
        summary="NaN detected in TASFT training",
        description=(
            "NaN or Inf values detected in model tensors. Training is likely "
            "diverging. Immediate investigation required — check learning rate, "
            "gradient scaling, and input data."
        ),
        labels={"component": "training"},
    ),
    AlertRule(
        alert="TASFTCheckpointFailed",
        expr='increase(tasft_errors_total{error_type="checkpoint_failed"}[5m]) > 0',
        for_duration="0m",
        severity="critical",
        summary="TASFT checkpoint save failed",
        description=(
            "A checkpoint save operation failed. This risks losing training "
            "progress. Check disk space, permissions, and storage backend health."
        ),
        labels={"component": "training"},
    ),
    AlertRule(
        alert="TASFTOOMRisk",
        expr="tasft_gpu_memory_used_bytes / 1073741824 > 0.9 * 80",
        for_duration="2m",
        severity="warning",
        summary="GPU memory usage exceeds 90% threshold",
        description=(
            "GPU memory usage has exceeded 90% of capacity for more than 2 minutes. "
            "OOM is imminent. Consider reducing batch size, enabling gradient "
            "checkpointing, or reducing the number of active gate layers."
        ),
        labels={"component": "infrastructure"},
    ),
    AlertRule(
        alert="TASFTHighStepLatency",
        expr="histogram_quantile(0.99, rate(tasft_step_duration_seconds_bucket[5m])) > 10",
        for_duration="5m",
        severity="warning",
        summary="TASFT training step p99 latency exceeds 10s",
        description=(
            "The 99th percentile training step duration has exceeded 10 seconds "
            "for more than 5 minutes. Check for GPU throttling, data loading "
            "bottlenecks, or increased sequence lengths."
        ),
        labels={"component": "training"},
    ),
    AlertRule(
        alert="TASFTHighErrorRate",
        expr="rate(tasft_errors_total[5m]) > 0.1",
        for_duration="5m",
        severity="warning",
        summary="TASFT error rate exceeds threshold",
        description=(
            "The error rate has exceeded 0.1 errors/second for more than 5 minutes. "
            "Investigate error types via tasft_errors_total labels."
        ),
        labels={"component": "training"},
    ),
]


def generate_alert_rules(output_path: str | Path) -> Path:
    """Generate Prometheus alerting rules YAML file.

    Writes a valid Prometheus alerting rules file from the TASFT_ALERT_RULES
    definitions. Uses manual YAML construction to avoid PyYAML dependency
    (which is not in our dependency tree).

    Preconditions: output_path parent directory exists and is writable.
    Postconditions: A valid Prometheus alerting rules YAML file is written.
    Complexity: O(r) where r = number of alert rules.
    Side effects: Writes a file to disk.

    Args:
        output_path: Path where the alerting rules YAML will be written.

    Returns:
        The resolved Path to the written file.
    """
    path = Path(output_path)
    rules_dicts = [rule.to_dict() for rule in TASFT_ALERT_RULES]

    # Build YAML manually to avoid PyYAML dependency
    lines: list[str] = [
        "# Auto-generated TASFT Prometheus alerting rules",
        "# Do not edit manually — regenerate via tasft.observability.alerts",
        "groups:",
        "  - name: tasft_alerts",
        "    rules:",
    ]

    for rule_dict in rules_dicts:
        lines.append(f"      - alert: {rule_dict['alert']}")
        lines.append(f"        expr: {json.dumps(rule_dict['expr'])}")
        lines.append(f"        for: {rule_dict['for']}")
        lines.append("        labels:")
        for lk, lv in rule_dict["labels"].items():
            lines.append(f"          {lk}: {json.dumps(lv)}")
        lines.append("        annotations:")
        for ak, av in rule_dict["annotations"].items():
            lines.append(f"          {ak}: {json.dumps(av)}")

    content = "\n".join(lines) + "\n"
    path.write_text(content, encoding="utf-8")
    return path.resolve()


__all__ = [
    "TASFT_ALERT_RULES",
    "AlertRule",
    "generate_alert_rules",
]
