"""
TASFT eval: evaluation harness for task quality, gate quality, and inference throughput.

This package contains:
- task_eval: Domain task evaluation (MedQA, HumanEval) with statistical comparison
- throughput_bench: Inference throughput benchmarking across batch/seq configurations
- gate_quality: Core ablation study — co-trained vs post-hoc gate KL divergence
- long_context_bench: Attention scaling benchmarks across sequence lengths
"""
from tasft.eval.gate_quality import (
    AblationResult,
    GateEvalError,
    GateQualityEvaluator,
    GateQualityResult,
)
from tasft.eval.long_context_bench import (
    LongContextBenchConfig,
    LongContextBenchError,
    LongContextResult,
    benchmark_attention_scaling,
    generate_scaling_report,
)
from tasft.eval.task_eval import (
    ComparisonResult,
    EvalError,
    TaskEvalResult,
    TaskEvaluator,
)
from tasft.eval.throughput_bench import (
    BenchmarkError,
    BenchmarkPoint,
    SpeedupMatrix,
    ThroughputBenchmark,
    ThroughputMatrix,
)

__all__ = [
    "AblationResult",
    "BenchmarkError",
    "BenchmarkPoint",
    "ComparisonResult",
    "EvalError",
    "GateEvalError",
    "GateQualityEvaluator",
    "GateQualityResult",
    "LongContextBenchConfig",
    "LongContextBenchError",
    "LongContextResult",
    "SpeedupMatrix",
    "TaskEvalResult",
    "TaskEvaluator",
    "ThroughputBenchmark",
    "ThroughputMatrix",
    "benchmark_attention_scaling",
    "generate_scaling_report",
]
