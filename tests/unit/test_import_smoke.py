"""Import smoke tests — verifies all modules and scripts can be imported without errors.

Every import crash in production is a P0. This test catches stale imports, wrong module
paths, and API drift between library internals and integration code (scripts, plugins).

Coverage target: 100% of public modules.
"""
from __future__ import annotations

import importlib

import pytest

# All library modules that must be importable
_LIBRARY_MODULES = [
    "tasft",
    "tasft.types",
    "tasft.exceptions",
    "tasft.modules.attn_gate",
    "tasft.modules.tasft_attention",
    "tasft.training.objectives",
    "tasft.training.layer_rotation",
    "tasft.training.trainer",
    "tasft.kernels.block_sparse_fa",
    "tasft.kernels.kernel_config",
    "tasft.inference.tasft_model",
    "tasft.inference.vllm_patch",
    "tasft.eval.task_eval",
    "tasft.eval.gate_quality",
    "tasft.eval.throughput_bench",
    "tasft.bundle.bundle_schema",
    "tasft.bundle.export",
    "tasft.observability.logging",
    "tasft.observability.metrics",
    "tasft.observability.tracing",
    "tasft.observability.alerts",
]

# Script modules that must parse without import errors
_SCRIPT_MODULES = [
    "scripts.train",
    "scripts.eval",
    "scripts.export_bundle",
]

# Plugin module
_PLUGIN_MODULES = [
    "axolotl_plugin.plugin",
]


class TestLibraryImports:
    """All library modules must import cleanly."""

    @pytest.mark.parametrize("module_name", _LIBRARY_MODULES)
    def test_library_module_imports(self, module_name: str) -> None:
        """Verify each library module can be imported without error."""
        mod = importlib.import_module(module_name)
        assert mod is not None


class TestScriptImports:
    """Script modules must parse without syntax errors.

    Scripts have heavy dependencies (datasets, peft, transformers) that may
    not be installed in the dev/test environment. We verify the module source
    compiles and that deferred import paths reference real tasft modules.
    """

    @pytest.mark.parametrize("module_name", _SCRIPT_MODULES)
    def test_script_source_compiles(self, module_name: str) -> None:
        """Verify each script's source compiles (catches syntax errors)."""
        module_path = module_name.replace(".", "/") + ".py"
        with open(module_path) as f:
            source = f.read()
        compile(source, module_path, "exec")

    def test_train_script_references_correct_bundle_module(self) -> None:
        """Verify train.py imports from tasft.bundle.export (not .exporter)."""
        with open("scripts/train.py") as f:
            source = f.read()
        assert "tasft.bundle.export" in source
        assert "tasft.bundle.exporter" not in source

    def test_eval_script_references_correct_modules(self) -> None:
        """Verify eval.py imports from correct eval submodules."""
        with open("scripts/eval.py") as f:
            source = f.read()
        assert "tasft.eval.task_eval" in source
        assert "tasft.eval.gate_quality" in source
        assert "tasft.eval.throughput_bench" in source
        assert "tasft.eval.task_evaluator" not in source
        assert "tasft.eval.gate_evaluator" not in source
        assert "tasft.eval.throughput_evaluator" not in source

    def test_export_script_references_correct_modules(self) -> None:
        """Verify export_bundle.py imports from tasft.bundle.export."""
        with open("scripts/export_bundle.py") as f:
            source = f.read()
        assert "tasft.bundle.export" in source
        assert "tasft.bundle.exporter" not in source


class TestPluginImports:
    """Plugin module must import cleanly."""

    @pytest.mark.parametrize("module_name", _PLUGIN_MODULES)
    def test_plugin_module_imports(self, module_name: str) -> None:
        """Verify plugin module can be imported without error."""
        mod = importlib.import_module(module_name)
        assert mod is not None

    def test_plugin_references_correct_bundle_module(self) -> None:
        """Verify plugin.py imports from tasft.bundle.export (not .exporter)."""
        with open("axolotl_plugin/plugin.py") as f:
            source = f.read()
        assert "tasft.bundle.export" in source
        assert "tasft.bundle.exporter" not in source


class TestPublicAPISurface:
    """Verify key classes and functions are accessible from their public modules."""

    def test_attn_gate_exports(self) -> None:
        from tasft.modules.attn_gate import AttnGate, GateOutput

        assert AttnGate is not None
        assert GateOutput is not None

    def test_tasft_attention_exports(self) -> None:
        from tasft.modules.tasft_attention import (
            GateConfig,
            TASFTAttention,
            patch_model_attention,
        )

        assert TASFTAttention is not None
        assert GateConfig is not None
        assert patch_model_attention is not None

    def test_objectives_exports(self) -> None:
        from tasft.training.objectives import ObjectiveLossOutput, TASFTObjective

        assert TASFTObjective is not None
        assert ObjectiveLossOutput is not None

    def test_trainer_exports(self) -> None:
        from tasft.training.trainer import TASFTTrainer, TASFTTrainingArguments

        assert TASFTTrainer is not None
        assert TASFTTrainingArguments is not None

    def test_bundle_exports(self) -> None:
        from tasft.bundle.export import BundleExporter, ExportConfig, ValidationResult

        assert BundleExporter is not None
        assert ExportConfig is not None
        assert ValidationResult is not None

    def test_bundle_schema_exports(self) -> None:
        from tasft.bundle.bundle_schema import (
            BundleManifest,
            BundleMetadata,
            EvalSummary,
            KernelConfig,
        )

        assert BundleManifest is not None
        assert BundleMetadata is not None
        assert EvalSummary is not None
        assert KernelConfig is not None

    def test_eval_exports(self) -> None:
        from tasft.eval.gate_quality import GateQualityEvaluator
        from tasft.eval.task_eval import TaskEvaluator
        from tasft.eval.throughput_bench import ThroughputBenchmark

        assert TaskEvaluator is not None
        assert GateQualityEvaluator is not None
        assert ThroughputBenchmark is not None

    def test_observability_exports(self) -> None:
        from tasft.observability.logging import configure_logging, get_logger
        from tasft.observability.metrics import TASFTMetrics

        assert get_logger is not None
        assert configure_logging is not None
        assert TASFTMetrics is not None

    def test_plugin_factory(self) -> None:
        from axolotl_plugin.plugin import TASFTPlugin, get_plugin

        plugin = get_plugin()
        assert isinstance(plugin, TASFTPlugin)
