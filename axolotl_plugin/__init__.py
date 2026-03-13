"""TASFT Axolotl Plugin: integration with the Axolotl training framework.

Provides hooks for injecting AttnGate modules and the dual-objective
training loop into Axolotl's training pipeline.

Usage in Axolotl config:
    plugins:
      - tasft

    tasft:
      gate:
        block_size: 64
        num_layers: 32
      objective:
        lambda_gate: 0.1
        tau_target: 0.7
"""
from axolotl_plugin.plugin import TASFTPlugin, get_plugin

__all__ = ["TASFTPlugin", "get_plugin"]
