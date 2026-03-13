"""
TASFT: Task-Aware Sparse Fine-Tuning

A co-training framework that simultaneously fine-tunes domain tasks and trains
sparse attention gates in a single training run, enabling 2-5x decode throughput
over standard fine-tuned models at inference time.

Architecture: SeerAttention-based AttnGate modules co-trained with LoRA adapters
via a dual objective (L_task + λ·L_gate) with layer-rotating gate calibration.

Reference: arxiv:2410.13276 (SeerAttention), arxiv:2409.15820 (attention head shifts)
"""
from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("tasft")
except PackageNotFoundError:
    __version__ = "0.1.0.dev0"
__all__ = ["__version__"]
