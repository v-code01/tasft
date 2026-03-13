# Changelog

All notable changes to TASFT are documented here.
Format: [Keep a Changelog](https://keepachangelog.com/en/1.0.0/)
Versioning: [Semantic Versioning](https://semver.org/spec/v2.0.0.html)

## [0.1.0] - 2025-03-13

### Added
- `AttnGate` module: block-level attention importance predictor based on SeerAttention
- `TASFTAttention`: patched attention layer with co-training hooks for LoRA + gate
- `TASFTObjective`: dual loss (L_task + lambda * L_gate) with sparsity regularization
- `LayerRotationScheduler`: memory-efficient gate calibration cycling (ROUND_ROBIN, RANDOM, PRIORITY_WEIGHTED)
- `TASFTTrainer`: HuggingFace Trainer subclass with co-training loop
- `BlockSparseFlashAttention`: Triton-based block-sparse kernel wrapper
- `TASFTInferenceModel`: deployment runtime with gate-driven sparse attention
- vLLM integration patch (`TASFTvLLMAttentionBackend`)
- Evaluation harness: MedQA, LegalBench, HumanEval, FinBench
- Gate quality ablation: co-trained vs post-hoc gate comparison
- Bundle export system with SHA256 checksums and atomic writes
- Training configs for Llama-3-8B (MedQA) and Qwen-2.5-7B (HumanEval)
- Axolotl plugin integration
- Observability: structlog, Prometheus metrics, OpenTelemetry tracing
- Full CI/CD pipeline with unit, integration, and benchmark tests
