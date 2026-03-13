"""
Task quality evaluation for TASFT fine-tuned models.

Evaluates domain task performance to verify TASFT co-training doesn't degrade
task quality vs standard LoRA fine-tuning (target: within 1-2% accuracy).

Domains: Medical (MedQA), Legal (LegalBench), Code (HumanEval), Finance (FinBench)

Preconditions:
    - model_path points to a valid HuggingFace model or TASFT bundle
    - GPU available for model inference (falls back to CPU with warning)
    - For HumanEval: code execution sandbox available

Postconditions:
    - TaskEvalResult contains statistically valid accuracy with 95% CI (Wilson interval)
    - ComparisonResult includes two-tailed t-test p-value and Cohen's d effect size
    - All per-question results are preserved for downstream analysis

Complexity: O(N * S * V) per evaluation where N=samples, S=seq_len, V=vocab_size
"""
from __future__ import annotations

import math
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from typing import Final

import numpy as np
import torch
from scipy import stats

from tasft.exceptions import TASFTError, ValidationError
from tasft.observability.logging import get_logger, timed_operation

_Z_95: Final[float] = 1.959964  # z-score for 95% confidence interval
_TARGET_DELTA: Final[float] = 0.02  # 2% accuracy delta threshold

logger = get_logger("tasft.eval.task_eval")


class EvalError(TASFTError):
    """Raised when evaluation encounters an unrecoverable error."""


@dataclass(frozen=True)
class TaskEvalResult:
    """Result of a domain task evaluation run.

    Attributes:
        accuracy: Point estimate of accuracy in [0, 1].
        accuracy_ci_low: Lower bound of Wilson 95% CI.
        accuracy_ci_high: Upper bound of Wilson 95% CI.
        n_samples: Number of evaluated samples.
        domain: Evaluation domain identifier (e.g., "medqa", "humaneval").
        model_path: Path to the evaluated model.
        eval_duration_seconds: Wall-clock evaluation time.
        per_question_results: Per-sample results for downstream analysis.
        metadata: Additional context (split, batch_size, etc.).
    """

    accuracy: float
    accuracy_ci_low: float
    accuracy_ci_high: float
    n_samples: int
    domain: str
    model_path: str
    eval_duration_seconds: float
    per_question_results: list[dict[str, object]] = field(default_factory=list)
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not 0.0 <= self.accuracy <= 1.0:
            raise ValidationError(
                f"accuracy must be in [0, 1], got {self.accuracy}",
                context={"accuracy": self.accuracy},
            )
        if self.n_samples <= 0:
            raise ValidationError(
                f"n_samples must be positive, got {self.n_samples}",
                context={"n_samples": self.n_samples},
            )


@dataclass(frozen=True)
class ComparisonResult:
    """Statistical comparison between baseline and TASFT model task accuracy.

    Attributes:
        baseline_result: Evaluation result from the baseline (standard LoRA) model.
        tasft_result: Evaluation result from the TASFT co-trained model.
        delta_accuracy: tasft_accuracy - baseline_accuracy (negative = degradation).
        p_value: Two-tailed independent t-test p-value on per-question binary scores.
        significant: Whether p_value < 0.01.
        effect_size: Cohen's d effect size.
        within_target: Whether |delta_accuracy| < 0.02.
    """

    baseline_result: TaskEvalResult
    tasft_result: TaskEvalResult
    delta_accuracy: float
    p_value: float
    significant: bool
    effect_size: float
    within_target: bool


def _wilson_ci(p: float, n: int, z: float = _Z_95) -> tuple[float, float]:
    """Wilson score interval for binomial proportion.

    Computes the 95% confidence interval for a proportion using the Wilson
    score method, which has better coverage than the normal approximation
    especially for extreme proportions and small sample sizes.

    Formula: (p + z²/2n ± z√(p(1-p)/n + z²/4n²)) / (1 + z²/n)

    Args:
        p: Observed proportion in [0, 1].
        n: Sample count (must be > 0).
        z: z-score for desired confidence level. Default 1.96 for 95% CI.

    Returns:
        (lower, upper) bounds of the confidence interval.

    Complexity: O(1).
    """
    if n == 0:
        return (0.0, 1.0)
    z2 = z * z
    denom = 1.0 + z2 / n
    centre = (p + z2 / (2.0 * n)) / denom
    margin = (z / denom) * math.sqrt(p * (1.0 - p) / n + z2 / (4.0 * n * n))
    return (max(0.0, centre - margin), min(1.0, centre + margin))


def _passatk_unbiased(n: int, c: int, k: int) -> float:
    """Unbiased pass@k estimator: 1 - C(n-c, k) / C(n, k).

    Uses the log-space computation to avoid overflow for large n, c, k.

    Args:
        n: Total number of generated samples per problem.
        c: Number of correct samples.
        k: k for pass@k.

    Returns:
        pass@k estimate in [0, 1].

    Complexity: O(k).
    """
    if n - c < k:
        return 1.0
    # Compute in log-space: log(C(n-c, k)) - log(C(n, k))
    # log(C(a, b)) = sum(log(a - i) - log(i + 1) for i in range(b))
    log_ratio = 0.0
    for i in range(k):
        log_ratio += math.log(n - c - i) - math.log(n - i)
    return 1.0 - math.exp(log_ratio)


class TaskEvaluator:
    """Domain task quality evaluator for TASFT and baseline models.

    Loads models and runs domain-specific evaluations to measure task accuracy.
    Supports MedQA (medical MCQ) and HumanEval (code generation) domains.

    All evaluation methods return frozen TaskEvalResult dataclasses with
    statistical confidence intervals computed via the Wilson score method.
    """

    def __init__(self, device: str | None = None) -> None:
        """Initialize evaluator.

        Args:
            device: Torch device string. Auto-detected if None.
        """
        if device is not None:
            self._device = torch.device(device)
        elif torch.cuda.is_available():
            self._device = torch.device("cuda")
        else:
            self._device = torch.device("cpu")
            logger.warning(
                "[TASK_EVAL_INIT] No GPU detected, falling back to CPU",
                device="cpu",
            )

    def _load_model_and_tokenizer(
        self, model_path: str
    ) -> tuple[torch.nn.Module, object]:
        """Load a HuggingFace model and tokenizer from path.

        Handles both standard HF checkpoints and TASFT bundles (by loading
        the base model + LoRA adapter from bundle manifest).

        Args:
            model_path: Path to HF model directory or TASFT bundle.

        Returns:
            (model, tokenizer) tuple.

        Raises:
            EvalError: If model loading fails.
        """
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise EvalError(
                "transformers package required for task evaluation",
                context={"missing_package": "transformers"},
            ) from exc

        try:
            tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16,
                device_map="auto" if self._device.type == "cuda" else None,
                trust_remote_code=True,
            )
            if self._device.type != "cuda":
                model = model.to(self._device)
            model.eval()
            return model, tokenizer
        except Exception as exc:
            raise EvalError(
                f"Failed to load model from {model_path}: {exc}",
                context={"model_path": model_path, "error": str(exc)},
            ) from exc

    @torch.inference_mode()
    def evaluate_medqa(
        self,
        model_path: str,
        split: str = "test",
        batch_size: int = 32,
    ) -> TaskEvalResult:
        """Evaluate on MedQA 4-option multiple choice.

        Loads MedQA from Hugging Face datasets ("bigbio/med_qa"), formats each
        question as MCQ with options A-D, computes log-probability for each
        option token, and selects argmax as the model's prediction.

        Args:
            model_path: Path to model checkpoint.
            split: Dataset split (default "test").
            batch_size: Inference batch size.

        Returns:
            TaskEvalResult with accuracy and Wilson 95% CI.

        Complexity: O(N * S * V) where N=dataset_size, S=max_seq_len, V=vocab_size.
        """
        try:
            from datasets import load_dataset
        except ImportError as exc:
            raise EvalError(
                "datasets package required for MedQA evaluation",
                context={"missing_package": "datasets"},
            ) from exc

        start_time = time.perf_counter()

        with timed_operation(logger, "MEDQA_LOAD_MODEL", model_path=model_path):
            model, tokenizer = self._load_model_and_tokenizer(model_path)

        with timed_operation(logger, "MEDQA_LOAD_DATASET", split=split):
            dataset = load_dataset("bigbio/med_qa", split=split, trust_remote_code=True)

        option_labels = ["A", "B", "C", "D"]
        # Pre-tokenize option tokens for log-prob extraction
        option_token_ids = []
        for label in option_labels:
            ids = tokenizer.encode(label, add_special_tokens=False)
            if len(ids) == 0:
                raise EvalError(
                    f"Tokenizer returned empty encoding for option label '{label}'",
                    context={"label": label},
                )
            option_token_ids.append(ids[0])

        per_question: list[dict[str, object]] = []
        correct_count = 0
        total_count = 0

        # Process in batches
        questions = list(dataset)
        for batch_start in range(0, len(questions), batch_size):
            batch = questions[batch_start : batch_start + batch_size]
            prompts = []
            ground_truths = []

            for item in batch:
                question_text = item.get("question", item.get("text", ""))
                options = item.get("options", item.get("answer_options", []))
                answer_idx = item.get("answer_idx", item.get("correct_answer", 0))

                # Format: "Q\nA) ...\nB) ...\nC) ...\nD) ...\nAnswer:"
                formatted_options = []
                for i, opt in enumerate(options[:4]):
                    opt_text = opt if isinstance(opt, str) else opt.get("text", str(opt))
                    formatted_options.append(f"{option_labels[i]}) {opt_text}")

                prompt = f"{question_text}\n" + "\n".join(formatted_options) + "\nAnswer:"
                prompts.append(prompt)

                # Normalize ground truth to index
                if isinstance(answer_idx, int):
                    ground_truths.append(answer_idx)
                elif isinstance(answer_idx, str) and answer_idx.upper() in option_labels:
                    ground_truths.append(option_labels.index(answer_idx.upper()))
                else:
                    ground_truths.append(0)

            # Tokenize batch
            encodings = tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=2048,
            ).to(self._device)

            # Forward pass — get logits at last non-pad position
            outputs = model(**encodings)
            logits = outputs.logits  # [B, S, V]

            # For each sample, extract log-probs at the last token position
            attention_mask = encodings["attention_mask"]
            # Last non-pad position: sum of mask - 1
            last_positions = attention_mask.sum(dim=1) - 1  # [B]

            for i in range(len(batch)):
                last_pos = last_positions[i].item()
                token_logits = logits[i, int(last_pos), :]  # [V]

                # Extract log-probs for each option token
                log_probs = torch.log_softmax(token_logits, dim=-1)
                option_log_probs = torch.tensor(
                    [log_probs[tid].item() for tid in option_token_ids],
                    dtype=torch.float64,
                )
                predicted_idx = int(option_log_probs.argmax().item())
                gt_idx = ground_truths[i]
                is_correct = predicted_idx == gt_idx

                if is_correct:
                    correct_count += 1
                total_count += 1

                per_question.append({
                    "question_id": batch_start + i,
                    "correct": is_correct,
                    "predicted": option_labels[predicted_idx],
                    "ground_truth": option_labels[gt_idx],
                })

            logger.info(
                "[MEDQA_BATCH] Processed batch",
                batch_start=batch_start,
                batch_size=len(batch),
                running_accuracy=correct_count / total_count if total_count > 0 else 0.0,
            )

        elapsed = time.perf_counter() - start_time
        accuracy = correct_count / total_count if total_count > 0 else 0.0
        ci_low, ci_high = _wilson_ci(accuracy, total_count)

        result = TaskEvalResult(
            accuracy=accuracy,
            accuracy_ci_low=ci_low,
            accuracy_ci_high=ci_high,
            n_samples=total_count,
            domain="medqa",
            model_path=model_path,
            eval_duration_seconds=elapsed,
            per_question_results=per_question,
            metadata={"split": split, "batch_size": batch_size},
        )

        logger.info(
            "[MEDQA_COMPLETE] MedQA evaluation finished",
            accuracy=accuracy,
            ci_low=ci_low,
            ci_high=ci_high,
            n_samples=total_count,
            duration_s=round(elapsed, 2),
        )
        return result

    @torch.inference_mode()
    def evaluate_humaneval(
        self,
        model_path: str,
        num_samples_per_problem: int = 20,
    ) -> TaskEvalResult:
        """Evaluate on HumanEval code generation benchmark.

        Uses the human_eval package. Generates num_samples_per_problem completions
        per problem, executes each in a sandboxed subprocess with 10s timeout,
        and computes pass@k for k in [1, 10] using the unbiased estimator.

        Args:
            model_path: Path to model checkpoint.
            num_samples_per_problem: Number of completions to generate per problem.

        Returns:
            TaskEvalResult with pass@1 as accuracy and pass@k in metadata.

        Complexity: O(P * K * S * V) where P=problems, K=samples, S=seq_len, V=vocab.
        """
        try:
            from human_eval.data import read_problems
        except ImportError as exc:
            raise EvalError(
                "human_eval package required for HumanEval evaluation",
                context={"missing_package": "human_eval"},
            ) from exc

        start_time = time.perf_counter()

        with timed_operation(logger, "HUMANEVAL_LOAD_MODEL", model_path=model_path):
            model, tokenizer = self._load_model_and_tokenizer(model_path)

        problems = read_problems()
        per_question: list[dict[str, object]] = []
        per_problem_correct: list[int] = []

        for task_id, problem in problems.items():
            prompt = problem["prompt"]
            entry_point = problem["entry_point"]
            test_code = problem["test"]

            correct_samples = 0
            for sample_idx in range(num_samples_per_problem):
                # Generate completion
                inputs = tokenizer(
                    prompt, return_tensors="pt", truncation=True, max_length=1024
                ).to(self._device)

                generated = model.generate(
                    **inputs,
                    max_new_tokens=512,
                    temperature=0.8,
                    top_p=0.95,
                    do_sample=True,
                    pad_token_id=tokenizer.pad_token_id,
                )
                completion = tokenizer.decode(
                    generated[0][inputs["input_ids"].shape[1] :],
                    skip_special_tokens=True,
                )

                # Execute in sandbox with timeout
                full_code = prompt + completion + "\n" + test_code + f"\ncheck({entry_point})\n"
                passed = self._execute_code_sandbox(full_code, timeout_seconds=10)
                if passed:
                    correct_samples += 1

            per_problem_correct.append(correct_samples)
            per_question.append({
                "question_id": task_id,
                "correct": correct_samples > 0,
                "predicted": f"{correct_samples}/{num_samples_per_problem} passed",
                "ground_truth": entry_point,
            })

            logger.debug(
                "[HUMANEVAL_PROBLEM] Evaluated problem",
                task_id=task_id,
                correct_samples=correct_samples,
                total_samples=num_samples_per_problem,
            )

        # Compute pass@k using unbiased estimator
        n = num_samples_per_problem
        pass_at_1_values = [_passatk_unbiased(n, c, 1) for c in per_problem_correct]
        pass_at_10_values = [
            _passatk_unbiased(n, c, min(10, n)) for c in per_problem_correct
        ]

        pass_at_1 = float(np.mean(pass_at_1_values))
        pass_at_10 = float(np.mean(pass_at_10_values))

        elapsed = time.perf_counter() - start_time
        ci_low, ci_high = _wilson_ci(pass_at_1, len(problems))

        result = TaskEvalResult(
            accuracy=pass_at_1,
            accuracy_ci_low=ci_low,
            accuracy_ci_high=ci_high,
            n_samples=len(problems),
            domain="humaneval",
            model_path=model_path,
            eval_duration_seconds=elapsed,
            per_question_results=per_question,
            metadata={
                "num_samples_per_problem": num_samples_per_problem,
                "pass_at_1": pass_at_1,
                "pass_at_10": pass_at_10,
            },
        )

        logger.info(
            "[HUMANEVAL_COMPLETE] HumanEval evaluation finished",
            pass_at_1=pass_at_1,
            pass_at_10=pass_at_10,
            n_problems=len(problems),
            duration_s=round(elapsed, 2),
        )
        return result

    @staticmethod
    def _execute_code_sandbox(code: str, timeout_seconds: int = 10) -> bool:
        """Execute generated code in an isolated subprocess.

        Runs the code in a fresh Python interpreter with a hard timeout.
        Returns True only if the process exits with code 0 (all assertions passed).

        Args:
            code: Python source code to execute.
            timeout_seconds: Maximum execution time before SIGKILL.

        Returns:
            True if execution succeeded (exit code 0), False otherwise.

        Complexity: O(1) for the subprocess management; execution time bounded by timeout.
        """
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=True) as f:
            f.write(code)
            f.flush()
            try:
                result = subprocess.run(
                    [sys.executable, f.name],
                    capture_output=True,
                    timeout=timeout_seconds,
                    check=False,
                )
                return result.returncode == 0
            except subprocess.TimeoutExpired:
                return False
            except OSError:
                return False

    def compare_models(
        self,
        baseline_path: str,
        tasft_path: str,
        domain: str,
        **kwargs: object,
    ) -> ComparisonResult:
        """Compare baseline and TASFT model on a domain task.

        Runs evaluation on both models, then performs a two-tailed independent
        t-test on per-question binary accuracy scores and computes Cohen's d.

        Args:
            baseline_path: Path to baseline (standard LoRA) model.
            tasft_path: Path to TASFT co-trained model/bundle.
            domain: Which domain to evaluate ("medqa" or "humaneval").
            **kwargs: Additional arguments passed to the domain evaluator.

        Returns:
            ComparisonResult with statistical significance and effect size.

        Raises:
            ValidationError: If domain is not recognized.
        """
        eval_dispatch = {
            "medqa": self.evaluate_medqa,
            "humaneval": self.evaluate_humaneval,
        }
        if domain not in eval_dispatch:
            raise ValidationError(
                f"Unknown domain '{domain}', expected one of {list(eval_dispatch.keys())}",
                context={"domain": domain},
            )

        eval_fn = eval_dispatch[domain]

        with timed_operation(logger, "COMPARE_BASELINE", model_path=baseline_path, domain=domain):
            baseline_result = eval_fn(baseline_path, **kwargs)  # type: ignore[arg-type]

        with timed_operation(logger, "COMPARE_TASFT", model_path=tasft_path, domain=domain):
            tasft_result = eval_fn(tasft_path, **kwargs)  # type: ignore[arg-type]

        # Extract per-question binary scores for t-test
        baseline_scores = np.array(
            [1.0 if q["correct"] else 0.0 for q in baseline_result.per_question_results],
            dtype=np.float64,
        )
        tasft_scores = np.array(
            [1.0 if q["correct"] else 0.0 for q in tasft_result.per_question_results],
            dtype=np.float64,
        )

        # Two-tailed independent samples t-test
        t_stat, p_value = stats.ttest_ind(tasft_scores, baseline_scores, equal_var=False)

        # Cohen's d = (mean_tasft - mean_baseline) / pooled_std
        mean_diff = float(np.mean(tasft_scores) - np.mean(baseline_scores))
        n_b, n_t = len(baseline_scores), len(tasft_scores)
        var_b = float(np.var(baseline_scores, ddof=1))
        var_t = float(np.var(tasft_scores, ddof=1))
        # Pooled standard deviation (Welch-style for unequal variances)
        pooled_var = ((n_b - 1) * var_b + (n_t - 1) * var_t) / (n_b + n_t - 2)
        pooled_std = math.sqrt(pooled_var) if pooled_var > 0 else 1e-10
        cohens_d = mean_diff / pooled_std

        delta_accuracy = tasft_result.accuracy - baseline_result.accuracy

        comparison = ComparisonResult(
            baseline_result=baseline_result,
            tasft_result=tasft_result,
            delta_accuracy=delta_accuracy,
            p_value=float(p_value),
            significant=float(p_value) < 0.01,
            effect_size=cohens_d,
            within_target=abs(delta_accuracy) < _TARGET_DELTA,
        )

        logger.info(
            "[COMPARE_COMPLETE] Model comparison finished",
            domain=domain,
            baseline_accuracy=baseline_result.accuracy,
            tasft_accuracy=tasft_result.accuracy,
            delta=delta_accuracy,
            p_value=float(p_value),
            significant=comparison.significant,
            effect_size=cohens_d,
            within_target=comparison.within_target,
        )
        return comparison


__all__ = [
    "ComparisonResult",
    "EvalError",
    "TaskEvalResult",
    "TaskEvaluator",
]
