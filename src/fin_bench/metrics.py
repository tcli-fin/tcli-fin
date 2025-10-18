"""
Evaluation metrics for financial QA benchmarks.
"""

import re
import math
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass
from abc import ABC, abstractmethod

from .types import MetricDict


@dataclass
class EvaluationResult:
    """Result of evaluating a single sample."""
    prediction: str
    ground_truth: str
    metrics: MetricDict
    is_correct: bool
    error: Optional[str] = None
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class BaseMetric(ABC):
    """Base class for evaluation metrics."""

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def compute(self, prediction: str, ground_truth: str, **kwargs) -> float:
        """Compute metric value."""
        pass

    def normalize_text(self, text: str) -> str:
        """Normalize text for comparison."""
        if not text:
            return ""

        text = str(text).strip()

        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)

        # Remove quotes
        if (text.startswith('"') and text.endswith('"')) or \
           (text.startswith("'") and text.endswith("'")):
            text = text[1:-1].strip()

        return text.lower()


class ExactMatchMetric(BaseMetric):
    """Exact match metric."""

    def __init__(self):
        super().__init__("exact_match")

    def compute(self, prediction: str, ground_truth: str, **kwargs) -> float:
        """Compute exact match score."""
        pred_norm = self.normalize_text(prediction)
        truth_norm = self.normalize_text(ground_truth)

        return 1.0 if pred_norm == truth_norm else 0.0


class F1Metric(BaseMetric):
    """F1 score metric."""

    def __init__(self):
        super().__init__("f1")

    def compute(self, prediction: str, ground_truth: str, **kwargs) -> float:
        """Compute token-level F1 score."""
        pred_norm = self.normalize_text(prediction)
        truth_norm = self.normalize_text(ground_truth)

        if not pred_norm and not truth_norm:
            return 1.0
        if not pred_norm or not truth_norm:
            return 0.0

        pred_tokens = pred_norm.split()
        truth_tokens = truth_norm.split()

        if not pred_tokens or not truth_tokens:
            return 0.0

        # Count token overlaps
        common = {}
        for token in truth_tokens:
            common[token] = common.get(token, 0) + 1

        matches = 0
        for token in pred_tokens:
            if common.get(token, 0) > 0:
                matches += 1
                common[token] -= 1

        if matches == 0:
            return 0.0

        precision = matches / len(pred_tokens)
        recall = matches / len(truth_tokens)

        return 2 * precision * recall / (precision + recall)


class NumericToleranceMetric(BaseMetric):
    """Numeric tolerance metric for financial values."""

    def __init__(self, rel_tol: float = 1e-3, abs_tol: float = 1e-6):
        super().__init__("numeric_tolerance")
        self.rel_tol = rel_tol
        self.abs_tol = abs_tol

    def compute(self, prediction: str, ground_truth: str, **kwargs) -> float:
        """Compute numeric tolerance match."""
        try:
            pred_val = self._parse_numeric(prediction)
            truth_val = self._parse_numeric(ground_truth)

            if pred_val is None or truth_val is None:
                return 0.0

            return 1.0 if math.isclose(pred_val, truth_val,
                                     rel_tol=self.rel_tol, abs_tol=self.abs_tol) else 0.0

        except Exception:
            return 0.0

    def _parse_numeric(self, text: str) -> Optional[float]:
        """Parse numeric value from text."""
        if not text:
            return None

        # Remove common formatting
        text = text.replace('$', '').replace(',', '').replace('%', '').strip()

        # Handle scale words
        scale_multipliers = {
            'thousand': 1000,
            'million': 1000000,
            'billion': 1000000000,
            'trillion': 1000000000000
        }

        multiplier = 1.0
        for word, mult in scale_multipliers.items():
            if f' {word}' in text.lower():
                text = text.lower().replace(word, '').strip()
                multiplier = mult
                break

        try:
            return float(text) * multiplier
        except ValueError:
            return None


class ProgramExecutionMetric(BaseMetric):
    """Program execution metric for FinQA-style datasets."""

    def __init__(self):
        super().__init__("program_execution")

    def compute(self, prediction: str, ground_truth: str, **kwargs) -> float:
        """Execute program and compare numeric result."""
        program = kwargs.get('program', '')
        context = kwargs.get('context', '')

        if not program:
            return 0.0

        try:
            # Execute the program
            result = self._execute_program(program)
            if result is None:
                return 0.0

            # Compare with ground truth
            truth_val = self._parse_numeric(ground_truth)
            if truth_val is None:
                return 0.0

            # Use numeric tolerance for comparison
            return 1.0 if math.isclose(result, truth_val, rel_tol=1e-3, abs_tol=1e-6) else 0.0

        except Exception:
            return 0.0

    def _execute_program(self, program: str) -> Optional[float]:
        """Execute Python program and extract answer."""
        safe_builtins = {
            'abs': abs, 'min': min, 'max': max, 'sum': sum,
            'len': len, 'range': range, 'round': round,
            'float': float, 'int': int, 'str': str
        }

        globals_dict = {'__builtins__': safe_builtins}
        locals_dict = {}

        try:
            exec(program, globals_dict, locals_dict)
            if 'answer' in locals_dict:
                return float(locals_dict['answer'])
        except Exception:
            pass

        return None

    def _parse_numeric(self, text: str) -> Optional[float]:
        """Parse numeric value from text."""
        if not text:
            return None

        text = text.replace('$', '').replace(',', '').replace('%', '').strip()

        try:
            return float(text)
        except ValueError:
            return None


class CompositeMetric(BaseMetric):
    """Composite metric that combines multiple metrics."""

    def __init__(self, metrics: List[BaseMetric]):
        super().__init__("composite")
        self.metrics = metrics

    def compute(self, prediction: str, ground_truth: str, **kwargs) -> float:
        """Compute composite score."""
        if not self.metrics:
            return 0.0

        scores = []
        for metric in self.metrics:
            try:
                score = metric.compute(prediction, ground_truth, **kwargs)
                scores.append(score)
            except Exception:
                scores.append(0.0)

        return sum(scores) / len(scores) if scores else 0.0


class MetricFactory:
    """Factory for creating metrics."""

    @staticmethod
    def create_exact_match() -> ExactMatchMetric:
        """Create exact match metric."""
        return ExactMatchMetric()

    @staticmethod
    def create_f1() -> F1Metric:
        """Create F1 metric."""
        return F1Metric()

    @staticmethod
    def create_numeric_tolerance(rel_tol: float = 1e-3, abs_tol: float = 1e-6) -> NumericToleranceMetric:
        """Create numeric tolerance metric."""
        return NumericToleranceMetric(rel_tol=rel_tol, abs_tol=abs_tol)

    @staticmethod
    def create_program_execution() -> ProgramExecutionMetric:
        """Create program execution metric."""
        return ProgramExecutionMetric()

    @staticmethod
    def create_composite(metrics: List[BaseMetric]) -> CompositeMetric:
        """Create composite metric."""
        return CompositeMetric(metrics)


def evaluate_prediction(
    prediction: str,
    ground_truth: str,
    metrics: List[BaseMetric],
    **kwargs
) -> EvaluationResult:
    """Evaluate a single prediction using multiple metrics."""
    metric_scores = {}

    for metric in metrics:
        try:
            score = metric.compute(prediction, ground_truth, **kwargs)
            metric_scores[metric.name] = score
        except Exception as e:
            metric_scores[metric.name] = 0.0

    # Determine if prediction is correct (at least one metric passes)
    is_correct = any(score > 0.5 for score in metric_scores.values())

    return EvaluationResult(
        prediction=prediction,
        ground_truth=ground_truth,
        metrics=metric_scores,
        is_correct=is_correct
    )


def aggregate_results(results: List[EvaluationResult]) -> Dict[str, Any]:
    """Aggregate evaluation results."""
    if not results:
        return {}

    total = len(results)
    correct = sum(1 for r in results if r.is_correct)
    errors = sum(1 for r in results if r.error)

    # Aggregate metrics
    metric_sums = {}
    metric_counts = {}
    # Optional judge aggregates (from result.metadata)
    judge_exact_sum = 0
    judge_approx_sum = 0

    for result in results:
        for metric_name, score in result.metrics.items():
            if metric_name not in metric_sums:
                metric_sums[metric_name] = 0.0
                metric_counts[metric_name] = 0
            metric_sums[metric_name] += score
            metric_counts[metric_name] += 1
        try:
            md = result.metadata or {}
            judge_exact_sum += int(md.get("judge_exact_match", 0) or 0)
            judge_approx_sum += int(md.get("judge_approximate_match", 0) or 0)
        except Exception:
            pass

    aggregated_metrics = {}
    for metric_name in metric_sums:
        count = metric_counts[metric_name]
        aggregated_metrics[metric_name] = metric_sums[metric_name] / count if count > 0 else 0.0

    payload = {
        "total_samples": total,
        "correct_samples": correct,
        "error_samples": errors,
        "accuracy": correct / total if total > 0 else 0.0,
        "error_rate": errors / total if total > 0 else 0.0,
        "metrics": aggregated_metrics
    }

    # Surface judge rates when present
    if total > 0 and (judge_exact_sum > 0 or judge_approx_sum > 0):
        aggregated_metrics["judge_exact"] = judge_exact_sum / total
        aggregated_metrics["judge_approximate"] = judge_approx_sum / total
        # Judge overall accuracy: percentage of samples marked as correct (exact OR approximate)
        aggregated_metrics["judge_accuracy"] = (judge_exact_sum + judge_approx_sum) / total

    return payload


def get_default_metrics_for_dataset(dataset_type: str) -> List[BaseMetric]:
    """Get default metrics for a dataset type."""
    base_metrics = [MetricFactory.create_exact_match(), MetricFactory.create_f1()]

    if dataset_type in ['finqa', 'docfinqa']:
        base_metrics.append(MetricFactory.create_program_execution())

    if dataset_type in ['financebench', 'tatqa']:
        base_metrics.append(MetricFactory.create_numeric_tolerance())

    return base_metrics
