#!/usr/bin/env python3
"""
Adapter for FinanceBench benchmark.

Handles both strict accuracy and failure rate evaluation for FinanceBench.
"""

import json
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass


def normalize_answer(text: str) -> str:
    """Normalize answer text for comparison."""
    if text is None:
        return ""

    text = str(text).strip()

    # Handle numeric values
    if any(char.isdigit() for char in text):
        # Remove common prefixes/suffixes
        text = text.replace('$', '').replace(',', '').replace('%', '').strip()
        # Handle "million", "billion" etc.
        multipliers = {
            'thousand': '000',
            'million': '000000',
            'billion': '000000000',
            'trillion': '000000000000'
        }
        for word, zeros in multipliers.items():
            if f' {word}' in text.lower() or f'{word} ' in text.lower():
                text = text.lower().replace(word, '').strip()
                text += zeros
                break

    # Lowercase for case-insensitive comparison
    return text.lower()


def strict_accuracy(pred: str, gold: str) -> bool:
    """Calculate strict accuracy (exact match after normalization)."""
    return normalize_answer(pred) == normalize_answer(gold)


def numeric_tolerance_accuracy(pred: str, gold: str, rel_tol: float = 0.01) -> bool:
    """Calculate accuracy with numeric tolerance."""
    try:
        pred_norm = normalize_answer(pred)
        gold_norm = normalize_answer(gold)

        # Try direct numeric comparison
        pred_val = float(pred_norm)
        gold_val = float(gold_norm)

        return abs(pred_val - gold_val) <= abs(gold_val * rel_tol)
    except (ValueError, TypeError):
        # Fall back to string comparison
        return strict_accuracy(pred, gold)


def is_refusal_or_error(text: str) -> bool:
    """Check if response indicates refusal or error."""
    if not text:
        return True

    text_lower = str(text).lower().strip()

    refusal_indicators = [
        "i don't know", "i cannot", "i can't", "not sure", "uncertain",
        "insufficient information", "not enough", "cannot determine",
        "not available", "n/a", "unknown", "sorry", "apologize",
        "error", "exception", "failed", "unable to"
    ]

    return any(indicator in text_lower for indicator in refusal_indicators)


def calculate_failure_rate(responses: List[str]) -> float:
    """Calculate failure rate based on refusals/errors."""
    if not responses:
        return 0.0

    failures = sum(1 for response in responses if is_refusal_or_error(response))
    return failures / len(responses)


@dataclass
class FinanceBenchEvaluation:
    """Results from FinanceBench evaluation."""
    strict_accuracy: float
    numeric_accuracy: float
    failure_rate: float
    total_samples: int
    strict_correct: int
    numeric_correct: int
    failures: int
    detailed_results: List[Dict[str, Any]]


def evaluate_financebench_predictions(predictions: List[Dict[str, Any]]) -> FinanceBenchEvaluation:
    """
    Evaluate FinanceBench predictions.

    Args:
        predictions: List of prediction dicts with 'predicted' and 'gold' fields

    Returns:
        FinanceBenchEvaluation object with results
    """
    strict_correct = 0
    numeric_correct = 0
    failures = 0
    results = []

    for i, pred in enumerate(predictions):
        predicted = pred.get('predicted', '')
        gold = pred.get('gold', '')

        result = {
            'index': i,
            'predicted': predicted,
            'gold': gold,
            'strict_match': False,
            'numeric_match': False,
            'is_failure': False
        }

        # Check strict accuracy
        if strict_accuracy(predicted, gold):
            strict_correct += 1
            result['strict_match'] = True

        # Check numeric accuracy
        if numeric_tolerance_accuracy(predicted, gold):
            numeric_correct += 1
            result['numeric_match'] = True

        # Check for failure
        if is_refusal_or_error(predicted):
            failures += 1
            result['is_failure'] = True

        results.append(result)

    total = len(predictions)
    if total == 0:
        return FinanceBenchEvaluation(0, 0, 0, 0, 0, 0, 0, [])

    return FinanceBenchEvaluation(
        strict_accuracy=strict_correct / total,
        numeric_accuracy=numeric_correct / total,
        failure_rate=failures / total,
        total_samples=total,
        strict_correct=strict_correct,
        numeric_correct=numeric_correct,
        failures=failures,
        detailed_results=results
    )


def save_financebench_results(evaluation: FinanceBenchEvaluation, output_path: str):
    """Save evaluation results to JSON file."""
    output_data = {
        'strict_accuracy': evaluation.strict_accuracy,
        'numeric_accuracy': evaluation.numeric_accuracy,
        'failure_rate': evaluation.failure_rate,
        'total_samples': evaluation.total_samples,
        'strict_correct': evaluation.strict_correct,
        'numeric_correct': evaluation.numeric_correct,
        'failures': evaluation.failures,
        'detailed_results': evaluation.detailed_results
    }

    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)


# Example usage
if __name__ == "__main__":
    # Example predictions
    example_predictions = [
        {
            'predicted': '$1,250 million',
            'gold': '1250000000'
        },
        {
            'predicted': '15.5%',
            'gold': '0.155'
        },
        {
            'predicted': 'I cannot determine this from the provided information',
            'gold': '500000'
        },
        {
            'predicted': 'Approximately 2.1 billion',
            'gold': '2100000000'
        },
        {
            'predicted': 'Unknown',
            'gold': '750000000'
        }
    ]

    # Evaluate
    evaluation = evaluate_financebench_predictions(example_predictions)

    print("FinanceBench Evaluation Results:")
    print(f"Strict Accuracy: {evaluation.strict_accuracy".3f"} ({evaluation.strict_correct}/{evaluation.total_samples})")
    print(f"Numeric Accuracy: {evaluation.numeric_accuracy".3f"} ({evaluation.numeric_correct}/{evaluation.total_samples})")
    print(f"Failure Rate: {evaluation.failure_rate".3f"} ({evaluation.failures}/{evaluation.total_samples})")

    # Save results
    save_financebench_results(evaluation, 'financebench_example_results.json')
    print("Results saved to financebench_example_results.json")
