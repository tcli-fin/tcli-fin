#!/usr/bin/env python3
"""
Adapter for DocFinQA benchmark.

Handles both program execution and numeric answer comparison for DocFinQA evaluation.
"""

import json
import math
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass


def exec_accuracy(pred_answers: List[float], gold_answers: List[float], tol: float = 1e-4) -> Tuple[int, int]:
    """
    Calculate execution accuracy for DocFinQA.

    Args:
        pred_answers: List of predicted numeric answers
        gold_answers: List of gold numeric answers
        tol: Tolerance for numeric comparison

    Returns:
        Tuple of (correct_count, total_count)
    """
    if len(pred_answers) != len(gold_answers):
        raise ValueError(f"Prediction and gold answer lists must have same length. Got {len(pred_answers)} vs {len(gold_answers)}")

    ok = 0
    for p, g in zip(pred_answers, gold_answers):
        if (isinstance(p, (int, float)) and isinstance(g, (int, float)) and
            (abs(p - g) <= tol or (abs(g) > tol and abs(p - g) / max(abs(g), tol) <= 1e-3))):
            ok += 1
    return ok, len(gold_answers)


def safe_exec_python(code: str) -> Tuple[bool, Optional[Any], Optional[str]]:
    """
    Execute Python code in a restricted namespace. The code should assign a variable `answer`.

    Returns: (ok, answer, error)
    """
    safe_builtins = {
        'abs': abs,
        'min': min,
        'max': max,
        'sum': sum,
        'len': len,
        'range': range,
        'round': round,
        'float': float,
        'int': int,
        'str': str,
        'print': print,
    }
    globals_dict: Dict[str, Any] = {'__builtins__': safe_builtins}
    locals_dict: Dict[str, Any] = {}

    try:
        exec(code, globals_dict, locals_dict)
    except Exception as e:
        return False, None, f"exec_error: {e}"

    if 'answer' not in locals_dict:
        return False, None, "answer_not_found"

    return True, locals_dict['answer'], None


def _to_float(val: Any) -> Optional[float]:
    """Convert value to float with error handling."""
    if val is None:
        return None
    if isinstance(val, (int, float)):
        return float(val)
    s = str(val).strip()
    if s.endswith('%'):
        try:
            return float(s[:-1]) / 100.0
        except Exception:
            return None
    # Remove commas and currency
    s = s.replace(',', '').replace('$', '')
    try:
        return float(s)
    except Exception:
        return None


def numeric_equal(a: Any, b: Any, rel_tol: float = 5e-3, abs_tol: float = 1e-4) -> bool:
    """Check if two numeric values are equal within tolerance."""
    fa = _to_float(a)
    fb = _to_float(b)
    if fa is None or fb is None:
        return False
    return math.isclose(fa, fb, rel_tol=rel_tol, abs_tol=abs_tol)


@dataclass
class DocFinQAEvaluation:
    """Results from DocFinQA evaluation."""
    correct: int
    total: int
    accuracy: float
    execution_errors: int
    results: List[Dict[str, Any]]


def evaluate_docfinqa_predictions(predictions: List[Dict[str, Any]]) -> DocFinQAEvaluation:
    """
    Evaluate DocFinQA predictions.

    Args:
        predictions: List of prediction dicts with 'pred_program' or 'pred_answer' and 'gold_answer'

    Returns:
        DocFinQAEvaluation object with results
    """
    correct = 0
    execution_errors = 0
    results = []

    for i, pred in enumerate(predictions):
        gold_answer = pred.get('gold_answer')
        pred_program = pred.get('pred_program')
        pred_answer = pred.get('pred_answer')

        result = {
            'index': i,
            'gold_answer': gold_answer,
            'pred_program': pred_program,
            'pred_answer': pred_answer,
            'execution_error': None,
            'is_correct': False
        }

        if pred_program:
            # Execute the program
            ok, exec_result, error = safe_exec_python(pred_program)
            if not ok:
                execution_errors += 1
                result['execution_error'] = error
            else:
                result['pred_answer'] = exec_result
                if numeric_equal(exec_result, gold_answer):
                    correct += 1
                    result['is_correct'] = True
        elif pred_answer is not None:
            # Direct numeric comparison
            if numeric_equal(pred_answer, gold_answer):
                correct += 1
                result['is_correct'] = True

        results.append(result)

    total = len(predictions)
    accuracy = correct / total if total > 0 else 0

    return DocFinQAEvaluation(
        correct=correct,
        total=total,
        accuracy=accuracy,
        execution_errors=execution_errors,
        results=results
    )


def save_docfinqa_results(evaluation: DocFinQAEvaluation, output_path: str):
    """Save evaluation results to JSON file."""
    output_data = {
        'accuracy': evaluation.accuracy,
        'correct': evaluation.correct,
        'total': evaluation.total,
        'execution_errors': evaluation.execution_errors,
        'detailed_results': evaluation.results
    }

    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2, default=str)


# Example usage
if __name__ == "__main__":
    # Example predictions with programs
    example_predictions = [
        {
            'gold_answer': 1000.0,
            'pred_program': 'answer = 500 + 500'
        },
        {
            'gold_answer': 0.25,
            'pred_program': 'answer = 1/4'
        },
        {
            'gold_answer': 1500000,
            'pred_program': 'answer = 1.5 * 1000000'
        },
        {
            'gold_answer': 1000.0,
            'pred_answer': 1000.0  # Direct answer
        }
    ]

    # Evaluate
    evaluation = evaluate_docfinqa_predictions(example_predictions)

    print("DocFinQA Evaluation Results:")
    print(f"Accuracy: {evaluation.accuracy".3f"} ({evaluation.correct}/{evaluation.total})")
    print(f"Execution errors: {evaluation.execution_errors}")

    # Save results
    save_docfinqa_results(evaluation, 'docfinqa_example_results.json')
    print("Results saved to docfinqa_example_results.json")
