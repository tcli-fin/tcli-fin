#!/usr/bin/env python3
"""
Adapter for TAT-QA benchmark.

Converts model predictions to the expected format: JSON dict {qid: [answer, scale]}
where scale may be "" for spans or a unit like "thousand/million/billion/percent".
"""

import json
from typing import Dict, Tuple, List, Any, Union


def to_tatqa_json(pred_dict: Dict[str, Tuple[Union[str, float], str]]) -> Dict[str, List[Union[str, float]]]:
    """
    Convert predictions to TAT-QA format.

    Args:
        pred_dict: {question_id: (answer, scale)} where scale may be "" for spans

    Returns:
        Dict with {id: [answer, scale_or_empty_or_list]} format
    """
    out = {}
    for qid, (ans, scale) in pred_dict.items():
        out[qid] = [ans, scale]
    return out


def parse_tatqa_predictions(pred_json_path: str) -> Dict[str, Tuple[Union[str, float], str]]:
    """
    Parse predictions from TAT-QA format JSON file.

    Args:
        pred_json_path: Path to predictions JSON file

    Returns:
        Dict of {question_id: (answer, scale)} tuples
    """
    with open(pred_json_path, 'r') as f:
        data = json.load(f)

    return {qid: tuple(pred) for qid, pred in data.items()}


def save_tatqa_predictions(pred_dict: Dict[str, Tuple[Union[str, float], str]], output_path: str):
    """
    Save predictions in TAT-QA format.

    Args:
        pred_dict: Dict of {question_id: (answer, scale)} tuples
        output_path: Path to save JSON file
    """
    tatqa_format = to_tatqa_json(pred_dict)
    with open(output_path, 'w') as f:
        json.dump(tatqa_format, f, indent=2)


def validate_tatqa_format(answer: Union[str, float], scale: str) -> bool:
    """
    Validate that answer and scale are in proper TAT-QA format.

    Args:
        answer: The predicted answer (string or numeric)
        scale: The scale/unit (e.g., "thousand", "million", "percent", "")

    Returns:
        True if format is valid
    """
    # Scale must be one of the valid options or empty
    valid_scales = {"", "thousand", "million", "billion", "percent"}
    if scale not in valid_scales:
        return False

    # If scale is specified, answer should be numeric
    if scale and not isinstance(answer, (int, float)):
        try:
            float(answer)
        except (ValueError, TypeError):
            return False

    return True


def normalize_tatqa_answer(answer: Union[str, float], scale: str) -> Tuple[float, str]:
    """
    Normalize TAT-QA answer to standard numeric format.

    Args:
        answer: Raw answer value
        scale: Scale/unit

    Returns:
        Tuple of (normalized_value, normalized_scale)
    """
    try:
        # Convert answer to float
        if isinstance(answer, (int, float)):
            val = float(answer)
        else:
            val = float(str(answer).replace(',', '').replace('$', ''))

        # Apply scale conversion
        scale_multipliers = {
            "thousand": 1000,
            "million": 1000000,
            "billion": 1000000000,
            "percent": 0.01,
            "": 1
        }

        normalized_val = val * scale_multipliers.get(scale, 1)
        return normalized_val, ""

    except (ValueError, TypeError):
        return 0.0, scale


# Example usage
if __name__ == "__main__":
    # Example predictions
    example_preds = {
        "tatqa_001": (125000, "thousand"),  # Should normalize to 125000000
        "tatqa_002": (15.5, "percent"),      # Should normalize to 0.155
        "tatqa_003": ("increased by 20%", "")  # String answer, no scale
    }

    # Convert to TAT-QA format
    tatqa_json = to_tatqa_json(example_preds)
    print("TAT-QA format:")
    print(json.dumps(tatqa_json, indent=2))

    # Validate examples
    for qid, (ans, scale) in example_preds.items():
        valid = validate_tatqa_format(ans, scale)
        print(f"Prediction {qid}: {'✓' if valid else '✗'}")

        if valid:
            normalized_val, normalized_scale = normalize_tatqa_answer(ans, scale)
            print(f"  Normalized: {normalized_val} (scale: {normalized_scale})")
