#!/usr/bin/env python3
"""
Adapter for FinQA and ConvFinQA benchmarks.

Converts model predictions to the expected format: JSON list of {id, predicted}
where predicted is a list of tokens ending with 'EOF'.
"""

import json
from typing import List, Tuple, Dict, Any


def to_finqa_program_json(preds: List[Tuple[str, List[str]]]) -> List[Dict[str, Any]]:
    """
    Convert predictions to FinQA/ConvFinQA format.

    Args:
        preds: List of (id, tokens_list) where tokens_list ends with 'EOF'

    Returns:
        List of dicts with {id, predicted} format for JSON serialization
    """
    return [{"id": pid, "predicted": toks} for pid, toks in preds]


def parse_finqa_predictions(pred_json_path: str) -> List[Tuple[str, List[str]]]:
    """
    Parse predictions from FinQA format JSON file.

    Args:
        pred_json_path: Path to predictions JSON file

    Returns:
        List of (id, tokens_list) tuples
    """
    with open(pred_json_path, 'r') as f:
        data = json.load(f)

    return [(item['id'], item['predicted']) for item in data]


def save_finqa_predictions(preds: List[Tuple[str, List[str]]], output_path: str):
    """
    Save predictions in FinQA format.

    Args:
        preds: List of (id, tokens_list) tuples
        output_path: Path to save JSON file
    """
    finqa_format = to_finqa_program_json(preds)
    with open(output_path, 'w') as f:
        json.dump(finqa_format, f, indent=2)


def validate_finqa_format(tokens: List[str]) -> bool:
    """
    Validate that tokens are in proper FinQA format.

    Args:
        tokens: List of program tokens

    Returns:
        True if format is valid
    """
    if not tokens:
        return False

    # Should end with EOF
    if tokens[-1] != 'EOF':
        return False

    # Should not be empty except for EOF
    if len(tokens) == 1:
        return False

    return True


# Example usage
if __name__ == "__main__":
    # Example predictions
    example_preds = [
        ("finqa_001", ["ADD", "NUM_100", "NUM_200", "EOF"]),
        ("finqa_002", ["MULTIPLY", "NUM_10", "NUM_5", "EOF"]),
        ("convfinqa_001", ["SUBTRACT", "revenue_2022", "revenue_2021", "EOF"])
    ]

    # Convert to FinQA format
    finqa_json = to_finqa_program_json(example_preds)
    print("FinQA format:")
    print(json.dumps(finqa_json, indent=2))

    # Validate examples
    for pred_id, tokens in example_preds:
        valid = validate_finqa_format(tokens)
        print(f"Prediction {pred_id}: {'✓' if valid else '✗'}")
