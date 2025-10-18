#!/usr/bin/env python3
"""
Evaluate DocFinQA JSON datasets using Gemini.

Reads a large JSON array where each element has fields: Context, Question, Answer, Program.
Sends Context + Question to Gemini, compares model output to Answer, and computes EM/F1.

Output: CSV with columns: split,row_index,em,f1,pred,gold

Environment variables:
- GEMINI_API_KEY (for direct Gemini API) OR Google login via gemini-cli is not used here.

Usage examples:
  python eval_docfinqa.py --split test --input DocFinQA/test.json --model gemini-2.5-pro --max-examples 100 --out results_test.csv
  python eval_docfinqa.py --split dev --input DocFinQA/dev.json --model gemini-2.5-flash --out results_dev.csv
"""

import argparse
import csv
import json
import math
import os
import re
import sys
from typing import Dict, Iterable, Tuple


def normalize_answer(text: str) -> str:
    if text is None:
        return ""
    text = text.strip()
    # Lowercase
    text = text.lower()
    # Normalize whitespace
    text = re.sub(r"\s+", " ", text)
    # Strip surrounding quotes
    if (text.startswith('"') and text.endswith('"')) or (text.startswith("'") and text.endswith("'")):
        text = text[1:-1].strip()
    return text


def compute_em_f1(prediction: str, ground_truth: str) -> Tuple[float, float]:
    p = normalize_answer(prediction)
    g = normalize_answer(ground_truth)
    em = 1.0 if p == g and g != "" else 0.0
    # Token F1
    p_tokens = p.split()
    g_tokens = g.split()
    if not p_tokens and not g_tokens:
        return em, 1.0
    if not p_tokens or not g_tokens:
        return em, 0.0
    common: Dict[str, int] = {}
    for t in g_tokens:
        common[t] = common.get(t, 0) + 1
    num_same = 0
    for t in p_tokens:
        if common.get(t, 0) > 0:
            num_same += 1
            common[t] -= 1
    if num_same == 0:
        return em, 0.0
    precision = num_same / len(p_tokens)
    recall = num_same / len(g_tokens)
    f1 = 2 * precision * recall / (precision + recall)
    return em, f1


def iter_json_array_stream(path: str) -> Iterable[Dict]:
    """Stream a large JSON array without loading entire file.

    This simple streaming parses objects between top-level { } assuming a JSON array at top.
    """
    with open(path, "r", encoding="utf-8") as f:
        buf = []
        in_array = False
        level = 0
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            for ch in chunk:
                if not in_array:
                    if ch == '[':
                        in_array = True
                    continue
                if ch == '{':
                    level += 1
                if level > 0:
                    buf.append(ch)
                if ch == '}':
                    level -= 1
                    if level == 0:
                        obj_text = ''.join(buf)
                        buf.clear()
                        try:
                            yield json.loads(obj_text)
                        except Exception:
                            # Attempt to repair common trailing commas
                            obj_text2 = re.sub(r",\s*}$", "}", obj_text)
                            yield json.loads(obj_text2)
                # End array
                if level == 0 and ch == ']':
                    return


def build_prompt(context: str, question: str) -> str:
    return (
        "You are a precise financial QA assistant.\n"
        "Answer the question using ONLY the provided context.\n"
        "If the answer is not in the context, say 'unknown'.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}\n"
        "Answer:"
    )


def load_gemini_client(model_name: str):
    try:
        import google.generativeai as genai
    except ImportError:
        print("Missing dependency google-generativeai. Install requirements and try again.", file=sys.stderr)
        sys.exit(2)
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("GEMINI_API_KEY is not set. Export your API key.", file=sys.stderr)
        sys.exit(2)
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(model_name)
    return model


def call_gemini(model, prompt: str, tokens: int) -> str:
    # Conservative safety settings; adjust as needed
    try:
        rsp = model.generate_content(prompt, generation_config={
            "temperature": 0.0,
            "max_output_tokens": tokens,
        })
        if hasattr(rsp, 'text') and rsp.text is not None:
            return rsp.text.strip()
        # fall back to candidates
        if getattr(rsp, 'candidates', None):
            parts = rsp.candidates[0].content.parts
            if parts:
                return str(parts[0].text).strip()
        return ""
    except Exception as e:
        return f"ERROR: {e}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to DocFinQA split JSON file")
    parser.add_argument("--split", default="test", help="Dataset split label to include in CSV")
    parser.add_argument("--model", default="gemini-2.5-pro", help="Gemini model name")
    parser.add_argument("--max-examples", type=int, default=0, help="Limit number of examples (0 = all)")
    parser.add_argument("--out", required=True, help="Output CSV path")
    parser.add_argument("--max-output-tokens", type=int, default=64)
    parser.add_argument("--start-index", type=int, default=0, help="Skip first N examples")
    args = parser.parse_args()

    model = load_gemini_client(args.model)

    total = 0
    sum_em = 0.0
    sum_f1 = 0.0

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as fout:
        writer = csv.writer(fout)
        writer.writerow(["split", "row_index", "em", "f1", "pred", "gold"])

        for idx, example in enumerate(iter_json_array_stream(args.input)):
            if idx < args.start_index:
                continue
            if args.max_examples and total >= args.max_examples:
                break
            context = example.get("Context", "")
            question = example.get("Question", "")
            gold = example.get("Answer", "")

            prompt = build_prompt(context, question)
            pred = call_gemini(model, prompt, tokens=args.max_output_tokens)

            em, f1 = compute_em_f1(pred, gold)
            writer.writerow([args.split, idx, f1 == 1.0 and em == 1.0 and 1.0 or em, f1, pred, gold])
            sum_em += em
            sum_f1 += f1
            total += 1

            if total % 25 == 0:
                print(f"Processed {total} examples... EM={sum_em/total:.3f} F1={sum_f1/total:.3f}")

    if total > 0:
        print(f"Done. N={total}  EM={sum_em/total:.4f}  F1={sum_f1/total:.4f}")
    else:
        print("No examples processed.")


if __name__ == "__main__":
    main()


