#!/usr/bin/env python3
"""Parallel DocFinQA evaluator with provider-agnostic clients, retries, and logging.

Processes a DocFinQA split with configurable concurrency, exponential backoff
retries, tqdm progress, and structured JSONL logging under
logs/parallel/{provider}/{model}/run-<timestamp>.jsonl.
"""

import argparse
import json
import os
import sys
import time
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import Dict, Iterable, List, Optional, Tuple

from tqdm import tqdm  # type: ignore

# Reuse utilities and provider sessions from the single-threaded script
from simple_gemini_eval import (
    ModelSession,
    compute_em_f1,
    generate_answer,
    judge_answer,
    read_examples,
    JudgeVerdict,
)


def normalize_provider_name(provider: str) -> str:
    name = (provider or "").strip().lower()
    if name == "openrouter" : # do not change
        return "openrouter"
    if name == "anthropic": # do not change
        return "anthropic-bedrock"
    return name


def build_log_path(base_dir: str, provider: str, model_name: str) -> str:
    safe_provider = provider.replace("/", "-")
    safe_model = model_name.replace("/", "-")
    run_id = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    directory = os.path.join(base_dir, safe_provider, safe_model)
    os.makedirs(directory, exist_ok=True)
    return os.path.join(directory, f"run-{run_id}.jsonl")


def write_jsonl(path: str, record: Dict[str, object]) -> None:
    record = dict(record)
    record.setdefault("timestamp", datetime.utcnow().isoformat() + "Z")
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, sort_keys=True) + "\n")


def load_dataset(path: str, start_index: int, limit: int) -> List[Dict]:
    items: List[Dict] = []
    for idx, obj in enumerate(read_examples(path)):
        if idx < start_index:
            continue
        items.append(obj)
        if limit and len(items) >= limit:
            break
    return items


def exponential_backoff_sleep(attempt_index: int, initial_backoff: float, max_backoff: float) -> None:
    delay = min(max_backoff, initial_backoff * (2 ** attempt_index))
    jitter = delay * 0.1 * random.random()
    time.sleep(delay + jitter)


def _is_rate_limit_error(message: str) -> bool:
    m = (message or "").lower()
    return (
        "429" in m
        or "too many requests" in m
        or "rate limit" in m
        or "rate-limit" in m
    )


def _sleep_with_rate_limit_awareness(
    attempt_index: int,
    initial_backoff: float,
    max_backoff: float,
    is_rate_limited: bool,
) -> None:
    if not is_rate_limited:
        exponential_backoff_sleep(attempt_index, initial_backoff, max_backoff)
        return
    # Apply extra scaling when rate limited
    scaled_initial = initial_backoff * 2.0
    delay = min(max_backoff, scaled_initial * (2 ** attempt_index))
    jitter = delay * 0.2 * random.random()
    # Add small extra random pause to reduce herd effects
    extra = 0.5 + random.random() * 1.5
    time.sleep(delay + jitter + extra)


def try_generate_with_retries(
    session: ModelSession,
    context: str,
    question: str,
    thinking_budget: Optional[int],
    retries: int,
    initial_backoff: float,
    max_backoff: float,
) -> str:
    last_text = ""
    for attempt in range(retries + 1):
        text = generate_answer(session, context, question, thinking_budget)
        last_text = text
        if text and not text.startswith("ERROR:"):
            return text
        if attempt < retries:
            _sleep_with_rate_limit_awareness(
                attempt,
                initial_backoff,
                max_backoff,
                is_rate_limited=_is_rate_limit_error(last_text),
            )
    return last_text


def try_judge_with_retries(
    session: ModelSession,
    context: str,
    question: str,
    gold: str,
    prediction: str,
    thinking_budget: Optional[int],
    retries: int,
    initial_backoff: float,
    max_backoff: float,
) -> Tuple[str, str, str]:
    last_tuple: Tuple[str, str, str] = ("error", "judge_not_run", "")
    for attempt in range(retries + 1):
        label, reason, raw = judge_answer(
            session=session,
            context=context,
            question=question,
            gold=gold,
            prediction=prediction,
            thinking_budget=thinking_budget,
        )
        last_tuple = (label, reason, raw)
        if label in {"exact", "approximate", "incorrect", "unknown"} and not (raw.startswith("ERROR:") if isinstance(raw, str) else False):
            return last_tuple
        if attempt < retries:
            _sleep_with_rate_limit_awareness(
                attempt,
                initial_backoff,
                max_backoff,
                is_rate_limited=_is_rate_limit_error(raw if isinstance(raw, str) else ""),
            )
    return last_tuple


def process_one(
    row_index: int,
    example: Dict[str, object],
    answer_session: ModelSession,
    judge_session: ModelSession,
    answer_thinking: Optional[int],
    judge_thinking: Optional[int],
    retries: int,
    initial_backoff: float,
    max_backoff: float,
) -> Dict[str, object]:
    context = str(example.get("Context", ""))
    question = str(example.get("Question", ""))
    gold = str(example.get("Answer", ""))

    full_output = try_generate_with_retries(
        session=answer_session,
        context=context,
        question=question,
        thinking_budget=answer_thinking,
        retries=retries,
        initial_backoff=initial_backoff,
        max_backoff=max_backoff,
    )

    # Parse numeric answer from the full output
    try:
        from simple_gemini_eval import parse_final_numeric_answer  # type: ignore
        parsed_answer = parse_final_numeric_answer(full_output)
    except Exception:
        parsed_answer = ""

    prediction = parsed_answer or full_output

    em, f1 = compute_em_f1(prediction, gold)

    label, reason, judge_raw = try_judge_with_retries(
        session=judge_session,
        context=context,
        question=question,
        gold=gold,
        prediction=full_output,
        thinking_budget=judge_thinking,
        retries=retries,
        initial_backoff=initial_backoff,
        max_backoff=max_backoff,
    )

    judge_exact_flag = 0
    judge_approx_flag = 0

    # Attempt to parse structured judge JSON to surface flags and normalized label/reason
    try:
        parsed = json.loads(judge_raw) if isinstance(judge_raw, str) else None
        if isinstance(parsed, dict):
            try:
                verdict = JudgeVerdict(
                    label=str(parsed.get("label", "")).strip().lower(),
                    reason=str(parsed.get("reason", "")),
                    exact_match=int(parsed.get("exact_match", 0)),
                    approximate_match=int(parsed.get("approximate_match", 0)),
                )
                label = verdict.label
                reason = verdict.reason
                judge_exact_flag = 1 if int(verdict.exact_match) else 0
                judge_approx_flag = 1 if int(verdict.approximate_match) else 0
            except Exception:
                pass
    except Exception:
        pass

    # Fallback if judge returned unknown: derive from lexical EM
    if label == "unknown":
        if em >= 1.0:
            label = "exact"
            reason = "fallback_lexical_exact"
            judge_exact_flag = 1
            judge_approx_flag = 0
        else:
            label = "incorrect"
            reason = "fallback_lexical_mismatch"
            judge_exact_flag = 0
            judge_approx_flag = 0

    return {
        "row": row_index,
        "question": question,
        "gold_answer": gold,
        "model_answer": prediction,
        "model_full_output": full_output,
        "model_error": prediction.startswith("ERROR:"),
        "judge_label": label,
        "judge_reason": reason,
        "judge_raw": judge_raw,
        "judge_exact_match": judge_exact_flag,
        "judge_approximate_match": judge_approx_flag,
        "exact_match": em,
        "f1": f1,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run parallel DocFinQA evaluation with retries")
    parser.add_argument("--input", default="DocFinQA/test.json", help="Path to DocFinQA split JSON")
    parser.add_argument(
        "--provider",
        default="gemini",
        choices=["gemini", "openrouter", "openorder", "openai", "anthropic-bedrock", "entropic"],
        help="Provider for answering",
    )
    parser.add_argument("--model", default="gemini-2.5-pro", help="Model name for answering")
    parser.add_argument(
        "--judge-provider",
        choices=["gemini", "openrouter", "openorder", "openai", "anthropic-bedrock", "entropic"],
        help="Provider for judging (defaults to --provider)",
    )
    parser.add_argument("--judge-model", default=None, help="Model name for judging (defaults to --model)")
    parser.add_argument("--start-index", type=int, default=0, help="Skip the first N examples")
    parser.add_argument("--limit", type=int, default=0, help="Max examples to evaluate (0 = all)")
    parser.add_argument("--concurrency", type=int, default=10, help="Number of concurrent requests")
    parser.add_argument("--retries", type=int, default=5, help="Max retries on error/empty outputs")
    parser.add_argument("--initial-backoff", type=float, default=1.0, help="Initial backoff seconds")
    parser.add_argument("--max-backoff", type=float, default=30.0, help="Max backoff seconds")
    parser.add_argument(
        "--thinking-budget",
        type=int,
        default=8000,
        help="Thinking budget for answering (Gemini only; set negative to omit)",
    )
    parser.add_argument(
        "--judge-thinking-budget",
        type=int,
        default=8000,
        help="Thinking budget for judging (Gemini only; set negative to omit)",
    )
    parser.add_argument("--log-dir", default="logs/parallel", help="Base directory for JSONL logs")
    # Anthropic Bedrock-specific overrides
    parser.add_argument(
        "--anthropic-max-tokens",
        type=int,
        default=64000,
        help="Default max tokens for Anthropic Bedrock generations",
    )
    parser.add_argument(
        "--anthropic-thinking",
        action="store_true",
        help="Enable Anthropic Bedrock thinking mode",
    )
    parser.add_argument(
        "--anthropic-thinking-budget",
        type=int,
        default=40000,
        help="Budget tokens for Anthropic Bedrock thinking mode",
    )

    args = parser.parse_args()

    answer_provider = normalize_provider_name(args.provider)
    judge_provider = normalize_provider_name(args.judge_provider or args.provider)
    judge_model_name = args.judge_model or args.model

    answer_session = ModelSession(answer_provider, args.model)
    judge_session = ModelSession(judge_provider, judge_model_name)

    answer_thinking = None if args.thinking_budget is not None and args.thinking_budget < 0 else args.thinking_budget
    judge_thinking = None if args.judge_thinking_budget is not None and args.judge_thinking_budget < 0 else args.judge_thinking_budget

    dataset = load_dataset(args.input, args.start_index, args.limit)
    total = len(dataset)
    if total == 0:
        print("No examples to evaluate.")
        return

    log_path = build_log_path(args.log_dir, answer_provider, args.model)

    # Configure Anthropic Bedrock defaults from CLI flags
    if answer_session.provider == "anthropic-bedrock":
        answer_session.anthropic_max_tokens = int(max(1, args.anthropic_max_tokens))
        answer_session.anthropic_thinking_enabled = bool(args.anthropic_thinking)
        answer_session.anthropic_thinking_budget = int(max(0, args.anthropic_thinking_budget))
    if judge_session.provider == "anthropic-bedrock":
        judge_session.anthropic_max_tokens = int(max(1, args.anthropic_max_tokens))
        judge_session.anthropic_thinking_enabled = bool(args.anthropic_thinking)
        judge_session.anthropic_thinking_budget = int(max(0, args.anthropic_thinking_budget))

    write_jsonl(
        log_path,
        {
            "type": "run_settings",
            "input": args.input,
            "total_examples": total,
            "provider": answer_provider,
            "model": args.model,
            "judge_provider": judge_provider,
            "judge_model": judge_model_name,
            "concurrency": args.concurrency,
            "retries": args.retries,
            "initial_backoff": args.initial_backoff,
            "max_backoff": args.max_backoff,
            "answer_thinking_budget": answer_thinking,
            "judge_thinking_budget": judge_thinking,
        },
    )

    results: List[Dict[str, object]] = []
    with ThreadPoolExecutor(max_workers=max(1, args.concurrency)) as executor:
        futures = []
        for offset, example in enumerate(dataset):
            row_index = args.start_index + offset
            futures.append(
                executor.submit(
                    process_one,
                    row_index,
                    example,
                    answer_session,
                    judge_session,
                    answer_thinking,
                    judge_thinking,
                    args.retries,
                    args.initial_backoff,
                    args.max_backoff,
                )
            )

        for future in tqdm(as_completed(futures), total=len(futures), desc="Evaluating"):
            try:
                result = future.result()
            except Exception as exc:
                result = {
                    "row": None,
                    "question": "",
                    "gold_answer": "",
                    "model_answer": f"ERROR: {exc}",
                    "model_error": True,
                    "judge_label": "error",
                    "judge_reason": "exception",
                    "judge_raw": str(exc),
                    "exact_match": 0.0,
                    "f1": 0.0,
                }
            write_jsonl(
                log_path,
                {
                    **result,
                    "type": "result",
                    "answer_provider": answer_provider,
                    "answer_model": args.model,
                    "judge_provider": judge_provider,
                    "judge_model": judge_model_name,
                },
            )
            results.append(result)

    exact_count = sum(1 for r in results if r.get("judge_label") == "exact" or r.get("judge_exact_match") == 1)
    approx_count = sum(1 for r in results if r.get("judge_label") == "approximate" or r.get("judge_approximate_match") == 1)
    print(f"Completed {len(results)} examples. Exact: {exact_count}  |  Approx: {approx_count}")


if __name__ == "__main__":
    main()


