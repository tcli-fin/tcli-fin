#!/usr/bin/env python3
"""Quick DocFinQA spot-check with Gemini answer + judge passes.

Loads examples from a DocFinQA JSON split, asks Gemini for an answer, and
then has Gemini act as a judge to determine whether the prediction matches
the gold answer. Lexical EM/F1 scores are still reported for reference.
"""

import argparse
import json
import os
import re
import sys
import urllib.error
import urllib.request
from datetime import datetime
from typing import Dict, Iterable, Tuple, Optional
try:
    from typing import Literal  # type: ignore
except Exception:
    Literal = str  # type: ignore

try:
    from pydantic import BaseModel, ValidationError  # type: ignore
except Exception:  # pragma: no cover - optional until requirements updated
    BaseModel = object  # type: ignore
    class ValidationError(Exception):  # type: ignore
        pass


def normalize_answer(text: str) -> str:
    if text is None:
        return ""
    text = text.strip()
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    if (text.startswith('"') and text.endswith('"')) or (text.startswith("'") and text.endswith("'")):
        text = text[1:-1].strip()
    return text


def compute_em_f1(prediction: str, ground_truth: str) -> Tuple[float, float]:
    p = normalize_answer(prediction)
    g = normalize_answer(ground_truth)
    em = 1.0 if p == g and g != "" else 0.0
    p_tokens = p.split()
    g_tokens = g.split()
    if not p_tokens and not g_tokens:
        return em, 1.0
    if not p_tokens or not g_tokens:
        return em, 0.0
    common: Dict[str, int] = {}
    for token in g_tokens:
        common[token] = common.get(token, 0) + 1
    overlap = 0
    for token in p_tokens:
        if common.get(token, 0) > 0:
            overlap += 1
            common[token] -= 1
    if overlap == 0:
        return em, 0.0
    precision = overlap / len(p_tokens)
    recall = overlap / len(g_tokens)
    f1 = 2 * precision * recall / (precision + recall)
    return em, f1


def parse_final_numeric_answer(text: str) -> str:
    """Extract the final numeric answer from a chain-of-thought style output.

    Strategy:
    1) Look for a sentence like: "Therefore, the answer is { ... }" (case-insensitive).
       - If braces found, take the content inside braces
       - Otherwise, take the remainder of the sentence and extract the first numeric token
    2) Fallback: take the last numeric token anywhere in the text

    Numeric tokens: optional sign, digits with optional commas and decimals, optional trailing %
    """
    if not text:
        return ""
    raw = str(text)
    lower = raw.lower()

    # Primary: find the "therefore, the answer is ..." segment
    import re as _re
    anchor = _re.search(r"therefore\s*,?\s*the\s+answer\s+is\s*(.*)$", lower, flags=_re.IGNORECASE | _re.MULTILINE)
    segment = None
    if anchor:
        # Recover the same slice from the original text to preserve symbols/casing
        start = anchor.start(1)
        segment = raw[start:]

    # Helper to extract first numeric from a string
    num_re = _re.compile(r"[-+]?\d{1,3}(?:,\d{3})*(?:\.\d+)?%?|[-+]?\d+(?:\.\d+)?%?")

    if segment:
        # If braces present, prefer inside the first {...}
        brace = _re.search(r"\{([^}]+)\}", segment)
        if brace:
            candidate = brace.group(1).strip()
            # If braces contain a number-like substring, extract it, else return raw inside braces
            m = num_re.search(candidate)
            return (m.group(0).strip() if m else candidate)
        # Else extract first numeric token from the segment
        m = num_re.search(segment)
        if m:
            return m.group(0).strip()

    # Fallback: last numeric token anywhere
    matches = list(num_re.finditer(raw))
    if matches:
        return matches[-1].group(0).strip()
    return ""


def read_examples(path: str) -> Iterable[Dict]:
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, list):
        raise ValueError("Expected a JSON array of DocFinQA examples")
    for obj in data:
        yield obj


def load_gemini_model(model_name: str):
    try:
        import google.generativeai as genai
    except ImportError as exc:
        print("Install google-generativeai via requirements.txt before running.", file=sys.stderr)
        raise SystemExit(2) from exc

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("GEMINI_API_KEY is not set. Export your Gemini API key first.", file=sys.stderr)
        raise SystemExit(2)

    genai.configure(api_key=api_key)
    return genai.GenerativeModel(model_name)


class ModelSession:
    def __init__(self, provider: str, model_name: str):
        provider = provider.lower()
        # Support both "openrouter" and the alias "openorder" (typo-friendly)
        if provider == "openrouter":
            provider = "openrouter"
        if provider not in {"gemini", "openrouter", "openai", "anthropic-bedrock"}:
            raise ValueError(f"Unsupported provider '{provider}'")
        self.provider = provider
        self.model_name = model_name
        # Default output cap for OpenAI/OpenRouter chat completions
        # Set to 32768 to match gpt-4.1 completion token limit.
        # Kept as an attribute to allow future CLI override if needed.
        self.openai_max_tokens: int = 32768

        if provider == "gemini":
            self.client = load_gemini_model(model_name)
            self.api_key: Optional[str] = None
        elif provider == "openai":
            try:
                from openai import OpenAI  # type: ignore
            except ImportError as exc:
                print("Install openai via requirements.txt before running.", file=sys.stderr)
                raise SystemExit(2) from exc
            api_key = os.environ.get("OPENAI_API_KEY")
            if not api_key:
                print("OPENAI_API_KEY is not set. Export your OpenAI API key first.", file=sys.stderr)
                raise SystemExit(2)
            self.client = OpenAI(api_key=api_key)
            self.api_key = None
        elif provider == "openrouter":
            try:
                from openai import OpenAI  # type: ignore
            except ImportError as exc:
                print("Install openai via requirements.txt before running.", file=sys.stderr)
                raise SystemExit(2) from exc
            api_key = os.environ.get("OPENROUTER_API_KEY")
            if not api_key:
                print(
                    "OPENROUTER_API_KEY is not set. Export your OpenRouter API key first.",
                    file=sys.stderr,
                )
                raise SystemExit(2)
            # Use OpenAI SDK against OpenRouter's OpenAI-compatible endpoint
            self.client = OpenAI(api_key=api_key, base_url="https://openrouter.ai/api/v1")
            self.api_key = None
        elif provider == "anthropic-bedrock":
            try:
                from anthropic import AnthropicBedrock  # type: ignore
            except ImportError as exc:
                print('Install "anthropic[bedrock]" via requirements.txt before running.', file=sys.stderr)
                raise SystemExit(2) from exc
            region = os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION")
            try:
                # Set a large timeout to avoid the SDK's non-streaming long-request guard
                # per docs: passing stream=True OR increasing timeout disables the error.
                # We do both (streaming in callers and higher timeout here) for robustness.
                if region:
                    self.client = AnthropicBedrock(aws_region=region, timeout=3600.0)
                else:
                    self.client = AnthropicBedrock(timeout=3600.0)
            except Exception as exc:
                print(f"Failed to create AnthropicBedrock client: {exc}", file=sys.stderr)
                raise SystemExit(2)
            self.api_key = None
        # Anthropic Bedrock defaults (used only when provider == "anthropic-bedrock")
        self.anthropic_max_tokens: int = 64000
        self.anthropic_thinking_enabled: bool = True
        self.anthropic_thinking_budget: int = 40000


class JudgeVerdict(BaseModel):
    # label must be one of: exact, approximate, incorrect, unknown
    label: str
    reason: str
    exact_match: int
    approximate_match: int


def call_gemini(model, system_prompt: Optional[str], user_prompt: str, thinking_budget: Optional[int]) -> str:
    generation_config = {
        "temperature": 0.0,
    }
    if thinking_budget is not None:
        generation_config["thinking_config"] = {"thinking_budget": thinking_budget}
    try:
        # Gemini doesn't require a formal system role; prepend if provided
        full_prompt = (
            (f"System instructions:\n{system_prompt}\n\n" if system_prompt else "")
            + user_prompt
        )
        response = model.generate_content(
            full_prompt,
            generation_config=generation_config,
        )
    except Exception as exc:
        return f"ERROR: {exc}"

    # Safely extract text without using response.text accessor (which can raise
    # when no valid Part is returned, e.g., safety block or empty output).
    try:
        candidates = getattr(response, "candidates", None) or []
        for candidate in candidates:
            content = getattr(candidate, "content", None)
            parts = getattr(content, "parts", None) if content is not None else None
            if parts:
                for part in parts:
                    text = getattr(part, "text", None)
                    if text:
                        return str(text).strip()
        # If we get here, there were no text parts. Surface finish_reason when available.
        if candidates:
            finish_reason = getattr(candidates[0], "finish_reason", None)
            return f"ERROR: empty_response finish_reason={finish_reason}"
    except Exception:
        # Fall through to empty string if parsing fails unexpectedly
        pass
    return ""


def call_openai_style_chat(client, model_name: str, system_prompt: Optional[str], user_prompt: str) -> str:
    try:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_prompt})
        response = client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=0.0,
        )
    except Exception as exc:
        return f"ERROR: {exc}"
    try:
        if getattr(response, "choices", None):
            message = response.choices[0].message
            content = getattr(message, "content", None)
            if content:
                return str(content).strip()
    except Exception:
        pass
    return ""


def _is_claude_4_model(model_id: str) -> bool:
    name = (model_id or "").lower()
    # Match common Claude 4 family identifiers across Bedrock model IDs
    return (
        "sonnet-4" in name
        or "haiku-4" in name
        or "opus-4" in name
        or "claude-4" in name
    )


def call_anthropic_bedrock(
    bedrock_client,
    model_id: str,
    system_prompt: Optional[str],
    user_prompt: str,
    max_tokens: int,
    thinking_enabled: bool,
    thinking_budget: int,
) -> str:
    # Use content blocks per Bedrock/Anthropic schema
    messages = [{"role": "user", "content": [{"type": "text", "text": user_prompt}]}]
    temperature_value = 1.0 if (thinking_enabled and thinking_budget and thinking_budget > 0) else 0.0

    # Prefer streaming to avoid the SDK's 10-minute cap on long requests
    try:
        extra_body = None
        if _is_claude_4_model(model_id):
            # Enable 1M context beta for Claude 4 family on Bedrock
            extra_body = {
                "anthropic_version": "bedrock-2023-05-31",
                "anthropic_beta": ["context-1m-2025-08-07"],
            }

        stream_kwargs = {
            "model": model_id,
            "max_tokens": max_tokens,
            "temperature": temperature_value,
            "messages": messages,
            "thinking": (
                {"type": "enabled", "budget_tokens": int(thinking_budget)}
                if thinking_enabled and thinking_budget and thinking_budget > 0
                else {"type": "disabled"}
            ),
        }
        if system_prompt:
            stream_kwargs["system"] = system_prompt
        if extra_body is not None:
            stream_kwargs["extra_body"] = extra_body

        texts: list[str] = []
        with bedrock_client.messages.stream(**stream_kwargs) as stream:
            for text in getattr(stream, "text_stream", []):
                if text:
                    texts.append(str(text))
            try:
                final_msg = stream.get_final_message()
            except Exception:
                final_msg = None

        if texts:
            return " ".join(t.strip() for t in texts if t).strip()

        if final_msg is not None:
            try:
                parts = getattr(final_msg, "content", None) or []
                fallback_texts: list[str] = []
                for part in parts:
                    t = getattr(part, "text", None)
                    if t:
                        fallback_texts.append(str(t))
                    elif isinstance(part, dict) and part.get("text"):
                        fallback_texts.append(str(part["text"]))
                if fallback_texts:
                    return " ".join(s.strip() for s in fallback_texts if s).strip()
            except Exception:
                pass
    except Exception:
        # Fallback 1: event iteration via create(stream=True)
        try:
            extra_body = None
            if _is_claude_4_model(model_id):
                extra_body = {
                    "anthropic_version": "bedrock-2023-05-31",
                    "anthropic_beta": ["context-1m-2025-08-07"],
                }
            create_stream_kwargs = {
                "model": model_id,
                "max_tokens": max_tokens,
                "temperature": temperature_value,
                "messages": messages,
                "thinking": (
                    {"type": "enabled", "budget_tokens": int(thinking_budget)}
                    if thinking_enabled and thinking_budget and thinking_budget > 0
                    else {"type": "disabled"}
                ),
                "stream": True,
            }
            if system_prompt:
                create_stream_kwargs["system"] = system_prompt
            if extra_body is not None:
                create_stream_kwargs["extra_body"] = extra_body

            event_stream = bedrock_client.messages.create(**create_stream_kwargs)
            texts: list[str] = []
            for event in event_stream:
                typ = getattr(event, "type", None)
                if typ is None and isinstance(event, dict):
                    typ = event.get("type")
                if typ == "content_block_delta":
                    delta = getattr(event, "delta", None)
                    if delta is None and isinstance(event, dict):
                        delta = event.get("delta")
                    piece = getattr(delta, "text", None)
                    if piece is None and isinstance(delta, dict):
                        piece = delta.get("text")
                    if piece:
                        texts.append(str(piece))
            if texts:
                return " ".join(t.strip() for t in texts if t).strip()
        except Exception:
            # Fallback 2: non-streaming create
            try:
                extra_body = None
                if _is_claude_4_model(model_id):
                    extra_body = {
                        "anthropic_version": "bedrock-2023-05-31",
                        "anthropic_beta": ["context-1m-2025-08-07"],
                    }
                create_kwargs = {
                    "model": model_id,
                    "max_tokens": max_tokens,
                    "temperature": temperature_value,
                    "messages": messages,
                    "thinking": (
                        {"type": "enabled", "budget_tokens": int(thinking_budget)}
                        if thinking_enabled and thinking_budget and thinking_budget > 0
                        else {"type": "disabled"}
                    ),
                }
                if system_prompt:
                    create_kwargs["system"] = system_prompt
                if extra_body is not None:
                    create_kwargs["extra_body"] = extra_body
                message = bedrock_client.messages.create(**create_kwargs)

                try:
                    parts = getattr(message, "content", None) or []
                    texts: list[str] = []
                    for part in parts:
                        text = getattr(part, "text", None)
                        if text:
                            texts.append(str(text))
                        elif isinstance(part, dict) and part.get("text"):
                            texts.append(str(part["text"]))
                    if texts:
                        return " ".join(t.strip() for t in texts if t).strip()
                except Exception:
                    pass
            except Exception as exc:
                return f"ERROR: {exc}"

    return ""


def call_model(
    session: ModelSession,
    system_prompt: Optional[str],
    user_prompt: str,
    thinking_budget: Optional[int],
) -> str:
    if session.provider == "gemini":
        return call_gemini(session.client, system_prompt, user_prompt, thinking_budget)
    if session.provider == "openrouter" or session.provider == "openai":
        return call_openai_style_chat(client=session.client, model_name=session.model_name, system_prompt=system_prompt, user_prompt=user_prompt)
    if session.provider == "anthropic-bedrock":
        return call_anthropic_bedrock(
            bedrock_client=session.client,
            model_id=session.model_name,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            max_tokens=session.anthropic_max_tokens,
            thinking_enabled=session.anthropic_thinking_enabled,
            thinking_budget=session.anthropic_thinking_budget,
        )
    return f"ERROR: unsupported provider {session.provider}"


def generate_answer(
    session: ModelSession,
    context: str,
    question: str,
    thinking_budget: Optional[int],
) -> str:
    system_prompt = (
        "You are a financial expert. Answer the question based on the provided financial document context."
        " First think step by step, documenting each necessary step."
        " Then conclude your response with the final numeric answer in your last sentence as:"
        " Therefore, the answer is { final answer }."
        " The final answer must be a numeric value."
    )
    user_prompt = (
        f"Context:\n{context}\n\n"
        f"Question: {question}\n"
        "Answer:"
    )
    full_output = call_model(session, system_prompt, user_prompt, thinking_budget)
    return full_output


def judge_answer(
    session: ModelSession,
    context: str,
    question: str,
    gold: str,
    prediction: str,
    thinking_budget: Optional[int],
) -> Tuple[str, str, str]:
    base_instructions = (
        "You are an evaluation assistant for DocFinQA.\n"
        "Treat the provided gold answer as authoritative ground truth.\n"
        "Normalization rules for EXACT match:\n"
        "- Trim whitespace, ignore case, remove thousands separators, ignore trailing zeros (13 == 13.0).\n"
        "- Percent formats are equivalent when numerically equal (33.3% == 33.30%).\n"
        "- Fractions equal to decimals are equivalent (33 1/3% ~= 33.33%).\n"
        "- Numeric tolerance: if both are percentages, abs diff <= 0.05 pp is exact; if decimals (no %), abs diff <= 0.01 is exact.\n"
        "Rules for APPROXIMATE match (set only when not exact):\n"
        "- Logically equivalent phrasing/sign/units (e.g., '-35' vs 'decrease 35 million'; '35 million decrease' vs '-35,000,000').\n"
        "- Minor unit wording differences with same magnitude/sign (e.g., 'million' vs 'm').\n"
        "- Rounding beyond exact tolerance but clearly same value (briefly justify).\n"
        "Labeling:\n"
        "- label must be one of: exact, approximate, incorrect, unknown.\n"
        "- EXACT implies approximate_match=0.\n"
        "- APPROXIMATE implies exact_match=0.\n"
        "Return ONLY a JSON object with keys: label, reason, exact_match, approximate_match.\n"
        "- exact_match and approximate_match must be 0 or 1, not both 1.\n"
        "Do not include markdown/code fences or any extra text.\n\n"
    )

    attempts = 2
    last_text = ""
    for _ in range(attempts):
        prompt = (
            base_instructions
            + f"Question: {question}\n"
            + f"Gold answer: {gold}\n"
            + f"Candidate answer: {prediction}\n"
            + "Judgement:"
        )

        text = call_model(session, None, prompt, thinking_budget)
        last_text = text

        if text.startswith("ERROR:"):
            continue
        if not text:
            continue

        raw = text.strip()

        # Try strict JSON parse, with substring fallback
        obj = None
        try:
            obj = json.loads(raw)
        except Exception:
            try:
                start = raw.find("{")
                end = raw.rfind("}")
                if start != -1 and end != -1 and end > start:
                    obj = json.loads(raw[start : end + 1])
            except Exception:
                obj = None

        if isinstance(obj, dict):
            label_value = str(obj.get("label", "")).strip().lower()
            reason_value = str(obj.get("reason", "")).strip()
            exact_flag = obj.get("exact_match", 1 if label_value == "exact" else 0)
            approx_flag = obj.get("approximate_match", 1 if label_value == "approximate" else 0)

            if label_value not in {"exact", "approximate", "incorrect", "unknown"}:
                label_value = "unknown"
            try:
                verdict = JudgeVerdict(
                    label=label_value,
                    reason=reason_value,
                    exact_match=int(1 if int(exact_flag) else 0),
                    approximate_match=int(1 if int(approx_flag) else 0),
                )
                # Enforce mutual exclusivity
                if verdict.exact_match == 1:
                    verdict.approximate_match = 0
                if verdict.approximate_match == 1:
                    verdict.exact_match = 0
                return verdict.label, verdict.reason, json.dumps(verdict.model_dump(), sort_keys=True)
            except Exception:
                # validation failed, retry
                continue

    # If we reach here, structured JSON failed; return error/unknown
    if last_text.startswith("ERROR:"):
        return "error", last_text, last_text
    if not last_text:
        return "unknown", "judge_returned_empty", last_text
    return "unknown", "judge_invalid_json", last_text


def open_log_file(path: Optional[str]):
    if path is None:
        return None
    log_path = path.strip()
    if not log_path:
        return None
    directory = os.path.dirname(log_path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    return open(log_path, "a", encoding="utf-8")


def write_log(fp, payload: Dict[str, object]) -> None:
    if fp is None:
        return
    payload = dict(payload)
    payload.setdefault("timestamp", datetime.utcnow().isoformat() + "Z")
    fp.write(json.dumps(payload, sort_keys=True) + "\n")
    fp.flush()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a minimal DocFinQA answer+judge loop")
    parser.add_argument("--input", default="DocFinQA/test.json", help="Path to DocFinQA split JSON")
    parser.add_argument(
        "--provider",
        default="gemini",
        choices=["gemini", "openrouter", "openorder", "openai", "anthropic-bedrock"],
        help="Model provider for answering (supports: gemini, openrouter|openorder, openai, anthropic-bedrock)",
    )
    parser.add_argument("--model", default="gemini-2.5-pro", help="Model name for answering")
    parser.add_argument(
        "--judge-provider",
        choices=["gemini", "openrouter", "openorder", "openai", "anthropic-bedrock"],
        help="Model provider for judging (defaults to --provider)",
    )
    parser.add_argument("--judge-model", default=None, help="Model name for judging (defaults to --model)")
    parser.add_argument("--start-index", type=int, default=0, help="Skip the first N examples")
    parser.add_argument("--limit", type=int, default=1, help="Number of examples to evaluate (0 = all)")
    parser.add_argument(
        "--thinking-budget",
        type=int,
        default=8000,
        help="Thinking budget for answer generation (Gemini only; set negative to omit)",
    )
    parser.add_argument(
        "--judge-thinking-budget",
        type=int,
        default=8000,
        help="Thinking budget for judging (Gemini only; set negative to omit)",
    )
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
        default=4000,
        help="Budget tokens for Anthropic Bedrock thinking mode",
    )
    parser.add_argument(
        "--log-file",
        default="logs/simple_gemini_eval.jsonl",
        help="Path to append JSONL logs (set to empty string to disable)",
    )
    args = parser.parse_args()

    answer_session = ModelSession(args.provider, args.model)
    judge_provider = args.judge_provider or args.provider
    judge_model_name = args.judge_model or args.model
    judge_session = ModelSession(judge_provider, judge_model_name)

    answer_thinking = None if args.thinking_budget is not None and args.thinking_budget < 0 else args.thinking_budget
    judge_thinking = None if args.judge_thinking_budget is not None and args.judge_thinking_budget < 0 else args.judge_thinking_budget

    log_fp = open_log_file(args.log_file)

    # Configure Anthropic Bedrock defaults from CLI flags
    if answer_session.provider == "anthropic-bedrock":
        answer_session.anthropic_max_tokens = int(max(1, args.anthropic_max_tokens))
        answer_session.anthropic_thinking_enabled = bool(args.anthropic_thinking)
        answer_session.anthropic_thinking_budget = int(max(0, args.anthropic_thinking_budget))
    if judge_session.provider == "anthropic-bedrock":
        judge_session.anthropic_max_tokens = int(max(1, args.anthropic_max_tokens))
        judge_session.anthropic_thinking_enabled = bool(args.anthropic_thinking)
        judge_session.anthropic_thinking_budget = int(max(0, args.anthropic_thinking_budget))

    processed = 0
    try:
        for idx, example in enumerate(read_examples(args.input)):
            if idx < args.start_index:
                continue
            if args.limit and processed >= args.limit:
                break

            context = example.get("Context", "")
            question = example.get("Question", "")
            gold = example.get("Answer", "")

            prediction = generate_answer(answer_session, context, question, answer_thinking)
            # Save full output and parsed answer
            model_full_output = prediction
            try:
                parsed = parse_final_numeric_answer(model_full_output)
            except Exception:
                parsed = ""
            display_answer = parsed or model_full_output
            em, f1 = compute_em_f1(display_answer, gold)
            label, reason, judge_raw = judge_answer(
                judge_session,
                context,
                question,
                gold,
                model_full_output,
                judge_thinking,
            )

            print("-" * 60)
            print(f"Row: {idx}")
            print(f"Question: {question}")
            print(f"Gold answer: {gold}")
            print(f"Model answer (parsed): {display_answer}")
            print(f"Model full output: {model_full_output}")
            print(f"LLM judge: {label}{' — ' + reason if reason else ''}")
            print(f"Exact match: {'yes' if em == 1.0 else 'no'}  |  F1: {f1:.2f}")

            write_log(
                log_fp,
                {
                    "row": idx,
                    "question": question,
                    "gold_answer": gold,
                    "model_answer": display_answer,
                    "model_full_output": model_full_output,
                    "model_error": prediction.startswith("ERROR:"),
                    "answer_provider": answer_session.provider,
                    "answer_model": answer_session.model_name,
                    "answer_thinking_budget": answer_thinking,
                    "judge_label": label,
                    "judge_reason": reason,
                    "judge_raw": judge_raw,
                    "judge_provider": judge_session.provider,
                    "judge_model": judge_session.model_name,
                    "judge_thinking_budget": judge_thinking,
                    "exact_match": em,
                    "f1": f1,
                },
            )

            processed += 1
    finally:
        if log_fp is not None:
            log_fp.close()

    if processed == 0:
        print("No examples evaluated. Adjust --start-index/--limit settings.")


if __name__ == "__main__":
    main()
