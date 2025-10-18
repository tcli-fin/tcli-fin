"""
LLM-as-a-Judge utilities.

Provides a provider-agnostic judge that labels a model's prediction as
exact, approximate, incorrect, or unknown and includes a brief reason.
"""

import json
import time
import logging
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

from .types import ModelConfig
from .models import ModelProviderFactory, BaseModelProvider, ModelResponse

logger = logging.getLogger(__name__)


@dataclass
class JudgeVerdict:
    label: str
    reason: str
    exact_match: int
    approximate_match: int


class LLMJudge:
    def __init__(self, judge_config: ModelConfig, max_retries: int = 3, retry_delay: float = 2.0):
        self.judge_config = judge_config
        self.provider: BaseModelProvider = ModelProviderFactory.create_provider(judge_config)
        self.max_retries = max_retries
        self.retry_delay = retry_delay

    def judge(self, *, context: str, question: str, gold: str, prediction: str) -> Dict[str, object]:
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
            "Labeling: label must be one of: exact, approximate, incorrect, unknown.\n"
            "- EXACT implies approximate_match=0.\n"
            "- APPROXIMATE implies exact_match=0.\n"
            "Return ONLY a JSON object with keys: label, reason, exact_match, approximate_match.\n"
            "- exact_match and approximate_match must be 0 or 1, not both 1.\n"
            "Do not include markdown/code fences or any extra text.\n\n"
        )

        user_prompt = (
            f"Question: {question}\n"
            f"Gold answer: {gold}\n"
            f"Candidate answer: {prediction}\n"
            "Judgement:"
        )

        messages = []
        # Use a system message when supported; providers that don't explicitly support it
        # will concatenate system+user internally (see provider implementations).
        messages.append({"role": "system", "content": base_instructions})
        messages.append({"role": "user", "content": user_prompt})

        # Retry loop for robustness against API failures and empty responses
        last_error = None
        response = None
        
        for attempt in range(self.max_retries):
            try:
                logger.info(f"Judge attempt {attempt + 1}/{self.max_retries}")
                response = self.provider.generate_with_retry(messages)
                
                # Check for errors
                if response.error:
                    last_error = response.error
                    logger.warning(f"Judge error on attempt {attempt + 1}: {response.error}")
                    if attempt < self.max_retries - 1:
                        delay = self.retry_delay * (2 ** attempt)
                        logger.info(f"Retrying judge in {delay}s...")
                        time.sleep(delay)
                        continue
                
                # Check for empty response
                raw_text = (response.text or "").strip()
                if not raw_text:
                    last_error = "empty_response"
                    logger.warning(f"Judge returned empty response on attempt {attempt + 1}")
                    if attempt < self.max_retries - 1:
                        delay = self.retry_delay * (2 ** attempt)
                        logger.info(f"Retrying judge in {delay}s...")
                        time.sleep(delay)
                        continue
                
                # Success - got a valid response
                break
                
            except Exception as e:
                last_error = str(e)
                logger.error(f"Judge exception on attempt {attempt + 1}: {e}")
                if attempt < self.max_retries - 1:
                    delay = self.retry_delay * (2 ** attempt)
                    logger.info(f"Retrying judge in {delay}s...")
                    time.sleep(delay)
                    continue
        
        # If all retries failed
        if response is None or (response.error and not response.text):
            return {
                "judge_label": "error",
                "judge_reason": f"judge_failed_after_{self.max_retries}_retries: {last_error}",
                "judge_raw": "",
                "judge_exact_match": 0,
                "judge_approximate_match": 0,
            }
        
        raw_text = response.text or ""
        
        if response.error and not raw_text:
            return {
                "judge_label": "error",
                "judge_reason": response.error,
                "judge_raw": raw_text or response.error,
                "judge_exact_match": 0,
                "judge_approximate_match": 0,
            }

        obj = None
        raw = raw_text.strip()
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
            try:
                verdict = JudgeVerdict(
                    label=(label_value if label_value in {"exact", "approximate", "incorrect", "unknown"} else "unknown"),
                    reason=reason_value,
                    exact_match=int(1 if int(exact_flag) else 0),
                    approximate_match=int(1 if int(approx_flag) else 0),
                )
                if verdict.exact_match == 1:
                    verdict.approximate_match = 0
                if verdict.approximate_match == 1:
                    verdict.exact_match = 0
                return {
                    "judge_label": verdict.label,
                    "judge_reason": verdict.reason,
                    "judge_raw": json.dumps({
                        "label": verdict.label,
                        "reason": verdict.reason,
                        "exact_match": verdict.exact_match,
                        "approximate_match": verdict.approximate_match,
                    }, sort_keys=True),
                    "judge_exact_match": verdict.exact_match,
                    "judge_approximate_match": verdict.approximate_match,
                }
            except Exception:
                pass

        if not raw:
            return {
                "judge_label": "unknown",
                "judge_reason": "judge_returned_empty",
                "judge_raw": raw,
                "judge_exact_match": 0,
                "judge_approximate_match": 0,
            }

        return {
            "judge_label": "unknown",
            "judge_reason": "judge_invalid_json",
            "judge_raw": raw,
            "judge_exact_match": 0,
            "judge_approximate_match": 0,
        }

