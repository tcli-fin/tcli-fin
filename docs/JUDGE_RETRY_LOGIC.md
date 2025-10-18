# LLM Judge Retry Logic

## Overview

The LLM judge now includes robust retry logic to handle API failures, timeouts, and empty responses.

## Problem Statement

In the initial runs, we observed a **40% judge failure rate** (2 out of 5 samples) with errors like:
```json
"judge_label": "unknown",
"judge_reason": "judge_returned_empty",
"judge_raw": ""
```

This was caused by:
- OpenRouter API timeouts
- Empty responses from the QwQ-32B model
- Network intermittent issues
- No retry mechanism

## Solution: Retry with Exponential Backoff

### Configuration

```python
LLMJudge(
    judge_config: ModelConfig,
    max_retries: int = 3,         # Default: 3 retries
    retry_delay: float = 2.0      # Initial delay: 2 seconds
)
```

### Retry Logic Flow

```
Attempt 1: Judge call → Error/Empty? → Wait 2s → Retry
Attempt 2: Judge call → Error/Empty? → Wait 4s → Retry  
Attempt 3: Judge call → Error/Empty? → Wait 8s → Final attempt
```

Exponential backoff formula: `delay = retry_delay * (2 ** attempt)`

### What Triggers a Retry?

1. **API Errors**: Network errors, HTTP errors, timeouts
2. **Empty Responses**: Model returns empty string
3. **Exceptions**: Any Python exception during judge call

### What Does NOT Trigger a Retry?

1. **Valid JSON response** (even if it's "incorrect" or "unknown")
2. **Invalid JSON** but non-empty response (returns "judge_invalid_json")

## Implementation Details

### Code Structure

```python
def judge(self, *, context: str, question: str, gold: str, prediction: str):
    # Build prompt (unchanged)
    messages = [system_prompt, user_prompt]
    
    # Retry loop
    for attempt in range(self.max_retries):
        try:
            response = self.provider.generate_with_retry(messages)
            
            # Check for errors
            if response.error:
                if attempt < max_retries - 1:
                    sleep(exponential_backoff)
                    continue
            
            # Check for empty response
            if not response.text.strip():
                if attempt < max_retries - 1:
                    sleep(exponential_backoff)
                    continue
            
            # Success - exit retry loop
            break
            
        except Exception as e:
            if attempt < max_retries - 1:
                sleep(exponential_backoff)
                continue
            else:
                return error_result
    
    # Parse and return result
    return parse_judge_response(response)
```

### Error Responses

#### After all retries exhausted:

**Empty Response:**
```json
{
  "judge_label": "unknown",
  "judge_reason": "judge_returned_empty_after_3_retries",
  "judge_raw": "",
  "judge_exact_match": 0,
  "judge_approximate_match": 0
}
```

**Exception:**
```json
{
  "judge_label": "error",
  "judge_reason": "judge_exception_after_3_retries: <error_message>",
  "judge_raw": "",
  "judge_exact_match": 0,
  "judge_approximate_match": 0
}
```

## Logging

The judge now logs retry attempts for debugging:

```
INFO: Judge attempt 1/3
WARNING: Judge returned empty response on attempt 1
INFO: Retrying judge in 2.0s...
INFO: Judge attempt 2/3
WARNING: Judge error on attempt 2: Connection timeout
INFO: Retrying judge in 4.0s...
INFO: Judge attempt 3/3
```

## Benefits

1. ✅ **Increased Reliability**: Transient failures are automatically retried
2. ✅ **Better Error Messages**: Clear indication of retry attempts in error messages
3. ✅ **Exponential Backoff**: Reduces load on API during high traffic
4. ✅ **Configurable**: Can adjust retries and delays per use case
5. ✅ **No Prompt Changes**: Judge prompt remains identical

## Usage

### Default (3 retries):
```python
judge = LLMJudge(judge_config)
result = judge.judge(context=ctx, question=q, gold=g, prediction=p)
```

### Custom retry settings:
```python
judge = LLMJudge(
    judge_config,
    max_retries=5,      # More retries for flaky APIs
    retry_delay=1.0     # Shorter initial delay
)
```

### No retries (testing):
```python
judge = LLMJudge(judge_config, max_retries=1)  # Only 1 attempt, no retries
```

## Performance Impact

| Metric | Before | After (with retries) |
|--------|--------|---------------------|
| Success Rate | 60% (3/5) | ~95%+ expected |
| Avg Latency | ~5s | ~5-15s (with retries) |
| API Calls | 1 per judge | 1-3 per judge |
| Cost Impact | Baseline | +0-200% (on retries only) |

## Testing

Test the retry logic:

```bash
# Run with judge to see retry behavior
PYTHONPATH=src python -m fin_bench.cli run docfinqa_claude_code_test --limit 5
```

Monitor logs for retry messages to verify it's working.

## Future Improvements

1. **Adaptive backoff**: Adjust delays based on error type
2. **Circuit breaker**: Stop retrying if API is consistently failing
3. **Metrics tracking**: Log retry success/failure rates
4. **Alternative judge**: Fallback to different judge model if one fails

