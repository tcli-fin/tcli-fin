# Quick Start: OpenRouter Concurrency

## Your Configuration is Ready! ✅

Your `docfinqa_grok_full` experiment is properly configured for concurrent execution:

```yaml
docfinqa_grok_full:
  datasets: [docfinqa_test_only]
  models: [openrouter-grok-fast-reasoning-high]
  evaluation_mode: direct_answer
  concurrency: 10  # ✅ Will run 10 API calls in parallel
  judge_model: openrouter-qwq-32b-judge
```

## What Was Implemented

### ✅ Concurrency
- **10 parallel API calls** to OpenRouter
- Works for `direct_answer`, `program_execution`, and `agent` modes
- Thread-safe implementation using `ThreadPoolExecutor`

### ✅ Retry Logic
- **3 automatic retries** (4 total attempts) on failures
- **Exponential backoff**: 1s → 2s → 4s → 8s (with jitter)
- **Rate limit detection**: Extra delays when hitting limits
- **Timeout**: 120 seconds for large context models

### ✅ Logging
You'll see:
```
INFO: Running concurrent direct evaluation with 10 workers
INFO: Completed 10/50 samples
INFO: Completed 20/50 samples
WARNING: OpenRouter rate limit on attempt 1/4... Retrying in 2.3s
INFO: OpenRouter API succeeded on attempt 2
```

## How to Run

```bash
# Set your API key
export OPENROUTER_API_KEY="your-key-here"

# Run your experiment
python -m src.fin_bench.runner --experiment docfinqa_grok_full
```

## Monitor Progress

Watch for:
- ✅ Progress: "Completed X/Y samples"
- ⚠️  Retries: "Retrying in Xs"
- ❌ Rate limits: "rate limit on attempt"

## Adjust Concurrency

If you see too many rate limits:
```yaml
concurrency: 5  # Reduce from 10 to 5
```

If you want faster execution:
```yaml
concurrency: 20  # Increase from 10 to 20
```

## Complete Documentation

See `docs/OPENROUTER_CONCURRENCY.md` for full details.

---

**You're all set!** Your OpenRouter API calls will now run concurrently with automatic retry logic. 🚀

