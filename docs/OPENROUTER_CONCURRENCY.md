# OpenRouter API Concurrency and Retry Implementation

## Overview

This document describes the concurrency and retry logic implementation for OpenRouter API models in the FinBench system.

## Implementation Summary

### 1. Concurrency Support

Concurrency is now fully implemented for all evaluation modes:

- **Direct Answer Mode** (`evaluation_mode: direct_answer`)
  - Uses `ThreadPoolExecutor` for parallel API calls
  - Configurable via `concurrency` parameter in experiment config
  - Example: `concurrency: 10` means up to 10 concurrent API calls

- **Program Execution Mode** (`evaluation_mode: program_execution`)
  - Same thread-based concurrency as direct answer mode
  - Parallel program generation and execution

- **Agent Mode** (`evaluation_mode: agent`)
  - Already supported concurrency (was implemented previously)

#### Configuration Example

```yaml
experiments:
  docfinqa_grok_full:
    datasets: [docfinqa_test_only]
    models: [openrouter-grok-fast-reasoning-high]
    evaluation_mode: direct_answer
    concurrency: 10  # Will run 10 samples in parallel
    judge_model: openrouter-qwq-32b-judge
```

### 2. Retry Logic

#### OpenRouter Provider Retry Features

1. **Automatic Retries**: 3 retries by default (4 total attempts)
2. **Exponential Backoff**: 
   - Initial delay: 1 second
   - Max delay: 30 seconds
   - Multiplier: 2x per attempt
   - Jitter: Random ±50% to prevent thundering herd

3. **Rate Limit Detection**: 
   - Automatically detects rate limit errors
   - Applies 2x delay multiplier for rate limits
   - Detects patterns: "rate limit", "429", "too many requests", etc.

4. **Enhanced Logging**:
   - Logs retry attempts with error details
   - Shows delay duration
   - Indicates success after retries

#### Retry Configuration

Default configuration in `runner.py`:

```python
retry_config = RetryConfig(
    max_retries=3,           # 3 retries (4 total attempts)
    initial_backoff=1.0,     # 1 second initial delay
    max_backoff=30.0,        # 30 seconds max delay
    backoff_multiplier=2.0,  # Exponential (2x per retry)
    jitter=True              # Add randomness to delays
)
```

#### Timeout Handling

- Default timeout: 120 seconds (2 minutes)
- Suitable for large context models like Grok (2M tokens)
- Configurable per request if needed

### 3. Error Handling

#### Detected Error Types

- **Rate Limits**: `rate limit`, `429`, `too many requests`, `quota exceeded`
- **OpenRouter Specific**: `openrouter rate limit`, `credits exhausted`, `insufficient_quota`
- **Connection Issues**: Timeout errors, connection failures
- **API Errors**: All other exceptions are caught and logged

#### Error Recovery Strategy

1. **First Attempt**: Normal API call
2. **If Error**: Check if rate limit or general error
3. **Retry with Backoff**: Wait exponentially longer
4. **Rate Limit Handling**: Extra 2x delay for rate limits
5. **Max Retries**: After 3 retries, return error to caller

### 4. Judge Model Concurrency

The judge model (OpenRouter QwQ-32B) also benefits from:
- Same retry logic as main model
- Independent retry loop (3 attempts)
- Exponential backoff
- Empty response detection and retry

### 5. Thread Safety

- Each thread gets its own API client instance
- No shared state between concurrent requests
- Results are collected in order (preserves sample ordering)

### 6. Progress Logging

During concurrent execution, you'll see:
```
INFO: Running concurrent direct evaluation with 10 workers
INFO: Completed 10/100 samples
INFO: Completed 20/100 samples
...
WARNING: OpenRouter rate limit on attempt 1/4: ... Retrying in 2.3s
INFO: OpenRouter API succeeded on attempt 2
```

## Best Practices

### 1. Concurrency Settings

- **Conservative**: `concurrency: 1-5` - For testing or limited credits
- **Moderate**: `concurrency: 10-20` - Balanced speed and safety
- **Aggressive**: `concurrency: 50+` - Maximum speed (watch rate limits!)

### 2. Rate Limit Management

OpenRouter has different rate limits per model. Monitor logs for rate limit warnings:

```
WARNING: OpenRouter rate limit on attempt 1/4: Rate limit exceeded. Retrying in 4.5s
```

If you see many rate limit errors, reduce `concurrency`.

### 3. Timeout Considerations

For very long contexts (like DocFinQA's 100k+ tokens), you may need longer timeouts:

```python
# In your code, pass custom timeout
response = provider.generate_with_retry(messages, timeout=300.0)  # 5 minutes
```

### 4. Cost Management

Concurrent execution speeds up evaluation but increases parallel API usage:
- Monitor OpenRouter dashboard for credit usage
- Start with lower concurrency for expensive models
- Use `limit` parameter to test on small samples first

## Example Usage

### Running with Concurrency

```bash
# Run experiment with 10 concurrent workers
python -m src.fin_bench.runner --experiment docfinqa_grok_full

# Or programmatically:
from src.fin_bench.runner import ExperimentRunner
from src.fin_bench.config import Config

config = Config("config.yaml")
runner = ExperimentRunner(config)
results = runner.run_experiment("docfinqa_grok_full")
```

### Monitoring Progress

Watch the logs for:
1. Concurrency level: "Running concurrent direct evaluation with X workers"
2. Progress updates: "Completed N/M samples"
3. Retry attempts: "OpenRouter API error on attempt X/4"
4. Rate limits: "OpenRouter rate limit... Retrying in Xs"

## Implementation Details

### Key Files Modified

1. **`src/fin_bench/runner.py`**:
   - Added `_evaluate_direct_concurrent()` 
   - Added `_evaluate_program_execution_concurrent()`
   - Uses `ThreadPoolExecutor` for parallel execution

2. **`src/fin_bench/models.py`**:
   - Enhanced `OpenRouterProvider.generate_with_retry()` with logging
   - Added timeout handling to `OpenRouterProvider.generate()`
   - Enhanced rate limit detection patterns
   - Better error messages with exception types

### Architecture

```
ExperimentRunner
  ├─> run_experiment()
  ├─> run_benchmark()
  └─> evaluate_samples()
       ├─> _evaluate_direct() [checks concurrency]
       │    ├─> _evaluate_direct_sequential() [concurrency <= 1]
       │    └─> _evaluate_direct_concurrent() [concurrency > 1]
       │         └─> ThreadPoolExecutor
       │              └─> Multiple threads call:
       │                   └─> provider.generate_with_retry()
       │                        └─> Retry loop with backoff
       └─> judge.judge() [per sample, also with retries]
```

## Troubleshooting

### Problem: Too Many Rate Limits

**Solution**: Reduce `concurrency` value in config

### Problem: Timeouts

**Solution**: Increase timeout in OpenRouterProvider.generate()

### Problem: Connection Errors

**Solution**: Check internet connection, verify API key is set

### Problem: Slow Progress

**Solution**: Increase `concurrency` (if not hitting rate limits)

## Configuration Reference

### Model Configuration

```yaml
openrouter-grok-fast-reasoning-high:
  provider: openrouter
  model_name: x-ai/grok-4-fast
  api_key_env: OPENROUTER_API_KEY
  max_tokens: 1000000
  temperature: 0.0
  provider_kwargs:
    reasoning:
      effort: "high"
```

### Experiment Configuration

```yaml
docfinqa_grok_full:
  datasets: [docfinqa_test_only]
  models: [openrouter-grok-fast-reasoning-high]
  evaluation_mode: direct_answer
  concurrency: 10                          # Concurrent workers
  judge_model: openrouter-qwq-32b-judge   # Judge also uses retries
```

## Performance Impact

### With Concurrency (concurrency: 10)
- **Speed**: ~10x faster than sequential
- **API Calls**: 10 parallel requests at once
- **Rate Limits**: Higher chance of hitting limits
- **Cost**: Same total cost, faster spending

### Sequential (concurrency: 1)
- **Speed**: Baseline (slowest)
- **API Calls**: One at a time
- **Rate Limits**: Minimal risk
- **Cost**: Same total cost, slower spending

## Summary

✅ **Concurrency**: Fully implemented for all evaluation modes
✅ **Retry Logic**: Robust with exponential backoff
✅ **Rate Limit Handling**: Automatic detection and extra delays
✅ **Timeout Handling**: 2-minute default for large contexts
✅ **Logging**: Comprehensive retry and progress logging
✅ **Thread Safety**: Each thread has isolated API client
✅ **Error Recovery**: Graceful handling of all error types

The system is now ready for high-performance concurrent evaluation with OpenRouter API models!

