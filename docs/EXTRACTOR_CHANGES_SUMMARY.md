# Extractor Disabled by Default - Summary

## ✅ Changes Made

### 1. **AgentProvider Base Class** (`src/fin_bench/models.py`)
Changed the default behavior from **extractor enabled** to **extractor disabled**.

**Before:**
```python
disable_extractor = False  # Default enabled
try:
    disable_extractor = bool((self.agent_config.provider_kwargs or {}).get("disable_extractor", False))
except Exception:
    disable_extractor = False
```

**After:**
```python
disable_extractor = True  # Default disabled
try:
    # Check if explicitly enabled in config (enable_extractor: true)
    enable_extractor = bool((self.agent_config.provider_kwargs or {}).get("enable_extractor", False))
    disable_extractor = not enable_extractor
except Exception:
    disable_extractor = True  # Default to disabled on error
```

### 2. **Configuration File Cleanup** (`config.yaml`)
Removed all `provider_kwargs: {disable_extractor: true}` entries since it's now the default.

**Cleaned up agents:**
- `claude-code-sonnet`
- `gemini-cli-pro`
- `gemini-cli-flash`

## How It Works Now

### Default Behavior (No Config):
```yaml
claude-code-sonnet:
  provider: agent
  agent_name: claude-code
  # ... other settings ...
  # No provider_kwargs needed - extractor disabled by default
```

**Result:** Agent sends full response text to judge, not just extracted numbers.

### Opt-in to Enable Extractor:
```yaml
some-agent:
  provider: agent
  # ... other settings ...
  provider_kwargs:
    enable_extractor: true  # Explicitly enable if you want numerical extraction
```

**Result:** Agent extracts numerical answers from response before sending to judge.

## Impact on All Agents

| Agent | Old Behavior | New Behavior |
|-------|--------------|--------------|
| `claude-code-sonnet` | Extract numbers | Full response text ✅ |
| `claude-code-haiku` | Extract numbers | Full response text ✅ |
| `gemini-cli-pro` | Extract numbers | Full response text ✅ |
| `gemini-cli-flash` | Extract numbers | Full response text ✅ |
| `codex-cli-default` | Extract numbers | Full response text ✅ |
| `trae-agent-*` | Extract numbers | Full response text ✅ |

## What the Judge Receives

### Before (with extractor):
```
Prediction: "42"
```

### After (without extractor):
```
Prediction: "**Answer: 42%**

Based on the facilities data in the SEC filing:
- Leased facilities: 8.5 million sq ft
- Total facilities: 20.2 million sq ft
- Percentage: (8.5 / 20.2) × 100 = 42%"
```

## Agent-Specific Parsing

Each agent provider can still customize how they process their output:

- **ClaudeCodeProvider**: Parses JSON and extracts "result" field
- **GeminiCLIProvider**: Parses JSON and extracts "response" field
- **CodexCLIProvider**: Extracts agent_message from JSON lines
- **TraeAgentProvider**: Extracts execution summary section

All of these set `metadata["judge_input"]` which is used by the judge.

## Benefits

1. ✅ **Better Judge Accuracy**: Judge sees reasoning, not just numbers
2. ✅ **Simpler Config**: No need to specify `disable_extractor` everywhere
3. ✅ **Consistent Behavior**: All agents work the same way by default
4. ✅ **Easy Override**: Can still enable extractor if needed with `enable_extractor: true`
5. ✅ **Agent-Specific Logic**: Each agent can parse its output format appropriately

## Migration Guide

If you were relying on the numerical extractor:

**Old config:**
```yaml
my-agent:
  provider: agent
  # ... settings ...
  # Extractor was enabled by default
```

**New config (to keep old behavior):**
```yaml
my-agent:
  provider: agent
  # ... settings ...
  provider_kwargs:
    enable_extractor: true  # Explicitly enable
```

## Testing

All existing experiments will now use full response text by default:

```bash
# All these now send full responses to judge:
PYTHONPATH=src python -m fin_bench.cli run docfinqa_claude_code_test --limit 1
PYTHONPATH=src python -m fin_bench.cli run docfinqa_gemini_cli_test_judge_openrouter --limit 1
```

No changes needed to commands - the behavior automatically improved!

