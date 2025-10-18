# Claude Code + DocFinQA Configuration Changes

## Summary of Changes

### ✅ What Was Fixed:

1. **Added `--dangerously-skip-permissions` flag** to Claude Code CLI command
   - Required for automated usage in trusted workspaces
   - Allows Claude Code to read `context.md` and other files without interactive permissions

2. **Disabled the numerical extractor** for Claude Code
   - Added `provider_kwargs: {disable_extractor: true}` to `claude-code-sonnet` config
   - Now sends the full Claude response to the judge, not just extracted numbers

3. **Added JSON parsing for Claude Code output**
   - Created `_parse_json_result()` method to extract the "result" field from Claude's JSON response
   - Sets `judge_input` metadata field with the parsed result
   - Ensures the LLM judge receives Claude's full answer text, not raw JSON

## Complete Data Flow:

```
1. DocFinQA sample loaded → SEC filing text extracted
2. context.md created with full filing text (100k+ tokens)
3. Claude Code CLI executed:
   Command: claude --output-format json --dangerously-skip-permissions -p <question>
   Working Dir: temp/claude_workspace/<session_id>/
   
4. Claude Code JSON response:
   {
     "type": "result",
     "subtype": "success", 
     "result": "**Answer: 42%**\n\nCalculation: ...",
     "num_turns": 14,
     "duration_ms": 150152
   }

5. Response processing:
   - Full JSON stored in metadata["full_stdout"]
   - "result" field extracted and set as metadata["judge_input"]
   - response.text = parsed result (clean answer text)

6. LLM Judge evaluation:
   - Receives parsed result from judge_input
   - Compares against gold answer
   - Returns: {label, reason, exact_match, approximate_match}

7. Final scoring:
   - Exact match accuracy
   - Approximate match accuracy
   - Judge metrics aggregated
```

## Configuration Files Modified:

### config.yaml
```yaml
claude-code-sonnet:
  provider: agent
  agent_name: claude-code
  model_name: null
  cli_command: claude
  cli_args: ["--output-format", "json"]
  workspace_dir: temp/claude_workspace
  timeout: 300
  provider_kwargs:
    disable_extractor: true  # ← NEW: Send full response to judge
```

### src/fin_bench/models.py
```python
class ClaudeCodeProvider(AgentProvider):
    def _parse_json_result(self, stdout: str) -> str:
        """Extract 'result' field from Claude Code JSON output."""
        # NEW: Parse JSON and extract result field
        
    def generate(self, messages: List[Dict[str, str]], **kwargs) -> ModelResponse:
        """Override to set judge_input with parsed result."""
        # NEW: Set judge_input metadata for LLM judge
        
    def _execute_cli_command(self, prompt: str, workspace: Path):
        # UPDATED: Added --dangerously-skip-permissions flag
```

## What the Judge Receives:

### Before (with extractor enabled):
```
Judge Input: "42"
```

### After (extractor disabled, JSON parsed):
```
Judge Input: "**Answer: 42%**

Based on the facilities data in the SEC filing:
- Leased facilities: 8.5 million sq ft
- Total facilities: 20.2 million sq ft
- Percentage: (8.5 / 20.2) × 100 = 42%"
```

## Benefits:

1. ✅ **Better Judge Accuracy**: Judge sees full reasoning, not just extracted numbers
2. ✅ **More Context**: Judge can evaluate calculation steps and logic
3. ✅ **Handles Edge Cases**: Works even when answer isn't a simple number
4. ✅ **Proper Permissions**: Claude Code has full file access in workspace
5. ✅ **Clean JSON Parsing**: Extracts meaningful content from structured output

## Testing:

Run with 1 sample to verify:
```bash
source finbench-env/bin/activate
export OPENROUTER_API_KEY="your_key"
PYTHONPATH=src python -m fin_bench.cli run docfinqa_claude_code_test --limit 1
```

Expected behavior:
- Claude Code receives context.md with SEC filing
- Answers question with full explanation
- Judge receives full answer text (not JSON)
- Proper scoring with reasoning

