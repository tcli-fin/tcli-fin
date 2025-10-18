# Claude Code CLI + DocFinQA Evaluation Guide

## ✅ Your Setup Status

- **Claude Code CLI**: ✅ Installed (v2.0.10)
- **Virtual Environment**: ✅ `finbench-env`
- **DocFinQA Test Set**: ✅ Available at `DocFinQA/test.json` (554 MB)
- **Configuration**: ✅ Ready in `config.yaml`

## 📋 Available Experiments

### 1. `docfinqa_claude_code_test` (with OpenRouter LLM Judge)
Full evaluation with LLM-as-a-judge scoring using OpenRouter's QwQ-32B model.

**Requirements:**
- `OPENROUTER_API_KEY` environment variable must be set

**Run command:**
```bash
source finbench-env/bin/activate
export OPENROUTER_API_KEY="your_key_here"
PYTHONPATH=src python -m fin_bench.cli run docfinqa_claude_code_test --limit 10
```

### 2. `docfinqa_claude_code_test_nojudge` (no judge - faster)
Quick testing without LLM judge, only basic metrics.

**Run command:**
```bash
source finbench-env/bin/activate
PYTHONPATH=src python -m fin_bench.cli run docfinqa_claude_code_test_nojudge --limit 10
```

## 🚀 Quick Start

### Test with 1 sample (no judge):
```bash
source finbench-env/bin/activate
PYTHONPATH=src python -m fin_bench.cli run docfinqa_claude_code_test_nojudge --limit 1
```

### Test with 5 samples (with OpenRouter judge):
```bash
source finbench-env/bin/activate
export OPENROUTER_API_KEY="your_key_here"
PYTHONPATH=src python -m fin_bench.cli run docfinqa_claude_code_test --limit 5
```

### Run full test set (all samples):
```bash
source finbench-env/bin/activate
export OPENROUTER_API_KEY="your_key_here"
PYTHONPATH=src python -m fin_bench.cli run docfinqa_claude_code_test
```

## 🔧 Useful Options

### Keep workspace for debugging:
```bash
PYTHONPATH=src python -m fin_bench.cli run docfinqa_claude_code_test_nojudge --limit 1 --keep-workspace
```
This preserves the temporary workspace folders created for Claude Code CLI so you can inspect them.

### Print context files:
```bash
PYTHONPATH=src python -m fin_bench.cli run docfinqa_claude_code_test_nojudge --limit 1 --print-context
```
This shows the context.md file content that's provided to Claude Code CLI.

### Custom workspace directory:
```bash
PYTHONPATH=src python -m fin_bench.cli run docfinqa_claude_code_test_nojudge --limit 1 --workspace-dir ./my_workspaces
```

## 📊 Results

Results are automatically saved to the `results/` directory with timestamp:
- `results/results_YYYYMMDD_HHMMSS.json`

## 🎯 Model Configuration

Your current Claude Code setup (`claude-code-sonnet` in config.yaml):
```yaml
claude-code-sonnet:
  provider: agent
  agent_name: claude-code
  model_name: null  # Uses default Claude model
  cli_command: claude
  cli_args: ["--output-format", "json"]
  workspace_dir: temp/claude_workspace
  timeout: 300
```

## 💡 Tips

1. **Start small**: Test with `--limit 1` or `--limit 5` first
2. **Monitor progress**: The CLI shows live progress and errors
3. **Check workspaces**: Use `--keep-workspace` to debug if Claude Code fails
4. **API costs**: OpenRouter judge adds cost per evaluation. Use `_nojudge` for testing
5. **Timeout**: Default is 300 seconds per sample. Adjust in config.yaml if needed

## 🔍 What Gets Evaluated?

Each DocFinQA sample contains:
- **Context**: Full SEC filing text (very long, ~100k tokens)
- **Question**: Financial question about the filing
- **Gold Answer**: Expected numerical answer

Claude Code CLI receives:
- A `context.md` file with the full filing text
- A prompt asking the question
- Access to read files, write code, and execute Python

The system evaluates:
- Numerical accuracy of the answer
- LLM judge assessment (if using judge experiment)

## ⚙️ How It Works

1. **Dataset Loading**: System loads DocFinQA test samples
2. **Workspace Creation**: Temporary workspace created for each sample
3. **Context File**: Full SEC filing written to `context.md`
4. **Claude Code Execution**: CLI runs with the question prompt
5. **Answer Extraction**: Numerical answer extracted from Claude's response
6. **Evaluation**: Answer compared to gold standard
7. **Judge (optional)**: LLM judge evaluates the quality
8. **Results Saved**: JSON results file created in `results/`

## 📝 Example Output

```
🚀 FinBench: Starting Experiment
==================================================

📊 Experiment Results
==================================================
Experiment: docfinqa_claude_code_test
Execution Time: 125.45 seconds
Total Benchmarks: 1
Total Samples: 5
Overall Accuracy: 0.800

📈 Per-Benchmark Results:
------------------------------
docfinqa_test_only + claude-code-sonnet:
  Accuracy: 0.800 (4/5)
  Judge: exact=0.600 approx=0.200

💾 Results saved to: results/results_20251008_225640.json
```

## 🆘 Troubleshooting

### "OPENROUTER_API_KEY not set"
→ Run: `export OPENROUTER_API_KEY="your_key"`

### "ModuleNotFoundError: No module named 'fin_bench'"
→ Make sure to use: `PYTHONPATH=src python -m fin_bench.cli ...`

### "Command timed out after 300 seconds"
→ Increase timeout in config.yaml or use `--limit 1` to test individual samples

### Claude Code CLI fails
→ Use `--keep-workspace` and check the temp workspace folders
→ Use `--print-context` to verify context.md is being created properly

## 🎓 Next Steps

Once you verify it works with a few samples, you can:
1. Run the full test set
2. Compare with other models (Gemini, GPT-4, etc.)
3. Analyze results in the JSON files
4. Customize prompts or evaluation metrics

