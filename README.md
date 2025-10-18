# FinBench: Unified Financial QA Benchmarking System

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A comprehensive, modular system for evaluating financial question-answering models across multiple benchmarks with consistent metrics and parallel execution.

## 🎯 **Overview**

FinBench transforms your existing financial QA evaluation scripts into a unified, research-grade benchmarking system. It provides:

- **🔥 Multi-dataset support**: FinQA, TAT-QA, DocFinQA, DocMath-Eval
- **🤖 Multi-provider models**: Gemini, OpenAI, Anthropic, OpenRouter
- **⚡ Parallel execution**: Concurrent evaluation with retry logic
- **🏗️ Modular architecture**: Clean separation of concerns
- **🔬 Research-oriented**: Designed for systematic experimentation

## 🚀 **Quick Start**

### 1. Installation
```bash
# Install dependencies
pip install -r requirements.txt

# Set up data and configuration
python tools/setup_data.py
python finbench.py config create config.yaml
```

### 2. Configure API Keys
```bash
export GEMINI_API_KEY="your_gemini_key"
export OPENAI_API_KEY="your_openai_key"
# AWS credentials for Anthropic Bedrock (if using)
export AWS_REGION="us-east-1"
```

### 3. Run Benchmarks
```bash
# Quick validation test
python finbench.py run quick_test --limit 5

# Run comprehensive evaluation
python finbench.py run comprehensive_eval

# List available configurations
python finbench.py list experiments
```

## 📊 **Supported Benchmarks**

| Benchmark | Type | Evaluation Mode | Metrics |
|-----------|------|-----------------|---------|
| **FinQA** | Numerical reasoning | Program execution | Exact match, program execution |
| **TAT-QA** | Table/text hybrid | Direct answer | Exact match, F1, numeric tolerance |
| **DocFinQA** | Long-context | Retrieval-augmented | Exact match, program execution |
| **DocMath-Eval** | Mathematical reasoning | Program execution | Exact match, program execution |

## 🏗️ **Repository Structure**

```
fin_code/
├── src/fin_bench/           # 🏗️ Core modular system
│   ├── __init__.py         # Package exports
│   ├── types.py            # Type definitions & enums
│   ├── config.py           # Configuration management
│   ├── datasets.py         # Dataset loaders (all benchmarks)
│   ├── models.py           # Model providers (all APIs)
│   ├── metrics.py          # Evaluation metrics
│   ├── runner.py           # Experiment orchestration
│   └── cli.py              # Command-line interface
├── benchmarks/              # 📊 Benchmark-specific files
│   ├── adapters/           # Dataset adapters
│   ├── metrics/            # Benchmark-specific metrics
│   ├── benchmark_docfinqa.py
│   ├── eval_docfinqa.py
│   ├── simple_gemini_eval.py
│   └── parallel_eval.py
├── tools/                  # 🛠️ Utility scripts
│   ├── setup_data.py
│   ├── create_sample_data.py
│   ├── download_financebench_pdfs.py
│   ├── validate_benchmarks.py
│   └── scripts/
├── examples/               # 💡 Usage examples
│   └── quick_start.py
├── data/                   # 📁 Dataset files
├── logs/                   # 📋 Log files
├── results/                # 📈 Result files
├── docs/                   # 📚 Documentation
├── finbench.py             # 🚀 Main entry point
├── config.yaml             # ⚙️ Configuration file
├── requirements.txt        # 📦 Dependencies
└── README.md               # 📖 This file
```

## ⚙️ **Configuration**

### Datasets
Each dataset is configured with:
- **Type**: Dataset format (docfinqa, finqa, tatqa, financebench)
- **Path**: Location of data files
- **Splits**: Available data splits
- **Fields**: Available sample fields (context, program, answer)

### Models
Models support multiple providers:
- **Gemini**: Google Gemini models
- **OpenAI**: GPT models
- **Anthropic**: Claude via Bedrock
- **OpenRouter**: Multi-provider access

### Experiments
Experiments define:
- **Datasets**: Which datasets to evaluate on
- **Models**: Which models to use
- **Mode**: Evaluation approach (direct, program execution, retrieval)
- **Concurrency**: Parallel execution settings

## 🔧 **Usage Examples**

### Basic Evaluation
```bash
# Run a quick test
python finbench.py run quick_test --limit 10

# Run specific dataset with specific model
python finbench.py run finqa_study

# Run with async execution
python finbench.py run comprehensive_eval --async --concurrency 15
```

### Validation & Information
```bash
# Validate system setup
python tools/validate_benchmarks.py

# List available configurations
python finbench.py list datasets
python finbench.py list models
python finbench.py list experiments

# Show system information
python finbench.py info
```

### Configuration Management
```bash
# Create default configuration
python finbench.py config create my_config.yaml

# Validate configuration
python finbench.py config validate config.yaml
```

## 📈 **Sample Output**

### Console Output
```
🚀 FinBench: Starting Experiment
==================================================
📊 Experiment Results
==================================================
Experiment: comprehensive_eval
Execution Time: 125.4 seconds
Total Benchmarks: 12
Total Samples: 240
Overall Accuracy: 0.742

📈 Per-Benchmark Results:
------------------------------
docfinqa + gemini-pro:
  Accuracy: 0.850 (17/20)
  Errors: 0
finqa + gemini-pro:
  Accuracy: 0.700 (14/20)
  Errors: 1
...

🏆 Aggregate Metrics:
  Average Exact Match: 0.742
  Average F1: 0.689

💾 Results saved to: results/results_20240920_143052.json
```

### Results File Structure
```json
{
  "experiment_config": {
    "name": "comprehensive_eval",
    "datasets": [...],
    "models": [...],
    "evaluation_mode": "direct_answer"
  },
  "results": [
    {
      "dataset_name": "docfinqa",
      "model_name": "gemini-pro",
      "metrics": {
        "exact_match": 0.85,
        "f1": 0.82
      },
      "total_samples": 20,
      "processed_samples": 20,
      "errors": 0,
      "execution_time": 45.2,
      "detailed_results": [...]
    }
  ],
  "summary": {
    "total_benchmarks": 12,
    "execution_time": 125.4
  }
}
```

## 🔬 **Research Features**

### Experiment Design
- **Modular experiments**: Define complex evaluation scenarios
- **Parameter sweeps**: Systematic variation of model/dataset combinations
- **Multi-metric evaluation**: Comprehensive performance assessment

### Extensibility
- **Custom datasets**: Add new financial QA datasets
- **Custom models**: Integrate new model providers
- **Custom metrics**: Define domain-specific evaluation metrics

### Reproducibility
- **Configuration files**: Complete experiment specification
- **Detailed logging**: Execution traces and debugging info
- **Result persistence**: Structured output for analysis

## 🗂️ **Directory Structure Guide**

### Core System (`src/fin_bench/`)
The core FinBench system with modular architecture:
- `types.py` - Type definitions and enums
- `config.py` - Configuration management
- `datasets.py` - Dataset loaders for all benchmarks
- `models.py` - Model providers (Gemini, OpenAI, Anthropic)
- `metrics.py` - Evaluation metrics
- `runner.py` - Experiment orchestration
- `cli.py` - Command-line interface

### Benchmarks (`benchmarks/`)
Benchmark-specific evaluation scripts:
- Individual benchmark evaluators
- Dataset adapters for different formats
- Benchmark-specific metrics

### Tools (`tools/`)
Utility scripts for setup and maintenance:
- `setup_data.py` - Download and organize datasets
- `validate_benchmarks.py` - Validate system setup
- `create_sample_data.py` - Generate sample data
- `download_financebench_pdfs.py` - Download FinanceBench PDFs

### Examples (`examples/`)
Usage examples and tutorials:
- `quick_start.py` - Basic usage example
- Tutorial scripts

## 🎛️ **Advanced Usage**

### Custom Configuration
```yaml
# config.yaml
experiments:
  my_research:
    datasets: [docfinqa, finqa]
    models: [gemini-pro, gpt4o]
    evaluation_mode: program_execution
    concurrency: 15
    limit: 100
    start_index: 50
```

### Programmatic Usage
```python
from src.fin_bench.runner import ExperimentRunner
from src.fin_bench.config import Config

# Load configuration
config = Config("config.yaml")
runner = ExperimentRunner(config)

# Run experiment
results = runner.run_experiment("my_research")

# Analyze results
summary = results.get_aggregate_metrics()
print(f"Overall accuracy: {summary['accuracy']:.3f}")
```

### Integration with Existing Scripts
Your existing evaluation scripts continue to work unchanged:
```bash
# Your original scripts still work
python benchmark_docfinqa.py --split test --limit 10
python eval_docfinqa.py --input DocFinQA/test.json --model gemini-2.5-pro
```

## 🆚 **Comparison with Existing System**

| Feature | Old System | FinBench |
|---------|------------|----------|
| **Multi-dataset** | Manual coordination | Unified interface |
| **Multi-model** | Provider-specific scripts | Provider abstraction |
| **Parallel execution** | Limited | Built-in concurrency |
| **Configuration** | Hardcoded parameters | YAML configuration |
| **Extensibility** | Script modification | Module system |
| **Research features** | Basic | Comprehensive |

## 🔍 **Migration Guide**

### From benchmark_docfinqa.py
```bash
# Old way
python benchmark_docfinqa.py --split test --limit 10

# New way
python finbench.py run docfinqa_study --limit 10
```

### From eval_docfinqa.py
```bash
# Old way
python eval_docfinqa.py --input DocFinQA/test.json --model gemini-2.5-pro

# New way
python finbench.py run quick_test

# Or use the moved file directly
python benchmarks/eval_docfinqa.py --input data/DocFinQA/test.json --model gemini-2.5-pro
```

### From setup_data.py
```bash
# Old way
python setup_data.py

# New way
python tools/setup_data.py
```

## 🛠️ **Development**

### Adding New Datasets
1. Create loader in `src/fin_bench/datasets.py`
2. Add dataset type to `types.py`
3. Update configuration schema
4. Add to `config.yaml`

### Adding New Models
1. Create provider in `src/fin_bench/models.py`
2. Add provider type to `types.py`
3. Update factory method
4. Add to `config.yaml`

### Adding New Metrics
1. Create metric in `src/fin_bench/metrics.py`
2. Add to factory methods
3. Update default metric selection

## 📝 **API Reference**

### Main Classes
- `ExperimentRunner`: Orchestrates benchmark execution
- `Config`: Manages system configuration
- `Dataset`: Dataset specification
- `ModelConfig`: Model configuration
- `BenchmarkResult`: Individual benchmark results

### Command Line Interface
- `finbench run <experiment>`: Run experiment
- `finbench list <type>`: List configurations
- `finbench config <command>`: Configuration management
- `finbench validate`: Validate setup
- `finbench info`: System information

## 🎉 **Benefits**

1. **🔥 Unified Interface**: Single command for all benchmarks
2. **🔬 Research Quality**: Designed for systematic experimentation
3. **📈 Extensible**: Easy to add new datasets, models, metrics
4. **🔄 Reproducible**: Configuration-driven experiments
5. **⚡ Scalable**: Parallel execution and efficient data handling
6. **🧹 Maintainable**: Clean, modular architecture
7. **📁 Well-Organized**: Logical file structure for easy navigation

## ✅ **Repository Restructuring Complete**

This repository has been restructured for better organization and maintainability:

### 🗂️ **Before vs After**

**Before (Scattered files):**
- Multiple files at root level
- Mixed utility scripts and core system files
- Benchmark-specific files mixed with system files
- No clear separation of concerns

**After (Clean structure):**
- **Root level**: Only essential system files
- **Core system**: Modular architecture in `src/fin_bench/`
- **Benchmark tools**: Organized in `benchmarks/`
- **Utility scripts**: Centralized in `tools/`
- **Examples**: Usage examples in `examples/`
- **Data**: Well-organized dataset files

### 📊 **Restructuring Summary**

✅ **Files moved to `tools/`:**
- `setup_data.py` - Dataset download and setup
- `validate_benchmarks.py` - System validation
- `create_sample_data.py` - Sample data generation
- `download_financebench_pdfs.py` - PDF download utility

✅ **Files moved to `benchmarks/`:**
- `eval_docfinqa.py` - DocFinQA evaluation
- `simple_gemini_eval.py` - Simple evaluation script
- `parallel_eval.py` - Parallel evaluation
- `adapters/` - Dataset adapters
- `metrics/` - Benchmark-specific metrics

✅ **New directories created:**
- `examples/` - Usage examples and tutorials
- `docs/` - Documentation files

✅ **Cleaned up:**
- Duplicate files removed
- `__pycache__` directories cleaned
- Better path handling for imports

### 🧪 **Verified Working**
- ✅ All system tests pass
- ✅ All tools work from new locations
- ✅ Main CLI functions correctly
- ✅ Configuration loading works
- ✅ Dataset loading successful

## 🛠️ **Setup and Tools**

### Complete Setup Workflow

#### **Step 1: Environment Setup**
```bash
# 1. Create virtual environment (recommended)
python3 -m venv finbench-env

# 2. Activate virtual environment
source finbench-env/bin/activate

# 3. Install dependencies
pip install -r requirements.txt
```

#### **Step 2: Dataset Setup (Two-Phase Approach)**

##### **Phase 1: Download from Official Repositories**
```bash
# Download datasets from official GitHub repositories
bash tools/scripts/fetch_all.sh
```
This downloads: FinQA and TAT-QA, and creates symlinks to data directories.

##### **Phase 2: Download from Hugging Face (Requires Authentication)**
```bash
# Set your Hugging Face token (get from https://huggingface.co/settings/tokens)
export HF_TOKEN="your_huggingface_token_here"

# Download Hugging Face datasets
python tools/setup_data.py
```
This downloads: DocFinQA and DocMath-Eval (requires approval)

#### **Step 3: Validation**
```bash
# Validate that all datasets are properly set up
python tools/validate_benchmarks.py
```

#### **Step 4: API Keys Setup**
```bash
# Set API keys for models you want to use
export GEMINI_API_KEY="your_gemini_api_key"
export OPENAI_API_KEY="your_openai_api_key"
# AWS credentials for Anthropic Bedrock (if using)
export AWS_REGION="us-east-1"
```

#### **Step 5: Test Setup**
```bash
# Run a quick validation test
python examples/quick_start.py

# Or run a small benchmark test
python finbench.py run quick_test --limit 5
```

### Available Tools

#### **Dataset Setup Tools**

##### `tools/scripts/fetch_all.sh`
Downloads datasets from official GitHub repositories:
```bash
bash tools/scripts/fetch_all.sh
```
**What it downloads:**
- FinQA (from czyssrs/FinQA)
- TAT-QA (from NExTplusplus/TAT-QA)
- Creates symlinks in `data/` directory

##### `tools/setup_data.py`
Downloads datasets from Hugging Face Hub:
```bash
export HF_TOKEN="your_token"
python tools/setup_data.py
```
**What it downloads:**
- DocFinQA (from kensho/DocFinQA)
- DocMath-Eval (from yale-nlp/DocMath-Eval) - requires approval

#### **Validation Tools**

##### `tools/validate_benchmarks.py`
Validates system setup and data integrity:
```bash
python tools/validate_benchmarks.py
```
**Checks:**
- Dataset file existence
- Data structure validity
- Sample loading tests
- Reports success/failure per dataset

##### `tools/create_sample_data.py`
Creates minimal sample datasets for testing:
```bash
python tools/create_sample_data.py
```

#### **Troubleshooting Common Issues**

##### **Hugging Face Authentication Errors**
```bash
# If you get "Invalid credentials" or "401 Unauthorized":
# 1. Get a new token from https://huggingface.co/settings/tokens
# 2. Set it in your environment:
export HF_TOKEN="hf_your_new_token_here"

# 3. Or use it directly:
HF_TOKEN="hf_your_token" python tools/setup_data.py
```

##### **Dataset Download Failures**
```bash
# If datasets fail to download:
# 1. Check your internet connection
# 2. Verify Hugging Face token is valid
# 3. Try downloading manually:
python -c "from datasets import load_dataset; ds = load_dataset('dataset_name'); ds.save_to_disk('data/dataset_name')"
```

##### **DocMath-Eval Access Issues**
- DocMath-Eval requires special approval from Yale NLP Lab
- Visit https://huggingface.co/datasets/yale-nlp/DocMath-Eval
- Contact: Yilun Zhao (yilun.zhao@yale.edu) for access

##### **Validation Failures**
```bash
# If validation shows missing datasets:
# 1. Run fetch_all.sh first
# 2. Then run setup_data.py
# 3. Copy any missing files manually if needed
```

#### **Setup Completion Checklist**
✅ **4 Financial QA Datasets Ready:**
- **DocFinQA**: Long-form financial QA with retrieval
- **FinQA**: Numerical reasoning over financial data
- **TAT-QA**: Table and text hybrid QA
- **DocMath-Eval**: Mathematical reasoning (requires approval)

✅ **Multi-Model Support:**
- Google Gemini (gemini-pro, gemini-flash)
- OpenAI GPT (gpt-4o, gpt-4o-mini)
- Anthropic Claude (via Bedrock)

✅ **Ready-to-Run Experiments:**
- Quick validation tests
- Comprehensive evaluations
- Model comparison studies
- Domain-specific experiments

### Benchmark-Specific Tools
Located in `benchmarks/` directory:
- `benchmarks/eval_docfinqa.py` - DocFinQA evaluation
- `benchmarks/simple_gemini_eval.py` - Simple evaluation with Gemini
- `benchmarks/parallel_eval.py` - Parallel evaluation script

## 📋 **Installation Requirements**

- Python 3.8+
- Dependencies listed in `requirements.txt`
- API keys for model providers (optional, depending on usage)

## 🤝 **Contributing**

This system is designed to be extended:

1. **Datasets**: Add new financial QA datasets in `src/fin_bench/datasets.py`
2. **Models**: Integrate new providers in `src/fin_bench/models.py`
3. **Metrics**: Define custom evaluation metrics in `src/fin_bench/metrics.py`
4. **Experiments**: Create new evaluation scenarios in `config.yaml`

## 📄 **License**

MIT License - see LICENSE file for details.

## 🙏 **Acknowledgments**

Built on top of existing financial QA benchmarks:
- [FinQA](https://github.com/czyssrs/FinQA)
- [TAT-QA](https://github.com/NExTplusplus/TAT-QA)
- [DocFinQA](https://huggingface.co/datasets/kensho/DocFinQA)
- [DocMath-Eval](https://huggingface.co/datasets/yale-nlp/DocMath-Eval)

---

This system transforms your existing evaluation scripts into a comprehensive research platform while maintaining full backward compatibility.

## 🧪 LLM-as-a-Judge (OpenRouter)

- Add a judge model under `models:` in `config.yaml` (this repo includes `openrouter-qwq-32b-judge`).
- Export `OPENROUTER_API_KEY` in your shell:
  - `export OPENROUTER_API_KEY="..."`
- Enable per-experiment by adding `judge_model: openrouter-qwq-32b-judge` to the experiment block.
- During runs, each sample's `detailed_results` includes `judge_label`, `judge_reason`, `judge_raw`, `judge_exact_match`, and `judge_approximate_match`.
- Aggregated `metrics` include `judge_exact` and `judge_approximate` rates when available.
