# Quick Reference: Context for Each Benchmark

## One-Line Summary for Each

| Benchmark | What Agent Gets | Paper Used | We Use |
|-----------|----------------|------------|---------|
| **DocFinQA** | Full SEC filing (~100k tokens) | Retrieval (chunks) | ✅ Full filing |
| **FinQA** | Full page (pre+table+post) | BERT top-3 facts | ✅ Full page |
| **ConvFinQA** | Full page + conversation history | Retrieved facts | ✅ Full + history |
| **TAT-QA** | Table + paragraphs (hybrid) | IO tagging | ✅ Clean hybrid |
| **FinanceBench** | Full evidence pages | Vector store RAG | ✅ Oracle setup |
| **EconLogicQA** | 4 events to order | News article (gen) | ✅ Events only |
| **BizBench** | Varies by task (8 tasks) | Varies | ✅ Task-specific |
| **DocMath-Eval** | Full document | RAG (embeddings) | ✅ Full doc |

---

## Context Sizes

```
DocFinQA      ~330,000 chars (~100k tokens)  [LARGEST]
FinQA         ~3,500 chars                    
ConvFinQA     ~4,000 chars
TAT-QA        ~1,900 chars
FinanceBench  ~3,400 chars
EconLogicQA   ~0 chars (empty)               [SMALLEST]
BizBench      0 - 2,352 chars (varies)
DocMath-Eval  ~4,000 chars
```

---

## Retrieval Status

✅ **ALL: NO RETRIEVAL**

Every benchmark provides complete context directly.
No external retrieval systems needed.

---

## Quick Verification Commands

```bash
# Test all benchmarks load correctly
python3 << 'EOF'
from src.fin_bench.config import Config
from src.fin_bench.datasets import DatasetFactory

config = Config("config.yaml")

for name in ['docfinqa', 'finqa', 'convfinqa', 'tatqa', 
             'financebench', 'econlogicqa', 'bizbench', 'docmath_eval']:
    try:
        dataset = config.get_dataset(name)
        loader = DatasetFactory.create_loader(dataset)
        samples = loader.load_split(dataset.splits[0])
        print(f"✅ {name:15} {len(samples):6} samples")
    except Exception as e:
        print(f"❌ {name:15} {str(e)[:40]}")
EOF
```

---

## For Each Benchmark

### DocFinQA
- Context: **Full SEC filing**
- Size: ~100k tokens
- Special: Reference program included

### FinQA  
- Context: **Full page** (pre_text + table + post_text)
- Size: ~1k tokens
- Special: Ignores `qa.model_input` (pre-retrieved facts)

### ConvFinQA
- Context: **Full page + conversation history**
- Size: ~1-2k tokens
- Special: Multi-turn with Q&A pairs

### TAT-QA
- Context: **Table + paragraphs** (hybrid)
- Size: ~500 tokens
- Special: Clean extracted text

### FinanceBench
- Context: **Full evidence pages** (Oracle)
- Size: ~1k tokens
- Special: Not using vector store

### EconLogicQA
- Context: **Empty** (events in options)
- Size: 0
- Special: Logic-based ordering

### BizBench
- Context: **Varies by task** (8 different tasks)
- Size: 0-2k tokens
- Special: Multi-task benchmark

### DocMath-Eval
- Context: **Full document** (long-context)
- Size: ~1k tokens
- Special: Not using RAG

---

## Ready to Use

```python
# Quick start
from src.fin_bench.runner import ExperimentRunner
from src.fin_bench.config import Config

config = Config("config.yaml")
runner = ExperimentRunner(config)

# Run on any benchmark
results = runner.run_experiment("agent_comparison", limit=10)
print(f"Accuracy: {results.accuracy:.3f}")
```

---

**🎉 All 8 benchmarks ready for pure coding agent evaluation!**




