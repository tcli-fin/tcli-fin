#!/bin/bash
# scripts/fetch_all.sh
# Download and setup all financial benchmarks

set -euo pipefail

echo "=== Setting up Financial Benchmarks ==="

# Create directories
mkdir -p external data metrics adapters scripts
mkdir -p data/finqa data/convfinqa data/tatqa data/financebench data/financebench_pdfs data/secqa data/docfinqa
mkdir -p metrics/finqa metrics/tatqa metrics/secqa

echo "1. Setting up FinQA..."
if [ ! -d "external/FinQA" ]; then
    git clone https://github.com/czyssrs/FinQA.git external/FinQA
fi
ln -sf "$(pwd)/external/FinQA/dataset" data/finqa
cp external/FinQA/code/evaluate/evaluate.py metrics/finqa/
echo "✓ FinQA setup complete"

echo "2. Setting up ConvFinQA..."
if [ ! -d "external/ConvFinQA" ]; then
    git clone https://github.com/czyssrs/ConvFinQA.git external/ConvFinQA
fi
mkdir -p data/convfinqa
unzip -n external/ConvFinQA/data.zip -d data/convfinqa 2>/dev/null || echo "ConvFinQA data already extracted"
echo "✓ ConvFinQA setup complete"

echo "3. Setting up TAT-QA..."
if [ ! -d "external/TAT-QA" ]; then
    git clone https://github.com/NExTplusplus/TAT-QA.git external/TAT-QA
fi
ln -sf "$(pwd)/external/TAT-QA/dataset_raw" data/tatqa
cp external/TAT-QA/tatqa_eval.py metrics/tatqa/
echo "✓ TAT-QA setup complete"

echo "4. Setting up FinanceBench..."
if [ ! -d "external/financebench" ]; then
    git clone https://github.com/patronus-ai/financebench.git external/financebench
fi
ln -sf "$(pwd)/external/financebench/data" data/financebench
ln -sf "$(pwd)/external/financebench/pdfs" data/financebench_pdfs
echo "✓ FinanceBench setup complete"

echo "5. Setting up DocFinQA..."
if [ ! -d "data/docfinqa" ]; then
    echo "DocFinQA requires manual download via HuggingFace datasets:"
    echo "  python -c \"from datasets import load_dataset; ds = load_dataset('kensho/DocFinQA'); ds.save_to_disk('data/docfinqa')\""
fi
echo "✓ DocFinQA setup instructions provided"

echo "6. Setting up SEC-QA framework..."
mkdir -p data/secqa
if [ ! -f "data/secqa/sample_tasks.json" ]; then
    echo "Creating sample SEC-QA tasks..."
    cat > data/secqa/sample_tasks.json << 'EOF'
[
  {
    "task_id": "single_revenue_2022",
    "category": "single",
    "question": "What was the total revenue for Apple Inc. in 2022?",
    "answer": "394328000000",
    "confidence": 1.0,
    "source_docs": ["AAPL_2022_10K"],
    "answer_type": "numeric"
  },
  {
    "task_id": "compound_growth_rate",
    "category": "compound",
    "question": "What was the year-over-year revenue growth rate from 2021 to 2022?",
    "answer": "0.078",
    "confidence": 0.9,
    "source_docs": ["AAPL_2021_10K", "AAPL_2022_10K"],
    "answer_type": "numeric"
  }
]
EOF
fi
echo "✓ SEC-QA framework setup complete"

echo "7. Setting up adapters..."
# Create __init__.py files to make adapters a proper package
touch adapters/__init__.py

echo "=== Setup Complete! ==="
echo ""
echo "Directory structure created:"
echo "  external/     - Git repositories"
echo "  data/         - Dataset symlinks and files"
echo "  metrics/      - Evaluation scripts"
echo "  adapters/     - Format conversion utilities"
echo "  scripts/      - Helper scripts"
echo ""
echo "Next steps:"
echo "1. Install dependencies: pip install -r requirements.txt"
echo "2. Download DocFinQA: python -c \"from datasets import load_dataset; ds = load_dataset('kensho/DocFinQA'); ds.save_to_disk('data/docfinqa')\""
echo "3. Run evaluations using the appropriate adapters and metrics"
echo ""
echo "Example evaluation commands:"
echo "  # FinQA"
echo "  python metrics/finqa/evaluate.py runs/finqa/preds.json data/finqa/test.json"
echo "  # TAT-QA"
echo "  python metrics/tatqa/tatqa_eval.py --gold_path data/tatqa/tatqa_dataset_dev.json --pred_path runs/tatqa/preds.json"
echo "  # DocFinQA (using existing scripts)"
echo "  python benchmark_docfinqa.py --split test --limit 10"
echo "  # SEC-QA"
echo "  python metrics/secqa/evaluate_secqa.py --pred_path runs/secqa/preds.json --gold_path data/secqa/sample_tasks.json"
