#!/usr/bin/env python3
"""
Data setup script for all financial benchmarks.

Downloads and organizes datasets for:
- FinQA, ConvFinQA, TAT-QA (from official repos when available)
- DocMath-Eval, EconLogicQA, BizBench-QA, DocFinQA (from Hugging Face)
"""

import os
import json
import shutil
from pathlib import Path
from typing import Dict, List, Optional
import requests
from datasets import load_dataset
import pandas as pd

# Set Hugging Face token if available
hf_token = os.environ.get("HF_TOKEN")
if hf_token:
    os.environ["HUGGINGFACE_TOKEN"] = hf_token
    os.environ["HF_TOKEN"] = hf_token

class DatasetDownloader:
    """Download and organize financial benchmark datasets."""

    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)

    def download_huggingface_dataset(self, dataset_id: str, save_path: str) -> bool:
        """Download dataset from Hugging Face and save to JSON."""
        try:
            print(f"📥 Downloading {dataset_id}...")
            dataset = load_dataset(dataset_id)
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)

            # Save each split as JSON
            for split_name, split_data in dataset.items():
                output_file = save_path / f"{split_name}.json"
                split_data.to_json(output_file)
                print(f"  ✅ Saved {split_name} split to {output_file}")

            return True
        except Exception as e:
            print(f"  ❌ Error downloading {dataset_id}: {e}")
            return False

    def download_docmath_eval(self) -> bool:
        """Download DocMath-Eval dataset."""
        print("\n🧮 Setting up DocMath-Eval...")

        # Create directory
        docmath_dir = self.data_dir / "docmath_eval"
        docmath_dir.mkdir(exist_ok=True)

        # Download dataset
        success = self.download_huggingface_dataset(
            "yale-nlp/DocMath-Eval",
            str(docmath_dir / "docmath_eval.json")
        )

        if success:
            print("  ✅ DocMath-Eval setup complete")
        return success

    def download_econlogicqa(self) -> bool:
        """Download EconLogicQA dataset."""
        print("\n📊 Setting up EconLogicQA...")

        # Create directory
        econlogic_dir = self.data_dir / "econlogicqa"
        econlogic_dir.mkdir(exist_ok=True)

        # Download dataset
        success = self.download_huggingface_dataset(
            "yinzhu-quan/econ_logic_qa",
            str(econlogic_dir / "econlogicqa.json")
        )

        if success:
            print("  ✅ EconLogicQA setup complete")
        return success

    def download_bizbench(self) -> bool:
        """Download BizBench-QA dataset."""
        print("\n💼 Setting up BizBench-QA...")

        # Create directory
        bizbench_dir = self.data_dir / "bizbench"
        bizbench_dir.mkdir(exist_ok=True)

        # Download dataset
        success = self.download_huggingface_dataset(
            "kensho/bizbench",
            str(bizbench_dir / "bizbench.json")
        )

        if success:
            print("  ✅ BizBench-QA setup complete")
        return success

    def download_docfinqa(self) -> bool:
        """Download DocFinQA dataset."""
        print("\n📄 Setting up DocFinQA...")

        # Create directory
        docfinqa_dir = self.data_dir / "docfinqa"
        docfinqa_dir.mkdir(exist_ok=True)

        # Download dataset
        success = self.download_huggingface_dataset(
            "kensho/DocFinQA",
            str(docfinqa_dir / "docfinqa.json")
        )

        if success:
            print("  ✅ DocFinQA setup complete")
        return success

    def setup_finqa_data(self) -> bool:
        """Set up FinQA data from existing files or download."""
        print("\n🧮 Setting up FinQA...")

        finqa_dir = self.data_dir / "finqa"
        finqa_dir.mkdir(exist_ok=True)

        # Check if data already exists
        if list(finqa_dir.glob("*.json")):
            print("  ✅ FinQA data already exists")
            return True

        # Download from Hugging Face if available
        try:
            print("  📥 Downloading FinQA from Hugging Face...")
            dataset = load_dataset("ChanceFocus/finqa")

            # Save each split
            for split_name, split_data in dataset.items():
                output_file = finqa_dir / f"{split_name}.json"
                split_data.to_json(output_file)
                print(f"    ✅ Saved {split_name} split to {output_file}")

            print("  ✅ FinQA setup complete")
            return True
        except Exception as e:
            print(f"  ❌ Error downloading FinQA: {e}")
            print("  💡 Note: FinQA data may need to be downloaded from the official repository")
            return False

    def setup_tatqa_data(self) -> bool:
        """Set up TAT-QA data."""
        print("\n📋 Setting up TAT-QA...")

        tatqa_dir = self.data_dir / "tatqa"
        tatqa_dir.mkdir(exist_ok=True)

        # Check if data already exists
        if list(tatqa_dir.glob("*.json")):
            print("  ✅ TAT-QA data already exists")
            return True

        # Download from Hugging Face if available
        try:
            print("  📥 Downloading TAT-QA from Hugging Face...")
            dataset = load_dataset("next-tat/TAT-QA")

            # Save each split
            for split_name, split_data in dataset.items():
                output_file = tatqa_dir / f"tatqa_dataset_{split_name}.json"
                split_data.to_json(output_file)
                print(f"    ✅ Saved {split_name} split to {output_file}")

            print("  ✅ TAT-QA setup complete")
            return True
        except Exception as e:
            print(f"  ❌ Error downloading TAT-QA: {e}")
            print("  💡 Note: TAT-QA data may need to be downloaded from the official repository")
            return False

    def setup_financebench_data(self) -> bool:
        """Set up FinanceBench data."""
        print("\n🏦 Setting up FinanceBench...")

        financebench_dir = self.data_dir / "financebench"
        financebench_dir.mkdir(exist_ok=True)

        # Check if data already exists
        if (financebench_dir / "financebench_open_source.jsonl").exists():
            print("  ✅ FinanceBench data already exists")
            return True

        # Download from Hugging Face
        try:
            print("  📥 Downloading FinanceBench from Hugging Face...")
            dataset = load_dataset("PatronusAI/financebench")

            # Save as JSONL
            output_file = financebench_dir / "financebench_open_source.jsonl"
            with open(output_file, 'w') as f:
                for split_name, split_data in dataset.items():
                    for item in split_data:
                        f.write(json.dumps(item) + '\n')

            print(f"  ✅ Saved to {output_file}")
            print("  ✅ FinanceBench setup complete")
            return True
        except Exception as e:
            print(f"  ❌ Error downloading FinanceBench: {e}")
            return False

    def create_dataset_summary(self) -> None:
        """Create a summary of all available datasets."""
        summary = {}

        for dataset_dir in self.data_dir.iterdir():
            if dataset_dir.is_dir():
                dataset_name = dataset_dir.name
                files = list(dataset_dir.glob("*.json")) + list(dataset_dir.glob("*.jsonl"))

                summary[dataset_name] = {
                    "path": str(dataset_dir),
                    "files": [f.name for f in files],
                    "file_count": len(files)
                }

        # Save summary
        summary_file = self.data_dir / "dataset_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)

        print("\n📊 Dataset Summary:")
        print(f"  Saved to: {summary_file}")
        for name, info in summary.items():
            print(f"  • {name}: {info['file_count']} files")

    def run_all_downloads(self) -> bool:
        """Run all dataset downloads."""
        print("🚀 Starting dataset download process...")
        print("=" * 50)

        downloads = [
            ("DocMath-Eval", self.download_docmath_eval),
            ("EconLogicQA", self.download_econlogicqa),
            ("BizBench-QA", self.download_bizbench),
            ("DocFinQA", self.download_docfinqa),
            ("FinQA", self.setup_finqa_data),
            ("TAT-QA", self.setup_tatqa_data),
            ("FinanceBench", self.setup_financebench_data),
        ]

        success_count = 0
        for name, download_func in downloads:
            try:
                if download_func():
                    success_count += 1
            except Exception as e:
                print(f"  ❌ Failed to download {name}: {e}")

        print("\n" + "=" * 50)
        print(f"📈 Download Summary: {success_count}/{len(downloads)} successful")

        if success_count > 0:
            self.create_dataset_summary()
            print("✅ Dataset setup completed!")
            return True
        else:
            print("❌ No datasets were successfully downloaded")
            return False


def main():
    """Main function."""
    downloader = DatasetDownloader()
    success = downloader.run_all_downloads()

    if success:
        print("\n🎉 All datasets have been set up successfully!")
        print("💡 You can now run: python finbench.py run quick_test")
    else:
        print("\n⚠️  Some datasets failed to download. Check the errors above.")


if __name__ == "__main__":
    main()
