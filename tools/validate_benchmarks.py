#!/usr/bin/env python3
"""
Validation script for all financial benchmarks.

Tests data loading, sample parsing, and basic functionality for all supported benchmarks.
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Any

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

try:
    from fin_bench.datasets import DatasetFactory, validate_dataset_structure, load_samples
    from fin_bench.types import DatasetType, Dataset
    from fin_bench.config import Config
except ImportError as e:
    print(f"❌ Error: Could not import FinBench modules: {e}")
    print("💡 Make sure you're running from the project root directory")
    sys.exit(1)


class BenchmarkValidator:
    """Validate all benchmark setups."""

    def __init__(self, config_path: str = None):
        if config_path is None:
            config_path = str(Path(__file__).parent.parent / "config.yaml")
        self.config = Config(config_path)
        self.results = {}

    def validate_all_datasets(self) -> bool:
        """Validate all datasets in configuration."""
        print("🔍 Validating all datasets...")
        print("=" * 50)

        all_valid = True
        dataset_names = self.config.list_datasets()

        for dataset_name in dataset_names:
            try:
                dataset = self.config.get_dataset(dataset_name)
                is_valid = self.validate_single_dataset(dataset)
                self.results[dataset_name] = is_valid
                if not is_valid:
                    all_valid = False
            except Exception as e:
                print(f"❌ Error validating {dataset_name}: {e}")
                self.results[dataset_name] = False
                all_valid = False

        return all_valid

    def validate_single_dataset(self, dataset: Dataset) -> bool:
        """Validate a single dataset."""
        print(f"\n📊 Validating {dataset.name}...")

        # Check path exists
        if not dataset.path.exists():
            print(f"  ❌ Dataset path does not exist: {dataset.path}")
            return False

        # Check structure
        if not validate_dataset_structure(dataset):
            print(f"  ❌ Invalid dataset structure for {dataset.name}")
            return False

        print(f"  ✅ Dataset path exists: {dataset.path}")

        # Try to load samples from each split
        samples_loaded = 0
        for split in dataset.splits:
            try:
                samples = load_samples(dataset, split, limit=5)  # Load only first 5 samples
                if samples:
                    print(f"  ✅ {split} split: {len(samples)} samples loaded")
                    if len(samples) > 0:
                        # Validate sample structure
                        sample = samples[0]
                        self.validate_sample_structure(sample, dataset)
                    samples_loaded += 1
                else:
                    print(f"  ⚠️  {split} split: No samples found")
            except Exception as e:
                print(f"  ❌ Error loading {split} split: {e}")
                return False

        if samples_loaded > 0:
            print(f"  ✅ Successfully loaded {samples_loaded}/{len(dataset.splits)} splits")
            return True
        else:
            print(f"  ❌ No splits could be loaded")
            return False

    def validate_sample_structure(self, sample, dataset: Dataset):
        """Validate that a sample has the expected structure."""
        print("    📋 Sample structure:")

        # Check required fields
        required_fields = ['id', 'question']
        for field in required_fields:
            if hasattr(sample, field) and getattr(sample, field):
                print(f"      ✅ {field}: {getattr(sample, field)[:50]}...")
            else:
                print(f"      ❌ Missing {field}")

        # Check optional fields based on dataset configuration
        if dataset.has_context and sample.context:
            print(f"      ✅ context: {len(sample.context)} characters")
        elif dataset.has_context:
            print("      ⚠️  context: None/empty")

        if dataset.has_program and sample.program:
            print(f"      ✅ program: {len(sample.program)} characters")
        elif dataset.has_program:
            print("      ⚠️  program: None/empty")

        if dataset.has_answer and sample.answer:
            print(f"      ✅ answer: {sample.answer}")
        elif dataset.has_answer:
            print("      ⚠️  answer: None/empty")

        # Check metadata
        if sample.metadata:
            print(f"      ✅ metadata: {list(sample.metadata.keys())}")

    def test_sample_loading(self) -> bool:
        """Test loading samples from different datasets."""
        print("\n🧪 Testing sample loading...")
        print("=" * 50)

        test_datasets = [
            ("docfinqa", "test", 3),
            ("finqa", "test", 3),
            ("econlogicqa", "test", 3),
            ("bizbench", "test", 3),
            ("financebench", "default", 3),
        ]

        success_count = 0

        for dataset_name, split, limit in test_datasets:
            try:
                if dataset_name not in self.config.list_datasets():
                    print(f"  ⚠️  Skipping {dataset_name} (not configured)")
                    continue

                dataset = self.config.get_dataset(dataset_name)
                samples = load_samples(dataset, split, limit=limit)

                if len(samples) >= limit:
                    print(f"  ✅ {dataset_name}: Successfully loaded {len(samples)} samples")
                    success_count += 1

                    # Show sample details
                    sample = samples[0]
                    print(f"     Sample ID: {sample.id}")
                    print(f"     Question: {sample.question[:100]}...")
                    if sample.context:
                        print(f"     Context: {len(sample.context)} characters")
                    if sample.answer:
                        print(f"     Answer: {sample.answer}")
                else:
                    print(f"  ⚠️  {dataset_name}: Only loaded {len(samples)}/{limit} samples")

            except Exception as e:
                print(f"  ❌ Error testing {dataset_name}: {e}")

        print(f"\n📊 Sample loading test: {success_count}/{len(test_datasets)} successful")
        return success_count > 0

    def generate_summary_report(self) -> str:
        """Generate a summary report of validation results."""
        print("\n📋 Validation Summary Report")
        print("=" * 50)

        total_datasets = len(self.results)
        valid_datasets = sum(1 for result in self.results.values() if result)

        print(f"Total datasets: {total_datasets}")
        print(f"Valid datasets: {valid_datasets}")
        print(f"Success rate: {valid_datasets/total_datasets*100:.1f}%")

        print("\n📈 Per-dataset results:")
        for dataset_name, is_valid in self.results.items():
            status = "✅" if is_valid else "❌"
            print(f"  {status} {dataset_name}")

        # Generate recommendations
        print("\n💡 Recommendations:")
        if valid_datasets == total_datasets:
            print("  🎉 All datasets are properly configured!")
        else:
            print("  📝 Issues found. Check the validation output above.")

            # Specific recommendations
            invalid_datasets = [name for name, valid in self.results.items() if not valid]
            if "docmath_eval" in invalid_datasets:
                print("  🔐 DocMath-Eval requires Hugging Face access approval")
                print("     Visit: https://huggingface.co/datasets/yale-nlp/DocMath-Eval")

        return f"Validation complete: {valid_datasets}/{total_datasets} datasets valid"

    def run_full_validation(self) -> bool:
        """Run complete validation suite."""
        print("🚀 Starting comprehensive benchmark validation...")
        print("=" * 60)

        # Step 1: Validate all datasets
        datasets_valid = self.validate_all_datasets()

        # Step 2: Test sample loading
        samples_valid = self.test_sample_loading()

        # Step 3: Generate report
        report = self.generate_summary_report()

        print(f"\n{report}")

        # Overall result
        overall_success = datasets_valid and samples_valid

        if overall_success:
            print("\n🎉 VALIDATION PASSED!")
            print("💡 You can now run: python finbench.py run quick_test")
        else:
            print("\n⚠️  VALIDATION FOUND ISSUES")
            print("💡 Fix the issues above before running experiments")

        return overall_success


def main():
    """Main function."""
    validator = BenchmarkValidator()
    success = validator.run_full_validation()

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
