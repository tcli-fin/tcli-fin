#!/usr/bin/env python3
"""
Quick Start Example for FinBench

This example shows how to use FinBench programmatically for basic evaluation.
"""

from src.fin_bench.runner import ExperimentRunner
from src.fin_bench.config import Config

def main():
    """Run a quick evaluation example."""

    # Load configuration
    config = Config("config.yaml")

    # Create runner
    runner = ExperimentRunner(config)

    # Run a quick test
    print("🚀 Running FinBench Quick Start Example")
    print("=" * 50)

    # Get a small sample from one dataset
    try:
        # Test with a single dataset and limit
        results = runner.run_experiment("quick_test", limit=3)

        print("✅ Quick test completed successfully!")
        print(f"Total samples processed: {results.total_samples}")
        print(f"Overall accuracy: {results.accuracy".3f"}")

    except Exception as e:
        print(f"❌ Error running quick test: {e}")
        print("💡 Make sure you have API keys configured")

if __name__ == "__main__":
    main()
