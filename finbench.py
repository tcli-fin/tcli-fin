#!/usr/bin/env python3
"""
FinBench: Unified Financial QA Benchmarking System

Main entry point for the FinBench system.
Provides a unified interface for running financial QA benchmarks.
"""

import sys
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

try:
    from fin_bench.cli import main
    main()
except ImportError as e:
    print("❌ Error: Could not import FinBench modules")
    print(f"   {e}")
    print("💡 Make sure you're running from the project root directory")
    sys.exit(1)
