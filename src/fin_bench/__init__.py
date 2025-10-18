"""
FinBench: Unified Financial QA Benchmarking System

A comprehensive, modular system for evaluating financial question-answering models
across multiple benchmarks with consistent metrics and parallel execution.
"""

__version__ = "0.1.0"
__author__ = "Financial QA Research Team"

from .config import Config, ExperimentConfig
from .types import Dataset, ModelProvider, BenchmarkResult, ExperimentResults

__all__ = [
    "Config",
    "ExperimentConfig",
    "Dataset",
    "ModelProvider",
    "BenchmarkResult",
    "ExperimentResults"
]
