"""
Core type definitions for FinBench.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Any, Optional, Union, Callable, TYPE_CHECKING
from pathlib import Path

if TYPE_CHECKING:
    from .rag import RAGConfig


class DatasetType(Enum):
    """Supported dataset types."""
    FINQA = "finqa"
    CONVFINQA = "convfinqa"
    TATQA = "tatqa"
    FINANCEBENCH = "financebench"
    DOCFINQA = "docfinqa"
    ECONLOGICQA = "econlogicqa"
    BIZBENCH = "bizbench"
    DOCMATH_EVAL = "docmath_eval"
    FINER139 = "finer139"


class ModelProvider(Enum):
    """Supported model providers."""
    GEMINI = "gemini"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    OPENROUTER = "openrouter"
    ANTHROPIC_BEDROCK = "anthropic-bedrock"
    AGENT = "agent"


class EvaluationMode(Enum):
    """Evaluation modes."""
    PROGRAM_EXECUTION = "program_execution"  # FinQA-style: generate program, execute
    DIRECT_ANSWER = "direct_answer"          # Direct context + question → answer
    RETRIEVAL_AUGMENTED = "retrieval_augmented"  # DocFinQA-style: retrieval + generation
    AGENT = "agent"                          # CLI-based coding agents


@dataclass
class Dataset:
    """Dataset configuration."""
    name: str
    type: DatasetType
    path: Path
    splits: List[str]
    has_context: bool = True
    has_program: bool = False
    has_answer: bool = True
    include_all_questions: bool = False  # For TAT-QA: include all questions from context or just the specific one
    description: str = ""

    @property
    def sample_fields(self) -> List[str]:
        """Fields present in dataset samples."""
        fields = []
        if self.has_context:
            fields.append("context")
        if self.has_program:
            fields.append("program")
        if self.has_answer:
            fields.append("answer")
        return fields


@dataclass
class ModelConfig:
    """Model configuration."""
    provider: ModelProvider
    model_name: str
    api_key_env: Optional[str] = None
    max_tokens: int = 32768
    temperature: float = 0.0
    thinking_budget: Optional[int] = None
    provider_kwargs: Dict[str, Any] = None

    def __post_init__(self):
        if self.provider_kwargs is None:
            self.provider_kwargs = {}


@dataclass
class AgentConfig(ModelConfig):
    """Configuration for CLI-based coding agents."""
    agent_name: str = ""              # "claude-code", "aider", etc.
    cli_command: str = ""             # "claude", "aider", etc.
    cli_args: List[str] = None        # ["--output-format", "json"]
    workspace_dir: str = "temp"       # Workspace directory
    timeout: int = 300                # CLI timeout in seconds
    keep_workspace: bool = False      # If True, do not delete temp workspace
    print_context: bool = False       # If True, print context.md path and contents

    def __post_init__(self):
        super().__post_init__()
        if self.cli_args is None:
            self.cli_args = []


@dataclass
class ExperimentConfig:
    """Experiment configuration."""
    name: str
    datasets: List[Dataset]
    model_configs: List[ModelConfig]
    evaluation_mode: EvaluationMode
    concurrency: int = 10
    max_retries: int = 3
    output_dir: Path = Path("results")
    log_dir: Path = Path("logs")
    start_index: int = 0
    limit: Optional[int] = None
    random_seed: int = 42
    # Optional LLM judge configuration (referenced model config)
    judge_model_config: Optional[ModelConfig] = None
    # Optional RAG configuration for retrieval-augmented evaluation
    rag_config: Optional["RAGConfig"] = None

    def __post_init__(self):
        self.output_dir = Path(self.output_dir)
        self.log_dir = Path(self.log_dir)


@dataclass
class BenchmarkResult:
    """Result from a single benchmark run."""
    dataset_name: str
    model_name: str
    split: str
    metrics: Dict[str, float]
    total_samples: int
    processed_samples: int
    errors: int
    execution_time: float
    detailed_results: List[Dict[str, Any]]
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class ExperimentResults:
    """Complete experiment results."""
    experiment_config: ExperimentConfig
    results: List[BenchmarkResult]
    summary: Dict[str, Any]

    def get_aggregate_metrics(self) -> Dict[str, float]:
        """Get aggregate metrics across all results."""
        if not self.results:
            return {}

        metrics = {}
        total_samples = sum(r.total_samples for r in self.results)
        processed_samples = sum(r.processed_samples for r in self.results)

        metrics["total_samples"] = total_samples
        metrics["processed_samples"] = processed_samples
        metrics["processing_rate"] = processed_samples / total_samples if total_samples > 0 else 0

        # Average metrics across datasets/models
        metric_keys = set()
        for result in self.results:
            metric_keys.update(result.metrics.keys())

        for key in metric_keys:
            values = [r.metrics[key] for r in self.results if key in r.metrics]
            if values:
                metrics[f"avg_{key}"] = sum(values) / len(values)

        return metrics


# Type aliases for commonly used types
SampleDict = Dict[str, Any]
MetricDict = Dict[str, float]
ProviderCallable = Callable[[str, str, Dict[str, Any]], str]
