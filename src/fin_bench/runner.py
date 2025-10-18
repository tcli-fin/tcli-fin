"""
Main experiment runner for FinBench.
Orchestrates dataset loading, model evaluation, and result aggregation.
"""

import asyncio
import time
import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict

from .config import Config, ConfigError
from .types import (
    Dataset, ModelConfig, ExperimentConfig, BenchmarkResult,
    ExperimentResults, EvaluationMode, SampleDict
)
from .datasets import DatasetFactory, DatasetSample, load_samples, validate_dataset_structure
from .models import ModelProviderFactory, BaseModelProvider, ModelResponse, RetryConfig
from .metrics import (
    BaseMetric, EvaluationResult, evaluate_prediction,
    aggregate_results, get_default_metrics_for_dataset
)
from .judge import LLMJudge


class ExperimentRunner:
    """Main experiment runner."""

    def __init__(self, config: Union[str, Path, Config]):
        if isinstance(config, (str, Path)):
            self.config = Config(config)
        else:
            self.config = config

        self.logger = logging.getLogger(__name__)
        self.setup_logging()

    def setup_logging(self):
        """Setup logging configuration."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )

    def validate_setup(self, experiment_name: Optional[str] = None) -> List[str]:
        """Validate experiment setup."""
        issues = []

        # Check API keys
        missing_keys = []
        try:
            if experiment_name:
                experiment_config = self.config.get_experiment(experiment_name)
                # Validate only models used in this experiment
                for model_config in experiment_config.model_configs:
                    if getattr(model_config, 'api_key_env', None):
                        import os
                        if not os.environ.get(model_config.api_key_env):
                            missing_keys.append(model_config.api_key_env)
                # Also validate judge model if present
                jm = getattr(experiment_config, 'judge_model_config', None)
                if jm and getattr(jm, 'api_key_env', None):
                    import os
                    if not os.environ.get(jm.api_key_env):
                        missing_keys.append(jm.api_key_env)
            else:
                missing_keys = self.config.validate_api_keys()
        except Exception as e:
            issues.append(f"API key validation failed: {e}")

        if missing_keys:
            issues.extend([f"Missing API key: {key}" for key in missing_keys])

        # Validate datasets - only check the ones used in the specific experiment if provided
        datasets_to_check = []
        if experiment_name:
            try:
                experiment_config = self.config.get_experiment(experiment_name)
                datasets_to_check = [d.name for d in experiment_config.datasets]
            except ConfigError:
                # If experiment not found, check all datasets
                datasets_to_check = self.config.list_datasets()
        else:
            datasets_to_check = self.config.list_datasets()

        for name in datasets_to_check:
            try:
                dataset = self.config.get_dataset(name)
                if not validate_dataset_structure(dataset):
                    issues.append(f"Invalid dataset structure: {name}")
            except Exception as e:
                issues.append(f"Dataset validation failed: {name} - {e}")

        return issues

    def run_experiment(self, experiment_name: str) -> ExperimentResults:
        """Run a complete experiment."""
        self.logger.info(f"Starting experiment: {experiment_name}")

        try:
            experiment_config = self.config.get_experiment(experiment_name)
        except ConfigError as e:
            raise ConfigError(f"Experiment not found: {experiment_name}") from e

        # Validate setup for this specific experiment
        issues = self.validate_setup(experiment_name)
        if issues:
            raise ConfigError(f"Setup validation failed: {'; '.join(issues)}")

        start_time = time.time()
        results = []

        # Run benchmarks for each dataset-model combination
        for dataset in experiment_config.datasets:
            for model_config in experiment_config.model_configs:
                try:
                    result = self.run_benchmark(
                        dataset=dataset,
                        model_config=model_config,
                        experiment_config=experiment_config
                    )
                    results.append(result)
                    self.logger.info(f"Completed: {dataset.name} + {model_config.model_name}")
                except Exception as e:
                    self.logger.error(f"Failed {dataset.name} + {model_config.model_name}: {e}")
                    # Continue with other combinations

        execution_time = time.time() - start_time

        # Create summary
        summary = {
            "experiment_name": experiment_name,
            "execution_time": execution_time,
            "total_benchmarks": len(results),
            "successful_benchmarks": len([r for r in results if r.processed_samples > 0]),
        }

        return ExperimentResults(
            experiment_config=experiment_config,
            results=results,
            summary=summary
        )

    def run_benchmark(
        self,
        dataset: Dataset,
        model_config: ModelConfig,
        experiment_config: ExperimentConfig
    ) -> BenchmarkResult:
        """Run a single benchmark."""
        self.logger.info(f"Running benchmark: {dataset.name} with {model_config.model_name}")

        start_time = time.time()

        try:
            # Load samples
            samples = self.load_dataset_samples(dataset, experiment_config)
            if not samples:
                raise ValueError(f"No samples loaded for {dataset.name}")

            # Create model provider
            provider = self.create_model_provider(model_config)

            # Get appropriate metrics
            metrics = self.get_metrics_for_dataset(dataset.type.value)

            # Set concurrency and experiment config for this benchmark run
            self._current_concurrency = experiment_config.concurrency
            self._current_experiment_config = experiment_config

            # Run evaluation
            evaluation_results = self.evaluate_samples(
                samples=samples,
                provider=provider,
                metrics=metrics,
                experiment_config=experiment_config
            )

            # Aggregate results
            aggregated = aggregate_results(evaluation_results)

            execution_time = time.time() - start_time

            return BenchmarkResult(
                dataset_name=dataset.name,
                model_name=model_config.model_name,
                split=experiment_config.name,  # Use experiment name as split identifier
                metrics=aggregated.get("metrics", {}),
                total_samples=aggregated.get("total_samples", len(samples)),
                processed_samples=len(evaluation_results),
                errors=aggregated.get("error_samples", 0),
                execution_time=execution_time,
                detailed_results=[asdict(r) for r in evaluation_results]
            )

        except Exception as e:
            self.logger.error(f"Benchmark failed: {e}")
            return BenchmarkResult(
                dataset_name=dataset.name,
                model_name=model_config.model_name,
                split=experiment_config.name,
                metrics={},
                total_samples=0,
                processed_samples=0,
                errors=1,
                execution_time=time.time() - start_time,
                detailed_results=[]
            )

    def load_dataset_samples(
        self,
        dataset: Dataset,
        experiment_config: ExperimentConfig
    ) -> List[DatasetSample]:
        """Load dataset samples."""
        try:
            samples = []
            for split in dataset.splits:
                split_samples = load_samples(
                    dataset=dataset,
                    split=split,
                    limit=experiment_config.limit
                )
                samples.extend(split_samples)

            # Apply start index and limit
            start_idx = experiment_config.start_index
            if experiment_config.limit:
                samples = samples[start_idx:start_idx + experiment_config.limit]
            else:
                samples = samples[start_idx:]

            self.logger.info(f"Loaded {len(samples)} samples for {dataset.name}")
            return samples

        except Exception as e:
            self.logger.error(f"Failed to load dataset {dataset.name}: {e}")
            return []

    def create_model_provider(self, model_config: ModelConfig) -> BaseModelProvider:
        """Create model provider instance."""
        retry_config = RetryConfig(
            max_retries=3,
            initial_backoff=1.0,
            max_backoff=30.0
        )

        return ModelProviderFactory.create_provider(model_config, retry_config)

    def get_metrics_for_dataset(self, dataset_type: str) -> List[BaseMetric]:
        """Get appropriate metrics for dataset type."""
        return get_default_metrics_for_dataset(dataset_type)

    def evaluate_samples(
        self,
        samples: List[DatasetSample],
        provider: BaseModelProvider,
        metrics: List[BaseMetric],
        experiment_config: ExperimentConfig
    ) -> List[EvaluationResult]:
        """Evaluate samples using the specified evaluation mode."""
        evaluation_results = []

        judge = None
        if getattr(experiment_config, 'judge_model_config', None):
            judge = LLMJudge(experiment_config.judge_model_config)

        if experiment_config.evaluation_mode == EvaluationMode.DIRECT_ANSWER:
            evaluation_results = self._evaluate_direct(samples, provider, metrics, judge)
        elif experiment_config.evaluation_mode == EvaluationMode.PROGRAM_EXECUTION:
            evaluation_results = self._evaluate_program_execution(samples, provider, metrics, judge)
        elif experiment_config.evaluation_mode == EvaluationMode.RETRIEVAL_AUGMENTED:
            evaluation_results = self._evaluate_retrieval_augmented(samples, provider, metrics, judge)
        elif experiment_config.evaluation_mode == EvaluationMode.AGENT:
            evaluation_results = self._evaluate_agent(samples, provider, metrics, judge)
        else:
            raise ValueError(f"Unsupported evaluation mode: {experiment_config.evaluation_mode}")

        return evaluation_results

    def _evaluate_direct(
        self,
        samples: List[DatasetSample],
        provider: BaseModelProvider,
        metrics: List[BaseMetric],
        judge: Optional[LLMJudge] = None
    ) -> List[EvaluationResult]:
        """Direct evaluation: context + question → answer."""
        # Get concurrency setting from experiment config
        concurrency = getattr(self, '_current_concurrency', 1)
        
        if concurrency <= 1:
            # Sequential evaluation
            return self._evaluate_direct_sequential(samples, provider, metrics, judge)
        else:
            # Concurrent evaluation
            return self._evaluate_direct_concurrent(samples, provider, metrics, judge, concurrency)
    
    def _evaluate_direct_sequential(
        self,
        samples: List[DatasetSample],
        provider: BaseModelProvider,
        metrics: List[BaseMetric],
        judge: Optional[LLMJudge] = None
    ) -> List[EvaluationResult]:
        """Sequential direct evaluation (original implementation)."""
        results = []
        total_samples = len(samples)

        for idx, sample in enumerate(samples, 1):
            self.logger.info(f"Processing sample {idx}/{total_samples} ({idx/total_samples*100:.1f}%)")
            try:
                # Prepare messages
                messages = []
                if sample.context:
                    messages.append({
                        "role": "system",
                        "content": f"Context: {sample.context}"
                    })
                messages.append({
                    "role": "user",
                    "content": f"Question: {sample.question}\nAnswer:"
                })

                # Generate response
                response = provider.generate_with_retry(messages)

                if response.error:
                    results.append(EvaluationResult(
                        prediction="",
                        ground_truth=sample.answer or "",
                        metrics={},
                        is_correct=False,
                        error=response.error
                    ))
                    continue

                # Evaluate prediction
                evaluation = evaluate_prediction(
                    prediction=response.text,
                    ground_truth=sample.answer or "",
                    metrics=metrics,
                    context=sample.context,
                    question=sample.question
                )

                if judge is not None:
                    try:
                        j = judge.judge(
                            context=sample.context or "",
                            question=sample.question or "",
                            gold=sample.answer or "",
                            prediction=response.text or "",
                        )
                    except Exception as _e:
                        j = {
                            "judge_label": "error",
                            "judge_reason": str(_e),
                            "judge_raw": "",
                            "judge_exact_match": 0,
                            "judge_approximate_match": 0,
                        }
                    evaluation.metadata.update(j)

                results.append(evaluation)

            except Exception as e:
                results.append(EvaluationResult(
                    prediction="",
                    ground_truth=sample.answer or "",
                    metrics={},
                    is_correct=False,
                    error=str(e)
                ))

        return results
    
    def _evaluate_direct_concurrent(
        self,
        samples: List[DatasetSample],
        provider: BaseModelProvider,
        metrics: List[BaseMetric],
        judge: Optional[LLMJudge] = None,
        max_workers: int = 5
    ) -> List[EvaluationResult]:
        """Concurrent direct evaluation using ThreadPoolExecutor."""
        from concurrent.futures import ThreadPoolExecutor, as_completed
        
        def evaluate_single_sample(sample: DatasetSample) -> EvaluationResult:
            """Evaluate a single sample (called in thread)."""
            try:
                # Prepare messages
                messages = []
                if sample.context:
                    messages.append({
                        "role": "system",
                        "content": f"Context: {sample.context}"
                    })
                messages.append({
                    "role": "user",
                    "content": f"Question: {sample.question}\nAnswer:"
                })

                # Generate response
                response = provider.generate_with_retry(messages)

                if response.error:
                    return EvaluationResult(
                        prediction="",
                        ground_truth=sample.answer or "",
                        metrics={},
                        is_correct=False,
                        error=response.error
                    )

                # Evaluate prediction
                evaluation = evaluate_prediction(
                    prediction=response.text,
                    ground_truth=sample.answer or "",
                    metrics=metrics,
                    context=sample.context,
                    question=sample.question
                )

                if judge is not None:
                    try:
                        j = judge.judge(
                            context=sample.context or "",
                            question=sample.question or "",
                            gold=sample.answer or "",
                            prediction=response.text or "",
                        )
                    except Exception as _e:
                        j = {
                            "judge_label": "error",
                            "judge_reason": str(_e),
                            "judge_raw": "",
                            "judge_exact_match": 0,
                            "judge_approximate_match": 0,
                        }
                    evaluation.metadata.update(j)

                return evaluation

            except Exception as e:
                return EvaluationResult(
                    prediction="",
                    ground_truth=sample.answer or "",
                    metrics={},
                    is_correct=False,
                    error=str(e)
                )
        
        # Run evaluation with thread pool
        results = [None] * len(samples)  # Preserve order
        self.logger.info(f"Running concurrent direct evaluation with {max_workers} workers")
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks with their index
            future_to_index = {
                executor.submit(evaluate_single_sample, sample): idx
                for idx, sample in enumerate(samples)
            }
            
            # Collect results as they complete
            completed = 0
            for future in as_completed(future_to_index):
                idx = future_to_index[future]
                try:
                    results[idx] = future.result()
                    completed += 1
                    if completed % 10 == 0 or completed == len(samples):
                        self.logger.info(f"Completed {completed}/{len(samples)} samples ({completed/len(samples)*100:.1f}%)")
                except Exception as e:
                    self.logger.error(f"Future failed for sample {idx}: {e}")
                    results[idx] = EvaluationResult(
                        prediction="",
                        ground_truth=samples[idx].answer or "",
                        metrics={},
                        is_correct=False,
                        error=str(e)
                    )
        
        return results

    def _evaluate_program_execution(
        self,
        samples: List[DatasetSample],
        provider: BaseModelProvider,
        metrics: List[BaseMetric],
        judge: Optional[LLMJudge] = None
    ) -> List[EvaluationResult]:
        """Program execution evaluation: generate + execute program."""
        # Get concurrency setting from experiment config
        concurrency = getattr(self, '_current_concurrency', 1)
        
        if concurrency <= 1:
            # Sequential evaluation
            return self._evaluate_program_execution_sequential(samples, provider, metrics, judge)
        else:
            # Concurrent evaluation
            return self._evaluate_program_execution_concurrent(samples, provider, metrics, judge, concurrency)
    
    def _evaluate_program_execution_sequential(
        self,
        samples: List[DatasetSample],
        provider: BaseModelProvider,
        metrics: List[BaseMetric],
        judge: Optional[LLMJudge] = None
    ) -> List[EvaluationResult]:
        """Sequential program execution evaluation (original implementation)."""
        results = []

        for sample in samples:
            try:
                # Prepare prompt for program generation
                prompt = self._build_program_generation_prompt(sample)

                messages = [{
                    "role": "user",
                    "content": prompt
                }]

                # Generate program
                response = provider.generate_with_retry(messages)

                if response.error:
                    results.append(EvaluationResult(
                        prediction="",
                        ground_truth=sample.answer or "",
                        metrics={},
                        is_correct=False,
                        error=response.error
                    ))
                    continue

                # Evaluate program execution
                evaluation = evaluate_prediction(
                    prediction=response.text,
                    ground_truth=sample.answer or "",
                    metrics=metrics,
                    program=response.text,
                    context=sample.context,
                    question=sample.question
                )

                if judge is not None:
                    try:
                        j = judge.judge(
                            context=sample.context or "",
                            question=sample.question or "",
                            gold=sample.answer or "",
                            prediction=response.text or "",
                        )
                    except Exception as _e:
                        j = {
                            "judge_label": "error",
                            "judge_reason": str(_e),
                            "judge_raw": "",
                            "judge_exact_match": 0,
                            "judge_approximate_match": 0,
                        }
                    evaluation.metadata.update(j)

                results.append(evaluation)

            except Exception as e:
                results.append(EvaluationResult(
                    prediction="",
                    ground_truth=sample.answer or "",
                    metrics={},
                    is_correct=False,
                    error=str(e)
                ))

        return results
    
    def _evaluate_program_execution_concurrent(
        self,
        samples: List[DatasetSample],
        provider: BaseModelProvider,
        metrics: List[BaseMetric],
        judge: Optional[LLMJudge] = None,
        max_workers: int = 5
    ) -> List[EvaluationResult]:
        """Concurrent program execution evaluation using ThreadPoolExecutor."""
        from concurrent.futures import ThreadPoolExecutor, as_completed
        
        def evaluate_single_sample(sample: DatasetSample) -> EvaluationResult:
            """Evaluate a single sample (called in thread)."""
            try:
                # Prepare prompt for program generation
                prompt = self._build_program_generation_prompt(sample)

                messages = [{
                    "role": "user",
                    "content": prompt
                }]

                # Generate program
                response = provider.generate_with_retry(messages)

                if response.error:
                    return EvaluationResult(
                        prediction="",
                        ground_truth=sample.answer or "",
                        metrics={},
                        is_correct=False,
                        error=response.error
                    )

                # Evaluate program execution
                evaluation = evaluate_prediction(
                    prediction=response.text,
                    ground_truth=sample.answer or "",
                    metrics=metrics,
                    program=response.text,
                    context=sample.context,
                    question=sample.question
                )

                if judge is not None:
                    try:
                        j = judge.judge(
                            context=sample.context or "",
                            question=sample.question or "",
                            gold=sample.answer or "",
                            prediction=response.text or "",
                        )
                    except Exception as _e:
                        j = {
                            "judge_label": "error",
                            "judge_reason": str(_e),
                            "judge_raw": "",
                            "judge_exact_match": 0,
                            "judge_approximate_match": 0,
                        }
                    evaluation.metadata.update(j)

                return evaluation

            except Exception as e:
                return EvaluationResult(
                    prediction="",
                    ground_truth=sample.answer or "",
                    metrics={},
                    is_correct=False,
                    error=str(e)
                )
        
        # Run evaluation with thread pool
        results = [None] * len(samples)  # Preserve order
        self.logger.info(f"Running concurrent program execution evaluation with {max_workers} workers")
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks with their index
            future_to_index = {
                executor.submit(evaluate_single_sample, sample): idx
                for idx, sample in enumerate(samples)
            }
            
            # Collect results as they complete
            completed = 0
            for future in as_completed(future_to_index):
                idx = future_to_index[future]
                try:
                    results[idx] = future.result()
                    completed += 1
                    if completed % 10 == 0 or completed == len(samples):
                        self.logger.info(f"Completed {completed}/{len(samples)} samples ({completed/len(samples)*100:.1f}%)")
                except Exception as e:
                    self.logger.error(f"Future failed for sample {idx}: {e}")
                    results[idx] = EvaluationResult(
                        prediction="",
                        ground_truth=samples[idx].answer or "",
                        metrics={},
                        is_correct=False,
                        error=str(e)
                    )
        
        return results

    def _evaluate_retrieval_augmented(
        self,
        samples: List[DatasetSample],
        provider: BaseModelProvider,
        metrics: List[BaseMetric],
        judge: Optional[LLMJudge] = None
    ) -> List[EvaluationResult]:
        """Retrieval-augmented evaluation using RAG pipeline."""
        # Get concurrency and RAG config from experiment config
        concurrency = getattr(self, '_current_concurrency', 1)
        experiment_config = getattr(self, '_current_experiment_config', None)

        # Check if RAG is configured and enabled
        if not experiment_config or not hasattr(experiment_config, 'rag_config'):
            self.logger.warning(
                "RAG config not found in experiment config. "
                "Falling back to direct evaluation."
            )
            return self._evaluate_direct(samples, provider, metrics, judge)

        rag_config = experiment_config.rag_config

        if not rag_config or not rag_config.enabled:
            self.logger.warning(
                "RAG is disabled. Falling back to direct evaluation."
            )
            return self._evaluate_direct(samples, provider, metrics, judge)

        # Initialize RAG pipeline
        from .rag import RAGPipeline
        rag_pipeline = RAGPipeline(rag_config)

        self.logger.info(
            f"Using RAG pipeline for evaluation "
            f"(embedding: {rag_config.embedding_model}, "
            f"chunks: {rag_config.chunk_size}, "
            f"top_k: {rag_config.top_k_rerank})"
        )

        if concurrency <= 1:
            # Sequential evaluation with RAG
            return self._evaluate_retrieval_augmented_sequential(
                samples, provider, metrics, judge, rag_pipeline
            )
        else:
            # Concurrent evaluation with RAG
            return self._evaluate_retrieval_augmented_concurrent(
                samples, provider, metrics, judge, rag_pipeline, concurrency
            )

    def _evaluate_retrieval_augmented_sequential(
        self,
        samples: List[DatasetSample],
        provider: BaseModelProvider,
        metrics: List[BaseMetric],
        judge: Optional[LLMJudge] = None,
        rag_pipeline=None
    ) -> List[EvaluationResult]:
        """Sequential retrieval-augmented evaluation."""
        results = []
        total_samples = len(samples)

        for idx, sample in enumerate(samples, 1):
            self.logger.info(
                f"Processing sample {idx}/{total_samples} ({idx/total_samples*100:.1f}%)"
            )

            try:
                # Index the full context
                context_id = rag_pipeline.index_context(sample.context)

                # Retrieve relevant chunks for the question
                retrieved_context = rag_pipeline.retrieve(
                    sample.question,
                    context_id
                )

                # Prepare messages with retrieved context
                messages = [
                    {
                        "role": "system",
                        "content": f"Context: {retrieved_context}"
                    },
                    {
                        "role": "user",
                        "content": f"Question: {sample.question}\nAnswer:"
                    }
                ]

                # Generate response
                response = provider.generate_with_retry(messages)

                if response.error:
                    results.append(EvaluationResult(
                        prediction="",
                        ground_truth=sample.answer or "",
                        metrics={},
                        is_correct=False,
                        error=response.error
                    ))
                    continue

                # Evaluate prediction
                evaluation = evaluate_prediction(
                    prediction=response.text,
                    ground_truth=sample.answer or "",
                    metrics=metrics,
                    context=retrieved_context,  # Use retrieved context for metrics
                    question=sample.question
                )

                # Add RAG metadata
                evaluation.metadata["rag_used"] = True
                evaluation.metadata["retrieved_context_length"] = len(retrieved_context)

                # Run judge if available
                if judge is not None:
                    try:
                        j = judge.judge(
                            context=retrieved_context,  # Use retrieved context
                            question=sample.question or "",
                            gold=sample.answer or "",
                            prediction=response.text or "",
                        )
                    except Exception as _e:
                        j = {
                            "judge_label": "error",
                            "judge_reason": str(_e),
                            "judge_raw": "",
                            "judge_exact_match": 0,
                            "judge_approximate_match": 0,
                        }
                    evaluation.metadata.update(j)

                results.append(evaluation)

            except Exception as e:
                self.logger.error(f"RAG evaluation failed for sample {idx}: {e}")
                results.append(EvaluationResult(
                    prediction="",
                    ground_truth=sample.answer or "",
                    metrics={},
                    is_correct=False,
                    error=str(e)
                ))

        # Print RAG statistics
        if hasattr(rag_pipeline, 'print_stats'):
            rag_pipeline.print_stats()

        return results

    def _evaluate_retrieval_augmented_concurrent(
        self,
        samples: List[DatasetSample],
        provider: BaseModelProvider,
        metrics: List[BaseMetric],
        judge: Optional[LLMJudge] = None,
        rag_pipeline=None,
        max_workers: int = 5
    ) -> List[EvaluationResult]:
        """Concurrent retrieval-augmented evaluation."""
        from concurrent.futures import ThreadPoolExecutor, as_completed

        def evaluate_single_sample(sample: DatasetSample) -> EvaluationResult:
            """Evaluate a single sample with RAG."""
            try:
                # Index and retrieve
                context_id = rag_pipeline.index_context(sample.context)
                retrieved_context = rag_pipeline.retrieve(sample.question, context_id)

                # Prepare messages
                messages = [
                    {
                        "role": "system",
                        "content": f"Context: {retrieved_context}"
                    },
                    {
                        "role": "user",
                        "content": f"Question: {sample.question}\nAnswer:"
                    }
                ]

                # Generate response
                response = provider.generate_with_retry(messages)

                if response.error:
                    return EvaluationResult(
                        prediction="",
                        ground_truth=sample.answer or "",
                        metrics={},
                        is_correct=False,
                        error=response.error
                    )

                # Evaluate prediction
                evaluation = evaluate_prediction(
                    prediction=response.text,
                    ground_truth=sample.answer or "",
                    metrics=metrics,
                    context=retrieved_context,
                    question=sample.question
                )

                # Add RAG metadata
                evaluation.metadata["rag_used"] = True
                evaluation.metadata["retrieved_context_length"] = len(retrieved_context)

                # Run judge if available
                if judge is not None:
                    try:
                        j = judge.judge(
                            context=retrieved_context,
                            question=sample.question or "",
                            gold=sample.answer or "",
                            prediction=response.text or "",
                        )
                    except Exception as _e:
                        j = {
                            "judge_label": "error",
                            "judge_reason": str(_e),
                            "judge_raw": "",
                            "judge_exact_match": 0,
                            "judge_approximate_match": 0,
                        }
                    evaluation.metadata.update(j)

                return evaluation

            except Exception as e:
                return EvaluationResult(
                    prediction="",
                    ground_truth=sample.answer or "",
                    metrics={},
                    is_correct=False,
                    error=str(e)
                )

        # Run with thread pool
        results = [None] * len(samples)
        self.logger.info(
            f"Running concurrent RAG evaluation with {max_workers} workers"
        )

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_index = {
                executor.submit(evaluate_single_sample, sample): idx
                for idx, sample in enumerate(samples)
            }

            completed = 0
            for future in as_completed(future_to_index):
                idx = future_to_index[future]
                try:
                    results[idx] = future.result()
                    completed += 1
                    if completed % 10 == 0 or completed == len(samples):
                        self.logger.info(
                            f"Completed {completed}/{len(samples)} samples "
                            f"({completed/len(samples)*100:.1f}%)"
                        )
                except Exception as e:
                    self.logger.error(f"Future failed for sample {idx}: {e}")
                    results[idx] = EvaluationResult(
                        prediction="",
                        ground_truth=samples[idx].answer or "",
                        metrics={},
                        is_correct=False,
                        error=str(e)
                    )

        # Print RAG statistics
        if hasattr(rag_pipeline, 'print_stats'):
            rag_pipeline.print_stats()

        return results

    def _evaluate_agent(
        self,
        samples: List[DatasetSample],
        provider: BaseModelProvider,
        metrics: List[BaseMetric],
        judge: Optional[LLMJudge] = None
    ) -> List[EvaluationResult]:
        """Agent-based evaluation: CLI agents with context files."""
        
        # Get concurrency setting from experiment config
        concurrency = getattr(self, '_current_concurrency', 1)
        
        if concurrency <= 1:
            # Sequential evaluation
            return self._evaluate_agent_sequential(samples, provider, metrics, judge)
        else:
            # Concurrent evaluation
            return self._evaluate_agent_concurrent(samples, provider, metrics, judge, concurrency)
    
    def _evaluate_agent_sequential(
        self,
        samples: List[DatasetSample],
        provider: BaseModelProvider,
        metrics: List[BaseMetric],
        judge: Optional[LLMJudge] = None
    ) -> List[EvaluationResult]:
        """Sequential agent evaluation (original implementation)."""
        results = []
        total_samples = len(samples)

        for idx, sample in enumerate(samples, 1):
            self.logger.info(f"Processing sample {idx}/{total_samples} ({idx/total_samples*100:.1f}%)")
            try:
                # Prepare messages for agent
                messages = []
                # Build a dataset-aware context payload for the agent's context.md
                context_payload = self._build_agent_context(sample)
                if context_payload:
                    messages.append({
                        "role": "system",
                        "content": context_payload
                    })
                messages.append({
                    "role": "user",
                    "content": f"Question: {sample.question}\nAnswer:"
                })

                # Generate response using agent
                response = provider.generate_with_retry(messages)

                if response.error:
                    results.append(EvaluationResult(
                        prediction="",
                        ground_truth=sample.answer or "",
                        metrics={},
                        is_correct=False,
                        error=response.error,
                        metadata=response.metadata
                    ))
                else:
                    # Evaluate prediction
                    result = evaluate_prediction(
                        prediction=response.text,
                        ground_truth=sample.answer or "",
                        metrics=metrics
                    )
                    result.metadata.update(response.metadata or {})
                    if judge is not None:
                        try:
                            # Use agent-specific judge input (hybrid approach)
                            # Prefer judge_input (e.g., trae-cli execution summary) over full_stdout
                            # This reduces token count while preserving relevant context
                            prediction_for_judge = response.metadata.get(
                                "judge_input",  # Agent-specific extraction (e.g., trae-cli summary)
                                response.metadata.get("full_stdout", response.text)  # Fallback to full or extracted
                            )
                            j = judge.judge(
                                context=sample.context or "",
                                question=sample.question or "",
                                gold=sample.answer or "",
                                prediction=prediction_for_judge or "",
                            )
                        except Exception as _e:
                            j = {
                                "judge_label": "error",
                                "judge_reason": str(_e),
                                "judge_raw": "",
                                "judge_exact_match": 0,
                                "judge_approximate_match": 0,
                            }
                        result.metadata.update(j)
                    results.append(result)

            except Exception as e:
                self.logger.error(f"Agent evaluation failed for sample: {e}")
                results.append(EvaluationResult(
                    prediction="",
                    ground_truth=sample.answer or "",
                    metrics={},
                    is_correct=False,
                    error=str(e),
                    metadata={}
                ))

        return results
    
    def _evaluate_agent_concurrent(
        self,
        samples: List[DatasetSample],
        provider: BaseModelProvider,
        metrics: List[BaseMetric],
        judge: Optional[LLMJudge] = None,
        max_workers: int = 5
    ) -> List[EvaluationResult]:
        """Concurrent agent evaluation using ThreadPoolExecutor."""
        from concurrent.futures import ThreadPoolExecutor, as_completed
        
        def evaluate_single_sample(sample: DatasetSample) -> EvaluationResult:
            """Evaluate a single sample (called in thread)."""
            try:
                # Prepare messages for agent
                messages = []
                context_payload = self._build_agent_context(sample)
                if context_payload:
                    messages.append({
                        "role": "system",
                        "content": context_payload
                    })
                messages.append({
                    "role": "user",
                    "content": f"Question: {sample.question}\nAnswer:"
                })

                # Generate response using agent
                response = provider.generate_with_retry(messages)

                if response.error:
                    return EvaluationResult(
                        prediction="",
                        ground_truth=sample.answer or "",
                        metrics={},
                        is_correct=False,
                        error=response.error,
                        metadata=response.metadata
                    )
                else:
                    # Evaluate prediction
                    result = evaluate_prediction(
                        prediction=response.text,
                        ground_truth=sample.answer or "",
                        metrics=metrics
                    )
                    result.metadata.update(response.metadata or {})
                    if judge is not None:
                        try:
                            prediction_for_judge = response.metadata.get(
                                "judge_input",
                                response.metadata.get("full_stdout", response.text)
                            )
                            j = judge.judge(
                                context=sample.context or "",
                                question=sample.question or "",
                                gold=sample.answer or "",
                                prediction=prediction_for_judge or "",
                            )
                        except Exception as _e:
                            j = {
                                "judge_label": "error",
                                "judge_reason": str(_e),
                                "judge_raw": "",
                                "judge_exact_match": 0,
                                "judge_approximate_match": 0,
                            }
                        result.metadata.update(j)
                    return result

            except Exception as e:
                self.logger.error(f"Agent evaluation failed for sample: {e}")
                return EvaluationResult(
                    prediction="",
                    ground_truth=sample.answer or "",
                    metrics={},
                    is_correct=False,
                    error=str(e),
                    metadata={}
                )
        
        # Run evaluation with thread pool
        results = [None] * len(samples)  # Preserve order
        self.logger.info(f"Running concurrent evaluation with {max_workers} workers")
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks with their index
            future_to_index = {
                executor.submit(evaluate_single_sample, sample): idx
                for idx, sample in enumerate(samples)
            }
            
            # Collect results as they complete
            completed = 0
            for future in as_completed(future_to_index):
                idx = future_to_index[future]
                try:
                    results[idx] = future.result()
                    completed += 1
                    self.logger.info(f"Completed {completed}/{len(samples)} samples")
                except Exception as e:
                    self.logger.error(f"Future failed for sample {idx}: {e}")
                    results[idx] = EvaluationResult(
                        prediction="",
                        ground_truth=samples[idx].answer or "",
                        metrics={},
                        is_correct=False,
                        error=str(e),
                        metadata={}
                    )
        
        return results

    def _build_program_generation_prompt(self, sample: DatasetSample) -> str:
        """Build prompt for program generation."""
        return f"""
You are given a context and a question, and must write Python code that computes the numeric answer.

Context: {sample.context}
Question: {sample.question}

Write Python code that:
- Uses readable variable names grounded in the context
- Assigns the final numeric result to a variable named 'answer'
- Only uses safe built-in functions: abs, min, max, sum, len, range, round, float, int, str

Return only the Python code without any explanation.
""".strip()

    def _build_agent_context(self, sample: DatasetSample) -> str:
        """Construct a rich, benchmark-aware context payload for context.md.

        The provider will wrap this with a top-level "# Context" header, so this
        content should use sub-headings (## ...) where appropriate and avoid
        repeating a top-level header.
        """
        sections: List[str] = []

        base_context = sample.context or ""
        if base_context.strip():
            sections.append(base_context.strip())
        
        meta = sample.metadata or {}
        
        # TAT-QA: Add question(s) section based on include_all_questions flag
        if meta.get('dataset') == 'tatqa':
            # Check if we should include all questions or just the specific one
            # Use metadata flag set by the loader based on dataset config
            include_all = meta.get('include_all_questions', False)
            
            if include_all and 'all_context_questions' in meta:
                # Include ALL questions from the context
                question_lines = ["## Questions"]
                question_lines.append("")
                for q_info in meta['all_context_questions']:
                    question_lines.append(f"{q_info['order']}. {q_info['question']}")
                sections.append("\n".join(question_lines))
            elif sample.question:
                # Include only the specific question being asked (TAT-LLM approach)
                sections.append(f"## Question\n\n{sample.question}")
        elif sample.question:
            # For non-TAT-QA datasets, add question if present
            sections.append(f"## Question\n\n{sample.question}")
        
        # ConvFinQA: Add conversation history if available
        if meta.get('conversational') and meta.get('has_conversation_history'):
            conversation_history = meta.get('conversation_history', [])
            if conversation_history:
                history_lines = ["## Conversation History"]
                history_lines.append("")
                for i, turn in enumerate(conversation_history):
                    history_lines.append(f"**Q{i+1}:** {turn['question']}")
                    # Only show previous answers, not the current one
                    if i < len(conversation_history) - 1:
                        ans = turn['answer']
                        # Handle answer references like A0, A1
                        if isinstance(ans, str) and ans.startswith('A') and ans[1:].isdigit():
                            ref_idx = int(ans[1:])
                            if ref_idx < len(conversation_history):
                                ans = f"{ans} (refers to answer {ref_idx+1})"
                        history_lines.append(f"**A{i+1}:** {ans}")
                        history_lines.append("")
                sections.append("\n".join(history_lines))

        # NOTE: Program is intentionally NOT included in context.md
        # The agent should solve the problem without seeing the reference solution
        # if getattr(sample, "program", None):
        #     prog = str(sample.program or "").strip()
        #     if prog:
        #         sections.append("## Reference Program\n\n```python\n" + prog + "\n```")

        # FinanceBench: include evidence and answer type if present
        ev = meta.get("evidence")
        if ev:
            sections.append("## Evidence\n\n" + str(ev).strip())
        # Only include answer_type for FinanceBench (not TAT-QA, DocMath, etc.)
        if meta.get("dataset") == "financebench":
            ans_type = meta.get("answer_type")
            if ans_type:
                sections.append(f"## Answer Type\n\n{ans_type}")

        # EconLogicQA: include multiple-choice options
        options = meta.get("options")
        if isinstance(options, list) and options:
            opt_lines = ["## Options"] + [f"- {str(o)}" for o in options]
            sections.append("\n\n".join([opt_lines[0], "\n".join(opt_lines[1:])]))

        # BizBench: include task metadata if present
        task_bits = []
        if meta.get("task_type"):
            task_bits.append(f"Task Type: {meta.get('task_type')}")
        if meta.get("domain"):
            task_bits.append(f"Domain: {meta.get('domain')}")
        if meta.get("difficulty"):
            task_bits.append(f"Difficulty: {meta.get('difficulty')}")
        if task_bits:
            sections.append("## Task Info\n\n" + "\n".join(task_bits))

        # DocMath-Eval: paragraph/table evidence removed to keep context minimal
        # Only the base context (paragraphs) from sample.context is included

        return "\n\n".join([s for s in sections if s.strip()])

    def save_results(self, results: ExperimentResults, output_path: Optional[Path] = None) -> Path:
        """Save experiment results to file."""
        if output_path is None:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            output_path = results.experiment_config.output_dir / f"results_{timestamp}.json"

        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Convert dataclasses to dicts for JSON serialization
        results_dict = asdict(results)
        results_dict["experiment_config"] = asdict(results.experiment_config)

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results_dict, f, indent=2, default=str)

        self.logger.info(f"Results saved to: {output_path}")
        return output_path

    def create_summary_report(self, results: ExperimentResults) -> str:
        """Create a human-readable summary report."""
        summary = results.get_aggregate_metrics()

        report = f"""
# Experiment Summary: {results.experiment_config.name}

## Overview
- **Total Benchmarks**: {len(results.results)}
- **Total Samples**: {summary.get('total_samples', 0)}
- **Overall Accuracy**: {summary.get('accuracy', 0):.3f}
- **Error Rate**: {summary.get('error_rate', 0):.3f}

## Per-Benchmark Results
"""

        for result in results.results:
            report += f"""
### {result.dataset_name} + {result.model_name}
- **Accuracy**: {result.metrics.get('exact_match', 0):.3f}
- **Samples**: {result.processed_samples}/{result.total_samples}
- **Errors**: {result.errors}
- **Execution Time**: {result.execution_time:.2f}s
"""

        if 'avg_exact_match' in summary:
            report += f"""
## Aggregate Metrics
- **Average Exact Match**: {summary['avg_exact_match']:.3f}
- **Average F1**: {summary.get('avg_f1', 0):.3f}
"""

        return report


class AsyncExperimentRunner(ExperimentRunner):
    """Async version of experiment runner for better concurrency."""

    def __init__(self, config: Union[str, Path, Config], max_concurrent: int = 10):
        super().__init__(config)
        self.max_concurrent = max_concurrent

    async def run_benchmark_async(
        self,
        dataset: Dataset,
        model_config: ModelConfig,
        experiment_config: ExperimentConfig
    ) -> BenchmarkResult:
        """Async version of run_benchmark."""
        # For now, just wrap the sync version
        # In a full implementation, this would use aiohttp for API calls
        loop = asyncio.get_event_loop()
        with ThreadPoolExecutor(max_workers=1) as executor:
            result = await loop.run_in_executor(
                executor,
                self.run_benchmark,
                dataset,
                model_config,
                experiment_config
            )
        return result

    async def run_experiment_async(self, experiment_name: str) -> ExperimentResults:
        """Async version of run_experiment."""
        experiment_config = self.config.get_experiment(experiment_name)

        # Create tasks for all dataset-model combinations
        tasks = []
        for dataset in experiment_config.datasets:
            for model_config in experiment_config.model_configs:
                task = self.run_benchmark_async(dataset, model_config, experiment_config)
                tasks.append(task)

        # Run with limited concurrency
        semaphore = asyncio.Semaphore(self.max_concurrent)

        async def run_with_semaphore(task):
            async with semaphore:
                return await task

        results = []
        for i in range(0, len(tasks), self.max_concurrent):
            batch = tasks[i:i + self.max_concurrent]
            batch_results = await asyncio.gather(*batch)
            results.extend(batch_results)
            self.logger.info(f"Completed batch {i//self.max_concurrent + 1}")

        # Create summary
        summary = {
            "experiment_name": experiment_name,
            "execution_time": 0,  # Would need proper timing
            "total_benchmarks": len(results),
            "successful_benchmarks": len([r for r in results if r.processed_samples > 0]),
        }

        return ExperimentResults(
            experiment_config=experiment_config,
            results=results,
            summary=summary
        )
