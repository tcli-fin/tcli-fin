"""
Configuration management for FinBench experiments.
"""

import os
import json
import yaml
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
from dataclasses import asdict

from .types import (
    Dataset, DatasetType, ModelProvider, EvaluationMode,
    ModelConfig, AgentConfig, ExperimentConfig
)


class ConfigError(Exception):
    """Configuration-related error."""
    pass


class Config:
    """Main configuration manager."""

    def __init__(self, config_path: Optional[Union[str, Path]] = None):
        self.config_path = Path(config_path) if config_path else None
        self._config_data: Dict[str, Any] = {}
        self._datasets: Dict[str, Dataset] = {}
        self._model_configs: Dict[str, ModelConfig] = {}
        self._rag_configs: Dict[str, Any] = {}  # RAG configurations
        self._experiment_configs: Dict[str, ExperimentConfig] = {}

        if config_path:
            self.load_config(config_path)

    def load_config(self, config_path: Union[str, Path]) -> None:
        """Load configuration from file."""
        config_path = Path(config_path)

        if not config_path.exists():
            raise ConfigError(f"Configuration file not found: {config_path}")

        with open(config_path, 'r') as f:
            if config_path.suffix in ['.yaml', '.yml']:
                self._config_data = yaml.safe_load(f)
            elif config_path.suffix == '.json':
                self._config_data = json.load(f)
            else:
                raise ConfigError(f"Unsupported config file format: {config_path.suffix}")

        self._parse_config()

    def _parse_config(self) -> None:
        """Parse configuration data into typed objects."""
        # Parse datasets
        datasets_config = self._config_data.get('datasets', {})
        for name, config in datasets_config.items():
            dataset = self._parse_dataset_config(name, config)
            self._datasets[name] = dataset

        # Parse models
        models_config = self._config_data.get('models', {})
        for name, config in models_config.items():
            model = self._parse_model_config(name, config)
            self._model_configs[name] = model

        # Parse RAG configs
        rag_configs_config = self._config_data.get('rag_configs', {})
        for name, config in rag_configs_config.items():
            rag_config = self._parse_rag_config(name, config)
            self._rag_configs[name] = rag_config

        # Parse experiments
        experiments_config = self._config_data.get('experiments', {})
        for name, config in experiments_config.items():
            experiment = self._parse_experiment_config(name, config)
            self._experiment_configs[name] = experiment

    def _parse_dataset_config(self, name: str, config: Dict[str, Any]) -> Dataset:
        """Parse dataset configuration."""
        try:
            dataset_type = DatasetType(config['type'])
            path = Path(config['path'])

            return Dataset(
                name=name,
                type=dataset_type,
                path=path,
                splits=config.get('splits', ['train', 'dev', 'test']),
                has_context=config.get('has_context', True),
                has_program=config.get('has_program', False),
                has_answer=config.get('has_answer', True),
                include_all_questions=config.get('include_all_questions', False),
                description=config.get('description', '')
            )
        except KeyError as e:
            raise ConfigError(f"Missing required field in dataset '{name}': {e}")

    def _parse_model_config(self, name: str, config: Dict[str, Any]) -> Union[ModelConfig, AgentConfig]:
        """Parse model configuration."""
        try:
            provider = ModelProvider(config['provider'])

            if provider == ModelProvider.AGENT:
                # Parse as AgentConfig
                return AgentConfig(
                    provider=provider,
                    model_name=config['model_name'],
                    api_key_env=config.get('api_key_env'),
                    max_tokens=config.get('max_tokens', 32768),
                    temperature=config.get('temperature', 0.0),
                    thinking_budget=config.get('thinking_budget'),
                    provider_kwargs=config.get('provider_kwargs', {}),
                    agent_name=config.get('agent_name', ''),
                    cli_command=config.get('cli_command', ''),
                    cli_args=config.get('cli_args', []),
                    workspace_dir=config.get('workspace_dir', 'temp'),
                    timeout=config.get('timeout', 300),
                    keep_workspace=config.get('keep_workspace', False),
                    print_context=config.get('print_context', False)
                )
            else:
                # Parse as regular ModelConfig
                return ModelConfig(
                    provider=provider,
                    model_name=config['model_name'],
                    api_key_env=config.get('api_key_env'),
                    max_tokens=config.get('max_tokens', 32768),
                    temperature=config.get('temperature', 0.0),
                    thinking_budget=config.get('thinking_budget'),
                    provider_kwargs=config.get('provider_kwargs', {})
                )
        except KeyError as e:
            raise ConfigError(f"Missing required field in model '{name}': {e}")

    def _parse_rag_config(self, name: str, config: Dict[str, Any]):
        """Parse RAG configuration."""
        from .rag import RAGConfig

        try:
            return RAGConfig.from_dict(config)
        except Exception as e:
            raise ConfigError(f"Error parsing RAG config '{name}': {e}")

    def _parse_experiment_config(self, name: str, config: Dict[str, Any]) -> ExperimentConfig:
        """Parse experiment configuration."""
        try:
            # Resolve dataset references
            dataset_names = config['datasets']
            datasets = [self._datasets[name] for name in dataset_names]

            # Resolve model references
            model_names = config['models']
            model_configs = [self._model_configs[name] for name in model_names]

            evaluation_mode = EvaluationMode(config['evaluation_mode'])

            exp = ExperimentConfig(
                name=name,
                datasets=datasets,
                model_configs=model_configs,
                evaluation_mode=evaluation_mode,
                concurrency=config.get('concurrency', 10),
                max_retries=config.get('max_retries', 3),
                output_dir=Path(config.get('output_dir', 'results')),
                log_dir=Path(config.get('log_dir', 'logs')),
                start_index=config.get('start_index', 0),
                limit=config.get('limit'),
                random_seed=config.get('random_seed', 42)
            )

            # Optional judge model reference by name (must exist under top-level models)
            judge_model_name = config.get('judge_model')
            if judge_model_name:
                if judge_model_name not in self._model_configs:
                    raise ConfigError(f"Judge model not found: {judge_model_name}")
                exp.judge_model_config = self._model_configs[judge_model_name]

            # Optional RAG config reference by name (must exist under top-level rag_configs)
            rag_config_name = config.get('rag_config')
            if rag_config_name:
                if rag_config_name not in self._rag_configs:
                    raise ConfigError(f"RAG config not found: {rag_config_name}")
                exp.rag_config = self._rag_configs[rag_config_name]

            return exp
        except KeyError as e:
            raise ConfigError(f"Missing required field in experiment '{name}': {e}")

    def get_dataset(self, name: str) -> Dataset:
        """Get dataset configuration by name."""
        if name not in self._datasets:
            raise ConfigError(f"Dataset not found: {name}")
        return self._datasets[name]

    def get_model(self, name: str) -> ModelConfig:
        """Get model configuration by name."""
        if name not in self._model_configs:
            raise ConfigError(f"Model not found: {name}")
        return self._model_configs[name]

    def get_experiment(self, name: str) -> ExperimentConfig:
        """Get experiment configuration by name."""
        if name not in self._experiment_configs:
            raise ConfigError(f"Experiment not found: {name}")
        return self._experiment_configs[name]

    def list_datasets(self) -> List[str]:
        """List available dataset names."""
        return list(self._datasets.keys())

    def list_models(self) -> List[str]:
        """List available model names."""
        return list(self._model_configs.keys())

    def list_experiments(self) -> List[str]:
        """List available experiment names."""
        return list(self._experiment_configs.keys())

    def validate_api_keys(self) -> List[str]:
        """Validate that required API keys are set."""
        missing_keys = []

        for model_config in self._model_configs.values():
            if model_config.api_key_env:
                env_value = os.environ.get(model_config.api_key_env)
                if not env_value:
                    missing_keys.append(model_config.api_key_env)

        return missing_keys

    def create_default_config(self, path: Union[str, Path]) -> None:
        """Create a default configuration file."""
        default_config = {
            'datasets': {
                'docfinqa': {
                    'type': 'docfinqa',
                    'path': 'DocFinQA',
                    'splits': ['train', 'dev', 'test'],
                    'has_context': True,
                    'has_program': True,
                    'has_answer': True,
                    'description': 'DocFinQA: Long-form financial QA with retrieval'
                },
                'finqa': {
                    'type': 'finqa',
                    'path': 'data/finqa',
                    'splits': ['train', 'dev', 'test'],
                    'has_context': True,
                    'has_program': True,
                    'has_answer': True,
                    'description': 'FinQA: Numerical reasoning over financial data'
                },
                'tatqa': {
                    'type': 'tatqa',
                    'path': 'data/tatqa',
                    'splits': ['train', 'dev', 'test'],
                    'has_context': True,
                    'has_program': False,
                    'has_answer': True,
                    'description': 'TAT-QA: Table and text hybrid QA'
                }
            },
            'models': {
                'gemini-pro': {
                    'provider': 'gemini',
                    'model_name': 'gemini-2.5-pro',
                    'api_key_env': 'GEMINI_API_KEY',
                    'max_tokens': 32768,
                    'temperature': 0.0
                },
                'gpt4o-mini': {
                    'provider': 'openai',
                    'model_name': 'gpt-4o-mini',
                    'api_key_env': 'OPENAI_API_KEY',
                    'max_tokens': 32768,
                    'temperature': 0.0
                },
                'claude-sonnet': {
                    'provider': 'anthropic-bedrock',
                    'model_name': 'us.anthropic.claude-sonnet-4-20250514-v1:0',
                    'api_key_env': None,
                    'max_tokens': 64000,
                    'temperature': 0.0,
                    'thinking_budget': 40000
                }
            },
            'experiments': {
                'quick_test': {
                    'datasets': ['docfinqa', 'finqa'],
                    'models': ['gemini-pro'],
                    'evaluation_mode': 'program_execution',
                    'concurrency': 5,
                    'limit': 10
                },
                'comprehensive_eval': {
                    'datasets': ['docfinqa', 'finqa', 'tatqa'],
                    'models': ['gemini-pro', 'gpt4o-mini', 'claude-sonnet'],
                    'evaluation_mode': 'direct_answer',
                    'concurrency': 10,
                    'max_retries': 5
                },
                'retrieval_study': {
                    'datasets': ['docfinqa'],
                    'models': ['gemini-pro'],
                    'evaluation_mode': 'retrieval_augmented',
                    'concurrency': 5,
                    'limit': 50
                }
            }
        }

        config_path = Path(path)
        config_path.parent.mkdir(parents=True, exist_ok=True)

        with open(config_path, 'w') as f:
            yaml.dump(default_config, f, default_flow_style=False, sort_keys=False)

        print(f"Default configuration created at: {config_path}")
