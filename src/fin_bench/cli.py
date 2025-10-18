#!/usr/bin/env python3
"""
Command-line interface for FinBench.
Provides a unified interface for running financial QA benchmarks.
"""

import argparse
import sys
import time
import os
from pathlib import Path
from typing import Optional

from .config import Config, ConfigError
from .runner import ExperimentRunner, AsyncExperimentRunner
from .types import EvaluationMode


class FinBenchCLI:
    """Command-line interface for FinBench."""

    def __init__(self):
        self.parser = self._create_parser()

    def _create_parser(self) -> argparse.ArgumentParser:
        """Create argument parser."""
        parser = argparse.ArgumentParser(
            description="FinBench: Unified Financial QA Benchmarking System",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
Examples:
  # Quick test run
  finbench run quick_test --limit 5

  # Run specific experiment
  finbench run comprehensive_eval --async

  # List available configurations
  finbench list

  # Create default config
  finbench config create config.yaml

  # Validate setup
  finbench validate
            """
        )

        subparsers = parser.add_subparsers(dest='command', help='Available commands')

        # Run command
        run_parser = subparsers.add_parser('run', help='Run experiment')
        run_parser.add_argument('experiment', help='Experiment name')
        run_parser.add_argument('--config', '-c', help='Configuration file path')
        run_parser.add_argument('--output', '-o', help='Output directory')
        run_parser.add_argument('--limit', '-l', type=int, help='Limit number of samples')
        run_parser.add_argument('--async', '-a', action='store_true', help='Use async runner')
        run_parser.add_argument('--concurrency', type=int, default=10, help='Max concurrent requests')
        run_parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
        run_parser.add_argument('--keep-workspace', action='store_true', help='Keep agent workspace directories for debugging')
        run_parser.add_argument('--print-context', action='store_true', help='Print context.md path and head for agent runs')
        run_parser.add_argument('--workspace-dir', type=str, help='Base directory to create agent workspaces in')

        # List command
        list_parser = subparsers.add_parser('list', help='List configurations')
        list_parser.add_argument('type', choices=['datasets', 'models', 'experiments'],
                               help='Type of configuration to list')

        # Config command
        config_parser = subparsers.add_parser('config', help='Configuration management')
        config_subparsers = config_parser.add_subparsers(dest='config_command')

        # Config create
        create_parser = config_subparsers.add_parser('create', help='Create default config')
        create_parser.add_argument('path', help='Output path for config file')

        # Config validate
        validate_parser = config_subparsers.add_parser('validate', help='Validate configuration')
        validate_parser.add_argument('path', help='Configuration file path')

        # Validate command (standalone)
        validate_parser = subparsers.add_parser('validate', help='Validate setup')
        validate_parser.add_argument('--config', '-c', help='Configuration file path')

        # Info command
        subparsers.add_parser('info', help='Show system information')

        return parser

    def run(self):
        """Run CLI."""
        args = self.parser.parse_args()

        if not args.command:
            self.parser.print_help()
            return

        try:
            if args.command == 'run':
                self._run_experiment(args)
            elif args.command == 'list':
                self._list_config(args)
            elif args.command == 'config':
                self._handle_config(args)
            elif args.command == 'validate':
                self._validate_setup(args)
            elif args.command == 'info':
                self._show_info()
            else:
                print(f"Unknown command: {args.command}")
                self.parser.print_help()

        except ConfigError as e:
            print(f"Configuration error: {e}", file=sys.stderr)
            sys.exit(1)
        except Exception as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)

    def _run_experiment(self, args):
        """Run experiment."""
        print("🚀 FinBench: Starting Experiment")
        print("=" * 50)

        # Load configuration
        config_path = args.config or "config.yaml"
        if not Path(config_path).exists():
            print(f"❌ Configuration file not found: {config_path}")
            print("💡 Create one with: finbench config create config.yaml")
            return

        config = Config(config_path)

        # Apply CLI debug overrides for agent models if provided
        if getattr(args, 'keep_workspace', False) or getattr(args, 'print_context', False) or getattr(args, 'workspace_dir', None):
            # We mutate the in-memory model configs after loading
            for name in config.list_models():
                try:
                    model = config.get_model(name)
                    # Only apply to agent configs
                    from .types import AgentConfig
                    if isinstance(model, AgentConfig):
                        if getattr(args, 'keep_workspace', False):
                            model.keep_workspace = True
                        if getattr(args, 'print_context', False):
                            model.print_context = True
                        if getattr(args, 'workspace_dir', None):
                            model.workspace_dir = args.workspace_dir
                except Exception:
                    continue

        # Resolve experiment to validate only relevant models
        try:
            experiment = config.get_experiment(args.experiment)
        except ConfigError as e:
            print(f"❌ Experiment not found: {args.experiment}")
            return

        # Apply CLI overrides to the in-memory experiment config
        # Mutating the returned object updates the Config mapping used by the runner.
        if getattr(args, 'limit', None) is not None:
            try:
                experiment.limit = args.limit
            except Exception:
                pass
        if getattr(args, 'output', None):
            try:
                from pathlib import Path as _Path
                experiment.output_dir = _Path(args.output)
            except Exception:
                pass

        # Validate API keys only for models used in this experiment (including optional judge)
        missing_keys = []
        for mc in experiment.model_configs:
            if getattr(mc, 'api_key_env', None):
                if not os.environ.get(mc.api_key_env):
                    missing_keys.append(mc.api_key_env)
        try:
            jm = getattr(experiment, 'judge_model_config', None)
            if jm is not None and getattr(jm, 'api_key_env', None):
                if not os.environ.get(jm.api_key_env):
                    missing_keys.append(jm.api_key_env)
        except Exception:
            pass

        if missing_keys:
            print("❌ Missing API keys:")
            for key in sorted(set(missing_keys)):
                print(f"   - {key}")
            print("💡 Set environment variables before running")
            return

        # Create runner
        if getattr(args, 'async', False):
            runner = AsyncExperimentRunner(config, max_concurrent=args.concurrency)
        else:
            runner = ExperimentRunner(config)

        # Run experiment
        start_time = time.time()
        results = runner.run_experiment(args.experiment)
        execution_time = time.time() - start_time

        # Display results
        self._display_results(results, execution_time)

        # Save results
        output_path = runner.save_results(results)
        print(f"💾 Results saved to: {output_path}")

    def _display_results(self, results, execution_time):
        """Display experiment results."""
        summary = results.get_aggregate_metrics()

        print("\n📊 Experiment Results")
        print("=" * 50)
        print(f"Experiment: {results.experiment_config.name}")
        print(f"Execution Time: {execution_time:.2f} seconds")
        print(f"Total Benchmarks: {len(results.results)}")
        print(f"Total Samples: {summary.get('total_samples', 0)}")
        print(f"Overall Accuracy: {summary.get('accuracy', 0):.3f}")

        if results.results:
            print("\n📈 Per-Benchmark Results:")
            print("-" * 30)

            for result in results.results:
                accuracy = result.metrics.get('exact_match', 0)
                correct_samples = int(accuracy * result.total_samples) if result.total_samples > 0 else 0
                print(f"{result.dataset_name} + {result.model_name}:")
                print(f"  Accuracy: {accuracy:.3f} ({correct_samples}/{result.total_samples})")
                if result.errors > 0:
                    print(f"  Errors: {result.errors}")
                judge_exact = result.metrics.get('judge_exact')
                judge_approx = result.metrics.get('judge_approximate')
                if judge_exact is not None or judge_approx is not None:
                    print(
                        "  Judge: "
                        + (f"exact={judge_exact:.3f} " if judge_exact is not None else "")
                        + (f"approx={judge_approx:.3f}" if judge_approx is not None else "")
                    )

        # Show aggregate metrics
        if 'avg_exact_match' in summary:
            print("\n🏆 Aggregate Metrics:")
            print(f"  Average Exact Match: {summary['avg_exact_match']:.3f}")
            if 'avg_f1' in summary:
                print(f"  Average F1: {summary['avg_f1']:.3f}")
            # Show judge aggregate metrics if available
            if 'avg_judge_accuracy' in summary:
                print(f"  Judge Accuracy: {summary['avg_judge_accuracy']:.3f}")
                # Show breakdown of exact vs approximate
                if 'avg_judge_exact' in summary or 'avg_judge_approximate' in summary:
                    judge_parts = []
                    if 'avg_judge_exact' in summary:
                        judge_parts.append(f"exact={summary['avg_judge_exact']:.3f}")
                    if 'avg_judge_approximate' in summary:
                        judge_parts.append(f"approx={summary['avg_judge_approximate']:.3f}")
                    print(f"    Breakdown: {' '.join(judge_parts)}")

    def _list_config(self, args):
        """List configuration items."""
        config_path = "config.yaml"
        if not Path(config_path).exists():
            print(f"❌ Configuration file not found: {config_path}")
            return

        config = Config(config_path)

        if args.type == 'datasets':
            print("📚 Available Datasets:")
            for name in config.list_datasets():
                dataset = config.get_dataset(name)
                print(f"  - {name}: {dataset.description}")

        elif args.type == 'models':
            print("🤖 Available Models:")
            for name in config.list_models():
                model = config.get_model(name)
                print(f"  - {name}: {model.provider.value} {model.model_name}")

        elif args.type == 'experiments':
            print("🧪 Available Experiments:")
            for name in config.list_experiments():
                experiment = config.get_experiment(name)
                datasets = [d.name for d in experiment.datasets]
                def _fmt_model(m):
                    val = getattr(m, 'model_name', None)
                    if val:
                        return str(val)
                    # Fallback to agent name for agent providers without explicit model
                    agent = getattr(m, 'agent_name', None)
                    return str(agent) if agent else 'n/a'
                models = [_fmt_model(m) for m in experiment.model_configs]
                print(f"  - {name}:")
                print(f"    Datasets: {', '.join(datasets)}")
                print(f"    Models: {', '.join(models)}")
                print(f"    Mode: {experiment.evaluation_mode.value}")

    def _handle_config(self, args):
        """Handle configuration commands."""
        if args.config_command == 'create':
            config = Config()
            config.create_default_config(args.path)
        elif args.config_command == 'validate':
            config = Config(args.path)
            issues = config.validate_api_keys()
            if issues:
                print("❌ Configuration issues found:")
                for issue in issues:
                    print(f"   - {issue}")
            else:
                print("✅ Configuration is valid!")

    def _validate_setup(self, args):
        """Validate system setup."""
        print("🔍 Validating FinBench Setup")
        print("=" * 50)

        config_path = args.config or "config.yaml"
        if not Path(config_path).exists():
            print(f"❌ Configuration file not found: {config_path}")
            return

        config = Config(config_path)

        # Check API keys
        print("🔑 Checking API keys...")
        missing_keys = config.validate_api_keys()
        if missing_keys:
            print("❌ Missing API keys:")
            for key in missing_keys:
                print(f"   - {key}")
        else:
            print("✅ All required API keys are set")

        # Check datasets
        print("\n📚 Checking datasets...")
        valid_datasets = 0
        for name in config.list_datasets():
            try:
                dataset = config.get_dataset(name)
                if validate_dataset_structure(dataset):
                    print(f"✅ {name}: Valid structure")
                    valid_datasets += 1
                else:
                    print(f"❌ {name}: Invalid structure")
            except Exception as e:
                print(f"❌ {name}: Error - {e}")

        print(f"\nDataset status: {valid_datasets}/{len(config.list_datasets())} valid")

        # Check models
        print("\n🤖 Checking model providers...")
        available_providers = []
        for name in config.list_models():
            try:
                model = config.get_model(name)
                provider = ModelProviderFactory.create_provider(model)
                available_providers.append(name)
                print(f"✅ {name}: {model.provider.value}")
            except Exception as e:
                print(f"❌ {name}: Error - {e}")

        print(f"\nModel status: {len(available_providers)}/{len(config.list_models())} available")

        # Overall status
        print("\n" + "=" * 50)
        if not missing_keys and valid_datasets == len(config.list_datasets()):
            print("🎉 Setup validation PASSED!")
            print("💡 You can now run experiments with: finbench run <experiment_name>")
        else:
            print("⚠️  Setup validation found issues")
            print("💡 Fix the issues above before running experiments")

    def _show_info(self):
        """Show system information."""
        print("ℹ️  FinBench Information")
        print("=" * 50)
        print("A unified system for financial QA benchmarking")
        print()
        print("Features:")
        print("  • Multi-dataset evaluation")
        print("  • Multi-model provider support")
        print("  • Parallel execution")
        print("  • Comprehensive metrics")
        print("  • Research-oriented design")
        print()
        print("Supported Datasets:")
        print("  • FinQA (program execution)")
        print("  • ConvFinQA (conversational)")
        print("  • TAT-QA (table/text hybrid)")
        print("  • FinanceBench (open-book QA)")
        print("  • DocFinQA (retrieval-augmented)")
        print()
        print("Supported Models:")
        print("  • Google Gemini")
        print("  • OpenAI GPT")
        print("  • Anthropic Claude")
        print("  • OpenRouter")
        print()
        print("For more information, see the documentation.")


def main():
    """Main entry point."""
    cli = FinBenchCLI()
    cli.run()


if __name__ == "__main__":
    main()
