"""
Model providers for different LLM APIs.
Extends the existing ModelSession with additional functionality.
"""

import os
import json
import time
import random
import re
import subprocess
import shutil
import tempfile
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, asdict
from pathlib import Path

from .types import ModelProvider, ModelConfig, AgentConfig
from .config import ConfigError


@dataclass
class ModelResponse:
    """Standardized model response."""
    text: str
    usage: Dict[str, int] = None
    metadata: Dict[str, Any] = None
    error: Optional[str] = None

    def __post_init__(self):
        if self.usage is None:
            self.usage = {}
        if self.metadata is None:
            self.metadata = {}


@dataclass
class RetryConfig:
    """Configuration for retry logic."""
    max_retries: int = 3
    initial_backoff: float = 1.0
    max_backoff: float = 30.0
    backoff_multiplier: float = 2.0
    jitter: bool = True

    def calculate_delay(self, attempt: int, is_rate_limited: bool = False) -> float:
        """Calculate delay for retry attempt."""
        delay = min(self.max_backoff, self.initial_backoff * (self.backoff_multiplier ** attempt))

        if is_rate_limited:
            delay *= 2.0  # Extra delay for rate limits

        if self.jitter:
            delay *= (0.5 + random.random() * 0.5)  # Add ±50% jitter

        return delay


class FinancialAnswerExtractor:
    """Extract financial answers from CLI agent responses."""

    def extract_answer(self, agent_response: str) -> Optional[str]:
        """Extract numerical answer using multiple strategies."""

        # Parse JSON response if applicable
        try:
            response_data = json.loads(agent_response)
            result_text = response_data.get("result", "")
        except json.JSONDecodeError:
            result_text = agent_response

        # Strategy 1: Bold answer patterns
        bold_patterns = [
            r'\*\*(\d+\.?\d*%?)\*\*',  # **37.69%**
            r'\*\*Answer:\*\*[^0-9]*(\d+\.?\d*%?)',  # **Answer:** 37.69%
            r'was \*\*(\d+\.?\d*%?)\*\*',  # was **6.18%**
            r'is \*\*(\d+\.?\d*%?)\*\*',  # is **1.89**
        ]

        for pattern in bold_patterns:
            match = re.search(pattern, result_text)
            if match:
                return self._clean_answer(match.group(1))

        # Strategy 2: Direct ratio/percentage statements
        direct_patterns = [
            r'ratio is (\d+\.?\d*)',  # ratio is 1.89
            r'margin.*?(\d+\.?\d*%)',  # margin for Q3 2023 is 37.69%
            r'growth.*?(\d+\.?\d*%)',  # growth from Q2 to Q3 2023 was 6.18%
            r'(\d+\.?\d*%)\s*\(',  # 37.69% (calculated as...
        ]

        for pattern in direct_patterns:
            match = re.search(pattern, result_text, re.IGNORECASE)
            if match:
                return self._clean_answer(match.group(1))

        # Strategy 3: Last number in parentheses (concise format)
        paren_match = re.search(r'\(.*?(\d+\.?\d*%?)\s*\)', result_text)
        if paren_match:
            return self._clean_answer(paren_match.group(1))

        # Strategy 4: Any percentage or decimal number as fallback
        number_match = re.search(r'(\d+\.?\d*%?)', result_text)
        if number_match:
            return self._clean_answer(number_match.group(1))

        return None

    def _clean_answer(self, answer: str) -> str:
        """Clean and standardize answer format."""
        return answer.strip().replace('$', '').replace(',', '')


class BaseModelProvider(ABC):
    """Base class for model providers."""

    def __init__(self, config: ModelConfig, retry_config: Optional[RetryConfig] = None):
        self.config = config
        self.retry_config = retry_config or RetryConfig()
        self._validate_config()

    def _validate_config(self):
        """Validate model configuration."""
        if self.config.api_key_env and self.config.api_key_env not in os.environ:
            raise ConfigError(f"Required environment variable not set: {self.config.api_key_env}")

    @abstractmethod
    def generate(self, messages: List[Dict[str, str]], **kwargs) -> ModelResponse:
        """Generate response from model."""
        pass

    @abstractmethod
    def generate_with_retry(self, messages: List[Dict[str, str]], **kwargs) -> ModelResponse:
        """Generate response with retry logic."""
        pass

    def _is_rate_limit_error(self, error: Exception) -> bool:
        """Check if error is a rate limit error."""
        error_msg = str(error).lower()
        return any(term in error_msg for term in [
            "rate limit", "too many requests", "429",
            "quota exceeded", "resource exhausted",
            "rate_limit_exceeded", "insufficient_quota",
            "openrouter rate limit", "credits exhausted"
        ])

    def _sleep_with_backoff(self, attempt: int, is_rate_limited: bool = False):
        """Sleep with exponential backoff."""
        delay = self.retry_config.calculate_delay(attempt, is_rate_limited)
        time.sleep(delay)


class GeminiProvider(BaseModelProvider):
    """Google Gemini provider."""

    def __init__(self, config: ModelConfig, retry_config: Optional[RetryConfig] = None):
        super().__init__(config, retry_config)
        self._import_dependencies()

    def _import_dependencies(self):
        """Import required dependencies."""
        try:
            import google.generativeai as genai
            self.genai = genai
        except ImportError:
            raise ConfigError("google-generativeai not installed. Install with: pip install google-generativeai")

    def generate(self, messages: List[Dict[str, str]], **kwargs) -> ModelResponse:
        """Generate response using Gemini."""
        try:
            # Configure API
            api_key = os.environ.get(self.config.api_key_env or "GEMINI_API_KEY")
            self.genai.configure(api_key=api_key)

            # Get model
            model = self.genai.GenerativeModel(self.config.model_name)

            # Prepare prompt
            system_prompt = ""
            user_prompt = ""

            for msg in messages:
                if msg["role"] == "system":
                    system_prompt = msg["content"]
                elif msg["role"] == "user":
                    user_prompt = msg["content"]

            full_prompt = system_prompt + "\n\n" + user_prompt if system_prompt else user_prompt

            # Generate response
            generation_config = {
                "temperature": self.config.temperature,
                "max_output_tokens": self.config.max_tokens,
            }

            if self.config.thinking_budget:
                generation_config["thinking_config"] = {
                    "thinking_budget": self.config.thinking_budget
                }

            response = model.generate_content(
                full_prompt,
                generation_config=generation_config
            )

            # Extract text
            text = self._extract_response_text(response)

            return ModelResponse(
                text=text,
                usage={"prompt_tokens": 0, "completion_tokens": 0},  # Gemini doesn't provide usage info
                metadata={"model": self.config.model_name}
            )

        except Exception as e:
            return ModelResponse(
                text="",
                error=str(e),
                metadata={"model": self.config.model_name}
            )

    def generate_with_retry(self, messages: List[Dict[str, str]], **kwargs) -> ModelResponse:
        """Generate with retry logic."""
        last_response = None

        for attempt in range(self.retry_config.max_retries + 1):
            response = self.generate(messages, **kwargs)
            last_response = response

            if response.error:
                if attempt < self.retry_config.max_retries:
                    is_rate_limited = self._is_rate_limit_error(Exception(response.error))
                    self._sleep_with_backoff(attempt, is_rate_limited)
                    continue
            else:
                # Successful response
                return response

        return last_response or ModelResponse(text="", error="Max retries exceeded")

    def _extract_response_text(self, response) -> str:
        """Extract text from Gemini response."""
        try:
            if hasattr(response, 'text') and response.text:
                return response.text.strip()

            # Try candidates
            if hasattr(response, 'candidates') and response.candidates:
                for candidate in response.candidates:
                    if hasattr(candidate, 'content') and candidate.content.parts:
                        for part in candidate.content.parts:
                            if hasattr(part, 'text') and part.text:
                                return part.text.strip()

            return ""
        except Exception:
            return ""


class OpenAIProvider(BaseModelProvider):
    """OpenAI-compatible provider."""

    def __init__(self, config: ModelConfig, retry_config: Optional[RetryConfig] = None):
        super().__init__(config, retry_config)
        self._import_dependencies()

    def _import_dependencies(self):
        """Import required dependencies."""
        try:
            from openai import OpenAI
            self.OpenAI = OpenAI
        except ImportError:
            raise ConfigError("openai not installed. Install with: pip install openai")

    def generate(self, messages: List[Dict[str, str]], **kwargs) -> ModelResponse:
        """Generate response using OpenAI."""
        try:
            api_key = os.environ.get(self.config.api_key_env or "OPENAI_API_KEY")
            client = self.OpenAI(api_key=api_key)

            response = client.chat.completions.create(
                model=self.config.model_name,
                messages=messages,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                **kwargs
            )

            if response.choices:
                text = response.choices[0].message.content or ""
                usage = {
                    "prompt_tokens": getattr(response.usage, 'prompt_tokens', 0) if response.usage else 0,
                    "completion_tokens": getattr(response.usage, 'completion_tokens', 0) if response.usage else 0,
                }
                return ModelResponse(
                    text=text.strip(),
                    usage=usage,
                    metadata={"model": self.config.model_name}
                )

            return ModelResponse(text="", error="No response choices")

        except Exception as e:
            return ModelResponse(text="", error=str(e))

    def generate_with_retry(self, messages: List[Dict[str, str]], **kwargs) -> ModelResponse:
        """Generate with retry logic."""
        last_response = None

        for attempt in range(self.retry_config.max_retries + 1):
            response = self.generate(messages, **kwargs)
            last_response = response

            if response.error:
                if attempt < self.retry_config.max_retries:
                    is_rate_limited = self._is_rate_limit_error(Exception(response.error))
                    self._sleep_with_backoff(attempt, is_rate_limited)
                    continue
            else:
                return response

        return last_response or ModelResponse(text="", error="Max retries exceeded")


class OpenRouterProvider(BaseModelProvider):
    """OpenRouter provider using OpenAI-compatible SDK and endpoint."""

    def __init__(self, config: ModelConfig, retry_config: Optional[RetryConfig] = None):
        super().__init__(config, retry_config)
        self._import_dependencies()

    def _import_dependencies(self):
        try:
            from openai import OpenAI
            self.OpenAI = OpenAI
        except ImportError:
            raise ConfigError("openai not installed. Install with: pip install openai")

    def generate(self, messages: List[Dict[str, str]], **kwargs) -> ModelResponse:
        try:
            api_key = os.environ.get(self.config.api_key_env or "OPENROUTER_API_KEY")
            
            # Use OpenAI SDK against OpenRouter's OpenAI-compatible endpoint
            # Set reasonable timeout (default 60s for long-context models like Grok)
            timeout = kwargs.pop('timeout', 120.0)  # 2 minutes for large context
            
            client = self.OpenAI(
                api_key=api_key, 
                base_url="https://openrouter.ai/api/v1",
                timeout=timeout,
                max_retries=0  # We handle retries ourselves
            )

            response = client.chat.completions.create(
                model=self.config.model_name,
                messages=messages,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                **kwargs
            )

            if response.choices:
                text = response.choices[0].message.content or ""
                usage = {
                    "prompt_tokens": getattr(response.usage, 'prompt_tokens', 0) if response.usage else 0,
                    "completion_tokens": getattr(response.usage, 'completion_tokens', 0) if response.usage else 0,
                }
                return ModelResponse(
                    text=text.strip(),
                    usage=usage,
                    metadata={"model": self.config.model_name}
                )

            return ModelResponse(text="", error="No response choices")

        except Exception as e:
            # Capture the full error for better debugging
            error_msg = str(e)
            if hasattr(e, '__class__'):
                error_msg = f"{e.__class__.__name__}: {error_msg}"
            return ModelResponse(text="", error=error_msg)

    def generate_with_retry(self, messages: List[Dict[str, str]], **kwargs) -> ModelResponse:
        import logging
        logger = logging.getLogger(__name__)
        
        last_response = None

        for attempt in range(self.retry_config.max_retries + 1):
            response = self.generate(messages, **kwargs)
            last_response = response

            if response.error:
                if attempt < self.retry_config.max_retries:
                    is_rate_limited = self._is_rate_limit_error(Exception(response.error))
                    delay = self.retry_config.calculate_delay(attempt, is_rate_limited)
                    
                    # Log retry with details
                    error_type = "rate limit" if is_rate_limited else "API error"
                    logger.warning(
                        f"OpenRouter {error_type} on attempt {attempt + 1}/{self.retry_config.max_retries + 1}: "
                        f"{response.error[:100]}... Retrying in {delay:.1f}s"
                    )
                    
                    self._sleep_with_backoff(attempt, is_rate_limited)
                    continue
                else:
                    # Max retries reached
                    logger.error(
                        f"OpenRouter API failed after {self.retry_config.max_retries + 1} attempts: "
                        f"{response.error[:100]}"
                    )
            else:
                # Success
                if attempt > 0:
                    logger.info(f"OpenRouter API succeeded on attempt {attempt + 1}")
                return response

        return last_response or ModelResponse(text="", error="Max retries exceeded")


class AnthropicBedrockProvider(BaseModelProvider):
    """Anthropic Bedrock provider."""

    def __init__(self, config: ModelConfig, retry_config: Optional[RetryConfig] = None):
        super().__init__(config, retry_config)
        self._import_dependencies()

    def _import_dependencies(self):
        """Import required dependencies."""
        try:
            from anthropic import AnthropicBedrock
            self.AnthropicBedrock = AnthropicBedrock
        except ImportError:
            raise ConfigError("anthropic[bedrock] not installed. Install with: pip install 'anthropic[bedrock]'")

    def generate(self, messages: List[Dict[str, str]], **kwargs) -> ModelResponse:
        """Generate response using Anthropic Bedrock."""
        try:
            # Get AWS credentials
            region = os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION")
            if not region:
                raise ConfigError("AWS_REGION or AWS_DEFAULT_REGION not set")

            client = self.AnthropicBedrock(aws_region=region)

            # Convert messages to Anthropic format
            system_prompt = ""
            anthropic_messages = []

            for msg in messages:
                if msg["role"] == "system":
                    system_prompt = msg["content"]
                else:
                    anthropic_messages.append({
                        "role": msg["role"],
                        "content": msg["content"]
                    })

            # Prepare request
            request_kwargs = {
                "model": self.config.model_name,
                "max_tokens": self.config.max_tokens,
                "temperature": self.config.temperature,
                "messages": anthropic_messages,
            }

            if system_prompt:
                request_kwargs["system"] = system_prompt

            # Generate response
            response = client.messages.create(**request_kwargs)

            # Extract text from response
            text_parts = []
            for content in response.content:
                if content.type == "text":
                    text_parts.append(content.text)

            text = " ".join(text_parts).strip()

            return ModelResponse(
                text=text,
                usage={"prompt_tokens": 0, "completion_tokens": 0},  # Not available in response
                metadata={"model": self.config.model_name}
            )

        except Exception as e:
            return ModelResponse(text="", error=str(e))

    def generate_with_retry(self, messages: List[Dict[str, str]], **kwargs) -> ModelResponse:
        """Generate with retry logic."""
        last_response = None

        for attempt in range(self.retry_config.max_retries + 1):
            response = self.generate(messages, **kwargs)
            last_response = response

            if response.error:
                if attempt < self.retry_config.max_retries:
                    is_rate_limited = self._is_rate_limit_error(Exception(response.error))
                    self._sleep_with_backoff(attempt, is_rate_limited)
                    continue
            else:
                return response

        return last_response or ModelResponse(text="", error="Max retries exceeded")


class AgentProvider(BaseModelProvider):
    """Base class for CLI-based coding agents."""

    def __init__(self, config: AgentConfig, retry_config: Optional[RetryConfig] = None):
        # Convert AgentConfig to ModelConfig for base class
        model_config = ModelConfig(
            provider=config.provider,
            model_name=config.model_name,
            api_key_env=config.api_key_env,
            max_tokens=config.max_tokens,
            temperature=config.temperature,
            thinking_budget=config.thinking_budget,
            provider_kwargs=config.provider_kwargs
        )
        super().__init__(model_config, retry_config)
        self.agent_config = config
        self.extractor = FinancialAnswerExtractor()

    def generate(self, messages: List[Dict[str, str]], **kwargs) -> ModelResponse:
        """Generate response using CLI agent."""
        workspace = self._create_workspace()

        try:
            # Prepare context file
            context_content = self._prepare_context(messages)
            context_file_path = self._write_context_file(context_content, workspace)

            # Optionally print context path/contents
            if getattr(self.agent_config, 'print_context', False):
                try:
                    print(f"DEBUG: context.md path: {context_file_path}")
                    sample = context_content[:1000]
                    print(f"DEBUG: context.md head (first 1000 chars):\n{sample}")
                except Exception:
                    pass

            # Build prompt with explicit context reference
            prompt = self._build_agent_prompt(messages)

            # Execute CLI command
            result = self._execute_cli_command(prompt, workspace)

            if result.returncode == 0:
                # By default, disable extractor and use raw CLI stdout as prediction
                # Only enable extractor if explicitly set to True in config
                disable_extractor = True  # Default to disabled
                try:
                    # Check if explicitly enabled in config (enable_extractor: true)
                    enable_extractor = bool((self.agent_config.provider_kwargs or {}).get("enable_extractor", False))
                    disable_extractor = not enable_extractor
                except Exception:
                    disable_extractor = True  # Default to disabled on error

                if disable_extractor:
                    prediction_text = result.stdout
                    metadata = {
                        "agent": self.agent_config.agent_name,
                        "model": self.agent_config.model_name,
                        "workspace": str(workspace),
                        "full_stdout": result.stdout,
                        "extraction": "disabled"
                    }
                else:
                    # Extract numerical answer for scoring convenience
                    extracted = self.extractor.extract_answer(result.stdout)
                    prediction_text = extracted or ""
                    metadata = {
                        "agent": self.agent_config.agent_name,
                        "model": self.agent_config.model_name,
                        "workspace": str(workspace),
                        "full_stdout": result.stdout,
                        "extracted_answer": extracted
                    }

                return ModelResponse(
                    text=prediction_text,
                    metadata=metadata
                )
            else:
                return ModelResponse(
                    text="",
                    error=f"CLI command failed (return code {result.returncode}): {result.stderr}",
                    metadata={"agent": self.agent_config.agent_name, "cmd": " ".join(result.args) if hasattr(result, 'args') else 'unknown', "workspace": str(workspace)}
                )

        except Exception as e:
            return ModelResponse(
                text="",
                error=str(e),
                metadata={"agent": self.agent_config.agent_name, "workspace": str(workspace)}
            )
        finally:
            # Clean up workspace unless flagged to keep
            if not getattr(self.agent_config, 'keep_workspace', False):
                self._cleanup_workspace(workspace)

    def generate_with_retry(self, messages: List[Dict[str, str]], **kwargs) -> ModelResponse:
        """Generate with retry logic for agents."""
        last_response = None

        for attempt in range(self.retry_config.max_retries + 1):
            response = self.generate(messages, **kwargs)
            last_response = response

            if response.error:
                if attempt < self.retry_config.max_retries:
                    self._sleep_with_backoff(attempt, False)  # Agents don't have rate limits
                    continue
            else:
                return response

        return last_response or ModelResponse(text="", error="Max retries exceeded")

    def _create_workspace(self) -> Path:
        """Create temporary workspace for agent."""
        # If a fixed workspace_dir is provided, create a new subdirectory inside it
        base_dir = Path(self.agent_config.workspace_dir or "temp")
        try:
            base_dir.mkdir(parents=True, exist_ok=True)
            workspace = Path(tempfile.mkdtemp(prefix=f"{self.agent_config.agent_name}_", dir=str(base_dir)))
        except Exception:
            # Fallback to system tmp
            workspace = Path(tempfile.mkdtemp(prefix=f"{self.agent_config.agent_name}_"))
        return workspace

    def _cleanup_workspace(self, workspace: Path):
        """Clean up workspace after execution."""
        try:
            shutil.rmtree(workspace)
        except Exception:
            pass  # Ignore cleanup errors

    def _prepare_context(self, messages: List[Dict[str, str]]) -> str:
        """Prepare context content from messages."""
        context_parts = []

        for msg in messages:
            if msg["role"] == "system":
                context_parts.append(f"# Context\n\n{msg['content']}")
            # Note: For TAT-QA and similar datasets, the question is already included
            # in the system message context, so we don't add it again from user message

        return "\n\n".join(context_parts)

    def _write_context_file(self, context_content: str, workspace: Path):
        """Write context to context.md file."""
        context_file = workspace / "context.md"
        context_file.write_text(context_content)
        return context_file

    def _build_agent_prompt(self, messages: List[Dict[str, str]]) -> str:
        """Build prompt for the agent."""
        # Extract question from messages
        question = ""
        for msg in messages:
            if msg["role"] == "user":
                content = msg["content"]
                if "Question:" in content:
                    question = content.split("Question:")[1].split("Answer:")[0].strip()
                else:
                    question = content
                break

        # Detect task type and use appropriate prompt wrapper
        # For tagging/classification tasks, use minimal wrapper
        if any(keyword in question.lower() for keyword in ["tag each token", "classify", "label each", "annotate"]):
            # Tagging/classification task - use question as-is with minimal context reference
            return f"Using the data in context.md, {question}"
        else:
            # Financial QA task - use full wrapper with calculation instruction
            return f"""Based on the financial data in context.md, {question}

Please provide a clear numerical answer with your calculation."""

    def _execute_cli_command(self, prompt: str, workspace: Path) -> subprocess.CompletedProcess:
        """Execute the CLI command. To be overridden by specific agent implementations."""
        raise NotImplementedError("Subclasses must implement _execute_cli_command")


class ClaudeCodeProvider(AgentProvider):
    """Claude Code CLI agent provider."""

    def _parse_json_result(self, stdout: str) -> str:
        """Parse Claude Code JSON output and extract the result field.
        
        Claude Code outputs JSON with structure:
        {"type":"result","subtype":"success","result":"<answer>", ...}
        
        Returns the result field content for judge evaluation.
        """
        import json
        try:
            data = json.loads(stdout.strip())
            if isinstance(data, dict):
                result = data.get("result", "")
                return str(result).strip()
        except json.JSONDecodeError:
            pass
        
        # Fallback to raw stdout if not valid JSON
        return stdout

    def generate(self, messages: List[Dict[str, str]], **kwargs) -> ModelResponse:
        """Override generate to parse JSON output and set judge_input."""
        response = super().generate(messages, **kwargs)

        # Parse JSON output and set judge_input for the LLM judge
        if not response.error and response.metadata and "full_stdout" in response.metadata:
            full_stdout = response.metadata["full_stdout"]
            # Extract the "result" field from Claude Code's JSON response
            parsed_result = self._parse_json_result(full_stdout)
            # Set judge_input to the parsed result (for judge evaluation)
            response.metadata["judge_input"] = parsed_result
            # Also update the main response text if extractor is disabled
            if response.metadata.get("extraction") == "disabled":
                response.text = parsed_result

        return response

    def _execute_cli_command(self, prompt: str, workspace: Path) -> subprocess.CompletedProcess:
        """Execute Claude Code CLI command."""
        # Build command based on agent configuration
        cmd = [self.agent_config.cli_command or "claude"]

        # Add model specification only if provided
        if self.agent_config.model_name:
            cmd.extend(["--model", self.agent_config.model_name])

        # Add configured CLI arguments
        cmd.extend(self.agent_config.cli_args or ["--output-format", "json"])
        
        # Skip permissions for automated usage in trusted workspace
        cmd.append("--dangerously-skip-permissions")

        # Add the prompt with -p flag
        cmd.extend(["-p", prompt])

        # Debug: log the command being executed
        print(f"DEBUG: Executing command: {' '.join(cmd)}")
        print(f"DEBUG: Working directory: {workspace}")
        print(f"DEBUG: Prompt: {prompt[:100]}...")

        # Execute command in workspace
        try:
            result = subprocess.run(
                cmd,
                cwd=workspace,
                capture_output=True,
                text=True,
                timeout=self.agent_config.timeout
            )
            print(f"DEBUG: Return code: {result.returncode}")
            print(f"DEBUG: Stdout: {result.stdout[:200]}...")
            print(f"DEBUG: Stderr: {result.stderr[:200]}...")
            return result
        except subprocess.TimeoutExpired:
            # Create a mock result for timeout
            result = subprocess.CompletedProcess(
                args=cmd,
                returncode=1,
                stdout="",
                stderr=f"Command timed out after {self.agent_config.timeout} seconds"
            )
            return result


class GeminiCLIProvider(AgentProvider):
    """Gemini CLI agent provider."""

    def _parse_json_output(self, stdout: str) -> tuple[str, dict]:
        """Parse JSON output from Gemini CLI.
        
        Returns:
            tuple: (response_text, stats_dict)
        """
        import json
        try:
            data = json.loads(stdout.strip())
            if isinstance(data, dict):
                response = data.get("response", "")
                stats = data.get("stats", {})
                error = data.get("error")
                
                if error:
                    error_msg = error.get("message", str(error))
                    return f"ERROR: {error_msg}", stats
                
                return str(response), stats
        except json.JSONDecodeError:
            # Fallback to raw stdout if not valid JSON
            pass
        
        return stdout, {}

    def generate(self, messages: List[Dict[str, str]], **kwargs) -> ModelResponse:
        """Override generate to parse JSON output with retry logic for API errors."""
        import time
        
        max_retries = 3
        retry_delay = 5  # seconds
        
        for attempt in range(max_retries):
            response = super().generate(messages, **kwargs)
            
            # Parse JSON output if available
            if not response.error and response.metadata and "full_stdout" in response.metadata:
                full_stdout = response.metadata["full_stdout"]
                parsed_response, stats = self._parse_json_output(full_stdout)
                
                # Check if response is empty (API error)
                if not parsed_response or parsed_response.strip() == "":
                    # Check if this was a Gemini API error (check all models in stats)
                    has_api_error = False
                    if stats and "models" in stats:
                        for model_stats in stats.get("models", {}).values():
                            if model_stats.get("api", {}).get("totalErrors", 0) > 0:
                                has_api_error = True
                                break
                    
                    if has_api_error:
                        if attempt < max_retries - 1:
                            wait_time = retry_delay * (2 ** attempt)  # Exponential backoff
                            print(f"WARNING: Gemini API error (empty response), retrying in {wait_time}s... (attempt {attempt + 1}/{max_retries})")
                            time.sleep(wait_time)
                            continue
                        else:
                            # Last attempt failed, mark as error
                            response.error = "Gemini API error: empty response after retries"
                            response.text = ""
                            return response
                
                # Update response text with parsed content
                response.text = parsed_response
                
                # Add stats to metadata
                if stats:
                    response.metadata["gemini_stats"] = stats
            
            return response
        
        # Should not reach here, but return last response if it does
        return response

    def _execute_cli_command(self, prompt: str, workspace: Path) -> subprocess.CompletedProcess:
        """Execute Gemini CLI command."""
        # Build command based on agent configuration
        cmd = [self.agent_config.cli_command or "gemini"]

        # Add model specification only if provided
        if self.agent_config.model_name:
            cmd.extend(["-m", self.agent_config.model_name])

        # Add configured CLI arguments
        cmd.extend(self.agent_config.cli_args or [])

        # Add the prompt with -p flag
        cmd.extend(["-p", prompt])

        # Debug: log the command being executed
        print(f"DEBUG: Executing command: {' '.join(cmd)}")
        print(f"DEBUG: Working directory: {workspace}")
        print(f"DEBUG: Prompt: {prompt[:100]}...")

        # Execute command in workspace
        try:
            result = subprocess.run(
                cmd,
                cwd=workspace,
                capture_output=True,
                text=True,
                timeout=self.agent_config.timeout
            )
            print(f"DEBUG: Return code: {result.returncode}")
            print(f"DEBUG: Stdout: {result.stdout[:200]}...")
            print(f"DEBUG: Stderr: {result.stderr[:200]}...")
            return result
        except subprocess.TimeoutExpired:
            # Create a mock result for timeout
            result = subprocess.CompletedProcess(
                args=cmd,
                returncode=1,
                stdout="",
                stderr=f"Command timed out after {self.agent_config.timeout} seconds"
            )
            return result


class CodexCLIProvider(AgentProvider):
    """Codex CLI agent provider."""

    def _extract_agent_messages(self, stdout: str) -> str:
        """Extract agent messages from codex-cli JSON output.

        Codex outputs JSON lines with different message types. We extract
        only substantial reasoning to reduce token count for judge.
        
        New format (current):
        - {"type": "item.completed", "item": {"type": "reasoning", "text": "..."}}
        - {"type": "result", "content": "..."}
        
        Old format (legacy):
        - {"msg": {"type": "agent_message", "message": "..."}}
        """
        import json
        import logging
        
        logger = logging.getLogger(__name__)
        agent_messages = []
        reasoning_items = []
        final_result = None

        for line in stdout.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                if not isinstance(data, dict):
                    continue
                
                # New format: Extract items
                if data.get("type") == "item.completed":
                    item = data.get("item", {})
                    
                    # Agent messages - THE ACTUAL ANSWER (highest priority)
                    if item.get("type") == "agent_message":
                        text = item.get("text", "")
                        if text:
                            agent_messages.append(text)
                    
                    # Reasoning text (for context if no agent_message)
                    elif item.get("type") == "reasoning":
                        text = item.get("text", "")
                        # Skip simple status updates like "**Opening file**"
                        # Keep only substantial reasoning (>100 chars or contains calculations)
                        if text and (len(text) > 100 or any(char in text for char in ['=', '+', '-', '*', '/', '%'])):
                            reasoning_items.append(text)
                
                # New format: Extract final result
                elif data.get("type") == "result":
                    content = data.get("content", "")
                    if content:
                        final_result = content
                
                # Old format: Extract agent_message (legacy support)
                elif "msg" in data:
                    msg = data["msg"]
                    if msg.get("type") == "agent_message":
                        message_text = msg.get("message", "")
                        if message_text:
                            agent_messages.append(message_text)
                            
            except json.JSONDecodeError:
                # Skip lines that aren't valid JSON
                continue

        # Build the extracted message - prioritize agent_messages (the actual answer)
        extracted_parts = []
        
        # Priority 1: Agent messages (the actual answer from Codex)
        if agent_messages:
            extracted_parts.extend(agent_messages)  # Keep all agent messages
        # Priority 2: Final result (if present)
        elif final_result:
            extracted_parts.append(f"**Final Answer:**\n{final_result}")
        # Priority 3: Reasoning items as fallback
        elif reasoning_items:
            extracted_parts.extend(reasoning_items[-5:])  # Keep last 5 reasoning items only
        
        if extracted_parts:
            result = "\n\n".join(extracted_parts)
            logger.info(f"Extracted {len(agent_messages)} agent messages, {len(reasoning_items)} reasoning items ({len(result)} chars total)")
            return result
        else:
            # Fallback: return truncated stdout if nothing extracted
            logger.warning("No substantial content extracted from codex output, returning truncated stdout")
            return stdout[:5000] + "\n\n[... output truncated to 5000 chars ...]" if len(stdout) > 5000 else stdout

    def generate(self, messages: List[Dict[str, str]], **kwargs) -> ModelResponse:
        """Override generate to extract and optimize for judge (Codex-specific optimization)."""
        response = super().generate(messages, **kwargs)

        # Codex-specific: Extract agent messages for judge evaluation (reduces token count)
        if not response.error and response.metadata and "full_stdout" in response.metadata:
            full_stdout = response.metadata["full_stdout"]
            # Extract the important parts
            extracted_content = self._extract_agent_messages(full_stdout)
            response.metadata["judge_input"] = extracted_content
            # Also update the prediction text to the extracted content
            response.text = extracted_content
            # Remove full_stdout to save space (Codex outputs are huge)
            del response.metadata["full_stdout"]

        return response

    def _execute_cli_command(self, prompt: str, workspace: Path) -> subprocess.CompletedProcess:
        """Execute Codex CLI command."""
        # Build command based on agent configuration
        cmd = [self.agent_config.cli_command or "codex", "exec"]

        # Add model specification only if provided
        if self.agent_config.model_name:
            cmd.extend(["-m", self.agent_config.model_name])

        # Add configured CLI arguments
        cmd.extend(self.agent_config.cli_args or [])

        # Add the prompt as the last argument
        cmd.append(prompt)

        # Debug: log the command being executed
        print(f"DEBUG: Executing command: {' '.join(cmd)}")
        print(f"DEBUG: Working directory: {workspace}")
        print(f"DEBUG: Prompt: {prompt}")

        # Execute command in workspace
        try:
            result = subprocess.run(
                cmd,
                cwd=workspace,
                capture_output=True,
                text=True,
                timeout=self.agent_config.timeout
            )
            print(f"DEBUG: Return code: {result.returncode}")
            print(f"DEBUG: Stdout: {result.stdout}")
            print(f"DEBUG: Stderr: {result.stderr}")
            return result
        except subprocess.TimeoutExpired:
            # Create a mock result for timeout
            result = subprocess.CompletedProcess(
                args=cmd,
                returncode=1,
                stdout="",
                stderr=f"Command timed out after {self.agent_config.timeout} seconds"
            )
            return result


class TraeAgentProvider(AgentProvider):
    """Trae Agent CLI provider."""

    def _extract_execution_summary(self, stdout: str) -> str:
        """Extract Execution Summary section onwards from trae-cli output.

        This reduces token count for LLM judge by removing verbose tool usage logs
        and keeping only the final result section.
        """
        marker = "Execution Summary"
        if marker in stdout:
            # Find the marker and extract everything from there onwards
            idx = stdout.find(marker)
            return stdout[idx:]
        # Fallback to full stdout if marker not found
        return stdout

    def generate(self, messages: List[Dict[str, str]], **kwargs) -> ModelResponse:
        """Override generate to add trae-cli specific extraction for judge."""
        # Call parent generate method
        response = super().generate(messages, **kwargs)

        # If successful, add extracted summary for judge
        if not response.error and response.metadata and "full_stdout" in response.metadata:
            full_stdout = response.metadata["full_stdout"]
            # Extract just the execution summary section for judge
            response.metadata["judge_input"] = self._extract_execution_summary(full_stdout)

        return response

    def _create_trae_config(self, config_file: Path) -> None:
        """Create a minimal trae config file."""
        config_content = """agents:
    trae_agent:
        enable_lakeview: true
        model: trae_agent_model
        max_steps: 40
        tools:
            - bash
            - str_replace_based_edit_tool
            - json_edit_tool
            - todo_write
            - task_done

lakeview:
    model: lakeview_model

model_providers:
    anthropic:
        api_key: ${ANTHROPIC_API_KEY}
        provider: anthropic
    openai:
        api_key: ${OPENAI_API_KEY}
        provider: openai

models:
    trae_agent_model:
        model_provider: anthropic
        model: claude-sonnet-4-20250514
        max_tokens: 4096
        temperature: 0.5
        top_p: 1
        top_k: 0
        max_retries: 10
        parallel_tool_calls: true
    lakeview_model:
        model_provider: anthropic
        model: claude-sonnet-4-20250514
        max_tokens: 4096
        temperature: 0.5
        top_p: 1
        top_k: 0
        max_retries: 10
        parallel_tool_calls: true
"""
        config_file.write_text(config_content)

    def _execute_cli_command(self, prompt: str, workspace: Path) -> subprocess.CompletedProcess:
        """Execute Trae Agent CLI command."""
        # Create a minimal config file for trae-cli if not exists
        config_file = workspace / "trae_config.yaml"
        if not config_file.exists():
            self._create_trae_config(config_file)

        # Build command based on agent configuration
        cmd = [self.agent_config.cli_command or "trae-cli", "run"]

        # Add config file
        cmd.extend(["--config-file", str(config_file)])

        # Add model specification only if provided
        if self.agent_config.model_name:
            cmd.extend(["-m", self.agent_config.model_name])

        # Add working directory
        cmd.extend(["-w", str(workspace)])

        # Add configured CLI arguments
        cmd.extend(self.agent_config.cli_args or [])

        # Add the prompt as the task argument
        cmd.append(prompt)

        # Debug: log the command being executed
        print(f"DEBUG: Executing command: {' '.join(cmd)}")
        print(f"DEBUG: Working directory: {workspace}")
        print(f"DEBUG: Prompt: {prompt}")

        # Execute command in workspace
        try:
            result = subprocess.run(
                cmd,
                cwd=workspace,
                capture_output=True,
                text=True,
                timeout=self.agent_config.timeout
            )
            print(f"DEBUG: Return code: {result.returncode}")
            print(f"DEBUG: Stdout: {result.stdout}")
            print(f"DEBUG: Stderr: {result.stderr}")
            return result
        except subprocess.TimeoutExpired:
            # Create a mock result for timeout
            result = subprocess.CompletedProcess(
                args=cmd,
                returncode=1,
                stdout="",
                stderr=f"Command timed out after {self.agent_config.timeout} seconds"
            )
            return result


class ModelProviderFactory:
    """Factory for creating model providers."""

    _providers = {
        ModelProvider.GEMINI: GeminiProvider,
        ModelProvider.OPENAI: OpenAIProvider,
        ModelProvider.OPENROUTER: OpenRouterProvider,
        ModelProvider.ANTHROPIC_BEDROCK: AnthropicBedrockProvider,
    }

    _agent_providers = {
        "claude-code": ClaudeCodeProvider,
        "gemini-cli": GeminiCLIProvider,
        "codex-cli": CodexCLIProvider,
        "trae-agent": TraeAgentProvider,
        # Future agents can be added here
        # "aider": AiderProvider,
        # "cursor": CursorProvider,
    }

    @classmethod
    def create_provider(cls, config: Union[ModelConfig, AgentConfig], retry_config: Optional[RetryConfig] = None) -> BaseModelProvider:
        """Create a model provider instance."""
        if config.provider == ModelProvider.AGENT:
            # Handle agent providers
            if not isinstance(config, AgentConfig):
                raise ConfigError("AGENT provider requires AgentConfig")

            agent_name = config.agent_name
            if agent_name not in cls._agent_providers:
                raise ConfigError(f"No agent provider available for: {agent_name}")

            provider_class = cls._agent_providers[agent_name]
            return provider_class(config, retry_config)

        else:
            # Handle standard LLM providers
            if config.provider not in cls._providers:
                raise ConfigError(f"No provider available for: {config.provider}")

            provider_class = cls._providers[config.provider]
            return provider_class(config, retry_config)

    @classmethod
    def get_supported_providers(cls) -> List[ModelProvider]:
        """Get list of supported providers."""
        return list(cls._providers.keys()) + [ModelProvider.AGENT]

    @classmethod
    def get_supported_agents(cls) -> List[str]:
        """Get list of supported agent names."""
        return list(cls._agent_providers.keys())
