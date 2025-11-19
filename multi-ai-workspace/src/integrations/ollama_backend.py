"""Ollama local LLM backend implementation for offline AI."""

import time
import httpx
from typing import Optional, Dict, Any, AsyncIterator, List
import asyncio

from ..core.backend import AIBackend, AIProvider, Capabilities, Context, Response
from ..utils.logger import get_logger

logger = get_logger(__name__)


class OllamaBackend(AIBackend):
    """
    Ollama backend implementation for local, offline AI.

    Supports running local LLMs like Mistral, Mixtral, Llama, CodeLlama, etc.
    Completely offline - no API keys or internet required.

    Features:
    - Mistral 7B (small, fast, ~4GB RAM)
    - Mixtral 8x7B MoE (~26GB RAM, ~47B params but ~13B active)
    - Custom model loading
    - Streaming support
    - Local privacy (no data leaves your machine)
    """

    # Model configurations
    MODELS = {
        "mistral": {
            "id": "mistral:7b",
            "display_name": "Mistral 7B",
            "params": "7B",
            "ram_required": "4-8GB",
            "description": "Small, fast Mistral model - great for offline use",
            "context_length": 8192
        },
        "mistral-small": {
            "id": "mistral:7b-instruct",
            "display_name": "Mistral 7B Instruct",
            "params": "7B",
            "ram_required": "4-8GB",
            "description": "Instruction-tuned Mistral - better for tasks",
            "context_length": 8192
        },
        "mixtral": {
            "id": "mixtral:8x7b",
            "display_name": "Mixtral 8x7B",
            "params": "47B (13B active)",
            "ram_required": "24-32GB",
            "description": "Mixtral MoE - powerful offline alternative",
            "context_length": 32768
        },
        "mixtral-instruct": {
            "id": "mixtral:8x7b-instruct-v0.1-q4_0",
            "display_name": "Mixtral 8x7B Instruct Q4",
            "params": "47B (13B active)",
            "ram_required": "20-26GB",
            "description": "Quantized Mixtral for lower RAM usage",
            "context_length": 32768
        },
        "codestral": {
            "id": "codestral:22b",
            "display_name": "Codestral 22B",
            "params": "22B",
            "ram_required": "16-20GB",
            "description": "Code-specialized Mistral model",
            "context_length": 32768
        },
        "llama3": {
            "id": "llama3:8b",
            "display_name": "Llama 3 8B",
            "params": "8B",
            "ram_required": "4-8GB",
            "description": "Meta's Llama 3 - good general model",
            "context_length": 8192
        }
    }

    def __init__(
        self,
        model: str = "mistral",
        name: str = "Ollama",
        base_url: str = "http://localhost:11434",
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize Ollama backend.

        Args:
            model: Model alias (mistral, mixtral, etc.) or full model ID
            name: Backend display name
            base_url: Ollama API URL (default: http://localhost:11434)
            config: Additional configuration options
        """
        # Resolve model ID
        if model in self.MODELS:
            self.model_info = self.MODELS[model]
            self.model_id = self.model_info["id"]
            self.model_alias = model
        else:
            # Custom model ID
            self.model_id = model
            self.model_alias = model
            self.model_info = {
                "id": model,
                "display_name": model,
                "description": "Custom Ollama model"
            }

        self.base_url = base_url.rstrip('/')
        self.name = name
        self.config = config or {}

        # HTTP client for Ollama API
        self.client = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=httpx.Timeout(300.0)  # 5 min timeout for large models
        )

        # Provider metadata
        self._provider = AIProvider.CUSTOM
        self._capabilities = Capabilities(
            streaming=True,
            function_calling=False,
            vision=False,
            tools=False
        )

        logger.info(
            f"Initialized Ollama backend: {self.model_info.get('display_name', self.model_id)} "
            f"(URL: {base_url})"
        )

    @property
    def provider(self) -> AIProvider:
        """Return provider type."""
        return self._provider

    @property
    def capabilities(self) -> Capabilities:
        """Return backend capabilities."""
        return self._capabilities

    async def check_health(self) -> bool:
        """
        Check if Ollama server is running and model is available.

        Returns:
            True if server is healthy and model exists
        """
        try:
            # Check server health
            response = await self.client.get("/api/tags")
            if response.status_code != 200:
                logger.warning(f"Ollama server returned status {response.status_code}")
                return False

            # Check if model is available
            data = response.json()
            available_models = [m.get("name", "") for m in data.get("models", [])]

            # Check if our model is in the list
            model_available = any(
                self.model_id in model or model.startswith(self.model_id.split(":")[0])
                for model in available_models
            )

            if not model_available:
                logger.warning(
                    f"Model {self.model_id} not found. Available models: {available_models}"
                )
                logger.info(f"Pull model with: ollama pull {self.model_id}")
                return False

            return True

        except httpx.ConnectError:
            logger.error(f"Cannot connect to Ollama at {self.base_url}")
            logger.info("Start Ollama with: ollama serve")
            return False
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return False

    async def send_message(
        self,
        prompt: str,
        context: Optional[Context] = None,
        **kwargs
    ) -> Response:
        """
        Send a message to Ollama and get response.

        Args:
            prompt: User prompt
            context: Optional context (conversation history, system prompt, etc.)
            **kwargs: Additional Ollama API parameters

        Returns:
            Response object with generated text
        """
        start_time = time.time()

        try:
            # Build messages array
            messages = []

            # Add system prompt if provided
            if context and context.system_prompt:
                messages.append({
                    "role": "system",
                    "content": context.system_prompt
                })

            # Add conversation history
            if context and context.conversation_history:
                for msg in context.conversation_history:
                    messages.append({
                        "role": msg.get("role", "user"),
                        "content": msg.get("content", "")
                    })

            # Add current prompt
            messages.append({
                "role": "user",
                "content": prompt
            })

            # Prepare request payload
            payload = {
                "model": self.model_id,
                "messages": messages,
                "stream": False,
                "options": {
                    "temperature": self.config.get("temperature", 0.7),
                    "num_predict": self.config.get("max_tokens", 2048),
                    "top_p": self.config.get("top_p", 0.9),
                    "top_k": self.config.get("top_k", 40),
                }
            }

            # Add any additional kwargs
            if kwargs:
                payload["options"].update(kwargs)

            # Make request to Ollama
            response = await self.client.post("/api/chat", json=payload)
            response.raise_for_status()

            data = response.json()

            # Extract response
            message = data.get("message", {})
            content = message.get("content", "")

            # Calculate metrics
            latency = time.time() - start_time

            # Extract token usage (if available)
            eval_count = data.get("eval_count", 0)
            prompt_eval_count = data.get("prompt_eval_count", 0)

            return Response(
                content=content,
                provider=self._provider,
                model=self.model_id,
                latency=latency,
                tokens_used=eval_count + prompt_eval_count,
                metadata={
                    "backend": self.name,
                    "model_info": self.model_info,
                    "eval_count": eval_count,
                    "prompt_eval_count": prompt_eval_count,
                    "eval_duration": data.get("eval_duration", 0),
                    "load_duration": data.get("load_duration", 0),
                    "offline": True,
                    "local": True
                }
            )

        except httpx.HTTPStatusError as e:
            logger.error(f"Ollama HTTP error: {e}")
            raise Exception(f"Ollama request failed: {e}")
        except httpx.ConnectError:
            raise Exception(
                f"Cannot connect to Ollama at {self.base_url}. "
                "Make sure Ollama is running: ollama serve"
            )
        except Exception as e:
            logger.error(f"Error sending message to Ollama: {e}")
            raise

    async def stream_message(
        self,
        prompt: str,
        context: Optional[Context] = None,
        **kwargs
    ) -> AsyncIterator[str]:
        """
        Stream a response from Ollama token by token.

        Args:
            prompt: User prompt
            context: Optional context
            **kwargs: Additional parameters

        Yields:
            Response tokens as they're generated
        """
        try:
            # Build messages
            messages = []

            if context and context.system_prompt:
                messages.append({
                    "role": "system",
                    "content": context.system_prompt
                })

            if context and context.conversation_history:
                for msg in context.conversation_history:
                    messages.append({
                        "role": msg.get("role", "user"),
                        "content": msg.get("content", "")
                    })

            messages.append({
                "role": "user",
                "content": prompt
            })

            # Prepare streaming request
            payload = {
                "model": self.model_id,
                "messages": messages,
                "stream": True,
                "options": {
                    "temperature": self.config.get("temperature", 0.7),
                    "num_predict": self.config.get("max_tokens", 2048),
                }
            }

            # Stream response
            async with self.client.stream("POST", "/api/chat", json=payload) as response:
                response.raise_for_status()

                async for line in response.aiter_lines():
                    if not line.strip():
                        continue

                    try:
                        import json
                        data = json.loads(line)

                        if "message" in data:
                            content = data["message"].get("content", "")
                            if content:
                                yield content

                        # Check if done
                        if data.get("done", False):
                            break

                    except json.JSONDecodeError:
                        continue

        except Exception as e:
            logger.error(f"Error streaming from Ollama: {e}")
            raise

    async def list_models(self) -> List[Dict[str, Any]]:
        """
        List all available models in Ollama.

        Returns:
            List of model information dictionaries
        """
        try:
            response = await self.client.get("/api/tags")
            response.raise_for_status()

            data = response.json()
            return data.get("models", [])

        except Exception as e:
            logger.error(f"Error listing Ollama models: {e}")
            return []

    async def pull_model(self, model: str) -> bool:
        """
        Pull/download a model from Ollama registry.

        Args:
            model: Model name (e.g., "mistral:7b", "mixtral:8x7b")

        Returns:
            True if successful
        """
        try:
            logger.info(f"Pulling model {model} from Ollama registry...")

            payload = {"name": model, "stream": False}
            response = await self.client.post("/api/pull", json=payload)
            response.raise_for_status()

            logger.info(f"Successfully pulled {model}")
            return True

        except Exception as e:
            logger.error(f"Error pulling model {model}: {e}")
            return False

    async def cleanup(self):
        """Clean up resources."""
        await self.client.aclose()
        logger.info(f"Ollama backend {self.name} cleaned up")

    def get_info(self) -> Dict[str, Any]:
        """
        Get backend information.

        Returns:
            Dictionary with backend details
        """
        return {
            "name": self.name,
            "provider": self._provider.value,
            "model": self.model_id,
            "model_alias": self.model_alias,
            "model_info": self.model_info,
            "base_url": self.base_url,
            "capabilities": {
                "streaming": self._capabilities.streaming,
                "function_calling": self._capabilities.function_calling,
                "vision": self._capabilities.vision,
                "tools": self._capabilities.tools
            },
            "offline": True,
            "local": True,
            "cost": "Free (runs locally)"
        }
