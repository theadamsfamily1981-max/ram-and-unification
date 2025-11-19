"""Pulse (Local Ollama) AI backend implementation."""

import time
import httpx
from typing import Optional, Dict, Any, AsyncIterator

from ..core.backend import AIBackend, AIProvider, Capabilities, Context, Response
from ..utils.logger import get_logger

logger = get_logger(__name__)


class PulseBackend(AIBackend):
    """
    Pulse (Local Ollama) backend implementation.

    Supports local LLMs via Ollama, enabling privacy-focused orchestration
    and routing without external API dependencies.
    """

    # Common Ollama models
    MODELS = {
        "llama3.2": {
            "id": "llama3.2",
            "max_tokens": 128000,
            "supports_vision": False
        },
        "llama3.1": {
            "id": "llama3.1",
            "max_tokens": 128000,
            "supports_vision": False
        },
        "mistral": {
            "id": "mistral",
            "max_tokens": 32768,
            "supports_vision": False
        },
        "phi3": {
            "id": "phi3",
            "max_tokens": 128000,
            "supports_vision": False
        }
    }

    def __init__(
        self,
        model: str = "llama3.2",
        name: str = "Pulse",
        base_url: str = "http://localhost:11434",
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize Pulse backend.

        Args:
            model: Ollama model name
            name: Display name
            base_url: Ollama server URL
            config: Additional configuration
        """
        # Resolve model config
        if model in self.MODELS:
            self._model_config = self.MODELS[model]
        else:
            self._model_config = {
                "id": model,
                "max_tokens": 128000,
                "supports_vision": False
            }

        super().__init__(
            name=name,
            provider=AIProvider.OLLAMA,
            model=model,
            api_key=None,  # Ollama doesn't use API keys
            config=config
        )

        self.base_url = base_url.rstrip("/")
        self.client = httpx.AsyncClient(timeout=120.0)

        logger.info(f"Pulse backend initialized: {model} at {base_url}")

    async def send_message(
        self,
        prompt: str,
        context: Optional[Context] = None
    ) -> Response:
        """
        Send a message to Pulse and get complete response.

        Args:
            prompt: User message
            context: Optional context

        Returns:
            Response object
        """
        start_time = time.time()
        context = context or Context()

        try:
            # Build request
            request_data = {
                "model": self.model,
                "prompt": self._build_prompt(prompt, context),
                "stream": False,
                "options": {
                    "temperature": context.temperature or self.config.get("temperature", 0.7),
                    "num_predict": context.max_tokens or self.config.get("max_tokens", 4096)
                }
            }

            # Call Ollama API
            response = await self.client.post(
                f"{self.base_url}/api/generate",
                json=request_data
            )
            response.raise_for_status()

            result = response.json()
            content = result.get("response", "")

            latency_ms = (time.time() - start_time) * 1000

            return Response(
                content=content,
                provider=AIProvider.OLLAMA,
                model=self.model,
                tokens_used=result.get("eval_count", 0),
                latency_ms=latency_ms,
                metadata={
                    "total_duration": result.get("total_duration"),
                    "load_duration": result.get("load_duration"),
                    "prompt_eval_count": result.get("prompt_eval_count"),
                    "eval_count": result.get("eval_count")
                }
            )

        except Exception as e:
            logger.error(f"Pulse API error: {e}")
            latency_ms = (time.time() - start_time) * 1000

            return Response(
                content="",
                provider=AIProvider.OLLAMA,
                model=self.model,
                latency_ms=latency_ms,
                error=str(e)
            )

    async def stream_message(
        self,
        prompt: str,
        context: Optional[Context] = None
    ) -> AsyncIterator[str]:
        """
        Stream Pulse response.

        Args:
            prompt: User message
            context: Optional context

        Yields:
            Response chunks
        """
        context = context or Context()

        try:
            # Build request
            request_data = {
                "model": self.model,
                "prompt": self._build_prompt(prompt, context),
                "stream": True,
                "options": {
                    "temperature": context.temperature or self.config.get("temperature", 0.7),
                    "num_predict": context.max_tokens or self.config.get("max_tokens", 4096)
                }
            }

            # Stream from Ollama
            async with self.client.stream(
                "POST",
                f"{self.base_url}/api/generate",
                json=request_data
            ) as response:
                async for line in response.aiter_lines():
                    if line:
                        import json
                        try:
                            chunk = json.loads(line)
                            if "response" in chunk:
                                yield chunk["response"]
                        except json.JSONDecodeError:
                            continue

        except Exception as e:
            logger.error(f"Pulse streaming error: {e}")
            yield f"[Error: {e}]"

    def get_capabilities(self) -> Capabilities:
        """Get Pulse capabilities."""
        if self._capabilities is None:
            self._capabilities = Capabilities(
                streaming=True,
                vision=self._model_config.get("supports_vision", False),
                function_calling=False,  # Ollama doesn't natively support function calling
                max_tokens=self._model_config.get("max_tokens", 128000),
                supports_system_prompt=True,
                rate_limit_rpm=None,  # Local, no rate limits
                cost_per_1k_tokens=0.0  # Free, local
            )

        return self._capabilities

    async def health_check(self) -> bool:
        """
        Check if Ollama server is accessible.

        Returns:
            True if healthy
        """
        try:
            response = await self.client.get(f"{self.base_url}/api/tags")
            return response.status_code == 200

        except Exception as e:
            logger.error(f"Pulse health check failed: {e}")
            return False

    def _build_prompt(
        self,
        prompt: str,
        context: Context
    ) -> str:
        """
        Build prompt string for Ollama.

        Args:
            prompt: Current user message
            context: Context with system prompt and history

        Returns:
            Formatted prompt string
        """
        parts = []

        # Add system prompt if present
        if context.system_prompt:
            parts.append(f"System: {context.system_prompt}\n")

        # Add conversation history
        for msg in context.conversation_history:
            role = msg.get("role", "user").capitalize()
            content = msg.get("content", "")
            parts.append(f"{role}: {content}\n")

        # Add current prompt
        parts.append(f"User: {prompt}\n")
        parts.append("Assistant:")

        return "\n".join(parts)

    async def __aenter__(self):
        """Async context manager enter."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.client.aclose()
