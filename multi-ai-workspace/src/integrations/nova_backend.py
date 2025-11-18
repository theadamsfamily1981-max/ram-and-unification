"""Nova (OpenAI ChatGPT) AI backend implementation."""

import time
from typing import Optional, Dict, Any, AsyncIterator
from openai import AsyncOpenAI

from ..core.backend import AIBackend, AIProvider, Capabilities, Context, Response
from ..utils.logger import get_logger

logger = get_logger(__name__)


class NovaBackend(AIBackend):
    """
    Nova (OpenAI ChatGPT) backend implementation.

    Supports GPT-4, GPT-4 Turbo, and GPT-3.5 models with streaming,
    vision, and function calling capabilities.
    """

    # Model configurations
    MODELS = {
        "gpt-4o": {
            "id": "gpt-4o",
            "max_tokens": 128000,
            "supports_vision": True,
            "cost_per_1k": 0.005
        },
        "gpt-4-turbo": {
            "id": "gpt-4-turbo-preview",
            "max_tokens": 128000,
            "supports_vision": True,
            "cost_per_1k": 0.01
        },
        "gpt-4": {
            "id": "gpt-4",
            "max_tokens": 8192,
            "supports_vision": False,
            "cost_per_1k": 0.03
        },
        "gpt-3.5-turbo": {
            "id": "gpt-3.5-turbo",
            "max_tokens": 16385,
            "supports_vision": False,
            "cost_per_1k": 0.001
        }
    }

    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4o",
        name: str = "Nova",
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize Nova backend.

        Args:
            api_key: OpenAI API key
            model: Model variant or full model ID
            name: Display name
            config: Additional configuration
        """
        # Resolve model ID
        if model in self.MODELS:
            model_id = self.MODELS[model]["id"]
            self._model_config = self.MODELS[model]
        else:
            model_id = model
            self._model_config = {
                "id": model,
                "max_tokens": 128000,
                "supports_vision": False,
                "cost_per_1k": 0.005
            }

        super().__init__(
            name=name,
            provider=AIProvider.OPENAI,
            model=model_id,
            api_key=api_key,
            config=config
        )

        # Initialize OpenAI client
        self.client = AsyncOpenAI(api_key=api_key)

        logger.info(f"Nova backend initialized: {model_id}")

    async def send_message(
        self,
        prompt: str,
        context: Optional[Context] = None
    ) -> Response:
        """
        Send a message to Nova and get complete response.

        Args:
            prompt: User message
            context: Optional context

        Returns:
            Response object
        """
        start_time = time.time()
        context = context or Context()

        try:
            # Build messages
            messages = self._build_messages(prompt, context)

            # Call API
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=context.max_tokens or self.config.get("max_tokens", 4096),
                temperature=context.temperature or self.config.get("temperature", 1.0)
            )

            # Extract content
            content = response.choices[0].message.content or ""

            latency_ms = (time.time() - start_time) * 1000

            return Response(
                content=content,
                provider=AIProvider.OPENAI,
                model=self.model,
                tokens_used=response.usage.total_tokens if response.usage else None,
                latency_ms=latency_ms,
                metadata={
                    "prompt_tokens": response.usage.prompt_tokens if response.usage else None,
                    "completion_tokens": response.usage.completion_tokens if response.usage else None,
                    "finish_reason": response.choices[0].finish_reason
                }
            )

        except Exception as e:
            logger.error(f"Nova API error: {e}")
            latency_ms = (time.time() - start_time) * 1000

            return Response(
                content="",
                provider=AIProvider.OPENAI,
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
        Stream Nova response.

        Args:
            prompt: User message
            context: Optional context

        Yields:
            Response chunks
        """
        context = context or Context()

        try:
            # Build messages
            messages = self._build_messages(prompt, context)

            # Stream API call
            stream = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=context.max_tokens or self.config.get("max_tokens", 4096),
                temperature=context.temperature or self.config.get("temperature", 1.0),
                stream=True
            )

            async for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content

        except Exception as e:
            logger.error(f"Nova streaming error: {e}")
            yield f"[Error: {e}]"

    def get_capabilities(self) -> Capabilities:
        """Get Nova capabilities."""
        if self._capabilities is None:
            self._capabilities = Capabilities(
                streaming=True,
                vision=self._model_config.get("supports_vision", False),
                function_calling=True,
                max_tokens=self._model_config.get("max_tokens", 128000),
                supports_system_prompt=True,
                rate_limit_rpm=self.config.get("rate_limit_rpm", 60),
                cost_per_1k_tokens=self._model_config.get("cost_per_1k", 0.005)
            )

        return self._capabilities

    async def health_check(self) -> bool:
        """
        Check if OpenAI API is accessible.

        Returns:
            True if healthy
        """
        try:
            # Simple test message
            response = await self.send_message(
                prompt="Hi",
                context=Context(max_tokens=10)
            )
            return response.success

        except Exception as e:
            logger.error(f"Nova health check failed: {e}")
            return False

    def _build_messages(
        self,
        prompt: str,
        context: Context
    ) -> list:
        """
        Build messages array for OpenAI API.

        Args:
            prompt: Current user message
            context: Context with history

        Returns:
            Messages list
        """
        messages = []

        # Add system prompt if present
        if context.system_prompt:
            messages.append({
                "role": "system",
                "content": context.system_prompt
            })

        # Add conversation history
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

        return messages
