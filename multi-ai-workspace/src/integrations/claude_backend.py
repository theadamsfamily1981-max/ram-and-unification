"""Claude (Anthropic) AI backend implementation."""

import time
from typing import Optional, Dict, Any, AsyncIterator
import anthropic
from anthropic import AsyncAnthropic

from core.backend import AIBackend, AIProvider, Capabilities, Context, Response
from utils.logger import get_logger

logger = get_logger(__name__)


class ClaudeBackend(AIBackend):
    """
    Claude (Anthropic) backend implementation.

    Supports Claude 3 models (Opus, Sonnet, Haiku) with streaming,
    vision, and function calling capabilities.
    """

    # Model configurations
    MODELS = {
        "opus": {
            "id": "claude-3-opus-20240229",
            "max_tokens": 200000,
            "supports_vision": True,
            "cost_per_1k": 0.015
        },
        "sonnet": {
            "id": "claude-3-5-sonnet-20241022",
            "max_tokens": 200000,
            "supports_vision": True,
            "cost_per_1k": 0.003
        },
        "haiku": {
            "id": "claude-3-5-haiku-20241022",
            "max_tokens": 200000,
            "supports_vision": True,
            "cost_per_1k": 0.001
        }
    }

    def __init__(
        self,
        api_key: str,
        model: str = "sonnet",
        name: str = "Claude",
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize Claude backend.

        Args:
            api_key: Anthropic API key
            model: Model variant (opus, sonnet, haiku) or full model ID
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
                "max_tokens": 200000,
                "supports_vision": False,
                "cost_per_1k": 0.003
            }

        super().__init__(
            name=name,
            provider=AIProvider.CLAUDE,
            model=model_id,
            api_key=api_key,
            config=config
        )

        # Initialize Anthropic client
        self.client = AsyncAnthropic(api_key=api_key)

        logger.info(f"Claude backend initialized: {model_id}")

    async def send_message(
        self,
        prompt: str,
        context: Optional[Context] = None
    ) -> Response:
        """
        Send a message to Claude and get complete response.

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
            response = await self.client.messages.create(
                model=self.model,
                max_tokens=context.max_tokens or self.config.get("max_tokens", 4096),
                temperature=context.temperature or self.config.get("temperature", 1.0),
                system=context.system_prompt or "",
                messages=messages
            )

            # Extract content
            content = ""
            for block in response.content:
                if hasattr(block, "text"):
                    content += block.text

            latency_ms = (time.time() - start_time) * 1000

            return Response(
                content=content,
                provider=AIProvider.CLAUDE,
                model=self.model,
                tokens_used=response.usage.input_tokens + response.usage.output_tokens,
                latency_ms=latency_ms,
                metadata={
                    "input_tokens": response.usage.input_tokens,
                    "output_tokens": response.usage.output_tokens,
                    "stop_reason": response.stop_reason
                }
            )

        except Exception as e:
            logger.error(f"Claude API error: {e}")
            latency_ms = (time.time() - start_time) * 1000

            return Response(
                content="",
                provider=AIProvider.CLAUDE,
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
        Stream Claude response.

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
            async with self.client.messages.stream(
                model=self.model,
                max_tokens=context.max_tokens or self.config.get("max_tokens", 4096),
                temperature=context.temperature or self.config.get("temperature", 1.0),
                system=context.system_prompt or "",
                messages=messages
            ) as stream:
                async for text in stream.text_stream:
                    yield text

        except Exception as e:
            logger.error(f"Claude streaming error: {e}")
            yield f"[Error: {e}]"

    def get_capabilities(self) -> Capabilities:
        """Get Claude capabilities."""
        if self._capabilities is None:
            self._capabilities = Capabilities(
                streaming=True,
                vision=self._model_config.get("supports_vision", False),
                function_calling=True,
                max_tokens=self._model_config.get("max_tokens", 200000),
                supports_system_prompt=True,
                rate_limit_rpm=self.config.get("rate_limit_rpm", 50),
                cost_per_1k_tokens=self._model_config.get("cost_per_1k", 0.003)
            )

        return self._capabilities

    async def health_check(self) -> bool:
        """
        Check if Claude API is accessible.

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
            logger.error(f"Claude health check failed: {e}")
            return False

    def _build_messages(
        self,
        prompt: str,
        context: Context
    ) -> list:
        """
        Build messages array for Anthropic API.

        Args:
            prompt: Current user message
            context: Context with history

        Returns:
            Messages list
        """
        messages = []

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
