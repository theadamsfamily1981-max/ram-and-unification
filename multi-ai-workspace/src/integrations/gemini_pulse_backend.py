"""Pulse (Google Gemini) AI backend implementation."""

import time
from typing import Optional, Dict, Any, AsyncIterator
import google.generativeai as genai

from ..core.backend import AIBackend, AIProvider, Capabilities, Context, Response
from ..utils.logger import get_logger

logger = get_logger(__name__)


class GeminiPulseBackend(AIBackend):
    """
    Pulse (Google Gemini) backend implementation.

    Pulse is the internal orchestrator/planner using Google's Gemini models.
    """

    # Model configurations
    MODELS = {
        "gemini-pro": {
            "id": "gemini-pro",
            "max_tokens": 32768,
            "supports_vision": False
        },
        "gemini-pro-vision": {
            "id": "gemini-pro-vision",
            "max_tokens": 16384,
            "supports_vision": True
        },
        "gemini-1.5-pro": {
            "id": "gemini-1.5-pro",
            "max_tokens": 1048576,  # 1M context
            "supports_vision": True
        },
        "gemini-1.5-flash": {
            "id": "gemini-1.5-flash",
            "max_tokens": 1048576,
            "supports_vision": True
        }
    }

    def __init__(
        self,
        api_key: str,
        model: str = "gemini-1.5-flash",
        name: str = "Pulse",
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize Pulse backend.

        Args:
            api_key: Google Gemini API key
            model: Model variant
            name: Display name
            config: Additional configuration
        """
        # Resolve model config
        if model in self.MODELS:
            model_id = self.MODELS[model]["id"]
            self._model_config = self.MODELS[model]
        else:
            model_id = model
            self._model_config = {
                "id": model,
                "max_tokens": 32768,
                "supports_vision": False
            }

        super().__init__(
            name=name,
            provider=AIProvider.CUSTOM,  # Gemini
            model=model_id,
            api_key=api_key,
            config=config
        )

        # Configure Gemini
        genai.configure(api_key=api_key)
        self.client = genai.GenerativeModel(model_id)

        logger.info(f"Pulse (Gemini) backend initialized: {model_id}")

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
            # Build full prompt with context
            full_prompt = self._build_prompt(prompt, context)

            # Generate response
            generation_config = {
                "max_output_tokens": context.max_tokens or self.config.get("max_tokens", 2048),
                "temperature": context.temperature or self.config.get("temperature", 0.9),
            }

            response = self.client.generate_content(
                full_prompt,
                generation_config=generation_config
            )

            content = response.text if hasattr(response, 'text') else ""

            latency_ms = (time.time() - start_time) * 1000

            # Try to get token count (may not be available)
            tokens_used = None
            if hasattr(response, 'usage_metadata'):
                tokens_used = (
                    response.usage_metadata.prompt_token_count +
                    response.usage_metadata.candidates_token_count
                )

            return Response(
                content=content,
                provider=AIProvider.CUSTOM,
                model=self.model,
                tokens_used=tokens_used,
                latency_ms=latency_ms,
                metadata={
                    "provider_name": "google_gemini",
                    "finish_reason": str(getattr(response, 'finish_reason', None)) if hasattr(response, 'finish_reason') else None
                }
            )

        except Exception as e:
            logger.error(f"Pulse (Gemini) API error: {e}")
            latency_ms = (time.time() - start_time) * 1000

            return Response(
                content="",
                provider=AIProvider.CUSTOM,
                model=self.model,
                latency_ms=latency_ms,
                error=str(e),
                metadata={"provider_name": "google_gemini"}
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
            # Build full prompt
            full_prompt = self._build_prompt(prompt, context)

            # Generate config
            generation_config = {
                "max_output_tokens": context.max_tokens or self.config.get("max_tokens", 2048),
                "temperature": context.temperature or self.config.get("temperature", 0.9),
            }

            # Stream response
            response = self.client.generate_content(
                full_prompt,
                generation_config=generation_config,
                stream=True
            )

            for chunk in response:
                if hasattr(chunk, 'text'):
                    yield chunk.text

        except Exception as e:
            logger.error(f"Pulse (Gemini) streaming error: {e}")
            yield f"[Error: {e}]"

    def get_capabilities(self) -> Capabilities:
        """Get Pulse capabilities."""
        if self._capabilities is None:
            self._capabilities = Capabilities(
                streaming=True,
                vision=self._model_config.get("supports_vision", False),
                function_calling=True,
                max_tokens=self._model_config.get("max_tokens", 32768),
                supports_system_prompt=True,
                rate_limit_rpm=self.config.get("rate_limit_rpm", 60),
                cost_per_1k_tokens=0.0  # Free tier available
            )

        return self._capabilities

    async def health_check(self) -> bool:
        """
        Check if Gemini API is accessible.

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
            logger.error(f"Pulse (Gemini) health check failed: {e}")
            return False

    def _build_prompt(
        self,
        prompt: str,
        context: Context
    ) -> str:
        """
        Build prompt string for Gemini.

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
