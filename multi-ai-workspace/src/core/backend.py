"""Core backend abstraction for AI services.

This module defines the abstract interface that all AI backends must implement,
enabling unified interaction with different AI services (Claude, ChatGPT, Grok, etc.).
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Any, AsyncIterator
from datetime import datetime
from enum import Enum


class AIProvider(Enum):
    """Supported AI providers."""
    CLAUDE = "claude"
    OPENAI = "openai"
    GROK = "grok"
    OLLAMA = "ollama"
    CUSTOM = "custom"


@dataclass
class Capabilities:
    """AI backend capabilities."""
    streaming: bool = False
    vision: bool = False
    function_calling: bool = False
    max_tokens: int = 4096
    supports_system_prompt: bool = True
    rate_limit_rpm: Optional[int] = None
    cost_per_1k_tokens: Optional[float] = None


@dataclass
class Context:
    """Context information for AI requests."""
    system_prompt: Optional[str] = None
    conversation_history: List[Dict[str, str]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "system_prompt": self.system_prompt,
            "conversation_history": self.conversation_history,
            "metadata": self.metadata,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature
        }


@dataclass
class Response:
    """AI response with metadata."""
    content: str
    provider: AIProvider
    model: str
    timestamp: datetime = field(default_factory=datetime.now)
    tokens_used: Optional[int] = None
    latency_ms: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "content": self.content,
            "provider": self.provider.value,
            "model": self.model,
            "timestamp": self.timestamp.isoformat(),
            "tokens_used": self.tokens_used,
            "latency_ms": self.latency_ms,
            "metadata": self.metadata,
            "error": self.error
        }

    @property
    def success(self) -> bool:
        """Check if response was successful."""
        return self.error is None


class AIBackend(ABC):
    """
    Abstract base class for AI backend implementations.

    All AI service integrations (Claude, ChatGPT, Grok, Ollama) must implement
    this interface to ensure consistent behavior across the system.
    """

    def __init__(
        self,
        name: str,
        provider: AIProvider,
        model: str,
        api_key: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize AI backend.

        Args:
            name: Human-readable name (e.g., "Claude", "Nova")
            provider: AI provider enum
            model: Model identifier
            api_key: API key for authentication
            config: Additional configuration parameters
        """
        self.name = name
        self.provider = provider
        self.model = model
        self.api_key = api_key
        self.config = config or {}
        self._capabilities: Optional[Capabilities] = None

    @abstractmethod
    async def send_message(
        self,
        prompt: str,
        context: Optional[Context] = None
    ) -> Response:
        """
        Send a message and get a complete response.

        Args:
            prompt: User message/prompt
            context: Optional context (system prompt, history, etc.)

        Returns:
            Response object with content and metadata
        """
        pass

    @abstractmethod
    async def stream_message(
        self,
        prompt: str,
        context: Optional[Context] = None
    ) -> AsyncIterator[str]:
        """
        Send a message and stream the response.

        Args:
            prompt: User message/prompt
            context: Optional context

        Yields:
            Response chunks as they arrive
        """
        pass

    @abstractmethod
    def get_capabilities(self) -> Capabilities:
        """
        Get backend capabilities.

        Returns:
            Capabilities object describing what this backend can do
        """
        pass

    @abstractmethod
    async def health_check(self) -> bool:
        """
        Check if backend is healthy and accessible.

        Returns:
            True if backend is operational
        """
        pass

    def __repr__(self) -> str:
        """String representation."""
        return f"{self.__class__.__name__}(name='{self.name}', provider={self.provider.value}, model='{self.model}')"
