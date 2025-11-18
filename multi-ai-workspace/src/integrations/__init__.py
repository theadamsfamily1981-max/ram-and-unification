"""AI backend integrations for various providers."""

from .claude_backend import ClaudeBackend
from .nova_backend import NovaBackend
from .pulse_backend import PulseBackend  # Ollama (legacy/optional)
from .gemini_pulse_backend import GeminiPulseBackend  # Pulse (Gemini)
from .grok_ara_backend import GrokAraBackend  # Ara (Grok)

__all__ = [
    "ClaudeBackend",
    "NovaBackend",
    "PulseBackend",  # Ollama
    "GeminiPulseBackend",  # Pulse
    "GrokAraBackend",  # Ara
]
