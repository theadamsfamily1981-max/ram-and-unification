"""Core abstractions and base classes for Multi-AI Workspace."""

from .backend import AIBackend, AIProvider, Capabilities, Context, Response
from .router import Router, RoutingStrategy, RoutingRule, RoutingResult

__all__ = [
    "AIBackend",
    "AIProvider",
    "Capabilities",
    "Context",
    "Response",
    "Router",
    "RoutingStrategy",
    "RoutingRule",
    "RoutingResult",
]
