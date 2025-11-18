"""Widgets for Multi-AI Workspace Phase 2."""

from .perspectives_mixer import PerspectivesMixer, PerspectiveComparison
from .context_packs import ContextPackManager, BUILTIN_PACKS
from .cross_posting import CrossPostingPanel

__all__ = [
    "PerspectivesMixer",
    "PerspectiveComparison",
    "ContextPackManager",
    "BUILTIN_PACKS",
    "CrossPostingPanel",
]
