"""Widgets for Multi-AI Workspace."""

from .perspectives_mixer import PerspectivesMixer, PerspectiveComparison
from .context_packs import ContextPackManager, BUILTIN_PACKS
from .cross_posting import CrossPostingPanel
from .github_autopilot import GitHubAutopilot
from .colab_offload import ColabOffload

__all__ = [
    "PerspectivesMixer",
    "PerspectiveComparison",
    "ContextPackManager",
    "BUILTIN_PACKS",
    "CrossPostingPanel",
    "GitHubAutopilot",
    "ColabOffload",
]
