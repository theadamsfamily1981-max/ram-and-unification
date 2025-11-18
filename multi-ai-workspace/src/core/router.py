"""Tag-based routing system for Multi-AI Workspace.

The Router intelligently routes prompts to appropriate AI backends based on
tags, capabilities, and routing rules defined in configuration.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
from enum import Enum

from .backend import AIBackend, Context, Response
from ..utils.logger import get_logger

logger = get_logger(__name__)


class RoutingStrategy(Enum):
    """Routing strategies for multi-AI collaboration."""
    SINGLE = "single"              # Route to single best AI
    PARALLEL = "parallel"          # Route to multiple AIs in parallel
    SEQUENTIAL = "sequential"      # Route to multiple AIs sequentially
    COMPETITIVE = "competitive"    # Route to multiple AIs, use best response


@dataclass
class RoutingRule:
    """A routing rule that maps tags to backends."""
    tags: List[str]
    backends: List[str]
    strategy: RoutingStrategy = RoutingStrategy.SINGLE
    priority: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RoutingResult:
    """Result of routing operation."""
    backends: List[AIBackend]
    strategy: RoutingStrategy
    matched_rule: Optional[RoutingRule] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class Router:
    """
    Multi-AI Router for intelligent prompt distribution.

    The Router analyzes tags in prompts and routes them to the most appropriate
    AI backend(s) based on configured rules and capabilities.

    Example tags:
        #code - Route to coding-focused AI
        #creative - Route to creative writing AI
        #fast - Prefer faster/cheaper models
        #multiverse - Route to all AIs for perspective comparison
    """

    def __init__(
        self,
        backends: Dict[str, AIBackend],
        rules: Optional[List[RoutingRule]] = None,
        default_backend: Optional[str] = None
    ):
        """
        Initialize Router.

        Args:
            backends: Dictionary of backend_name -> AIBackend
            rules: List of routing rules
            default_backend: Name of default backend if no rules match
        """
        self.backends = backends
        self.rules = sorted(rules or [], key=lambda r: r.priority, reverse=True)
        self.default_backend = default_backend or (list(backends.keys())[0] if backends else None)

        logger.info(f"Router initialized with {len(backends)} backends, {len(self.rules)} rules")

    def route(
        self,
        prompt: str,
        tags: Optional[List[str]] = None,
        force_backend: Optional[str] = None
    ) -> RoutingResult:
        """
        Route a prompt to appropriate backend(s).

        Args:
            prompt: User prompt
            tags: Optional explicit tags (overrides auto-detection)
            force_backend: Force specific backend (bypass routing)

        Returns:
            RoutingResult with selected backends and strategy
        """
        # Force specific backend if requested
        if force_backend:
            if force_backend not in self.backends:
                logger.warning(f"Forced backend '{force_backend}' not found, using default")
                backend_name = self.default_backend
            else:
                backend_name = force_backend

            return RoutingResult(
                backends=[self.backends[backend_name]],
                strategy=RoutingStrategy.SINGLE,
                metadata={"forced": True}
            )

        # Extract tags from prompt if not provided
        if tags is None:
            tags = self._extract_tags(prompt)

        logger.debug(f"Routing with tags: {tags}")

        # Find matching rule
        for rule in self.rules:
            if self._rule_matches(rule, tags):
                logger.info(f"Matched routing rule: {rule.tags} -> {rule.backends} ({rule.strategy.value})")

                # Get backends for rule
                selected_backends = []
                for backend_name in rule.backends:
                    if backend_name in self.backends:
                        selected_backends.append(self.backends[backend_name])
                    else:
                        logger.warning(f"Backend '{backend_name}' in rule not found")

                if not selected_backends:
                    logger.warning("No valid backends in matched rule, using default")
                    selected_backends = [self.backends[self.default_backend]]

                return RoutingResult(
                    backends=selected_backends,
                    strategy=rule.strategy,
                    matched_rule=rule,
                    metadata=rule.metadata
                )

        # No rule matched, use default
        logger.info(f"No rule matched, using default backend: {self.default_backend}")
        return RoutingResult(
            backends=[self.backends[self.default_backend]],
            strategy=RoutingStrategy.SINGLE,
            metadata={"default": True}
        )

    async def execute(
        self,
        prompt: str,
        tags: Optional[List[str]] = None,
        context: Optional[Context] = None,
        force_backend: Optional[str] = None
    ) -> List[Response]:
        """
        Route and execute prompt on selected backend(s).

        Args:
            prompt: User prompt
            tags: Optional explicit tags
            context: Optional context
            force_backend: Force specific backend

        Returns:
            List of Response objects (one per backend)
        """
        # Route to backends
        routing = self.route(prompt, tags, force_backend)

        logger.info(f"Executing on {len(routing.backends)} backend(s) with {routing.strategy.value} strategy")

        # Execute based on strategy
        if routing.strategy == RoutingStrategy.SINGLE:
            response = await routing.backends[0].send_message(prompt, context)
            return [response]

        elif routing.strategy == RoutingStrategy.PARALLEL:
            # Execute on all backends in parallel
            import asyncio
            tasks = [backend.send_message(prompt, context) for backend in routing.backends]
            responses = await asyncio.gather(*tasks, return_exceptions=True)

            # Convert exceptions to error responses
            results = []
            for i, response in enumerate(responses):
                if isinstance(response, Exception):
                    backend = routing.backends[i]
                    results.append(Response(
                        content="",
                        provider=backend.provider,
                        model=backend.model,
                        error=str(response)
                    ))
                else:
                    results.append(response)

            return results

        elif routing.strategy == RoutingStrategy.SEQUENTIAL:
            # Execute on backends sequentially, passing previous response as context
            responses = []
            current_context = context or Context()

            for backend in routing.backends:
                response = await backend.send_message(prompt, current_context)
                responses.append(response)

                # Add response to context for next backend
                if response.success:
                    current_context.conversation_history.append({
                        "role": "assistant",
                        "content": f"[{backend.name}]: {response.content}"
                    })

            return responses

        elif routing.strategy == RoutingStrategy.COMPETITIVE:
            # Execute on all backends, return only the best
            import asyncio
            tasks = [backend.send_message(prompt, context) for backend in routing.backends]
            responses = await asyncio.gather(*tasks, return_exceptions=True)

            # Filter valid responses
            valid_responses = [r for r in responses if isinstance(r, Response) and r.success]

            if not valid_responses:
                logger.error("No valid responses in competitive routing")
                return [responses[0] if responses else Response(
                    content="",
                    provider=routing.backends[0].provider,
                    model=routing.backends[0].model,
                    error="All backends failed"
                )]

            # Select best response (for now, just fastest)
            best = min(valid_responses, key=lambda r: r.latency_ms or float('inf'))
            return [best]

        return []

    def _extract_tags(self, prompt: str) -> List[str]:
        """
        Extract hashtags from prompt.

        Args:
            prompt: User prompt

        Returns:
            List of tags (without #)
        """
        import re
        tags = re.findall(r'#(\w+)', prompt)
        return [tag.lower() for tag in tags]

    def _rule_matches(self, rule: RoutingRule, tags: List[str]) -> bool:
        """
        Check if routing rule matches given tags.

        Args:
            rule: Routing rule
            tags: User tags

        Returns:
            True if rule matches
        """
        # Rule matches if any of its tags are in user tags
        return any(rule_tag in tags for rule_tag in rule.tags)

    def add_rule(self, rule: RoutingRule):
        """
        Add a routing rule.

        Args:
            rule: Routing rule to add
        """
        self.rules.append(rule)
        self.rules.sort(key=lambda r: r.priority, reverse=True)
        logger.info(f"Added routing rule: {rule.tags} -> {rule.backends}")

    def remove_rule(self, tags: List[str]):
        """
        Remove routing rule by tags.

        Args:
            tags: Rule tags to match
        """
        self.rules = [r for r in self.rules if r.tags != tags]
        logger.info(f"Removed routing rule for tags: {tags}")

    def get_backend(self, name: str) -> Optional[AIBackend]:
        """
        Get backend by name.

        Args:
            name: Backend name

        Returns:
            AIBackend or None
        """
        return self.backends.get(name)

    def list_backends(self) -> List[str]:
        """
        List all available backend names.

        Returns:
            List of backend names
        """
        return list(self.backends.keys())

    def get_routing_info(self) -> Dict[str, Any]:
        """
        Get routing configuration info.

        Returns:
            Dictionary with routing information
        """
        return {
            "backends": {
                name: {
                    "provider": backend.provider.value,
                    "model": backend.model,
                    "capabilities": backend.get_capabilities().__dict__
                }
                for name, backend in self.backends.items()
            },
            "rules": [
                {
                    "tags": rule.tags,
                    "backends": rule.backends,
                    "strategy": rule.strategy.value,
                    "priority": rule.priority
                }
                for rule in self.rules
            ],
            "default_backend": self.default_backend
        }
