"""Perspectives Mixer Widget - Compare responses from multiple AIs.

The Perspectives Mixer enables side-by-side comparison of AI responses,
helping users understand different perspectives and approaches.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from datetime import datetime

from ..core.router import Router
from ..core.backend import Context, Response
from ..storage.database import ResponseStore
from ..utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class PerspectiveComparison:
    """A multi-AI perspective comparison."""
    prompt: str
    perspectives: List[Dict[str, Any]] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "prompt": self.prompt,
            "perspectives": self.perspectives,
            "tags": self.tags,
            "created_at": self.created_at.isoformat(),
            "metadata": self.metadata
        }

    def get_summary(self) -> Dict[str, Any]:
        """Get comparison summary statistics."""
        if not self.perspectives:
            return {
                "total_perspectives": 0,
                "avg_latency_ms": 0,
                "total_tokens": 0,
                "backends": []
            }

        total_latency = sum(
            p.get("latency_ms", 0)
            for p in self.perspectives
            if p.get("latency_ms")
        )
        total_tokens = sum(
            p.get("tokens_used", 0)
            for p in self.perspectives
            if p.get("tokens_used")
        )

        return {
            "total_perspectives": len(self.perspectives),
            "avg_latency_ms": round(total_latency / len(self.perspectives), 2),
            "total_tokens": total_tokens,
            "backends": [p.get("backend") for p in self.perspectives],
            "successful": sum(1 for p in self.perspectives if not p.get("error")),
            "failed": sum(1 for p in self.perspectives if p.get("error"))
        }


class PerspectivesMixer:
    """
    Perspectives Mixer Widget.

    Enables multi-AI comparison by:
    - Querying multiple AIs with the same prompt
    - Side-by-side response presentation
    - Response quality analysis
    - Voting and ranking
    """

    def __init__(
        self,
        router: Router,
        store: Optional[ResponseStore] = None
    ):
        """
        Initialize Perspectives Mixer.

        Args:
            router: Router instance
            store: Optional response store
        """
        self.router = router
        self.store = store
        logger.info("PerspectivesMixer initialized")

    async def compare(
        self,
        prompt: str,
        backends: Optional[List[str]] = None,
        context: Optional[Context] = None,
        save_to_store: bool = True
    ) -> PerspectiveComparison:
        """
        Compare responses from multiple AIs.

        Args:
            prompt: Prompt to send to all AIs
            backends: List of backend names (None = all available)
            context: Optional context
            save_to_store: Save comparison to database

        Returns:
            PerspectiveComparison with results
        """
        # Determine backends to query
        if backends is None:
            backends = self.router.list_backends()
        else:
            # Validate backends exist
            available = self.router.list_backends()
            backends = [b for b in backends if b in available]

        if not backends:
            logger.warning("No valid backends for comparison")
            return PerspectiveComparison(prompt=prompt)

        logger.info(f"Comparing perspectives from {len(backends)} backends: {backends}")

        # Query all backends in parallel
        import asyncio
        tasks = []
        for backend_name in backends:
            backend = self.router.get_backend(backend_name)
            if backend:
                tasks.append(self._get_perspective(backend, prompt, context))

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Build perspectives
        perspectives = []
        for i, result in enumerate(results):
            backend_name = backends[i]
            backend = self.router.get_backend(backend_name)

            if isinstance(result, Exception):
                # Handle exception
                perspectives.append({
                    "backend": backend_name,
                    "provider": backend.provider.value if backend else "unknown",
                    "model": backend.model if backend else "unknown",
                    "content": "",
                    "error": str(result),
                    "latency_ms": None,
                    "tokens_used": None
                })
            elif isinstance(result, Response):
                # Successful response
                perspectives.append({
                    "backend": backend_name,
                    "provider": result.provider.value,
                    "model": result.model,
                    "content": result.content,
                    "error": result.error,
                    "latency_ms": result.latency_ms,
                    "tokens_used": result.tokens_used,
                    "metadata": result.metadata
                })

        # Create comparison
        comparison = PerspectiveComparison(
            prompt=prompt,
            perspectives=perspectives,
            tags=context.metadata.get("tags", []) if context else [],
            metadata={
                "context": context.to_dict() if context else None
            }
        )

        # Save to store if requested
        if save_to_store and self.store:
            await self._save_comparison(comparison)

        logger.info(f"Comparison complete: {len(perspectives)} perspectives")
        return comparison

    async def _get_perspective(
        self,
        backend,
        prompt: str,
        context: Optional[Context]
    ) -> Response:
        """
        Get perspective from a single backend.

        Args:
            backend: AI backend
            prompt: Prompt
            context: Optional context

        Returns:
            Response
        """
        try:
            return await backend.send_message(prompt, context)
        except Exception as e:
            logger.error(f"Error getting perspective from {backend.name}: {e}")
            raise

    async def _save_comparison(self, comparison: PerspectiveComparison):
        """Save comparison to database."""
        if not self.store:
            return

        try:
            # Create conversation for this comparison
            conversation = self.store.create_conversation(
                title=f"Perspectives: {comparison.prompt[:50]}...",
                metadata={
                    "type": "perspectives_comparison",
                    "tags": comparison.tags
                }
            )

            # Add user message
            self.store.add_message(
                conversation_id=conversation.id,
                role="user",
                content=comparison.prompt,
                tags=comparison.tags
            )

            # Add each perspective as a message
            for perspective in comparison.perspectives:
                self.store.add_message(
                    conversation_id=conversation.id,
                    role="assistant",
                    content=perspective.get("content", ""),
                    backend_name=perspective.get("backend"),
                    provider=perspective.get("provider"),
                    model=perspective.get("model"),
                    tokens_used=perspective.get("tokens_used"),
                    latency_ms=perspective.get("latency_ms"),
                    error=perspective.get("error"),
                    metadata=perspective.get("metadata", {})
                )

            logger.debug(f"Saved comparison to conversation {conversation.id}")

        except Exception as e:
            logger.error(f"Failed to save comparison: {e}")

    def analyze_perspectives(
        self,
        comparison: PerspectiveComparison
    ) -> Dict[str, Any]:
        """
        Analyze perspectives for insights.

        Args:
            comparison: Perspective comparison

        Returns:
            Analysis results
        """
        if not comparison.perspectives:
            return {"error": "No perspectives to analyze"}

        successful = [p for p in comparison.perspectives if not p.get("error")]

        if not successful:
            return {"error": "All perspectives failed"}

        # Length analysis
        lengths = [len(p.get("content", "")) for p in successful]

        # Speed analysis
        latencies = [
            p.get("latency_ms", 0)
            for p in successful
            if p.get("latency_ms")
        ]

        # Token usage
        tokens = [
            p.get("tokens_used", 0)
            for p in successful
            if p.get("tokens_used")
        ]

        # Find fastest, longest, most concise
        fastest = None
        longest = None
        most_concise = None

        if latencies:
            fastest_idx = latencies.index(min(latencies))
            fastest = successful[fastest_idx].get("backend")

        if lengths:
            longest_idx = lengths.index(max(lengths))
            longest = successful[longest_idx].get("backend")

            shortest_idx = lengths.index(min(lengths))
            most_concise = successful[shortest_idx].get("backend")

        return {
            "summary": comparison.get_summary(),
            "length_analysis": {
                "avg_length": round(sum(lengths) / len(lengths), 0) if lengths else 0,
                "min_length": min(lengths) if lengths else 0,
                "max_length": max(lengths) if lengths else 0,
                "longest_backend": longest
            },
            "speed_analysis": {
                "avg_latency_ms": round(sum(latencies) / len(latencies), 2) if latencies else 0,
                "min_latency_ms": min(latencies) if latencies else 0,
                "max_latency_ms": max(latencies) if latencies else 0,
                "fastest_backend": fastest
            },
            "token_analysis": {
                "total_tokens": sum(tokens),
                "avg_tokens": round(sum(tokens) / len(tokens), 0) if tokens else 0
            },
            "recommendations": {
                "fastest": fastest,
                "longest_response": longest,
                "most_concise": most_concise
            }
        }

    async def compare_with_voting(
        self,
        prompt: str,
        backends: Optional[List[str]] = None,
        context: Optional[Context] = None
    ) -> Dict[str, Any]:
        """
        Compare perspectives and enable voting.

        Args:
            prompt: Prompt
            backends: Backend names
            context: Optional context

        Returns:
            Comparison with voting data
        """
        comparison = await self.compare(prompt, backends, context)
        analysis = self.analyze_perspectives(comparison)

        return {
            "comparison": comparison.to_dict(),
            "analysis": analysis,
            "voting": {
                "enabled": True,
                "vote_options": [
                    {
                        "backend": p.get("backend"),
                        "preview": p.get("content", "")[:100] + "..."
                    }
                    for p in comparison.perspectives
                    if not p.get("error")
                ]
            }
        }
