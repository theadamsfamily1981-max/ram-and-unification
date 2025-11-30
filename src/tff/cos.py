"""CognitiveOffloadingSubsystem (COS) - Niche Construction Economics.

This module implements the Cognitive Offloading Subsystem based on
Niche Construction Economics (NCE) principles:

1. Decision policy for when to offload computation
2. Cost-benefit analysis for tool use vs. internal computation
3. Environmental scaffolding management
4. Resource allocation optimization

Key insight: Intelligent agents should treat their environment as an
extended cognitive workspace, strategically offloading computation
when the energetic/accuracy tradeoff favors external resources.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Callable
from enum import Enum, auto
from collections import deque

import torch
import torch.nn as nn
import torch.nn.functional as F


class OffloadTarget(Enum):
    """Targets for cognitive offloading."""

    INTERNAL = auto()  # Keep in working memory
    CACHE = auto()  # Local cache/scratchpad
    TOOL = auto()  # External tool (calculator, search, etc.)
    MEMORY = auto()  # Long-term memory store
    HUMAN = auto()  # Defer to human
    PARALLEL = auto()  # Spawn parallel process


class OffloadReason(Enum):
    """Reasons for offloading decision."""

    CAPACITY = "capacity"  # Working memory at capacity
    EFFICIENCY = "efficiency"  # Tool is more efficient
    ACCURACY = "accuracy"  # Tool is more accurate
    LATENCY = "latency"  # Need faster response
    COST = "cost"  # Internal compute too expensive
    UNCERTAINTY = "uncertainty"  # High uncertainty, need verification


@dataclass
class ResourceCosts:
    """Cost structure for different resources."""

    # Internal computation
    internal_compute: float = 1.0  # Base cost per operation
    internal_memory: float = 0.5  # Working memory usage
    internal_energy: float = 0.3  # Energy/attention cost

    # External resources
    tool_latency: float = 0.5  # Time cost for tool call
    tool_accuracy_bonus: float = 0.9  # Expected accuracy improvement
    tool_energy: float = 0.1  # Energy for orchestration

    # Memory operations
    memory_write: float = 0.2
    memory_read: float = 0.1
    memory_search: float = 0.3

    # Human-in-the-loop
    human_latency: float = 10.0  # High latency
    human_accuracy: float = 0.99  # But high accuracy
    human_cost: float = 5.0  # Expensive resource


@dataclass
class NCEConfig:
    """Configuration for NCE decision policy."""

    # Thresholds
    capacity_threshold: float = 0.8  # Trigger offload above this
    efficiency_gain_threshold: float = 0.3  # Min efficiency gain to offload
    confidence_threshold: float = 0.7  # Below this, consider offload

    # Weights for cost-benefit calculation
    accuracy_weight: float = 1.0
    latency_weight: float = 0.5
    energy_weight: float = 0.3
    cost_weight: float = 0.2

    # Resource costs
    resource_costs: ResourceCosts = field(default_factory=ResourceCosts)

    # Policy parameters
    exploration_rate: float = 0.1  # Random exploration
    ema_decay: float = 0.95  # For tracking statistics


@dataclass
class OffloadDecision:
    """Result of offload decision."""

    target: OffloadTarget
    reason: OffloadReason
    confidence: float
    expected_cost: float
    expected_benefit: float
    net_value: float  # benefit - cost
    metadata: Dict[str, Any] = field(default_factory=dict)


class WorkingMemoryState:
    """Tracks working memory state for capacity decisions."""

    def __init__(self, capacity: int = 7):
        self.capacity = capacity
        self.items: List[Dict[str, Any]] = []
        self.attention_weights: List[float] = []

    def add(self, item: Dict[str, Any], attention: float = 1.0) -> bool:
        """Add item to working memory.

        Returns False if at capacity.
        """
        if len(self.items) >= self.capacity:
            return False

        self.items.append(item)
        self.attention_weights.append(attention)
        return True

    def remove(self, index: int) -> Optional[Dict[str, Any]]:
        """Remove item by index."""
        if 0 <= index < len(self.items):
            self.attention_weights.pop(index)
            return self.items.pop(index)
        return None

    def get_load(self) -> float:
        """Get current load as fraction of capacity."""
        return len(self.items) / self.capacity

    def get_attention_entropy(self) -> float:
        """Get entropy of attention distribution."""
        if not self.attention_weights:
            return 0.0

        total = sum(self.attention_weights)
        if total == 0:
            return 0.0

        probs = [w / total for w in self.attention_weights]
        entropy = -sum(p * math.log(p + 1e-8) for p in probs)

        return entropy

    def evict_lowest_attention(self) -> Optional[Dict[str, Any]]:
        """Evict item with lowest attention."""
        if not self.items:
            return None

        min_idx = min(range(len(self.attention_weights)),
                      key=lambda i: self.attention_weights[i])
        return self.remove(min_idx)

    def clear(self) -> None:
        """Clear working memory."""
        self.items = []
        self.attention_weights = []


class ToolRegistry:
    """Registry of available tools with cost/capability profiles."""

    def __init__(self):
        self.tools: Dict[str, Dict[str, Any]] = {}

    def register(
        self,
        name: str,
        capability: str,
        accuracy: float,
        latency: float,
        cost: float,
        handler: Optional[Callable] = None,
    ) -> None:
        """Register a tool."""
        self.tools[name] = {
            "capability": capability,
            "accuracy": accuracy,
            "latency": latency,
            "cost": cost,
            "handler": handler,
            "usage_count": 0,
            "success_rate": 1.0,
        }

    def get(self, name: str) -> Optional[Dict[str, Any]]:
        """Get tool info."""
        return self.tools.get(name)

    def find_by_capability(self, capability: str) -> List[str]:
        """Find tools with given capability."""
        return [
            name for name, info in self.tools.items()
            if info["capability"] == capability
        ]

    def update_stats(self, name: str, success: bool) -> None:
        """Update tool usage statistics."""
        if name in self.tools:
            tool = self.tools[name]
            tool["usage_count"] += 1
            # EMA update of success rate
            alpha = 0.1
            tool["success_rate"] = (
                (1 - alpha) * tool["success_rate"] +
                alpha * (1.0 if success else 0.0)
            )

    def get_best_tool(
        self,
        capability: str,
        prioritize: str = "accuracy",
    ) -> Optional[str]:
        """Get best tool for capability.

        Args:
            capability: Required capability
            prioritize: "accuracy", "latency", or "cost"
        """
        candidates = self.find_by_capability(capability)
        if not candidates:
            return None

        def score(name: str) -> float:
            tool = self.tools[name]
            if prioritize == "accuracy":
                return tool["accuracy"] * tool["success_rate"]
            elif prioritize == "latency":
                return -tool["latency"]
            else:
                return -tool["cost"]

        return max(candidates, key=score)


class NCEPolicy(nn.Module):
    """Neural policy for NCE offloading decisions.

    Takes state features and outputs offload decision.
    """

    def __init__(
        self,
        state_dim: int = 64,
        hidden_dim: int = 128,
        num_targets: int = len(OffloadTarget),
    ):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # Target prediction head
        self.target_head = nn.Linear(hidden_dim, num_targets)

        # Value head (for training)
        self.value_head = nn.Linear(hidden_dim, 1)

        # Confidence head
        self.confidence_head = nn.Sequential(
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        state: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass.

        Args:
            state: State features (batch, state_dim)

        Returns:
            Dict with target logits, value, confidence
        """
        hidden = self.encoder(state)

        target_logits = self.target_head(hidden)
        value = self.value_head(hidden).squeeze(-1)
        confidence = self.confidence_head(hidden).squeeze(-1)

        return {
            "target_logits": target_logits,
            "target_probs": F.softmax(target_logits, dim=-1),
            "value": value,
            "confidence": confidence,
        }


class CognitiveOffloadingSubsystem(nn.Module):
    """Main COS class implementing NCE decision making.

    Combines:
    1. Working memory management
    2. Tool registry
    3. Cost-benefit analysis
    4. Neural policy for learned decisions
    """

    def __init__(self, config: NCEConfig):
        super().__init__()
        self.config = config

        # State components
        self.working_memory = WorkingMemoryState()
        self.tool_registry = ToolRegistry()

        # Neural policy
        self.policy = NCEPolicy()

        # Statistics tracking
        self.decision_history: deque = deque(maxlen=1000)
        self.register_buffer("total_decisions", torch.tensor(0))
        self.register_buffer("offload_count", torch.tensor(0))

        # EMA statistics
        self.register_buffer("ema_internal_accuracy", torch.tensor(0.9))
        self.register_buffer("ema_offload_accuracy", torch.tensor(0.95))
        self.register_buffer("ema_latency_ratio", torch.tensor(1.0))

        # Register default tools
        self._register_default_tools()

    def _register_default_tools(self) -> None:
        """Register commonly available tools."""
        self.tool_registry.register(
            "calculator",
            capability="arithmetic",
            accuracy=0.99,
            latency=0.01,
            cost=0.01,
        )
        self.tool_registry.register(
            "search",
            capability="information_retrieval",
            accuracy=0.85,
            latency=0.5,
            cost=0.1,
        )
        self.tool_registry.register(
            "memory_store",
            capability="storage",
            accuracy=1.0,
            latency=0.05,
            cost=0.02,
        )

    def build_state_features(
        self,
        task_complexity: float,
        task_uncertainty: float,
        time_pressure: float,
        available_tools: List[str],
    ) -> torch.Tensor:
        """Build state feature vector for policy.

        Args:
            task_complexity: Estimated task complexity [0, 1]
            task_uncertainty: Uncertainty about task [0, 1]
            time_pressure: Time urgency [0, 1]
            available_tools: List of available tool names

        Returns:
            State feature tensor (64,)
        """
        features = []

        # Task features
        features.extend([
            task_complexity,
            task_uncertainty,
            time_pressure,
        ])

        # Working memory features
        features.extend([
            self.working_memory.get_load(),
            self.working_memory.get_attention_entropy(),
            len(self.working_memory.items) / 10.0,  # Normalized count
        ])

        # Tool availability (one-hot style)
        tool_features = [0.0] * 10  # Fixed size
        for i, tool_name in enumerate(available_tools[:10]):
            tool = self.tool_registry.get(tool_name)
            if tool:
                tool_features[i] = tool["accuracy"]
        features.extend(tool_features)

        # Historical statistics
        features.extend([
            self.ema_internal_accuracy.item(),
            self.ema_offload_accuracy.item(),
            self.ema_latency_ratio.item(),
            self.offload_count.float().item() / (self.total_decisions.float().item() + 1),
        ])

        # Pad to 64 dimensions
        while len(features) < 64:
            features.append(0.0)

        return torch.tensor(features[:64], dtype=torch.float32)

    def compute_cost_benefit(
        self,
        target: OffloadTarget,
        task_complexity: float,
        task_uncertainty: float,
        tool_name: Optional[str] = None,
    ) -> Tuple[float, float]:
        """Compute expected cost and benefit for offload target.

        Returns:
            (expected_cost, expected_benefit)
        """
        costs = self.config.resource_costs

        if target == OffloadTarget.INTERNAL:
            # Internal computation
            cost = (
                costs.internal_compute * task_complexity +
                costs.internal_memory * self.working_memory.get_load() +
                costs.internal_energy
            )
            benefit = self.ema_internal_accuracy.item() * (1 - task_uncertainty)

        elif target == OffloadTarget.TOOL and tool_name:
            # Tool use
            tool = self.tool_registry.get(tool_name)
            if tool:
                cost = (
                    costs.tool_latency * tool["latency"] +
                    costs.tool_energy +
                    tool["cost"]
                )
                benefit = (
                    tool["accuracy"] * tool["success_rate"] *
                    costs.tool_accuracy_bonus
                )
            else:
                cost, benefit = float('inf'), 0.0

        elif target == OffloadTarget.CACHE:
            # Cache lookup
            cost = costs.memory_read
            benefit = 0.8  # Assuming cached result is relevant

        elif target == OffloadTarget.MEMORY:
            # Long-term memory
            cost = costs.memory_search + costs.memory_read
            benefit = 0.7 * (1 - task_uncertainty)  # Less certain for old memories

        elif target == OffloadTarget.HUMAN:
            # Human-in-the-loop
            cost = costs.human_cost + costs.human_latency
            benefit = costs.human_accuracy

        elif target == OffloadTarget.PARALLEL:
            # Parallel execution
            cost = costs.internal_compute * 0.5 + costs.internal_energy * 2
            benefit = self.ema_internal_accuracy.item() * 1.2  # Parallelism bonus

        else:
            cost, benefit = 1.0, 0.5

        return cost, benefit

    def heuristic_decision(
        self,
        task_complexity: float,
        task_uncertainty: float,
        time_pressure: float,
        capability_needed: Optional[str] = None,
    ) -> OffloadDecision:
        """Make offloading decision using heuristics.

        Args:
            task_complexity: Task complexity [0, 1]
            task_uncertainty: Uncertainty [0, 1]
            time_pressure: Time urgency [0, 1]
            capability_needed: Required capability (for tool selection)

        Returns:
            OffloadDecision
        """
        wm_load = self.working_memory.get_load()

        # Check capacity constraint
        if wm_load > self.config.capacity_threshold:
            # Must offload - working memory at capacity
            if capability_needed:
                tool = self.tool_registry.get_best_tool(capability_needed)
                if tool:
                    cost, benefit = self.compute_cost_benefit(
                        OffloadTarget.TOOL, task_complexity, task_uncertainty, tool
                    )
                    return OffloadDecision(
                        target=OffloadTarget.TOOL,
                        reason=OffloadReason.CAPACITY,
                        confidence=0.9,
                        expected_cost=cost,
                        expected_benefit=benefit,
                        net_value=benefit - cost,
                        metadata={"tool": tool},
                    )

            # Offload to memory
            cost, benefit = self.compute_cost_benefit(
                OffloadTarget.MEMORY, task_complexity, task_uncertainty
            )
            return OffloadDecision(
                target=OffloadTarget.MEMORY,
                reason=OffloadReason.CAPACITY,
                confidence=0.8,
                expected_cost=cost,
                expected_benefit=benefit,
                net_value=benefit - cost,
            )

        # Check uncertainty threshold
        if task_uncertainty > (1 - self.config.confidence_threshold):
            # High uncertainty - consider external verification
            if capability_needed:
                tool = self.tool_registry.get_best_tool(
                    capability_needed, prioritize="accuracy"
                )
                if tool:
                    cost, benefit = self.compute_cost_benefit(
                        OffloadTarget.TOOL, task_complexity, task_uncertainty, tool
                    )
                    if benefit - cost > self.config.efficiency_gain_threshold:
                        return OffloadDecision(
                            target=OffloadTarget.TOOL,
                            reason=OffloadReason.UNCERTAINTY,
                            confidence=0.7,
                            expected_cost=cost,
                            expected_benefit=benefit,
                            net_value=benefit - cost,
                            metadata={"tool": tool},
                        )

        # Check efficiency gain
        internal_cost, internal_benefit = self.compute_cost_benefit(
            OffloadTarget.INTERNAL, task_complexity, task_uncertainty
        )

        if capability_needed:
            tool = self.tool_registry.get_best_tool(capability_needed)
            if tool:
                tool_cost, tool_benefit = self.compute_cost_benefit(
                    OffloadTarget.TOOL, task_complexity, task_uncertainty, tool
                )

                # Compare net values
                internal_net = internal_benefit - internal_cost
                tool_net = tool_benefit - tool_cost

                if tool_net - internal_net > self.config.efficiency_gain_threshold:
                    return OffloadDecision(
                        target=OffloadTarget.TOOL,
                        reason=OffloadReason.EFFICIENCY,
                        confidence=0.8,
                        expected_cost=tool_cost,
                        expected_benefit=tool_benefit,
                        net_value=tool_net,
                        metadata={"tool": tool},
                    )

        # Default: internal computation
        return OffloadDecision(
            target=OffloadTarget.INTERNAL,
            reason=OffloadReason.EFFICIENCY,
            confidence=0.9,
            expected_cost=internal_cost,
            expected_benefit=internal_benefit,
            net_value=internal_benefit - internal_cost,
        )

    def neural_decision(
        self,
        state: torch.Tensor,
    ) -> OffloadDecision:
        """Make offloading decision using neural policy.

        Args:
            state: State feature tensor

        Returns:
            OffloadDecision
        """
        with torch.no_grad():
            output = self.policy(state.unsqueeze(0))

        target_probs = output["target_probs"].squeeze(0)
        confidence = output["confidence"].item()

        # Sample or argmax based on exploration
        if torch.rand(1).item() < self.config.exploration_rate:
            target_idx = torch.multinomial(target_probs, 1).item()
        else:
            target_idx = target_probs.argmax().item()

        target = list(OffloadTarget)[target_idx]

        # Estimate costs (simplified for neural path)
        cost = 1.0 - target_probs[target_idx].item()
        benefit = target_probs[target_idx].item()

        return OffloadDecision(
            target=target,
            reason=OffloadReason.EFFICIENCY,
            confidence=confidence,
            expected_cost=cost,
            expected_benefit=benefit,
            net_value=benefit - cost,
            metadata={"probs": target_probs.tolist()},
        )

    def forward(
        self,
        task_complexity: float,
        task_uncertainty: float,
        time_pressure: float = 0.5,
        capability_needed: Optional[str] = None,
        use_neural: bool = False,
    ) -> OffloadDecision:
        """Make offloading decision.

        Args:
            task_complexity: Task complexity [0, 1]
            task_uncertainty: Uncertainty [0, 1]
            time_pressure: Time urgency [0, 1]
            capability_needed: Required capability
            use_neural: Whether to use neural policy

        Returns:
            OffloadDecision
        """
        self.total_decisions += 1

        if use_neural:
            state = self.build_state_features(
                task_complexity,
                task_uncertainty,
                time_pressure,
                list(self.tool_registry.tools.keys()),
            )
            decision = self.neural_decision(state)
        else:
            decision = self.heuristic_decision(
                task_complexity,
                task_uncertainty,
                time_pressure,
                capability_needed,
            )

        # Track offloading
        if decision.target != OffloadTarget.INTERNAL:
            self.offload_count += 1

        # Record decision
        self.decision_history.append({
            "target": decision.target.name,
            "reason": decision.reason.value,
            "confidence": decision.confidence,
            "net_value": decision.net_value,
        })

        return decision

    def update_statistics(
        self,
        decision: OffloadDecision,
        actual_accuracy: float,
        actual_latency: float,
    ) -> None:
        """Update EMA statistics based on outcome.

        Args:
            decision: The decision that was made
            actual_accuracy: Achieved accuracy
            actual_latency: Actual latency
        """
        alpha = 1 - self.config.ema_decay

        if decision.target == OffloadTarget.INTERNAL:
            self.ema_internal_accuracy = (
                (1 - alpha) * self.ema_internal_accuracy +
                alpha * actual_accuracy
            )
        else:
            self.ema_offload_accuracy = (
                (1 - alpha) * self.ema_offload_accuracy +
                alpha * actual_accuracy
            )

        # Update tool stats if applicable
        if decision.target == OffloadTarget.TOOL and "tool" in decision.metadata:
            self.tool_registry.update_stats(
                decision.metadata["tool"],
                success=actual_accuracy > 0.5,
            )

    def get_statistics(self) -> Dict[str, Any]:
        """Get current statistics."""
        total = self.total_decisions.item()
        offloaded = self.offload_count.item()

        return {
            "total_decisions": total,
            "offload_count": offloaded,
            "offload_rate": offloaded / total if total > 0 else 0.0,
            "ema_internal_accuracy": self.ema_internal_accuracy.item(),
            "ema_offload_accuracy": self.ema_offload_accuracy.item(),
            "working_memory_load": self.working_memory.get_load(),
            "num_tools": len(self.tool_registry.tools),
        }


def create_cos(
    capacity_threshold: float = 0.8,
    exploration_rate: float = 0.1,
) -> CognitiveOffloadingSubsystem:
    """Factory function for COS with common defaults."""
    config = NCEConfig(
        capacity_threshold=capacity_threshold,
        exploration_rate=exploration_rate,
    )
    return CognitiveOffloadingSubsystem(config)
