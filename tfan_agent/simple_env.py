# tfan_agent/simple_env.py
# Simple Homeostatic Environment for Phase 1 Testing
#
# A toy environment where:
#   - The agent has internal needs that drift/deplete over time
#   - Actions can satisfy or exacerbate different needs
#   - Observations provide information about the environment state
#   - The goal is to maintain homeostasis (minimize free energy)
#
# This is NOT meant to be a realistic environment, just a playground
# to verify the homeostatic RL machinery works.

from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple, Dict, Any, Optional

import torch


@dataclass
class SimpleHomeostaticEnvConfig:
    """
    Configuration for the toy homeostatic environment.
    """
    obs_dim: int = 4              # External observation dimension
    num_actions: int = 4          # Number of discrete actions
    num_needs: int = 8            # Number of internal needs
    need_drift: float = 0.02      # How much needs naturally increase per step
    action_effect_scale: float = 0.1  # Scale of action effects on needs
    obs_noise: float = 0.1        # Observation noise level
    max_steps: int = 100          # Max steps per episode
    device: str = "cpu"


class SimpleHomeostaticEnv:
    """
    Toy environment for testing homeostatic RL.

    The environment simulates a simple world where:
    1. Internal needs naturally drift upward (entropy/depletion)
    2. Each action affects a subset of needs (some positively, some negatively)
    3. External observations provide contextual information

    The agent's goal is to select actions that keep needs low,
    maintaining homeostasis and maximizing the intrinsic reward
    from the HomeostatL1 module.
    """

    def __init__(self, config: SimpleHomeostaticEnvConfig):
        self.config = config
        self.device = torch.device(config.device)

        # Action -> need_delta mapping
        # Each action affects needs differently
        # Positive values = increases need (bad), negative = satisfies need (good)
        self.action_to_need_delta = self._generate_action_effects()

        # Current state
        self.obs = torch.zeros(config.obs_dim, device=self.device)
        self.step_count = 0
        self.done = False

        # Track which needs are "urgent" (changes observation)
        self.urgency_threshold = 0.5

    def _generate_action_effects(self) -> torch.Tensor:
        """
        Generate action -> need delta mapping.

        Each action primarily satisfies 1-2 needs but may have side effects.
        This creates a multi-objective optimization problem.
        """
        num_actions = self.config.num_actions
        num_needs = self.config.num_needs
        scale = self.config.action_effect_scale

        effects = torch.zeros(num_actions, num_needs, device=self.device)

        for a in range(num_actions):
            # Primary effect: strongly satisfies one need
            primary_need = a % num_needs
            effects[a, primary_need] = -scale * 2.0

            # Secondary effect: mildly satisfies another need
            secondary_need = (a + 1) % num_needs
            effects[a, secondary_need] = -scale * 0.5

            # Side effect: slightly increases a different need (cost)
            side_effect_need = (a + 3) % num_needs
            effects[a, side_effect_need] = scale * 0.3

        return effects

    def reset(self, seed: Optional[int] = None) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Reset environment for new episode.

        Returns:
            obs: Initial observation
            info: Additional info dict
        """
        if seed is not None:
            torch.manual_seed(seed)

        # Reset observation to small random values
        self.obs = torch.randn(self.config.obs_dim, device=self.device) * 0.1
        self.step_count = 0
        self.done = False

        info = {
            "step": self.step_count,
            "need_drift": torch.zeros(self.config.num_needs, device=self.device),
        }

        return self.obs.clone(), info

    def step(self, action: int) -> Tuple[torch.Tensor, torch.Tensor, bool, bool, Dict[str, Any]]:
        """
        Take one step in the environment.

        Args:
            action: int in [0, num_actions-1]

        Returns:
            obs_next: Next observation
            need_delta: Change in needs (to pass to HomeostatL1)
            terminated: Whether episode ended normally
            truncated: Whether episode was truncated (max steps)
            info: Additional info
        """
        assert 0 <= action < self.config.num_actions, f"Invalid action {action}"

        self.step_count += 1

        # Compute need delta from action
        action_effect = self.action_to_need_delta[action].clone()

        # Add natural drift (needs increase over time)
        drift = torch.full(
            (self.config.num_needs,),
            self.config.need_drift,
            device=self.device
        )

        # Some randomness in drift
        drift = drift + torch.randn_like(drift) * 0.01

        # Total need delta
        need_delta = action_effect + drift

        # Update observation (random walk with action-dependent bias)
        obs_change = torch.randn(self.config.obs_dim, device=self.device) * self.config.obs_noise
        # Action slightly biases observation
        obs_change[action % self.config.obs_dim] += 0.1
        self.obs = self.obs + obs_change

        # Check termination
        terminated = False
        truncated = self.step_count >= self.config.max_steps

        self.done = terminated or truncated

        info = {
            "step": self.step_count,
            "action_effect": action_effect,
            "drift": drift,
            "need_delta": need_delta,
        }

        return self.obs.clone(), need_delta, terminated, truncated, info

    def get_action_descriptions(self) -> Dict[int, str]:
        """Get human-readable action descriptions."""
        descriptions = {}
        for a in range(self.config.num_actions):
            effects = self.action_to_need_delta[a]
            primary = effects.argmin().item()
            descriptions[a] = f"Action {a}: primarily satisfies need {primary}"
        return descriptions


class NeedsBasedObsEnv(SimpleHomeostaticEnv):
    """
    Extended environment where observations include need-related signals.

    This makes the environment more "embodied" - the agent can perceive
    its internal state through external-like observations.
    """

    def __init__(self, config: SimpleHomeostaticEnvConfig):
        super().__init__(config)

        # Store current needs for observation generation
        self.current_needs = torch.zeros(config.num_needs, device=self.device)

    def reset(self, seed: Optional[int] = None, init_needs: Optional[torch.Tensor] = None):
        obs, info = super().reset(seed)

        if init_needs is not None:
            self.current_needs = init_needs.clone().to(self.device)
        else:
            self.current_needs = torch.zeros(self.config.num_needs, device=self.device)

        # Modify observation to include need signals
        obs = self._add_need_signals(obs)

        return obs, info

    def step(self, action: int):
        obs, need_delta, terminated, truncated, info = super().step(action)

        # Update internal needs tracking
        self.current_needs = (self.current_needs + need_delta).clamp(min=0)

        # Add need signals to observation
        obs = self._add_need_signals(obs)

        return obs, need_delta, terminated, truncated, info

    def _add_need_signals(self, obs: torch.Tensor) -> torch.Tensor:
        """Add need-derived signals to observation."""
        # Example: first obs_dim features are external, we add need-based ones
        # In a real implementation, config would specify this

        # For simplicity, we just return the original obs
        # The HomeostatL1 handles internal state separately
        return obs


if __name__ == "__main__":
    # Quick sanity check
    print("=== SimpleHomeostaticEnv Sanity Check ===")

    config = SimpleHomeostaticEnvConfig(
        obs_dim=4,
        num_actions=4,
        num_needs=4,
        device="cpu",
    )
    env = SimpleHomeostaticEnv(config)

    print(f"Action descriptions:")
    for a, desc in env.get_action_descriptions().items():
        print(f"  {desc}")

    print(f"\nAction -> Need delta matrix:")
    print(env.action_to_need_delta)

    # Run a few steps
    obs, info = env.reset(seed=42)
    print(f"\nInitial obs: {obs}")

    total_need_delta = torch.zeros(config.num_needs)
    for step in range(5):
        action = step % config.num_actions
        obs, need_delta, term, trunc, info = env.step(action)
        total_need_delta += need_delta
        print(f"Step {step+1}: action={action}, need_delta sum={need_delta.sum():.3f}")

    print(f"\nTotal need delta after 5 steps: {total_need_delta}")
    print(f"Mean per-need drift: {total_need_delta / 5}")

    print("\nSimpleHomeostaticEnv sanity check passed!")
