#!/usr/bin/env python3
"""
ara_loop_toy_env.py
Ara Agent Loop in Toy Environment

Demonstrates the full Ara agent loop:
1. Observe environment
2. Update identity manifold
3. Compute homeostatic needs
4. Generate action via L5 control
5. Apply DAU corrections if needed

The toy environment is a simple grid world where the agent must
navigate to goals while maintaining identity coherence.

Usage:
    python ara_loop_toy_env.py --steps 1000
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from tgsfn_core import TGSFNNetwork, compute_piq
from ara_agent import (
    IdentityManifold,
    HyperbolicEncoder,
    L5Controller,
    HomeostaticRegulator,
    NeedsConfig,
    DAULite,
)


class ToyGridWorld:
    """
    Simple grid world environment.

    The agent navigates a grid to reach goals while avoiding hazards.
    Observations include position, goal location, and hazard proximity.
    """

    def __init__(
        self,
        size: int = 10,
        n_goals: int = 3,
        n_hazards: int = 5,
    ):
        self.size = size
        self.n_goals = n_goals
        self.n_hazards = n_hazards

        # Agent state
        self.agent_pos = np.array([size // 2, size // 2])

        # Goals and hazards
        self.goals = []
        self.hazards = []
        self._place_objects()

        # Observation dimension
        self.obs_dim = 8  # [x, y, goal_dx, goal_dy, hazard_dist, time, energy, novelty]

        # Internal state
        self.time = 0
        self.energy = 1.0
        self.novelty = 0.5
        self.visited = np.zeros((size, size))

    def _place_objects(self) -> None:
        """Place goals and hazards randomly."""
        self.goals = []
        self.hazards = []

        for _ in range(self.n_goals):
            pos = np.random.randint(0, self.size, 2)
            self.goals.append(pos)

        for _ in range(self.n_hazards):
            pos = np.random.randint(0, self.size, 2)
            self.hazards.append(pos)

    def reset(self) -> np.ndarray:
        """Reset environment."""
        self.agent_pos = np.array([self.size // 2, self.size // 2])
        self._place_objects()
        self.time = 0
        self.energy = 1.0
        self.novelty = 0.5
        self.visited = np.zeros((self.size, self.size))
        return self._get_obs()

    def _get_obs(self) -> np.ndarray:
        """Get current observation."""
        x, y = self.agent_pos / self.size  # Normalize

        # Nearest goal direction
        if self.goals:
            goal_dists = [np.linalg.norm(self.agent_pos - g) for g in self.goals]
            nearest_goal = self.goals[np.argmin(goal_dists)]
            goal_dx = (nearest_goal[0] - self.agent_pos[0]) / self.size
            goal_dy = (nearest_goal[1] - self.agent_pos[1]) / self.size
        else:
            goal_dx, goal_dy = 0, 0

        # Nearest hazard distance
        if self.hazards:
            hazard_dists = [np.linalg.norm(self.agent_pos - h) for h in self.hazards]
            hazard_dist = min(hazard_dists) / self.size
        else:
            hazard_dist = 1.0

        return np.array([
            x, y,
            goal_dx, goal_dy,
            hazard_dist,
            self.time / 1000,  # Normalized time
            self.energy,
            self.novelty,
        ], dtype=np.float32)

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Take action in environment.

        Args:
            action: Movement direction (2D, will be clipped to [-1, 1])

        Returns:
            (observation, reward, done, info)
        """
        # Clip and scale action
        action = np.clip(action[:2], -1, 1)
        move = (action * 2).astype(int)  # -2 to +2 movement

        # Update position
        new_pos = np.clip(self.agent_pos + move, 0, self.size - 1)
        self.agent_pos = new_pos

        # Update internal state
        self.time += 1
        self.energy = max(0, self.energy - 0.001)  # Energy decay

        # Novelty based on visit history
        if self.visited[new_pos[0], new_pos[1]] == 0:
            self.novelty = min(1, self.novelty + 0.1)
        else:
            self.novelty = max(0, self.novelty - 0.02)
        self.visited[new_pos[0], new_pos[1]] += 1

        # Compute reward
        reward = 0.0
        done = False

        # Check goals
        for i, goal in enumerate(self.goals):
            if np.array_equal(self.agent_pos, goal):
                reward += 1.0
                self.goals.pop(i)
                self.energy = min(1, self.energy + 0.2)  # Energy boost
                break

        # Check hazards
        for hazard in self.hazards:
            if np.array_equal(self.agent_pos, hazard):
                reward -= 0.5
                self.energy = max(0, self.energy - 0.1)

        # Done conditions
        if len(self.goals) == 0:
            done = True
            reward += 5.0  # Bonus for completing all goals
        if self.energy <= 0:
            done = True
            reward -= 2.0

        info = {
            "goals_remaining": len(self.goals),
            "energy": self.energy,
            "novelty": self.novelty,
        }

        return self._get_obs(), reward, done, info


class AraAgent(nn.Module):
    """
    Ara Agent integrating TGSFN backend with identity/control systems.

    Components:
    1. TGSFNNetwork: Spiking neural network for state processing
    2. IdentityManifold: Hyperbolic identity embedding
    3. HyperbolicEncoder: Maps observations to identity space
    4. HomeostaticRegulator: Manages internal needs
    5. L5Controller: Generates thermodynamically-safe actions
    6. DAULite: Ensures stability
    """

    def __init__(
        self,
        obs_dim: int = 8,
        action_dim: int = 4,
        identity_dim: int = 16,
        snn_neurons: int = 128,
    ):
        super().__init__()

        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.identity_dim = identity_dim

        # TGSFN backend
        self.snn = TGSFNNetwork(
            input_dim=obs_dim,
            n_neurons=snn_neurons,
            output_dim=identity_dim,
            ei_ratio=0.8,
            connectivity=0.1,
        )

        # Identity manifold
        self.identity = IdentityManifold(
            dim=identity_dim,
            curvature=1.0,
            init_scale=0.01,
        )

        # Observation encoder
        self.encoder = HyperbolicEncoder(
            input_dim=obs_dim,
            hidden_dim=64,
            output_dim=identity_dim,
            curvature=1.0,
        )

        # Homeostatic regulator
        self.homeostasis = HomeostaticRegulator(
            NeedsConfig(n_needs=4, beta=1.0)
        )

        # L5 controller
        self.controller = L5Controller(
            identity_dim=identity_dim,
            action_dim=action_dim,
        )

        # DAU for stability
        self.dau = DAULite()
        self.dau.register_module(self.snn, prefix="snn")

        # Track state
        self.step_count = 0

    def reset(self) -> None:
        """Reset agent state."""
        self.snn.reset_state()
        self.identity.set_identity(torch.randn(self.identity_dim) * 0.01)
        self.step_count = 0

    def forward(
        self,
        obs: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Full agent forward pass.

        Args:
            obs: Environment observation

        Returns:
            (action, info_dict)
        """
        # Ensure batch dimension
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)

        # 1. Process through SNN
        snn_out, spikes = self.snn(obs)

        # 2. Encode observation to hyperbolic space
        z_obs = self.encoder(obs)

        # 3. Update identity (blend with observation encoding)
        z_current = self.identity.z
        direction = self.identity.logmap(z_obs.squeeze(0), z_current)
        self.identity.update_identity(direction, step_size=0.01)

        # 4. Update homeostasis from observation
        # Map observation to need changes
        needs_delta = torch.zeros(4)
        needs_delta[0] = obs[0, 7] - 0.5  # Energy -> need 0
        needs_delta[1] = obs[0, 6] - 0.5  # Novelty -> need 1
        needs_delta[2] = obs[0, 4] - 0.5  # Safety (hazard distance)
        needs_delta[3] = 0.0  # Progress (computed below)

        self.homeostasis.needs.update_needs(needs_delta)
        F_int, drive = self.homeostasis()

        # 5. Compute Pi_q for throttling
        J_proxy = self.snn.compute_jacobian_proxy()
        pi_q = compute_piq(spikes.unsqueeze(1), J_proxy)

        # 6. Generate action via L5 controller
        action, control_info = self.controller(
            z=self.identity.z,
            observation=obs.squeeze(0)[:self.action_dim],
            pi_q=pi_q,
        )

        # 7. Apply DAU if needed
        dau_adjustments = self.dau(J_proxy)

        # 8. Collect info
        info = {
            "z_norm": self.identity.z.norm().item(),
            "coherence": self.identity.identity_coherence().item(),
            "F_int": F_int.item(),
            "pi_q": pi_q.item(),
            "throttle": control_info["throttle"].item(),
            "action_norm": action.norm().item(),
            "dau_adjustments": len(dau_adjustments),
            "J_proxy": J_proxy.item(),
        }

        self.step_count += 1

        return action.squeeze(0), info


def run_episode(
    agent: AraAgent,
    env: ToyGridWorld,
    max_steps: int = 500,
    verbose: bool = True,
) -> Dict:
    """
    Run single episode.

    Returns:
        Episode statistics
    """
    obs = env.reset()
    agent.reset()

    total_reward = 0.0
    step_infos = []

    for step in range(max_steps):
        obs_tensor = torch.from_numpy(obs).float().unsqueeze(0)

        # Agent forward pass
        action, info = agent(obs_tensor)

        # Convert to numpy and step
        action_np = action.detach().numpy()
        obs, reward, done, env_info = env.step(action_np)

        total_reward += reward
        step_infos.append({**info, **env_info, "reward": reward})

        if verbose and step % 100 == 0:
            print(f"  Step {step}: reward={reward:.2f}, "
                  f"coherence={info['coherence']:.3f}, "
                  f"throttle={info['throttle']:.3f}, "
                  f"goals={env_info['goals_remaining']}")

        if done:
            break

    return {
        "total_reward": total_reward,
        "steps": step + 1,
        "goals_completed": env.n_goals - len(env.goals),
        "final_coherence": step_infos[-1]["coherence"],
        "mean_throttle": np.mean([s["throttle"] for s in step_infos]),
        "mean_F_int": np.mean([s["F_int"] for s in step_infos]),
        "mean_pi_q": np.mean([s["pi_q"] for s in step_infos]),
        "dau_adjustments": sum(s["dau_adjustments"] for s in step_infos),
    }


def main():
    parser = argparse.ArgumentParser(description="Ara Agent Toy Environment")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--max_steps", type=int, default=500)
    parser.add_argument("--grid_size", type=int, default=10)
    parser.add_argument("--snn_neurons", type=int, default=128)
    parser.add_argument("--identity_dim", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    # Set seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print("=" * 60)
    print("Ara Agent - Toy Environment Loop")
    print("=" * 60)
    print(f"Episodes: {args.episodes}")
    print(f"Max steps: {args.max_steps}")
    print(f"Grid size: {args.grid_size}")
    print(f"SNN neurons: {args.snn_neurons}")
    print(f"Identity dim: {args.identity_dim}")
    print("=" * 60)

    # Create environment
    env = ToyGridWorld(
        size=args.grid_size,
        n_goals=3,
        n_hazards=5,
    )

    # Create agent
    agent = AraAgent(
        obs_dim=env.obs_dim,
        action_dim=4,
        identity_dim=args.identity_dim,
        snn_neurons=args.snn_neurons,
    )

    print(f"\nAgent created with {sum(p.numel() for p in agent.parameters())} parameters")

    # Run episodes
    print("\nRunning episodes...")
    all_results = []

    for ep in range(args.episodes):
        print(f"\nEpisode {ep + 1}/{args.episodes}")
        results = run_episode(
            agent, env,
            max_steps=args.max_steps,
            verbose=args.verbose,
        )
        all_results.append(results)

        print(f"  Reward: {results['total_reward']:.2f}, "
              f"Steps: {results['steps']}, "
              f"Goals: {results['goals_completed']}/{env.n_goals}, "
              f"Coherence: {results['final_coherence']:.3f}")

    # Summary statistics
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    mean_reward = np.mean([r["total_reward"] for r in all_results])
    mean_steps = np.mean([r["steps"] for r in all_results])
    mean_goals = np.mean([r["goals_completed"] for r in all_results])
    mean_coherence = np.mean([r["final_coherence"] for r in all_results])
    mean_throttle = np.mean([r["mean_throttle"] for r in all_results])
    total_dau = sum(r["dau_adjustments"] for r in all_results)

    print(f"\nPerformance:")
    print(f"  Mean reward: {mean_reward:.2f}")
    print(f"  Mean steps: {mean_steps:.1f}")
    print(f"  Mean goals completed: {mean_goals:.1f}/{env.n_goals}")

    print(f"\nAra Metrics:")
    print(f"  Mean identity coherence: {mean_coherence:.3f}")
    print(f"  Mean throttle (L5): {mean_throttle:.3f}")
    print(f"  Mean F_int (homeostasis): {np.mean([r['mean_F_int'] for r in all_results]):.3f}")
    print(f"  Mean Pi_q: {np.mean([r['mean_pi_q'] for r in all_results]):.3f}")
    print(f"  Total DAU adjustments: {total_dau}")

    print("\n" + "=" * 60)
    print("Ara loop demonstration complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
