# tfan_agent/buffers.py
# Trajectory and Experience Buffers for RL Training
#
# Provides simple data structures for collecting and processing
# trajectories in the Phase 1 training loop.

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

import torch


@dataclass
class Trajectory:
    """
    Container for a single episode trajectory.

    Stores the sequence of observations, actions, rewards, and auxiliary
    information needed for policy gradient training.
    """
    obs: List[torch.Tensor] = field(default_factory=list)
    actions: List[torch.Tensor] = field(default_factory=list)
    log_probs: List[torch.Tensor] = field(default_factory=list)
    rewards: List[float] = field(default_factory=list)
    values: List[torch.Tensor] = field(default_factory=list)
    dones: List[bool] = field(default_factory=list)

    # Optional: internal state logs for visualization
    homeostat_states: List[Dict[str, float]] = field(default_factory=list)
    gate_values: List[Dict[str, float]] = field(default_factory=list)
    appraisals: List[torch.Tensor] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.rewards)

    def add_step(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        log_prob: torch.Tensor,
        reward: float,
        value: Optional[torch.Tensor] = None,
        done: bool = False,
        homeostat_state: Optional[Dict[str, float]] = None,
        gate_values: Optional[Dict[str, float]] = None,
        appraisal: Optional[torch.Tensor] = None,
    ):
        """Add a single step to the trajectory."""
        self.obs.append(obs.detach().cpu())
        self.actions.append(action.detach().cpu())
        self.log_probs.append(log_prob.detach().cpu())
        self.rewards.append(reward)
        self.dones.append(done)

        if value is not None:
            self.values.append(value.detach().cpu())
        if homeostat_state is not None:
            self.homeostat_states.append(homeostat_state)
        if gate_values is not None:
            self.gate_values.append(gate_values)
        if appraisal is not None:
            self.appraisals.append(appraisal.detach().cpu())

    def to_tensors(self, device: str = "cpu") -> Dict[str, torch.Tensor]:
        """
        Convert trajectory to batched tensors for training.

        Returns:
            Dict with:
                - obs: (T, obs_dim)
                - actions: (T,)
                - log_probs: (T,)
                - rewards: (T,)
                - values: (T,) if available
        """
        result = {
            "obs": torch.stack(self.obs).to(device),
            "actions": torch.stack(self.actions).to(device),
            "log_probs": torch.stack(self.log_probs).to(device),
            "rewards": torch.tensor(self.rewards, dtype=torch.float32, device=device),
        }

        if self.values:
            result["values"] = torch.stack(self.values).squeeze(-1).to(device)

        if self.appraisals:
            result["appraisals"] = torch.stack(self.appraisals).to(device)

        return result

    def compute_returns(
        self,
        gamma: float = 0.99,
        normalize: bool = True,
    ) -> torch.Tensor:
        """
        Compute discounted returns for each timestep.

        G_t = r_t + gamma * r_{t+1} + gamma^2 * r_{t+2} + ...

        Args:
            gamma: discount factor
            normalize: whether to normalize returns (zero mean, unit std)

        Returns:
            returns: (T,) tensor of discounted returns
        """
        returns = []
        G = 0.0

        for r in reversed(self.rewards):
            G = r + gamma * G
            returns.append(G)

        returns = torch.tensor(list(reversed(returns)), dtype=torch.float32)

        if normalize and len(returns) > 1:
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        return returns

    def compute_gae(
        self,
        gamma: float = 0.99,
        lam: float = 0.95,
        normalize: bool = True,
    ) -> torch.Tensor:
        """
        Compute Generalized Advantage Estimation (GAE).

        A_t = sum_{l=0}^{inf} (gamma * lam)^l * delta_{t+l}
        where delta_t = r_t + gamma * V_{t+1} - V_t

        Args:
            gamma: discount factor
            lam: GAE lambda parameter
            normalize: whether to normalize advantages

        Returns:
            advantages: (T,) tensor
        """
        if not self.values:
            # Fall back to simple returns - baseline
            return self.compute_returns(gamma, normalize)

        values = torch.stack(self.values).squeeze(-1)
        rewards = torch.tensor(self.rewards, dtype=torch.float32)

        advantages = []
        gae = 0.0

        for t in reversed(range(len(self.rewards))):
            if t == len(self.rewards) - 1:
                next_value = 0.0  # Terminal state
            else:
                next_value = values[t + 1].item()

            delta = rewards[t] + gamma * next_value - values[t].item()
            gae = delta + gamma * lam * gae
            advantages.append(gae)

        advantages = torch.tensor(list(reversed(advantages)), dtype=torch.float32)

        if normalize and len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        return advantages

    def get_summary(self) -> Dict[str, float]:
        """Get summary statistics for logging."""
        return {
            "length": len(self.rewards),
            "total_reward": sum(self.rewards),
            "mean_reward": sum(self.rewards) / max(1, len(self.rewards)),
            "max_reward": max(self.rewards) if self.rewards else 0.0,
            "min_reward": min(self.rewards) if self.rewards else 0.0,
        }


@dataclass
class RolloutBuffer:
    """
    Buffer for collecting multiple trajectories for batch training.
    """
    trajectories: List[Trajectory] = field(default_factory=list)
    current_trajectory: Trajectory = field(default_factory=Trajectory)

    def start_trajectory(self):
        """Start a new trajectory."""
        self.current_trajectory = Trajectory()

    def add_step(self, **kwargs):
        """Add step to current trajectory."""
        self.current_trajectory.add_step(**kwargs)

    def end_trajectory(self):
        """Finish current trajectory and add to buffer."""
        if len(self.current_trajectory) > 0:
            self.trajectories.append(self.current_trajectory)
        self.current_trajectory = Trajectory()

    def clear(self):
        """Clear all trajectories."""
        self.trajectories = []
        self.current_trajectory = Trajectory()

    def __len__(self) -> int:
        return len(self.trajectories)

    def get_batch(
        self,
        gamma: float = 0.99,
        lam: float = 0.95,
        device: str = "cpu",
    ) -> Dict[str, torch.Tensor]:
        """
        Get all trajectories as a single batch for training.

        Returns:
            Dict with concatenated tensors from all trajectories
        """
        all_obs = []
        all_actions = []
        all_log_probs = []
        all_advantages = []
        all_returns = []

        for traj in self.trajectories:
            tensors = traj.to_tensors(device)
            advantages = traj.compute_gae(gamma, lam, normalize=False)
            returns = traj.compute_returns(gamma, normalize=False)

            all_obs.append(tensors["obs"])
            all_actions.append(tensors["actions"])
            all_log_probs.append(tensors["log_probs"])
            all_advantages.append(advantages.to(device))
            all_returns.append(returns.to(device))

        # Concatenate all trajectories
        batch = {
            "obs": torch.cat(all_obs, dim=0),
            "actions": torch.cat(all_actions, dim=0),
            "log_probs": torch.cat(all_log_probs, dim=0),
            "advantages": torch.cat(all_advantages, dim=0),
            "returns": torch.cat(all_returns, dim=0),
        }

        # Normalize advantages across full batch
        batch["advantages"] = (batch["advantages"] - batch["advantages"].mean()) / (
            batch["advantages"].std() + 1e-8
        )

        return batch

    def get_summary(self) -> Dict[str, float]:
        """Get summary statistics for all trajectories."""
        if not self.trajectories:
            return {}

        total_steps = sum(len(t) for t in self.trajectories)
        total_reward = sum(sum(t.rewards) for t in self.trajectories)
        mean_length = total_steps / len(self.trajectories)
        mean_reward = total_reward / len(self.trajectories)

        return {
            "num_trajectories": len(self.trajectories),
            "total_steps": total_steps,
            "mean_length": mean_length,
            "mean_episode_reward": mean_reward,
        }


if __name__ == "__main__":
    # Quick sanity check
    print("=== Buffers Sanity Check ===")

    # Test single trajectory
    traj = Trajectory()
    for i in range(10):
        traj.add_step(
            obs=torch.randn(4),
            action=torch.tensor(i % 3),
            log_prob=torch.tensor(-0.5),
            reward=float(i) * 0.1,
            value=torch.tensor([[0.5]]),
        )

    tensors = traj.to_tensors()
    print(f"Trajectory length: {len(traj)}")
    print(f"Obs shape: {tensors['obs'].shape}")
    print(f"Returns shape: {traj.compute_returns().shape}")
    print(f"GAE shape: {traj.compute_gae().shape}")
    print(f"Summary: {traj.get_summary()}")

    # Test rollout buffer
    buffer = RolloutBuffer()
    for ep in range(3):
        buffer.start_trajectory()
        for t in range(5 + ep):
            buffer.add_step(
                obs=torch.randn(4),
                action=torch.tensor(t % 3),
                log_prob=torch.tensor(-0.5),
                reward=0.1,
                value=torch.tensor([[0.5]]),
            )
        buffer.end_trajectory()

    batch = buffer.get_batch()
    print(f"\nBuffer summary: {buffer.get_summary()}")
    print(f"Batch obs shape: {batch['obs'].shape}")
    print(f"Batch advantages shape: {batch['advantages'].shape}")

    print("\nBuffers sanity check passed!")
