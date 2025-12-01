# tfan_agent/training_loop.py
# Phase 1 Training Loop: REINFORCE with Homeostatic Reward
#
# This module ties together all Phase 1 components:
#   - L1: HomeostatL1 (homeostatic core)
#   - L2: AppraisalHeadL2 (cognitive appraisal)
#   - L3: GatingControllerL3 (adaptive gating)
#   - Policy: SimpleSpikingPolicy
#
# The training loop uses REINFORCE (policy gradient) with the
# homeostatic reward signal from L1.

from __future__ import annotations
from typing import Dict, Optional
import time

import torch
import torch.nn as nn
import torch.optim as optim

from .homeostat import HomeostatL1, HomeostatConfig, HomeostatState
from .appraisal import AppraisalHeadL2, AppraisalConfig
from .gating import GatingControllerL3, GatingConfig
from .snn_policy import SimpleSpikingPolicy, PolicyConfig
from .buffers import Trajectory, RolloutBuffer
from .simple_env import SimpleHomeostaticEnv, SimpleHomeostaticEnvConfig


class TfanAgent(nn.Module):
    """
    Complete Phase 1 T-FAN Agent.

    Integrates all four layers:
        - L1: Homeostatic Core (generates intrinsic reward)
        - L2: Appraisal Engine (assesses situations)
        - L3: Gating Controller (modulates policy)
        - Policy: Spiking policy network (selects actions)

    The agent's motivation comes entirely from maintaining homeostasis,
    not from external task rewards. This is the core of embodied AI.
    """

    def __init__(
        self,
        obs_dim: int,
        num_needs: int,
        num_actions: int,
        hidden_dim: int = 128,
        device: str = "cpu",
    ):
        super().__init__()
        self.device = torch.device(device)
        self.obs_dim = obs_dim
        self.num_needs = num_needs
        self.num_actions = num_actions

        # L1: Homeostatic Core
        self.homeostat = HomeostatL1(
            HomeostatConfig(num_needs=num_needs, device=device)
        )

        # L2: Appraisal Engine
        self.appraisal = AppraisalHeadL2(
            AppraisalConfig(
                obs_dim=obs_dim,
                num_needs=num_needs,
                goal_dim=0,
                hidden_dim=hidden_dim,
                device=device,
            )
        )

        # L3: Gating Controller
        self.gating = GatingControllerL3(
            GatingConfig(appraisal_dim=8, hidden_dim=hidden_dim // 2, device=device)
        )

        # Policy Network
        self.policy = SimpleSpikingPolicy(
            PolicyConfig(
                obs_dim=obs_dim,
                gating_dim=4,      # temperature, lr_scale, mem_write_p, aux
                hidden_dim=hidden_dim,
                num_actions=num_actions,
                device=device,
            )
        )

        # Move to device
        self.to(self.device)

    def reset(self, init_needs: Optional[torch.Tensor] = None) -> HomeostatState:
        """
        Reset agent state for new episode.

        Args:
            init_needs: Optional initial needs vector

        Returns:
            Initial HomeostatState
        """
        hs_state = self.homeostat.reset(init_needs)
        self.policy.reset_state(batch_size=1)
        return hs_state

    def forward_step(
        self,
        obs: torch.Tensor,
        epistemic_unc: float = 0.0,
        aleatoric_unc: float = 0.0,
    ) -> Dict[str, torch.Tensor]:
        """
        Single forward pass for one decision step.

        This is the main inference path:
        1. Get current needs from L1
        2. Compute appraisal from obs + needs (L2)
        3. Get PAD affect from L1
        4. Compute gating from PAD + uncertainty + appraisal (L3)
        5. Get action distribution from policy

        Args:
            obs: (B, obs_dim) observation batch
            epistemic_unc: Epistemic uncertainty estimate
            aleatoric_unc: Aleatoric uncertainty estimate

        Returns:
            Dict with probs, logits, gates, value, appraisal
        """
        B = obs.size(0)
        device = self.device

        # Get current needs
        needs = self.homeostat.n.unsqueeze(0).expand(B, -1)

        # L2: Compute appraisal
        appraisal = self.appraisal(obs.to(device), needs)

        # L1: Get PAD affect
        V, A, D, d = self.homeostat._compute_pad_and_drive(self.homeostat.n)

        # Convert to batched tensors
        V_t = torch.full((B,), float(V), device=device)
        A_t = torch.full((B,), float(A), device=device)
        D_t = torch.full((B,), float(D), device=device)
        epistemic_t = torch.tensor([epistemic_unc], device=device)
        aleatoric_t = torch.tensor([aleatoric_unc], device=device)

        # L3: Compute gating
        gates_dict = self.gating(
            V_t, A_t, D_t, epistemic_t, aleatoric_t, appraisal
        )

        # Build gate vector for policy
        gates_vec = torch.cat([
            gates_dict["temperature"],
            gates_dict["lr_scale"],
            gates_dict["mem_write_p"],
            gates_dict["aux_gain"],
        ], dim=-1)

        # Policy: Get action distribution
        policy_out = self.policy(obs.to(device), gates_vec, gates_dict["temperature"])

        return {
            "probs": policy_out["probs"],
            "logits": policy_out["logits"],
            "value": policy_out["value"],
            "gates": gates_dict,
            "appraisal": appraisal,
            "homeostat_n": self.homeostat.n.clone(),
        }

    def get_action(
        self,
        obs: torch.Tensor,
        epistemic_unc: float = 0.0,
        aleatoric_unc: float = 0.0,
        deterministic: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Sample action from policy.

        Returns:
            Dict with action, log_prob, value, etc.
        """
        out = self.forward_step(obs, epistemic_unc, aleatoric_unc)

        if deterministic:
            action = out["probs"].argmax(dim=-1)
            log_prob = torch.log(out["probs"].gather(-1, action.unsqueeze(-1)) + 1e-8).squeeze(-1)
        else:
            dist = torch.distributions.Categorical(probs=out["probs"])
            action = dist.sample()
            log_prob = dist.log_prob(action)

        return {
            "action": action,
            "log_prob": log_prob,
            "value": out["value"],
            "probs": out["probs"],
            "gates": out["gates"],
            "appraisal": out["appraisal"],
        }

    def update_homeostat(self, need_delta: torch.Tensor) -> Dict[str, float | HomeostatState]:
        """
        Update homeostatic state with need delta.

        Args:
            need_delta: Change in needs from environment

        Returns:
            Dict with reward and state
        """
        return self.homeostat(need_delta)


def train_phase1(
    num_episodes: int = 200,
    max_steps_per_episode: int = 64,
    batch_episodes: int = 4,
    gamma: float = 0.99,
    lr: float = 3e-4,
    entropy_coef: float = 0.01,
    value_coef: float = 0.5,
    device: str = "cpu",
    verbose: bool = True,
) -> Dict[str, list]:
    """
    Phase 1 training loop using REINFORCE with homeostatic reward.

    This demonstrates the core idea: the agent learns to take actions
    that reduce its internal free energy (maintain homeostasis).

    Args:
        num_episodes: Total episodes to train
        max_steps_per_episode: Max steps per episode
        batch_episodes: Episodes per training batch
        gamma: Discount factor
        lr: Learning rate
        entropy_coef: Entropy bonus coefficient (encourages exploration)
        value_coef: Value loss coefficient (for actor-critic)
        device: Device to train on
        verbose: Print progress

    Returns:
        Dict with training history (losses, rewards, etc.)
    """
    # Create environment
    env_cfg = SimpleHomeostaticEnvConfig(
        obs_dim=4,
        num_actions=4,
        num_needs=8,
        max_steps=max_steps_per_episode,
        device=device,
    )
    env = SimpleHomeostaticEnv(env_cfg)

    # Create agent
    agent = TfanAgent(
        obs_dim=env_cfg.obs_dim,
        num_needs=env_cfg.num_needs,
        num_actions=env_cfg.num_actions,
        device=device,
    )

    # Optimizer
    optimizer = optim.Adam(agent.parameters(), lr=lr)

    # Training history
    history = {
        "episode": [],
        "loss": [],
        "policy_loss": [],
        "value_loss": [],
        "entropy": [],
        "mean_reward": [],
        "mean_free_energy": [],
        "mean_valence": [],
        "mean_arousal": [],
    }

    # Training loop
    buffer = RolloutBuffer()
    episode = 0

    start_time = time.time()

    while episode < num_episodes:
        # Collect batch of episodes
        buffer.clear()

        for _ in range(batch_episodes):
            if episode >= num_episodes:
                break

            # Reset for new episode
            obs, _ = env.reset()
            agent.reset()

            traj = Trajectory()
            episode_free_energy = []
            episode_valence = []
            episode_arousal = []

            for t in range(max_steps_per_episode):
                obs_batch = obs.unsqueeze(0)

                # Get action
                action_out = agent.get_action(obs_batch)
                action = action_out["action"].item()
                log_prob = action_out["log_prob"]
                value = action_out["value"]

                # Step environment
                obs_next, need_delta, terminated, truncated, info = env.step(action)

                # Update homeostatic state
                hs_out = agent.update_homeostat(need_delta)
                reward = hs_out["reward"]
                hs_state = hs_out["state"]

                # Store step
                traj.add_step(
                    obs=obs_batch.squeeze(0),
                    action=action_out["action"],
                    log_prob=log_prob.squeeze(0),
                    reward=reward,
                    value=value,
                    done=terminated or truncated,
                )

                # Track internal state
                episode_free_energy.append(hs_state.free_energy)
                episode_valence.append(hs_state.valence)
                episode_arousal.append(hs_state.arousal)

                obs = obs_next

                if terminated or truncated:
                    break

            buffer.trajectories.append(traj)
            episode += 1

        # Train on collected batch
        if len(buffer) > 0:
            batch = buffer.get_batch(gamma=gamma, device=device)

            # Recompute policy outputs for gradient
            agent.policy.reset_state(batch["obs"].size(0))
            needs = agent.homeostat.n.unsqueeze(0).expand(batch["obs"].size(0), -1)
            appraisal = agent.appraisal(batch["obs"], needs)

            V, A, D, _ = agent.homeostat._compute_pad_and_drive(agent.homeostat.n)
            V_t = torch.full((batch["obs"].size(0),), float(V), device=device)
            A_t = torch.full((batch["obs"].size(0),), float(A), device=device)
            D_t = torch.full((batch["obs"].size(0),), float(D), device=device)

            gates_dict = agent.gating(
                V_t, A_t, D_t,
                torch.tensor([0.0], device=device),
                torch.tensor([0.0], device=device),
                appraisal
            )

            gates_vec = torch.cat([
                gates_dict["temperature"],
                gates_dict["lr_scale"],
                gates_dict["mem_write_p"],
                gates_dict["aux_gain"],
            ], dim=-1)

            policy_out = agent.policy(batch["obs"], gates_vec)

            # Compute losses
            dist = torch.distributions.Categorical(probs=policy_out["probs"])
            new_log_probs = dist.log_prob(batch["actions"])
            entropy = dist.entropy().mean()

            # Policy loss (REINFORCE)
            policy_loss = -(new_log_probs * batch["advantages"]).mean()

            # Value loss
            value_loss = (policy_out["value"].squeeze(-1) - batch["returns"]).pow(2).mean()

            # Total loss
            loss = policy_loss + value_coef * value_loss - entropy_coef * entropy

            # Update
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(agent.parameters(), max_norm=0.5)
            optimizer.step()

            # Log
            summary = buffer.get_summary()
            history["episode"].append(episode)
            history["loss"].append(loss.item())
            history["policy_loss"].append(policy_loss.item())
            history["value_loss"].append(value_loss.item())
            history["entropy"].append(entropy.item())
            history["mean_reward"].append(summary["mean_episode_reward"])
            history["mean_free_energy"].append(sum(episode_free_energy) / max(1, len(episode_free_energy)))
            history["mean_valence"].append(sum(episode_valence) / max(1, len(episode_valence)))
            history["mean_arousal"].append(sum(episode_arousal) / max(1, len(episode_arousal)))

            if verbose and episode % 10 == 0:
                elapsed = time.time() - start_time
                print(
                    f"[Episode {episode}/{num_episodes}] "
                    f"loss={loss.item():.3f} "
                    f"reward={summary['mean_episode_reward']:.3f} "
                    f"F_int={history['mean_free_energy'][-1]:.3f} "
                    f"V={history['mean_valence'][-1]:.3f} "
                    f"A={history['mean_arousal'][-1]:.3f} "
                    f"({elapsed:.1f}s)"
                )

    if verbose:
        total_time = time.time() - start_time
        print(f"\nTraining complete in {total_time:.1f}s")
        print(f"Final mean reward: {history['mean_reward'][-1]:.3f}")

    return history


if __name__ == "__main__":
    print("=== T-FAN Phase 1 Training ===\n")

    # Quick training run
    history = train_phase1(
        num_episodes=50,
        max_steps_per_episode=32,
        batch_episodes=4,
        device="cpu",
        verbose=True,
    )

    print("\n=== Training History Summary ===")
    print(f"Episodes: {len(history['episode'])}")
    print(f"Final loss: {history['loss'][-1]:.4f}")
    print(f"Final mean reward: {history['mean_reward'][-1]:.4f}")
    print(f"Final mean free energy: {history['mean_free_energy'][-1]:.4f}")
