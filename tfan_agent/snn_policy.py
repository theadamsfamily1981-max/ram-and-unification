# tfan_agent/snn_policy.py
# Phase 1: Simple "Spiking-ish" Policy Network
#
# This is a rate-coded approximation of a spiking neural network (SNN).
# It maintains membrane potentials that leaky-integrate inputs and
# produce spike-like activations via a surrogate gradient.
#
# Key features:
#   - LIF (Leaky Integrate-and-Fire) dynamics
#   - Surrogate spike activation (fast sigmoid)
#   - Stateful (maintains V across timesteps within episode)
#   - Accepts gating parameters to modulate behavior
#
# Phase 2 will replace this with a proper SNN (Norse/snnTorch).

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class PolicyConfig:
    """
    Configuration for the Phase 1 'spiking-ish' policy network.

    For now this is an MLP with simple LIF-like state dynamics,
    which we can later replace with a proper SNN.
    """
    obs_dim: int
    gating_dim: int = 4             # number of gates we feed in (temp, lr, mem, aux)
    hidden_dim: int = 128
    num_actions: int = 4
    tau: float = 0.9                # membrane time constant
    v_threshold: float = 1.0        # spike threshold
    device: str = "cpu"


class SurrogateSpike(torch.autograd.Function):
    """
    Surrogate gradient for spike function.

    Forward: step function (v > threshold)
    Backward: smooth gradient (fast sigmoid derivative)
    """

    @staticmethod
    def forward(ctx, v: torch.Tensor, threshold: float = 1.0, scale: float = 5.0):
        ctx.save_for_backward(v)
        ctx.threshold = threshold
        ctx.scale = scale
        return (v > threshold).float()

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        v, = ctx.saved_tensors
        # Fast sigmoid surrogate gradient
        x = ctx.scale * (v - ctx.threshold)
        sigmoid = torch.sigmoid(x)
        grad = grad_output * ctx.scale * sigmoid * (1 - sigmoid)
        return grad, None, None


def surrogate_spike(v: torch.Tensor, threshold: float = 1.0, scale: float = 5.0) -> torch.Tensor:
    """Apply surrogate spike function with gradient."""
    return SurrogateSpike.apply(v, threshold, scale)


class LIFLayer(nn.Module):
    """
    Leaky Integrate-and-Fire layer with surrogate gradients.

    This layer maintains membrane potentials that:
    1. Leak toward zero (tau decay)
    2. Integrate weighted inputs
    3. Produce spikes when crossing threshold
    4. Reset after spiking
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        tau: float = 0.9,
        v_threshold: float = 1.0,
        device: str = "cpu",
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.tau = tau
        self.v_threshold = v_threshold
        self.device = torch.device(device)

        # Synaptic weights
        self.fc = nn.Linear(in_features, out_features).to(self.device)

        # Membrane potential state
        self.register_buffer(
            "V",
            torch.zeros(out_features, device=self.device)
        )

    def reset_state(self, batch_size: int = 1):
        """Reset membrane potentials between episodes."""
        self.V = torch.zeros(batch_size, self.out_features, device=self.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with LIF dynamics.

        Args:
            x: (B, in_features) input

        Returns:
            spikes: (B, out_features) spike output in [0, 1]
        """
        B = x.size(0)

        # Handle batch size mismatch
        if self.V.dim() == 1 or self.V.size(0) != B:
            self.reset_state(B)

        # Compute synaptic input
        I = self.fc(x.to(self.device))

        # Leaky integration: V = tau * V + (1-tau) * I
        self.V = self.tau * self.V + (1.0 - self.tau) * I

        # Spike with surrogate gradient
        spikes = surrogate_spike(self.V, self.v_threshold)

        # Soft reset: reduce V proportional to spike
        self.V = self.V * (1.0 - 0.5 * spikes)

        return spikes


class SimpleSpikingPolicy(nn.Module):
    """
    Phase 1 approximation of a spiking policy network.

    Architecture:
        [obs, gates] -> LIF hidden -> LIF hidden -> Linear -> action logits

    The network maintains membrane potentials across timesteps within
    an episode, giving it a simple form of temporal integration.

    Gating parameters from L3 can modulate behavior:
        - Temperature affects action selection (via logits scaling)
        - Other gates available for future use

    Later we can:
        - Replace with a real SNN (Norse/snnTorch)
        - Embed on manifolds (hyperbolic output space)
        - Add recurrent connections
    """

    def __init__(self, config: PolicyConfig):
        super().__init__()
        self.config = config
        self.device = torch.device(config.device)

        in_dim = config.obs_dim + config.gating_dim
        hidden = config.hidden_dim

        # LIF layers
        self.lif1 = LIFLayer(in_dim, hidden, config.tau, config.v_threshold, config.device)
        self.lif2 = LIFLayer(hidden, hidden, config.tau, config.v_threshold, config.device)

        # Output layer (rate-coded, not spiking)
        self.fc_out = nn.Linear(hidden, config.num_actions).to(self.device)

        # Value head for actor-critic (optional)
        self.value_head = nn.Linear(hidden, 1).to(self.device)

    def reset_state(self, batch_size: int = 1):
        """
        Reset membrane potentials between episodes.

        Call this at the start of each episode to clear temporal state.
        """
        self.lif1.reset_state(batch_size)
        self.lif2.reset_state(batch_size)

    def forward(
        self,
        obs: torch.Tensor,
        gates: torch.Tensor,
        temperature: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass to compute action distribution.

        Args:
            obs:   (B, obs_dim) - observation from environment
            gates: (B, gating_dim) - gating parameters from L3
            temperature: (B, 1) or None - optional temperature override

        Returns:
            dict with:
                - "logits": (B, num_actions) - raw action scores
                - "probs":  (B, num_actions) - action probabilities
                - "value":  (B, 1) - state value estimate (for A2C)
        """
        # Concatenate obs and gates
        x = torch.cat([obs.to(self.device), gates.to(self.device)], dim=-1)

        # Forward through LIF layers
        h1 = self.lif1(x)
        h2 = self.lif2(h1)

        # Compute logits
        logits = self.fc_out(h2)

        # Apply temperature if provided
        if temperature is not None:
            logits = logits / (temperature.to(self.device) + 1e-8)

        # Action probabilities
        probs = F.softmax(logits, dim=-1)

        # Value estimate
        value = self.value_head(h2)

        return {
            "logits": logits,
            "probs": probs,
            "value": value,
        }

    def get_action(
        self,
        obs: torch.Tensor,
        gates: torch.Tensor,
        temperature: Optional[torch.Tensor] = None,
        deterministic: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Sample action from policy.

        Args:
            obs: observation
            gates: gating parameters
            temperature: optional temperature
            deterministic: if True, take argmax instead of sampling

        Returns:
            dict with action, log_prob, probs, value
        """
        out = self.forward(obs, gates, temperature)
        probs = out["probs"]

        if deterministic:
            action = probs.argmax(dim=-1)
            log_prob = torch.log(probs.gather(-1, action.unsqueeze(-1)) + 1e-8).squeeze(-1)
        else:
            dist = torch.distributions.Categorical(probs=probs)
            action = dist.sample()
            log_prob = dist.log_prob(action)

        return {
            "action": action,
            "log_prob": log_prob,
            "probs": probs,
            "value": out["value"],
            "logits": out["logits"],
        }


class RateCodingPolicy(nn.Module):
    """
    Simpler rate-coded policy without explicit LIF dynamics.

    This is a fallback for when we don't need temporal dynamics
    but want the same interface.
    """

    def __init__(self, config: PolicyConfig):
        super().__init__()
        self.config = config
        self.device = torch.device(config.device)

        in_dim = config.obs_dim + config.gating_dim
        hidden = config.hidden_dim

        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        ).to(self.device)

        self.action_head = nn.Linear(hidden, config.num_actions).to(self.device)
        self.value_head = nn.Linear(hidden, 1).to(self.device)

    def reset_state(self, batch_size: int = 1):
        """No state to reset in rate coding."""
        pass

    def forward(
        self,
        obs: torch.Tensor,
        gates: torch.Tensor,
        temperature: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        x = torch.cat([obs.to(self.device), gates.to(self.device)], dim=-1)
        h = self.net(x)

        logits = self.action_head(h)
        if temperature is not None:
            logits = logits / (temperature.to(self.device) + 1e-8)

        probs = F.softmax(logits, dim=-1)
        value = self.value_head(h)

        return {"logits": logits, "probs": probs, "value": value}


if __name__ == "__main__":
    # Quick sanity check
    print("=== SimpleSpikingPolicy Sanity Check ===")

    config = PolicyConfig(
        obs_dim=8,
        gating_dim=4,
        hidden_dim=32,
        num_actions=4,
        device="cpu",
    )
    policy = SimpleSpikingPolicy(config)

    # Test forward pass
    B = 4
    obs = torch.randn(B, 8)
    gates = torch.rand(B, 4)
    temperature = torch.ones(B, 1) * 1.5

    policy.reset_state(B)

    # Run several steps to test temporal dynamics
    for step in range(5):
        out = policy(obs, gates, temperature)
        print(f"Step {step+1}: probs mean={out['probs'].mean():.3f}, value mean={out['value'].mean():.3f}")

    # Test action sampling
    action_out = policy.get_action(obs, gates, temperature)
    print(f"\nSampled actions: {action_out['action'].tolist()}")
    print(f"Log probs: {action_out['log_prob'].tolist()}")

    # Test deterministic mode
    action_det = policy.get_action(obs, gates, temperature, deterministic=True)
    print(f"Deterministic actions: {action_det['action'].tolist()}")

    # Check gradient flow through surrogate
    loss = out["probs"].sum()
    loss.backward()
    grad_norm = sum(p.grad.norm().item() for p in policy.parameters() if p.grad is not None)
    print(f"\nGradient norm: {grad_norm:.4f}")

    print("\nSimpleSpikingPolicy sanity check passed!")
