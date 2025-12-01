"""
Layer 1: Homeostatic Core

Implements:
- Need vector n(t) ∈ ℝ^k
- Drive vector d(t) = n(t) - n* (deviation from setpoint)
- Internal free energy F_int(n) = ½ d^T Σ^{-1} d
- HRRL reward r_t = -ΔF_int = -(F_int(t+1) - F_int(t))
- PAD affect: Valence, Arousal, Dominance

Dynamics: dn/dt = -Γ d(t) + ξ(t) + u(t)
"""
import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Tuple, Optional, NamedTuple

from .config import L1Config


class HomeostatState(NamedTuple):
    """State returned by homeostat step."""
    n: torch.Tensor  # Need vector
    d: torch.Tensor  # Drive vector
    f_int: float  # Internal free energy
    reward: float  # HRRL reward -ΔF_int
    valence: float  # PAD: Pleasure
    arousal: float  # PAD: Arousal
    dominance: float  # PAD: Dominance
    n_dot: torch.Tensor  # Rate of change


class HomeostatL1(nn.Module):
    """
    Homeostatic Core Layer.

    Maintains internal need state and computes HRRL reward signal.

    The free energy F_int(n) represents how far the agent is from
    homeostatic equilibrium. The HRRL reward r = -ΔF_int encourages
    actions that reduce this deviation.
    """

    def __init__(self, config: L1Config):
        super().__init__()
        self.config = config
        self.k = config.num_needs

        # Setpoint n* (optimal need levels)
        self.register_buffer(
            'setpoint',
            torch.full((self.k,), config.setpoint_default)
        )

        # Precision matrix (inverse covariance) for F_int
        # Diagonal for simplicity: Σ^{-1} = diag(inv_sigma)
        self.register_buffer(
            'inv_sigma',
            torch.full((self.k,), config.inv_sigma_default)
        )

        # Current need state n(t)
        self.register_buffer(
            'n',
            torch.full((self.k,), config.setpoint_default)
        )

        # Previous free energy for reward computation
        self._f_int_prev: float = 0.0

        # Initialize
        self.reset()

    def reset(self, n_init: Optional[torch.Tensor] = None):
        """Reset to initial state."""
        if n_init is not None:
            self.n.copy_(n_init)
        else:
            self.n.fill_(self.config.setpoint_default)

        self._f_int_prev = self._compute_free_energy(self.n).item()

    @property
    def drive(self) -> torch.Tensor:
        """Current drive vector d(t) = n(t) - n*."""
        return self.n - self.setpoint

    @property
    def free_energy(self) -> float:
        """Current internal free energy F_int."""
        return self._compute_free_energy(self.n).item()

    def _compute_free_energy(self, n: torch.Tensor) -> torch.Tensor:
        """
        Compute internal free energy.

        F_int(n) = ½ d^T Σ^{-1} d = ½ Σ_i (n_i - n*_i)² / σ_i²
        """
        d = n - self.setpoint
        return 0.5 * torch.sum(d * self.inv_sigma * d)

    def _compute_free_energy_gradient(self, n: torch.Tensor) -> torch.Tensor:
        """
        Gradient of free energy w.r.t. needs.

        ∇F_int = Σ^{-1} d = Σ^{-1} (n - n*)
        """
        d = n - self.setpoint
        return self.inv_sigma * d

    def step(
        self,
        u: torch.Tensor,
        xi: Optional[torch.Tensor] = None,
        sigma_unc: Optional[float] = None
    ) -> HomeostatState:
        """
        Perform one homeostatic timestep.

        Dynamics: dn/dt = -Γ d(t) + ξ(t) + u(t)

        Args:
            u: Control input (actions affecting needs)
            xi: External disturbance (optional, sampled if None)
            sigma_unc: Uncertainty scaling for noise (optional)

        Returns:
            HomeostatState with all state variables and reward
        """
        # Current drive
        d = self.drive.clone()

        # Process noise
        if xi is None:
            sigma = self.config.sigma_process
            if sigma_unc is not None:
                sigma = sigma * (1.0 + sigma_unc)
            xi = torch.randn_like(self.n) * sigma

        # Compute rate of change: dn/dt = -Γd + ξ + u
        n_dot = -self.config.gamma * d + xi + u

        # Euler integration
        n_next = self.n + self.config.dt * n_dot

        # Clamp to valid range [0, ∞)
        n_next = n_next.clamp(min=0.0)

        # Compute free energies
        f_int_prev = self._f_int_prev
        f_int_next = self._compute_free_energy(n_next).item()

        # HRRL reward: r_t = -ΔF_int
        reward = -(f_int_next - f_int_prev)

        # Compute PAD affect
        valence, arousal, dominance = self._compute_pad(d, n_dot, u)

        # Update state
        self.n.copy_(n_next)
        self._f_int_prev = f_int_next

        return HomeostatState(
            n=self.n.clone(),
            d=self.n - self.setpoint,
            f_int=f_int_next,
            reward=reward,
            valence=valence,
            arousal=arousal,
            dominance=dominance,
            n_dot=n_dot
        )

    def _compute_pad(
        self,
        d: torch.Tensor,
        n_dot: torch.Tensor,
        u: torch.Tensor
    ) -> Tuple[float, float, float]:
        """
        Compute PAD (Pleasure-Arousal-Dominance) affect.

        Valence (V): Pleasure from drive reduction
            V = -d · ṅ (positive when drives decrease)

        Arousal (A): Overall activation level
            A = ||d|| + ||ṅ||

        Dominance (D): Sense of control
            D = ||u|| / (||d|| + ε) (control relative to distress)
        """
        eps = 1e-6

        # Valence: pleasure from drive reduction
        valence = -torch.dot(d, n_dot).item()

        # Arousal: activation level
        d_norm = torch.norm(d).item()
        n_dot_norm = torch.norm(n_dot).item()
        arousal = d_norm + n_dot_norm

        # Dominance: sense of control
        u_norm = torch.norm(u).item()
        dominance = u_norm / (d_norm + eps)

        # Normalize to [-1, 1] or [0, 1] range
        valence = torch.tanh(torch.tensor(valence)).item()
        arousal = torch.sigmoid(torch.tensor(arousal)).item()
        dominance = torch.sigmoid(torch.tensor(dominance)).item()

        return valence, arousal, dominance

    def set_setpoint(self, new_setpoint: torch.Tensor):
        """Update homeostatic setpoint n*."""
        self.setpoint.copy_(new_setpoint)

    def set_precision(self, new_inv_sigma: torch.Tensor):
        """Update precision matrix diagonal."""
        self.inv_sigma.copy_(new_inv_sigma)

    def get_state_dict(self) -> dict:
        """Get serializable state."""
        return {
            'n': self.n.clone(),
            'setpoint': self.setpoint.clone(),
            'inv_sigma': self.inv_sigma.clone(),
            'f_int_prev': self._f_int_prev
        }

    def load_state_dict_custom(self, state: dict):
        """Load from serialized state."""
        self.n.copy_(state['n'])
        self.setpoint.copy_(state['setpoint'])
        self.inv_sigma.copy_(state['inv_sigma'])
        self._f_int_prev = state['f_int_prev']


class BatchedHomeostatL1(nn.Module):
    """
    Batched version of HomeostatL1 for parallel environments.

    All operations support batch dimension for efficiency.
    """

    def __init__(self, config: L1Config, batch_size: int):
        super().__init__()
        self.config = config
        self.k = config.num_needs
        self.batch_size = batch_size

        # Setpoints [B, k]
        self.register_buffer(
            'setpoint',
            torch.full((batch_size, self.k), config.setpoint_default)
        )

        # Precision [B, k]
        self.register_buffer(
            'inv_sigma',
            torch.full((batch_size, self.k), config.inv_sigma_default)
        )

        # Need state [B, k]
        self.register_buffer(
            'n',
            torch.full((batch_size, self.k), config.setpoint_default)
        )

        # Previous F_int [B]
        self.register_buffer(
            'f_int_prev',
            torch.zeros(batch_size)
        )

        self.reset()

    def reset(self, mask: Optional[torch.Tensor] = None):
        """Reset state (optionally only for masked indices)."""
        if mask is None:
            self.n.fill_(self.config.setpoint_default)
            self.f_int_prev.zero_()
        else:
            self.n[mask] = self.config.setpoint_default
            self.f_int_prev[mask] = 0.0

    @property
    def drive(self) -> torch.Tensor:
        """Drive vectors [B, k]."""
        return self.n - self.setpoint

    def _compute_free_energy(self, n: torch.Tensor) -> torch.Tensor:
        """Compute F_int for batch [B]."""
        d = n - self.setpoint
        return 0.5 * torch.sum(d * self.inv_sigma * d, dim=-1)

    def step(
        self,
        u: torch.Tensor,
        xi: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Batched step.

        Args:
            u: Control inputs [B, k]
            xi: Disturbances [B, k] (optional)

        Returns:
            rewards: HRRL rewards [B]
            n: New need states [B, k]
            d: New drive states [B, k]
        """
        d = self.drive

        if xi is None:
            xi = torch.randn_like(self.n) * self.config.sigma_process

        n_dot = -self.config.gamma * d + xi + u
        n_next = (self.n + self.config.dt * n_dot).clamp(min=0.0)

        f_int_next = self._compute_free_energy(n_next)
        rewards = -(f_int_next - self.f_int_prev)

        self.n.copy_(n_next)
        self.f_int_prev.copy_(f_int_next)

        return rewards, self.n.clone(), self.drive


if __name__ == "__main__":
    # Sanity check
    config = L1Config(num_needs=4)
    homeostat = HomeostatL1(config)

    print("Initial state:")
    print(f"  n = {homeostat.n}")
    print(f"  d = {homeostat.drive}")
    print(f"  F_int = {homeostat.free_energy:.4f}")

    # Simulate disturbance
    u = torch.zeros(4)
    for t in range(10):
        state = homeostat.step(u)
        print(f"Step {t+1}: F_int={state.f_int:.4f}, r={state.reward:.4f}, "
              f"V={state.valence:.2f}, A={state.arousal:.2f}, D={state.dominance:.2f}")

    print("\nBatched version:")
    batched = BatchedHomeostatL1(config, batch_size=8)
    u_batch = torch.zeros(8, 4)
    for t in range(5):
        rewards, n, d = batched.step(u_batch)
        print(f"Step {t+1}: mean_r={rewards.mean():.4f}")
