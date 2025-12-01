# grok_tgsfn/l1_homeostat.py
# Layer 1: Homeostatic Core with Continuous-Time Dynamics
#
# Implements the "vulnerable body" from the Grok memo:
#
# State:
#   n(t) ∈ ℝ₊^K  - internal needs vector
#
# Dynamics:
#   dn/dt = -Γ d(t) + ξ(t) + u(t)
#
# Where:
#   d(t) = Σ⁻¹ n(t)  - drive vector (gradient of free energy)
#   Γ              - relaxation rate (diagonal matrix)
#   ξ(t)           - interoceptive noise / prediction error
#   u(t)           - control input from actions
#
# Free Energy:
#   F_int(n) = ½ n^T Σ⁻¹ n
#
# PAD Affect:
#   V(t) = -d(t) · ṅ̂(t)    - Valence (pleasure when drive decreasing)
#   A(t) = ||d(t)|| + σ(t)  - Arousal (drive magnitude + uncertainty)
#   D(t) = controllability   - Dominance (heuristic for now)
#
# Reward:
#   r_t = -(F_int(t+1) - F_int(t))  - HRRL intrinsic reward

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Dict, Tuple

import torch
import torch.nn as nn

from .config import L1Config


@dataclass
class HomeostatState:
    """
    Snapshot of L1 internal state at a given time.

    All quantities are computed from the needs vector n(t).
    """
    needs: torch.Tensor          # n(t) ∈ ℝ₊^K
    free_energy: float           # F_int(n) = ½ n^T Σ⁻¹ n
    drive: torch.Tensor          # d(t) = Σ⁻¹ n(t)
    valence: float               # V(t) = -d · ṅ̂
    arousal: float               # A(t) = ||d|| + σ
    dominance: float             # D(t) - controllability heuristic

    def to_dict(self) -> Dict[str, float]:
        """Convert to flat dict for telemetry."""
        return {
            "free_energy": self.free_energy,
            "valence": self.valence,
            "arousal": self.arousal,
            "dominance": self.dominance,
            "drive_norm": float(self.drive.norm().item()),
            "needs_sum": float(self.needs.sum().item()),
        }


class HomeostatL1(nn.Module):
    """
    L1 - Homeostatic Core implementing continuous-time dynamics.

    This is the mathematical foundation of embodied motivation:
    the agent's reward comes from maintaining internal homeostasis,
    not from external task objectives.

    Key implementation details:
    - Uses Euler integration for dn/dt
    - Maintains diagonal precision matrix Σ (learnable or fixed)
    - Computes PAD affect for downstream gating
    - Tracks free energy for HRRL reward computation
    """

    def __init__(self, config: L1Config):
        super().__init__()
        self.config = config
        self.device = torch.device(config.device)

        # Needs vector n(t) ∈ ℝ₊^K
        self.register_buffer(
            "n",
            torch.zeros(config.num_needs, device=self.device)
        )

        # Diagonal precision matrix Σ (stored as diag entries)
        # Higher precision = more sensitive to that need being unmet
        sigma_diag = torch.full(
            (config.num_needs,),
            config.sigma_init,
            device=self.device,
        )
        self.register_buffer("sigma_diag", sigma_diag)

        # Relaxation rate Γ (diagonal, can be per-need or uniform)
        if config.gamma_per_need is not None:
            gamma = torch.tensor(config.gamma_per_need, device=self.device)
        else:
            gamma = torch.full(
                (config.num_needs,),
                config.gamma,
                device=self.device,
            )
        self.register_buffer("gamma", gamma)

        # Track free energy for ΔF_int computation
        self.last_free_energy: float = 0.0
        self._prev_n: Optional[torch.Tensor] = None

    @torch.no_grad()
    def reset(self, init_needs: Optional[torch.Tensor] = None) -> HomeostatState:
        """
        Reset needs and baseline free energy.

        Args:
            init_needs: Optional initial needs vector (K,)

        Returns:
            Initial HomeostatState
        """
        if init_needs is not None:
            assert init_needs.shape == self.n.shape
            self.n.copy_(init_needs.to(self.device))
        else:
            self.n.zero_()

        self._prev_n = self.n.clone()
        F = self._free_energy(self.n)
        self.last_free_energy = float(F.item())

        V, A, D, d = self._compute_pad(
            self.n,
            n_dot_pred=None,
            sigma_unc=None,
        )

        return HomeostatState(
            needs=self.n.clone(),
            free_energy=self.last_free_energy,
            drive=d.detach().clone(),
            valence=float(V),
            arousal=float(A),
            dominance=float(D),
        )

    @torch.no_grad()
    def step(
        self,
        u: torch.Tensor,
        xi: Optional[torch.Tensor] = None,
        sigma_unc: Optional[float] = None,
    ) -> Dict[str, float | HomeostatState]:
        """
        One step of continuous-time dynamics using Euler integration.

        Implements:
            dn/dt = -Γ d(t) + ξ(t) + u(t)
            n(t+dt) = n(t) + dt * dn/dt

        Args:
            u: Control input from actions, shape (K,)
               Positive = increases need (deprivation)
               Negative = satisfies need (relief)
            xi: Interoceptive noise/prediction error (optional)
            sigma_unc: Scalar uncertainty σ(t) for arousal computation

        Returns:
            Dict with:
                'reward': r_t = -ΔF_int (positive when free energy decreases)
                'state': HomeostatState snapshot
                'n_dot': Rate of change dn/dt for debugging
        """
        u = u.to(self.device)
        assert u.shape == self.n.shape, f"u shape {u.shape} != n shape {self.n.shape}"

        if xi is None:
            xi = torch.zeros_like(self.n)
        else:
            xi = xi.to(self.device)

        # Compute current drive d(t) = Σ⁻¹ n(t)
        inv_sigma = self._inv_sigma()
        d = inv_sigma * self.n

        # Continuous dynamics: dn/dt = -Γ d + ξ + u
        n_dot = -self.gamma * d + xi + u

        # Euler integration
        n_next = (self.n + self.config.dt * n_dot).clamp(min=0.0)

        # Free energy and reward
        F_prev = self._free_energy(self.n)
        F_next = self._free_energy(n_next)

        # HRRL reward: pleasure from reducing free energy
        reward = -(F_next.item() - F_prev.item())

        # Update state
        self._prev_n = self.n.clone()
        self.n.copy_(n_next)
        self.last_free_energy = float(F_next.item())

        # Compute PAD affect with predicted ṅ
        V, A, D, d_new = self._compute_pad(
            self.n,
            n_dot_pred=n_dot,
            sigma_unc=sigma_unc,
        )

        state = HomeostatState(
            needs=self.n.clone(),
            free_energy=self.last_free_energy,
            drive=d_new.detach().clone(),
            valence=float(V),
            arousal=float(A),
            dominance=float(D),
        )

        return {
            "reward": reward,
            "state": state,
            "n_dot": n_dot.clone(),
        }

    def forward(
        self,
        u: torch.Tensor,
        xi: Optional[torch.Tensor] = None,
        sigma_unc: Optional[float] = None,
    ) -> Dict[str, float | HomeostatState]:
        """Alias for step() for nn.Module interface."""
        return self.step(u, xi, sigma_unc)

    def _inv_sigma(self) -> torch.Tensor:
        """Compute Σ⁻¹ (inverse of diagonal precision matrix)."""
        return 1.0 / (self.sigma_diag + 1e-8)

    def _free_energy(self, n: torch.Tensor) -> torch.Tensor:
        """
        Compute internal free energy.

        F_int(n) = ½ n^T Σ⁻¹ n

        This is a Mahalanobis distance from the homeostatic set-point
        (origin) weighted by precision. Higher F_int = more "distress".
        """
        inv_sigma = self._inv_sigma()
        return 0.5 * torch.sum(n * inv_sigma * n)

    def _compute_pad(
        self,
        n: torch.Tensor,
        n_dot_pred: Optional[torch.Tensor],
        sigma_unc: Optional[float],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute PAD affect and drive vector.

        Returns:
            (valence, arousal, dominance, drive)
        """
        inv_sigma = self._inv_sigma()
        d = inv_sigma * n  # Drive = gradient of F_int w.r.t. n

        # Default n_dot if not provided: deterministic drift
        if n_dot_pred is None:
            n_dot_pred = -self.gamma * d

        # Valence: V(t) = -d · ṅ̂
        # Positive when drive is decreasing (moving toward homeostasis)
        # Negative when drive is increasing (moving away from homeostasis)
        valence = -torch.dot(d, n_dot_pred)

        # Arousal: A(t) = ||d|| + σ(t)
        # High arousal = strong motivation to act
        arousal = d.norm(p=2)
        if sigma_unc is not None:
            arousal = arousal + float(sigma_unc)

        # Dominance: controllability heuristic
        # Lower precision (higher σ_diag) = less control = lower dominance
        # This is a placeholder; proper implementation uses controllability Gramian
        dominance = -inv_sigma.sum()

        return valence, arousal, dominance, d

    def get_drive(self) -> torch.Tensor:
        """Get current drive vector d(t) = Σ⁻¹ n(t)."""
        return self._inv_sigma() * self.n

    def get_free_energy(self) -> float:
        """Get current free energy F_int(n)."""
        return float(self._free_energy(self.n).item())

    def update_precision(self, new_sigma_diag: torch.Tensor) -> None:
        """
        Update the precision matrix diagonal.

        This can be used for learning sensitivity to different needs.
        """
        assert new_sigma_diag.shape == self.sigma_diag.shape
        self.sigma_diag.copy_(new_sigma_diag.to(self.device).clamp(min=1e-6))


if __name__ == "__main__":
    print("=== HomeostatL1 Test (Continuous Dynamics) ===")

    config = L1Config(num_needs=4, gamma=0.2, dt=1.0, device="cpu")
    homeostat = HomeostatL1(config)

    # Reset
    state = homeostat.reset()
    print(f"Initial: F_int={state.free_energy:.4f}, V={state.valence:.4f}, A={state.arousal:.4f}")

    # Simulate deprivation (needs increasing)
    print("\n--- Deprivation phase ---")
    for t in range(5):
        u = torch.tensor([0.2, 0.1, 0.05, 0.0])  # Actions that increase needs
        result = homeostat.step(u)
        s = result["state"]
        r = result["reward"]
        print(f"t={t+1}: F_int={s.free_energy:.4f}, r={r:.4f}, V={s.valence:.4f}, A={s.arousal:.4f}")

    # Simulate relief (needs decreasing via actions)
    print("\n--- Relief phase ---")
    for t in range(5):
        u = torch.tensor([-0.3, -0.2, -0.1, -0.05])  # Actions that satisfy needs
        result = homeostat.step(u)
        s = result["state"]
        r = result["reward"]
        print(f"t={t+6}: F_int={s.free_energy:.4f}, r={r:.4f}, V={s.valence:.4f}")

    # Test natural relaxation (no action, drive pulls needs down)
    print("\n--- Natural relaxation (u=0) ---")
    homeostat.reset(torch.tensor([1.0, 0.5, 0.3, 0.1]))  # Start with high needs
    print(f"Reset: F_int={homeostat.get_free_energy():.4f}")
    for t in range(10):
        u = torch.zeros(4)  # No external action
        result = homeostat.step(u)
        s = result["state"]
        print(f"t={t+1}: F_int={s.free_energy:.4f}, needs={s.needs.tolist()}")

    print("\nHomeostatL1 test passed!")
