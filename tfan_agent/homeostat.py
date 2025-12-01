# tfan_agent/homeostat.py
# Layer 1: Homeostatic Core / Vulnerable Body
#
# Implements the HRRL (Homeostatic Reinforcement Learning) foundation:
#   - Maintains internal needs vector n(t) ∈ R_+^K
#   - Computes free energy F_int(n) = 0.5 * n^T Σ^{-1} n
#   - Provides drive d = Σ^{-1} n
#   - Computes PAD-like affect scalars (Valence, Arousal, Dominance)
#   - Generates intrinsic reward r_t = -(F_int(t+1) - F_int(t))
#
# This is the "body" that makes the agent care about its internal state.

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict

import torch
import torch.nn as nn


@dataclass
class HomeostatConfig:
    """
    Configuration for the L1 Homeostatic Core.

    K = number of internal needs (e.g., energy, integrity, affiliation, certainty, etc.)
    """
    num_needs: int = 8
    gamma: float = 0.1          # relaxation rate in dn/dt = -Gamma * d + ...
    sigma_init: float = 1.0     # initial diagonal of precision matrix Σ
    device: str = "cpu"


@dataclass
class HomeostatState:
    """
    Container for logging the internal homeostatic state at a given time step.
    """
    needs: torch.Tensor              # (K,)
    free_energy: float               # scalar F_int
    drive: torch.Tensor              # (K,)
    valence: float                   # scalar V (pleasure/displeasure)
    arousal: float                   # scalar A (activation level)
    dominance: float                 # scalar D (control/agency)


class HomeostatL1(nn.Module):
    """
    Layer 1: Homeostatic Core / Vulnerable Body.

    This is the foundation of affective computing in the T-FAN architecture.
    It implements a Free Energy Principle-inspired internal state that:

    1. Maintains needs n(t) that represent unmet internal requirements
    2. Computes free energy F_int as a scalar measure of overall "distress"
    3. Generates intrinsic reward from free energy reduction (pleasure)
    4. Provides PAD-style affect for downstream gating/appraisal

    The key insight from HRRL: the agent's reward comes from maintaining
    homeostasis, not from external task rewards. This creates genuinely
    embodied motivation.

    Equations:
        F_int(n) = 0.5 * n^T Σ^{-1} n     (free energy)
        d = Σ^{-1} n                       (drive vector)
        r_t = -(F_int(t+1) - F_int(t))    (homeostatic reward)
        V = -d·n                           (valence: negative when driven)
        A = ||d||                          (arousal: magnitude of drive)
        D = -trace(Σ^{-1})                (dominance: placeholder)
    """

    def __init__(self, config: HomeostatConfig):
        super().__init__()
        self.config = config
        self.device = torch.device(config.device)

        # Needs vector n(t) - the "unmet requirements" of the vulnerable body
        self.register_buffer("n", torch.zeros(config.num_needs, device=self.device))

        # Precision matrix Σ (for now, diagonal)
        # Σ^{-1} is just 1 / diag for the diagonal case.
        # Higher precision = more sensitive to that need being unmet
        sigma_diag = torch.full((config.num_needs,), config.sigma_init, device=self.device)
        self.register_buffer("sigma_diag", sigma_diag)

        # Track last free energy for ΔF reward computation
        self.last_free_energy: float = 0.0

    @torch.no_grad()
    def reset(self, init_needs: torch.Tensor | None = None) -> HomeostatState:
        """
        Reset needs to zero or provided vector, reset free energy baseline.

        Call this at the start of each episode to initialize the body state.
        """
        if init_needs is not None:
            assert init_needs.shape == self.n.shape
            self.n = init_needs.to(self.device).clone()
        else:
            self.n.zero_()

        self.last_free_energy = self._compute_free_energy(self.n).item()
        V, A, D, d = self._compute_pad_and_drive(self.n)

        return HomeostatState(
            needs=self.n.clone(),
            free_energy=self.last_free_energy,
            drive=d.detach().clone(),
            valence=float(V),
            arousal=float(A),
            dominance=float(D),
        )

    def forward(self, need_delta: torch.Tensor) -> Dict[str, float | torch.Tensor | HomeostatState]:
        """
        Apply one time-step update to the homeostatic state.

        This is the core "body dynamics" update. The environment and actions
        produce need_delta, which shifts the internal state. The reward comes
        from whether this shift reduced or increased free energy.

        Args:
            need_delta: tensor of shape (K,)
                Change in needs induced by environment + action at this step
                (positive = more unmet need / deprivation,
                 negative = need satisfaction / relief).

        Returns:
            dict containing:
                - "reward": scalar r_t = -(F_next - F_prev)
                            Positive when free energy decreases (relief)
                            Negative when free energy increases (distress)
                - "state": HomeostatState snapshot for logging/visualization
        """
        need_delta = need_delta.to(self.device)
        assert need_delta.shape == self.n.shape, "need_delta must match needs shape"

        # Update needs (simple Euler integration of dn/dt)
        # Here we treat need_delta as already the effective step for this dt.
        # Clamp to non-negative (can't have negative unmet needs)
        self.n = (self.n + need_delta).clamp(min=0.0)

        # Compute new free energy
        F_int = self._compute_free_energy(self.n)

        # Homeostatic reward: pleasure from reducing free energy
        # r_t = -(F_int(t+1) - F_int(t))
        reward = -(F_int.item() - self.last_free_energy)
        self.last_free_energy = F_int.item()

        # Compute PAD affect and drive for downstream use
        V, A, D, d = self._compute_pad_and_drive(self.n)

        state = HomeostatState(
            needs=self.n.clone(),
            free_energy=float(F_int.item()),
            drive=d.detach().clone(),
            valence=float(V),
            arousal=float(A),
            dominance=float(D),
        )

        return {
            "reward": reward,
            "state": state,
        }

    def _compute_free_energy(self, n: torch.Tensor) -> torch.Tensor:
        """
        Compute internal free energy F_int(n) = 0.5 * n^T Σ^{-1} n

        This is a Mahalanobis distance from the origin (ideal homeostatic state)
        weighted by the precision matrix. Higher precision on a need means
        deviations in that need contribute more to overall distress.

        For diagonal Σ, Σ^{-1} is simply (1 / sigma_diag).
        """
        inv_sigma = 1.0 / (self.sigma_diag + 1e-8)
        return 0.5 * torch.sum(n * inv_sigma * n)

    def _compute_pad_and_drive(self, n: torch.Tensor):
        """
        Compute drive vector and PAD-style affect scalars.

        Drive d = Σ^{-1} n
            The gradient of free energy w.r.t. needs. Points toward
            the direction of steepest descent (what would most reduce F_int).

        PAD (Pleasure-Arousal-Dominance) model:
            V (Valence) ~ -d·n
                Negative when driven (unpleasant), positive when satisfied
            A (Arousal) ~ ||d||
                High when there's strong drive to act
            D (Dominance) ~ controllability
                Placeholder: will be replaced by controllability Gramian in Phase 2

        Returns:
            (valence, arousal, dominance, drive_vector)
        """
        inv_sigma = 1.0 / (self.sigma_diag + 1e-8)
        d = inv_sigma * n

        # Valence: negative dot product between drive and state
        # When d·n is large (high drive, high needs), valence is negative (unpleasant)
        # When both are near zero, valence is near zero (neutral)
        valence = -torch.dot(d, n).clamp(min=-100.0, max=100.0)

        # Arousal: magnitude of drive vector
        # High arousal = strong motivation to act
        arousal = d.norm(p=2)

        # Dominance: placeholder heuristic (negative sum of precision = low control)
        # In Phase 2, this will be replaced by controllability analysis
        dominance = -inv_sigma.sum()

        return valence, arousal, dominance, d

    def get_state_dict_for_telemetry(self) -> Dict[str, float]:
        """
        Get current state as a flat dict for telemetry/HUD.
        """
        V, A, D, d = self._compute_pad_and_drive(self.n)
        F_int = self._compute_free_energy(self.n)

        result = {
            "free_energy": float(F_int.item()),
            "valence": float(V.item()) if torch.is_tensor(V) else float(V),
            "arousal": float(A.item()) if torch.is_tensor(A) else float(A),
            "dominance": float(D.item()) if torch.is_tensor(D) else float(D),
            "drive_magnitude": float(d.norm().item()),
        }

        # Add individual needs
        for i in range(self.config.num_needs):
            result[f"need_{i}"] = float(self.n[i].item())
            result[f"drive_{i}"] = float(d[i].item())

        return result


if __name__ == "__main__":
    # Quick sanity check
    print("=== HomeostatL1 Sanity Check ===")

    config = HomeostatConfig(num_needs=4, device="cpu")
    homeostat = HomeostatL1(config)

    # Reset
    state = homeostat.reset()
    print(f"Initial state: F_int={state.free_energy:.4f}, V={state.valence:.4f}, A={state.arousal:.4f}")

    # Apply some need deltas
    for step in range(5):
        # Simulate environment pushing needs up
        delta = torch.tensor([0.1, 0.05, 0.0, -0.02])
        result = homeostat(delta)
        s = result["state"]
        r = result["reward"]
        print(f"Step {step+1}: F_int={s.free_energy:.4f}, reward={r:.4f}, V={s.valence:.4f}, A={s.arousal:.4f}")

    # Simulate relief
    print("\n--- Relief phase ---")
    for step in range(5):
        delta = torch.tensor([-0.15, -0.1, -0.05, -0.01])
        result = homeostat(delta)
        s = result["state"]
        r = result["reward"]
        print(f"Step {step+6}: F_int={s.free_energy:.4f}, reward={r:.4f}, V={s.valence:.4f}")

    print("\nHomeostatL1 sanity check passed!")
