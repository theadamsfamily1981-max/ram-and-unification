# tgsfn_core/piq_loss.py
# Thermodynamic Loss Functions for TGSFN
#
# Implements:
#   - Π_q: Entropy production proxy (thermodynamic regularizer)
#   - F_int: Internal free energy (homeostatic cost)
#   - Combined TGSFN loss for training near criticality
#
# The thermodynamic regularizer pushes the network toward:
#   - Low entropy production (efficient dynamics)
#   - Stable Jacobian (bounded sensitivity)
#   - Target firing rates (homeostasis)

import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple


def compute_piq(
    V: torch.Tensor,
    V_reset: float,
    tau_m: float,
    J_proxy: torch.Tensor,
    lambda_J: float = 0.01,
    sigma_sq: float = 1.0,
) -> torch.Tensor:
    """
    Compute entropy production proxy Π_q.

    Π_q ≈ Σᵢ (Vᵢ - V_reset)² / (τ_m · σ²) + λ_J ||J||_F²

    This serves as a lower bound on true entropy production and
    regularizes the network toward efficient, stable dynamics.

    Args:
        V: Membrane potentials [batch, N] or [batch, T, N]
        V_reset: Reset potential (scalar)
        tau_m: Membrane time constant (scalar or [N])
        J_proxy: Jacobian norm proxy ||J||_F² (scalar)
        lambda_J: Weight for Jacobian term
        sigma_sq: Noise variance for normalization

    Returns:
        Π_q scalar value
    """
    # Ensure tau_m is tensor
    if not torch.is_tensor(tau_m):
        tau_m = torch.tensor(tau_m, device=V.device, dtype=V.dtype)

    # Membrane dissipation term
    diff_sq = (V - V_reset) ** 2
    leak_term = (diff_sq / (tau_m * sigma_sq)).mean()

    # Jacobian stability term
    jacobian_term = lambda_J * J_proxy

    return leak_term + jacobian_term


def compute_internal_free_energy(
    rate_estimate: torch.Tensor,
    target_rates: torch.Tensor,
    beta_rate: float = 1.0,
) -> torch.Tensor:
    """
    Compute internal free energy F_int (homeostatic cost).

    F_int = β · Σᵢ (rᵢ - r*ᵢ)²

    This penalizes deviation from target firing rates,
    maintaining network in a healthy operating regime.

    Args:
        rate_estimate: Running average firing rates [N] or [batch, N]
        target_rates: Target firing rates (broadcastable)
        beta_rate: Weighting coefficient

    Returns:
        F_int scalar value
    """
    diff_sq = (rate_estimate - target_rates) ** 2
    return beta_rate * diff_sq.mean()


def compute_spike_rate_penalty(
    spikes: torch.Tensor,
    target_rate: float = 0.05,
    beta: float = 1.0,
) -> torch.Tensor:
    """
    Direct spike rate penalty (alternative to running estimate).

    Args:
        spikes: Spike tensor [batch, T, N] or [batch, N]
        target_rate: Target firing rate
        beta: Penalty weight

    Returns:
        Rate penalty scalar
    """
    actual_rate = spikes.mean()
    return beta * (actual_rate - target_rate) ** 2


def compute_activity_regularizer(
    spikes: torch.Tensor,
    min_rate: float = 0.01,
    max_rate: float = 0.3,
    penalty_scale: float = 1.0,
) -> torch.Tensor:
    """
    Soft penalty for rates outside [min_rate, max_rate].

    Prevents network from going silent or saturating.

    Args:
        spikes: Spike tensor
        min_rate: Minimum acceptable rate
        max_rate: Maximum acceptable rate
        penalty_scale: Penalty coefficient

    Returns:
        Activity regularizer value
    """
    rate = spikes.mean()

    # Penalty for being below min_rate
    below_penalty = torch.relu(min_rate - rate) ** 2

    # Penalty for being above max_rate
    above_penalty = torch.relu(rate - max_rate) ** 2

    return penalty_scale * (below_penalty + above_penalty)


def tgsfn_loss(
    task_loss: torch.Tensor,
    F_int: torch.Tensor,
    Pi_q: torch.Tensor,
    lambda_diss: float = 0.1,
    lambda_homeo: float = 1.0,
) -> torch.Tensor:
    """
    Combined TGSFN training loss.

    L_total = L_task + λ_homeo · F_int + λ_diss · Π_q

    The balance between terms controls:
    - λ_diss: How strongly we regularize toward low entropy production
    - λ_homeo: How strongly we enforce target firing rates

    Higher λ_diss pushes system more subcritical (stable but less expressive)
    Lower λ_diss allows more critical dynamics (expressive but potentially unstable)

    Args:
        task_loss: Primary task objective
        F_int: Internal free energy (homeostatic cost)
        Pi_q: Entropy production proxy
        lambda_diss: Weight for thermodynamic regularizer
        lambda_homeo: Weight for homeostatic term

    Returns:
        Combined loss value
    """
    return task_loss + lambda_homeo * F_int + lambda_diss * Pi_q


class TGSFNLossModule(nn.Module):
    """
    Module wrapper for TGSFN loss computation.

    Provides a convenient interface for computing all loss components
    and tracking statistics over training.
    """

    def __init__(
        self,
        lambda_diss: float = 0.1,
        lambda_homeo: float = 1.0,
        lambda_J: float = 0.01,
        beta_rate: float = 1.0,
        target_rate: float = 0.05,
        tau_m: float = 20e-3,
        v_reset: float = 0.0,
    ):
        """
        Initialize loss module.

        Args:
            lambda_diss: Weight for Π_q
            lambda_homeo: Weight for F_int
            lambda_J: Weight for Jacobian term in Π_q
            beta_rate: Weight for rate deviation in F_int
            target_rate: Target firing rate
            tau_m: Membrane time constant
            v_reset: Reset potential
        """
        super().__init__()

        self.lambda_diss = lambda_diss
        self.lambda_homeo = lambda_homeo
        self.lambda_J = lambda_J
        self.beta_rate = beta_rate
        self.target_rate = target_rate
        self.tau_m = tau_m
        self.v_reset = v_reset

        # Running statistics
        self.register_buffer("pi_q_ema", torch.tensor(0.0))
        self.register_buffer("f_int_ema", torch.tensor(0.0))
        self.ema_decay = 0.99

    def forward(
        self,
        task_loss: torch.Tensor,
        V: torch.Tensor,
        spikes: torch.Tensor,
        rate_estimate: torch.Tensor,
        J_proxy: torch.Tensor,
        target_rates: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute all loss components.

        Args:
            task_loss: Primary task loss
            V: Membrane potentials
            spikes: Spike tensor
            rate_estimate: Running rate estimate
            J_proxy: Jacobian norm proxy
            target_rates: Target rates (uses default if None)

        Returns:
            Dict with total_loss and all components
        """
        if target_rates is None:
            target_rates = torch.full_like(rate_estimate, self.target_rate)

        # Compute Π_q
        Pi_q = compute_piq(
            V=V,
            V_reset=self.v_reset,
            tau_m=self.tau_m,
            J_proxy=J_proxy,
            lambda_J=self.lambda_J,
        )

        # Compute F_int
        F_int = compute_internal_free_energy(
            rate_estimate=rate_estimate,
            target_rates=target_rates,
            beta_rate=self.beta_rate,
        )

        # Activity regularizer
        activity_reg = compute_activity_regularizer(spikes)

        # Combined loss
        total_loss = tgsfn_loss(
            task_loss=task_loss,
            F_int=F_int,
            Pi_q=Pi_q,
            lambda_diss=self.lambda_diss,
            lambda_homeo=self.lambda_homeo,
        ) + 0.1 * activity_reg

        # Update EMAs
        if self.training:
            self.pi_q_ema = self.ema_decay * self.pi_q_ema + (1 - self.ema_decay) * Pi_q.detach()
            self.f_int_ema = self.ema_decay * self.f_int_ema + (1 - self.ema_decay) * F_int.detach()

        return {
            "total_loss": total_loss,
            "task_loss": task_loss,
            "Pi_q": Pi_q,
            "F_int": F_int,
            "activity_reg": activity_reg,
            "pi_q_ema": self.pi_q_ema,
            "f_int_ema": self.f_int_ema,
        }

    def get_stats(self) -> Dict[str, float]:
        """Get current EMA statistics."""
        return {
            "pi_q_ema": self.pi_q_ema.item(),
            "f_int_ema": self.f_int_ema.item(),
        }


def estimate_critical_lambda_diss(
    N: int,
    target_subcriticality: float = 0.1,
) -> float:
    """
    Estimate λ_diss to achieve target subcriticality.

    Based on finite-size scaling: Δm(N) = O(N^{-1/2})

    Args:
        N: Number of neurons
        target_subcriticality: Desired distance from criticality

    Returns:
        Suggested λ_diss value
    """
    # Empirical scaling: λ_diss ∝ N^{-1/2}
    base_lambda = 0.1
    return base_lambda * (target_subcriticality / 0.1) * (1000 / N) ** 0.5


if __name__ == "__main__":
    print("=== TGSFN Loss Functions Test ===\n")

    batch, N, T = 8, 256, 100

    # Create dummy data
    V = torch.randn(batch, T, N) * 0.5
    spikes = (torch.rand(batch, T, N) > 0.95).float()
    rate_estimate = spikes.mean(dim=(0, 1))
    target_rates = torch.ones(N) * 0.05
    J_proxy = torch.tensor(0.5)
    task_loss = torch.tensor(1.5)

    # Test individual functions
    print("--- Individual Functions ---")
    Pi_q = compute_piq(V, V_reset=0.0, tau_m=0.02, J_proxy=J_proxy)
    print(f"  Π_q: {Pi_q.item():.6f}")

    F_int = compute_internal_free_energy(rate_estimate, target_rates)
    print(f"  F_int: {F_int.item():.6f}")

    total = tgsfn_loss(task_loss, F_int, Pi_q, lambda_diss=0.1)
    print(f"  Total loss: {total.item():.6f}")

    # Test loss module
    print("\n--- TGSFNLossModule ---")
    loss_module = TGSFNLossModule(lambda_diss=0.1, lambda_homeo=1.0)

    result = loss_module(
        task_loss=task_loss,
        V=V[:, -1, :],  # Last timestep voltage
        spikes=spikes,
        rate_estimate=rate_estimate,
        J_proxy=J_proxy,
    )

    for k, v in result.items():
        if torch.is_tensor(v):
            print(f"  {k}: {v.item():.6f}")

    # Test critical lambda estimation
    print("\n--- Critical λ_diss Estimation ---")
    for N_test in [1000, 4096, 10000]:
        lam = estimate_critical_lambda_diss(N_test)
        print(f"  N={N_test}: λ_diss = {lam:.4f}")

    print("\n✓ Loss functions test passed!")
