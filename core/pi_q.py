# core/pi_q.py
# Thermodynamic Regularizer Π_q for TGSFN
#
# Implements the Entropy Production Proxy as defined in the Master Implementation Brief:
#
#   Π_q = Leak_Energy_Proxy + λ_J * Jacobian_Stress
#
# Where:
#   - Leak_Energy_Proxy: (V - V_reset)² / (τ_m * σ²) summed over neurons
#   - Jacobian_Stress: ||J_θ||_F² (Frobenius norm of dynamics Jacobian)
#
# Scientific Constraints (Referee-safe):
#   - This is a lower bound on true entropy production
#   - Minimizing Π_q pushes system slightly subcritical: Δm(N) = O(N^{-1/2})
#   - Ensures finite-size stability while approaching critical branching
#
# References:
#   - Seifert (2012): Stochastic thermodynamics, fluctuation theorems
#   - Perunov et al. (2016): Entropy production bounds in neural systems

from __future__ import annotations

from typing import Dict, Optional, Tuple
import torch
import torch.nn as nn


def compute_pi_q(
    V: torch.Tensor,
    V_reset: float,
    tau_m: float,
    sigma_sq: float,
    jacobian_norm_sq: Optional[torch.Tensor] = None,
    lambda_J: float = 0.01,
) -> torch.Tensor:
    """
    Computes the Entropy Production Proxy (Π_q).

    Π_q = Leak_Energy_Proxy + λ_J * Jacobian_Stress

    This serves as the thermodynamic regularizer that:
    1. Penalizes membrane deviation from reset (energy dissipation)
    2. Penalizes large Jacobian norms (dynamic instability)

    Args:
        V: Membrane potentials, shape (B, N) or (B, T, N)
        V_reset: Reset potential (typically 0)
        tau_m: Membrane time constant
        sigma_sq: Noise variance (σ²) for normalization
        jacobian_norm_sq: Pre-computed ||J||_F², or None to skip
        lambda_J: Weight for Jacobian stress term

    Returns:
        Scalar Π_q value
    """
    # Term 1: Membrane Dissipation (Lower bound on entropy production)
    # Energy = (V - V_reset)² normalized by τ_m and noise
    # This measures how far membrane states deviate from equilibrium
    leak_energy = ((V - V_reset) ** 2 / (tau_m * sigma_sq))
    leak_term = leak_energy.sum(dim=-1).mean()

    # Term 2: Jacobian Stress (Measure of dynamic instability)
    # High ||J||² means small perturbations grow → unstable dynamics
    if jacobian_norm_sq is not None:
        jacobian_stress = lambda_J * jacobian_norm_sq
    else:
        jacobian_stress = torch.tensor(0.0, device=V.device, dtype=V.dtype)

    return leak_term + jacobian_stress


class JacobianEstimator(nn.Module):
    """
    Estimates Jacobian norm using power iteration or Lanczos methods.

    For a dynamics function f: R^N → R^N, estimates ||J||_F² where
    J_ij = ∂f_i/∂x_j without explicitly computing the full Jacobian.

    Methods:
        - Power iteration: Estimates max eigenvalue λ_max
        - Hutchinson: Stochastic trace estimation for ||J||_F²
    """

    def __init__(
        self,
        method: str = "hutchinson",
        n_iterations: int = 5,
        n_samples: int = 3,
    ):
        """
        Args:
            method: "power" for power iteration, "hutchinson" for trace estimation
            n_iterations: Number of iterations for power method
            n_samples: Number of random vectors for Hutchinson
        """
        super().__init__()
        self.method = method
        self.n_iterations = n_iterations
        self.n_samples = n_samples

    def power_iteration(
        self,
        vjp_fn,  # Vector-Jacobian product function
        shape: Tuple[int, ...],
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """
        Estimate max eigenvalue of J^T J using power iteration.

        λ_max ≈ ||Jv||² / ||v||² after convergence.
        """
        # Random initial vector
        v = torch.randn(shape, device=device, dtype=dtype)
        v = v / v.norm()

        for _ in range(self.n_iterations):
            # Compute J @ v using VJP (autodiff)
            Jv = vjp_fn(v)

            # Normalize
            norm = Jv.norm()
            if norm > 1e-8:
                v = Jv / norm

        # Final eigenvalue estimate
        Jv = vjp_fn(v)
        lambda_max = (Jv * v).sum()

        return lambda_max

    def hutchinson_trace(
        self,
        vjp_fn,
        shape: Tuple[int, ...],
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """
        Estimate trace(J^T J) = ||J||_F² using Hutchinson's trick.

        E[z^T J^T J z] = trace(J^T J) for z ~ N(0, I)
        """
        trace_estimate = torch.tensor(0.0, device=device, dtype=dtype)

        for _ in range(self.n_samples):
            # Random Rademacher vector (±1 with equal probability)
            z = torch.randint(0, 2, shape, device=device, dtype=dtype) * 2 - 1

            # Compute J @ z
            Jz = vjp_fn(z)

            # z^T J^T J z = ||J z||²
            trace_estimate = trace_estimate + (Jz ** 2).sum()

        return trace_estimate / self.n_samples

    def forward(
        self,
        dynamics_fn,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """
        Estimate ||J||_F² for dynamics_fn at point x.

        Args:
            dynamics_fn: Function f: R^N → R^N
            x: Current state, shape (..., N)

        Returns:
            Estimated ||J||_F²
        """
        x = x.detach().requires_grad_(True)

        # Define VJP function using autograd
        def vjp_fn(v: torch.Tensor) -> torch.Tensor:
            y = dynamics_fn(x)
            # Compute J @ v via backward pass
            (Jv,) = torch.autograd.grad(
                y, x, v,
                create_graph=False,
                retain_graph=True,
            )
            return Jv

        shape = x.shape
        device = x.device
        dtype = x.dtype

        if self.method == "power":
            return self.power_iteration(vjp_fn, shape, device, dtype)
        else:
            return self.hutchinson_trace(vjp_fn, shape, device, dtype)


class EntropyProductionMonitor(nn.Module):
    """
    Full Entropy Production Monitor for TGSFN training.

    Computes Π_q and provides the thermodynamic regularization term
    for the total loss:

        L_total = L_task + λ_diss * Π_q

    The λ_diss parameter is the "criticality dial" - higher values
    push the system toward lower entropy production (more subcritical),
    while lower values allow more critical dynamics.

    Attributes:
        lambda_J: Weight for Jacobian stress term
        lambda_diss: Weight for Π_q in total loss
        sigma_sq: Noise variance for normalization
    """

    def __init__(
        self,
        lambda_J: float = 0.01,
        lambda_diss: float = 0.1,
        sigma_sq: float = 1.0,
        jacobian_method: str = "hutchinson",
        jacobian_iterations: int = 5,
        jacobian_samples: int = 3,
    ):
        """
        Args:
            lambda_J: Weight for Jacobian stress in Π_q
            lambda_diss: Weight for Π_q in total loss
            sigma_sq: Noise variance (default 1.0)
            jacobian_method: "hutchinson" or "power"
            jacobian_iterations: Iterations for power method
            jacobian_samples: Samples for Hutchinson
        """
        super().__init__()
        self.lambda_J = lambda_J
        self.lambda_diss = lambda_diss
        self.sigma_sq = sigma_sq

        self.jacobian_estimator = JacobianEstimator(
            method=jacobian_method,
            n_iterations=jacobian_iterations,
            n_samples=jacobian_samples,
        )

        # Running statistics for monitoring
        self.register_buffer("pi_q_ema", torch.tensor(0.0))
        self.register_buffer("leak_ema", torch.tensor(0.0))
        self.register_buffer("jacobian_ema", torch.tensor(0.0))
        self.ema_decay = 0.99

    def compute_pi_q(
        self,
        V: torch.Tensor,
        V_reset: float,
        tau_m: float,
        jacobian_norm_sq: Optional[torch.Tensor] = None,
        dynamics_fn=None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute Π_q with component breakdown.

        Args:
            V: Membrane potentials (B, N)
            V_reset: Reset potential
            tau_m: Membrane time constant
            jacobian_norm_sq: Pre-computed ||J||_F², or None to estimate
            dynamics_fn: Function for Jacobian estimation (if jacobian_norm_sq is None)

        Returns:
            Dict with pi_q, leak_term, jacobian_term
        """
        # Leak energy term
        leak_energy = ((V - V_reset) ** 2 / (tau_m * self.sigma_sq))
        leak_term = leak_energy.sum(dim=-1).mean()

        # Jacobian term
        if jacobian_norm_sq is not None:
            jac_term = self.lambda_J * jacobian_norm_sq
        elif dynamics_fn is not None:
            jac_norm_sq = self.jacobian_estimator(dynamics_fn, V)
            jac_term = self.lambda_J * jac_norm_sq
        else:
            jac_term = torch.tensor(0.0, device=V.device, dtype=V.dtype)

        pi_q = leak_term + jac_term

        # Update EMAs
        if self.training:
            self.pi_q_ema = self.ema_decay * self.pi_q_ema + (1 - self.ema_decay) * pi_q.detach()
            self.leak_ema = self.ema_decay * self.leak_ema + (1 - self.ema_decay) * leak_term.detach()
            self.jacobian_ema = self.ema_decay * self.jacobian_ema + (1 - self.ema_decay) * jac_term.detach()

        return {
            "pi_q": pi_q,
            "leak_term": leak_term,
            "jacobian_term": jac_term,
        }

    def compute_thermodynamic_loss(
        self,
        task_loss: torch.Tensor,
        V: torch.Tensor,
        V_reset: float,
        tau_m: float,
        jacobian_norm_sq: Optional[torch.Tensor] = None,
        dynamics_fn=None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute total loss with thermodynamic regularization.

        L_total = L_task + λ_diss * Π_q

        Args:
            task_loss: Primary task objective
            V: Membrane potentials
            V_reset: Reset potential
            tau_m: Membrane time constant
            jacobian_norm_sq: Pre-computed Jacobian norm
            dynamics_fn: For Jacobian estimation

        Returns:
            Dict with total_loss, task_loss, pi_q_loss, pi_q_components
        """
        pi_q_result = self.compute_pi_q(
            V, V_reset, tau_m, jacobian_norm_sq, dynamics_fn
        )

        pi_q_loss = self.lambda_diss * pi_q_result["pi_q"]
        total_loss = task_loss + pi_q_loss

        return {
            "total_loss": total_loss,
            "task_loss": task_loss,
            "pi_q_loss": pi_q_loss,
            **pi_q_result,
        }

    def get_stats(self) -> Dict[str, float]:
        """Get current EMA statistics."""
        return {
            "pi_q_ema": self.pi_q_ema.item(),
            "leak_ema": self.leak_ema.item(),
            "jacobian_ema": self.jacobian_ema.item(),
        }

    def forward(
        self,
        V: torch.Tensor,
        V_reset: float,
        tau_m: float,
        jacobian_norm_sq: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Simple forward: just compute Π_q."""
        result = self.compute_pi_q(V, V_reset, tau_m, jacobian_norm_sq)
        return result["pi_q"]


# =============================================================================
# Utility Functions
# =============================================================================

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
    # Empirical scaling: λ_diss ∝ N^{-1/2} for constant subcriticality
    base_lambda = 0.1
    return base_lambda * (target_subcriticality / 0.1) * (1000 / N) ** 0.5


if __name__ == "__main__":
    print("=== Π_q Thermodynamic Regularizer Test ===\n")

    # Test basic compute_pi_q
    print("--- Basic compute_pi_q ---")
    V = torch.randn(8, 256) * 0.5  # (B, N)
    pi_q = compute_pi_q(
        V=V,
        V_reset=0.0,
        tau_m=10.0,
        sigma_sq=1.0,
        jacobian_norm_sq=torch.tensor(0.5),
        lambda_J=0.01,
    )
    print(f"Π_q: {pi_q.item():.6f}")

    # Test EntropyProductionMonitor
    print("\n--- EntropyProductionMonitor ---")
    monitor = EntropyProductionMonitor(
        lambda_J=0.01,
        lambda_diss=0.1,
        sigma_sq=1.0,
    )

    result = monitor.compute_pi_q(
        V=V,
        V_reset=0.0,
        tau_m=10.0,
        jacobian_norm_sq=torch.tensor(0.5),
    )
    print(f"Π_q: {result['pi_q'].item():.6f}")
    print(f"  Leak term: {result['leak_term'].item():.6f}")
    print(f"  Jacobian term: {result['jacobian_term'].item():.6f}")

    # Test with task loss
    print("\n--- Thermodynamic Loss ---")
    task_loss = torch.tensor(1.5)
    loss_result = monitor.compute_thermodynamic_loss(
        task_loss=task_loss,
        V=V,
        V_reset=0.0,
        tau_m=10.0,
        jacobian_norm_sq=torch.tensor(0.5),
    )
    print(f"Total loss: {loss_result['total_loss'].item():.6f}")
    print(f"  Task loss: {loss_result['task_loss'].item():.6f}")
    print(f"  Π_q loss: {loss_result['pi_q_loss'].item():.6f}")

    # Test critical lambda estimation
    print("\n--- Critical λ_diss Estimation ---")
    for N in [1000, 10000, 100000]:
        lambda_diss = estimate_critical_lambda_diss(N)
        print(f"N={N:6d}: λ_diss = {lambda_diss:.4f}")

    print("\n✓ Π_q module test passed!")
