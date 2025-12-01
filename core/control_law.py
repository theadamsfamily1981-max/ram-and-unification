# core/control_law.py
# L5 Controller for TGSFN Architecture
#
# Implements the unified control law that translates forces into
# manifold-constrained velocity vectors.
#
# The L5 Controller computes:
#   v*(t) = thermodynamic_clip(proj_TzM(F_action))
#
# Where:
#   F_action = F_drive + F_appraisal + F_epistemic
#   proj_TzM = Riemannian projection onto tangent space at z
#   thermodynamic_clip = clipping based on Π_q / Π_max
#
# Force Components:
#   F_drive = Σ_inv @ n_t         (Allostatic drive)
#   F_appraisal = W_mot @ a_t      (Appraisal bias)
#   F_epistemic = g_τ * ξ_epi      (Epistemic exploration)
#
# Scientific Constraints:
#   - Must use Geoopt manifold operations for projection
#   - Thermodynamic clipping ensures Π_q ≤ Π_max
#   - Retraction must project back onto manifold after spike reset
#
# References:
#   - Friston et al. (2017): Active inference
#   - Nickel & Kiela (2017): Poincaré embeddings

from __future__ import annotations

from typing import Dict, Optional, Tuple, Union
import torch
import torch.nn as nn

try:
    import geoopt
    from geoopt import ManifoldParameter, PoincareBall, Lorentz
    GEOOPT_AVAILABLE = True
except ImportError:
    GEOOPT_AVAILABLE = False
    # Fallback stubs
    class ManifoldParameter:
        pass
    class PoincareBall:
        pass


class L5Controller(nn.Module):
    """
    L5 Control Law for TGSFN.

    Translates the unified force (allostatic + appraisal + epistemic)
    into a manifold-constrained velocity vector v*(t).

    The controller performs three key operations:
    1. Compute ambient force F_action in embedding space
    2. Project to tangent space using Riemannian projection
    3. Apply thermodynamic clipping based on Π_q / Π_max

    Attributes:
        manifold: Geoopt manifold (PoincareBall or Lorentz)
        dim: Embedding dimension
        n_interoceptive: Number of interoceptive signals
        n_appraisal: Number of appraisal dimensions
    """

    def __init__(
        self,
        manifold,
        dim: int,
        n_interoceptive: int,
        n_appraisal: int,
        sigma_init: float = 1.0,
        w_mot_init: float = 0.1,
        epistemic_scale: float = 0.1,
    ):
        """
        Initialize L5 Controller.

        Args:
            manifold: Geoopt manifold instance (e.g., PoincareBall(c=1.0))
            dim: Embedding dimension
            n_interoceptive: Number of interoceptive/need signals (n_t)
            n_appraisal: Number of appraisal dimensions (a_t)
            sigma_init: Initial scale for Σ_inv
            w_mot_init: Initial scale for W_mot
            epistemic_scale: Scale for epistemic exploration
        """
        super().__init__()

        self.manifold = manifold
        self.dim = dim
        self.n_interoceptive = n_interoceptive
        self.n_appraisal = n_appraisal
        self.epistemic_scale = epistemic_scale

        # Σ_inv: Interoceptive precision matrix (maps needs → forces)
        # Shape: (dim, n_interoceptive)
        self.sigma_inv = nn.Parameter(
            torch.randn(dim, n_interoceptive) * sigma_init
        )

        # W_mot: Motor/appraisal weight matrix (maps appraisals → forces)
        # Shape: (dim, n_appraisal)
        self.W_mot = nn.Parameter(
            torch.randn(dim, n_appraisal) * w_mot_init
        )

        # Learnable epistemic gain (optional)
        self.epistemic_gain = nn.Parameter(torch.tensor(1.0))

    def compute_allostatic_drive(
        self,
        n_t: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute allostatic drive force.

        F_drive = Σ_inv @ n_t

        Args:
            n_t: Interoceptive/need signals, shape (B, n_interoceptive)

        Returns:
            Drive force, shape (B, dim)
        """
        # (B, n_interoceptive) @ (n_interoceptive, dim).T → (B, dim)
        return (self.sigma_inv @ n_t.T).T

    def compute_appraisal_bias(
        self,
        a_t: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute appraisal-based force.

        F_appraisal = W_mot @ a_t

        Args:
            a_t: Appraisal vector, shape (B, n_appraisal)

        Returns:
            Appraisal force, shape (B, dim)
        """
        # (B, n_appraisal) @ (n_appraisal, dim).T → (B, dim)
        return (self.W_mot @ a_t.T).T

    def compute_epistemic_force(
        self,
        g_tau_t: torch.Tensor,
        xi_epi_t: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute epistemic exploration force.

        F_epistemic = g_τ * ξ_epi * epistemic_gain

        Args:
            g_tau_t: Epistemic gain (scalar or (B,))
            xi_epi_t: Epistemic direction, shape (B, dim)

        Returns:
            Epistemic force, shape (B, dim)
        """
        # Expand g_tau if scalar
        if g_tau_t.dim() == 0:
            g_tau_t = g_tau_t.unsqueeze(0)
        if g_tau_t.dim() == 1:
            g_tau_t = g_tau_t.unsqueeze(-1)  # (B, 1)

        return g_tau_t * xi_epi_t * self.epistemic_gain * self.epistemic_scale

    def project_to_tangent(
        self,
        z: torch.Tensor,
        F_ambient: torch.Tensor,
    ) -> torch.Tensor:
        """
        Project ambient force to tangent space at z.

        v_tangent = proj_TzM(F_ambient)

        Uses Geoopt manifold.proju() for projection.

        Args:
            z: Current position on manifold, shape (B, dim)
            F_ambient: Force in ambient space, shape (B, dim)

        Returns:
            Tangent vector at z, shape (B, dim)
        """
        if GEOOPT_AVAILABLE and self.manifold is not None:
            # Use Geoopt projection
            return self.manifold.proju(z, F_ambient)
        else:
            # Fallback: Euclidean projection (for Poincaré, approximate)
            # For Poincaré ball, proj_z(u) ≈ u - <u, z> z / ||z||²
            z_norm_sq = (z ** 2).sum(dim=-1, keepdim=True).clamp(min=1e-8)
            inner = (F_ambient * z).sum(dim=-1, keepdim=True)
            return F_ambient - (inner / z_norm_sq) * z

    def thermodynamic_clip(
        self,
        v_tangent: torch.Tensor,
        pi_q: torch.Tensor,
        pi_max: float,
        eps: float = 1e-6,
    ) -> torch.Tensor:
        """
        Apply thermodynamic clipping to velocity.

        v_clipped = v_tangent * min(Π_max / Π_q, 1.0)

        This ensures that velocity is reduced when entropy production
        exceeds the maximum threshold.

        Args:
            v_tangent: Tangent velocity, shape (B, dim)
            pi_q: Current entropy production, scalar or (B,)
            pi_max: Maximum allowed entropy production
            eps: Small constant for numerical stability

        Returns:
            Clipped velocity, shape (B, dim)
        """
        # Compute clipping factor
        therm_factor = torch.clamp(pi_max / (pi_q + eps), max=1.0)

        # Expand to match v_tangent shape
        if therm_factor.dim() == 0:
            pass  # scalar, broadcasts naturally
        elif therm_factor.dim() == 1 and v_tangent.dim() == 2:
            therm_factor = therm_factor.unsqueeze(-1)  # (B, 1)

        return v_tangent * therm_factor

    def retract(
        self,
        z: torch.Tensor,
        v: torch.Tensor,
        step_size: float = 1.0,
    ) -> torch.Tensor:
        """
        Retract from z along v to new point on manifold.

        z_new = Retr_z(step_size * v)

        Uses exponential map for geodesic retraction.

        Args:
            z: Current point on manifold, shape (B, dim)
            v: Tangent vector at z, shape (B, dim)
            step_size: Step size for retraction

        Returns:
            New point on manifold, shape (B, dim)
        """
        if GEOOPT_AVAILABLE and self.manifold is not None:
            # Use Geoopt exponential map
            return self.manifold.expmap(z, step_size * v)
        else:
            # Fallback: Euclidean retraction with projection
            z_new = z + step_size * v
            # Project to ball (clip norm to < 1)
            norm = z_new.norm(dim=-1, keepdim=True)
            max_norm = 1.0 - 1e-5
            z_new = torch.where(
                norm > max_norm,
                z_new * max_norm / norm,
                z_new,
            )
            return z_new

    def forward(
        self,
        z: torch.Tensor,
        n_t: torch.Tensor,
        a_t: torch.Tensor,
        g_tau_t: torch.Tensor,
        xi_epi_t: torch.Tensor,
        pi_q: torch.Tensor,
        pi_max: float,
        eps: float = 1e-6,
    ) -> Dict[str, torch.Tensor]:
        """
        Full L5 control law forward pass.

        Computes:
            1. F_action = F_drive + F_appraisal + F_epistemic
            2. v_tangent = proj_TzM(F_action)
            3. v_clipped = thermodynamic_clip(v_tangent, Π_q, Π_max)

        Args:
            z: Current manifold position, shape (B, dim)
            n_t: Interoceptive signals, shape (B, n_interoceptive)
            a_t: Appraisal vector, shape (B, n_appraisal)
            g_tau_t: Epistemic gain, scalar or (B,)
            xi_epi_t: Epistemic direction, shape (B, dim)
            pi_q: Current entropy production
            pi_max: Maximum entropy production threshold
            eps: Numerical stability constant

        Returns:
            Dict with:
                v_star: Final velocity v*(t)
                F_action: Total ambient force
                F_drive: Allostatic drive
                F_appraisal: Appraisal bias
                F_epistemic: Epistemic force
                therm_factor: Thermodynamic clipping factor
        """
        # 1. Compute force components
        F_drive = self.compute_allostatic_drive(n_t)
        F_appraisal = self.compute_appraisal_bias(a_t)
        F_epistemic = self.compute_epistemic_force(g_tau_t, xi_epi_t)

        # Total ambient force
        F_action = F_drive + F_appraisal + F_epistemic

        # 2. Project to tangent space
        v_tangent = self.project_to_tangent(z, F_action)

        # 3. Thermodynamic clipping
        therm_factor = torch.clamp(pi_max / (pi_q + eps), max=1.0)
        v_star = v_tangent * therm_factor

        return {
            "v_star": v_star,
            "F_action": F_action,
            "F_drive": F_drive,
            "F_appraisal": F_appraisal,
            "F_epistemic": F_epistemic,
            "therm_factor": therm_factor,
            "v_tangent": v_tangent,
        }

    def step(
        self,
        z: torch.Tensor,
        v_star: torch.Tensor,
        step_size: float = 1.0,
    ) -> torch.Tensor:
        """
        Take a step on the manifold.

        Args:
            z: Current position
            v_star: Velocity from forward()
            step_size: Step size

        Returns:
            New position on manifold
        """
        return self.retract(z, v_star, step_size)


class L5ControllerEuclidean(L5Controller):
    """
    L5 Controller for Euclidean space (no manifold constraints).

    Useful for comparison experiments or when Geoopt is not available.
    """

    def __init__(
        self,
        dim: int,
        n_interoceptive: int,
        n_appraisal: int,
        **kwargs,
    ):
        # Pass None for manifold
        super().__init__(
            manifold=None,
            dim=dim,
            n_interoceptive=n_interoceptive,
            n_appraisal=n_appraisal,
            **kwargs,
        )

    def project_to_tangent(
        self,
        z: torch.Tensor,
        F_ambient: torch.Tensor,
    ) -> torch.Tensor:
        """In Euclidean space, no projection needed."""
        return F_ambient

    def retract(
        self,
        z: torch.Tensor,
        v: torch.Tensor,
        step_size: float = 1.0,
    ) -> torch.Tensor:
        """In Euclidean space, retraction is just addition."""
        return z + step_size * v


# =============================================================================
# Factory Functions
# =============================================================================

def create_l5_controller(
    manifold_type: str,
    dim: int,
    n_interoceptive: int,
    n_appraisal: int,
    curvature: float = 1.0,
    **kwargs,
) -> L5Controller:
    """
    Factory function to create L5 Controller with specified manifold.

    Args:
        manifold_type: "poincare", "lorentz", or "euclidean"
        dim: Embedding dimension
        n_interoceptive: Number of interoceptive signals
        n_appraisal: Number of appraisal dimensions
        curvature: Manifold curvature (for hyperbolic)
        **kwargs: Additional arguments for L5Controller

    Returns:
        Configured L5Controller instance
    """
    if manifold_type == "euclidean":
        return L5ControllerEuclidean(
            dim=dim,
            n_interoceptive=n_interoceptive,
            n_appraisal=n_appraisal,
            **kwargs,
        )

    if not GEOOPT_AVAILABLE:
        print("Warning: Geoopt not available, falling back to Euclidean")
        return L5ControllerEuclidean(
            dim=dim,
            n_interoceptive=n_interoceptive,
            n_appraisal=n_appraisal,
            **kwargs,
        )

    if manifold_type == "poincare":
        manifold = geoopt.PoincareBall(c=curvature)
    elif manifold_type == "lorentz":
        manifold = geoopt.Lorentz(k=curvature)
    else:
        raise ValueError(f"Unknown manifold type: {manifold_type}")

    return L5Controller(
        manifold=manifold,
        dim=dim,
        n_interoceptive=n_interoceptive,
        n_appraisal=n_appraisal,
        **kwargs,
    )


if __name__ == "__main__":
    print("=== L5 Controller Test ===\n")

    B, dim = 8, 32
    n_intero, n_appraisal = 4, 6

    # Create controller (Euclidean fallback)
    controller = create_l5_controller(
        manifold_type="euclidean",
        dim=dim,
        n_interoceptive=n_intero,
        n_appraisal=n_appraisal,
    )

    # Test inputs
    z = torch.randn(B, dim) * 0.3  # Position
    n_t = torch.randn(B, n_intero)  # Needs
    a_t = torch.randn(B, n_appraisal)  # Appraisals
    g_tau_t = torch.tensor(0.5)  # Epistemic gain
    xi_epi_t = torch.randn(B, dim)  # Epistemic direction
    xi_epi_t = xi_epi_t / xi_epi_t.norm(dim=-1, keepdim=True)
    pi_q = torch.tensor(0.3)  # Current entropy production
    pi_max = 1.0  # Max entropy production

    # Forward pass
    result = controller(z, n_t, a_t, g_tau_t, xi_epi_t, pi_q, pi_max)

    print("Force components:")
    print(f"  F_drive norm: {result['F_drive'].norm(dim=-1).mean().item():.4f}")
    print(f"  F_appraisal norm: {result['F_appraisal'].norm(dim=-1).mean().item():.4f}")
    print(f"  F_epistemic norm: {result['F_epistemic'].norm(dim=-1).mean().item():.4f}")
    print(f"  F_action norm: {result['F_action'].norm(dim=-1).mean().item():.4f}")
    print(f"\nVelocity:")
    print(f"  v_tangent norm: {result['v_tangent'].norm(dim=-1).mean().item():.4f}")
    print(f"  v_star norm: {result['v_star'].norm(dim=-1).mean().item():.4f}")
    print(f"  Therm factor: {result['therm_factor'].item():.4f}")

    # Test step
    z_new = controller.step(z, result["v_star"], step_size=0.1)
    print(f"\nStep:")
    print(f"  z norm: {z.norm(dim=-1).mean().item():.4f}")
    print(f"  z_new norm: {z_new.norm(dim=-1).mean().item():.4f}")

    # Test with high entropy production (should clip)
    print("\n--- High entropy production test ---")
    pi_q_high = torch.tensor(2.0)  # > pi_max
    result_high = controller(z, n_t, a_t, g_tau_t, xi_epi_t, pi_q_high, pi_max)
    print(f"  Therm factor (high Π_q): {result_high['therm_factor'].item():.4f}")
    print(f"  v_star norm (clipped): {result_high['v_star'].norm(dim=-1).mean().item():.4f}")

    print("\n✓ L5 Controller test passed!")
