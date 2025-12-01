#!/usr/bin/env python3
"""
ara_v0_minimal.py
Minimal runnable Ara core: Product manifold geometry + LIF dynamics + thermodynamic clip

This is the ONLY running piece of the Ara architecture.
Everything else (training, real Jacobians, DAU, hardware) is future work.

Key components:
    - ProductManifold: Euclidean × PoincaréBall for latent state z
    - TGSFNNeuron: LIF membrane dynamics with geometric state update
    - L5Controller: Thermodynamically-constrained tangent velocity
    - compute_piq: Global entropy production proxy

IMPORTANT NOTES:
    1. In v0, z does not yet affect firing; this is a testbed for stable
       manifold dynamics and Π_q proxies. Minimal coupling via geom_drive
       is included but can be disabled.
    2. J_norm_sq is mocked; real Jacobian estimation is future work.
    3. Tested with Geoopt 0.6; if upgrading, verify PoincareBall signature
       and ProductManifold.proju/retr semantics.

Usage:
    python ara_v0_minimal.py
"""

from __future__ import annotations

import torch
import torch.nn as nn

# Geoopt for Riemannian manifold operations
try:
    import geoopt
    from geoopt import Euclidean, PoincareBall, ProductManifold
    HAS_GEOOPT = True
except ImportError:
    HAS_GEOOPT = False
    print("Warning: geoopt not installed. Using fallback (limited functionality).")


# =============================================================================
# Entropy Production Proxy
# =============================================================================

def compute_piq(
    v: torch.Tensor,
    J_norm_sq: float = 1.0,
    lambda_J: float = 1.2,
    n_neurons: int | None = None,
) -> torch.Tensor:
    """
    Compute global Π_q (entropy production proxy).

    Π_q = leak_term + λ_J * J_norm_sq / N

    Args:
        v: Membrane potentials, shape (N,) or (N, d)
        J_norm_sq: Jacobian norm squared (MOCKED in v0)
        lambda_J: Jacobian penalty weight
        n_neurons: Number of neurons (defaults to v.numel())

    Returns:
        Scalar Π_q value

    Note:
        J_norm_sq is currently mocked as a constant. Real Jacobian
        estimation via power iteration is future work.
    """
    if n_neurons is None:
        n_neurons = v.numel()

    # Leak term: deviation from resting potential
    V_rest = -70.0
    tau_m = 20.0
    leak = ((v - V_rest) ** 2 / tau_m).mean()

    # Jacobian term (mocked)
    jacob = lambda_J * J_norm_sq / n_neurons

    return leak + jacob


# =============================================================================
# Product Manifold (Euclidean × Hyperbolic)
# =============================================================================

class AraProductManifold:
    """
    Product manifold: Euclidean(d_E) × PoincaréBall(d_H).

    The latent state z lives on this product space:
        z = [z_eucl | z_hyp] ∈ ℝ^d_E × B^d_H

    Where B^d_H is the d_H-dimensional Poincaré ball.

    Note:
        Tested with Geoopt 0.6. The PoincareBall constructor signature
        may vary across versions. Current usage assumes:
            PoincareBall(c=curvature) with dim inferred from tensors.
    """

    def __init__(
        self,
        eucl_dim: int = 16,
        hyp_dim: int = 16,
        c: float = 1.0,
    ):
        """
        Args:
            eucl_dim: Dimension of Euclidean component
            hyp_dim: Dimension of hyperbolic (Poincaré ball) component
            c: Curvature of Poincaré ball (c > 0)
        """
        self.eucl_dim = eucl_dim
        self.hyp_dim = hyp_dim
        self.total_dim = eucl_dim + hyp_dim
        self.c = c

        if HAS_GEOOPT:
            self.eucl = Euclidean()
            self.hyp = PoincareBall(c=c)
            # ProductManifold for joint operations
            self.manifold = ProductManifold((self.eucl, eucl_dim), (self.hyp, hyp_dim))
        else:
            self.eucl = None
            self.hyp = None
            self.manifold = None

    def origin(
        self,
        n_neurons: int,
        device: torch.device | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        """
        Return origin point on the product manifold.

        Args:
            n_neurons: Batch size (number of neurons)
            device: Target device
            dtype: Data type

        Returns:
            Tensor of shape (n_neurons, eucl_dim + hyp_dim)
        """
        if HAS_GEOOPT:
            # Geoopt origin methods
            o_e = torch.zeros(n_neurons, self.eucl_dim, device=device, dtype=dtype)
            o_h = torch.zeros(n_neurons, self.hyp_dim, device=device, dtype=dtype)
            return torch.cat([o_e, o_h], dim=-1)
        else:
            return torch.zeros(n_neurons, self.total_dim, device=device, dtype=dtype)

    def proju(self, z: torch.Tensor, F: torch.Tensor) -> torch.Tensor:
        """
        Project ambient vector F to tangent space at z.

        Args:
            z: Point on manifold, shape (N, total_dim)
            F: Ambient force vector, shape (N, total_dim)

        Returns:
            Tangent vector at z, shape (N, total_dim)
        """
        if HAS_GEOOPT and self.manifold is not None:
            return self.manifold.proju(z, F)
        else:
            # Fallback: just return F (Euclidean approximation)
            return F

    def retr(self, z: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """
        Retraction: move from z along tangent vector v.

        Args:
            z: Point on manifold, shape (N, total_dim)
            v: Tangent vector at z, shape (N, total_dim)

        Returns:
            New point on manifold, shape (N, total_dim)
        """
        if HAS_GEOOPT and self.manifold is not None:
            return self.manifold.retr(z, v)
        else:
            # Fallback: Euclidean retraction
            z_new = z + v
            # Project hyperbolic part to ball interior
            z_hyp = z_new[:, self.eucl_dim:]
            norm = z_hyp.norm(dim=-1, keepdim=True)
            max_norm = 1.0 - 1e-5
            z_hyp = torch.where(norm > max_norm, z_hyp * max_norm / norm, z_hyp)
            z_new = torch.cat([z_new[:, :self.eucl_dim], z_hyp], dim=-1)
            return z_new


# =============================================================================
# TGSFN Neuron Layer
# =============================================================================

class TGSFNNeuron(nn.Module):
    """
    TGSFN neuron with geometric latent state and LIF membrane dynamics.

    State variables (buffers, not trained parameters):
        z: Latent state on product manifold, shape (N, d_E + d_H)
           Updated each step via Riemannian retraction. Not trained.
        v: Membrane potential, shape (N,)
           Standard LIF dynamics with spike-and-reset.

    The geometric state z evolves according to L5 controller forces,
    while spiking follows classic LIF dynamics with optional geometric
    coupling via geom_drive.
    """

    def __init__(
        self,
        n_neurons: int = 1024,
        eucl_dim: int = 16,
        hyp_dim: int = 16,
        c: float = 1.0,
        tau_m: float = 20.0,
        V_thresh: float = -55.0,
        V_reset: float = -70.0,
        geom_coupling: float = 0.0,  # Set > 0 for geometry → spiking coupling
    ):
        """
        Args:
            n_neurons: Number of neurons
            eucl_dim: Euclidean manifold dimension
            hyp_dim: Hyperbolic manifold dimension
            c: Poincaré ball curvature
            tau_m: Membrane time constant (ms)
            V_thresh: Spike threshold (mV)
            V_reset: Reset potential (mV)
            geom_coupling: Strength of geometry → firing coupling (0 = decoupled)
        """
        super().__init__()

        self.n_neurons = n_neurons
        self.tau_m = tau_m
        self.V_thresh = V_thresh
        self.V_reset = V_reset
        self.geom_coupling = geom_coupling

        # Product manifold
        self.manifold = AraProductManifold(eucl_dim, hyp_dim, c)

        # State buffers (not learned, just dynamical state)
        # z: latent state, updated each step via retraction
        self.register_buffer("z", self.manifold.origin(n_neurons))
        # v: membrane state, pure LIF simulation
        self.register_buffer("v", torch.full((n_neurons,), V_reset))

    def reset_state(self, device: torch.device | None = None) -> None:
        """Reset state to initial conditions."""
        if device is None:
            device = self.z.device
        self.z = self.manifold.origin(self.n_neurons, device=device)
        self.v = torch.full((self.n_neurons,), self.V_reset, device=device)

    def forward(
        self,
        F_geom_tangent: torch.Tensor,
        I_ext: torch.Tensor,
        dt: float = 1.0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Single timestep update.

        Args:
            F_geom_tangent: Tangent force for geometric update, shape (N, d_total)
            I_ext: External current, shape (N,)
            dt: Timestep (ms)

        Returns:
            (v_new, spikes): Updated membrane potential and spike indicators
        """
        # === 1. Geometric update (no gradients, pure state evolution) ===
        with torch.no_grad():
            # Project force to tangent space
            dz = self.manifold.proju(self.z, F_geom_tangent)
            # Retract along tangent
            self.z = self.manifold.retr(self.z, dz * dt)

        # === 2. Optional geometry → spiking coupling ===
        if self.geom_coupling > 0:
            geom_drive = self.z.norm(dim=-1)  # Shape (N,)
            I_total = I_ext + self.geom_coupling * geom_drive
        else:
            I_total = I_ext

        # === 3. LIF membrane dynamics ===
        # dv/dt = (-(v - V_reset) + I) / τ_m
        dv = (-(self.v - self.V_reset) + I_total) / self.tau_m
        v_new = self.v + dv * dt

        # Spike detection and reset
        spikes = (v_new >= self.V_thresh).float()
        v_new = torch.where(spikes > 0, torch.full_like(v_new, self.V_reset), v_new)

        # Update state buffer
        with torch.no_grad():
            self.v = v_new

        return v_new, spikes


# =============================================================================
# L5 Controller
# =============================================================================

class L5Controller(nn.Module):
    """
    L5 Controller: Thermodynamically-constrained action generation.

    Maps drive and appraisal signals to geometric tangent forces,
    then clips by global entropy production rate Π_q.

    Control law:
        v_tangent = proj_{T_z M}(F_raw) * min(1, 1/Π_q)

    Where F_raw = W_drive @ drive + W_appraisal @ appraisal + noise
    """

    def __init__(
        self,
        n_neurons: int = 1024,
        drive_dim: int = 8,
        appraisal_dim: int = 8,
        geom_dim: int = 32,
        noise_scale: float = 0.01,
    ):
        """
        Args:
            n_neurons: Number of neurons
            drive_dim: Dimension of drive input
            appraisal_dim: Dimension of appraisal input
            geom_dim: Total geometric dimension (eucl + hyp)
            noise_scale: Scale of epigenetic noise
        """
        super().__init__()

        self.n_neurons = n_neurons
        self.geom_dim = geom_dim
        self.noise_scale = noise_scale

        # Linear maps from drive/appraisal to geometry
        self.W_drive = nn.Linear(drive_dim, geom_dim, bias=False)
        self.W_appraisal = nn.Linear(appraisal_dim, geom_dim, bias=False)

        # Initialize with small weights
        nn.init.normal_(self.W_drive.weight, std=0.01)
        nn.init.normal_(self.W_appraisal.weight, std=0.01)

    def forward(
        self,
        z: torch.Tensor,
        drive: torch.Tensor,
        appraisal: torch.Tensor,
        epigenetic_noise: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Generate thermodynamically-constrained geometric force.

        Args:
            z: Current geometric state, shape (N, geom_dim)
            drive: Drive signals, shape (N, drive_dim)
            appraisal: Appraisal signals, shape (N, appraisal_dim)
            epigenetic_noise: Optional noise, shape (N, geom_dim)

        Returns:
            (F_clipped, Pi_q): Clipped tangent force and entropy production
        """
        # Raw force from drive and appraisal
        F_drive = self.W_drive(drive)
        F_appraisal = self.W_appraisal(appraisal)
        F_raw = F_drive + F_appraisal

        # Add epigenetic noise
        if epigenetic_noise is not None:
            F_raw = F_raw + self.noise_scale * epigenetic_noise

        # Compute Π_q proxy (using z norm as cheap stand-in for Jacobian)
        # NOTE: This is a placeholder. Real Jacobian estimation via power
        # iteration is future work.
        J_norm_sq = z.norm(dim=-1).pow(2).mean()
        Pi_q = compute_piq(z.norm(dim=-1), J_norm_sq.item(), n_neurons=z.shape[0])

        # Thermodynamic clip: slow down when Π_q is high
        therm_clip = torch.clamp(1.0 / (Pi_q + 1e-6), max=1.0)

        # Apply clip to force
        F_clipped = F_raw * therm_clip

        return F_clipped, Pi_q


# =============================================================================
# Demo
# =============================================================================

def demo():
    """
    Minimal demo: 100 steps of geometric + spiking dynamics.

    This demonstrates:
        1. Manifold state z evolving via L5 controller
        2. LIF spiking with optional geometric coupling
        3. Global Π_q monitoring
    """
    print("=" * 60)
    print("Ara v0 Minimal Demo")
    print("=" * 60)

    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Geoopt available: {HAS_GEOOPT}")

    N = 1024  # neurons
    eucl_dim = 16
    hyp_dim = 16
    geom_dim = eucl_dim + hyp_dim

    # Create components
    neuron = TGSFNNeuron(
        n_neurons=N,
        eucl_dim=eucl_dim,
        hyp_dim=hyp_dim,
        geom_coupling=0.1,  # Enable mild geometry → spiking coupling
    ).to(device)

    controller = L5Controller(
        n_neurons=N,
        geom_dim=geom_dim,
    ).to(device)

    print(f"\nNetwork: {N} neurons, geometry dim = {geom_dim}")
    print(f"Geometry coupling: {neuron.geom_coupling}")
    print("-" * 60)

    # Run dynamics
    n_steps = 100
    dt = 1.0

    spike_history = []
    piq_history = []
    z_norm_history = []

    for step in range(n_steps):
        # Random inputs (would be from sensors/internal drives in real system)
        drive = torch.randn(N, 8, device=device) * 0.5
        appraisal = torch.randn(N, 8, device=device) * 0.5
        epigenetic_noise = torch.randn(N, geom_dim, device=device)
        I_ext = torch.randn(N, device=device) * 5.0  # External current

        # L5 controller generates geometric force
        F_geom, Pi_q = controller(neuron.z, drive, appraisal, epigenetic_noise)

        # Neuron update (geometry + LIF)
        v_new, spikes = neuron(F_geom, I_ext, dt=dt)

        # Track stats
        n_spikes = spikes.sum().item()
        z_norm = neuron.z.norm(dim=-1).mean().item()
        spike_history.append(n_spikes)
        piq_history.append(Pi_q.item())
        z_norm_history.append(z_norm)

        # Print progress
        if (step + 1) % 20 == 0:
            recent_rate = sum(spike_history[-20:]) / (20 * N) * 1000  # Hz
            print(f"Step {step+1:3d}: spikes={n_spikes:4.0f}, "
                  f"rate={recent_rate:.1f}Hz, "
                  f"Π_q={Pi_q.item():.4f}, "
                  f"||z||={z_norm:.4f}")

    # Summary
    print("-" * 60)
    print(f"Total spikes: {sum(spike_history):.0f}")
    print(f"Mean rate: {sum(spike_history) / (n_steps * N) * 1000:.1f} Hz")
    print(f"Mean Π_q: {sum(piq_history) / n_steps:.4f}")
    print(f"Final ||z||: {z_norm_history[-1]:.4f}")
    print("=" * 60)

    return neuron, controller, spike_history, piq_history


if __name__ == "__main__":
    demo()
