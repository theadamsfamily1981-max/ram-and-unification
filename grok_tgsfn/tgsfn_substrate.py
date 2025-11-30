# grok_tgsfn/tgsfn_substrate.py
# TGSFN: Thermodynamic-Geometric Spiking Field Network Substrate
#
# Implements the neural substrate from the Grok memo:
#
# Manifold:
#   M = ℝⁿ × Sᵖ × ℍᵐ  (Euclidean × Spherical × Hyperbolic)
#
# Neuron i has:
#   z_i ∈ M           - position on manifold
#   x_i(t) ∈ T_{z_i}M - tangent state (membrane potential analog)
#
# Dynamics (GCSD - Geometric Continual Spiking Dynamics):
#   dx_i/dt = -∇_z V(z_i) + Σ_j w_ij log_{z_i}(spike_j direction) + D Δ_R x_i
#
# Where:
#   V(z)    - potential field on manifold
#   w_ij    - synaptic weights
#   log_{z} - Riemannian logarithm (direction to spike origin)
#   Δ_R     - Riemannian Laplacian (diffusion on manifold)
#
# Spike condition:
#   if ||x_i|| > θ_i:
#       emit spike
#       z_i ← Exp_{z_i}(x_i / ||x_i||)  (move on manifold)
#       x_i ← x_i - θ_i · unit_vector   (soft reset)

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Dict
import math

import torch
import torch.nn as nn

from .config import TGSFNConfig


@dataclass
class TGSFNState:
    """
    State of a TGSFN layer at time t.

    z: (B, N, D) positions on manifold M
    x: (B, N, D) tangent states (membrane potentials)

    Where:
        B = batch size
        N = number of neurons
        D = D_euclid + D_sphere + D_hyper (total manifold dimension)
    """
    z: torch.Tensor   # Manifold positions
    x: torch.Tensor   # Tangent states


class TGSFNLayer(nn.Module):
    """
    TGSFN Layer: Spiking neural network on product manifold.

    This is a structural implementation with placeholders for full
    Riemannian operations. Key features:

    1. Neurons live on M = ℝⁿ × Sᵖ × ℍᵐ
    2. Membrane dynamics integrate in tangent space
    3. Spikes cause movement on manifold
    4. Recurrent connections use parallel transport

    Phase 1 simplification:
    - Use Euclidean approximations for non-hyperbolic parts
    - Implement hyperbolic exp/log via Poincaré ball
    - Leave Riemannian Laplacian as local averaging
    """

    def __init__(self, config: TGSFNConfig, num_inputs: int):
        super().__init__()
        self.config = config
        self.device = torch.device(config.device)

        self.num_neurons = config.num_neurons
        self.num_inputs = num_inputs

        # Manifold dimensions
        self.D_euclid = config.euclid_dim
        self.D_sphere = config.sphere_dim
        self.D_hyper = config.hyper_dim
        self.D_total = self.D_euclid + self.D_sphere + self.D_hyper

        # Synaptic weights: recurrent (N x N) and input (N x input_dim)
        self.W_rec = nn.Parameter(
            torch.randn(self.num_neurons, self.num_neurons, device=self.device) * 0.01
        )
        self.W_in = nn.Parameter(
            torch.randn(self.num_neurons, num_inputs, device=self.device) * 0.1
        )

        # Per-neuron spike thresholds
        self.theta = nn.Parameter(
            torch.full((self.num_neurons,), config.spike_threshold, device=self.device)
        )

        # Diffusion coefficient D
        self.D_coeff = nn.Parameter(
            torch.tensor(config.diffusion_coeff, device=self.device)
        )

        # Membrane time constant τ_m
        self.tau_m = config.tau_membrane

    def init_state(self, batch_size: int = 1) -> TGSFNState:
        """
        Initialize manifold positions z and tangent states x.

        Positions are initialized:
        - Euclidean part: near origin
        - Spherical part: uniformly on sphere
        - Hyperbolic part: near origin of Poincaré ball
        """
        device = self.device
        N = self.num_neurons

        # Initialize z on manifold
        z = torch.zeros(batch_size, N, self.D_total, device=device)

        # Initialize tangent states x to zero
        x = torch.zeros_like(z)

        return TGSFNState(z=z, x=x)

    def forward(
        self,
        state: TGSFNState,
        external_input: torch.Tensor,  # (B, num_inputs)
        dt: float = 1.0,
    ) -> Tuple[TGSFNState, torch.Tensor]:
        """
        One step of TGSFN dynamics.

        Implements:
            dx_i/dt = -∇V + W_rec @ spikes + D·Δx + W_in @ input
            spike if ||x_i|| > θ_i
            reset: x_i -= θ_i · x_i/||x_i||, z_i moves

        Args:
            state: TGSFNState with z, x of shape (B, N, D)
            external_input: (B, num_inputs) external drive
            dt: Integration timestep

        Returns:
            (new_state, spikes)
            spikes: (B, N) spike indicators (0 or 1)
        """
        z = state.z
        x = state.x
        B, N, D = z.shape
        device = self.device

        # Input projection: (B, num_inputs) -> (B, N, D)
        # Broadcast input to all tangent dimensions
        input_contrib = torch.einsum('bi,ni->bn', external_input, self.W_in)  # (B, N)
        input_contrib = input_contrib.unsqueeze(-1).expand(B, N, D)

        # Potential gradient (placeholder: simple quadratic attractor to origin)
        grad_V = self._compute_potential_gradient(z)

        # Recurrent contribution from previous spikes
        # For now, use simple weight matrix on tangent norms
        x_norm = x.norm(dim=-1)  # (B, N)
        recurrent_activation = torch.matmul(x_norm, self.W_rec.T)  # (B, N)
        recurrent_contrib = recurrent_activation.unsqueeze(-1).expand(B, N, D)

        # Diffusion (Laplacian approximation: local averaging)
        laplacian = self._approximate_laplacian(x)

        # Total derivative
        dx_dt = (-grad_V
                + 0.1 * input_contrib
                + 0.05 * recurrent_contrib
                + self.D_coeff * laplacian)

        # Leaky integration with time constant
        alpha = dt / self.tau_m
        x_new = (1 - alpha) * x + alpha * dx_dt

        # Spike detection
        x_new_norm = x_new.norm(dim=-1)  # (B, N)
        theta_expanded = self.theta.view(1, N).expand(B, N)
        spikes = (x_new_norm > theta_expanded).float()  # (B, N)

        # Reset and manifold update for spiking neurons
        z_new, x_new = self._spike_reset(z, x_new, spikes, x_new_norm, theta_expanded)

        return TGSFNState(z=z_new, x=x_new), spikes

    def _compute_potential_gradient(self, z: torch.Tensor) -> torch.Tensor:
        """
        Compute gradient of potential V(z).

        Phase 1: Simple quadratic potential pulling toward origin.
        Future: More complex landscape based on task structure.
        """
        # V(z) = 0.5 * ||z||^2 → ∇V = z
        return 0.1 * z

    def _approximate_laplacian(self, x: torch.Tensor) -> torch.Tensor:
        """
        Approximate Riemannian Laplacian.

        Phase 1: Local averaging with neighboring neurons.
        Full implementation would use geodesic neighborhoods.
        """
        B, N, D = x.shape

        # Simple averaging: each neuron averages with neighbors
        # This is a placeholder for proper Riemannian diffusion
        avg_x = x.mean(dim=1, keepdim=True).expand(B, N, D)
        laplacian = avg_x - x

        return laplacian

    def _spike_reset(
        self,
        z: torch.Tensor,
        x: torch.Tensor,
        spikes: torch.Tensor,
        x_norm: torch.Tensor,
        theta: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Handle spike reset and manifold update.

        For spiking neurons:
        1. Move z on manifold: z_new = Exp_z(x / ||x||)
        2. Reset x: x_new = x - θ * (x / ||x||)
        """
        B, N, D = z.shape
        device = self.device

        # Compute unit direction
        unit_x = x / (x_norm.unsqueeze(-1).clamp_min(1e-8))

        # Create spike mask (B, N, 1) for broadcasting
        spike_mask = spikes.unsqueeze(-1)

        # Move z on manifold for spiking neurons
        # Phase 1: Linear step in tangent direction (Euclidean approximation)
        # Full implementation: Riemannian exponential map
        step_size = 0.1  # Movement magnitude on manifold
        z_delta = step_size * unit_x
        z_new = z + spike_mask * z_delta

        # Clamp hyperbolic part to stay in Poincaré ball
        if self.D_hyper > 0:
            hyp_start = self.D_euclid + self.D_sphere
            hyp_end = self.D_total
            z_hyp = z_new[:, :, hyp_start:hyp_end]
            z_hyp_norm = z_hyp.norm(dim=-1, keepdim=True)
            max_norm = 0.99  # Stay inside unit ball
            z_hyp = torch.where(
                z_hyp_norm > max_norm,
                z_hyp * max_norm / z_hyp_norm,
                z_hyp,
            )
            z_new = torch.cat([
                z_new[:, :, :hyp_start],
                z_hyp,
            ], dim=-1) if hyp_start > 0 else z_hyp

        # Reset x for spiking neurons
        reset_amount = theta.unsqueeze(-1) * unit_x
        x_new = x - spike_mask * reset_amount

        return z_new, x_new

    def get_state_summary(self, state: TGSFNState) -> Dict[str, float]:
        """Get summary statistics of current state."""
        z_norm = state.z.norm(dim=-1).mean().item()
        x_norm = state.x.norm(dim=-1).mean().item()
        return {
            "mean_z_norm": z_norm,
            "mean_x_norm": x_norm,
            "max_x_norm": state.x.norm(dim=-1).max().item(),
        }


if __name__ == "__main__":
    print("=== TGSFNLayer Test ===")

    config = TGSFNConfig(
        euclid_dim=0,
        sphere_dim=0,
        hyper_dim=32,
        num_neurons=64,
        diffusion_coeff=0.01,
        spike_threshold=1.0,
        tau_membrane=10.0,
        device="cpu",
    )

    layer = TGSFNLayer(config, num_inputs=8)

    # Initialize state
    B = 4
    state = layer.init_state(B)
    print(f"Initial state: z={state.z.shape}, x={state.x.shape}")

    # Run some steps
    print("\n--- Running dynamics ---")
    total_spikes = 0
    for t in range(50):
        external_input = torch.randn(B, 8) * (0.5 if t < 20 else 0.1)
        state, spikes = layer(state, external_input, dt=1.0)
        n_spikes = spikes.sum().item()
        total_spikes += n_spikes

        if (t + 1) % 10 == 0:
            summary = layer.get_state_summary(state)
            print(f"t={t+1}: spikes={n_spikes:.0f}, "
                  f"mean_x={summary['mean_x_norm']:.4f}, "
                  f"max_x={summary['max_x_norm']:.4f}")

    print(f"\nTotal spikes over 50 steps: {total_spikes:.0f}")
    print("TGSFNLayer test passed!")
