# grok_tgsfn/thermodynamics.py
# Thermodynamic Monitor for TGSFN
#
# Computes entropy production Π_q and thermodynamic loss terms:
#
# Entropy Production (from memo):
#   Π_q ≈ Σ_spikes (V_m - V_reset)² / τ_m
#         + λ_J trace(J^T J)
#         + I(spike_train; input)
#
# Where:
#   V_m, V_reset - membrane potential and reset value
#   τ_m          - membrane time constant
#   J            - Jacobian of dynamics
#   I(·;·)       - mutual information (spike predictability)
#
# Total Loss:
#   L_total = VFE + λ_diss Π_q + λ_geom K_sectional
#
# Where:
#   VFE          - Variational Free Energy (prediction error)
#   K_sectional  - Sectional curvature penalty (stability)

from __future__ import annotations

from typing import Dict, Optional
import torch
import torch.nn as nn

from .config import ThermoConfig


class ThermodynamicMonitor(nn.Module):
    """
    Monitors and computes thermodynamic quantities for TGSFN.

    The entropy production Π_q measures how far the system is from
    thermodynamic equilibrium. Lower Π_q means more efficient computation.

    Components of Π_q:
    1. Energy dissipation from spikes: (V_m - V_reset)² / τ_m
    2. Jacobian norm: stability/sensitivity measure
    3. Mutual information: spike predictability (optional)

    The thermodynamic loss encourages:
    - Low entropy production (efficient dynamics)
    - Bounded curvature (stable manifold geometry)
    - Good predictions (low VFE)
    """

    def __init__(self, config: ThermoConfig):
        super().__init__()
        self.config = config
        self.device = torch.device(config.device)

        self.lambda_diss = config.lambda_diss
        self.lambda_geom = config.lambda_geom
        self.lambda_jacobian = config.lambda_jacobian

        # History for mutual information estimation
        self._spike_history: list = []
        self._input_history: list = []
        self._history_maxlen = 100

    def compute_spike_dissipation(
        self,
        membrane_potentials: torch.Tensor,  # (B, N) or (B, T, N)
        spikes: torch.Tensor,               # (B, N) or (B, T, N)
        tau_m: float,
        v_reset: float = 0.0,
    ) -> torch.Tensor:
        """
        Compute energy dissipation from spikes.

        Σ_spikes (V_m - V_reset)² / τ_m

        This measures how much "energy" is lost when neurons spike and reset.

        Args:
            membrane_potentials: Pre-spike membrane potentials
            spikes: Binary spike indicators
            tau_m: Membrane time constant
            v_reset: Reset potential (usually 0)

        Returns:
            Scalar dissipation term
        """
        # Energy at spike = (V_m - V_reset)^2
        energy = (membrane_potentials - v_reset) ** 2

        # Only count spiking neurons
        spike_energy = energy * spikes

        # Normalize by time constant
        dissipation = spike_energy.sum() / tau_m

        return dissipation

    def compute_jacobian_term(
        self,
        jacobian: Optional[torch.Tensor] = None,
        jacobian_norm: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute Jacobian contribution to entropy production.

        λ_J trace(J^T J) = λ_J ||J||_F²

        This measures sensitivity to perturbations. High Jacobian norm
        means small input changes cause large output changes.

        Args:
            jacobian: (N, N) or (B, N, N) Jacobian matrix
            jacobian_norm: Pre-computed Frobenius norm

        Returns:
            Scalar Jacobian term
        """
        if jacobian_norm is not None:
            return self.lambda_jacobian * jacobian_norm ** 2

        if jacobian is not None:
            # Frobenius norm squared = trace(J^T J)
            return self.lambda_jacobian * (jacobian ** 2).sum()

        return torch.tensor(0.0, device=self.device)

    def estimate_mutual_information(
        self,
        spikes: torch.Tensor,
        inputs: torch.Tensor,
    ) -> torch.Tensor:
        """
        Estimate mutual information I(spike_train; input).

        High MI means spikes are predictable from inputs.
        This is a rough estimate using correlation as proxy.

        Phase 1: Simple correlation-based approximation.
        Full implementation would use proper MI estimation.

        Args:
            spikes: (B, N) spike indicators
            inputs: (B, D) input vectors

        Returns:
            Scalar MI estimate
        """
        # Store history for better estimation
        self._spike_history.append(spikes.detach().mean(dim=0))
        self._input_history.append(inputs.detach().mean(dim=0))

        if len(self._spike_history) > self._history_maxlen:
            self._spike_history.pop(0)
            self._input_history.pop(0)

        if len(self._spike_history) < 10:
            return torch.tensor(0.0, device=self.device)

        # Compute correlation between spike rates and input magnitudes
        spike_rates = torch.stack(self._spike_history)  # (T, N)
        input_mags = torch.stack([x.norm() for x in self._input_history])  # (T,)

        # Mean spike rate correlation with input magnitude
        mean_spike_rate = spike_rates.mean(dim=1)  # (T,)

        # Pearson correlation as MI proxy
        if mean_spike_rate.std() > 1e-6 and input_mags.std() > 1e-6:
            corr = torch.corrcoef(
                torch.stack([mean_spike_rate, input_mags])
            )[0, 1]
            # Convert correlation to [0, 1] range
            mi_proxy = (corr.abs() ** 2).clamp(0, 1)
        else:
            mi_proxy = torch.tensor(0.0, device=self.device)

        return mi_proxy

    def compute_entropy_production(
        self,
        membrane_potentials: torch.Tensor,
        spikes: torch.Tensor,
        tau_m: float,
        jacobian_norm: Optional[torch.Tensor] = None,
        inputs: Optional[torch.Tensor] = None,
        v_reset: float = 0.0,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute full entropy production Π_q.

        Π_q = spike_dissipation + jacobian_term + mi_term

        Args:
            membrane_potentials: (B, N) membrane states
            spikes: (B, N) spike indicators
            tau_m: Membrane time constant
            jacobian_norm: Pre-computed Jacobian norm
            inputs: (B, D) inputs for MI estimation
            v_reset: Reset potential

        Returns:
            Dict with Pi_q and component breakdown
        """
        # Spike dissipation
        dissipation = self.compute_spike_dissipation(
            membrane_potentials, spikes, tau_m, v_reset
        )

        # Jacobian term
        jacobian_term = self.compute_jacobian_term(
            jacobian_norm=jacobian_norm
        )

        # Mutual information (optional)
        if inputs is not None:
            mi_term = self.estimate_mutual_information(spikes, inputs)
        else:
            mi_term = torch.tensor(0.0, device=self.device)

        # Total entropy production
        Pi_q = dissipation + jacobian_term + mi_term

        return {
            "Pi_q": Pi_q,
            "dissipation": dissipation,
            "jacobian_term": jacobian_term,
            "mi_term": mi_term,
        }

    def compute_curvature_penalty(
        self,
        z_positions: torch.Tensor,
        curvature: float = 1.0,
    ) -> torch.Tensor:
        """
        Compute sectional curvature penalty.

        For stability, we want bounded curvature. On Poincaré ball,
        curvature is -c everywhere. We penalize positions near the
        boundary where geodesics diverge rapidly.

        Args:
            z_positions: (B, N, D) positions on manifold
            curvature: Base curvature c

        Returns:
            Scalar curvature penalty
        """
        # Distance from origin (higher = closer to boundary = more unstable)
        norms = z_positions.norm(dim=-1)  # (B, N)

        # Penalty increases as we approach boundary
        # For Poincaré ball: positions near boundary have extreme curvature
        max_norm = 0.99
        boundary_proximity = (norms / max_norm).clamp(0, 1)

        # Quadratic penalty on boundary proximity
        penalty = (boundary_proximity ** 2).mean()

        return penalty * curvature

    def compute_total_loss(
        self,
        vfe: torch.Tensor,                 # Variational free energy
        Pi_q: torch.Tensor,                # Entropy production
        curvature_penalty: torch.Tensor,   # Sectional curvature term
    ) -> Dict[str, torch.Tensor]:
        """
        Compute total thermodynamic loss.

        L_total = VFE + λ_diss Π_q + λ_geom K_sectional

        Args:
            vfe: Variational free energy (prediction error)
            Pi_q: Entropy production
            curvature_penalty: Curvature regularization

        Returns:
            Dict with total loss and components
        """
        loss = (
            vfe
            + self.lambda_diss * Pi_q
            + self.lambda_geom * curvature_penalty
        )

        return {
            "loss": loss,
            "vfe": vfe,
            "diss_loss": self.lambda_diss * Pi_q,
            "geom_loss": self.lambda_geom * curvature_penalty,
        }

    def forward(
        self,
        membrane_potentials: torch.Tensor,
        spikes: torch.Tensor,
        z_positions: torch.Tensor,
        vfe: torch.Tensor,
        tau_m: float,
        jacobian_norm: Optional[torch.Tensor] = None,
        inputs: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Full forward pass: compute all thermodynamic quantities and loss.

        Returns:
            Dict with Pi_q, curvature_penalty, total_loss, and components
        """
        # Entropy production
        pi_q_result = self.compute_entropy_production(
            membrane_potentials, spikes, tau_m, jacobian_norm, inputs
        )

        # Curvature penalty
        curv_penalty = self.compute_curvature_penalty(z_positions)

        # Total loss
        loss_result = self.compute_total_loss(
            vfe, pi_q_result["Pi_q"], curv_penalty
        )

        return {
            **pi_q_result,
            "curvature_penalty": curv_penalty,
            **loss_result,
        }


if __name__ == "__main__":
    print("=== ThermodynamicMonitor Test ===")

    config = ThermoConfig(
        lambda_diss=0.1,
        lambda_geom=0.1,
        lambda_jacobian=0.01,
        device="cpu",
    )
    monitor = ThermodynamicMonitor(config)

    # Test entropy production
    print("\n--- Entropy production test ---")
    B, N = 4, 64
    membrane = torch.randn(B, N) * 0.5
    spikes = (torch.rand(B, N) > 0.9).float()
    inputs = torch.randn(B, 8)

    pi_q_result = monitor.compute_entropy_production(
        membrane, spikes, tau_m=10.0,
        jacobian_norm=torch.tensor(0.5),
        inputs=inputs,
    )
    print(f"Π_q: {pi_q_result['Pi_q'].item():.6f}")
    print(f"  Dissipation: {pi_q_result['dissipation'].item():.6f}")
    print(f"  Jacobian: {pi_q_result['jacobian_term'].item():.6f}")

    # Test curvature penalty
    print("\n--- Curvature penalty test ---")
    z = torch.randn(B, N, 32) * 0.3  # Positions inside ball
    curv_pen = monitor.compute_curvature_penalty(z)
    print(f"Curvature penalty (inside): {curv_pen.item():.6f}")

    z_boundary = torch.randn(B, N, 32) * 0.9  # Positions near boundary
    curv_pen_boundary = monitor.compute_curvature_penalty(z_boundary)
    print(f"Curvature penalty (boundary): {curv_pen_boundary.item():.6f}")

    # Test total loss
    print("\n--- Total loss test ---")
    vfe = torch.tensor(1.5)
    loss_result = monitor.compute_total_loss(
        vfe, pi_q_result["Pi_q"], curv_pen
    )
    print(f"Total loss: {loss_result['loss'].item():.6f}")
    print(f"  VFE: {loss_result['vfe'].item():.6f}")
    print(f"  Diss: {loss_result['diss_loss'].item():.6f}")
    print(f"  Geom: {loss_result['geom_loss'].item():.6f}")

    print("\nThermodynamicMonitor test passed!")
