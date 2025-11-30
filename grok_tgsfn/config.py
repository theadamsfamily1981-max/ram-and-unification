# grok_tgsfn/config.py
# Configuration dataclasses for the Grok TGSFN framework
#
# All hyperparameters and structural settings are centralized here
# to maintain alignment with the theoretical specification.

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class L1Config:
    """
    Config for L1 - Homeostatic Core.

    Implements the "vulnerable body" with continuous-time dynamics:
        dn/dt = -Γ d(t) + ξ(t) + u(t)

    Where:
        n(t) ∈ ℝ₊^K are the internal needs
        Γ is the relaxation rate matrix (diagonal for now)
        d(t) = Σ⁻¹ n(t) is the drive vector
        ξ(t) is interoceptive noise/prediction error
        u(t) is the control input from actions
    """
    num_needs: int = 8              # K - dimensionality of needs vector
    gamma: float = 0.1              # Γ - relaxation rate (scalar for diagonal case)
    sigma_init: float = 1.0         # Initial diagonal entries of precision Σ
    dt: float = 1.0                 # Integration timestep for Euler method
    device: str = "cpu"

    # Optional: per-need gamma values for non-uniform relaxation
    gamma_per_need: Optional[List[float]] = None


@dataclass
class L2Config:
    """
    Config for L2 - Hyperbolic Appraisal Engine.

    Implements cognitive appraisal on Poincaré ball:
        z_s = Exp_0(s)              - state embedding
        z_b = Exp_0(beliefs)        - belief embedding
        z = z_s ⊕ z_b              - Möbius binding
        a(t) = W_app · log_0(z)    - appraisal readout

    The 8-dimensional appraisal vector follows CoRE:
        [pleasantness, relevance, certainty, control,
         coping_potential, urgency, agency, norm_compatibility]
    """
    obs_dim: int = 8                # Observation dimension
    num_needs: int = 8              # Needs dimension (from L1)
    goal_dim: int = 0               # Optional goal embedding dimension
    app_dim: int = 8                # Appraisal vector dimension
    hyp_dim: int = 64               # Hyperbolic embedding dimension m
    curvature: float = 1.0          # Poincaré ball curvature c
    device: str = "cpu"


@dataclass
class L3Config:
    """
    Config for L3 - Gating Controller.

    Implements explicit gating equations from the memo:
        τ(t) = σ(3 - 2A + U_epi)
        η_scale(t) = σ(D + cop - A + U_epi)
        mem_write_p = σ(3V + 2r + u + U_epi - A)
        att_gain = σ(D + ctrl + ag - U_conflict)
        p_sleep = σ(A + ||d|| - cop - V)

    No learnable parameters - pure functional mapping.
    """
    device: str = "cpu"


@dataclass
class L4Config:
    """
    Config for L4 - Memory & Personalization.

    Key equations:
        sal(t) = ||d|| · ||a|| + β U_epi          (salience)
        p_replay(i) ∝ sal_i · exp(-λ_diss Π_q)   (replay distribution)
                         · exp(-λ_id Δidentity)
        ℒ_homeo = λ_homeo · ½ ||Δn||²_Σ          (personalization loss)
    """
    beta_epistemic: float = 1.0     # β - epistemic weight in salience
    lambda_diss: float = 0.1        # λ_diss - thermodynamic penalty in replay
    lambda_id: float = 0.1          # λ_id - identity preservation in replay
    lambda_homeo: float = 0.1       # λ_homeo - homeostatic regularizer weight
    memory_capacity: int = 10000    # Max stored experiences
    device: str = "cpu"


@dataclass
class TGSFNConfig:
    """
    Config for TGSFN - Thermodynamic-Geometric Spiking Field Network.

    The manifold is M = ℝⁿ × Sᵖ × ℍᵐ:
        - ℝⁿ: Euclidean component (n = euclid_dim)
        - Sᵖ: Spherical component (p = sphere_dim)
        - ℍᵐ: Hyperbolic component (m = hyper_dim)

    Neuron i has:
        - Position z_i ∈ M
        - Tangent state x_i(t) ∈ T_{z_i}M

    Dynamics (GCSD):
        dx_i/dt = -∇_z V(z_i) + Σ_j w_ij log_{z_i}(spike_j) + D Δ_R x_i
    """
    euclid_dim: int = 0             # Euclidean manifold dimension
    sphere_dim: int = 0             # Spherical manifold dimension
    hyper_dim: int = 64             # Hyperbolic manifold dimension
    num_neurons: int = 128          # Number of neurons in layer
    diffusion_coeff: float = 0.01   # D - diffusion coefficient
    spike_threshold: float = 1.0    # Default spike threshold θ
    tau_membrane: float = 10.0      # Membrane time constant
    device: str = "cpu"


@dataclass
class ThermoConfig:
    """
    Config for Thermodynamic Monitor.

    Computes entropy production Π_q and total loss:
        Π_q ≈ Σ_spikes (V_m - V_reset)² / τ_m
              + λ_J trace(J^T J)
              + I(spike_train; input)

        L_total = VFE + λ_diss Π_q + λ_geom sectional_K
    """
    lambda_diss: float = 0.1        # Weight on entropy production
    lambda_geom: float = 0.1        # Weight on sectional curvature penalty
    lambda_jacobian: float = 0.01   # Weight on Jacobian trace term
    device: str = "cpu"


@dataclass
class CouplingConfig:
    """
    Config for Affective Coupling.

    Maps L1/L2 outputs to neuromodulatory signals:
        V → threshold bias (dopamine-like)
        A → noise scale / inverse temperature
        D → precision weighting (self vs other)

    Prosocial coupling:
        n_eff = n_self + w_empathy · n_human(inferred)
    """
    init_empathy_w: float = 0.5     # Initial empathy weight
    threshold_scale: float = 0.1    # Scale factor for V → threshold mapping
    noise_scale_min: float = 0.01   # Minimum noise scale
    device: str = "cpu"
