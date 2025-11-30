"""
Configuration dataclasses for HRRL Agent.

All hyperparameters are centralized here for easy tuning.
"""
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class L1Config:
    """Layer 1: Homeostatic Core configuration."""
    num_needs: int = 8
    dt: float = 0.1  # Integration timestep
    gamma: float = 0.1  # Drive decay rate Γ in dn/dt = -Γd + ξ + u
    sigma_process: float = 0.01  # Process noise σ_ξ

    # Setpoint configuration (optimal need levels)
    setpoint_default: float = 0.5

    # Free energy precision (inverse covariance diagonal)
    inv_sigma_default: float = 1.0


@dataclass
class L2Config:
    """Layer 2: Hyperbolic Appraisal configuration."""
    # Poincaré ball parameters
    hyperbolic_dim: int = 128  # ℍ^128 appraisal space
    curvature: float = 1.0  # Negative curvature -c

    # Input dimensions
    sensory_dim: int = 64  # z_s dimension
    belief_dim: int = 64  # z_b dimension

    # Appraisal output
    appraisal_dim: int = 8  # a(t) dimension

    # Uncertainty estimation
    uncertainty_hidden: int = 64

    # Numerical stability
    eps: float = 1e-5
    max_norm: float = 0.99  # Clip to stay inside ball


@dataclass
class L3Config:
    """Layer 3: MLP Gating Controller configuration."""
    # Input: [V, A, D, appraisal, epistemic, aleatoric]
    # V, A, D = 3
    # appraisal = appraisal_dim (8)
    # epistemic, aleatoric = 2
    # Total default = 3 + 8 + 2 = 13

    hidden_dims: List[int] = field(default_factory=lambda: [64, 32])
    dropout: float = 0.1

    # Output bounds (for stability)
    tau_min: float = 0.1
    tau_max: float = 1.0
    lr_scale_min: float = 0.01
    lr_scale_max: float = 2.0


@dataclass
class L4Config:
    """Layer 4: Memory & Personalization configuration."""
    # Memory buffer
    buffer_capacity: int = 10000
    min_samples_for_replay: int = 100

    # Salience computation
    beta_epistemic: float = 0.5  # β in salience = ||d||·||a|| + βU_epi

    # Replay distribution
    lambda_dissipation: float = 0.1  # λ_diss for Π_q penalty
    lambda_identity: float = 0.5  # λ_id for identity distance
    replay_temperature: float = 1.0

    # LoRA adapter configuration
    lora_rank: int = 8
    lora_alpha: float = 16.0
    lora_dropout: float = 0.0

    # Homeostatic rejection threshold
    f_int_rejection_threshold: float = 0.5  # Reject if ΔF_int > threshold

    # Personalization regularization
    personalization_lambda: float = 0.01


@dataclass
class IdentityConfig:
    """EXPERIMENTAL: Hyperbolic Identity Manifold configuration."""
    enabled: bool = True

    # Identity embedding dimension
    identity_dim: int = 32
    curvature: float = 1.0

    # Identity attack detection thresholds (TUNABLE)
    drift_warning_threshold: float = 0.3
    drift_critical_threshold: float = 0.5
    drift_reject_threshold: float = 0.7

    # Core values (protected from modification)
    num_core_values: int = 8
    core_value_protection_weight: float = 10.0

    # Logging
    log_all_updates: bool = True


@dataclass
class DAUConfig:
    """EXPERIMENTAL: Dynamic Architecture Update configuration.

    GUARDED: Very conservative defaults with hard bans.
    """
    enabled: bool = False  # Disabled by default - must explicitly enable

    # Tiny step sizes (conservative)
    step_size: float = 0.001  # Very small
    max_step_size: float = 0.01  # Hard cap

    # Hard bans
    ban_identity_modification: bool = True  # NEVER modify identity
    ban_value_axiom_modification: bool = True  # NEVER modify core values
    banned_parameter_patterns: List[str] = field(default_factory=lambda: [
        "identity", "core_value", "axiom", "ethics"
    ])

    # Safety thresholds
    max_param_change_norm: float = 0.1
    require_approval_above: float = 0.05  # Log warning above this

    # Extensive logging
    log_all_proposals: bool = True
    log_all_rejections: bool = True
    log_parameter_norms: bool = True

    # Rollback capability
    keep_checkpoints: int = 10
    enable_auto_rollback: bool = True


@dataclass
class ThermodynamicsConfig:
    """Π_q-based criticality measurement configuration."""
    # Entropy production computation
    tau_membrane: float = 10.0  # Membrane time constant
    v_reset: float = 0.0  # Reset potential

    # Jacobian regularization
    lambda_jacobian: float = 0.01

    # Mutual information estimation (simplified)
    mi_bins: int = 20

    # Criticality tuning (MEASUREMENT ONLY by default)
    enable_auto_tuning: bool = False  # Disabled - measurement first
    target_pi_q_range: tuple = (0.1, 1.0)  # Target criticality range

    # Logging
    log_interval: int = 100  # Log every N steps
    log_detailed: bool = True


@dataclass
class TrainingConfig:
    """Online and Sleep loop configuration."""
    # Online learning
    online_lr: float = 1e-4
    online_batch_size: int = 1
    online_grad_clip: float = 1.0

    # Sleep/consolidation
    sleep_lr: float = 1e-3
    sleep_batch_size: int = 32
    sleep_epochs: int = 10
    sleep_replay_samples: int = 256

    # Interval
    steps_between_sleep: int = 1000

    # Optimizer
    optimizer: str = "adamw"
    weight_decay: float = 0.01

    # HRRL specific
    hrrl_reward_scale: float = 1.0


@dataclass
class HRRLConfig:
    """Master configuration for HRRL Agent."""
    l1: L1Config = field(default_factory=L1Config)
    l2: L2Config = field(default_factory=L2Config)
    l3: L3Config = field(default_factory=L3Config)
    l4: L4Config = field(default_factory=L4Config)
    identity: IdentityConfig = field(default_factory=IdentityConfig)
    dau: DAUConfig = field(default_factory=DAUConfig)
    thermo: ThermodynamicsConfig = field(default_factory=ThermodynamicsConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    # Device
    device: str = "cuda"

    # Random seed
    seed: Optional[int] = 42
