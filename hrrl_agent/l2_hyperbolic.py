"""
Layer 2: Hyperbolic Appraisal

Implements appraisal on the Poincaré ball model of hyperbolic space.

Key equations:
- Appraisal space: M = ℍ^128
- Appraisal: a(t) = W_app · log_0(z_s ⊕ z_b)
- Where z_s = sensory embedding, z_b = belief embedding
- ⊕ is Möbius addition on Poincaré ball
- log_0 is logarithmic map at origin

The hyperbolic space naturally represents hierarchical/tree-like
structures in cognitive appraisal (near origin = abstract,
near boundary = concrete/specific).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, NamedTuple
import math

from .config import L2Config


class AppraisalOutput(NamedTuple):
    """Output from hyperbolic appraisal layer."""
    appraisal: torch.Tensor  # a(t) appraisal vector
    z_combined: torch.Tensor  # z_s ⊕ z_b on Poincaré ball
    epistemic: torch.Tensor  # Epistemic uncertainty
    aleatoric: torch.Tensor  # Aleatoric uncertainty
    hyperbolic_norm: torch.Tensor  # Distance from origin (abstraction level)


class PoincareOperations:
    """
    Operations on the Poincaré ball model of hyperbolic space.

    The Poincaré ball B^n_c = {x ∈ ℝ^n : c||x||² < 1} with curvature -c.

    Key operations:
    - Möbius addition: x ⊕ y
    - Exponential map: exp_x(v) maps tangent vector to manifold
    - Logarithmic map: log_x(y) maps manifold point to tangent space
    """

    def __init__(self, c: float = 1.0, eps: float = 1e-5):
        """
        Args:
            c: Curvature parameter (hyperbolic curvature is -c)
            eps: Numerical stability constant
        """
        self.c = c
        self.eps = eps

    def mobius_add(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Möbius addition on Poincaré ball.

        x ⊕ y = ((1 + 2c⟨x,y⟩ + c||y||²)x + (1 - c||x||²)y) /
                (1 + 2c⟨x,y⟩ + c²||x||²||y||²)

        Args:
            x, y: Points on Poincaré ball [..., d]

        Returns:
            x ⊕ y on Poincaré ball
        """
        c = self.c

        x2 = torch.sum(x * x, dim=-1, keepdim=True).clamp(min=self.eps)
        y2 = torch.sum(y * y, dim=-1, keepdim=True).clamp(min=self.eps)
        xy = torch.sum(x * y, dim=-1, keepdim=True)

        num = (1 + 2 * c * xy + c * y2) * x + (1 - c * x2) * y
        denom = 1 + 2 * c * xy + c * c * x2 * y2

        return num / denom.clamp(min=self.eps)

    def mobius_scalar_mul(self, r: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        Möbius scalar multiplication.

        r ⊗ x = tanh(r · artanh(√c ||x||)) · x / (√c ||x||)
        """
        c = self.c
        sqrt_c = math.sqrt(c)

        x_norm = torch.norm(x, dim=-1, keepdim=True).clamp(min=self.eps)
        x_normalized = x / x_norm

        # artanh(√c ||x||)
        sqrt_c_norm = (sqrt_c * x_norm).clamp(max=1.0 - self.eps)
        artanh_val = 0.5 * torch.log((1 + sqrt_c_norm) / (1 - sqrt_c_norm + self.eps))

        # tanh(r · artanh(...))
        new_norm = torch.tanh(r * artanh_val) / sqrt_c

        return new_norm * x_normalized

    def exp_map_zero(self, v: torch.Tensor) -> torch.Tensor:
        """
        Exponential map at origin.

        exp_0(v) = tanh(√c ||v||) · v / (√c ||v||)

        Maps tangent vector at origin to point on manifold.
        """
        c = self.c
        sqrt_c = math.sqrt(c)

        v_norm = torch.norm(v, dim=-1, keepdim=True).clamp(min=self.eps)
        v_normalized = v / v_norm

        return torch.tanh(sqrt_c * v_norm) * v_normalized / sqrt_c

    def exp_map(self, x: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """
        Exponential map at point x.

        exp_x(v) = x ⊕ (tanh(√c λ_x ||v|| / 2) · v / (√c ||v||))

        where λ_x = 2 / (1 - c||x||²) is the conformal factor.
        """
        c = self.c
        sqrt_c = math.sqrt(c)

        x2 = torch.sum(x * x, dim=-1, keepdim=True)
        lambda_x = 2.0 / (1 - c * x2).clamp(min=self.eps)

        v_norm = torch.norm(v, dim=-1, keepdim=True).clamp(min=self.eps)
        v_normalized = v / v_norm

        second_term = torch.tanh(sqrt_c * lambda_x * v_norm / 2) * v_normalized / sqrt_c

        return self.mobius_add(x, second_term)

    def log_map_zero(self, y: torch.Tensor) -> torch.Tensor:
        """
        Logarithmic map at origin.

        log_0(y) = artanh(√c ||y||) · y / (√c ||y||)

        Maps point on manifold to tangent vector at origin.
        """
        c = self.c
        sqrt_c = math.sqrt(c)

        y_norm = torch.norm(y, dim=-1, keepdim=True).clamp(min=self.eps)
        y_normalized = y / y_norm

        # Clamp to stay inside ball
        sqrt_c_norm = (sqrt_c * y_norm).clamp(max=1.0 - self.eps)

        # artanh
        artanh_val = 0.5 * torch.log((1 + sqrt_c_norm) / (1 - sqrt_c_norm + self.eps))

        return artanh_val * y_normalized / sqrt_c

    def log_map(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Logarithmic map at point x.

        log_x(y) = (2 / (√c λ_x)) · artanh(√c ||-x ⊕ y||) · (-x ⊕ y) / ||-x ⊕ y||
        """
        c = self.c
        sqrt_c = math.sqrt(c)

        # -x ⊕ y
        neg_x = -x
        diff = self.mobius_add(neg_x, y)

        diff_norm = torch.norm(diff, dim=-1, keepdim=True).clamp(min=self.eps)
        diff_normalized = diff / diff_norm

        # Conformal factor at x
        x2 = torch.sum(x * x, dim=-1, keepdim=True)
        lambda_x = 2.0 / (1 - c * x2).clamp(min=self.eps)

        # artanh
        sqrt_c_norm = (sqrt_c * diff_norm).clamp(max=1.0 - self.eps)
        artanh_val = 0.5 * torch.log((1 + sqrt_c_norm) / (1 - sqrt_c_norm + self.eps))

        return (2.0 / (sqrt_c * lambda_x)) * artanh_val * diff_normalized

    def distance(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Hyperbolic distance on Poincaré ball.

        d(x, y) = (2/√c) · artanh(√c ||-x ⊕ y||)
        """
        c = self.c
        sqrt_c = math.sqrt(c)

        diff = self.mobius_add(-x, y)
        diff_norm = torch.norm(diff, dim=-1).clamp(min=self.eps)

        sqrt_c_norm = (sqrt_c * diff_norm).clamp(max=1.0 - self.eps)
        artanh_val = 0.5 * torch.log((1 + sqrt_c_norm) / (1 - sqrt_c_norm + self.eps))

        return (2.0 / sqrt_c) * artanh_val

    def project(self, x: torch.Tensor, max_norm: float = 0.99) -> torch.Tensor:
        """Project point to stay inside Poincaré ball."""
        c = self.c
        max_norm_c = max_norm / math.sqrt(c)

        x_norm = torch.norm(x, dim=-1, keepdim=True)
        cond = x_norm > max_norm_c
        x_normalized = x / x_norm.clamp(min=self.eps)

        return torch.where(cond, x_normalized * max_norm_c, x)


class HyperbolicLinear(nn.Module):
    """
    Linear layer in hyperbolic space.

    Maps from tangent space at origin to tangent space,
    then projects via exp_0.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        c: float = 1.0,
        bias: bool = True
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.c = c

        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias', None)

        self.poincare = PoincareOperations(c)
        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.weight, gain=0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        1. Map to tangent space: v = log_0(x)
        2. Linear transform: v' = W @ v + b
        3. Map back: y = exp_0(v')
        """
        # Log map to tangent space
        v = self.poincare.log_map_zero(x)

        # Linear transform in tangent space
        v_out = F.linear(v, self.weight, self.bias)

        # Exp map back to manifold
        y = self.poincare.exp_map_zero(v_out)

        return self.poincare.project(y)


class HyperbolicAppraisalL2(nn.Module):
    """
    Layer 2: Hyperbolic Cognitive Appraisal.

    Computes appraisal vector by combining sensory and belief
    embeddings in hyperbolic space, then projecting to appraisal
    dimensions.

    a(t) = W_app · log_0(z_s ⊕ z_b)
    """

    def __init__(self, config: L2Config):
        super().__init__()
        self.config = config
        self.poincare = PoincareOperations(config.curvature, config.eps)

        # Encoders: map inputs to Poincaré ball
        self.sensory_encoder = nn.Sequential(
            nn.Linear(config.sensory_dim, config.hyperbolic_dim),
            nn.Tanh(),  # Keep bounded for exp_0
        )

        self.belief_encoder = nn.Sequential(
            nn.Linear(config.belief_dim, config.hyperbolic_dim),
            nn.Tanh(),
        )

        # Appraisal projection: W_app
        # Maps from tangent space (log_0 output) to appraisal vector
        self.W_app = nn.Linear(config.hyperbolic_dim, config.appraisal_dim)

        # Uncertainty heads
        self.uncertainty_net = nn.Sequential(
            nn.Linear(config.hyperbolic_dim, config.uncertainty_hidden),
            nn.ReLU(),
            nn.Linear(config.uncertainty_hidden, 2)  # [epistemic, aleatoric]
        )

    def forward(
        self,
        sensory: torch.Tensor,
        belief: torch.Tensor
    ) -> AppraisalOutput:
        """
        Compute hyperbolic appraisal.

        Args:
            sensory: Sensory input [..., sensory_dim]
            belief: Belief state [..., belief_dim]

        Returns:
            AppraisalOutput with appraisal vector and uncertainties
        """
        # Encode to tangent space, then project to Poincaré ball
        z_s_tangent = self.sensory_encoder(sensory)
        z_b_tangent = self.belief_encoder(belief)

        # Project to Poincaré ball via exp_0
        z_s = self.poincare.exp_map_zero(z_s_tangent)
        z_b = self.poincare.exp_map_zero(z_b_tangent)

        # Ensure we stay inside ball
        z_s = self.poincare.project(z_s, self.config.max_norm)
        z_b = self.poincare.project(z_b, self.config.max_norm)

        # Möbius addition: z_s ⊕ z_b
        z_combined = self.poincare.mobius_add(z_s, z_b)
        z_combined = self.poincare.project(z_combined, self.config.max_norm)

        # Log map back to tangent space at origin
        v_combined = self.poincare.log_map_zero(z_combined)

        # Appraisal: a(t) = W_app · log_0(z_s ⊕ z_b)
        appraisal = self.W_app(v_combined)

        # Uncertainty estimation from hyperbolic embedding
        uncertainties = self.uncertainty_net(v_combined)
        epistemic = F.softplus(uncertainties[..., 0])
        aleatoric = F.softplus(uncertainties[..., 1])

        # Hyperbolic norm (distance from origin = abstraction level)
        # Points near origin = abstract, near boundary = concrete
        hyperbolic_norm = torch.norm(z_combined, dim=-1)

        return AppraisalOutput(
            appraisal=appraisal,
            z_combined=z_combined,
            epistemic=epistemic,
            aleatoric=aleatoric,
            hyperbolic_norm=hyperbolic_norm
        )

    def compute_hyperbolic_distance(
        self,
        z1: torch.Tensor,
        z2: torch.Tensor
    ) -> torch.Tensor:
        """Compute hyperbolic distance between two points."""
        return self.poincare.distance(z1, z2)


class HyperbolicAppraisalWithDrive(nn.Module):
    """
    Extended appraisal that also conditions on homeostatic drive.

    Incorporates drive state into the appraisal computation,
    allowing needs to influence cognitive evaluation.
    """

    def __init__(self, config: L2Config, num_needs: int):
        super().__init__()
        self.config = config
        self.num_needs = num_needs
        self.poincare = PoincareOperations(config.curvature, config.eps)

        # Extended encoder includes drive
        self.sensory_encoder = nn.Sequential(
            nn.Linear(config.sensory_dim + num_needs, config.hyperbolic_dim),
            nn.Tanh(),
        )

        self.belief_encoder = nn.Sequential(
            nn.Linear(config.belief_dim + num_needs, config.hyperbolic_dim),
            nn.Tanh(),
        )

        self.W_app = nn.Linear(config.hyperbolic_dim, config.appraisal_dim)

        self.uncertainty_net = nn.Sequential(
            nn.Linear(config.hyperbolic_dim, config.uncertainty_hidden),
            nn.ReLU(),
            nn.Linear(config.uncertainty_hidden, 2)
        )

    def forward(
        self,
        sensory: torch.Tensor,
        belief: torch.Tensor,
        drive: torch.Tensor
    ) -> AppraisalOutput:
        """
        Compute drive-conditioned hyperbolic appraisal.

        Args:
            sensory: Sensory input
            belief: Belief state
            drive: Homeostatic drive vector

        Returns:
            AppraisalOutput
        """
        # Concatenate drive with inputs
        sensory_aug = torch.cat([sensory, drive], dim=-1)
        belief_aug = torch.cat([belief, drive], dim=-1)

        # Encode to tangent space
        z_s_tangent = self.sensory_encoder(sensory_aug)
        z_b_tangent = self.belief_encoder(belief_aug)

        # Project to Poincaré ball
        z_s = self.poincare.exp_map_zero(z_s_tangent)
        z_b = self.poincare.exp_map_zero(z_b_tangent)

        z_s = self.poincare.project(z_s, self.config.max_norm)
        z_b = self.poincare.project(z_b, self.config.max_norm)

        # Combine
        z_combined = self.poincare.mobius_add(z_s, z_b)
        z_combined = self.poincare.project(z_combined, self.config.max_norm)

        # Log map and appraisal
        v_combined = self.poincare.log_map_zero(z_combined)
        appraisal = self.W_app(v_combined)

        # Uncertainties
        uncertainties = self.uncertainty_net(v_combined)
        epistemic = F.softplus(uncertainties[..., 0])
        aleatoric = F.softplus(uncertainties[..., 1])

        hyperbolic_norm = torch.norm(z_combined, dim=-1)

        return AppraisalOutput(
            appraisal=appraisal,
            z_combined=z_combined,
            epistemic=epistemic,
            aleatoric=aleatoric,
            hyperbolic_norm=hyperbolic_norm
        )


if __name__ == "__main__":
    # Sanity checks
    print("Testing Poincaré operations...")

    poincare = PoincareOperations(c=1.0)

    # Test exp and log are inverses
    v = torch.randn(10, 128) * 0.1
    y = poincare.exp_map_zero(v)
    v_recovered = poincare.log_map_zero(y)
    error = torch.norm(v - v_recovered).item()
    print(f"  exp_0/log_0 inverse error: {error:.6f}")

    # Test Möbius addition
    x = torch.randn(10, 128) * 0.1
    x = poincare.project(poincare.exp_map_zero(x))
    y = torch.randn(10, 128) * 0.1
    y = poincare.project(poincare.exp_map_zero(y))
    z = poincare.mobius_add(x, y)
    print(f"  Möbius add result norm: {torch.norm(z, dim=-1).mean():.4f}")

    # Test HyperbolicAppraisalL2
    print("\nTesting HyperbolicAppraisalL2...")
    config = L2Config()
    model = HyperbolicAppraisalL2(config)

    sensory = torch.randn(4, config.sensory_dim)
    belief = torch.randn(4, config.belief_dim)

    output = model(sensory, belief)
    print(f"  Appraisal shape: {output.appraisal.shape}")
    print(f"  z_combined shape: {output.z_combined.shape}")
    print(f"  Epistemic: {output.epistemic}")
    print(f"  Hyperbolic norm: {output.hyperbolic_norm}")

    # Test with drive
    print("\nTesting HyperbolicAppraisalWithDrive...")
    model_drive = HyperbolicAppraisalWithDrive(config, num_needs=8)
    drive = torch.randn(4, 8) * 0.5

    output_drive = model_drive(sensory, belief, drive)
    print(f"  Appraisal shape: {output_drive.appraisal.shape}")
    print(f"  Epistemic: {output_drive.epistemic}")
