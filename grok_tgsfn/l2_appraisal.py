# grok_tgsfn/l2_appraisal.py
# Layer 2: Hyperbolic Cognitive Appraisal Engine
#
# Implements cognitive appraisal theory on the Poincaré ball:
#
# State vector:
#   s(t) = [o(t), n(t), g(t)]  - observation, needs, goals
#
# Hyperbolic embeddings:
#   z_s = Exp_0(W_s · s)       - state embedding
#   z_b = Exp_0(beliefs)       - belief embedding (learned)
#
# Hyperbolic binding (Möbius addition):
#   z = z_s ⊕ z_b
#
# Appraisal readout:
#   a(t) = W_app · log_0(z)
#
# The 8-dimensional appraisal vector (CoRE-inspired):
#   [0] pleasantness    - intrinsic hedonic tone
#   [1] relevance       - personal importance
#   [2] certainty       - confidence in situation
#   [3] control         - perceived agency
#   [4] coping_potential - ability to handle
#   [5] urgency         - need for immediate action
#   [6] agency          - self vs external causation
#   [7] norm_compat     - social/moral fit
#
# The hyperbolic geometry enables:
#   - Hierarchical concept representation
#   - Efficient binding/unbinding operations
#   - Natural uncertainty representation (distance from origin)

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict, Tuple
import math

import torch
import torch.nn as nn

from .config import L2Config

# Attempt to import geoopt for proper hyperbolic ops
try:
    import geoopt
    from geoopt import ManifoldParameter
    from geoopt.manifolds import PoincareBall
    HAS_GEOOPT = True
except ImportError:
    HAS_GEOOPT = False
    PoincareBall = None
    ManifoldParameter = nn.Parameter


# Appraisal dimension names
APPRAISAL_DIMS = [
    "pleasantness",      # [0] Intrinsic hedonic quality
    "relevance",         # [1] Personal relevance/importance
    "certainty",         # [2] Situational certainty
    "control",           # [3] Perceived control/agency
    "coping_potential",  # [4] Ability to cope/handle
    "urgency",           # [5] Temporal urgency
    "agency",            # [6] Self vs external attribution
    "norm_compatibility", # [7] Social/moral norm fit
]


class HyperbolicAppraisalL2(nn.Module):
    """
    L2 - Cognitive Appraisal Engine with Hyperbolic VSA.

    Uses the Poincaré ball model for hyperbolic geometry, which provides:
    - Natural hierarchy: distance from origin encodes abstraction level
    - Efficient binding: Möbius addition combines concepts
    - Uncertainty: points near boundary have high uncertainty

    The appraisal process:
    1. Encode state s(t) = [obs, needs, goals] to tangent space
    2. Map to hyperbolic space via exponential map
    3. Bind with learned belief embedding via Möbius addition
    4. Read out appraisal via log map + linear projection
    """

    def __init__(self, config: L2Config):
        super().__init__()
        self.config = config
        self.device = torch.device(config.device)
        self.curvature = config.curvature  # c in Poincaré ball

        # Input dimension: obs + needs + goals
        in_dim = config.obs_dim + config.num_needs + config.goal_dim

        # Linear encoder to tangent space at origin
        self.linear_state = nn.Linear(in_dim, config.hyp_dim).to(self.device)

        # Initialize with small weights for stability
        nn.init.xavier_uniform_(self.linear_state.weight, gain=0.1)
        nn.init.zeros_(self.linear_state.bias)

        # Hyperbolic manifold
        if HAS_GEOOPT:
            self.manifold = PoincareBall(c=config.curvature)
            # Beliefs as manifold parameter
            self.beliefs = ManifoldParameter(
                torch.zeros(config.hyp_dim, device=self.device),
                manifold=self.manifold,
            )
        else:
            self.manifold = None
            self.beliefs = nn.Parameter(
                torch.zeros(config.hyp_dim, device=self.device)
            )

        # Appraisal projection: tangent space → appraisal vector
        self.W_app = nn.Linear(config.hyp_dim, config.app_dim, bias=True).to(self.device)

        # Initialize appraisal layer
        nn.init.xavier_uniform_(self.W_app.weight, gain=0.5)
        nn.init.zeros_(self.W_app.bias)

    def forward(
        self,
        obs: torch.Tensor,
        needs: torch.Tensor,
        goals: Optional[torch.Tensor] = None,
        beliefs_override: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute appraisal vector from observation, needs, and goals.

        Args:
            obs: (B, obs_dim) - external observation
            needs: (B, num_needs) or (num_needs,) - current needs from L1
            goals: (B, goal_dim) or None - optional goal representation
            beliefs_override: (hyp_dim,) or (B, hyp_dim) - optional external beliefs

        Returns:
            appraisal: (B, app_dim) - 8-dimensional appraisal vector
        """
        B = obs.size(0)
        device = self.device

        # Handle unbatched needs
        if needs.dim() == 1:
            needs = needs.unsqueeze(0).expand(B, -1)

        # Handle missing or unbatched goals
        if goals is None and self.config.goal_dim > 0:
            goals = torch.zeros(B, self.config.goal_dim, device=device)
        elif goals is not None and goals.dim() == 1:
            goals = goals.unsqueeze(0).expand(B, -1)

        # Construct state vector s = [obs, needs, goals]
        if self.config.goal_dim > 0 and goals is not None:
            s = torch.cat([obs.to(device), needs.to(device), goals.to(device)], dim=-1)
        else:
            s = torch.cat([obs.to(device), needs.to(device)], dim=-1)

        # Encode to tangent space
        v_state = self.linear_state(s)  # (B, hyp_dim)

        # Map to Poincaré ball: z_s = Exp_0(v_state)
        z_s = self._exp_map_zero(v_state)

        # Get belief embedding
        if beliefs_override is not None:
            if beliefs_override.dim() == 1:
                z_b = self._exp_map_zero(beliefs_override.view(1, -1)).expand(B, -1)
            else:
                z_b = self._exp_map_zero(beliefs_override.to(device))
        else:
            z_b = self._exp_map_zero(self.beliefs.view(1, -1)).expand(B, -1)

        # Hyperbolic binding: z = z_s ⊕ z_b
        z_bound = self._mobius_add(z_s, z_b)

        # Map back to tangent space: v = log_0(z)
        v_tangent = self._log_map_zero(z_bound)

        # Linear readout to appraisal
        appraisal = self.W_app(v_tangent)

        return appraisal

    # =========================================================================
    #  Hyperbolic Operations (Poincaré Ball, curvature c)
    # =========================================================================

    def _exp_map_zero(self, v: torch.Tensor) -> torch.Tensor:
        """
        Exponential map at origin: T_0 M → M

        For Poincaré ball with curvature c:
            Exp_0(v) = tanh(sqrt(c) ||v||) * v / (sqrt(c) ||v||)

        Maps tangent vectors to points on the manifold.
        """
        if self.manifold is not None:
            return self.manifold.expmap0(v)

        # Manual implementation
        c = self.curvature
        sqrt_c = math.sqrt(c)
        norm = v.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        return torch.tanh(sqrt_c * norm) * v / (sqrt_c * norm)

    def _log_map_zero(self, z: torch.Tensor) -> torch.Tensor:
        """
        Logarithmic map at origin: M → T_0 M

        For Poincaré ball with curvature c:
            log_0(z) = arctanh(sqrt(c) ||z||) * z / (sqrt(c) ||z||)

        Maps points on manifold back to tangent space.
        """
        if self.manifold is not None:
            return self.manifold.logmap0(z)

        # Manual implementation
        c = self.curvature
        sqrt_c = math.sqrt(c)
        norm = z.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        # Clamp norm to avoid numerical issues at boundary
        norm = norm.clamp(max=1.0 / sqrt_c - 1e-5)
        return torch.atanh(sqrt_c * norm) * z / (sqrt_c * norm)

    def _mobius_add(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Möbius addition on Poincaré ball with curvature c.

        x ⊕ y = ((1 + 2c⟨x,y⟩ + c||y||²) x + (1 - c||x||²) y) /
                 (1 + 2c⟨x,y⟩ + c²||x||²||y||²)

        This is the hyperbolic analogue of vector addition.
        """
        if self.manifold is not None:
            return self.manifold.mobius_add(x, y)

        # Manual implementation
        c = self.curvature
        x2 = (x * x).sum(dim=-1, keepdim=True)  # ||x||²
        y2 = (y * y).sum(dim=-1, keepdim=True)  # ||y||²
        xy = (x * y).sum(dim=-1, keepdim=True)  # ⟨x, y⟩

        num = (1 + 2 * c * xy + c * y2) * x + (1 - c * x2) * y
        den = 1 + 2 * c * xy + c * c * x2 * y2

        return num / den.clamp_min(1e-8)

    def _hyperbolic_distance(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Hyperbolic distance on Poincaré ball.

        d(x, y) = (2/sqrt(c)) arctanh(sqrt(c) ||(-x) ⊕ y||)
        """
        c = self.curvature
        sqrt_c = math.sqrt(c)

        # Compute (-x) ⊕ y
        neg_x = -x
        diff = self._mobius_add(neg_x, y)
        diff_norm = diff.norm(dim=-1)

        return (2.0 / sqrt_c) * torch.atanh(sqrt_c * diff_norm.clamp(max=1.0 - 1e-5))

    # =========================================================================
    #  Utilities
    # =========================================================================

    def get_appraisal_dict(self, appraisal: torch.Tensor) -> Dict[str, float]:
        """
        Convert appraisal tensor to named dictionary.

        Args:
            appraisal: (app_dim,) or (1, app_dim)

        Returns:
            Dict mapping dimension names to float values
        """
        if appraisal.dim() == 2:
            appraisal = appraisal.squeeze(0)

        return {
            name: float(appraisal[i].item())
            for i, name in enumerate(APPRAISAL_DIMS[:len(appraisal)])
        }

    def get_uncertainty_from_position(self, z: torch.Tensor) -> torch.Tensor:
        """
        Compute uncertainty from position on Poincaré ball.

        Points near the boundary (||z|| → 1/sqrt(c)) have high uncertainty.
        Points near origin have low uncertainty.

        Returns uncertainty in [0, 1].
        """
        c = self.curvature
        max_norm = 1.0 / math.sqrt(c) - 1e-5
        norm = z.norm(dim=-1)
        return norm / max_norm


if __name__ == "__main__":
    print("=== HyperbolicAppraisalL2 Test ===")
    print(f"Geoopt available: {HAS_GEOOPT}")

    config = L2Config(
        obs_dim=8,
        num_needs=4,
        goal_dim=0,
        app_dim=8,
        hyp_dim=32,
        curvature=1.0,
        device="cpu",
    )
    appraisal_engine = HyperbolicAppraisalL2(config)

    # Test forward pass
    B = 4
    obs = torch.randn(B, 8)
    needs = torch.tensor([0.5, 0.2, 0.8, 0.1])

    appraisal = appraisal_engine(obs, needs)
    print(f"Appraisal shape: {appraisal.shape}")
    print(f"Appraisal[0]: {appraisal[0].tolist()}")

    # Get named appraisal
    named = appraisal_engine.get_appraisal_dict(appraisal[0])
    print(f"\nNamed appraisal:")
    for k, v in named.items():
        print(f"  {k}: {v:.4f}")

    # Test hyperbolic operations
    print("\n--- Hyperbolic ops test ---")
    v = torch.randn(1, 32) * 0.5
    z = appraisal_engine._exp_map_zero(v)
    v_recovered = appraisal_engine._log_map_zero(z)
    print(f"Exp-Log roundtrip error: {(v - v_recovered).abs().max().item():.6f}")

    # Test Möbius addition associativity (approximately)
    a = appraisal_engine._exp_map_zero(torch.randn(1, 32) * 0.3)
    b = appraisal_engine._exp_map_zero(torch.randn(1, 32) * 0.3)
    ab = appraisal_engine._mobius_add(a, b)
    print(f"Möbius add result norm: {ab.norm().item():.4f}")

    print("\nHyperbolicAppraisalL2 test passed!")
