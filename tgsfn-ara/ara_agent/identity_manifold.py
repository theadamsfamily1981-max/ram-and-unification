# ara_agent/identity_manifold.py
# Identity Manifold for Ara Agent
#
# Implements hyperbolic (Poincaré ball) embedding for agent identity.
# The identity state z ∈ B^d (d-dimensional Poincaré ball) represents
# the agent's coherent self-model.
#
# Key Properties:
#   - Hierarchical: tree-like structures have low distortion
#   - Bounded: all points lie within unit ball ||z|| < 1
#   - Riemannian: geodesics and parallel transport well-defined
#
# Scientific Constraints:
#   - Uses geoopt for proper Riemannian optimization
#   - Exponential/logarithmic maps for manifold operations
#   - Numerical stability via safe projections
#
# References:
#   - Nickel & Kiela (2017): Poincaré embeddings
#   - Ganea et al. (2018): Hyperbolic neural networks

from __future__ import annotations

from typing import Optional, Tuple
import math
import torch
import torch.nn as nn

# Optional geoopt import
try:
    import geoopt
    from geoopt import PoincareBall
    HAS_GEOOPT = True
except ImportError:
    HAS_GEOOPT = False
    PoincareBall = None


# =============================================================================
# Poincaré Ball Operations (Pure PyTorch fallback)
# =============================================================================

def poincare_distance(u: torch.Tensor, v: torch.Tensor, c: float = 1.0) -> torch.Tensor:
    """
    Compute Poincaré distance between points.

    d(u, v) = (2/√c) * arctanh(√c * ||−u ⊕ v||)

    where ⊕ is the Möbius addition.

    Args:
        u: Point in ball, shape (..., d)
        v: Point in ball, shape (..., d)
        c: Curvature (default 1.0)

    Returns:
        Poincaré distance, shape (...)
    """
    # Möbius addition: -u ⊕ v
    neg_u = -u
    diff = mobius_add(neg_u, v, c)

    # Norm
    norm = diff.norm(dim=-1).clamp(max=1.0 - 1e-5)

    # Distance
    sqrt_c = math.sqrt(c)
    dist = (2.0 / sqrt_c) * torch.atanh(sqrt_c * norm)

    return dist


def mobius_add(u: torch.Tensor, v: torch.Tensor, c: float = 1.0) -> torch.Tensor:
    """
    Möbius addition in Poincaré ball.

    u ⊕ v = ((1 + 2c<u,v> + c||v||²)u + (1 - c||u||²)v) /
            (1 + 2c<u,v> + c²||u||²||v||²)

    Args:
        u, v: Points in ball, shape (..., d)
        c: Curvature

    Returns:
        Möbius sum u ⊕ v
    """
    u_sq = (u * u).sum(dim=-1, keepdim=True)
    v_sq = (v * v).sum(dim=-1, keepdim=True)
    uv = (u * v).sum(dim=-1, keepdim=True)

    num = (1 + 2*c*uv + c*v_sq) * u + (1 - c*u_sq) * v
    denom = 1 + 2*c*uv + c*c*u_sq*v_sq

    return num / denom.clamp(min=1e-8)


def expmap0(v: torch.Tensor, c: float = 1.0) -> torch.Tensor:
    """
    Exponential map from origin.

    exp_0(v) = tanh(√c ||v||) * v / (√c ||v||)

    Args:
        v: Tangent vector at origin, shape (..., d)
        c: Curvature

    Returns:
        Point in ball
    """
    sqrt_c = math.sqrt(c)
    v_norm = v.norm(dim=-1, keepdim=True).clamp(min=1e-8)

    return torch.tanh(sqrt_c * v_norm) * v / (sqrt_c * v_norm)


def logmap0(y: torch.Tensor, c: float = 1.0) -> torch.Tensor:
    """
    Logarithmic map to origin.

    log_0(y) = arctanh(√c ||y||) * y / (√c ||y||)

    Args:
        y: Point in ball, shape (..., d)
        c: Curvature

    Returns:
        Tangent vector at origin
    """
    sqrt_c = math.sqrt(c)
    y_norm = y.norm(dim=-1, keepdim=True).clamp(min=1e-8, max=1.0 - 1e-5)

    return torch.atanh(sqrt_c * y_norm) * y / (sqrt_c * y_norm)


def project_to_ball(x: torch.Tensor, c: float = 1.0, eps: float = 1e-5) -> torch.Tensor:
    """
    Project point to interior of Poincaré ball.

    Args:
        x: Point, shape (..., d)
        c: Curvature
        eps: Margin from boundary

    Returns:
        Projected point with ||x|| < 1/√c - eps
    """
    max_norm = 1.0 / math.sqrt(c) - eps
    norm = x.norm(dim=-1, keepdim=True)

    # Scale down if outside ball
    scale = torch.where(
        norm > max_norm,
        max_norm / norm,
        torch.ones_like(norm)
    )

    return x * scale


def parallel_transport(v: torch.Tensor, x: torch.Tensor, y: torch.Tensor, c: float = 1.0) -> torch.Tensor:
    """
    Parallel transport vector v from x to y.

    Args:
        v: Tangent vector at x, shape (..., d)
        x: Source point, shape (..., d)
        y: Target point, shape (..., d)
        c: Curvature

    Returns:
        Transported vector at y
    """
    # Conformal factor
    lambda_x = 2.0 / (1 - c * (x * x).sum(dim=-1, keepdim=True))
    lambda_y = 2.0 / (1 - c * (y * y).sum(dim=-1, keepdim=True))

    # Transport via gyration
    return v * lambda_x / lambda_y


# =============================================================================
# Identity Manifold Classes
# =============================================================================

class IdentityManifold(nn.Module):
    """
    Poincaré ball manifold for agent identity embedding.

    The identity state z ∈ B^d encodes the agent's coherent self-model
    in a hyperbolic space that naturally captures hierarchical structure.

    Attributes:
        dim: Dimension of the Poincaré ball
        curvature: Negative curvature parameter c > 0
        z: Current identity point (learnable)
    """

    def __init__(
        self,
        dim: int = 32,
        curvature: float = 1.0,
        init_scale: float = 0.01,
    ):
        """
        Args:
            dim: Dimension of identity space
            curvature: Ball curvature (higher = more curved)
            init_scale: Scale for initial identity (near origin)
        """
        super().__init__()

        self.dim = dim
        self.curvature = curvature
        self.init_scale = init_scale

        # Identity state (initialized near origin for stability)
        if HAS_GEOOPT:
            self.ball = PoincareBall(c=curvature)
            # Register as manifold parameter
            z_init = torch.randn(dim) * init_scale
            z_init = self.ball.projx(z_init)
            self.z = geoopt.ManifoldParameter(z_init, manifold=self.ball)
        else:
            # Pure PyTorch fallback
            z_init = torch.randn(dim) * init_scale
            z_init = project_to_ball(z_init, curvature)
            self.z = nn.Parameter(z_init)
            self.ball = None

    def project(self, x: torch.Tensor) -> torch.Tensor:
        """Project point to ball interior."""
        if self.ball is not None:
            return self.ball.projx(x)
        return project_to_ball(x, self.curvature)

    def distance(self, u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Compute Poincaré distance."""
        if self.ball is not None:
            return self.ball.dist(u, v)
        return poincare_distance(u, v, self.curvature)

    def distance_from_identity(self, x: torch.Tensor) -> torch.Tensor:
        """Compute distance from current identity."""
        return self.distance(self.z, x)

    def expmap(self, v: torch.Tensor, base: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Exponential map.

        Args:
            v: Tangent vector
            base: Base point (default: origin)

        Returns:
            Point on manifold
        """
        if base is None:
            return expmap0(v, self.curvature)

        if self.ball is not None:
            return self.ball.expmap(base, v)

        # For non-origin base, use Möbius operations
        v_at_origin = self.transport_to_origin(v, base)
        y_at_origin = expmap0(v_at_origin, self.curvature)
        return mobius_add(base, y_at_origin, self.curvature)

    def logmap(self, y: torch.Tensor, base: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Logarithmic map.

        Args:
            y: Point on manifold
            base: Base point (default: origin)

        Returns:
            Tangent vector at base
        """
        if base is None:
            return logmap0(y, self.curvature)

        if self.ball is not None:
            return self.ball.logmap(base, y)

        # Move to origin, compute log, transport back
        y_from_base = mobius_add(-base, y, self.curvature)
        v_at_origin = logmap0(y_from_base, self.curvature)
        return self.transport_from_origin(v_at_origin, base)

    def transport_to_origin(self, v: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Parallel transport from x to origin."""
        origin = torch.zeros_like(x)
        return parallel_transport(v, x, origin, self.curvature)

    def transport_from_origin(self, v: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Parallel transport from origin to x."""
        origin = torch.zeros_like(x)
        return parallel_transport(v, origin, x, self.curvature)

    def update_identity(
        self,
        direction: torch.Tensor,
        step_size: float = 0.01,
    ) -> torch.Tensor:
        """
        Update identity along geodesic direction.

        Args:
            direction: Tangent vector at current identity
            step_size: Step size along geodesic

        Returns:
            New identity point
        """
        # Scale direction
        v = direction * step_size

        # Move along geodesic
        new_z = self.expmap(v, self.z)

        # Update (detach gradient for in-place update)
        with torch.no_grad():
            self.z.copy_(self.project(new_z))

        return self.z

    def identity_coherence(self) -> torch.Tensor:
        """
        Compute identity coherence metric.

        Returns value in [0, 1] where 1 = maximally coherent (near origin).
        """
        dist_from_origin = self.z.norm()
        max_dist = 1.0 / math.sqrt(self.curvature) - 1e-5

        # Coherence decreases as we move toward boundary
        coherence = 1.0 - (dist_from_origin / max_dist).clamp(0, 1)

        return coherence

    def get_identity(self) -> torch.Tensor:
        """Get current identity embedding."""
        return self.z.detach().clone()

    def set_identity(self, z: torch.Tensor) -> None:
        """Set identity embedding."""
        with torch.no_grad():
            self.z.copy_(self.project(z))

    def forward(self, x: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass returns distance to identity.

        Args:
            x: Query point (default: origin)

        Returns:
            Distance from x to identity
        """
        if x is None:
            x = torch.zeros_like(self.z)
        return self.distance_from_identity(x)


class HyperbolicEncoder(nn.Module):
    """
    Encode observations into Poincaré ball.

    Maps high-dimensional observations to hyperbolic identity space
    using exponential map projection.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        output_dim: int = 32,
        curvature: float = 1.0,
        n_layers: int = 2,
    ):
        """
        Args:
            input_dim: Observation dimension
            hidden_dim: Hidden layer dimension
            output_dim: Poincaré ball dimension
            curvature: Ball curvature
            n_layers: Number of hidden layers
        """
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.curvature = curvature

        # Euclidean encoder
        layers = []
        in_dim = input_dim
        for _ in range(n_layers):
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
            ])
            in_dim = hidden_dim

        # Final projection to tangent space
        layers.append(nn.Linear(hidden_dim, output_dim))

        self.encoder = nn.Sequential(*layers)

        # Scale for tangent vector (learnable)
        self.scale = nn.Parameter(torch.ones(1) * 0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode observation to Poincaré ball.

        Args:
            x: Observation, shape (..., input_dim)

        Returns:
            Point in Poincaré ball, shape (..., output_dim)
        """
        # Encode to tangent space at origin
        v = self.encoder(x)

        # Scale and clamp for numerical stability
        v = v * self.scale
        v = v.clamp(-10, 10)

        # Project via exponential map
        z = expmap0(v, self.curvature)

        return z


class HyperbolicMLP(nn.Module):
    """
    MLP that operates in hyperbolic space.

    Uses Möbius operations for addition and applies
    non-linearities in tangent space.
    """

    def __init__(
        self,
        dim: int,
        hidden_dim: int = 64,
        curvature: float = 1.0,
    ):
        """
        Args:
            dim: Poincaré ball dimension
            hidden_dim: Hidden tangent space dimension
            curvature: Ball curvature
        """
        super().__init__()

        self.dim = dim
        self.curvature = curvature

        # Linear layers in tangent space
        self.W1 = nn.Linear(dim, hidden_dim, bias=False)
        self.W2 = nn.Linear(hidden_dim, dim, bias=False)

        # Bias terms (added via Möbius addition)
        self.b1 = nn.Parameter(torch.zeros(hidden_dim) * 0.01)
        self.b2 = nn.Parameter(torch.zeros(dim) * 0.01)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through hyperbolic MLP.

        Args:
            z: Point in Poincaré ball, shape (..., dim)

        Returns:
            Transformed point, shape (..., dim)
        """
        # Map to tangent space at origin
        v = logmap0(z, self.curvature)

        # First layer (in tangent space)
        h = self.W1(v)
        h = h + self.b1
        h = torch.tanh(h)

        # Second layer
        out = self.W2(h)
        out = out + self.b2

        # Map back to ball
        z_out = expmap0(out, self.curvature)

        return z_out


if __name__ == "__main__":
    print("=== Identity Manifold Test ===\n")

    # Create manifold
    manifold = IdentityManifold(dim=32, curvature=1.0)
    print(f"Identity dim: {manifold.dim}")
    print(f"Curvature: {manifold.curvature}")
    print(f"Initial identity norm: {manifold.z.norm().item():.4f}")
    print(f"Initial coherence: {manifold.identity_coherence().item():.4f}")

    # Test distance computation
    print("\n--- Distance Tests ---")
    origin = torch.zeros(32)
    z = manifold.z.detach()
    d = manifold.distance(origin, z)
    print(f"Distance from origin: {d.item():.4f}")

    # Test exp/log maps
    print("\n--- Exp/Log Map Tests ---")
    v = torch.randn(32) * 0.1
    y = manifold.expmap(v)
    print(f"Exp map ||v||={v.norm().item():.4f} -> ||y||={y.norm().item():.4f}")

    v_back = manifold.logmap(y)
    print(f"Log map reconstruction error: {(v - v_back).norm().item():.6f}")

    # Test identity update
    print("\n--- Identity Update ---")
    direction = torch.randn(32)
    old_z = manifold.z.clone()
    new_z = manifold.update_identity(direction, step_size=0.1)
    move_dist = manifold.distance(old_z, new_z)
    print(f"Moved distance: {move_dist.item():.4f}")
    print(f"New coherence: {manifold.identity_coherence().item():.4f}")

    # Test encoder
    print("\n--- Hyperbolic Encoder ---")
    encoder = HyperbolicEncoder(input_dim=64, output_dim=32)
    x = torch.randn(8, 64)  # Batch of observations
    z_encoded = encoder(x)
    print(f"Input shape: {x.shape}")
    print(f"Encoded shape: {z_encoded.shape}")
    print(f"Encoded norms: {z_encoded.norm(dim=-1).mean().item():.4f} (should be < 1)")

    # Test Möbius addition
    print("\n--- Möbius Addition ---")
    u = torch.randn(32) * 0.3
    v = torch.randn(32) * 0.3
    u = project_to_ball(u)
    v = project_to_ball(v)
    w = mobius_add(u, v)
    print(f"||u||={u.norm().item():.4f}, ||v||={v.norm().item():.4f}, ||u⊕v||={w.norm().item():.4f}")

    # Test hyperbolic MLP
    print("\n--- Hyperbolic MLP ---")
    mlp = HyperbolicMLP(dim=32)
    z_in = torch.randn(8, 32) * 0.3
    z_in = project_to_ball(z_in)
    z_out = mlp(z_in)
    print(f"MLP input norms: {z_in.norm(dim=-1).mean().item():.4f}")
    print(f"MLP output norms: {z_out.norm(dim=-1).mean().item():.4f}")

    # Verify all outputs in ball
    assert (z_out.norm(dim=-1) < 1.0).all(), "MLP output outside ball!"

    print("\n✓ Identity manifold test passed!")
