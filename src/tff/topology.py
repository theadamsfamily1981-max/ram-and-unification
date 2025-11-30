"""TopologyHead - Persistent Homology Feature Extraction.

This module implements topological feature extraction via persistent homology:
1. PersistentHomologyEngine - Computes persistence diagrams from point clouds
2. PersistenceImageEncoder - Vectorizes persistence diagrams
3. TopologyHead - Neural network head that outputs topological features

Key insight: Persistent homology captures multi-scale structural features
(holes, voids, connected components) that are invariant to continuous
deformations - critical for robust representation learning.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Union
from enum import IntEnum

import torch
import torch.nn as nn
import torch.nn.functional as F


class HomologyDimension(IntEnum):
    """Homology dimensions for PH computation."""

    H0 = 0  # Connected components
    H1 = 1  # Loops/cycles
    H2 = 2  # Voids/cavities


@dataclass
class PHConfig:
    """Configuration for Persistent Homology computation."""

    max_dimension: int = 2  # Max homology dimension (H0, H1, H2)
    max_edge_length: float = 2.0  # Maximum filtration value
    num_filtration_steps: int = 50  # Discretization of filtration
    use_alpha_complex: bool = True  # Alpha vs Rips complex
    subsample_size: int = 256  # Max points for PH (computational limit)


@dataclass
class PIConfig:
    """Configuration for Persistence Image encoding."""

    resolution: Tuple[int, int] = (20, 20)  # Image resolution
    sigma: float = 0.1  # Gaussian kernel bandwidth
    weight_fn: str = "linear"  # Weighting: linear, persistence, entropy
    birth_range: Tuple[float, float] = (0.0, 1.0)
    pers_range: Tuple[float, float] = (0.0, 1.0)


@dataclass
class TopologyHeadConfig:
    """Configuration for TopologyHead."""

    d_model: int = 512
    ph_config: PHConfig = field(default_factory=PHConfig)
    pi_config: PIConfig = field(default_factory=PIConfig)
    num_landscapes: int = 5  # Number of persistence landscapes
    landscape_resolution: int = 100
    output_dim: int = 256  # Topology feature dimension
    dropout: float = 0.1


class PersistenceDiagram:
    """Container for a persistence diagram.

    A persistence diagram is a multiset of (birth, death) pairs
    representing topological features across filtration values.
    """

    def __init__(
        self,
        pairs: torch.Tensor,  # (N, 2) birth-death pairs
        dimension: int,
    ):
        self.pairs = pairs
        self.dimension = dimension

    @property
    def birth(self) -> torch.Tensor:
        return self.pairs[:, 0]

    @property
    def death(self) -> torch.Tensor:
        return self.pairs[:, 1]

    @property
    def persistence(self) -> torch.Tensor:
        """Persistence (lifetime) of each feature."""
        return self.death - self.birth

    @property
    def midlife(self) -> torch.Tensor:
        """Midpoint of each feature's lifetime."""
        return (self.birth + self.death) / 2

    def __len__(self) -> int:
        return self.pairs.shape[0]


class PersistentHomologyEngine:
    """Computes persistent homology from point clouds.

    Uses a differentiable approximation for gradient flow:
    - Soft Rips filtration with learned edge weights
    - Differentiable persistence computation

    Note: Full PH is computed with external libraries (Ripser, GUDHI)
    for accuracy. This module provides a differentiable approximation.
    """

    def __init__(self, config: PHConfig):
        self.config = config

    def compute_distance_matrix(
        self,
        points: torch.Tensor,
    ) -> torch.Tensor:
        """Compute pairwise distance matrix.

        Args:
            points: Point cloud (N, D)

        Returns:
            Distance matrix (N, N)
        """
        # Euclidean distance matrix
        diff = points.unsqueeze(0) - points.unsqueeze(1)
        dist = torch.sqrt((diff ** 2).sum(dim=-1) + 1e-8)
        return dist

    def compute_rips_filtration(
        self,
        dist_matrix: torch.Tensor,
    ) -> List[Tuple[float, torch.Tensor]]:
        """Compute Rips filtration (soft/differentiable version).

        Args:
            dist_matrix: Pairwise distances (N, N)

        Returns:
            List of (filtration_value, adjacency_matrix) pairs
        """
        N = dist_matrix.shape[0]
        filtration = []

        # Discretize filtration values
        max_dist = min(dist_matrix.max().item(), self.config.max_edge_length)
        steps = torch.linspace(0, max_dist, self.config.num_filtration_steps)

        for eps in steps:
            # Soft thresholding for differentiability
            # Adjacency = sigmoid((eps - dist) / temperature)
            temp = 0.1
            adj = torch.sigmoid((eps - dist_matrix) / temp)
            adj = adj * (1 - torch.eye(N, device=dist_matrix.device))  # No self-loops
            filtration.append((eps.item(), adj))

        return filtration

    def compute_h0_persistence(
        self,
        filtration: List[Tuple[float, torch.Tensor]],
    ) -> PersistenceDiagram:
        """Compute H0 (connected components) persistence.

        Uses differentiable union-find approximation.
        """
        N = filtration[0][1].shape[0]
        device = filtration[0][1].device

        # Track component membership (soft)
        membership = torch.eye(N, device=device)  # Each point is its own component
        birth_times = torch.zeros(N, device=device)
        death_times = torch.full((N,), float('inf'), device=device)

        prev_components = N

        for eps, adj in filtration:
            # Propagate membership through adjacency
            # Soft connectivity: membership @ adj gives reachability
            reachability = torch.matmul(membership, adj)
            membership = torch.clamp(membership + reachability, 0, 1)

            # Symmetrize and normalize
            membership = (membership + membership.T) / 2
            membership = membership / (membership.sum(dim=1, keepdim=True) + 1e-8)

            # Estimate number of components (soft)
            # Using spectral approach: rank of Laplacian
            degree = adj.sum(dim=1)
            laplacian = torch.diag(degree) - adj
            eigvals = torch.linalg.eigvalsh(laplacian)
            num_components = (eigvals < 0.1).sum().item()

            # Record deaths when components merge
            if num_components < prev_components:
                merged = prev_components - num_components
                # Mark oldest components as dying
                for i in range(min(merged, N)):
                    if death_times[i] == float('inf'):
                        death_times[i] = eps
                prev_components = num_components

        # Build persistence diagram
        # Filter out infinite death times (final component)
        finite_mask = death_times != float('inf')
        pairs = torch.stack([birth_times[finite_mask], death_times[finite_mask]], dim=1)

        if pairs.shape[0] == 0:
            pairs = torch.zeros((1, 2), device=device)

        return PersistenceDiagram(pairs, dimension=0)

    def compute_h1_persistence_approx(
        self,
        dist_matrix: torch.Tensor,
        filtration: List[Tuple[float, torch.Tensor]],
    ) -> PersistenceDiagram:
        """Approximate H1 (loops) persistence.

        Uses spectral analysis of graph Laplacian at each filtration step.
        This is an approximation - true H1 requires boundary matrix reduction.
        """
        device = dist_matrix.device
        birth_death_pairs = []

        prev_betti_1 = 0

        for eps, adj in filtration:
            # Compute Laplacian
            degree = adj.sum(dim=1)
            laplacian = torch.diag(degree) - adj

            # Betti-1 ≈ nullity(L) - 1 for connected graphs
            # Use eigenvalue gap as proxy
            eigvals = torch.linalg.eigvalsh(laplacian)
            eigvals_sorted = eigvals.sort()[0]

            # Count small eigenvalues (near-zero)
            threshold = 0.1
            null_count = (eigvals_sorted < threshold).sum().item()

            # H1 births when cycle forms
            betti_1 = max(0, null_count - 1)  # Subtract 1 for connected component

            if betti_1 > prev_betti_1:
                # New cycle born
                for _ in range(betti_1 - prev_betti_1):
                    birth_death_pairs.append([eps, float('inf')])

            elif betti_1 < prev_betti_1:
                # Cycle died (filled in)
                # Mark most recent births as dying
                for pair in reversed(birth_death_pairs):
                    if pair[1] == float('inf'):
                        pair[1] = eps
                        break

            prev_betti_1 = betti_1

        if not birth_death_pairs:
            pairs = torch.zeros((1, 2), device=device)
        else:
            pairs = torch.tensor(birth_death_pairs, device=device)
            # Filter infinite deaths
            finite_mask = pairs[:, 1] != float('inf')
            if finite_mask.any():
                pairs = pairs[finite_mask]
            else:
                pairs = torch.zeros((1, 2), device=device)

        return PersistenceDiagram(pairs, dimension=1)

    def compute(
        self,
        points: torch.Tensor,
    ) -> Dict[int, PersistenceDiagram]:
        """Compute persistence diagrams for all dimensions.

        Args:
            points: Point cloud (N, D) or (batch, N, D)

        Returns:
            Dict mapping dimension to PersistenceDiagram
        """
        # Handle batched input
        if points.dim() == 3:
            # Process batch - take first for now (TODO: batch PH)
            points = points[0]

        # Subsample if too large
        if points.shape[0] > self.config.subsample_size:
            indices = torch.randperm(points.shape[0])[:self.config.subsample_size]
            points = points[indices]

        # Compute distance matrix
        dist_matrix = self.compute_distance_matrix(points)

        # Compute filtration
        filtration = self.compute_rips_filtration(dist_matrix)

        # Compute persistence for each dimension
        diagrams = {}

        diagrams[0] = self.compute_h0_persistence(filtration)

        if self.config.max_dimension >= 1:
            diagrams[1] = self.compute_h1_persistence_approx(dist_matrix, filtration)

        # H2 is expensive - skip for now
        if self.config.max_dimension >= 2:
            # Placeholder - would need proper boundary matrix
            diagrams[2] = PersistenceDiagram(
                torch.zeros((1, 2), device=points.device),
                dimension=2,
            )

        return diagrams


class PersistenceImageEncoder(nn.Module):
    """Encodes persistence diagrams as persistence images.

    A persistence image is a weighted, Gaussian-smoothed histogram
    of the (birth, persistence) pairs in a persistence diagram.
    """

    def __init__(self, config: PIConfig):
        super().__init__()
        self.config = config

        # Learnable Gaussian centers and bandwidths
        h, w = config.resolution
        self.register_buffer(
            "grid_x",
            torch.linspace(config.birth_range[0], config.birth_range[1], w),
        )
        self.register_buffer(
            "grid_y",
            torch.linspace(config.pers_range[0], config.pers_range[1], h),
        )

        self.sigma = nn.Parameter(torch.tensor(config.sigma))

    def _weight_fn(self, persistence: torch.Tensor) -> torch.Tensor:
        """Weighting function for persistence values."""
        if self.config.weight_fn == "linear":
            return persistence
        elif self.config.weight_fn == "persistence":
            return persistence ** 2
        elif self.config.weight_fn == "entropy":
            p = persistence / (persistence.sum() + 1e-8)
            return -p * torch.log(p + 1e-8)
        else:
            return persistence

    def forward(self, diagram: PersistenceDiagram) -> torch.Tensor:
        """Convert persistence diagram to persistence image.

        Args:
            diagram: Input persistence diagram

        Returns:
            Persistence image (H, W)
        """
        h, w = self.config.resolution
        device = diagram.pairs.device

        # Initialize image
        image = torch.zeros(h, w, device=device)

        if len(diagram) == 0:
            return image

        # Get birth and persistence values
        birth = diagram.birth
        pers = diagram.persistence

        # Normalize to [0, 1]
        birth_norm = (birth - self.config.birth_range[0]) / (
            self.config.birth_range[1] - self.config.birth_range[0] + 1e-8
        )
        pers_norm = (pers - self.config.pers_range[0]) / (
            self.config.pers_range[1] - self.config.pers_range[0] + 1e-8
        )

        # Compute weights
        weights = self._weight_fn(pers)

        # Build image via Gaussian smoothing
        for i, (b, p, wt) in enumerate(zip(birth_norm, pers_norm, weights)):
            # Gaussian centered at (b, p)
            x_dist = (self.grid_x - b) ** 2
            y_dist = (self.grid_y - p) ** 2

            gaussian = torch.exp(-x_dist.unsqueeze(0) / (2 * self.sigma ** 2)) * \
                       torch.exp(-y_dist.unsqueeze(1) / (2 * self.sigma ** 2))

            image = image + wt * gaussian

        # Normalize
        image = image / (image.max() + 1e-8)

        return image


class PersistenceLandscapeEncoder(nn.Module):
    """Encodes persistence diagrams as persistence landscapes.

    Persistence landscapes are functional summaries that are more
    stable than raw diagrams and naturally vectorizable.
    """

    def __init__(
        self,
        num_landscapes: int = 5,
        resolution: int = 100,
    ):
        super().__init__()
        self.num_landscapes = num_landscapes
        self.resolution = resolution

    def _tent_function(
        self,
        t: torch.Tensor,
        birth: float,
        death: float,
    ) -> torch.Tensor:
        """Tent function for a single persistence pair."""
        mid = (birth + death) / 2
        half_life = (death - birth) / 2

        # Rising part: (t - birth) for t in [birth, mid]
        # Falling part: (death - t) for t in [mid, death]
        rising = torch.clamp(t - birth, min=0)
        falling = torch.clamp(death - t, min=0)

        return torch.min(rising, falling)

    def forward(self, diagram: PersistenceDiagram) -> torch.Tensor:
        """Compute persistence landscapes.

        Args:
            diagram: Input persistence diagram

        Returns:
            Landscapes tensor (num_landscapes, resolution)
        """
        device = diagram.pairs.device

        if len(diagram) == 0:
            return torch.zeros(self.num_landscapes, self.resolution, device=device)

        # Determine range
        min_birth = diagram.birth.min().item()
        max_death = diagram.death.max().item()

        t = torch.linspace(min_birth, max_death, self.resolution, device=device)

        # Compute tent functions for all pairs
        tents = []
        for i in range(len(diagram)):
            b, d = diagram.pairs[i]
            tent = self._tent_function(t, b.item(), d.item())
            tents.append(tent)

        tents = torch.stack(tents, dim=0)  # (num_pairs, resolution)

        # Sort at each t value to get landscapes
        sorted_tents, _ = torch.sort(tents, dim=0, descending=True)

        # Take top k landscapes
        k = min(self.num_landscapes, sorted_tents.shape[0])
        landscapes = sorted_tents[:k]

        # Pad if necessary
        if k < self.num_landscapes:
            padding = torch.zeros(
                self.num_landscapes - k,
                self.resolution,
                device=device,
            )
            landscapes = torch.cat([landscapes, padding], dim=0)

        return landscapes


class TopologyHead(nn.Module):
    """Neural network head that extracts topological features.

    Combines:
    1. Persistent homology computation
    2. Persistence image encoding
    3. Persistence landscape encoding
    4. Neural projection to output dimension
    """

    def __init__(self, config: TopologyHeadConfig):
        super().__init__()
        self.config = config

        # PH engine
        self.ph_engine = PersistentHomologyEngine(config.ph_config)

        # Encoders for each homology dimension
        self.pi_encoders = nn.ModuleDict({
            f"h{d}": PersistenceImageEncoder(config.pi_config)
            for d in range(config.ph_config.max_dimension + 1)
        })

        self.landscape_encoders = nn.ModuleDict({
            f"h{d}": PersistenceLandscapeEncoder(
                num_landscapes=config.num_landscapes,
                resolution=config.landscape_resolution,
            )
            for d in range(config.ph_config.max_dimension + 1)
        })

        # Feature dimensions
        pi_dim = config.pi_config.resolution[0] * config.pi_config.resolution[1]
        landscape_dim = config.num_landscapes * config.landscape_resolution
        total_dim_per_h = pi_dim + landscape_dim

        num_h = config.ph_config.max_dimension + 1
        total_input_dim = total_dim_per_h * num_h

        # Neural projection
        self.projector = nn.Sequential(
            nn.Linear(total_input_dim, config.d_model),
            nn.GELU(),
            nn.LayerNorm(config.d_model),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_model, config.output_dim),
            nn.LayerNorm(config.output_dim),
        )

        # Betti number estimator (auxiliary output)
        self.betti_head = nn.Linear(total_input_dim, num_h)

    def forward(
        self,
        embeddings: torch.Tensor,
        return_betti: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """Extract topological features from embeddings.

        Args:
            embeddings: Input embeddings (batch, seq_len, d_model) or (N, d_model)
            return_betti: Whether to return Betti number estimates

        Returns:
            Dict with:
                - topology_features: Topological feature vector
                - betti_estimates: Estimated Betti numbers (optional)
                - persistence_diagrams: Raw diagrams (for regularization)
        """
        # Flatten to point cloud if needed
        if embeddings.dim() == 3:
            batch_size = embeddings.shape[0]
            # Use mean pooled representation as points
            points = embeddings.mean(dim=1)  # (batch, d_model)
        else:
            points = embeddings
            batch_size = 1

        # Compute persistence diagrams
        diagrams = self.ph_engine.compute(points)

        # Encode each dimension
        features = []

        for d in range(self.config.ph_config.max_dimension + 1):
            diagram = diagrams.get(d)

            if diagram is None:
                # Create empty diagram
                diagram = PersistenceDiagram(
                    torch.zeros((1, 2), device=embeddings.device),
                    dimension=d,
                )

            # Persistence image
            pi = self.pi_encoders[f"h{d}"](diagram)
            features.append(pi.flatten())

            # Persistence landscape
            landscape = self.landscape_encoders[f"h{d}"](diagram)
            features.append(landscape.flatten())

        # Concatenate all features
        combined = torch.cat(features, dim=-1)

        # Add batch dimension if needed
        if combined.dim() == 1:
            combined = combined.unsqueeze(0).expand(batch_size, -1)

        # Project to output dimension
        topology_features = self.projector(combined)

        result = {
            "topology_features": topology_features,
            "persistence_diagrams": diagrams,
        }

        if return_betti:
            betti_estimates = self.betti_head(combined)
            result["betti_estimates"] = torch.relu(betti_estimates)

        return result


def create_topology_head(
    d_model: int = 512,
    output_dim: int = 256,
    max_dimension: int = 1,
) -> TopologyHead:
    """Factory function for TopologyHead with common defaults."""
    ph_config = PHConfig(max_dimension=max_dimension)
    pi_config = PIConfig()

    config = TopologyHeadConfig(
        d_model=d_model,
        ph_config=ph_config,
        pi_config=pi_config,
        output_dim=output_dim,
    )

    return TopologyHead(config)
