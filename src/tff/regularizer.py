"""TopologicalRegularizer - Stability constraint via persistent homology.

This module implements topological regularization that:
1. Encourages stable Betti number evolution during training
2. Penalizes drastic changes in topological structure
3. Uses EMA-smoothed targets for stable optimization

Key insight: Representations with stable topology are more robust
to perturbations and transfer better across domains.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from collections import deque

import torch
import torch.nn as nn
import torch.nn.functional as F

from .topology import (
    PersistenceDiagram,
    PersistentHomologyEngine,
    PHConfig,
)


@dataclass
class RegularizerConfig:
    """Configuration for TopologicalRegularizer."""

    # Betti stability
    betti_weight: float = 0.1  # Weight for Betti stability loss
    betti_ema_decay: float = 0.99  # EMA decay for target Betti numbers

    # Persistence stability
    persistence_weight: float = 0.05  # Weight for persistence stability
    min_persistence_threshold: float = 0.01  # Ignore small features

    # Wasserstein loss (diagram distance)
    wasserstein_weight: float = 0.02
    wasserstein_p: int = 2  # p-Wasserstein distance

    # Entropy regularization
    entropy_weight: float = 0.01  # Encourage diverse persistence values

    # Dimension weights
    dimension_weights: Dict[int, float] = field(
        default_factory=lambda: {0: 1.0, 1: 2.0, 2: 1.0}
    )

    # Update frequency (PH is expensive)
    update_every: int = 10  # Update targets every N steps


class BettiTracker:
    """Tracks Betti numbers with EMA smoothing."""

    def __init__(
        self,
        num_dimensions: int = 3,
        ema_decay: float = 0.99,
        history_size: int = 100,
    ):
        self.num_dimensions = num_dimensions
        self.ema_decay = ema_decay

        # EMA targets
        self.targets = torch.zeros(num_dimensions)

        # History for variance estimation
        self.history: List[deque] = [
            deque(maxlen=history_size) for _ in range(num_dimensions)
        ]

        self.initialized = False
        self.step_count = 0

    def update(self, betti_numbers: torch.Tensor) -> None:
        """Update EMA targets with new Betti numbers."""
        if not self.initialized:
            self.targets = betti_numbers.clone()
            self.initialized = True
        else:
            self.targets = (
                self.ema_decay * self.targets +
                (1 - self.ema_decay) * betti_numbers
            )

        # Update history
        for d in range(min(len(betti_numbers), self.num_dimensions)):
            self.history[d].append(betti_numbers[d].item())

        self.step_count += 1

    def get_targets(self) -> torch.Tensor:
        """Get current EMA targets."""
        return self.targets

    def get_variance(self) -> torch.Tensor:
        """Get variance estimates from history."""
        variances = []
        for d in range(self.num_dimensions):
            if len(self.history[d]) > 1:
                values = torch.tensor(list(self.history[d]))
                variances.append(values.var().item())
            else:
                variances.append(1.0)
        return torch.tensor(variances)


class TopologicalRegularizer(nn.Module):
    """Regularizer that encourages topological stability.

    Loss components:
    1. Betti stability: ||β_k - EMA(β_k)||² per dimension
    2. Persistence stability: variance of persistence values
    3. Wasserstein: distance between current and reference diagrams
    4. Entropy: encourages diverse but stable persistence distribution
    """

    def __init__(self, config: RegularizerConfig):
        super().__init__()
        self.config = config

        # Betti tracker
        self.betti_tracker = BettiTracker(
            num_dimensions=3,
            ema_decay=config.betti_ema_decay,
        )

        # Reference diagrams (EMA-smoothed)
        self._reference_diagrams: Optional[Dict[int, PersistenceDiagram]] = None
        self._diagram_ema_decay = 0.95

        # Step counter
        self.register_buffer("step", torch.tensor(0))

    def compute_betti_numbers(
        self,
        diagrams: Dict[int, PersistenceDiagram],
    ) -> torch.Tensor:
        """Compute Betti numbers from persistence diagrams.

        Betti_k = number of features with persistence > threshold
        """
        threshold = self.config.min_persistence_threshold
        betti = []

        for d in range(3):
            if d in diagrams:
                diagram = diagrams[d]
                persistent_features = (diagram.persistence > threshold).sum()
                betti.append(persistent_features.float())
            else:
                betti.append(torch.tensor(0.0))

        return torch.stack(betti)

    def betti_stability_loss(
        self,
        betti_numbers: torch.Tensor,
    ) -> torch.Tensor:
        """Compute Betti stability loss.

        Penalizes deviation from EMA-smoothed targets.
        """
        targets = self.betti_tracker.get_targets().to(betti_numbers.device)

        # Weighted MSE per dimension
        weights = torch.tensor([
            self.config.dimension_weights.get(d, 1.0)
            for d in range(len(betti_numbers))
        ]).to(betti_numbers.device)

        loss = ((betti_numbers - targets) ** 2 * weights).mean()

        return loss

    def persistence_stability_loss(
        self,
        diagrams: Dict[int, PersistenceDiagram],
    ) -> torch.Tensor:
        """Compute persistence stability loss.

        Penalizes high variance in persistence values within each dimension.
        """
        total_loss = 0.0
        count = 0

        for d, diagram in diagrams.items():
            if len(diagram) < 2:
                continue

            pers = diagram.persistence
            weight = self.config.dimension_weights.get(d, 1.0)

            # Variance of persistence values
            variance = pers.var()
            total_loss += weight * variance
            count += 1

        if count > 0:
            return total_loss / count
        return torch.tensor(0.0)

    def wasserstein_loss(
        self,
        diagrams: Dict[int, PersistenceDiagram],
    ) -> torch.Tensor:
        """Compute approximate Wasserstein distance to reference diagrams.

        Uses greedy matching for efficiency (not true Wasserstein).
        """
        if self._reference_diagrams is None:
            return torch.tensor(0.0)

        total_loss = 0.0
        count = 0

        for d in diagrams:
            if d not in self._reference_diagrams:
                continue

            current = diagrams[d]
            reference = self._reference_diagrams[d]

            if len(current) == 0 or len(reference) == 0:
                continue

            # Compute pairwise distances in (birth, death) space
            current_pts = current.pairs  # (N, 2)
            ref_pts = reference.pairs  # (M, 2)

            # Distance matrix
            dist = torch.cdist(current_pts.float(), ref_pts.float(), p=2)

            # Greedy matching (approximation)
            matched_dist = 0.0
            matched_count = 0

            for i in range(min(len(current), len(reference))):
                if dist.numel() == 0:
                    break
                min_val = dist.min()
                matched_dist += min_val ** self.config.wasserstein_p
                matched_count += 1

                # Remove matched pair
                min_idx = (dist == min_val).nonzero()[0]
                dist = torch.cat([
                    dist[:min_idx[0]],
                    dist[min_idx[0]+1:],
                ], dim=0)
                dist = torch.cat([
                    dist[:, :min_idx[1]],
                    dist[:, min_idx[1]+1:],
                ], dim=1)

            if matched_count > 0:
                weight = self.config.dimension_weights.get(d, 1.0)
                total_loss += weight * (matched_dist / matched_count) ** (1 / self.config.wasserstein_p)
                count += 1

        if count > 0:
            return total_loss / count
        return torch.tensor(0.0)

    def entropy_loss(
        self,
        diagrams: Dict[int, PersistenceDiagram],
    ) -> torch.Tensor:
        """Compute entropy regularization.

        Encourages diverse persistence values (avoid collapse to single value).
        """
        total_entropy = 0.0
        count = 0

        for d, diagram in diagrams.items():
            if len(diagram) < 2:
                continue

            pers = diagram.persistence + 1e-8
            # Normalize to probability distribution
            probs = pers / pers.sum()

            # Entropy (negative because we want to maximize)
            entropy = -(probs * torch.log(probs)).sum()

            weight = self.config.dimension_weights.get(d, 1.0)
            total_entropy += weight * entropy
            count += 1

        if count > 0:
            # Return negative entropy (we minimize loss, so this maximizes entropy)
            return -total_entropy / count
        return torch.tensor(0.0)

    def update_references(
        self,
        diagrams: Dict[int, PersistenceDiagram],
    ) -> None:
        """Update reference diagrams with EMA."""
        if self._reference_diagrams is None:
            self._reference_diagrams = diagrams
            return

        # Simple EMA update on diagram points
        for d in diagrams:
            if d in self._reference_diagrams:
                ref = self._reference_diagrams[d]
                new = diagrams[d]

                # Match by closest point and interpolate
                if len(ref) > 0 and len(new) > 0:
                    # Truncate to same size
                    min_len = min(len(ref), len(new))
                    ref_pts = ref.pairs[:min_len]
                    new_pts = new.pairs[:min_len]

                    updated = (
                        self._diagram_ema_decay * ref_pts +
                        (1 - self._diagram_ema_decay) * new_pts
                    )

                    self._reference_diagrams[d] = PersistenceDiagram(
                        updated, dimension=d
                    )
            else:
                self._reference_diagrams[d] = diagrams[d]

    def forward(
        self,
        diagrams: Dict[int, PersistenceDiagram],
        update_targets: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """Compute total regularization loss.

        Args:
            diagrams: Persistence diagrams per dimension
            update_targets: Whether to update EMA targets

        Returns:
            Dict with loss components and total
        """
        self.step += 1

        # Compute Betti numbers
        betti = self.compute_betti_numbers(diagrams)

        # Update tracker
        if update_targets and self.step % self.config.update_every == 0:
            self.betti_tracker.update(betti)
            self.update_references(diagrams)

        # Compute losses
        losses = {}

        # Betti stability
        if self.config.betti_weight > 0:
            losses["betti_stability"] = self.config.betti_weight * self.betti_stability_loss(betti)

        # Persistence stability
        if self.config.persistence_weight > 0:
            losses["persistence_stability"] = (
                self.config.persistence_weight * self.persistence_stability_loss(diagrams)
            )

        # Wasserstein loss
        if self.config.wasserstein_weight > 0:
            losses["wasserstein"] = self.config.wasserstein_weight * self.wasserstein_loss(diagrams)

        # Entropy regularization
        if self.config.entropy_weight > 0:
            losses["entropy"] = self.config.entropy_weight * self.entropy_loss(diagrams)

        # Total loss
        total = sum(losses.values())
        losses["total"] = total

        # Add Betti numbers for logging
        losses["betti_0"] = betti[0]
        losses["betti_1"] = betti[1]
        losses["betti_2"] = betti[2] if len(betti) > 2 else torch.tensor(0.0)

        return losses


class TopologicalRegularizerWrapper(nn.Module):
    """Wrapper that combines PH computation with regularization.

    Convenience class that handles the full pipeline from
    embeddings to regularization loss.
    """

    def __init__(
        self,
        ph_config: Optional[PHConfig] = None,
        reg_config: Optional[RegularizerConfig] = None,
    ):
        super().__init__()

        self.ph_config = ph_config or PHConfig()
        self.reg_config = reg_config or RegularizerConfig()

        self.ph_engine = PersistentHomologyEngine(self.ph_config)
        self.regularizer = TopologicalRegularizer(self.reg_config)

    def forward(
        self,
        embeddings: torch.Tensor,
        update_targets: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """Compute regularization loss from embeddings.

        Args:
            embeddings: Input embeddings (batch, seq_len, d_model) or (N, d_model)
            update_targets: Whether to update EMA targets

        Returns:
            Dict with loss components
        """
        # Flatten to point cloud
        if embeddings.dim() == 3:
            points = embeddings.mean(dim=1)  # (batch, d_model)
        else:
            points = embeddings

        # Compute persistence diagrams
        diagrams = self.ph_engine.compute(points)

        # Compute regularization loss
        return self.regularizer(diagrams, update_targets=update_targets)


def create_topological_regularizer(
    betti_weight: float = 0.1,
    persistence_weight: float = 0.05,
) -> TopologicalRegularizerWrapper:
    """Factory function for TopologicalRegularizerWrapper."""
    reg_config = RegularizerConfig(
        betti_weight=betti_weight,
        persistence_weight=persistence_weight,
    )
    return TopologicalRegularizerWrapper(reg_config=reg_config)
